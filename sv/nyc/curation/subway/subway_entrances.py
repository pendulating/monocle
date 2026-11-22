"""Top-level orchestration: fetch → normalize → buffer points → validate → write.

Subway entrances are mostly sidewalk stairs and other street furniture,
not building features — so this build skips :mod:`..geom.attach_geometry`
(BIN match → nearest building → point) and buffers the entrance point
directly. Each row's geometry is the entrance ``(lat, lon)`` point
buffered by ``buffer_ft`` in EPSG:2263 (US feet) and reprojected to
WGS84.

Output dir layout:

    <out>/
      by_source/subway_raw.parquet
      entrances.parquet         — publishable rows with WKB geometry
      entrances.geojson         — same, GeoJSON FeatureCollection
      coverage.geojson          — unary_union of buffered points
      manifest.json
      summary.md
      validation_report.parquet

The ``materialize-cyclomedia`` tool's per-unit attribution code recognizes
``entrances.parquet`` (auto-detect updated alongside this build) and looks
up unit ID + name as ``uid`` + ``facname``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import geopandas as gpd
import polars as pl
from shapely.geometry import Point, mapping
from shapely.geometry.base import BaseGeometry

from .entrance_types import validate_entrance_types
from .fetch import fetch_subway_entrances
from .normalize import BOROUGH_MAP, COMMON_COLUMNS, normalize_subway_entrances
from .validation import SubwayValidationError, ValidationResult, run_validation

__all__ = ["build", "SubwayBuildResult"]

log = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"
NYC_SP_CRS = "EPSG:2263"
WGS84_CRS = "EPSG:4326"

VALID_DIVISIONS: tuple[str, ...] = ("IRT", "IND", "BMT", "SIR", "IRT/BMT", "IND/BMT")
VALID_BOROUGH_CODES: tuple[str, ...] = ("M", "B", "Bx", "Q", "SI")


@dataclass
class SubwayBuildResult:
    output_root: str
    total_publishable: int
    raw_rows: int
    funnel: dict[str, int]
    pagination: dict[str, Any]
    summary_path: str
    report_path: str
    entrances_parquet: str
    entrances_geojson: str
    coverage_geojson: str
    manifest_path: str
    filters: dict[str, list[str]]
    elapsed_s: float = 0.0
    validation: Optional[ValidationResult] = None


def _git_sha() -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _buffer_points(df: pl.DataFrame, buffer_ft: float) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame whose geometry is each row's lat/lon buffered.

    Buffer is applied in EPSG:2263 (US feet) and reprojected to WGS84.
    """
    pdf = df.to_pandas()
    points = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(pdf["longitude"], pdf["latitude"])],
        crs=WGS84_CRS,
    )
    pts_sp = points.to_crs(NYC_SP_CRS)
    bufs_sp = pts_sp.buffer(float(buffer_ft))
    bufs = bufs_sp.to_crs(WGS84_CRS)
    gdf = gpd.GeoDataFrame(pdf, geometry=bufs, crs=WGS84_CRS)
    gdf["geom_source"] = "point"
    gdf["match_dist_ft"] = 0.0
    return gdf


def _write_entrances_parquet(gdf: gpd.GeoDataFrame, path: str) -> None:
    from shapely.wkb import dumps as wkb_dumps
    wkbs = [wkb_dumps(g) if g is not None and not g.is_empty else None for g in gdf.geometry]
    df = pl.from_pandas(gdf.drop(columns=["geometry"]))
    df = df.with_columns(pl.Series("geom_wkb", wkbs, dtype=pl.Binary))
    df.write_parquet(path)


def _write_entrances_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
    gdf.to_file(path, driver="GeoJSON")


def _write_coverage_geojson(coverage: BaseGeometry, path: str) -> None:
    from shapely.geometry import MultiPolygon as _MP
    if isinstance(coverage, _MP):
        polys = list(coverage.geoms)
    elif coverage is None or coverage.is_empty:
        polys = []
    else:
        polys = [coverage]
    features = [
        {"type": "Feature",
         "properties": {"name": "subway_coverage", "part": i},
         "geometry": mapping(p)}
        for i, p in enumerate(polys)
    ]
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def _write_manifest(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _validate_divisions(values: list[str]) -> list[str]:
    legal = {d.upper(): d for d in VALID_DIVISIONS}
    out: list[str] = []
    for v in values:
        key = v.strip().upper()
        if key in legal:
            out.append(legal[key])
        else:
            raise ValueError(
                f"unknown division {v!r}; expected one of {list(VALID_DIVISIONS)}"
            )
    return out


def _validate_boroughs(values: list[str]) -> list[str]:
    """Canonicalize borough filter to the dataset's single/two-letter codes.

    Accepts both the source codes (M / B / Bx / Q / SI) and full names
    (Manhattan / Brooklyn / Bronx / Queens / 'Staten Island').
    """
    code_lower = {c.lower(): c for c in VALID_BOROUGH_CODES}
    name_to_code = {
        "manhattan": "M",
        "brooklyn": "B", "bk": "B",
        "bronx": "Bx",
        "queens": "Q", "qns": "Q",
        "staten island": "SI", "richmond": "SI",
    }
    out: list[str] = []
    for v in values:
        key = v.strip().lower()
        if key in code_lower:
            out.append(code_lower[key])
        elif key in name_to_code:
            out.append(name_to_code[key])
        else:
            raise ValueError(
                f"unknown borough {v!r}; expected one of {list(VALID_BOROUGH_CODES)} "
                "or full names (Manhattan, Brooklyn, Bronx, Queens, 'Staten Island')"
            )
    return out


def build(
    out: str,
    *,
    entrance_types: Optional[Iterable[str]] = None,
    divisions: Optional[Iterable[str]] = None,
    boroughs: Optional[Iterable[str]] = None,
    routes: Optional[Iterable[str]] = None,
    buffer_ft: float = 80.0,
    refresh: bool = False,
) -> SubwayBuildResult:
    """Build a subway-entrances curation sub-dataset at ``out``.

    Args:
        out: Output directory.
        entrance_types: Optional list of ``entrance_type`` values
            (e.g. ``['Stair', 'Elevator']``). Validated against the frozen
            MTA vocab.
        divisions: Optional list of division codes (IRT / BMT / IND / SIR /
            IRT/BMT / IND/BMT).
        boroughs: Optional list of borough codes (M / B / Bx / Q / SI) or
            full names (Manhattan / etc.).
        routes: Optional list of route IDs (e.g. ``['L', '4', 'Q']``).
            Matched as whole tokens against the space-separated
            ``daytime_routes`` column.
        buffer_ft: Buffer distance in feet around each entrance point.
            Default 80 (matches the other curation families).
        refresh: When True, ignore the on-disk Socrata cache.
    """
    t0 = time.monotonic()
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "by_source"), exist_ok=True)

    filters: dict[str, list[str]] = {}
    if entrance_types:
        filters["entrance_type"] = validate_entrance_types(list(entrance_types))
    if divisions:
        filters["division"] = _validate_divisions(list(divisions))
    if boroughs:
        filters["borough"] = _validate_boroughs(list(boroughs))
    if routes:
        # Tokens are user-facing IDs ("L", "4", "Q"); only sanity-check shape.
        cleaned = [r.strip() for r in routes if r and r.strip()]
        if not cleaned:
            raise ValueError("routes filter is empty after stripping whitespace")
        filters["route"] = cleaned

    log.info("build: subway-entrances filters = %s", filters or "(none — full pull)")

    cache_path = os.path.join(out, "by_source", "subway_raw.parquet")
    log.info("build: fetching MTA subway entrances")
    r = fetch_subway_entrances(
        entrance_types=filters.get("entrance_type"),
        divisions=filters.get("division"),
        boroughs=filters.get("borough"),
        routes=filters.get("route"),
        cache_path=cache_path,
        refresh=refresh,
    )
    pagination = {
        "subway": {
            "pages": r.pages,
            "page_rows": r.page_rows,
            "total_rows": r.total_rows,
            "truncated_likely": r.truncated_likely,
            "cached": r.cached,
        },
    }

    log.info("build: normalizing %d raw rows", r.total_rows)
    normalized = normalize_subway_entrances(r.df)

    NYC_LAT_MIN, NYC_LAT_MAX = 40.40, 41.00
    NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.60
    in_bbox = normalized.filter(
        pl.col("raw_latitude").is_between(NYC_LAT_MIN, NYC_LAT_MAX)
        & pl.col("raw_longitude").is_between(NYC_LON_MIN, NYC_LON_MAX)
    )
    have_name = in_bbox.filter(
        pl.col("facname").is_not_null() & (pl.col("facname") != "")
    )

    log.info("build: buffering %d entrance points by %.1f ft", have_name.height, buffer_ft)
    publishable = _buffer_points(have_name, buffer_ft=buffer_ft).reset_index(drop=True)

    funnel = {
        "raw_rows": int(r.total_rows),
        "after_normalize": int(normalized.height),
        "after_in_nyc_bbox": int(in_bbox.height),
        "after_has_facname": int(have_name.height),
        "publishable": int(len(publishable)),
    }

    log.info("build: validating %d publishable rows", len(publishable))
    try:
        validation = run_validation(
            publishable, out,
            funnel=funnel, pagination=pagination,
            filters=filters, buffer_ft=buffer_ft,
        )
    except SubwayValidationError as exc:
        manifest_path = os.path.join(out, "manifest.json")
        _write_manifest(manifest_path, {
            "schema_version": SCHEMA_VERSION,
            "built_at": datetime.now(tz=timezone.utc).isoformat(),
            "status": "FATAL",
            "fatal_violations": list(str(exc).split("; ")),
            "filters": filters,
            "buffer_ft": buffer_ft,
            "pagination": pagination,
            "funnel": funnel,
            "git_sha": _git_sha(),
        })
        raise

    entrances_parquet = os.path.join(out, "entrances.parquet")
    entrances_geojson = os.path.join(out, "entrances.geojson")
    coverage_geojson = os.path.join(out, "coverage.geojson")
    manifest_path = os.path.join(out, "manifest.json")

    for p in (entrances_geojson, coverage_geojson):
        if os.path.isfile(p):
            os.remove(p)

    _write_entrances_parquet(publishable, entrances_parquet)
    _write_entrances_geojson(publishable, entrances_geojson)
    _write_coverage_geojson(validation.coverage, coverage_geojson)

    _write_manifest(manifest_path, {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": "OK",
        "filters": filters,
        "buffer_ft": buffer_ft,
        "raw_rows": int(r.total_rows),
        "publishable_rows": int(len(publishable)),
        "unique_stations": validation.metrics.get("unique_stations"),
        "unique_complexes": validation.metrics.get("unique_complexes"),
        "coverage_area_km2": validation.metrics.get("coverage_area_km2"),
        "funnel": funnel,
        "pagination": pagination,
        "columns": list(COMMON_COLUMNS) + ["geom_source", "match_dist_ft", "geom_wkb"],
        "git_sha": _git_sha(),
    })

    elapsed = time.monotonic() - t0
    log.info(
        "build: done — %d entrances in %.1fs (coverage %.3f km²)",
        len(publishable), elapsed,
        validation.metrics.get("coverage_area_km2", 0),
    )

    return SubwayBuildResult(
        output_root=out,
        total_publishable=int(len(publishable)),
        raw_rows=int(r.total_rows),
        funnel=funnel,
        pagination=pagination,
        summary_path=validation.summary_path,
        report_path=validation.report_path,
        entrances_parquet=entrances_parquet,
        entrances_geojson=entrances_geojson,
        coverage_geojson=coverage_geojson,
        manifest_path=manifest_path,
        filters=filters,
        elapsed_s=elapsed,
        validation=validation,
    )
