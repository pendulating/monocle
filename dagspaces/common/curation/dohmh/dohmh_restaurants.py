"""Top-level orchestration: fetch → normalize → attach geometry → validate → write.

Mirrors :mod:`..facdb.facdb_facilities`, with one important difference: the
output is **inspection-level**, not restaurant-level. Each row is one
``(camis, inspection_date)`` pair (violations within an inspection are
collapsed; the most-Critical violation row wins). Multi-year inspection
history is preserved on purpose — the user opts into camis-level dedup
explicitly via :mod:`.aggregate` (CLI: ``aggregate-restaurants``).

Output dir layout:

    <out>/
      by_source/dohmh_raw.parquet
      restaurants.parquet       — inspection-level rows, with WKB geometry
      restaurants.geojson       — same, GeoJSON FeatureCollection
      coverage.geojson          — unary_union of buffered polygons (per-polygon features)
      manifest.json
      summary.md
      validation_report.parquet

After running ``aggregate-restaurants``, a sibling ``restaurants_aggregated.parquet``
is written. The downstream ``materialize-cyclomedia`` tool **prefers** the
aggregated parquet (each ``unit_uid`` is unique → clean spatial join) and
errors out if only the inspection-level one is present.
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
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from ..geom import (
    DEFAULT_BUILDINGS_PATH,
    DEFAULT_NEAREST_MAX_FT,
    attach_geometry,
)
from .cuisines import validate_cuisines
from .fetch import fetch_dohmh
from .normalize import COMMON_COLUMNS, normalize_dohmh
from .validation import DohmhValidationError, ValidationResult, run_validation

__all__ = ["build", "DohmhBuildResult"]

log = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"

VALID_BOROUGHS: tuple[str, ...] = (
    "Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island",
)


@dataclass
class DohmhBuildResult:
    output_root: str
    total_publishable: int
    raw_rows: int
    funnel: dict[str, int]
    pagination: dict[str, Any]
    summary_path: str
    report_path: str
    restaurants_parquet: str
    restaurants_geojson: str
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


def _write_restaurants_parquet(gdf: gpd.GeoDataFrame, path: str) -> None:
    from shapely.wkb import dumps as wkb_dumps
    wkbs = [wkb_dumps(g) if g is not None and not g.is_empty else None for g in gdf.geometry]
    df = pl.from_pandas(gdf.drop(columns=["geometry"]))
    df = df.with_columns(pl.Series("geom_wkb", wkbs, dtype=pl.Binary))
    df.write_parquet(path)


def _write_restaurants_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
    # GeoJSON can't carry Python datetime values directly; cast to ISO strings.
    out = gdf.copy()
    for col in ("inspection_date", "record_date", "grade_date"):
        if col in out.columns:
            out[col] = out[col].astype("string")
    out.to_file(path, driver="GeoJSON")


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
         "properties": {"name": "dohmh_coverage", "part": i},
         "geometry": mapping(p)}
        for i, p in enumerate(polys)
    ]
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def _write_manifest(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _validate_boroughs(values: list[str]) -> list[str]:
    """Canonicalize borough filter against DOHMH's own ``boro`` column.

    DOHMH stores boroughs as title-case strings (``"Manhattan"``, etc.) and
    uses ``"0"`` for unknown. Here we accept common synonyms (``MN``, ``BK``,
    full uppercase) and emit the title-case forms for the SoQL clause.
    """
    canonical_lower = {b.lower(): b for b in VALID_BOROUGHS}
    aliases = {
        "mn": "Manhattan", "ny": "Manhattan",
        "bx": "Bronx",
        "bk": "Brooklyn", "bklyn": "Brooklyn",
        "qn": "Queens", "qns": "Queens",
        "si": "Staten Island", "sit": "Staten Island", "richmond": "Staten Island",
    }
    out: list[str] = []
    for v in values:
        key = v.strip().lower()
        if key in canonical_lower:
            out.append(canonical_lower[key])
        elif key in aliases:
            out.append(aliases[key])
        else:
            raise ValueError(
                f"unknown borough {v!r}; expected one of "
                f"{list(VALID_BOROUGHS)} (or aliases: MN/BX/BK/QN/SI)"
            )
    return out


def build(
    out: str,
    *,
    cuisines: Optional[Iterable[str]] = None,
    boroughs: Optional[Iterable[str]] = None,
    buffer_ft: float = 80.0,
    buildings_path: str = DEFAULT_BUILDINGS_PATH,
    refresh: bool = False,
    bin_match_warn_threshold: float = 0.85,
    nearest_max_ft: float = DEFAULT_NEAREST_MAX_FT,
    drop_placeholder_only: bool = False,
) -> DohmhBuildResult:
    """Build a DOHMH restaurant curation sub-dataset at ``out``.

    Args:
        out: Output directory.
        cuisines: Optional list of ``cuisine_description`` values. Validated
            case-insensitively against the frozen DOHMH vocab in
            :mod:`.cuisines` — typos raise :class:`UnknownCuisineError`.
        boroughs: Optional list of borough names / aliases (Manhattan / MN /
            Bronx / BX / etc.).
        drop_placeholder_only: When True, drop CAMIS that only have
            placeholder inspection rows (registered but never inspected).
            Default False — we want every restaurant in the output by
            default since this curation is "all NYC restaurants" by design.
    """
    t0 = time.monotonic()
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "by_source"), exist_ok=True)

    # ---- Validate filter values -----------------------------------------
    filters: dict[str, list[str]] = {}
    if cuisines:
        filters["cuisine_description"] = validate_cuisines(list(cuisines))
    if boroughs:
        filters["boro"] = _validate_boroughs(list(boroughs))

    log.info("build: DOHMH filters = %s", filters or "(none — full pull)")

    # ---- Fetch -----------------------------------------------------------
    cache_path = os.path.join(out, "by_source", "dohmh_raw.parquet")
    log.info("build: fetching DOHMH inspection rows")
    r = fetch_dohmh(
        cuisines=filters.get("cuisine_description"),
        boroughs=filters.get("boro"),
        cache_path=cache_path,
        refresh=refresh,
    )
    pagination = {
        "dohmh": {
            "pages": r.pages,
            "page_rows": r.page_rows,
            "total_rows": r.total_rows,
            "truncated_likely": r.truncated_likely,
            "cached": r.cached,
        },
    }

    # ---- Normalize: collapse violation rows, keep all inspections -------
    log.info("build: normalizing %d raw inspection rows", r.total_rows)
    normalized = normalize_dohmh(r.df)

    # Optionally drop placeholder rows (CAMIS that are registered but never
    # actually inspected). Multi-inspection CAMIS may have a mix; only the
    # placeholder rows are dropped here, not the whole CAMIS.
    if drop_placeholder_only and normalized.height > 0:
        before = normalized.height
        normalized = normalized.filter(~pl.col("is_placeholder_inspection"))
        log.info("build: dropped %d placeholder rows", before - normalized.height)

    # Drop rows that have no plausible geocoding source.
    NYC_LAT_MIN, NYC_LAT_MAX = 40.40, 41.00
    NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.60
    with_any_geom = normalized.filter(
        (pl.col("bin").is_not_null() & (pl.col("bin") != "") & (pl.col("bin") != "0"))
        | (
            pl.col("raw_latitude").is_between(NYC_LAT_MIN, NYC_LAT_MAX)
            & pl.col("raw_longitude").is_between(NYC_LON_MIN, NYC_LON_MAX)
        )
    )

    # Drop rows with null/empty facname (every restaurant must have a DBA).
    have_name = with_any_geom.filter(
        pl.col("facname").is_not_null() & (pl.col("facname") != "")
    )

    # ---- Attach geometry -------------------------------------------------
    log.info(
        "build: attaching geometry (buffer=%.1f ft, nearest_max=%.1f ft)",
        buffer_ft, nearest_max_ft,
    )
    with_geom = attach_geometry(
        have_name,
        buffer_ft=buffer_ft,
        buildings_path=buildings_path,
        nearest_max_ft=nearest_max_ft,
        id_col="permit_id",
        bin_col="bin",
        lat_col="raw_latitude",
        lon_col="raw_longitude",
    )
    publishable = with_geom[
        with_geom["geom_source"].isin(["bin_polygon", "nearest_polygon", "point"])
    ].copy().reset_index(drop=True)

    funnel = {
        "raw_rows": int(r.total_rows),
        "after_normalize_inspections": int(normalized.height),
        "after_has_geom_source": int(with_any_geom.height),
        "after_has_facname": int(have_name.height),
        "after_attach_geometry": int(len(with_geom)),
        "publishable": int(len(publishable)),
    }

    # ---- Validate --------------------------------------------------------
    log.info("build: validating %d publishable rows", len(publishable))
    try:
        validation = run_validation(
            publishable, out,
            funnel=funnel, pagination=pagination,
            filters=filters, buffer_ft=buffer_ft,
            bin_match_warn_threshold=bin_match_warn_threshold,
        )
    except DohmhValidationError as exc:
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

    # ---- Write outputs ---------------------------------------------------
    restaurants_parquet = os.path.join(out, "restaurants.parquet")
    restaurants_geojson = os.path.join(out, "restaurants.geojson")
    coverage_geojson = os.path.join(out, "coverage.geojson")
    manifest_path = os.path.join(out, "manifest.json")

    for p in (restaurants_geojson, coverage_geojson):
        if os.path.isfile(p):
            os.remove(p)

    _write_restaurants_parquet(publishable, restaurants_parquet)
    _write_restaurants_geojson(publishable, restaurants_geojson)
    _write_coverage_geojson(validation.coverage, coverage_geojson)

    geom_source_counts = publishable["geom_source"].value_counts().to_dict()

    _write_manifest(manifest_path, {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "status": "OK",
        "filters": filters,
        "buffer_ft": buffer_ft,
        "nearest_max_ft": nearest_max_ft,
        "drop_placeholder_only": bool(drop_placeholder_only),
        "raw_rows": int(r.total_rows),
        "publishable_rows": int(len(publishable)),
        "geom_source_counts": {str(k): int(v) for k, v in geom_source_counts.items()},
        "funnel": funnel,
        "pagination": pagination,
        "unique_camis": validation.metrics.get("unique_camis"),
        "polygon_match_rate_overall_pct": validation.metrics.get("polygon_match_rate_overall_pct"),
        "bin_exact_rate_overall_pct": validation.metrics.get("bin_exact_rate_overall_pct"),
        "nearest_polygon_rate_overall_pct": validation.metrics.get("nearest_polygon_rate_overall_pct"),
        "coverage_area_km2": validation.metrics.get("coverage_area_km2"),
        "placeholder_inspections": validation.metrics.get("placeholder_inspections"),
        "camis_only_placeholder": validation.metrics.get("camis_only_placeholder"),
        "columns": list(COMMON_COLUMNS) + ["geom_source", "match_dist_ft", "geom_wkb"],
        "git_sha": _git_sha(),
    })

    elapsed = time.monotonic() - t0
    log.info(
        "build: done — %d publishable rows in %.1fs "
        "(polygon %.2f%% = bin_exact %.2f%% + nearest %.2f%%)",
        len(publishable), elapsed,
        validation.metrics.get("polygon_match_rate_overall_pct", float("nan")),
        validation.metrics.get("bin_exact_rate_overall_pct", float("nan")),
        validation.metrics.get("nearest_polygon_rate_overall_pct", float("nan")),
    )

    return DohmhBuildResult(
        output_root=out,
        total_publishable=int(len(publishable)),
        raw_rows=int(r.total_rows),
        funnel=funnel,
        pagination=pagination,
        summary_path=validation.summary_path,
        report_path=validation.report_path,
        restaurants_parquet=restaurants_parquet,
        restaurants_geojson=restaurants_geojson,
        coverage_geojson=coverage_geojson,
        manifest_path=manifest_path,
        filters=filters,
        elapsed_s=elapsed,
        validation=validation,
    )
