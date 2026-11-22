"""Top-level orchestration: fetch → normalize → attach geometry → validate → write.

Mirrors :mod:`..facdb.facdb_facilities`. Each NYC outdoor-dining license is
joined to a building polygon via BIN (with nearest-building + point fallback,
shared :mod:`..geom.attach_geometry`) and buffered by ``buffer_ft`` so the
coverage captures the sidewalk / roadway dining setup in front of the
restaurant. Output dir layout:

    <out>/
      by_source/open_restaurants_raw.parquet
      open_restaurants.parquet  — publishable rows with WKB geometry
      open_restaurants.geojson  — same, GeoJSON FeatureCollection
      coverage.geojson          — unary_union of buffered polygons
      manifest.json
      summary.md
      validation_report.parquet

``materialize-cyclomedia`` auto-detects ``open_restaurants.parquet`` (one row
per license, keyed ``uid`` + ``facname``) for per-unit attribution.
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
from .fetch import fetch_open_restaurants
from .license_types import validate_license_types
from .normalize import COMMON_COLUMNS, normalize_open_restaurants
from .validation import OpenRestaurantsValidationError, ValidationResult, run_validation

__all__ = ["build", "OpenRestaurantsBuildResult"]

log = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"

# Canonical borough names accepted on the --borough filter.
VALID_BOROUGHS: tuple[str, ...] = (
    "MANHATTAN", "BRONX", "BROOKLYN", "QUEENS", "STATEN ISLAND",
)
_BOROUGH_ALIASES: dict[str, str] = {
    "MN": "MANHATTAN", "MANHATTAN": "MANHATTAN",
    "BX": "BRONX", "BRONX": "BRONX",
    "BK": "BROOKLYN", "BROOKLYN": "BROOKLYN",
    "QN": "QUEENS", "QUEENS": "QUEENS",
    "SI": "STATEN ISLAND", "STATEN ISLAND": "STATEN ISLAND", "RICHMOND": "STATEN ISLAND",
}


@dataclass
class OpenRestaurantsBuildResult:
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


def _validate_boroughs(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        key = v.strip().upper()
        if key in _BOROUGH_ALIASES:
            out.append(_BOROUGH_ALIASES[key])
        else:
            raise ValueError(
                f"unknown borough {v!r}; expected one of {list(VALID_BOROUGHS)} "
                "or aliases MN/BX/BK/QN/SI"
            )
    return out


def _write_restaurants_parquet(gdf: gpd.GeoDataFrame, path: str) -> None:
    from shapely.wkb import dumps as wkb_dumps
    wkbs = [wkb_dumps(g) if g is not None and not g.is_empty else None for g in gdf.geometry]
    df = pl.from_pandas(gdf.drop(columns=["geometry"]))
    df = df.with_columns(pl.Series("geom_wkb", wkbs, dtype=pl.Binary))
    df.write_parquet(path)


def _write_restaurants_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
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
         "properties": {"name": "open_restaurants_coverage", "part": i},
         "geometry": mapping(p)}
        for i, p in enumerate(polys)
    ]
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def _write_manifest(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def build(
    out: str,
    *,
    license_types: Optional[Iterable[str]] = None,
    boroughs: Optional[Iterable[str]] = None,
    buffer_ft: float = 80.0,
    buildings_path: str = DEFAULT_BUILDINGS_PATH,
    refresh: bool = False,
    bin_match_warn_threshold: float = 0.85,
    nearest_max_ft: float = DEFAULT_NEAREST_MAX_FT,
) -> OpenRestaurantsBuildResult:
    """Build an open-restaurants curation sub-dataset at ``out``.

    Passing all ``None`` pulls every issued license (~1.3k rows). Each filter
    accepts multiple values (OR within a filter, AND across filters).
    ``license_types`` is validated against the frozen vocab (Sidewalk/Roadway);
    typos raise :class:`.license_types.UnknownLicenseTypeError`.
    """
    t0 = time.monotonic()
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "by_source"), exist_ok=True)

    # ---- Validate / canonicalize filters --------------------------------
    filters: dict[str, list[str]] = {}
    if license_types:
        filters["license_type"] = validate_license_types(list(license_types))
    if boroughs:
        filters["borough"] = _validate_boroughs(list(boroughs))

    log.info("build: open-restaurants filters = %s", filters or "(none — full pull)")

    # ---- Fetch -----------------------------------------------------------
    cache_path = os.path.join(out, "by_source", "open_restaurants_raw.parquet")
    log.info("build: fetching Open Restaurants (Dining Out NYC) licenses")
    r = fetch_open_restaurants(
        license_types=filters.get("license_type"),
        boroughs=filters.get("borough"),
        cache_path=cache_path,
        refresh=refresh,
    )
    pagination = {
        "open_restaurants": {
            "pages": r.pages,
            "page_rows": r.page_rows,
            "total_rows": r.total_rows,
            "truncated_likely": r.truncated_likely,
            "cached": r.cached,
        },
    }

    # ---- Normalize -------------------------------------------------------
    log.info("build: normalizing %d rows", r.total_rows)
    normalized = normalize_open_restaurants(r.df)

    have_name = normalized.filter(
        pl.col("facname").is_not_null() & (pl.col("facname") != "")
    )

    # Drop rows that can't produce a usable NYC polygon: a handful of licenses
    # carry bad geocoding (no BIN/BBL plus a lat/lon out on Long Island or in
    # Massachusetts Bay). Keep a row only if it has a real BIN OR a lat/lon
    # plausibly inside NYC — otherwise it would fail the NYC-bbox fatal check.
    NYC_LAT_MIN, NYC_LAT_MAX = 40.40, 41.00
    NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.60
    in_nyc = have_name.filter(
        (pl.col("bin").is_not_null() & (pl.col("bin") != "") & (pl.col("bin") != "0"))
        | (
            pl.col("raw_latitude").is_between(NYC_LAT_MIN, NYC_LAT_MAX)
            & pl.col("raw_longitude").is_between(NYC_LON_MIN, NYC_LON_MAX)
        )
    )

    # ---- Attach geometry -------------------------------------------------
    log.info(
        "build: attaching geometry (buffer=%.1f ft, nearest_max=%.1f ft)",
        buffer_ft, nearest_max_ft,
    )
    with_geom = attach_geometry(
        in_nyc,
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
        "after_normalize": int(normalized.height),
        "after_has_facname": int(have_name.height),
        "after_in_nyc_or_bin": int(in_nyc.height),
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
    except OpenRestaurantsValidationError as exc:
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
    restaurants_parquet = os.path.join(out, "open_restaurants.parquet")
    restaurants_geojson = os.path.join(out, "open_restaurants.geojson")
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
        "raw_rows": int(r.total_rows),
        "publishable_rows": int(len(publishable)),
        "geom_source_counts": {str(k): int(v) for k, v in geom_source_counts.items()},
        "funnel": funnel,
        "pagination": pagination,
        "polygon_match_rate_overall_pct": validation.metrics.get("polygon_match_rate_overall_pct"),
        "bin_exact_rate_overall_pct": validation.metrics.get("bin_exact_rate_overall_pct"),
        "nearest_polygon_rate_overall_pct": validation.metrics.get("nearest_polygon_rate_overall_pct"),
        "coverage_area_km2": validation.metrics.get("coverage_area_km2"),
        "columns": list(COMMON_COLUMNS) + ["geom_source", "match_dist_ft", "geom_wkb"],
        "git_sha": _git_sha(),
    })

    elapsed = time.monotonic() - t0
    log.info(
        "build: done — %d publishable licenses in %.1fs "
        "(polygon %.2f%% = bin_exact %.2f%% + nearest %.2f%%)",
        len(publishable), elapsed,
        validation.metrics.get("polygon_match_rate_overall_pct", float("nan")),
        validation.metrics.get("bin_exact_rate_overall_pct", float("nan")),
        validation.metrics.get("nearest_polygon_rate_overall_pct", float("nan")),
    )

    return OpenRestaurantsBuildResult(
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
