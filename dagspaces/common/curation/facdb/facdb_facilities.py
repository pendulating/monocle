"""Top-level orchestration: fetch → normalize → attach geometry → validate → write.

Mirrors :mod:`..permits.scaffolding_permits`. Output dir layout:

    <out>/
      by_source/facdb_raw.parquet
      facilities.parquet        — publishable rows with WKB geometry
      facilities.geojson        — same, GeoJSON FeatureCollection
      coverage.geojson          — unary_union of buffered polygons (per-polygon features)
      manifest.json
      summary.md
      validation_report.parquet
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
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
from .categorization import HIERARCHY_LEVELS, validate_filter_values
from .fetch import fetch_facdb
from .normalize import COMMON_COLUMNS, normalize_facdb
from .validation import FacdbValidationError, ValidationResult, run_validation

__all__ = ["build", "FacdbBuildResult"]

log = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"


@dataclass
class FacdbBuildResult:
    output_root: str
    total_publishable: int
    raw_rows: int
    funnel: dict[str, int]
    pagination: dict[str, Any]
    summary_path: str
    report_path: str
    facilities_parquet: str
    facilities_geojson: str
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


def _write_facilities_parquet(gdf: gpd.GeoDataFrame, path: str) -> None:
    from shapely.wkb import dumps as wkb_dumps
    wkbs = [wkb_dumps(g) if g is not None and not g.is_empty else None for g in gdf.geometry]
    df = pl.from_pandas(gdf.drop(columns=["geometry"]))
    df = df.with_columns(pl.Series("geom_wkb", wkbs, dtype=pl.Binary))
    df.write_parquet(path)


def _write_facilities_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
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
         "properties": {"name": "facdb_coverage", "part": i},
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
    facdomain: Optional[Iterable[str]] = None,
    facgroup: Optional[Iterable[str]] = None,
    facsubgrp: Optional[Iterable[str]] = None,
    factype: Optional[Iterable[str]] = None,
    buffer_ft: float = 80.0,
    buildings_path: str = DEFAULT_BUILDINGS_PATH,
    refresh: bool = False,
    bin_match_warn_threshold: float = 0.75,
    nearest_max_ft: float = DEFAULT_NEAREST_MAX_FT,
) -> FacdbBuildResult:
    """Build a FacDB curation sub-dataset at ``out``.

    At least one filter level is usually provided; passing all ``None``
    pulls the full FacDB (~34.7k rows). Each level accepts multiple values
    (logical OR within the level, AND across levels).

    Filter values are validated against the frozen FacDB dictionary at
    :mod:`.categorization` — typos raise :class:`.UnknownCategoryError`.
    """
    t0 = time.monotonic()
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "by_source"), exist_ok=True)

    # ---- Validate filter values against the dictionary ------------------
    filters: dict[str, list[str]] = {}
    if facdomain:
        filters["facdomain"] = validate_filter_values("facdomain", list(facdomain))
    if facgroup:
        filters["facgroup"] = validate_filter_values("facgroup", list(facgroup))
    if facsubgrp:
        filters["facsubgrp"] = validate_filter_values("facsubgrp", list(facsubgrp))
    if factype:
        filters["factype"] = validate_filter_values("factype", list(factype))

    log.info("build: FacDB filters = %s", filters or "(none — full pull)")

    # ---- Fetch -----------------------------------------------------------
    cache_path = os.path.join(out, "by_source", "facdb_raw.parquet")
    log.info("build: fetching FacDB")
    r = fetch_facdb(
        facdomain=filters.get("facdomain"),
        facgroup=filters.get("facgroup"),
        facsubgrp=filters.get("facsubgrp"),
        factype=filters.get("factype"),
        cache_path=cache_path,
        refresh=refresh,
    )
    pagination = {
        "facdb": {
            "pages": r.pages,
            "page_rows": r.page_rows,
            "total_rows": r.total_rows,
            "truncated_likely": r.truncated_likely,
            "cached": r.cached,
        },
    }

    # ---- Normalize -------------------------------------------------------
    log.info("build: normalizing %d rows", r.total_rows)
    normalized = normalize_facdb(r.df)

    # FacDB has a small number of rows with bad geocoding — most commonly
    # `bin='0'` plus `latitude=longitude=0.0` (seen e.g. for some library
    # branches). Keep the row only if it has a non-zero BIN OR a lat/lon
    # that plausibly sits in NYC. Anything else can't produce a usable
    # polygon and would fail the NYC-bbox fatal check downstream.
    NYC_LAT_MIN, NYC_LAT_MAX = 40.40, 41.00
    NYC_LON_MIN, NYC_LON_MAX = -74.30, -73.60
    with_any_geom = normalized.filter(
        (pl.col("bin").is_not_null() & (pl.col("bin") != "") & (pl.col("bin") != "0"))
        | (
            pl.col("raw_latitude").is_between(NYC_LAT_MIN, NYC_LAT_MAX)
            & pl.col("raw_longitude").is_between(NYC_LON_MIN, NYC_LON_MAX)
        )
    )

    # Drop rows with null facdomain (fatal invariant; surface here so we
    # don't break the geom attach by including them).
    have_domain = with_any_geom.filter(
        pl.col("facdomain").is_not_null() & (pl.col("facdomain") != "")
    )

    # ---- Attach geometry -------------------------------------------------
    log.info(
        "build: attaching geometry (buffer=%.1f ft, nearest_max=%.1f ft)",
        buffer_ft, nearest_max_ft,
    )
    with_geom = attach_geometry(
        have_domain,
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
        "after_has_geom_source": int(with_any_geom.height),
        "after_has_facdomain": int(have_domain.height),
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
    except FacdbValidationError as exc:
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
    facilities_parquet = os.path.join(out, "facilities.parquet")
    facilities_geojson = os.path.join(out, "facilities.geojson")
    coverage_geojson = os.path.join(out, "coverage.geojson")
    manifest_path = os.path.join(out, "manifest.json")

    for p in (facilities_geojson, coverage_geojson):
        if os.path.isfile(p):
            os.remove(p)

    _write_facilities_parquet(publishable, facilities_parquet)
    _write_facilities_geojson(publishable, facilities_geojson)
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
        "build: done — %d publishable rows in %.1fs "
        "(polygon %.2f%% = bin_exact %.2f%% + nearest %.2f%%)",
        len(publishable), elapsed,
        validation.metrics.get("polygon_match_rate_overall_pct", float("nan")),
        validation.metrics.get("bin_exact_rate_overall_pct", float("nan")),
        validation.metrics.get("nearest_polygon_rate_overall_pct", float("nan")),
    )

    return FacdbBuildResult(
        output_root=out,
        total_publishable=int(len(publishable)),
        raw_rows=int(r.total_rows),
        funnel=funnel,
        pagination=pagination,
        summary_path=validation.summary_path,
        report_path=validation.report_path,
        facilities_parquet=facilities_parquet,
        facilities_geojson=facilities_geojson,
        coverage_geojson=coverage_geojson,
        manifest_path=manifest_path,
        filters=filters,
        elapsed_s=elapsed,
        validation=validation,
    )
