"""Top-level orchestration: fetch → normalize → buffer → validate → write.

Produces a versioned sub-dataset dir under ``--out``:

- ``permits.parquet``, ``permits.geojson`` — one row/feature per publishable permit.
- ``coverage.geojson`` — ``unary_union`` of every buffered polygon.
- ``by_source/dob_now_raw.parquet``, ``bis_raw.parquet`` — raw Socrata responses.
- ``manifest.json`` — schema version, cutoff, row counts, git SHA, funnel.
- ``validation_report.parquet``, ``summary.md`` — from validation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
import datetime as _dt
from typing import Any, Optional

import geopandas as gpd
import polars as pl
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from .buffer import (
    DEFAULT_BUILDINGS_PATH,
    DEFAULT_NEAREST_MAX_FT,
    WGS84_CRS,
    attach_geometry,
)
from .fetch import fetch_bis, fetch_dob_now
from .normalize import COMMON_COLUMNS, normalize_to_common
from .validation import PermitValidationError, ValidationResult, run_validation

__all__ = ["build", "PermitBuildResult"]

log = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"


@dataclass
class PermitBuildResult:
    output_root: str
    cutoff: str
    buffer_ft: float
    total_publishable: int
    dob_now_rows: int
    bis_rows: int
    funnel: dict[str, int]
    pagination: dict[str, Any]
    summary_path: str
    report_path: str
    permits_parquet: str
    permits_geojson: str
    coverage_geojson: str
    manifest_path: str
    elapsed_s: float = 0.0
    validation: Optional[ValidationResult] = None
    skipped_fatal: list[str] = field(default_factory=list)
    since: Optional[str] = None


def _parse_cutoff_to_datetime(cutoff: str) -> _dt.datetime:
    s = cutoff.strip()
    if "T" in s:
        return _dt.datetime.fromisoformat(s)
    return _dt.datetime.strptime(s, "%Y-%m-%d").replace(hour=23, minute=59, second=59)


def _parse_since_to_datetime(since: str) -> _dt.datetime:
    s = since.strip()
    if "T" in s:
        return _dt.datetime.fromisoformat(s)
    return _dt.datetime.strptime(s, "%Y-%m-%d")


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


def _write_permits_parquet(gdf: gpd.GeoDataFrame, path: str) -> None:
    """Serialize the full permit frame (WKB geometry) as parquet."""
    from shapely.wkb import dumps as wkb_dumps

    wkbs = [wkb_dumps(g) if g is not None and not g.is_empty else None for g in gdf.geometry]
    df = pl.from_pandas(gdf.drop(columns=["geometry"]))
    df = df.with_columns(pl.Series("geom_wkb", wkbs, dtype=pl.Binary))
    df.write_parquet(path)


def _write_permits_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
    """GeoJSON FeatureCollection, one feature per permit."""
    # Use GeoPandas' native writer (handles property serialization cleanly) —
    # then post-process dates, since GeoJSON doesn't know datetimes.
    writable = gdf.copy()
    for c in ("issue_date", "expiration_date", "signoff_date"):
        if c in writable.columns:
            writable[c] = writable[c].apply(
                lambda v: v.isoformat() if v is not None and not _is_na(v) else None
            )
    writable.to_file(path, driver="GeoJSON")


def _is_na(v: Any) -> bool:
    try:
        import pandas as pd
        return bool(pd.isna(v))
    except Exception:
        return v is None


def _write_coverage_geojson(coverage: BaseGeometry, path: str) -> None:
    """Emit ``coverage`` as a FeatureCollection of per-Polygon features.

    The input is typically a single dissolved ``MultiPolygon`` from
    ``unary_union``. Splitting into per-Polygon features keeps each GeoJSON
    feature under pyogrio's default size limit while preserving semantics —
    downstream consumers (like :class:`CyclomediaCatalog`) re-dissolve via
    ``unary_union`` on the way in, so the fragmentation is invisible.
    """
    from shapely.geometry import MultiPolygon as _MP
    if isinstance(coverage, _MP):
        polys = list(coverage.geoms)
    elif coverage is None or coverage.is_empty:
        polys = []
    else:
        polys = [coverage]
    features = [
        {
            "type": "Feature",
            "properties": {"name": "scaffolding_permit_coverage", "part": i},
            "geometry": mapping(p),
        }
        for i, p in enumerate(polys)
    ]
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(fc, f)


def _write_manifest(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def build(
    out: str,
    *,
    cutoff: str = "2025-12-31",
    since: Optional[str] = None,
    buffer_ft: float = 80.0,
    buildings_path: str = DEFAULT_BUILDINGS_PATH,
    refresh: bool = False,
    bin_match_warn_threshold: float = 0.85,
    nearest_max_ft: float = DEFAULT_NEAREST_MAX_FT,
) -> PermitBuildResult:
    """Build a scaffolding-permits sub-dataset at ``out``.

    Args:
        out: Output directory; created if missing.
        cutoff: ``YYYY-MM-DD`` — permits with ``issue_date`` past this are excluded.
        since: Optional ``YYYY-MM-DD`` lower bound. ``None`` → no lower bound
            (BIS goes back to the 1990s, DOB NOW starts ~2017).
        buffer_ft: Buffer distance (feet) applied to each permit's geometry.
        buildings_path: ``nyc_buildings.parquet`` location.
        refresh: If True, ignore Socrata caches and re-fetch.
        bin_match_warn_threshold: Overall polygon-match rate below this fires a warn.
    """
    t0 = time.monotonic()
    out = os.path.abspath(out)
    os.makedirs(os.path.join(out, "by_source"), exist_ok=True)

    # ---------------------------------------------------------------- fetch
    dob_now_cache = os.path.join(out, "by_source", "dob_now_raw.parquet")
    bis_cache = os.path.join(out, "by_source", "bis_raw.parquet")

    log.info("build: fetching DOB NOW (since=%s cutoff=%s)", since, cutoff)
    dob_now = fetch_dob_now(cutoff=cutoff, since=since, cache_path=dob_now_cache, refresh=refresh)
    log.info("build: fetching BIS (since=%s cutoff=%s)", since, cutoff)
    bis = fetch_bis(cutoff=cutoff, since=since, cache_path=bis_cache, refresh=refresh)

    pagination = {
        "dob_now": {
            "pages": dob_now.pages,
            "page_rows": dob_now.page_rows,
            "total_rows": dob_now.total_rows,
            "truncated_likely": dob_now.truncated_likely,
            "cached": dob_now.cached,
        },
        "bis": {
            "pages": bis.pages,
            "page_rows": bis.page_rows,
            "total_rows": bis.total_rows,
            "truncated_likely": bis.truncated_likely,
            "cached": bis.cached,
        },
    }

    # ---------------------------------------------------------------- normalize
    log.info("build: normalizing %d dob_now + %d bis", dob_now.total_rows, bis.total_rows)
    normalized = normalize_to_common(dob_now.df, bis.df)
    n_after_normalize = normalized.height

    # Client-side date clip. BIS's issuance_date is a plain-text MM/DD/YYYY
    # column so the server filter can only prune by year; enforce the exact
    # [since, cutoff] bounds here after parsing. DOB NOW is already
    # server-filtered correctly but we re-apply for consistency / defense.
    cutoff_dt = _parse_cutoff_to_datetime(cutoff)
    since_dt = _parse_since_to_datetime(since) if since is not None else None
    date_pred = pl.col("issue_date").is_not_null() & (pl.col("issue_date") <= cutoff_dt)
    if since_dt is not None:
        date_pred = date_pred & (pl.col("issue_date") >= since_dt)
    date_clipped = normalized.filter(date_pred)
    valid_date = date_clipped
    valid_stype = valid_date.filter(pl.col("scaffold_type").is_not_null())

    # ---------------------------------------------------------------- geometry
    log.info(
        "build: attaching geometry (buffer=%.1f ft, nearest_max=%.1f ft)",
        buffer_ft, nearest_max_ft,
    )
    with_geom = attach_geometry(
        valid_stype,
        buffer_ft=buffer_ft,
        buildings_path=buildings_path,
        nearest_max_ft=nearest_max_ft,
    )
    n_after_geom = len(with_geom)

    # Drop unpublishable (geom_source == "none") rows. Tracked in the funnel.
    publishable = with_geom[
        with_geom["geom_source"].isin(["bin_polygon", "nearest_polygon", "point"])
    ].copy()
    publishable = publishable.reset_index(drop=True)

    funnel = {
        "dob_now_raw": int(dob_now.total_rows),
        "bis_raw": int(bis.total_rows),
        "after_normalize": int(n_after_normalize),
        "after_date_clip": int(date_clipped.height),
        "after_valid_scaffold_type": int(valid_stype.height),
        "after_attach_geometry": int(n_after_geom),
        "publishable": int(len(publishable)),
        # Informational — surfaced in validation:
        "dob_now_dropped_null_first_permit_date": 0,  # server-filtered; see note
    }

    # ---------------------------------------------------------------- validate
    log.info("build: validating %d publishable rows", len(publishable))
    skipped_fatal: list[str] = []
    validation: Optional[ValidationResult] = None
    try:
        validation = run_validation(
            publishable,
            out,
            funnel=funnel,
            pagination=pagination,
            cutoff=cutoff,
            since=since,
            buffer_ft=buffer_ft,
            bin_match_warn_threshold=bin_match_warn_threshold,
        )
    except PermitValidationError as exc:
        skipped_fatal = list(str(exc).split("; "))
        log.error("build: fatal validation errors — NOT writing permits outputs")
        # Re-raise after writing the diagnostic manifest at the bottom.
        manifest_path = os.path.join(out, "manifest.json")
        _write_manifest(manifest_path, {
            "schema_version": SCHEMA_VERSION,
            "built_at": datetime.now(tz=timezone.utc).isoformat(),
            "cutoff": cutoff,
            "since": since,
            "buffer_ft": buffer_ft,
            "status": "FATAL",
            "fatal_violations": skipped_fatal,
            "pagination": pagination,
            "funnel": funnel,
            "git_sha": _git_sha(),
        })
        raise

    # ---------------------------------------------------------------- write outputs
    permits_parquet = os.path.join(out, "permits.parquet")
    permits_geojson = os.path.join(out, "permits.geojson")
    coverage_geojson = os.path.join(out, "coverage.geojson")
    manifest_path = os.path.join(out, "manifest.json")

    # Remove any stale geojson before writing — GeoPandas' GeoJSON driver appends.
    for p in (permits_geojson, coverage_geojson):
        if os.path.isfile(p):
            os.remove(p)

    _write_permits_parquet(publishable, permits_parquet)
    _write_permits_geojson(publishable, permits_geojson)
    _write_coverage_geojson(validation.coverage, coverage_geojson)

    geom_source_counts = (
        publishable["geom_source"].value_counts().to_dict()
    )

    _write_manifest(manifest_path, {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "cutoff": cutoff,
        "since": since,
        "buffer_ft": buffer_ft,
        "nearest_max_ft": nearest_max_ft,
        "status": "OK",
        "source_row_counts": {
            "dob_now": int(dob_now.total_rows),
            "bis": int(bis.total_rows),
        },
        "publishable_rows": int(len(publishable)),
        "geom_source_counts": {str(k): int(v) for k, v in geom_source_counts.items()},
        "funnel": funnel,
        "pagination": pagination,
        "polygon_match_rate_overall_pct": validation.metrics.get("polygon_match_rate_overall_pct"),
        "bin_exact_rate_overall_pct": validation.metrics.get("bin_exact_rate_overall_pct"),
        "nearest_polygon_rate_overall_pct": validation.metrics.get("nearest_polygon_rate_overall_pct"),
        "coverage_area_km2": validation.metrics.get("coverage_area_km2"),
        "cross_source_bin_overlap": validation.metrics.get("cross_source_bin_overlap"),
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

    return PermitBuildResult(
        output_root=out,
        cutoff=cutoff,
        since=since,
        buffer_ft=buffer_ft,
        total_publishable=int(len(publishable)),
        dob_now_rows=int(dob_now.total_rows),
        bis_rows=int(bis.total_rows),
        funnel=funnel,
        pagination=pagination,
        summary_path=validation.summary_path,
        report_path=validation.report_path,
        permits_parquet=permits_parquet,
        permits_geojson=permits_geojson,
        coverage_geojson=coverage_geojson,
        manifest_path=manifest_path,
        elapsed_s=elapsed,
        validation=validation,
        skipped_fatal=skipped_fatal,
    )
