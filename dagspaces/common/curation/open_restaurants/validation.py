"""Post-build sanity checks for the open-restaurants curation.

Mirrors :mod:`..facdb.validation` (BIN-polygon geometry with nearest/point
fallback). Produces ``validation_report.parquet`` + ``summary.md`` per build.
Raises :class:`OpenRestaurantsValidationError` on fatal checks.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import polars as pl
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

__all__ = [
    "run_validation",
    "OpenRestaurantsValidationError",
    "ValidationResult",
    "NYC_BBOX_WGS84",
]

log = logging.getLogger(__name__)

NYC_BBOX_WGS84 = {"lon_min": -74.30, "lon_max": -73.60, "lat_min": 40.40, "lat_max": 41.00}
NYC_LAND_AREA_KM2 = 778.2


class OpenRestaurantsValidationError(AssertionError):
    """Raised when a fatal invariant is violated."""


@dataclass
class ValidationResult:
    summary_path: str
    report_path: str
    coverage: BaseGeometry
    fatal_violations: list[str]
    warn_notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _pct(n: int, d: int) -> float:
    return float("nan") if d == 0 else 100.0 * n / d


def _nyc_bbox_polygon() -> Polygon:
    b = NYC_BBOX_WGS84
    return Polygon([
        (b["lon_min"], b["lat_min"]),
        (b["lon_max"], b["lat_min"]),
        (b["lon_max"], b["lat_max"]),
        (b["lon_min"], b["lat_max"]),
    ])


def run_validation(
    buffered: gpd.GeoDataFrame,
    output_root: str,
    *,
    funnel: dict[str, int],
    pagination: dict[str, Any],
    filters: dict[str, list[str]],
    buffer_ft: float,
    bin_match_warn_threshold: float = 0.85,
) -> ValidationResult:
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "validation_report.parquet")
    summary_path = os.path.join(output_root, "summary.md")

    total = len(buffered)
    fatal: list[str] = []
    warns: list[str] = []
    metrics: dict[str, Any] = {}
    gdf = buffered.copy()
    metrics["unique_bbl"] = int(gdf["bbl"].nunique()) if total > 0 else 0

    # Fatal #1: Socrata returned non-empty (filters may legitimately zero it).
    raw_rows = funnel.get("raw_rows", 0)
    if raw_rows == 0:
        msg = "Open Restaurants Socrata returned 0 rows"
        if filters:
            warns.append(f"{msg} — your filters may be too narrow.")
        else:
            fatal.append(msg)

    # Fatal #2: no duplicate uid
    dup_mask = gdf.duplicated(subset=["uid"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        fatal.append(f"duplicate uid rows: {n_dup}")

    # Fatal #3: every publishable row has a supported geom_source
    allowed = ["bin_polygon", "nearest_polygon", "point"]
    bad_gs = ~gdf["geom_source"].isin(allowed)
    n_bad_gs = int(bad_gs.sum())
    if n_bad_gs > 0:
        fatal.append(f"rows with geom_source not in {{{', '.join(allowed)}}}: {n_bad_gs}")

    # Fatal #4: no null facname (every restaurant has a name)
    null_name = gdf["facname"].isna() | (gdf["facname"] == "")
    n_null_name = int(null_name.sum())
    if n_null_name > 0:
        fatal.append(f"rows with null facname: {n_null_name}")

    # Fatal #5: no null license_type
    null_type = gdf["license_type"].isna() | (gdf["license_type"] == "")
    n_null_type = int(null_type.sum())
    if n_null_type > 0:
        fatal.append(f"rows with null license_type: {n_null_type}")

    # Fatal #6: all geometries valid
    is_valid = gdf.geometry.is_valid & gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_invalid = int((~is_valid).sum())
    if n_invalid > 0:
        fatal.append(f"invalid / empty / null buffered geometries: {n_invalid}")

    # Fatal #7: all inside NYC bbox
    bbox_poly = _nyc_bbox_polygon()
    if total > 0:
        in_bbox = gdf.geometry.apply(lambda g: bbox_poly.contains(g) if g and not g.is_empty else False)
    else:
        in_bbox = gdf.geometry
    n_out_bbox = int((~in_bbox).sum()) if total > 0 else 0
    if n_out_bbox > 0:
        fatal.append(f"buffered geometries outside NYC bbox: {n_out_bbox}")

    # Fatal #8: coverage non-empty + valid
    valid_geoms = gdf.loc[is_valid, "geometry"].tolist() if total > 0 else []
    coverage = unary_union(valid_geoms) if valid_geoms else MultiPolygon()
    coverage_valid = coverage is not None and not coverage.is_empty and coverage.is_valid
    if total > 0 and not coverage_valid:
        fatal.append("coverage (unary_union of licenses) is empty or invalid")

    # --- warn metrics ---
    is_poly = gdf["geom_source"].isin(["bin_polygon", "nearest_polygon"])
    is_bin = gdf["geom_source"] == "bin_polygon"
    is_near = gdf["geom_source"] == "nearest_polygon"
    poly_pct = _pct(int(is_poly.sum()), total)
    metrics["polygon_match_rate_overall_pct"] = poly_pct
    metrics["bin_exact_rate_overall_pct"] = _pct(int(is_bin.sum()), total)
    metrics["nearest_polygon_rate_overall_pct"] = _pct(int(is_near.sum()), total)
    if total > 0 and poly_pct / 100.0 < bin_match_warn_threshold:
        warns.append(
            f"polygon match rate {poly_pct:.2f}% below "
            f"{bin_match_warn_threshold*100:.0f}% threshold"
        )

    metrics["per_license_type"] = (
        gdf.groupby("license_type", dropna=False).size().sort_values(ascending=False).to_dict()
    )
    metrics["per_license_status"] = (
        gdf.groupby("license_status", dropna=False).size().sort_values(ascending=False).to_dict()
    )
    metrics["per_borough"] = gdf.groupby("borough", dropna=False).size().to_dict()
    metrics["per_geom_source"] = gdf["geom_source"].value_counts().to_dict()

    if coverage_valid:
        covdf = gpd.GeoDataFrame(geometry=[coverage], crs="EPSG:4326").to_crs("EPSG:2263")
        area_m2 = float(covdf.area.iloc[0] * (0.3048 ** 2))
        metrics["coverage_area_km2"] = area_m2 / 1e6
        metrics["coverage_pct_of_nyc"] = (metrics["coverage_area_km2"] / NYC_LAND_AREA_KM2) * 100.0
    else:
        metrics["coverage_area_km2"] = 0.0
        metrics["coverage_pct_of_nyc"] = 0.0

    # Per-row report
    report_cols = {
        "uid": gdf["uid"].astype(str),
        "facname": gdf["facname"].astype(str),
        "license_type": gdf["license_type"].astype(str),
        "license_status": gdf["license_status"].astype(str),
        "borough": gdf["borough"].astype(str),
        "bin": gdf["bin"].fillna("").astype(str),
        "geom_source": gdf["geom_source"].astype(str),
        "chk_unique_uid": (~dup_mask).astype(bool),
        "chk_facname_not_null": (~null_name).astype(bool),
        "chk_license_type_not_null": (~null_type).astype(bool),
        "chk_geometry_valid": is_valid.astype(bool),
        "chk_geometry_in_nyc_bbox": in_bbox.astype(bool) if total > 0 else in_bbox,
    }
    pl.DataFrame({k: v.tolist() for k, v in report_cols.items()}).write_parquet(report_path)

    lines = _build_summary_md(
        total=total, filters=filters, funnel=funnel, pagination=pagination,
        buffer_ft=buffer_ft,
        fatal=fatal, warns=warns, metrics=metrics,
        n_dup=n_dup, n_bad_gs=n_bad_gs, n_null_name=n_null_name,
        n_null_type=n_null_type, n_invalid=n_invalid, n_out_bbox=n_out_bbox,
        coverage_valid=coverage_valid,
    )
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    for m in warns:
        log.warning("validation: %s", m)
    for m in fatal:
        log.error("validation: FATAL %s", m)

    if fatal:
        raise OpenRestaurantsValidationError("; ".join(fatal))

    return ValidationResult(
        summary_path=summary_path,
        report_path=report_path,
        coverage=coverage,
        fatal_violations=fatal,
        warn_notes=warns,
        metrics=metrics,
    )


def _build_summary_md(
    *, total, filters, funnel, pagination, buffer_ft,
    fatal, warns, metrics, n_dup, n_bad_gs, n_null_name, n_null_type,
    n_invalid, n_out_bbox, coverage_valid,
) -> list[str]:
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# Open Restaurants (Dining Out NYC) curation — validation summary\n"]
    lines.append(f"- Buffer: **{buffer_ft:.0f} ft** (applied in EPSG:2263)")
    lines.append(f"- Publishable rows (one per license): **{total:,}**")
    lines.append(f"- Unique tax lots (BBL): **{metrics.get('unique_bbl', 0):,}**")
    lines.append(f"- Built at: {now}")
    if filters:
        lines.append("- Applied filters:")
        for level, vals in filters.items():
            if vals:
                preview = ", ".join(vals[:5]) + (f"  (+{len(vals)-5} more)" if len(vals) > 5 else "")
                lines.append(f"    - `{level}`: {preview}")
    else:
        lines.append("- Applied filters: **none** (full issued-license pull)")
    lines.append("")

    lines.append("## Fatal checks")
    lines.append("")
    lines.append("| # | Check | Status |")
    lines.append("|---|-------|--------|")
    lines.append(f"| 1 | Socrata returned ≥1 row | "
                 f"{'PASS' if funnel.get('raw_rows', 0) > 0 else 'FAIL'} |")
    lines.append(f"| 2 | Unique `uid` | {'PASS' if n_dup == 0 else f'{n_dup} dupes'} |")
    lines.append(f"| 3 | Every row has geom_source ∈ {{bin_polygon, nearest_polygon, point}} | "
                 f"{'PASS' if n_bad_gs == 0 else f'{n_bad_gs} bad'} |")
    lines.append(f"| 4 | No null `facname` | {'PASS' if n_null_name == 0 else f'{n_null_name} null'} |")
    lines.append(f"| 5 | No null `license_type` | {'PASS' if n_null_type == 0 else f'{n_null_type} null'} |")
    lines.append(f"| 6 | All buffered geometries valid | "
                 f"{'PASS' if n_invalid == 0 else f'{n_invalid} invalid'} |")
    lines.append(f"| 7 | All geometries inside NYC bbox | "
                 f"{'PASS' if n_out_bbox == 0 else f'{n_out_bbox} outside'} |")
    lines.append(f"| 8 | `coverage` non-empty + valid | "
                 f"{'PASS' if coverage_valid else 'FAIL'} |")
    lines.append("")

    lines.append("## Warn checks")
    lines.append("")
    lines.append(f"- Polygon match rate: **{metrics.get('polygon_match_rate_overall_pct', 0):.2f}%** "
                 f"(bin_exact {metrics.get('bin_exact_rate_overall_pct', 0):.2f}% + "
                 f"nearest {metrics.get('nearest_polygon_rate_overall_pct', 0):.2f}%)")
    lines.append(f"- Coverage: **{metrics.get('coverage_area_km2', 0):.3f} km²** "
                 f"({metrics.get('coverage_pct_of_nyc', 0):.3f}% of NYC land)")
    for w in warns:
        lines.append(f"- ⚠️  {w}")
    lines.append("")

    lines.append("## Socrata pagination")
    lines.append("")
    lines.append("| rows | pages | last_page | truncated_likely |")
    lines.append("|-----:|------:|----------:|:-----------------|")
    p = pagination.get("open_restaurants", {})
    page_rows = p.get("page_rows", [])
    last = page_rows[-1] if page_rows else 0
    lines.append(f"| {p.get('total_rows', 0):,} | {p.get('pages', 0)} | "
                 f"{last:,} | {'YES' if p.get('truncated_likely') else 'no'} |")
    lines.append("")

    lines.append("## Per `license_type`")
    lines.append("")
    lines.append("| license_type | rows |")
    lines.append("|--------------|-----:|")
    for t, n in metrics.get("per_license_type", {}).items():
        lines.append(f"| {t} | {int(n):,} |")
    lines.append("")

    lines.append("## Per `license_status`")
    lines.append("")
    lines.append("| license_status | rows |")
    lines.append("|----------------|-----:|")
    for s, n in metrics.get("per_license_status", {}).items():
        lines.append(f"| {s} | {int(n):,} |")
    lines.append("")

    lines.append("## Per borough")
    lines.append("")
    lines.append("| borough | rows |")
    lines.append("|---------|-----:|")
    for b, n in metrics.get("per_borough", {}).items():
        lines.append(f"| {b} | {int(n):,} |")
    lines.append("")

    lines.append("## Per `geom_source`")
    lines.append("")
    lines.append("| geom_source | rows |")
    lines.append("|-------------|-----:|")
    for src, n in metrics.get("per_geom_source", {}).items():
        lines.append(f"| {src} | {int(n):,} |")
    lines.append("")

    if fatal:
        lines.append("## Fatal violations (publication blocked)")
        lines.append("")
        for m in fatal:
            lines.append(f"- {m}")
        lines.append("")
    return lines
