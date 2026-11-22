"""Post-build sanity checks for DOHMH restaurant curation.

Mirrors :mod:`..facdb.validation`. Produces ``validation_report.parquet``
+ ``summary.md`` per build. Raises :class:`DohmhValidationError` on fatal
checks (unique camis, non-null identity, geom validity, bbox, coverage).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import geopandas as gpd
import polars as pl
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

__all__ = [
    "run_validation",
    "DohmhValidationError",
    "ValidationResult",
    "NYC_BBOX_WGS84",
]

log = logging.getLogger(__name__)

NYC_BBOX_WGS84 = {"lon_min": -74.30, "lon_max": -73.60, "lat_min": 40.40, "lat_max": 41.00}
NYC_LAND_AREA_KM2 = 778.2


class DohmhValidationError(AssertionError):
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
    """Run checks, write artifacts, raise on fatal."""
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "validation_report.parquet")
    summary_path = os.path.join(output_root, "summary.md")

    total = len(buffered)
    fatal: list[str] = []
    warns: list[str] = []
    metrics: dict[str, Any] = {}
    gdf = buffered.copy()
    n_unique_camis = int(gdf["camis"].nunique()) if total > 0 else 0
    metrics["unique_camis"] = n_unique_camis

    # Fatal #1: Socrata returned non-empty (with filters: warn instead).
    raw_rows = funnel.get("raw_rows", 0)
    if raw_rows == 0:
        msg = "DOHMH Socrata returned 0 rows"
        if filters:
            warns.append(f"{msg} — your filters may be too narrow.")
        else:
            fatal.append(msg)

    # Fatal #2: unique (camis, inspection_date). Multiple inspections per
    # camis are expected — collapse to one row per restaurant via the
    # aggregate-restaurants subcommand.
    dup_mask = gdf.duplicated(subset=["sample_id"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        fatal.append(f"duplicate (camis, inspection_date) rows: {n_dup}")

    # Fatal #3: every publishable row has a supported geom_source
    allowed = ["bin_polygon", "nearest_polygon", "point"]
    bad_gs = ~gdf["geom_source"].isin(allowed)
    n_bad_gs = int(bad_gs.sum())
    if n_bad_gs > 0:
        fatal.append(f"rows with geom_source not in {{{', '.join(allowed)}}}: {n_bad_gs}")

    # Fatal #4: no null facname (restaurant must have a DBA name)
    null_name = gdf["facname"].isna() | (gdf["facname"] == "")
    n_null_name = int(null_name.sum())
    if n_null_name > 0:
        fatal.append(f"rows with null DBA / facname: {n_null_name}")

    # Fatal #5: all geometries valid
    is_valid = gdf.geometry.is_valid & gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_invalid = int((~is_valid).sum())
    if n_invalid > 0:
        fatal.append(f"invalid / empty / null buffered geometries: {n_invalid}")

    # Fatal #6: all inside NYC bbox
    bbox_poly = _nyc_bbox_polygon()
    if total > 0:
        in_bbox = gdf.geometry.apply(
            lambda g: bbox_poly.contains(g) if g and not g.is_empty else False
        )
    else:
        in_bbox = gdf.geometry
    n_out_bbox = int((~in_bbox).sum()) if total > 0 else 0
    if n_out_bbox > 0:
        fatal.append(f"buffered geometries outside NYC bbox: {n_out_bbox}")

    # Fatal #7: coverage non-empty + valid
    valid_geoms = gdf.loc[is_valid, "geometry"].tolist() if total > 0 else []
    coverage = unary_union(valid_geoms) if valid_geoms else MultiPolygon()
    coverage_valid = coverage is not None and not coverage.is_empty and coverage.is_valid
    if total > 0 and not coverage_valid:
        fatal.append("coverage (unary_union of restaurants) is empty or invalid")

    # ---- warn metrics ---------------------------------------------------
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

    # placeholder-inspection rate (per row, since each row is an inspection)
    if total > 0:
        n_placeholder = int(gdf["is_placeholder_inspection"].sum())
        metrics["placeholder_inspections"] = n_placeholder
        metrics["placeholder_inspections_pct"] = _pct(n_placeholder, total)
        # Camis that only ever got placeholder rows (= "registered but never inspected").
        camis_with_real = set(
            gdf.loc[~gdf["is_placeholder_inspection"], "camis"].astype(str).unique()
        )
        all_camis = set(gdf["camis"].astype(str).unique())
        camis_only_placeholder = sorted(all_camis - camis_with_real)
        metrics["camis_only_placeholder"] = len(camis_only_placeholder)
        if camis_only_placeholder:
            warns.append(
                f"{len(camis_only_placeholder)} CAMIS "
                f"({_pct(len(camis_only_placeholder), n_unique_camis):.2f}% of "
                "unique restaurants) have only placeholder inspection rows "
                "(registered but never inspected)"
            )

    # inspections-per-camis distribution
    if total > 0 and n_unique_camis > 0:
        per_camis = gdf.groupby("camis").size()
        metrics["inspections_per_camis_p50"] = float(per_camis.median())
        metrics["inspections_per_camis_p95"] = float(per_camis.quantile(0.95))
        metrics["inspections_per_camis_max"] = int(per_camis.max())
        metrics["inspections_per_camis_mean"] = float(per_camis.mean())

    # Per-cuisine / per-borough / per-grade
    metrics["per_cuisine"] = (
        gdf.groupby("cuisine_description", dropna=False).size()
           .sort_values(ascending=False).head(20).to_dict()
    )
    metrics["per_borough"] = gdf.groupby("borough", dropna=False).size().to_dict()
    metrics["per_grade"] = gdf.groupby("grade", dropna=False).size().to_dict()
    metrics["per_geom_source"] = gdf["geom_source"].value_counts().to_dict()

    # Coverage area
    if coverage_valid:
        covdf = gpd.GeoDataFrame(geometry=[coverage], crs="EPSG:4326").to_crs("EPSG:2263")
        area_m2 = float(covdf.area.iloc[0] * (0.3048 ** 2))
        metrics["coverage_area_km2"] = area_m2 / 1e6
        metrics["coverage_pct_of_nyc"] = (metrics["coverage_area_km2"] / NYC_LAND_AREA_KM2) * 100.0
    else:
        metrics["coverage_area_km2"] = 0.0
        metrics["coverage_pct_of_nyc"] = 0.0

    # Per-row report parquet
    report_cols = {
        "sample_id": gdf["sample_id"].astype(str),
        "camis": gdf["camis"].astype(str),
        "facname": gdf["facname"].astype(str),
        "cuisine_description": gdf["cuisine_description"].astype(str),
        "borough": gdf["borough"].astype(str),
        "bin": gdf["bin"].fillna("").astype(str),
        "geom_source": gdf["geom_source"].astype(str),
        "grade": gdf["grade"].astype(str),
        "is_placeholder_inspection": gdf["is_placeholder_inspection"].astype(bool),
        "chk_unique_sample_id": (~dup_mask).astype(bool),
        "chk_facname_not_null": (~null_name).astype(bool),
        "chk_geometry_valid": is_valid.astype(bool),
        "chk_geometry_in_nyc_bbox": in_bbox.astype(bool) if total > 0 else in_bbox,
    }
    pl.DataFrame({k: v.tolist() for k, v in report_cols.items()}).write_parquet(report_path)

    lines = _build_summary_md(
        total=total, filters=filters, funnel=funnel, pagination=pagination,
        buffer_ft=buffer_ft,
        fatal=fatal, warns=warns, metrics=metrics,
        n_dup=n_dup, n_bad_gs=n_bad_gs, n_null_name=n_null_name,
        n_invalid=n_invalid, n_out_bbox=n_out_bbox, coverage_valid=coverage_valid,
    )
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    for m in warns:
        log.warning("validation: %s", m)
    for m in fatal:
        log.error("validation: FATAL %s", m)

    if fatal:
        raise DohmhValidationError("; ".join(fatal))

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
    fatal, warns, metrics, n_dup, n_bad_gs, n_null_name,
    n_invalid, n_out_bbox, coverage_valid,
) -> list[str]:
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# DOHMH restaurant curation — validation summary\n"]
    lines.append(f"- Buffer: **{buffer_ft:.0f} ft** (applied in EPSG:2263)")
    lines.append(f"- Publishable rows (one per camis × inspection_date): **{total:,}**")
    lines.append(f"- Unique CAMIS (restaurants): **{metrics.get('unique_camis', 0):,}**")
    lines.append(f"- Built at: {now}")
    if filters:
        lines.append("- Applied filters:")
        for level, vals in filters.items():
            if vals:
                preview = ", ".join(vals[:5]) + (f"  (+{len(vals)-5} more)" if len(vals) > 5 else "")
                lines.append(f"    - `{level}`: {preview}")
    else:
        lines.append("- Applied filters: **none** (full DOHMH pull)")
    lines.append("")

    lines.append("## Fatal checks")
    lines.append("")
    lines.append("| # | Check | Status |")
    lines.append("|---|-------|--------|")
    lines.append(f"| 1 | DOHMH returned ≥1 row | "
                 f"{'PASS' if funnel.get('raw_rows', 0) > 0 else 'FAIL'} |")
    lines.append(f"| 2 | Unique `(camis, inspection_date)` | {'PASS' if n_dup == 0 else f'{n_dup} dupes'} |")
    lines.append(f"| 3 | Every row has geom_source ∈ {{bin_polygon, nearest_polygon, point}} | "
                 f"{'PASS' if n_bad_gs == 0 else f'{n_bad_gs} bad'} |")
    lines.append(f"| 4 | No null DBA / `facname` | "
                 f"{'PASS' if n_null_name == 0 else f'{n_null_name} null'} |")
    lines.append(f"| 5 | All buffered geometries valid | "
                 f"{'PASS' if n_invalid == 0 else f'{n_invalid} invalid'} |")
    lines.append(f"| 6 | All geometries inside NYC bbox | "
                 f"{'PASS' if n_out_bbox == 0 else f'{n_out_bbox} outside'} |")
    lines.append(f"| 7 | `coverage` non-empty + valid | "
                 f"{'PASS' if coverage_valid else 'FAIL'} |")
    lines.append("")

    lines.append("## Warn checks")
    lines.append("")
    lines.append(f"- Polygon match rate: **{metrics.get('polygon_match_rate_overall_pct', 0):.2f}%** "
                 f"(bin_exact {metrics.get('bin_exact_rate_overall_pct', 0):.2f}% + "
                 f"nearest {metrics.get('nearest_polygon_rate_overall_pct', 0):.2f}%)")
    lines.append(f"- Coverage: **{metrics.get('coverage_area_km2', 0):.2f} km²** "
                 f"({metrics.get('coverage_pct_of_nyc', 0):.2f}% of NYC land)")
    if "placeholder_inspections_pct" in metrics:
        lines.append(
            f"- Placeholder inspection rows: "
            f"**{metrics.get('placeholder_inspections', 0):,}** "
            f"({metrics.get('placeholder_inspections_pct', 0):.2f}%)"
        )
        if metrics.get("camis_only_placeholder", 0):
            lines.append(
                f"- CAMIS with **only** placeholder rows (never inspected): "
                f"**{metrics['camis_only_placeholder']:,}**"
            )
    if "inspections_per_camis_p50" in metrics:
        lines.append(
            f"- Inspections per CAMIS — "
            f"mean {metrics['inspections_per_camis_mean']:.1f}, "
            f"p50 {metrics['inspections_per_camis_p50']:.0f}, "
            f"p95 {metrics['inspections_per_camis_p95']:.0f}, "
            f"max {metrics['inspections_per_camis_max']:,}"
        )
    for w in warns:
        lines.append(f"- ⚠️  {w}")
    lines.append("")

    # Funnel
    lines.append("## Dropped-row funnel")
    lines.append("")
    lines.append("| Step | Rows |")
    lines.append("|------|-----:|")
    for k, v in funnel.items():
        lines.append(f"| {k} | {int(v):,} |")
    lines.append("")

    # Socrata pagination
    lines.append("## Socrata pagination")
    lines.append("")
    lines.append("| rows | pages | last_page | truncated_likely |")
    lines.append("|-----:|------:|----------:|:-----------------|")
    p = pagination.get("dohmh", {})
    page_rows = p.get("page_rows", [])
    last = page_rows[-1] if page_rows else 0
    lines.append(f"| {p.get('total_rows', 0):,} | {p.get('pages', 0)} | "
                 f"{last:,} | {'YES' if p.get('truncated_likely') else 'no'} |")
    lines.append("")

    # Top cuisines
    lines.append("## Top 20 cuisines")
    lines.append("")
    lines.append("| cuisine_description | rows |")
    lines.append("|---------------------|-----:|")
    for c, n in metrics.get("per_cuisine", {}).items():
        lines.append(f"| {c} | {int(n):,} |")
    lines.append("")

    # Per-borough
    lines.append("## Per borough")
    lines.append("")
    lines.append("| borough | rows |")
    lines.append("|---------|-----:|")
    for b, n in metrics.get("per_borough", {}).items():
        lines.append(f"| {b} | {int(n):,} |")
    lines.append("")

    # Per-grade
    lines.append("## Per inspection grade (all rows)")
    lines.append("")
    lines.append("| grade | rows |")
    lines.append("|-------|-----:|")
    for g, n in metrics.get("per_grade", {}).items():
        lines.append(f"| {g} | {int(n):,} |")
    lines.append("")

    # geom_source
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
