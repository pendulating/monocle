"""Post-build sanity checks for the scaffold-permits curation pipeline.

Modeled on ``dagspaces.common.cyclomedia_catalog.validation``. Every build runs
8 fatal + 12 warn checks against the buffered permit frame. Fatals refuse to
publish outputs; warns only log.

Artifacts written per run (even on fatal, so the operator can diagnose):

- ``validation_report.parquet`` — one row per publishable permit with a boolean
  column per check.
- ``summary.md`` — human-readable scoreboard + per-source × per-borough BIN
  match rates + dropped-permit funnel + top BIN frequencies + coverage area.
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
    "PermitValidationError",
    "ValidationResult",
    "NYC_BBOX_WGS84",
]

log = logging.getLogger(__name__)

NYC_BBOX_WGS84 = {"lon_min": -74.30, "lon_max": -73.60, "lat_min": 40.40, "lat_max": 41.00}
NYC_LAND_AREA_KM2 = 778.2  # used only for "% of NYC" surface-area framing

ISSUE_DATE_FLOOR = dt.datetime(1990, 1, 1)


class PermitValidationError(AssertionError):
    """Raised when a fatal invariant is violated."""


@dataclass
class ValidationResult:
    summary_path: str
    report_path: str
    coverage: BaseGeometry                # unary_union of all buffered geometries
    fatal_violations: list[str]           # empty on success
    warn_notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def _fmt_pct(n: int, d: int) -> str:
    if d == 0:
        return "  n/a "
    return f"{100.0 * n / d:6.2f}%"


def _pct(n: int, d: int) -> float:
    if d == 0:
        return float("nan")
    return 100.0 * n / d


def run_validation(
    buffered: gpd.GeoDataFrame,
    output_root: str,
    *,
    funnel: dict[str, int],
    pagination: dict[str, Any],
    cutoff: str,
    buffer_ft: float,
    since: Optional[str] = None,
    bin_match_warn_threshold: float = 0.85,
) -> ValidationResult:
    """Run all checks, write artifacts, raise on fatal.

    Args:
        buffered: Publishable GeoDataFrame in EPSG:4326 with the common schema +
            ``geometry`` + ``geom_source`` ∈ {``bin_polygon``, ``point``}. Rows
            with ``geom_source == "none"`` should be filtered out upstream and
            counted in ``funnel``.
        output_root: Dir where ``summary.md`` + ``validation_report.parquet`` land.
        funnel: ``{"dob_now_raw": n1, "bis_raw": n2, "after_normalize": n3,
            "after_valid_issue_date": n4, "after_valid_scaffold_type": n5,
            "after_attach_geometry": n6, "publishable": n7}``.
        pagination: ``{"dob_now": {...}, "bis": {...}}`` from FetchResult; keys
            include ``pages``, ``page_rows``, ``truncated_likely``.
        cutoff: ISO date string, echoed into summary.
        buffer_ft: Buffer distance in feet, echoed into summary.
        bin_match_warn_threshold: Polygon-match rate below which we raise a warn.

    Returns: :class:`ValidationResult`. Raises :class:`PermitValidationError`
    if any fatal check fails (after writing artifacts).
    """
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "validation_report.parquet")
    summary_path = os.path.join(output_root, "summary.md")

    total = len(buffered)
    fatal: list[str] = []
    warns: list[str] = []
    metrics: dict[str, Any] = {}

    # ------------------------------------------------------------------ row checks
    # Buffer the GeoDataFrame work through pandas to avoid reprojecting twice.
    gdf = buffered.copy()

    # Fatal #2: no duplicate (source, permit_id)
    dup_mask = gdf.duplicated(subset=["source", "permit_id"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        fatal.append(f"duplicate (source, permit_id) rows: {n_dup}")

    # Fatal #3: every publishable row has a geom_source in the allowed set
    allowed_geom_sources = ["bin_polygon", "nearest_polygon", "point"]
    bad_geom_source = ~gdf["geom_source"].isin(allowed_geom_sources)
    n_bad_gs = int(bad_geom_source.sum())
    if n_bad_gs > 0:
        fatal.append(
            f"rows with geom_source not in {{{', '.join(allowed_geom_sources)}}}: {n_bad_gs}"
        )

    # Fatal #4: no null issue_date
    null_issue = gdf["issue_date"].isna()
    n_null_issue = int(null_issue.sum())
    if n_null_issue > 0:
        fatal.append(f"rows with null issue_date: {n_null_issue}")

    # Fatal #5: no null scaffold_type
    null_stype = gdf["scaffold_type"].isna()
    n_null_stype = int(null_stype.sum())
    if n_null_stype > 0:
        fatal.append(f"rows with null scaffold_type: {n_null_stype}")

    # Fatal #6: all geometries valid
    is_valid = gdf.geometry.is_valid & gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_invalid = int((~is_valid).sum())
    if n_invalid > 0:
        fatal.append(f"invalid / empty / null buffered geometries: {n_invalid}")

    # Fatal #7: all geometries inside NYC bbox
    bbox_poly = _nyc_bbox_polygon()
    if total > 0:
        in_bbox = gdf.geometry.apply(lambda g: bbox_poly.contains(g) if g and not g.is_empty else False)
    else:
        in_bbox = gdf.geometry  # empty series
    n_out_bbox = int((~in_bbox).sum()) if total > 0 else 0
    if n_out_bbox > 0:
        fatal.append(f"buffered geometries outside NYC bbox: {n_out_bbox}")

    # Fatal #8: coverage = unary_union(geoms) non-empty and valid
    valid_geoms = gdf.loc[is_valid, "geometry"].tolist() if total > 0 else []
    coverage = unary_union(valid_geoms) if valid_geoms else MultiPolygon()
    coverage_valid = coverage is not None and not coverage.is_empty and coverage.is_valid
    if total > 0 and not coverage_valid:
        fatal.append("coverage (unary_union of permits) is empty or invalid")

    # Fatal #1: both sources non-empty after pagination
    n_dob = funnel.get("dob_now_raw", 0)
    n_bis = funnel.get("bis_raw", 0)
    if n_dob == 0:
        fatal.append("DOB NOW returned 0 rows — Socrata API or where-clause failure")
    if n_bis == 0:
        fatal.append("BIS returned 0 rows — Socrata API or where-clause failure")

    # ------------------------------------------------------------------ warn checks
    # Warn #9 + #20: per-stage polygon match rate, per source × per borough.
    # A "polygon" match is either exact-BIN or nearest-building fallback; both
    # return an authoritative building footprint (not an artificial point).
    is_poly = gdf["geom_source"].isin(["bin_polygon", "nearest_polygon"])
    is_bin = (gdf["geom_source"] == "bin_polygon")
    is_nearest = (gdf["geom_source"] == "nearest_polygon")
    match_by_group = (
        gdf.assign(
            _is_poly=is_poly.astype(int),
            _is_bin=is_bin.astype(int),
            _is_nearest=is_nearest.astype(int),
        )
        .groupby(["source", "borough"], dropna=False)
        .agg(
            total=("permit_id", "count"),
            polygon=("_is_poly", "sum"),
            bin_exact=("_is_bin", "sum"),
            nearest=("_is_nearest", "sum"),
        )
        .reset_index()
    )
    match_by_group["rate"] = match_by_group["polygon"] / match_by_group["total"].clip(lower=1)
    metrics["bin_match_by_group"] = match_by_group.to_dict(orient="records")

    poly_rate_overall = _pct(int(is_poly.sum()), total)
    bin_rate_overall = _pct(int(is_bin.sum()), total)
    nearest_rate_overall = _pct(int(is_nearest.sum()), total)
    metrics["polygon_match_rate_overall_pct"] = poly_rate_overall
    metrics["bin_exact_rate_overall_pct"] = bin_rate_overall
    metrics["nearest_polygon_rate_overall_pct"] = nearest_rate_overall
    # Back-compat key (older manifests / external tools).
    metrics["bin_match_rate_overall_pct"] = poly_rate_overall
    if poly_rate_overall / 100.0 < bin_match_warn_threshold:
        warns.append(
            f"overall polygon match rate {poly_rate_overall:.2f}% below "
            f"{bin_match_warn_threshold*100:.0f}% threshold"
        )

    # Nearest-polygon distance distribution — surfaces bad fallback matches.
    if is_nearest.any() and "match_dist_ft" in gdf.columns:
        near_dists = gdf.loc[is_nearest, "match_dist_ft"].dropna()
        if len(near_dists) > 0:
            metrics["nearest_polygon_dist_ft_p50"] = float(near_dists.quantile(0.50))
            metrics["nearest_polygon_dist_ft_p95"] = float(near_dists.quantile(0.95))
            metrics["nearest_polygon_dist_ft_p99"] = float(near_dists.quantile(0.99))
            metrics["nearest_polygon_dist_ft_max"] = float(near_dists.max())

    # Warn #11: Socrata pagination truncation
    for src in ("dob_now", "bis"):
        p = pagination.get(src, {})
        if p.get("truncated_likely"):
            warns.append(
                f"{src}: last page returned exactly the limit — likely truncated"
            )

    # Warn #12: null first_permit_date dropout (recorded by the orchestrator;
    # surfaced here if funnel carries the count).
    n_null_fpd = funnel.get("dob_now_dropped_null_first_permit_date", 0)
    metrics["dob_now_dropped_null_first_permit_date"] = n_null_fpd

    # Warn #13: scaffold_type distribution per source
    stype_dist = (
        gdf.groupby(["source", "scaffold_type"], dropna=False)
           .size()
           .reset_index(name="count")
    )
    metrics["scaffold_type_distribution"] = stype_dist.to_dict(orient="records")

    # Warn #14: permit_status distribution per source
    pstatus_dist = (
        gdf.groupby(["source", "permit_status"], dropna=False)
           .size()
           .reset_index(name="count")
    )
    metrics["permit_status_distribution"] = pstatus_dist.to_dict(orient="records")

    # Warn #15: issue_date floor/ceiling outliers
    issue_dt = gdf["issue_date"]
    ceiling = _parse_cutoff(cutoff)
    too_old = int((issue_dt < ISSUE_DATE_FLOOR).sum())
    too_new = int((issue_dt > ceiling).sum())
    metrics["issue_date_too_old_pre_1990"] = too_old
    metrics["issue_date_after_cutoff"] = too_new
    if too_new > 0:
        warns.append(
            f"{too_new} permits have issue_date > cutoff {cutoff} (server filter leak?)"
        )

    # Warn #16: top BIN frequencies
    bin_counts = (
        gdf.dropna(subset=["bin"])
           .groupby("bin").size().reset_index(name="count")
           .sort_values("count", ascending=False)
           .head(20)
    )
    # Attach representative borough + address
    if len(bin_counts) > 0:
        bin_examples = (
            gdf.dropna(subset=["bin"])
               .drop_duplicates(subset=["bin"], keep="first")
               .set_index("bin")[["borough", "address"]]
        )
        bin_counts = bin_counts.merge(
            bin_examples, left_on="bin", right_index=True, how="left"
        )
    metrics["top_bins"] = bin_counts.to_dict(orient="records")

    # Warn #17: cross-source BIN overlap
    dob_bins = set(gdf.loc[gdf["source"] == "dob_now", "bin"].dropna())
    bis_bins = set(gdf.loc[gdf["source"] == "bis", "bin"].dropna())
    overlap = dob_bins & bis_bins
    metrics["cross_source_bin_overlap"] = len(overlap)
    metrics["dob_now_unique_bins"] = len(dob_bins)
    metrics["bis_unique_bins"] = len(bis_bins)

    # Warn #18: per-permit buffered area distribution (in m²)
    if total > 0 and is_valid.any():
        # Reproject a copy to a metric CRS for area computation.
        area_m2 = (
            gpd.GeoDataFrame(geometry=gdf.loc[is_valid, "geometry"], crs=gdf.crs)
               .to_crs("EPSG:2263")
               .area
               .mul(0.3048 ** 2)  # ft² → m²
        )
        metrics["buffered_area_m2_p50"] = float(area_m2.quantile(0.50))
        metrics["buffered_area_m2_p95"] = float(area_m2.quantile(0.95))
        metrics["buffered_area_m2_p99"] = float(area_m2.quantile(0.99))

    # Warn #19: total coverage area
    if coverage_valid:
        coverage_gdf = gpd.GeoDataFrame(
            geometry=[coverage], crs="EPSG:4326"
        ).to_crs("EPSG:2263")
        coverage_area_m2 = float(coverage_gdf.area.iloc[0] * (0.3048 ** 2))
        coverage_km2 = coverage_area_m2 / 1e6
        metrics["coverage_area_km2"] = coverage_km2
        metrics["coverage_pct_of_nyc"] = (coverage_km2 / NYC_LAND_AREA_KM2) * 100.0
    else:
        metrics["coverage_area_km2"] = 0.0
        metrics["coverage_pct_of_nyc"] = 0.0

    # ------------------------------------------------------------------ report + summary
    # Per-row boolean columns for drill-down. Defensive with NaN geometries.
    report_cols = {
        "source": gdf["source"].astype(str),
        "permit_id": gdf["permit_id"].astype(str),
        "bin": gdf["bin"].fillna("").astype(str),
        "borough": gdf["borough"].astype(str),
        "geom_source": gdf["geom_source"].astype(str),
        "chk_unique_permit_id": (~dup_mask).astype(bool),
        "chk_issue_date_not_null": (~null_issue).astype(bool),
        "chk_scaffold_type_not_null": (~null_stype).astype(bool),
        "chk_geometry_valid": is_valid.astype(bool),
        "chk_geometry_in_nyc_bbox": in_bbox.astype(bool) if total > 0 else in_bbox,
        "chk_issue_date_in_range": (
            (issue_dt >= ISSUE_DATE_FLOOR) & (issue_dt <= ceiling)
        ).astype(bool),
    }
    report_df = pl.DataFrame({k: v.tolist() for k, v in report_cols.items()})
    report_df.write_parquet(report_path)

    lines = _build_summary_md(
        total=total,
        funnel=funnel,
        pagination=pagination,
        cutoff=cutoff,
        since=since,
        buffer_ft=buffer_ft,
        fatal=fatal,
        warns=warns,
        metrics=metrics,
        n_dup=n_dup,
        n_bad_gs=n_bad_gs,
        n_null_issue=n_null_issue,
        n_null_stype=n_null_stype,
        n_invalid=n_invalid,
        n_out_bbox=n_out_bbox,
        coverage_valid=coverage_valid,
    )
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    result = ValidationResult(
        summary_path=summary_path,
        report_path=report_path,
        coverage=coverage,
        fatal_violations=fatal,
        warn_notes=warns,
        metrics=metrics,
    )

    for m in warns:
        log.warning("validation: %s", m)
    for m in fatal:
        log.error("validation: FATAL %s", m)

    if fatal:
        raise PermitValidationError("; ".join(fatal))

    return result


# ---------------------------------------------------------------------- helpers

def _nyc_bbox_polygon() -> Polygon:
    b = NYC_BBOX_WGS84
    return Polygon([
        (b["lon_min"], b["lat_min"]),
        (b["lon_max"], b["lat_min"]),
        (b["lon_max"], b["lat_max"]),
        (b["lon_min"], b["lat_max"]),
    ])


def _parse_cutoff(cutoff: str) -> dt.datetime:
    s = cutoff.strip()
    if "T" in s:
        return dt.datetime.fromisoformat(s)
    return dt.datetime.strptime(s, "%Y-%m-%d").replace(hour=23, minute=59, second=59)


def _build_summary_md(
    *,
    total: int,
    funnel: dict[str, int],
    pagination: dict[str, Any],
    cutoff: str,
    buffer_ft: float,
    since: Optional[str] = None,
    fatal: list[str],
    warns: list[str],
    metrics: dict[str, Any],
    n_dup: int,
    n_bad_gs: int,
    n_null_issue: int,
    n_null_stype: int,
    n_invalid: int,
    n_out_bbox: int,
    coverage_valid: bool,
) -> list[str]:
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Scaffolding permits curation — validation summary\n")
    if since is not None:
        lines.append(f"- Issue-date range: **{since}** → **{cutoff}**")
    else:
        lines.append(f"- Issue-date range: **(no lower bound)** → **{cutoff}**")
    lines.append(f"- Buffer: **{buffer_ft:.0f} ft** (applied in EPSG:2263)")
    lines.append(f"- Publishable rows: **{total:,}**")
    lines.append(f"- Built at: {now}")
    lines.append("")

    # Fatal checks
    lines.append("## Fatal checks")
    lines.append("")
    lines.append("| # | Check | Status |")
    lines.append("|---|-------|--------|")
    lines.append(f"| 1 | Both sources returned ≥1 row | "
                 f"{'PASS' if funnel.get('dob_now_raw', 0) > 0 and funnel.get('bis_raw', 0) > 0 else 'FAIL'} |")
    lines.append(f"| 2 | Unique `(source, permit_id)` | "
                 f"{'PASS' if n_dup == 0 else f'{n_dup} dupes'} |")
    lines.append(f"| 3 | Every row has `geom_source` ∈ {{bin_polygon, nearest_polygon, point}} | "
                 f"{'PASS' if n_bad_gs == 0 else f'{n_bad_gs} bad'} |")
    lines.append(f"| 4 | No null `issue_date` | "
                 f"{'PASS' if n_null_issue == 0 else f'{n_null_issue} null'} |")
    lines.append(f"| 5 | No null `scaffold_type` | "
                 f"{'PASS' if n_null_stype == 0 else f'{n_null_stype} null'} |")
    lines.append(f"| 6 | All buffered geometries valid | "
                 f"{'PASS' if n_invalid == 0 else f'{n_invalid} invalid'} |")
    lines.append(f"| 7 | All geometries inside NYC bbox | "
                 f"{'PASS' if n_out_bbox == 0 else f'{n_out_bbox} outside'} |")
    lines.append(f"| 8 | `coverage` (unary_union) non-empty + valid | "
                 f"{'PASS' if coverage_valid else 'FAIL'} |")
    lines.append("")

    # Warn summary
    lines.append("## Warn checks")
    lines.append("")
    poly_pct = metrics.get("polygon_match_rate_overall_pct", float("nan"))
    bin_pct = metrics.get("bin_exact_rate_overall_pct", float("nan"))
    near_pct = metrics.get("nearest_polygon_rate_overall_pct", float("nan"))
    lines.append(
        f"- **Polygon match rate (overall):** {poly_pct:.2f}%  "
        f"(bin_exact {bin_pct:.2f}% + nearest_polygon {near_pct:.2f}%)"
    )
    if "nearest_polygon_dist_ft_p50" in metrics:
        lines.append(
            f"- Nearest-building fallback distances (ft): "
            f"p50={metrics['nearest_polygon_dist_ft_p50']:.0f}  "
            f"p95={metrics['nearest_polygon_dist_ft_p95']:.0f}  "
            f"p99={metrics['nearest_polygon_dist_ft_p99']:.0f}  "
            f"max={metrics['nearest_polygon_dist_ft_max']:.0f}"
        )
    lines.append(f"- Coverage area: **{metrics.get('coverage_area_km2', 0):.2f} km²** "
                 f"({metrics.get('coverage_pct_of_nyc', 0):.2f}% of NYC land)")
    lines.append(f"- Cross-source BIN overlap: **{metrics.get('cross_source_bin_overlap', 0):,}** BINs "
                 f"(dob_now unique={metrics.get('dob_now_unique_bins', 0):,}, "
                 f"bis unique={metrics.get('bis_unique_bins', 0):,})")
    lines.append(f"- DOB NOW filings dropped for null `first_permit_date`: "
                 f"**{metrics.get('dob_now_dropped_null_first_permit_date', 0):,}** "
                 "(intentional per design decision #1)")
    lines.append(f"- issue_date outside [1990-01-01, cutoff]: "
                 f"too_old={metrics.get('issue_date_too_old_pre_1990', 0)}, "
                 f"after_cutoff={metrics.get('issue_date_after_cutoff', 0)}")
    if "buffered_area_m2_p50" in metrics:
        lines.append(f"- Per-permit buffered area (m²): "
                     f"p50={metrics['buffered_area_m2_p50']:,.0f}, "
                     f"p95={metrics['buffered_area_m2_p95']:,.0f}, "
                     f"p99={metrics['buffered_area_m2_p99']:,.0f}")
    for w in warns:
        lines.append(f"- ⚠️  {w}")
    lines.append("")

    # Pagination
    lines.append("## Socrata pagination")
    lines.append("")
    lines.append("| source | rows | pages | last_page | truncated_likely |")
    lines.append("|--------|-----:|------:|----------:|:-----------------|")
    for src in ("dob_now", "bis"):
        p = pagination.get(src, {})
        page_rows = p.get("page_rows", [])
        last = page_rows[-1] if page_rows else 0
        lines.append(
            f"| {src} | {p.get('total_rows', 0):,} | {p.get('pages', 0)} | "
            f"{last:,} | {'YES' if p.get('truncated_likely') else 'no'} |"
        )
    lines.append("")

    # Polygon match rate per source × borough, split by stage
    lines.append("## Polygon match rate by source × borough")
    lines.append("")
    lines.append("| source | borough | total | bin_exact | nearest | point | polygon % |")
    lines.append("|--------|---------|------:|----------:|--------:|------:|----------:|")
    for row in metrics.get("bin_match_by_group", []):
        total_ = int(row["total"])
        poly_ = int(row["polygon"])
        bin_ = int(row.get("bin_exact", row["polygon"]))
        near_ = int(row.get("nearest", 0))
        pt_ = total_ - poly_
        rate = 100.0 * poly_ / total_ if total_ else 0.0
        lines.append(
            f"| {row['source']} | {row['borough']} | {total_:,} | "
            f"{bin_:,} | {near_:,} | {pt_:,} | {rate:.2f}% |"
        )
    lines.append("")

    # Dropped-permit funnel
    lines.append("## Dropped-permit funnel")
    lines.append("")
    lines.append("| step | rows | Δ from previous |")
    lines.append("|------|-----:|----------------:|")
    funnel_order = [
        ("dob_now_raw", "DOB NOW raw fetch"),
        ("bis_raw", "BIS raw fetch"),
        ("after_normalize", "after normalize"),
        ("after_date_clip", "after client-side date clip (≤ cutoff)"),
        ("after_valid_scaffold_type", "after scaffold_type non-null"),
        ("after_attach_geometry", "after geometry attach (polygon ∪ point)"),
        ("publishable", "publishable (fed to validation)"),
    ]
    prev = None
    for key, label in funnel_order:
        v = funnel.get(key)
        if v is None:
            continue
        delta = "" if prev is None else f"{v - prev:+,}"
        lines.append(f"| {label} | {v:,} | {delta} |")
        prev = v
    lines.append("")

    # Scaffold type distribution
    lines.append("## `scaffold_type` distribution")
    lines.append("")
    lines.append("| source | scaffold_type | count |")
    lines.append("|--------|---------------|------:|")
    for row in metrics.get("scaffold_type_distribution", []):
        lines.append(f"| {row['source']} | {row['scaffold_type']} | {int(row['count']):,} |")
    lines.append("")

    # Permit status distribution
    lines.append("## `permit_status` distribution")
    lines.append("")
    lines.append("| source | permit_status | count |")
    lines.append("|--------|---------------|------:|")
    for row in metrics.get("permit_status_distribution", []):
        lines.append(f"| {row['source']} | {row['permit_status']} | {int(row['count']):,} |")
    lines.append("")

    # Top BINs
    lines.append("## Top 20 BINs by permit count")
    lines.append("")
    lines.append("| bin | permits | borough | address |")
    lines.append("|-----|--------:|---------|---------|")
    for row in metrics.get("top_bins", []):
        lines.append(
            f"| {row.get('bin', '')} | {int(row.get('count', 0)):,} | "
            f"{row.get('borough', '')} | {row.get('address', '')} |"
        )
    lines.append("")

    if fatal:
        lines.append("## Fatal violations (publication blocked)")
        lines.append("")
        for m in fatal:
            lines.append(f"- {m}")
        lines.append("")

    return lines
