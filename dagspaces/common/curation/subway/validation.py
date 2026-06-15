"""Post-build sanity checks for subway-entrances curation.

Mirrors :mod:`..facdb.validation` and :mod:`..dohmh.validation`. Produces
``validation_report.parquet`` + ``summary.md`` per build. Raises
:class:`SubwayValidationError` on fatal checks.
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
    "SubwayValidationError",
    "ValidationResult",
    "NYC_BBOX_WGS84",
]

log = logging.getLogger(__name__)

NYC_BBOX_WGS84 = {"lon_min": -74.30, "lon_max": -73.60, "lat_min": 40.40, "lat_max": 41.00}
NYC_LAND_AREA_KM2 = 778.2


class SubwayValidationError(AssertionError):
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
) -> ValidationResult:
    os.makedirs(output_root, exist_ok=True)
    report_path = os.path.join(output_root, "validation_report.parquet")
    summary_path = os.path.join(output_root, "summary.md")

    total = len(buffered)
    fatal: list[str] = []
    warns: list[str] = []
    metrics: dict[str, Any] = {}
    gdf = buffered.copy()
    metrics["unique_stations"] = (
        int(gdf["station_id"].nunique()) if total > 0 else 0
    )
    metrics["unique_complexes"] = (
        int(gdf["complex_id"].nunique()) if total > 0 else 0
    )

    raw_rows = funnel.get("raw_rows", 0)
    if raw_rows == 0:
        msg = "Subway entrances Socrata returned 0 rows"
        if filters:
            warns.append(f"{msg} — your filters may be too narrow.")
        else:
            fatal.append(msg)

    # Fatal #2: no duplicate uid
    dup_mask = gdf.duplicated(subset=["uid"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        fatal.append(f"duplicate entrance uid rows: {n_dup}")

    # Fatal #3: every row has a non-null lat/lon (already filtered by normalize,
    # but assert as a sanity check).
    null_coords = gdf["latitude"].isna() | gdf["longitude"].isna()
    n_null_coords = int(null_coords.sum())
    if n_null_coords > 0:
        fatal.append(f"rows with null lat/lon: {n_null_coords}")

    # Fatal #4: no null entrance_type
    null_type = gdf["entrance_type"].isna() | (gdf["entrance_type"] == "")
    n_null_type = int(null_type.sum())
    if n_null_type > 0:
        fatal.append(f"rows with null entrance_type: {n_null_type}")

    # Fatal #5: no null facname (every entrance must have a name)
    null_name = gdf["facname"].isna() | (gdf["facname"] == "")
    n_null_name = int(null_name.sum())
    if n_null_name > 0:
        fatal.append(f"rows with null facname: {n_null_name}")

    # Fatal #6: all geometries valid + non-empty
    is_valid = gdf.geometry.is_valid & gdf.geometry.notna() & ~gdf.geometry.is_empty
    n_invalid = int((~is_valid).sum())
    if n_invalid > 0:
        fatal.append(f"invalid / empty / null buffered geometries: {n_invalid}")

    # Fatal #7: all inside NYC bbox
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

    # Fatal #8: coverage non-empty + valid
    valid_geoms = gdf.loc[is_valid, "geometry"].tolist() if total > 0 else []
    coverage = unary_union(valid_geoms) if valid_geoms else MultiPolygon()
    coverage_valid = coverage is not None and not coverage.is_empty and coverage.is_valid
    if total > 0 and not coverage_valid:
        fatal.append("coverage (unary_union of entrances) is empty or invalid")

    # ---- warn metrics ---------------------------------------------------
    metrics["per_entrance_type"] = (
        gdf.groupby("entrance_type", dropna=False).size()
           .sort_values(ascending=False).to_dict()
    )
    metrics["per_division"] = (
        gdf.groupby("division", dropna=False).size().to_dict()
    )
    metrics["per_borough"] = (
        gdf.groupby("borough", dropna=False).size().to_dict()
    )

    # entries-per-station distribution
    if total > 0 and metrics["unique_stations"] > 0:
        per_station = gdf.groupby("station_id").size()
        metrics["entrances_per_station_p50"] = float(per_station.median())
        metrics["entrances_per_station_p95"] = float(per_station.quantile(0.95))
        metrics["entrances_per_station_max"] = int(per_station.max())
        metrics["entrances_per_station_mean"] = float(per_station.mean())

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
        "uid": gdf["uid"].astype(str),
        "facname": gdf["facname"].astype(str),
        "station_id": gdf["station_id"].astype(str),
        "entrance_type": gdf["entrance_type"].astype(str),
        "division": gdf["division"].astype(str),
        "borough": gdf["borough"].astype(str),
        "chk_unique_uid": (~dup_mask).astype(bool),
        "chk_coords_not_null": (~null_coords).astype(bool),
        "chk_entrance_type_not_null": (~null_type).astype(bool),
        "chk_facname_not_null": (~null_name).astype(bool),
        "chk_geometry_valid": is_valid.astype(bool),
        "chk_geometry_in_nyc_bbox": in_bbox.astype(bool) if total > 0 else in_bbox,
    }
    pl.DataFrame({k: v.tolist() for k, v in report_cols.items()}).write_parquet(report_path)

    lines = _build_summary_md(
        total=total, filters=filters, funnel=funnel, pagination=pagination,
        buffer_ft=buffer_ft,
        fatal=fatal, warns=warns, metrics=metrics,
        n_dup=n_dup, n_null_coords=n_null_coords, n_null_type=n_null_type,
        n_null_name=n_null_name,
        n_invalid=n_invalid, n_out_bbox=n_out_bbox, coverage_valid=coverage_valid,
    )
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    for m in warns:
        log.warning("validation: %s", m)
    for m in fatal:
        log.error("validation: FATAL %s", m)

    if fatal:
        raise SubwayValidationError("; ".join(fatal))

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
    fatal, warns, metrics, n_dup, n_null_coords, n_null_type, n_null_name,
    n_invalid, n_out_bbox, coverage_valid,
) -> list[str]:
    now = dt.datetime.now(tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# Subway entrances curation — validation summary\n"]
    lines.append(f"- Buffer: **{buffer_ft:.0f} ft** around each entrance point (EPSG:2263)")
    lines.append(f"- Publishable rows (one per entrance): **{total:,}**")
    lines.append(f"- Unique stations: **{metrics.get('unique_stations', 0):,}** "
                 f"(complexes: {metrics.get('unique_complexes', 0):,})")
    lines.append(f"- Built at: {now}")
    if filters:
        lines.append("- Applied filters:")
        for level, vals in filters.items():
            if vals:
                preview = ", ".join(vals[:5]) + (f"  (+{len(vals)-5} more)" if len(vals) > 5 else "")
                lines.append(f"    - `{level}`: {preview}")
    else:
        lines.append("- Applied filters: **none** (full subway entrances pull)")
    lines.append("")

    lines.append("## Fatal checks")
    lines.append("")
    lines.append("| # | Check | Status |")
    lines.append("|---|-------|--------|")
    lines.append(f"| 1 | Socrata returned ≥1 row | "
                 f"{'PASS' if funnel.get('raw_rows', 0) > 0 else 'FAIL'} |")
    lines.append(f"| 2 | Unique entrance `uid` | {'PASS' if n_dup == 0 else f'{n_dup} dupes'} |")
    lines.append(f"| 3 | All rows have non-null lat/lon | "
                 f"{'PASS' if n_null_coords == 0 else f'{n_null_coords} null'} |")
    lines.append(f"| 4 | All rows have non-null `entrance_type` | "
                 f"{'PASS' if n_null_type == 0 else f'{n_null_type} null'} |")
    lines.append(f"| 5 | All rows have non-null `facname` | "
                 f"{'PASS' if n_null_name == 0 else f'{n_null_name} null'} |")
    lines.append(f"| 6 | All buffered geometries valid | "
                 f"{'PASS' if n_invalid == 0 else f'{n_invalid} invalid'} |")
    lines.append(f"| 7 | All geometries inside NYC bbox | "
                 f"{'PASS' if n_out_bbox == 0 else f'{n_out_bbox} outside'} |")
    lines.append(f"| 8 | `coverage` non-empty + valid | "
                 f"{'PASS' if coverage_valid else 'FAIL'} |")
    lines.append("")

    lines.append("## Warn checks")
    lines.append("")
    lines.append(f"- Coverage: **{metrics.get('coverage_area_km2', 0):.3f} km²** "
                 f"({metrics.get('coverage_pct_of_nyc', 0):.3f}% of NYC land)")
    if "entrances_per_station_p50" in metrics:
        lines.append(
            f"- Entrances per station — "
            f"mean {metrics['entrances_per_station_mean']:.1f}, "
            f"p50 {metrics['entrances_per_station_p50']:.0f}, "
            f"p95 {metrics['entrances_per_station_p95']:.0f}, "
            f"max {metrics['entrances_per_station_max']:,}"
        )
    for w in warns:
        lines.append(f"- ⚠️  {w}")
    lines.append("")

    lines.append("## Dropped-row funnel")
    lines.append("")
    lines.append("| Step | Rows |")
    lines.append("|------|-----:|")
    for k, v in funnel.items():
        lines.append(f"| {k} | {int(v):,} |")
    lines.append("")

    lines.append("## Socrata pagination")
    lines.append("")
    lines.append("| rows | pages | last_page | truncated_likely |")
    lines.append("|-----:|------:|----------:|:-----------------|")
    p = pagination.get("subway", {})
    page_rows = p.get("page_rows", [])
    last = page_rows[-1] if page_rows else 0
    lines.append(f"| {p.get('total_rows', 0):,} | {p.get('pages', 0)} | "
                 f"{last:,} | {'YES' if p.get('truncated_likely') else 'no'} |")
    lines.append("")

    lines.append("## Per `entrance_type`")
    lines.append("")
    lines.append("| entrance_type | rows |")
    lines.append("|---------------|-----:|")
    for t, n in metrics.get("per_entrance_type", {}).items():
        lines.append(f"| {t} | {int(n):,} |")
    lines.append("")

    lines.append("## Per `division`")
    lines.append("")
    lines.append("| division | rows |")
    lines.append("|----------|-----:|")
    for d, n in metrics.get("per_division", {}).items():
        lines.append(f"| {d} | {int(n):,} |")
    lines.append("")

    lines.append("## Per borough")
    lines.append("")
    lines.append("| borough | rows |")
    lines.append("|---------|-----:|")
    for b, n in metrics.get("per_borough", {}).items():
        lines.append(f"| {b} | {int(n):,} |")
    lines.append("")

    if fatal:
        lines.append("## Fatal violations (publication blocked)")
        lines.append("")
        for m in fatal:
            lines.append(f"- {m}")
        lines.append("")
    return lines
