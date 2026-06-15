"""Sanity checks for the built catalog.

Eleven invariants, split by severity:

  Fatal (1-4): indexer will not publish a catalog that violates these.
  Warn  (5-11): logged + shown in summary.md; catalog is still written.

Called at the tail of `build_catalog` with the fully joined rows.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import polars as pl

from .schema import NYC_BBOX

__all__ = ["run_validation", "ValidationError"]

log = logging.getLogger(__name__)


class ValidationError(AssertionError):
    """Raised when a fatal invariant is violated."""


def _count(df: pl.DataFrame, expr: pl.Expr) -> int:
    return int(df.select(expr.sum()).item() or 0)


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "  n/a"
    return f"{100.0 * n / d:6.2f}%"


def run_validation(
    df: pl.DataFrame,
    output_root: str,
    raw_root: str,
) -> str:
    """Run all checks. Emit validation_report.parquet + summary.md.

    Returns the summary.md path.
    """
    total = df.height
    if total == 0:
        raise ValidationError("catalog is empty; refusing to publish")

    # Build a per-row boolean column for each check. Then aggregate for summary.
    checks = df.with_columns(
        # Fatal 1: manifest.imageId == basename(recording_path). `recording_id`
        # is manifest imageId when manifest_ok else dirname; if manifest_ok and
        # recording_id != dirname, flag.
        # The walker stores recording_dir as the basename; we joined it into the
        # catalog implicitly via `recording_id` fallback. For this check we
        # re-derive from image_path.
        pl.col("image_path").str.split("/").list.get(-3).alias("_rec_from_path"),
    ).with_columns(
        # Fatal 2+3: sample_id shape + face matches filename.
        (pl.col("sample_id") == pl.col("recording_id") + pl.lit("_") + pl.col("face").cast(pl.Utf8))
            .alias("chk_sample_id_shape"),
        pl.col("face").cast(pl.Utf8).is_in(list("FBLRUD")).alias("chk_face_in_set"),
        (pl.col("image_path").str.split("/").list.get(-1).str.slice(0, 1) == pl.col("face").cast(pl.Utf8))
            .alias("chk_face_matches_filename"),
        # Fatal 1: if manifest_ok then recording_id == _rec_from_path.
        (~pl.col("manifest_ok").fill_null(False) | (pl.col("recording_id") == pl.col("_rec_from_path")))
            .alias("chk_imageid_vs_dirname"),
        # Warn 6: NYC bbox
        (
            (pl.col("latitude").is_between(NYC_BBOX["lat_min"], NYC_BBOX["lat_max"])) &
            (pl.col("longitude").is_between(NYC_BBOX["lon_min"], NYC_BBOX["lon_max"]))
        ).fill_null(False).alias("chk_coord_in_nyc_bbox"),
        # Warn 7: bearing is:
        #   - NULL on U/D faces (always)
        #   - [0,360) on F/B/L/R when recorderDirection was available
        #   - NULL on F/B/L/R when recorderDirection was missing (can't compute)
        # Anything else is a violation.
        (
            (pl.col("face").cast(pl.Utf8).is_in(["U", "D"]) & pl.col("bearing").is_null()) |
            (
                ~pl.col("face").cast(pl.Utf8).is_in(["U", "D"]) &
                (
                    pl.col("bearing").is_between(0.0, 360.0, closed="left") |
                    (pl.col("bearing").is_null() & pl.col("recorderDirection").is_null())
                )
            )
        ).alias("chk_bearing_range"),
        # Warn 8: file size plausibility (>50KB)
        (pl.col("file_size") > 50_000).alias("chk_file_size_ok"),
        # Warn 10: path escape
        pl.col("image_path").str.starts_with(os.path.abspath(raw_root) + os.sep).alias("chk_path_in_raw_root"),
    )

    # Fatal 4: one row per (dataset, recording_id, face). `dataset` partitions
    # the raw tree — the same recording_id legitimately appears in two borough
    # dirs when their pull bboxes overlap at the edge, so uniqueness must be
    # scoped to a dataset.
    dup_count = (
        checks.group_by(["dataset", "recording_id", "face"])
              .len()
              .filter(pl.col("len") > 1)
              .height
    )

    # Warn 11: cross-dataset overlap. Same (recording_id, face) appearing in
    # more than one dataset means the underlying image file is present in
    # multiple raw borough dirs (bbox edge overlap). Not fatal — callers can
    # dedupe at query time — but surface the count so it stays visible.
    cross_ds_overlap = (
        checks.group_by(["recording_id", "face"])
              .agg(pl.col("dataset").n_unique().alias("n_ds"))
              .filter(pl.col("n_ds") > 1)
              .height
    )

    # Warn 5: catalog_hit rate per dataset.
    ds_catalog_hit = (
        checks.group_by("dataset")
              .agg(
                  pl.len().alias("rows"),
                  pl.col("catalog_hit").fill_null(False).sum().alias("hits"),
              )
              .with_columns((pl.col("hits") / pl.col("rows")).alias("rate"))
              .sort("dataset")
    )

    # Warn 9: every (dataset, year) partition non-empty. Trivially true since
    # we only wrote partitions we computed rows for, but assert anyway.
    ds_year_nonempty = (
        checks.group_by(["dataset", "year"])
              .len()
              .filter(pl.col("len") == 0)
              .height == 0
    )

    # --- Aggregate counters for summary ---
    n_manifest_ok = _count(checks, pl.col("manifest_ok").fill_null(False).cast(pl.Int64))
    n_sample_id = _count(checks, pl.col("chk_sample_id_shape").cast(pl.Int64))
    n_face_in_set = _count(checks, pl.col("chk_face_in_set").cast(pl.Int64))
    n_face_matches = _count(checks, pl.col("chk_face_matches_filename").cast(pl.Int64))
    n_imageid_ok = _count(checks, pl.col("chk_imageid_vs_dirname").cast(pl.Int64))
    n_catalog_hit = _count(checks, pl.col("catalog_hit").fill_null(False).cast(pl.Int64))
    n_nyc_bbox = _count(checks, pl.col("chk_coord_in_nyc_bbox").cast(pl.Int64))
    n_bearing_ok = _count(checks, pl.col("chk_bearing_range").cast(pl.Int64))
    n_file_size_ok = _count(checks, pl.col("chk_file_size_ok").cast(pl.Int64))
    n_path_in_root = _count(checks, pl.col("chk_path_in_raw_root").cast(pl.Int64))

    # --- Enforce fatal checks ---
    fatal_msgs: list[str] = []
    if n_sample_id != total:
        fatal_msgs.append(
            f"sample_id shape mismatch: {total - n_sample_id}/{total} rows"
        )
    if n_face_in_set != total:
        fatal_msgs.append(
            f"face outside {{F,B,L,R,U,D}}: {total - n_face_in_set}/{total} rows"
        )
    if n_face_matches != total:
        fatal_msgs.append(
            f"face filename mismatch: {total - n_face_matches}/{total} rows"
        )
    # fatal 1 (imageid_vs_dirname) only enforced among rows with manifest_ok
    mf_ok_total = _count(checks, pl.col("manifest_ok").fill_null(False).cast(pl.Int64))
    mf_ok_good = _count(
        checks,
        (pl.col("manifest_ok").fill_null(False) & pl.col("chk_imageid_vs_dirname")).cast(pl.Int64),
    )
    if mf_ok_total != mf_ok_good:
        fatal_msgs.append(
            f"manifest imageId vs dirname mismatch: {mf_ok_total - mf_ok_good}/{mf_ok_total} (manifest_ok rows)"
        )
    if dup_count > 0:
        fatal_msgs.append(f"duplicate (dataset, recording_id, face) pairs: {dup_count}")

    # --- Write validation_report.parquet ---
    report_cols = [
        "sample_id", "recording_id", "face", "dataset", "group", "borough",
        "manifest_ok", "catalog_hit",
        "chk_sample_id_shape", "chk_face_in_set", "chk_face_matches_filename",
        "chk_imageid_vs_dirname", "chk_coord_in_nyc_bbox",
        "chk_bearing_range", "chk_file_size_ok", "chk_path_in_raw_root",
    ]
    report_df = checks.select([c for c in report_cols if c in checks.columns])
    report_df.write_parquet(os.path.join(output_root, "validation_report.parquet"))

    # --- Human-readable summary.md ---
    lines: list[str] = []
    lines.append(f"# Cyclomedia catalog validation summary\n")
    lines.append(f"- Total rows: **{total:,}**")
    lines.append(f"- Datasets: **{checks['dataset'].n_unique()}**")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| # | Check | Severity | Pass rate |")
    lines.append("|---|-------|----------|-----------|")
    lines.append(f"| 1 | `manifest.imageId == basename(recording_path)` | fatal | {_pct(mf_ok_good, mf_ok_total)} (of manifest_ok rows) |")
    lines.append(f"| 2 | `sample_id == {{recording_id}}_{{face}}` + face in set | fatal | {_pct(min(n_sample_id, n_face_in_set), total)} |")
    lines.append(f"| 3 | face letter matches filename | fatal | {_pct(n_face_matches, total)} |")
    lines.append(f"| 4 | unique `(dataset, recording_id, face)` | fatal | {'PASS' if dup_count == 0 else f'{dup_count} dupes'} |")
    lines.append(f"| 5 | WFS catalog join hit rate | warn (≥95%) | {_pct(n_catalog_hit, total)} overall |")
    lines.append(f"| 6 | lat/lon in NYC bbox | warn | {_pct(n_nyc_bbox, total)} |")
    lines.append(f"| 7 | bearing range + NULL on U/D | warn | {_pct(n_bearing_ok, total)} |")
    lines.append(f"| 8 | file_size > 50 KB | warn | {_pct(n_file_size_ok, total)} |")
    lines.append(f"| 9 | every `(dataset, year)` partition non-empty | warn | {'PASS' if ds_year_nonempty else 'FAIL'} |")
    lines.append(f"| 10 | image_path starts with raw_root | warn | {_pct(n_path_in_root, total)} |")
    lines.append(f"| 11 | cross-dataset `(recording_id, face)` overlap | warn | {cross_ds_overlap:,} pairs in ≥2 datasets |")
    lines.append("")
    lines.append("## Per-dataset catalog hit rate")
    lines.append("")
    lines.append("| dataset | rows | catalog hits | rate |")
    lines.append("|---------|-----:|-------------:|-----:|")
    for row in ds_catalog_hit.to_dicts():
        lines.append(f"| {row['dataset']} | {row['rows']:,} | {row['hits']:,} | {row['rate']:.4f} |")
    lines.append("")
    if fatal_msgs:
        lines.append("## Fatal violations")
        for m in fatal_msgs:
            lines.append(f"- {m}")
        lines.append("")
    lines.append(f"manifest_ok rows: {n_manifest_ok:,} / {total:,} ({_pct(n_manifest_ok, total).strip()})")

    summary_path = os.path.join(output_root, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))

    # --- Log warns ---
    for row in ds_catalog_hit.to_dicts():
        rate = float(row["rate"]) if row["rows"] else 1.0
        if rate < 0.95:
            log.warning("validation: WFS catalog hit rate low for %s: %.2f%%", row["dataset"], 100 * rate)
    if n_nyc_bbox < total:
        log.warning("validation: %d/%d rows outside NYC bbox", total - n_nyc_bbox, total)
    if n_file_size_ok < total:
        log.warning("validation: %d/%d rows have suspicious file_size ≤ 50KB", total - n_file_size_ok, total)
    if n_path_in_root < total:
        log.warning("validation: %d/%d rows have image_path outside raw_root", total - n_path_in_root, total)
    if cross_ds_overlap > 0:
        log.warning(
            "validation: %d (recording_id, face) pairs appear in multiple datasets "
            "(borough bbox edge overlap)", cross_ds_overlap
        )

    # --- Raise on fatals ---
    if fatal_msgs:
        raise ValidationError("; ".join(fatal_msgs))

    return summary_path
