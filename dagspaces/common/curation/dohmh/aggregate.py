"""Aggregate inspection-level ``restaurants.parquet`` → one row per CAMIS.

The build step (:mod:`.dohmh_restaurants`) preserves multi-year inspection
history: one row per ``(camis, inspection_date)``. This module collapses
that history to **one row per restaurant**, picking the most-recent
non-placeholder inspection's metadata as the row's "current state."

Aggregation is deliberately a separate, **opt-in** step. Two reasons:

1. Downstream consumers may want the full inspection history (e.g. plotting
   grade trajectories, computing inspection cadence). Collapsing eagerly
   destroys that.
2. The downstream ``materialize-cyclomedia`` tool requires a unique
   ``unit_uid`` per row (otherwise the spatial join produces N attribution
   rows per camera point). Aggregated output is what feeds that step.

CLI: ``python -m dagspaces.common.curation aggregate-restaurants ...``.

Output schema is the FacDB-shaped one-row-per-restaurant view, mirroring
the column names downstream tools (``materialize-cyclomedia``,
``filter-facing``, ``sample-images``) already understand:

    permit_id, uid          str — = camis
    camis                   str
    facname, dba            str
    address, building, street, city, zipcode, borough  str
    cuisine_description     str
    facdomain, facgroup, facsubgrp, factype  str  (FacDB-shaped aliases)

    last_inspection_date    timestamp
    last_inspection_type, last_action, last_grade, last_critical_flag,
    last_violation_code, last_violation_description  str
    last_score              f64

    n_inspections           int   — non-placeholder inspections seen
    n_placeholder_inspections int — placeholder rows for this CAMIS
    first_inspection_date   timestamp
    n_grade_a, n_grade_b, n_grade_c, n_grade_other int

    bin, bbl, latitude, longitude, raw_latitude, raw_longitude
    nta, community_board, council_district, census_tract
    datasource              str

    geom_source, match_dist_ft, geom_wkb     — copied verbatim from input
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import polars as pl

__all__ = ["aggregate_restaurants", "AggregateResult"]

log = logging.getLogger(__name__)


@dataclass
class AggregateResult:
    input_parquet: str
    output_parquet: str
    manifest_path: str
    in_rows: int
    out_rows: int
    unique_camis: int
    n_only_placeholder: int
    elapsed_s: float = 0.0


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def aggregate_restaurants(
    input_parquet: str,
    output_parquet: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> AggregateResult:
    """Collapse inspection-level rows to one row per CAMIS.

    Args:
        input_parquet: Path to ``restaurants.parquet`` written by
            :func:`.dohmh_restaurants.build`.
        output_parquet: Output path. Defaults to
            ``<input_dir>/restaurants_aggregated.parquet`` next to the input.
        overwrite: When False (default), existing output_parquet raises
            :class:`FileExistsError`.
    """
    t0 = time.monotonic()
    if not os.path.isfile(input_parquet):
        raise FileNotFoundError(f"input parquet not found: {input_parquet}")

    if output_parquet is None:
        output_parquet = os.path.join(
            os.path.dirname(os.path.abspath(input_parquet)),
            "restaurants_aggregated.parquet",
        )
    if os.path.isfile(output_parquet) and not overwrite:
        raise FileExistsError(
            f"output parquet exists: {output_parquet} — pass --overwrite to replace"
        )

    df = pl.read_parquet(input_parquet)
    in_rows = df.height
    if in_rows == 0:
        raise ValueError(f"{input_parquet} is empty — nothing to aggregate")

    required = {"camis", "facname", "geom_wkb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{input_parquet} missing required columns {sorted(missing)} — "
            "expected output of dohmh-restaurants build"
        )

    # ---- 1. compute per-row sort keys for "most-recent real inspection wins"
    # is_placeholder_inspection is a build-time column. Older builds may
    # lack it; fall back to inspection_date IS NULL as the placeholder signal.
    if "is_placeholder_inspection" in df.columns:
        is_placeholder = pl.col("is_placeholder_inspection").cast(pl.Boolean)
    elif "inspection_date" in df.columns:
        is_placeholder = pl.col("inspection_date").is_null()
    else:
        raise ValueError(
            f"{input_parquet} has neither is_placeholder_inspection nor "
            "inspection_date columns; can't tell real vs placeholder rows"
        )

    df = df.with_columns(
        is_placeholder.alias("_is_placeholder"),
        pl.when(is_placeholder).then(0).otherwise(1).alias("_real_rank"),
    )

    # ---- 2. precompute per-camis aggregate stats -----------------------
    grade = pl.col("grade") if "grade" in df.columns else pl.lit(None, dtype=pl.Utf8)
    grade_upper = grade.cast(pl.Utf8).str.to_uppercase().str.strip_chars()
    insp_date = (
        pl.col("inspection_date") if "inspection_date" in df.columns
        else pl.lit(None, dtype=pl.Datetime("us"))
    )

    stats = df.group_by("camis").agg(
        (~pl.col("_is_placeholder")).sum().alias("n_inspections"),
        pl.col("_is_placeholder").sum().alias("n_placeholder_inspections"),
        pl.when(~pl.col("_is_placeholder")).then(insp_date).otherwise(None)
          .min().alias("first_inspection_date"),
        pl.when(~pl.col("_is_placeholder")).then(insp_date).otherwise(None)
          .max().alias("last_inspection_date_agg"),
        ((~pl.col("_is_placeholder")) & (grade_upper == pl.lit("A"))).sum().alias("n_grade_a"),
        ((~pl.col("_is_placeholder")) & (grade_upper == pl.lit("B"))).sum().alias("n_grade_b"),
        ((~pl.col("_is_placeholder")) & (grade_upper == pl.lit("C"))).sum().alias("n_grade_c"),
        (
            (~pl.col("_is_placeholder"))
            & ~grade_upper.is_in(["A", "B", "C"])
            & grade_upper.is_not_null()
        ).sum().alias("n_grade_other"),
    )

    # ---- 3. pick the "headline row" per CAMIS --------------------------
    # Sort: real inspections > placeholder; latest inspection_date; Critical >
    # Non-Critical (so the carried violation_* columns are substantive).
    crit_rank = (
        pl.when(pl.col("critical_flag").cast(pl.Utf8).str.to_uppercase() == "CRITICAL")
          .then(1).otherwise(0)
        if "critical_flag" in df.columns
        else pl.lit(0)
    )
    df = df.with_columns(crit_rank.alias("_critical_rank"))
    df_sorted = df.sort(
        ["camis", "_real_rank", "inspection_date" if "inspection_date" in df.columns else "_real_rank", "_critical_rank"],
        descending=[False, True, True, True],
        nulls_last=True,
    )
    headline = df_sorted.group_by("camis", maintain_order=True).agg(pl.all().first())

    # ---- 4. rename "this row's inspection metadata" to last_* ----------
    rename_map = {
        "inspection_date": "last_inspection_date",
        "inspection_type": "last_inspection_type",
        "action": "last_action",
        "grade": "last_grade",
        "score": "last_score",
        "critical_flag": "last_critical_flag",
        "violation_code": "last_violation_code",
        "violation_description": "last_violation_description",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in headline.columns}
    headline = headline.rename(rename_map)

    # last_inspection_date / last_grade for "only placeholder" CAMIS:
    # leave the date null but flag the grade.
    if "last_inspection_date" in headline.columns:
        headline = headline.with_columns(
            pl.when(pl.col("_is_placeholder"))
              .then(pl.lit(None, dtype=pl.Datetime("us")))
              .otherwise(pl.col("last_inspection_date"))
              .alias("last_inspection_date"),
        )
    if "last_grade" in headline.columns:
        headline = headline.with_columns(
            pl.when(pl.col("_is_placeholder"))
              .then(pl.lit("NOT YET INSPECTED"))
              .otherwise(pl.col("last_grade"))
              .alias("last_grade"),
        )
    if "last_score" in headline.columns:
        headline = headline.with_columns(
            pl.when(pl.col("_is_placeholder"))
              .then(pl.lit(None, dtype=pl.Float64))
              .otherwise(pl.col("last_score"))
              .alias("last_score"),
        )

    # ---- 5. join + drop helper columns + drop redundant sample_id ------
    out = headline.join(stats, on="camis", how="left").drop(
        [c for c in (
            "_is_placeholder", "_real_rank", "_critical_rank",
            "is_placeholder_inspection", "sample_id",
            "last_inspection_date_agg",
        ) if c in headline.columns or c in stats.columns]
    )

    # Prefer the aggregate's max date as canonical last_inspection_date
    # (the headline row's date should already match, but be explicit).
    if (
        "last_inspection_date" in out.columns
        and "last_inspection_date_agg" in out.columns
    ):
        out = out.with_columns(
            pl.coalesce("last_inspection_date_agg", "last_inspection_date")
              .alias("last_inspection_date"),
        ).drop("last_inspection_date_agg")

    n_only_placeholder = int(out.filter(
        (pl.col("n_inspections") == 0) & (pl.col("n_placeholder_inspections") > 0)
    ).height)

    # ---- 6. write output + manifest ------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)) or ".", exist_ok=True)
    out.write_parquet(output_parquet)
    elapsed = time.monotonic() - t0

    manifest_path = output_parquet.replace(".parquet", "_manifest.json")
    _atomic_write_json(manifest_path, {
        "tool": "aggregate-restaurants",
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "input_parquet": os.path.abspath(input_parquet),
        "output_parquet": os.path.abspath(output_parquet),
        "in_rows": in_rows,
        "out_rows": int(out.height),
        "unique_camis": int(out.height),
        "camis_only_placeholder": n_only_placeholder,
        "elapsed_s": round(elapsed, 2),
    })

    log.info(
        "aggregate: %d → %d rows (one per CAMIS) in %.1fs "
        "(%d CAMIS only had placeholder rows)",
        in_rows, out.height, elapsed, n_only_placeholder,
    )

    return AggregateResult(
        input_parquet=os.path.abspath(input_parquet),
        output_parquet=os.path.abspath(output_parquet),
        manifest_path=manifest_path,
        in_rows=in_rows,
        out_rows=int(out.height),
        unique_camis=int(out.height),
        n_only_placeholder=n_only_placeholder,
        elapsed_s=elapsed,
    )
