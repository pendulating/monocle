"""Per-restaurant inspection history from the DOHMH inspection table.

Why this module exists
----------------------
:mod:`.aggregate` collapses the inspection history to one row for each CAMIS.
It keeps the metadata of the most recent inspection under the ``last_*`` names.
``last_score`` is thus one draw of a noisy measurement, and a proxy that reads
it carries that noise.

This module reads the same inspection-level table and gives a HISTORY mean
instead. It writes a sidecar parquet keyed by ``camis``. It does not touch
``restaurants_aggregated.parquet``, and no existing column changes.

Why a sidecar and not more columns on the aggregate
---------------------------------------------------
``restaurants_aggregated.parquet`` is the input of 3 artifacts that already
exist: ``cyclomedia_near_restaurants.parquet``, the facing filter beside it,
and the pairs manifest of the restaurants case. A new column changes the schema
of that file and makes every consumer read it again. The sidecar adds the same
numbers with no risk to those bytes. A consumer joins it on ``camis``.

Warning: do not fold these columns into :func:`.aggregate_restaurants` until a
rebuild of the curation sub-dataset is scheduled.

Why the inspection type matters
-------------------------------
A DOHMH score counts violation points, and a LOW score is a CLEAN restaurant.
The city gives a re-inspection ONLY after an initial inspection fails. Thus a
re-inspection score is conditional on an earlier failure, and a mean over mixed
inspection types compares restaurants on different conditions.

This module restricts the mean to one inspection type. The default is
``Cycle Inspection / Initial Inspection``: the unconditional, routine visit that
every operating restaurant receives on a cycle.

| Inspection type | Rows in the 2026-04-27 pull |
|-----------------|----------------------------:|
| Cycle Inspection / Initial Inspection | 45,679 |
| Cycle Inspection / Re-inspection | 17,822 |
| Pre-permit (Operational) / Initial Inspection | 10,558 |
| Pre-permit (Operational) / Re-inspection | 3,232 |
| every other type | 5,826 |

Why the date floor
------------------
The source publishes a rolling window. The pull spans 2007-08-10 to
2026-04-25, but 98.6% of the rows fall on 2022-01-01 or later. The default
floor of 2019-01-01 drops 335 of the 45,679 cycle initial inspections (0.73%)
and 36 of the 21,573 restaurants. A pre-2019 inspection describes a restaurant
before the pandemic, and often a different operator.

Entry points:
- :func:`inspection_history` — programmatic, returns a frame
- :func:`build_inspection_history` — writes the sidecar parquet and a manifest
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import polars as pl

__all__ = [
    "inspection_history",
    "build_inspection_history",
    "InspectionHistoryResult",
    "CYCLE_INITIAL",
    "HISTORY_SINCE",
    "HISTORY_SIDECAR",
]

log = logging.getLogger(__name__)

#: The routine, unconditional inspection. Read the module docstring.
CYCLE_INITIAL = "Cycle Inspection / Initial Inspection"

#: The date floor of the history window, as an ISO date.
HISTORY_SINCE = "2019-01-01"

#: The default file name of the sidecar.
HISTORY_SIDECAR = "restaurants_inspection_history.parquet"


@dataclass
class InspectionHistoryResult:
    """What :func:`build_inspection_history` did."""

    input_parquet: str
    output_parquet: str
    manifest_path: str
    inspection_type: str
    since: Optional[str]
    in_rows: int
    kept_rows: int
    out_rows: int
    n_with_2_or_more: int
    elapsed_s: float = 0.0


def _atomic_write_json(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def inspection_history(
    input_parquet: str,
    *,
    inspection_type: str = CYCLE_INITIAL,
    since: Optional[str] = HISTORY_SINCE,
) -> pl.DataFrame:
    """Give the inspection history of each CAMIS as one row.

    Args:
        input_parquet: The inspection-level ``restaurants.parquet`` that
            :func:`.dohmh_restaurants.build` writes. It holds one row for each
            (camis, inspection_date).
        inspection_type: Keep only the inspections of this exact type. Pass
            None to keep every type, which mixes conditional and unconditional
            visits. Read the warning in the module docstring first.
        since: An ISO date. Drop an inspection before it. Pass None to keep the
            whole span of the file.

    Returns a frame with these columns:

    | Column | Holds |
    |--------|-------|
    | ``camis`` | The restaurant id |
    | ``history_mean_score`` | The mean violation score over the window |
    | ``history_median_score`` | The median violation score |
    | ``history_sd_score`` | The standard deviation, null under 2 inspections |
    | ``history_n_inspections`` | The count of inspections in the mean |
    | ``history_n_grade_a`` | The count of grade A results in the window |
    | ``history_first_date`` | The first inspection date in the window |
    | ``history_last_date`` | The last inspection date in the window |

    Warning: a LOW score is a CLEAN restaurant. This function applies no sign.
    """
    if not os.path.isfile(input_parquet):
        raise FileNotFoundError(f"input parquet not found: {input_parquet}")

    df = pl.read_parquet(
        input_parquet,
        columns=[
            c
            for c in (
                "camis",
                "inspection_date",
                "inspection_type",
                "score",
                "grade",
                "is_placeholder_inspection",
            )
        ],
    )
    in_rows = df.height
    if in_rows == 0:
        raise ValueError(f"{input_parquet} is empty — nothing to summarize")

    required = {"camis", "inspection_date", "inspection_type", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{input_parquet} missing required columns {sorted(missing)} — "
            "expected the inspection-level output of dohmh-restaurants build"
        )

    # A placeholder row stands for a restaurant with no inspection. It carries
    # no score, thus it never enters a mean.
    if "is_placeholder_inspection" in df.columns:
        df = df.filter(~pl.col("is_placeholder_inspection").cast(pl.Boolean))

    if inspection_type is not None:
        df = df.filter(pl.col("inspection_type") == pl.lit(inspection_type))
    if since is not None:
        df = df.filter(pl.col("inspection_date") >= pl.lit(since).str.to_datetime())

    df = df.filter(pl.col("score").is_not_null())

    grade = (
        pl.col("grade") if "grade" in df.columns else pl.lit(None, dtype=pl.Utf8)
    )
    grade_upper = grade.cast(pl.Utf8).str.to_uppercase().str.strip_chars()

    out = (
        df.group_by("camis")
        .agg(
            pl.col("score").mean().alias("history_mean_score"),
            pl.col("score").median().alias("history_median_score"),
            pl.col("score").std().alias("history_sd_score"),
            pl.len().alias("history_n_inspections"),
            (grade_upper == pl.lit("A")).sum().alias("history_n_grade_a"),
            pl.col("inspection_date").min().alias("history_first_date"),
            pl.col("inspection_date").max().alias("history_last_date"),
        )
        .sort("camis")
    )
    return out.with_columns(
        pl.col("history_n_inspections").cast(pl.Int64),
        pl.col("history_n_grade_a").cast(pl.Int64),
    )


def build_inspection_history(
    input_parquet: str,
    output_parquet: Optional[str] = None,
    *,
    inspection_type: str = CYCLE_INITIAL,
    since: Optional[str] = HISTORY_SINCE,
    overwrite: bool = False,
) -> InspectionHistoryResult:
    """Write the inspection-history sidecar beside the inspection table.

    Args:
        input_parquet: The inspection-level ``restaurants.parquet``.
        output_parquet: The output path. It defaults to
            ``<input_dir>/restaurants_inspection_history.parquet``.
        overwrite: When False, an existing output raises
            :class:`FileExistsError`.

    The manifest beside the parquet names the inspection type and the window.
    Read it before you read a number that came from this file.
    """
    t0 = time.monotonic()
    if output_parquet is None:
        output_parquet = os.path.join(
            os.path.dirname(os.path.abspath(input_parquet)), HISTORY_SIDECAR
        )
    if os.path.isfile(output_parquet) and not overwrite:
        raise FileExistsError(
            f"output parquet exists: {output_parquet} — pass overwrite=True to replace"
        )

    in_rows = pl.read_parquet(input_parquet, columns=["camis"]).height
    out = inspection_history(
        input_parquet, inspection_type=inspection_type, since=since
    )
    kept = int(out["history_n_inspections"].sum())
    n2 = int((out["history_n_inspections"] >= 2).sum())

    os.makedirs(os.path.dirname(os.path.abspath(output_parquet)) or ".", exist_ok=True)
    out.write_parquet(output_parquet)
    elapsed = time.monotonic() - t0

    manifest_path = output_parquet.replace(".parquet", "_manifest.json")
    _atomic_write_json(
        manifest_path,
        {
            "tool": "dohmh-inspection-history",
            "built_at": datetime.now(tz=timezone.utc).isoformat(),
            "input_parquet": os.path.abspath(input_parquet),
            "output_parquet": os.path.abspath(output_parquet),
            "inspection_type": inspection_type,
            "since": since,
            "in_rows": in_rows,
            "kept_inspections": kept,
            "out_rows": int(out.height),
            "camis_with_2_or_more": n2,
            "window_first": str(out["history_first_date"].min()),
            "window_last": str(out["history_last_date"].max()),
            "elapsed_s": round(elapsed, 2),
            "note": (
                "A sidecar. It adds no column to restaurants_aggregated.parquet. "
                "Join it on camis. A LOW score is a CLEAN restaurant."
            ),
        },
    )

    log.info(
        "inspection-history: %d rows → %d CAMIS (%d inspections of type %r "
        "since %s) in %.1fs",
        in_rows,
        out.height,
        kept,
        inspection_type,
        since,
        elapsed,
    )

    return InspectionHistoryResult(
        input_parquet=os.path.abspath(input_parquet),
        output_parquet=os.path.abspath(output_parquet),
        manifest_path=manifest_path,
        inspection_type=inspection_type,
        since=since,
        in_rows=in_rows,
        kept_rows=kept,
        out_rows=int(out.height),
        n_with_2_or_more=n2,
        elapsed_s=elapsed,
    )
