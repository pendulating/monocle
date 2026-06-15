"""Unify per-stage detection parquets into a single pseudo-label parquet.

Reads every parquet from ``context.inputs`` (each produced by an upstream
detector stage with the standard ``DETECTION_COLUMNS`` schema),
concatenates them, and writes a single output parquet partitioned by
``(dataset, face)`` so downstream queries can prune by borough / camera
direction.

Sentinel rows: each upstream stage already emits a row with
``class=None`` for images that produced zero detections. After
concatenation, an image may appear as a sentinel from one stage and as
real detections from another (e.g., a person was found but no plate);
in that case we drop the sentinels for that image. Images that are
sentinels in *all* upstream stages keep exactly one sentinel row.
"""

from __future__ import annotations

import os
import time
from typing import Dict

import pandas as pd

from dagspaces.common.orchestrator import StageExecutionContext, StageResult
from dagspaces.common.runners.base import StageRunner

from ._common import DETECTION_COLUMNS, ensure_output_dir


def _coerce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any missing DETECTION_COLUMNS as NaN and drop unexpected ones."""
    for col in DETECTION_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[DETECTION_COLUMNS]


def _drop_redundant_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """For each sample_id, keep sentinels only if there are no real detections."""
    if df.empty:
        return df
    has_real = df[df["class"].notna()]["sample_id"].unique()
    keep = (df["class"].notna()) | (~df["sample_id"].isin(has_real))
    out = df[keep].copy()
    # If multiple sentinels for the same sample_id (one per stage), keep one.
    sentinels = out[out["class"].isna()]
    if not sentinels.empty:
        first_sentinel_idx = sentinels.drop_duplicates(subset=["sample_id"]).index
        out = pd.concat(
            [out[out["class"].notna()], out.loc[first_sentinel_idx]],
            ignore_index=True,
        )
    return out


class UnifyRunner(StageRunner):
    stage_name = "unify"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        out_path = ensure_output_dir(context.output_paths, "detections")
        inputs: Dict[str, str] = dict(context.inputs)

        if not inputs:
            print("[unify] no upstream inputs supplied")
            empty = pd.DataFrame(columns=DETECTION_COLUMNS)
            empty.to_parquet(out_path, index=False)
            return StageResult(outputs={"detections": out_path}, metadata={"rows": 0})

        t0 = time.time()
        frames = []
        per_source_rows: Dict[str, int] = {}
        for source_key, path in inputs.items():
            if not path:
                continue
            if not os.path.exists(path):
                print(f"[unify] missing input '{source_key}' at {path}; skipping")
                continue
            df = pd.read_parquet(path)
            df = _coerce_columns(df)
            df["source"] = source_key
            frames.append(df)
            per_source_rows[source_key] = len(df)
            print(f"[unify] loaded '{source_key}' ({len(df)} rows) from {path}")

        if not frames:
            empty = pd.DataFrame(columns=DETECTION_COLUMNS)
            empty.to_parquet(out_path, index=False)
            return StageResult(outputs={"detections": out_path}, metadata={"rows": 0})

        merged = pd.concat(frames, ignore_index=True)
        before_dedup = len(merged)
        merged = _drop_redundant_sentinels(merged.drop(columns=["source"]))

        # Hive-partition by dataset/face. ``face`` is the cube-face label
        # (F/B/L/R/U/D) — distinct from the per-detection ``class`` column.
        partition_cols = [c for c in ("dataset", "face") if c in merged.columns]
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        if partition_cols and merged[partition_cols].notna().all().all():
            # Use the parquet path as a directory root for partitioned output.
            root_dir = out_path[:-len(".parquet")] if out_path.endswith(".parquet") else out_path
            os.makedirs(root_dir, exist_ok=True)
            merged.to_parquet(
                root_dir,
                index=False,
                partition_cols=partition_cols,
            )
            # Also write a single flat file at out_path for callers that
            # prefer one parquet (e.g., the OPF training Dataset).
            merged.to_parquet(out_path, index=False)
        else:
            merged.to_parquet(out_path, index=False)

        elapsed = time.time() - t0
        n_real = int(merged["class"].notna().sum())
        n_sentinel = int(merged["class"].isna().sum())
        n_images = merged["sample_id"].nunique()
        print(
            f"[unify] {n_images} images, {n_real} real detections, "
            f"{n_sentinel} sentinels (was {before_dedup} pre-dedup) "
            f"in {elapsed:.1f}s -> {out_path}"
        )
        return StageResult(
            outputs={"detections": out_path},
            metadata={
                "rows": int(len(merged)),
                "real_detections": n_real,
                "sentinels": n_sentinel,
                "images": n_images,
                "per_source_rows": per_source_rows,
            },
        )
