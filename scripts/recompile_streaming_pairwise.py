"""Recompile a pairwise-VQA final parquet from leftover streaming chunks.

When a urbanpairvqa run is interrupted (or the monitor dies) after the DP
workers have written their per-chunk parquets under
``outputs/pairwise/streaming/urbanpairvqa_pairwise/`` but before the
orchestrator wrote the consolidated ``<dataset>_mvp_<ts>.parquet``, this
script rebuilds that final artifact.

The streaming chunks hold the 15 *raw* stage columns (pair_id, is_swapped,
answer, ...). The consolidated output additionally carries the 5 derived
label/score columns (presented_answer/label/score, relative_label/score).
Those are produced by ``_derive_labels`` in the pairwise stage, applied to the
raw ``answer`` JSON string — so recompiling is exactly:

    concat(chunks, in true row order)  ->  _derive_labels(df)

This reproduces the consolidated output byte-for-column-identically with the
orchestrator's own postprocessing (including its label-canonicalization
behavior), so the recompiled restaurants/schools parquets are mutually
consistent.

Usage:
    python scripts/recompile_streaming_pairwise.py \
        --pairwise-dir multirun/2026-06-08_URBANPAIRVQA/12-21-06/0/outputs/pairwise \
        --out-name restaurants_mvp_20260608_122106.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

from dagspaces.urbanpairvqa.stages.pairwise_vqa import _derive_labels

# Canonical column order of the consolidated output (15 raw + 5 derived).
_FINAL_COLUMNS = [
    "pair_id", "sample_id", "canonical_pair_id", "repeat_idx",
    "sample_id_a", "sample_id_b", "image_path_a", "image_path_b",
    "presented_left_path", "presented_right_path", "presented_order",
    "is_swapped", "answer", "model_response", "model_reasoning",
    "presented_answer", "presented_label", "presented_score",
    "relative_label", "relative_score",
]

_ROW_RANGE = re.compile(r"rows(\d+)-(\d+)")


def _chunk_sort_key(path: str):
    """Sort chunks by their starting global row index (from the filename),
    falling back to lexicographic order."""
    m = _ROW_RANGE.search(os.path.basename(path))
    return (int(m.group(1)) if m else 1 << 62, os.path.basename(path))


def recompile(pairwise_dir: str, out_name: str | None) -> str:
    streaming = os.path.join(pairwise_dir, "streaming", "urbanpairvqa_pairwise")
    chunks = sorted(glob.glob(os.path.join(streaming, "*.parquet")), key=_chunk_sort_key)
    if not chunks:
        raise SystemExit(f"No streaming chunks found under {streaming}")

    frames = [pd.read_parquet(c) for c in chunks]
    raw = pd.concat(frames, ignore_index=True)
    print(f"Concatenated {len(chunks)} chunks -> {len(raw)} rows")

    out = _derive_labels(raw)
    # Reorder to the canonical layout; keep any extras at the end.
    ordered = [c for c in _FINAL_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in _FINAL_COLUMNS]
    out = out[ordered + extras]

    if out_name is None:
        out_name = "pairwise_recompiled.parquet"
    out_path = os.path.join(pairwise_dir, out_name)
    out.to_parquet(out_path, index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} cols -> {out_path}")
    print("relative_label distribution:")
    print(out["relative_label"].value_counts())
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairwise-dir", required=True,
                    help="Path to the run's outputs/pairwise directory")
    ap.add_argument("--out-name", default=None,
                    help="Filename for the consolidated parquet (written inside --pairwise-dir)")
    args = ap.parse_args()
    recompile(args.pairwise_dir, args.out_name)


if __name__ == "__main__":
    main()
