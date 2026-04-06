from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_sample_id(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _normalize_counterbalance_mode(value: Any) -> str:
    mode = str(value or "none").strip().lower()
    if mode not in {"none", "random", "balanced"}:
        return "none"
    return mode


def _build_pair_row(
    left: pd.Series,
    right: pd.Series,
    pair_id: str,
    canonical_pair_id: str,
    repeat_idx: int,
    swap_presented: bool,
    metadata_columns: List[str],
) -> Dict[str, Any]:
    presented_left_path = str(right["image_path"]) if swap_presented else str(left["image_path"])
    presented_right_path = str(left["image_path"]) if swap_presented else str(right["image_path"])
    presented_order = "B_then_A" if swap_presented else "A_then_B"
    row: Dict[str, Any] = {
        "pair_id": pair_id,
        "canonical_pair_id": canonical_pair_id,
        "repeat_idx": int(repeat_idx),
        "sample_id_a": _safe_sample_id(left.get("sample_id"), f"{canonical_pair_id}_a"),
        "sample_id_b": _safe_sample_id(right.get("sample_id"), f"{canonical_pair_id}_b"),
        "image_path_a": str(left["image_path"]),
        "image_path_b": str(right["image_path"]),
        "presented_left_path": presented_left_path,
        "presented_right_path": presented_right_path,
        "presented_order": presented_order,
        "is_swapped": bool(swap_presented),
    }
    for col in metadata_columns:
        row[f"{col}_a"] = left.get(col)
        row[f"{col}_b"] = right.get(col)
    return row


def build_global_random_pairs(
    manifest_df: pd.DataFrame,
    *,
    max_pairs: Optional[int] = None,
    seed: int = 777,
    allow_replacement: bool = False,
    counterbalance_mode: str = "none",
    repeat_count: int = 0,
    repeat_fraction: float = 0.0,
    metadata_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build globally-random image pairs from Cyclomedia manifest rows."""
    if manifest_df is None or manifest_df.empty:
        return pd.DataFrame(
            columns=[
                "pair_id",
                "sample_id_a",
                "sample_id_b",
                "image_path_a",
                "image_path_b",
                "presented_left_path",
                "presented_right_path",
                "presented_order",
                "is_swapped",
                "canonical_pair_id",
                "repeat_idx",
            ]
        )

    if "image_path" not in manifest_df.columns:
        raise ValueError("Manifest must contain an 'image_path' column before pair sampling.")

    work_df = manifest_df.dropna(subset=["image_path"]).reset_index(drop=True)
    if work_df.empty:
        raise ValueError("No valid image rows available to form pairs.")

    if "sample_id" not in work_df.columns:
        work_df = work_df.copy()
        work_df["sample_id"] = [f"sample_{i:08d}" for i in range(len(work_df))]

    if metadata_columns is None:
        excluded = {"sample_id", "image_path"}
        metadata_columns = [c for c in work_df.columns if c not in excluded]

    counterbalance_mode = _normalize_counterbalance_mode(counterbalance_mode)
    repeat_count = max(0, int(repeat_count or 0))
    repeat_fraction = max(0.0, float(repeat_fraction or 0.0))

    rng = np.random.default_rng(seed)
    n = len(work_df)
    base_pairs: List[Tuple[int, int]] = []

    if allow_replacement:
        if n < 2:
            raise ValueError("At least two rows are required when allow_replacement=true.")
        target_pairs = max_pairs if max_pairs is not None else n // 2
        for pair_idx in range(int(target_pairs)):
            left_idx = int(rng.integers(0, n))
            right_idx = int(rng.integers(0, n))
            while right_idx == left_idx:
                right_idx = int(rng.integers(0, n))
            base_pairs.append((left_idx, right_idx))
    else:
        perm = rng.permutation(n)
        pair_count = n // 2
        if max_pairs is not None:
            pair_count = min(pair_count, int(max_pairs))
        for pair_idx in range(pair_count):
            left_idx = int(perm[pair_idx * 2])
            right_idx = int(perm[pair_idx * 2 + 1])
            base_pairs.append((left_idx, right_idx))

    if not base_pairs:
        raise ValueError("Pair sampler produced zero base pairs; adjust max_pairs or input dataset size.")

    extra_repeats = repeat_count
    if extra_repeats <= 0 and repeat_fraction > 0:
        extra_repeats = int(round(len(base_pairs) * repeat_fraction))

    repeat_cursor: Dict[int, int] = {}
    observations: List[Tuple[int, int]] = [(idx, 0) for idx in range(len(base_pairs))]
    for _ in range(extra_repeats):
        canonical_idx = int(rng.integers(0, len(base_pairs)))
        next_repeat = repeat_cursor.get(canonical_idx, 0) + 1
        repeat_cursor[canonical_idx] = next_repeat
        observations.append((canonical_idx, next_repeat))

    rows: List[Dict[str, Any]] = []
    for obs_idx, (canonical_idx, repeat_idx) in enumerate(observations):
        left_idx, right_idx = base_pairs[canonical_idx]
        canonical_pair_id = f"pair_{canonical_idx:08d}"
        pair_id = f"{canonical_pair_id}_r{repeat_idx}" if repeat_idx > 0 else canonical_pair_id
        if counterbalance_mode == "balanced":
            swap_presented = bool(obs_idx % 2)
        elif counterbalance_mode == "random":
            swap_presented = bool(rng.integers(0, 2))
        else:
            swap_presented = False
        rows.append(
            _build_pair_row(
                work_df.iloc[left_idx],
                work_df.iloc[right_idx],
                pair_id,
                canonical_pair_id,
                repeat_idx,
                swap_presented,
                metadata_columns,
            )
        )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise ValueError("Pair sampler produced zero pairs; adjust max_pairs or input dataset size.")
    return pair_df

