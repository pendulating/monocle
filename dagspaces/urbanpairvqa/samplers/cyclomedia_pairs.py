from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "build_global_random_pairs",
    "build_unit_random_pairs",
]


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


# Memory bound for the enumerate-and-shuffle branch in
# ``_sample_distinct_canonical_pairs``. At 10M canonical pairs, the two int64
# arrays from ``np.triu_indices`` plus the permutation cost ~240 MB.
_DISTINCT_PAIR_DENSE_THRESHOLD = 10_000_000


def _sample_distinct_canonical_pairs(
    rng: np.random.Generator,
    n: int,
    target: int,
) -> List[Tuple[int, int]]:
    """Draw ``target`` distinct unordered index pairs from ``[0, n)``.

    Each returned ``(a, b)`` has ``a != b`` and no two output pairs share the
    same unordered key ``{a, b}``. Output is capped at the canonical-pair
    budget ``C(n, 2) = n * (n - 1) / 2``.

    Uses enumerate-and-shuffle when the canonical-pair set is small
    (``≤ _DISTINCT_PAIR_DENSE_THRESHOLD``) and rejection sampling otherwise.
    Rejection is intended for the regime where ``target ≪ C(n, 2)`` — e.g.
    50k pairs from 18,488 units (target/budget ≈ 3e-4).
    """
    if n < 2:
        raise ValueError(f"Need at least 2 items to sample distinct pairs; got n={n}.")
    max_canonical = n * (n - 1) // 2
    target = int(target)
    if target <= 0:
        return []
    if target > max_canonical:
        target = max_canonical

    if max_canonical <= _DISTINCT_PAIR_DENSE_THRESHOLD:
        iu0, iu1 = np.triu_indices(n, k=1)
        chosen = rng.permutation(max_canonical)[:target]
        flip = rng.integers(0, 2, size=target).astype(bool)
        return [
            (int(iu1[k]), int(iu0[k])) if flip[t] else (int(iu0[k]), int(iu1[k]))
            for t, k in enumerate(chosen)
        ]

    seen: set = set()
    pairs: List[Tuple[int, int]] = []
    attempts = 0
    # Expected attempts under uniform rejection = target * C / (C - target/2).
    # Cap at 8*target leaves wide safety margin for any target/C ≤ ~0.85,
    # which is far past where this branch is used (sparse-only by design).
    attempt_cap = max(8 * target, target + 1024)
    while len(pairs) < target and attempts < attempt_cap:
        attempts += 1
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    if len(pairs) < target:
        raise RuntimeError(
            f"Distinct-pair sampler stalled at {len(pairs)}/{target} pairs after "
            f"{attempts} attempts (n={n}, max_canonical={max_canonical}). "
            "Reduce max_pairs or raise _DISTINCT_PAIR_DENSE_THRESHOLD."
        )
    return pairs


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
        if max_pairs is None:
            # Default: perfect matching over the row pool — each row appears in
            # at most one pair, total ``n // 2``.
            perm = rng.permutation(n)
            for pair_idx in range(n // 2):
                left_idx = int(perm[pair_idx * 2])
                right_idx = int(perm[pair_idx * 2 + 1])
                base_pairs.append((left_idx, right_idx))
        else:
            # Sample distinct canonical pairs (rows may recur across pairs)
            # up to ``min(max_pairs, C(n, 2))``.
            base_pairs.extend(
                _sample_distinct_canonical_pairs(rng, n, int(max_pairs))
            )

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


def build_unit_random_pairs(
    manifest_df: pd.DataFrame,
    *,
    unit_column: str = "unit_uid",
    unit_name_column: Optional[str] = "unit_name",
    max_pairs: Optional[int] = None,
    seed: int = 777,
    allow_replacement: bool = False,
    counterbalance_mode: str = "none",
    repeat_count: int = 0,
    repeat_fraction: float = 0.0,
    metadata_columns: Optional[List[str]] = None,
    weight_column: Optional[str] = None,
) -> pd.DataFrame:
    """Pair images at the **unit** level: sample 2 distinct units, then one
    image from each.

    Designed for curated parquets where each row carries a unit identifier
    (e.g. ``unit_uid`` = the library/facility/permit the image was materialized
    under). A single image that sits inside multiple overlapping buffers
    appears multiple times in the manifest — once per unit it's attributed
    to — and the sampler honors that: each unit draws from its own full
    image pool.

    The returned pair frame has the same shape as
    :func:`build_global_random_pairs` output plus ``unit_uid_a``,
    ``unit_name_a``, ``unit_uid_b``, ``unit_name_b``. ``canonical_pair_id``
    is keyed on the ordered unit pair (so the same two units drawn with
    different image samples are still distinguished by ``pair_id``).

    Args:
        manifest_df: Rows with ``image_path``, ``sample_id``, and the
            ``unit_column`` key.
        unit_column: Column naming the unit. Default ``unit_uid``.
        unit_name_column: Optional column for a human-readable unit label
            to persist into pair rows (``unit_name_a``/``unit_name_b``).
        max_pairs: Number of unit pairs to sample. With
            ``allow_replacement=False``: ``None`` falls back to a perfect
            matching (``n_units // 2`` pairs, each unit appears at most once);
            an explicit value samples distinct canonical unit pairs (units may
            recur across pairs) up to ``min(max_pairs, C(n_units, 2))``.
            With ``allow_replacement=True``: targets ``max_pairs`` exactly,
            sampling pair endpoints independently.
        allow_replacement: If True, sample unit pairs with replacement;
            each pair still has two distinct units (we reject self-pairs).
        counterbalance_mode / repeat_count / repeat_fraction: Same semantics
            as :func:`build_global_random_pairs`.
        metadata_columns: Extra columns on the manifest to persist as
            ``<col>_a`` / ``<col>_b``. Default: all non-internal columns.
        weight_column: Optional column of non-negative weights used to bias
            **within-unit** image selection (unit-level pair choice stays
            uniform). Typically ``"attribution_confidence"`` emitted by the
            per-unit facing filter (see [[concept-facing-filter]]). Missing
            column → uniform with a one-line warning. NaN/negative/zero
            values are clipped to 0; units whose entire pool ends up at 0
            fall back to uniform for that unit only.
    """
    if manifest_df is None or manifest_df.empty:
        return pd.DataFrame(columns=[
            "pair_id", "sample_id_a", "sample_id_b", "image_path_a", "image_path_b",
            "presented_left_path", "presented_right_path", "presented_order",
            "is_swapped", "canonical_pair_id", "repeat_idx",
            "unit_uid_a", "unit_name_a", "unit_uid_b", "unit_name_b",
        ])

    if "image_path" not in manifest_df.columns:
        raise ValueError("Manifest must contain an 'image_path' column before pair sampling.")
    if unit_column not in manifest_df.columns:
        raise ValueError(
            f"Manifest missing unit_column {unit_column!r}. "
            "Did you re-materialize after the unit-attribution change? "
            f"Available: {list(manifest_df.columns)[:20]}..."
        )

    work = manifest_df.dropna(subset=["image_path", unit_column]).copy()
    work = work[work[unit_column].astype(str).str.len() > 0].reset_index(drop=True)
    if "sample_id" not in work.columns:
        work["sample_id"] = [f"sample_{i:08d}" for i in range(len(work))]

    if metadata_columns is None:
        excluded = {"sample_id", "image_path"}
        metadata_columns = [c for c in work.columns if c not in excluded]

    counterbalance_mode = _normalize_counterbalance_mode(counterbalance_mode)
    repeat_count = max(0, int(repeat_count or 0))
    repeat_fraction = max(0.0, float(repeat_fraction or 0.0))

    # Group rows by unit, keep only units with ≥1 image.
    unit_to_indices: Dict[Any, List[int]] = {}
    for i, u in enumerate(work[unit_column].tolist()):
        unit_to_indices.setdefault(u, []).append(i)
    # Drop empty (shouldn't happen after the dropna) and sort for determinism.
    units = sorted([u for u, idxs in unit_to_indices.items() if idxs], key=str)
    n_units = len(units)
    if n_units < 2:
        raise ValueError(
            f"Unit sampler needs ≥ 2 distinct units; got {n_units} in column {unit_column!r}."
        )

    # Build per-unit weight vectors when requested. rng.choice(..., p=None) is
    # uniform, so a unit mapped to None (column missing or degenerate weights)
    # silently falls back — same behavior as the non-weighted path.
    unit_to_weights: Dict[Any, Optional[np.ndarray]] = {}
    n_fallback_units = 0
    weight_column_used = False
    if weight_column:
        if weight_column not in work.columns:
            print(
                f"[build_unit_random_pairs] weight_column={weight_column!r} not in "
                f"manifest — falling back to uniform sampling for all units. "
                f"Did you re-materialize with the per-unit facing filter?",
                flush=True,
            )
        else:
            weight_column_used = True
            raw = pd.to_numeric(work[weight_column], errors="coerce").to_numpy(dtype=float)
            raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
            for unit, idxs in unit_to_indices.items():
                w = raw[idxs]
                total = float(w.sum())
                if total > 0.0:
                    unit_to_weights[unit] = w / total
                else:
                    unit_to_weights[unit] = None
                    n_fallback_units += 1
            n_usable = sum(1 for v in unit_to_weights.values() if v is not None)
            print(
                f"[build_unit_random_pairs] weighted image sampling on "
                f"{weight_column!r}: {n_usable}/{n_units} units using weights, "
                f"{n_fallback_units} units fell back to uniform (all-zero weights).",
                flush=True,
            )

    rng = np.random.default_rng(seed)
    unit_pairs: List[Tuple[Any, Any]] = []

    if allow_replacement:
        target = max_pairs if max_pairs is not None else n_units // 2
        for _ in range(int(target)):
            a_idx = int(rng.integers(0, n_units))
            b_idx = int(rng.integers(0, n_units))
            while b_idx == a_idx:
                b_idx = int(rng.integers(0, n_units))
            unit_pairs.append((units[a_idx], units[b_idx]))
    else:
        if max_pairs is None:
            # Default: perfect matching over the unit pool — each unit appears
            # in at most one pair, total ``n_units // 2``.
            perm = rng.permutation(n_units)
            for k in range(n_units // 2):
                unit_pairs.append((units[perm[2 * k]], units[perm[2 * k + 1]]))
        else:
            # Sample distinct canonical unit pairs (a unit may recur across
            # pairs but no canonical pair repeats) up to
            # ``min(max_pairs, C(n_units, 2))``.
            for a_idx, b_idx in _sample_distinct_canonical_pairs(
                rng, n_units, int(max_pairs)
            ):
                unit_pairs.append((units[a_idx], units[b_idx]))

    if not unit_pairs:
        raise ValueError("Unit sampler produced zero pairs; adjust max_pairs or input dataset size.")

    # Counterbalance / repeat plan.
    extra_repeats = repeat_count
    if extra_repeats <= 0 and repeat_fraction > 0:
        extra_repeats = int(round(len(unit_pairs) * repeat_fraction))

    observations: List[Tuple[int, int]] = [(i, 0) for i in range(len(unit_pairs))]
    repeat_cursor: Dict[int, int] = {}
    for _ in range(extra_repeats):
        canonical_idx = int(rng.integers(0, len(unit_pairs)))
        nxt = repeat_cursor.get(canonical_idx, 0) + 1
        repeat_cursor[canonical_idx] = nxt
        observations.append((canonical_idx, nxt))

    rows: List[Dict[str, Any]] = []
    for obs_idx, (canonical_idx, repeat_idx) in enumerate(observations):
        unit_a, unit_b = unit_pairs[canonical_idx]
        # Sample one image per unit for this observation (every observation gets
        # a fresh draw so repeats genuinely re-sample within-unit).
        # p=None → uniform; p=weights → biased toward higher-weight images.
        idx_a = int(rng.choice(unit_to_indices[unit_a], p=unit_to_weights.get(unit_a)))
        idx_b = int(rng.choice(unit_to_indices[unit_b], p=unit_to_weights.get(unit_b)))
        left = work.iloc[idx_a]
        right = work.iloc[idx_b]
        canonical_pair_id = f"unit_{canonical_idx:08d}"
        pair_id = f"{canonical_pair_id}_r{repeat_idx}" if repeat_idx > 0 else canonical_pair_id
        if counterbalance_mode == "balanced":
            swap_presented = bool(obs_idx % 2)
        elif counterbalance_mode == "random":
            swap_presented = bool(rng.integers(0, 2))
        else:
            swap_presented = False
        row = _build_pair_row(
            left, right, pair_id, canonical_pair_id, repeat_idx,
            swap_presented, metadata_columns,
        )
        # Always persist unit identity (even if unit_column was in metadata_columns
        # and got copied as <unit_column>_a already — we also add canonical aliases).
        row["unit_uid_a"] = str(left[unit_column])
        row["unit_uid_b"] = str(right[unit_column])
        if unit_name_column and unit_name_column in work.columns:
            row["unit_name_a"] = str(left.get(unit_name_column, ""))
            row["unit_name_b"] = str(right.get(unit_name_column, ""))
        rows.append(row)

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise ValueError("Unit pair sampler produced zero pairs; adjust max_pairs.")
    return pair_df
