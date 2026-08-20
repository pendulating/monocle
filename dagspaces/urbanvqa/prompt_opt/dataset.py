from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf


GEPA_ROW_COLUMNS: Tuple[str, ...] = (
    "prompt",
    "expected_answer",
    "sample_id",
    "image_path",
    "image_url",
    "image_base64",
    "metadata",
)


class MissingColumnError(ValueError):
    """Raised when the configured answer or prompt columns are missing."""


def _clone_with_dataset_overrides(cfg: DictConfig, split_cfg: DictConfig) -> DictConfig:
    """Clone the Hydra config and apply dataset overrides for a split."""
    local_cfg = deepcopy(cfg)

    if getattr(split_cfg, "parquet_path", None):
        OmegaConf.update(local_cfg, "data.parquet_path", split_cfg.parquet_path, merge=True)
    if getattr(split_cfg, "image_path", None):
        OmegaConf.update(local_cfg, "data.image_path", split_cfg.image_path, merge=True)
    if getattr(split_cfg, "prompt_column", None):
        OmegaConf.update(local_cfg, "data.columns.prompt", split_cfg.prompt_column, merge=True)
    if getattr(split_cfg, "answer_column", None):
        OmegaConf.update(local_cfg, "data.columns.expected_answer", split_cfg.answer_column, merge=True)

    return local_cfg


def _to_pandas(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        return df.to_pandas()
    return df


def materialize_supervised_frame(
    cfg: DictConfig,
    split: str,
) -> pd.DataFrame:
    """Return a pandas DataFrame suitable for GEPA training or validation.

    The function clones the provided Hydra configuration, applies per-split
    overrides, and then delegates to ``prepare_stage_input`` to reuse the
    existing VQA ingestion logic.
    """
    split_cfg = getattr(cfg.gepa.dataset, split, None)
    if split_cfg is None:
        raise ValueError(f"cfg.gepa.dataset has no entry for split '{split}'")

    local_cfg = _clone_with_dataset_overrides(cfg, split_cfg)
    parquet_path = getattr(local_cfg.data, "parquet_path", "") or ""
    # Lazy import: prepare_stage_input predates the trawler orchestrator
    # refactor. Importing here keeps the package importable (the pairwise GEPA
    # path never calls this) while surfacing a clear error if the legacy
    # single-image path is exercised against the refactored orchestrator.
    from dagspaces.urbanvqa.orchestrator import prepare_stage_input
    df, ds, use_streaming = prepare_stage_input(local_cfg, parquet_path, stage=f"gepa_{split}")
    frame = _to_pandas(df)

    prompt_col = getattr(split_cfg, "prompt_column", "prompt")
    answer_col = getattr(split_cfg, "answer_column", "expected_answer")

    for col in (prompt_col, answer_col):
        if col not in frame.columns:
            raise MissingColumnError(f"Required column '{col}' not found in dataset for split '{split}'")

    # Ensure expected columns exist even if empty so downstream consumers can rely on schema.
    for column in GEPA_ROW_COLUMNS:
        if column not in frame.columns:
            frame[column] = None

    return frame


def stratified_sample(
    frame: pd.DataFrame,
    *,
    label_column: str,
    num_rows: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Return a stratified sample that preserves class balance."""
    if label_column not in frame.columns:
        raise MissingColumnError(f"Stratification column '{label_column}' is missing from dataset")

    if num_rows is None or num_rows >= len(frame):
        return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    grouped = frame.groupby(label_column, dropna=False, sort=False)
    counts = grouped.size()
    proportions = counts / counts.sum()

    base_counts = (proportions * num_rows).apply(math.floor).astype(int)
    remainder = num_rows - base_counts.sum()

    if remainder > 0:
        fractional = (proportions * num_rows) - base_counts
        order = fractional.sort_values(ascending=False).index.tolist()
        for label in order[:remainder]:
            base_counts[label] += 1

    samples: List[pd.DataFrame] = []
    for label, group in grouped:
        n = min(base_counts.get(label, 0), len(group))
        if n <= 0:
            continue
        samples.append(group.sample(n=n, random_state=seed))

    sampled = pd.concat(samples).sample(frac=1.0, random_state=seed, ignore_index=True)
    return sampled


def iter_minibatches(
    frame: pd.DataFrame,
    *,
    batch_size: int,
) -> Iterable[pd.DataFrame]:
    """Yield deterministic minibatches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(frame), batch_size):
        yield frame.iloc[start : start + batch_size]


def build_supervised_minibatches(
    cfg: DictConfig,
    split: str,
) -> List[pd.DataFrame]:
    """Materialize stratified minibatches suitable for GEPA optimization."""
    frame = materialize_supervised_frame(cfg, split)
    split_cfg = getattr(cfg.gepa.dataset, split)
    limit = getattr(split_cfg, "limit", None)
    stratify_by = getattr(split_cfg, "stratify_by", None)
    seed = getattr(cfg, "seed", 0)

    if stratify_by and len(frame) > 0:
        frame = stratified_sample(
            frame,
            label_column=stratify_by,
            num_rows=limit,
            seed=seed,
        )
    elif limit is not None and limit < len(frame):
        frame = frame.sample(n=limit, random_state=seed).reset_index(drop=True)
    else:
        frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    cache_dir = getattr(cfg.gepa.sampler, "cache_dir", None)
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path / f"{split}.parquet", index=False)

    batch_size = getattr(cfg.gepa.sampler, "batch_size", 64)
    return list(iter_minibatches(frame, batch_size=batch_size))


