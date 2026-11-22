"""Classify stage: apply trained heads to cached features → scores parquet.

Cheap (features are small, heads are linear). Re-runs are fast — use this
stage to evaluate a new head, threshold, or post-hoc ensemble without
re-running the expensive feature-extraction step.
"""
from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .train import _build_heads, _resolve_heads


def run_classify_stage(cfg: DictConfig) -> Dict[str, Any]:
    features_input = str(cfg.classify.features_input or "")
    if not features_input or not os.path.exists(features_input):
        raise ValueError(f"classify.features_input must exist; got {features_input!r}")

    output_path = _resolve_output_path(cfg)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load heads from training checkpoint
    checkpoint_dir = str(cfg.training.checkpoint_dir or "")
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"training.checkpoint_dir must exist; got {checkpoint_dir!r}")

    feature_dim = int(cfg.model.feature_dim)
    heads = _build_heads(cfg, feature_dim)
    heads_path = os.path.join(checkpoint_dir, "heads_final.pt")
    if not os.path.exists(heads_path):
        heads_path = os.path.join(checkpoint_dir, "heads_best.pt")
    if not os.path.exists(heads_path):
        raise FileNotFoundError(
            f"No heads checkpoint under {checkpoint_dir} (looked for "
            f"heads_final.pt, heads_best.pt). Train first, or point "
            f"training.checkpoint_dir at a trained run."
        )
    heads.load_state_dict(torch.load(heads_path, map_location="cpu"))
    heads = heads.eval()

    head_specs = _resolve_heads(cfg)
    heads_subset = cfg.classify.get("heads_subset", None)
    if heads_subset is not None:
        wanted = set(str(h) for h in heads_subset)
        head_specs = [h for h in head_specs if h["name"] in wanted]

    # Enumerate features parquets
    if os.path.isdir(features_input):
        parquet_files = sorted(glob.glob(os.path.join(features_input,
                                                       "features-*.parquet")))
        if not parquet_files:
            parquet_files = sorted(glob.glob(os.path.join(features_input,
                                                           "*.parquet")))
    else:
        parquet_files = [features_input]

    if not parquet_files:
        raise FileNotFoundError(f"No feature parquets in {features_input}")

    batch_size = int(cfg.classify.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = heads.to(device)

    feature_col = "feature"
    t0 = time.time()
    total_rows = 0
    all_chunks: List[pd.DataFrame] = []

    for pq in parquet_files:
        df = pd.read_parquet(pq)
        if feature_col not in df.columns:
            # Try alternate column name
            for cand in ("features", "embedding", "feat"):
                if cand in df.columns:
                    feature_col = cand
                    break

        feats = np.asarray(df[feature_col].tolist(), dtype=np.float32)
        scores = _classify_array(feats, heads, head_specs, batch_size, device)

        out_df = df[[c for c in df.columns if c != feature_col]].copy()
        for name, payload in scores.items():
            out_df[f"{name}_pred"] = payload["pred"]
            for ci, col in enumerate(payload["probs_cols"]):
                out_df[f"{name}_{col}"] = payload["probs"][:, ci]
        all_chunks.append(out_df)
        total_rows += len(out_df)

    result = pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()
    result.to_parquet(output_path, index=False)
    duration = time.time() - t0

    print(f"[classify] {total_rows} samples × {len(head_specs)} heads → "
          f"{output_path} in {duration:.1f}s", flush=True)
    return {
        "scores_path": output_path,
        "metrics": {
            "samples": total_rows,
            "heads": len(head_specs),
            "duration_s": duration,
        },
        "metadata": {"parquets_in": len(parquet_files)},
    }


def _classify_array(feats: np.ndarray, heads: nn.ModuleDict,
                    head_specs: List[Dict[str, Any]], batch_size: int,
                    device) -> Dict[str, Dict[str, np.ndarray]]:
    n = len(feats)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for spec in head_specs:
        name = spec["name"]
        num_classes = int(spec["num_classes"])
        out[name] = {
            "pred": np.empty(n, dtype=np.int64),
            "probs": np.empty((n, num_classes), dtype=np.float32),
            "probs_cols": [f"score_{i}" for i in range(num_classes)]
            if num_classes != 2 else ["score_0", "score_1"],
        }

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            x = torch.from_numpy(feats[start:end]).to(device)
            for spec in head_specs:
                name = spec["name"]
                logits = heads[name](x)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                preds = probs.argmax(axis=-1)
                out[name]["pred"][start:end] = preds
                out[name]["probs"][start:end] = probs

    return out


def _resolve_output_path(cfg: DictConfig) -> str:
    explicit = cfg.classify.get("output_path", None)
    if explicit:
        return os.path.abspath(str(explicit))
    return os.path.abspath(os.path.join("outputs", "urbanvit", "scores.parquet"))
