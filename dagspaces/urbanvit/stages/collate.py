"""Collate stage: concat per-shard scores parquets + optional /scratch offload.

- If `scores_input` is a directory, concats all `*.parquet` files inside into
  a single `output_path` parquet.
- If `scores_input` is a single parquet file, passes through (copy).
- If `collate.offload.enabled`, archives the shards dir (read from
  `shard.output_dir`) into `offload.archive_dir` after the concat completes.
"""
from __future__ import annotations

import glob
import os
import shutil
import time
from typing import Any, Dict, List

import pandas as pd
from omegaconf import DictConfig


def run_collate_stage(cfg: DictConfig) -> Dict[str, Any]:
    scores_input = str(cfg.collate.scores_input or "")
    if not scores_input or not os.path.exists(scores_input):
        raise ValueError(f"collate.scores_input must exist; got {scores_input!r}")

    output_path = _resolve_output_path(cfg)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    if os.path.isdir(scores_input):
        files = sorted(glob.glob(os.path.join(scores_input, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files in {scores_input}")
        frames: List[pd.DataFrame] = [pd.read_parquet(f) for f in files]
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(output_path, index=False)
        print(f"[collate] merged {len(files)} parquets → {output_path} "
              f"({len(df)} rows)", flush=True)
        total_rows = len(df)
        n_parquets = len(files)
    else:
        shutil.copyfile(scores_input, output_path)
        total_rows = len(pd.read_parquet(output_path))
        n_parquets = 1
        print(f"[collate] passthrough {scores_input} → {output_path} "
              f"({total_rows} rows)", flush=True)

    # Optional offload
    offload_cfg = cfg.collate.get("offload", {})
    offloaded = False
    if bool(offload_cfg.get("enabled", False)):
        archive_dir = str(offload_cfg.get("archive_dir") or "")
        shards_dir = str(cfg.shard.output_dir or "")
        if archive_dir and shards_dir and os.path.isdir(shards_dir):
            os.makedirs(archive_dir, exist_ok=True)
            target = os.path.join(archive_dir,
                                  os.path.basename(os.path.normpath(shards_dir)))
            if os.path.exists(target):
                print(f"[collate] offload target {target} already exists — skipping",
                      flush=True)
            else:
                print(f"[collate] offloading {shards_dir} → {target}", flush=True)
                shutil.move(shards_dir, target)
                offloaded = True
        else:
            print(f"[collate] offload enabled but archive_dir or shard.output_dir "
                  f"missing — skipping", flush=True)

    duration = time.time() - t0
    return {
        "final_path": output_path,
        "metrics": {
            "rows": total_rows,
            "parquets_in": n_parquets,
            "duration_s": duration,
            "offloaded": int(offloaded),
        },
        "metadata": {"offloaded": offloaded},
    }


def _resolve_output_path(cfg: DictConfig) -> str:
    explicit = cfg.collate.get("output_path", None)
    if explicit:
        return os.path.abspath(str(explicit))
    return os.path.abspath(os.path.join("outputs", "urbanvit", "final.parquet"))
