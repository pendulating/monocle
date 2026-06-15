"""Extract stage: cached 1024-dim features for each image.

Streams WebDataset tar shards through a compiled bf16 backbone and writes
one features parquet per shard under `output_dir`. Downstream `classify`
stage concats these. Accepts an optional `backbone_checkpoint` (LoRA adapter
dir from the train stage); when null, uses the pretrained backbone.

Designed to be run per-shard (or per-shard-subset) so a corpus of 4M–36M
images can be split across independent SLURM jobs. Use
`extract.shard_subset=[start, end]` to process a contiguous slice.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .train import _build_backbone, _extract_features, _maybe_attach_lora


def run_extract_stage(cfg: DictConfig) -> Dict[str, Any]:
    import torch

    output_dir = _resolve_output_dir(cfg)
    os.makedirs(output_dir, exist_ok=True)

    shards_dir = str(cfg.shard.output_dir or "")
    if not shards_dir or not os.path.isdir(shards_dir):
        raise ValueError(
            f"shard.output_dir must point to existing shard dir; got {shards_dir!r}"
        )

    # Build backbone (optionally with LoRA adapter from training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, feature_dim, image_size = _build_backbone(cfg)
    ckpt = cfg.extract.get("backbone_checkpoint", None)
    if ckpt:
        backbone = _load_backbone_checkpoint(backbone, str(ckpt), cfg)
    backbone = backbone.to(device).eval()

    # Compile
    try:
        backbone = torch.compile(backbone, mode="max-autotune")
    except Exception as e:
        print(f"[extract] torch.compile failed, running eager: {e}", flush=True)

    # Enumerate shards
    splits = list(cfg.extract.get("splits", ["train", "val", "test"]))
    shard_files: List[Dict[str, str]] = []
    for fn in sorted(os.listdir(shards_dir)):
        if not fn.endswith(".tar"):
            continue
        for split in splits:
            if fn.startswith(f"{split}-"):
                shard_files.append({"split": split,
                                    "path": os.path.join(shards_dir, fn),
                                    "name": fn})
                break

    # Apply shard_subset slicing
    subset = cfg.extract.get("shard_subset", None)
    if subset is not None:
        start, end = int(subset[0]), int(subset[1])
        shard_files = shard_files[start:end]
        print(f"[extract] Processing shard subset [{start}:{end}] → "
              f"{len(shard_files)} shards", flush=True)

    if not shard_files:
        print(f"[extract] No shards matched splits={splits}", flush=True)
        return {
            "features_dir": output_dir,
            "metrics": {"shards": 0, "samples": 0},
            "metadata": {"splits": splits},
        }

    batch_size = int(cfg.extract.batch_size)
    num_workers = int(cfg.extract.num_workers)
    feature_col = str(cfg.extract.feature_column)

    total_samples = 0
    t0 = time.time()
    for idx, shard in enumerate(shard_files):
        out_path = os.path.join(
            output_dir,
            shard["name"].replace(".tar", ".parquet").replace(
                f"{shard['split']}-", f"features-{shard['split']}-"
            ),
        )
        if os.path.exists(out_path):
            print(f"[extract] skip existing {out_path}", flush=True)
            continue

        t_shard = time.time()
        rows = _extract_one_shard(
            backbone=backbone, shard_path=shard["path"],
            split=shard["split"], cfg=cfg, device=device,
            batch_size=batch_size, num_workers=num_workers,
            image_size=image_size, feature_col=feature_col,
        )
        if not rows:
            print(f"[extract] WARN: 0 rows from {shard['path']}", flush=True)
            continue
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        total_samples += len(rows)
        print(f"[extract] [{idx+1}/{len(shard_files)}] {shard['name']}: "
              f"{len(rows)} samples in {time.time()-t_shard:.1f}s → {out_path}",
              flush=True)

    duration = time.time() - t0
    return {
        "features_dir": output_dir,
        "metrics": {
            "shards": len(shard_files),
            "samples": total_samples,
            "duration_s": duration,
            "throughput_per_s": total_samples / max(1.0, duration),
        },
        "metadata": {"splits": splits, "feature_dim": feature_dim},
    }


# ---------------------------------------------------------------------------
# Per-shard extraction loop
# ---------------------------------------------------------------------------

def _extract_one_shard(backbone, shard_path: str, split: str, cfg: DictConfig,
                       device, batch_size: int, num_workers: int,
                       image_size: int, feature_col: str) -> List[Dict[str, Any]]:
    import torch
    import webdataset as wds
    from torch.utils.data import DataLoader

    means = list(cfg.model.mean)
    stds = list(cfg.model.std)
    mean_t = torch.tensor(means).view(3, 1, 1)
    std_t = torch.tensor(stds).view(3, 1, 1)

    def _decode(sample):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BICUBIC)
        arr = torch.from_numpy(np.array(img, dtype=np.uint8))
        arr = arr.permute(2, 0, 1).float().div_(255.0)
        arr = (arr - mean_t) / std_t
        meta = json.loads(sample["json"])
        return {"pixel": arr, "meta": meta}

    dataset = (
        wds.WebDataset([shard_path], shardshuffle=False,
                       handler=wds.warn_and_continue)
        .map(_decode, handler=wds.warn_and_continue)
    )

    def _collate(batch):
        pixels = torch.stack([b["pixel"] for b in batch])
        metas = [b["meta"] for b in batch]
        return {"pixel": pixels, "meta": metas}

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, pin_memory=True)

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            pixel = batch["pixel"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats = _extract_features(backbone, pixel)
            feats_np = feats.float().cpu().numpy()
            for i, meta in enumerate(batch["meta"]):
                r = {
                    "sample_id": meta.get("sample_id"),
                    "recording_id": meta.get("recording_id"),
                    "split": split,
                    feature_col: feats_np[i].tolist(),
                }
                # Preserve any other metadata for reference
                for k, v in meta.items():
                    if k not in r:
                        r[k] = v
                rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Checkpoint loading for fine-tuned backbone
# ---------------------------------------------------------------------------

def _load_backbone_checkpoint(backbone, checkpoint_dir: str, cfg: DictConfig):
    adapter_dir = os.path.join(checkpoint_dir, "lora_adapter")
    if os.path.isdir(adapter_dir):
        try:
            from peft import PeftModel
            backbone = _maybe_attach_lora(backbone, cfg)  # ensure lora-wrapped
            if hasattr(backbone, "load_adapter"):
                backbone.load_adapter(adapter_dir, adapter_name="default")
            else:
                backbone = PeftModel.from_pretrained(backbone, adapter_dir)
            print(f"[extract] Loaded LoRA adapter from {adapter_dir}", flush=True)
        except Exception as e:
            print(f"[extract] Failed to load LoRA adapter: {e} — using base backbone",
                  flush=True)
    return backbone


def _resolve_output_dir(cfg: DictConfig) -> str:
    explicit = cfg.extract.get("output_dir", None)
    if explicit:
        return os.path.abspath(str(explicit))
    return os.path.abspath(os.path.join("outputs", "urbanvit", "features"))
