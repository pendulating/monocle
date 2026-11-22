"""Eval stage: load a trained checkpoint, run on a held-out split, write metrics.

Computes per-head accuracy + count, plus a small per-head confusion-matrix
table. Output is a single parquet row per head.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import pandas as pd
from omegaconf import DictConfig

from .train import (
    _build_backbone,
    _build_heads,
    _maybe_attach_lora,
    _BackboneWithHeads,
    _make_dataloader,
    _resolve_heads,
)


def run_eval_stage(cfg: DictConfig) -> Dict[str, Any]:
    import torch

    checkpoint_dir = str(cfg.training.checkpoint_dir or "")
    if not os.path.isdir(checkpoint_dir):
        raise ValueError(f"Invalid training.checkpoint_dir: {checkpoint_dir}")

    split = str(cfg.eval.get("split", "val"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model → attach LoRA adapter → load heads
    backbone, feature_dim, image_size = _build_backbone(cfg)
    backbone = _maybe_attach_lora(backbone, cfg)
    adapter_path = os.path.join(checkpoint_dir, "lora_adapter")
    if os.path.isdir(adapter_path) and hasattr(backbone, "load_adapter"):
        try:
            backbone.load_adapter(adapter_path, adapter_name="default")
        except Exception as e:
            print(f"[eval] load_adapter failed ({e}); reloading LoRA from disk "
                  f"via from_pretrained instead", flush=True)
            try:
                from peft import PeftModel
                backbone = PeftModel.from_pretrained(backbone, adapter_path)
            except Exception as e2:
                print(f"[eval] PeftModel.from_pretrained also failed: {e2}",
                      flush=True)

    heads = _build_heads(cfg, feature_dim)
    heads_path = os.path.join(checkpoint_dir, "heads_final.pt")
    if not os.path.exists(heads_path):
        heads_path = os.path.join(checkpoint_dir, "heads_best.pt")
    if os.path.exists(heads_path):
        heads.load_state_dict(torch.load(heads_path, map_location="cpu"))
    else:
        print(f"[eval] WARNING: no heads checkpoint found under {checkpoint_dir} "
              f"— using randomly-initialized heads", flush=True)

    model = _BackboneWithHeads(backbone=backbone, heads=heads).to(device).eval()

    loader = _make_dataloader(cfg, split=split, rank=0, world_size=1,
                              image_size=image_size)

    head_specs = _resolve_heads(cfg)

    t0 = time.time()
    confusion: Dict[str, Dict[str, int]] = {
        h["name"]: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "correct": 0, "total": 0}
        for h in head_specs
    }
    amp_dtype = torch.bfloat16

    with torch.no_grad():
        for batch in loader:
            pixel = batch["pixel"].to(device, non_blocking=True)
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pixel)

            for h in head_specs:
                name = h["name"]
                if name not in logits:
                    continue
                pred = logits[name].argmax(dim=-1)
                valid = masks[name].bool()
                y = labels[name]
                if int(valid.sum().item()) == 0:
                    continue
                yv = y[valid]
                pv = pred[valid]
                confusion[name]["total"] += int(valid.sum().item())
                confusion[name]["correct"] += int((pv == yv).sum().item())
                # Binary head tp/fp/tn/fn (for num_classes==2; else only total/correct)
                if int(h["num_classes"]) == 2:
                    confusion[name]["tp"] += int(((pv == 1) & (yv == 1)).sum().item())
                    confusion[name]["fp"] += int(((pv == 1) & (yv == 0)).sum().item())
                    confusion[name]["tn"] += int(((pv == 0) & (yv == 0)).sum().item())
                    confusion[name]["fn"] += int(((pv == 0) & (yv == 1)).sum().item())

    duration = time.time() - t0

    rows: List[Dict[str, Any]] = []
    metric_summary: Dict[str, float] = {}
    for h in head_specs:
        name = h["name"]
        c = confusion[name]
        acc = (c["correct"] / c["total"]) if c["total"] > 0 else None
        row = {
            "head": name,
            "split": split,
            "total": c["total"],
            "correct": c["correct"],
            "accuracy": acc,
        }
        if int(h["num_classes"]) == 2 and c["total"] > 0:
            precision = c["tp"] / max(1, c["tp"] + c["fp"])
            recall = c["tp"] / max(1, c["tp"] + c["fn"])
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            row.update({
                "tp": c["tp"], "fp": c["fp"], "tn": c["tn"], "fn": c["fn"],
                "precision": precision, "recall": recall, "f1": f1,
            })
            if acc is not None:
                metric_summary[f"eval/acc_{name}"] = acc
                metric_summary[f"eval/f1_{name}"] = f1
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = _resolve_metrics_path(cfg)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Also dump json for human inspection
    json_path = output_path.replace(".parquet", ".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "duration_s": duration}, fh, indent=2, default=str)

    print(f"[eval] split={split}, {len(rows)} heads, duration={duration:.1f}s → "
          f"{output_path}", flush=True)
    return {
        "metrics_path": output_path,
        "metrics": metric_summary,
        "metadata": {"split": split, "duration_s": duration},
    }


def _resolve_metrics_path(cfg: DictConfig) -> str:
    explicit = cfg.eval.get("metrics_path", None)
    if explicit:
        return os.path.abspath(str(explicit))
    return os.path.abspath(os.path.join("outputs", "urbanvit", "eval_metrics.parquet"))
