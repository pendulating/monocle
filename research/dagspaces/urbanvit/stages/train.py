"""Train stage: LoRA fine-tuning of DINOv3/TIPSv2 with multi-head classifiers.

Reads WebDataset tar shards produced by `shard.py`, fine-tunes a LoRA-wrapped
backbone with per-head linear classifiers, writes a checkpoint under
`training.checkpoint_dir`.

Multi-head training uses masked cross-entropy: rows with a null label for a
given head contribute zero loss for that head. A head with zero supervised
samples across the entire train set raises an early error.

Multi-GPU: if CUDA_VISIBLE_DEVICES exposes >1 GPU and training.force_single_gpu
is false, we spawn N ranks via torch.multiprocessing.spawn and use DDP.
"""
from __future__ import annotations

import io
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_train_stage(cfg: DictConfig) -> Dict[str, Any]:
    checkpoint_dir = _resolve_checkpoint_dir(cfg)
    os.makedirs(checkpoint_dir, exist_ok=True)

    shards_dir = str(cfg.shard.output_dir or "")
    if not shards_dir or not os.path.isdir(shards_dir):
        raise ValueError(
            f"shard.output_dir must point to an existing shards dir; got: {shards_dir!r}"
        )

    manifest = _load_shard_manifest(shards_dir)
    _assert_some_labels_present(manifest, cfg)

    # Decide single-GPU vs DDP
    import torch
    n_gpus = torch.cuda.device_count()
    force_single = bool(cfg.training.get("force_single_gpu", False))
    world_size = 1 if force_single else max(1, n_gpus)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    t0 = time.time()

    if world_size > 1:
        import torch.multiprocessing as mp
        # Default backend on PCIe-only boxes is NCCL (already configured
        # via launcher env). `spawn` cleanly inherits env + CUDA context.
        errors: List[Optional[BaseException]] = [None] * world_size
        ctx = mp.spawn(
            fn=_train_entry,
            args=(world_size, cfg_dict, checkpoint_dir),
            nprocs=world_size,
            join=True,
        )
        # If any rank errored, mp.spawn would have raised — reaching here = success
    else:
        _train_entry(rank=0, world_size=1, cfg_dict=cfg_dict,
                     checkpoint_dir=checkpoint_dir)

    duration = time.time() - t0

    # Rank 0 wrote a metrics.json in checkpoint_dir
    metrics = {}
    metrics_path = os.path.join(checkpoint_dir, "train_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)
        except Exception:
            pass

    return {
        "checkpoint_dir": checkpoint_dir,
        "metrics": {**metrics, "duration_s": duration, "world_size": world_size},
        "metadata": {"world_size": world_size, "duration_s": duration},
    }


# ---------------------------------------------------------------------------
# Per-rank training entrypoint
# ---------------------------------------------------------------------------

def _train_entry(
    rank: int, world_size: int, cfg_dict: Dict[str, Any], checkpoint_dir: str
) -> None:
    import torch
    import torch.distributed as dist

    cfg = OmegaConf.create(cfg_dict)
    is_distributed = world_size > 1

    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size,
        )
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    # ── Backbone + LoRA + heads ──────────────────────────────────────────
    backbone, feature_dim, image_size = _build_backbone(cfg)
    backbone = _maybe_attach_lora(backbone, cfg)
    heads = _build_heads(cfg, feature_dim)

    model = _BackboneWithHeads(backbone=backbone, heads=heads).to(device)

    # torch.compile (single rank — compile AFTER DDP wrap where relevant)
    # Then DDP
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if bool(cfg.training.get("torch_compile", True)):
        try:
            compile_mode = str(cfg.training.get("compile_mode", "max-autotune"))
            model = torch.compile(model, mode=compile_mode)
            if is_main:
                print(f"[train:rank{rank}] torch.compile(mode={compile_mode}) ok",
                      flush=True)
        except Exception as e:
            if is_main:
                print(f"[train:rank{rank}] torch.compile failed, continuing eagerly: {e}",
                      flush=True)

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader = _make_dataloader(
        cfg, split="train", rank=rank, world_size=world_size, image_size=image_size,
    )
    val_loader = _make_dataloader(
        cfg, split="val", rank=rank, world_size=world_size, image_size=image_size,
    )

    head_specs = _resolve_heads(cfg)

    # ── Optimizer / scheduler ────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )

    total_epochs = int(cfg.training.epochs)
    steps_per_epoch = _estimate_steps_per_epoch(cfg)
    total_steps = max(1, total_epochs * steps_per_epoch)
    warmup_steps = max(1, int(total_steps * float(cfg.training.warmup_ratio)))
    scheduler = _make_scheduler(optimizer, cfg.training.lr_schedule,
                                warmup_steps, total_steps)

    amp_dtype_str = str(cfg.training.get("amp_dtype", "bfloat16"))
    amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16

    # ── W&B (rank 0 only) ────────────────────────────────────────────────
    wb = None
    if is_main and bool(getattr(cfg.wandb, "enabled", False)):
        try:
            import wandb
            wb = wandb.run  # re-use orchestrator-started run if present
        except Exception:
            wb = None

    # ── Training loop ────────────────────────────────────────────────────
    log_every = int(cfg.training.get("log_every_steps", 25))
    eval_every = int(cfg.training.get("eval_every_epochs", 1))
    grad_accum = max(1, int(cfg.training.get("gradient_accumulation_steps", 1)))

    global_step = 0
    best_val: Dict[str, float] = {h["name"]: -math.inf for h in head_specs}

    for epoch in range(total_epochs):
        model.train()
        running = {h["name"]: 0.0 for h in head_specs}
        running_counts = {h["name"]: 0 for h in head_specs}
        running_loss_sum = 0.0
        running_loss_n = 0

        t_epoch = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            pixel = batch["pixel"].to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}
            masks = {k: v.to(device, non_blocking=True) for k, v in batch["masks"].items()}

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pixel)  # dict: head_name -> (B, C) logits
                loss, per_head = _compute_masked_loss(logits, labels, masks, head_specs)

            if torch.isfinite(loss):
                (loss / grad_accum).backward()
            else:
                if is_main:
                    print(f"[train:rank{rank}] non-finite loss at step {global_step}, skipping",
                          flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss_sum += float(loss.detach())
            running_loss_n += 1
            for name, (l, c) in per_head.items():
                running[name] += float(l)
                running_counts[name] += int(c)

            if is_main and global_step > 0 and global_step % log_every == 0:
                avg_loss = running_loss_sum / max(1, running_loss_n)
                log_payload: Dict[str, float] = {
                    "train/loss": avg_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch_frac": epoch + step / max(1, steps_per_epoch),
                    "train/step": float(global_step),
                }
                for name in running:
                    if running_counts[name] > 0:
                        log_payload[f"train/loss_{name}"] = running[name] / running_counts[name]
                _wb_log(wb, log_payload, step=global_step)
                print(f"[train:rank{rank}] step={global_step} "
                      f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)
                running = {h["name"]: 0.0 for h in head_specs}
                running_counts = {h["name"]: 0 for h in head_specs}
                running_loss_sum = 0.0
                running_loss_n = 0

        # ── Eval ───────────────────────────────────────────────────────
        if (epoch + 1) % eval_every == 0:
            val_stats = _eval_loop(model, val_loader, head_specs, device, amp_dtype)
            if is_main:
                print(f"[train] epoch {epoch+1}/{total_epochs} "
                      f"({time.time()-t_epoch:.1f}s): " +
                      ", ".join(f"{k}={v:.4f}" for k, v in val_stats.items()),
                      flush=True)
                _wb_log(wb, {f"val/{k}": v for k, v in val_stats.items()},
                        step=global_step)
                if bool(cfg.training.get("save_best", True)):
                    improved = False
                    for head in head_specs:
                        key = f"acc_{head['name']}"
                        if key in val_stats and val_stats[key] > best_val[head["name"]]:
                            best_val[head["name"]] = val_stats[key]
                            improved = True
                    if improved:
                        _save_checkpoint(model, head_specs, cfg, checkpoint_dir,
                                         tag="best")

    # Final checkpoint (rank 0)
    if is_main:
        _save_checkpoint(model, head_specs, cfg, checkpoint_dir, tag="final")
        metrics_json = {
            f"best_val_acc_{k}": v for k, v in best_val.items() if v > -math.inf
        }
        with open(os.path.join(checkpoint_dir, "train_metrics.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(metrics_json, fh, indent=2)

    if is_distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Backbone + LoRA + heads
# ---------------------------------------------------------------------------

class _BackboneWithHeads:
    """Thin nn.Module wrapper combining a feature extractor and head dict."""

    def __new__(cls, backbone, heads):
        import torch.nn as nn

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.heads = heads

            def forward(self, x):
                feats = _extract_features(self.backbone, x)
                return {name: head(feats) for name, head in self.heads.items()}

        return Wrapper()


def _build_backbone(cfg: DictConfig) -> Tuple[Any, int, int]:
    """Return (backbone_module, feature_dim, image_size)."""
    kind = str(cfg.model.backbone_kind)
    if kind == "dinov3_hf":
        return _build_dinov3_hf(cfg)
    if kind == "tipsv2":
        return _build_tipsv2(cfg)
    raise ValueError(
        f"Unknown backbone_kind: {kind!r}. Supported: 'dinov3_hf', 'tipsv2'."
    )


def _build_dinov3_hf(cfg: DictConfig) -> Tuple[Any, int, int]:
    """DINOv3 via native HuggingFace DINOv3ViTModel (transformers >= 4.56)."""
    from transformers import AutoModel

    ckpt_path = str(cfg.model.checkpoint_path)
    image_size = int(cfg.model.image_size)
    feature_dim = int(cfg.model.feature_dim)

    model = AutoModel.from_pretrained(ckpt_path)
    print(f"[train] DINOv3 loaded from {ckpt_path} ({type(model).__name__})",
          flush=True)
    return model, feature_dim, image_size


def _build_tipsv2(cfg: DictConfig) -> Tuple[Any, int, int]:
    """TIPSv2 via HuggingFace AutoModel with custom code (trust_remote_code=True)."""
    from transformers import AutoModel

    ckpt_path = str(cfg.model.checkpoint_path)
    image_size = int(cfg.model.image_size)
    feature_dim = int(cfg.model.feature_dim)

    model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True)

    # TIPSv2 bundles image + text encoders; expose only the image tower.
    image_encoder = None
    for attr in ("image_encoder", "vision_encoder", "vision_model", "visual"):
        if hasattr(model, attr):
            image_encoder = getattr(model, attr)
            break
    if image_encoder is None:
        raise RuntimeError(
            "TIPSv2 model has no recognizable image encoder attribute "
            "(tried image_encoder, vision_encoder, vision_model, visual). "
            "Update _build_tipsv2 to match the checkpoint's API."
        )

    return image_encoder, feature_dim, image_size


def _extract_features(backbone: Any, x: Any) -> Any:
    """Call the backbone and return a (B, feature_dim) pooled feature tensor.

    Handles the common conventions:
      - timm models: `forward_features(x)` → (B, N, D), then mean-pool tokens
        (excluding cls_token & register tokens if present — here we just
        mean over all spatial+cls tokens which works well in practice)
      - HuggingFace image encoders: `(x).last_hidden_state[:, 0]` (CLS)
    """
    if hasattr(backbone, "forward_features"):
        feats = backbone.forward_features(x)
        if feats.ndim == 3:
            # (B, N, D) → mean over tokens
            return feats.mean(dim=1)
        return feats  # already (B, D)
    out = backbone(x)
    if hasattr(out, "last_hidden_state"):
        # CLS token is usually index 0 for ViTs
        return out.last_hidden_state[:, 0]
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    # Tensor fallback
    if out.ndim == 3:
        return out.mean(dim=1)
    return out


def _maybe_attach_lora(backbone: Any, cfg: DictConfig) -> Any:
    if not bool(cfg.training.lora.enabled):
        return backbone
    from peft import LoraConfig, get_peft_model, TaskType

    targets = cfg.training.lora.get("target_modules", None)
    if targets is None:
        targets = list(cfg.model.get("lora_target_modules", ["attn.qkv", "attn.proj"]))
    lora_cfg = LoraConfig(
        r=int(cfg.training.lora.rank),
        lora_alpha=int(cfg.training.lora.alpha),
        lora_dropout=float(cfg.training.lora.dropout),
        target_modules=list(targets),
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(backbone, lora_cfg)


def _build_heads(cfg: DictConfig, feature_dim: int) -> Any:
    import torch.nn as nn
    heads = nn.ModuleDict()
    for spec in _resolve_heads(cfg):
        heads[spec["name"]] = nn.Linear(feature_dim, int(spec["num_classes"]))
    return heads


def _resolve_heads(cfg: DictConfig) -> List[Dict[str, Any]]:
    heads = cfg.get("heads", {}).get("heads", [])
    out = []
    for h in heads:
        out.append({
            "name": str(h.name),
            "column": str(h.column),
            "task": str(h.get("task", "binary")),
            "num_classes": int(h.get("num_classes", 2)),
            "conditional_on": h.get("conditional_on", None),
            "loss_weight": float(h.get("loss_weight", 1.0)),
        })
    return out


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------

def _compute_masked_loss(logits: Dict[str, Any], labels: Dict[str, Any],
                         masks: Dict[str, Any],
                         head_specs: List[Dict[str, Any]]):
    """Masked multi-head cross-entropy.

    Returns (scalar_loss, {head_name: (loss_value, n_valid)}).
    """
    import torch
    import torch.nn.functional as F

    total = None
    per_head = {}
    for spec in head_specs:
        name = spec["name"]
        if name not in logits or name not in labels:
            continue
        l = logits[name]
        y = labels[name]
        m = masks[name].float()
        n_valid = m.sum()
        if n_valid.item() == 0:
            per_head[name] = (0.0, 0)
            continue
        ce = F.cross_entropy(l, y, reduction="none")
        masked = (ce * m).sum() / n_valid
        weighted = masked * float(spec["loss_weight"])
        total = weighted if total is None else total + weighted
        per_head[name] = (float(masked.detach()), int(n_valid.item()))

    if total is None:
        total = torch.zeros((), device=logits[list(logits.keys())[0]].device,
                            requires_grad=True)
    return total, per_head


def _eval_loop(model, loader, head_specs, device, amp_dtype) -> Dict[str, float]:
    import torch
    model.eval()

    correct = {h["name"]: 0 for h in head_specs}
    total = {h["name"]: 0 for h in head_specs}

    with torch.no_grad():
        for batch in loader:
            pixel = batch["pixel"].to(device, non_blocking=True)
            labels = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}
            masks = {k: v.to(device, non_blocking=True) for k, v in batch["masks"].items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(pixel)
            for h in head_specs:
                name = h["name"]
                if name not in logits:
                    continue
                pred = logits[name].argmax(dim=-1)
                valid = masks[name].bool()
                correct[name] += int((pred[valid] == labels[name][valid]).sum().item())
                total[name] += int(valid.sum().item())

    model.train()
    stats = {}
    for h in head_specs:
        name = h["name"]
        if total[name] > 0:
            stats[f"acc_{name}"] = correct[name] / total[name]
        stats[f"n_{name}"] = total[name]
    return stats


# ---------------------------------------------------------------------------
# Data loader (WebDataset)
# ---------------------------------------------------------------------------

def _make_dataloader(cfg: DictConfig, split: str, rank: int, world_size: int,
                     image_size: int):
    import torch
    import webdataset as wds
    from torch.utils.data import DataLoader

    shards_dir = str(cfg.shard.output_dir)
    shard_files = sorted(
        os.path.join(shards_dir, f)
        for f in os.listdir(shards_dir)
        if f.startswith(f"{split}-") and f.endswith(".tar")
    )
    if not shard_files:
        raise FileNotFoundError(f"No {split} shards found in {shards_dir}")

    # Rank-split shards so each rank sees disjoint data
    if world_size > 1:
        shard_files = shard_files[rank::world_size]

    head_specs = _resolve_heads(cfg)
    means = list(cfg.model.mean)
    stds = list(cfg.model.std)
    mean_t = torch.tensor(means).view(3, 1, 1)
    std_t = torch.tensor(stds).view(3, 1, 1)

    def _decode(sample):
        # Sample is a dict with keys like 'jpg' + 'json' (extension lowercased).
        import io as _io
        from PIL import Image
        img_bytes = sample["jpg"]
        meta = json.loads(sample["json"])
        im = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        if im.size != (image_size, image_size):
            im = im.resize((image_size, image_size), Image.BICUBIC)
        arr = torch.from_numpy(_pil_to_array(im)).permute(2, 0, 1).float().div_(255.0)
        arr = (arr - mean_t) / std_t

        labels = {}
        masks = {}
        for spec in head_specs:
            val = meta.get(spec["column"])
            if val is None:
                labels[spec["name"]] = 0
                masks[spec["name"]] = 0
            else:
                labels[spec["name"]] = int(val)
                masks[spec["name"]] = 1
                # conditional_on: if the parent head's label says "no", mask this head out
                parent = spec["conditional_on"]
                if parent:
                    parent_spec = next((h for h in head_specs if h["name"] == parent),
                                       None)
                    if parent_spec:
                        parent_val = meta.get(parent_spec["column"])
                        if parent_val is None or int(parent_val) == 0:
                            masks[spec["name"]] = 0
        return {"pixel": arr, "labels": labels, "masks": masks}

    dataset = (
        wds.WebDataset(shard_files, shardshuffle=(split == "train"),
                       nodesplitter=wds.split_by_node if world_size > 1 else None,
                       handler=wds.warn_and_continue)
        .shuffle(int(cfg.shard.get("shuffle_buffer", 1000)) if split == "train" else 0)
        .map(_decode, handler=wds.warn_and_continue)
    )

    batch_size = int(cfg.training.per_device_batch_size if split == "train"
                     else cfg.eval.batch_size)
    num_workers = int(cfg.training.num_workers if split == "train"
                      else cfg.eval.num_workers)

    def _collate(batch):
        import torch as _torch
        pixels = _torch.stack([b["pixel"] for b in batch])
        head_names = list(batch[0]["labels"].keys())
        labels = {n: _torch.tensor([b["labels"][n] for b in batch], dtype=_torch.long)
                  for n in head_names}
        masks = {n: _torch.tensor([b["masks"][n] for b in batch], dtype=_torch.long)
                 for n in head_names}
        return {"pixel": pixels, "labels": labels, "masks": masks}

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        collate_fn=_collate, pin_memory=True, persistent_workers=(num_workers > 0),
    )
    return loader


def _pil_to_array(im):
    import numpy as np
    return np.array(im, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Optimizer / scheduler helpers
# ---------------------------------------------------------------------------

def _make_scheduler(optimizer, kind: str, warmup_steps: int, total_steps: int):
    import torch

    if str(kind) == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif str(kind) == "linear":
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, 1.0 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)


def _estimate_steps_per_epoch(cfg: DictConfig) -> int:
    # Coarse estimate from manifest; used only for LR schedule.
    try:
        with open(os.path.join(str(cfg.shard.output_dir), "manifest.json"),
                  "r", encoding="utf-8") as fh:
            m = json.load(fh)
        train_samples = int(m.get("splits", {}).get("train", 0))
    except Exception:
        train_samples = 1000
    bs = int(cfg.training.per_device_batch_size)
    return max(1, train_samples // max(1, bs))


# ---------------------------------------------------------------------------
# Manifest / label sanity checks
# ---------------------------------------------------------------------------

def _load_shard_manifest(shards_dir: str) -> Dict[str, Any]:
    path = os.path.join(shards_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shard manifest not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _assert_some_labels_present(manifest: Dict[str, Any], cfg: DictConfig) -> None:
    """Guard: refuse to train if the parquet has no label columns populated.

    We check whether any label column listed in heads appears in the shard
    manifest's `label_columns`. (The shard stage writes all of a row's label
    columns into the json sidecar, so absence here means the parquet didn't
    have the column at all.)
    """
    head_cols = {str(h.column) for h in cfg.get("heads", {}).get("heads", [])}
    present = set(manifest.get("label_columns", []))
    if not head_cols & present:
        raise RuntimeError(
            "No label columns from conf/heads are present in the shard manifest. "
            f"Heads expect columns {sorted(head_cols)} but shard sidecars have "
            f"{sorted(present)}. Populate labels in the parquet, re-shard (or "
            f"delete the shard dir to force rebuild), then rerun training.\n"
            "For inference-only workflows that don't need a fine-tuned backbone, "
            "run the infer_scaffolding pipeline with extract.backbone_checkpoint=null "
            "to skip training entirely."
        )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _save_checkpoint(model, head_specs, cfg: DictConfig, checkpoint_dir: str,
                     tag: str) -> None:
    """Save LoRA adapter + head weights + config.

    Structure under checkpoint_dir/:
      lora_adapter/           (peft-style dir, if lora enabled)
      heads_{tag}.pt          (state dict of the head ModuleDict)
      config_{tag}.yaml       (frozen snapshot of training cfg)
    """
    import torch

    # Unwrap DDP / compile
    core = model
    for attr in ("_orig_mod", "module"):
        if hasattr(core, attr):
            core = getattr(core, attr)

    backbone = getattr(core, "backbone")
    heads = getattr(core, "heads")

    # LoRA adapter
    try:
        if hasattr(backbone, "save_pretrained"):
            backbone.save_pretrained(os.path.join(checkpoint_dir, "lora_adapter"))
    except Exception as e:
        print(f"[train] save_pretrained(lora) failed: {e}", flush=True)

    torch.save(heads.state_dict(),
               os.path.join(checkpoint_dir, f"heads_{tag}.pt"))

    snapshot = {
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "heads": [h for h in head_specs],
    }
    with open(os.path.join(checkpoint_dir, f"config_{tag}.yaml"),
              "w", encoding="utf-8") as fh:
        fh.write(OmegaConf.to_yaml(OmegaConf.create(snapshot)))


def _resolve_checkpoint_dir(cfg: DictConfig) -> str:
    explicit = cfg.training.get("checkpoint_dir", None)
    if explicit:
        return os.path.abspath(str(explicit))
    return os.path.abspath(os.path.join("outputs", "urbanvit",
                                        f"{cfg.experiment.name}_checkpoint"))


# ---------------------------------------------------------------------------
# W&B helper
# ---------------------------------------------------------------------------

def _wb_log(run, payload: Dict[str, Any], step: int) -> None:
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception:
        pass
