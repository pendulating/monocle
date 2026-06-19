---
title: "UrbanViT — ViT Training + Inference"
category: dagspace
created: 2026-04-21
updated: 2026-04-21
tags:
  - dagspace
  - vit
  - dinov3
  - tipsv2
  - lora
  - feature-extraction
  - classification
  - scaffolding
---

# UrbanViT — ViT Training + Inference

UrbanViT is the dagspace for **pure-ViT workloads over urban imagery** — LoRA fine-tuning of pretrained DINOv3 / TIPSv2 backbones, multi-head linear classifiers, and optimized batch feature extraction over millions of Cyclomedia street-view images. It's the home for any vision-only (non-VLM) inference in the project.

First use case: 3-way scaffolding classification (no scaffold / standard green / fancy white) over the `scaffolding_permits_2020_through_2025` corpus (3.97 M Cyclomedia rows near DOB permits).

## Purpose

- LoRA fine-tune ViT-B/16 (DINOv3) or ViT-B/14 (TIPSv2) on curated labeled data
- Cache 1024-dim features once, reuse for unlimited task heads (scaffolding today, sidewalk quality tomorrow, …)
- Run inference over 3 M – 36 M image corpora with sharded SLURM jobs on the `klara` A6000 partition
- Multi-head training with **per-head masked loss**: rows with null labels don't contribute to that head's gradient

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanvit/cli.py` | Hydra CLI entry point |
| `dagspaces/urbanvit/orchestrator.py` | DAG executor; defines six `StageRunner` subclasses |
| `dagspaces/urbanvit/stages/shard.py` | Parquet → WebDataset tar shards on `/scratch`, group-aware stratified split |
| `dagspaces/urbanvit/stages/train.py` | DDP + LoRA + multi-head training loop, bf16 autocast, `torch.compile` |
| `dagspaces/urbanvit/stages/eval.py` | Held-out split evaluation → per-head metrics parquet |
| `dagspaces/urbanvit/stages/extract.py` | Sharded feature extraction → features parquet per shard |
| `dagspaces/urbanvit/stages/classify.py` | Apply trained heads to cached features → scores parquet |
| `dagspaces/urbanvit/stages/collate.py` | Concat shard outputs + optional `/scratch` → shared offload |
| `dagspaces/urbanvit/conf/` | Hydra configs (config + data + model + shard + heads + pipeline) |

## Pipelines

Two pipeline YAMLs, both composed on top of the base `config.yaml`:

### `pipeline=train_scaffolding`

```
shard → train → eval
```

- **shard** (1 × A6000, CPU-heavy): reads parquet, computes group-aware split, writes tar shards to `/scratch/$USER/urbanvit/shards/<experiment>/`
- **train** (4 × A6000 DDP): spawns ranks via `torch.multiprocessing.spawn`, LoRA-fine-tunes backbone, saves `lora_adapter/` + `heads_final.pt` + `heads_best.pt` + `train_metrics.json`
- **eval** (1 × A6000): loads checkpoint, runs val split, writes `eval_metrics.parquet`

### `pipeline=infer_scaffolding`

```
shard → extract → classify → collate
```

- **shard**: same as train (skipped if shard dir already exists — see `shard.skip_if_exists`)
- **extract** (1 × A6000): compiled bf16 backbone → features parquet per shard. Accepts `extract.backbone_checkpoint=<dir>` to apply a fine-tuned LoRA adapter; null = pretrained.
- **classify** (1 × A6000): loads trained heads, applies to cached features → scores parquet per input parquet
- **collate** (monitor node): concat + optional `/scratch` archival

Extract is the throughput bottleneck. For 10M+ corpora, submit multiple parallel runs with `extract.shard_subset=[start,end]` to shard the work across GPUs.

## Models

| Backbone | Source | Image size | Feature dim | Params |
|----------|--------|-----------|-------------|--------|
| DINOv3 ViT-L/16 | `/share/pierson/matt/zoo/models/dinov3-vitl16-pretrain-lvd1689m/` (HF native) | 224 | 1024 | 305 M |
| TIPSv2 ViT-B/14 | `/share/pierson/matt/zoo/models/tipsv2-b14/` (HF w/ custom code) | 448 | 768 | ~86 M |

DINOv3 uses the native `DINOv3ViTModel` class from transformers ≥ 4.56 — no `trust_remote_code`, no timm surrogate arch, no strict=False key mismatches. Attention Linears follow HF naming (`q_proj`, `k_proj`, `v_proj`, `o_proj`), so LoRA targets are the standard substring patterns. Features are the CLS token from `last_hidden_state[:, 0]` (201 tokens = 1 CLS + 4 register tokens + 196 patches at 224/16).

TIPSv2 is loaded via `AutoModel.from_pretrained(trust_remote_code=True)` and we pluck the image encoder from whatever attribute the custom class exposes.

Switch via `model=dinov3_vitl16` (default) or `model=tipsv2_vitb14`.

### LoRA trainable-param budget

With `r=16, alpha=32` on Q/K/V across all 24 layers of DINOv3 ViT-L/16:  **2.36 M trainable / 305 M total (0.77%)**.

## Multi-head configuration

Heads live in `conf/heads/*.yaml` as an ordered list. Each entry:

```yaml
- name: fancy_vs_standard
  column: label_fancy          # parquet column holding the 0/1 label
  task: binary
  num_classes: 2
  conditional_on: scaffolding_any   # optional: only train on rows where parent head == 1
  loss_weight: 1.0
```

Rows with null in `column` are masked out of that head's loss and eval. Adding a new head (e.g., `sidewalk_condition`) is pure config — no Python change required.

## Split strategy

**Group-aware split on `recording_id`, stratified by `(borough, year)`.** All 4 faces (L/B/R/F) of a single recording always land in the same split — prevents trivial leakage where faces of the same pano appear in both train and val. See [[concept-recording-id-splits]] for the leakage example and the coarser `group`-level alternative (deferred; needs H3-based balancing).

## Shard format

WebDataset tar shards (pure Python, no compiled deps). Per sample: `{sample_id}.jpg` + `{sample_id}.json` (metadata + all label columns). Configured via `shard=webdataset` (default).

FFCV (`shard=ffcv`) is a stub that raises `NotImplementedError` — FFCV install requires libjpeg-turbo + opencv compiled from source, deferred until WebDataset+torchvision is profiled and proven insufficient.

## Storage

- **Shards** → `/scratch/$USER/urbanvit/shards/<experiment>/` (up to 4 TB quota). After a run, `collate.offload.enabled=true` moves them to shared for archival.
- **Checkpoints + features parquets** → `outputs/urbanvit/<experiment>/` on shared.

At 4 M images × ~500 KB/JPEG the raw JPEG footprint (~2 TB) fits on `/scratch`. Features (1024-dim fp16) are ~8 GB for 4 M rows, ~72 GB for 36 M — stored on shared.

## Launchers

UrbanViT targets the **klara** partition (RTX A6000, sm_86, PCIe-only). Dedicated launchers:

| Launcher | GPUs | Partition |
|----------|------|-----------|
| `slurm_gpu_klara_1x` | 1 × A6000 | klara |
| `slurm_gpu_klara_2x` | 2 × A6000 | klara |
| `slurm_gpu_klara_4x` | 4 × A6000 | klara |

All three set `TORCH_CUDA_ARCH_LIST=8.6` and disable NCCL P2P/IB (required for PCIe-only boxes). Monitor node uses `slurm_monitor`.

## Related

- [[urban-embed]] — embeds are the label-acquisition path (top/bottom-scoring images → weak labels that populate parquet columns consumed by UrbanViT training)
- [[scaffolding-permits-curation]] — source of the first curated corpus
- [[cyclomedia-catalog]] — full image catalog; curated subsets derive from it
- [[slurm-deployment]] — launcher conventions
- [[shared-infrastructure]] — common modules (config_schema, orchestrator, wandb_logger) that UrbanViT reuses

## Open items

- Label acquisition via embedding retrieval (populate `label_scaffolding`, `label_fancy` columns in the parquet). Until labels exist, the train stage aborts with a helpful error.
- DALI GPU JPEG decode path inside `extract.py` — deferred, WebDataset + torchvision is the current baseline.
- H3-based spatial partitioning for splits — deferred; currently only `recording_id`-level grouping.
- TensorRT + INT8 for production inference — deferred until measured throughput demands it.
