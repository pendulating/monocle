# UrbanViT Dagspace — Implementation Plan

**Status:** proposed — awaiting scaffold
**Owner:** mllmsci
**Target path:** `dagspaces/urbanvit/`
**Purpose:** High-throughput ViT training and inference over Cyclomedia street view imagery. First concrete use case: scaffolding classification (three-way: none / standard green / fancy white).

## Why a new dagspace

All existing dagspaces wrap VLM inference via vLLM. UrbanViT is the home for pure-ViT workloads — frozen or LoRA-fine-tuned DINOv3 / TIPSv2 backbones, multi-head classifiers, and the optimized data pipeline needed to run them over 3M–36M images. Keeping ViT-specific patterns (DALI/FFCV, `torch.compile`, LoRA, DDP training) in one dagspace prevents cross-contamination with the VLM inference stack.

## Goals

1. **Feature cache once, task-head forever.** Run the (optionally fine-tuned) backbone once per image, cache 1024-dim features to parquet. Every future classifier head reuses the cache. Training stage also writes checkpoints that `extract` can pick up.
2. **Train and infer in the same dagspace.** LoRA fine-tuning (multi-head, multi-task with per-head masked loss) plus large-scale inference over the curated Cyclomedia corpus.
3. **Scale-aware shard format.** FFCV for labeled training sets (fits /scratch); WebDataset + DALI for 10M+ inference corpora (stream sequentially from shared FS).
4. **Multi-head from day one.** Configurable list of heads; adding a new urban concept later is a config change, not a code change.

## Scope & constraints

| Axis | Decision |
|------|----------|
| Models | DINOv3 ViT-L/16 HF native (`/share/pierson/matt/zoo/models/dinov3-vitl16-pretrain-lvd1689m/`), TIPSv2 ViT-B/14 (`/share/pierson/matt/zoo/models/tipsv2-b14/`) |
| Fine-tuning | LoRA on attention projections (q/k/v); backbone otherwise frozen |
| Precision | bf16 autocast, `torch.compile` (`max-autotune`) |
| Compute | A6000 on the `klara` SLURM partition (sm_86, PCIe-only, no NVLink) |
| Storage (fast) | `/scratch` (up to 4 TB), shards evicted to shared once per-run quota exceeded |
| Storage (long-term) | `/share/pierson/matt/mllmsci/...` for features parquets and checkpoints |
| First corpus | `curation/scaffolding_permits_2020_through_2025/cyclomedia_near_permits.parquet` (3.97M rows, 52 columns) |
| Label acquisition | Labels appended to the same parquet in a future pass (bootstrapped via embedding-retrieval: top/bottom scoring examples → weak labels) |
| Multi-GPU regime | DDP via `torchrun` for training; SLURM array of independent workers for inference (shard-the-work, not shard-the-model) |

## Directory layout

```
dagspaces/urbanvit/
├── __init__.py
├── __main__.py
├── cli.py
├── config_schema.py          # ViT-specific extensions to PipelineGraphSpec
├── orchestrator.py
├── stages/
│   ├── __init__.py
│   ├── shard.py              # labeled/unlabeled parquet → FFCV or WebDataset shards on /scratch
│   ├── train.py              # DDP LoRA + multi-head with masked loss, bf16 + compile
│   ├── eval.py               # per-head metrics on held-out split
│   ├── extract.py            # sharded → compiled backbone → features parquet
│   ├── classify.py           # trained heads over cached features → scores parquet
│   └── collate.py            # concat shard outputs + optional /scratch → shared offload
└── conf/
    ├── config.yaml
    ├── data/
    │   └── cyclomedia_scaffolding.yaml
    ├── model/
    │   ├── dinov3_vitb16.yaml
    │   └── tipsv2_vitb14.yaml
    ├── shard/
    │   ├── ffcv.yaml
    │   └── webdataset.yaml
    ├── heads/
    │   └── scaffolding_3way.yaml
    ├── pipeline/
    │   ├── train_scaffolding.yaml
    │   └── infer_scaffolding.yaml
    └── hydra/launcher/
        ├── slurm_gpu_klara_1x.yaml
        ├── slurm_gpu_klara_2x.yaml
        └── slurm_gpu_klara_4x.yaml
```

## Pipeline DAGs

**Training** (`train_scaffolding.yaml`)

```
shard  →  train  →  eval
```

- `shard`: labeled parquet → FFCV train/val/test shards on `/scratch/$USER/urbanvit/shards/<experiment>/`
- `train`: DDP across N GPUs via `torchrun` within a single submitit task; writes LoRA adapter + per-head classifier weights to `${paths.output_dir}/checkpoint/`
- `eval`: runs the trained model on held-out split, dumps per-head metrics to parquet

**Inference** (`infer_scaffolding.yaml`)

```
shard  →  extract  →  classify  →  collate
```

- `shard`: inference parquet → WebDataset shards (default for large corpora) or FFCV (small)
- `extract`: SLURM array (1 task / shard) → DALI loader → compiled backbone (bf16) → features parquet per shard. Accepts an optional backbone checkpoint (defaults to pretrained).
- `classify`: loads trained heads and applies them to cached features → (image_id, head, score) parquet. Cheap — repeat as heads are added.
- `collate`: concat shard outputs + optional offload of `/scratch` shards to shared for archival

## Multi-head configuration

Heads are declared as a list; training masks the loss per-sample where a given label column is null.

```yaml
# conf/heads/scaffolding_3way.yaml
heads:
  - name: scaffolding_any
    column: label_scaffolding          # future parquet column
    task: binary
    num_classes: 2
  - name: fancy_vs_standard
    column: label_fancy
    task: binary
    num_classes: 2
    conditional_on: scaffolding_any    # only train on rows where scaffolding_any == 1
```

Adding "sidewalk_condition" or any future urban concept = append to this list; no code change.

## Split strategy — **DECISION**

**Split by `recording_id`** (default). Each Cyclomedia recording has 4 faces (L/B/R/F) captured simultaneously from the same pano at the same (lat, lon). A random row-level split puts different faces of the same recording into train and val, leaking the same scene across splits and inflating val metrics.

### Why not the coarser `group` unit

The `group` column (the 5-char `recording_id` prefix, e.g., `W0E6C`) clusters multiple nearby recordings — typically same block / same drive-through. Splitting by `group` would close the "nearby panos, different cars" leakage channel too.

**Rejected for v1** because `group`-level splits are not well-balanced: group sizes vary substantially (some streets have many recordings, others few), so a random group-level split produces lumpy train/val/test sizes and skewed class distributions. The balance problem isn't insurmountable but needs a more deliberate strategy than we want to build right now.

### Future upgrade path

When we have real labels and can see whether val metrics look suspicious, revisit with an **H3-based spatial partitioning** scheme: bin recordings by H3 cell (resolution ~9 or ~10), then split by cell. H3 gives balanced, deterministic spatial bins — you can hash cells into folds with controlled size distribution. Roughly:

```
split_fold(recording) = hash(h3_cell(lat, lon, resolution=9)) % n_folds
```

This closes the nearby-pano leakage channel while keeping splits balanced. Defer until the simpler `recording_id` split shows a measurable leakage signal.

### Implementation

- Split column: `recording_id`
- Stratification within splits: `borough` × `year` (so train/val/test each see the same mix of boroughs and capture years)
- Ratios: 80/10/10 train/val/test by default, configurable

## Models & LoRA

- Loader abstraction in `stages/train.py` (`_build_backbone`): dispatches on `model.backbone_kind`.
- DINOv3 ViT-L/16: `AutoModel.from_pretrained(...)` using the native `DINOv3ViTModel` class (transformers ≥ 4.56). 224² input, 1024-dim CLS feature, 305 M params.
- TIPSv2 ViT-B/14: `AutoModel.from_pretrained(..., trust_remote_code=True)` — custom-code HF model. Image encoder plucked from the wrapper module. 448² input, 768-dim features.
- LoRA via `peft` on `q_proj`, `k_proj`, `v_proj` (substring-matched across all 24 layers for DINOv3). Default r=16, α=32, dropout 0.05 → 2.36 M trainable (0.77% of 305 M).
- Classifier heads: single `Linear(feature_dim, num_classes)` per head in an `nn.ModuleDict`.

## Shard stage — format selection

| Corpus size | Default shard format | Rationale |
|-------------|---------------------|-----------|
| ≤ 4M (fits /scratch) | FFCV `.beton` | Pre-decoded packed tensors → zero CPU work at read time |
| > 4M | WebDataset `.tar` + DALI GPU decode | Sequential reads from shared FS, GPU JPEG decode |

Selection is a Hydra override: `shard=ffcv` or `shard=webdataset`.

## Launchers (to be created)

`dagspaces/common/conf/hydra/launcher/slurm_gpu_klara_{1x,2x,4x}.yaml` — clones of the existing `slurm_gpu_Nx.yaml` with `partition: klara` pinned and `gres: gpu:N` set for A6000s. The existing 4x launcher already has `TORCH_CUDA_ARCH_LIST=8.6` and correct NCCL flags for PCIe-only boxes.

## Open items

- Label acquisition pipeline — bootstrapped via embedding retrieval from `urbanembed`. Out of scope for urbanvit itself; label parquet is an input.
- H3 spatial partitioning for splits — deferred, see "Future upgrade path" above.
- TensorRT / INT8 path for production inference — deferred until throughput need is measured.
- FP8 — not available on A6000 (Ampere), skip.

## Success criteria

1. `shard` + `extract` + `classify` run end-to-end on the unlabeled 3.97M parquet with pretrained DINOv3 (no heads yet, just features + dummy scores).
2. Training stage runs end-to-end on a small synthetic labeled sample (e.g., 10k rows with injected random labels) and produces a checkpoint that `extract` + `classify` consume.
3. Per-head masked loss correctly handles rows where a label column is null.
4. Splits are verified non-leaking: no `recording_id` appears in more than one split.
