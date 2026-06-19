---
title: Configuration System
category: config
created: 2026-04-06
updated: 2026-04-06
tags:
  - hydra
  - config
  - yaml
---

# Configuration System

MLLMSCI uses Hydra 1.3 for hierarchical configuration composition. Each dagspace has its own `conf/` directory with a root `config.yaml`, and shared configs are resolved via a common searchpath.

## Root Config Structure

Every dagspace's `config.yaml` follows this pattern (example from `dagspaces/urbanvqa/conf/config.yaml`):

```yaml
defaults:
  - _self_
  - data: inputs
  - prompt: classify
  - model: vllm_qwen3-30b
  - optional pipeline: cluster_topic

experiment:
  name: URBANVQA

runtime:
  debug: false
  sample_n: null
  output_root: null
  # ... stage-specific options

sampling_params:
  seed: 777
  temperature: 0.0
  # ...

wandb:
  enabled: true
  project: ${oc.env:WANDB_PROJECT,URBANVQA}
  entity: ${oc.env:WANDB_ENTITY,urbanekg}

hydra:
  searchpath:
    - pkg://dagspaces.common.conf
  job:
    name: ${experiment.name}
  run:
    dir: ${oc.env:HYDRA_RUN_DIR,outputs}/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: ${oc.env:HYDRA_SWEEP_DIR,multirun}/${now:%Y-%m-%d}_${experiment.name}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

### Shared Config Resolution

The `hydra.searchpath: [pkg://dagspaces.common.conf]` directive makes configs in `dagspaces/common/conf/` available to all dagspaces. Dagspace-local configs in `conf/` take precedence when both exist.

---

## Config Groups

### data/ -- Input Data

Defines the input dataset path and column mappings. Shared data configs live in `dagspaces/common/conf/data/` and are accessible to all dagspaces via the Hydra searchpath. A root-level symlink `datasets/ -> dagspaces/common/conf/data/` provides convenient access, following the same pattern as `models/` and `launchers/`.

| Field | Type | Description |
|-------|------|-------------|
| `parquet_path` | `str` | Path to input parquet file |
| `handler` | `str` | Data handler identifier (optional) |
| `columns` | `Dict[str, str]` | Column rename mapping (source -> target) |
| `metadata_columns` | `List[str]` | Columns to preserve through pipeline (optional) |

Example (`conf/data/inputs.yaml`):

```yaml
parquet_path: ${oc.env:DATA_PATH,/share/pierson/data/dataset.parquet}
columns:
  image_url: image_path
  caption: text
```

#### Shared data configs (dagspaces/common/conf/data/)

16 configs shared across all dagspaces:

| Config | Description |
|--------|-------------|
| `inputs.yaml` | Generic input spec |
| `multimodal_inputs.yaml` | Multimodal input spec |
| `vqa_inputs.yaml` | VQA input spec |
| `generic_images.yaml` | Generic image dataset |
| `cyclomedia_manhattan.yaml` | Cyclomedia Manhattan base |
| `cyclomedia_manhattan_2025_1.yaml` | Cyclomedia Manhattan 2025 batch 1 |
| `nexar_dashcam.yaml` | Nexar dashcam dataset |
| `bayflood.yaml` | BayFlood base dataset |
| `bayflood_1k.yaml` | BayFlood 1k sample |
| `bayflood_relative.yaml` | BayFlood relative comparison |
| `bayflood_nearby_floodnet.yaml` | BayFlood nearby FloodNet |
| `bayflood_sep29all.yaml` | BayFlood Sep 29 full |
| `bayflood_sep29all_relative_basic.yaml` | BayFlood Sep 29 relative (basic) |
| `bayflood_sep29all_relative_advanced.yaml` | BayFlood Sep 29 relative (advanced) |
| `bayflood_sep29all_gepaopt_query_1ft.yaml` | BayFlood Sep 29 GEPA query (1ft) |
| `bayflood_sep29all_gepaopt_query_flooded.yaml` | BayFlood Sep 29 GEPA query (flooded) |

#### Dagspace-local data configs

Some dagspaces retain local data configs that are specific to their pipeline:

- **urbanvqa**: `flattened_rules.yaml` (external project reference)
- **urbanocr**: `cyclomedia.yaml`, `cyclomedia_manhattan.yaml`, `cyclomedia_manhattan_small.yaml`, `cyclomedia_manhattan_tiny.yaml`, `cyclomedia_test_tiny.yaml`, `cyclomedia_test_w0etz.yaml` (OCR-specific handler configs)
- **urbanpairvqa**: `cyclomedia_pairwise_manhattan_2025_1.yaml` (pairwise-specific)
- **urbanroamvqa**: `cyclomedia_manhattan_2025.yaml` (roaming-specific with image_pattern/yaw fields)
- **urbanembed**: none (resolves all data configs from common)

### model/ -- vLLM Model Settings

Located in `dagspaces/common/conf/model/`. Available models:

| Config File | Model | Notes |
|-------------|-------|-------|
| `vllm_multimodal_qwen3_vl_2b.yaml` | Qwen3-VL-2B | Small VLM |
| `vllm_multimodal_qwen3_vl_2b_awq.yaml` | Qwen3-VL-2B AWQ | Quantized |
| `vllm_multimodal_qwen3_vl_2b_gptq.yaml` | Qwen3-VL-2B GPTQ | Quantized |
| `vllm_multimodal_qwen3_vl_30b.yaml` | Qwen3-VL-30B-A3B MoE | 2x GPU, TP=2 |
| `vllm_multimodal_qwen2.5_vl_3b.yaml` | Qwen2.5-VL-3B | Previous gen |
| `vllm_multimodal_qwen2.5_vl_7b.yaml` | Qwen2.5-VL-7B | Previous gen |
| `vllm_multimodal_internvl.yaml` | InternVL | Large VLM |
| `vllm_multimodal_phi3.yaml` | Phi-3 Vision | Microsoft |
| `vllm_multimodal_smolvlm.yaml` | SmolVLM | Compact |
| `vllm_multimodal_cambrian1_13b.yaml` | Cambrian-1 13B | Multi-encoder |
| `vllm_qwen3-30b.yaml` | Qwen3-30B (text-only) | MoE text model |

#### Model config structure

```yaml
model_source: /share/pierson/matt/zoo/models/Qwen3-VL-30B-A3B-Instruct

engine_kwargs:
  tensor_parallel_size: 2       # Number of GPUs for tensor parallelism
  max_model_len: 8192           # Context window limit
  max_num_seqs: 4               # Max concurrent sequences
  enable_chunked_prefill: true  # Essential for long contexts
  max_num_batched_tokens: 4096  # Throughput/memory balance
  gpu_memory_utilization: 0.80  # VRAM fraction (leave headroom)
  trust_remote_code: true       # Required for custom architectures
  dtype: bfloat16               # Model native dtype
  enforce_eager: true           # Disable CUDA graphs for stability
  disable_custom_all_reduce: false
  guided_decoding_backend: auto
  limit_mm_per_prompt:
    image: 1                    # Max images per prompt
  mm_processor_kwargs:
    min_pixels: 256
    max_pixels: 1003520

batch_size: 8
concurrency: 1
has_image: true                 # Explicit multimodal flag
```

See [[vllm-inference#Engine Configuration]] for how these are processed.

### prompt/ -- Jinja2 Templates

Each dagspace defines prompt templates in `conf/prompt/`:

| Field | Description |
|-------|-------------|
| `system_prompt` | System message content (Jinja2 template) |
| `prompt_template` | User message template with variable injection |
| `task` | Task identifier for W&B tracking |

Templates support Jinja2 variable injection from row data. Multiple prompt configs can be composed using Hydra's `@` syntax:

```yaml
defaults:
  - prompt: classify
  - prompt@prompt_taxonomy: taxonomy
  - prompt@prompt_synthesis: synthesis
  - optional prompt@prompt_decompose: decompose
```

### pipeline/ -- DAG Node Definitions

Pipeline configs define the execution graph in `conf/pipeline/*.yaml`:

```yaml
sources:
  raw_data:
    path: ${data.parquet_path}
    type: parquet

graph:
  nodes:
    classify:
      stage: classify
      depends_on: []
      inputs:
        dataset: raw_data
      outputs:
        result:
          path: classify/classified.parquet
      overrides:
        prompt: ${prompt}
        sampling_params: ${sampling_params_classify}
      launcher: slurm_gpu_2x
      max_attempts: 2
      retry_backoff_s: 30.0

    taxonomy:
      stage: taxonomy
      depends_on:
        - classify
      inputs:
        dataset: classify.result
      outputs:
        result:
          path: taxonomy/categorized.parquet
      overrides:
        prompt: ${prompt_taxonomy}
      launcher: slurm_gpu_2x
      parallel_group: inference

    decompose:
      stage: decompose
      depends_on:
        - taxonomy
      inputs:
        dataset: taxonomy.result
      outputs:
        result:
          path: decompose/decomposed.parquet
          optional: true
      wandb_suffix: decompose
```

#### Node fields

| Field | Type | Description |
|-------|------|-------------|
| `stage` | `str` | Stage type name (maps to a `StageRunner` subclass) |
| `depends_on` | `List[str]` | Node keys that must complete first |
| `inputs` | `Dict[str, str]` | Input artifact references (`alias: source_name` or `alias: node.output`) |
| `outputs` | `Dict[str, OutputSpec]` | Output artifacts with `path` (relative to output_root) and optional `optional: true` |
| `overrides` | `Dict[str, Any]` | Hydra config overrides applied to this node's config |
| `launcher` | `Optional[str]` | SLURM launcher config name (e.g., `slurm_gpu_2x`). When set, node executes as a separate SLURM job |
| `parallel_group` | `Optional[str]` | Group name for concurrent execution of independent nodes |
| `max_attempts` | `int` | Retry count (default 1) |
| `retry_backoff_s` | `float` | Seconds between retries (default 0.0) |
| `wandb_suffix` | `Optional[str]` | Custom suffix for W&B run name |

### hydra/launcher/ -- SLURM Configs

See [[slurm-deployment]] for the full launcher config reference.

---

## Runtime Config

The `runtime` section controls execution behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `debug` | `bool` | `false` | Debug mode flag |
| `sample_n` | `Optional[int]` | `null` | Sample N rows from input (seed from `MLLMSCI_SAMPLE_SEED`, default 777) |
| `output_root` | `Optional[str]` | `null` | Override output directory |
| `max_errored_blocks` | `int` | `0` | Ray Data error tolerance |
| `multimodal_enabled` | `Optional[bool]` | _(auto-detected)_ | Force multimodal on/off |
| `batch_inference` | `bool` | `true` | Enable batch processing |
| `job_memory_gb` | `int` | `32` | Job memory for launcher and Ray object store sizing |
| `suppress_child_warnings` | `bool` | `true` | Suppress multiprocessing resource_tracker warnings |
| `streaming_io` | `bool` | `false` | Enable streaming I/O mode |
| `rows_per_block` | `int` | `4000` | Ray Data block size |

---

## Sampling Parameters

Default sampling params are set at the root level, with stage-specific overrides:

```yaml
# Global defaults
sampling_params:
  seed: 777
  temperature: 0.0
  top_p: 1.0
  top_k: -1
  max_tokens: 16384

# Stage overrides (applied via pipeline node overrides)
sampling_params_classify:
  max_tokens: 4
  detokenize: false
  guided_decoding:
    choice: ["YES", "NO"]

sampling_params_vqa:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 512
  stop: []
```

### Supported sampling fields

| Field | Description |
|-------|-------------|
| `temperature` | Sampling temperature (0.0 = greedy) |
| `top_p` | Nucleus sampling threshold |
| `top_k` | Top-k sampling (-1 = disabled) |
| `max_tokens` | Maximum generation length |
| `seed` | Random seed for reproducibility |
| `stop` | Stop token sequences |
| `guided_decoding` | Structured output constraints (see [[vllm-inference#Guided Decoding]]) |
| `n` | Number of samples per prompt |
| `detokenize` | Whether to detokenize output |

---

## Environment and Interpolation

### server.env

Site-specific settings loaded by `ensure_dotenv()` (see [[shared-infrastructure#stage_utils.py]]):

```env
SLURM_PARTITION=pierson
MLLMSCI_PROJECT_ROOT=/share/pierson/matt/mllmsci
MLLMSCI_VENV_ACTIVATE=/share/pierson/matt/mllmsci/.venv/bin/activate
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_CUMEM_HOST_ENABLE=0
```

### OmegaConf interpolation

Configs use `${oc.env:VAR,default}` for environment variable interpolation:

```yaml
partition: ${oc.env:SLURM_PARTITION,pierson}
project: ${oc.env:WANDB_PROJECT,URBANVQA}
```

This allows the same config to work across different machines by changing environment variables rather than editing YAML files.

---

## CLI Usage

### Basic run

```bash
python -m dagspaces.urbanvqa.cli pipeline=vqa_cyclomedia_scaffolding runtime.debug=true runtime.sample_n=100
```

### SLURM submission

```bash
python -m dagspaces.urbanvqa.cli hydra/launcher=slurm_gpu_4x pipeline=vqa_cyclomedia_scaffolding
```

### Multi-run sweep (Hydra multirun)

```bash
python -m dagspaces.urbanvqa.cli -m \
  data=bayflood_1k,bayflood_nearby \
  model=vllm_multimodal_qwen3_vl_2b,vllm_multimodal_qwen3_vl_30b
```

This runs the Cartesian product: 2 datasets x 2 models = 4 runs.

### Override examples

```bash
# Change model and batch size
python -m dagspaces.urbanvqa.cli model=vllm_multimodal_qwen3_vl_2b model.batch_size=16

# Override sampling params
python -m dagspaces.urbanvqa.cli sampling_params.temperature=0.7 sampling_params.max_tokens=1024

# Custom output directory
python -m dagspaces.urbanvqa.cli runtime.output_root=/scratch/user/experiment_001

# Force multimodal mode
python -m dagspaces.urbanvqa.cli runtime.multimodal_enabled=true
```

### Entry points

Each dagspace has its own CLI:

| Command | Dagspace |
|---------|----------|
| `python -m dagspaces.urbanvqa.cli` | Visual Question Answering |
| `python -m dagspaces.urbanocr.cli` | OCR / text spotting |
| `python -m dagspaces.urbanpairvqa.cli` | Pairwise comparison VQA |
| `python -m dagspaces.urbanroamvqa.cli` | Multi-step street traversal VQA |
| `python -m dagspaces.urbanembed.cli` | Embedding inference |

All entry points clean inherited SLURM env vars before Hydra init (prevents corruption when launching from interactive SLURM sessions). See [[slurm-deployment]] for details.

---

## See Also

- [[architecture]] -- Overall system architecture
- [[shared-infrastructure]] -- Shared module documentation
- [[guide-bootstrapping]] -- Getting started guide
