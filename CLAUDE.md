# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLLMSCI / UAIR (Urban AI Risks Assessment Framework) — a scalable pipeline framework for large-scale multimodal LLM inference over urban datasets. Uses Ray Data + vLLM for distributed GPU inference, Hydra for configuration, and W&B for experiment tracking. Runs on SLURM clusters.

Python 3.12, managed with `uv`.

## Common Commands

```bash
# Install dependencies
uv pip install -e .

# Run a pipeline (VQA example)
python -m dagspaces.urbanvqa.cli pipeline=vqa_cyclomedia_scaffolding runtime.debug=true runtime.sample_n=100

# Run on SLURM with GPUs
python -m dagspaces.urbanvqa.cli hydra/launcher=slurm_gpu_4x pipeline=vqa_cyclomedia_scaffolding

# Run tests
pytest tests/test_vqa.py -v

# Type checking
pyright
```

Each dagspace has its own CLI entry point:
- `python -m dagspaces.urbanvqa.cli` — Visual Question Answering
- `python -m dagspaces.urbanocr.cli` — OCR / text spotting
- `python -m dagspaces.urbanpairvqa.cli` — Pairwise comparison VQA
- `python -m dagspaces.urbanroamvqa.cli` — Multi-step street traversal VQA
- `python -m dagspaces.urbanembed.cli` — Embedding inference

All accept Hydra overrides: `model.batch_size=32 runtime.debug=true data.parquet_path=/path/to/data.parquet`

## Architecture

### Dagspaces

Five independent pipeline systems under `dagspaces/`, each following the same structure:

| Dagspace | Purpose |
|----------|---------|
| `urbanvqa` | Multimodal VQA inference (prompt + images → answers) |
| `urbanocr` | Text spotting with bounding boxes, automatic tiling for large images |
| `urbanpairvqa` | Pairwise relative comparison of image pairs |
| `urbanroamvqa` | Multi-step street traversal VQA |
| `urbanembed` | Embedding inference |

Each dagspace contains:
- `cli.py` — Entry point; cleans SLURM env vars before Hydra init
- `orchestrator.py` — DAG execution engine; topologically sorts pipeline nodes, defines `get_stage_registry()` with dagspace-specific `StageRunner` subclasses
- `stages/` — Processing stage implementations (vqa.py, ocr.py, etc.)
- `conf/` — Hydra configs (data, model, prompt, pipeline); dagspace-local launcher overrides only where needed

### Shared Infrastructure (`dagspaces/common/`)

All dagspaces import shared modules directly from `dagspaces.common`:
- `config_schema.py` — `PipelineGraphSpec`, `PipelineNodeSpec`, `ArtifactSpec` dataclasses
- `orchestrator.py` — DAG utilities, `ArtifactRegistry`, `StageExecutionContext`, `StageResult`, SLURM helpers
- `runners/base.py` — `StageRunner` base class
- `wandb_logger.py` — Distributed W&B integration with auto-tagging
- `vllm_inference.py` — vLLM inference utilities
- `multiprocessing_utils.py` — Ray init, SLURM detection, asyncio setup
- `stage_utils.py` — `ensure_dotenv()`, `extract_last_json()`, `sanitize_for_json()`
- `conf/model/` — Shared model configs (symlinked from root as `models/`)
- `conf/hydra/launcher/` — Shared SLURM launcher configs (symlinked from root as `launchers/`)

Shared configs are resolved by all dagspaces via Hydra searchpath (`pkg://dagspaces.common.conf`). Dagspace-local `conf/hydra/launcher/` overrides take precedence when present.

### Pipeline Execution Model

Pipelines are DAGs defined in YAML (`conf/pipeline/*.yaml`). Each node specifies a stage type, dependencies, inputs, and outputs. The orchestrator resolves the DAG topologically and executes stages sequentially, passing parquet data between them.

Stage types include: `vqa`, `ocr`, `pairwise_vqa`, `classify`, `taxonomy`, `decompose`, `topic`, `verify`.

### VQA Stage Data Flow (primary inference path)

```
Input Parquet → Ray Data (streaming)
  → _load_images_batch (map_batches)
  → _preprocess (map, Jinja2 prompt rendering)
  → Ray Data LLM API (vLLMEngineProcessorConfig)
  → _postprocess (map, JSON extraction)
  → Output Parquet
```

### Configuration System

Hydra-based hierarchical config composition (version 1.3). Key config groups:
- `data/` — Input data paths
- `model/` — vLLM model settings (batch_size, engine_kwargs, tensor_parallel_size)
- `prompt/` — Jinja2 prompt templates
- `pipeline/` — DAG node definitions
- `hydra/launcher/` — SLURM launcher configs (shared via `dagspaces/common/conf/`, dagspace-local overrides where needed)

Shared configs resolved via `hydra.searchpath: [pkg://dagspaces.common.conf]` in each dagspace's `config.yaml`. Site-specific settings (SLURM partition, project paths, NCCL) are in `server.env`, loaded automatically by `ensure_dotenv()`.

Supports `${oc.env:VAR}` interpolation and optional defaults.

### Key Infrastructure

- **Ray Data** — Streaming distributed processing; handles partitioned parquet I/O
- **vLLM** — GPU inference with batching, guided decoding, multimodal support
- **SLURM** — Cluster job submission via hydra-submitit-launcher
- **W&B** — Experiment tracking (in-process mode for SLURM/Ray compatibility)

### Image Input Formats

The VQA stage accepts images as: PIL objects, local file paths, base64 strings, or HTTP URLs. Format is auto-detected in `_prepare_image_content()`.

## Known Performance Issues

Documented in `PIPELINE_ISSUES.md`:
1. **vLLM batch size collapse** — First batch=64, subsequent=2. Caused by upstream starvation + object store backpressure.
2. **Object store growth** — Can grow to 124+ GiB causing spilling. Tune `override_object_store_memory_limit_fraction` early.
3. **Job hanging** — Cascading stalls from spilled data disk reads.

## File Locations

- `pipeline_manifest.json` — Tracks execution results and metadata for completed pipeline runs
- `notebooks/` — Analysis notebooks (prompt optimization, OCR verification)
- `scripts/` — Dataset creation and utility scripts
- `documentation/` and `docs/` — Detailed guides (user guide, config guide, custom stages)
- `tests/test_vqa.py` — Unit tests (template rendering, image prep, JSON extraction, data validation)
