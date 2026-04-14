# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Wiki Maintenance

This project has an Obsidian wiki at `vault/context/`. You are responsible for keeping it accurate and up to date. The full schema is in `vault/context/WIKI_SCHEMA.md` — read it before making wiki changes. Key rules:

**Structure:**
- `vault/context/index.md` — Content catalog. Read this first to find relevant pages. Update it whenever you create, rename, or delete a page.
- `vault/context/log.md` — Append-only activity log. Add an entry (`## [YYYY-MM-DD] action | Subject`) for every wiki change.
- `vault/context/wiki/` — All wiki pages live here. You own this directory entirely.
- `vault/context/sources/`, `vault/context/raw/` — Immutable source material. Read from these but never modify them.

**When to update the wiki:**
- After any structural change to the codebase (new dagspace, new stage, new config group, renamed module)
- After adding, removing, or significantly modifying a pipeline stage or shared utility
- After resolving a bug or performance issue that's documented in the troubleshooting page
- When the user asks you to document something or you discover the wiki is stale

**How to update:**
1. Read `vault/context/index.md` to find the relevant page(s)
2. Read the page(s) and the current source code
3. Edit the wiki page to reflect the current state — update the `updated:` frontmatter date
4. Add/fix `[[wikilinks]]` cross-references
5. Update `index.md` if the page scope changed or a new page was created
6. Append to `log.md`

**Page conventions:** Frontmatter with title/category/created/updated/tags. Filenames in `kebab-case.md`. Concept pages prefixed `concept-`, guide pages prefixed `guide-`. Use `[[wikilinks]]` (no `.md` extension). Prefer tables and bullets over prose.

**Do not** update the wiki for trivial changes (typo fixes, comment edits, test-only changes). The wiki documents architecture, not every commit.

## Project Overview

MLLMSCI / UAIR (Urban AI Risks Assessment Framework) — a scalable pipeline framework for large-scale multimodal LLM inference over urban datasets. Uses vLLM for GPU inference, Hydra for configuration, and W&B for experiment tracking. Runs on SLURM clusters.

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

**IMPORTANT:** Always use `-m` (multirun/submitit) when giving CLI commands to run a dagspace pipeline. Pipelines are designed to be orchestrated via SLURM — the monitor node submits stage jobs. Example: `python -m dagspaces.urbanembed.cli -m pipeline=browser_index_cyclomedia ...`

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
- `multiprocessing_utils.py` — SLURM detection, warning suppression
- `stage_utils.py` — `ensure_dotenv()`, `extract_last_json()`, `sanitize_for_json()`, `resolve_thinking_mode()`, `maybe_silence_vllm_logs()`
- `conf/data/` — Shared data configs (symlinked from root as `datasets/`)
- `conf/model/` — Shared model configs (symlinked from root as `models/`)
- `conf/hydra/launcher/` — Shared SLURM launcher configs (symlinked from root as `launchers/`)

Shared configs are resolved by all dagspaces via Hydra searchpath (`pkg://dagspaces.common.conf`). Dagspace-local `conf/data/`, `conf/model/`, or `conf/hydra/launcher/` overrides take precedence when present.

### Pipeline Execution Model

Pipelines are DAGs defined in YAML (`conf/pipeline/*.yaml`). Each node specifies a stage type, dependencies, inputs, and outputs. The orchestrator resolves the DAG topologically and executes stages sequentially, passing parquet data between them.

Stage types include: `vqa`, `ocr`, `pairwise_vqa`, `roaming_vqa`, `embed`, `classify`, `taxonomy`, `decompose`, `topic`, `verify`.

### VQA Stage Data Flow (primary inference path)

```
Input Parquet → pandas DataFrame
  → preprocess (Jinja2 prompt rendering, image loading)
  → run_vllm_inference (direct vLLM LLM.generate() with multimodal support)
  → postprocess (JSON extraction)
  → Output Parquet
```

### Configuration System

Hydra-based hierarchical config composition (version 1.3). Key config groups:
- `data/` — Input data paths and dataset definitions (shared via `dagspaces/common/conf/data/`, dagspace-local overrides where needed)
- `model/` — vLLM model settings (shared via `dagspaces/common/conf/model/`)
- `prompt/` — Jinja2 prompt templates (dagspace-local)
- `pipeline/` — DAG node definitions (dagspace-local)
- `hydra/launcher/` — SLURM launcher configs (shared via `dagspaces/common/conf/hydra/launcher/`, dagspace-local overrides where needed)

Shared configs resolved via `hydra.searchpath: [pkg://dagspaces.common.conf]` in each dagspace's `config.yaml`. Site-specific settings (SLURM partition, project paths, NCCL) are in `server.env`, loaded automatically by `ensure_dotenv()`.

Supports `${oc.env:VAR}` interpolation and optional defaults.

### Key Infrastructure

- **vLLM** — GPU inference with batching, guided decoding, multimodal support
- **SLURM** — Cluster job submission via hydra-submitit-launcher
- **W&B** — Experiment tracking (in-process mode for SLURM compatibility)

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
- `tests/test_vqa.py` — VQA unit tests (template rendering, image prep, JSON extraction, data validation)
- `tests/test_roaming_vqa.py` — Roaming VQA tests (graph construction, bearings, checkpoints)
- `vault/context/` — Project knowledge wiki (Obsidian vault with index, schema, and wiki pages)
