# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language: ASD-STE100

Write all text in ASD-STE100 Simplified Technical English (STE).

This rule applies to:

- Replies to the user
- Code comments and docstrings
- Comments in config files
- Documentation in `vlm-narratives-docs/` and commit messages

### Writing rules

| Rule | Requirement |
|------|-------------|
| Words | Use only approved words. Technical names and technical verbs are permitted. |
| One meaning | Give each word one meaning and one part of speech. |
| Sentence length | Instructions: 20 words maximum. Descriptions: 25 words maximum. |
| Paragraphs | Write one topic in one paragraph. Use 6 sentences maximum. |
| Voice | Use the active voice. |
| Tense | Use the simple tenses. Do not use the perfect or the progressive tenses. |
| Verbs that end in -ing | Do not use them, unless they are a technical name. Write a clause. |
| Instructions | Write one instruction in one sentence. |
| Articles | Write `the` or `a` where you can. |
| Noun clusters | Use 3 words maximum. |
| Lists | Put complex data in a vertical list. |
| Warnings | Write the warning before the step, not after it. |

### Words to replace

| Do not write | Write |
|--------------|-------|
| utilize | use |
| perform | do |
| prior to | before |
| in order to | to |
| due to | because of |
| ensure | make sure |
| via | with, by |
| obtain | get |
| attempt | try |
| terminate | stop |
| initiate | start |
| additional | more |
| approximately | about |
| however | but |
| therefore | thus |

### Examples

| Do not write | Write |
|--------------|-------|
| Removing the persona changed the judgments. | The judgments changed when we removed the persona. |
| The cache is being shared by concurrent jobs. | Concurrent jobs share the cache. |
| It's uninterpretable without an anchor. | You cannot read this value without an anchor. |
| This has been fixed. | We fixed this. |

**Note:** The STE dictionary has about 900 approved words. Follow the rules and
the style above. If a word is necessary for accuracy and no approved word is
equivalent, use it as a technical name.

## Documentation

Project documentation lives in `vlm-narratives-docs/`. Write all project
information there.

**Warning:** The Obsidian wiki at `vault/context/` is no longer maintained.
Do not write to it. Its content is stale. Read it only for history.

**When to write documentation:**

- After a structural change to the codebase. This includes a new dagspace, a new
  stage, a new config group, or a module that you rename.
- After you add, remove, or change a pipeline stage or a shared utility.
- After you correct a bug or a performance problem.
- When the user asks you to document something.

**Do not** write documentation for a small change. A typo, a comment, or a change
to a test does not need documentation. Document the architecture, not each commit.

**Conventions:** Use `kebab-case.md` for file names. Prefer tables and lists.
Write the text in STE (see the section above).

## Project Overview

MLLMSCI / UAIR (Urban AI Risks Assessment Framework) — a scalable pipeline framework for large-scale multimodal LLM inference over urban datasets. Uses vLLM for GPU inference, Hydra for configuration, and W&B for experiment tracking. Runs on SLURM clusters.

Python 3.12, managed with `uv`.

## Common Commands

```bash
# Install dependencies
uv pip install -e .

# Run a pipeline (VQA example)
python -m dagspaces.vqa.cli pipeline=vqa_cyclomedia_scaffolding runtime.debug=true runtime.sample_n=100

# Run on SLURM with GPUs
python -m dagspaces.vqa.cli hydra/launcher=slurm_gpu_4x pipeline=vqa_cyclomedia_scaffolding

# Run tests
pytest tests/test_vqa.py -v

# Type checking
pyright
```

Each dagspace has its own CLI entry point:
- `python -m dagspaces.vqa.cli` — Visual Question Answering
- `python -m dagspaces.ocr.cli` — OCR / text spotting
- `python -m dagspaces.pairwisevqa.cli` — Pairwise comparison VQA
- `python -m research.dagspaces.urbanroamvqa.cli` — Multi-step street traversal VQA
- `python -m dagspaces.embed.cli` — Embedding inference
- `python -m dagspaces.speech.cli` — Speech recognition over video clips

All accept Hydra overrides: `model.batch_size=32 runtime.debug=true data.parquet_path=/path/to/data.parquet`

**IMPORTANT:** Always use `-m` (multirun/submitit) when giving CLI commands to run a dagspace pipeline. Pipelines are designed to be orchestrated via SLURM — the monitor node submits stage jobs. Example: `python -m dagspaces.embed.cli -m pipeline=browser_index_cyclomedia ...`

## Architecture

### Dagspaces

The framework has five pipeline systems under `dagspaces/`. Each has the same structure:

| Dagspace | Purpose |
|----------|---------|
| `vqa` | Multimodal VQA inference (prompt + images → answers) |
| `ocr` | Text spotting with bounding boxes, automatic tiling for large images |
| `pairwisevqa` | Pairwise relative comparison of image pairs |
| `embed` | Embedding inference |
| `speech` | Speech recognition over video clips (ffmpeg audio isolation + granite-speech via vLLM, JU partition) |

The experimental dagspaces are in `research/dagspaces/`: `urbanroamvqa`,
`urbanvit`, `opf_vision_labels` and `artifact_gen`. They import the framework,
but the framework does not import them.

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
- `image_pack.py` — Read images from a pack of blob files (`runtime.image_pack`) instead of the raw NFS tree. Build with `scripts/build_image_pack.py`. See `vlm-narratives-docs/nfs-load-pairvqa-sweeps.md`.
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

Shared configs resolved via `hydra.searchpath: [pkg://dagspaces.common.conf]` in each dagspace's `config.yaml`.

The dataset configs (Cyclomedia, NYC, bayflood, nexar) are in `sv/conf/` and
`sv/nyc/conf/`, not in `dagspaces/`. The Hydra plugin
`hydra_plugins/mllmsci_searchpath/` adds them to the search path of each
dagspace. Thus `pipeline=vqa_bayflood` and `data=cyclomedia_manhattan` work
as before. Run the commands from the repo root, or Hydra does not find the
plugin. See `vlm-narratives-docs/project-config-roots.md`.

Site-specific settings (SLURM partition, project paths, NCCL) are in `server.env`, loaded automatically by `ensure_dotenv()`.

Supports `${oc.env:VAR}` interpolation and optional defaults.

### Key Infrastructure

- **vLLM** — GPU inference with batching, guided decoding, multimodal support
- **SLURM** — Cluster job submission via hydra-submitit-launcher
- **W&B** — Experiment tracking (in-process mode for SLURM compatibility)

### Environments and node-local mirrors

`.venv-vllm025cu129` (vLLM 0.25.0, torch 2.11.0+cu129) is the default for all
launchers. `server.env` sets it. `.venv-3.12` (vLLM 0.19) and `.venv-nightly`
(vLLM 0.23) stay on disk; override `MLLMSCI_VENV_ACTIVATE` to use one.

A stage job starts from a node-local `/scratch` mirror of the venv when the
node holds one. Weight loads go to a `/scratch` model mirror by the same rule.
Each mirror carries a `.sync_complete` marker, and a user of the mirror trusts
it only when the marker names the same source. If it does not, the stage falls
back to NFS.

| Command | Purpose |
|---------|---------|
| `bash scripts/build_venv_vllm025.sh` | Build the default environment again |
| `bash scripts/sync_venv_to_scratch.sh` | Deploy the venv mirror on this node |
| `bash scripts/sync_model_registry_to_scratch.sh` | Deploy the model mirror |

See `vlm-narratives-docs/scratch-mirrors.md` and
`vlm-narratives-docs/vllm-025-upgrade.md`.

### Image Input Formats

The VQA stage accepts images as: PIL objects, local file paths, base64 strings, or HTTP URLs. Format is auto-detected in `_prepare_image_content()`.

## Current Performance State

The Ray Data path is gone. The VQA stages now send chunked batches through
in-process `llm.chat` calls. Results stream to parquet during the run, and
resume chunks align with batch boundaries. See
`vlm-narratives-docs/million-pair-battery.md` for details on shards and
resume chunks.

## File Locations

- `<output_root>/pipeline_manifest.json` — The orchestrator writes it in each run. It holds the stage results and the metadata. Git ignores it.
- `scripts/` — The generic tools of the framework (venv build, scratch mirrors, image packs, shard merge).
- `sv/` — The street-view code. The framework does not import it.
  - `sv/cyclomedia/`, `sv/cyclomedia_catalog/` — The Cyclomedia readers and the catalog builder.
  - `sv/ocr/` — The Cyclomedia OCR handler (`handler: sv.ocr.cyclomedia_handler:CyclomediaHandler`).
  - `sv/nyc/curation/` — The NYC Socrata curation (`python -m sv.nyc.curation`).
  - `sv/conf/`, `sv/nyc/conf/` — The dataset configs, in `common/` (all dagspaces) and `<dagspace>/` folders.
  - `sv/scripts/`, `sv/nyc/scripts/`, `sv/tests/`, `sv/nyc/tests/` — The data scripts and their tests.
- `research/` — The research code. The framework does not import it. Run its code from the repo root, because the editable install maps only `dagspaces`.
  - `research/monocle/` — The interpretability code (the J-lens). Import it as `research.monocle`.
  - `research/notebooks/` — The analysis notebooks (cvpr, facct).
  - `research/scripts/` — The experiment scripts (f4 to f10, batteries, exports).
  - `research/figures/`, `research/tables_export/` — The outputs of the notebooks and the scripts.
  - `research/papers/`, `research/notes/` — The paper drafts and the research notes.
- `docs/` — The framework guides (user guide, config guide, custom stages). They are stale: they describe `dagspaces/uair` and Ray Data. Phase 5 of `vlm-narratives-docs/repo-cleanup-plan.md` writes them again.
- `vlm-narratives-docs/archive/documentation/` — Old development history (phase reports, implementation notes). Read it only for history.
- `tests/test_vqa.py` — VQA unit tests (template rendering, image prep, JSON extraction, data validation)
- `tests/test_roaming_vqa.py` — Roaming VQA tests (graph construction, bearings, checkpoints)
- `vlm-narratives-docs/` — Project documentation. Write project information here.
- `artificial-aesthetic/` — The FAccT paper repo. The LaTeX is the submodule `paper/`. `paperkit` copies the figures, tables and numbers into it, and checks them. See `vlm-narratives-docs/paper-repo-sync.md`.
- `vault/context/` — Obsidian wiki. NOT MAINTAINED. Stale. Read only for history.
