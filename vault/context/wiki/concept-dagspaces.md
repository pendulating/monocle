---
title: Dagspaces Architecture
category: concept
created: 2026-06-16
updated: 2026-06-16
tags: [architecture, dagspace, hydra, slurm, vllm, pipeline]
---

# Dagspaces Architecture

A **dagspace** is an independent pipeline system under `dagspaces/`. Each runs
multimodal LLM inference over urban datasets as a DAG of stages, orchestrated on
SLURM with Hydra config and vLLM GPU inference. Part of MLLMSCI / UAIR.

## The six dagspaces

| Dagspace | Purpose |
|----------|---------|
| `urbanvqa` | Multimodal VQA (prompt + images → answers) — primary inference path |
| `urbanocr` | Text spotting with bounding boxes, automatic tiling |
| `urbanpairvqa` | Pairwise relative comparison of image pairs |
| `urbanroamvqa` | Multi-step street traversal VQA |
| `urbanembed` | Embedding inference |
| `urbanspeech` | Speech recognition over video clips — see [[concept-urbanspeech]] |

## Common structure

Each dagspace contains:
- `cli.py` — entry point; cleans SLURM env vars before Hydra init.
- `orchestrator.py` — DAG engine; topologically sorts pipeline nodes, defines
  `get_stage_registry()` with dagspace-specific `StageRunner` subclasses.
- `stages/` — processing stage implementations.
- `conf/` — Hydra configs (data, model, prompt, pipeline).

Always launch with `-m` (submitit/multirun) — pipelines are orchestrated via
SLURM, with a monitor node submitting per-stage jobs.

## Shared infrastructure (`dagspaces/common/`)

`config_schema.py` (pipeline/node/artifact specs), `orchestrator.py` (DAG
utilities, `ArtifactRegistry`, SLURM helpers), `runners/base.py` (`StageRunner`),
`wandb_logger.py`, `vllm_inference.py`, `stage_utils.py`. Shared `conf/data`,
`conf/model`, `conf/hydra/launcher` are resolved by every dagspace via the Hydra
searchpath `pkg://dagspaces.common.conf`; dagspace-local overrides win.

## Execution model

Pipelines are DAGs in `conf/pipeline/*.yaml`. Each node names a stage type,
dependencies, inputs, and outputs. The orchestrator resolves the DAG
topologically and runs stages sequentially, passing parquet data between them.
Two-tier SLURM: a lightweight monitor job runs the orchestrator, which submits
each stage as its own job via that stage's launcher.

## See also
- [[concept-urbanspeech]] — speech dagspace + ASR hallucination control
