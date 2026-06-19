---
title: Project Overview
category: overview
created: 2026-04-06
updated: 2026-04-08
tags: [mllmsci, uair, overview]
sources: []
---

# Project Overview

**MLLMSCI / UAIR** (Urban AI Risks Assessment Framework) is a scalable pipeline framework for large-scale multimodal LLM inference over urban datasets. It orchestrates distributed GPU inference across SLURM clusters, processing street-level imagery through configurable DAG-based pipelines.

## Core Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Language | Python 3.12 (managed with `uv`) | Runtime |
| Inference | vLLM | Batched GPU inference with multimodal support |
| Distribution | Ray Data | Streaming distributed data processing |
| Configuration | Hydra 1.3 | Hierarchical config composition |
| Experiment tracking | Weights & Biases | Logging, tagging, artifact management |
| Cluster | SLURM | Job scheduling via hydra-submitit-launcher |

## Dagspaces

The framework is organized into five independent pipeline systems ("dagspaces"), each under `dagspaces/`. All share a common infrastructure layer but define their own stage registries and CLI entry points.

| Dagspace | CLI Entry | Purpose |
|----------|-----------|---------|
| [[urban-vqa\|urbanvqa]] | `python -m dagspaces.urbanvqa.cli` | Multimodal VQA inference (prompt + images -> answers) |
| [[urban-ocr\|urbanocr]] | `python -m dagspaces.urbanocr.cli` | Text spotting with bounding boxes, automatic tiling for large images |
| [[urban-pairvqa\|urbanpairvqa]] | `python -m dagspaces.urbanpairvqa.cli` | Pairwise relative comparison of image pairs |
| [[urban-roamvqa\|urbanroamvqa]] | `python -m dagspaces.urbanroamvqa.cli` | Multi-step agent-driven street traversal VQA |
| [[urban-embed\|urbanembed]] | `python -m dagspaces.urbanembed.cli` | Batch image embedding inference |

Each dagspace contains:
- `cli.py` -- Entry point; cleans SLURM env vars before Hydra init
- `orchestrator.py` -- DAG execution engine; topologically sorts pipeline nodes, defines `get_stage_registry()`
- `stages/` -- Processing stage implementations
- `conf/` -- Hydra configs (data, model, prompt, pipeline)

## Design Principles

- **DAG-based pipelines** -- Pipelines are directed acyclic graphs defined in YAML. The orchestrator resolves execution order via topological sort. See [[architecture#Pipeline Execution Model]].
- **Shared infrastructure** -- Common modules (`dagspaces/common/`) provide config schema, orchestration utilities, vLLM helpers, W&B integration, and Ray/SLURM setup. See [[shared-infrastructure]].
- **Hydra config composition** -- Hierarchical config system with shared searchpath (`pkg://dagspaces.common.conf`). Dagspace-local overrides take precedence. See [[config-system]].
- **Pluggable stages** -- Each stage type is a `StageRunner` subclass registered in the dagspace's stage registry. New stages can be added without modifying the orchestrator.
- **Parquet-native data flow** -- All inter-stage data passes through partitioned Parquet files, enabling streaming reads and columnar access.

## Data Domains

The framework processes imagery from multiple urban data sources:

- **Cyclomedia** -- Street-view panoramas with metadata (location, timestamp, camera parameters)
- **Nexar** -- Dashcam imagery from rideshare vehicles
- **BayFlood** -- Flood event imagery for damage assessment

## Models

| Model Family | Variants | Primary Use |
|-------------|----------|-------------|
| Qwen3-VL | 2B, 30B | Primary VQA and OCR inference |
| Cambrian | -- | Multimodal reasoning |
| Phi3 | -- | Lightweight VQA |
| SmolVLM | -- | Efficient small-scale inference |
| InternVL | -- | Visual understanding |

Model configurations live in `dagspaces/common/conf/model/` (symlinked from project root as `models/`).

## Quick Start

```bash
# Install dependencies
uv pip install -e .

# Run a VQA pipeline in debug mode
python -m dagspaces.urbanvqa.cli pipeline=vqa_cyclomedia_scaffolding runtime.debug=true runtime.sample_n=100

# Run on SLURM with 4 GPUs
python -m dagspaces.urbanvqa.cli hydra/launcher=slurm_gpu_4x pipeline=vqa_cyclomedia_scaffolding
```

All CLI entry points accept Hydra overrides: `model.batch_size=32 runtime.debug=true data.parquet_path=/path/to/data.parquet`

## Key File Locations

| Path | Description |
|------|-------------|
| `dagspaces/` | All dagspace implementations |
| `dagspaces/common/` | Shared infrastructure modules |
| `dagspaces/common/conf/` | Shared Hydra configs (models, launchers) |
| `pipeline_manifest.json` | Tracks execution results and metadata |
| `tests/test_vqa.py` | Unit tests |
| `notebooks/` | Analysis notebooks |
| `scripts/` | Dataset creation and utility scripts |
| `server.env` | Site-specific settings (SLURM partition, paths, NCCL) |

## Downstream Projects

| Project | Location | Description |
|---------|----------|-------------|
| **Shedfolio** | `/share/ju/matt/shedfolio/` | NYC scaffolding/sidewalk shed research project. Uses MLLMSCI dagspaces (urbanvqa, urbanocr, urbanpairvqa, urbanembed, urbanroamvqa, artifact_gen) for citywide scaffold detection, type classification, temporal analysis, and DOB permit validation. Has its own Obsidian wiki at `vault/`. |

## See Also

- [[architecture]] -- System architecture and execution model
- [[guide-bootstrapping]] -- Getting started guide
- [[config-system]] -- Configuration system details
- [[shared-infrastructure]] -- Common modules reference
