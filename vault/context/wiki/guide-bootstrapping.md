---
title: "Bootstrapping Guide"
category: guide
created: 2026-04-06
updated: 2026-04-06
tags:
  - getting-started
  - setup
  - guide
---

# Bootstrapping Guide

A complete newcomer guide to getting the MLLMSCI / UAIR framework running from scratch.

## Prerequisites

Before you begin, ensure you have access to:

- **Python 3.12** -- the project is pinned to this version
- **uv** package manager -- used for dependency management and virtual environments
- **SLURM cluster with GPUs** -- the inference pipeline runs distributed GPU workloads (e.g., 4x A6000 on `klara.tech.cornell.edu`)
- **Model weights** -- download or locate the vLLM-compatible model weights (e.g., `Qwen2.5-VL-3B-Instruct-AWQ`) on shared storage accessible from compute nodes
- **W&B account** (optional) -- for experiment tracking via Weights & Biases

## Installation

Clone the repository and install in editable mode:

```bash
git clone <repo-url> mllmsci
cd mllmsci
uv pip install -e .
```

This installs all Python dependencies, including Ray, vLLM, Hydra, and the dagspaces package itself.

## Environment Setup

1. Copy the example environment file:

```bash
cp server.env.example server.env
```

2. Edit `server.env` and fill in site-specific values:

| Variable | Description | Example |
|----------|-------------|---------|
| `SLURM_PARTITION` | GPU partition name on your cluster | `gpu` |
| `VENV` | Path to the Python virtual environment | `/share/pierson/matt/mllmsci/.venv` |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL comms | `eth0` |
| `NCCL_DEBUG` | NCCL debug level | `WARN` |
| `WANDB_API_KEY` | W&B API key (optional) | `wand_xxxx...` |

The `ensure_dotenv()` utility in `dagspaces/common/stage_utils.py` loads `server.env` automatically at runtime.

## First Run (Local Debug)

Run a small VQA pipeline locally without SLURM to verify the installation:

```bash
python -m dagspaces.urbanvqa.cli \
  pipeline=vqa_cyclomedia_scaffolding \
  runtime.debug=true \
  runtime.sample_n=100
```

This processes 100 sample images through the VQA pipeline in debug mode. It will:
- Load data from the parquet path defined in the pipeline config
- Initialize a local vLLM engine (requires a GPU on the current machine)
- Write results to the `outputs/` directory

## First SLURM Run

Submit a full pipeline to the cluster:

```bash
python -m dagspaces.urbanvqa.cli \
  hydra/launcher=slurm_gpu_4x \
  pipeline=vqa_cyclomedia_scaffolding
```

The `slurm_gpu_4x` launcher config (in `dagspaces/common/conf/hydra/launcher/`) requests 4 GPUs, appropriate CPU/memory, and sets NCCL environment variables. Hydra uses the submitit launcher to submit the job to SLURM.

## Dagspace CLI Entry Points

Each dagspace has its own CLI module. All accept Hydra overrides on the command line.

| Dagspace | Entry Point | Purpose |
|----------|------------|---------|
| UrbanVQA | `python -m dagspaces.urbanvqa.cli` | Multimodal Visual Question Answering |
| UrbanOCR | `python -m dagspaces.urbanocr.cli` | OCR / text spotting with bounding boxes |
| UrbanPairVQA | `python -m dagspaces.urbanpairvqa.cli` | Pairwise relative comparison of image pairs |
| UrbanRoamVQA | `python -m dagspaces.urbanroamvqa.cli` | Multi-step street traversal VQA |
| UrbanEmbed | `python -m dagspaces.urbanembed.cli` | Embedding inference |

Each CLI module (`cli.py`) cleans SLURM environment variables before Hydra initializes, preventing conflicts between SLURM's env and Hydra's launcher.

## Understanding Output

After a pipeline completes:

- **`outputs/`** -- the default output directory tree. Each pipeline run creates a subdirectory containing output parquet files, one per pipeline stage.
- **`pipeline_manifest.json`** -- tracks execution results and metadata for completed pipeline runs. Check this to confirm a run completed successfully.
- **W&B dashboard** -- if W&B is configured, the run logs metrics, config snapshots, and artifact metadata. The `dagspaces/common/wandb_logger.py` module handles distributed W&B integration with auto-tagging.
- **Output parquets** -- each stage writes its results as parquet files. Downstream stages consume the upstream parquet as input. Columns include the original data plus model answers, extracted JSON, and any verification scores.

## Running Tests

```bash
pytest tests/test_vqa.py -v
```

Tests cover template rendering, image preparation, JSON extraction, and data validation.

## Type Checking

```bash
pyright
```

The project uses Pyright for static type analysis.

## Common Hydra Overrides

All dagspace CLIs accept Hydra overrides. The most frequently used:

| Override | Description | Example |
|----------|-------------|---------|
| `runtime.debug=true` | Enable debug mode (verbose logging, smaller batches) | |
| `runtime.sample_n=100` | Limit input to N rows for testing | |
| `model.batch_size=32` | Set vLLM batch size | |
| `data.parquet_path=/path/to/data.parquet` | Override input data path | |
| `hydra/launcher=slurm_gpu_4x` | Use SLURM launcher with 4 GPUs | |

Overrides can be combined:

```bash
python -m dagspaces.urbanvqa.cli \
  pipeline=vqa_cyclomedia_scaffolding \
  runtime.debug=true \
  runtime.sample_n=500 \
  model.batch_size=32
```

## Next Steps

- [[project-overview]] -- understand the high-level architecture and design goals
- [[config-system]] -- deep dive into Hydra configuration composition
- [[cli-reference]] -- full CLI reference for all dagspaces
- [[guide-custom-stages]] -- learn to extend the framework with new stages
