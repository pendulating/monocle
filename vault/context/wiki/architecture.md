---
title: System Architecture
category: architecture
created: 2026-04-06
updated: 2026-06-09
tags: [architecture, pipeline, dag]
sources: []
---

# System Architecture

## High-Level Flow

```
YAML Config
    |
    v
PipelineGraphSpec          (dagspaces/common/config_schema.py)
    |
    v
Orchestrator               (dagspaces/<dagspace>/orchestrator.py)
    | topological_order()
    v
StageRunner.run(context)   (dagspaces/common/runners/base.py)
    |
    v
StageResult                (dagspaces/common/orchestrator.py)
    |
    v
Output Parquet + Manifest
```

Each dagspace implements this flow independently, sharing the common infrastructure layer for config parsing, stage execution, and result handling.

## Pipeline Execution Model

Pipelines are DAGs defined in YAML under each dagspace's `conf/pipeline/` directory. The execution lifecycle:

1. **Config loading** -- Hydra composes a unified config from dagspace-local and shared (`pkg://dagspaces.common.conf`) config groups
2. **Graph construction** -- `PipelineGraphSpec.from_cfg()` parses the pipeline YAML into a graph of `PipelineNodeSpec` objects
3. **Topological sort** -- `PipelineGraphSpec.topological_order()` resolves execution order respecting node dependencies (Kahn's algorithm)
4. **Stage dispatch** -- For each node in order, the orchestrator looks up the `StageRunner` from `get_stage_registry()` and calls `runner.run(context)`
5. **Result collection** -- Each stage returns a `StageResult` with status, output paths, and metadata. Results are written to `pipeline_manifest.json`

### Node Specification

Each pipeline node (`PipelineNodeSpec`) declares:

| Field | Type | Purpose |
|-------|------|---------|
| `key` | `str` | Unique node identifier |
| `stage` | `str` | Stage type (maps to a registered `StageRunner`) |
| `depends_on` | `List[str]` | Node keys this stage depends on |
| `inputs` | `Dict[str, str]` | Input artifact mappings |
| `outputs` | `Dict[str, str]` | Output artifact declarations |
| `params` | `Dict[str, Any]` | Stage-specific parameters |
| `parallel_group` | `Optional[str]` | Group tag for parallel execution |
| `max_attempts` | `int` | Retry count (default 1) |
| `retry_backoff_s` | `float` | Backoff between retries (default 0.0) |

### Stage Types

Available stage types vary by dagspace:

| Stage Type | Dagspace(s) | Description |
|-----------|-------------|-------------|
| `vqa` | urbanvqa, urbanpairvqa, urbanroamvqa | Visual question answering |
| `ocr` | urbanocr | Text spotting with bounding boxes |
| `embed` | urbanembed | Embedding extraction |
| `extract_audio` | urbanspeech | ffmpeg audio isolation from video clips |
| `asr` | urbanspeech | granite-speech transcription via vLLM |
| `classify` | urbanvqa | Classification from VQA outputs |
| `taxonomy` | urbanvqa | Taxonomy-guided categorization |
| `decompose` | urbanvqa | Question decomposition |
| `topic` | urbanvqa | Topic extraction |
| `verify` | urbanvqa | Answer verification |

## Stage Registration

Each dagspace's `orchestrator.py` defines a `get_stage_registry()` function returning `Dict[str, StageRunner]`:

```python
# dagspaces/urbanvqa/orchestrator.py
_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "vqa": VQAStageRunner(),
    "classify": ClassifyStageRunner(),
    ...
}

def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)
```

### StageRunner Base Class

Defined in `dagspaces/common/runners/base.py`:

- **`stage_name`** class attribute -- must match the stage name in config
- **`run(context: StageExecutionContext) -> StageResult`** -- the execution method each subclass implements
- Context provides: config, input/output paths, runtime parameters, W&B logger

## VQA Data Flow (Primary Inference Path)

The VQA stage is the most common inference path. Data flows through Ray Data operations:

```
Input Parquet
    |  ray.data.read_parquet()
    v
Ray Dataset (streaming)
    |  map_batches(_load_images_batch)
    v
Dataset with PIL images loaded
    |  map(_preprocess) -- Jinja2 prompt rendering
    v
Dataset with formatted prompts
    |  Ray Data LLM API (vLLMEngineProcessorConfig)
    v
Dataset with raw model outputs
    |  map(_postprocess) -- JSON extraction via extract_last_json()
    v
Output Parquet
    write_parquet()
```

Key details:
- Images accepted as PIL objects, local file paths, base64 strings, or HTTP URLs (auto-detected in `_prepare_image_content()`)
- Prompt templates are Jinja2 files under `conf/prompt/`
- JSON extraction handles malformed model outputs gracefully
- Output is partitioned Parquet for downstream consumption

## Configuration Architecture

Hydra 1.3 hierarchical composition with multiple config groups:

```
dagspaces/<dagspace>/conf/
    config.yaml          <- root config with defaults list
    data/                <- input data paths
    model/               <- (overrides shared model configs)
    prompt/              <- Jinja2 prompt templates
    pipeline/            <- DAG node definitions
    hydra/launcher/      <- (overrides shared SLURM launchers)

dagspaces/common/conf/   <- shared configs (via searchpath)
    model/               <- model configs (Qwen3-VL, Phi3, etc.)
    hydra/launcher/      <- SLURM launcher presets
```

Resolution order:
1. CLI overrides (`model.batch_size=32`)
2. Dagspace-local configs (`dagspaces/<dagspace>/conf/`)
3. Shared configs (`dagspaces/common/conf/` via `pkg://dagspaces.common.conf`)

Supports `${oc.env:VAR}` interpolation for environment variables. Site-specific settings (SLURM partition, project paths, NCCL tunables) live in `server.env`, loaded by `ensure_dotenv()`.

See [[config-system]] for full details.

## SLURM Integration

Pipeline stages can run as individual SLURM jobs via hydra-submitit-launcher:

- **Launcher configs** in `dagspaces/common/conf/hydra/launcher/` define GPU counts, partitions, time limits
- **`execute_stage_job(context_data)`** is the entrypoint submitted to SLURM -- each dagspace defines this in its `orchestrator.py`
- The orchestrator submits jobs via a submitit executor, cleans SLURM env vars to avoid conflicts, and polls for completion
- Dagspace-local launcher overrides in `conf/hydra/launcher/` take precedence over shared configs

## Shared Infrastructure Layer

All dagspaces import from `dagspaces.common`. See [[shared-infrastructure]] for full reference.

| Module | Key Exports | Purpose |
|--------|------------|---------|
| `config_schema.py` | `PipelineGraphSpec`, `PipelineNodeSpec`, `ArtifactSpec` | Pipeline DAG data model |
| `orchestrator.py` | `ArtifactRegistry`, `StageExecutionContext`, `StageResult` | Execution context and results |
| `runners/base.py` | `StageRunner` | Base class for all stage implementations |
| `vllm_inference.py` | vLLM engine setup, batch inference | GPU inference utilities |
| `wandb_logger.py` | `WandbLogger` | Distributed W&B with auto-tagging (in-process mode for SLURM/Ray) |
| `stage_utils.py` | `ensure_dotenv()`, `extract_last_json()`, `sanitize_for_json()` | Common stage helpers |
| `multiprocessing_utils.py` | Ray init, SLURM detection, asyncio setup | Runtime environment setup |

## Key Classes

### PipelineGraphSpec
- Defined in `dagspaces/common/config_schema.py`
- Holds `Dict[str, PipelineNodeSpec]` mapping node keys to node specs
- `topological_order() -> List[str]` -- Kahn's algorithm for execution ordering
- `from_cfg(cfg)` -- constructs from Hydra config
- `allow_partial: bool` -- whether to allow partial graph execution

### StageExecutionContext
- Defined in `dagspaces/common/orchestrator.py`
- Carries everything a stage needs: full config, node spec, resolved input/output paths, runtime flags
- Passed to `StageRunner.run()`

### StageResult
- Defined in `dagspaces/common/orchestrator.py`
- Fields: `status` (success/failure), `outputs` (path dict), `metadata` (row counts, timing, etc.)
- Written to `pipeline_manifest.json` after each stage

### WandbLogger
- Defined in `dagspaces/common/wandb_logger.py`
- Runs in in-process mode for SLURM/Ray compatibility
- Auto-tags runs with pipeline, node, model, and data identifiers

## Error Handling

- Each `PipelineNodeSpec` supports `max_attempts` (default 1) and `retry_backoff_s` (default 0.0)
- On stage failure, the orchestrator retries up to `max_attempts` times with configurable backoff
- Failed stages record error details in the pipeline manifest
- `allow_partial` on `PipelineGraphSpec` controls whether downstream nodes execute after an upstream failure

## Known Performance Issues

See [[troubleshooting-performance]] for details on:
- vLLM batch size collapse (first batch=64, subsequent=2)
- Object store memory growth (can reach 124+ GiB)
- Cascading job hangs from spilled data disk reads

## See Also

- [[project-overview]] -- High-level project summary
- [[pipeline-execution-model]] -- Detailed execution model walkthrough
- [[config-system]] -- Configuration system reference
- [[vllm-inference]] -- vLLM integration details
- [[shared-infrastructure]] -- Common modules reference
