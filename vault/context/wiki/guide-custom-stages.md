---
title: "Writing Custom Stages"
category: guide
created: 2026-04-06
updated: 2026-04-06
tags:
  - guide
  - stages
  - extending
---

# Writing Custom Stages

This guide walks through creating a new pipeline stage for any dagspace in the MLLMSCI framework.

## Overview

Every pipeline stage follows the same pattern:

1. A **StageRunner subclass** implements the processing logic
2. The runner is **registered** in the dagspace's orchestrator
3. A **pipeline YAML** node references the stage by name

The orchestrator resolves the DAG topologically and calls each runner's `run()` method in dependency order, passing parquet data between stages.

## Step 1: Create a StageRunner Subclass

Create a new file in `dagspaces/<dagspace>/stages/` (e.g., `dagspaces/urbanvqa/stages/my_stage.py`).

The base class lives at `dagspaces/common/runners/base.py`:

```python
from dagspaces.common.runners.base import StageRunner


class MyStageRunner(StageRunner):
    """Custom stage that does something useful."""

    stage_name = "my_stage"

    def run(self, context):
        """Execute the stage.

        Args:
            context: StageExecutionContext with:
                - context.cfg: Full Hydra DictConfig
                - context.node: PipelineNodeSpec for this stage
                - context.inputs: Dict[str, str] mapping input names to parquet paths
                - context.output_paths: Dict[str, str] mapping output names to paths
                - context.output_dir: Directory for this stage's outputs
                - context.output_root: Root output directory for the pipeline

        Returns:
            StageResult with:
                - outputs: Dict[str, str] mapping output names to written file paths
                - metadata: Dict[str, Any] with timing, row counts, etc.
        """
        import pandas as pd
        from dagspaces.common.orchestrator import StageResult

        # Read input parquet from an upstream stage
        input_path = context.inputs["input_data"]
        df = pd.read_parquet(input_path)

        # --- Your processing logic here ---
        df["new_column"] = df["some_column"].apply(transform_fn)

        # Write output parquet
        output_path = context.output_paths["output_data"]
        df.to_parquet(output_path, index=False)

        return StageResult(
            outputs={"output_data": output_path},
            metadata={"n_rows": len(df)},
        )
```

### Key requirements

- **`stage_name`** class attribute must be a unique string matching what you use in pipeline YAML
- **`run()`** must accept a `StageExecutionContext` and return a `StageResult`
- Input/output names in `context.inputs` and `context.output_paths` are defined by the pipeline YAML node's `inputs` and `outputs` fields

## Step 2: Register in the Orchestrator

Each dagspace has an `orchestrator.py` with a `get_stage_registry()` function that returns a dict mapping stage names to runner instances.

Edit `dagspaces/<dagspace>/orchestrator.py`:

```python
from dagspaces.<dagspace>.stages.my_stage import MyStageRunner

_STAGE_REGISTRY = {
    # ... existing stages ...
    "my_stage": MyStageRunner(),
}

def get_stage_registry():
    return dict(_STAGE_REGISTRY)
```

The orchestrator calls `get_stage_registry()` at pipeline startup, resolves the DAG via `graph_spec.topological_order()`, and executes each node by looking up its `stage` field in this registry.

## Step 3: Create a Pipeline YAML

Add a pipeline config in `dagspaces/<dagspace>/conf/pipeline/` that references your stage:

```yaml
# dagspaces/urbanvqa/conf/pipeline/my_pipeline.yaml
graph:
  nodes:
    - key: "load_data"
      stage: "vqa"
      inputs: {}
      outputs:
        main: "vqa_output.parquet"

    - key: "my_processing"
      stage: "my_stage"
      depends_on:
        - "load_data"
      inputs:
        input_data: "load_data.main"
      outputs:
        output_data: "my_stage_output.parquet"
```

Run it:

```bash
python -m dagspaces.urbanvqa.cli pipeline=my_pipeline runtime.debug=true
```

## Step 4: Add Dagspace-Specific Config (Optional)

If your stage needs its own configuration, add a config group:

```yaml
# dagspaces/urbanvqa/conf/my_stage/default.yaml
my_stage:
  threshold: 0.5
  max_items: 1000
```

Then reference it in your runner via `context.cfg.my_stage.threshold`.

## Using run_vllm_inference() for LLM-Powered Stages

If your custom stage needs to call a vision-language model, use the shared VQA inference path rather than building your own vLLM integration:

```python
from dagspaces.urbanvqa.stages.vqa import run_vqa_stage

def run(self, context):
    df = pd.read_parquet(context.inputs["input_data"])

    # run_vqa_stage handles:
    #   - Ray Data streaming
    #   - Image loading via _load_images_batch
    #   - Jinja2 prompt rendering via _preprocess
    #   - vLLM engine initialization and batched inference
    #   - JSON extraction via _postprocess
    result_df = run_vqa_stage(df, context.cfg)

    output_path = context.output_paths["output_data"]
    result_df.to_parquet(output_path, index=False)
    return StageResult(outputs={"output_data": output_path}, metadata={})
```

The `run_vqa_stage()` function encapsulates the full data flow:

```
Input DataFrame -> Ray Data (streaming)
  -> _load_images_batch (map_batches)
  -> _preprocess (map, Jinja2 prompt rendering)
  -> Ray Data LLM API (vLLMEngineProcessorConfig)
  -> _postprocess (map, JSON extraction)
  -> Output DataFrame
```

## Using the StageExecutionContext

The context object provides everything a stage needs:

| Field | Type | Description |
|-------|------|-------------|
| `cfg` | `DictConfig` | Full Hydra configuration |
| `node` | `PipelineNodeSpec` | This node's spec (key, stage, depends_on, inputs, outputs) |
| `inputs` | `Dict[str, str]` | Resolved input artifact paths from upstream stages |
| `output_paths` | `Dict[str, str]` | Pre-computed output paths based on node's `outputs` spec |
| `output_dir` | `str` | Directory for this stage's outputs |
| `output_root` | `str` | Root output directory for the entire pipeline run |

The `PipelineNodeSpec` and related dataclasses are defined in `dagspaces/common/config_schema.py`.

## Existing Stage Types for Reference

The framework includes these built-in stage types across dagspaces:

| Stage | Dagspace | Description |
|-------|----------|-------------|
| `vqa` | urbanvqa | Multimodal VQA inference |
| `classify` | urbanvqa | Classification with taxonomy |
| `taxonomy` | urbanvqa | Taxonomy decomposition |
| `decompose` | urbanvqa | Question decomposition |
| `topic` | urbanvqa | Topic extraction |
| `verify` | urbanvqa | Answer verification |
| `ocr` | urbanocr | Text spotting with bounding boxes |
| `pairwise_vqa` | urbanpairvqa | Pairwise image comparison |
| `embed` | urbanembed | Embedding inference |

Study these implementations in their respective `dagspaces/<dagspace>/stages/` directories for patterns.

## See Also

- [[architecture]] -- overall system architecture and DAG execution model
- [[shared-infrastructure]] -- common modules available to all stages
- [[config-system]] -- Hydra configuration composition and overrides
