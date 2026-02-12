# GEPA Adapter Integration (November 2025)

## Overview
- Added native support for **GEPA prompt optimization** inside the UrbanVQA stack without requiring DSPy.
- Wrapped existing Ray + vLLM VQA stage so GEPA can evolve both `system` and `user` prompts deterministically.
- Provided Hydra configs, dataset samplers, and a CLI entry point to run optimization jobs or validation-only checks on annotated image datasets (e.g., 5k flooded vs. non-flooded street-view images).
- Enabled the main VQA pipeline to consume GEPA-exported prompts through `runtime.prompt_override_path`.

## Environment & Dependencies
- New Python dependencies: `gepa` (>=0.2.0) and `litellm` (>=1.64.0).
- Hydra config tree now includes `dagspaces/urbanvqa/conf/prompt_opt/gepa_vqa.yaml` describing datasets, optimization knobs, and local/remote LM endpoints.
- `llm.task` / `llm.reflection` sections align with LiteLLM/OpenAI-compatible kwargs, supporting either cloud APIs or locally served vLLM models (e.g., Qwen-30B via `llm.reflection.provider: vllm`).

## Core Modules
- `dagspaces/urbanvqa/prompt_opt/dataset.py`
  - Reuses `prepare_stage_input` to materialize supervised DataFrames from the flood dataset.
  - Provides deterministic stratified sampling + cached minibatch generation for GEPA.
- `dagspaces/urbanvqa/prompt_opt/gepa_adapter.py`
  - Implements `GEPAVQAAdapter` that clones Hydra configs, injects candidate prompts, runs `run_vqa_stage`, and returns exact-match metrics plus structured traces.
  - Resolves task/reflection LM clients via `prompt_opt/lm_resolver.py` so optimization can target either local vLLM servers or external APIs.
- `dagspaces/urbanvqa/prompt_opt/runner.py`
  - Orchestrates GEPA runs, persists artifacts (`best_prompts.yaml`, metrics, optional traces), and supports `gepa.mode=validate` for deterministic regression checks against saved prompts.
- `dagspaces/urbanvqa/gepa_cli.py`
  - New entry point: `python -m dagspaces.urbanvqa.gepa_cli` (default config `prompt_opt/gepa_vqa`).

## Running GEPA Optimization
```bash
python -m dagspaces.urbanvqa.gepa_cli \
  gepa.mode=optimize \
  gepa.dataset.train.parquet_path=/path/to/train.parquet \
  gepa.dataset.val.parquet_path=/path/to/val.parquet \
  llm.reflection.api_base=http://localhost:8000/v1 \
  vllm.model_config=/share/pierson/matt/zoo/models/Qwen3-30B-A3B-Instruct-2507
```

Artifacts:
- `outputs/gepa/<hydra_run>/best_prompts.yaml`
- `metrics.json` summarizing best score, call budget, dataset sizes.
- Optional `traces.jsonl` with per-sample prompt/answer context for reflection review.

## Validation / Regression Check
```bash
python -m dagspaces.urbanvqa.gepa_cli \
  gepa.mode=validate \
  gepa.validation.prompt_path=outputs/gepa/.../best_prompts.yaml \
  gepa.validation.expected_score=0.912 \
  gepa.validation.tolerance=0.01
```
- Validates deterministic accuracy on the configured split (`val` by default, optional limit for quick smoke tests).
- Writes `validation_metrics.json` + `validation_traces.jsonl` under the same artifact directory.

## Applying Optimized Prompts to VQA Pipeline
- Any standard pipeline run can merge GEPA results by setting:
  ```yaml
  runtime:
    prompt_override_path: /abs/path/to/best_prompts.yaml
  ```
- `orchestrator.prepare_node_config` reads the YAML and overrides `prompt.system` + `prompt.user_template` before executing the stage.

## Ray/vLLM Integration Notes
- Metadata enrichment for `ray.data.read_images()` now uses a Python `map` UDF (instead of PyArrow column appends) to preserve tensor-backed image storage while adding `image_path`, `sample_id`, and default prompts.
- Metadata joins stay inside Ray Data (no pandas fallback) for better GPU throughput; annotated parquet columns (e.g., flooded/non-flooded labels) remain available downstream for stratified sampling and metric computation.

## Future Work
- Expand `gepa.components` config to optimize multi-stage prompt chains (CoT, ReAct) using existing unified prompting hooks.
- Add pytest regressions that mock minimal datasets and verify prompt override application without GPU dependencies.


