---
title: "UrbanVQA — Visual Question Answering"
category: dagspace
created: 2026-04-06
tags:
  - dagspace
  - vqa
  - inference
  - multimodal
  - prompt-engineering
---

# UrbanVQA — Visual Question Answering

UrbanVQA is the core dagspace for **multimodal Visual Question Answering** inference. It takes an input parquet dataset of images (with optional prompts) and runs them through a vision-language model via vLLM, producing structured JSON answers. It is the most feature-rich dagspace, with support for prompt optimization, answer verification, guided decoding, topic modeling, and multi-stage event-detection pipelines.

## Purpose

- General-purpose VQA inference: prompt + images -> structured answers
- Supports single-image classification, open-ended QA, domain-specific event detection, taxonomy extraction, topic modeling, and synthesis
- Entry point for all text-generating multimodal inference in the MLLMSCI framework

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanvqa/cli.py` | Hydra CLI entry point; cleans SLURM env vars before init |
| `dagspaces/urbanvqa/orchestrator.py` | DAG execution engine; defines `VQARunner(StageRunner)` and `get_stage_registry()` |
| `dagspaces/urbanvqa/stages/vqa.py` | Core inference stage: `run_vqa_stage()`, preprocess/postprocess callbacks |
| `dagspaces/urbanvqa/verification_core.py` | Post-inference answer verification (embedding, NLI, combo methods) |
| `dagspaces/urbanvqa/schema_builders.py` | JSON schema builder helpers for guided decoding |
| `dagspaces/urbanvqa/prompts/unified.py` | Unified dynamic prompting framework (`unified_preprocess`, `preprocess_simple`) |
| `dagspaces/urbanvqa/prompts/decision_tree.py` | Decision-tree prompting strategy |
| `dagspaces/urbanvqa/prompts/techniques.py` | Prompting technique implementations |
| `dagspaces/urbanvqa/prompt_opt/` | GEPA prompt optimization suite |

## Data Flow

```
Input Parquet
  -> _load_vqa_input (pandas DataFrame, column validation)
  -> run_vqa_stage(df, cfg)
     -> _make_preprocess(cfg)  -- builds per-row preprocessing closure
        -> unified_preprocess() or preprocess_simple()
           -> Jinja2 template rendering
           -> PIL image resolution (_resolve_pil_image)
           -> Guided decoding config injection
           -> Chat message formatting (system/user/image blocks)
     -> dagspaces.common.vllm_inference.run_vllm_inference()
        -> vLLM engine batched inference on GPU
     -> _make_postprocess(cfg)  -- JSON extraction, label parsing
  -> Output Parquet (saved via VQARunner)
  -> W&B table logging (sample_id, prompt, answer, model_response)
```

## Prompt System

UrbanVQA supports multiple prompting strategies, resolved in priority order by `_make_preprocess`:

| Strategy | Config Flag | Description |
|----------|------------|-------------|
| **Adaptive** | `prompt.adaptive.enabled` | Adjusts prompt based on input characteristics |
| **Retrieval-Augmented** | `prompt.retrieval_augmented.enabled` | Injects retrieved context into prompt |
| **Chain of Thought** | `prompt.chain_of_thought.enabled` | Step-by-step reasoning before answer |
| **ReAct** | `prompt.react.enabled` | Reason + Act iterative prompting |
| **Contextual** | `prompt.contextual.enabled` | Context-aware prompt augmentation |
| **Hierarchical** | `prompt.hierarchical.enabled` | Multi-level decomposed prompting |
| **Decision Tree** | `prompt.decision_tree.enabled` | Branching conditional prompt logic |
| **Simple (default)** | always | `preprocess_simple()` with Jinja2 user template |

All strategies flow through `dagspaces/urbanvqa/prompts/unified.py`, which dispatches to the appropriate technique module. Prompt templates are Jinja2 strings rendered via `render_prompt_template()` with row-level variable injection.

## Guided Decoding

Guided decoding constrains model output to valid JSON schemas or enumerated choices. Configured via `sampling_params_vqa.structured_output` or `structured_output_schema` in the Hydra config.

Key functions in `stages/vqa.py`:

- `_ensure_json_schema_dict(schema)` -- normalizes OmegaConf/dict schemas to plain Python dicts
- `_extract_enum_choices(schema)` -- recursively finds enum values in a JSON schema
- `_build_guided_decoding_config(schema)` -- returns `{"choice": [...]}` for enums or `{"json": schema}` for full schemas

Schema builders in `schema_builders.py` provide helpers:

- `string_enum(values)` -- `{"type": "string", "enum": [...]}`
- `nullable_string_enum(values)` -- anyOf with null
- `object_schema(properties, required)` -- full object schema
- `array_of_strings()`, `string_or_null()`

See [[concept-guided-decoding]] for details.

## Verification Layer

`verification_core.py` provides post-inference answer filtering to reject low-confidence or contradictory answers.

**`VerificationConfig` dataclass:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | `"combo"` | One of: `off`, `embed`, `nli`, `combo`, `combo_judge` |
| `top_k` | `3` | Number of candidate answers to consider |
| `sim_threshold` | `0.55` | Embedding cosine similarity threshold |
| `entail_threshold` | `0.85` | NLI entailment probability threshold |
| `contra_max` | `0.05` | Maximum contradiction probability allowed |
| `embed_model_name` | `intfloat/multilingual-e5-base` | Embedding model for similarity |
| `nli_model_name` | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | NLI model |

Models are lazily initialized and cached globally. See [[concept-verification]] for methodology.

## Prompt Optimization (GEPA)

The `dagspaces/urbanvqa/prompt_opt/` directory contains the **Genetic Evolutionary Prompt Adaptation** suite for automated prompt optimization:

| File | Role |
|------|------|
| `prompt_opt/runner.py` | GEPA optimization loop runner |
| `prompt_opt/gepa_adapter.py` | Adapter connecting GEPA to VQA pipeline |
| `prompt_opt/dataset.py` | Evaluation dataset handling |
| `prompt_opt/lm_resolver.py` | Language model resolution for optimization |
| `prompt_opt/multimodal_reflection.py` | Multimodal reflection-based optimization |
| `prompt_opt/visualize.py` | Visualization of optimization results |
| `dagspaces/urbanvqa/gepa_cli.py` | Dedicated GEPA CLI entry point |

## Configuration

### Data Configs (`dagspaces/urbanvqa/conf/data/`)

| Config | Dataset |
|--------|---------|
| `bayflood.yaml`, `bayflood_1k.yaml` | Bay Area flood imagery |
| `bayflood_nearby_floodnet.yaml` | FloodNet nearby flood images |
| `bayflood_relative.yaml`, `bayflood_sep29all_*.yaml` | Relative flood assessment variants |
| `cyclomedia_manhattan.yaml`, `cyclomedia_manhattan_2025_1.yaml` | Cyclomedia street-view imagery |
| `nexar_dashcam.yaml` | Nexar dashcam dataset |
| `vqa_inputs.yaml`, `inputs.yaml`, `multimodal_inputs.yaml` | Generic input configs |

### Prompt Configs (`dagspaces/urbanvqa/conf/prompt/`)

| Config | Domain |
|--------|--------|
| `vqa.yaml` | General VQA |
| `classify.yaml`, `classify_risks_and_benefits.yaml` | Classification prompts |
| `scaffolding_detection.yaml` | Construction scaffolding detection |
| `dominant_language.yaml`, `dominant_religion.yaml` | Demographic/cultural classification |
| `eu_ai_act_classification.yaml` | EU AI Act risk classification |
| `chain_of_thought.yaml`, `decision_tree.yaml`, `react.yaml` | Technique-specific prompts |
| `adaptive.yaml`, `contextual.yaml`, `hierarchical.yaml` | Advanced strategy prompts |
| `bayflood_*.yaml` | Flood-domain optimized prompts |
| `taxonomy.yaml`, `synthesis.yaml` | Topic taxonomy and synthesis |

### Pipeline Configs (`dagspaces/urbanvqa/conf/pipeline/`)

| Config | Pipeline Type |
|--------|--------------|
| `vqa.yaml` | Single-node VQA inference |
| `vqa_cyclomedia_scaffolding.yaml` | Scaffolding detection on Cyclomedia |
| `vqa_bayflood*.yaml` | Various flood assessment pipelines |
| `vqa_nexar.yaml` | Dashcam VQA |
| `classify_decompose.yaml` | Classify then decompose |
| `full_event_pipeline.yaml`, `full_event_pipeline_us.yaml` | Multi-stage event detection DAG |
| `topic_modeling_of_relevant_classifications.yaml` | Topic modeling |
| `topic_with_synthesis.yaml`, `topic_synthesis_from_classify.yaml` | Topic + synthesis |
| `event_synthesis_from_classify.yaml` | Event synthesis |
| `taxonomy_full.yaml` | Full taxonomy extraction |
| `verify_nbl_from_decompose.yaml` | Verification pipeline |

## Unique Features

- **Answer verification** with embedding similarity and NLI entailment filtering
- **Prompt optimization** via GEPA (genetic evolutionary search over prompt variants)
- **Topic modeling** pipelines with clustering and synthesis stages
- **Multi-prompt variants** (adaptive, hierarchical, decision tree, chain of thought, ReAct)
- **Multi-stage DAGs** for event detection, decomposition, classification, and synthesis
- **Guided decoding** with JSON schema enforcement and enum choice constraints

## Related Pages

- [[architecture]] -- overall pipeline architecture
- [[vllm-inference]] -- shared vLLM inference engine
- [[concept-guided-decoding]] -- structured output enforcement
- [[concept-verification]] -- answer verification methodology
- [[shared-infrastructure]] -- common modules used by all dagspaces
- [[urban-ocr]] -- OCR dagspace
- [[urban-pair-vqa]] -- pairwise comparison dagspace
- [[urban-roam-vqa]] -- street traversal dagspace
- [[urban-embed]] -- embedding dagspace
