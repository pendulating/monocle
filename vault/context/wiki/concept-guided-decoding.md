---
title: "Guided Decoding (Structured Output)"
category: concept
created: 2026-04-06
updated: 2026-04-06
tags:
  - concept
  - vllm
  - structured-output
  - json-schema
---

# Guided Decoding (Structured Output)

How the framework enforces structured LLM output using vLLM's constrained decoding.

## Overview

Large language models produce free-form text by default. For pipeline stages that require machine-parseable output -- classification labels, ordinal scales, structured JSON -- the framework uses vLLM's guided decoding feature. This constrains the token generation process at inference time so that the model can only produce output matching a specified schema.

## How It Works

vLLM's guided decoding modifies the token sampling step. At each generation step, tokens that would violate the schema constraints are masked out (assigned -infinity logit), so the model can only sample from valid continuations. This guarantees syntactically valid output without post-hoc parsing failures.

Two constraint types are supported:

| Type | Description | Use Case |
|------|-------------|----------|
| `choice` | Constrain output to one of N string values | Classification labels, ordinal scales |
| `json` | Constrain output to match a JSON schema | Structured responses with multiple fields |

## Key Functions

All in `dagspaces/urbanvqa/stages/vqa.py`:

### `_ensure_json_schema_dict(schema)`

Converts OmegaConf `DictConfig` objects into plain Python dicts suitable for vLLM. Handles:

- `DictConfig` -- calls `OmegaConf.to_container(schema, resolve=True)`
- `dict` -- returns a deep copy
- `None` -- returns `None`

This is necessary because vLLM expects native Python dicts, not OmegaConf containers.

### `_extract_enum_choices(schema)`

Recursively walks a JSON schema dict to find `enum` fields. Returns the first list of enum values found, converted to strings. This enables automatic detection of enum-style schemas so they can use the more efficient `choice` constraint mode.

Example: given `{"type": "string", "enum": ["low", "medium", "high"]}`, returns `["low", "medium", "high"]`.

### `_build_guided_decoding_config(schema)`

Builds the guided decoding config payload for vLLM:

1. If the schema contains enum choices (detected by `_extract_enum_choices()`), returns `{"choice": choices}` -- this is faster and more reliable for simple classification
2. Otherwise, returns `{"json": schema}` -- full JSON schema constraint

Returns `None` if no schema is provided (free-form generation).

## Schema Builders

The `dagspaces/urbanvqa/schema_builders.py` module provides helper functions for constructing JSON schemas:

| Function | Output Schema | Example Use |
|----------|---------------|-------------|
| `string_enum(values)` | `{"type": "string", "enum": [...]}` | Classification labels |
| `nullable_string_enum(values)` | `{"anyOf": [string_enum, {"type": "null"}]}` | Labels with "not applicable" option |
| `string_or_null()` | `{"anyOf": [{"type": "string"}, {"type": "null"}]}` | Free-form text with null option |
| `array_of_strings()` | `{"type": "array", "items": {"type": "string"}}` | Multi-label outputs |
| `object_schema(properties, required)` | Full object schema | Multi-field structured responses |

### Example: Object Schema

```python
from dagspaces.urbanvqa.schema_builders import object_schema, string_enum

schema = object_schema(
    properties={
        "category": string_enum(["residential", "commercial", "industrial"]),
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    required=["category", "confidence"],
    additional_properties=False,
)
```

This forces the model to produce output like:

```json
{"category": "residential", "confidence": 0.85}
```

## Configuration

Guided decoding is configured through the Hydra config under `sampling_params`:

```yaml
sampling_params:
  guided_decoding:
    json_schema:
      type: "string"
      enum: ["low", "medium", "high"]
```

Or via `structured_output_schema` at the top level of the config. The VQA stage resolves the schema from either location:

```python
structured_schema = _ensure_json_schema_dict(
    getattr(getattr(cfg, "sampling_params_vqa", None), "structured_output", None)
    or getattr(cfg, "structured_output_schema", None)
)
```

## Performance Considerations

- **Choice mode** (enum) is faster than full JSON schema constraint because the valid token set is smaller and can be pre-computed
- Guided decoding adds minimal latency for simple schemas but can slow generation for complex nested JSON schemas
- The framework automatically selects choice mode when it detects an enum schema, falling back to JSON mode for complex schemas

## See Also

- [[urban-vqa]] -- the UrbanVQA dagspace that uses guided decoding
- [[vllm-inference]] -- vLLM inference engine configuration
