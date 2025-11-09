# URBANVQA Implementation Plan: Refactor from News Article Processing to Visual Question Answering

## Overview

This document outlines the complete refactoring plan to transform URBANVQA from a news article processing pipeline into a Visual Question Answering (VQA) system. The core goal is to:

1. **Remove** all news article processing logic (article_text, article_id, etc.)
2. **Replace** with VQA-specific inputs: custom text prompt + image (one-to-one relationship)
3. **Support** batch inference for multiple independent prompt+image pairs efficiently
4. **Plan** for future enhancements: dynamic prompts and hierarchical prompts



#### vLLM Compatibility

1. **Message Format**: All prompts must use OpenAI chat format:
   ```python
   # Best Practice: Use OpenAI chat format with proper content types
   messages = [
       {"role": "system", "content": "..."},  # System message supported
       {
           "role": "user",
           "content": [
               {"type": "text", "text": "..."},  # Text content
               # Image formats (choose one):
               {"type": "image", "image": PIL.Image},  # PIL Image object
               {"type": "image_url", "image_url": {"url": "https://..."}},  # URL
               {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # Base64
               {"type": "image_embeds", "image_embeds": torch.Tensor},  # Pre-computed embeddings
           ]
       }
   ]
   ```
   - **Best Practice**: Content must be an array (list) of content parts, not a string
   - **Best Practice**: Use `{"type": "image_url", "image_url": {"url": "..."}}` for URLs and base64
   - **Best Practice**: Use `{"type": "image", "image": PIL.Image}` for PIL Image objects
   - **Best Practice**: For models like Qwen2-VL, may need additional parameters like `image_grid_thw`
   - **Note**: vLLM automatically processes images; no need for `<image>` tokens in prompt text

2. **Multimodal Support**: 
   - Use `has_image=True` in `vLLMEngineProcessorConfig` - **Best Practice**: Required for multimodal models
   - Set `limit_mm_per_prompt={"image": 1}` in `engine_kwargs` - **Best Practice**: Enforce single image per prompt
   - Images can be PIL Images, URLs (http/https), base64-encoded strings, or pre-computed embeddings
   - Single image per prompt (one-to-one relationship) - enforced via `limit_mm_per_prompt`
   - **Best Practice**: Set `trust_remote_code=True` for custom vision models (e.g., Phi-3.5-Vision, Qwen2-VL)
   - **Best Practice**: Use `mm_processor_kwargs` for model-specific image processing parameters

3. **Batch Inference**:
   - vLLM handles batching automatically via Ray Data's continuous batching
   - vLLM uses continuous batching (PagedAttention) for efficient GPU utilization
   - Each row processes one prompt + one image independently
   - Batch size controlled by Ray Data `batch_size` parameter
   - **Best Practice**: Use `max_num_batched_tokens` in `engine_kwargs` to tune throughput vs latency
   - **Best Practice**: Use `max_num_seqs` to control concurrent decode slots (balance latency/throughput)
   - **Best Practice**: vLLM automatically batches requests; no manual batching needed
   - **Note**: vLLM's continuous batching dynamically batches requests of different lengths efficiently

4. **Engine Configuration**:
   ```python
   engine_kwargs = {
       "limit_mm_per_prompt": {"image": 1},  # Single image per prompt
       "mm_processor_kwargs": {...},  # Model-specific image processing
       "max_num_batched_tokens": 8192,  # Tune for throughput (default: adaptive)
       "max_model_len": 4096,  # Maximum context length
       "tensor_parallel_size": 2,  # For multi-GPU models
       "trust_remote_code": True,  # Required for custom vision models
       "gpu_memory_utilization": 0.9,  # GPU memory usage (0.0-1.0)
   }
   ```
   - **Best Practice**: Set `max_model_len` based on model's context window (e.g., 4096 for Phi-3.5-Vision)
   - **Best Practice**: Tune `max_num_batched_tokens` for optimal throughput (higher = more throughput, higher latency)
   - **Best Practice**: Set `tensor_parallel_size` when model doesn't fit on one GPU (e.g., Qwen-30B on 2 GPUs)
   - **Best Practice**: Use `gpu_memory_utilization` (0.0-1.0) to control GPU memory usage (default: 0.9)
   - **Best Practice**: Set `trust_remote_code=True` for custom vision models (required for Phi-3.5-Vision, Qwen2-VL, etc.)

5. **Structured Output**:
   - Use `guided_decoding={"json": json_schema}` in `sampling_params` for offline inference (Ray Data)
   - Use `response_format` with `json_schema` for OpenAI API format (online serving)
   - Set `guided_decoding_backend` in `engine_kwargs` ("auto", "xgrammar", "outlines", "jsonformer")
   - **Best Practice**: Use "auto" to let vLLM select backend automatically
   - **Best Practice**: Use "xgrammar" for best JSON schema performance
   - **Best Practice**: Format is `guided_decoding={"json": schema}` or `guided_decoding={"choice": [...]}` or `guided_decoding={"grammar": "..."}`
   - **Best Practice**: Provide JSON schema in prompt text for better results (not strictly required but recommended)

6. **Model-Specific Considerations**:
   - **Qwen2-VL**: May require `image_grid_thw` parameter for positional encoding
   - **MiniCPM-V**: May require `image_sizes` parameter for sliced images
   - **Phi-3.5-Vision**: Requires `trust_remote_code=True` and `--runner generate`
   - **Custom Models**: Always set `trust_remote_code=True` for models not in vLLM's default registry

#### Ray Data Compatibility

1. **LLM Processor Architecture**:
   - Use `build_llm_processor` with `vLLMEngineProcessorConfig` for stateless preprocessing
   - **Best Practice**: Preprocess function: `row -> {"messages": [...], "sampling_params": {...}}`
   - **Best Practice**: Postprocess function: `row -> {"sample_id": ..., "answer": ..., ...}`
   - **Best Practice**: Processor is lazy: `processor(dataset)` creates lazy pipeline (no execution until materialization)
   - **Best Practice**: For stateful operations (loading models/configs), use class-based processors with `map_batches` instead
   - **Note**: `build_llm_processor` accepts callable functions (`preprocess`, `postprocess`), not class instances directly

2. **Batch Processing**:
   - `map_batches` for stateful operations (e.g., decision tree traversal) - **Best Practice**: Use class-based predictors
   - `map` for stateless row transformations
   - Use `concurrency` to control parallel workers - **Best Practice**: Set equal to number of model replicas, NOT total GPU count
   - **Important**: When using tensor parallelism (`tensor_parallel_size > 1`), concurrency = `total_gpus / tensor_parallel_size`
     - Example: 2 GPUs, `tensor_parallel_size=2` → `concurrency=1` (one replica using both GPUs)
     - Example: 4 GPUs, `tensor_parallel_size=2` → `concurrency=2` (two replicas, each using 2 GPUs)
     - Example: 8 GPUs, `tensor_parallel_size=1` → `concurrency=8` (eight replicas, one GPU each)
   - Use `batch_size` to control batch size per worker - **Best Practice**: 16-64 for LLMs, adjust based on GPU memory
   - **Best Practice**: For stateful operations, use class-based approach with `__init__` to load models/configs once per worker

3. **Streaming Datasets**:
   - All transformations are lazy until materialization
   - Use `.materialize()` to force execution and cache in object store (for datasets that fit in memory)
   - Use `.take_all()` to trigger execution and get all results as a list
   - Use `.take(n)` to trigger execution and get first n results
   - Use `.show(n)` to trigger execution and display first n results
   - **Best Practice**: Only call `.materialize()` if dataset fits in memory; otherwise use `.take_all()` or `.take(n)` for streaming
   - **Best Practice**: For large datasets, prefer `.take_all()` or iterating over batches rather than materializing

4. **State Management**:
   - Use Ray actors for shared state (e.g., `ContextTracker`) - **Best Practice**: Use named actors for easy retrieval
   - Use metadata columns for per-row state
   - Decision trees/configs should be loaded once per worker via class initialization (not in preprocessing functions)
   - **Best Practice**: For stateful operations, use class-based processors with `__init__` to load models/configs once per worker
   - **Best Practice**: Avoid loading large objects (like decision trees) in preprocessing functions; use class-based approach instead

#### Hydra Compatibility

1. **Configuration Composition**:
   - Use config groups for each prompting technique:
     ```yaml
     defaults:
       - _self_  # Best Practice: Include _self_ for composition order control
       - prompt/decision_tree: decision_tree
       - prompt/hierarchical: hierarchical
     ```
   - Override via command line: `prompt.decision_tree.enabled=true`
   - Use `# @package _global_` for nested configs (note space after #)
   - **Best Practice**: Place `_self_` first in defaults list if you want defaults to override primary config, or last (or omit) if primary config should override defaults

2. **Dynamic Configuration**:
   - Use `DictConfig` for runtime access
   - Use `OmegaConf.select()` for nested access
   - Use `OmegaConf.update()` for dynamic updates
   - **Best Practice**: Use `OmegaConf.resolve()` to resolve interpolations before accessing values

3. **Configuration Validation**:
   - Use Structured Configs for type safety (optional)
   - Validate configs in preprocessing functions
   - Use `OmegaConf.to_container()` for JSON serialization
   - **Best Practice**: Use `OmegaConf.is_missing()` to check for MISSING values before access

4. **Command-Line Overrides**:
   - Use dot notation for nested keys: `prompt.decision_tree.enabled=true`
   - **Best Practice**: Quote paths and special characters: `'prompt.decision_tree.tree_path=conf/prompts/tree.yaml'`
   - **Best Practice**: Quote list values: `'prompt.priority=["a","b"]'`
   - **Best Practice**: Quote interpolations: `'dir=/root/${name}'`
   - Use `++` prefix to add new fields: `++prompt.new_field=value`

5. **Config Paths**:
   - Use relative paths from config directory: `conf/prompts/decision_trees/tree.yaml`
   - Use absolute package paths: `dagspaces/urbanvqa/conf/prompts/decision_trees/tree.yaml`
   - **Best Practice**: Use config groups for organization: `prompt/decision_tree: decision_tree`

**Usage Example**:
```python
# Processor returns lazy dataset
ds = processor(dataset)  # Lazy - no execution yet

# Option 1: Materialize if dataset fits in memory (caches in object store)
ds_materialized = ds.materialize()  # Triggers execution, caches results
results = ds_materialized.take_all()

# Option 2: Stream results without materialization (for large datasets)
results = ds.take_all()  # Triggers execution, returns all results as list

# Option 3: Iterate over batches (for very large datasets)
for batch in ds.iter_batches(batch_size=100):
    process_batch(batch)

# Option 4: Show first few results (for debugging)
ds.show(limit=5)  # Triggers execution, displays first 5 results
```

**Key Ray & vLLM Integration Best Practices Summary**:
1. **Message Format**: Always use OpenAI chat format with `content` as array: `[{"type": "text", "text": "..."}, {"type": "image", "image": PIL.Image}]`
2. **Structured Output**: Use `guided_decoding={"json": schema}` in `sampling_params`, NOT `guided_json`
3. **Processor Architecture**: Use `build_llm_processor` for stateless preprocessing; use class-based `map_batches` for stateful operations
4. **Concurrency**: Set to number of model replicas (`total_gpus / tensor_parallel_size`), NOT total GPU count
5. **Batch Size**: Set `batch_size` in `vLLMEngineProcessorConfig` (16-64 typical for LLMs)
6. **Multimodal**: Always set `has_image=True` and `limit_mm_per_prompt={"image": 1}` in `engine_kwargs`
7. **Error Handling**: Include try-except blocks in preprocessing functions for graceful failures
8. **State Management**: Use Ray actors for shared state, class `__init__` for per-worker state
9. **Lazy Execution**: Processors return lazy datasets; use `.materialize()`, `.take_all()`, or `.show()` to trigger execution
10. **Trust Remote Code**: Set `trust_remote_code=True` for custom vision models (Qwen2-VL, Phi-3.5-Vision, etc.)


## Phase 9: Documentation Updates

### 9.1 User Guide

**New File**: `documentation/VQA_USER_GUIDE.md`

**Sections**:
- Overview of URBANVQA
- Data Format Requirements
- Configuration Guide
- Running VQA Inference
- Batch Inference
- Output Format
- Troubleshooting

### 9.2 API Documentation

**Update**: Function docstrings for VQA-specific functions

---

## Implementation Order

1. **Phase 1**: Data Schema Refactoring (foundation)
2. **Phase 2**: Refactor Classify Stage to VQA Stage (core functionality)
3. **Phase 3**: Orchestrator Refactoring (integration)
4. **Phase 4**: Configuration Refactoring (user-facing)
5. **Phase 5**: Batch Inference Support (optimization)
6. **Phase 7**: Code Refactoring and Cleanup (maintenance)
7. **Phase 8**: Testing and Validation (quality assurance)
8. **Phase 9**: Documentation Updates (usability)

**Phase 6** (Future Enhancements) will be implemented after Phase 8 is complete.

**Note**: Other stages (taxonomy, decompose, topic, synthesis, verify) remain unchanged for now. A separate plan will handle their refactoring to support VQA-style inputs if needed.

---

## Feature Support Summary

### ✅ Supported Features

1. **Structured JSON Output**: Supported via vLLM's guided decoding with JSON schema (Pydantic models or JSON Schema).
   - See section 6.6 for implementation details
   - Uses `guided_decoding_backend` with JSON schema support
   - Pydantic models or JSON Schema definitions supported

2. **Jinja2 Templates**: Supported for dynamic prompt generation with variable substitution, conditionals, and loops.
   - See section 6.1 for implementation details
   - Full Jinja2 syntax support: variables, conditionals, loops, filters
   - Strict mode enabled (raises error on undefined variables)

### ❌ Not Supported Features

1. **Caching**: No support for caching model responses or image processing results
2. **Multi-Image Support**: Only single image per prompt supported (one-to-one relationship)
3. **Answer Validation**: No built-in answer validation or scoring mechanisms

---

## Summary

This refactoring transforms URBANVQA's `classify.py` stage into a focused Visual Question Answering (`vqa.py`) stage while preserving other stages (taxonomy, decompose, etc.) for future refactoring. The key changes are:

1. **Simplified Input**: prompt + image (one-to-one) instead of article_text + metadata
2. **VQA Stage**: Refactored classify.py to vqa.py with VQA-specific functionality
3. **Batch Support**: Efficient processing of multiple independent prompt+image pairs
4. **Structured Output**: JSON schema-guided output using vLLM's guided decoding
5. **Template Support**: Jinja2-style templates for dynamic prompt generation
6. **Future-Ready**: Foundation for dynamic and hierarchical prompts

**Note**: Other stages (taxonomy, decompose, topic, synthesis, verify) remain unchanged. A separate plan will handle their refactoring to support VQA-style inputs if needed.

The implementation maintains the existing infrastructure (Ray Data, vLLM, multimodal support) while removing article-specific complexity from the classify stage.

