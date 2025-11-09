
## Phase 4: Configuration Refactoring

### 4.1 Update Base Configuration

**File**: `dagspaces/urbanvqa/conf/config.yaml`

**Remove**:
- `prefilter_mode` (relevance filtering)
- `keyword_buffering` (keyword buffering)
- `keyword_window_words` (keyword buffering)
- `gate_on_relevance` (relevance filtering)
- `topic.*` (topic modeling config)
- `sampling_params_classify` (classify-specific)
- `sampling_params_decompose` (decompose-specific)
- `sampling_params_taxonomy` (taxonomy-specific)
- `taxonomy_json` (taxonomy config)
- `tfidf_stopwords_path` (topic modeling)
- `verify.*` (verification config)

**Add**:
```yaml
runtime:
  # VQA-specific options
  batch_inference: true  # Process multiple prompt+image pairs efficiently
  dynamic_prompts: false  # Future: allow per-sample prompts
  hierarchical_prompts: false  # Future: multi-step reasoning

# VQA prompt configuration
prompt:
  system: "You are a helpful assistant that answers questions about images."
  # Future: support prompt templates with variables

# VQA sampling parameters
sampling_params_vqa:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 512  # Longer for detailed answers
  stop: []  # No specific stop tokens for VQA
```

### 4.2 Create VQA Pipeline Configuration

**New File**: `dagspaces/urbanvqa/conf/pipeline/vqa.yaml`

```yaml
# VQA Pipeline Configuration
# Processes prompt + image(s) -> answer(s)

defaults:
  - _self_  # Best Practice: Include _self_ for explicit composition order control
  - override /data: vqa_inputs
  - override /model: vllm_multimodal
  - override /prompt: vqa

runtime:
  multimodal_enabled: true
  image_fallback: false  # VQA requires images; fail if missing
  batch_inference: true
  streaming_io: true  # Recommended for large datasets
  
model:
  model_source: "Qwen/Qwen2.5-VL-3B-Instruct"
  batch_size: 16
  concurrency: 1
  has_image: true
  engine_kwargs:
    # Best Practice: Always set multimodal limits
    limit_mm_per_prompt: {"image": 1}  # Single image per prompt
    trust_remote_code: true  # Required for custom vision models like Qwen2-VL
    max_model_len: 4096  # Set based on model's context window
    max_num_batched_tokens: 8192  # Tune for throughput (optional, default: adaptive)
    tensor_parallel_size: 1  # Set to 2+ if model doesn't fit on one GPU
    gpu_memory_utilization: 0.9  # GPU memory usage (0.0-1.0)

data:
  columns:
    prompt: prompt
    sample_id: sample_id
    image_path: image_path
    image_url: image_url
    image_base64: image_base64
```

### 4.3 Update Prompt Configuration

**New File**: `dagspaces/urbanvqa/conf/prompt/vqa.yaml`

```yaml
# VQA Prompt Configuration

system: "You are a helpful assistant that answers questions about images accurately and concisely."

# User prompt template (prompt from data column is inserted here)
user_template: "{{prompt}}"

# Future: Support for hierarchical prompts
# hierarchical:
#   enabled: false
#   steps:
#     - type: "observation"
#       prompt: "First, describe what you see in the image."
#     - type: "reasoning"
#       prompt: "Based on your observation, {{reasoning_prompt}}"
#     - type: "answer"
#       prompt: "Finally, answer: {{prompt}}"
```
