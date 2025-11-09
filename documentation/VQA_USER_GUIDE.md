# URBANVQA User Guide: Visual Question Answering Pipeline

## Overview

URBANVQA is a Visual Question Answering (VQA) pipeline that processes images paired with text prompts/questions to generate answers using vision-language models (VLMs) and multimodal language models (MLLMs). The pipeline is built on Ray Data for distributed batch inference and vLLM for efficient model serving.

### Key Features

- **Prompt + Image → Answer**: Simple one-to-one relationship between prompts and images
- **Batch Inference**: Efficient processing of multiple independent prompt+image pairs
- **Jinja2 Template Support**: Dynamic prompt generation with variable substitution
- **Structured JSON Output**: Optional guided decoding for structured responses
- **Streaming Support**: Process large datasets efficiently with Ray Data streaming
- **GPU Management**: Automatic GPU detection and configuration

---

## Data Format Requirements

### Input Parquet Schema

Your input dataset must be a Parquet file with the following columns:

**Required Columns:**
- `prompt` (string): The text question/prompt to ask about the image
- At least one image source column:
  - `image_path` (string): Path to local image files
  - `image_url` (string): URL to remote images (http/https)
  - `image_base64` (string): Base64-encoded image strings

**Optional Columns:**
- `sample_id` (string): Unique identifier for each sample (auto-generated if missing)

### Example Input Data

```python
import pandas as pd

df = pd.DataFrame({
    "prompt": [
        "What type of building is visible in this image?",
        "Describe the urban planning characteristics.",
        "What is the primary land use?"
    ],
    "image_path": [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
        "/path/to/image3.jpg"
    ],
    "sample_id": ["sample_001", "sample_002", "sample_003"]  # Optional
})

df.to_parquet("vqa_dataset.parquet", index=False)
```

### Image Formats Supported

- **Local Files**: Paths to image files (JPEG, PNG, etc.)
- **Remote URLs**: HTTP/HTTPS URLs to images
- **Base64**: Base64-encoded image strings (with or without data URI prefix)

---

## Configuration Guide

### Basic Configuration

Create or modify your Hydra configuration file:

```yaml
# dagspaces/urbanvqa/conf/pipeline/vqa.yaml

defaults:
  - _self_
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
    limit_mm_per_prompt: {"image": 1}  # Single image per prompt
    trust_remote_code: true
    max_model_len: 4096
    tensor_parallel_size: 1  # Set to 2+ if model doesn't fit on one GPU

data:
  parquet_path: ${oc.env:DATA_ROOT,/path/to/data}/vqa_dataset.parquet
  columns:
    prompt: prompt
    sample_id: sample_id
    image_path: image_path
```

### Prompt Configuration

```yaml
# dagspaces/urbanvqa/conf/prompt/vqa.yaml

system: "You are a helpful assistant that answers questions about images accurately and concisely."

user_template: "{{prompt}}"
```

### Sampling Parameters

```yaml
# In config.yaml

sampling_params_vqa:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 512  # Longer for detailed answers
  stop: []  # No specific stop tokens for VQA
```

### Prompt Grouping Optimization

Enable prompt grouping to improve efficiency when the same prompt is used for multiple images:

```yaml
runtime:
  group_by_prompt: true  # Group same prompts together for efficiency
```

This optimization reorders rows so that rows with the same prompt are processed together, allowing prompt tokenization to be reused.

---

## Running VQA Inference

### Basic Usage

```bash
python -m dagspaces.urbanvqa.cli pipeline=vqa
```

### With Custom Data Path

```bash
python -m dagspaces.urbanvqa.cli \
  pipeline=vqa \
  data.parquet_path=/path/to/your/vqa_dataset.parquet
```

### With Custom Model

```bash
python -m dagspaces.urbanvqa.cli \
  pipeline=vqa \
  model.model_source=Qwen/Qwen2.5-VL-7B-Instruct \
  model.tensor_parallel_size=2  # For models that don't fit on one GPU
```

### With Prompt Grouping

```bash
python -m dagspaces.urbanvqa.cli \
  pipeline=vqa \
  runtime.group_by_prompt=true
```

### With Jinja2 Templates

If you have Jinja2 templates configured:

```yaml
# In prompt config
prompt:
  template: "Focus on {{focus_area}}. Question: {{prompt}}"
  template_vars:
    focus_area: "urban planning"
```

Then use it:

```bash
python -m dagspaces.urbanvqa.cli pipeline=vqa
```

### With Structured JSON Output

Configure structured output:

```yaml
prompt:
  structured_output:
    enabled: true
    json_schema:
      type: object
      properties:
        answer:
          type: string
        confidence:
          type: number
      required:
        - answer
```

---

## Batch Inference

### Independent Prompt+Image Pairs

By default, the pipeline processes each prompt+image pair independently:

```python
# Each row is processed independently
df = pd.DataFrame({
    "prompt": ["Q1", "Q2", "Q3"],
    "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"]
})
```

### Prompt Grouping

When multiple images share the same prompt, enable grouping for efficiency:

```python
# Same prompt, different images
df = pd.DataFrame({
    "prompt": ["What type of building?", "What type of building?", "What type of building?"],
    "image_path": ["img1.jpg", "img2.jpg", "img3.jpg"]
})
```

With `group_by_prompt: true`, these will be grouped together, avoiding redundant prompt tokenization.

### Batch Size Configuration

Adjust batch size based on GPU memory:

```yaml
model:
  batch_size: 16  # Number of prompt+image pairs per batch
  concurrency: 1  # Number of parallel workers
```

**Note**: When using tensor parallelism (`tensor_parallel_size > 1`), `concurrency` should match the number of model replicas, not total GPU count:
- Example: 2 GPUs, `tensor_parallel_size=2` → `concurrency=1` (one replica using both GPUs)
- Example: 4 GPUs, `tensor_parallel_size=2` → `concurrency=2` (two replicas, each using 2 GPUs)

---

## Output Format

### Standard Output Schema

The pipeline returns a DataFrame with the following columns:

```python
{
    "sample_id": str,          # Unique identifier
    "prompt": str,             # Original prompt
    "image_path": str,         # Image source (path/URL/base64)
    "answer": str,             # Model's answer (extracted from response)
    "model_response": str,     # Full model response
    "metadata": dict,          # Additional metadata (timing, tokens, etc.)
}
```

### Structured JSON Output

When structured output is enabled, additional fields are extracted:

```python
{
    "sample_id": "sample_001",
    "prompt": "What type of building?",
    "answer": "Residential building",
    "confidence": 0.95,
    "reasoning": "The image shows a multi-story residential structure...",
    "model_response": "{'answer': 'Residential building', 'confidence': 0.95, ...}",
    "metadata": {...}
}
```

---

## Troubleshooting

### Image Loading Failures

**Problem**: Images fail to load.

**Solutions**:
- Verify image paths are correct and accessible
- Check file permissions
- For URLs, ensure they are publicly accessible or use authentication
- Enable `image_fallback: true` for graceful degradation (not recommended for VQA)

### Out of Memory Errors

**Problem**: GPU runs out of memory.

**Solutions**:
- Reduce `batch_size` in model configuration
- Reduce `max_model_len` if using a smaller context window
- Enable tensor parallelism: `tensor_parallel_size: 2` (requires multiple GPUs)
- Reduce `gpu_memory_utilization` (default: 0.9)

### Model Not Found

**Problem**: Model source not found.

**Solutions**:
- Verify model path or HuggingFace model ID
- Check `MODEL_ZOO_BASE` environment variable if using local models
- Ensure `trust_remote_code: true` for custom vision models

### Slow Performance

**Problem**: Inference is slow.

**Solutions**:
- Enable `group_by_prompt: true` if using repeated prompts
- Increase `batch_size` (within GPU memory limits)
- Use `streaming_io: true` for large datasets
- Adjust `concurrency` based on GPU count and tensor parallelism
- Check `max_num_batched_tokens` in `engine_kwargs` for throughput tuning

### Ray Initialization Errors

**Problem**: Ray fails to initialize.

**Solutions**:
- Ensure Ray is installed: `pip install ray`
- Check GPU availability: `ray status`
- Verify CUDA_VISIBLE_DEVICES environment variable

---

## Advanced Configuration

### Jinja2 Templates

Enable dynamic prompt generation:

```yaml
prompt:
  template: |
    {% if focus_area %}
    Focus on {{focus_area}}.
    {% endif %}
    Question: {{prompt}}
  template_vars:
    focus_area: "urban planning"
```

Template variables can come from:
- Config file (`template_vars`)
- Data columns (all columns available as template variables)
- Row metadata

### Structured Output with Pydantic

Define a Pydantic model for structured output:

```python
# dagspaces/urbanvqa/schemas/vqa_answer.py
from pydantic import BaseModel
from typing import Optional, List

class VQAAnswer(BaseModel):
    answer: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    entities: Optional[List[str]] = None
```

Then reference in config:

```yaml
prompt:
  structured_output:
    enabled: true
    schema_type: "pydantic"
    schema_path: "dagspaces/urbanvqa/schemas/vqa_answer.py"
```

### Model Zoo Integration

Use local models from model zoo:

```bash
export MODEL_ZOO_BASE=/share/pierson/matt/zoo/models
python -m dagspaces.urbanvqa.cli pipeline=vqa
```

The pipeline will check the local zoo first, then fall back to HuggingFace Hub.

---

## Example Workflows

### Single Image VQA

```python
import pandas as pd
from dagspaces.urbanvqa.stages.vqa import run_vqa_stage
from omegaconf import OmegaConf

# Create input
df = pd.DataFrame({
    "prompt": ["What type of building is this?"],
    "image_path": ["/path/to/building.jpg"],
    "sample_id": ["test_001"]
})

# Load config
cfg = OmegaConf.load("dagspaces/urbanvqa/conf/pipeline/vqa.yaml")

# Run inference
results = run_vqa_stage(df, cfg)
print(results["answer"].iloc[0])
```

### Batch Processing

```python
# Process multiple images
df = pd.DataFrame({
    "prompt": [
        "What type of building?",
        "Describe the scene.",
        "What is the land use?"
    ],
    "image_path": [
        "/path/to/img1.jpg",
        "/path/to/img2.jpg",
        "/path/to/img3.jpg"
    ]
})

results = run_vqa_stage(df, cfg)
for idx, row in results.iterrows():
    print(f"Sample {row['sample_id']}: {row['answer']}")
```

### Using URLs

```python
df = pd.DataFrame({
    "prompt": ["What is in this image?"],
    "image_url": ["https://example.com/image.jpg"],
    "sample_id": ["url_test"]
})

results = run_vqa_stage(df, cfg)
```

---

## Best Practices

1. **Use Streaming for Large Datasets**: Enable `streaming_io: true` for datasets that don't fit in memory.

2. **Enable Prompt Grouping**: If you have repeated prompts, use `group_by_prompt: true` for efficiency.

3. **Optimize Batch Size**: Start with `batch_size: 16` and adjust based on GPU memory and throughput.

4. **Set Concurrency Correctly**: With tensor parallelism, set `concurrency = total_gpus / tensor_parallel_size`.

5. **Validate Input Data**: Ensure all required columns exist and images are accessible before running.

6. **Monitor GPU Usage**: Use `nvidia-smi` or Ray dashboard to monitor GPU utilization.

7. **Use Structured Output**: For downstream processing, enable structured JSON output for easier parsing.

8. **Error Handling**: Enable `image_fallback: false` for VQA (images are required) to catch errors early.

---

## Getting Help

For issues or questions:
- Check the troubleshooting section above
- Review `documentation/urbanvqa_implementation.md` for implementation details
- Check Ray and vLLM documentation for framework-specific issues

