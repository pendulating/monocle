# Multimodal Language Model (MLLM) Support Implementation Plan

## Overview

This document outlines the implementation plan for adding **image classification** support to UAIR using Vision-Language Models (VLMs) and Multimodal Language Models (MLLMs) via Ray Data's batch inference capabilities and vLLM's multimodal support.

## Current State Analysis

### Existing Architecture

1. **Data Pipeline**:
   - Input format: Parquet files with columns: `article_text`, `article_id`, `article_path`, `country`, `year`
   - Supports both pandas DataFrame and Ray Dataset (streaming mode)
   - Column mapping via `data.columns` config

2. **Classification Stage** (`dagspaces/uair/stages/classify.py`):
   - Text-only classification using vLLM via Ray Data LLM processor
   - Supports multiple profiles: `relevance`, `eu_ai_act`, `risks_and_benefits`
   - Uses `build_llm_processor` with `vLLMEngineProcessorConfig`
   - Preprocessing/postprocessing functions format text prompts
   - Streaming support for large datasets

3. **Orchestrator** (`dagspaces/uair/orchestrator.py`):
   - Stage runners (`ClassificationRunner`, `ClassificationEUActRunner`, etc.)
   - Handles data loading via `prepare_stage_input()`
   - Manages Ray initialization and resource allocation
   - Supports streaming datasets via `_prepare_streaming_dataset()`

### Current Limitations

- **No image support**: Only processes `article_text` column
- **Text-only prompts**: Messages format assumes text-only content
- **No image loading**: No mechanism to load images from paths or URLs
- **No multimodal preprocessing**: Cannot handle PIL Images or image arrays

## Implementation Plan

### Phase 1: Data Schema Extensions

#### 1.1 Extend Column Mapping Configuration

**File**: `dagspaces/uair/orchestrator.py`

**Changes**:
- Extend `_load_parquet_dataset()` to support image-related columns:
  - `image_path`: Path to image file (local or URL)
  - `image_url`: URL to image (alternative to path)
  - `image_base64`: Base64-encoded image (for inline storage)
  - `image_bytes`: Raw image bytes (for Parquet storage)

**New column mapping**:
```python
col_map = {
    columns.get("article_text", "article_text"): "article_text",
    columns.get("article_path", "article_path"): "article_path",
    columns.get("image_path", "image_path"): "image_path",
    columns.get("image_url", "image_url"): "image_url",
    columns.get("image_base64", "image_base64"): "image_base64",
    # ... existing columns
}
```

#### 1.2 Image Loading Utilities

**File**: `dagspaces/uair/stages/classify.py`

**New functions**:
```python
def _load_image_from_row(row: Dict[str, Any]) -> Optional[PIL.Image.Image]:
    """Load image from various sources in row.
    
    Priority:
    1. image_path (local file or URL)
    2. image_url
    3. image_base64
    4. image_bytes (if already loaded)
    5. image (if already a PIL Image or numpy array)
    
    Returns PIL.Image.Image in RGB format, or None on failure.
    """
    # Implementation with PIL Image.open() and URL fetching
    pass

def _ensure_image_column(ds_or_df: Union[pd.DataFrame, ray.data.Dataset]) -> Union[pd.DataFrame, ray.data.Dataset]:
    """Ensure 'image' column exists with PIL.Image objects."""
    # Convert paths/URLs/base64 to PIL Images
    pass
```

**Important Notes**:
- Ray Data's `read_images()` returns numpy arrays (not PIL Images) - conversion required
- All images must be converted to RGB format for vLLM compatibility
- Handle `include_paths=True` option when reading images from directories
- Support lazy loading to avoid loading all images into memory at once

### Phase 2: Ray Data Image Support

#### 2.1 Update Streaming Dataset Preparation

**File**: `dagspaces/uair/orchestrator.py`

**Changes**:
- Extend `_prepare_streaming_dataset()` to handle image columns
- Use `ray.data.read_images()` for image directory inputs (returns numpy arrays in 'image' column)
- Use `ray.data.read_images()` with `include_paths=True` to get file paths
- Merge image data with text data when both are present
- Convert numpy arrays to PIL Images in a map function

**New function**:
```python
def _prepare_multimodal_dataset(
    dataset_path: str,
    image_path: Optional[str],
    columns: Mapping[str, str],
    cfg: DictConfig,
    stage: str
) -> tuple[Optional[ray.data.Dataset], bool]:
    """Prepare Ray Dataset with both text and image support.
    
    Handles multiple scenarios:
    1. Images stored in Parquet (image_path, image_url, image_base64 columns)
    2. Images in separate directory (image_path parameter)
    3. Images already loaded as numpy arrays from ray.data.read_images()
    
    Returns Ray Dataset with 'image' column containing PIL.Image objects.
    """
    # Load parquet for text/metadata
    ds_text = ray.data.read_parquet(dataset_path)
    
    # Option 1: Images in separate directory
    if image_path:
        ds_images = ray.data.read_images(image_path, include_paths=True)
        # Convert numpy arrays to PIL Images
        ds_images = ds_images.map(_convert_numpy_to_pil)
        # Merge on paths/article_id
        ds = ds_text.join(ds_images, on="article_path", how="left")
    
    # Option 2: Images in parquet (paths/URLs)
    elif any(col in ds_text.columns() for col in ["image_path", "image_url"]):
        # Load images in map function (lazy)
        ds = ds_text.map(_load_images_map)
    
    # Option 3: Already have numpy arrays
    else:
        ds = ds_text.map(_convert_numpy_to_pil_if_needed)
    
    return ds, True
```

**Key Implementation Details**:
- `ray.data.read_images()` returns `numpy.ndarray` in 'image' column, not PIL Images
- Must convert numpy arrays to PIL Images: `PIL.Image.fromarray(arr).convert("RGB")`
- Use `include_paths=True` to get file paths for merging with metadata
- Lazy loading: images loaded on-demand in map functions, not at dataset creation

#### 2.2 Image Column Mapping in Ray Dataset

**File**: `dagspaces/uair/orchestrator.py`

**Changes**:
- Extend `_ensure_canon()` mapping function to handle image columns
- Support both PIL Images and image paths (lazy loading)

### Phase 3: vLLM Multimodal Configuration

#### 3.1 Extend vLLM Engine Configuration

**File**: `dagspaces/uair/stages/classify.py`

**Changes**:
- Update `vLLMEngineProcessorConfig` to support multimodal models:
  - Add `has_image=True` parameter
  - Configure `limit_mm_per_prompt={"image": 1}` in engine_kwargs
  - Set `mm_processor_kwargs` for image processing settings

**New configuration function**:
```python
def _configure_multimodal_engine(
    cfg: DictConfig,
    model_source: str
) -> vLLMEngineProcessorConfig:
    """Configure vLLM engine for multimodal inference."""
    ek = dict(getattr(cfg.model, "engine_kwargs", {}))
    
    # Resolve model path from zoo or HuggingFace Hub
    resolved_model_source = _resolve_model_path(model_source)
    
    # Detect if model supports images
    is_multimodal = _is_multimodal_model(resolved_model_source)
    
    if is_multimodal:
        # Required: limit multimodal inputs per prompt
        ek.setdefault("limit_mm_per_prompt", {"image": 1})
        
        # Model-specific image processing settings
        # Qwen models use min_pixels/max_pixels
        # InternVL uses max_dynamic_patch
        # Phi-3 uses num_crops
        ek.setdefault("mm_processor_kwargs", {
            "min_pixels": 28 * 28,  # 784
            "max_pixels": 1280 * 28 * 28,  # 9830400 for Qwen
        })
        
        # Increase max_num_batched_tokens for multimodal (default: 5120)
        # vs text-only (default: 2048)
        ek.setdefault("max_num_batched_tokens", 5120)
    
    # Runtime environment for HuggingFace token (if needed)
    runtime_env = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        runtime_env["env_vars"] = {"HF_TOKEN": hf_token}
    
    return vLLMEngineProcessorConfig(
        model_source=resolved_model_source,  # Use resolved path
        engine_kwargs=ek,
        has_image=is_multimodal,  # Required for multimodal models
        runtime_env=runtime_env if runtime_env else None,
        accelerator_type=getattr(cfg.model, "accelerator_type", None),  # e.g., "L4"
        # ... other config (batch_size, concurrency, etc.)
    )
```

**Critical Configuration Details**:
- `has_image=True` is **required** in `vLLMEngineProcessorConfig` for multimodal models
- `limit_mm_per_prompt={"image": 1}` must be in `engine_kwargs`, not top-level
- `max_num_batched_tokens` defaults to 5120 for multimodal (vs 2048 for text-only)
- `mm_processor_kwargs` varies by model:
  - Qwen: `min_pixels`, `max_pixels`, `fps` (for video)
  - InternVL: `max_dynamic_patch`
  - Phi-3: `num_crops` (16 for single frame, 4 for multi-frame)
  - SmolVLM: `max_image_size={"longest_edge": 384}`
- `accelerator_type` can be specified (e.g., "L4", "A100") for Ray cluster optimization
- `runtime_env` needed for HuggingFace token when accessing private models

#### 3.2 Model Path Resolution and Zoo Integration

**File**: `dagspaces/uair/stages/classify.py`

**New functions**:
```python
import os
import re
import json
from pathlib import Path

# Model zoo base path
MODEL_ZOO_BASE = "/share/pierson/matt/zoo/models"

def _resolve_model_path(model_source: str) -> str:
    """Resolve model path from zoo or HuggingFace Hub.
    
    Priority:
    1. If absolute path exists: use as-is
    2. If relative path in zoo: resolve to zoo/models/{model_source}
    3. If model name matches zoo directory: resolve to zoo/models/{name}
    4. Otherwise: use as HuggingFace Hub identifier
    
    Args:
        model_source: Model path/name from config (e.g., "Qwen2.5-VL-3B-Instruct" 
                     or "/share/pierson/matt/zoo/models/Qwen3-30B-A3B-Instruct-2507")
    
    Returns:
        Resolved path (absolute path or HuggingFace Hub identifier)
    """
    # Already an absolute path
    if os.path.isabs(model_source) and os.path.exists(model_source):
        return model_source
    
    # Check if it's in the zoo
    zoo_path = os.path.join(MODEL_ZOO_BASE, model_source)
    if os.path.exists(zoo_path):
        return zoo_path
    
    # Try fuzzy matching in zoo directory
    if os.path.exists(MODEL_ZOO_BASE):
        try:
            zoo_dirs = [d for d in os.listdir(MODEL_ZOO_BASE) 
                       if os.path.isdir(os.path.join(MODEL_ZOO_BASE, d))]
            # Check for exact match or partial match
            for dir_name in zoo_dirs:
                if model_source.lower() in dir_name.lower() or dir_name.lower() in model_source.lower():
                    resolved = os.path.join(MODEL_ZOO_BASE, dir_name)
                    if os.path.exists(os.path.join(resolved, "config.json")):
                        return resolved
        except Exception:
            pass  # Fallback to HuggingFace Hub
    
    # Fallback to HuggingFace Hub
    return model_source

def _is_multimodal_model(model_source: str) -> bool:
    """Detect if model supports multimodal inputs.
    
    Checks:
    1. Model name patterns (e.g., "Qwen2.5-VL", "InternVL", "Phi-3.5-vision")
    2. Explicit config flag: runtime.multimodal_enabled
    3. Model config metadata (if available)
    4. Config.json in model directory for local models
    
    Returns True if multimodal capabilities detected.
    """
    # Resolve model path first
    resolved_path = _resolve_model_path(model_source)
    
    # Check explicit flag first
    if hasattr(cfg.runtime, "multimodal_enabled"):
        return bool(cfg.runtime.multimodal_enabled)
    
    # For local models, check config.json
    if os.path.isabs(resolved_path) and os.path.exists(resolved_path):
        config_path = os.path.join(resolved_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                    # Check for multimodal indicators in config
                    model_type = config.get("model_type", "").lower()
                    arch = config.get("architectures", [])
                    if any("vision" in str(a).lower() or "multimodal" in str(a).lower() 
                           for a in arch):
                        return True
                    if "vision" in model_type or "multimodal" in model_type:
                        return True
            except Exception:
                pass  # Fallback to pattern matching
    
    # Pattern matching for common multimodal models
    multimodal_patterns = [
        r"Qwen.*VL",
        r"InternVL",
        r"Phi-3.*vision",
        r"SmolVLM",
        r"LLaVA",
        r"CLIP",
        r"vision",
        r"multimodal",
    ]
    
    model_lower = model_source.lower()
    return any(re.search(pattern, model_lower, re.IGNORECASE) for pattern in multimodal_patterns)
```

### Phase 4: Multimodal Preprocessing

#### 4.1 Extend Preprocessing Function

**File**: `dagspaces/uair/stages/classify.py`

**Changes**:
- Modify `_pre()` function to handle multimodal messages
- Support OpenAI Chat API format with image content:
  ```python
  {
      "role": "user",
      "content": [
          {"type": "text", "text": "Question text"},
          {"type": "image", "image": PIL.Image.Image}
      ]
  }
  ```

**New preprocessing logic**:
```python
def _pre_multimodal(row: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess row for multimodal inference.
    
    Supports multiple image input formats:
    - PIL.Image.Image: Direct use
    - numpy.ndarray: Convert to PIL Image
    - image_path: Load from local file or URL
    - image_url: Load from URL
    - image_base64: Decode base64 string
    - image_bytes: Load from bytes
    
    Message format follows OpenAI Chat API:
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "image": PIL.Image.Image}  # Direct PIL Image
            # OR
            {"type": "image_url", "image_url": {"url": "..."}}  # URL format
        ]
    }
    """
    _maybe_silence_vllm_logs()
    
    # Ensure article_id
    # ... existing article_id logic
    
    # Load and normalize image if present
    image = None
    image_url = None
    
    if "image" in row:
        img_val = row["image"]
        if isinstance(img_val, PIL.Image.Image):
            image = img_val.convert("RGB")
        elif isinstance(img_val, np.ndarray):
            image = PIL.Image.fromarray(img_val).convert("RGB")
        elif isinstance(img_val, bytes):
            image = PIL.Image.open(BytesIO(img_val)).convert("RGB")
    elif "image_path" in row and row["image_path"]:
        path = row["image_path"]
        if path.startswith(("http://", "https://")):
            image_url = path  # Use URL format for remote images
        else:
            image = _load_image_from_path(path)
    elif "image_url" in row and row["image_url"]:
        image_url = row["image_url"]
    elif "image_base64" in row and row["image_base64"]:
        image = _load_image_from_base64(row["image_base64"])
    
    # Build multimodal messages
    user_content = []
    
    # Add text prompt
    if is_eu_profile:
        user_text = _format_user_eu(row)
    elif is_risks_benefits_profile:
        user_text = _format_user_risks_benefits(row)
    else:
        user_text = _format_user(row.get("article_text"), row)
    
    user_content.append({"type": "text", "text": user_text})
    
    # Add image if available (prefer PIL Image, fallback to URL)
    if image is not None:
        user_content.append({"type": "image", "image": image})
    elif image_url is not None:
        # Use URL format (vLLM handles fetching)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url},
            # Optional UUID for caching (use article_id if available)
            "uuid": row.get("article_id") or image_url,
        })
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # ... rest of preprocessing (sampling params, etc.)
    return {
        **base,
        "messages": messages,
        "sampling_params": sp_local,
        "ts_start": _dt.now().timestamp(),
    }
```

**Important Message Format Details**:
- vLLM accepts PIL Image objects directly: `{"type": "image", "image": PIL.Image.Image}`
- Also supports URL format: `{"type": "image_url", "image_url": {"url": "..."}}`
- Base64 format: `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`
- UUID optional for caching: `{"uuid": "unique_id"}` can improve performance
- Content array order matters: text first, then images
- Mixed content: can have multiple text and image items in same message

#### 4.2 Image Format Conversion

**File**: `dagspaces/uair/stages/classify.py`

**New utilities**:
```python
def _load_image_from_path(path: str) -> Optional[PIL.Image.Image]:
    """Load image from local path or URL.
    
    Handles:
    - Local file paths (absolute or relative)
    - HTTP/HTTPS URLs
    - Data URLs (data:image/...;base64,...)
    
    Always converts to RGB format for vLLM compatibility.
    """
    try:
        if path.startswith(("http://", "https://")):
            import requests
            response = requests.get(path, timeout=10, headers={"User-Agent": "UAIR/1.0"})
            response.raise_for_status()
            return PIL.Image.open(BytesIO(response.content)).convert("RGB")
        elif path.startswith("data:image/"):
            # Handle data URL: data:image/jpeg;base64,...
            import base64
            header, encoded = path.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return PIL.Image.open(BytesIO(image_bytes)).convert("RGB")
        else:
            return PIL.Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to load image from {path}: {e}", flush=True)
        return None

def _load_image_from_base64(base64_str: str) -> Optional[PIL.Image.Image]:
    """Load image from base64-encoded string."""
    try:
        import base64
        # Handle with or without data URL prefix
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(base64_str)
        return PIL.Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to decode base64 image: {e}", flush=True)
        return None

def _normalize_image(image: Any) -> Optional[PIL.Image.Image]:
    """Normalize various image formats to PIL.Image.
    
    Supports:
    - PIL.Image.Image: Direct conversion to RGB
    - numpy.ndarray: Convert array to PIL Image
    - bytes: Load from bytes buffer
    - str: Try base64 decode, then path/URL load
    
    All images converted to RGB format.
    """
    if image is None:
        return None
    
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    elif isinstance(image, np.ndarray):
        # Handle different array shapes and dtypes
        if image.dtype != np.uint8:
            # Normalize float arrays to [0, 255]
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        # Handle different channel orders (HWC vs CHW)
        if len(image.shape) == 3 and image.shape[0] < image.shape[2]:
            # Likely CHW format, transpose to HWC
            image = image.transpose(1, 2, 0)
        return PIL.Image.fromarray(image).convert("RGB")
    elif isinstance(image, bytes):
        return PIL.Image.open(BytesIO(image)).convert("RGB")
    elif isinstance(image, str):
        # Try base64 decode first
        if image.startswith("data:image/") or len(image) > 100:
            try:
                return _load_image_from_base64(image)
            except Exception:
                pass
        # Fallback to path/URL
        return _load_image_from_path(image)
    return None

def _convert_numpy_to_pil(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy array in 'image' column to PIL Image.
    
    Used when ray.data.read_images() returns numpy arrays.
    """
    if "image" in row and isinstance(row["image"], np.ndarray):
        row["image"] = _normalize_image(row["image"])
    return row

def _convert_numpy_to_pil_if_needed(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays to PIL Images if present."""
    return _convert_numpy_to_pil(row)
```

**Critical Implementation Notes**:
- **Always convert to RGB**: vLLM expects RGB format, not RGBA or grayscale
- **Handle numpy array shapes**: Ray Data returns HWC format, but must handle CHW too
- **Normalize array dtypes**: Convert float arrays [0,1] to uint8 [0,255]
- **URL fetching**: Include timeout and proper headers for HTTP requests
- **Data URL support**: Handle `data:image/jpeg;base64,...` format
- **Error handling**: Log warnings but don't fail entire batch on single image error

### Phase 5: Ray Dataset Image Handling

#### 5.1 Map Functions for Image Loading

**File**: `dagspaces/uair/stages/classify.py`

**New map functions**:
```python
def _load_images_map(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ray Dataset map function to load images from paths/URLs.
    
    This is called lazily during Ray Dataset execution, not at dataset creation.
    Handles multiple input formats and converts to PIL Image.
    """
    image = None
    
    # Priority order: existing image > path > URL > base64
    if "image" in row and row["image"] is not None:
        image = _normalize_image(row["image"])
    elif "image_path" in row and row["image_path"]:
        image = _load_image_from_path(row["image_path"])
    elif "image_url" in row and row["image_url"]:
        # For URLs, we can either load now or pass URL to vLLM
        # Loading now is safer for batch processing
        image = _load_image_from_path(row["image_url"])
    elif "image_base64" in row and row["image_base64"]:
        image = _load_image_from_base64(row["image_base64"])
    
    # Set image column (None if loading failed)
    row["image"] = image
    return row

def _ensure_images_map(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure image column exists (for compatibility).
    
    Called before preprocessing to ensure consistent schema.
    """
    if "image" not in row:
        row["image"] = None
    return row

def _convert_numpy_to_pil_map(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy array images to PIL Images.
    
    Called after ray.data.read_images() which returns numpy arrays.
    """
    if "image" in row:
        if isinstance(row["image"], np.ndarray):
            row["image"] = _normalize_image(row["image"])
        elif not isinstance(row["image"], PIL.Image.Image):
            # Try to normalize other formats
            row["image"] = _normalize_image(row["image"])
    return row
```

**Performance Considerations**:
- **Lazy loading**: Images loaded on-demand in map functions, not at dataset creation
- **Parallel execution**: Ray Dataset maps execute in parallel across workers
- **Memory management**: Only load images needed for current batch
- **Caching**: Consider using UUID in messages for vLLM's internal caching
- **Batch size**: Smaller batches may be needed for large images (configurable)

#### 5.2 Update Streaming Path

**File**: `dagspaces/uair/stages/classify.py`

**Changes**:
- In `run_classification_stage()`, add image loading step before LLM processing:
  ```python
  if is_ray_ds:
      # ... existing preprocessing (article_id, keyword filtering, etc.)
      
      # Convert numpy arrays to PIL Images if ray.data.read_images() was used
      # Check if 'image' column exists and contains numpy arrays
      try:
        sample = ds_in.take(1)
        if sample and "image" in sample[0]:
            if isinstance(sample[0]["image"], np.ndarray):
                ds_in = ds_in.map(_convert_numpy_to_pil_map)
      except Exception:
          pass  # Continue if check fails
      
      # Load images from paths/URLs if present
      if any(col in ds_in.columns() for col in ["image_path", "image_url", "image_base64"]):
          ds_in = ds_in.map(_load_images_map)
      
      # Ensure image column exists (even if None)
      ds_in = ds_in.map(_ensure_images_map)
      
      # Continue with LLM processing
      processor = build_llm_processor(engine_config, preprocess=_pre_multimodal, postprocess=_post)
      ds_llm_results = processor(ds_in)
  ```

**Implementation Details**:
- Check for numpy arrays before converting (avoid unnecessary conversion)
- Handle mixed batches: some rows with images, some without
- Images can be None (will be handled gracefully in preprocessing)
- Preserve existing columns while adding/updating image column

### Phase 6: Configuration Updates

#### 6.1 Model Configuration

**File**: `dagspaces/uair/conf/model/`

**New config file**: `vllm_multimodal.yaml` (example)
**Existing config file**: `vllm_qwen3-30b.yaml` (already uses zoo)

**Model Zoo Structure**:
The model zoo is located at `/share/pierson/matt/zoo/models/` and contains downloaded HuggingFace models. Each model is stored in its own directory with the standard HuggingFace model structure (config.json, tokenizer files, model weights, etc.).

**Example Zoo Models**:
- `Qwen3-30B-A3B-Instruct-2507/` - Already configured in `vllm_qwen3-30b.yaml`
- `intfloat_e5_base/` - Embedding model
- `mdeberta_v3_base_xnli_multilingual_nli_2mil7/` - NLI model

**New config file**: `vllm_multimodal.yaml`
```yaml
# Model source: Can be HuggingFace Hub identifier or local zoo path
# Examples:
#   - HuggingFace Hub: "Qwen/Qwen2.5-VL-3B-Instruct"
#   - Absolute zoo path: "/share/pierson/matt/zoo/models/Qwen2.5-VL-3B-Instruct"
#   - Relative zoo name: "Qwen2.5-VL-3B-Instruct" (auto-resolved to zoo)
model_source: "Qwen/Qwen2.5-VL-3B-Instruct"  # Example multimodal model

engine_kwargs:
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  max_model_len: 4096
  max_num_seqs: 5
  enable_chunked_prefill: true
  max_num_batched_tokens: 5120  # Higher for multimodal (default: 5120 vs 2048)
  limit_mm_per_prompt:
    image: 1  # Maximum images per prompt
  mm_processor_kwargs:
    # Qwen-specific settings
    min_pixels: 784  # 28*28
    max_pixels: 9830400  # 1280*28*28 (for Qwen2.5-VL)
    fps: 1  # For video support (if needed)
  # Optional: trust_remote_code may be needed for some models
  trust_remote_code: false
  
batch_size: 16  # May need to reduce for large images
concurrency: 1
has_image: true  # Required for multimodal models
accelerator_type: "L4"  # Optional: GPU type hint for Ray

# Runtime environment (if HuggingFace token needed)
runtime_env:
  env_vars:
    HF_TOKEN: ${oc.env:HF_TOKEN,}  # From environment variable
    VLLM_USE_V1: "1"  # Optional: use v1 backend if needed
```

**Model-Specific Configurations**:

**For Qwen models** (`Qwen/Qwen2.5-VL-*`, `Qwen/Qwen3-VL-*`):
```yaml
mm_processor_kwargs:
  min_pixels: 784  # 28*28
  max_pixels: 9830400  # 1280*28*28
  fps: 1  # For video
```

**For InternVL models** (`OpenGVLab/InternVL*`):
```yaml
mm_processor_kwargs:
  max_dynamic_patch: 4  # Reduce for memory savings
trust_remote_code: true  # Usually required
```

**For Phi-3 models** (`microsoft/Phi-3.5-vision-instruct`):
```yaml
mm_processor_kwargs:
  num_crops: 16  # 16 for single frame, 4 for multi-frame
trust_remote_code: true  # Required
```

**For SmolVLM models** (`HuggingFaceTB/SmolVLM2-*`):
```yaml
mm_processor_kwargs:
  max_image_size:
    longest_edge: 384
enforce_eager: true  # May be needed for small GPUs
```

#### 6.2 Model Zoo Configuration Examples

**File**: `dagspaces/uair/conf/model/`

**Example 1: Using Zoo Model (Absolute Path)**
```yaml
# vllm_multimodal_zoo.yaml
model_source: /share/pierson/matt/zoo/models/Qwen2.5-VL-3B-Instruct

engine_kwargs:
  tensor_parallel_size: 1
  max_model_len: 4096
  max_num_batched_tokens: 5120
  limit_mm_per_prompt:
    image: 1
  mm_processor_kwargs:
    min_pixels: 784
    max_pixels: 9830400
  trust_remote_code: false  # Usually not needed for zoo models

batch_size: 16
concurrency: 1
has_image: true
```

**Example 2: Using Zoo Model (Auto-Resolved)**
```yaml
# vllm_multimodal_zoo_auto.yaml
# Model name will be auto-resolved to zoo if directory exists
model_source: Qwen2.5-VL-3B-Instruct  # Auto-resolved to zoo/models/Qwen2.5-VL-3B-Instruct

engine_kwargs:
  # ... same as above
```

**Example 3: Using HuggingFace Hub (Fallback)**
```yaml
# vllm_multimodal_hf.yaml
# If model not in zoo, falls back to HuggingFace Hub
model_source: Qwen/Qwen2.5-VL-3B-Instruct

engine_kwargs:
  # ... same as above
  trust_remote_code: true  # May be needed for Hub models
```

**Adding Models to Zoo**:
1. Download model using HuggingFace CLI or Python:
   ```bash
   # Using HuggingFace CLI
   huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /share/pierson/matt/zoo/models/Qwen2.5-VL-3B-Instruct
   
   # Or using Python script (see zoo/download_*.py examples)
   ```
2. Update config to use absolute path or model name (if auto-resolution enabled)
3. Ensure model directory contains `config.json` for proper detection

#### 6.3 Data Configuration

**File**: `dagspaces/uair/conf/data/`

**Extended config**: `inputs.yaml`
```yaml
parquet_path: ${oc.env:DATA_ROOT}/multimodal_dataset.parquet
columns:
  article_text: article_text
  article_id: article_id
  image_path: image_path  # New: path to image files
  image_url: image_url    # Alternative: URL to images
```

#### 6.3 Runtime Configuration

**File**: `dagspaces/uair/orchestrator.py`

**New runtime options**:
```python
cfg.runtime.multimodal_enabled = True
cfg.runtime.image_column = "image_path"  # or "image_url", "image_base64"
cfg.runtime.image_fallback = True  # Fallback to text-only if image missing
```

### Phase 7: Stage Runner Updates

#### 7.1 Update ClassificationRunner

**File**: `dagspaces/uair/orchestrator.py`

**Changes**:
- Detect multimodal mode from config or model source
- Enable multimodal preprocessing when appropriate
- Handle image loading failures gracefully

**Modified `ClassificationRunner.run()`**:
```python
def run(self, context: StageExecutionContext) -> StageResult:
    dataset_path = context.inputs.get("dataset")
    cfg = context.cfg
    
    # Check if multimodal mode enabled
    multimodal_enabled = bool(getattr(cfg.runtime, "multimodal_enabled", False))
    model_source = getattr(cfg.model, "model_source", "")
    is_multimodal = multimodal_enabled or _is_multimodal_model(model_source)
    
    # Prepare input (with image support)
    df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
    
    # If multimodal, ensure images are loaded
    if is_multimodal and use_streaming:
        # Add image loading map
        pass
    
    # Run classification
    in_obj = ds if use_streaming and ds is not None else df
    out = run_classification_relevance(in_obj, cfg)
    
    # ... rest of method
```

### Phase 8: Error Handling and Fallbacks

#### 8.1 Graceful Degradation

**File**: `dagspaces/uair/stages/classify.py`

**New logic**:
- If image loading fails, fallback to text-only classification
- Log warnings when images are missing
- Support mixed batches (some rows with images, some without)
- Handle None images gracefully in preprocessing

**Implementation**:
```python
def _pre_multimodal_with_fallback(row: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess with automatic fallback to text-only.
    
    Always returns valid messages, even if image loading fails.
    Configurable via runtime.image_fallback setting.
    """
    image = None
    image_load_failed = False
    
    try:
        # Try to load image
        image = _load_image_from_row(row)
        if image is None:
            image_load_failed = True
    except Exception as e:
        image_load_failed = True
        if not getattr(cfg.runtime, "image_fallback", True):
            raise RuntimeError(f"Image loading failed and fallback disabled: {e}") from e
        # Log warning but continue
        article_id = row.get("article_id", "unknown")
        print(f"Warning: Failed to load image for article_id={article_id}: {e}", flush=True)
    
    # Build messages (with or without image)
    user_content = []
    
    # Add text prompt (always present)
    if is_eu_profile:
        user_text = _format_user_eu(row)
    elif is_risks_benefits_profile:
        user_text = _format_user_risks_benefits(row)
    else:
        user_text = _format_user(row.get("article_text"), row)
    
    user_content.append({"type": "text", "text": user_text})
    
    # Add image only if successfully loaded
    if image is not None:
        user_content.append({"type": "image", "image": image})
    elif image_load_failed and getattr(cfg.runtime, "image_fallback", True):
        # Add metadata flag for downstream tracking
        row["image_load_failed"] = True
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # ... rest of preprocessing
    return {
        **base,
        "messages": messages,
        "sampling_params": sp_local,
        "ts_start": _dt.now().timestamp(),
    }
```

**Additional Error Handling**:
- **Mixed batches**: Handle rows with and without images in same batch
- **Image validation**: Check image dimensions before processing (avoid oversized images)
- **Memory errors**: Catch CUDA OOM and suggest reducing batch_size
- **Timeout handling**: Add timeouts for URL fetching (default 10 seconds)
- **Format validation**: Verify image format before processing (skip unsupported formats)
- **Logging**: Track image loading success/failure rates for monitoring

### Phase 9: Testing and Validation

#### 9.1 Unit Tests

**New test file**: `tests/test_multimodal_classify.py`

**Test cases**:
- Image loading from various sources (path, URL, base64)
- Multimodal message formatting
- Fallback to text-only on image errors
- Ray Dataset image processing
- Model detection logic

#### 9.2 Integration Tests

**Test scenarios**:
- End-to-end classification with images
- Mixed batches (text + images)
- Streaming mode with images
- Large dataset processing

### Phase 10: Documentation

#### 10.1 User Guide Updates

**File**: `docs/USER_GUIDE.md`

**New sections**:
- Multimodal Classification Guide
- Image Input Formats
- Model Selection for Multimodal Tasks
- Performance Considerations

#### 10.2 Configuration Examples

**New file**: `docs/examples/multimodal_classification.yaml`

**Example pipeline config**:
```yaml
defaults:
  - /pipeline: classify_multimodal
  - /model: vllm_multimodal
  - /data: multimodal_inputs

runtime:
  multimodal_enabled: true
  image_column: image_path
  image_fallback: true
```

## Implementation Order

1. **Phase 1**: Data schema extensions (columns, loading utilities)
2. **Phase 2**: Ray Data image support (streaming datasets)
3. **Phase 3**: vLLM multimodal configuration (engine setup)
4. **Phase 4**: Multimodal preprocessing (message formatting)
5. **Phase 5**: Ray Dataset image handling (map functions)
6. **Phase 6**: Configuration updates (YAML configs)
7. **Phase 7**: Stage runner updates (orchestrator integration)
8. **Phase 8**: Error handling (fallbacks, graceful degradation)
9. **Phase 9**: Testing (unit and integration tests)
10. **Phase 10**: Documentation (user guides, examples)

## Key Design Decisions

### 1. Backward Compatibility
- Text-only classification remains unchanged
- Multimodal features are opt-in via config
- Existing pipelines continue to work without modification

### 2. Image Storage Options
- Support multiple input formats (path, URL, base64, bytes)
- Flexible column mapping via config
- Lazy loading for large datasets

### 3. Model Detection and Zoo Integration
- Automatic detection of multimodal models (checks config.json for local models)
- Model path resolution: zoo → HuggingFace Hub fallback
- Manual override via `runtime.multimodal_enabled`
- Clear error messages for unsupported models
- Support for absolute paths, relative zoo paths, and HuggingFace Hub identifiers

### 4. Performance Considerations
- Use Ray Data's `read_images()` for directory-based inputs (returns numpy arrays)
- Convert numpy arrays to PIL Images in map functions (lazy conversion)
- Parallel image loading via Ray Dataset maps
- Lazy evaluation to minimize memory usage
- Batch size tuning: multimodal models may need smaller batches due to image memory
- GPU memory: Images consume significant GPU memory; adjust `gpu_memory_utilization` if needed
- `max_num_batched_tokens`: Use 5120 for multimodal (vs 2048 for text-only)
- Image caching: Use UUID in messages for vLLM's internal caching of processed images

### 5. Error Handling
- Graceful fallback to text-only on image errors
- Configurable fallback behavior
- Detailed logging for debugging

## Supported Models

Based on vLLM documentation, the following models are supported:

- **LLaVA variants**: `llava-hf/llava-1.5-7b-hf`, `llava-hf/llava-v1.6-mistral-7b-hf`
- **Qwen VL**: `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`
- **Gemma**: `google/gemma-3-4b-it`
- **InternVL**: `OpenGVLab/InternVL3-2B`, `internlm/Intern-S1-mini`
- **Llama-4**: `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- **Others**: `moonshotai/Kimi-VL-A3B-Instruct`, `lightonai/LightOnOCR-1B`, etc.

## Configuration Example

```yaml
# dagspaces/uair/conf/pipeline/classify_multimodal.yaml
defaults:
  - /pipeline: classify
  - override /model: vllm_multimodal
  - override /data: multimodal_inputs

runtime:
  multimodal_enabled: true
  image_column: image_path
  image_fallback: true
  classification_profile: relevance

model:
  model_source: "Qwen/Qwen2.5-VL-3B-Instruct"
  engine_kwargs:
    limit_mm_per_prompt:
      image: 1
  batch_size: 16
  concurrency: 1
  has_image: true

data:
  columns:
    article_text: article_text
    article_id: article_id
    image_path: image_path
```

## Additional Implementation Details

### Image Format Handling

1. **Ray Data Image Reading**:
   - `ray.data.read_images()` returns numpy arrays in 'image' column
   - Use `include_paths=True` to get file paths for merging
   - Convert numpy arrays to PIL Images before vLLM processing
   - Images stored as numpy arrays consume more memory than paths

2. **Message Format Options**:
   - **PIL Image direct**: `{"type": "image", "image": PIL.Image.Image}` (preferred)
   - **URL format**: `{"type": "image_url", "image_url": {"url": "..."}}`
   - **Base64 format**: `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`
   - **UUID caching**: Optional `"uuid"` field for vLLM internal caching

3. **Memory Management**:
   - Multimodal models require more GPU memory
   - Consider reducing `batch_size` for large images
   - Use `gpu_memory_utilization` to control GPU memory allocation
   - Image preprocessing happens in CPU, then moved to GPU

4. **Batch Size Considerations**:
   - Default batch_size may be too large for multimodal models
   - Start with smaller batches (8-16) and increase if memory allows
   - Monitor GPU memory usage during inference
   - Consider `max_num_seqs` limitation for concurrent requests

5. **Image Preprocessing**:
   - All images converted to RGB format automatically
   - vLLM handles resizing via `mm_processor_kwargs`
   - Model-specific preprocessing handled by vLLM's multimodal processor
   - No manual resizing needed unless preprocessing before Ray Dataset

### Edge Cases and Error Handling

1. **Missing Images**:
   - Rows without images: fallback to text-only classification
   - Configurable via `runtime.image_fallback` setting
   - Track missing images in metadata for analysis

2. **Invalid Image Formats**:
   - Skip unsupported formats with warning
   - Log format errors for debugging
   - Continue processing remaining rows

3. **Large Images**:
   - vLLM handles resizing via `mm_processor_kwargs`
   - Consider pre-resizing very large images to reduce memory
   - Monitor `max_pixels` setting for memory constraints

4. **Network Failures**:
   - URL fetching timeout: 10 seconds default
   - Retry logic: configurable retry attempts
   - Fallback to text-only on persistent failures

5. **Mixed Batches**:
   - Handle rows with and without images in same batch
   - None images are filtered out in preprocessing
   - vLLM processes mixed batches correctly

## Future Enhancements

1. **Video Support**: Extend to video inputs using vLLM's video modality support
   - Use `limit_mm_per_prompt={"video": 1}` in engine_kwargs
   - Support video URL format: `{"type": "video_url", "video_url": {"url": "..."}}`
   - Configure `mm_processor_kwargs` with `fps` parameter

2. **Multiple Images**: Support multiple images per article
   - Increase `limit_mm_per_prompt={"image": N}` where N > 1
   - Add multiple image items to content array
   - Handle image ordering and context

3. **Image Preprocessing**: Built-in image augmentation and normalization
   - Resize before vLLM processing (reduce memory)
   - Normalize pixel values
   - Apply transformations (crop, rotate, etc.)

4. **OCR Integration**: Automatic text extraction from images
   - Use OCR models (e.g., LightOnOCR) to extract text
   - Merge extracted text with article_text
   - Support for text-heavy images

5. **Embedding Extraction**: Support for multimodal embeddings (E5-V, CLIP)
   - Use `task_type="embed"` in vLLMEngineProcessorConfig
   - Extract embeddings for similarity search
   - Combine with text embeddings for hybrid search

## Critical Implementation Notes

### Dependencies

Required packages:
```bash
pip install "ray[data]" "vllm>=0.7.2" pillow requests numpy
```

Optional for advanced features:
```bash
pip install pybase64  # For base64 encoding/decoding
huggingface_hub  # For downloading models to zoo (if not already installed)
```

### Model Zoo Integration

**Zoo Location**: `/share/pierson/matt/zoo/models/`

**Model Path Resolution**:
- **Absolute paths**: Used as-is if they exist
- **Zoo paths**: Relative model names are resolved to `/share/pierson/matt/zoo/models/{model_name}`
- **HuggingFace Hub**: Falls back to HuggingFace Hub if model not found in zoo

**Benefits**:
- **Faster loading**: Local models load faster than downloading from Hub
- **Offline support**: Works without internet access
- **Version control**: Specific model versions stored locally
- **Cost savings**: Avoids repeated downloads

**Model Detection**:
- For local zoo models, checks `config.json` for multimodal capabilities
- Falls back to pattern matching on model name if config unavailable
- Supports explicit override via `runtime.multimodal_enabled` config flag

### Ray Data Image Format

- `ray.data.read_images()` returns `numpy.ndarray` in 'image' column
- Must convert to PIL Image: `PIL.Image.fromarray(arr).convert("RGB")`
- Use `include_paths=True` to get file paths for merging with metadata
- Images are read lazily; conversion happens in map functions

### vLLM Configuration Requirements

- **`has_image=True`**: Required in `vLLMEngineProcessorConfig` for multimodal
- **`limit_mm_per_prompt`**: Must be in `engine_kwargs`, not top-level config
- **`max_num_batched_tokens`**: Use 5120 for multimodal (vs 2048 for text)
- **`mm_processor_kwargs`**: Model-specific settings (varies by model)
- **`runtime_env`**: Needed for HuggingFace token access

### Message Format Specifications

vLLM supports multiple image input formats:

1. **Direct PIL Image** (preferred):
   ```python
   {"type": "image", "image": PIL.Image.Image}
   ```

2. **Image URL**:
   ```python
   {"type": "image_url", "image_url": {"url": "https://..."}}
   ```

3. **Base64 encoded**:
   ```python
   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
   ```

4. **With UUID caching** (optional):
   ```python
   {"type": "image", "image": PIL.Image.Image, "uuid": "unique_id"}
   ```

### Performance Tuning

1. **Batch Size**: Start small (8-16) for multimodal, increase if memory allows
2. **Concurrency**: Adjust based on GPU memory and model size
3. **Memory**: Monitor GPU memory; reduce batch_size if OOM occurs
4. **Image Size**: Use `mm_processor_kwargs` to limit image dimensions
5. **Caching**: Use UUIDs for repeated images to leverage vLLM caching

### Troubleshooting

**Common Issues**:

1. **CUDA OOM**: Reduce `batch_size` or `max_num_seqs`
2. **Image loading fails**: Check file paths, URLs, network connectivity
3. **Wrong format**: Ensure RGB conversion (`convert("RGB")`)
4. **Model not found**: 
   - Verify `model_source` path is correct
   - Check if model exists in zoo: `ls /share/pierson/matt/zoo/models/`
   - Verify `config.json` exists in model directory
   - For HuggingFace Hub models, check `HF_TOKEN` if private model
5. **Ray Dataset errors**: Check numpy array conversion, ensure PIL Images
6. **Zoo model not detected**: Ensure model directory contains `config.json` with proper structure

**Debugging Tips**:

- Enable verbose logging: `ray.init(log_to_driver=True)`
- Check Ray Data schema: `ds.schema()` or `ds.take(1)`
- Verify image format: `print(type(image))` should be `PIL.Image.Image`
- Monitor GPU memory: `nvidia-smi` during inference
- Check vLLM logs: Look for multimodal processing messages
- Verify model path resolution: Add logging to `_resolve_model_path()` to see resolved path
- Check zoo model structure: Ensure `config.json` exists and contains model metadata
- Test model loading: Try loading model directly with `AutoTokenizer.from_pretrained()` before vLLM

## References

- [Ray Data Batch Inference Guide](https://docs.ray.io/en/latest/data/batch_inference.html)
- [Ray Data Working with Images](https://docs.ray.io/en/latest/data/working-with-images.html)
- [Ray Data LLM Integration](https://docs.ray.io/en/latest/data/working-with-llms.html)
- [vLLM Vision Language Models](https://docs.vllm.ai/en/latest/examples/offline_inference/vision_language.html)
- [vLLM Multimodal Inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs.html)
- [vLLM Configuration Guide](https://docs.vllm.ai/en/latest/configuration/conserving_memory.html)
- [Multimodal AI Examples](https://github.com/anyscale/multimodal-ai/blob/main/notebooks/01-Batch-Inference.ipynb)

