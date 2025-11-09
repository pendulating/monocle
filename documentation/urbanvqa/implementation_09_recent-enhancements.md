# Phase 9: Recent Enhancements and Fixes

## Overview

This document captures recent enhancements and critical fixes made to the VQA pipeline after the initial refactoring phases. These changes improve flexibility, fix serialization issues, and add metadata support.

---

## 9.1 Directory-Based Image Loading (No Parquet Required)

### Problem
Previously, the pipeline required a parquet file even when images were stored in a directory. This was redundant when images could be read directly.

### Solution
Refactored `_prepare_streaming_dataset()` in `orchestrator.py` to **always** use `ray.data.read_images()` when `data.image_path` points to a directory, eliminating the need for a parquet file in this use case.

### Implementation

**File**: `dagspaces/urbanvqa/orchestrator.py`

**Key Changes**:
1. **Priority Logic**: If `data.image_path` points to an existing directory AND `parquet_path` is empty/None or doesn't exist, **always** use `ray.data.read_images()`.
2. **Metadata Addition**: Automatically adds `prompt` (from `data.default_prompt`) and `sample_id` (generated from filename) to image dataset.
3. **Parquet Reserved For**: URLs, base64-encoded images, or per-image custom metadata.

**Code Flow**:
```python
# In _prepare_streaming_dataset():
if image_path_config and os.path.isdir(image_path_config):
    # ALWAYS use ray.data.read_images() for directory-based images
    ds = ray.data.read_images(image_path_config, include_paths=True)
    
    # Add prompt and sample_id columns from config
    def _add_vqa_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize path column to image_path
        row["image_path"] = os.path.abspath(row["path"])
        
        # Generate sample_id from filename
        filename = os.path.basename(row["path"])
        sample_id = os.path.splitext(filename)[0]
        sample_id = re.sub(r'[^a-zA-Z0-9_]', '_', sample_id)
        row["sample_id"] = sample_id
        
        # Add prompt from config
        row["prompt"] = default_prompt or "What do you see in this image?"
        
        return row
    
    ds = ds.map(_add_vqa_metadata)
    return ds, True
```

**Configuration Example**:
```yaml
# dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml
parquet_path: ""  # Empty - images will be read directly from directory
image_path: /share/ju/nexar_data/2023/2023-08-20/604222321527357439/frames
default_prompt: "What urban planning features are visible in this dashcam image?"
```

**Benefits**:
- No need to create parquet files for simple directory-based datasets
- Faster startup (no parquet reading overhead)
- Automatic prompt and sample_id generation
- Images loaded as NumPy arrays (PyArrow-serializable)

---

## 9.2 Per-Image Metadata Support from Parquet Files

### Problem
When images are stored in a directory, there was no way to attach per-image metadata (e.g., location, timestamp, camera_id) for use in Jinja2 templates or downstream processing.

### Solution
Added support for loading metadata columns from parquet files and joining them with images read from directories. Metadata columns are automatically available in Jinja2 templates and preserved through the pipeline.

### Implementation

**File**: `dagspaces/urbanvqa/orchestrator.py` (`_prepare_streaming_dataset`)

**Key Features**:
1. **Metadata Detection**: Automatically detects non-standard columns in parquet (excludes `prompt`, `sample_id`, `image_path`, `image_url`, `image_base64`, `image`).
2. **Join Logic**: Joins metadata with images on `image_path` or `sample_id`.
3. **Explicit Specification**: Supports `data.metadata_columns` config to explicitly specify which columns to load.
4. **Per-Image Prompts**: If parquet contains a `prompt` column, it overrides the default prompt from config.

**Code Flow**:
```python
# After reading images from directory:
if dataset_path and os.path.exists(dataset_path):
    # Inspect parquet schema to identify metadata columns
    parquet_columns = [field.name for field in schema]
    
    # Identify metadata columns (exclude standard VQA columns)
    standard_cols = {"prompt", "sample_id", "image_path", "image_url", "image_base64", "image"}
    if metadata_columns:
        # Use explicitly specified metadata columns
        metadata_cols_to_read = [col for col in metadata_columns if col in parquet_columns]
    else:
        # Auto-detect: all columns except standard VQA columns
        metadata_cols_to_read = [col for col in parquet_columns if col not in standard_cols]
    
    # Load metadata and join with images
    if metadata_cols_to_read:
        metadata_ds = ray.data.read_parquet(dataset_path, columns=metadata_cols_to_read)
        ds = ds.join(metadata_ds, on="image_path", how="left")  # or "sample_id"
```

**Configuration Example**:
```yaml
# dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml
parquet_path: /path/to/metadata.parquet  # Contains: image_path, location, timestamp, camera_id
image_path: /share/ju/nexar_data/2023/2023-08-20/604222321527357439/frames
default_prompt: "What do you see?"

# Optional: explicitly specify metadata columns
metadata_columns:
  - location
  - timestamp
  - camera_id
  - weather_conditions
```

**Jinja2 Template Usage**:
```yaml
# dagspaces/urbanvqa/conf/prompt/vqa.yaml
template: |
  Location: {{location}}
  Camera: {{camera_id}}
  Timestamp: {{timestamp}}
  Question: {{prompt}}
```

**Metadata Preservation**:
- Metadata columns are preserved through all preprocessing/postprocessing stages
- Available in Jinja2 templates via `{{column_name}}`
- Included in final output parquet
- Filtered to exclude only large/complex objects (images, arrays, PIL objects)

---

## 9.3 Fixed ArrowConversionError (PIL Image Serialization)

### Problem
PIL Images were leaking into dataset rows, causing `ArrowConversionError: Error converting data to Arrow` when Ray Data tried to serialize them. The error showed `'image': [<PIL.Image.Image ...>]` wrapped in a list.

### Root Cause
In `unified_preprocess()` in `dagspaces/urbanvqa/prompts/unified.py`, when CoT/ReAct techniques returned messages, the code was spreading **all** columns from `current_row` back into the result, including the `image` column (numpy array from `ray.data.read_images()`). This numpy array was then being converted to PIL Images, which leaked into the dataset rows.

### Solution
Filtered out image columns (`image`, `image_array`, `image_data`, `path`) when spreading metadata back in CoT/ReAct paths and when calling `preprocess_simple()`.

**File**: `dagspaces/urbanvqa/prompts/unified.py`

**Key Changes**:

1. **CoT/ReAct Paths** (lines 218-263):
```python
if cot_enabled:
    result = preprocess_cot(current_row, cfg)
    if result and "messages" in result:
        # CRITICAL: Only spread lightweight metadata, NOT image columns
        excluded_cols = {"messages", "sampling_params", "image", "image_array", "image_data", "path"}
        lightweight_metadata = {}
        for k, v in current_row.items():
            if k in excluded_cols:
                continue
            # Only include simple, serializable types
            if isinstance(v, (str, int, float, type(None))):
                lightweight_metadata[k] = v
            # ... (dict/list filtering)
        return {
            **lightweight_metadata,
            **result,
        }
```

2. **Fallback to preprocess_simple** (lines 276-301):
```python
# CRITICAL: Only pass lightweight metadata to preprocess_simple, NOT image columns
lightweight_row = {}
excluded_cols = {"image", "image_array", "image_data", "path", "messages", "sampling_params"}
for k, v in current_row.items():
    if k in excluded_cols:
        continue
    # Only include simple, serializable types
    # ... (filtering logic)
    # For image_path/image_url/image_base64, keep them as strings (paths/URLs)
    elif k in {"image_path", "image_url", "image_base64"} and isinstance(v, str):
        lightweight_row[k] = v

return preprocess_simple(lightweight_row, cfg, is_multimodal)
```

**Best Practice Maintained**:
- Images remain as NumPy arrays in dataset rows (PyArrow-serializable)
- Conversion to PIL Images happens **only** when building `messages` structure for vLLM
- PIL Images are **only** inside `messages`, where Ray Data LLM handles them specially
- `preprocess_simple()` can still load images via `_load_image_from_row()`, which checks `image_path`/`image_url`/`image_base64` strings

**Impact**:
- ✅ Eliminates `ArrowConversionError`
- ✅ Maintains functionality (images still loaded correctly)
- ✅ Preserves metadata columns (filtered appropriately)
- ✅ Follows Ray Data best practices

---

## 9.4 Made Dataset Input Optional in VQARunner

### Problem
The `VQARunner` required a `dataset` input even when images were being read directly from a directory, causing errors when the parquet file was removed from pipeline configs.

### Solution
Made the `dataset` input optional in `VQARunner`. If no `dataset` input is provided, the runner checks for `data.image_path` pointing to a directory.

**File**: `dagspaces/urbanvqa/orchestrator.py` (`VQARunner.run()`)

**Key Changes**:
```python
class VQARunner(StageRunner):
    stage_name = "vqa"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        
        # Dataset input is optional - VQA can read images directly from directory
        dataset_path = context.inputs.get("dataset")
        if dataset_path:
            # Update parquet_path if dataset input is provided
            OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        else:
            # No dataset input - check if we have image_path configured
            image_path_config = getattr(cfg.data, "image_path", None)
            if not image_path_config or not isinstance(image_path_config, str) or not image_path_config.strip():
                raise ValueError(
                    f"Node '{context.node.key}' requires either 'dataset' input or 'data.image_path' "
                    f"pointing to an image directory"
                )
            # Set parquet_path to empty string to indicate directory-based images
            OmegaConf.update(cfg, "data.parquet_path", "", merge=True)
        
        # Load data (handles both parquet and directory-based images)
        parquet_path = getattr(cfg.data, "parquet_path", None) or dataset_path or ""
        df, ds, use_streaming = prepare_stage_input(cfg, parquet_path, self.stage_name)
        # ... rest of processing
```

**Pipeline Configuration**:
```yaml
# dagspaces/urbanvqa/conf/pipeline/vqa_nexar.yaml
pipeline:
  graph:
    nodes:
      vqa:
        stage: vqa
        depends_on: []  # No dataset input required!
        outputs:
          results: outputs/vqa/results.parquet
```

**Use Cases**:
1. **Directory-based images** (no parquet): Set `data.image_path` to directory, no `dataset` input needed
2. **Parquet with metadata**: Provide `dataset` input pointing to parquet file with metadata columns
3. **URLs/base64**: Provide `dataset` input pointing to parquet file with `image_url` or `image_base64` columns

---

## 9.5 Metadata Column Preservation Through Pipeline

### Problem
Metadata columns loaded from parquet needed to be preserved through preprocessing/postprocessing stages and available in Jinja2 templates.

### Solution
Updated all preprocessing and postprocessing functions to preserve lightweight metadata columns while filtering out large/complex objects.

**Files Modified**:
- `dagspaces/urbanvqa/stages/vqa.py`:
  - `_pre_hierarchical()`: Preserves all lightweight metadata
  - `_pre_parallel()`: Preserves all lightweight metadata
  - `_pre_tree_node()`: Preserves all lightweight metadata
  - `_post()`: Preserves all lightweight metadata
  - `_post_parallel()`: Preserves all lightweight metadata
  - `_final_post()`: Preserves all lightweight metadata
  - `_final_post_tree()`: Preserves all lightweight metadata

**Filtering Logic**:
```python
# Preserve all lightweight, serializable metadata columns
# Exclude image arrays, PIL Images, and complex objects
excluded_cols = {"image", "image_array", "image_data", "messages", "sampling_params", "path"}
lightweight_metadata = {}
for key, val in row.items():
    if key in excluded_cols:
        continue
    # Only include if it's a simple type
    if isinstance(val, (str, int, float, type(None))):
        lightweight_metadata[key] = val
    elif isinstance(val, dict):
        # Only include if dict contains only simple types
        if all(isinstance(v, (str, int, float, type(None))) for v in val.values()):
            lightweight_metadata[key] = val
    elif isinstance(val, list):
        # Only include if list contains only simple types
        if all(isinstance(v, (str, int, float, type(None))) for v in val):
            lightweight_metadata[key] = val
```

**Jinja2 Template Access**:
Metadata columns are automatically available in Jinja2 templates via `**current_row` spread in `unified_preprocess()`:
```python
context = {
    "prompt": prompt,
    "user_question": prompt,
    **current_row  # Include all row data as template variables
}
```

---

## 9.6 Summary of Changes

### Files Modified

1. **`dagspaces/urbanvqa/orchestrator.py`**:
   - `_prepare_streaming_dataset()`: Always use `ray.data.read_images()` for directory-based images
   - `_prepare_streaming_dataset()`: Added metadata loading and joining logic
   - `VQARunner.run()`: Made `dataset` input optional

2. **`dagspaces/urbanvqa/prompts/unified.py`**:
   - `unified_preprocess()`: Filter out image columns when spreading metadata back
   - `unified_preprocess()`: Filter out image columns when calling `preprocess_simple()`

3. **`dagspaces/urbanvqa/stages/vqa.py`**:
   - All preprocessing functions: Preserve lightweight metadata columns
   - All postprocessing functions: Preserve lightweight metadata columns

4. **`dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml`**:
   - Updated documentation for metadata support
   - Added `metadata_columns` option

### Configuration Changes

**New Config Option**:
```yaml
# Optional: Explicitly specify metadata columns to load from parquet
metadata_columns:
  - location
  - timestamp
  - camera_id
```

**Pipeline Config**:
- `dataset` input is now optional in pipeline node definitions
- Can rely solely on `data.image_path` for directory-based images

### Benefits

1. **Flexibility**: Support both directory-based images and parquet-based images/metadata
2. **Performance**: No need to create parquet files for simple directory-based datasets
3. **Metadata Support**: Per-image metadata available in Jinja2 templates and preserved through pipeline
4. **Stability**: Fixed critical `ArrowConversionError` preventing pipeline execution
5. **Simplicity**: Cleaner pipeline configs without redundant parquet files

---

## 9.7 Testing Recommendations

### Test Case 1: Directory-Based Images (No Parquet)
```yaml
data:
  image_path: /path/to/images/
  default_prompt: "What do you see?"
  parquet_path: ""
```
**Expected**: Images loaded directly, prompt and sample_id auto-generated.

### Test Case 2: Directory + Metadata Parquet
```yaml
data:
  image_path: /path/to/images/
  parquet_path: /path/to/metadata.parquet
  default_prompt: "What do you see?"
  metadata_columns:
    - location
    - timestamp
```
**Expected**: Images loaded from directory, metadata joined from parquet, available in templates.

### Test Case 3: Parquet with URLs/Base64
```yaml
data:
  parquet_path: /path/to/urls.parquet
  # parquet contains: prompt, image_url, sample_id
```
**Expected**: Images loaded from URLs/base64 in parquet.

### Test Case 4: Jinja2 Template with Metadata
```yaml
prompt:
  template: |
    Location: {{location}}
    Camera: {{camera_id}}
    Question: {{prompt}}
```
**Expected**: Template renders with metadata values from parquet.

---

## 9.8 Migration Guide

### From Old Config (Required Parquet)
```yaml
# OLD
pipeline:
  graph:
    nodes:
      vqa:
        inputs:
          dataset: /path/to/dataset.parquet
```

### To New Config (Directory-Based)
```yaml
# NEW
data:
  image_path: /path/to/images/
  default_prompt: "What do you see?"
  parquet_path: ""

pipeline:
  graph:
    nodes:
      vqa:
        # No inputs needed!
        outputs:
          results: outputs/vqa/results.parquet
```

### To New Config (Directory + Metadata)
```yaml
# NEW
data:
  image_path: /path/to/images/
  parquet_path: /path/to/metadata.parquet
  default_prompt: "What do you see?"
  metadata_columns:
    - location
    - timestamp
```

---

## 9.9 Known Limitations

1. **Metadata Join Key**: Currently joins on `image_path` or `sample_id`. Both must match exactly (normalized paths).
2. **Large Metadata**: Very large metadata columns (e.g., >100MB) may cause memory issues.
3. **Complex Types**: Only simple types (str, int, float, None) and simple dicts/lists are preserved. Complex nested structures are filtered out.

---

## 9.10 Future Enhancements

1. **Flexible Join Keys**: Support custom join key specification in config
2. **Metadata Validation**: Validate metadata schema against expected columns
3. **Metadata Caching**: Cache metadata lookups for performance
4. **Multi-Image Metadata**: Support metadata for multi-image prompts (future feature)

