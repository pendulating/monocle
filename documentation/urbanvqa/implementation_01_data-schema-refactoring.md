## Phase 1: Data Schema Refactoring

### 1.1 Remove Article-Specific Columns

**Files to Modify**:
- `dagspaces/urbanvqa/orchestrator.py` (`_load_parquet_dataset`, `_prepare_streaming_dataset`)
- `dagspaces/urbanvqa/conf/data/inputs.yaml`
- `dagspaces/urbanvqa/conf/data/multimodal_inputs.yaml`

**Changes**:

Remove these column mappings and references:
- `article_text` → Remove (replaced by `prompt`)
- `article_id` → Replace with `sample_id` (optional identifier)
- `article_path` → Remove (not needed for VQA)
- `country` → Remove (not needed for VQA)
- `year` → Remove (not needed for VQA)
- `chunk_text` → Remove (not needed for VQA)

**New Required Columns**:
- `prompt` (required): The text question/prompt to ask about the image
- `image_path` OR `image_url` OR `image_base64` (required): At least one image source must be present
- `sample_id` (optional): Unique identifier for each sample (auto-generated if missing)

**Implementation**:

```python
# In orchestrator.py _load_parquet_dataset():
col_map = {
    columns.get("prompt", "prompt"): "prompt",
    columns.get("sample_id", "sample_id"): "sample_id",
    # Image columns (at least one required)
    columns.get("image_path", "image_path"): "image_path",
    columns.get("image_url", "image_url"): "image_url",
    columns.get("image_base64", "image_base64"): "image_base64",
}

# Validation: Ensure prompt exists
if "prompt" not in df.columns:
    raise RuntimeError("Parquet missing required column: prompt")

# Validation: Ensure at least one image column exists
image_cols = ["image_path", "image_url", "image_base64"]
if not any(col in df.columns for col in image_cols):
    raise RuntimeError(f"Parquet missing required image column. Must have one of: {image_cols}")

# Generate sample_id if missing
if "sample_id" not in df.columns:
    def _gen_sample_id(row):
        import hashlib
        # Use image path/URL/base64 + prompt to generate deterministic ID
        img_src = row.get("image_path") or row.get("image_url") or row.get("image_base64") or ""
        prompt_val = row.get("prompt", "")
        combined = f"{img_src}|{prompt_val}"
        return hashlib.sha1(combined.encode("utf-8")).hexdigest()
    df["sample_id"] = df.apply(_gen_sample_id, axis=1)
```

### 1.2 Update Data Configuration Files

**New File**: `dagspaces/urbanvqa/conf/data/vqa_inputs.yaml`

```yaml
# VQA Data Configuration
# Input format: prompt + image(s)

parquet_path: ${oc.env:DATA_ROOT,/path/to/data}/vqa_dataset.parquet

columns:
  # Required: text prompt/question
  prompt: prompt
  
  # Optional: sample identifier
  sample_id: sample_id
  
  # Required: at least one image source
  image_path: image_path      # Path to local image files
  image_url: image_url        # URL to remote images
  image_base64: image_base64  # Base64-encoded images

# Optional: Path to directory containing images (if images are in separate directory)
# If set, images will be loaded from this directory and merged with parquet data
# image_path: /path/to/images/directory
```

**Update**: `dagspaces/urbanvqa/conf/data/multimodal_inputs.yaml` → Deprecate or update to VQA format

### 1.3 Directory-Based Image Loading (Recent Enhancement)

**Status**: ✅ Implemented (see `implementation_09_recent-enhancements.md`)

The pipeline now supports reading images directly from a directory without requiring a parquet file:

```yaml
# dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml
parquet_path: ""  # Empty - images read directly from directory
image_path: /path/to/images/directory
default_prompt: "What do you see in this image?"
```

When `data.image_path` points to a directory and `parquet_path` is empty, the pipeline:
1. Uses `ray.data.read_images()` to load images as NumPy arrays
2. Automatically generates `sample_id` from filenames
3. Adds `prompt` from `data.default_prompt`
4. Optionally joins metadata from parquet if `parquet_path` is provided

### 1.4 Per-Image Metadata Support (Recent Enhancement)

**Status**: ✅ Implemented (see `implementation_09_recent-enhancements.md`)

Support for loading per-image metadata from parquet files and joining with directory-based images:

```yaml
# dagspaces/urbanvqa/conf/data/nexar_dashcam.yaml
parquet_path: /path/to/metadata.parquet  # Contains: image_path, location, timestamp, camera_id
image_path: /path/to/images/directory
default_prompt: "What do you see?"

# Optional: explicitly specify metadata columns
metadata_columns:
  - location
  - timestamp
  - camera_id
```

Metadata columns are:
- Automatically detected (all non-standard columns) or explicitly specified
- Joined with images on `image_path` or `sample_id`
- Available in Jinja2 templates via `{{column_name}}`
- Preserved through all preprocessing/postprocessing stages

---
