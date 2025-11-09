# Implementation Plan: Ray Data + vLLM Image Handling Refactoring

## Executive Summary

This document outlines a systematic refactoring plan to properly integrate Ray Data image loading with vLLM multimodal inference, eliminating PIL Image serialization errors and ensuring optimal batch processing.

## Research Findings

### Key Insights from Documentation

1. **Ray Data LLM Processor Preprocess Function:**
   - MUST return ONLY `messages` and `sampling_params` (plus lightweight metadata)
   - PIL Images are handled specially when inside `messages` structure
   - Ray Data LLM processor bypasses Arrow serialization for PIL Images in messages
   - Should NEVER return PIL Images as separate row columns

2. **Ray Data Image Loading:**
   - `ray.data.read_images()` returns numpy arrays in `'image'` column
   - Numpy arrays are PyArrow-serializable (no conversion needed)
   - Use `include_paths=True` to get file paths for metadata joining
   - Images remain as numpy arrays throughout dataset rows

3. **vLLM Image Input Requirements:**
   - Requires PIL Images in messages structure: `{"type": "image", "image": PIL_Image}`
   - Alternative formats: `{"type": "image_pil", "image_pil": PIL_Image}` or `{"type": "image_url", "image_url": {"url": "..."}}`
   - Does NOT accept numpy arrays directly for images (only for videos)

4. **Critical Pattern:**
   - Keep numpy arrays in dataset rows (PyArrow-serializable)
   - Convert numpy → PIL ONLY when building messages structure
   - PIL Images exist ONLY inside messages, never in dataset rows

## Current Problems

1. **PIL Images Leaking into Dataset Rows:**
   - PIL Images are being returned as columns from preprocessing functions
   - Causes `ArrowConversionError` when Ray Data tries to serialize rows
   - Error: `Could not convert <PIL.Image.Image ...> with type Image`

2. **Incorrect Preprocess Return Format:**
   - Some preprocessing functions return image columns alongside messages
   - Should return ONLY `messages`, `sampling_params`, and lightweight metadata

3. **Premature PIL Conversion:**
   - Images converted to PIL too early in pipeline
   - Should remain as numpy arrays until messages construction

## Implementation Plan

### Phase 1: Image Loading Refactoring

#### 1.1 Refactor `_prepare_streaming_dataset` in `orchestrator.py`

**Current State:**
- Uses `ray.data.read_images()` which returns numpy arrays
- Adds metadata via `.map(_add_vqa_metadata)`
- May convert numpy to PIL prematurely

**Target State:**
- Keep images as numpy arrays throughout dataset
- Only store image paths/URLs/base64 as strings in rows
- No PIL conversion at dataset level

**Changes:**
```python
# In _prepare_streaming_dataset:
# 1. Use ray.data.read_images() with include_paths=True
ds = ray.data.read_images(image_dir, include_paths=True)

# 2. Add metadata WITHOUT converting images
def _add_vqa_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    # Keep 'image' as numpy array (PyArrow-serializable)
    # Add image_path, prompt, sample_id as strings
    return {
        **row,  # Includes 'image' as numpy array
        "image_path": os.path.abspath(row["path"]),
        "prompt": default_prompt,
        "sample_id": generate_sample_id(row["path"]),
    }

# 3. Join with metadata parquet if needed (images stay as numpy)
```

**Files to Modify:**
- `dagspaces/urbanvqa/orchestrator.py` (lines ~700-790)

---

### Phase 2: Preprocessing Function Refactoring

#### 2.1 Refactor `preprocess_simple` in `unified.py`

**Current State:**
- Converts numpy arrays to PIL Images
- Returns PIL Images in messages structure (correct)
- May leak PIL Images back into row

**Target State:**
- Convert numpy → PIL ONLY when building messages
- Return ONLY `messages` and `sampling_params`
- Never include image columns in return

**Changes:**
```python
def preprocess_simple(row: Dict[str, Any], cfg: DictConfig, is_multimodal: bool = True) -> Dict[str, Any]:
    """Preprocess row - returns ONLY messages and sampling_params."""
    
    prompt = str(row.get("prompt", "")).strip()
    
    # Load image - convert numpy to PIL ONLY here
    image = None
    if is_multimodal:
        if "image" in row and row["image"] is not None:
            img_val = row["image"]
            if isinstance(img_val, np.ndarray):
                # Convert numpy → PIL ONLY for messages
                image = _normalize_image(img_val)
            elif hasattr(img_val, "convert"):  # Already PIL
                image = img_val
        elif "image_path" in row:
            image = _load_image_from_path(row["image_path"])
        # ... handle URLs, base64
    
    # Build messages with PIL Image
    if is_multimodal and image is not None:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image", "image": image}  # PIL Image in messages
        ]
    else:
        user_content = prompt
    
    # CRITICAL: Return ONLY messages and sampling_params
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "sampling_params": sampling_params,
        # Optional: lightweight metadata (sample_id, prompt as strings)
        "sample_id": row.get("sample_id"),
        "prompt": prompt,
    }
```

**Files to Modify:**
- `dagspaces/urbanvqa/prompts/unified.py` (lines ~17-132)

---

#### 2.2 Refactor `unified_preprocess` in `unified.py`

**Current State:**
- May spread image columns back into result
- Filters image columns but may miss edge cases

**Target State:**
- Strictly filter out ALL image columns when spreading metadata
- Ensure preprocess_simple receives lightweight row only

**Changes:**
```python
def unified_preprocess(...) -> Dict[str, Any]:
    # ... existing logic ...
    
    # When returning from CoT/ReAct:
    if cot_enabled:
        result = preprocess_cot(current_row, cfg)
        if result and "messages" in result:
            # CRITICAL: Filter out ALL image-related columns
            excluded_cols = {
                "messages", "sampling_params",
                "image", "image_array", "image_data", "path",
                # Add any other image-related columns
            }
            lightweight_metadata = {
                k: v for k, v in current_row.items()
                if k not in excluded_cols
                and isinstance(v, (str, int, float, type(None)))
            }
            return {
                **lightweight_metadata,
                **result,  # Contains messages and sampling_params
            }
    
    # When calling preprocess_simple:
    lightweight_row = {
        k: v for k, v in current_row.items()
        if k not in {"image", "image_array", "image_data", "path"}
        and isinstance(v, (str, int, float, type(None)))
    }
    lightweight_row["prompt"] = prompt
    # Include image_path/image_url/image_base64 as strings
    for col in ["image_path", "image_url", "image_base64"]:
        if col in current_row and isinstance(current_row[col], str):
            lightweight_row[col] = current_row[col]
    
    return preprocess_simple(lightweight_row, cfg, is_multimodal)
```

**Files to Modify:**
- `dagspaces/urbanvqa/prompts/unified.py` (lines ~135-322)

---

#### 2.3 Refactor `_pre` in `vqa.py`

**Current State:**
- Calls `unified_preprocess`
- May include image columns in return

**Target State:**
- Ensure return contains ONLY messages, sampling_params, lightweight metadata
- Explicitly filter out any image columns

**Changes:**
```python
def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess row - returns ONLY messages and sampling_params."""
    
    unified_result = unified_preprocess(row, cfg, is_multimodal, ...)
    
    if unified_result and "messages" in unified_result:
        # CRITICAL: Return ONLY what Ray Data LLM processor expects
        result = {
            "messages": unified_result["messages"],
            "sampling_params": unified_result["sampling_params"],
        }
        
        # Only include lightweight, serializable metadata
        for key in ["sample_id", "prompt"]:
            if key in row and isinstance(row[key], (str, int, float, type(None))):
                result[key] = row[key]
        
        result["ts_start"] = datetime.now().timestamp()
        return result
    
    # Fallback
    return preprocess_simple(row, cfg, is_multimodal)
```

**Files to Modify:**
- `dagspaces/urbanvqa/stages/vqa.py` (lines ~354-408)

---

### Phase 3: Postprocessing Refactoring

#### 3.1 Ensure Postprocessing Filters Image Columns

**Current State:**
- Already filters large columns, but verify completeness

**Target State:**
- Explicitly filter ALL image-related columns
- Ensure no PIL Images leak into output

**Changes:**
```python
def _post(row: Dict[str, Any]) -> Dict[str, Any]:
    """Postprocess - filter out all image columns."""
    
    excluded_cols = {
        "image", "image_array", "image_data", "path",  # Image data
        "messages", "sampling_params",  # Internal processing
        "llm_output", "generated_text",  # Already extracted
        # ... other excluded columns
    }
    
    result = {}
    for col, val in row.items():
        if col in excluded_cols:
            continue
        # Only include serializable types
        if isinstance(val, (str, int, float, type(None))):
            result[col] = val
        # ... handle dicts/lists
    
    return result
```

**Files to Modify:**
- `dagspaces/urbanvqa/stages/vqa.py` (lines ~410-494)
- All other postprocessing functions (`_post_hierarchical`, `_post_parallel`, `_post_tree_node`, etc.)

---

### Phase 4: Image Loading Utilities Refactoring

#### 4.1 Ensure `_load_image_from_row` Handles Numpy Arrays

**Current State:**
- May convert numpy to PIL immediately

**Target State:**
- Accept numpy arrays and convert to PIL only when needed
- Support image_path, image_url, image_base64

**Changes:**
```python
def _load_image_from_row(row: Dict[str, Any]) -> Optional[PIL.Image.Image]:
    """Load image from row - converts numpy to PIL if needed."""
    
    # Priority: image_path > image_url > image_base64 > image (numpy)
    
    if "image_path" in row and row["image_path"]:
        return _load_image_from_path(row["image_path"])
    
    if "image_url" in row and row["image_url"]:
        return _load_image_from_path(row["image_url"])  # Handles URLs
    
    if "image_base64" in row and row["image_base64"]:
        return _load_image_from_base64(row["image_base64"])
    
    # Last resort: convert numpy array to PIL
    if "image" in row and row["image"] is not None:
        return _normalize_image(row["image"])
    
    return None
```

**Files to Modify:**
- `dagspaces/urbanvqa/stages/classify.py` (lines ~180-218)

---

### Phase 5: Testing and Validation

#### 5.1 Unit Tests

**Test Cases:**
1. Test that `ray.data.read_images()` returns numpy arrays
2. Test that `preprocess_simple` converts numpy → PIL only in messages
3. Test that preprocess returns ONLY messages and sampling_params
4. Test that PIL Images never appear in dataset rows
5. Test metadata preservation through pipeline

**Files to Create:**
- `tests/test_image_loading.py`
- `tests/test_preprocessing.py`
- `tests/test_ray_data_integration.py`

---

#### 5.2 Integration Tests

**Test Cases:**
1. End-to-end pipeline with directory-based images
2. End-to-end pipeline with parquet metadata
3. Verify no ArrowConversionError occurs
4. Verify PIL Images only in messages structure

---

### Phase 6: Documentation Updates

#### 6.1 Update Implementation Docs

**Files to Update:**
- `documentation/urbanvqa/implementation_01_data-schema-refactoring.md`
- `documentation/urbanvqa/implementation_02_refactor-classify-to-vqa.md`
- `documentation/urbanvqa/implementation_09_recent-enhancements.md`

**Add Sections:**
- Ray Data Image Loading Best Practices
- Preprocess Function Return Format Requirements
- PIL Image Handling Guidelines

---

## Implementation Checklist

### Phase 1: Image Loading
- [ ] Refactor `_prepare_streaming_dataset` to keep numpy arrays
- [ ] Update `_add_vqa_metadata` to preserve numpy arrays
- [ ] Test image loading from directory
- [ ] Test metadata joining with parquet

### Phase 2: Preprocessing
- [ ] Refactor `preprocess_simple` to return only messages/sampling_params
- [ ] Refactor `unified_preprocess` to filter image columns strictly
- [ ] Refactor `_pre` to ensure correct return format
- [ ] Update all preprocessing functions (hierarchical, parallel, tree)

### Phase 3: Postprocessing
- [ ] Verify `_post` filters all image columns
- [ ] Update all postprocessing functions
- [ ] Test metadata preservation

### Phase 4: Utilities
- [ ] Update `_load_image_from_row` to handle numpy arrays
- [ ] Ensure `_normalize_image` handles numpy correctly
- [ ] Test image loading utilities

### Phase 5: Testing
- [ ] Write unit tests for image loading
- [ ] Write unit tests for preprocessing
- [ ] Write integration tests
- [ ] Run full pipeline test

### Phase 6: Documentation
- [ ] Update implementation docs
- [ ] Add best practices guide
- [ ] Update code comments

---

## Critical Code Patterns

### Pattern 1: Preprocess Function Return Format

```python
def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
    """MUST return ONLY messages and sampling_params."""
    
    # Convert numpy → PIL ONLY here
    image = _load_and_convert_image(row)
    
    return {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}  # PIL Image
            ]}
        ],
        "sampling_params": {...},
        # Optional lightweight metadata (strings/numbers only)
        "sample_id": row.get("sample_id"),
    }
    # NEVER return image columns or PIL Images as separate fields
```

### Pattern 2: Filtering Image Columns

```python
# When spreading metadata back:
excluded_cols = {
    "image", "image_array", "image_data", "path",
    "messages", "sampling_params",
}
lightweight_metadata = {
    k: v for k, v in row.items()
    if k not in excluded_cols
    and isinstance(v, (str, int, float, type(None)))
}
```

### Pattern 3: Image Loading Flow

```python
# 1. Load images as numpy arrays (Ray Data)
ds = ray.data.read_images(image_dir, include_paths=True)
# ds has 'image' column as numpy arrays (PyArrow-serializable)

# 2. Add metadata (keep numpy arrays)
ds = ds.map(_add_vqa_metadata)
# Still numpy arrays in 'image' column

# 3. Preprocess (convert numpy → PIL only in messages)
def _pre(row):
    image_pil = _normalize_image(row["image"])  # numpy → PIL
    return {
        "messages": [..., {"type": "image", "image": image_pil}],
        "sampling_params": {...},
    }
    # PIL Image ONLY in messages, NOT in row

# 4. Ray Data LLM processor handles PIL in messages specially
# 5. Postprocess filters out all image columns
```

---

## Expected Outcomes

1. **Eliminate ArrowConversionError:**
   - No PIL Images in dataset rows
   - Only numpy arrays (PyArrow-serializable) in rows
   - PIL Images only in messages structure

2. **Proper Batch Processing:**
   - Ray Data can serialize all rows
   - vLLM receives PIL Images in correct format
   - Efficient batching and GPU utilization

3. **Metadata Preservation:**
   - All lightweight metadata preserved
   - Image paths/URLs preserved as strings
   - Custom metadata from parquet preserved

4. **Performance:**
   - No unnecessary conversions
   - Efficient memory usage
   - Optimal GPU utilization

---

## Risk Mitigation

1. **Backward Compatibility:**
   - Maintain support for image_path/image_url/image_base64
   - Ensure existing configs still work

2. **Error Handling:**
   - Graceful fallback if image loading fails
   - Clear error messages for debugging

3. **Testing:**
   - Comprehensive unit tests
   - Integration tests with real data
   - Performance benchmarking

---

## Timeline Estimate

- **Phase 1:** 2-3 hours
- **Phase 2:** 4-5 hours
- **Phase 3:** 2-3 hours
- **Phase 4:** 1-2 hours
- **Phase 5:** 3-4 hours
- **Phase 6:** 1-2 hours

**Total:** ~13-19 hours

---

## References

- Ray Data LLM Integration: https://docs.ray.io/en/latest/data/working-with-llms.html
- vLLM Multimodal Inputs: https://docs.vllm.ai/en/latest/features/multimodal_inputs.html
- Ray Data Batch Inference: https://docs.vllm.ai/en/latest/examples/offline_inference/batch_llm_inference.html

