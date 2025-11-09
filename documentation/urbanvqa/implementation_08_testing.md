## Phase 8: Testing and Validation

### 8.1 Unit Tests

**New File**: `tests/test_vqa.py`

**Test Cases**:
- Data loading with prompt + image columns
- Missing prompt column (should raise error)
- Missing image columns (should raise error)
- Sample ID generation
- Image loading from path/URL/base64
- Batch inference (independent prompt+image pairs)
- Jinja2 template rendering
- Structured JSON output parsing

### 8.2 Integration Tests

**Test Scenarios**:
- End-to-end VQA pipeline with parquet input
- Streaming mode with images
- Batch inference with independent prompt+image pairs
- Error handling for missing images
- Jinja2 template rendering
- Structured JSON output
- Large dataset processing

### 8.3 Example Datasets

**Create**: `examples/vqa_dataset.parquet`

**Schema**:
```python
{
    "prompt": ["What is in this image?", "Describe the scene.", ...],
    "image_path": ["/path/to/image1.jpg", "/path/to/image2.jpg", ...],
    "sample_id": ["id1", "id2", ...]  # Optional
}
```
