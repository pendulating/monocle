## Phase 3: Orchestrator Refactoring

### 3.1 Remove Article-Specific Processing

**File**: `dagspaces/urbanvqa/orchestrator.py`

**Changes**:

1. **Update `_load_parquet_dataset()`**:
   - Remove article_text, article_path, country, year, article_id checks
   - Require prompt column
   - Require at least one image column
   - Generate sample_id if missing

2. **Update `_prepare_streaming_dataset()`**:
   - Update column mapping for VQA schema
   - Remove article-specific preprocessing
   - **Recent Enhancement**: Always use `ray.data.read_images()` for directory-based images
   - **Recent Enhancement**: Support loading metadata from parquet and joining with directory-based images

3. **Update `prepare_stage_input()`**:
   - Remove article-specific validation
   - Update streaming compatibility check for VQA stage (other stages may have different requirements)

### 3.2 Remove Unnecessary Features

**Remove** (from classify.py/vqa.py):
- Keyword buffering logic (not needed for VQA)
- Relevance filtering (prefilter_mode)
- EU Act classification support
- Risks/Benefits classification support
- Article text processing

**Keep** (in classify.py/vqa.py):
- Multimodal image loading
- Ray Data streaming support
- Batch inference
- GPU management
- W&B logging

### 3.3 VQARunner Updates (Recent Enhancement)

**Status**: ✅ Implemented (see `implementation_09_recent-enhancements.md`)

**File**: `dagspaces/urbanvqa/orchestrator.py` (`VQARunner`)

**Key Changes**:
- Made `dataset` input optional - VQA can work with just `data.image_path` pointing to a directory
- If no `dataset` input provided, checks for `data.image_path` configuration
- Supports both directory-based images and parquet-based images/metadata

**Note**: Other stages (taxonomy, decompose, topic, synthesis, verify) remain unchanged for now. A separate plan will handle their refactoring.
