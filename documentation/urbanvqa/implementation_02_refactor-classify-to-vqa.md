

## Phase 2: Refactor Classify Stage to VQA Stage

### 2.1 Refactor classify.py to vqa.py

**Goal**: Transform `classify.py` into `vqa.py` to support Visual Question Answering workflows.

**File**: `dagspaces/urbanvqa/stages/classify.py` → `dagspaces/urbanvqa/stages/vqa.py`

**Key Changes**:
- Remove article-specific processing (`article_text`, article metadata)
- Remove keyword buffering logic
- Remove relevance filtering (prefilter_mode)
- Remove EU Act classification profiles
- Remove Risks/Benefits classification profiles
- Add support for prompt + image input
- Simplify to direct VQA flow: prompt + image → answer

**Keep**:
- Multimodal image loading (from path, URL, base64)
- Ray Data streaming support
- Batch inference infrastructure
- vLLM integration
- GPU management
- W&B logging

### 2.2 Create New VQA Stage

**New File**: `dagspaces/urbanvqa/stages/vqa.py` (refactored from `classify.py`)

**Core Functionality**:
- Accept prompt + image as input
- Support batch inference (independent prompt+image pairs)
- Return answers in consistent format
- Support structured JSON output (via guided decoding)
- Support Jinja2 template rendering

**Key Functions**:

```python
def run_vqa_stage(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Run VQA inference on a dataset.
    
    Args:
        df: DataFrame with columns: prompt, image_path/image_url/image_base64, sample_id
        cfg: Configuration object
        
    Returns:
        DataFrame with columns: sample_id, prompt, answer, model_response, metadata
    """
    # Implementation refactored from classify.py
    # - Remove article processing
    # - Remove keyword buffering
    # - Remove EU/Risks-Benefits profiles
    # - Simple prompt + image -> answer flow
    # - Support Jinja2 templates
    # - Support structured JSON output
    pass

def _pre_vqa(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """
    Preprocess a single row for VQA.
    
    Builds messages in format:
    - system: VQA system prompt (from config)
    - user: [{"type": "text", "text": prompt}, {"type": "image", "image": PIL.Image}]
    
    Supports:
    - Jinja2 template rendering if enabled
    - Dynamic prompt generation
    """
    # Load image from row
    # Render Jinja2 template if enabled
    # Build multimodal message
    # Return formatted message
    pass

def _post_vqa(row: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """
    Postprocess VQA response.
    
    Handles:
    - Structured JSON output parsing
    - Answer extraction
    - Metadata collection
    """
    # Parse JSON if structured output enabled
    # Extract answer
    # Return formatted result
    pass
```

**Differences from `classify.py`**:
- **No article_text processing**: Prompt comes directly from `prompt` column
- **No keyword buffering**: Not needed for VQA
- **No relevance filtering**: All samples are processed
- **Simplified prompt formatting**: No EU/Risks-Benefits profiles
- **Direct answer extraction**: Model response is the answer
- **Jinja2 template support**: Dynamic prompt generation with metadata variable substitution
- **Structured JSON output**: Optional guided decoding support
- **Metadata preservation**: All lightweight metadata columns preserved through preprocessing/postprocessing (Recent Enhancement - see `implementation_09_recent-enhancements.md`)

### 2.3 Update Stage Registry

**File**: `dagspaces/urbanvqa/orchestrator.py`

**Add VQA Runner** (keep existing runners):
```python
class VQARunner(StageRunner):
    stage_name = "vqa"
    
    def run(self, context: StageExecutionContext) -> StageResult:
        # Load data with prompt + image columns
        # Run VQA stage
        # Return results
        pass

# Add to existing registry (keep other runners)
_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "vqa": VQARunner(),
    # Keep existing: taxonomy, decompose, etc.
    # (separate plan will handle refactoring these)
}
```