
## Phase 7: Code Refactoring and Cleanup

### 7.1 Refactor classify.py to vqa.py

**File**: `dagspaces/urbanvqa/stages/classify.py` → `dagspaces/urbanvqa/stages/vqa.py`

**Changes**:
- Remove article-specific processing (`article_text`, article metadata)
- Remove keyword buffering logic
- Remove relevance filtering (prefilter_mode)
- Remove EU Act classification profiles
- Remove Risks/Benefits classification profiles
- Add support for prompt + image input
- Add Jinja2 template rendering support
- Add structured JSON output support (via guided decoding)
- Simplify to direct VQA flow: prompt + image → answer

**Keep**:
- Multimodal image loading infrastructure
- Ray Data streaming support
- Batch inference infrastructure
- vLLM integration
- GPU management
- W&B logging

**Note**: Other stage files (`taxonomy.py`, `decompose.py`, `topic.py`, etc.) remain unchanged. A separate plan will handle their refactoring.

### 7.2 Update Prompt Configurations

**Create**:
- `dagspaces/urbanvqa/conf/prompt/vqa.yaml` (new VQA prompt config)

**Update**:
- `dagspaces/urbanvqa/conf/prompt/classify.yaml` → Deprecate or update to VQA format

**Note**: Other prompt configurations remain unchanged for now.

### 7.3 Update Pipeline Configurations

**Create/Update**:
- `dagspaces/urbanvqa/conf/pipeline/vqa.yaml` → New VQA pipeline config
- `dagspaces/urbanvqa/conf/pipeline/classify_multimodal.yaml` → Update to VQA format OR deprecate

**Note**: Other pipeline configurations remain unchanged for now. A separate plan will handle taxonomy, decompose, and other stage pipelines.

### 7.4 Update Support Files

**Note**: Support files for other stages (taxonomy, topic, synthesis, etc.) remain unchanged for now. A separate plan will determine if these need updates for VQA compatibility.
