---
title: "Guide — Browser-Based Image Search"
category: guide
created: 2026-04-07
updated: 2026-04-07
status: mvp-implemented
tags:
  - guide
  - retrieval
  - browser
  - embedding
  - search
  - urbanembed
---

# Guide — Browser-Based Image Search

Design plan for a pure client-side image search UI over pre-computed Qwen3-VL-Embedding vectors. No server required — all inference and search runs in the browser.

## Problem

The full rerank pipeline (Phase 1 embedding retrieval + Phase 2 cross-encoder reranking) takes minutes on a GPU cluster. We want interactive search with <1s latency, running entirely in the browser.

## Key Constraints

| Constraint | Implication |
|------------|-------------|
| No server | Cross-encoder reranker (8B params) is eliminated entirely |
| Browser model budget | ~70-130MB ONNX model is the practical max |
| Query encoding | Qwen3-VL-Embedding-8B can't run in browser; need a proxy |
| Index size | 1M images × 4096d float32 = 16GB — must be reduced |

## Architecture

### Offline Prep (one-time, GPU cluster)

**Step 1 — Dimensionality reduction (PCA)**

Fit PCA on the 1M Qwen3-VL-Embedding-8B vectors (4096d → 256d). Retain the projection matrix `W_pca` (4096 × 256) and mean vector `μ` (4096).

- Source: embed stage output at `outputs/embed/cyclomedia_20260407_060223/`
- PCA retains ~90%+ variance at 256d for these embeddings
- Script: `scripts/build_browser_index.py`

**Step 2 — Scalar quantization**

Quantize the PCA-reduced embeddings from float32 to uint8 per dimension (track min/max per dim for dequantization). This is the static index shipped to the browser.

| Dims | Precision | Index size | Transfer (gzip) |
|------|-----------|-----------|-----------------|
| 256 | float16 | 512MB | ~350MB |
| 256 | uint8 | 256MB | ~180MB |
| 128 | uint8 | 128MB | ~80MB |

**Step 3 — Train query projection**

The browser-side text encoder (e.g., `BAAI/bge-small-en-v1.5`, 33M params, 384d output) lives in a different embedding space than Qwen3-VL-Embedding. Bridge the gap with a learned linear projection.

Training data:
- Sample ~50K diverse text descriptions (captions, query templates, scene descriptions)
- Encode each with Qwen3-VL-Embedding-8B → `qwen_emb` (4096d)
- Encode each with bge-small → `small_emb` (384d)
- Project `qwen_emb` through PCA → `target` (256d)
- Train `W_proj` (256 × 384) via MSE: `W_proj @ small_emb ≈ PCA(qwen_emb)`

This is a simple linear regression — trains in seconds on CPU.

Script: `scripts/train_query_projection.py`

**Step 4 — Export artifacts**

| Artifact | Format | Size | Description |
|----------|--------|------|-------------|
| `index.bin` | uint8 flat binary | ~256MB | 1M × 256 quantized embeddings |
| `index_meta.json` | JSON | <1KB | dim, count, min/max per dim for dequant |
| `pca_components.bin` | float32 | ~4MB | PCA projection matrix (4096 × 256) |
| `pca_mean.bin` | float32 | ~16KB | PCA mean vector (4096) |
| `W_proj.bin` | float32 | ~384KB | Projection matrix (256 × 384) |
| `image_manifest.json` | JSON | ~50MB | Row index → image path, recording_id, lat/lon, etc. |
| ONNX text model | ONNX | ~70MB | bge-small-en-v1.5 quantized |

Total static assets: ~380MB (cacheable, loaded once).

### Browser Runtime

```
User types query
  → bge-small ONNX encode (Transformers.js)     ~50-100ms
  → W_proj @ small_emb                           <1ms
  → dot product vs uint8 index (typed arrays)    ~50-100ms
  → argsort top-K                                ~10ms
  → render thumbnail grid                        ~50ms
                                         TOTAL:  ~200-300ms
```

**Libraries:**
- [Transformers.js](https://huggingface.co/docs/transformers.js) or [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) for text encoding
- Vanilla JS typed arrays (`Uint8Array`, `Float32Array`) for index search
- WebGPU matmul as optional acceleration path (not required at 256d)

### Search Flow Detail

```javascript
// One-time load
const index = new Uint8Array(await fetch('index.bin').then(r => r.arrayBuffer()));
const meta = await fetch('index_meta.json').then(r => r.json());
const Wproj = new Float32Array(await fetch('W_proj.bin').then(r => r.arrayBuffer()));
const manifest = await fetch('image_manifest.json').then(r => r.json());
const model = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5');

// Per query
const smallEmb = await model(queryText);          // 384d float32
const queryVec = matmul(Wproj, smallEmb);         // 256d float32
const scores = dotProductUint8(index, queryVec, meta);  // 1M scores
const topK = argsortDesc(scores).slice(0, k);     // top-K indices
const results = topK.map(i => manifest[i]);        // image paths + metadata
```

## Training Text Generation

The `train_query_projection` stage generates 50K synthetic texts via template expansion. 16 sentence templates with 5 slot types (17 scenes, 32 objects, 14 conditions, 10 actions, 14 descriptors) produce ~17M unique combinations — more than enough for 50K deduplicated samples.

**Strengths:** Good coverage of urban/streetscape vocabulary matching the Cyclomedia domain. The linear projection is robust and mainly needs diversity to span the embedding space.

**Limitations:** Synthetic templates won't capture natural query phrasing ("show me places where people jaywalk"), abstract/subjective queries ("dangerous looking area"), or colloquial language. If real query logs or captions become available, they would improve projection quality — the stage accepts any text list internally and could be extended with a `--text-file` input.

Acceptable for MVP; revisit if retrieval quality degrades on natural-language queries.

## Quality Considerations

| Component | Quality impact | Notes |
|-----------|---------------|-------|
| PCA 4096 → 256 | Small (~5-10% recall loss) | Qwen embeddings are high-rank; PCA preserves most structure |
| uint8 quantization | Negligible | 256 quantization levels per dim is sufficient for ranking |
| Learned projection | Moderate | Main quality bottleneck — linear map between embedding spaces loses some semantic nuance |
| No cross-encoder reranking | Moderate | Top-1 precision drops ~5-10% vs full pipeline; top-20 recall stays strong |

**Overall**: interactive search results will be relevant and useful for browsing. For publication-quality retrieval, use the full GPU pipeline.

## Alternative: Image Query Support

For image-based queries (instead of text), the browser could:
- Accept a user-uploaded image
- Encode it via a small CLIP-like model in browser (e.g., `Xenova/clip-vit-base-patch32`, ~340MB)
- Project into the same PCA space via a separate projection matrix trained on image pairs
- Same dot-product search path

This would require a second projection matrix and a second ONNX model, but the search infrastructure is identical.

## Implementation

### Pipeline Stages (recommended)

Both offline prep steps are registered as urbanembed pipeline stages and can be run via the DAG orchestrator:

```bash
# Full pipeline: embed → build_browser_index → train_query_projection
python -m dagspaces.urbanembed.cli pipeline=browser_search_cyclomedia

# Or with pre-computed embeddings
python -m dagspaces.urbanembed.cli pipeline=browser_search_cyclomedia \
  browser_index.embeddings_input_path=/path/to/embed/output
```

See [[urban-embed]] for config group details (`browser_index`, `query_projection`).

### Standalone Scripts (alternative)

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/build_browser_index.py` | PCA fit + uint8 quantize + export index.bin, metadata, manifest | Done |
| `scripts/train_query_projection.py` | Generate text pairs, train W_proj, export projection matrix | Done (requires GPU) |

### Web App — `viz/embedding_search/`

Vite + React app with deck.gl map view. Key files:

| File | Purpose |
|------|---------|
| `src/lib/search-engine.js` | `SearchEngine` class: artifact loading with progress, full search pipeline |
| `src/lib/math-utils.js` | Optimized matmul, uint8 dot product search, top-K |
| `src/lib/text-encoder.js` | Transformers.js wrapper for `Xenova/bge-small-en-v1.5` |
| `src/hooks/useSearchEngine.js` | React hook for engine lifecycle |
| `src/components/MapView.jsx` | deck.gl `ScatterplotLayer` colored by similarity score |
| `src/components/ResultGrid.jsx` | CSS grid of thumbnail result cards |
| `src/components/SearchBar.jsx` | Text input with 300ms debounce + latency badge |
| `src/components/LoadingOverlay.jsx` | Staged progress bars during artifact loading |

**Image serving:** Custom Vite middleware maps `/images/<relative_path>` to filesystem reads from `IMAGE_ROOT` env var (defaults to `/share/ju/cyclomedia/raw/manhattan_2025_1k`).

**Setup:**
```bash
# 1. Build index artifacts (no GPU needed)
python scripts/build_browser_index.py \
    --embed-dir outputs/embed/cyclomedia_YYYYMMDD_HHMMSS \
    --output-dir viz/embedding_search/public/data \
    --image-root /share/ju/cyclomedia/raw/manhattan_2025_1k

# 2. Train projection (GPU required)
python scripts/train_query_projection.py \
    --pca-dir viz/embedding_search/public/data \
    --output-dir viz/embedding_search/public/data

# 3. Run web app
cd viz/embedding_search && npm install && npm run dev
```

## Related Pages

- [[urban-embed]] — Embedding pipeline and rerank stage
- [[vllm-inference]] — GPU inference infrastructure (offline pipeline)
