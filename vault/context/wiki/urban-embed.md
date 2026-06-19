---
title: "UrbanEmbed — Image Embedding"
category: dagspace
created: 2026-04-06
updated: 2026-04-13
tags:
  - dagspace
  - embedding
  - reranking
  - retrieval
  - clustering
  - qwen3-vl
  - vllm
---

# UrbanEmbed — Image Embedding

UrbanEmbed is the dagspace for **batch dense embedding of images** for downstream retrieval, clustering, and similarity analysis. It uses vLLM with `runner="pooling"` and `llm.embed()` for GPU-accelerated embedding extraction.

## Purpose

- Batch computation of dense image embeddings at scale
- Support for retrieval, clustering, and semantic similarity workflows
- Multi-GPU scaling via vLLM's native `data_parallel_size`
- Streaming checkpoints for fault tolerance on long-running jobs

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanembed/cli.py` | Hydra CLI entry point |
| `dagspaces/urbanembed/__main__.py` | Alternative module entry point |
| `dagspaces/urbanembed/orchestrator.py` | DAG execution engine; defines `EmbedRunner` and `RerankRunner` |
| `dagspaces/urbanembed/stages/embed.py` | Core embedding stage: `run_embed_stage()` with vLLM pooling |
| `dagspaces/urbanembed/stages/rerank.py` | Two-phase retrieval: embedding recall + cross-encoder reranking |
| `dagspaces/urbanembed/conf/model/` | Model configs (Qwen3-VL-Embedding 2B/8B, Qwen3-VL-Reranker 8B) |
| `dagspaces/urbanembed/conf/pipeline/` | Pipeline definitions |

## Architecture

UrbanEmbed uses vLLM's pooling model API, matching the infrastructure pattern of other dagspaces:

| Aspect | Description |
|--------|-------------|
| Inference engine | vLLM `LLM(runner="pooling")` with `llm.embed()` |
| Output type | Dense float vectors (numpy arrays) |
| Multi-GPU | vLLM native `data_parallel_size` (auto-detected from GPU count) |
| Image handling | PIL images loaded lazily per-batch via `Image.open()` |
| Checkpointing | Two-layer: worker-side chunk pickles (every 50 batches) + stage-level streaming parquet parts (every `checkpoint_interval` rows) |

### Execution Modes

Two entry points exist with the same core logic:

1. **`run_embed_stage()`** in `stages/embed.py` — standalone stage with streaming I/O support, reads/writes parquet directly
2. **`run_vllm_embed()`** in `dagspaces/common/vllm_inference.py` — DataFrame-in/DataFrame-out API matching `run_vllm_inference()` contract

Both use `_build_engine_kwargs(cfg)` with `runner="pooling"` and pass `data_parallel_size` through to vLLM natively.

## Multi-GPU Scaling

UrbanEmbed uses data-parallel inference via `_run_data_parallel_embed()` for multi-GPU scaling:

- **Auto-detection**: `_build_engine_kwargs()` computes `dp_size = total_gpus // tp_size` when `data_parallel_size` is not explicitly set. A 2B model (TP=1) on 4 GPUs gets DP=4 automatically.
- **Manual override**: `model.engine_kwargs.data_parallel_size=4` via CLI or config
- **Subprocess isolation**: Each DP rank runs as a fresh Python process with `VLLM_DP_*` env vars set, following the vLLM 0.19 DP pattern. Workers create their own `LLM(runner="pooling")` and call `llm.embed()` on their shard.
- **Image handling**: Image file paths (strings) are passed to workers; PIL images are loaded lazily per-batch to avoid serialization and OOM.
- **Worker-side chunk streaming**: Every `CHUNK_BATCHES=50` batches (~800 embeddings at `batch_size=16`) each worker atomically writes `chunk{idx:05d}.pkl` into `{TMPDIR}/embed_dp{rank}_result.pkl.chunks/`. A kill at 95% of the run preserves 95% of the data — unlike the old single-pickle-at-end model, where a watchdog-killed rank left nothing on disk.
- **Partial-failure recovery**: On worker timeout or crash, the parent loads every chunk from every rank, returns `(all_embeddings, errors)` with `None` placeholders in the gaps, and leaves the failing ranks' chunk directories intact for manual recovery. The stage flushes what it has to parquet parts *before* re-raising. See [[#Fault Tolerance and Partial Recovery]] below.
- **Watchdog timeout**: Default `timeout=255600s` (~71h), sized to fit under the `slurm_gpu_4x` 72h SLURM limit with headroom for final cleanup. Before this was fixed, the watchdog fired at 24h and wiped out ranks 1/2/3 that were minutes from completing — see the 2026-04-12 incident in [[#Fault Tolerance and Partial Recovery]].

This is the natural scaling strategy for small embedding models where tensor parallelism is unnecessary.

## Data Flow

```
Input: parquet with image_path column
  -> pd.read_parquet()
  -> Optional: debug sampling (runtime.sample_n)
  -> Preprocess: build chat messages with instruction + image reference
  -> tokenizer.apply_chat_template() for prompt text
  -> Batch loop:
       -> Lazy PIL image loading per batch
       -> llm.embed(batch_inputs) with multi_modal_data
       -> Postprocess: normalize, truncate to output_dim
       -> Optional: streaming checkpoint to parquet part files
  -> Output: parquet with embedding column
```

## Preprocessing

`_make_preprocess(cfg)` builds chat messages for each row:

```python
messages = [
    {"role": "system", "content": [{"type": "text", "text": instruction}]},
    {"role": "user", "content": [{"type": "image", "image": image_ref, ...}]},
]
```

The `instruction` field (from `cfg.embedding.instruction`) provides task-specific context for the embedding model.

## Postprocessing

`_make_postprocess(cfg)` normalizes and truncates embeddings:

1. Convert to numpy float32
2. Optional: truncate to `output_dim` (e.g., 1536 from 2048)
3. Optional: L2-normalize (default: True)

## Output Format

| Column | Type | Description |
|--------|------|-------------|
| `embedding` | numpy array (float32) | Dense embedding vector |
| `embedding_dim` | int32 | Dimensionality of the embedding |
| `model_source` | string | Model identifier used |

Plus all original input columns.

## Configuration

### Model Configs (`dagspaces/urbanembed/conf/model/`)

| Config | Model | TP | Notes |
|--------|-------|----|-------|
| `qwen3_vl_embedding_2b.yaml` | Qwen3-VL-Embedding 2B | 1 | ~4GB VRAM, high DP headroom |
| `qwen3_vl_embedding_8b.yaml` | Qwen3-VL-Embedding 8B | 1 | ~16GB VRAM |
| `qwen3_vl_reranker_8b.yaml` | Qwen3-VL-Reranker 8B | 1 | Cross-encoder scoring via `llm.score()` |

### Embedding Parameters

| Parameter | Source | Description |
|-----------|--------|-------------|
| `instruction` | `embedding.instruction` | Text instruction for embedding context |
| `normalize` | `embedding.normalize` | L2-normalize embeddings (default: True) |
| `output_dim` | `embedding.output_dim` | Optional dimension truncation |
| `min_pixels` | `embedding.min_pixels` | Minimum image resolution |
| `max_pixels` | `embedding.max_pixels` | Maximum image resolution |
| `checkpoint_interval` | `embedding.checkpoint_interval` | Rows between streaming flushes (default: 50000) |

### Streaming I/O

When `runtime.streaming_io=True`, results are flushed to numbered parquet part files every `checkpoint_interval` rows. Two layers of fault tolerance:

1. **Worker-side chunk pickles** (always on for the DP path): each DP worker writes `chunk{idx:05d}.pkl` to its tmpdir every 50 batches via atomic temp+fsync+rename. A killed worker leaves these chunks for the parent to pick up.
2. **Stage-side parquet parts**: after the DP call returns, `run_embed_stage()` walks the embeddings (including `None` placeholders for any missing positions), merges with the preprocessed rows, and writes `part-{NNNNN}.parquet` every `checkpoint_interval` rows (default 50,000). On partial failure it still flushes everything it has *before* raising — the raised exception names the output directory that holds the recovered data.

### Fault Tolerance and Partial Recovery

Background: on 2026-04-12 an embed job of 1,038,932 rows on `slurm_gpu_4x` had rank 0 straggle past the (old) 24h Python watchdog while ranks 1/2/3 finished at 25.3/25.5/25.9h. The watchdog killed rank 0, the parent then unconditionally `os.unlink()`-ed every worker pickle in cleanup, and the stage raised with zero rows on disk — 25 hours of A6000 time discarded. The fix has three layers:

| Layer | What changed | Where |
|---|---|---|
| SLURM limit | `slurm_gpu_4x` bumped from 48h to 72h (`timeout_min: 4320`) | `dagspaces/common/conf/hydra/launcher/slurm_gpu_4x.yaml` |
| Python watchdog | `_run_data_parallel_embed` default `timeout` bumped from 86400s (24h) to 255600s (~71h) to sit under SLURM | `dagspaces/common/vllm_inference.py` |
| Data durability | Workers stream chunk pickles; parent returns `(embeddings, errors)` with `None` placeholders; stage flushes parquet parts before raising; failed-rank chunk dirs are preserved (not unlinked) | `_DP_EMBED_WORKER_SCRIPT`, `_run_data_parallel_embed`, `stages/embed.py` |

If a rank times out now, the parent logs:
```
[embed] DP embed completed with N error(s):
  - DP rank 0: incomplete: 245013/259733 embeddings (partial chunks preserved at /scratch/$USER/embed_dp0_result.pkl.chunks)
  - DP rank 0: process (pid=...) timed out after 255600s, killed
[embed] Recoverable chunk dirs (NOT deleted):
  - /scratch/$USER/embed_dp0_result.pkl.chunks
```
and the stage still writes `outputs/embed/<stamp>/part-*.parquet` for every row that *did* embed (including `None` for the gaps), then raises `RuntimeError` with the error list and the output directory. A rerun over just the missing rows can then finish the job.

## Rerank Stage

The `rerank` stage performs **two-phase retrieval** over pre-computed embeddings:

### Phase 1 — Embedding Retrieval (CPU)

- Loads all embedding parquet parts from the embed stage output
- Stacks the `embedding` column into an `(N, 4096)` float32 matrix
- Computes cosine similarity via `embeddings @ query_embedding` (L2-normalized → dot product)
- Selects top-K candidates by score

### Phase 2 — Cross-Encoder Reranking (GPU)

- Loads Qwen3-VL-Reranker-8B via vLLM with `runner="pooling"` and `hf_overrides` to swap architecture to `Qwen3VLForSequenceClassification`
- For each (query, candidate-image) pair, calls `llm.score()` to get a relevance score (sigmoid of yes/no logit difference)
- Supports vLLM's `1→N` batch mode for efficient scoring
- Sorts results by rerank score; optional score threshold and top-N filtering

### Reranking Configuration

| Parameter | Source | Description |
|-----------|--------|-------------|
| `query_text` | `reranking.query_text` | Text search query |
| `query_image_path` | `reranking.query_image_path` | Optional image query |
| `instruction` | `reranking.instruction` | Task instruction for the reranker |
| `top_k` | `reranking.top_k` | Candidates from Phase 1 (default: 100) |
| `rerank_top_n` | `reranking.rerank_top_n` | Final output limit (null = all) |
| `rerank_batch_size` | `reranking.rerank_batch_size` | Documents per `llm.score()` batch |
| `embeddings_input_path` | `reranking.embeddings_input_path` | Path to embed stage output |
| `query_embedding_path` | `reranking.query_embedding_path` | Pre-computed query embedding (.npy) |
| `score_threshold` | `reranking.score_threshold` | Minimum rerank score filter |

### Rerank Output Format

| Column | Type | Description |
|--------|------|-------------|
| `retrieval_score` | float32 | Phase 1 cosine similarity |
| `rerank_score` | float32 | Phase 2 cross-encoder relevance score |
| `rerank_rank` | int | 1-indexed rank by rerank_score |

Plus all original columns (minus `embedding`).

### Pipeline Usage

```bash
# Standalone rerank on pre-computed embeddings
python -m dagspaces.urbanembed.cli pipeline=rerank_cyclomedia \
  reranking.query_text="pedestrian infrastructure" \
  reranking.embeddings_input_path=/path/to/embed/output
```

## Browser Search Index Stages

Two pipeline stages export pre-computed embeddings into a format optimized for the browser-based search app (`viz/embedding_search/`).

### build_browser_index

Reads embed output parquet, fits PCA for dimensionality reduction (4096d → 256d), quantizes to uint8, and exports static artifacts.

| File | Role |
|------|------|
| `dagspaces/urbanembed/stages/build_browser_index.py` | Stage implementation |
| `dagspaces/urbanembed/conf/pipeline/browser_search_cyclomedia.yaml` | Pipeline config |

**Config group:** `browser_index`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embeddings_input_path` | null | Path to embed stage output (chained or manual) |
| `n_components` | 256 | PCA target dimensionality |
| `image_root` | null | Image path prefix to strip (null = use `data.image_path`) |

**Outputs:** `index.bin`, `index_meta.json`, `pca_components.bin`, `pca_mean.bin`, `image_manifest.json`

Runs on CPU only (`slurm_cpu_beefy` launcher).

### train_query_projection

Generates diverse text descriptions, encodes with both Qwen3-VL-Embedding (GPU) and bge-small-en-v1.5 (CPU), trains a linear projection bridging the two embedding spaces after PCA reduction.

| File | Role |
|------|------|
| `dagspaces/urbanembed/stages/train_query_projection.py` | Stage implementation |

**Config group:** `query_projection`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pca_input_path` | null | Path to build_browser_index output (chained or manual) |
| `n_texts` | 50000 | Number of training texts to generate |
| `pca_dim` | 256 | Must match `browser_index.n_components` |
| `qwen_model_path` | null | Qwen model (null = use `model.model_source`) |
| `bge_model` | BAAI/bge-small-en-v1.5 | Browser-side text encoder |
| `alpha` | 1.0 | Ridge regression regularization |

**Outputs:** `W_proj.bin`, `projection_summary.json`

Requires GPU for Qwen text encoding (`slurm_gpu_2x` launcher).

### Pipeline Usage

```bash
# Full pipeline: embed → build index → train projection
python -m dagspaces.urbanembed.cli pipeline=browser_search_cyclomedia

# With pre-computed embeddings (skip embed)
python -m dagspaces.urbanembed.cli pipeline=browser_search_cyclomedia \
  browser_index.embeddings_input_path=/path/to/embed/output
```

## Related Pages

- [[vllm-inference]] -- vLLM integration details (native DP, engine kwargs)
- [[architecture]] -- overall pipeline architecture
- [[shared-infrastructure]] -- common modules
- [[config-system]] -- Hydra configuration system
- [[guide-browser-search]] -- Browser-based image search design and web app
