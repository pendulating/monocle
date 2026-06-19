---
title: vLLM Inference Engine
category: infrastructure
created: 2026-04-06
updated: 2026-04-10
tags:
  - vllm
  - inference
  - gpu
  - multimodal
---

# vLLM Inference Engine

Deep dive on `dagspaces/common/vllm_inference.py` -- the shared inference engine used by all dagspaces for GPU-accelerated LLM generation via vLLM.

## Core Function: run_vllm_inference()

```python
def run_vllm_inference(
    df: pd.DataFrame,
    cfg,
    preprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    postprocess: Callable[[Dict[str, Any]], Dict[str, Any]],
    stage_name: str = "vllm_inference",
) -> pd.DataFrame:
```

### How It Works

1. **Input:** Receives a DataFrame (or Ray Dataset, auto-converted to pandas)
2. **Server check:** If `cfg.model.vllm_server_url` or `VLLM_SERVER_URL` env var is set, routes to `_run_server_inference()` (OpenAI-compatible API client mode)
3. **Preprocess:** Calls `preprocess(row_dict)` for each row. The preprocessor must set:
   - `row["messages"]` -- list of chat message dicts
   - `row["sampling_params"]` -- plain dict of generation parameters
4. **Engine init:** Builds vLLM `LLM` engine from config via `_build_engine_kwargs(cfg)`
5. **Prompt construction:** Applies chat template via `tokenizer.apply_chat_template()` with thinking mode support
6. **Generation:** Calls `llm.generate(prompts, sampling_params)` in configurable batches
7. **Reasoning split:** For thinking models, extracts reasoning via `_split_reasoning()` and populates `row["generated_reasoning"]`
8. **Postprocess:** Sets `row["generated_text"]` and usage info, then calls `postprocess(row)`
9. **Cleanup:** Explicitly shuts down vLLM engine workers via `_shutdown_llm()` to prevent SLURM job hangs
10. **Output:** Returns a DataFrame of all postprocessed rows

---

## Multimodal Support

Messages can contain image content blocks alongside text:

```python
{"role": "user", "content": [
    {"type": "image", "image": <PIL.Image>},
    {"type": "text", "text": "What do you see in this image?"}
]}
```

### Image handling pipeline

| Step | Function | Description |
|------|----------|-------------|
| Detection | `_is_multimodal_model(model_source, cfg)` | Checks `runtime.multimodal_enabled`, `model.multimodal` config flags, or pattern-matches model name (qwen-vl, internvl, phi-vision, llava, cambrian, smolvlm, gemma-4, etc.) |
| Extraction | `_extract_images_from_messages(messages)` | Walks all message content blocks, collects PIL Image objects from `{"type": "image"}` blocks |
| Flattening | `_flatten_messages_for_template(messages)` | Converts multimodal blocks to text-only for chat template (replaces image blocks with `<image>` placeholder tokens) |
| Passing | via `multi_modal_data={"image": images}` | Images passed to vLLM's `TokensPrompt` alongside tokenized prompt IDs |

### Multimodal engine kwargs

Key model config fields for multimodal:

```yaml
engine_kwargs:
  limit_mm_per_prompt:
    image: 1           # Max images per prompt
  mm_processor_kwargs:
    min_pixels: 256    # 16x16 single patch minimum
    max_pixels: 1003520  # 980x1024 max resolution
    fps: 1             # Video frame rate (if applicable)
```

---

## LoRA Adaptation

### _remap_lora_keys_for_vlm(lora_path, model_source, stage_name)

Adapters trained on `AutoModelForCausalLM` have keys like `base_model.model.model.layers.X...` which vLLM maps to `model.layers.X...`. But VLM architectures (e.g., `Qwen3_5ForConditionalGeneration`) expect `model.language_model.layers.X...`.

**Remapping logic:**
1. Checks if base model's safetensors contain `language_model.layers.` prefix
2. Checks if adapter keys use `base_model.model.model.layers.` (CausalLM format)
3. If mismatch detected, creates a remapped copy in `_vlm_remapped/` subdirectory
4. Copies `adapter_config.json`, `tokenizer_config.json`, and other metadata
5. Caches remapped adapter for subsequent runs

---

## Reasoning Model Support

### _detect_reasoning_parser(model_source) -> Optional[str]

Maps model paths to vLLM `ReasoningParserManager` parser names:

| Pattern | Parser | Models |
|---------|--------|--------|
| `gemma-4`, `gemma4` | `gemma4` | Gemma 4 family |
| `gpt-oss` | `gptoss` | GPT-OSS models |
| `deepseek-r1`, `deepseek_r1`, `deepseek-v3` | `deepseek_r1` | DeepSeek reasoning models |
| `qwen3` | `qwen3` | Qwen3, Qwen3.5, Qwen3-VL, etc. |
| Phi-4, Llama-3.x, Gemma-3, Qwen2.5 | `None` | Non-thinking families (no parser needed) |

### _split_reasoning(text, model_source, thinking_enabled, tokenizer) -> (reasoning, content)

**Primary path:** Uses vLLM's family-specific `ReasoningParser` (e.g., Qwen3's `<think>...</think>`, Gemma-4's `thought\n...\n`). These are maintained upstream alongside each model's chat template.

**Fallback path:** Regex-based `_fallback_strip_reasoning()` when:
- No parser matches the model family
- The parser raises an exception
- The parser returns content that still contains raw reasoning tags

Handles multiple formats:
- `<think>...</think>` -- Qwen3+, DeepSeek-R1, open-source reasoning models
- `<|begin_of_thought|>...<|end_of_thought|>` -- context-reasoner-ppo, PPO models
- Unterminated blocks (model ran out of tokens mid-reasoning)

### resolve_thinking_mode(cfg_model, default=True) -> bool

Single source of truth for the thinking mode contract. Defined in `stage_utils.py` but central to inference behavior.

**Priority (highest wins):**

1. `cfg_model.thinking_mode` -- preferred field. Accepts `"on"`, `"off"`, `"auto"`, booleans, ints
2. `cfg_model.chat_template_kwargs.enable_thinking` -- legacy boolean, honored for backwards compatibility
3. `default` kwarg (defaults to `True`)

The resolved value is passed to `tokenizer.apply_chat_template()` via `chat_template_kwargs={"enable_thinking": thinking_enabled}` and to the reasoning parser for correct truncated-output classification.

---

## Guided Decoding

Supports JSON schema and enum choice constraints passed to vLLM.

### In-process mode

Uses `_build_sampling_params()` which handles both vLLM API versions:

| vLLM Version | Parameter Class | Config Key |
|--------------|-----------------|------------|
| >= 0.12 | `StructuredOutputsParams` | `structured_outputs` |
| <= 0.11 | `GuidedDecodingParams` | `guided_decoding` |

### Server mode

Translates to OpenAI API `extra_body` fields via `_sp_to_openai_kwargs()`:

| Input Key | Output | Description |
|-----------|--------|-------------|
| `guided_decoding.json` | `extra_body.guided_json` | JSON schema constraint |
| `guided_decoding.regex` | `extra_body.guided_regex` | Regex constraint |
| `guided_decoding.choice` | `extra_body.guided_choice` | Enum choice list |
| `guided_decoding.grammar` | `extra_body.guided_grammar` | Grammar constraint |

### Example: classify stage sampling params

```yaml
sampling_params_classify:
  max_tokens: 4
  detokenize: false
  guided_decoding:
    choice:
      - "YES"
      - "NO"
```

---

## Inference Modes

### In-process batch mode (default)

Direct `vLLM.LLM.generate()` calls in the same process. Engine is initialized once, processes all rows, then explicitly shut down.

### Server mode

Routes through an OpenAI-compatible vLLM server (`_run_server_inference()`):
- Triggered by `cfg.model.vllm_server_url` or `VLLM_SERVER_URL` env var
- Uses `openai.OpenAI` client with `base_url` pointing to the server
- Thread pool concurrency (default 32 workers, configurable via `VLLM_SERVER_CLIENT_CONCURRENCY`)
- Server handles continuous batching internally

### Data-parallel mode

Spawns `dp_size` fully-isolated subprocess workers following the vLLM 0.19 DP pattern (`_run_data_parallel()`):
- Each worker is a fresh Python process with `VLLM_DP_*` env vars set (`VLLM_DP_RANK`, `VLLM_DP_SIZE`, `VLLM_DP_MASTER_IP`, `VLLM_DP_MASTER_PORT`)
- Workers create their own `LLM()` instance (without `data_parallel_size` in kwargs — vLLM reads DP config from env vars)
- vLLM handles GPU assignment and NCCL coordination internally per the env vars
- **Multimodal support**: image file paths (strings) are passed alongside prompts; workers load images lazily via PIL — avoids serializing large PIL objects across process boundaries
- **Auto-detection**: when `data_parallel_size` is not explicitly set and `total_gpus > tensor_parallel_size`, auto-computes `dp_size = total_gpus // tp_size`
- Triggered when `engine_kwargs.data_parallel_size > 1` (explicit or auto-detected)
- Communication via pickle files in temp directory; results reassembled in original order

---

## Chunked DP Worker

The default DP path for multimodal stages (`run_vllm_inference` → `_run_data_parallel_full` → `_DP_FULL_WORKER_SCRIPT`). Replaces the original "render the entire shard then call `llm.chat(big_list)` once" design that hit a 12+ hour render bottleneck and engine-core OOM around row ~3,000 — see [[troubleshooting#Issue 4 Multimodal Rendering Bottleneck Engine Core OOM urbanpairvqa]] for the incident and [[concept-chunked-dp-worker]] for the distilled reasoning (chunk-size tradeoffs, pre-resize rationale, mandatory engine-kwarg defaults, streaming-shard path resolution, why `use_tqdm=False`, how it replaces Ray).

### Why the previous approach broke

`vllm.LLM.chat()` does not pipeline multimodal preprocessing with generation. When called on a list of conversations it first renders every conversation through the HuggingFace processor (PIL decode + resize + chat-template + tokenize) — single-threaded, ~5 it/s for 2 × 1024² images on Qwen3-VL — and only then begins generation. For ~120k pairs per DP worker that is ~6.7 hours of pure CPU work before any GPU work happens, with every rendered prompt + multimodal tensor accumulating in the engine-core process. The engine-core gets OOM-killed by the SLURM cgroup well before generation starts.

### What the chunked worker does

For each DP rank, in a fresh subprocess:

1. **Boot vLLM once** with `LLM(**engine_kwargs)`.
2. **Iterate the rank's row shard in chunks of `chunk_size`** (default 256).
3. **Per chunk:**
   1. Call `preprocess_fn(row)` for each row (cheap — just builds the message dict).
   2. Walk the chunk's messages collecting all `image_url` (file://) and `image` (str path) blocks.
   3. Decode + RGB-convert + resize-to-`max_pixels` every image in parallel via a `ThreadPoolExecutor` (`image_load_workers=16` threads, PIL releases the GIL).
   4. Rewrite each image block in place to vLLM's `image_pil` format. The HF processor inside vLLM no longer has to decode/resize; it just runs the vision encoder.
   5. `llm.chat(chunk_messages, sampling_params=..., use_tqdm=False)` — generation runs on already-rendered tensors.
   6. Postprocess outputs into `results[chunk_start:chunk_end]`.
   7. `del` chunk-local conversations + outputs; the engine-core RSS does not grow over time.
4. **Streaming progress** is logged whenever `processed - last_log_at >= log_every` rows (default 1000), with elapsed, rate, ETA, and a per-chunk breakdown of image-decode time vs generation time:

   ```
   [urbanpairvqa_pairwise] DP rank 0/4: 1024/121200 (0.8%) | chunk 38.2s (img 2.1s, gen 36.1s) | 26.8 rows/s | elapsed 38s | ETA 4486s
   ```
5. **Failed-image rows** get tagged with `__image_load_error__` and are excluded from the `llm.chat()` batch but still flow through `postprocess_fn` so the result row count matches the input row count.

### Knobs (all configured under `cfg.model.*`)

| Field | Default | Notes |
|-------|---------|-------|
| `model.chunk_size` | 256 | Rows per `llm.chat()` call. Smaller → bounded memory, more frequent progress, slightly more Python overhead. Larger → fewer scheduler gaps but more peak RSS. |
| `model.log_every` | 1000 | Minimum rows between streaming progress lines. |
| `model.image_load_workers` | 16 | Threads in the per-chunk decode + resize pool. PIL releases the GIL, so threads scale to CPU count. |
| `model.engine_kwargs.mm_processor_kwargs.max_pixels` | per model | Used to pre-resize images on the worker side; same value vLLM would have used downstream. |
| `model.engine_kwargs.mm_processor_cache_gb` | 2 | Auto-set for multimodal models in `_build_engine_kwargs`. Caps the LRU cache that leaked in vLLM #15294, #35191. |
| `model.engine_kwargs.mm_encoder_tp_mode` | `"data"` | Auto-set for multimodal models. Vision-encoder data parallelism across GPUs (vLLM 0.19+). |

### Engine-kwargs auto-tuning

`_build_engine_kwargs(cfg)` now applies multimodal-specific defaults whenever `_is_multimodal_model(model_source, cfg)` returns true, unless the user has set them explicitly:

```python
ek.setdefault("mm_processor_cache_gb", 2)
ek.setdefault("mm_encoder_tp_mode", "data")
```

`filter_vllm_engine_kwargs` will silently drop these if the installed vLLM doesn't recognise them, so older vLLM versions are unaffected.

### When NOT to use the chunked path

Single-process (non-DP) inference still goes through the original `llm.generate(prompts, sampling_params)` path inside `run_vllm_inference`. The chunked worker is only invoked when `dp_size > 1`. Text-only DP jobs (no `image_*` blocks in the messages) also benefit from chunking — image hydration is a no-op when no image blocks are found.

---

## Engine Configuration

### _build_engine_kwargs(cfg) -> Dict

Builds vLLM `LLM` constructor kwargs from Hydra config:

| Setting | Source | Default |
|---------|--------|---------|
| `model` | `cfg.model.model_source` | _(required)_ |
| `tensor_parallel_size` | `cfg.model.engine_kwargs.tensor_parallel_size` | Auto-detected via `detect_num_gpus()` |
| `max_model_len` | `cfg.model.engine_kwargs.max_model_len` | Model default |
| `max_num_seqs` | `cfg.model.engine_kwargs.max_num_seqs` | GPU-type-aware default |
| `gpu_memory_utilization` | `cfg.model.engine_kwargs.gpu_memory_utilization` | vLLM default (0.9) |
| `trust_remote_code` | -- | `True` |
| `distributed_executor_backend` | -- | `"mp"` |
| `disable_custom_all_reduce` | -- | `True` when TP > 1 |
| `quantization` | Auto-detected | `"awq"` if model path contains "awq" |

### GPU-aware defaults (apply_gpu_aware_settings)

| GPU Type | batch_size | max_num_seqs |
|----------|-----------|-------------|
| RTX A6000 | 4 | 4 |
| RTX A5000 | 2 | 2 |
| A100 | 8 | 8 |
| H100 | 16 | 16 |
| V100 | 4 | 4 |
| A40 | 4 | 4 |

### GPU detection priority (detect_num_gpus)

1. `MLLMSCI_TENSOR_PARALLEL_SIZE` env var
2. `CUDA_VISIBLE_DEVICES` device count
3. SLURM GPU env vars (`SLURM_GPUS_PER_NODE`, `SLURM_GPUS_ON_NODE`)
4. `nvidia-smi -L` output line count
5. Fallback: 1

### PCIe NCCL environment (get_pcie_nccl_env_vars)

For machines without NVLink (like RTX A6000 setups):

```
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_CUMEM_HOST_ENABLE=0
NCCL_DEBUG=WARN
VLLM_WORKER_MULTIPROC_METHOD=spawn
```

---

## Engine Lifecycle

### _shutdown_llm(llm, stage_name)

Explicitly shuts down vLLM engine workers to prevent SLURM job hangs. `vllm.LLM` does not define `__del__`, and `v1.LLMEngine.__del__` only cleans up the DP process group -- it does not stop `EngineCore` / `WorkerProc_TP*` child processes.

**Shutdown sequence:**
1. Access `llm.llm_engine.engine_core.shutdown()` (V1 multiproc mode)
2. Delete LLM reference and run `gc.collect()` to trigger `LLMEngine.__del__`
3. Terminate any surviving `multiprocessing.active_children()`

---

## Known Issues

- **Multimodal rendering bottleneck (FIXED 2026-04-10)** -- `llm.chat(big_list)` rendered every multimodal conversation single-threaded before any generation, taking ~6.7h per DP worker for ~120k 2-image pairs and OOM-killing the engine core after a few thousand rows. Replaced by the chunked DP worker above. See [[troubleshooting#Issue 4 Multimodal Rendering Bottleneck Engine Core OOM urbanpairvqa]].
- **Batch size collapse (Ray-era)** -- First batch processed ~64 rows, subsequent batches collapsed to ~2. Caused by upstream starvation and object store backpressure. No longer applicable post-Ray removal but documented in [[troubleshooting]] for historical context.
- **Object store growth (Ray-era)** -- Could grow to 124+ GiB causing spilling to disk. Resolved by removing Ray; the chunked DP worker bounds RSS structurally now.
- **Job hanging** -- Cascading stalls from spilled data disk reads (Ray-era) or orphaned vLLM worker processes (current). `_shutdown_llm()` addresses the latter.

---

## See Also

- [[shared-infrastructure]] -- Overview of all shared modules
- [[config-system#Model Configs]] -- Model configuration YAML structure
- [[troubleshooting]] -- Known performance issues and mitigations
