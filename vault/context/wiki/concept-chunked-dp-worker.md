---
title: Chunked DP-full Worker for Multimodal vLLM
category: concept
created: 2026-04-10
updated: 2026-04-13
tags:
  - vllm
  - multimodal
  - inference
  - throughput
  - patterns
  - gpu
  - performance
---

# Chunked DP-full Worker for Multimodal vLLM

Distilled lessons from the 2026-04-10 rewrite of [[vllm-inference#Chunked DP Worker|`_DP_FULL_WORKER_SCRIPT`]] in `dagspaces/common/vllm_inference.py`. The implementation details live in [[vllm-inference]]; this page is the *why* — the non-obvious reasoning you would otherwise have to re-derive from vLLM GitHub issues and a long debugging session.

## The failure this fixes

Calling `vllm.LLM.chat(list_of_N_conversations, ...)` on a large `N` (e.g. 121k) with two full-resolution images per conversation blows up two independent ways that look wrong together — first observed on `pipeline=pairwise_cyclomedia_wealth_large` against Qwen3-VL-8B-Thinking, see [[troubleshooting#Issue 4 Multimodal Rendering Bottleneck Engine Core OOM urbanpairvqa|Issue 4]]:

1. **Serial rendering before any generation.** vLLM 0.19's `llm.chat()` runs the HuggingFace multimodal processor (PIL decode → resize → patch extraction → chat template apply → tokenize) **single-threaded, once per conversation, up front**. For 2×1024² images on one CPU this is ~5 it/s. At 121,200 conversations per DP worker that's ~6.7 hours of pure CPU work while every GPU in the job sits idle. The "Rendering conversations" tqdm bar that appears is not a progress indicator through generation — it is a progress indicator through render-only work.
2. **Engine-core RSS climbs until the cgroup kills it.** Rendered prompts + multimodal tensors accumulate in the engine-core process because `llm.chat` buffers the entire batch, and the multimodal-processor LRU cache leaks memory across rendered requests ([vllm #15294](https://github.com/vllm-project/vllm/issues/15294), [vllm #35191](https://github.com/vllm-project/vllm/issues/35191)). Under the SLURM memory cgroup the engine-core subprocess gets OOM-killed a few thousand rows in, producing `EngineCore died unexpectedly` on stderr.

The misleading part is that a comment in early vLLM code suggests `llm.chat` pipelines rendering with generation via a generator. It does not, for multimodal.

## The pattern

Inside each data-parallel subprocess (`_DP_FULL_WORKER_SCRIPT`), process the row shard in **small fixed-size chunks** — default `chunk_size=64` — and for each chunk:

1. Call the stage's `preprocess_fn(row)` for every row in the chunk. Cheap; just builds the chat-message dict.
2. Walk the chunk's messages and collect every `{"type": "image_url", ...}` and string-valued `{"type": "image", ...}` block, dereferencing their `file://` paths.
3. Decode + RGB-convert + resize-to-`max_pixels` every image in a `ThreadPoolExecutor` with ~16 workers. PIL releases the GIL during `open`/`load`/`convert`/`resize`, so threads actually scale across CPUs.
4. Rewrite each block in place to vLLM's native `image_pil` block type. The HF processor inside vLLM then no longer has to decode or resize — it just runs the chat template and the vision encoder.
5. Call `llm.chat(chunk_messages, sampling_params=..., use_tqdm=False, **chat_kwargs)` on the *already-hydrated* chunk.
6. Postprocess outputs into a pre-allocated `results[chunk_start:chunk_end]` array so the original row order is preserved.
7. `del chunk_rows, chunk_pp_rows, chunk_messages, ok_messages, outputs` before the next iteration. Engine-core RSS does not grow over the life of the worker.

At the top of the function a **one-row peek** extracts the stage's `sampling_params` dict → `SamplingParams` object exactly once, because stages use uniform sampling across rows and the alternative (rebuild per row) burns CPU for no gain.

## Chunk size tuning

Tradeoffs observed while tuning on Qwen3-VL-8B-Thinking with 2×1024² images on A6000:

| `chunk_size` | effect |
|---|---|
| 1024+ | Peak RSS climbs again; engine-core buffering returns; cold-start first-chunk visible latency is minutes, which reads as "is it hung?" |
| 256 | Works but the first log line appears after ~5 minutes on thinking mode — bad UX. |
| **64** | Sweet spot. Per-call Python overhead is negligible; progress lines arrive every ~30–60 s on a warm engine; cold-start visible within ~2 min. |
| 8 | Python/per-chunk overhead dominates; lose ~10–15% throughput to function-call churn. |

64 is a default, not a constant. Stages with tiny images (thumbnails) or very short answers can go higher. Tune via `cfg.model.chunk_size`.

## Why pre-resize on the application side

`max_pixels` in `mm_processor_kwargs` tells vLLM what to downscale to, but vLLM runs that resize *single-threaded* inside the HF processor at rendering time. If you hand vLLM already-resized PIL images via `image_pil` blocks, the processor's resize becomes a no-op and the only remaining work is the vision-encoder forward pass. The 16-thread decode pool then becomes your parallel image pipeline. Empirically on real Cyclomedia JPEGs this takes ~100 images/sec per worker vs vLLM's ~10 images/sec — a ~10× win on the CPU-bound phase.

## Engine-kwarg defaults that should always be set for multimodal

In `_build_engine_kwargs`, whenever `_is_multimodal_model(...)` is true:

```python
ek.setdefault("mm_processor_cache_gb", 2)       # cap the leaky LRU
ek.setdefault("mm_encoder_tp_mode", "data")     # batch-level vision DP (vLLM 0.19+)
```

`filter_vllm_engine_kwargs` silently drops these on older vLLM builds that don't recognise them, so it is always safe to set them unconditionally.

`max_num_seqs` should be set based on the KV-cache concurrency that vLLM reports at startup — the line `Maximum concurrency for N tokens per request: X.XXx`. You want `max_num_seqs` comfortably below that ceiling, with headroom for prefill spikes under chunked prefill. For Qwen3-VL-8B-Thinking on an A6000 with `max_model_len=8192`, vLLM reports ~22.77x concurrency; `max_num_seqs=16` with `max_num_batched_tokens=16384` is the observed sweet spot. The earlier default of `max_num_seqs=8` left most of the KV cache idle.

## Streaming parquet shards

The chunked loop is also where partial results flush to disk. After every `flush_every` (default 1000) newly-completed rows, `results[flushed_upto:processed]` is written as a shard file under `<sweep_dir>/streaming/<stage_name>/rank{NN}_part{IIII}_rows{START}-{END}.parquet`. Each worker receives its absolute `row_offset` in the full DataFrame so shard filenames carry stable global row ranges, not per-rank-relative ones.

The shard directory is resolved from `HydraConfig.sweep.dir` first, then `HydraConfig.runtime.output_dir`, then `cfg.runtime.output_dir`, with `os.getcwd()` as the last-resort fallback. The `os.getcwd()` path is explicitly wrong under mllmsci because Hydra runs with `chdir: null` and cwd stays at the project root — hence the cascade.

The final `result.pkl` that the DP-full orchestrator collects at the end of each rank is unchanged. Streaming shards are additive observability + crash-resilience, not a replacement protocol.

## Why `use_tqdm=False`

vLLM's per-conversation tqdm writes multi-line-per-row progress updates to stderr when redirected to a file (as it is under SLURM / submitit) because there is no carriage-return collapsing across line buffers. This pollutes the stage's error log to the point of uselessness — tens of thousands of lines of tqdm noise drowning any real warning. The per-chunk "starting chunk N" + "N/total | chunk Xs | img Ys | gen Zs | rate | ETA" + per-flush "flushed N rows →" log lines give enough visibility on stdout without the spam.

## How this replaces Ray

Before the 2026-04 [trawler-alignment refactor](../log.md), multimodal throughput came from Ray Data: a pool of preprocessing actors kept the vLLM actors fed, and Ray's object store mediated backpressure. That worked until the object store filled (~120 GB of spilled images) and cascading stalls killed the pipeline — see [[troubleshooting]] Issues 1–3. The chunked DP-full worker achieves the same "keep the GPU fed without hoarding images" goal without Ray by:

- sharding rows across OS processes instead of actor workers,
- replacing Ray's object store with per-chunk CPU-thread image hydration,
- using ordinary pickle files + `subprocess.Popen` for inter-process transport,
- cloudpickling the stage's `preprocess_fn` / `postprocess_fn` into each subprocess so stage code does not care which DP rank it runs on.

No shared state means no contention, no backpressure protocol, no object-store-full errors, no spilling. The downside is slightly more Python overhead per chunk; the upside is that "multimodal VQA over 500k rows" stops being an operations problem.

## When NOT to use the chunked path

- **Single-GPU text-only runs.** vLLM's built-in batching is fine; the chunked path adds no value. `run_vllm_inference` keeps a non-DP single-process path for exactly this case.
- **Runs with <1000 rows total.** Cold-start dominates; just call `llm.generate` directly or accept the single unchunked call.
- **Server mode (`vllm serve` + OpenAI client).** Server mode has its own continuous batching and the OpenAI protocol streams naturally; the chunked DP worker is an offline-only pattern. `run_vllm_inference` routes to `_run_server_inference` when `cfg.model.vllm_server_url` is set.

## See also

- [[vllm-inference]] — implementation of the chunked DP worker, the embed DP worker variant, and the dispatcher
- [[troubleshooting]] — Issue 4 (multimodal rendering bottleneck + engine-core OOM) is the concrete incident this pattern fixes
- [[urban-pair-vqa]] — the canonical use-site (hundreds of thousands of 2-image pairs)
- [[urban-ocr]] — also multimodal, inherits the same path
- [[urban-embed]] — embedding inference has its own DP worker variant (`_run_data_parallel_embed`); as of 2026-04-13 it also streams chunk pickles to disk (atomic temp+fsync+rename every 50 batches) and the parent returns `(embeddings, errors)` with `None` placeholders so the stage can flush partial parquet before re-raising. See [[urban-embed#Fault Tolerance and Partial Recovery]] for the incident that drove the fix.
- [[shared-infrastructure]] — overview of `dagspaces/common/`
