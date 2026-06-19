---
title: "Troubleshooting & Known Issues"
category: troubleshooting
created: 2026-04-06
updated: 2026-06-18
tags:
  - troubleshooting
  - performance
  - issues
---

# Troubleshooting & Known Issues

Known performance issues, common errors, and debugging strategies for the MLLMSCI pipeline framework. Issues 1-3 (the original Ray-era starvation/spilling problems) are documented in detail in `PIPELINE_ISSUES.md`. Issue 4 (multimodal rendering bottleneck) is the current concern for the post-Ray DP-full-worker path and is fixed by the chunked worker added 2026-04-10 — see [[vllm-inference#Chunked DP Worker]].

---

## Issue 1: vLLM Batch Size Collapse

### Symptom

Despite configuring `batch_size=64`, vLLM processes only the first batch at full size. All subsequent batches collapse to size 2:

```
[vLLM] Elapsed time for batch ... with size 64: 20.95s   <- first batch only
[vLLM] Elapsed time for batch ... with size 2: 20.95s    <- all subsequent
[vLLM] Elapsed time for batch ... with size 2: 34.01s
```

### Root Cause

Three factors combine to starve the vLLM engine:

1. **Upstream starvation** -- `_preprocess` and `_load_images_batch` are CPU-bound and compete for the same 16 CPUs as the ChatTemplateUDF actors. With 10-15.5/16 CPUs active, the pipeline cannot feed vLLM fast enough to fill batches.

2. **Object store backpressure** -- once the Ray object store exceeds its memory budget (e.g., 37.3 GiB), spilling kicks in. Downstream consumers (vLLM actors) must wait for data to be read back from disk, reducing effective batch size.

3. **Block size mismatch** -- `_load_images_batch` produces blocks of 64 rows, but Ray Data's internal block splitting/merging may fragment these before they reach the vLLM operator.

### Mitigation

- Check `target_max_block_size` on the Ray `DataContext` -- the default may be too small for image-heavy rows, causing block splits
- Try `enforce_eager=True` in vLLM engine kwargs to skip CUDA graph capture. The warmup stall (~100s) causes the initial burst that overwhelms the object store
- Review the autoscaler warning about `max_concurrency` vs `max_tasks_in_flight_per_actor` -- the effective utilization threshold may be misconfigured
- Tune `max_tasks_in_flight_per_actor` and prefetch settings

---

## Issue 2: Uncontrolled Object Store Growth

### Symptom

The Ray object store grows linearly at ~3 GiB every 5 seconds during vLLM warmup, far exceeding budget:

```
08:04:10   7.2 GiB / 37.3 GiB
08:04:45  37.8 GiB / 37.3 GiB  <- exceeds budget after ~40s
08:05:26  61.9 GiB / 37.3 GiB
08:06:01  83.8 GiB / 37.3 GiB  <- vLLM warmup ends here
08:06:37 102.9 GiB / 37.3 GiB  <- still growing
08:07:37 124.0 GiB / 37.3 GiB  <- job starts hanging
```

Spilling throughput reaches 499 MiB/s with hundreds of objects being offloaded to disk.

### Root Cause

During vLLM warmup (CUDA graph capture), the engine cannot consume data, but upstream operators continue producing image batches that accumulate in the object store. The upstream operators are not throttled because backpressure signals do not propagate fast enough.

### Mitigation

- Tune `override_object_store_memory_limit_fraction` early in the pipeline config. Setting this lower limits how much RAM Ray allocates to the object store.
- Reduce the number of concurrent upstream map_batches operators to limit production rate during warmup
- Monitor object store usage via Ray dashboard or logs

---

## Issue 4: Multimodal Rendering Bottleneck + Engine-Core OOM (urbanpairvqa)

### Symptom

A large pairwise VQA run (e.g. `pipeline=pairwise_cyclomedia_wealth_large` with `pair_sampler.max_pairs=480000`) hangs in the "Rendering conversations" tqdm phase for hours per DP worker, then dies with `EngineCore died unexpectedly` after only ~3,000 rows have been rendered. Stage stderr shows:

```
Rendering conversations:   2%|▏ | 2928/121200 [11:44<7:54:14, 4.16it/s]
...
ERROR core_client.py:667 Engine core proc EngineCore died unexpectedly, shutting down client.
```

### Root Cause

The original `_DP_FULL_WORKER_SCRIPT` called `llm.chat(big_list_of_conversations)` once on the worker's entire shard (~121k pairs at a time). vLLM 0.19's `llm.chat()` does not pipeline multimodal preprocessing with generation — it serially renders every conversation through the HuggingFace processor (decode + resize 2 images + apply chat template + tokenize) before any generation begins. At ~5 it/s on one CPU, that is ~6.7 hours per worker just to reach generation. Meanwhile every rendered prompt + multimodal tensor accumulates in the engine-core process; the multimodal LRU cache also leaks (vLLM GH issues #15294, #35191) so RSS climbs until the SLURM cgroup OOM-kills the engine core.

### Mitigation (in tree)

The shared `dagspaces/common/vllm_inference.py` was rewritten to use a chunked DP worker. See [[vllm-inference#Chunked DP Worker]] for the implementation and [[concept-chunked-dp-worker]] for the distilled reasoning behind each design choice. Summary:

- Each worker iterates its shard in chunks of `chunk_size` (default 256).
- Per chunk, both PIL images on each row are decoded and resized to `mm_processor_kwargs.max_pixels` in parallel CPU threads (`image_load_workers=16`) before `llm.chat()` is called.
- Image blocks are rewritten to vLLM's `image_pil` format so the HF processor does not redo decode/resize.
- Chunk-local conversations and outputs are dropped before the next iteration → engine-core RSS stays bounded.
- Progress (rate, ETA, image-load time, generation time) is logged every `log_every` rows (default 1000).
- `_build_engine_kwargs` now sets `mm_processor_cache_gb=2` and `mm_encoder_tp_mode="data"` by default for multimodal models.
- Pairwise model config (`vllm_multimodal_qwen3_vl_8b_thinking.yaml`) bumped from `max_num_seqs=8 / max_num_batched_tokens=8192` to `16 / 16384` so generation can actually use the KV cache the engine reports (~22.7x concurrency).

### How to verify on a small sample

```bash
python -m dagspaces.urbanpairvqa.cli -m \
  pipeline=pairwise_cyclomedia_wealth_large \
  runtime.sample_n=200
```

Expect: each DP worker logs `... N/total | chunk Xs (img Ys, gen Zs) | rate rows/s | ETA s` lines, no `EngineCore died` errors, and final results parquet written.

### Knobs

| Setting | Default | What it controls |
|---------|---------|------------------|
| `model.chunk_size` | 256 | Rows per `llm.chat()` call inside each DP worker. |
| `model.log_every` | 1000 | Minimum rows between streaming progress lines. |
| `model.image_load_workers` | 16 | Threads in the per-chunk image-decode pool. |
| `model.engine_kwargs.mm_processor_cache_gb` | 2 | Cap on vLLM multimodal LRU cache (RAM). |
| `model.engine_kwargs.mm_encoder_tp_mode` | `data` | Vision-encoder DP across GPUs (vLLM 0.19+). |

---

## Issue 5: Interrupted urbanpairvqa Run — Recompile From Streaming Chunks

### Symptom

A pairwise run appears to finish (inference completed, GPUs idle) but the consolidated `outputs/pairwise/<dataset>_mvp_<ts>.parquet` is **missing**. Only the per-worker streaming chunks exist:

```
outputs/pairwise/
  pairs.parquet                      # written up front — present
  pairs.meta.json
  streaming/urbanpairvqa_pairwise/
    rank00_part0000_rows00000000-00001024.parquet
    rank00_part0001_rows00001024-00002048.parquet
    ...                              # one per chunk, across DP ranks
```

This happens when the monitor/orchestrator dies (or is killed) after the DP-full workers stream their chunks but before the final consolidation + postprocess step runs.

### Root Cause

The DP-full worker writes each chunk's **raw** stage output as it goes (15 columns: `pair_id`, `is_swapped`, `answer`, `model_response`, ...). The consolidated output additionally carries **5 derived columns** — `presented_answer`, `presented_label`, `presented_score`, `relative_label`, `relative_score` — produced by `_derive_labels()` in `dagspaces/urbanpairvqa/stages/pairwise_vqa.py`, applied to the raw `answer` JSON string. That derivation only runs in the orchestrator's consolidation step, so the chunks on disk are **pre-postprocessing** and lack everything downstream analysis needs (`relative_score` etc.).

### Recovery

Recompile = concat all chunks (in true row order, parsed from the `rows<start>-<end>` span in each filename) + re-apply `_derive_labels`. The reusable utility does exactly this:

```bash
source /share/pierson/matt/mllmsci/.venv/bin/activate
python scripts/recompile_streaming_pairwise.py \
  --pairwise-dir <run>/0/outputs/pairwise \
  --out-name <dataset>_mvp_<ts>.parquet
```

It reproduces the orchestrator's own postprocessing column-for-column (including the label-canonicalization behavior), so a recompiled output is mutually consistent with a normally-written one. Verify: row count matches `pairs.meta.json` `rows`, and `relative_label` is roughly symmetric (`Less ≈ More`, `MuchLess ≈ MuchMore`) — the signature of correct de-swapping ([[concept-counterbalancing]]).

Downstream geo work (e.g. [[guide-neighborhood-aggregation]]) joins the recompiled output to the always-present `pairs.parquet` on `pair_id`.

---

## Issue 3: Job Hanging from Spilled Data

### Symptom

After prolonged object store spilling, the entire job stalls. No progress is made, and the SLURM job appears hung.

### Root Cause

Cascading stalls from disk reads. When the object store has spilled tens of GiB to disk, every data access requires a disk read. If multiple vLLM actors simultaneously need spilled data, disk I/O becomes the bottleneck. Combined with backpressure, the pipeline enters a deadlock-like state.

### Mitigation

- Prevent the object store from growing unchecked (see Issue 2)
- Use faster local storage (NVMe) for spilling rather than network-attached storage
- Set explicit memory limits and kill jobs that exceed them rather than allowing indefinite spilling

---

## General Debugging

### Enable Debug Mode

```bash
python -m dagspaces.urbanvqa.cli pipeline=<pipeline> runtime.debug=true runtime.sample_n=100
```

Debug mode enables verbose logging and limits input size, making it easier to diagnose issues.

### Check W&B Logs

If W&B is configured, check the run dashboard for:
- Stage timing and row counts
- Error logs and stack traces
- Config snapshots (verify overrides were applied)

The `dagspaces/common/wandb_logger.py` module logs distributed execution metadata with auto-tagging.

### Inspect Output Parquets

Use pandas or pyarrow to inspect intermediate outputs:

```python
import pandas as pd
df = pd.read_parquet("outputs/<run>/<stage>/output.parquet")
df.head()
df["answer"].value_counts()
```

Check for empty answers, parse failures, or unexpected distributions.

### Check pipeline_manifest.json

The manifest at the project root tracks completed pipeline runs with metadata. Check it to confirm whether a run completed or which stage it failed at.

---

## SLURM Issues

### Server Environment

Ensure `server.env` is correctly configured. The `ensure_dotenv()` function loads it at startup. Missing or incorrect values cause silent failures.

Key settings to verify:
- `SLURM_PARTITION` -- must match an available partition on your cluster
- `VENV` -- path to the Python virtual environment must be accessible from compute nodes
- NCCL settings (`NCCL_SOCKET_IFNAME`, `NCCL_DEBUG`) -- incorrect network interface causes multi-GPU communication failures

### GPU Allocation

- Verify GPUs are available: `sinfo -p <partition> -o "%G %D %t"`
- Check that `tensor_parallel_size` in model config matches the number of GPUs requested in the SLURM launcher
- If using multi-node, ensure NCCL can communicate across nodes (check firewall rules and network interfaces)

### Launcher Configs

SLURM launcher configs live in `dagspaces/common/conf/hydra/launcher/` (shared) and can be overridden per-dagspace in `dagspaces/<dagspace>/conf/hydra/launcher/`. Verify the launcher matches your cluster's resource limits.

---

## Common Errors

### OmegaConf Resolution Failures

```
omegaconf.errors.InterpolationResolutionError: ... ${oc.env:SOME_VAR}
```

A config file references an environment variable that is not set. Check `server.env` and ensure the variable is defined. The `${oc.env:VAR}` syntax in Hydra configs requires the variable to exist at resolution time.

### Model Not Found

```
ValueError: Model ... not found
```

The vLLM model path in the model config does not exist or is not accessible from the compute node. Verify the path in `dagspaces/common/conf/model/<model>.yaml` and ensure it is on shared storage visible to SLURM workers.

### GPU Out of Memory

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

The model does not fit in GPU memory with the current batch size and tensor parallel configuration. Solutions:
- Reduce `model.batch_size`
- Increase `model.engine_kwargs.tensor_parallel_size` to shard across more GPUs
- Use a quantized model variant (e.g., AWQ)
- Reduce `model.engine_kwargs.max_model_len`

### Ray Object Store Errors

```
ray.exceptions.ObjectStoreFullError
```

The object store is full and cannot spill fast enough. See Issue 2 above. Reduce upstream concurrency or increase the object store memory limit.

---

## vLLM Nightly / gemma4_unified Stack (2026-06)

Running the `Gemma4UnifiedForConditionalGeneration` models (gemma-4-12B / -12B-it — encoder-free, `model_type: gemma4_unified`, distinct from the e2b/e4b `gemma4` arch) needed a stack bump that introduces a few new error classes. All affect any GPU vLLM job once the venv is on the nightly.

### `ImportError: libcudart.so.13: cannot open shared object file`

The PyPI stable `vllm==0.23.0` is a **CUDA-13** build. vLLM ≤0.22.1 cannot load `gemma4_unified` at all, and only the **cu129 nightly index** ships compatible wheels (the cu128 nightly index is empty, so `--index-strategy unsafe-best-match` silently grabs the CUDA-13 PyPI build → this mismatch). Install with both cu129 indexes:

```bash
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

Result: `vllm 0.23.1rc1.dev*+cu129` + `torch 2.11.0+cu129`. Pre-upgrade snapshot for rollback: `/share/pierson/matt/cache/venv_snapshot_pre_vllm_nightly_*.txt`. Also requires `transformers ≥5.12` to recognize the arch.

### `ImportError: GLIBCXX_3.4.32 not found` (flashinfer sampling kernels)

flashinfer ≥0.6.12 JIT kernels need `GLIBCXX_3.4.32`, but every launcher's `source ~/.bashrc` conda-activates anaconda base, whose **older** `libstdc++.so.6` shadows the system one (which has 3.4.32). Fix — force the system libstdc++ ahead via `LD_PRELOAD` in the launcher `setup:`, exported **before** python execs (the loader reads `LD_PRELOAD` at process start, so server.env is too late):

```yaml
- export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

Propagated 2026-06-18 to all 12 GPU launchers (`slurm_gpu_{1x,2x,3x,4x,6x}`, `slurm_gpu_ju_{1x,2x,4x}`, `slurm_gpu_klara_{1x,2x,4x}`, `urbanocr/slurm_gpu_4x`). **Never** write bash `${...}` expansion in launcher `setup:` YAML — OmegaConf treats it as a config interpolation and the orchestrator silently skips the whole stage (monitor "completes" in seconds, no stage submitted). See [[slurm-deployment]].

### `AssertionError: Failed to apply prompt replacement for mm_items['image'][0]`

Multimodal preprocessing dies for `gemma4_unified` only. Its image token is `<|image|>` (id 258880) and its `chat_template.jinja` inserts it only for `{"type":"image"}` content blocks — but stages send OpenAI `{"type":"image_url"}` and the single-process path uses `tokenizer.apply_chat_template`, whose default template omits images. Fixed in `dagspaces/common/vllm_inference.py` via `_gemma4_unified_chat_template()` (loads the model's own `chat_template.jinja`, cached) + `_to_image_type_blocks()`, **gated to the unified arch** so qwen/gemma-e2b/e4b are not regressed. Validated end-to-end on the klara_1x single-process path (110k-pair schools run). **Still pending:** the DP-full worker (`_DP_FULL_WORKER_SCRIPT`, used by `klara_2x`) is not yet gated, so DP launchers still fail for unified models — run them single-process for now. See [[vllm-inference]].

---

## See Also

- [[vllm-inference]] -- vLLM engine configuration and tuning
- [[slurm-deployment]] -- SLURM launcher configuration
- [[config-system]] -- Hydra configuration system and overrides
