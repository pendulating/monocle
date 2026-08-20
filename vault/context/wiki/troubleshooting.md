---
title: "Troubleshooting & Known Issues"
category: troubleshooting
created: 2026-04-06
updated: 2026-08-11
tags:
  - troubleshooting
  - performance
  - issues
  - interpretability
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

## Issue 6: Every SLURM Launch Dies at Setup — Gutted `.venv`

### Symptom

Every job on every dagspace fails immediately in the submitit setup block; the
`.err` shows `source: No such file or directory` for
`/share/pierson/matt/mllmsci/.venv/bin/activate`. Nothing reaches Python, so
there is no traceback and no W&B run — it does not look like a pipeline bug.

### Root Cause

`.venv` was gutted: empty `site-packages` (2 entries), no `bin/` at all. Every
launcher's setup does
`source ${oc.env:MLLMSCI_VENV_ACTIVATE,/share/pierson/matt/mllmsci/.venv/bin/activate}`,
and **both** `server.env` and the hardcoded fallback default pointed there — so
there was no surviving path.

### Fix (applied 2026-08-11)

Repointed to `.venv-3.12` (vllm 0.19.0, torch 2.10.0+cu128) in `server.env`,
`server.env.example`, and the fallback default in all 16 launcher configs.
Verify the resolved setup line rather than trusting the config text — the value
is baked into the sbatch script at submit time, after `ensure_dotenv()` loads
`server.env`:

```python
from dagspaces.common.stage_utils import ensure_dotenv; ensure_dotenv()
# compose with return_hydra_config=True, then read cfg.hydra.launcher.setup
```

The nightly env stays an explicit per-run override (gemma-4-12b needs vLLM 0.23):

```bash
export MLLMSCI_VENV_ACTIVATE=/share/pierson/matt/mllmsci/.venv-nightly/bin/activate
```

---

## Issue 7: Shared TMPDIR — Fixed, and Must Stay Shared

### Symptom (historical)

Concurrent DP sweep jobs on one node silently ran the **wrong model** — the
2026-06-13 schools sweep had two of six models never actually run, with result
dirs duplicating a neighbour. Tell-tale: the parent logs the correct
`streaming_output_dir`, but the per-rank `DP rank N/2: starting (...)` line
shows a *different* subdir.

### Root Cause + Fix

DP workers wrote task/result pickles to a fixed, non-unique path inside the
shared `TMPDIR` (`/scratch/$USER`), so concurrent jobs clobbered each other.
**Fixed 2026-06-14**: every DP path now allocates a unique per-invocation
`tempfile.mkdtemp()` subdir inside TMPDIR and `rmtree`s it —
`vllm_inference.py:1781`, `1960`, `2378`, `3389`. Concurrent sweeps
(`array_parallelism > 1`) are safe.

### ⚠️ Do not "re-fix" this by isolating TMPDIR per job

Re-verified 2026-08-11: `/scratch/$USER` holds **zero** orphaned dp workdirs, so
cleanup works. The shared TMPDIR is also where torchinductor caches compiled
kernels (`/scratch/$USER/torchinductor_$USER`, ~213 MB). Setting
`TMPDIR=/scratch/$USER/$SLURM_JOB_ID` would fragment that cache and force a
fresh `torch.compile` on every run (~256 s for qwen3.5-9b) — a straight
regression. The correct design is the current one: **share the caches, isolate
the mutable per-run state.**

Note also that non-DP runs (`data_parallel_size` unset, `concurrency=1` — which
is what every urbanpairvqa case pipeline pins) never touch the task-pickle path
at all, so they were never exposed even before the fix.

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

Hits **all gemma-4 models** on the single-process (single-GPU) path — for two related reasons. The root cause is shared: stages send OpenAI `{"type":"image_url"}` blocks, but the single-process path builds the prompt with `tokenizer.apply_chat_template` → `tokenizer.encode` → `TokensPrompt` + `multi_modal_data`, and gemma's chat templates emit **zero** image placeholders for `image_url` blocks → N images, 0 placeholders → the assertion. The **DP path never had this bug** because it calls `llm.chat()`, which does vLLM's first-class multimodal placeholder insertion end-to-end.

1. **`gemma4_unified` (12B)** — image token `<|image|>` (id 258880); its `chat_template.jinja` only emits it for `{"type":"image"}` blocks. Fixed via `_gemma4_unified_chat_template()` (loads the model's own `chat_template.jinja`, cached) + `_to_image_type_blocks()`, gated to the unified arch.
2. **`gemma4` (e2b/e4b)** — same symptom on multi-image prompts (e.g. pairwise VQA's 2-image inputs). These ran fine in production only because the prior sweeps used `klara_2x` (DP/`llm.chat`); they crash on `klara_1x` (single-process). Fixed 2026-06-25 in `run_vllm_inference`: the single-process multimodal branch now **routes the non-unified gemma-4 family through `llm.chat()`** (new `_hydrate_messages_for_chat()` converts `image_url`→`image_pil` blocks, mirroring the DP path's `_hydrate_chunk_images`). Gated to `model_family == "gemma-4"` and not unified, so qwen / phi / unified keep their existing path. Validated: gemma-4-e2b 2-image smoke run on `.venv` (vLLM 0.19.1) + `klara_1x` renders + writes results (was crashing on both 0.19.1 and the 0.23 nightly — so this was **not** a venv issue; nightly does not fix it).

**Still pending:** the DP-full worker (`_DP_FULL_WORKER_SCRIPT`, used by `klara_2x`) is not yet gated for `gemma4_unified` (12B) — run unified models single-process for now. The e2b/e4b fix above is single-process, so those run on `klara_1x` (1 GPU). See [[vllm-inference]].

---

## Analysis Traps (interpretability / prompt-search)

These are not crashes — they are results that look real and are wrong. Both cost real time; one put wrong numbers in a paper draft.

### Gemma residual-stream cosine is always ≈ 1.0 (massive activations)

**Symptom.** You extract hidden states from gemma-4 and the pairwise cosine between activations for *completely different* inputs comes back `mean=1.0000, min=0.9993`. Read literally: "every activation is identical, the signal is dead, the probe is impossible."

**It is an artifact.** Gemma has *massive activations* — a handful of residual dimensions with enormous magnitude. At layer 24 of gemma-4-12B, **dim 1750 alone has |mean| = 246.9** while the median dimension is **0.25**; `||mean vector|| = 251.8` vs `mean ||x − mean|| = 1.7`. The shared component is ~**150×** the input-specific signal, so cosine is dominated by it and everything looks parallel.

**Fix.** **Center (and standardize) before probing or plotting.** Centered, the same states give mean cosine `0.0005`, range `−0.85 … 0.92` — pair-specific and near-orthogonal, exactly as they should be.

**Tell.** If the model emits *different outputs* for those inputs, the activations *cannot* be identical. Trust the behavior, distrust the raw cosine. See [[concept-activation-probe]].

### GEPA's last `[val-eval]` line is not the returned best

**Symptom.** You read the final `[val-eval]` score from a GEPA log and report it as the optimized result. It is wrong — and plausibly wrong, which is worse.

**Cause.** `evaluate()` logs a `[val-eval]` line on **every** full-val candidate evaluation. The last line is therefore the **last candidate tried**, not the winner GEPA returns.

**Fix.** Read `max(val_aggregate_scores)` from `metrics.json` (task evals run at temperature 0, so they are deterministic). `gepa_pairwise.py` now also writes `best_val_ordinal` directly. See [[concept-self-explanation-ladder]].

### Cyclomedia depth: do not calibrate against `groundLevelOffset`

**Symptom.** You decode a depth PNG (`code = R*256 + G`), fit the down-face ground plane against the catalog's `groundLevelOffset` (≈ 2.23 m), and get a clean scale of 245.76 codes/m. The fit looks excellent (R² = 0.997) and the implied range tops out at a suspiciously round 200 m.

**It is wrong by ~2%.** The depth render places the camera at a **fixed nominal ~2.18 m above the road for every vehicle fleet** — the down-face nadir code is ~16930 whether the catalog says 2.2259 m or 2.9856 m. So `groundLevelOffset` is not the rendered camera height and cannot anchor the scale. The correct value, measured from known camera baselines, is 249.86 ± 0.17 codes/m — **24 σ away**.

Calibrating on the down face alone is also **not identifiable at all**: it spans only ~2.2–3.9 m, over which linear, log and inverse fits are indistinguishable while diverging wildly on extrapolation.

**Fix.** Use `to_metres()`: `range_m = (code - 16384) / 250`. Anchor on *camera baselines*, not on the ground plane — if camera B sits on camera A's ray, `range_A - range_B == |AB|`, which is known from the catalog and needs no feature matching. See [[concept-cyclomedia-depth-maps]].

Two more depth gotchas: `0` is the *no-return* sentinel, not "very close" (it would decode to −65.5 m — mask it, `to_metres` returns NaN); and the encoding is **not** IEEE float16, despite the down face landing at a plausible-looking 3.06–3.88 m under that reading.

### DuckDB `USING SAMPLE` runs *before* `WHERE`

**Symptom.** `SELECT * FROM t WHERE <bbox> USING SAMPLE 5000 ROWS` returns ~20 rows instead of 5000, and they cluster oddly.

**Cause.** DuckDB applies the sample right after `FROM`, before the filter. It sampled 5000 rows from all 5.2M, then the bbox filter discarded almost all of them.

**Fix.** Wrap the filtered set: `SELECT * FROM (SELECT * FROM t WHERE <bbox>) USING SAMPLE 5000 ROWS`. Related: `QUALIFY` requires a window function, so a plain computed-column filter (e.g. `dist_m <= r`) must go in a subquery.

### Do not walk the Cyclomedia image tree

**Symptom.** A `find` or recursive `ls` over `/share/ju/cyclomedia/raw/<dataset>` hangs; a 2-minute timeout is not enough.

**Cause.** NFS directory listing over millions of small directories. Never enumerate it.

**Fix.** Construct paths from the catalog instead — `{RAW_ROOT}/{dataset}/{group}/{recording_id}/` (see `catalog.recording_dir()` in [[guide-cyclomedia-browser]]). The catalog already carries `image_path` and `depthmap_present`, so the filesystem never needs to be asked.

---

## See Also

- [[vllm-inference]] -- vLLM engine configuration and tuning
- [[slurm-deployment]] -- SLURM launcher configuration
- [[config-system]] -- Hydra configuration system and overrides
- [[concept-activation-probe]] -- HF hidden-state extraction from `gemma4_unified`; the massive-activation trap
- [[concept-self-explanation-ladder]] -- GEPA self-distillation; the `[val-eval]` scoring trap
