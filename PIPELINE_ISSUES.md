# VQA Pipeline Issues — 2026-02-10

## Status

Job `675068` on `klara.tech.cornell.edu` (4× A6000, 16 CPU, 887 GiB RAM).
Pipeline processes 2.16M Cyclomedia images with Qwen2.5-VL-3B-Instruct-AWQ.

**The `map_batches` refactor (separate `_load_images_batch` operator) is working correctly** — no more 9.6 GiB per-task warnings, and the execution plan shows proper operator separation:

```
ReadParquet -> MapBatches(_load_images_batch) -> Map(_preprocess) -> ChatTemplateUDF -> vLLMEngineStageUDF -> DetokenizeUDF -> Map(_postprocess)
```

However, two critical issues remain:

---

## Issue 1: vLLM batch size collapse (size 2 instead of 64)

### Symptom

Despite `batch_size=64` in config, vLLM consistently processes batches of size 2:

```
[vLLM] Elapsed time for batch ... with size 64: 20.95s   ← only the FIRST batch
[vLLM] Elapsed time for batch ... with size 2: 20.95s    ← immediately after
[vLLM] Elapsed time for batch ... with size 2: 34.01s    ← all subsequent
[vLLM] Elapsed time for batch ... with size 2: 33.16s
[vLLM] Elapsed time for batch ... with size 2: 39.88s
...
```

The very first batch on one GPU is size 64, but every subsequent batch across all 4 GPUs is size 2.

### Root cause hypothesis

The batch size in `vLLMEngineProcessorConfig` controls how many rows Ray Data sends to each vLLM actor per task. But the `max_tasks_in_flight_per_actor` defaults to 4, and during startup, the upstream operators haven't produced enough completed batches. The batch-of-2 likely comes from:

1. **Upstream starvation**: `_preprocess` and `_load_images_batch` are CPU-bound and competing for the same 16 CPUs as the ChatTemplateUDF actors. With 10-15.5/16 CPUs active, the pipeline can't feed vLLM fast enough.
2. **Object store backpressure**: Once the store exceeds its budget (37.3 GiB), spilling kicks in. Downstream consumers (vLLM) then have to wait for data to be read back from disk, reducing effective batch size.
3. **Block size mismatch**: `_load_images_batch` uses `batch_size=64`, producing blocks of 64 rows. But Ray Data's internal block splitting/merging may fragment these before they reach vLLM.

### Things to investigate

- Check if `target_max_block_size` on the DataContext is set appropriately (default may be too small for image-heavy rows, causing block splits).
- The `concurrency` parameter is deprecated in Ray 2.51+ — the warning suggests using `compute` instead. This may affect how actor pools are sized.
- Try `enforce_eager=True` in vLLM engine kwargs to skip CUDA graph capture entirely. The 3B AWQ model is small enough that eager mode may be comparable, and it eliminates the ~100s warmup stall that causes the initial burst.
- Consider whether the autoscaler warning is relevant:
  ```
  ⚠️  Actor Pool configuration of MapBatches(vLLMEngineStageUDF) will not allow it to scale up:
  configured utilization threshold (200.0%) couldn't be reached with max_concurrency=8
  and max_tasks_in_flight_per_actor=4 (max utilization = 50%)
  ```
  The `max_concurrency=8` is surprising given we set `concurrency=4`. This may indicate the config is being doubled internally.

---

## Issue 2: Uncontrolled object store growth + spilling

### Symptom

Object store grows linearly at ~3 GiB/5s during vLLM warmup, far exceeding the 37.3 GiB budget:

```
08:04:10  7.2 GiB / 37.3 GiB
08:04:45  37.8 GiB / 37.3 GiB  ← exceeds budget, ~40s in
08:05:26  61.9 GiB / 37.3 GiB
08:06:01  83.8 GiB / 37.3 GiB  ← vLLM warmup ends here
08:06:37  102.9 GiB / 37.3 GiB ← still growing even with vLLM running
08:07:37  124.0 GiB / 37.3 GiB ← job starts hanging
```

Spilling logs confirm disk offload at high throughput:
```
Spilled  2,688 MiB (40 objects)  @ 413 MiB/s
Spilled  4,560 MiB (68 objects)  @ 442 MiB/s
Spilled  8,461 MiB (126 objects) @ 495 MiB/s
Spilled 16,897 MiB (252 objects) @ 488 MiB/s
Spilled 32,985 MiB (491 objects) @ 499 MiB/s
```

### The 0.8 fraction override is NOT working

We set `ctx.override_object_store_memory_limit_fraction = 0.8` which should give a budget of `74.5 GiB × 0.8 = 59.6 GiB`. But the progress bar consistently shows `37.3 GiB` = `74.5 × 0.5`, the default fraction.

**The attribute exists** (confirmed via `dir(ctx)`) but the value is apparently not being picked up by the `ResourceManager`. Possible causes:

1. The ResourceManager may snapshot the fraction at construction time (when the first dataset executes), before `run_vqa_stage` sets it.
2. The override may need to be set **before** `ray.init()` or before any dataset execution.
3. Ray 2.53 may have changed the semantics of this attribute.

### Why backpressure isn't working

Even with the budget at 37.3 GiB, Ray Data should apply backpressure to upstream operators when the object store exceeds the limit. The fact that it grows to 124+ GiB suggests:

1. **During vLLM CUDA graph capture (~100s)**, vLLM actors are alive and hold their GPU resources, but produce zero output. The streaming executor sees the actors as "busy" and keeps feeding upstream operators.
2. The `_load_images_batch` operator with `num_cpus=1` (default) can run up to 10 parallel tasks, each producing 64 images × ~2 MB = ~128 MB per task. That's 1.28 GiB/s of production with no consumption.
3. The ChatTemplateUDF actors (pool of 2) transform rows faster than vLLM can consume them, adding further buffering.

### Things to investigate / fix

1. **Move the fraction override earlier** — set it right after `ray.init()` in `_ensure_ray_init()`, before any dataset is created.
2. **Reduce upstream parallelism during warmup**: Set `_load_images_batch` resources higher (e.g., `num_cpus=2`) to limit concurrent image-loading tasks from 10 to 5.
3. **Set `target_max_block_size`** on the DataContext to limit how much data each block carries (e.g., 64 MB instead of the default 128 MB).
4. **Consider `enforce_eager=True`** to eliminate the 100s CUDA graph capture stall entirely.
5. **Investigate `ctx.execution_options.resource_limits`** — explicitly cap the object store memory limit:
   ```python
   from ray.data import ExecutionResources
   ctx.execution_options.resource_limits = ExecutionResources(
       object_store_memory=60 * 1024**3  # 60 GiB hard cap
   )
   ```

---

## Issue 3: Job hanging

### Symptom

At `08:07:35`, Ray warns that `_preprocess` task #104 has been running for 60s (avg is 1.63s) and multiple `_postprocess` tasks have been running for 30s (avg is 0.44s).

After `08:07:37` (124 GiB in object store), the job appears to hang with progress stuck at 924 / 2,381,742.

### Root cause

This is almost certainly caused by the extreme object store pressure. At 124 GiB in-use with a 74.5 GiB physical object store, roughly 50 GiB is spilled to disk. Every object access that hits spilled data requires a disk read, causing tasks to stall waiting for their inputs. This cascades: `_preprocess` waits for image data from spilled blocks, `_postprocess` waits for vLLM outputs that are spilled, and vLLM waits for preprocessed inputs that are spilled.

The fix is the same as Issue 2 — control object store growth.

---

## Summary of changes already made (this session)

| File | Change | Status |
|------|--------|--------|
| `orchestrator.py` | Replaced `.map(_load_image_from_path)` with `.map_batches(_load_images_batch, batch_size=64)` | Working — separate operator in execution plan |
| `stages/vqa.py` | Added `preprocess_map_kwargs={"num_cpus": 0.5}` to `build_processor()` | Applied |
| `stages/vqa.py` | Set `ctx.override_object_store_memory_limit_fraction = 0.8` | Applied but NOT effective (budget still 37.3 GiB) |
| `g2_slurm_pierson_4x.yaml` | `RAY_OBJECT_STORE_MEMORY=80000000000` (80 GB) | Applied |
| `prompts/unified.py` | Added `_resolve_pil_image()` fallback (loads from path if no in-memory image) | Applied |

## Recommended next steps (priority order)

1. **Fix the fraction override** — move it before any dataset creation, or use `ctx.execution_options.resource_limits` with an explicit `object_store_memory` cap instead.
2. **Investigate the batch-size-2 problem** at the Ray Data LLM / vLLM integration layer — trace how `batch_size=64` in `vLLMEngineProcessorConfig` maps to actual batch dispatch. The `concurrency` deprecation warning may be masking a misconfiguration.
3. **Consider `enforce_eager=True`** for this small (3B AWQ) model to eliminate the warmup stall.
4. **Reduce upstream production rate** — increase CPU resource requests for image-loading tasks, or reduce the number of ReadParquet + _load_images_batch concurrent tasks.
