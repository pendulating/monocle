# The 1,000,000-pair battery

On 2026-08-18 the validation-by-proxy battery grew from 100,000 pairs for each
case to 1,000,000. The prompt, the model, the seed and the sampling do not
change. Only the number of pairs changes.

The cost is about 1,700 GPU-hours, which is 10 times the "looks like" battery of
2026-08-14. That does not fit the 8 GPUs of `pierson`, thus the battery runs on
the preemptable `gpu` partition, and the pairwise stage learned 2 new things to
make that safe.

## What is new in the code

| Mechanism | Where | What it does |
|-----------|-------|--------------|
| Shards | `pair_sampler.shard_count`, `shard_index` | Cuts 1 case into N jobs |
| Resume | `runtime.resume`, `runtime.resume_chunk_rows` | Keeps the work of a preempted job |
| Prebuilt pairs | `pair_sampler.pairs_path` | Draws the pair table once, not in each job |
| Scratch cleanup | `scripts/prepare_scratch.sh` | Makes room on the node before the model loads |

### Shards

A case of 1,100,000 rows needs about 165 GPU-hours for qwen3.5-9b. It does not
fit 1 job, and its prompts alone fill the memory of 1 job.

Every shard uses the SAME full pair table, then keeps every N-th canonical
pair. Thus the shards partition the table exactly, and no shard reads another
shard. `pair_sampler.pairs_path` says where the table comes from (see
**Prebuilt pair tables**).

The cut goes by CANONICAL PAIR, not by row. A canonical pair holds 2
counterbalanced presentations and any repeat draws, and the repeat diagnostic
compares those rows against each other. A cut by row would put the halves of a
comparison in 2 jobs and make the diagnostic empty.

```bash
python -m dagspaces.urbanpairvqa.cli -m pipeline=pairwise_schools_mvp \
    pair_sampler.max_pairs=1000000 \
    pair_sampler.shard_count=92 pair_sampler.shard_index=range(0,92)
```

### Prebuilt pair tables

A draw of 1,100,000 pairs costs about 195 seconds. The draw is deterministic,
thus every shard of a case made the SAME table and then threw away 91/92 of it.
Over the 966 jobs that is about 51 GPU-hours, and a preemption made a job pay
it again.

The table does not depend on the model. Thus 1 file for each case serves the
644 qwen jobs AND the 322 gemma jobs.

`scripts/prebuild_pair_tables.py` draws the 7 tables in parallel, about 215
seconds in total, and writes:

| File | Holds |
|------|-------|
| `<case>_pairs.parquet` | The table, 64 MB to 213 MB |
| `<case>_pairs.json` | The sampler settings that drew it |

A shard reads the parquet in about 1 to 6 seconds. Measured on the same job:
job start to prompt prep fell from about 172 seconds to about 84 seconds.

**Warning:** a table from other settings must not change the science quietly.
The runner compares the sidecar against its own config and STOPS when they
differ, and it names the settings that differ. A missing file also stops the
job. `pairs_path: null` draws the table in the job, as before.

`scripts/launch_million_battery.sh` runs the prebuild before it submits. The
step is idempotent: a table whose sidecar already matches is kept. Use
`--no-prebuild` to go back to the draw in each job.

```bash
python scripts/prebuild_pair_tables.py --sweep million_proxy_qwen9b \
    --out-dir multirun/pair_tables_1m
```

### Room on the node-local scratch

A stage writes its torch and triton JIT caches to `TMPDIR`, which is
`/scratch/$USER` when the node holds a `/scratch`. On a node whose `/scratch`
is full, torch dies before the model loads:

```
OSError: [Errno 28] No space left on device: '/scratch/mwf62'
```

On 2026-08-18 that killed 5 of 113 tasks. `scripts/prepare_scratch.sh` runs in
the launcher setup, before the free-space test, and removes the JIT caches that
earlier runs left. The launcher then falls back to `/tmp` only when the node is
still short of 10 GB.

**Warning:** the cleanup NEVER touches `/scratch/$USER/venvs/` or
`/scratch/$USER/registry/`. Those are the venv and model mirrors, and they are
the reason the node-local path is fast. It removes only named cache dirs, and
only entries older than 2 days, thus a live job keeps its cache. 1 job for each
node holds the lock, and the others go straight on.

```bash
bash scripts/prepare_scratch.sh --dry-run    # report only
```

### A shard needs more than 180 minutes

`slurm_gpu_preempt` gives 180 minutes. The first pass of 2026-08-18 showed that
most shards need more: 268 of 644 qwen tasks and 23 of 322 gemma tasks ended in
TIMEOUT at 3 hours and 2 minutes.

**SLURM does not requeue a TIMEOUT.** `--requeue` covers a preemption and a node
failure, not a walltime. The array held 268 terminal TIMEOUTs and only 48
pending tasks, thus nothing was going to run them again.

No work was lost. Each shard keeps its resume chunks, and a new job continues
from the chunk that follows the last one.

The rate is nearly the same for every case, 2,200 to 3,700 rows for each hour.
The wall simply sits inside the distribution of the shard time:

| Rater | Rows in a shard | Hours needed | New walltime |
|-------|-----------------|--------------|--------------|
| qwen | 11,957 | 3.2 to 5.3 | 360 min |
| gemma | 23,913 | 3.2 to 6.4 | 480 min |

### How to run the missing shards again

`submit_shards.py` takes 2 new options:

| Option | What it does |
|--------|--------------|
| `--only-missing` | Submits only a shard that holds no final parquet. |
| `--timeout-min` | Sets the walltime, and leaves the shared launcher alone. |

`--only-missing` asks each composed config where its parquet goes, and then
looks for that name with the time removed. The name of a result carries the
time the job started, thus a test of the exact path always fails.

`scripts/resubmit_million_battery.sh` does the whole procedure:

```bash
bash scripts/resubmit_million_battery.sh --dry-run   # count what is missing
bash scripts/resubmit_million_battery.sh             # cancel, then submit
```

**Warning:** cancel the old array before you submit. A task that runs now and a
task that the script submits write to the SAME directory. The cancel costs 1
chunk for each running task, and no more, because a chunk lands with an atomic
rename.

## Resume

**Warning:** before this change the single-GPU path wrote NOTHING until the end.
It held every output in memory and wrote 1 parquet when the last row finished.
A preemption threw away the whole job. The streaming shards that
`recompile_streaming_pairwise.py` reads come from the 2-GPU path only.

With `runtime.resume=true` the stage writes a chunk parquet for each row range
into `<results dir>/resume/`. A job that starts again reads the chunks and
generates only the rest.

The design holds 4 properties:

- The chunk holds the RAW text and the token counts, not the postprocessed row.
  Thus a change to a postprocess function does not make a chunk stale.
- A guard file stores a fingerprint of the prompts. A prompt change, a seed
  change or a shard change makes the stage stop, and not mix 2 runs in 1 file.
- A chunk lands through a temporary name and an atomic rename. Thus a
  preemption during the write leaves a whole file or no file.
- A chunk that is short or damaged is dropped, and its rows run again.

The resume dir carries no timestamp, because a requeued job renders
`${now:...}` again and would otherwise lose its own chunks.

### Where a shard writes

**Warning:** `HYDRA_SWEEP_DIR` moves the MONITOR dir only. It does NOT move the
dir a stage job writes to. Without an override a stage writes under
`multirun/<date>_URBANPAIRVQA/<launch second>/<job num>/`, thus the 2 raters
start in the same second, take the same job numbers, and 1 silently overwrites
the other. Measured on 2026-08-18: a qwen shard and a gemma shard both wrote
`14-50-19/0/outputs/pairwise/schools_mvp_20260818_145028.parquet`.

`scripts/launch_million_battery.sh` sets
`runtime.output_root=<sweep>/<rater>/runs/${hydra:job.num}`, which
`resolve_output_root` reads before it asks Hydra. The rater name separates the
sweeps and the job number separates the shards.

## The 2 sweeps

| Sweep | Rater | Shards for each case | Jobs | GPU-hours | Chunk |
|-------|-------|----------------------|------|-----------|-------|
| `million_proxy_qwen9b` | qwen3.5-9b/instruct | 92 | 644 | about 1,155 | 1,024 rows |
| `million_proxy_gemma12b` | gemma-4-12b/instruct | 46 | 322 | about 539 | 2,048 rows |

The shard counts differ because the raters differ in rate. A shard holds about
110 minutes of work inside a 180-minute walltime, thus it fills about 64%.

Measured on the `gpu` partition, 2026-08-18, on a smoke run of 2,200 rows:

| Rater | Rate | 1 shard | Start-up |
|-------|------|---------|----------|
| qwen3.5-9b | 0.48 to 0.56 s for each row | 12,000 rows, about 112 min | 3 to 4 min |
| gemma-4-12b | about 0.13 s for each row | 24,000 rows, about 52 min | about 3 min |

The start-up was 3 to 4 minutes, and NOT the 13 minutes that
`slurm_gpu_preempt.yaml` warns about for a cold NFS import. Do not depend on
that: a node with a cold page cache pays more. The gemma shard is sized against
the slower rate of 0.25 s, thus it holds a wide margin.

The cost of the whole battery is about 1,400 to 1,700 GPU-hours, and the spread
comes from the gemma rate.

`million_proxy_gemma12b` keeps the 2 requirements of the "looks like" battery:
1 GPU for each job, and `prompt.image_layout=interleaved_labels`.

## How to launch

```bash
bash scripts/launch_million_battery.sh --dry-run
bash scripts/launch_million_battery.sh --smoke      # 1 case, 2 shards
bash scripts/launch_million_battery.sh              # both raters
bash scripts/launch_million_battery.sh --parallel 48
```

**Warning:** run the smoke test first when the venv, a model config, or the
inference path changed. A bad launch wastes about 1,700 GPU-hours.

## How to join the shards

```bash
python scripts/merge_pairwise_shards.py --sweep-dir <sweep>/qwen --dry-run
python scripts/merge_pairwise_shards.py --sweep-dir <sweep>/qwen \
    --out-dir <sweep>/merged_qwen
```

The merge REFUSES to write while a shard is missing, while a shard index
repeats, or while a `pair_id` occurs twice. A run of 91 of 92 shards looks
complete in a row count and is not. Do not pass `--allow-incomplete` to a run
that feeds the results table.

## What 1,000,000 pairs buys, and what it does not

The pair ceiling is not the same for every case. `C(n, 2)` is the number of
canonical unit pairs that exist.

| Case | Units | C(n,2) | 1,000,000 pairs is |
|------|-------|--------|--------------------|
| libraries | 236 | 27,730 | 36 draws of each pair, with fresh images |
| subway safety | 1,990 | 1,979,055 | 51% of every pair that exists |
| parks and plazas | 2,064 | 2,129,016 | 47% |
| schools | 2,287 | 2,614,041 | 38% |
| restaurants | 18,488 | 170,893,828 | 0.6% |
| road quality | 164,780 images | 13.6 billion | a small sample |
| street photography | 164,780 images | 13.6 billion | a small sample |

Libraries uses `allow_replacement: true`, and a library holds 47 photos in the
median. Thus a repeated library pair draws 2 fresh photos and samples the
image variation more deeply. It does not repeat a prompt.

For subway, parks and schools, a 1,000,000-pair draw approaches a census of the
pair space. That is not wrong, but say it: those 3 cases stop being a sample.

**What the extra pairs sharpen.** Each unit gets about 10 times the
comparisons, thus a unit score is about 3.2 times more precise.

**What they do not sharpen.** The headline correlation against the proxy runs
over UNITS. Its sampling error comes from the number of units, not the number
of pairs. Expect that number to move very little.

## Related

- [looks-like-rerun.md](looks-like-rerun.md) — the 100,000-pair battery whose
  prompts these sweeps reuse
- [canonical-run-registry.md](canonical-run-registry.md) — the gate a merged run
  must pass before a notebook reads it
