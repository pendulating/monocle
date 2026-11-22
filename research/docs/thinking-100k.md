# The 100,000-pair thinking runs

Launched 2026-09-08 00:15 with `research/scripts/launch_thinking100k.sh`. Both
raters, all 7 cases, 100,000 pairs for each case, thinking mode, on the
preemptable `gpu` partition.

## What it is

The 10x scale-up of the 10,000-pair trace corpus of 2026-08-14
(`looks_thinking_*`). The sweeps are
`research/conf/pairwisevqa/sweep/thinking100k_gemma12b.yaml` and
`thinking100k_qwen9b.yaml`. Each keeps the thinking contract of its
10,000-pair parent: the reasoning parser, sampling at temperature 1.0,
`allow_not_sure`, and the gemma layout `interleaved_labels`.

| | gemma-4-12b | qwen3.5-9b |
|---|---|---|
| Model config | `gemma-4-12b/instruct_thinking` | `qwen3.5-9b/instruct_thinking` |
| Layout | `interleaved_labels` | `images_then_text` (default) |
| max_tokens | 4,096 | 8,192 |
| Shards for each case | 32 (about 3,440 rows) | 40 (about 2,750 rows) |
| Jobs | 224 | 280 |
| Cost for each case | 73 to 100 GPU-hours | 100 to 140 GPU-hours |
| Cost for the 7 cases | about 640 GPU-hours | 700 to 980 GPU-hours |

## The pairs are the cut of the 1,000,000-pair table

The rows come from `multirun/pair_tables_100k_of_1m`, cut row for row out
of the canonical 1,000,000-pair table. Thus:

- Every image is in the image pack (`runtime.image_pack`), and no shard
  reads /share/ju. A fresh draw takes other photographs of the same units
  and the pack does not hold them. See
  [streaming-and-resume.md](../../docs/streaming-and-resume.md).
- Every row joins the greedy canonical battery on `pair_id` at 100%: the
  same photographs, the same A/B order. A trace sits beside the greedy label
  of its pair. The 10,000-pair corpus did not have this.
- The 10% repeat draws of the parent stay, thus about 110,000 rows for each
  case and a self-consistency estimate inside the thinking arm.

## Where it runs

Arrays 545289 (gemma) and 545331 (qwen). Base dir: `multirun/thinking100k_20260908_001705`. Under it, `gemma/` and `qwen/` hold `runs/<case>__<shard>/`
with the resume chunks and the final parquet of each shard, and the array
logs.

`submit_shards` submits 1 SLURM array for each rater from the login node,
with no monitor jobs. Walltime 480 minutes, throttle 128, chunks of 512
rows. A preemption costs 1 chunk and SLURM requeues the task. A walltime
kill does not requeue: run the missing shards again on the same base dir.

```bash
# progress
ls multirun/thinking100k_20260908_001705/*/runs/*/outputs/pairwise/*_mvp_*.parquet | wc -l   # of 504
# the shards that timed out, on the same base
.venv-mllmsci-vllm025cu129/bin/python -m dagspaces.pairwisevqa.submit_shards \
    --sweep thinking100k_gemma12b --base-dir multirun/thinking100k_20260908_001705/gemma \
    --pairs-dir multirun/pair_tables_100k_of_1m --only-missing --timeout-min 480 \
    --set runtime.image_pack=/share/ju/matt/mllmsci/multirun/pair_tables_1m/image_pack
```

## Can it finish overnight?

The work is 1,350 to 1,650 GPU-hours. At midnight on 2026-09-08 the clean
nodes of the constraint held 382 GPUs with 48 free. Ten hours of the night
finish the work only with 135 to 165 GPUs the whole time. The arrays hold
their place in the queue, every shard resumes, and the runs complete when
the cluster gives the hours. Read the progress line above in the morning.

## Warning: not canonical, and a register would take them

The model strings are the thinking configs and the sampling is not greedy,
thus `_provenance.py` drops them from the proxy table. They are trace runs.
A `register` call that includes this base dir makes them the trace runs of
the registry, because 110,000 rows beat 11,000 in the newest-wins rule. The
4 extraction corpora of 2026-08-28 to 2026-09-08 read the 11,000-row runs.
Register on purpose, and re-run the extraction after.
