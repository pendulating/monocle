# The GPU census tool

`scripts/gpu_census.py` measures the GPU supply of a SLURM partition. It tells
you what GPUs exist, how many are free, how often they are free, how long a job
waits, and how often a job loses its node to a preemption.

The tool needs no library. It calls `sinfo`, `scontrol`, and `sacct`, and the
system `python3` runs it.

## Why it exists

The `gpu` partition holds 143 nodes and 27 GPU models. A launcher picks a
subset with a feature constraint, but nothing told us how large that subset is,
or how much of it is free at the hour we launch. The 1,000,000-pair battery
made the question concrete: we hold 47 slots and we asked for 160.

## The subcommands

| Command | What it gives |
|---------|---------------|
| `inventory` | The GPUs of the partition now: model, memory, architecture, nodes, busy, free. |
| `history` | How many GPUs were free over a window, from a replay of the accounting database. |
| `waits` | The delay from submit to start, and the preemption rate. |
| `report` | All of the above, as a markdown page. |
| `snapshot` | 1 CSV row for each model, from the live state. Made for a cron. |

Every subcommand takes `--partition`, `--constraint`, `--min-vram`, and
`--json`.

`--min-vram GB` selects the nodes by the MEMORY of their GPUs instead of a
hand-written list of models. `inventory` then prints the SLURM constraint that
reaches the same nodes. Use it to test a constraint that you wrote by hand:
the cluster adds a card, and a hand-written list does not know it.

```bash
# What can my launcher constraint reach?
python3 scripts/gpu_census.py inventory --constraint 'a6000|6000ada|a40|a100|l40s'

# How often is it free, and at what hour?
python3 scripts/gpu_census.py history --days 14 --by-hour \
    --constraint 'a6000|6000ada|a40|a100|l40s'

# How long do I wait, and how often am I preempted?
python3 scripts/gpu_census.py waits --days 14

# Which nodes hold a GPU of 40 GB or more, and what constraint reaches them?
python3 scripts/gpu_census.py inventory --min-vram 40

# Write the whole census as a page.
python3 scripts/gpu_census.py report --days 30 \
    --constraint 'a6000|6000ada|a40|a100|l40s' \
    --out vlm-narratives-docs/gpu-partition-census.md
```

`vlm-narratives-docs/gpu-partition-census.md` holds the last census. Build it
again when you want new numbers.

## How the replay works

SLURM keeps no record of the state of a node in the past. But `sacct` keeps
about 90 days of jobs, and each job carries its node list, its allocated TRES,
its start time, and its end time. The tool:

1. Reads every job that touched a node of the partition, with `sacct -N`.
2. Makes an event for each start (`+n` GPUs) and each end (`-n` GPUs).
3. Sorts the events and adds them up, which gives a step function of time.
4. Samples that function every 300 seconds, at the MIDDLE of each period.

A poll loop is thus unnecessary: the tool measures the past on the first run.

**The tool counts EVERY job on the node, not only the jobs of your partition.**
A node of the `gpu` partition also belongs to the partition of its owner, and
the owner is where most of the competition comes from.

### The check that the replay is correct

The last sample of the replay must equal what `scontrol show node` reports now.
We measured 558 GPUs busy from the replay against 559 from `scontrol`, over 24
GPU models. The 1 GPU of difference is a job that started after the sample.

**Warning:** sample at the middle of the period, never at its edge. A job
starts and ends on a whole minute, and 300 divides a whole minute. A sample on
the edge counts a job that ends exactly there as already gone, and the last
sample of the window then reads 0.

## The limits

| Limit | Effect |
|-------|--------|
| A job on more than 1 node | The tool gives its GPUs to the nodes in equal parts. |
| The capacity is the capacity of today | A node that the cluster added last week counts over the whole window. |
| A job that still waits has no start time | The delay table reports less than the truth. |
| A constraint selects a NODE, not a GPU | A node that matches can hold a second model, and an untyped `--gres=gpu:1` can land on it. |
| `scontrol show partition` hides what you may not use | The owner partition that preempts you is often absent from the neighbour table. |
| A constraint with a count, such as `[a6000*2]` | Not supported. The tool stops and tells you. |

## A longitudinal record

The replay covers what `sacct` keeps. To hold a record that outlives it, run
`snapshot` from a cron:

```cron
*/10 * * * * \
  /usr/bin/python3 scripts/gpu_census.py snapshot --quiet \
      --csv multirun/gpu_snapshots.csv
```

The CSV carries 1 row for each model: nodes, GPUs, busy, free, and the number
of nodes that hold no job at all.

## What the first census found

Measured over 30 days to 2026-08-18, for the constraint that
`slurm_gpu_preempt` uses:

| Fact | Value |
|------|-------|
| Nodes the constraint reaches | 64 |
| GPUs on them | 396 |
| Mean occupancy | 62.6% |
| Mean free GPUs | 148 |
| Part of the window with 64 or more free | 91% |
| Preemption rate on `gpu` | 5.0% |
| Median life of a job before a preemption | 2 minutes |
| Median delay from submit to start | 11 minutes |

The supply is thus NOT the limit. About 148 GPUs of the pool held no job on
average, and we held 47. The limit is the queue: the `gpu` partition sits at
priority tier 15, under every owner partition at tier 20.

The daily profile is shallow. The best hour (08:00) gives about 171 free GPUs
and the worst hour (18:00) about 126. A launch time thus changes little.

Preemption is cheap here. 5% of jobs lose the node, and the median loss is 2
minutes of work. This supports the design of `slurm_gpu_preempt`: many small
shards that resume.

## The constraint of `slurm_gpu_preempt` is too narrow

`slurm_gpu_preempt` asks for `a6000|6000ada|a40|a100|l40s`, which its comment
explains as "40 GB or more". The list is written by hand, and the cluster added
larger cards after somebody wrote it. `inventory --min-vram 40` finds:

| Selection | Nodes | GPUs |
|-----------|-------|------|
| The constraint of the launcher | 64 | 396 |
| Every GPU of 40 GB or more | 74 | 454 |

The 58 GPUs that the launcher cannot reach are the largest on the cluster:
b200 (180 GB), h200 (141 GB), 6000maxq and blackwell (96 GB), h100 (94 GB).

They do not all behave the same. Measured over 30 days to 2026-08-18:

| Pool | GPUs | Mean free | Preemption rate | Median wait |
|------|------|-----------|-----------------|-------------|
| `a6000\|6000ada\|a40\|a100\|l40s` (now) | 396 | 148 | 5.0% | 11 min |
| `blackwell\|6000maxq` | 24 | 11.5 | **0.1%** | 1.7 min |
| `b200\|h200\|h100` | 34 | 8.5 | **51.3%** | 87.9 min |

- **`blackwell` and `6000maxq` are worth adding.** 24 GPUs of 96 GB, 8 or more
  free 78% of the time, and almost no preemption. The owners of
  `owens-compute-01`, `owens-compute-02`, and `jjs533-compute-03` took a job
  back 5 times in 30 days.
- **`b200`, `h200`, and `h100` are not.** Their owners take a job back 51% of
  the time, and the pool is busy 75% of the time. `lil-compute-05` alone did it 373 times in 30 days.

Both venvs compile the kernels these cards need. `torch.cuda.get_arch_list()`
gives `sm_100` and `sm_120` in `.venv-3.12` (torch 2.10+cu128) and in
`.venv-mllmsci-vllm025cu129` (torch 2.11+cu129).

**Warning:** test 1 shard on a blackwell node before you change a launcher that
a battery uses. A compiled arch is not proof that every vLLM kernel runs.

### A tag is not the hardware

`bhattacharjee-compute-02` holds 2 blackwell max-q GPUs, but its features name
only `6000ada`. A job that asks for `6000maxq` never reaches those 2 GPUs, and
a job that asks for `6000ada` can land on one. Read the models of a node with
`inventory --by-node`, not its tags.

## Related

- `dagspaces/common/conf/hydra/launcher/slurm_gpu_preempt.yaml` — the launcher
  whose constraint this census measures.
- `vlm-narratives-docs/million-pair-battery.md` — the run that raised the
  question.
- `vlm-narratives-docs/gpu-partition-census.md` — the census itself.
