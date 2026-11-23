# The GPU census of this cluster

`scripts/gpu_census.py` of image-forward measures the GPU supply of a SLURM
partition. See `docs/gpu-census-tool.md` in image-forward for how the tool
works. This page holds the results for the `gpu` partition of this cluster.

The `gpu` partition holds 143 nodes and 27 GPU models. The 1,000,000-pair
battery made the question concrete: we held 47 slots and we asked for 160.

To make the census again, run this from the monocle checkout. The page goes to
the terminal:

```bash
python3 scripts/gpu_census.py report --days 30 --constraint 'a6000|6000ada|a40|a100|l40s'
```

## The check of the replay

The last sample of the replay must equal what `scontrol show node` reports now.
We measured 558 GPUs busy from the replay against 559 from `scontrol`, over 24
GPU models. The 1 GPU of difference is a job that started after the sample.

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
  the time, and the pool is busy 75% of the time. `lil-compute-05` alone did
  it 373 times in 30 days.

Both venvs compile the kernels these cards need. `torch.cuda.get_arch_list()`
gives `sm_100` and `sm_120` in `.venv-3.12` (torch 2.10+cu128) and in
the vLLM 0.25 venv of this repo (torch 2.11+cu129).

**Warning:** test 1 shard on a blackwell node before you change a launcher that
a battery uses. A compiled arch is not proof that every vLLM kernel runs.

## A tag is not the hardware

`bhattacharjee-compute-02` holds 2 blackwell max-q GPUs, but its features name
only `6000ada`. A job that asks for `6000maxq` never reaches those 2 GPUs, and
a job that asks for `6000ada` can land on one.
