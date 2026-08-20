# GPU census — partition `gpu`

Written by `scripts/gpu_census.py report` on 2026-08-18 19:47. The window is 30 days, from 2026-07-19 to 2026-08-18.

Constraint: `a6000|6000ada|a40|a100|l40s`.

## What the partition is

| field        | value     |
|--------------|-----------|
| PriorityTier | 15        |
| PreemptMode  | REQUEUE   |
| MaxTime      | UNLIMITED |
| DefaultTime  | 04:00:00  |
| TotalNodes   | 143       |
| State        | UP        |

A partition of a HIGHER priority tier takes a node back from a lower one. These partitions share the nodes of this one. `scontrol` shows only the partitions that you may use, thus the owner partition that preempts you is often absent from this table:

| partition         | tier | shared nodes | effect                    |
|-------------------|------|--------------|---------------------------|
| pierson           | 20   | 1            | takes the node from you   |
| default_partition | 10   | 64           | you take the node from it |

## What GPUs it holds

| model    | memory | arch      | nodes | GPUs | free now | gres type                                               |
|----------|--------|-----------|-------|------|----------|---------------------------------------------------------|
| a6000    | 48 GB  | ampere    | 39    | 275  | 7        | nvidia_rtx_a6000                                        |
| 6000ada  | 48 GB  | ada       | 14    | 83   | 0        | nvidia_rtx_6000_ada_generation                          |
| a40      | 48 GB  | ampere    | 6     | 16   | 3        | nvidia_a40                                              |
| a100     | 80 GB  | ampere    | 1     | 8    | 0        | nvidia_a100-sxm4-80gb                                   |
| a100     | 80 GB  | ampere    | 3     | 6    | 0        | nvidia_a100_80gb_pcie                                   |
| 6000maxq | 96 GB  | blackwell | 1     | 2    | 1        | nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition |
| l40s     | 48 GB  | ada       | 1     | 2    | 0        | nvidia_l40s                                             |
| a100     | 40 GB  | ampere    | 1     | 2    | 0        | nvidia_a100-pcie-40gb                                   |
| 2080ti   | 11 GB  | turing    | 1     | 2    | 1        | nvidia_geforce_rtx_2080_ti                              |

Total: 64 nodes and 396 GPUs.

## How often a GPU is free

The tool replayed 160001 jobs and sampled the result every 300 seconds. It counts EVERY job on the node, thus the owner of the node is in these numbers too.

| model    | GPUs | mean free | p10 | median | p90 | ≥1 free | ≥8 free | ≥32 free | ≥64 free |
|----------|------|-----------|-----|--------|-----|---------|---------|----------|----------|
| a6000    | 275  | 99.8      | 38  | 84     | 174 | 100%    | 100%    | 94%      | 66%      |
| 6000ada  | 83   | 34.1      | 11  | 32     | 56  | 100%    | 94%     | 50%      | 9%       |
| a40      | 16   | 6.5       | 2   | 7      | 13  | 100%    | 49%     | 0%       | 0%       |
| a100     | 8    | 3.1       | 0   | 2      | 8   | 67%     | 15%     | 0%       | 0%       |
| a100     | 6    | 2.1       | 0   | 1      | 6   | 82%     | 0%      | 0%       | 0%       |
| 6000maxq | 2    | 0.0       | 0   | 0      | 0   | 1%      | 0%      | 0%       | 0%       |
| l40s     | 2    | 0.5       | 0   | 0      | 2   | 31%     | 0%      | 0%       | 0%       |
| a100     | 2    | 0.8       | 0   | 1      | 2   | 58%     | 0%      | 0%       | 0%       |
| 2080ti   | 2    | 1.1       | 0   | 2      | 2   | 58%     | 0%      | 0%       | 0%       |
| POOL     | 396  | 148.1     | 64  | 126    | 236 | 100%    | 100%    | 100%     | 91%      |

Mean occupancy of the pool: 62.6%.

### By hour of the day (local time)

| hour  | mean free | p10 | p90 |                   |
|-------|-----------|-----|-----|-------------------|
| 00:00 | 142.6     | 64  | 197 | ##############    |
| 01:00 | 143.3     | 63  | 188 | ##############    |
| 02:00 | 147.6     | 63  | 213 | ###############   |
| 03:00 | 143.2     | 63  | 238 | ##############    |
| 04:00 | 148.0     | 64  | 238 | ###############   |
| 05:00 | 155.6     | 66  | 213 | ################  |
| 06:00 | 158.1     | 73  | 225 | ################  |
| 07:00 | 165.3     | 81  | 235 | ################# |
| 08:00 | 170.6     | 78  | 238 | ################# |
| 09:00 | 162.5     | 69  | 238 | ################  |
| 10:00 | 154.7     | 73  | 231 | ################  |
| 11:00 | 156.1     | 66  | 227 | ################  |
| 12:00 | 155.3     | 70  | 229 | ################  |
| 13:00 | 157.0     | 78  | 330 | ################  |
| 14:00 | 139.7     | 61  | 247 | ##############    |
| 15:00 | 137.0     | 61  | 207 | ##############    |
| 16:00 | 137.5     | 60  | 215 | ##############    |
| 17:00 | 131.1     | 59  | 191 | #############     |
| 18:00 | 126.1     | 53  | 200 | #############     |
| 19:00 | 130.2     | 54  | 212 | #############     |
| 20:00 | 146.5     | 62  | 212 | ###############   |
| 21:00 | 152.6     | 66  | 209 | ###############   |
| 22:00 | 149.0     | 73  | 203 | ###############   |
| 23:00 | 145.5     | 69  | 192 | ###############   |

## What a job of yours can expect

| GPU jobs | median wait | p90 wait  | preempted | median life before preemption | median job life |
|----------|-------------|-----------|-----------|-------------------------------|-----------------|
| 60359    | 10.8 min    | 284.9 min | 5.0%      | 2m00s                         | 10m41s          |

The nodes that took a job back most often:

| node                | preemptions | model |
|---------------------|-------------|-------|
| rush-compute-03     | 794         | a6000 |
| rush-compute-02     | 513         | a6000 |
| ellis-compute-02    | 238         | a6000 |
| klara               | 180         | a6000 |
| nikola-compute-18   | 93          | a6000 |
| kuleshov-compute-02 | 83          | a6000 |
| sablab-gpu-12       | 72          | a6000 |
| sablab-gpu-11       | 69          | a6000 |
| elor-compute-01     | 67          | a6000 |
| nikola-compute-15   | 66          | a6000 |

## How to read this

- A free GPU holds no job. Your job must still outrank the queue.
- A constraint selects a NODE, not a GPU. A node that matches can hold a second model, and an untyped `--gres=gpu:1` can land on it. The table above shows every model of every node that matched.
- A job that spans more than 1 node gives its GPUs to the nodes in equal parts. This is an approximation.
- The capacity is the capacity of today. A node that the cluster added last week counts over the whole window.
- A job that still waits has no start time, thus the delay table reports less than the truth.
- Build this page again with:

```bash
python3 scripts/gpu_census.py report --days 30 \
      --constraint 'a6000|6000ada|a40|a100|l40s' \
      --out vlm-narratives-docs/gpu-partition-census.md
```
