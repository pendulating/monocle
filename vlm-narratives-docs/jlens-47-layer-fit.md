# The 47-layer multimodal J-lens fit

Started 2026-08-25. It fits a Jacobian lens at EVERY source block of
gemma-4-12B, not at the 8 blocks of the first fit.

## Why more layers

The fitted causal reader (`vlm-narratives-docs/fitted-causal-reader.md`)
reads a patch from `g * h`. It needs a lens at the layer it reads. The first
multimodal fit covered 8 blocks, so a question about another depth had no
answer. This fit covers blocks 0 to 46, with block 47 as the target.

## The shape of the run

| Item | Value |
|---|---|
| source blocks | 0 to 46 (47 of them) |
| target block | 47 |
| images | 128, in 16 shards of 8 |
| `dim_batch` | 4 |
| cost | 450 to 520 s for each image |
| output | `outputs/_monocle/jlens/mm_all47/` |

The layer count is almost free. 1 backward pass gives the gradient at every
recorded block, and the accumulator sits on the CPU. So 47 blocks cost about
the same GPU memory as 8.

## Why the run can lose a node

The `gpu` partition is cluster-wide and preemptable. A long job on it dies.
So the fit writes `fit_ckpt_shard<i>of16.pt` after every 2 images, to a
temporary name and then a rename. A job that dies during a write leaves the
last good checkpoint. On restart the fit reads `next_idx` and goes on.

A preemption thus costs 2 images, about 15 minutes.

## Warning: name the card in `--gres`, not in `--constraint`

A node with 2 card types carries the FEATURE of both. We asked for
`--constraint=a6000|6000ada|...` and SLURM gave an 11 GB 2080 Ti, and the
fit ran out of memory.

Write `--gres=gpu:nvidia_rtx_a6000:1`. The submit script also reads
`nvidia-smi` and stops at once if the card holds less than 40000 MiB.

## The tools

| Tool | Purpose |
|---|---|
| `research/monocle/jlens_fit_mm.py --all-layers` | fit 1 shard |
| `research/monocle/jlens_fit_preempt.sub` | the preemptable submit script |
| `research/scripts/jlens_shard_watchdog.sh` | keep the 16 shards alive |
| `research/monocle/jlens_merge_mm.py` | merge the shards into 1 lens |
| `research/scripts/jlens_auto_merge.sh` | merge when the shards land, then check |
| `research/scripts/jlens_verify_lens.py` | check that a lens file is usable |

### Why the project holds the watchdog

The first watchdog lived in the scratchpad under `/tmp`. A cleanup removed
it and its state while the fit ran. The watchdog now lives in `scripts/`,
and it keeps its state in `outputs/_monocle/jlens/mm_all47/.watchdog/`.

**Warning:** do not put the state of a long run under `/tmp`.

### Why the merge does not use `jlens_fit --merge`

`jlens_fit.py --merge` loads every shard at the same time. At 47 layers 1
shard holds 47 x 3840 x 3840 fp16 values, which is 1.4 GB, so 16 shards need
more than 22 GB.

`jlens_merge_mm.py` reads 1 shard at a time into a float32 accumulator. The
peak is about 4 GB. It also reads a live CHECKPOINT, which holds the sum over
the images that shard finished. So you can merge before every shard is
complete, and still get a correct mean over fewer images.

    python -m research.monocle.jlens_merge_mm \
        --in-dir outputs/_monocle/jlens/mm_all47 \
        --out outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt

Add `--final-only` to refuse a partial shard.

## The check on a fitted shard

`research/scripts/jlens_verify_lens.py` reports the layer count, any layer that is not
finite, and the diagonal mass by depth. The diagonal mass must RISE toward
the output, because a block near the end changes the residual less.

## The result, 2026-08-26

The fit is complete. All 16 shards ran to the end and merged.

| Item | Value |
|---|---|
| file | `outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt` |
| layers | 47 (blocks 0 to 46) |
| images | 128 |
| d_model | 3840 |
| non-finite layers | none |
| all-zero layers | none |
| verdict | PASS |

The merged lens gives:

| Block | diagonal mass |
|---|---|
| 0 | 0.0288 |
| 8 | 0.0140 |
| 16 | 0.0252 |
| 24 | 0.0004 |
| 32 | 0.0031 |
| 40 | 0.0197 |
| 46 | 0.0507 |

The low value at block 24 is not a fault. It agrees with the window that
`r-lens.md` reports: the lens does not read in the middle of the stack.
