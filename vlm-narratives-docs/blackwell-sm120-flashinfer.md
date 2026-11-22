# Blackwell (sm_120) and the FlashInfer CUDA gate

Date: 2026-08-20. A shard of the 1,000,000-pair battery died on a Blackwell
node in start-up. The message named the wrong cause, and the true cause is a
toolkit version.

## What you see

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

on `jjs533-compute-03`, whose card is an RTX PRO 6000 Blackwell — sm_120, far
above sm75. The task FAILS. It does not requeue: `--requeue` covers a
preemption and a node failure, not an exit code.

## What really happens

| Step | Where |
|---|---|
| FlashInfer maps sm 12.0 to the family target `compute_120f` | `compilation_context.py::_normalize_cuda_arch` |
| That target needs nvcc 12.9 or later, thus it raises | same function |
| `__init__` catches the error and only WARNS | `CompilationContext.__init__` |
| The target-arch set is then EMPTY | same |
| The later check sees no eligible arch and reports sm75 | `jit/core.py::check_cuda_arch` |

The card is never the problem. A card below sm80 is what the message
describes, and this card is above it.

Measured with the system toolkit (12.8):

| Target | nvcc 12.8 |
|---|---|
| `compute_120,code=sm_120` | builds |
| `compute_120a,code=sm_120a` | builds |
| `compute_120f,code=sm_120f` | `nvcc fatal: Unsupported gpu architecture` |

Thus 12.8 builds Blackwell kernels, but NOT the family variant FlashInfer asks
for.

## The fix

`scripts/install_cuda_129_toolkit.sh` assembles a CUDA 12.9 toolkit from the
NVIDIA redistributable tarballs into `/share/ju/matt/cuda-12.9`. It needs
no root and takes about 290 MB: nvcc, cudart, curand, cccl, nvrtc, profiler
API. libcublas stays out at 934 MB — the sampling kernels do not include it.

`scripts/activate_stage_venv.sh` exports that path as `CUDA_HOME` for every
launcher, thus every node compiles the same way.

Verified 2026-08-20:

| Arch | Result |
|---|---|
| `12.0f` (Blackwell) | builds and loads |
| `8.6` (A6000) | builds and loads |

The shared JIT cache under `.cache/flashinfer/0.6.13/` now holds `80`, `86`,
`89` and `120f`, thus a node reuses the build instead of making its own.

The toolkit also MATCHES torch, which is built against CUDA 12.9. The 12.8
system toolkit never did.

## Two traps

**The wheel is not a toolkit.** `nvidia-cuda-nvcc-cu12` 12.9.86 on PyPI ships
`ptxas` ONLY, with no nvcc. The venv also holds `nvidia-cuda-nvcc` 13.2, whose
nvcc cannot read the CUDA 12 headers: "CUDA compiler and CUDA toolkit headers
are incompatible". Take the redistributable tarball.

**`~/.bashrc` sets `CUDA_HOME` first.** The launcher setup blocks source
`~/.bashrc` before `activate_stage_venv.sh`, and that sets
`CUDA_HOME=/usr/local/cuda-12.8`. A test of the form
`[ -z "${CUDA_HOME:-}" ] && export CUDA_HOME=...` therefore NEVER fires on a
compute node, and the bug stays. Our toolkit wins whenever it is present.
`IMAGE_FORWARD_CUDA_HOME` in `server.env` names the toolkit. Since phase 3d of
`repo-cleanup-plan.md` the script has no default path: when the key is not set,
the job uses the CUDA of the node.

## What it buys

The 4 Blackwell nodes carry 24 GPUs and are the best nodes in the pool:

| Pool | GPUs | Preempted | Median wait |
|---|---|---|---|
| blackwell / 6000maxq | 24 | 0.1% | 1.7 min |
| the older 5 tags | 396 | 5.0% | 11.0 min |

`slurm_gpu_preempt` keeps `blackwell|6000maxq` in its constraint. Do not drop
the tags again — fix the toolkit instead.

## Also note

The libraries sit in `lib/` in the redistributable tarballs, and build systems
link `-L$CUDA_HOME/lib64`. The install script adds a `lib64 -> lib` symlink.
Without it every object compiles and the LINK fails with "cannot find
-lcudart".
