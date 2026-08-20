# Node-local /scratch mirrors

A stage job starts much more quickly when the venv and the model weights are on
node-local disk. This page tells you how the mirrors work, and how to deploy
them.

## The problem

The venv holds about 90k mostly-small files on NFS. The NFS bandwidth is
adequate (about 170 MB/s sequential), but the per-file round trips control the
speed. A python import walks the tree file-by-file, so it is latency-bound.

Measured on ju-compute-01 on 2026-08-12, with the vLLM 0.25 environment. The
two runs import the same modules, from the same node, in the same order.

| Import of torch + vllm + flashinfer + project deps | Time |
|----------------------------------------------------|------|
| From the NFS venv | 171.1 s |
| From the `/scratch` mirror | 10.5 s |

The mirror is thus about 16 times faster. A vLLM stage spawns 3 processes
(parent, EngineCore, Worker), so each stage saves about 8 minutes. On a node
with a cold page cache the NFS number is much worse: UAIR measured about 13
minutes for each process spawn.

The deploy costs are small against that: the first sync took 341 s, and the
bootstrap tarball is 5.2 GB and took 49 s to build.

vLLM 0.25 makes the problem larger than vLLM 0.19 did. vLLM 0.25 imports
FlashInfer at start, and `flashinfer_cubin` alone is 1.9 GB in about 20k files.
vLLM 0.19 never loaded it.

A symlink does not help. The bytes must be on the local disk.

## The four parts

| Part | File | Role |
|------|------|------|
| Venv mirror | `scripts/sync_venv_to_scratch.sh` | Copies the NFS venv to `/scratch/$USER/venvs/<name>` |
| Activation | `scripts/activate_stage_venv.sh` | Selects the mirror in the launcher setup block |
| Interpreter | `dagspaces/common/orchestrator.py` | Moves the interpreter choice to job runtime |
| Model mirror | `dagspaces/common/model_registry.py` | Sends weight reads to a node-local mirror |

`activate_stage_venv.sh` also prepends the FFmpeg library directory, which
torchcodec needs before `import vllm` on 0.25. Each launcher sources the
script for that reason, and `slurm_monitor.yaml` sets
`MLLMSCI_SKIP_SCRATCH_VENV=1` to keep the FFmpeg path but skip the venv swap.
See [vllm-025-upgrade.md](vllm-025-upgrade.md).

## The safety marker

Each mirror carries a `.sync_complete` file with a line `src=<source path>`.
The sync scripts write this file last. A user of the mirror trusts it only when
the marker exists **and** names the same source. This rule gives two
guarantees:

- A killed or partial sync is never used. There is no marker.
- A mirror of a different venv or a different model is never used. The
  `src=` line does not agree.

On any doubt, the stage falls back to the NFS path and runs as it did before.

## Deploy a mirror

Run these commands one time on each node. Repeat them after you change the
shared venv.

```bash
# on the node (or as a small sbatch job pinned to that node)
bash scripts/sync_venv_to_scratch.sh
bash scripts/sync_venv_to_scratch.sh --make-tarball   # first node only
bash scripts/sync_model_registry_to_scratch.sh
```

The tarball is a bootstrap for the other nodes. A first deploy prefers the
tarball, because one sequential NFS stream takes about 2 minutes. Without a
tarball the script uses a 24-way parallel rsync fan-out, because latency-bound
NFS work scales with the number of streams. Both paths end with a serial
`rsync --delete` pass, which gives exact 1:1 parity with the live venv.

To mirror a model that is not in the canonical set, give the zoo name:

```bash
bash scripts/sync_model_registry_to_scratch.sh Qwen3-VL-30B-A3B-Instruct
```

## How a stage selects the mirror

1. `_create_submitit_executor` puts `export MLLMSCI_DRIVER_VENV=<sys.prefix>`
   first in the setup block. It also sets the submitit interpreter to
   `"${MLLMSCI_STAGE_PYTHON:-<sys.executable>}"`.
2. The launcher setup block sources `scripts/activate_stage_venv.sh`.
3. That script derives the mirror path from the driver venv name, tests the
   marker, and exports `MLLMSCI_STAGE_PYTHON` only when the marker agrees.
4. The shell expands the interpreter at job runtime, on the node that SLURM
   assigned. If there is no valid mirror, the expansion gives the same
   `sys.executable` that submitit would have used.

**Warning:** `slurm_monitor.yaml` does not source `activate_stage_venv.sh`. The
monitor job runs the driver. If the monitor activated the mirror, `sys.prefix`
would become a `/scratch` path, and no stage node could match the marker.

**Warning:** do not put `MLLMSCI_SCRATCH_VENV` in `server.env`. A stage reads
that file with `ensure_dotenv()`, which runs inside python, but
`activate_stage_venv.sh` runs in the SLURM shell before python starts.

**Warning:** start the driver from the same venv that `server.env` names.
`MLLMSCI_DRIVER_VENV` comes from the `sys.prefix` of the driver process, not
from the launcher setup block. If you start the driver from `.venv-3.12` while
the mirrors hold `.venv-vllm025cu129`, the marker does not agree and each stage
falls back to NFS. The result is correct but slow.

To confirm which path a stage took, read the stage log. The script prints one
line of the two below:

```
[stage_venv] node-local venv on klara: /scratch/mwf62/venvs/venv-vllm025cu129 (...)
[stage_venv] no matching scratch mirror on klara (driver venv: ...); the stage runs from NFS
```

## The model registry

`MLLMSCI_MODEL_REGISTRY` in `server.env` gives the registry root. An empty
value switches the feature off, so a machine without a mirror is not affected.

The contract:

- Resolution happens at the load boundary only. The two boundaries are the
  vLLM engine kwargs (`_build_engine_kwargs`) and the transformers fallback.
- The Hydra configs, the W&B records, and the run metadata keep the canonical
  `/share` path.
- The mirror keeps the zoo basename, so each test that reads the path gives
  the same result. This covers the AWQ test, the multimodal test, and the
  gemma4-unified test.
- Zoo models do not change after the download. The mirrors do not test
  freshness. If you replace a model in place, sync each node again.

The canonical set is the urbanpairvqa roster: `Qwen3.5-9B`, `gemma-4-12B-it`,
`Gemma-4-E2B-it`, `Gemma-4-E4B-it`, `Qwen3.5-2B` and `Qwen3.5-4B`. Phi-4 is
absent on purpose, because it gives degenerate output on each pairvqa task.

## Tests

`tests/test_model_registry.py` covers the resolution rules. The important
cases are the refusals: a mirror with no marker, and a marker that names a
different source.
