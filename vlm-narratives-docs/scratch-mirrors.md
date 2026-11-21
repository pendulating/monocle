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

## The sync on demand (2026-08-20)

Before this date a mirror came only from a hand-run deploy, thus only klara had
one. Every job on the `gpu` partition — 66 nodes — paid a cold NFS import. A
gemma shard measured 158 seconds to load the weights against 60 seconds on
klara, and a preempted task paid it again.

`scripts/activate_stage_venv.sh` now BUILDS the mirror at the start of a job
when the node holds none. Four guards keep it safe:

| Guard | What it stops |
|---|---|
| An flock on `/scratch/$USER/venvs/.autosync.lock` | 10 tasks on 1 node syncing at once. One syncs; the others wait, then use it. |
| The bootstrap tarball must exist | 60 nodes each walking 90k NFS files by rsync |
| 25 GB free on `/scratch` | A sync that fills the disk and kills the job |
| Any failure falls back to NFS | A node problem stopping the stage |

Set `MLLMSCI_SCRATCH_AUTOSYNC=0` to turn it off. `MLLMSCI_SCRATCH_AUTOSYNC_WAIT`
sets how long a task waits for another task's sync (900 seconds by default).

**Warning:** the tarball is per venv, and the name drops the leading dot.
`.venv-mllmsci-vllm025cu129` needs `venv-mllmsci-vllm025cu129.tar.zst`. Until
2026-08-20 only `venv-vllm025cu129.tar.zst` existed — a tarball of a DIFFERENT
venv — thus the sync had no fast path at all. Build the tarball again each time
the venv changes:

```bash
# as a job pinned to a node that already holds a complete mirror
bash scripts/sync_venv_to_scratch.sh --make-tarball \
     /share/ju/matt/mllmsci/.venv-mllmsci-vllm025cu129
```

## The CUDA toolkit (2026-08-20)

The same script exports `CUDA_HOME=/share/ju/matt/cuda-12.9` for every
launcher. FlashInfer needs nvcc 12.9 to build Blackwell kernels, and the system
toolkit is 12.8. See `blackwell-sm120-flashinfer.md`.

**Warning:** the launchers `source ~/.bashrc` BEFORE this script, and that sets
`CUDA_HOME=/usr/local/cuda-12.8`. A test of the form "set it only when it is
empty" never fires on a compute node. Our toolkit wins whenever it is present.

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

## The model sync on demand (2026-08-28)

Before this date only the venv built its own mirror. A model mirror came from
`scripts/sync_model_registry_to_scratch.sh`, and a person ran that script one
time on each node. Only klara and ju ever got one. Thus every job on the `gpu`
partition read the weights off NFS.

`resolve_model_source` now builds the mirror itself. A node that holds none
runs the sync script for that one model, then loads from local disk. The
guards match the venv autosync:

| Guard | Effect |
|---|---|
| The lock in the sync script | 1 job syncs. A second job on the node reads NFS. |
| Free space | The copy needs the model size and 20 percent more. |
| A time limit of 30 minutes | A slow NFS read does not hold the GPU. |
| Every failure falls back | The stage always runs. |

Set `MLLMSCI_MODEL_AUTOSYNC=0` to turn it off.

A mirror of a DIFFERENT source keeps its marker. The code does not touch it,
and the stage reads NFS. This is the same refusal that the sync script makes.

## urbanpairvqa did not read server.env (2026-08-28)

`MLLMSCI_MODEL_REGISTRY` lives in `server.env`. A stage job reads that file
only when its dagspace sets `ensure_dotenv_first=True`. urbanembed,
artifact_gen, urbanvit and urbanspeech set it. urbanpairvqa did not.

Thus the registry root was empty in each urbanpairvqa stage job, and
`resolve_model_source` returned the zoo path with no message. The model mirror
never worked for the largest dagspace, and no log line said so. A 2026-08-28
extraction job showed the cost: 195 seconds of weight load for each engine
start.

`dagspaces/urbanpairvqa/orchestrator.py` now sets `ensure_dotenv_first=True`.

**Warning:** urbanvqa, urbanocr and urbanroamvqa still do not set it. They
have the same fault.

## monocle and the nightly venv (2026-08-22)

The analysis jobs under `monocle/` did NOT use either mirror until this date.
`monocle/monocle.sub` activated `.venv-nightly` off NFS by hand, and
`attn_route.load_eager` read the weights from the zoo path.

The symptom hid itself. A monocle job printed the GPU banner, then nothing at
all, for 10 to 15 minutes. On a busy day the same job started in 46 seconds,
because an earlier monocle job left the venv in the page cache of the node. A
quiet day gave a cold import, and the job looked hung.

| Mirror | What monocle did | Cost |
|---|---|---|
| model registry | read the zoo path | 24 GB off NFS |
| venv | activated the NFS venv | a cold import, about 13 minutes |

The venv is the larger half. `.venv-nightly` holds about 90k small files, and
the import is latency-bound, not bandwidth-bound.

**The fixes.**

1. `monocle.sub` sources `scripts/activate_stage_venv.sh` with
   `MLLMSCI_DRIVER_VENV=/share/ju/matt/mllmsci/.venv-nightly`. monocle
   needs the nightly venv, so the driver venv is named there and NOT taken
   from `server.env`, which names the vllm025 venv.
2. `monocle.sub` sources `server.env`, so `MLLMSCI_MODEL_REGISTRY` is set.
3. `attn_route.load_eager` and `extract.load_model` call
   `resolve_model_source`. `load_eager` is the single load point for every
   monocle GPU entry point except `extract`.

**Warning:** the helper is written for launcher setup blocks, which do not run
under `set -eu`. `monocle.sub` turns the flags off around the source and back
on after it. An unset-variable probe must not kill the job.

**Warning: `source <mirror>/bin/activate` does NOT move the interpreter.**
`sync_venv_to_scratch.sh` is a plain `rsync -a`, so the mirror keeps the
`bin/activate` of the source venv, and that script re-exports the NFS
`VIRTUAL_ENV`. The helper still prints `node-local venv on klara`, because the
marker is valid. The interpreter is the NFS one.

The mechanism that works is `MLLMSCI_STAGE_PYTHON`, which the helper exports
next to the source line. The stage launchers put it on the submitit command
line, and `monocle.sub` now runs `"${MLLMSCI_STAGE_PYTHON:-python}" -m`.

This cost 3 jobs and about 40 minutes to find, because the success message and
the true state disagree. To check a job, read the line that `monocle.sub`
prints:

```
[monocle] interpreter: /scratch/mwf62/venvs/venv-nightly/bin/python
```

The numbers, on 1 monocle job:

| Path | torch import | model load | total to the first result |
|---|---|---|---|
| NFS venv | 155 s, and transformers not done at 420 s | never reached | over 14 min, killed |
| mirror venv + mirror weights | seconds | 35 s | **77 s** |

**monocle is klara-only, by decision (2026-08-22).** `.venv-mirrors/` holds a
bootstrap tarball for `venv-mllmsci-vllm025cu129` and `venv-vllm025cu129`
only. Without `venv-nightly.tar.zst` the autosync path refuses to build the
nightly mirror, so a node other than klara falls back to NFS and pays the cold
import. We do not build the nightly tarball, because monocle runs on klara.

`monocle.sub` sets `#SBATCH --nodelist=klara`, so the decision holds by
itself. Warning: a job that overrides the node runs 13 minutes slower and
gives no error, because the fallback to NFS is silent by design.

To put monocle on a new node, either build the tarball once with
`sync_venv_to_scratch.sh --make-tarball`, which turns the autosync on for
every node, or build that 1 mirror by hand:

```bash
sbatch -p pierson -w klara -c 24 --mem=32G --time=01:00:00 \
  --wrap 'bash scripts/sync_venv_to_scratch.sh \
          /share/ju/matt/mllmsci/.venv-nightly'
```

## The move to /share/ju (2026-10-07)

`/share/pierson` is being retired. The project now runs from
`/share/ju/matt/mllmsci`. The copy at `/share/pierson/matt/mllmsci` is a stale
duplicate; do not launch from it.

The copied venvs held the old path in about 100 `bin/` scripts each (the
shebangs and `VIRTUAL_ENV` in `activate`) and in the editable-install finder
(`__editable___mllmsci_0_1_0_finder.py` mapped `dagspaces` to the pierson
copy). On 2026-10-07 these paths were rewritten in place in all five venvs.
Backups of the old files are in
`/share/ju/matt/.pierson_migration/backups_20261007/mllmsci_venvs_pre_repoint.tgz`.

Every mirror on `/scratch` that was synced before the move has a marker with
`src=/share/pierson/matt/mllmsci/...`. The launcher refuses such a mirror,
because the driver venv is now `/share/ju/matt/mllmsci/...`. Before the fix,
the sync on demand then also refused ("a mirror of a different venv"), and the
stage ran from NFS on every such node, with no error.

`scripts/sync_venv_to_scratch.sh` now accepts a marker whose path differs only
by the `/share/pierson/matt` to `/share/ju/matt` move. It re-syncs that mirror
in place (rsync `--delete` parity) and writes a new marker. Thus the first
stage on each node pays one incremental sync, then uses the mirror again. On
ju-compute-01 on 2026-10-07 that sync took 1,281 s for the 15 GB venv, and the
tarball rebuild took 213 s (5.2 GB).

The tarball `.venv-mirrors/venv-mllmsci-vllm025cu129.tar.zst` was rebuilt on
2026-10-07 from a ju-compute-01 mirror of the rewritten venv. The older
`.venv-mirrors/venv-vllm025cu129.tar.zst` is from 2026-08-12 and holds pierson
paths; no launcher uses it.

To check a node: `cat /scratch/$USER/venvs/venv-mllmsci-vllm025cu129/.sync_complete`.
After the first stage on that node it must show `src=/share/ju/...`. The stage
log prints `[stage_venv] node-local venv on <host>` when the mirror is in use.

`.venv-cambrian` and `.venv-labelstudio` use a uv-managed Python under
`/share/pierson/matt_ai/usr_data/uv/python/` (their `pyvenv.cfg` `home`). These
two venvs stop working when `/share/pierson` goes away, unless that Python is
copied or the venvs are rebuilt.

## Tests

`tests/test_model_registry.py` covers the resolution rules. The important
cases are the refusals: a mirror with no marker, and a marker that names a
different source.
