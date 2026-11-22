# Models and node-local mirrors

## The model configs

A model config in `dagspaces/common/conf/model/` gives the model and the vLLM
engine settings:

```yaml
model_source: "Qwen/Qwen2.5-VL-3B-Instruct"   # a Hugging Face ID or a local path
engine_kwargs:
  max_model_len: 8192
  gpu_memory_utilization: 0.85
  tensor_parallel_size: 1
batch_size: 16
```

| Key | Content |
|-----|---------|
| `model_source` | A Hugging Face ID, or a folder of weights |
| `engine_kwargs` | The arguments of `vllm.LLM` |
| `batch_size` | The rows of one `llm.chat` call |

Select a config with `model=<name>`, or change one key with
`model.model_source=/path/to/model`.

A Hugging Face ID downloads the weights to `HF_HOME` at the first run. On a
cluster, download the weights one time to a shared folder, and set
`IMAGE_FORWARD_MODEL_ZOO` to it. A config then writes:

```yaml
model_source: ${oc.env:IMAGE_FORWARD_MODEL_ZOO}/Qwen2.5-VL-3B-Instruct
```

`scripts/download_mllm_model.py` downloads a model into the zoo.

## Why a mirror

A venv with vLLM holds about 90,000 small files. On NFS, an import walks the
tree file by file, thus the import time depends on the latency, not on the
bandwidth. We measured these times on one node:

| Import of torch, vLLM and FlashInfer | Time |
|--------------------------------------|------|
| From the NFS venv | 171 s |
| From a node-local copy | 10.5 s |

A vLLM stage starts 3 processes, thus a mirror saves about 8 minutes for each
stage. The weights have the same problem on a smaller scale.

**Note:** a symlink does not help. The bytes must be on the local disk.

## The parts

| Part | File | Role |
|------|------|------|
| Venv mirror | `scripts/sync_venv_to_scratch.sh` | Copies the venv to `/scratch/$USER/venvs/<name>` |
| Activation | `scripts/activate_stage_venv.sh` | Selects the mirror in the job shell |
| Model mirror | `scripts/sync_model_registry_to_scratch.sh` | Copies models to `IMAGE_FORWARD_MODEL_REGISTRY` |
| Model lookup | `dagspaces/common/model_registry.py` | Sends the weight reads to the mirror |

## The safety marker

Each mirror has a `.sync_complete` file with the line `src=<source path>`. The
sync scripts write it last. A job uses a mirror only when the marker exists and
names the same source. Thus:

- A job does not use a partial copy. A partial copy has no marker.
- A job does not use a copy of a different venv or model.

When a job has a doubt, it uses the NFS path. The result is the same, but the
start is slower.

## Deploy a mirror

Run these commands on each node, one time. Run them again after a change to
the venv.

```bash
bash scripts/sync_venv_to_scratch.sh
bash scripts/sync_venv_to_scratch.sh --make-tarball   # on the first node only
bash scripts/sync_model_registry_to_scratch.sh Qwen2.5-VL-3B-Instruct
```

The tarball makes the copy to the next nodes fast: one sequential read in
place of many small reads.

## The sync on demand

A job can make the mirror itself, when the node has none:

| Mirror | When | Guards |
|--------|------|--------|
| Venv | `activate_stage_venv.sh`, at the job start | A lock (one job copies), the tarball must exist, 25 GB free |
| Model | `resolve_model_source`, before the weight load | A lock, the model size plus 20 percent free, 30 minutes maximum |

Each failure falls back to NFS. Set `IMAGE_FORWARD_SCRATCH_AUTOSYNC=0` to stop
the venv sync on demand.

## How a job selects the venv mirror

1. The submission puts `IMAGE_FORWARD_DRIVER_VENV=<the venv of the monitor>` in
   the job setup. The job interpreter is `${IMAGE_FORWARD_STAGE_PYTHON:-<the python of the monitor>}`.
2. The launcher sources `scripts/activate_stage_venv.sh`.
3. The script finds the mirror of the driver venv and checks its marker. When
   the marker agrees, it sets `IMAGE_FORWARD_STAGE_PYTHON`.
4. The job starts the stage with that python.

The stage log tells you which path the job took:

```
[stage_venv] node-local venv on node01: /scratch/me/venvs/venv (...)
[stage_venv] no matching scratch mirror on node01 (driver venv: ...); the stage runs from NFS
```

**Warning:** start the monitor from the venv that `server.env` names. The job
looks for the mirror of the monitor venv. With a different venv, no marker
agrees and each stage reads NFS.

**Warning:** `slurm_monitor` does not use the mirror. If the monitor used it,
the monitor venv would be a `/scratch` path, and no stage node would find a
mirror for it.

## The model mirror

`IMAGE_FORWARD_MODEL_REGISTRY` names the mirror root on each node. With no
value, the model mirror is off.

- The lookup changes the path only at the load: the vLLM engine arguments and
  the transformers fallback. The config, W&B and the run metadata keep the zoo
  path.
- The mirror keeps the folder name of the zoo model.
- The lookup does not check that a mirror is current. If you replace a model
  in the zoo, sync each node again.
