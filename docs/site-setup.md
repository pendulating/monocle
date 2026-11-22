# Site setup and launchers

A site is one cluster: its SLURM partitions, its paths and its CUDA. The
framework reads the site settings from `server.env`. It needs no change to the
code.

## 1. Write server.env

```bash
cp server.env.example server.env
```

Set the keys for your cluster. Git ignores `server.env`. Put API keys in `.env`,
not in `server.env`.

| Key | Content | Necessary |
|-----|---------|-----------|
| `IMAGE_FORWARD_VENV_ACTIVATE` | The `activate` script of the venv. Each job sources it. | On SLURM |
| `IMAGE_FORWARD_SLURM_PARTITION` | The partition of the jobs | No |
| `IMAGE_FORWARD_SLURM_ACCOUNT`, `_QOS`, `_CONSTRAINT`, `_EXCLUDE` | More sbatch options | No |
| `IMAGE_FORWARD_SLURM_BIN` | The folder of `sbatch`, when a node does not have it on `PATH` | No |
| `IMAGE_FORWARD_JOB_PREFIX` | The first part of each job name. The default is `$USER`. | No |
| `IMAGE_FORWARD_SITE_SETUP` | A shell script that each job sources first (module loads) | No |
| `IMAGE_FORWARD_PROJECT_ROOT` | The repo root. The default is the folder of `server.env`. | No |
| `IMAGE_FORWARD_MODEL_ZOO` | A folder of local model weights | No |
| `IMAGE_FORWARD_MODEL_REGISTRY` | A node-local mirror of the zoo (see [models-and-mirrors.md](models-and-mirrors.md)) | No |
| `IMAGE_FORWARD_CUDA_HOME` | A CUDA toolkit, when the node has no usable copy | No |
| `IMAGE_FORWARD_FFMPEG_LIBDIR` | The FFmpeg libraries for torchcodec | No |
| `WANDB_ENTITY`, `WANDB_PROJECT` | W&B. With no entity, W&B is off. | No |
| `NCCL_*` | The GPU interconnect settings. Set them to 1 on a node with PCIe only. | No |

**Note:** do not use the names `SLURM_PARTITION` and so on. The CLI removes each
`SLURM*` variable before a launch, and SLURM sets them in each job.

`ensure_dotenv()` loads `server.env` and `.env` before Hydra composes a config,
and again in each stage job. `scripts/load_server_env.sh` loads `server.env` in
the job shell. A variable that the shell already sets does not change.

## 2. Check the config

```bash
dagspace vqa -m pipeline=example_images --cfg job | head -40
```

## 3. Run a stage on SLURM

```bash
dagspace vqa -m hydra/launcher=local pipeline=example_images runtime.stage_launcher=slurm_gpu_1x
```

The monitor runs in your shell. It submits one GPU job and waits for it. The
job log is in `multirun/<date>_<EXPERIMENT>/<time>/0/.slurm_jobs/<node>/`.

To run the monitor as a SLURM job too, use `hydra/launcher=slurm_monitor`.

## The generic launchers

The framework has these launchers in `dagspaces/common/conf/hydra/launcher/`:

| Launcher | Job |
|----------|-----|
| `local` | No SLURM: the run stays in this process |
| `slurm_monitor` | A small CPU job for the monitor |
| `slurm_cpu` | A CPU stage |
| `slurm_gpu_1x`, `slurm_gpu_2x`, `slurm_gpu_4x` | A stage with 1, 2 or 4 GPUs |

Each launcher reads the partition and the other sbatch options from
`server.env`. Its setup lines source `scripts/load_server_env.sh`, the site
setup script, the venv and `scripts/activate_stage_venv.sh`, in that order.

A node gives its launcher with `launcher:`. A dagspace can override a launcher
in `dagspaces/<name>/conf/hydra/launcher/`.

## The site configs

A cluster that needs more than `server.env` puts its own launchers in
`site/conf/common/hydra/launcher/`. A launcher there takes the place of the
generic launcher with the same name. The plugin
`hydra_plugins/mllmsci_searchpath/` adds `site/conf/` to the search path.

This repo holds the launchers of the Cornell Tech cluster in `site/`. On a
different cluster, delete `site/conf/` or write your own launchers there. See
`site/README.md`.

## The venv on a cluster

Each job activates the venv of `IMAGE_FORWARD_VENV_ACTIVATE`. Build the venv on
a shared disk:

```bash
bash scripts/build_venv.sh --extra wandb
```

A job on NFS can take minutes to import vLLM. A node-local copy of the venv
makes the start faster. See [models-and-mirrors.md](models-and-mirrors.md).
