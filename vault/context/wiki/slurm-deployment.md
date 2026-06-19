---
title: SLURM Deployment
category: infrastructure
created: 2026-04-06
updated: 2026-06-18
tags:
  - slurm
  - deployment
  - cluster
  - gpu
---

# SLURM Deployment

MLLMSCI pipelines run on SLURM clusters using `hydra-submitit-launcher` to submit pipeline stages as batch jobs. This page covers the full SLURM integration stack.

## How It Works

1. **User launches** a pipeline from a login node (or interactive SLURM session)
2. **Hydra** composes the config and resolves the pipeline DAG
3. **Orchestrator** topologically sorts pipeline nodes and iterates them
4. For nodes with a `launcher` field, the orchestrator:
   - Loads the launcher config via `_load_launcher_config()`
   - Creates a `submitit.AutoExecutor` via `_create_submitit_executor()`
   - Serializes the `StageExecutionContext` as a dict
   - Submits `execute_stage_job(context_data)` as a SLURM job
   - Polls the job until completion or failure
5. For nodes without a `launcher`, execution happens in-process

---

## server.env

**Path:** `server.env` (project root)

Site-specific environment variables loaded by `ensure_dotenv()` at startup. These are NOT overridden if already set in the environment.

```env
# SLURM partition for job submission
SLURM_PARTITION=pierson

# Project paths
MLLMSCI_PROJECT_ROOT=/share/pierson/matt/mllmsci
MLLMSCI_VENV_ACTIVATE=/share/pierson/matt/mllmsci/.venv/bin/activate

# NCCL settings for PCIe-only machines (no NVLink)
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
NCCL_SHM_DISABLE=1
NCCL_CUMEM_HOST_ENABLE=0
```

### NCCL Settings Explained

The cluster uses RTX A6000 (pierson/klara) and RTX A5000 (ju partition) GPUs connected via PCIe (no NVLink) — both sm_86. Without these settings, NCCL attempts P2P/IB/SHM transfers that fail or deadlock:

| Variable | Value | Reason |
|----------|-------|--------|
| `NCCL_P2P_DISABLE=1` | Disable peer-to-peer | No NVLink/NVSwitch between GPUs |
| `NCCL_IB_DISABLE=1` | Disable InfiniBand | No IB fabric |
| `NCCL_SHM_DISABLE=1` | Disable shared memory | Prevents SHM-related hangs on PCIe |
| `NCCL_CUMEM_HOST_ENABLE=0` | Disable CUDA managed memory for host | Stability on PCIe setups |

---

## Launcher Configs

**Path:** `dagspaces/common/conf/hydra/launcher/`

Shared across all dagspaces via the Hydra searchpath. Dagspace-local `conf/hydra/launcher/` overrides take precedence when present.

### Available Launchers

| Config | GPUs | CPUs | Memory | Timeout | Use Case |
|--------|------|------|--------|---------|----------|
| `ray_local.yaml` | 2 | 8 | -- | -- | Single-machine Ray (no SLURM) |
| `slurm_cpu.yaml` | 0 | varies | varies | varies | CPU-only stages (topic modeling, verification) |
| `slurm_cpu_beefy.yaml` | 0 | more | more | longer | Memory-intensive CPU stages |
| `slurm_gpu_1x.yaml` | 1 | varies | varies | varies | Small models (2B-3B) |
| `slurm_gpu_2x.yaml` | 2 | varies | varies | varies | Medium models (7B), TP=2 |
| `slurm_gpu_3x.yaml` | 3 | varies | varies | varies | Larger models with TP=3 |
| `slurm_gpu_4x.yaml` | 4 | 8 | `${runtime.job_memory_gb}` | 2880 min | Large models (30B MoE), TP=4 |
| `slurm_gpu_6x.yaml` | 6 | varies | varies | varies | Very large models, TP=6 |
| `slurm_gpu_klara_{1,2,4}x.yaml` | 1/2/4 | 8 | `${runtime.job_memory_gb}` | 14400 min | klara node (8x A6000) in pierson partition, pinned via nodelist |
| `slurm_cpu_ju.yaml` | 0 | 4 | 32 GB | 2880 min | JU-partition CPU stages (e.g. urbanspeech audio extraction) |
| `slurm_gpu_ju_{1,2,4}x.yaml` | 1/2/4 | 4/8/16 | `${runtime.job_memory_gb}` | 14400 min | JU partition (ju-compute-01, 4x RTX A5000 24 GB) |
| `slurm_monitor.yaml` | varies | varies | varies | varies | Monitoring/dashboard jobs |

### Launcher Config Structure

Example: `slurm_gpu_4x.yaml`

```yaml
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher
submitit_folder: ${hydra.sweep.dir}/.submitit/%j
timeout_min: 2880          # 48 hours
nodes: 1
tasks_per_node: 1
cpus_per_task: 8
mem_gb: ${runtime.job_memory_gb}
partition: ${oc.env:SLURM_PARTITION,pierson}
array_parallelism: 1
gpus_per_node: 4
name: ${experiment.name}

additional_parameters:
  gres: gpu:4

setup:
  - export HYDRA_FULL_ERROR=1
  - source ~/.bashrc
  - source ${oc.env:MLLMSCI_VENV_ACTIVATE,/share/pierson/matt/mllmsci/.venv/bin/activate}
  # Force system libstdc++ ahead of anaconda's (which `source ~/.bashrc` activates
  # and which lacks GLIBCXX_3.4.32 that flashinfer >=0.6.12 needs). All 12 GPU
  # launchers carry this as of 2026-06-18. See [[troubleshooting]].
  - export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
  - # CUDA device enumeration from SLURM allocation
  - if [ -n "$SLURM_GPUS_ON_NODE" ] && [ "$SLURM_GPUS_ON_NODE" -gt 0 ]; then
      export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SLURM_GPUS_ON_NODE - 1)));
    else unset CUDA_VISIBLE_DEVICES; fi
  - export PYTHONUNBUFFERED=1
  # GPU sanity check
  - nvidia-smi -L || true
  - python -c 'import torch; print("cuda.is_available=", torch.cuda.is_available())' || true
  # W&B configuration
  - unset WANDB_DISABLED
  - export WANDB_SILENT=true
  - export WANDB_DISABLE_SERVICE=true
  # vLLM multiprocessing
  - export VLLM_WORKER_MULTIPROC_METHOD=spawn
  # NCCL for PCIe
  - export NCCL_P2P_DISABLE=1
  - export NCCL_IB_DISABLE=1
  - export NCCL_SHM_DISABLE=1
  - export NCCL_CUMEM_HOST_ENABLE=0
  # CUDA/PyTorch tuning
  - export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
  - export TOKENIZERS_PARALLELISM=false
  - export TORCH_CUDA_ARCH_LIST=8.6    # RTX A6000 = sm_86
  # Ray object store cap
  - export RAY_OBJECT_STORE_MEMORY=100000000000  # ~93 GB
  # Temp directory (local disk, not network)
  - if [ -d /scratch ]; then export TMPDIR=/scratch/$USER;
    else export TMPDIR=/tmp/$USER; fi
  - mkdir -p $TMPDIR
  - export SLURM_CPU_BIND=none
```

### ray_local.yaml

For single-machine execution without SLURM:

```yaml
_target_: hydra_plugins.hydra_ray_launcher.ray_launcher.RayLauncher
ray:
  init:
    address: null
  remote:
    num_cpus: 8
    num_gpus: 2
```

---

## SLURM Env Var Cleanup

**Path:** Each dagspace's `cli.py` (e.g., `dagspaces/urbanvqa/cli.py`)

When launching from an interactive SLURM session, parent SLURM env vars leak into Hydra's submitit launcher and corrupt job tracking / result-pickle resolution. Each CLI entry point strips them:

```python
if not os.environ.get("SUBMITIT_EXECUTOR"):
    for _k in list(os.environ):
        if _k.startswith("SLURM") or _k.startswith("SBATCH"):
            os.environ.pop(_k)
```

The guard `SUBMITIT_EXECUTOR` ensures that inside a submitit-managed SLURM job, the vars are preserved (they are correct, set by the scheduler for that job).

---

## submitit Executor Creation

**Path:** `dagspaces/common/orchestrator.py` -- `_create_submitit_executor()`

Creates a `submitit.AutoExecutor` from launcher config:

```python
def _create_submitit_executor(
    launcher_cfg: DictConfig,
    job_name: str,
    log_folder: str,
    use_srun: bool = False,
) -> submitit.AutoExecutor:
```

### Parameters mapped from launcher config

| Launcher YAML Field | submitit Parameter | Description |
|---------------------|-------------------|-------------|
| `timeout_min` | `timeout_min` | Job walltime in minutes |
| `partition` | `slurm_partition` | SLURM partition name |
| `mem_gb` | `slurm_mem` | Memory request (formatted as `"{N}GB"`) |
| `cpus_per_task` | `slurm_cpus_per_task` | CPU cores per task |
| `gpus_per_node` | `slurm_gpus_per_node` | GPUs requested |
| `nodes` | `slurm_nodes` | Number of nodes |
| `tasks_per_node` | `slurm_tasks_per_node` | Tasks per node |
| `array_parallelism` | `slurm_array_parallelism` | Max concurrent array tasks |
| `additional_parameters` | `slurm_additional_parameters` | Extra SLURM params (e.g., `gres`) |
| `setup` | `slurm_setup` | Shell setup commands run before job |

The executor is created inside a `_clean_slurm_env()` context manager that temporarily removes all `SLURM*` and `SBATCH*` env vars to prevent nesting corruption.

Job names are prefixed: `matt-{job_name}`.

---

## Job Entrypoint

Each dagspace's `orchestrator.py` defines an `execute_stage_job(context_data)` function:

1. Deserializes `StageExecutionContext` from the pickled dict
2. Calls `_sanitize_cuda_visible_devices()` to probe and validate GPUs
3. Initializes W&B logger
4. Looks up the stage runner from `get_stage_registry()`
5. Calls `runner.run(context)` and returns `StageResult`

This function is what submitit pickles and submits to the SLURM scheduler.

---

## SLURM Detection Utilities

**Path:** `dagspaces/common/multiprocessing_utils.py`

| Function | Description |
|----------|-------------|
| `_detect_slurm_cpus() -> Optional[int]` | Reads `SLURM_CPUS_PER_TASK` first, falls back to `SLURM_CPUS_ON_NODE`. Used by `ensure_ray_init()` to set Ray's CPU count. |
| `_parse_cpus_on_node(val) -> int` | Parses `SLURM_CPUS_ON_NODE` formats: `"8"`, `"4(x2)"` (multiplied), `"4,4"` (comma-separated). Returns -1 on parse failure. |

---

## GPU Sanitization

**Path:** `dagspaces/common/orchestrator.py` -- `_sanitize_cuda_visible_devices()`

Before each stage starts, GPUs are validated:

1. Reads `CUDA_VISIBLE_DEVICES`
2. For each device, spawns a subprocess that imports torch, creates tensors, runs a matmul, and synchronizes
3. Removes failed devices from `CUDA_VISIBLE_DEVICES`
4. Updates SLURM GPU env vars (`SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, `SLURM_GPUS_ON_NODE`, etc.)
5. Clamps `tensor_parallel_size` in both env var and Hydra config
6. Raises `RuntimeError` if ALL devices fail (configurable skip via `MLLMSCI_SKIP_GPU_SANITIZE=1`)
7. Logs structured JSON status for debugging

Helper functions:
- `_probe_single_gpu(device) -> Dict` -- Subprocess GPU probe
- `_update_slurm_gpu_envs(valid_devices)` -- Updates SLURM env vars
- `_adjust_tensor_parallel_env(valid_count)` -- Clamps TP env var
- `_log_gpu_environment(reason)` -- Structured status output

---

## Submission Scripts

**Path:** `scripts/`

Standalone SLURM submission scripts for common model configurations:

| Script | Description |
|--------|-------------|
| `scripts/run_qwen30b_vllm.sub` | Qwen3-30B text model |
| `scripts/run_qwen3vl30b_vllm.sub` | Qwen3-VL-30B multimodal model |

These are alternative to Hydra launcher submission for cases where direct `sbatch` is preferred.

---

## Typical Deployment Flow

### 1. Interactive development

```bash
# Small sample, local execution, no SLURM
python -m dagspaces.urbanvqa.cli \
  pipeline=vqa_cyclomedia_scaffolding \
  runtime.debug=true \
  runtime.sample_n=100
```

### 2. Single-node GPU job

```bash
# Submit via Hydra launcher
python -m dagspaces.urbanvqa.cli \
  hydra/launcher=slurm_gpu_4x \
  pipeline=vqa_cyclomedia_scaffolding
```

### 3. Multi-config sweep

```bash
# Hydra multirun: 2 datasets x 2 models = 4 SLURM jobs
python -m dagspaces.urbanvqa.cli -m \
  hydra/launcher=slurm_gpu_2x \
  data=bayflood_1k,bayflood_nearby \
  model=vllm_multimodal_qwen3_vl_2b,vllm_multimodal_qwen3_vl_30b
```

### 4. Pipeline with per-node launchers

When pipeline YAML specifies `launcher` per node, each node gets its own SLURM job:

```yaml
graph:
  nodes:
    classify:
      stage: classify
      launcher: slurm_gpu_2x     # 2 GPUs for small model
    synthesize:
      stage: vqa
      depends_on: [classify]
      launcher: slurm_gpu_4x     # 4 GPUs for large model
```

The orchestrator submits `classify` first, waits for completion, then submits `synthesize`.

---

## Troubleshooting

### Job hangs at exit

vLLM engine workers may not terminate cleanly. The `_shutdown_llm()` function in [[vllm-inference]] addresses this by explicitly shutting down engine core and terminating surviving multiprocessing children.

### GPU probe failures

If `_sanitize_cuda_visible_devices()` reports probe failures, check:
- GPU health via `nvidia-smi`
- CUDA driver compatibility
- Set `MLLMSCI_SKIP_GPU_SANITIZE=1` to bypass (not recommended)

### SLURM env var corruption

Symptoms: job tracking fails, result pickles not found. Cause: launching from inside an existing SLURM allocation. Solution: the CLI entry points strip inherited SLURM vars (see above).

### NCCL errors with tensor parallelism

Ensure all NCCL env vars from `server.env` are set. The launcher setup scripts apply them, but manual runs may need:

```bash
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

---

## See Also

- [[config-system]] -- Full Hydra configuration reference
- [[shared-infrastructure]] -- Shared module documentation
- [[troubleshooting]] -- Known performance issues
