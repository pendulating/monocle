---
title: Shared Infrastructure
category: infrastructure
created: 2026-04-06
tags:
  - common
  - infrastructure
  - shared
---

# Shared Infrastructure

All five dagspaces (`urbanvqa`, `urbanocr`, `urbanpairvqa`, `urbanroamvqa`, `urbanembed`) import shared modules from `dagspaces/common/`. This page documents every module in that shared layer.

## Module Overview

| Module | File | Purpose |
|--------|------|---------|
| `config_schema` | `dagspaces/common/config_schema.py` | Pipeline graph dataclasses and topology resolution |
| `orchestrator` | `dagspaces/common/orchestrator.py` | DAG execution context, config utilities, artifact registry, SLURM helpers |
| `runners/base` | `dagspaces/common/runners/base.py` | Abstract base class for stage runners |
| `vllm_inference` | `dagspaces/common/vllm_inference.py` | Direct vLLM inference with multimodal, LoRA, reasoning, data-parallel support |
| `wandb_logger` | `dagspaces/common/wandb_logger.py` | Distributed W&B integration with SLURM-safe in-process mode |
| `stage_utils` | `dagspaces/common/stage_utils.py` | Dotenv loading, JSON extraction, thinking mode resolution |
| `multiprocessing_utils` | `dagspaces/common/multiprocessing_utils.py` | Ray init, SLURM CPU detection, object store sizing |
| `logging_filters` | `dagspaces/common/logging_filters.py` | vLLM log throttling filters |
| `resource_tracker_patch` | `dagspaces/common/resource_tracker_patch.py` | Multiprocessing resource_tracker workaround for SLURM |

---

## config_schema.py

**Path:** `dagspaces/common/config_schema.py`

Defines the typed dataclasses that represent a pipeline DAG parsed from Hydra YAML configs.

### Dataclasses

| Class | Fields | Description |
|-------|--------|-------------|
| `ArtifactSpec` | `key`, `type`, `path`, `optional` | A data artifact (parquet, csv, json, etc.). Type is auto-inferred from file extension via `_infer_artifact_type()`. |
| `SourceSpec` | _(inherits ArtifactSpec)_ | A pipeline source input (e.g., raw parquet dataset). |
| `OutputSpec` | _(inherits ArtifactSpec)_ | A pipeline output artifact. |
| `PipelineNodeSpec` | `key`, `stage`, `depends_on`, `inputs`, `outputs`, `overrides`, `launcher`, `parallel_group`, `max_attempts`, `retry_backoff_s`, `wandb_suffix` | A single stage node in the DAG. |
| `PipelineGraphSpec` | `sources`, `nodes`, `output_root`, `allow_partial` | Top-level pipeline definition containing source artifacts and the node graph. |

### Functions

- **`load_pipeline_graph(cfg: DictConfig) -> PipelineGraphSpec`** -- Parses the `pipeline` section of a Hydra config into a typed `PipelineGraphSpec`. Reads `pipeline.sources`, `pipeline.graph.nodes`, `pipeline.output_root`, and `pipeline.allow_partial`.
- **`resolve_output_root(graph_spec, cfg) -> str`** -- Resolves the output directory with fallback priority: `graph_spec.output_root` > `cfg.runtime.output_root` > Hydra run dir > CWD.
- **`iter_topologically(nodes) -> Iterable[PipelineNodeSpec]`** -- Yields nodes in dependency order using Kahn's algorithm (BFS topological sort). Raises `ValueError` on cycles.
- **`PipelineGraphSpec.topological_order() -> List[str]`** -- Returns node keys in execution order. Detects cycles and unknown dependencies.

---

## orchestrator.py

**Path:** `dagspaces/common/orchestrator.py`

The shared DAG execution engine utilities. Each dagspace's `orchestrator.py` imports from here.

### Dataclasses

| Class | Fields | Description |
|-------|--------|-------------|
| `StageExecutionContext` | `cfg`, `node`, `inputs`, `output_paths`, `output_dir`, `output_root`, `logger` | Everything a stage runner needs to execute. Passed to `StageRunner.run()`. |
| `StageResult` | `outputs`, `metadata` | Return type from stage execution: output file paths and optional metadata dict. |
| `_NoOpLogger` | -- | Drop-in replacement for `WandbLogger` when W&B is disabled. Implements `log_metrics()`, `log_table()`, `set_summary()`, `set_config()`, and context manager protocol. |

### Config Utilities

- **`clone_config(cfg) -> DictConfig`** -- Deep-copies a Hydra config without resolving interpolations.
- **`merge_overrides(base_cfg, overrides) -> DictConfig`** -- Applies per-node overrides to a cloned config via `OmegaConf.update(merge=True)`.
- **`prepare_node_config(base_cfg, node, output_dir) -> DictConfig`** -- Clones config, merges node overrides, and injects `runtime.stage` and `runtime.output_dir`.
- **`build_run_config(cfg, node, inputs, output_paths, dagspace_name) -> Dict`** -- Builds a standardized metadata dict for W&B logging with node/stage/model/pipeline/eval_task/checkpoint info.
- **`_load_launcher_config(cfg, launcher_name, config_dir) -> DictConfig`** -- Loads a launcher YAML from the dagspace's `conf/hydra/launcher/` directory, falling back to `dagspaces/common/conf/hydra/launcher/` if not found locally.

### Stage I/O

- **`_node_inputs(node, registry) -> Dict[str, str]`** -- Resolves input artifact references to file paths via `ArtifactRegistry`.
- **`_node_output_paths(node, registry, output_root) -> Dict[str, str]`** -- Resolves output spec paths relative to `output_root`, creating directories.
- **`_collect_outputs(context, optional) -> Dict[str, str]`** -- Verifies expected outputs exist after stage execution. Checks for fallback extensions (`.csv`, `.pkl`) when parquet is missing. Raises `FileNotFoundError` for missing non-optional outputs.
- **`_save_stage_outputs(out, output_paths)`** -- Saves DataFrame to parquet with automatic fallback to CSV then pickle on serialization errors.

### ArtifactRegistry

Tracks artifact paths by name. Sources are registered as `name`, node outputs as `node_key.output_name`. Used to resolve input references like `classify.result` across pipeline nodes.

### SLURM / Launcher Utilities

- **`_create_submitit_executor(launcher_cfg, job_name, log_folder, use_srun=False)`** -- Creates a `submitit.AutoExecutor` with SLURM parameters from launcher config: `timeout_min`, `partition`, `mem_gb`, `cpus_per_task`, `gpus_per_node`, `nodes`, `tasks_per_node`, `array_parallelism`, plus `additional_parameters` and `setup` scripts.
- **`_clean_slurm_env()`** -- Context manager that temporarily removes all `SLURM*` and `SBATCH*` env vars to prevent incorrect inheritance when nesting submitit jobs.
- **`_submit_slurm_job(executor, execute_fn, context_data, node_key, launcher_name)`** -- Submits a SLURM job with structured error logging.
- **`_sanitize_cuda_visible_devices(reason, env_prefix, cfg)`** -- Probes each GPU in a subprocess, removes broken devices from `CUDA_VISIBLE_DEVICES`, and clamps `tensor_parallel_size` to the valid count.

### Data Loading

- **`_load_parquet_dataset(parquet_path, columns, debug, sample_n) -> pd.DataFrame`** -- Reads a parquet file, applies column renames, and optionally samples `sample_n` rows with seed from `MLLMSCI_SAMPLE_SEED` (default 777).
- **`prepare_stage_input(cfg, dataset_path, stage) -> (df, None, False)`** -- Convenience wrapper that loads a DataFrame from config parameters.

---

## runners/base.py

**Path:** `dagspaces/common/runners/base.py`

Defines the abstract base class that all stage runners must implement.

```python
class StageRunner:
    stage_name: str  # Class attribute identifying the stage type

    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError
```

Each dagspace defines concrete runners inheriting from `StageRunner`. The `stage_name` class attribute is used by the orchestrator's `get_stage_registry()` to map stage types (from pipeline YAML) to runner classes.

**Protocol types:** `StageExecutionContext` and `StageResult` are defined as `typing.Protocol` classes for static type checking, ensuring that runner code is decoupled from the specific dataclass implementations.

---

## vllm_inference.py

**Path:** `dagspaces/common/vllm_inference.py`

Deep dive documented in [[vllm-inference]].

Core function: `run_vllm_inference(df, cfg, preprocess, postprocess, stage_name)` -- runs vLLM batch inference on a DataFrame with multimodal support, LoRA remapping, reasoning model parsing, data-parallel mode, and server-mode inference.

---

## wandb_logger.py

**Path:** `dagspaces/common/wandb_logger.py`

Provides unified W&B logging across all dagspaces with SLURM-safe execution.

### Key Components

| Component | Description |
|-----------|-------------|
| `WandbConfig` | Dataclass extracted from Hydra config via `from_hydra_config()`. Fields: `enabled`, `project`, `entity`, `group`, `tags`, `table_sample_rows`, `table_sample_seed`, plus dagspace-specific knobs (`full_column_stages`, `extra_internal_columns`, `classify_variant_field`, etc.). |
| `WandbLogger` | Context manager for run lifecycle. Methods: `log_metrics()`, `log_table()`, `set_summary()`, `set_config()`. |
| `ensure_local_tmpdir(dagspace_name)` | Redirects `TMPDIR` from `/share` network paths to `/scratch` or `/tmp` to avoid socket-file issues across SLURM nodes. |
| `_apply_wandb_settings_defaults()` | Loads `wandb/settings` (INI format) from repo root before wandb import. Sets `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_BASE_URL` if not already present. |
| `build_wandb_tags(cfg, dagspace_name)` | Auto-generates tags from model source, stage, dagspace, and SLURM job info. |
| `_derive_checkpoint_name(lora_path, model_source)` | Derives a human-readable checkpoint name from LoRA adapter path. |

### SLURM Compatibility

- Service daemon disabled via `WANDB_DISABLE_SERVICE` environment variable
- In-process mode used instead of service daemon
- `WANDB_DIR` defaults to `SLURM_SUBMIT_DIR` or CWD
- Online mode by default (`WANDB_MODE` configurable)

---

## stage_utils.py

**Path:** `dagspaces/common/stage_utils.py`

Utility functions used across all stage implementations.

| Function | Description |
|----------|-------------|
| `ensure_dotenv()` | Loads `.env` and `server.env` from project root into `os.environ`. Walks up from `common/` to find repo root. Does not override existing vars. Idempotent. |
| `maybe_silence_vllm_logs()` | Sets vLLM logger levels from `VLLM_LOGGING_LEVEL` env var (default `WARNING`). Installs `PatternModuloFilter` to throttle "Elapsed time for batch" messages (period from `MLLMSCI_VLLM_LOG_EVERY`, default 10). |
| `to_json_str(value) -> Optional[str]` | Safe JSON serialization with `str()` fallback. |
| `serialize_arrow_unfriendly_in_row(row, columns)` | In-place converts `dict`, `list`, `tuple`, `GuidedDecodingParams`, `SamplingParams` values to JSON strings for Arrow/parquet compatibility. |
| `extract_last_json(text) -> Optional[Dict]` | Extracts the last JSON object from model output text. Tries full parse first, then regex `{...}` blocks from last to first. |
| `sanitize_for_json(value) -> Any` | Recursively converts arbitrary Python values (including numpy arrays, vLLM params) to JSON-serializable builtins. |
| `resolve_thinking_mode(cfg_model, default=True) -> bool` | Single source of truth for thinking mode. Priority: `cfg_model.thinking_mode` > `cfg_model.chat_template_kwargs.enable_thinking` > `default`. Accepts strings like `"on"`, `"off"`, `"auto"`, booleans, and ints. |

---

## multiprocessing_utils.py

**Path:** `dagspaces/common/multiprocessing_utils.py`

Handles Ray initialization and SLURM resource detection.

| Function | Description |
|----------|-------------|
| `_worker_process_setup_hook()` | Ray worker process setup: configures asyncio event loop, applies resource_tracker patch, suppresses multiprocessing warnings. Controlled by `MLLMSCI_SUPPRESS_WARNINGS` env var. |
| `get_suppress_child_warnings(cfg) -> bool` | Reads `cfg.runtime.suppress_child_warnings` (default `True`). |
| `_parse_cpus_on_node(val) -> int` | Parses SLURM `CPUS_ON_NODE` formats: plain int, `N(xM)` multiplied format, comma-separated. |
| `_detect_slurm_cpus() -> Optional[int]` | Reads `SLURM_CPUS_PER_TASK` first, then falls back to `SLURM_CPUS_ON_NODE`. |
| `_compute_object_store_bytes(cfg) -> int` | Computes Ray object store size. Priority: `RAY_OBJECT_STORE_MEMORY` env > `cfg.runtime.object_store_proportion` fraction of RAM > 50% of `cfg.runtime.job_memory_gb`. |
| `ensure_ray_init(cfg, caller)` | Unified Ray initialization: SLURM-aware CPU count, computed object store, runtime env with PYTHONPATH, worker setup hooks, warning suppression. Safe to call multiple times (no-op if already running). |

---

## logging_filters.py

**Path:** `dagspaces/common/logging_filters.py`

Two logging filters for throttling high-frequency vLLM log messages:

| Filter | Description |
|--------|-------------|
| `ModuloFilter(mod=10)` | Passes every Nth log record (below WARNING). WARNING+ always passes. |
| `PatternModuloFilter(mod=10, pattern="Elapsed time for batch")` | Like `ModuloFilter` but only applies to messages matching the pattern string. Non-matching messages always pass. |

Both are installed by `stage_utils.maybe_silence_vllm_logs()` on the root `vllm` logger.

---

## resource_tracker_patch.py

**Path:** `dagspaces/common/resource_tracker_patch.py`

Hardens Python's `multiprocessing.resource_tracker` for SLURM/Ray environments where tracker processes die unexpectedly.

### Two layers of protection

1. **Cleanup function wrapping** (`_wrap_cleanup_funcs`) -- Wraps all registered cleanup functions (for shared memory, semaphores, etc.) to suppress `FileNotFoundError`, `ProcessLookupError`, and `OSError` when resources are already cleaned up.

2. **Custom tracker entrypoint** (`run_patched_resource_tracker`) -- Replaces the default resource tracker's `main()` with a version that:
   - Uses `discard()` instead of `remove()` for UNREGISTER (no `KeyError` on double-unregister)
   - Swallows `KeyError` exceptions from malformed commands
   - Uses the wrapped cleanup functions for final resource release

### Patch installation

`apply_patch()` is idempotent. It monkey-patches `ResourceTracker.ensure_running` to spawn the custom tracker process using the module's own `run_patched_resource_tracker` entrypoint. Called from `_worker_process_setup_hook()` in every Ray worker.

---

## Shared Config Directory

**Path:** `dagspaces/common/conf/`

Contains shared configuration files resolved by all dagspaces via the Hydra searchpath `pkg://dagspaces.common.conf`:

- `model/` -- Model configs for vLLM (symlinked from root as `models/`). See [[config-system#Model Configs]].
- `hydra/launcher/` -- SLURM launcher configs. See [[slurm-deployment]].

---

## See Also

- [[architecture]] -- Overall system architecture
- [[vllm-inference]] -- Deep dive on `vllm_inference.py`
- [[config-system]] -- Hydra configuration system
