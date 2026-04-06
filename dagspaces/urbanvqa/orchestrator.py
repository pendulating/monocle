"""UrbanVQA pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanvqa-specific pipeline execution logic (W&B group inheritance,
SLURM job dispatch with error recovery, VQA stage runner).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from hydra.core.hydra_config import HydraConfig

# -- Common infrastructure imports ------------------------------------------
from dagspaces.common.config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    OutputSpec,
    load_pipeline_graph,
    resolve_output_root,
)
from dagspaces.common.orchestrator import (
    ArtifactRegistry,
    StageExecutionContext,
    StageResult,
    _NoOpLogger,
    _clean_slurm_env,
    _collect_outputs,
    _create_submitit_executor,
    _load_launcher_config,
    _log_gpu_environment,
    _node_inputs,
    _node_output_paths,
    _print_status,
    _safe_log_table,
    _sanitize_cuda_visible_devices,
    _save_stage_outputs,
    build_run_config,
    clone_config,
    common_parent,
    merge_overrides,
    prepare_node_config,
)
from dagspaces.common.runners.base import StageRunner

# -- Dagspace-local imports -------------------------------------------------
from dagspaces.common.wandb_logger import WandbLogger
from .stages.vqa import run_vqa_stage

# Config dir for prompt/launcher resolution
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


# ---------------------------------------------------------------------------
# VQA Stage Runner
# ---------------------------------------------------------------------------

class VQARunner(StageRunner):
    stage_name = "vqa"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        # Resolve dataset path from inputs or config
        dataset_path = context.inputs.get("dataset")
        if dataset_path:
            OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        else:
            parquet_path_cfg = getattr(cfg.data, "parquet_path", None)
            image_path_cfg = getattr(cfg.data, "image_path", None)
            if not (parquet_path_cfg and str(parquet_path_cfg).strip()) and \
               not (image_path_cfg and str(image_path_cfg).strip()):
                raise ValueError(
                    f"Node '{context.node.key}' requires either 'dataset' input, "
                    f"'data.parquet_path', or 'data.image_path'"
                )

        # Load input data
        parquet_path = getattr(cfg.data, "parquet_path", None) or dataset_path or ""
        df = _load_vqa_input(cfg, str(parquet_path))

        _print_status({
            "vqa_runner": {
                "event": "invoke_run_vqa_stage",
                "rows": len(df) if df is not None else 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

        # Run VQA inference
        out = run_vqa_stage(df, cfg)

        # Save outputs
        if isinstance(out, pd.DataFrame) and "results" in context.output_paths:
            from dagspaces.common.orchestrator import _clean_df_for_parquet
            out_clean = _clean_df_for_parquet(out.copy())
            out_clean.to_parquet(context.output_paths["results"], index=False)
            print(f"[vqa_runner] Saved {len(out_clean)} rows to "
                  f"{context.output_paths['results']}")

        # Log results to W&B
        if isinstance(out, pd.DataFrame) and context.logger:
            prefer_cols = [
                c for c in ["sample_id", "prompt", "answer", "model_response",
                            "image_path", "image_url"]
                if c in out.columns
            ]
            _safe_log_table(
                context.logger, out, "vqa/results",
                prefer_cols=prefer_cols,
                panel_group="inspect_results",
            )

        row_count = len(out) if isinstance(out, pd.DataFrame) else None
        _print_status({
            "vqa_runner": {
                "event": "stage_completed",
                "rows": row_count,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata={"rows": row_count})


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "vqa": VQARunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Data loading (VQA-specific: requires prompt + image columns)
# ---------------------------------------------------------------------------

def _load_vqa_input(cfg: DictConfig, parquet_path: str) -> pd.DataFrame:
    """Load VQA dataset with image column validation."""
    from dagspaces.common.orchestrator import _load_parquet_dataset

    debug = bool(getattr(cfg.runtime, "debug", False))
    sample_n = getattr(cfg.runtime, "sample_n", None)
    columns = dict(getattr(cfg.data, "columns", {})) if getattr(cfg, "data", None) else {}

    if parquet_path and not os.path.isabs(parquet_path):
        parquet_path = os.path.abspath(parquet_path)

    df = _load_parquet_dataset(parquet_path, columns, debug=debug, sample_n=sample_n)
    return df


# ---------------------------------------------------------------------------
# SLURM job entrypoint
# ---------------------------------------------------------------------------

def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage — designed to be submitted as a SLURM job."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]

    # Reconstruct PipelineNodeSpec
    outputs = {}
    for out_key, out_val in node_dict.get("outputs", {}).items():
        outputs[out_key] = OutputSpec.from_config(out_key, out_val)

    node = PipelineNodeSpec(
        key=node_dict["key"],
        stage=node_dict["stage"],
        depends_on=node_dict.get("depends_on", []),
        inputs=node_dict.get("inputs", {}),
        outputs=outputs,
        overrides=node_dict.get("overrides", {}),
        launcher=node_dict.get("launcher"),
        parallel_group=node_dict.get("parallel_group"),
        max_attempts=node_dict.get("max_attempts", 1),
        retry_backoff_s=node_dict.get("retry_backoff_s", 0.0),
        wandb_suffix=node_dict.get("wandb_suffix"),
    )

    _sanitize_cuda_visible_devices(reason=f"job:{node.key}", env_prefix="MLLMSCI")
    _log_gpu_environment(reason=f"job:{node.key}", env_prefix="MLLMSCI")

    context = StageExecutionContext(
        cfg=cfg,
        node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )

    # Get the stage runner
    stage_registry = get_stage_registry()
    runner = stage_registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")

    # Execute with W&B logging
    wandb_run_id = node.wandb_suffix or node.key
    run_config = build_run_config(
        cfg, node, context.inputs, context.output_paths,
        dagspace_name="urbanvqa",
    )

    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            context.logger = logger
            _print_status({"node": node.key, "stage": node.stage, "status": "running",
                           "inputs": context.inputs})
            stage_start = time.time()

            result = runner.run(context)

            duration_s = time.time() - stage_start
            try:
                logger.set_summary(f"{node.stage}/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                f"{node.stage}/duration_s": duration_s,
                f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
            })

            return {"outputs": result.outputs, "metadata": result.metadata}
        except Exception as e:
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
                logger.set_summary(f"{node.stage}/error", str(e))
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig) -> None:
    """Execute the full urbanvqa pipeline."""
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor",
                     run_config={"type": "pipeline"}) as logger:
        try:
            # Set W&B group for child SLURM jobs
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                os.environ["WANDB_GROUP"] = parent_group
                print(f"[orchestrator] Setting WANDB_GROUP={parent_group} for child stages",
                      flush=True)

            graph_spec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)

            registry = ArtifactRegistry()
            for source_key, source in graph_spec.sources.items():
                path = source.path
                if not os.path.isabs(path):
                    path = os.path.abspath(os.path.expanduser(path))
                registry.register_source(source_key, path)

            manifest: Dict[str, Any] = {"output_root": output_root, "nodes": {}}
            stage_registry = get_stage_registry()
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()

            logger.log_metrics({"orchestrator/total_nodes": len(ordered_nodes)})
            try:
                logger.set_config({
                    "orchestrator": {
                        "node_order": ordered_nodes,
                        "total_nodes": len(ordered_nodes),
                    }
                })
            except Exception:
                pass

            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(
                        f"No runner registered for stage '{node.stage}' (node '{node.key}')"
                    )

                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values())
                if not output_dir:
                    output_dir = os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)
                node_cfg = prepare_node_config(cfg, node, output_dir)

                context = StageExecutionContext(
                    cfg=node_cfg, node=node, inputs=inputs,
                    output_paths=output_paths, output_dir=output_dir,
                    output_root=output_root,
                )

                node_start = time.time()

                if node.launcher:
                    # ---- SLURM submission path --------------------------------
                    result = _submit_and_wait(
                        cfg=cfg,
                        node=node,
                        node_cfg=node_cfg,
                        inputs=inputs,
                        output_paths=output_paths,
                        output_dir=output_dir,
                        output_root=output_root,
                        parent_group=parent_group,
                    )
                else:
                    # ---- Local execution path ---------------------------------
                    _print_status({"node": node.key, "stage": node.stage,
                                   "status": "running", "inputs": inputs})
                    wandb_run_id = node.wandb_suffix or node.key
                    stage_run_config = build_run_config(
                        node_cfg, node, inputs, output_paths,
                        dagspace_name="urbanvqa",
                    )
                    stage_logger = WandbLogger(
                        node_cfg, stage=node.stage,
                        run_id=wandb_run_id, run_config=stage_run_config,
                    )
                    stage_logger.start()
                    context.logger = stage_logger
                    try:
                        _sanitize_cuda_visible_devices(
                            reason=f"node:{node.key}", env_prefix="MLLMSCI"
                        )
                        _log_gpu_environment(
                            reason=f"node:{node.key}", env_prefix="MLLMSCI"
                        )
                        result = runner.run(context)
                        stage_duration = time.time() - node_start
                        try:
                            stage_logger.set_summary(f"{node.stage}/status", "completed")
                        except Exception:
                            pass
                        stage_logger.log_metrics({
                            f"{node.stage}/duration_s": stage_duration,
                            f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
                        })
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage,
                                       "status": "failed", "error": str(exc)})
                        try:
                            stage_logger.set_summary(f"{node.stage}/status", "failed")
                            stage_logger.set_summary(f"{node.stage}/error", str(exc))
                        except Exception:
                            pass
                        raise
                    finally:
                        stage_logger.finish()

                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage,
                    "inputs": inputs,
                    "outputs": result.outputs,
                    "metadata": result.metadata,
                    "duration_s": duration,
                }
                _print_status({
                    "node": node.key, "stage": node.stage, "status": "completed",
                    "duration_s": round(duration, 3), "outputs": result.outputs,
                })

            # Write pipeline manifest
            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
            except Exception:
                pass

            total_duration = time.time() - pipeline_start
            try:
                logger.set_summary("orchestrator/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                "orchestrator/total_duration_s": round(total_duration, 3),
                "orchestrator/nodes_completed": len(manifest["nodes"]),
            })
            _print_status({
                "pipeline": {
                    "output_root": output_root,
                    "nodes": ordered_nodes,
                    "duration_s": round(total_duration, 3),
                    "manifest": manifest_path,
                }
            })
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# SLURM submission with error recovery
# ---------------------------------------------------------------------------

def _submit_and_wait(
    cfg: DictConfig,
    node: PipelineNodeSpec,
    node_cfg: DictConfig,
    inputs: Dict[str, str],
    output_paths: Dict[str, str],
    output_dir: str,
    output_root: str,
    parent_group: Optional[str],
) -> StageResult:
    """Submit a stage as a SLURM job and wait for completion with recovery."""
    _print_status({"node": node.key, "stage": node.stage,
                   "status": "submitting", "launcher": node.launcher,
                   "inputs": inputs})

    launcher_cfg = _load_launcher_config(cfg, node.launcher, _CONFIG_DIR)

    # Determine log folder
    log_folder = None
    try:
        hydra_cfg = HydraConfig.get()
        if hydra_cfg and hydra_cfg.runtime and hydra_cfg.runtime.output_dir:
            log_folder = os.path.join(hydra_cfg.runtime.output_dir,
                                      ".slurm_jobs", node.key)
    except Exception:
        pass
    if not log_folder:
        log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
    log_folder = os.path.abspath(log_folder)
    os.makedirs(log_folder, exist_ok=True)

    job_name = f"URBANVQA-{node.key}"
    executor = _create_submitit_executor(launcher_cfg, job_name, log_folder)

    # Inject W&B group into SLURM setup for child job grouping
    if parent_group:
        try:
            current_setup = list(launcher_cfg.get("setup", []))
            insert_idx = 0
            for i, cmd in enumerate(current_setup):
                if "source" in cmd or "export HYDRA_FULL_ERROR" in cmd:
                    insert_idx = i + 1
            wandb_group_export = f"export WANDB_GROUP={parent_group}"
            if wandb_group_export not in current_setup:
                current_setup.insert(insert_idx, wandb_group_export)
                executor.update_parameters(slurm_setup=current_setup)
        except Exception as e:
            _print_status({"debug": "failed_to_inject_wandb_group", "error": str(e)})

    # Serialize context for submitit
    context_data = {
        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
        "node": {
            "key": node.key,
            "stage": node.stage,
            "depends_on": node.depends_on,
            "inputs": node.inputs,
            "outputs": {
                k: {"path": v.path, "type": v.type, "optional": v.optional}
                for k, v in node.outputs.items()
            },
            "overrides": node.overrides,
            "launcher": node.launcher,
            "parallel_group": node.parallel_group,
            "max_attempts": node.max_attempts,
            "retry_backoff_s": node.retry_backoff_s,
            "wandb_suffix": node.wandb_suffix,
        },
        "inputs": inputs,
        "output_paths": output_paths,
        "output_dir": output_dir,
        "output_root": output_root,
    }

    # Submit
    with _clean_slurm_env():
        job = executor.submit(execute_stage_job, context_data)
    _print_status({"node": node.key, "stage": node.stage,
                   "status": "submitted", "job_id": job.job_id})

    # Wait with error recovery
    try:
        job_result = job.result()
        return StageResult(
            outputs=job_result["outputs"],
            metadata=job_result["metadata"],
        )
    except Exception as exc:
        return _recover_slurm_job(job, node, exc)


def _recover_slurm_job(
    job, node: PipelineNodeSpec, original_exc: Exception
) -> StageResult:
    """Attempt to recover a SLURM job result when submitit reports failure."""
    import pickle

    try:
        check = subprocess.run(
            ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
            capture_output=True, text=True, timeout=10,
        )
        job_state = check.stdout.strip()

        if job_state in ("R", "PD", "CG"):
            _print_status({
                "debug": "job_misreported_as_failed",
                "job_id": job.job_id, "state": job_state,
                "original_error": str(original_exc),
            })
            # Poll until done
            while True:
                time.sleep(30)
                check = subprocess.run(
                    ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
                    capture_output=True, text=True, timeout=10,
                )
                if not check.stdout.strip() or check.stdout.strip() not in ("R", "PD", "CG"):
                    break

            # Read result from pickle
            if hasattr(job, "paths") and hasattr(job.paths, "result_pickle"):
                if os.path.exists(job.paths.result_pickle):
                    with open(job.paths.result_pickle, "rb") as f:
                        _outcome, _result = pickle.load(f)
                    _print_status({"debug": "recovered_result_from_pickle",
                                   "job_id": job.job_id})
                    return StageResult(
                        outputs=_result["outputs"],
                        metadata=_result["metadata"],
                    )

        _print_status({
            "node": node.key, "stage": node.stage,
            "status": "failed", "job_id": job.job_id,
            "error": str(original_exc),
        })
        raise original_exc
    except Exception as inner_exc:
        if isinstance(inner_exc, type(original_exc)) and str(inner_exc) == str(original_exc):
            raise
        _print_status({
            "node": node.key, "stage": node.stage,
            "status": "failed", "job_id": job.job_id,
            "error": f"{original_exc} (recovery failed: {inner_exc})",
        })
        raise original_exc from inner_exc
