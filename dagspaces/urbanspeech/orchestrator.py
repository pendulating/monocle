"""Urban Speech pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanspeech-specific stage runners for audio extraction and ASR.

Follows the same two-tier SLURM pattern as urbanvqa/urbanembed:
  - Orchestrator runs on slurm_monitor (lightweight CPU job)
  - Stage nodes submit their own jobs via per-node launcher
    (extract_audio -> CPU launcher, asr -> JU GPU launcher)
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any, Dict, Optional

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
    _clean_slurm_env,
    _create_submitit_executor,
    _load_launcher_config,
    _log_gpu_environment,
    _node_inputs,
    _node_output_paths,
    _print_status,
    _sanitize_cuda_visible_devices,
    build_run_config,
    common_parent,
    prepare_node_config,
)
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.wandb_logger import WandbLogger

# -- Dagspace-local imports -------------------------------------------------
from .stages.extract_audio import run_extract_audio_stage
from .stages.asr import run_asr_stage

# Config dir for launcher resolution
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

class ExtractAudioRunner(StageRunner):
    stage_name = "extract_audio"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        manifest_path = context.output_paths.get("audio_manifest")
        if manifest_path:
            OmegaConf.update(cfg, "runtime.output_path", manifest_path, merge=True)

        output_path = run_extract_audio_stage(cfg)

        metadata = {"output_path": output_path}
        if context.logger:
            try:
                context.logger.log_grouped_metrics("extract_audio", metadata)
            except Exception:
                pass

        outputs = {}
        if manifest_path and os.path.exists(manifest_path):
            outputs["audio_manifest"] = manifest_path
        elif output_path and os.path.exists(output_path):
            outputs["audio_manifest"] = output_path

        return StageResult(outputs=outputs, metadata=metadata)


class AsrRunner(StageRunner):
    stage_name = "asr"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        # Wire input from a prior extract_audio stage (chained pipeline)
        manifest_input = context.inputs.get("audio_manifest")
        if manifest_input:
            OmegaConf.update(cfg, "asr.audio_manifest_path", manifest_input, merge=True)

        results_path = context.output_paths.get("results")
        if results_path:
            OmegaConf.update(cfg, "runtime.output_path", results_path, merge=True)

        output_path = run_asr_stage(cfg)

        metadata = {
            "output_path": output_path,
            "model_source": str(cfg.model.model_source),
        }

        if context.logger:
            try:
                context.logger.log_grouped_metrics(
                    "asr",
                    {"model_source": str(cfg.model.model_source)},
                )
            except Exception:
                pass

        outputs = {}
        if results_path and os.path.exists(results_path):
            outputs["results"] = results_path
        elif output_path and os.path.exists(output_path):
            outputs["results"] = output_path

        return StageResult(outputs=outputs, metadata=metadata)


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "extract_audio": ExtractAudioRunner(),
    "asr": AsrRunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# SLURM job entrypoint
# ---------------------------------------------------------------------------

def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage — designed to be submitted as a SLURM job."""
    from dagspaces.common.stage_utils import ensure_dotenv
    ensure_dotenv()

    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]

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
        cfg=cfg, node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )

    registry = get_stage_registry()
    runner = registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner for stage '{node.stage}' (node '{node.key}')")

    wandb_run_id = node.wandb_suffix or node.key
    run_config = build_run_config(
        cfg, node, context.inputs, context.output_paths,
        dagspace_name="urbanspeech",
    )

    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            context.logger = logger
            _print_status({"node": node.key, "stage": node.stage, "status": "running",
                           "inputs": context.inputs})
            t0 = time.time()
            result = runner.run(context)
            duration = time.time() - t0
            try:
                logger.set_summary(f"{node.stage}/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                f"{node.stage}/duration_s": duration,
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
    """Execute speech recognition pipeline."""
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor",
                     run_config={"type": "pipeline"}) as logger:
        try:
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
                        f"No runner for stage '{node.stage}' (node '{node.key}')"
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
                    _print_status({"node": node.key, "stage": node.stage,
                                   "status": "running", "inputs": inputs})
                    wandb_run_id = node.wandb_suffix or node.key
                    stage_run_config = build_run_config(
                        node_cfg, node, inputs, output_paths,
                        dagspace_name="urbanspeech",
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

    # Determine log folder — prefer Hydra runtime output dir
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

    job_name = f"URBANSPEECH-{node.key}"
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
            while True:
                time.sleep(30)
                check = subprocess.run(
                    ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
                    capture_output=True, text=True, timeout=10,
                )
                if not check.stdout.strip() or check.stdout.strip() not in ("R", "PD", "CG"):
                    break

        # Whether the job just left the queue or was already gone: fast jobs
        # can reach COMPLETED before their result pickle is visible over NFS,
        # making submitit raise UncompletedJobError on a successful job.
        # Poll briefly for the pickle before giving up.
        if hasattr(job, "paths") and hasattr(job.paths, "result_pickle"):
            deadline = time.time() + 120
            while time.time() < deadline:
                if os.path.exists(job.paths.result_pickle):
                    with open(job.paths.result_pickle, "rb") as f:
                        _outcome, _result = pickle.load(f)
                    if _outcome == "success":
                        _print_status({"debug": "recovered_result_from_pickle",
                                       "job_id": job.job_id})
                        return StageResult(
                            outputs=_result["outputs"],
                            metadata=_result["metadata"],
                        )
                    break
                time.sleep(5)

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
