"""Shared SLURM submission core and pipeline drivers for dagspaces.

The dagspace orchestrators repeated one execute-stage-job body, one
submit-wait-recover block, and one pipeline loop. This module holds a single
copy of each. A :class:`DagspaceSettings` instance selects the exact behavior
variant of a dagspace: job-name prefix, GPU preflight, W&B summary detail,
recovery mode, manifest fields, and the config directory.

Each orchestrator keeps its public names (``execute_stage_job`` and
``run_experiment``) and delegates to the functions here. submitit pickles the submitted callable by
module path, so the per-dagspace wrapper functions stay the submit targets.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from dagspaces.common.config_schema import (
    PipelineNodeSpec,
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

try:
    from hydra.core.hydra_config import HydraConfig
except ImportError:  # pragma: no cover - hydra is a hard runtime dependency
    HydraConfig = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Behavior switches
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DagspaceSettings:
    """Select the exact behavior variant of one dagspace.

    Attributes mirror the differences between the copied orchestrators:

    dagspace_name:
        Name written into the W&B run_config.
    job_prefix:
        Prefix for SLURM job names, e.g. ``VQA``.
    config_dir:
        Absolute path to the dagspace ``conf/`` directory. The launcher
        config loader resolves prompts and launcher YAML against it.
    registry_provider:
        Returns the dagspace stage registry. Called in the driver process
        and again in each SLURM child process.
    gpu_preflight:
        Run ``_sanitize_cuda_visible_devices`` plus the GPU env log before a
        stage starts. Off for the CPU-only pipelines.
    ensure_dotenv_first:
        Call ``ensure_dotenv()`` at the start of a SLURM child job. On by
        default: the configs read ``server.env`` keys (``IMAGE_FORWARD_MODEL_ZOO``).
    status_includes_inputs:
        Add ``inputs`` to the ``running`` status payloads.
    log_rows_metric:
        Log the ``rows_processed`` metric next to the duration metric.
    full_wandb_summaries:
        True: log the completed summary, and the error summary on failure.
        False: log only the failed status summary (pairvqa/roamvqa variant).
    registered_runner_message:
        Use the ``No runner registered for stage`` message text instead of
        ``No runner for stage``.
    recovery_mode:
        ``basic`` reads the result pickle after the job leaves the queue.
        ``nfs_poll`` also polls for the pickle for up to 120 s and checks
        the outcome, because a fast job can finish before NFS shows the
        pickle (speech variant).
    pipeline_style:
        ``monitor`` runs a per-stage W&B run, GPU preflight, and the
        submit-wait-recover path. ``inline`` submits each node and blocks
        on ``job.result()`` with no recovery (ocr/pairvqa/roamvqa variant).
    announce_wandb_group:
        Print a line when the driver exports ``WANDB_GROUP``.
    log_orchestrator_set_config:
        Send the node-order block to ``logger.set_config``.
    inline_announce_submit:
        Print the ``submitting`` and ``submitted`` statuses in the inline
        style (ocr prints them; pairvqa/roamvqa do not).
    manifest_include_inputs:
        Put ``inputs`` in each manifest node entry.
    """

    dagspace_name: str
    job_prefix: str
    config_dir: str
    registry_provider: Callable[[], Dict[str, StageRunner]]
    gpu_preflight: bool = False
    ensure_dotenv_first: bool = True
    status_includes_inputs: bool = False
    log_rows_metric: bool = False
    full_wandb_summaries: bool = True
    registered_runner_message: bool = False
    recovery_mode: str = "basic"
    pipeline_style: str = "monitor"
    announce_wandb_group: bool = False
    log_orchestrator_set_config: bool = False
    inline_announce_submit: bool = False
    manifest_include_inputs: bool = True


# ---------------------------------------------------------------------------
# Node serialization
# ---------------------------------------------------------------------------

def serialize_node(node: PipelineNodeSpec) -> Dict[str, Any]:
    """Serialize a node to the dict form that travels to the SLURM job."""
    return {
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
    }


def deserialize_node(node_dict: Dict[str, Any]) -> PipelineNodeSpec:
    """Rebuild a :class:`PipelineNodeSpec` from its serialized dict form.

    ``PipelineNodeSpec.from_config`` applies the same field defaults and
    coercions as the field-by-field rebuild that the orchestrators repeated.
    """
    return PipelineNodeSpec.from_config(node_dict["key"], node_dict)


# ---------------------------------------------------------------------------
# Atomic JSON writes
# ---------------------------------------------------------------------------

def write_json_atomic(path: str, payload: Any, *, indent: int = 2) -> None:
    """Write JSON to a temp file, then rename it over the target path.

    A crash mid-write leaves the old file intact instead of a torn file.
    The fsync call pushes the bytes to disk before the rename.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# SLURM child entrypoint
# ---------------------------------------------------------------------------

def stage_registry_for(settings: DagspaceSettings, cfg: DictConfig) -> Dict[str, StageRunner]:
    """Return the dagspace stage registry plus the runners that the pipeline names.

    A project adds a stage without a change to the framework. The pipeline
    config names the runner by import path:

        pipeline:
          stage_runners:
            my_stage: my_project.stages:MyStageRunner

    The value is a ``StageRunner`` subclass or instance. A pipeline runner
    replaces a dagspace runner with the same stage name.
    """
    import importlib

    registry = dict(settings.registry_provider())
    extra = OmegaConf.select(cfg, "pipeline.stage_runners", default=None) or {}
    for stage, path in dict(extra).items():
        module_name, sep, attr = str(path).partition(":")
        if not sep or not attr:
            raise ValueError(
                f"pipeline.stage_runners.{stage}: write 'package.module:ClassName', not {path!r}"
            )
        obj = getattr(importlib.import_module(module_name), attr)
        runner = obj() if isinstance(obj, type) else obj
        if not isinstance(runner, StageRunner):
            raise TypeError(f"pipeline.stage_runners.{stage}: {path} is not a StageRunner")
        registry[str(stage)] = runner
    return registry


def _missing_runner_message(node: PipelineNodeSpec, settings: DagspaceSettings) -> str:
    if settings.registered_runner_message:
        return f"No runner registered for stage '{node.stage}' (node '{node.key}')"
    return f"No runner for stage '{node.stage}' (node '{node.key}')"


def execute_stage_job(
    context_data: Dict[str, Any], settings: DagspaceSettings
) -> Dict[str, Any]:
    """Run one stage inside a SLURM job.

    This is the shared core of the per-dagspace ``execute_stage_job``
    functions. ``settings`` selects the variant details, so each copy keeps
    its own behavior.
    """
    if settings.ensure_dotenv_first:
        from dagspaces.common.stage_utils import ensure_dotenv
        ensure_dotenv()

    cfg = OmegaConf.create(context_data["cfg"])
    node = deserialize_node(context_data["node"])

    if settings.gpu_preflight:
        _sanitize_cuda_visible_devices(reason=f"job:{node.key}", env_prefix="IMAGE_FORWARD")
        _log_gpu_environment(reason=f"job:{node.key}", env_prefix="IMAGE_FORWARD")

    context = StageExecutionContext(
        cfg=cfg, node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )

    runner = stage_registry_for(settings, cfg).get(node.stage)
    if runner is None:
        raise ValueError(_missing_runner_message(node, settings))

    wandb_run_id = node.wandb_suffix or node.key
    run_config = build_run_config(
        cfg, node, context.inputs, context.output_paths,
        dagspace_name=settings.dagspace_name,
    )

    status_payload: Dict[str, Any] = {
        "node": node.key, "stage": node.stage, "status": "running",
    }
    if settings.status_includes_inputs:
        status_payload["inputs"] = context.inputs

    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id,
                     run_config=run_config) as logger:
        try:
            context.logger = logger
            _print_status(status_payload)
            t0 = time.time()
            result = runner.run(context)
            duration = time.time() - t0
            if settings.full_wandb_summaries:
                try:
                    logger.set_summary(f"{node.stage}/status", "completed")
                except Exception:
                    pass
            metrics: Dict[str, Any] = {f"{node.stage}/duration_s": duration}
            if settings.log_rows_metric:
                metrics[f"{node.stage}/rows_processed"] = result.metadata.get("rows", 0)
            logger.log_metrics(metrics)
            return {"outputs": result.outputs, "metadata": result.metadata}
        except Exception as e:
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
                if settings.full_wandb_summaries:
                    logger.set_summary(f"{node.stage}/error", str(e))
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# Submit, wait, recover
# ---------------------------------------------------------------------------

def submit_and_wait(
    cfg: DictConfig,
    node: PipelineNodeSpec,
    node_cfg: DictConfig,
    inputs: Dict[str, str],
    output_paths: Dict[str, str],
    output_dir: str,
    output_root: str,
    parent_group: Optional[str],
    *,
    settings: DagspaceSettings,
    execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> StageResult:
    """Submit one stage as a SLURM job and wait for completion with recovery."""
    _print_status({"node": node.key, "stage": node.stage,
                   "status": "submitting", "launcher": node.launcher,
                   "inputs": inputs})

    launcher_cfg = _load_launcher_config(cfg, node.launcher, settings.config_dir)

    # Prefer the Hydra runtime dir for job logs. Fall back to the pipeline
    # output root when Hydra is not initialized.
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

    executor = _create_submitit_executor(
        launcher_cfg, f"{settings.job_prefix}-{node.key}", log_folder
    )

    # Add the W&B group export to the SLURM setup lines. Child jobs then
    # join the parent W&B group.
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

    context_data = {
        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
        "node": serialize_node(node),
        "inputs": inputs,
        "output_paths": output_paths,
        "output_dir": output_dir,
        "output_root": output_root,
    }

    with _clean_slurm_env():
        job = executor.submit(execute_fn, context_data)
    _print_status({"node": node.key, "stage": node.stage,
                   "status": "submitted", "job_id": job.job_id})

    try:
        job_result = job.result()
        return StageResult(
            outputs=job_result["outputs"],
            metadata=job_result["metadata"],
        )
    except Exception as exc:
        return recover_slurm_job(job, node, exc, mode=settings.recovery_mode)


def recover_slurm_job(
    job,
    node: PipelineNodeSpec,
    original_exc: Exception,
    *,
    mode: str = "basic",
) -> StageResult:
    """Try to recover a SLURM job result when submitit reports failure.

    mode="basic":
        Poll squeue while the job shows R/PD/CG, then read the result
        pickle when it exists.
    mode="nfs_poll":
        As basic, but the pickle read also runs when the job already left
        the queue. The read polls for up to 120 s and accepts only an
        outcome of "success".
    """
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
            # Poll until the job leaves the queue.
            while True:
                time.sleep(30)
                check = subprocess.run(
                    ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
                    capture_output=True, text=True, timeout=10,
                )
                if not check.stdout.strip() or check.stdout.strip() not in ("R", "PD", "CG"):
                    break

            if mode == "basic":
                # Read the result from the pickle.
                if hasattr(job, "paths") and hasattr(job.paths, "result_pickle"):
                    if os.path.exists(job.paths.result_pickle):
                        with open(job.paths.result_pickle, "rb") as f:
                            _outcome, _result = pickle.load(f)
                        # A failed job writes ("error", trace text). Only a
                        # "success" pickle holds the result of the stage.
                        if _outcome == "success":
                            _print_status({"debug": "recovered_result_from_pickle",
                                           "job_id": job.job_id})
                            return StageResult(
                                outputs=_result["outputs"],
                                metadata=_result["metadata"],
                            )

        if mode == "nfs_poll":
            # A fast job can reach COMPLETED before its result pickle is
            # visible over NFS. submitit then raises on a good job. Poll
            # for the pickle for a short time before we give up.
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


# ---------------------------------------------------------------------------
# Pipeline drivers
# ---------------------------------------------------------------------------

def run_pipeline_experiment(
    cfg: DictConfig,
    settings: DagspaceSettings,
    *,
    execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Run the full pipeline for one dagspace.

    ``pipeline_style="monitor"`` selects the two-tier variant with a
    per-stage W&B run and job recovery. ``pipeline_style="inline"``
    selects the compact variant that blocks on ``job.result()``.
    """
    if settings.pipeline_style == "monitor":
        _run_monitor_experiment(cfg, settings, execute_fn)
    elif settings.pipeline_style == "inline":
        _run_inline_experiment(cfg, settings, execute_fn)
    else:
        raise ValueError(f"Unknown pipeline_style: {settings.pipeline_style!r}")


def _register_graph_sources(graph_spec, registry: ArtifactRegistry) -> None:
    for source_key, source in graph_spec.sources.items():
        path = source.path
        if not os.path.isabs(path):
            path = os.path.abspath(os.path.expanduser(path))
        registry.register_source(source_key, path)


def _run_monitor_experiment(
    cfg: DictConfig,
    settings: DagspaceSettings,
    execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Two-tier driver: per-stage W&B run, GPU preflight, job recovery."""
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor",
                     run_config={"type": "pipeline"}) as logger:
        try:
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                os.environ["WANDB_GROUP"] = parent_group
                if settings.announce_wandb_group:
                    print(f"[orchestrator] Setting WANDB_GROUP={parent_group} "
                          f"for child stages", flush=True)

            graph_spec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)

            registry = ArtifactRegistry()
            _register_graph_sources(graph_spec, registry)

            manifest: Dict[str, Any] = {"output_root": output_root, "nodes": {}}
            stage_registry = stage_registry_for(settings, cfg)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()

            logger.log_metrics({"orchestrator/total_nodes": len(ordered_nodes)})
            if settings.log_orchestrator_set_config:
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
                    raise ValueError(_missing_runner_message(node, settings))

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
                    result = submit_and_wait(
                        cfg=cfg,
                        node=node,
                        node_cfg=node_cfg,
                        inputs=inputs,
                        output_paths=output_paths,
                        output_dir=output_dir,
                        output_root=output_root,
                        parent_group=parent_group,
                        settings=settings,
                        execute_fn=execute_fn,
                    )
                else:
                    result = _run_node_locally(
                        context, node, node_cfg, inputs, output_paths,
                        node_start, runner, settings,
                    )

                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                if settings.manifest_include_inputs:
                    manifest["nodes"][node.key] = {
                        "stage": node.stage,
                        "inputs": inputs,
                        "outputs": result.outputs,
                        "metadata": result.metadata,
                        "duration_s": duration,
                    }
                else:
                    manifest["nodes"][node.key] = {
                        "stage": node.stage,
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
                write_json_atomic(manifest_path, manifest)
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


def _run_node_locally(
    context: StageExecutionContext,
    node: PipelineNodeSpec,
    node_cfg: DictConfig,
    inputs: Dict[str, str],
    output_paths: Dict[str, str],
    node_start: float,
    runner: StageRunner,
    settings: DagspaceSettings,
) -> StageResult:
    """Run one node in the driver process with its own W&B run."""
    running_payload: Dict[str, Any] = {
        "node": node.key, "stage": node.stage, "status": "running",
    }
    if settings.status_includes_inputs:
        running_payload["inputs"] = inputs
    _print_status(running_payload)

    wandb_run_id = node.wandb_suffix or node.key
    stage_run_config = build_run_config(
        node_cfg, node, inputs, output_paths,
        dagspace_name=settings.dagspace_name,
    )
    stage_logger = WandbLogger(
        node_cfg, stage=node.stage,
        run_id=wandb_run_id, run_config=stage_run_config,
    )
    stage_logger.start()
    context.logger = stage_logger
    try:
        if settings.gpu_preflight:
            _sanitize_cuda_visible_devices(
                reason=f"node:{node.key}", env_prefix="IMAGE_FORWARD"
            )
            _log_gpu_environment(
                reason=f"node:{node.key}", env_prefix="IMAGE_FORWARD"
            )
        result = runner.run(context)
        stage_duration = time.time() - node_start
        try:
            stage_logger.set_summary(f"{node.stage}/status", "completed")
        except Exception:
            pass
        metrics: Dict[str, Any] = {f"{node.stage}/duration_s": stage_duration}
        if settings.log_rows_metric:
            metrics[f"{node.stage}/rows_processed"] = result.metadata.get("rows", 0)
        stage_logger.log_metrics(metrics)
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
    return result


def _run_inline_experiment(
    cfg: DictConfig,
    settings: DagspaceSettings,
    execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> None:
    """Compact driver: plain submit, block on the job, no recovery."""
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor",
                     run_config={"type": "pipeline"}) as logger:
        try:
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                os.environ["WANDB_GROUP"] = parent_group

            graph_spec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)

            registry = ArtifactRegistry()
            _register_graph_sources(graph_spec, registry)

            manifest: Dict[str, Any] = {"output_root": output_root, "nodes": {}}
            stage_registry = stage_registry_for(settings, cfg)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()

            logger.log_metrics({"orchestrator/total_nodes": len(ordered_nodes)})

            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(_missing_runner_message(node, settings))

                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values()) or \
                    os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)
                node_cfg = prepare_node_config(cfg, node, output_dir)

                context = StageExecutionContext(
                    cfg=node_cfg, node=node, inputs=inputs,
                    output_paths=output_paths, output_dir=output_dir,
                    output_root=output_root,
                )
                node_start = time.time()

                if node.launcher:
                    if settings.inline_announce_submit:
                        _print_status({"node": node.key, "status": "submitting",
                                       "launcher": node.launcher})
                    launcher_cfg = _load_launcher_config(
                        cfg, node.launcher, settings.config_dir
                    )
                    log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
                    os.makedirs(log_folder, exist_ok=True)
                    executor = _create_submitit_executor(
                        launcher_cfg, f"{settings.job_prefix}-{node.key}", log_folder
                    )

                    context_data = {
                        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
                        "node": serialize_node(node),
                        "inputs": inputs, "output_paths": output_paths,
                        "output_dir": output_dir, "output_root": output_root,
                    }
                    with _clean_slurm_env():
                        job = executor.submit(execute_fn, context_data)
                    if settings.inline_announce_submit:
                        _print_status({"node": node.key, "status": "submitted",
                                       "job_id": job.job_id})
                    job_result = job.result()
                    result = StageResult(
                        outputs=job_result["outputs"],
                        metadata=job_result["metadata"],
                    )
                else:
                    _print_status({"node": node.key, "stage": node.stage,
                                   "status": "running"})
                    result = runner.run(context)

                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                if settings.manifest_include_inputs:
                    manifest["nodes"][node.key] = {
                        "stage": node.stage, "inputs": inputs,
                        "outputs": result.outputs, "metadata": result.metadata,
                        "duration_s": duration,
                    }
                else:
                    manifest["nodes"][node.key] = {
                        "stage": node.stage, "outputs": result.outputs,
                        "metadata": result.metadata, "duration_s": duration,
                    }
                _print_status({"node": node.key, "status": "completed",
                               "duration_s": round(duration, 3)})

            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                write_json_atomic(manifest_path, manifest)
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
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# Parameterized stage runners
# ---------------------------------------------------------------------------

InputWiring = Tuple[Tuple[str, str], ...]
MetadataFn = Callable[[DictConfig, Optional[str]], Dict[str, Any]]
LogPayloadFn = Callable[[DictConfig, Dict[str, Any]], Dict[str, Any]]


class PathStageRunner(StageRunner):
    """Runner for a stage function that returns an output path.

    The runner wires each ``(input_key, cfg_key)`` pair from
    ``context.inputs`` into the config, sets ``runtime.output_path`` from
    the stage output, calls the stage function, logs grouped metrics, and
    registers the output file. The output falls back to the path returned
    by the stage function when the expected output file does not exist.
    """

    def __init__(
        self,
        stage_name: str,
        stage_fn: Callable[[DictConfig], Optional[str]],
        *,
        input_wiring: InputWiring = (),
        output_key: str = "results",
        metadata_fn: MetadataFn,
        log_payload_fn: Optional[LogPayloadFn] = None,
    ) -> None:
        self.stage_name = stage_name
        self._stage_fn = stage_fn
        self._input_wiring = tuple(input_wiring)
        self._output_key = output_key
        self._metadata_fn = metadata_fn
        self._log_payload_fn = log_payload_fn

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        for input_key, cfg_key in self._input_wiring:
            value = context.inputs.get(input_key)
            if value:
                OmegaConf.update(cfg, cfg_key, value, merge=True)

        results_path = context.output_paths.get(self._output_key)
        if results_path:
            OmegaConf.update(cfg, "runtime.output_path", results_path, merge=True)

        output_path = self._stage_fn(cfg)

        metadata = self._metadata_fn(cfg, output_path)

        if context.logger:
            try:
                payload = (
                    self._log_payload_fn(cfg, metadata)
                    if self._log_payload_fn
                    else metadata
                )
                context.logger.log_grouped_metrics(self.stage_name, payload)
            except Exception:
                pass

        outputs: Dict[str, str] = {}
        if results_path and os.path.exists(results_path):
            outputs[self._output_key] = results_path
        elif output_path and os.path.exists(output_path):
            outputs[self._output_key] = output_path

        return StageResult(outputs=outputs, metadata=metadata)


class DictResultStageRunner(StageRunner):
    """Runner for a stage function that returns a dict.

    ``output_keys`` maps each registered output name to the key that holds
    its path in the returned dict. The dict also holds optional extra
    outputs, a ``metrics`` dict for grouped logging, and a ``metadata``
    dict for the stage result.
    """

    def __init__(
        self,
        stage_name: str,
        stage_fn: Callable[[DictConfig], Dict[str, Any]],
        *,
        input_wiring: InputWiring = (),
        output_wiring: Tuple[str, str],
        output_keys: Mapping[str, str],
        extra_optional_outputs: Sequence[str] = (),
    ) -> None:
        self.stage_name = stage_name
        self._stage_fn = stage_fn
        self._input_wiring = tuple(input_wiring)
        self._output_wiring = output_wiring
        self._output_keys = dict(output_keys)
        self._extra_optional_outputs = tuple(extra_optional_outputs)

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        for input_key, cfg_key in self._input_wiring:
            value = context.inputs.get(input_key)
            if value:
                OmegaConf.update(cfg, cfg_key, value, merge=True)

        out_path = context.output_paths.get(self._output_wiring[0])
        if out_path:
            OmegaConf.update(cfg, self._output_wiring[1], out_path, merge=True)

        result = self._stage_fn(cfg)

        outputs: Dict[str, str] = {
            out_key: result[result_key]
            for out_key, result_key in self._output_keys.items()
        }
        for key in self._extra_optional_outputs:
            if key in result:
                outputs[key] = result[key]

        if context.logger:
            try:
                context.logger.log_grouped_metrics(
                    self.stage_name, result.get("metrics", {})
                )
            except Exception:
                pass

        return StageResult(outputs=outputs, metadata=result.get("metadata", {}))
