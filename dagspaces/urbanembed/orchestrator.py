"""Urban Embed pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanembed-specific stage runner for batch image embedding.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

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
    _collect_outputs,
    _create_submitit_executor,
    _load_launcher_config,
    _node_inputs,
    _node_output_paths,
    _print_status,
    build_run_config,
    common_parent,
    prepare_node_config,
)
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.wandb_logger import WandbLogger

# -- Dagspace-local imports -------------------------------------------------
from .stages.embed import run_embed_stage

try:
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HydraConfig = None

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


# ---------------------------------------------------------------------------
# Embed Stage Runner
# ---------------------------------------------------------------------------

class EmbedRunner(StageRunner):
    stage_name = "embed"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        results_path = context.output_paths.get("results")
        if results_path:
            OmegaConf.update(cfg, "runtime.output_path", results_path, merge=True)

        output_path = run_embed_stage(cfg)

        metadata = {
            "output_path": output_path,
            "model_source": str(cfg.model.model_source),
        }

        if context.logger:
            try:
                context.logger.log_grouped_metrics(
                    "embed",
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
    "embed": EmbedRunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# SLURM job entrypoint
# ---------------------------------------------------------------------------

def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage — designed to be submitted as a SLURM job."""
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
        dagspace_name="urbanembed",
    )

    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            context.logger = logger
            _print_status({"node": node.key, "stage": node.stage, "status": "running"})
            t0 = time.time()
            result = runner.run(context)
            duration = time.time() - t0
            try:
                logger.set_summary(f"{node.stage}/status", "completed")
            except Exception:
                pass
            logger.log_metrics({f"{node.stage}/duration_s": duration})
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
    """Execute embedding pipeline."""
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

            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(f"No runner for stage '{node.stage}' (node '{node.key}')")

                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values()) or os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)
                node_cfg = prepare_node_config(cfg, node, output_dir)

                context = StageExecutionContext(
                    cfg=node_cfg, node=node, inputs=inputs,
                    output_paths=output_paths, output_dir=output_dir,
                    output_root=output_root,
                )
                node_start = time.time()

                if node.launcher:
                    launcher_cfg = _load_launcher_config(cfg, node.launcher, _CONFIG_DIR)
                    log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
                    os.makedirs(log_folder, exist_ok=True)
                    executor = _create_submitit_executor(launcher_cfg, f"URBANEMBED-{node.key}", log_folder)

                    context_data = {
                        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
                        "node": {
                            "key": node.key, "stage": node.stage,
                            "depends_on": node.depends_on, "inputs": node.inputs,
                            "outputs": {k: {"path": v.path, "type": v.type, "optional": v.optional}
                                        for k, v in node.outputs.items()},
                            "overrides": node.overrides, "launcher": node.launcher,
                            "parallel_group": node.parallel_group,
                            "max_attempts": node.max_attempts,
                            "retry_backoff_s": node.retry_backoff_s,
                            "wandb_suffix": node.wandb_suffix,
                        },
                        "inputs": inputs, "output_paths": output_paths,
                        "output_dir": output_dir, "output_root": output_root,
                    }
                    with _clean_slurm_env():
                        job = executor.submit(execute_stage_job, context_data)
                    job_result = job.result()
                    result = StageResult(outputs=job_result["outputs"], metadata=job_result["metadata"])
                else:
                    _print_status({"node": node.key, "stage": node.stage, "status": "running"})
                    result = runner.run(context)

                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage, "outputs": result.outputs,
                    "metadata": result.metadata, "duration_s": duration,
                }
                _print_status({"node": node.key, "status": "completed",
                               "duration_s": round(duration, 3)})

            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                with open(manifest_path, "w") as fh:
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
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
