"""Urban Roaming VQA pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanroamvqa-specific stage runner for agent-driven urban navigation.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
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
from dagspaces.common.resource_tracker_patch import apply_patch as _apply_resource_tracker_patch
from dagspaces.common.wandb_logger import WandbLogger

_apply_resource_tracker_patch()

# -- Dagspace-local imports -------------------------------------------------
from .graph.builder import build_street_graph
from .graph.street_graph import compute_graph_diagnostics
from .samplers.seed_sampler import sample_walk_seeds
from .stages.roaming_vqa import run_roaming_vqa_stage

try:
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HydraConfig = None

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _validate_config(cfg: Any, parquet_override: Optional[str] = None) -> None:
    """Fail fast on invalid roaming configs before any expensive work.

    parquet_override skips the data.parquet_path lookup when the metadata
    parquet comes from a pipeline input instead.
    """
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        raise ValueError("Missing 'data' config section")
    roaming_cfg = getattr(cfg, "roaming", None)
    if roaming_cfg is None:
        raise ValueError("Missing 'roaming' config section")

    parquet_path = parquet_override or str(getattr(data_cfg, "parquet_path", "") or "")
    if not parquet_path:
        raise ValueError("data.parquet_path is not set")
    if not os.path.exists(parquet_path):
        raise ValueError(f"Metadata parquet does not exist: {parquet_path}")

    termination_mode = str(getattr(roaming_cfg, "termination_mode", "fixed"))
    if termination_mode not in ("fixed", "independent"):
        raise ValueError(
            f"roaming.termination_mode must be 'fixed' or 'independent', got '{termination_mode}'"
        )

    max_steps = int(getattr(roaming_cfg, "max_steps", 10))
    if max_steps < 1:
        raise ValueError(f"roaming.max_steps must be >= 1, got {max_steps}")

    graph_cfg = getattr(cfg, "graph", None)
    if graph_cfg is not None:
        tolerance = float(getattr(graph_cfg, "bearing_tolerance_deg", 45.0))
        if not (5.0 <= tolerance <= 170.0):
            raise ValueError(
                f"graph.bearing_tolerance_deg must be within [5, 170], got {tolerance}"
            )


def _roaming_diagnostics(traces_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute diagnostics for roaming walks."""
    diagnostics: Dict[str, Any] = {}
    if traces_df is None or traces_df.empty:
        return diagnostics

    total_steps = len(traces_df)
    diagnostics["total_steps"] = total_steps

    walk_ids = traces_df["walk_id"].unique()
    diagnostics["n_walks"] = len(walk_ids)

    walk_lengths = traces_df.groupby("walk_id").size()
    diagnostics["walk_length_mean"] = float(walk_lengths.mean())
    diagnostics["walk_length_median"] = float(walk_lengths.median())
    diagnostics["walk_length_std"] = float(walk_lengths.std()) if len(walk_lengths) > 1 else 0.0
    diagnostics["walk_length_min"] = int(walk_lengths.min())
    diagnostics["walk_length_max"] = int(walk_lengths.max())

    unique_recordings = traces_df["recording_id"].nunique()
    diagnostics["unique_recordings_visited"] = unique_recordings
    diagnostics["revisit_rate"] = _safe_ratio(float(total_steps - unique_recordings), float(total_steps))

    if "face_chosen" in traces_df.columns:
        face_counts = traces_df["face_chosen"].dropna().value_counts()
        total_choices = float(face_counts.sum())
        for face, count in face_counts.items():
            diagnostics[f"face_pref/{face}"] = _safe_ratio(float(count), total_choices)

    if "distance_m" in traces_df.columns:
        walk_distances = traces_df.groupby("walk_id")["distance_m"].sum()
        diagnostics["total_distance_mean_m"] = float(walk_distances.mean())
        diagnostics["total_distance_median_m"] = float(walk_distances.median())

    if "termination_reason" in traces_df.columns:
        last_steps = traces_df.loc[traces_df.groupby("walk_id")["step_n"].idxmax()]
        term_counts = last_steps["termination_reason"].fillna("unknown").value_counts()
        total_term = float(term_counts.sum())
        for reason, count in term_counts.items():
            diagnostics[f"termination/{reason}"] = _safe_ratio(float(count), total_term)

    return diagnostics


# ---------------------------------------------------------------------------
# Roaming VQA Stage Runner
# ---------------------------------------------------------------------------

class RoamingVQARunner(StageRunner):
    stage_name = "roaming_vqa"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        data_cfg = getattr(cfg, "data", {})
        graph_cfg = getattr(cfg, "graph", {})
        roaming_cfg = getattr(cfg, "roaming", {})

        dataset_input = context.inputs.get("dataset")
        metadata_parquet = (
            dataset_input
            or str(getattr(graph_cfg, "metadata_parquet", "") or "")
            or str(getattr(data_cfg, "metadata_parquet", "") or "")
            or str(getattr(data_cfg, "parquet_path", "") or "")
        )

        _validate_config(cfg, parquet_override=metadata_parquet)

        graph = build_street_graph(metadata_parquet, graph_cfg)
        graph_diagnostics = compute_graph_diagnostics(graph)
        print(f"[roaming_runner] Graph diagnostics: "
              f"{json.dumps(graph_diagnostics, sort_keys=True)}", flush=True)

        n_walks = int(getattr(roaming_cfg, "n_walks", 100))
        walk_seed = int(getattr(roaming_cfg, "walk_seed", 42))
        seed_strategy = str(getattr(roaming_cfg, "seed_strategy", "random"))
        initial_face = str(getattr(roaming_cfg, "initial_face", "") or "")
        min_neighbors = int(getattr(roaming_cfg, "min_neighbors", 1))

        sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
        if sample_n:
            n_walks = min(n_walks, int(sample_n))

        seeds_df = sample_walk_seeds(
            graph, n_walks, walk_seed,
            strategy=seed_strategy,
            initial_face=initial_face,
            min_neighbors=min_neighbors,
        )

        traces_df = run_roaming_vqa_stage(seeds_df, cfg, graph=graph)
        diagnostics = _roaming_diagnostics(traces_df)
        diagnostics.update(graph_diagnostics)

        if "traces" in context.output_paths:
            traces_df.to_parquet(context.output_paths["traces"], index=False)

        if isinstance(traces_df, pd.DataFrame) and context.logger:
            try:
                prefer_cols = [
                    c for c in [
                        "walk_id", "step_n", "recording_id", "arrival_face",
                        "face_chosen", "reasoning", "lat", "lon",
                        "next_recording_id", "distance_m", "termination_reason",
                    ] if c in traces_df.columns
                ]
                context.logger.log_table(
                    traces_df, "roaming/traces", prefer_cols=prefer_cols,
                    panel_group="inspect_results",
                )
                if diagnostics:
                    diag_df = pd.DataFrame([
                        {"metric": k, "value": v}
                        for k, v in sorted(diagnostics.items())
                    ])
                    context.logger.log_table(
                        diag_df, "roaming/diagnostics",
                        prefer_cols=["metric", "value"],
                        panel_group="inspect_results",
                    )
            except Exception as exc:
                print(f"[roaming_runner] Warning: failed to log to wandb: {exc}", flush=True)

        metadata = {
            "rows": int(len(traces_df)),
            "n_walks": n_walks,
            "walk_seed": walk_seed,
            "seed_strategy": seed_strategy,
            "diagnostics": diagnostics,
        }

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "roaming_vqa": RoamingVQARunner(),
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
        key=node_dict["key"], stage=node_dict["stage"],
        depends_on=node_dict.get("depends_on", []),
        inputs=node_dict.get("inputs", {}), outputs=outputs,
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

    run_config = build_run_config(
        cfg, node, context.inputs, context.output_paths,
        dagspace_name="urbanroamvqa",
    )
    wandb_run_id = node.wandb_suffix or node.key

    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            context.logger = logger
            _print_status({"node": node.key, "stage": node.stage, "status": "running"})
            t0 = time.time()
            result = runner.run(context)
            logger.log_metrics({f"{node.stage}/duration_s": time.time() - t0})
            return {"outputs": result.outputs, "metadata": result.metadata}
        except Exception as e:
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig) -> None:
    """Execute roaming VQA pipeline."""
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
                    executor = _create_submitit_executor(launcher_cfg, f"ROAMVQA-{node.key}", log_folder)

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
