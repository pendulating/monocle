"""Urban Roaming VQA pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanroamvqa-specific stage runner for agent-driven urban navigation.
The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig

# -- Common infrastructure imports ------------------------------------------
from dagspaces.common import stage_submission
from dagspaces.common.config_schema import PipelineNodeSpec
from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _print_status,
)
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.resource_tracker_patch import apply_patch as _apply_resource_tracker_patch

_apply_resource_tracker_patch()

# -- Dagspace-local imports -------------------------------------------------
from .graph.builder import build_street_graph
from .graph.street_graph import compute_graph_diagnostics
from .samplers.seed_sampler import sample_walk_seeds
from .stages.roaming_vqa import run_roaming_vqa_stage

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
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name="urbanroamvqa",
    job_prefix="ROAMVQA",
    config_dir=_CONFIG_DIR,
    registry_provider=get_stage_registry,
    gpu_preflight=False,
    status_includes_inputs=False,
    log_rows_metric=False,
    full_wandb_summaries=False,
    registered_runner_message=False,
    pipeline_style="inline",
    inline_announce_submit=False,
    manifest_include_inputs=False,
)


# ---------------------------------------------------------------------------
# SLURM job entrypoint
# ---------------------------------------------------------------------------

def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run one stage. Submitit calls this function in the SLURM job."""
    return stage_submission.execute_stage_job(context_data, _STAGE_JOB_SETTINGS)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig) -> None:
    """Execute roaming VQA pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )
