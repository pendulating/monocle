"""Artifact generation pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
artifact_gen-specific stage runners for geospatial raster generation.

Follows the same two-tier SLURM pattern as other dagspaces:
  - Orchestrator runs on slurm_monitor (lightweight CPU job)
  - Stage nodes submit their own jobs via per-node launcher

The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from omegaconf import DictConfig

# -- Common infrastructure imports ------------------------------------------
from dagspaces.common import stage_submission
from dagspaces.common.runners.base import StageRunner

# -- Dagspace-local imports -------------------------------------------------
from .stages.raster import run_raster_stage
from .stages.combine import run_combine_stage

# Config dir for launcher resolution
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


# ---------------------------------------------------------------------------
# Stage registry — one parameterized runner per stage
#
# Each PathStageRunner wires its pipeline inputs into named config keys,
# sets runtime.output_path, calls the stage function, logs grouped metrics,
# and registers the results output. The wiring and metadata below match the
# previous per-stage runner classes exactly.
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "raster": stage_submission.PathStageRunner(
        "raster",
        run_raster_stage,
        input_wiring=(
            ("embeddings", "raster.embeddings_input_path"),
            ("pca_artifacts", "raster.pca_artifacts_path"),
            ("query_projection", "raster.projection_matrix_path"),
        ),
        metadata_fn=lambda cfg, output_path: {
            "output_path": output_path,
            "query_text": str(cfg.raster.query_text),
            "resolution_m": float(cfg.raster.resolution_m),
            "interpolation": "idw",
        },
    ),
    "combine": stage_submission.PathStageRunner(
        "combine",
        run_combine_stage,
        input_wiring=(
            ("raster_a", "combine.raster_a_path"),
            ("raster_b", "combine.raster_b_path"),
        ),
        metadata_fn=lambda cfg, output_path: {
            "output_path": output_path,
            "label_a": str(cfg.combine.label_a),
            "label_b": str(cfg.combine.label_b),
            "operations": list(cfg.combine.operations),
        },
    ),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name="artifact_gen",
    job_prefix="ARTIFACT_GEN",
    config_dir=_CONFIG_DIR,
    registry_provider=get_stage_registry,
    gpu_preflight=True,
    ensure_dotenv_first=True,
    status_includes_inputs=True,
    log_rows_metric=False,
    full_wandb_summaries=True,
    registered_runner_message=False,
    recovery_mode="basic",
    pipeline_style="monitor",
    announce_wandb_group=True,
    log_orchestrator_set_config=True,
    manifest_include_inputs=True,
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
    """Execute artifact generation pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )

