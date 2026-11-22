"""opf_vision_labels pipeline orchestrator.

Registers the dagspace-local stages and runs a topological DAG of detectors
whose outputs feed a unified pseudo-label parquet for the OPF vision head.
The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from omegaconf import DictConfig

from dagspaces.common import stage_submission
from dagspaces.common.runners.base import StageRunner

from .stages.blur_region import BlurRegionRunner
from .stages.classify_blur import ClassifyBlurRunner
from .stages.face import FaceDetectorRunner
from .stages.house_number import HouseNumberRunner
from .stages.person import PersonDetectorRunner
from .stages.plate import PlateDetectorRunner
from .stages.unify import UnifyRunner
from .stages.vehicle import VehicleDetectorRunner

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
_DAGSPACE_NAME = "opf_vision_labels"


_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "person": PersonDetectorRunner(),
    "vehicle": VehicleDetectorRunner(),
    "blur_region": BlurRegionRunner(),
    "classify_blur": ClassifyBlurRunner(),
    "house_number": HouseNumberRunner(),
    "face": FaceDetectorRunner(),
    "plate": PlateDetectorRunner(),
    "unify": UnifyRunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name=_DAGSPACE_NAME,
    job_prefix="OPFVL",
    config_dir=_CONFIG_DIR,
    registry_provider=get_stage_registry,
    gpu_preflight=True,
    status_includes_inputs=True,
    log_rows_metric=True,
    full_wandb_summaries=True,
    registered_runner_message=True,
    recovery_mode="basic",
    pipeline_style="monitor",
    announce_wandb_group=False,
    log_orchestrator_set_config=False,
    manifest_include_inputs=True,
)


# ---------------------------------------------------------------------------
# SLURM job entrypoint
# ---------------------------------------------------------------------------

def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point used by submitit when a node runs on SLURM."""
    return stage_submission.execute_stage_job(context_data, _STAGE_JOB_SETTINGS)


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig) -> None:
    """Execute the full pseudo-label pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )

