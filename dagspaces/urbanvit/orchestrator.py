"""UrbanViT pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds urbanvit-specific
stage runners for ViT training, feature extraction, and classification.

Follows the same two-tier SLURM pattern as urbanvqa/urbanembed:
  - Orchestrator runs on slurm_monitor (lightweight CPU job)
  - Stage nodes submit their own GPU jobs via per-node launcher

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
from .stages.shard import run_shard_stage
from .stages.train import run_train_stage
from .stages.eval import run_eval_stage
from .stages.extract import run_extract_stage
from .stages.classify import run_classify_stage
from .stages.collate import run_collate_stage

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")
_DAGSPACE_NAME = "urbanvit"
_JOB_PREFIX = "URBANVIT"


# ---------------------------------------------------------------------------
# Stage registry — one parameterized runner per stage
#
# Each DictResultStageRunner wires its pipeline inputs into named config
# keys, wires the stage output path into its config key, calls the stage
# function, and registers the outputs from the returned dict. The wiring
# below matches the previous per-stage runner classes exactly.
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "shard": stage_submission.DictResultStageRunner(
        "shard",
        run_shard_stage,
        output_wiring=("shards_dir", "shard.output_dir"),
        output_keys={"shards_dir": "shards_dir"},
        extra_optional_outputs=("manifest",),
    ),
    "train": stage_submission.DictResultStageRunner(
        "train",
        run_train_stage,
        input_wiring=(("shards_dir", "shard.output_dir"),),
        output_wiring=("checkpoint", "training.checkpoint_dir"),
        output_keys={"checkpoint": "checkpoint_dir"},
    ),
    "eval": stage_submission.DictResultStageRunner(
        "eval",
        run_eval_stage,
        input_wiring=(
            ("shards_dir", "shard.output_dir"),
            ("checkpoint", "training.checkpoint_dir"),
        ),
        output_wiring=("metrics", "eval.metrics_path"),
        output_keys={"metrics": "metrics_path"},
    ),
    "extract": stage_submission.DictResultStageRunner(
        "extract",
        run_extract_stage,
        input_wiring=(
            ("shards_dir", "shard.output_dir"),
            ("checkpoint", "extract.backbone_checkpoint"),
        ),
        output_wiring=("features", "extract.output_dir"),
        output_keys={"features": "features_dir"},
    ),
    "classify": stage_submission.DictResultStageRunner(
        "classify",
        run_classify_stage,
        input_wiring=(
            ("features", "classify.features_input"),
            ("checkpoint", "training.checkpoint_dir"),
        ),
        output_wiring=("scores", "classify.output_path"),
        output_keys={"scores": "scores_path"},
    ),
    "collate": stage_submission.DictResultStageRunner(
        "collate",
        run_collate_stage,
        input_wiring=(("scores", "collate.scores_input"),),
        output_wiring=("final", "collate.output_path"),
        output_keys={"final": "final_path"},
    ),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name=_DAGSPACE_NAME,
    job_prefix=_JOB_PREFIX,
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
    """Execute the configured urbanvit pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )

