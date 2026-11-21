"""Urban Embed pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanembed-specific stage runners for batch image embedding.

Follows the same two-tier SLURM pattern as urbanvqa:
  - Orchestrator runs on slurm_monitor (lightweight CPU job)
  - Stage nodes submit their own GPU jobs via per-node launcher

The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig

# -- Common infrastructure imports ------------------------------------------
from dagspaces.common import stage_submission
from dagspaces.common.config_schema import PipelineNodeSpec
from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
)
from dagspaces.common.runners.base import StageRunner

# -- Dagspace-local imports -------------------------------------------------
from .stages.embed import run_embed_stage
from .stages.rerank import run_rerank_stage
from .stages.build_browser_index import run_build_browser_index_stage
from .stages.train_query_projection import run_train_query_projection_stage

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
    "embed": stage_submission.PathStageRunner(
        "embed",
        run_embed_stage,
        metadata_fn=lambda cfg, output_path: {
            "output_path": output_path,
            "model_source": str(cfg.model.model_source),
        },
        log_payload_fn=lambda cfg, metadata: {
            "model_source": str(cfg.model.model_source),
        },
    ),
    "rerank": stage_submission.PathStageRunner(
        "rerank",
        run_rerank_stage,
        input_wiring=(("embeddings", "reranking.embeddings_input_path"),),
        metadata_fn=lambda cfg, output_path: {
            "output_path": output_path,
            "model_source": str(cfg.model.model_source),
            "top_k": int(cfg.reranking.top_k),
            "query_text": str(cfg.reranking.query_text),
        },
        log_payload_fn=lambda cfg, metadata: {
            "model_source": str(cfg.model.model_source),
            "top_k": int(cfg.reranking.top_k),
        },
    ),
    "build_browser_index": stage_submission.PathStageRunner(
        "build_browser_index",
        run_build_browser_index_stage,
        input_wiring=(("embeddings", "browser_index.embeddings_input_path"),),
        metadata_fn=lambda cfg, output_path: {"output_path": output_path},
    ),
    "train_query_projection": stage_submission.PathStageRunner(
        "train_query_projection",
        run_train_query_projection_stage,
        input_wiring=(("pca_artifacts", "query_projection.pca_input_path"),),
        metadata_fn=lambda cfg, output_path: {"output_path": output_path},
    ),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name="urbanembed",
    job_prefix="URBANEMBED",
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
    """Execute embedding pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )


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
    return stage_submission.submit_and_wait(
        cfg=cfg,
        node=node,
        node_cfg=node_cfg,
        inputs=inputs,
        output_paths=output_paths,
        output_dir=output_dir,
        output_root=output_root,
        parent_group=parent_group,
        settings=_STAGE_JOB_SETTINGS,
        execute_fn=execute_stage_job,
    )


def _recover_slurm_job(
    job, node: PipelineNodeSpec, original_exc: Exception
) -> StageResult:
    """Try to recover a SLURM job result when submitit reports failure."""
    return stage_submission.recover_slurm_job(
        job, node, original_exc, mode=_STAGE_JOB_SETTINGS.recovery_mode
    )
