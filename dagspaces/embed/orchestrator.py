"""Urban Embed pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
embed-specific stage runners for batch image embedding.

Follows the same two-tier SLURM pattern as vqa:
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
    dagspace_name="embed",
    job_prefix="EMBED",
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

