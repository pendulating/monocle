"""Urban Speech pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanspeech-specific stage runners for audio extraction and ASR.

Follows the same two-tier SLURM pattern as urbanvqa/urbanembed:
  - Orchestrator runs on slurm_monitor (lightweight CPU job)
  - Stage nodes submit their own jobs via per-node launcher
    (extract_audio -> CPU launcher, asr -> JU GPU launcher)

The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

# -- Common infrastructure imports ------------------------------------------
from dagspaces.common import stage_submission
from dagspaces.common.config_schema import PipelineNodeSpec
from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
)
from dagspaces.common.runners.base import StageRunner

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
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name="urbanspeech",
    job_prefix="URBANSPEECH",
    config_dir=_CONFIG_DIR,
    registry_provider=get_stage_registry,
    gpu_preflight=True,
    ensure_dotenv_first=True,
    status_includes_inputs=True,
    log_rows_metric=False,
    full_wandb_summaries=True,
    registered_runner_message=False,
    recovery_mode="nfs_poll",
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
    """Execute speech recognition pipeline."""
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
