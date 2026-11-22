"""Urban OCR pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
ocr-specific stage runner and pipeline execution.
The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

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
from .stages.ocr import run_ocr_stage
from .data_handlers.base import OCRDataHandler

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


# ---------------------------------------------------------------------------
# OCR Stage Runner
# ---------------------------------------------------------------------------

class OCRRunner(StageRunner):
    """Stage runner for OCR text spotting."""
    stage_name = "ocr"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        handler_name = getattr(getattr(cfg, "data", None), "handler", "generic")
        handler = OCRDataHandler.get_handler(handler_name)
        df = handler.load_dataset(cfg)

        out = run_ocr_stage(df, cfg)

        row_count = len(out) if isinstance(out, pd.DataFrame) else 0
        unique_images = 0
        if isinstance(out, pd.DataFrame) and "sample_id" in out.columns:
            unique_images = out["sample_id"].nunique()

        _print_status({
            "ocr_runner": {
                "event": "stage_completed",
                "rows": row_count,
                "unique_images": unique_images,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

        if isinstance(out, pd.DataFrame) and "results" in context.output_paths:
            out.to_parquet(context.output_paths["results"], index=False)

        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = [
                    c for c in [
                        "sample_id", "image_path", "text",
                        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                        "confidence", "text_type", "location_id", "face",
                    ] if c in out.columns
                ]
                context.logger.log_table(
                    out, "ocr/results", prefer_cols=prefer_cols,
                    panel_group="inspect_results",
                )
            except Exception as e:
                print(f"Warning: Failed to log OCR results to wandb: {e}", flush=True)

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(
            outputs=outputs,
            metadata={"rows": row_count, "unique_images": unique_images},
        )


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "ocr": OCRRunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name="ocr",
    job_prefix="OCR",
    config_dir=_CONFIG_DIR,
    registry_provider=get_stage_registry,
    gpu_preflight=False,
    status_includes_inputs=False,
    log_rows_metric=True,
    full_wandb_summaries=True,
    registered_runner_message=True,
    pipeline_style="inline",
    inline_announce_submit=True,
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
    """Execute OCR pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )
