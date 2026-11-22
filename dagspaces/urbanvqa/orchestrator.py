"""UrbanVQA pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanvqa-specific pipeline execution logic (VQA stage runner).
The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# -- Common infrastructure imports ------------------------------------------
from dagspaces.common import stage_submission
from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _print_status,
    _safe_log_table,
)
from dagspaces.common.runners.base import StageRunner

# -- Dagspace-local imports -------------------------------------------------
from dagspaces.common.wandb_logger import WandbLogger  # noqa: F401 (re-export)
from .stages.vqa import run_vqa_stage

# Config dir for prompt/launcher resolution
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


# ---------------------------------------------------------------------------
# VQA Stage Runner
# ---------------------------------------------------------------------------

class VQARunner(StageRunner):
    stage_name = "vqa"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg

        # Resolve dataset path from inputs or config
        dataset_path = context.inputs.get("dataset")
        if dataset_path:
            OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        else:
            parquet_path_cfg = getattr(cfg.data, "parquet_path", None)
            image_path_cfg = getattr(cfg.data, "image_path", None)
            if not (parquet_path_cfg and str(parquet_path_cfg).strip()) and \
               not (image_path_cfg and str(image_path_cfg).strip()):
                raise ValueError(
                    f"Node '{context.node.key}' requires either 'dataset' input, "
                    f"'data.parquet_path', or 'data.image_path'"
                )

        # Load input data
        parquet_path = getattr(cfg.data, "parquet_path", None) or dataset_path or ""
        df = _load_vqa_input(cfg, str(parquet_path))

        _print_status({
            "vqa_runner": {
                "event": "invoke_run_vqa_stage",
                "rows": len(df) if df is not None else 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

        # Run VQA inference
        out = run_vqa_stage(df, cfg)

        # Save outputs
        if isinstance(out, pd.DataFrame) and "results" in context.output_paths:
            from dagspaces.common.orchestrator import _clean_df_for_parquet
            out_clean = _clean_df_for_parquet(out.copy())
            out_clean.to_parquet(context.output_paths["results"], index=False)
            print(f"[vqa_runner] Saved {len(out_clean)} rows to "
                  f"{context.output_paths['results']}")

        # Log results to W&B
        if isinstance(out, pd.DataFrame) and context.logger:
            prefer_cols = [
                c for c in ["sample_id", "prompt", "answer", "model_response",
                            "image_path", "image_url"]
                if c in out.columns
            ]
            _safe_log_table(
                context.logger, out, "vqa/results",
                prefer_cols=prefer_cols,
                panel_group="inspect_results",
            )

        row_count = len(out) if isinstance(out, pd.DataFrame) else None
        _print_status({
            "vqa_runner": {
                "event": "stage_completed",
                "rows": row_count,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata={"rows": row_count})


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "vqa": VQARunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name="urbanvqa",
    job_prefix="URBANVQA",
    config_dir=_CONFIG_DIR,
    registry_provider=get_stage_registry,
    gpu_preflight=True,
    status_includes_inputs=True,
    log_rows_metric=True,
    full_wandb_summaries=True,
    registered_runner_message=True,
    recovery_mode="basic",
    pipeline_style="monitor",
    announce_wandb_group=True,
    log_orchestrator_set_config=True,
    manifest_include_inputs=True,
)


# ---------------------------------------------------------------------------
# Data loading (VQA-specific: requires prompt + image columns)
# ---------------------------------------------------------------------------

def _load_vqa_input(cfg: DictConfig, parquet_path: str) -> pd.DataFrame:
    """Load VQA dataset with image column validation."""
    from dagspaces.common.orchestrator import _load_parquet_dataset

    debug = bool(getattr(cfg.runtime, "debug", False))
    sample_n = getattr(cfg.runtime, "sample_n", None)
    columns = dict(getattr(cfg.data, "columns", {})) if getattr(cfg, "data", None) else {}

    if parquet_path and not os.path.isabs(parquet_path):
        parquet_path = os.path.abspath(parquet_path)

    df = _load_parquet_dataset(parquet_path, columns, debug=debug, sample_n=sample_n)
    return df


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
    """Execute the full urbanvqa pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )

