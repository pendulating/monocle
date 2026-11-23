"""The stage runner of ic_extract.

The pairwisevqa framework does not register this stage. A pipeline adds it:

    pipeline:
      stage_runners:
        ic_extract: research.ic.runner:ICExtractRunner
"""

from __future__ import annotations

import os

import pandas as pd

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    collect_outputs,
)
from dagspaces.common.runners.base import StageRunner
from research.ic.extract_stage import run_ic_extract_stage


class ICExtractRunner(StageRunner):
    """Extract the Integrative Complexity ingredients from a run's traces.

    The stage reads a results parquet of a THINKING run. It needs no dataset and
    no pair sampler, thus it takes its input from `ic_extract.results_path`.
    See `research/docs/ic-ingredient-extraction.md`.
    """

    stage_name = "ic_extract"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        out_paths = getattr(context, "output_paths", {}) or {}
        results_path = out_paths.get("ingredients")
        output_dir = (
            os.path.dirname(results_path) if results_path else context.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)

        out, metadata = run_ic_extract_stage(cfg, output_dir)

        if results_path:
            out.to_parquet(results_path, index=False)

        if context.logger:
            try:
                context.logger.log_metrics(
                    {
                        f"ic_extract/{k}": v
                        for k, v in metadata.items()
                        if isinstance(v, (int, float))
                    }
                )
                if isinstance(out, pd.DataFrame) and not out.empty:
                    prefer_cols = [
                        c for c in [
                            "pair_id", "case", "presented_label", "ingredient_type",
                            "name", "quote", "attrs_json", "quote_method",
                            "char_start", "char_end",
                        ] if c in out.columns
                    ]
                    context.logger.log_table(
                        out.head(2000), "ic_extract/ingredients",
                        prefer_cols=prefer_cols, panel_group="inspect_results",
                    )
            except Exception as exc:
                print(f"[ic_extract] Warning: failed to log to wandb: {exc}",
                      flush=True)

        outputs = collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)
