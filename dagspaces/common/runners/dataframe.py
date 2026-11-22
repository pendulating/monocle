"""A stage runner for a stage function that maps a DataFrame to a DataFrame.

Most inference stages read a table, call a model on each row and write a
table. ``DataFrameStageRunner`` does the parts around the stage function:

1. It reads the input table: the ``dataset`` input of the node, or else
   ``data.parquet_path``. ``data.columns`` renames the columns, and
   ``runtime.sample_n`` takes a sample.
2. It makes each relative ``image_path`` absolute with ``data.image_root``.
3. It calls ``stage_fn(df, cfg)``.
4. It writes the result to the ``results`` output of the node, and logs a
   sample of the rows to W&B.

Example::

    from dagspaces.common.runners.dataframe import DataFrameStageRunner

    def run_my_stage(df, cfg):
        ...
        return out_df

    stages = {"my_stage": DataFrameStageRunner("my_stage", run_my_stage)}
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.orchestrator import (
    StageExecutionContext,
    StageResult,
    _clean_df_for_parquet,
    _collect_outputs,
    _load_parquet_dataset,
    _safe_log_table,
)
from dagspaces.common.runners.base import StageRunner

StageFn = Callable[[pd.DataFrame, DictConfig], pd.DataFrame]

_NOT_A_FILE = ("http://", "https://", "data:", "file://")


def apply_image_root(df: pd.DataFrame, cfg: DictConfig, column: str = "image_path") -> pd.DataFrame:
    """Join ``data.image_root`` to each relative path in ``column``.

    An absolute path, a URL and an empty value do not change. When
    ``data.image_root`` is not set, the function returns ``df`` with no change.
    """
    root = OmegaConf.select(cfg, "data.image_root", default=None)
    if not root or column not in df.columns:
        return df
    root = str(root)

    def _join(value: Any) -> Any:
        if not isinstance(value, str) or not value:
            return value
        if os.path.isabs(value) or value.startswith(_NOT_A_FILE):
            return value
        return os.path.join(root, value)

    df = df.copy()
    df[column] = df[column].map(_join)
    return df


def load_stage_input(cfg: DictConfig, parquet_path: str) -> pd.DataFrame:
    """Read the input table of a stage, with the ``data`` and ``runtime`` settings."""
    columns = OmegaConf.select(cfg, "data.columns", default=None)
    columns = dict(columns) if columns else {}
    df = _load_parquet_dataset(
        str(parquet_path),
        columns,
        debug=bool(OmegaConf.select(cfg, "runtime.debug", default=False)),
        sample_n=OmegaConf.select(cfg, "runtime.sample_n", default=None),
    )
    return apply_image_root(df, cfg)


class DataFrameStageRunner(StageRunner):
    """Run ``stage_fn(df, cfg) -> DataFrame`` as a pipeline stage."""

    def __init__(
        self,
        stage_name: str,
        stage_fn: StageFn,
        *,
        input_key: str = "dataset",
        output_key: str = "results",
        table_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.stage_name = stage_name
        self._stage_fn = stage_fn
        self._input_key = input_key
        self._output_key = output_key
        self._table_columns = list(table_columns) if table_columns else None

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        parquet_path = context.inputs.get(self._input_key) or OmegaConf.select(
            cfg, "data.parquet_path", default=None
        )
        if not parquet_path:
            raise ValueError(
                f"Node '{context.node.key}' needs the input '{self._input_key}' "
                "or 'data.parquet_path'."
            )
        df = load_stage_input(cfg, str(parquet_path))
        print(f"[{self.stage_name}] {len(df)} input rows from {parquet_path}", flush=True)

        out = self._stage_fn(df, cfg)
        rows = len(out) if isinstance(out, pd.DataFrame) else None

        out_path = context.output_paths.get(self._output_key)
        if isinstance(out, pd.DataFrame) and out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            _clean_df_for_parquet(out.copy()).to_parquet(out_path, index=False)
            print(f"[{self.stage_name}] wrote {rows} rows to {out_path}", flush=True)

        if isinstance(out, pd.DataFrame) and context.logger:
            prefer = self._table_columns or [
                c for c in ("sample_id", "image_path", "prompt", "answer") if c in out.columns
            ]
            _safe_log_table(context.logger, out, f"{self.stage_name}/results", prefer_cols=prefer)

        outputs = _collect_outputs(
            context, {name: spec.optional for name, spec in context.node.outputs.items()}
        )
        return StageResult(outputs=outputs, metadata={"rows": rows})
