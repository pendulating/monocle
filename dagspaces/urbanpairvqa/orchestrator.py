"""Urban Pairwise VQA pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanpairvqa-specific stage runner for pairwise comparison VQA.
The SLURM submission core lives in dagspaces.common.stage_submission;
the functions below delegate to it and keep the public names stable.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

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
from .samplers.cyclomedia_pairs import (
    build_global_random_pairs,
    build_unit_random_pairs,
)
from .stages.ic_extract import run_ic_extract_stage
from .stages.pairwise_vqa import run_pairwise_vqa_stage
from .stages.trace_extract import run_trace_extract_stage

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


def pair_fingerprint(cfg: DictConfig, manifest_rows: int) -> Dict[str, Any]:
    """Describe everything that decides the pair table.

    A prebuilt table is only usable by a job whose sampler settings match. This
    is the same guard idea as the resume cache: a table built under other
    settings must stop the job, not quietly change the science.

    `shard_index` is NOT part of this. Every shard reads the same full table and
    then keeps its own share.
    """
    pc = getattr(cfg, "pair_sampler", {})
    return {
        "mode": str(getattr(pc, "mode", "image")),
        "unit_column": str(getattr(pc, "unit_column", "")),
        "unit_name_column": str(getattr(pc, "unit_name_column", "")),
        "max_pairs": int(getattr(pc, "max_pairs", 0) or 0),
        "pair_seed": int(getattr(pc, "pair_seed", 777)),
        "allow_replacement": bool(getattr(pc, "allow_replacement", False)),
        "counterbalance_mode": str(getattr(pc, "counterbalance_mode", "none")),
        "repeat_count": int(getattr(pc, "repeat_count", 0) or 0),
        "repeat_fraction": float(getattr(pc, "repeat_fraction", 0.0) or 0.0),
        "weight_column": str(getattr(pc, "weight_column", "") or ""),
        "metadata_columns": sorted(
            str(c) for c in getattr(getattr(cfg, "data", {}), "metadata_columns", []) or []
        ),
        "manifest_rows": int(manifest_rows),
    }


def build_pair_table(cfg: DictConfig, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Draw the full pair table for a case, before any shard cut.

    This is the expensive step: 1,100,000 pairs take about 195 seconds, and it
    gives the SAME table in every job, because the seed fixes the draw. Thus
    `scripts/prebuild_pair_tables.py` runs it once for each case and the shards
    read the parquet in about 3 seconds. See `_load_pair_table`.
    """
    pair_cfg = getattr(cfg, "pair_sampler", {})
    mode = str(getattr(pair_cfg, "mode", "image")).strip().lower() or "image"
    max_pairs = getattr(pair_cfg, "max_pairs", None)
    metadata_columns = list(getattr(getattr(cfg, "data", {}), "metadata_columns", []))
    weight_column = getattr(pair_cfg, "weight_column", None)
    if weight_column is not None:
        weight_column = str(weight_column).strip() or None

    common = dict(
        max_pairs=int(max_pairs) if max_pairs is not None else None,
        seed=int(getattr(pair_cfg, "pair_seed", 777)),
        allow_replacement=bool(getattr(pair_cfg, "allow_replacement", False)),
        counterbalance_mode=str(getattr(pair_cfg, "counterbalance_mode", "none")),
        repeat_count=int(getattr(pair_cfg, "repeat_count", 0) or 0),
        repeat_fraction=float(getattr(pair_cfg, "repeat_fraction", 0.0) or 0.0),
        metadata_columns=metadata_columns or None,
    )
    if mode == "unit":
        print(f"[pairwise_runner] pair_sampler.mode=unit — grouping by "
              f"{str(getattr(pair_cfg, 'unit_column', 'unit_uid'))!r}", flush=True)
        unit_name_column = str(getattr(pair_cfg, "unit_name_column", "unit_name"))
        return build_unit_random_pairs(
            manifest_df,
            unit_column=str(getattr(pair_cfg, "unit_column", "unit_uid")),
            unit_name_column=unit_name_column if unit_name_column else None,
            weight_column=weight_column,
            **common,
        )
    if mode == "image":
        return build_global_random_pairs(manifest_df, **common)
    raise ValueError(f"pair_sampler.mode must be 'image' or 'unit', got {mode!r}")


def _subset_table_matches(side_doc: Dict[str, Any], want: Dict[str, Any],
                          diff: List[str]) -> bool:
    """True when a SUBSET table serves a job that asks for fewer pairs.

    `scripts/subset_pair_tables.py` cuts a smaller table out of a larger draw,
    thus every row is an exact row of the parent. Its sidecar keeps the
    PARENT fingerprint and adds a `subset` block. Such a table serves a job
    whose settings equal the parent's in everything but `max_pairs`, and whose
    `max_pairs` equals the number of canonical pairs the cut kept. Anything
    else is another draw, and the job must stop.
    """
    sub = side_doc.get("subset") or {}
    if not sub:
        return False
    if diff != ["max_pairs"]:
        return False
    return int(want.get("max_pairs") or 0) == int(sub.get("canonical_pairs") or -1)


def _load_pair_table(cfg: DictConfig, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Read the prebuilt pair table when there is one, or draw it.

    `pair_sampler.pairs_path` names a parquet that
    `scripts/prebuild_pair_tables.py` wrote. A sidecar JSON beside it holds the
    sampler settings, and this refuses a table that another setting produced.
    """
    pair_cfg = getattr(cfg, "pair_sampler", {})
    pairs_path = getattr(pair_cfg, "pairs_path", None)
    if not pairs_path:
        return build_pair_table(cfg, manifest_df)

    pairs_path = str(pairs_path)
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(
            f"pair_sampler.pairs_path names {pairs_path}, and there is no file "
            "there. Run scripts/prebuild_pair_tables.py first, or clear the "
            "setting to draw the table in the job."
        )

    want = pair_fingerprint(cfg, len(manifest_df))
    side = os.path.splitext(pairs_path)[0] + ".json"
    side_doc: Dict[str, Any] = {}
    if os.path.exists(side):
        try:
            with open(side) as fh:
                side_doc = json.load(fh) or {}
            have = side_doc.get("fingerprint")
        except Exception:
            have = None
        if have is not None and have != want:
            diff = [k for k in want if have.get(k) != want.get(k)]
            if not _subset_table_matches(side_doc, want, diff):
                raise ValueError(
                    f"the prebuilt table {pairs_path} was drawn under other "
                    f"sampler settings. These differ: {diff}. Build it again."
                )
            print(f"[pairwise_runner] {pairs_path} is a subset of a larger "
                  f"draw: {side_doc['subset'].get('selected_by', '?')}",
                  flush=True)
    t0 = time.time()
    pairs_df = pd.read_parquet(pairs_path)
    print(f"[pairwise_runner] read {len(pairs_df)} prebuilt pairs in "
          f"{time.time() - t0:.1f}s from {pairs_path}", flush=True)
    return pairs_df


def _shard_pairs(pairs_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Keep the part of the pair table that belongs to this job.

    A 1,000,000-pair case is too large for 1 GPU job: it needs about 168 GPU
    hours for qwen3.5-9b, and its prompts alone fill the memory of the job. The
    work splits perfectly, because no pair needs another pair.

    The cut goes by CANONICAL PAIR, not by row. A canonical pair holds 2
    counterbalanced presentations and any repeat draws, and the repeat
    diagnostics compare those rows against each other. A cut by row would put
    the 2 halves of a comparison in 2 jobs and make the diagnostic empty.

    Each job builds the same full pair table from the same seed, then keeps
    every N-th canonical pair. Thus the shards partition the table exactly, and
    no shard needs to read another shard.

    ```bash
    python -m dagspaces.urbanpairvqa.cli -m pipeline=pairwise_schools_mvp \\
        pair_sampler.max_pairs=1000000 \\
        pair_sampler.shard_count=96 pair_sampler.shard_index=0,1,2,...,95
    ```
    """
    pair_cfg = getattr(cfg, "pair_sampler", {})
    count = int(getattr(pair_cfg, "shard_count", 1) or 1)
    index = int(getattr(pair_cfg, "shard_index", 0) or 0)
    if count <= 1:
        return pairs_df
    if not 0 <= index < count:
        raise ValueError(f"shard_index must be in [0, {count}), got {index}")
    if "canonical_pair_id" not in pairs_df.columns:
        raise ValueError(
            "pair_sampler.shard_count needs a canonical_pair_id column. "
            "The sampler always writes one; a custom pair table must too."
        )
    # `factorize` numbers the canonical pairs in the order they appear, thus the
    # stride keeps each shard mixed over the whole city. A contiguous cut would
    # give 1 shard the pairs of 1 corner of the draw.
    codes, _ = pd.factorize(pairs_df["canonical_pair_id"], sort=False)
    keep = pairs_df.loc[codes % count == index].reset_index(drop=True)
    if len(keep) == 0:
        raise ValueError(
            f"shard {index} of {count} holds no pair. Lower shard_count."
        )
    print(
        f"[pairwise_runner] shard {index + 1} of {count}: "
        f"{len(keep)} rows of {len(pairs_df)}",
        flush=True,
    )
    return keep


def _persist_pairs(
    pairs_df: pd.DataFrame,
    context: "StageExecutionContext",
    *,
    mode: str,
) -> Optional[str]:
    """Write the sampled pair manifest to the stage's output dir.

    Lands at ``<run_dir>/pairs.parquet`` — same dir as the stage's `results`
    output. A SIGTERM mid-inference still leaves this file on disk so a
    downstream rerun can skip sampling and go straight to inference.
    """
    if pairs_df is None or len(pairs_df) == 0:
        return None
    out_paths = getattr(context, "output_paths", {}) or {}
    # Prefer the dir of the `results` output; fall back to cwd.
    results_path = out_paths.get("results")
    if results_path:
        run_dir = os.path.dirname(os.path.abspath(results_path)) or "."
    else:
        run_dir = os.getcwd()
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "pairs.parquet")
    try:
        pairs_df.to_parquet(path, index=False)
    except Exception as exc:
        print(f"[pairwise_runner] WARN: failed to persist pairs.parquet: {exc}", flush=True)
        return None
    # Sidecar with minimal metadata for quick inspection.
    meta = {
        "mode": mode,
        "rows": int(len(pairs_df)),
        "columns": list(pairs_df.columns),
        "written_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        with open(os.path.join(run_dir, "pairs.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    except OSError:
        pass
    return path


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_pairwise_manifest(cfg: DictConfig, dataset_override: Optional[str]) -> pd.DataFrame:
    data_cfg = getattr(cfg, "data", {})
    parquet_path = dataset_override or str(getattr(data_cfg, "parquet_path", "")).strip()
    if not parquet_path:
        raise ValueError("Pairwise stage requires a parquet manifest via data.parquet_path or dataset input.")
    parquet_path = os.path.abspath(os.path.expanduser(parquet_path))
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Dataset manifest not found: {parquet_path}")

    columns_map = getattr(data_cfg, "columns", {})
    sample_col = str(getattr(columns_map, "sample_id", "sample_id"))
    image_col = str(getattr(columns_map, "image_path", "image_path"))
    metadata_columns = list(getattr(data_cfg, "metadata_columns", []))

    # Auto-inject pair_sampler.weight_column so data configs don't have to
    # duplicate it in metadata_columns. If the column is missing from the
    # parquet, the sampler's own fallback kicks in with a clear warning.
    pair_cfg = getattr(cfg, "pair_sampler", {})
    weight_column = getattr(pair_cfg, "weight_column", None)
    if weight_column is not None:
        weight_column = str(weight_column).strip() or None
    if weight_column and weight_column not in metadata_columns:
        metadata_columns.append(weight_column)

    try:
        import pyarrow.parquet as pq
        schema_cols = [f.name for f in pq.ParquetFile(parquet_path).schema_arrow]
    except Exception:
        schema_cols = []
    if schema_cols:
        missing = [c for c in (sample_col, image_col) if c not in schema_cols]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")
        existing_meta = [c for c in metadata_columns if c in schema_cols]
        selected = [sample_col, image_col, *existing_meta]
    else:
        selected = [c for c in [sample_col, image_col, *metadata_columns] if c]
    manifest_df = pd.read_parquet(parquet_path, columns=selected)
    rename_map = {}
    if sample_col in manifest_df.columns and sample_col != "sample_id":
        rename_map[sample_col] = "sample_id"
    if image_col in manifest_df.columns and image_col != "image_path":
        rename_map[image_col] = "image_path"
    if rename_map:
        manifest_df = manifest_df.rename(columns=rename_map)
    return manifest_df


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _pairwise_diagnostics(out_df: pd.DataFrame) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    if out_df is None or out_df.empty:
        return diagnostics

    total = float(len(out_df))
    diagnostics["rows_total"] = int(total)

    if "relative_label" in out_df.columns:
        label_counts = out_df["relative_label"].fillna("MISSING").astype(str).value_counts(dropna=False)
        for label, count in label_counts.items():
            key = str(label).replace(" ", "_")
            diagnostics[f"label_count/{key}"] = int(count)
            diagnostics[f"label_prop/{key}"] = _safe_ratio(float(count), total)
        probs = [float(c) / total for c in label_counts.values if c > 0]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        diagnostics["label_entropy_nats"] = float(entropy)
        diagnostics["majority_ratio"] = _safe_ratio(float(label_counts.max()), total)

    if "answer" in out_df.columns:
        ans = out_df["answer"].astype(str).str.strip()
        diagnostics["answer_missing_rate"] = _safe_ratio(float((ans == "").sum()), total)

    if "presented_order" in out_df.columns and "presented_label" in out_df.columns:
        by_order = (
            out_df.groupby("presented_order", dropna=False)["presented_label"]
            .value_counts(normalize=True).rename("prop").reset_index()
        )
        for _, row in by_order.iterrows():
            order = str(row["presented_order"]).replace(" ", "_")
            label = str(row["presented_label"]).replace(" ", "_")
            diagnostics[f"presented_order_prop/{order}/{label}"] = float(row["prop"])

    if "canonical_pair_id" in out_df.columns and "relative_score" in out_df.columns:
        repeats_df = out_df.groupby("canonical_pair_id", dropna=False).filter(lambda g: len(g) > 1)
        if not repeats_df.empty:
            grouped = repeats_df.groupby("canonical_pair_id", dropna=False)
            exact_matches = 0
            weighted_acc = []
            for _, grp in grouped:
                labels = grp["relative_label"].astype(str).tolist() if "relative_label" in grp.columns else []
                scores = grp["relative_score"].astype(float).tolist()
                if labels and len(set(labels)) == 1:
                    exact_matches += 1
                score_span = max(scores) - min(scores) if scores else 0.0
                weighted_acc.append(max(0.0, 1.0 - (score_span / 4.0)))
            diagnostics["repeat_groups"] = int(grouped.ngroups)
            diagnostics["repeat_exact_agreement_rate"] = _safe_ratio(float(exact_matches), float(grouped.ngroups))
            diagnostics["repeat_weighted_agreement_mean"] = float(sum(weighted_acc) / max(1, len(weighted_acc)))

    return diagnostics


# ---------------------------------------------------------------------------
# Pairwise VQA Stage Runner
# ---------------------------------------------------------------------------

class PairwiseVQARunner(StageRunner):
    stage_name = "pairwise_vqa"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        dataset_input = context.inputs.get("dataset")
        manifest_df = _load_pairwise_manifest(cfg, dataset_input)

        runtime_cfg = getattr(cfg, "runtime", {})
        max_rows = getattr(runtime_cfg, "sample_n", None)
        pair_cfg = getattr(cfg, "pair_sampler", {})
        pair_seed = int(getattr(pair_cfg, "pair_seed", 777))
        # Random subsample (not head) so dry runs cover all boroughs, not just
        # the first chunk in materialize order. Seed reuses pair_seed so the
        # same runtime.sample_n produces the same subsample across reruns.
        if max_rows and int(max_rows) < len(manifest_df):
            manifest_df = (
                manifest_df.sample(n=int(max_rows), random_state=pair_seed)
                           .reset_index(drop=True)
            )
        # `build_pair_table` reads the rest of the sampler settings itself.
        # Only what the metadata and the pairs sidecar need stays here.
        counterbalance_mode = str(getattr(pair_cfg, "counterbalance_mode", "none"))
        mode = str(getattr(pair_cfg, "mode", "image")).strip().lower() or "image"

        # Read the prebuilt table when the sweep points at one, or draw it here.
        # Drawing 1,100,000 pairs costs about 195 seconds, and every shard of a
        # case would otherwise repeat it. See `_load_pair_table`.
        pairs_df = _load_pair_table(cfg, manifest_df)

        # Keep only this job's share. The full table is built first and from
        # the same seed in every shard, thus the shards partition it exactly.
        shard_count = int(getattr(pair_cfg, "shard_count", 1) or 1)
        pairs_total = len(pairs_df)
        pairs_df = _shard_pairs(pairs_df, cfg)

        # Persist the sampled pairs alongside the stage output for audit +
        # resume-with-different-model. Written before inference so a SIGTERM
        # mid-inference still leaves the pair manifest on disk.
        pairs_parquet_path = _persist_pairs(pairs_df, context, mode=mode)
        if pairs_parquet_path:
            print(f"[pairwise_runner] wrote {len(pairs_df)} pairs → "
                  f"{pairs_parquet_path}", flush=True)

        # Resume across a preemption. `runtime.resume=true` gives this shard a
        # dir of its own beside its results parquet, and the inference path
        # writes a chunk parquet for each row range there. A job that starts
        # again reads the chunks and generates only the rest.
        #
        # The dir must NOT hold a timestamp: a requeued job renders
        # `${now:...}` again and would otherwise lose its own chunks. The
        # parent of the results path carries no timestamp, thus it is stable.
        if bool(getattr(runtime_cfg, "resume", False)):
            results_path = (getattr(context, "output_paths", {}) or {}).get("results")
            base = os.path.dirname(os.path.abspath(results_path)) if results_path \
                else context.output_dir
            resume_dir = os.path.join(base, "resume")
            os.makedirs(resume_dir, exist_ok=True)
            chunk_rows = int(getattr(runtime_cfg, "resume_chunk_rows", 0) or 0)
            # The model config group is a struct, thus a new key needs the
            # struct flag off. The inference path reads both keys with getattr
            # and does nothing when they are absent.
            OmegaConf.set_struct(cfg, False)
            OmegaConf.update(cfg, "model.resume_dir", resume_dir, merge=True)
            if chunk_rows > 0:
                OmegaConf.update(cfg, "model.resume_chunk_rows", chunk_rows,
                                 merge=True)
            OmegaConf.set_struct(cfg, True)
            print(f"[pairwise_runner] resume dir: {resume_dir} "
                  f"(chunk {chunk_rows} rows)", flush=True)

        out_any = run_pairwise_vqa_stage(pairs_df, cfg)
        out = out_any.to_pandas() if hasattr(out_any, "to_pandas") else out_any
        diagnostics = _pairwise_diagnostics(out)

        if "results" in context.output_paths:
            out.to_parquet(context.output_paths["results"], index=False)

        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = [
                    c for c in [
                        "pair_id", "sample_id_a", "sample_id_b", "answer",
                        "presented_answer", "presented_label", "presented_score",
                        "relative_label", "relative_score", "presented_order",
                        "is_swapped", "repeat_idx", "image_path_a", "image_path_b",
                    ] if c in out.columns
                ]
                context.logger.log_table(
                    out, "pairwise/results", prefer_cols=prefer_cols,
                    panel_group="inspect_results",
                )
                if diagnostics:
                    diag_df = pd.DataFrame([
                        {"metric": k, "value": v}
                        for k, v in sorted(diagnostics.items())
                    ])
                    context.logger.log_table(
                        diag_df, "pairwise/diagnostics",
                        prefer_cols=["metric", "value"],
                        panel_group="inspect_results",
                    )
            except Exception as exc:
                print(f"[pairwise_runner] Warning: failed to log to wandb: {exc}", flush=True)

        metadata = {
            "rows": int(len(out)),
            "pairs_sampled": int(len(pairs_df)),
            "pairs_total": int(pairs_total),
            "shard_count": shard_count,
            "shard_index": int(getattr(pair_cfg, "shard_index", 0) or 0),
            "seed": pair_seed,
            "counterbalance_mode": counterbalance_mode,
            "diagnostics": diagnostics,
        }

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


# ---------------------------------------------------------------------------
# Trace Extraction Stage Runner
# ---------------------------------------------------------------------------

class TraceExtractRunner(StageRunner):
    """Turn the reasoning traces of a pairvqa run into typed extractions.

    The stage reads a results parquet of a THINKING run. It needs no dataset and
    no pair sampler, thus it takes its input from `trace_extract.results_path`.
    See `vlm-narratives-docs/langextract-trace-extraction.md`.
    """

    stage_name = "trace_extract"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        out_paths = getattr(context, "output_paths", {}) or {}
        results_path = out_paths.get("extractions")
        output_dir = (
            os.path.dirname(results_path) if results_path else context.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)

        out, metadata = run_trace_extract_stage(cfg, output_dir)

        if results_path:
            out.to_parquet(results_path, index=False)

        if context.logger:
            try:
                context.logger.log_metrics(
                    {
                        f"trace_extract/{k}": v
                        for k, v in metadata.items()
                        if isinstance(v, (int, float))
                    }
                )
                if isinstance(out, pd.DataFrame) and not out.empty:
                    prefer_cols = [
                        c for c in [
                            "pair_id", "case", "presented_label", "extraction_class",
                            "extraction_text", "attributes_json", "alignment_status",
                            "char_start", "char_end",
                        ] if c in out.columns
                    ]
                    context.logger.log_table(
                        out.head(2000), "trace_extract/extractions",
                        prefer_cols=prefer_cols, panel_group="inspect_results",
                    )
            except Exception as exc:
                print(f"[trace_extract] Warning: failed to log to wandb: {exc}",
                      flush=True)

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class ICExtractRunner(StageRunner):
    """Extract the Integrative Complexity ingredients from a run's traces.

    The stage reads a results parquet of a THINKING run. It needs no dataset and
    no pair sampler, thus it takes its input from `ic_extract.results_path`.
    See `vlm-narratives-docs/ic-ingredient-extraction.md`.
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

        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "pairwise_vqa": PairwiseVQARunner(),
    "trace_extract": TraceExtractRunner(),
    "ic_extract": ICExtractRunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# Shared SLURM core settings
# ---------------------------------------------------------------------------

_STAGE_JOB_SETTINGS = stage_submission.DagspaceSettings(
    dagspace_name="urbanpairvqa",
    job_prefix="PAIRVQA",
    config_dir=_CONFIG_DIR,
    registry_provider=get_stage_registry,
    # The stage job must read `server.env`. Without it MLLMSCI_MODEL_REGISTRY
    # is empty in the job, and `resolve_model_source` cannot find the
    # node-local model mirror. The weight read then goes to NFS each time.
    ensure_dotenv_first=True,
    gpu_preflight=False,
    status_includes_inputs=False,
    log_rows_metric=False,
    full_wandb_summaries=False,
    registered_runner_message=False,
    pipeline_style="inline",
    inline_announce_submit=False,
    manifest_include_inputs=False,
)


# ---------------------------------------------------------------------------
# SLURM job entrypoint
# ---------------------------------------------------------------------------

def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run one stage. Submitit calls this function in the SLURM job."""
    return stage_submission.execute_stage_job(context_data, _STAGE_JOB_SETTINGS)


def run_shard_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run 1 stage and write its manifest, with no monitor job above it.

    `execute_stage_job` gives its outputs back to the caller. A monitor job
    normally waits for that and writes `pipeline_manifest.json`. A single-node
    graph does not need a monitor: it holds 1 stage with no dependency, thus a
    whole CPU job would only block on `job.result()`. 966 of those hold about
    1,900 CPU-hours of a node for nothing.

    Thus this wrapper writes the manifest from inside the GPU job, and the
    submitter exits as soon as it submits. `scripts/merge_pairwise_shards.py`
    reads the shard index and the shard count from that manifest.
    """
    t0 = time.time()
    result = execute_stage_job(context_data)
    output_root = context_data["output_root"]
    node_key = context_data["node"]["key"]
    manifest = {
        "output_root": output_root,
        "nodes": {
            node_key: {
                "stage": context_data["node"]["stage"],
                "outputs": result["outputs"],
                "metadata": result["metadata"],
                "duration_s": round(time.time() - t0, 3),
            }
        },
    }
    try:
        os.makedirs(output_root, exist_ok=True)
        stage_submission.write_json_atomic(
            os.path.join(output_root, "pipeline_manifest.json"), manifest
        )
    except Exception as exc:
        print(f"[run_shard_job] WARN: could not write the manifest: {exc}",
              flush=True)
    return result


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig) -> None:
    """Execute pairwise VQA pipeline."""
    stage_submission.run_pipeline_experiment(
        cfg, _STAGE_JOB_SETTINGS, execute_fn=execute_stage_job
    )
