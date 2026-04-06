"""Urban Pairwise VQA pipeline orchestrator.

Imports shared infrastructure from dagspaces.common and adds
urbanpairvqa-specific stage runner for pairwise comparison VQA.
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# -- Common infrastructure imports ------------------------------------------
from dagspaces.common.config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    OutputSpec,
    load_pipeline_graph,
    resolve_output_root,
)
from dagspaces.common.orchestrator import (
    ArtifactRegistry,
    StageExecutionContext,
    StageResult,
    _clean_slurm_env,
    _collect_outputs,
    _create_submitit_executor,
    _load_launcher_config,
    _node_inputs,
    _node_output_paths,
    _print_status,
    build_run_config,
    common_parent,
    prepare_node_config,
)
from dagspaces.common.runners.base import StageRunner
from dagspaces.common.resource_tracker_patch import apply_patch as _apply_resource_tracker_patch
from dagspaces.common.wandb_logger import WandbLogger

_apply_resource_tracker_patch()

# -- Dagspace-local imports -------------------------------------------------
from .samplers.cyclomedia_pairs import build_global_random_pairs
from .stages.pairwise_vqa import run_pairwise_vqa_stage

try:
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HydraConfig = None

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf")


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
        if max_rows:
            manifest_df = manifest_df.head(int(max_rows))

        pair_cfg = getattr(cfg, "pair_sampler", {})
        max_pairs = getattr(pair_cfg, "max_pairs", None)
        pair_seed = int(getattr(pair_cfg, "pair_seed", 777))
        allow_replacement = bool(getattr(pair_cfg, "allow_replacement", False))
        counterbalance_mode = str(getattr(pair_cfg, "counterbalance_mode", "none"))
        repeat_count = int(getattr(pair_cfg, "repeat_count", 0) or 0)
        repeat_fraction = float(getattr(pair_cfg, "repeat_fraction", 0.0) or 0.0)

        metadata_columns = list(getattr(getattr(cfg, "data", {}), "metadata_columns", []))
        pairs_df = build_global_random_pairs(
            manifest_df,
            max_pairs=int(max_pairs) if max_pairs is not None else None,
            seed=pair_seed,
            allow_replacement=allow_replacement,
            counterbalance_mode=counterbalance_mode,
            repeat_count=repeat_count,
            repeat_fraction=repeat_fraction,
            metadata_columns=metadata_columns or None,
        )

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
# Stage registry
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "pairwise_vqa": PairwiseVQARunner(),
}


def get_stage_registry() -> Dict[str, StageRunner]:
    return dict(_STAGE_REGISTRY)


# ---------------------------------------------------------------------------
# SLURM job entrypoint
# ---------------------------------------------------------------------------

def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage — designed to be submitted as a SLURM job."""
    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]

    outputs = {}
    for out_key, out_val in node_dict.get("outputs", {}).items():
        outputs[out_key] = OutputSpec.from_config(out_key, out_val)

    node = PipelineNodeSpec(
        key=node_dict["key"], stage=node_dict["stage"],
        depends_on=node_dict.get("depends_on", []),
        inputs=node_dict.get("inputs", {}), outputs=outputs,
        overrides=node_dict.get("overrides", {}),
        launcher=node_dict.get("launcher"),
        parallel_group=node_dict.get("parallel_group"),
        max_attempts=node_dict.get("max_attempts", 1),
        retry_backoff_s=node_dict.get("retry_backoff_s", 0.0),
        wandb_suffix=node_dict.get("wandb_suffix"),
    )

    context = StageExecutionContext(
        cfg=cfg, node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )

    registry = get_stage_registry()
    runner = registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner for stage '{node.stage}' (node '{node.key}')")

    run_config = build_run_config(
        cfg, node, context.inputs, context.output_paths,
        dagspace_name="urbanpairvqa",
    )
    wandb_run_id = node.wandb_suffix or node.key

    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            context.logger = logger
            _print_status({"node": node.key, "stage": node.stage, "status": "running"})
            t0 = time.time()
            result = runner.run(context)
            logger.log_metrics({f"{node.stage}/duration_s": time.time() - t0})
            return {"outputs": result.outputs, "metadata": result.metadata}
        except Exception as e:
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
            except Exception:
                pass
            raise


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_experiment(cfg: DictConfig) -> None:
    """Execute pairwise VQA pipeline."""
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor",
                     run_config={"type": "pipeline"}) as logger:
        try:
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                os.environ["WANDB_GROUP"] = parent_group

            graph_spec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)

            registry = ArtifactRegistry()
            for source_key, source in graph_spec.sources.items():
                path = source.path
                if not os.path.isabs(path):
                    path = os.path.abspath(os.path.expanduser(path))
                registry.register_source(source_key, path)

            manifest: Dict[str, Any] = {"output_root": output_root, "nodes": {}}
            stage_registry = get_stage_registry()
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()

            logger.log_metrics({"orchestrator/total_nodes": len(ordered_nodes)})

            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(f"No runner for stage '{node.stage}' (node '{node.key}')")

                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values()) or os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)
                node_cfg = prepare_node_config(cfg, node, output_dir)

                context = StageExecutionContext(
                    cfg=node_cfg, node=node, inputs=inputs,
                    output_paths=output_paths, output_dir=output_dir,
                    output_root=output_root,
                )
                node_start = time.time()

                if node.launcher:
                    launcher_cfg = _load_launcher_config(cfg, node.launcher, _CONFIG_DIR)
                    log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
                    os.makedirs(log_folder, exist_ok=True)
                    executor = _create_submitit_executor(launcher_cfg, f"PAIRVQA-{node.key}", log_folder)

                    context_data = {
                        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
                        "node": {
                            "key": node.key, "stage": node.stage,
                            "depends_on": node.depends_on, "inputs": node.inputs,
                            "outputs": {k: {"path": v.path, "type": v.type, "optional": v.optional}
                                        for k, v in node.outputs.items()},
                            "overrides": node.overrides, "launcher": node.launcher,
                            "parallel_group": node.parallel_group,
                            "max_attempts": node.max_attempts,
                            "retry_backoff_s": node.retry_backoff_s,
                            "wandb_suffix": node.wandb_suffix,
                        },
                        "inputs": inputs, "output_paths": output_paths,
                        "output_dir": output_dir, "output_root": output_root,
                    }
                    with _clean_slurm_env():
                        job = executor.submit(execute_stage_job, context_data)
                    job_result = job.result()
                    result = StageResult(outputs=job_result["outputs"], metadata=job_result["metadata"])
                else:
                    _print_status({"node": node.key, "stage": node.stage, "status": "running"})
                    result = runner.run(context)

                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage, "outputs": result.outputs,
                    "metadata": result.metadata, "duration_s": duration,
                }
                _print_status({"node": node.key, "status": "completed",
                               "duration_s": round(duration, 3)})

            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                with open(manifest_path, "w") as fh:
                    json.dump(manifest, fh, indent=2)
            except Exception:
                pass

            total_duration = time.time() - pipeline_start
            try:
                logger.set_summary("orchestrator/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                "orchestrator/total_duration_s": round(total_duration, 3),
                "orchestrator/nodes_completed": len(manifest["nodes"]),
            })
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
