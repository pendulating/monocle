from __future__ import annotations

import json
from datetime import datetime
import contextlib
import os
import random
import re
import subprocess
import sys
import time
import base64
from io import BytesIO
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from .config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    load_pipeline_graph,
    resolve_output_root,
)
from .stages.vqa import run_vqa_stage, _apply_ray_data_resource_limits
from .wandb_logger import WandbLogger
from .resource_tracker_patch import apply_patch as _apply_resource_tracker_patch

_apply_resource_tracker_patch()

try:
    import ray  # type: ignore

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover - Ray optional dependency
    ray = None  # type: ignore
    _RAY_AVAILABLE = False

try:
    import submitit  # type: ignore
    _SUBMITIT_AVAILABLE = True
except Exception:
    submitit = None  # type: ignore
    _SUBMITIT_AVAILABLE = False

# Check PIL availability
try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    PILImage = None


_STREAMING_COMPATIBLE_STAGES = {"vqa", "gepa_train", "gepa_val", "gepa_test", "gepa_validate"}


def _probe_single_gpu(device: str) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device
    code = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
        "available = torch.cuda.is_available()\n"
        "count = torch.cuda.device_count() if available else 0\n"
        "if not (available and count >= 1):\n"
        "    sys.exit(1)\n"
        "try:\n"
        "    torch.cuda.set_device(0)\n"
        "    x = torch.randn((8, 8), device='cuda')\n"
        "    y = torch.randn((8, 8), device='cuda')\n"
        "    _ = torch.mm(x, y)\n"
        "    torch.cuda.synchronize()\n"
        "except Exception:\n"
        "    sys.exit(2)\n"
        "sys.exit(0)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable or "python", "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _update_slurm_gpu_envs(valid_devices: List[str]) -> None:
    count = len(valid_devices)
    if count <= 0:
        return
    gpu_list = ",".join(valid_devices)
    for var in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE"):
        val = os.environ.get(var)
        if not val:
            continue
        if "," in val:
            os.environ[var] = gpu_list
        elif ":" in val:
            prefix = val.split(":", 1)[0]
            os.environ[var] = f"{prefix}:{count}"
        else:
            try:
                int(val)
                os.environ[var] = str(count)
            except Exception:
                os.environ[var] = gpu_list
    for var in ("SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK"):
        val = os.environ.get(var)
        if not val:
            continue
        if ":" in val:
            prefix = val.split(":", 1)[0]
            os.environ[var] = f"{prefix}:{count}"
        else:
            try:
                current = int(val)
                os.environ[var] = str(min(count, current))
            except Exception:
                os.environ[var] = str(count)


def _adjust_tensor_parallel_env(valid_count: int) -> None:
    tp_env = os.environ.get("URBANVQA_TENSOR_PARALLEL_SIZE")
    if not tp_env:
        return
    try:
        tp_val = max(1, int(tp_env))
        if valid_count > 0 and tp_val > valid_count:
            os.environ["URBANVQA_TENSOR_PARALLEL_SIZE"] = str(valid_count)
    except Exception:
        pass


def _log_gpu_environment(reason: str) -> None:
    try:
        cuda_visible = [d.strip() for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
        dropped = [d.strip() for d in os.environ.get("URBANVQA_SANITIZED_DROPPED_GPUS", "").split(",") if d.strip()]
        original = [d.strip() for d in os.environ.get("URBANVQA_GPU_SANITIZE_ORIGINAL", "").split(",") if d.strip()]
        payload: Dict[str, Any] = {
            "reason": reason,
            "cuda_visible_devices": cuda_visible,
        }
        if original:
            payload["sanitized_original"] = original
        if dropped:
            payload["sanitized_dropped"] = dropped
        tp_env = os.environ.get("URBANVQA_TENSOR_PARALLEL_SIZE")
        if tp_env:
            try:
                payload["tensor_parallel_size"] = int(tp_env)
            except Exception:
                payload["tensor_parallel_size"] = tp_env
        _print_status({"gpu_env": payload})
    except Exception:
        pass


def _sanitize_cuda_visible_devices(reason: str = "") -> None:
    if os.environ.get("URBANVQA_SKIP_GPU_SANITIZE"):
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not current:
        return
    devices = [d.strip() for d in current.split(",") if d.strip()]
    if len(devices) <= 1:
        return
    normalized = ",".join(devices)
    valid: List[str] = []
    invalid: List[str] = []
    for dev in devices:
        if _probe_single_gpu(dev):
            valid.append(dev)
        else:
            invalid.append(dev)
    if not invalid:
        return
    if not valid:
        # If everything failed, do not modify CUDA_VISIBLE_DEVICES but log once
        os.environ["URBANVQA_GPU_SANITIZE_REASON"] = reason or "stage_start"
        os.environ["URBANVQA_GPU_SANITIZE_TS"] = str(int(time.time()))
        os.environ.pop("URBANVQA_GPU_SANITIZE_ORIGINAL", None)
        os.environ.pop("URBANVQA_SANITIZED_DROPPED_GPUS", None)
        _print_status({
            "gpu_sanitize": {
                "reason": reason or "stage_start",
                "original": normalized,
                "error": "all_devices_failed",
            }
        })
        return
    new_devices = ",".join(valid)
    os.environ["CUDA_VISIBLE_DEVICES"] = new_devices
    os.environ["URBANVQA_SANITIZED_DROPPED_GPUS"] = ",".join(invalid)
    os.environ["URBANVQA_GPU_SANITIZE_REASON"] = reason or "stage_start"
    os.environ["URBANVQA_GPU_SANITIZE_TS"] = str(int(time.time()))
    os.environ["URBANVQA_GPU_SANITIZE_ORIGINAL"] = normalized
    _update_slurm_gpu_envs(valid)
    _adjust_tensor_parallel_env(len(valid))
    _print_status({
        "gpu_sanitize": {
            "reason": reason or "stage_start",
            "original": normalized,
            "sanitized": new_devices,
            "dropped": ",".join(invalid),
        }
    })
    _log_gpu_environment(reason or "stage_start")


@dataclass
class StageExecutionContext:
    cfg: DictConfig
    node: PipelineNodeSpec
    inputs: Dict[str, str]
    output_paths: Dict[str, str]
    output_dir: str
    output_root: str
    logger: Optional['WandbLogger'] = None


@dataclass
class StageResult:
    outputs: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _convert_to_pandas_if_needed(out: Any) -> pd.DataFrame:
    """Convert Ray Dataset to pandas DataFrame if needed."""
    if hasattr(out, "to_pandas"):
        return out.to_pandas()
    return out


def _build_stratified_reservoir(
    class_reservoirs: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Flatten per-class reservoirs into a single list for logging."""
    combined: List[Dict[str, Any]] = []
    for rows in class_reservoirs.values():
        combined.extend(rows)
    return combined


def _materialize_streaming_results(
    ds: Any,
    output_path: Optional[str],
    logger: Optional['WandbLogger'],
    cfg: DictConfig,
    prefer_cols: Optional[List[str]] = None,
) -> int:
    """Iterate over a lazy Ray Dataset in batches, writing results to parquet
    incrementally and streaming progress metrics + a class-balanced,
    reservoir-sampled results table to wandb in real time.

    Reservoir sampling is **stratified by a class column** (default: ``answer``)
    so that rare classes are over-represented in the inspection table.  Configure
    via ``runtime.reservoir_weights`` (dict mapping class values to relative
    weights, e.g. ``{Yes: 0.5, No: 0.5}``).  If weights are omitted, each
    observed class gets an equal share of the reservoir.

    Args:
        ds: Lazy Ray Dataset returned by ``build_llm_processor``/``run_vqa_stage``.
        output_path: Destination parquet file path (single file, not a directory).
        logger: Optional ``WandbLogger`` for streaming metrics/tables.
        cfg: Hydra config – used to read ``runtime.streaming_batch_size``.
        prefer_cols: Column subset for the wandb results table.

    Returns:
        Total number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    runtime_cfg = getattr(cfg, "runtime", None)
    batch_size: int = int(
        getattr(runtime_cfg, "streaming_batch_size", 256)
        if runtime_cfg is not None
        else 256
    )
    reservoir_size: int = 100
    table_log_interval_s: float = 60.0

    # ---- Stratified reservoir config ----------------------------------------
    # reservoir_class_key: which output column to stratify on (default "answer")
    # reservoir_weights:   {class_value: relative_weight} – if omitted, equal
    #                      share across all observed classes
    reservoir_class_key: str = str(
        getattr(runtime_cfg, "reservoir_class_key", "answer")
        if runtime_cfg is not None
        else "answer"
    )
    raw_weights: Optional[Dict[str, float]] = None
    try:
        rw = getattr(runtime_cfg, "reservoir_weights", None) if runtime_cfg else None
        if rw is not None:
            raw_weights = {str(k): float(v) for k, v in dict(rw).items()}
    except Exception:
        raw_weights = None

    # Per-class state: {class_val -> [reservoir rows]}
    class_reservoirs: Dict[str, List[Dict[str, Any]]] = {}
    class_counts: Dict[str, int] = {}  # total rows seen per class

    def _class_limit(cls: str) -> int:
        """Compute the reservoir slot budget for a given class."""
        if raw_weights:
            total_w = sum(raw_weights.values())
            w = raw_weights.get(cls, 0.0)
            if total_w > 0 and w > 0:
                return max(1, int(reservoir_size * w / total_w))
            # Class not in weights → give it a minimal allocation
            known_alloc = sum(
                max(1, int(reservoir_size * ww / total_w))
                for ww in raw_weights.values()
                if total_w > 0
            )
            remainder = max(1, reservoir_size - known_alloc)
            n_unknown = len(class_reservoirs) - len(raw_weights)
            return max(1, remainder // max(1, n_unknown))
        # No weights → equal split across observed classes
        n_classes = max(1, len(class_reservoirs))
        return max(1, reservoir_size // n_classes)

    total_rows: int = 0
    batch_count: int = 0
    writer: Optional[pq.ParquetWriter] = None
    start_time = time.time()
    last_table_log = start_time

    print(
        f"[streaming] Starting incremental materialisation "
        f"(batch_size={batch_size}, reservoir={reservoir_size}, "
        f"class_key={reservoir_class_key!r}, "
        f"weights={raw_weights or 'equal'}, "
        f"table_log_interval={table_log_interval_s}s)",
        flush=True,
    )

    try:
        for batch_df in ds.iter_batches(
            batch_size=batch_size, batch_format="pandas"
        ):
            batch_count += 1
            n = len(batch_df)

            # ---- Write batch to parquet ----------------------------------------
            try:
                table = pa.Table.from_pandas(batch_df, preserve_index=False)
                if writer is None and output_path:
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    writer = pq.ParquetWriter(output_path, table.schema)
                if writer is not None:
                    writer.write_table(table)
            except Exception as exc:
                print(
                    f"[streaming] Warning: parquet write failed on batch "
                    f"{batch_count}: {exc}",
                    flush=True,
                )

            # ---- Stratified reservoir sampling (per-class Algorithm R) ----------
            for idx in range(n):
                total_rows += 1
                row_dict = {
                    col: batch_df.iat[idx, ci]
                    for ci, col in enumerate(batch_df.columns)
                }
                cls_val = str(row_dict.get(reservoir_class_key, "_unknown_"))
                class_counts[cls_val] = class_counts.get(cls_val, 0) + 1
                cls_n = class_counts[cls_val]

                if cls_val not in class_reservoirs:
                    class_reservoirs[cls_val] = []
                cls_pool = class_reservoirs[cls_val]
                limit = _class_limit(cls_val)

                if len(cls_pool) < limit:
                    cls_pool.append(row_dict)
                else:
                    j = random.randint(0, cls_n - 1)
                    if j < limit:
                        cls_pool[j] = row_dict

            # ---- Wandb progress metrics (every batch) --------------------------
            elapsed = time.time() - start_time
            throughput = total_rows / elapsed if elapsed > 0 else 0.0
            if logger:
                try:
                    metrics: Dict[str, Any] = {
                        "vqa/rows_processed": total_rows,
                        "vqa/batches_completed": batch_count,
                        "vqa/throughput_rows_per_sec": round(throughput, 1),
                        "vqa/elapsed_s": round(elapsed, 1),
                    }
                    # Per-class counts as metrics for live monitoring
                    for ck, cv in class_counts.items():
                        safe_key = ck.replace(" ", "_").lower()
                        metrics[f"vqa/class_{safe_key}_count"] = cv
                    logger.log_metrics(metrics)
                except Exception:
                    pass

            # ---- Periodic console progress --------------------------------------
            if batch_count % 20 == 1 or batch_count == 1:
                cls_summary = ", ".join(
                    f"{k}={v}" for k, v in sorted(class_counts.items())
                )
                print(
                    f"[streaming] batch {batch_count}: "
                    f"{total_rows:,} rows, "
                    f"{throughput:.1f} rows/s, "
                    f"elapsed {elapsed:.0f}s  "
                    f"[{cls_summary}]",
                    flush=True,
                )

            # ---- Wandb table (periodic) ----------------------------------------
            now = time.time()
            reservoir = _build_stratified_reservoir(class_reservoirs)
            if (
                logger
                and reservoir
                and (now - last_table_log) >= table_log_interval_s
            ):
                try:
                    _safe_log_table(
                        logger,
                        pd.DataFrame(reservoir),
                        "vqa/results",
                        prefer_cols=prefer_cols,
                        panel_group="inspect_results",
                    )
                except Exception:
                    pass
                last_table_log = now

    finally:
        if writer is not None:
            writer.close()
        elif output_path and total_rows == 0:
            # No batches arrived – write an empty parquet so _collect_outputs
            # finds the expected file (consistent with non-streaming path).
            try:
                empty_df = pd.DataFrame()
                empty_df.to_parquet(output_path, index=False)
            except Exception:
                pass

    total_elapsed = time.time() - start_time

    # ---- Final wandb table + summary metrics --------------------------------
    reservoir = _build_stratified_reservoir(class_reservoirs)
    if logger and reservoir:
        try:
            _safe_log_table(
                logger,
                pd.DataFrame(reservoir),
                "vqa/results",
                prefer_cols=prefer_cols,
                panel_group="inspect_results",
            )
        except Exception:
            pass
        try:
            summary: Dict[str, Any] = {
                "vqa/total_rows": total_rows,
                "vqa/total_duration_s": round(total_elapsed, 1),
                "vqa/final_throughput_rows_per_sec": round(
                    total_rows / total_elapsed if total_elapsed > 0 else 0, 1
                ),
            }
            for ck, cv in class_counts.items():
                safe_key = ck.replace(" ", "_").lower()
                summary[f"vqa/class_{safe_key}_total"] = cv
                summary[f"vqa/class_{safe_key}_reservoir"] = len(
                    class_reservoirs.get(ck, [])
                )
            logger.log_metrics(summary)
        except Exception:
            pass

    cls_detail = ", ".join(
        f"{k}: {len(class_reservoirs.get(k, []))}/{class_counts.get(k, 0)}"
        for k in sorted(class_counts)
    )
    if total_elapsed > 0:
        print(
            f"[streaming] Materialisation complete: "
            f"{total_rows:,} rows in {batch_count} batches, "
            f"{total_elapsed:.1f}s total "
            f"({total_rows / total_elapsed:.1f} rows/s)  "
            f"reservoir [{cls_detail}]",
            flush=True,
        )
    else:
        print(
            f"[streaming] Materialisation complete: {total_rows:,} rows  "
            f"reservoir [{cls_detail}]",
            flush=True,
        )

    return total_rows


def _save_stage_outputs(out: pd.DataFrame, output_paths: Dict[str, str]) -> None:
    """Save DataFrame outputs to disk."""
    if isinstance(out, pd.DataFrame):
        for output_name, output_path in output_paths.items():
            out.to_parquet(output_path, index=False)


def _safe_log_table(
    logger: Optional['WandbLogger'],
    df: pd.DataFrame,
    key: str,
    prefer_cols: Optional[List[str]] = None,
    panel_group: str = "inspect_results"
) -> None:
    """Safely log DataFrame to wandb."""
    if logger and isinstance(df, pd.DataFrame):
        try:
            logger.log_table(df, key, prefer_cols=prefer_cols, panel_group=panel_group)
        except Exception as e:
            print(f"Warning: Failed to log {key} to wandb: {e}", flush=True)


def _compute_doc_level_verification(
    out: pd.DataFrame,
    results_path: Optional[str]
) -> Optional[pd.DataFrame]:
    """Compute document-level verification aggregation from row-level results.
    
    Returns:
        DataFrame with columns: article_id, doc_any_component_verified, core_tuple_verified
        or None if computation fails
    """
    import pandas as _pd
    
    # Preferred: read docs_verification written by stage implementation (if present)
    docs_df = None
    try:
        if results_path:
            out_dir = os.path.dirname(results_path)
            cand_file = os.path.join(out_dir, "docs_verification.parquet")
            cand_dir = os.path.join(out_dir, "docs_verification")
            if os.path.exists(cand_file):
                docs_df = _pd.read_parquet(cand_file)
            elif os.path.isdir(cand_dir):
                docs_df = _pd.read_parquet(cand_dir)
    except Exception:
        docs_df = None
    
    # Fallback: compute simple doc-level view from the results DataFrame
    if docs_df is None and "article_id" in out.columns:
        try:
            def _reduce(df_in: _pd.DataFrame) -> _pd.DataFrame:
                if df_in.empty:
                    return _pd.DataFrame([])
                any_tuple = bool(df_in.get("ver_tuples_any_verified", _pd.Series([], dtype=bool)).any())
                # core tuple verified: require per-field verified flags
                core_ok = True
                for f in ("deployment_domain", "deployment_purpose", "deployment_capability"):
                    col = f"ver_tuple_{f}_verified"
                    if col in df_in.columns:
                        try:
                            v_any = bool(df_in[col].astype(bool).any())
                        except Exception:
                            v_any = False
                        core_ok = core_ok and v_any
                    else:
                        core_ok = False
                return _pd.DataFrame([
                    {
                        "article_id": df_in.get("article_id", _pd.Series([None])).iloc[0],
                        "doc_any_component_verified": any_tuple,
                        "core_tuple_verified": bool(core_ok),
                    }
                ])
            docs_df = out.groupby("article_id", dropna=False).apply(_reduce).reset_index(drop=True)
        except Exception:
            docs_df = None
    
    return docs_df


def _inject_prompt_from_file(cfg: DictConfig, prompt_filename: str) -> None:
    """Inject prompt from YAML file into cfg.prompt."""
    try:
        base_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(base_dir, "conf", "prompt", prompt_filename)
        if os.path.exists(prompt_path):
            prompt_cfg = OmegaConf.load(prompt_path)
            ensure_section(cfg, "prompt")
            sys_p = prompt_cfg.get("system_prompt")
            usr_p = prompt_cfg.get("prompt_template")
            if sys_p:
                OmegaConf.update(cfg, "prompt.system_prompt", sys_p, merge=True)
            if usr_p:
                OmegaConf.update(cfg, "prompt.prompt_template", usr_p, merge=True)
    except Exception:
        pass  # Non-critical, stage may have defaults


class StageRunner:
    stage_name: str

    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError


class ArtifactRegistry:
    def __init__(self) -> None:
        self._artifacts: Dict[str, str] = {}

    def register_source(self, name: str, path: str) -> None:
        self._artifacts[name] = path

    def register_outputs(self, node_key: str, outputs: Mapping[str, str]) -> None:
        for out_name, out_path in outputs.items():
            self._artifacts[f"{node_key}.{out_name}"] = out_path

    def resolve(self, ref: str) -> str:
        if ref in self._artifacts:
            return self._artifacts[ref]
        candidate = os.path.abspath(os.path.expanduser(ref))
        if os.path.exists(candidate) or os.path.isabs(ref):
            return candidate
        raise KeyError(f"Unknown artifact reference '{ref}'")

    def resolve_output_path(self, path: str, output_root: str, node_key: str) -> str:
        if not path:
            raise ValueError(f"Node '{node_key}' output path is empty")
        resolved = path
        if not os.path.isabs(resolved):
            resolved = os.path.join(output_root, resolved)
        return os.path.abspath(resolved)


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))  # type: ignore[return-value]


def merge_overrides(base_cfg: DictConfig, overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    if not overrides:
        return base_cfg
    # Apply each override using OmegaConf.update to properly handle dot notation
    for key, value in overrides.items():
        OmegaConf.update(base_cfg, key, value, merge=True)
    return base_cfg


def ensure_section(cfg: DictConfig, section: str) -> None:
    if OmegaConf.select(cfg, section) is None:
        OmegaConf.update(cfg, section, {}, merge=True)


def common_parent(paths: Iterable[str]) -> Optional[str]:
    try:
        parents = [os.path.dirname(p) for p in paths]
        if not parents:
            return None
        return os.path.commonpath(parents)
    except Exception:
        return None


def prepare_node_config(base_cfg: DictConfig, node: PipelineNodeSpec, output_dir: str) -> DictConfig:
    cfg_copy = clone_config(base_cfg)
    cfg_copy = merge_overrides(cfg_copy, node.overrides)
    ensure_section(cfg_copy, "runtime")
    OmegaConf.update(cfg_copy, "runtime.stage", node.stage, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_dir", output_dir, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_csv", None, merge=True)
    _apply_prompt_override(cfg_copy)
    return cfg_copy


def _apply_prompt_override(cfg: DictConfig) -> None:
    override_path = OmegaConf.select(cfg, "runtime.prompt_override_path")
    if not override_path:
        return
    if not isinstance(override_path, str):
        raise ValueError("runtime.prompt_override_path must be a string if provided")
    resolved = os.path.abspath(os.path.expanduser(override_path))
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Prompt override file not found: {resolved}")
    override_cfg = OmegaConf.load(resolved)
    if override_cfg is None:
        return
    system_prompt = override_cfg.get("system_prompt")
    if system_prompt is not None:
        OmegaConf.update(cfg, "prompt.system", system_prompt, merge=True)
    user_prompt = override_cfg.get("user_prompt")
    if user_prompt is not None:
        OmegaConf.update(cfg, "prompt.user_template", user_prompt, merge=True)


def _load_parquet_dataset(parquet_path: str, columns: Mapping[str, str], debug: bool, sample_n: Optional[int], sample_ratio: Optional[float] = None, sample_seed: Optional[int] = None) -> pd.DataFrame:
    if not isinstance(parquet_path, str) or parquet_path.strip() == "":
        raise ValueError("data.parquet_path is required")
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.abspath(parquet_path)
    df = pd.read_parquet(parquet_path)
    
    # VQA column mapping - prompt is required, at least one image column is required
    col_map = {
        columns.get("prompt", "prompt"): "prompt",
        columns.get("sample_id", "sample_id"): "sample_id",
        # Image columns (at least one required)
        columns.get("image_path", "image_path"): "image_path",
        columns.get("image_url", "image_url"): "image_url",
        columns.get("image_base64", "image_base64"): "image_base64",
    }
    
    present = {src: dst for src, dst in col_map.items() if src in df.columns}
    if present:
        df = df.rename(columns=present)
    
    # Validation: Ensure prompt exists
    if "prompt" not in df.columns:
        raise RuntimeError("Parquet missing required column: prompt")
    
    # Validation: Ensure at least one image column exists
    image_cols = ["image_path", "image_url", "image_base64"]
    if not any(col in df.columns for col in image_cols):
        raise RuntimeError(f"Parquet missing required image column. Must have one of: {image_cols}")

    def _safe_str(x: Any) -> str:
        if x is None:
            return ""
        try:
            return "" if (isinstance(x, float) and pd.isna(x)) else str(x).strip()
        except Exception:
            return str(x) if x is not None else ""

    # Ensure image columns exist and are strings
    for column in image_cols:
        if column not in df.columns:
            df[column] = None
        else:
            try:
                df[column] = df[column].apply(_safe_str)
            except Exception:
                pass
    
    # Generate sample_id if missing
    if "sample_id" not in df.columns:
        import hashlib
        def _gen_sample_id(row):
            # Use image path/URL/base64 + prompt to generate deterministic ID
            img_src = row.get("image_path") or row.get("image_url") or row.get("image_base64") or ""
            prompt_val = row.get("prompt", "")
            combined = f"{img_src}|{prompt_val}"
            return hashlib.sha1(combined.encode("utf-8")).hexdigest()
        df["sample_id"] = df.apply(_gen_sample_id, axis=1)
    else:
        # Ensure sample_id is string
        try:
            df["sample_id"] = df["sample_id"].apply(_safe_str)
        except Exception:
            pass
    
    # Ensure prompt is string
    try:
        df["prompt"] = df["prompt"].apply(_safe_str)
    except Exception:
        pass

    # Apply random sampling if requested
    if sample_ratio is not None and isinstance(sample_ratio, (int, float)) and 0 < sample_ratio < 1:
        try:
            seed = sample_seed if sample_seed is not None else 777
            df = df.sample(frac=float(sample_ratio), random_state=seed).reset_index(drop=True)
            print(
                json.dumps(
                    {
                        "_load_parquet_dataset": {
                            "event": "random_sample_applied",
                            "sample_ratio": float(sample_ratio),
                            "sample_seed": seed,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            print(f"[_load_parquet_dataset] Warning: failed to apply sample_ratio ({exc})", flush=True)

    if isinstance(sample_n, int) and sample_n > 0:
        try:
            n = min(int(sample_n), int(len(df)))
        except Exception:
            n = int(sample_n)
        try:
            seed_env = os.environ.get("URBANVQA_SAMPLE_SEED", "777")
            seed = int(seed_env) if seed_env is not None else (sample_seed if sample_seed is not None else 777)
        except Exception:
            seed = 777
        try:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        except Exception:
            df = df.head(n)
        print(
            json.dumps(
                {
                    "_load_parquet_dataset": {
                        "event": "sample_limit_applied",
                        "sample_n": n,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                }
            ),
            flush=True,
        )
    return df


def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE value which can be in various formats."""
    try:
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            import re as _re
            m = _re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                return max(1, int(m.group(1)) * int(m.group(2)))
        if "," in v:
            acc = 0
            for p in v.split(","):
                acc += int(p)
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _ensure_ray_init_with_cpu_limits(cfg: DictConfig) -> None:
    """Initialize Ray with SLURM-aware CPU limits for orchestrator use.

    Delegates to the unified ``ensure_ray_init`` in ``multiprocessing_utils``
    which also applies ``DataContext`` resource limits after ``ray.init()``.
    """
    if not _RAY_AVAILABLE or ray.is_initialized():
        # Even if Ray is already up, ensure resource limits are applied.
        try:
            _apply_ray_data_resource_limits(cfg)
        except Exception:
            pass
        return
    try:
        from .multiprocessing_utils import ensure_ray_init
        ensure_ray_init(cfg, caller="orchestrator")
    except Exception as exc:
        print(f"[_ensure_ray_init_with_cpu_limits] Fallback: {exc}", flush=True)
        # Minimal fallback — just start Ray
        try:
            ray.init(log_to_driver=True)
        except Exception:
            pass


def _log_parquet_metadata(dataset_path: str) -> tuple[Optional[int], Optional[int], Optional[float]]:
    """Log parquet metadata for debugging. Returns metadata tuple for compatibility."""
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(dataset_path)
        metadata = pf.metadata
        num_row_groups = metadata.num_row_groups
        
        row_group_sizes = []
        for i in range(num_row_groups):
            rg = metadata.row_group(i)
            total_bytes = rg.total_byte_size
            row_group_sizes.append(total_bytes)
        
        if not row_group_sizes:
            return None, None, None
        
        max_size = max(row_group_sizes)
        avg_size = sum(row_group_sizes) / len(row_group_sizes)
        max_rg_size_mb = max_size / (1024 ** 2)
        avg_rg_size_mb = avg_size / (1024 ** 2)
        
        print(f"[_prepare_streaming_dataset] Parquet metadata: {num_row_groups} row groups, "
              f"max {max_rg_size_mb:.1f} MB, avg {avg_rg_size_mb:.1f} MB per row group", flush=True)
        
        return num_row_groups, max_size, avg_size
    except Exception:
        return None, None, None


def _calculate_target_blocks(
    num_row_groups: Optional[int],
    max_rg_size_bytes: Optional[int],
    size_bytes: Optional[int],
    cfg: DictConfig
) -> int:
    """Calculate target number of blocks for repartitioning based on file size and row group structure."""
    target_block_size_bytes = 64 * 1024 * 1024  # 64MB decompressed target (reduced from 128MB)
    
    # If we have row group metadata, use it for better estimation
    if num_row_groups is not None and max_rg_size_bytes is not None:
        max_rg_size_mb = max_rg_size_bytes / (1024 ** 2)
        
        # If row groups are too large (>128MB compressed), split aggressively
        # Reduced threshold from 256MB to 128MB for more aggressive splitting
        if max_rg_size_mb > 128:
            # Estimate decompressed size (conservative: assume 3x expansion)
            estimated_decompressed_mb = max_rg_size_mb * 3
            # Target: split each large row group into multiple blocks (64MB target)
            blocks_per_row_group = max(12, int(estimated_decompressed_mb / 64))  # More blocks per row group
            target_blocks = max(150, num_row_groups * blocks_per_row_group)  # Increased minimum
            target_blocks = min(target_blocks, 600)  # Increased cap to 600 blocks
            print(f"[_prepare_streaming_dataset] Large row groups detected. Splitting into {target_blocks} blocks "
                  f"({blocks_per_row_group} blocks per row group)", flush=True)
            return target_blocks
    
    # Standard calculation based on file size
    # Use smaller target block size (64MB instead of 128MB) for more aggressive splitting
    target_block_size_bytes = 64 * 1024 * 1024  # 64MB decompressed target
    if size_bytes and size_bytes > 0:
        size_gb = size_bytes / float(1024 ** 3)
        estimated_decompressed_bytes = size_bytes * 3  # Assume 3x expansion
        target_blocks = max(100, int(estimated_decompressed_bytes / target_block_size_bytes))
        target_blocks = min(max(100, target_blocks), 300)  # Increased max to 300 for more blocks
        estimated_block_size_mb = estimated_decompressed_bytes / target_blocks / (1024**2)
        print(f"[_prepare_streaming_dataset] Dataset: {size_gb:.2f} GB compressed, estimated {estimated_decompressed_bytes/(1024**3):.2f} GB decompressed, targeting {target_blocks} blocks (~{estimated_block_size_mb:.1f} MB per block)", flush=True)
        return target_blocks
    
    # Fallback: use CPU-based defaults
    try:
        cpus_alloc = None
        cpt = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpt is not None and str(cpt).strip() != "":
            cpus_alloc = int(cpt)
        else:
            con = os.environ.get("SLURM_CPUS_ON_NODE")
            if con is not None and str(con).strip() != "":
                cpus_alloc = _parse_cpus_on_node(con)
        target_blocks = max(100, (cpus_alloc if cpus_alloc and cpus_alloc > 0 else 8) * 15)
        return min(target_blocks, 200)
    except Exception:
        return 100  # Safe default


def _prepare_streaming_dataset(dataset_path: str, columns: Mapping[str, str], cfg: DictConfig, stage: str) -> tuple[Optional[Any], bool]:
    """Prepare streaming dataset for VQA stage.
    
    Two modes:
      1. Parquet-manifest mode (preferred): When dataset_path points to a parquet file
         that contains an 'image_path' column, read the manifest and load images lazily
         via a map step. This avoids expensive directory scanning for large/nested datasets.
      2. Directory-scan mode (fallback): Uses ray.data.read_images() on cfg.data.image_path.
    
    Args:
        dataset_path: Path to parquet manifest or metadata file
        columns: Column mapping configuration
        cfg: Configuration object
        stage: Stage name
        
    Returns:
        Tuple of (Ray Dataset, use_streaming flag)
    """
    if not _RAY_AVAILABLE:
        return None, False
    if stage not in _STREAMING_COMPATIBLE_STAGES:
        return None, False
    
    try:
        image_path_config = getattr(cfg.data, "image_path", None)
        default_prompt = getattr(cfg.data, "default_prompt", None)
        metadata_columns = getattr(cfg.data, "metadata_columns", None)
        partitioning_cfg = getattr(cfg.data, "partitioning", None)

        def _normalize_storage_path(path_val: Optional[str]) -> Optional[str]:
            if not path_val or not isinstance(path_val, str):
                return None
            path_str = path_val.strip()
            if not path_str:
                return None
            # Treat URI-style schemes (s3://, gs://, etc.) as remote paths
            if re.match(r"^[a-zA-Z0-9+\-.]+://", path_str):
                return path_str
            if os.path.isabs(path_str):
                return path_str
            return os.path.abspath(path_str)

        # Ensure Ray is initialized (needed for both modes)
        _ensure_ray_init_with_cpu_limits(cfg)
        if not ray.is_initialized():
            namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "urbanvqa"
            try:
                ray.init(log_to_driver=True, namespace=str(namespace))
            except Exception:
                ray.init(log_to_driver=True)
        
        # Configure Ray Data context
        # NOTE: target_min_block_size and target_max_block_size are
        # intentionally NOT set here.  They are managed exclusively by
        # _apply_ray_data_resource_limits() (in stages/vqa.py) which
        # sets a large target_max_block_size (2 GB) so that image-heavy
        # blocks are not fragmented below vLLM's batch_size of 64.
        try:
            ctx = ray.data.DataContext.get_current()
            ctx.execution_options.verbose_progress = False
            ctx.enable_fallback_to_arrow_object_ext_type = True
        except Exception:
            pass

        # ── Helper functions (shared by both modes) ────────────────────────
        def _normalize_dataset_path(path_val: Any) -> Optional[str]:
            if path_val is None:
                return None
            path_str = str(path_val).strip()
            if not path_str:
                return None
            if re.match(r"^[a-zA-Z0-9+\-.]+://", path_str):
                return path_str
            return os.path.abspath(path_str)

        def _derive_sample_id_from_path(path_val: Optional[str]) -> Optional[str]:
            if not path_val:
                return None
            base_name = os.path.basename(path_val.rstrip("/"))
            if not base_name:
                sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", path_val.strip("/"))
                return sanitized or None
            stem, _ = os.path.splitext(base_name)
            candidate = stem or base_name
            sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", candidate)
            return sanitized or None

        def _sanitize_prompt_value(value: Any, fallback: str) -> str:
            if value is None:
                return fallback
            if isinstance(value, str):
                sanitized = value.strip()
            else:
                sanitized = str(value).strip()
            return sanitized or fallback

        def _merge_sample_id(existing: Any, fallback: Optional[str]) -> Optional[str]:
            if existing is None:
                return fallback
            if isinstance(existing, str):
                candidate = existing.strip()
            else:
                candidate = str(existing).strip()
            if not candidate or candidate.lower() in {"nan", "none"}:
                return fallback
            return candidate

        DEFAULT_PROMPT = ""
        if isinstance(default_prompt, str) and default_prompt.strip():
            fallback_prompt = default_prompt.strip()
        else:
            fallback_prompt = DEFAULT_PROMPT

        # ── Mode 1: Parquet-manifest mode ──────────────────────────────────
        # When dataset_path is a parquet file containing an 'image_path' column,
        # read the manifest directly and load images lazily via map.
        # This avoids the extremely slow directory scan for large/nested datasets.
        _parquet_path = None
        if dataset_path and dataset_path.strip():
            _parquet_candidate = os.path.abspath(dataset_path.strip()) if not os.path.isabs(dataset_path.strip()) else dataset_path.strip()
            if os.path.isfile(_parquet_candidate):
                # Try to read as parquet regardless of extension (some manifests have no .parquet ext)
                try:
                    import pyarrow.parquet as pq
                    _pf = pq.ParquetFile(_parquet_candidate)
                    _parquet_cols = [f.name for f in _pf.schema_arrow]
                    if "image_path" in _parquet_cols:
                        _parquet_path = _parquet_candidate
                except Exception:
                    pass

        if _parquet_path is not None:
            print(
                f"[_prepare_streaming_dataset] Using parquet manifest: {_parquet_path} (skipping directory scan)",
                flush=True,
            )
            ds = ray.data.read_parquet(_parquet_path)

            try:
                info_dict = {
                    "event": "parquet_manifest_loaded",
                    "path": _parquet_path,
                    "cols": ds.schema().names if hasattr(ds, "schema") else [],
                    "count": None,
                }
                try:
                    info_dict["count"] = ds.count()
                except Exception:
                    info_dict["count"] = None
                print(
                    json.dumps({"_prepare_streaming_dataset": {**info_dict, "timestamp": datetime.utcnow().isoformat()}}),
                    flush=True,
                )
            except Exception:
                print(f"[_prepare_streaming_dataset] Parquet schema: {ds.schema()}", flush=True)

            # ── Image loading via map_batches ────────────────────────────────
            # CRITICAL: Use map_batches(batch_size=64) instead of map().
            #
            # map_batches(batch_size=64) batches the rows BEFORE calling the
            # UDF, so each invocation loads only 64 images (~50 MB).
            # Batches are processed sequentially within the task, so per-task
            # peak memory stays manageable regardless of input block size.
            # Block splitting uses a large threshold (2 GB, set by
            # _apply_ray_data_resource_limits) so that downstream vLLM
            # operators always receive full-sized blocks (>= batch_size 64).
            def _load_images_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
                """Load images from image_path into numpy arrays for a batch."""
                from PIL import Image as PILImage

                images = []
                prompts = []
                sample_ids = []
                image_paths = batch.get("image_path", [])
                raw_prompts = batch.get("prompt", [None] * len(image_paths))
                raw_sample_ids = batch.get("sample_id", [None] * len(image_paths))

                for i, img_path in enumerate(image_paths):
                    img_path_str = str(img_path) if img_path is not None else None
                    # Load image
                    if img_path_str and os.path.isfile(img_path_str):
                        try:
                            pil_img = PILImage.open(img_path_str)
                            pil_img.load()
                            images.append(np.asarray(pil_img.convert("RGB")))
                        except Exception:
                            images.append(None)
                    else:
                        images.append(None)
                    # Sanitize prompt
                    prompts.append(
                        _sanitize_prompt_value(
                            raw_prompts[i] if i < len(raw_prompts) else None,
                            fallback_prompt,
                        )
                    )
                    # Merge sample_id
                    sample_ids.append(
                        _merge_sample_id(
                            raw_sample_ids[i] if i < len(raw_sample_ids) else None,
                            _derive_sample_id_from_path(img_path_str),
                        )
                    )

                result = dict(batch)
                result["image"] = images
                result["prompt"] = prompts
                result["sample_id"] = sample_ids
                return result

            # num_cpus=2 limits concurrent image-loading tasks to ~8 (with
            # 16 CPUs) instead of the default ~16, halving the upstream
            # production rate so vLLM can keep up during warmup.
            ds = ds.map_batches(_load_images_batch, batch_size=64, num_cpus=2)

            try:
                sample = ds.take(2)
                print(
                    json.dumps(
                        {
                            "_prepare_streaming_dataset": {
                                "event": "parquet_manifest_sample",
                                "sample": [
                                    {k: ("<ndarray>" if isinstance(v, np.ndarray) else v) for k, v in item.items()}
                                    for item in sample
                                ],
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        }
                    ),
                    flush=True,
                )
            except Exception as sample_exc:
                print(f"[_prepare_streaming_dataset] Warning: sample failed: {sample_exc}", flush=True)

            # Apply debug/sample limits
            debug = bool(getattr(cfg.runtime, "debug", False))
            sample_n = getattr(cfg.runtime, "sample_n", None)
            sample_ratio = getattr(cfg.runtime, "sample_ratio", None)
            sample_seed = getattr(cfg.runtime, "sample_seed", None)

            if sample_ratio is not None and isinstance(sample_ratio, (int, float)) and 0 < sample_ratio < 1:
                try:
                    ds = ds.random_sample(fraction=float(sample_ratio), seed=sample_seed)
                    print(f"[_prepare_streaming_dataset] Applied sample_ratio={sample_ratio}", flush=True)
                except Exception as exc:
                    print(f"[_prepare_streaming_dataset] Warning: failed to apply sample_ratio ({exc})", flush=True)

            if isinstance(sample_n, int) and sample_n > 0:
                try:
                    ds = ds.limit(max(1, int(sample_n)))
                    print(f"[_prepare_streaming_dataset] Applied sample_n={sample_n}", flush=True)
                except Exception as exc:
                    print(f"[_prepare_streaming_dataset] Warning: failed to apply sample_n ({exc})", flush=True)

            return ds, True

        # ── Mode 2: Directory-scan mode (fallback) ─────────────────────────
        image_path_config = _normalize_storage_path(image_path_config)

        if not image_path_config:
            raise ValueError("data.image_path must point to a directory containing image files (no parquet manifest found)")

        is_remote_path = bool(re.match(r"^[a-zA-Z0-9+\-.]+://", image_path_config))
        if not is_remote_path and not os.path.isdir(image_path_config):
            raise ValueError(f"data.image_path must be a directory, got: {image_path_config}")

        if is_remote_path:
            print(
                json.dumps(
                    {
                        "_prepare_streaming_dataset": {
                            "event": "image_read_start",
                            "path": image_path_config,
                            "is_remote": True,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        else:
            print(
                f"[_prepare_streaming_dataset] Reading images from directory: {image_path_config}",
                flush=True,
            )
        
        # Configure optional directory partitioning
        partitioning_obj = None
        if partitioning_cfg:
            try:
                from ray.data.datasource.partitioning import Partitioning  # type: ignore

                partitioning_type = str(getattr(partitioning_cfg, "type", "dir") or "dir").lower()
                field_names = list(getattr(partitioning_cfg, "field_names", []))
                partition_base_dir = getattr(partitioning_cfg, "base_dir", None)
                partition_base_dir = _normalize_storage_path(partition_base_dir) or image_path_config

                if partitioning_type == "dir":
                    if not field_names:
                        raise ValueError("partitioning.field_names must be provided when partitioning.type='dir'")
                    partitioning_obj = Partitioning("dir", field_names=field_names, base_dir=partition_base_dir)
                else:
                    raise ValueError(f"Unsupported partitioning.type '{partitioning_type}' for image ingestion")

                print(
                    json.dumps(
                        {
                            "_prepare_streaming_dataset": {
                                "event": "partitioning_enabled",
                                "type": partitioning_type,
                                "fields": field_names,
                                "base_dir": partition_base_dir,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[_prepare_streaming_dataset] Warning: Failed to configure partitioning "
                    f"({exc}). Continuing without partitioning.",
                    flush=True,
                )
                partitioning_obj = None

        # Read images. ray.data.read_images() produces ArrowTensorType columns that remain Arrow-native
        # until materialized. Further coercion happens lazily downstream in preprocessing.
        ds = ray.data.read_images(image_path_config, include_paths=True, partitioning=partitioning_obj)

        try:
            info_dict = {
                "event": "dataset_loaded",
                "cols": ds.schema().names if hasattr(ds, "schema") else [],
                "count": None,
            }
            try:
                info_dict["count"] = ds.count()
            except Exception:
                info_dict["count"] = None
            print(
                json.dumps(
                    {
                        "_prepare_streaming_dataset": {
                            **info_dict,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        except Exception:
            print(f"[_prepare_streaming_dataset] Dataset schema: {ds.schema()}", flush=True)
        
        # Load metadata from external source if provided
        metadata_ds = None
        metadata_cols_available: set[str] = set()
        if dataset_path and dataset_path.strip():
            dataset_path = os.path.abspath(dataset_path)
        if dataset_path and os.path.exists(dataset_path):
            try:
                _, metadata_ext = os.path.splitext(dataset_path)
                metadata_ext = metadata_ext.lower()
                if metadata_ext in {".parquet", ".pq"}:
                    print(f"[_prepare_streaming_dataset] Loading metadata from parquet: {dataset_path}", flush=True)
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(dataset_path)
                    schema = pf.schema_arrow
                    parquet_columns = [field.name for field in schema]
                    
                    # Identify metadata columns (exclude standard VQA columns)
                    standard_cols = {"prompt", "sample_id", "image_path", "image_url", "image_base64", "image"}
                    if metadata_columns:
                        metadata_cols_to_read = [col for col in metadata_columns if col in parquet_columns]
                    else:
                        metadata_cols_to_read = [col for col in parquet_columns if col not in standard_cols]
                    
                    if metadata_cols_to_read:
                        if "image_path" in parquet_columns and "image_path" not in metadata_cols_to_read:
                            metadata_cols_to_read.append("image_path")
                        if "sample_id" in parquet_columns and "sample_id" not in metadata_cols_to_read:
                            metadata_cols_to_read.append("sample_id")
                        print(f"[_prepare_streaming_dataset] Loading metadata columns: {metadata_cols_to_read}", flush=True)
                        metadata_ds = ray.data.read_parquet(dataset_path, columns=metadata_cols_to_read)
                        metadata_cols_available = set(metadata_cols_to_read)
                    else:
                        metadata_ds = ray.data.read_parquet(dataset_path)
                        metadata_cols_available = set(metadata_ds.schema().names)
                elif metadata_ext in {".csv"}:
                    print(f"[_prepare_streaming_dataset] Loading metadata from csv: {dataset_path}", flush=True)
                    metadata_ds = ray.data.read_csv(dataset_path)
                    metadata_cols_available = set(metadata_ds.schema().names)
                    if metadata_columns:
                        selected_cols = [col for col in metadata_columns if col in metadata_cols_available]
                        if selected_cols:
                            metadata_ds = metadata_ds.select_columns(selected_cols)
                            metadata_cols_available = set(metadata_ds.schema().names)
                else:
                    print(f"[_prepare_streaming_dataset] Warning: Unsupported metadata file type for {dataset_path}", flush=True)
                    metadata_ds = None
                
                if metadata_ds is not None:
                    # Derive sample_id from image basename when available
                    if "image" in metadata_ds.schema().names and "sample_id" not in metadata_ds.schema().names:
                        def _attach_sample_id(row: Dict[str, Any]) -> Dict[str, Any]:
                            row_out = dict(row)
                            sample_val = _derive_sample_id_from_path(row_out.get("image"))
                            row_out["sample_id"] = sample_val
                            return row_out
                        metadata_ds = metadata_ds.map(_attach_sample_id)
                    metadata_cols_available = set(metadata_ds.schema().names)
            except Exception as e:
                print(f"[_prepare_streaming_dataset] Warning: Failed to load metadata: {e}", flush=True)

        def _enrich_vqa_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
            row_out = dict(row)
            path_val = _normalize_dataset_path(row_out.get("path"))
            row_out["image_path"] = path_val

            derived_sample = _derive_sample_id_from_path(path_val)
            row_out["sample_id"] = _merge_sample_id(row_out.get("sample_id"), derived_sample)

            row_out["prompt"] = _sanitize_prompt_value(row_out.get("prompt"), fallback_prompt)
            return row_out

        ds = ds.map(_enrich_vqa_metadata)

        try:
            materialized_sample = ds.take(3)
            print(
                json.dumps(
                    {
                        "_prepare_streaming_dataset": {
                            "event": "post_enrich_sample",
                            "sample": [
                                {
                                    k: ("<ndarray>" if isinstance(v, np.ndarray) else v)
                                    for k, v in item.items()
                                }
                                for item in materialized_sample
                            ],
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        except Exception as sample_exc:
            print(
                f"[_prepare_streaming_dataset] Warning: failed to materialize sample after enrichment: {sample_exc}",
                flush=True,
            )

        debug = bool(getattr(cfg.runtime, "debug", False))
        print(f"[_prepare_streaming_dataset] debug mode = {debug}", flush=True)

        # Join with metadata if available
        if metadata_ds is not None:
            try:
                sample = ds.take(1)
                if sample and len(sample) > 0:
                    sample_row = sample[0]
                    join_candidates: List[str] = []
                    if "sample_id" in sample_row and "sample_id" in metadata_cols_available:
                        join_candidates.append("sample_id")
                    if "image_path" in sample_row and "image_path" in metadata_cols_available:
                        join_candidates.append("image_path")
                    
                    if join_candidates:
                        join_key = join_candidates[0]
                        print(f"[_prepare_streaming_dataset] Joining images with metadata on {join_key}", flush=True)
                        from ray.data.dataset import MaterializedDataset

                        if not isinstance(ds, MaterializedDataset):
                            ds = ds.materialize()

                        left_names = set(ds.schema().names)
                        right_names = set(metadata_ds.schema().names)
                        drop_cols = [col for col in right_names if col in left_names and col != join_key]
                        if drop_cols:
                            metadata_ds = metadata_ds.drop_columns(drop_cols)

                        if not isinstance(metadata_ds, MaterializedDataset):
                            metadata_ds = metadata_ds.materialize()

                        left_blocks = ds.num_blocks()
                        right_blocks = metadata_ds.num_blocks()
                        num_partitions = max(left_blocks or 1, right_blocks or 1)
                        ds = ds.join(metadata_ds, "left_outer", num_partitions, on=(join_key,))
                        print(f"[_prepare_streaming_dataset] Successfully joined metadata", flush=True)
                    else:
                        print(f"[_prepare_streaming_dataset] Warning: Metadata missing join key columns", flush=True)
            except Exception as e:
                print(f"[_prepare_streaming_dataset] Warning: Failed to join metadata: {e}", flush=True)
        
        print(f"[_prepare_streaming_dataset] Successfully created dataset from images", flush=True)
        
        # Apply debug/sample limits if configured
        debug = bool(getattr(cfg.runtime, "debug", False))
        sample_n = getattr(cfg.runtime, "sample_n", None)
        sample_ratio = getattr(cfg.runtime, "sample_ratio", None)
        sample_seed = getattr(cfg.runtime, "sample_seed", None)

        if sample_ratio is not None and isinstance(sample_ratio, (int, float)) and 0 < sample_ratio < 1:
            try:
                ds = ds.random_sample(fraction=float(sample_ratio), seed=sample_seed)
                print(
                    json.dumps(
                        {
                            "_prepare_streaming_dataset": {
                                "event": "random_sample_applied",
                                "sample_ratio": float(sample_ratio),
                                "sample_seed": sample_seed,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[_prepare_streaming_dataset] Warning: failed to apply sample_ratio ({exc})",
                    flush=True,
                )

        if isinstance(sample_n, int) and sample_n > 0:
            try:
                limit_val = max(1, int(sample_n))
                ds = ds.limit(limit_val)
                print(
                    json.dumps(
                        {
                            "_prepare_streaming_dataset": {
                                "event": "sample_limit_applied",
                                "sample_n": limit_val,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[_prepare_streaming_dataset] Warning: failed to apply sample_n limit ({exc})",
                    flush=True,
                )
        
        return ds, True
    except Exception as exc:
        try:
            print(
                json.dumps(
                    {
                        "_prepare_streaming_dataset": {
                            "event": "error",
                            "message": str(exc),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        except Exception:
            print(f"[_prepare_streaming_dataset] Error: {exc}", flush=True)
        return None, False


def _estimate_dataset_size_bytes(path: str) -> Optional[int]:
    if not path:
        return None
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        if os.path.isdir(path):
            total = 0
            for root, _, files in os.walk(path):
                for name in files:
                    try:
                        total += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        continue
            return total
    except OSError:
        return None
    return None


def prepare_stage_input(cfg: DictConfig, dataset_path: str, stage: str) -> tuple[Optional[pd.DataFrame], Optional[Any], bool]:
    debug = bool(getattr(cfg.runtime, "debug", False))
    sample_n = getattr(cfg.runtime, "sample_n", None)
    columns = dict(getattr(cfg.data, "columns", {})) if getattr(cfg, "data", None) else {}
    runtime_cfg = getattr(cfg, "runtime", None)

    if dataset_path and not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(dataset_path)

    streaming_enabled = bool(getattr(runtime_cfg, "streaming_io", False)) if runtime_cfg is not None else False
    auto_stream_attempted = False

    if stage in _STREAMING_COMPATIBLE_STAGES and not streaming_enabled:
        auto_streaming_enabled = True if runtime_cfg is None else bool(getattr(runtime_cfg, "auto_streaming_io", True))
        raw_threshold = 1.0 if runtime_cfg is None else getattr(runtime_cfg, "auto_streaming_min_file_gb", 1.0)
        threshold_gb: Optional[float]
        try:
            threshold_candidate = float(raw_threshold)
            threshold_gb = threshold_candidate if threshold_candidate > 0 else None
        except Exception:
            threshold_gb = None

        if auto_streaming_enabled and threshold_gb is not None:
            size_bytes = _estimate_dataset_size_bytes(dataset_path)
            if size_bytes is not None:
                size_gb = size_bytes / float(1024 ** 3)
                if size_gb >= threshold_gb:
                    streaming_enabled = True
                    auto_stream_attempted = True
                    print(
                        f"[prepare_stage_input] Auto-enabled streaming IO for stage '{stage}' on dataset '{dataset_path}' (size {size_gb:.2f} GB >= threshold {threshold_gb:.2f} GB)",
                        flush=True,
                    )
            else:
                print(
                    f"[prepare_stage_input] Unable to determine dataset size for '{dataset_path}'; continuing without auto streaming",
                    flush=True,
                )

    ds = None
    use_streaming = False
    if streaming_enabled:
        ds, use_streaming = _prepare_streaming_dataset(dataset_path, columns, cfg, stage)
        try:
            print(
                json.dumps(
                    {
                        "prepare_stage_input": {
                            "event": "streaming_result",
                            "use_streaming": bool(use_streaming),
                            "ds_type": type(ds).__name__ if ds is not None else None,
                            "dataset_path": dataset_path,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        except Exception:
            pass
        if auto_stream_attempted and not use_streaming:
            print(
                f"[prepare_stage_input] Streaming IO requested for stage '{stage}' but Ray streaming is unavailable; falling back to pandas load",
                flush=True,
            )

    df = None
    if not use_streaming and dataset_path and str(dataset_path).strip():
        sample_ratio = getattr(runtime_cfg, "sample_ratio", None) if runtime_cfg is not None else None
        sample_seed = getattr(runtime_cfg, "sample_seed", None) if runtime_cfg is not None else None
        df = _load_parquet_dataset(
            dataset_path, 
            columns, 
            debug=debug, 
            sample_n=sample_n,
            sample_ratio=sample_ratio,
            sample_seed=sample_seed
        )
    return df, ds, use_streaming


def _collect_outputs(context: StageExecutionContext, optional: Mapping[str, bool]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, path in context.output_paths.items():
        if os.path.exists(path):
            resolved[key] = path
        else:
            if optional.get(key, False):
                continue
            raise FileNotFoundError(
                f"Expected output '{key}' for node '{context.node.key}' at '{path}' not found"
            )
    return resolved


class VQARunner(StageRunner):
    stage_name = "vqa"

    def run(self, context: StageExecutionContext) -> StageResult:
        cfg = context.cfg
        
        # Dataset input is optional - VQA can read images directly from directory
        # If dataset is provided, use it; otherwise rely on data.image_path for directory-based images
        dataset_path = context.inputs.get("dataset")
        if dataset_path:
            # Update parquet_path if dataset input is provided
            OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        else:
            # No dataset input - check if we have either parquet_path or image_path
            parquet_path_cfg = getattr(cfg.data, "parquet_path", None)
            image_path_cfg = getattr(cfg.data, "image_path", None)
            
            if not (parquet_path_cfg and str(parquet_path_cfg).strip()) and \
               not (image_path_cfg and str(image_path_cfg).strip()):
                raise ValueError(
                    f"Node '{context.node.key}' requires either 'dataset' input, "
                    f"'data.parquet_path', or 'data.image_path'"
                )
        
        # Load data with prompt + image columns
        # prepare_stage_input will handle both parquet and directory-based images
        parquet_path = getattr(cfg.data, "parquet_path", None) or dataset_path or ""
        df, ds, use_streaming = prepare_stage_input(cfg, parquet_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        try:
            print(
                json.dumps(
                    {
                        "vqa_runner": {
                            "event": "invoke_run_vqa_stage",
                            "use_streaming": bool(use_streaming),
                            "input_type": type(in_obj).__name__ if in_obj is not None else None,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        except Exception:
            pass
        
        # Run VQA stage
        out = run_vqa_stage(in_obj, cfg)
        
        # ----------------------------------------------------------------
        # Streaming path: iterate in batches, write parquet incrementally,
        # and stream progress + sampled results to wandb in real time.
        # ----------------------------------------------------------------
        if use_streaming and hasattr(out, "iter_batches"):
            prefer_cols = [
                "sample_id",
                "answer",
                "model_response",
                "image_path",
            ]
            row_count = _materialize_streaming_results(
                out,
                context.output_paths.get("results"),
                context.logger,
                cfg,
                prefer_cols=prefer_cols,
            )
        else:
            # ---- Non-streaming path (unchanged) ----------------------------
            out = _convert_to_pandas_if_needed(out)

            # Calculate row count
            row_count = None
            if isinstance(out, pd.DataFrame):
                row_count = len(out)
            elif isinstance(out, pd.Series):
                row_count = len(out)
            elif hasattr(out, "__len__"):
                try:
                    row_count = len(out)
                except Exception:
                    pass

            # Save outputs to disk
            if isinstance(out, pd.DataFrame):
                if "results" in context.output_paths:
                    out.to_parquet(context.output_paths["results"], index=False)

            # Log results table to wandb
            if isinstance(out, pd.DataFrame) and context.logger:
                try:
                    non_stream_prefer = [
                        "sample_id",
                        "prompt",
                        "answer",
                        "model_response",
                        "image_path",
                        "image_url",
                    ]
                    non_stream_prefer = [
                        c for c in non_stream_prefer if c in out.columns
                    ]
                    _safe_log_table(
                        context.logger, out, "vqa/results",
                        prefer_cols=non_stream_prefer,
                        panel_group="inspect_results",
                    )
                except Exception as e:
                    print(
                        f"Warning: Failed to log VQA results to wandb: {e}",
                        flush=True,
                    )

        try:
            print(
                json.dumps(
                    {
                        "vqa_runner": {
                            "event": "stage_completed",
                            "rows": row_count,
                            "streaming": bool(use_streaming),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    }
                ),
                flush=True,
            )
        except Exception:
            pass
        
        metadata: Dict[str, Any] = {
            "rows": row_count,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "vqa": VQARunner(),
}


def _ensure_output_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _node_optional_outputs(node: PipelineNodeSpec) -> Dict[str, bool]:
    return {name: spec.optional for name, spec in node.outputs.items()}


def _node_output_paths(node: PipelineNodeSpec, registry: ArtifactRegistry, output_root: str) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for out_name, spec in node.outputs.items():
        resolved[out_name] = registry.resolve_output_path(spec.path, output_root, node.key)
    _ensure_output_dirs(resolved.values())
    return resolved


def _node_inputs(node: PipelineNodeSpec, registry: ArtifactRegistry) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for alias, ref in node.inputs.items():
        resolved[alias] = registry.resolve(ref)
    return resolved


def _print_status(payload: Dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, indent=2))
    except Exception:
        pass


def _load_launcher_config(cfg: DictConfig, launcher_name: str) -> Optional[DictConfig]:
    """Load a launcher configuration from Hydra config."""
    try:
        # Find the config path - use the location of this file as reference
        config_path = os.path.join(os.path.dirname(__file__), "conf")
        
        if not os.path.exists(config_path):
            # Try to get from hydra runtime
            hydra_cfg = getattr(cfg, "hydra", None)
            if hydra_cfg:
                runtime_cfg = getattr(hydra_cfg, "runtime", None)
                if runtime_cfg:
                    sources = getattr(runtime_cfg, "config_sources", [])
                    for source in sources:
                        if hasattr(source, "provider") and source.provider == "main":
                            config_path = source.path
                            break
        
        if not config_path or not os.path.exists(config_path):
            raise ValueError(f"Could not find config directory")
            
        launcher_file = os.path.join(config_path, "hydra", "launcher", f"{launcher_name}.yaml")
        if not os.path.exists(launcher_file):
            raise ValueError(f"Launcher config file not found: {launcher_file}")
        
        # Load the launcher config
        launcher_cfg = OmegaConf.load(launcher_file)
        # Resolve interpolations with the main config as context
        launcher_cfg = OmegaConf.merge({"runtime": cfg.get("runtime", {})}, launcher_cfg)
        return launcher_cfg
    except Exception as e:
        raise ValueError(f"Failed to load launcher config '{launcher_name}': {e}") from e


@contextlib.contextmanager
def _clean_slurm_env():
    """Temporarily remove Slurm environment variables to prevent incorrect inheritance when nesting.

    When submitit submits child jobs from within a SLURM job, the parent's SLURM environment
    variables can leak into the child job's submission context, causing issues like:
    - Incorrect resource allocation
    - Job dependency conflicts
    - Namespace collisions

    This context manager temporarily removes all SLURM/SBATCH env vars during job submission.
    """
    slurm_vars = {k: v for k, v in os.environ.items() if k.startswith("SLURM") or k.startswith("SBATCH")}
    for k in slurm_vars:
        del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(slurm_vars)


def _create_submitit_executor(launcher_cfg: DictConfig, job_name: str, log_folder: str) -> Any:
    """Create a submitit executor from launcher configuration."""
    if not _SUBMITIT_AVAILABLE or submitit is None:
        raise RuntimeError("submitit is not available but is required for SLURM job submission")
    
    # Clean SLURM env to prevent inheritance issues from parent job
    with _clean_slurm_env():
        executor = submitit.AutoExecutor(folder=log_folder)
    
    # Map launcher config to submitit parameters
    executor.update_parameters(
        timeout_min=int(launcher_cfg.get("timeout_min", 120)),
        slurm_partition=str(launcher_cfg.get("partition", "pierson")),
        slurm_mem=f"{int(launcher_cfg.get('mem_gb', 8))}GB",
        slurm_cpus_per_task=int(launcher_cfg.get("cpus_per_task", 2)),
        slurm_gpus_per_node=int(launcher_cfg.get("gpus_per_node", 0)),
        slurm_nodes=int(launcher_cfg.get("nodes", 1)),
        slurm_tasks_per_node=int(launcher_cfg.get("tasks_per_node", 1)),
        slurm_array_parallelism=int(launcher_cfg.get("array_parallelism", 1)),
        name=job_name,
        slurm_additional_parameters=launcher_cfg.get("additional_parameters", {}),
        slurm_setup=launcher_cfg.get("setup", []),
    )
    
    return executor


def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage - designed to be submitted as a SLURM job."""
    # Reconstruct context from serialized data
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]
    
    # Reconstruct PipelineNodeSpec
    from .config_schema import PipelineNodeSpec, OutputSpec
    outputs = {}
    for out_key, out_val in node_dict.get("outputs", {}).items():
        outputs[out_key] = OutputSpec.from_config(out_key, out_val)
    
    node = PipelineNodeSpec(
        key=node_dict["key"],
        stage=node_dict["stage"],
        depends_on=node_dict.get("depends_on", []),
        inputs=node_dict.get("inputs", {}),
        outputs=outputs,
        overrides=node_dict.get("overrides", {}),
        launcher=node_dict.get("launcher"),
        parallel_group=node_dict.get("parallel_group"),
        max_attempts=node_dict.get("max_attempts", 1),
        retry_backoff_s=node_dict.get("retry_backoff_s", 0.0),
        wandb_suffix=node_dict.get("wandb_suffix"),
    )
    
    _sanitize_cuda_visible_devices(reason=f"job:{node.key}")
    _log_gpu_environment(reason=f"job:{node.key}")

    context = StageExecutionContext(
        cfg=cfg,
        node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )
    
    # Get the stage runner
    stage_registry = dict(_STAGE_REGISTRY)
    runner = stage_registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")
    
    # Execute stage with wandb logging context
    wandb_run_id = node.wandb_suffix or node.key
    run_config = {
        "node": node.key,
        "stage": node.stage,
        "inputs": list(context.inputs.keys()),
        "outputs": list(context.output_paths.keys()),
    }
    
    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            # Update context with logger
            context.logger = logger
            
            # Execute the stage
            _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": context.inputs})
            stage_start = time.time()
            
            result = runner.run(context)
            
            # Log completion metrics
            duration_s = time.time() - stage_start
            try:
                logger.set_summary(f"{node.stage}/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                f"{node.stage}/duration_s": duration_s,
                f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
            })
            
            return {
                "outputs": result.outputs,
                "metadata": result.metadata,
            }
        except Exception as e:
            # Log failure
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
                logger.set_summary(f"{node.stage}/error", str(e))
            except Exception:
                pass
            raise


def run_experiment(cfg: DictConfig) -> None:
    # Execute entire pipeline with wandb logging context
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor", run_config={"type": "pipeline"}) as logger:
        try:
            # Get the parent/monitor group ID to pass to child jobs
            # This ensures all stages in one pipeline run are grouped together
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                # Set in environment so child jobs can inherit it
                os.environ["WANDB_GROUP"] = parent_group
                print(f"[orchestrator] Setting WANDB_GROUP={parent_group} for child stages", flush=True)
            
            graph_spec: PipelineGraphSpec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)
            registry = ArtifactRegistry()
            for source_key, source in graph_spec.sources.items():
                path = source.path
                if not os.path.isabs(path):
                    path = os.path.abspath(os.path.expanduser(path))
                registry.register_source(source_key, path)
            manifest: Dict[str, Any] = {
                "output_root": output_root,
                "nodes": {},
            }
            stage_registry = dict(_STAGE_REGISTRY)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()
            
            # Log pipeline structure to wandb: numeric to charts; structure to config
            logger.log_metrics({
                "orchestrator/total_nodes": len(ordered_nodes),
            })
            try:
                logger.set_config({
                    "orchestrator": {
                        "node_order": ordered_nodes,
                        "total_nodes": len(ordered_nodes),
                    }
                })
            except Exception:
                pass
            
            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")
                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values())
                if not output_dir:
                    output_dir = os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)
                node_cfg = prepare_node_config(cfg, node, output_dir)
                context = StageExecutionContext(
                    cfg=node_cfg,
                    node=node,
                    inputs=inputs,
                    output_paths=output_paths,
                    output_dir=output_dir,
                    output_root=output_root,
                )
                
                node_start = time.time()
                
                # Check if this node should be launched as a separate SLURM job
                if node.launcher:
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitting", "launcher": node.launcher, "inputs": inputs})
                    try:
                        launcher_cfg = _load_launcher_config(cfg, node.launcher)
                    except ValueError as e:
                        raise ValueError(f"Could not load launcher config '{node.launcher}' for node '{node.key}': {e}") from e
                    
                    # Create submitit executor - store logs in the Hydra multirun directory
                    # Structure: multirun/YYYY-MM-DD/HH-MM-SS/0/.slurm_jobs/STAGE_NAME/
                    log_folder = None
                    try:
                        # Priority 1: Use HydraConfig to get runtime output directory
                        hydra_cfg = HydraConfig.get()
                        if hydra_cfg and hydra_cfg.runtime and hydra_cfg.runtime.output_dir:
                            hydra_output_dir = hydra_cfg.runtime.output_dir
                            log_folder = os.path.join(hydra_output_dir, ".slurm_jobs", node.key)
                            _print_status({"debug": "using_hydra_output_dir", "log_folder": log_folder})
                    except Exception as e:
                        _print_status({"debug": "hydra_config_error", "error": str(e)})
                    
                    # Priority 2: Fall back to output_root
                    if not log_folder:
                        log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
                        _print_status({"debug": "using_output_root_fallback", "log_folder": log_folder, "output_root": output_root})
                    
                    log_folder = os.path.abspath(log_folder)
                    os.makedirs(log_folder, exist_ok=True)
                    job_name = f"URBANVQA-{node.key}"
                    executor = _create_submitit_executor(launcher_cfg, job_name, log_folder)
                    
                    # Ensure child job uses parent's W&B group for proper grouping
                    # Submitit doesn't auto-inherit env vars, so we need to explicitly set them
                    if parent_group:
                        # Method 1: Set environment variable on executor
                        # This ensures it's available in the SLURM job's environment
                        try:
                            # Get current setup commands and prepend WANDB_GROUP export
                            current_setup = list(launcher_cfg.get("setup", []))
                            # Insert explicit WANDB_GROUP export at the beginning (after shebang/source commands)
                            # Find insertion point (after source commands)
                            insert_idx = 0
                            for i, cmd in enumerate(current_setup):
                                if "source" in cmd or "export HYDRA_FULL_ERROR" in cmd:
                                    insert_idx = i + 1
                            # Insert WANDB_GROUP export
                            wandb_group_export = f"export WANDB_GROUP={parent_group}"
                            if wandb_group_export not in current_setup:
                                current_setup.insert(insert_idx, wandb_group_export)
                                executor.update_parameters(slurm_setup=current_setup)
                                _print_status({"debug": "injected_wandb_group", "group": parent_group, "node": node.key})
                        except Exception as e:
                            _print_status({"debug": "failed_to_inject_wandb_group", "error": str(e)})
                    
                    # Prepare serializable context data
                    context_data = {
                        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
                        "node": {
                            "key": node.key,
                            "stage": node.stage,
                            "depends_on": node.depends_on,
                            "inputs": node.inputs,
                            "outputs": {k: {"path": v.path, "type": v.type, "optional": v.optional} for k, v in node.outputs.items()},
                            "overrides": node.overrides,
                            "launcher": node.launcher,
                            "parallel_group": node.parallel_group,
                            "max_attempts": node.max_attempts,
                            "retry_backoff_s": node.retry_backoff_s,
                            "wandb_suffix": node.wandb_suffix,
                        },
                        "inputs": inputs,
                        "output_paths": output_paths,
                        "output_dir": output_dir,
                        "output_root": output_root,
                    }
                    
                    # Submit the job (clean SLURM env to prevent inheritance issues)
                    with _clean_slurm_env():
                        job = executor.submit(execute_stage_job, context_data)
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitted", "job_id": job.job_id})
                    
                    # Wait for the job to complete with error recovery
                    try:
                        job_result = job.result()  # This blocks until the job completes
                        result = StageResult(
                            outputs=job_result["outputs"],
                            metadata=job_result["metadata"],
                        )
                    except Exception as exc:
                        # Check if the job was actually cancelled or just misreported by submitit
                        # This can happen due to signal handling issues or timing races
                        try:
                            import subprocess
                            # Use squeue to check if the job is still alive
                            check = subprocess.run(
                                ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
                                capture_output=True, text=True, timeout=10
                            )
                            job_state = check.stdout.strip()
                            
                            if job_state in ("R", "PD", "CG"):
                                # Job is still running/pending/completing - wait for it manually
                                _print_status({
                                    "debug": "job_misreported_as_failed",
                                    "job_id": job.job_id,
                                    "state": job_state,
                                    "original_error": str(exc)
                                })
                                
                                # Manual polling fallback
                                while True:
                                    time.sleep(30)
                                    check = subprocess.run(
                                        ["squeue", "-j", str(job.job_id), "-h", "-o", "%t"],
                                        capture_output=True, text=True, timeout=10
                                    )
                                    current_state = check.stdout.strip()
                                    if not current_state or current_state not in ("R", "PD", "CG"):
                                        break
                                
                                # Try to read result from pickle file (submitit's fallback)
                                if hasattr(job, 'paths') and hasattr(job.paths, 'result_pickle'):
                                    if os.path.exists(job.paths.result_pickle):
                                        import pickle
                                        with open(job.paths.result_pickle, "rb") as f:
                                            _outcome, _result = pickle.load(f)
                                            job_result = _result
                                            _print_status({
                                                "debug": "recovered_result_from_pickle",
                                                "job_id": job.job_id
                                            })
                                            result = StageResult(
                                                outputs=job_result["outputs"],
                                                metadata=job_result["metadata"],
                                            )
                                    else:
                                        _print_status({
                                            "node": node.key, "stage": node.stage,
                                            "status": "failed", "job_id": job.job_id,
                                            "error": f"Job completed but no result pickle found: {exc}"
                                        })
                                        raise
                                else:
                                    _print_status({
                                        "node": node.key, "stage": node.stage,
                                        "status": "failed", "job_id": job.job_id,
                                        "error": f"Job completed but cannot access result paths: {exc}"
                                    })
                                    raise
                            else:
                                # Job actually failed
                                _print_status({
                                    "node": node.key, "stage": node.stage,
                                    "status": "failed", "job_id": job.job_id,
                                    "error": str(exc)
                                })
                                raise
                        except subprocess.TimeoutExpired:
                            _print_status({
                                "node": node.key, "stage": node.stage,
                                "status": "failed", "job_id": job.job_id,
                                "error": f"squeue timeout while checking job status: {exc}"
                            })
                            raise
                        except Exception as inner_exc:
                            if 'result' not in locals():
                                _print_status({
                                    "node": node.key, "stage": node.stage,
                                    "status": "failed", "job_id": job.job_id,
                                    "error": f"{exc} (recovery failed: {inner_exc})"
                                })
                                raise exc from inner_exc
                else:
                    # Run locally in the current process
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    # Create a stage-level wandb logger so VQARunner (and others)
                    # can log tables/metrics even when running without a SLURM launcher.
                    wandb_run_id = node.wandb_suffix or node.key
                    stage_run_config = {
                        "node": node.key,
                        "stage": node.stage,
                        "inputs": list(inputs.keys()),
                        "outputs": list(output_paths.keys()),
                    }
                    stage_logger = WandbLogger(
                        node_cfg, stage=node.stage,
                        run_id=wandb_run_id, run_config=stage_run_config,
                    )
                    stage_logger.start()
                    context.logger = stage_logger
                    try:
                        _sanitize_cuda_visible_devices(reason=f"node:{node.key}")
                        _log_gpu_environment(reason=f"node:{node.key}")
                        result = runner.run(context)
                        # Log completion metrics
                        stage_duration = time.time() - node_start
                        try:
                            stage_logger.set_summary(f"{node.stage}/status", "completed")
                        except Exception:
                            pass
                        stage_logger.log_metrics({
                            f"{node.stage}/duration_s": stage_duration,
                            f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
                        })
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "error": str(exc)})
                        try:
                            stage_logger.set_summary(f"{node.stage}/status", "failed")
                            stage_logger.set_summary(f"{node.stage}/error", str(exc))
                        except Exception:
                            pass
                        raise
                    finally:
                        stage_logger.finish()
                
                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage,
                    "inputs": inputs,
                    "outputs": result.outputs,
                    "metadata": result.metadata,
                    "duration_s": duration,
                }
                _print_status({
                    "node": node.key,
                    "stage": node.stage,
                    "status": "completed",
                    "duration_s": round(duration, 3),
                    "outputs": result.outputs,
                })
            
            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
            except Exception:
                pass
            total_duration = time.time() - pipeline_start
            
            # Log final pipeline metrics to wandb
            try:
                logger.set_summary("orchestrator/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                "orchestrator/total_duration_s": round(total_duration, 3),
                "orchestrator/nodes_completed": len(manifest["nodes"]),
            })
            
            _print_status({
                "pipeline": {
                    "output_root": output_root,
                    "nodes": ordered_nodes,
                    "duration_s": round(total_duration, 3),
                    "manifest": manifest_path,
                }
            })
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
