"""OCR (Text Spotting) stage for Urban OCR pipeline.

This module provides text spotting functionality using Qwen3-VL:
- Text detection with bounding boxes (bbox_2d)
- Confidence scores and text type classification
- Batch inference with Ray Data
- Flat parquet output (one row per detection)
- Automatic tiling for large images (e.g., 8192x8192 Cyclomedia faces)

Uses the same vLLM + Ray Data + Hydra pipeline infrastructure as urbanvqa.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os
import re
import logging

import pandas as pd
from omegaconf import DictConfig

try:
    import numpy as np
except ImportError:
    np = None

try:
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    _RAY_OK = True
except Exception:
    _RAY_OK = False

from ..prompts.ocr_preprocessing import (
    ocr_preprocess,
    ocr_postprocess,
    parse_ocr_response,
    flatten_detections,
    get_system_prompt,
    get_user_prompt,
    get_sampling_params,
    _normalize_sampling_params,
    OCR_OUTPUT_SCHEMA,
)
from ..tiling import (
    tile_image,
    needs_tiling,
    get_tiling_config,
    transform_bbox_to_original,
    get_global_bbox_range,
    TileInfo,
)
from ..multiprocessing_utils import get_suppress_child_warnings
from ..resource_tracker_patch import apply_patch as _apply_resource_tracker_patch

_apply_resource_tracker_patch()

# Model zoo base path
MODEL_ZOO_BASE = "/share/pierson/matt/zoo/models"

# vLLM logging state
_VLLM_LOGS_SILENCED = False


def _maybe_silence_vllm_logs() -> None:
    """Silence verbose vLLM logs."""
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        from ..logging_filters import PatternModuloFilter
        lg = logging.getLogger("vllm")
        try:
            n = int(os.environ.get("URBANOCR_VLLM_LOG_EVERY", "10") or "10")
        except Exception:
            n = 10
        lg.setLevel(logging.INFO)
        try:
            existing_filters = getattr(lg, "filters", [])
            if not any(getattr(f, "__class__", object).__name__ == "PatternModuloFilter" for f in existing_filters):
                lg.addFilter(PatternModuloFilter(mod=n, pattern="Elapsed time for batch"))
        except Exception:
            pass
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def _suppress_multiprocessing_warnings(cfg: Optional[DictConfig] = None) -> None:
    """Suppress multiprocessing warnings."""
    suppress = get_suppress_child_warnings(cfg)
    if suppress:
        import warnings
        os.environ["URBANOCR_SUPPRESS_WARNINGS"] = "true"
        _apply_resource_tracker_patch()
        warnings.filterwarnings(
            "ignore",
            message="resource_tracker: process died unexpectedly",
            category=UserWarning,
            module="multiprocessing.resource_tracker"
        )


def _parse_cpus_on_node(val: str) -> int:
    """Parse SLURM_CPUS_ON_NODE value."""
    if not isinstance(val, str):
        return -1
    try:
        v = val.strip()
        if "(x" in v and v.endswith(")"):
            m = re.match(r"^(\d+)\(x(\d+)\)$", v)
            if m:
                return max(1, int(m.group(1)) * int(m.group(2)))
        if "," in v:
            acc = sum(int(p) for p in v.split(",") if p.strip())
            return max(1, acc)
        return max(1, int(v))
    except Exception:
        return -1


def _ensure_ray_init(cfg: DictConfig) -> None:
    """Initialize Ray with SLURM-aware limits."""
    if not _RAY_OK or ray.is_initialized():
        return
    
    try:
        suppress_warnings = get_suppress_child_warnings(cfg)
        
        # Detect SLURM CPU allocation
        cpus_alloc = None
        try:
            cpt = os.environ.get("SLURM_CPUS_PER_TASK")
            if cpt and str(cpt).strip():
                cpus_alloc = int(cpt)
            else:
                con = os.environ.get("SLURM_CPUS_ON_NODE")
                if con and str(con).strip():
                    cpus_alloc = _parse_cpus_on_node(con)
        except Exception:
            cpus_alloc = None
        
        # Memory configuration
        try:
            job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
        except Exception:
            job_mem_gb = 64
        
        obj_store_bytes = int(max(1, job_mem_gb) * (1024 ** 3) * 0.90)
        
        # Runtime environment
        runtime_env = {}
        if suppress_warnings:
            runtime_env["env_vars"] = {"URBANOCR_SUPPRESS_WARNINGS": "true"}
        
        namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "urbanocr"
        
        init_kwargs = {
            "log_to_driver": True,
            "object_store_memory": obj_store_bytes,
            "namespace": str(namespace),
        }
        if runtime_env:
            init_kwargs["runtime_env"] = runtime_env
        if cpus_alloc and cpus_alloc > 0:
            init_kwargs["num_cpus"] = cpus_alloc
        
        ray.init(**init_kwargs)
        
    except Exception:
        try:
            ray.init(log_to_driver=True)
        except Exception:
            pass


def _resolve_model_path(model_source: str) -> str:
    """Resolve model path from zoo or HuggingFace Hub."""
    if os.path.isabs(model_source) and os.path.exists(model_source):
        return model_source
    
    zoo_path = os.path.join(MODEL_ZOO_BASE, model_source)
    if os.path.exists(zoo_path):
        return zoo_path
    
    if os.path.exists(MODEL_ZOO_BASE):
        try:
            zoo_dirs = [d for d in os.listdir(MODEL_ZOO_BASE)
                       if os.path.isdir(os.path.join(MODEL_ZOO_BASE, d))]
            for dir_name in zoo_dirs:
                if model_source.lower() in dir_name.lower() or dir_name.lower() in model_source.lower():
                    resolved = os.path.join(MODEL_ZOO_BASE, dir_name)
                    if os.path.exists(os.path.join(resolved, "config.json")):
                        return resolved
        except Exception:
            pass
    
    return model_source


def _detect_num_gpus() -> int:
    """Detect number of available GPUs."""
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            return len([x.strip() for x in cuda_visible.split(",") if x.strip()])
    except Exception:
        pass
    
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm_gpus:
            if ":" in slurm_gpus:
                return int(slurm_gpus.split(":")[-1])
            return int(slurm_gpus)
    except Exception:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    
    return 1


def _filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Filter unsupported vLLM engine kwargs."""
    try:
        import vllm
        accepted = None
        try:
            fields = getattr(getattr(vllm, "AsyncEngineArgs", None), "__dataclass_fields__", None)
            if isinstance(fields, dict) and fields:
                accepted = set(fields.keys())
        except Exception:
            pass
        
        if accepted:
            return {k: v for k, v in ek.items() if k in accepted}
    except Exception:
        pass
    
    ek = dict(ek)
    for k in ("use_v2_block_manager",):
        ek.pop(k, None)
    return ek


def run_ocr_stage(ds: Any, cfg: DictConfig) -> pd.DataFrame:
    """Run OCR inference using vLLM+Ray pipeline.
    
    Args:
        ds: Ray Dataset with images (from data handler)
        cfg: Configuration object
        
    Returns:
        DataFrame with flat text detections (one row per detection)
    """
    _suppress_multiprocessing_warnings(cfg)
    _ensure_ray_init(cfg)
    
    # Enable fallback to Arrow object extension types
    if _RAY_OK:
        try:
            from ray.data import DataContext
            DataContext.get_current().enable_fallback_to_arrow_object_ext_type = True
        except Exception:
            pass
    
    # Check if input is Ray Dataset
    is_ray_ds = hasattr(ds, "map_batches") and hasattr(ds, "count") and _RAY_OK
    
    if not is_ray_ds:
        if ds is None or (hasattr(ds, "__len__") and len(ds) == 0):
            return pd.DataFrame(columns=[
                "sample_id", "image_path", "text", 
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                "confidence", "confidence_numeric", "text_type", "detection_count"
            ])
        if not _RAY_OK:
            raise RuntimeError("Ray is required for OCR stage")
        ds = ray.data.from_pandas(ds)
    
    # Log dataset ready (lazy - don't force materialization)
    print(f"[ocr_stage] ✓ Dataset received from handler", flush=True)
    print(
        json.dumps({
            "ocr_stage": {
                "event": "dataset_ready",
                "timestamp": datetime.utcnow().isoformat(),
            }
        }),
        flush=True,
    )
    
    if np is None:
        raise RuntimeError("NumPy is required for OCR stage")
    
    # Get tiling configuration
    print(f"[ocr_stage] Loading tiling configuration...", flush=True)
    tiling_cfg = get_tiling_config(cfg)
    tiling_enabled = tiling_cfg.get("enabled", True)
    tile_size = tiling_cfg.get("tile_size", 1024)
    tile_overlap = tiling_cfg.get("overlap", 64)
    max_dimension = tiling_cfg.get("max_dimension", 2048)
    
    print(f"[ocr_stage] Tiling: enabled={tiling_enabled}, tile_size={tile_size}, overlap={tile_overlap}, max_dim={max_dimension}", flush=True)
    print(
        json.dumps({
            "ocr_stage": {
                "event": "tiling_config",
                "enabled": tiling_enabled,
                "tile_size": tile_size,
                "overlap": tile_overlap,
                "max_dimension": max_dimension,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }),
        flush=True,
    )
    
    # Tiling expansion function - expands one image row into multiple tile rows
    def _expand_to_tiles(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand a single image into tiles if needed."""
        image = row.get("image")
        if image is None:
            return [row]  # Pass through if no image
        
        # Ensure numpy array
        if not isinstance(image, np.ndarray):
            try:
                import pyarrow as pa
                if isinstance(image, pa.Tensor):
                    image = image.to_numpy()
                else:
                    image = np.asarray(image)
            except Exception:
                return [row]  # Can't tile, pass through
        
        # Check if tiling is needed
        if not tiling_enabled or not needs_tiling(image, max_dimension):
            # No tiling needed - add empty tile_info
            row_out = dict(row)
            row_out["_tile_info"] = None
            return [row_out]
        
        # Tile the image
        tiles = tile_image(image, tile_size=tile_size, overlap=tile_overlap)
        
        # Create a row for each tile
        tile_rows = []
        for tile_array, tile_info in tiles:
            tile_row = {}
            # Copy metadata from original row
            for key in ["sample_id", "image_path", "location_group", "location_id", "face", "path"]:
                if key in row:
                    tile_row[key] = row[key]
            
            # Replace image with tile
            tile_row["image"] = tile_array
            
            # Store tile info for coordinate transformation
            tile_row["_tile_info"] = tile_info.to_dict()
            
            # Add tile identifiers
            tile_row["tile_idx"] = tile_info.tile_idx
            tile_row["tile_row"] = tile_info.row
            tile_row["tile_col"] = tile_info.col
            
            tile_rows.append(tile_row)
        
        return tile_rows
    
    # Apply tiling expansion
    print(f"[ocr_stage] Applying tiling expansion to dataset...", flush=True)
    ds = ds.flat_map(_expand_to_tiles)
    print(f"[ocr_stage] ✓ Tiling expansion added to pipeline (lazy)", flush=True)
    
    # Log tiling applied (lazy - count happens during execution)
    print(
        json.dumps({
            "ocr_stage": {
                "event": "tiling_applied",
                "tile_size": tile_size,
                "overlap": tile_overlap,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }),
        flush=True,
    )
    
    # Resolve model path
    print(f"[ocr_stage] Resolving model path...", flush=True)
    model_source_raw = getattr(cfg.model, "model_source", "")
    resolved_model_source = _resolve_model_path(model_source_raw)
    
    print(f"[ocr_stage] ✓ Model resolved: {resolved_model_source}", flush=True)
    
    # Build engine config
    print(f"[ocr_stage] Building vLLM engine configuration...", flush=True)
    engine_kwargs = dict(getattr(cfg.model, "engine_kwargs", {}))
    engine_kwargs = _filter_vllm_engine_kwargs(engine_kwargs)
    
    # Set multimodal defaults
    engine_kwargs.setdefault("limit_mm_per_prompt", {"image": 1})
    engine_kwargs.setdefault("trust_remote_code", True)
    
    num_gpus = _detect_num_gpus()
    batch_size = getattr(cfg.model, "batch_size", 4)
    concurrency = getattr(cfg.model, "concurrency", 1)
    
    # Adjust for tensor parallelism
    tp_val = engine_kwargs.get("tensor_parallel_size", 1)
    if tp_val > 1:
        concurrency = max(1, num_gpus // tp_val)
    
    # Runtime environment
    runtime_env_vars = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        runtime_env_vars["HF_TOKEN"] = hf_token
    
    suppress_warnings = get_suppress_child_warnings(cfg)
    runtime_env_vars["URBANOCR_SUPPRESS_WARNINGS"] = "true" if suppress_warnings else "false"
    
    runtime_env = {"env_vars": runtime_env_vars} if runtime_env_vars else None
    
    accelerator_type = getattr(cfg.model, "accelerator_type", None)
    
    # Build vLLM processor config
    engine_config = vLLMEngineProcessorConfig(
        model_source=resolved_model_source,
        engine_kwargs=engine_kwargs,
        concurrency=concurrency,
        batch_size=batch_size,
        tokenize=False,
        apply_chat_template=True,
        has_image=True,
        accelerator_type=accelerator_type,
        runtime_env=runtime_env,
    )
    
    # Preprocessing function
    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess row for OCR inference."""
        _suppress_multiprocessing_warnings(cfg)
        _maybe_silence_vllm_logs()
        return ocr_preprocess(row, cfg)
    
    # Postprocessing function
    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess OCR response with tile coordinate transformation."""
        result = ocr_postprocess(row, cfg)
        
        # Transform bounding boxes from tile-local to full-resolution global coordinates
        tile_info_dict = row.get("_tile_info")
        if tile_info_dict is not None:
            tile_info = TileInfo.from_dict(tile_info_dict)
            detections = result.get("_detections", [])
            
            transformed_detections = []
            for det in detections:
                det_copy = dict(det)
                bbox = det_copy.get("bbox_2d") or det_copy.get("bbox")
                if bbox and isinstance(bbox, list) and len(bbox) >= 4:
                    det_copy["bbox_2d"] = transform_bbox_to_original(bbox, tile_info)
                transformed_detections.append(det_copy)
            
            result["_detections"] = transformed_detections
            
            # Preserve tile metadata
            result["tile_idx"] = tile_info.tile_idx
            result["tile_row"] = tile_info.row
            result["tile_col"] = tile_info.col
            result["n_tiles_x"] = tile_info.n_cols
            result["n_tiles_y"] = tile_info.n_rows
            
            # Add global coordinate range (max values for bbox coords)
            # Global coords are 0 to (N * 999) where N is tiles in that dimension
            bbox_max_x, bbox_max_y = get_global_bbox_range(tile_info)
            result["bbox_max_x"] = bbox_max_x
            result["bbox_max_y"] = bbox_max_y
        else:
            # Non-tiled image: standard 0-999 range
            result["n_tiles_x"] = 1
            result["n_tiles_y"] = 1
            result["bbox_max_x"] = 999
            result["bbox_max_y"] = 999
        
        return result
    
    # Build and run processor
    print(f"[ocr_stage] Building LLM processor (this may take a moment to load model)...", flush=True)
    processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
    print(f"[ocr_stage] ✓ LLM processor built successfully", flush=True)
    
    print(
        json.dumps({
            "ocr_stage": {
                "event": "processor_built",
                "model": resolved_model_source,
                "batch_size": batch_size,
                "concurrency": concurrency,
                "num_gpus": num_gpus,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }),
        flush=True,
    )
    
    print(f"[ocr_stage] Starting inference pipeline...", flush=True)
    ds_results = processor(ds)
    print(f"[ocr_stage] ✓ Inference pipeline created (lazy)", flush=True)
    
    # Flatten detections to individual rows
    def _flatten_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten detections from a single image/tile to multiple rows."""
        detections = row.get("_detections", [])
        
        # Collect metadata including tile info and coordinate ranges
        metadata = {}
        for key in ["sample_id", "image_path", "location_group", "location_id", "face", 
                    "ts_processed", "tile_idx", "tile_row", "tile_col",
                    "n_tiles_x", "n_tiles_y", "bbox_max_x", "bbox_max_y"]:
            if key in row:
                metadata[key] = row[key]
        
        return flatten_detections(detections, metadata)
    
    # Materialize results
    print(f"[ocr_stage] Materializing results (this is where actual processing happens)...", flush=True)
    results_list = []
    batch_count = 0
    
    for batch in ds_results.iter_batches(batch_size=100):
        batch_count += 1
        if batch_count % 10 == 1:
            print(f"[ocr_stage] Processing batch {batch_count}...", flush=True)
        if isinstance(batch, dict):
            # Convert dict of lists to list of dicts
            keys = list(batch.keys())
            if keys:
                n_rows = len(batch[keys[0]])
                for i in range(n_rows):
                    row = {k: batch[k][i] for k in keys}
                    flat_rows = _flatten_row(row)
                    results_list.extend(flat_rows)
        else:
            # Assume it's a pandas DataFrame or similar
            for _, row in batch.iterrows():
                row_dict = row.to_dict()
                flat_rows = _flatten_row(row_dict)
                results_list.extend(flat_rows)
    
    # Build output DataFrame
    if results_list:
        df_results = pd.DataFrame(results_list)
    else:
        df_results = pd.DataFrame(columns=[
            "sample_id", "image_path", "text",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "confidence", "confidence_numeric", "text_type", "detection_count"
        ])
    
    # Deduplicate: remove duplicate detections per image (model can repeat same detection)
    rows_before_dedup = len(df_results)
    dedup_cols = ["image_path", "text", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    # Only deduplicate if we have actual detections (not null rows)
    has_detections = df_results["text"].notna()
    df_with_text = df_results[has_detections].drop_duplicates(subset=dedup_cols, keep="first")
    df_no_text = df_results[~has_detections]
    df_results = pd.concat([df_with_text, df_no_text], ignore_index=True)
    
    rows_after_dedup = len(df_results)
    duplicates_removed = rows_before_dedup - rows_after_dedup
    
    if duplicates_removed > 0:
        print(f"[ocr_stage] Deduplication: removed {duplicates_removed} duplicate detections ({100*duplicates_removed/rows_before_dedup:.1f}%)", flush=True)
    
    # Update detection_count to reflect unique detections per image
    if len(df_results) > 0 and "image_path" in df_results.columns:
        detection_counts = df_results[df_results["text"].notna()].groupby("image_path").size()
        df_results["detection_count"] = df_results["image_path"].map(detection_counts).fillna(0).astype(int)
    
    print(
        json.dumps({
            "ocr_stage": {
                "event": "stage_completed",
                "output_rows": len(df_results),
                "rows_before_dedup": rows_before_dedup,
                "duplicates_removed": duplicates_removed,
                "timestamp": datetime.utcnow().isoformat(),
            }
        }),
        flush=True,
    )
    
    return df_results

