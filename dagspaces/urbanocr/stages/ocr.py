"""OCR (Text Spotting) stage for Urban OCR pipeline.

This module provides text spotting functionality using Qwen3-VL:
- Text detection with bounding boxes (bbox_2d)
- Confidence scores and text type classification
- Batch inference with vLLM
- Flat parquet output (one row per detection)
- Automatic tiling for large images (e.g., 8192x8192 Cyclomedia faces)

Uses the same vLLM + Hydra pipeline infrastructure as urbanvqa.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os
import logging

import pandas as pd
from omegaconf import DictConfig

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

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
from dagspaces.common.multiprocessing_utils import get_suppress_child_warnings
from dagspaces.common.resource_tracker_patch import apply_patch as _apply_resource_tracker_patch

_apply_resource_tracker_patch()

# Model zoo base path
MODEL_ZOO_BASE = "/share/pierson/matt/zoo/models"

# vLLM logging state
_VLLM_LOGS_SILENCED = False

# Default chunk size for chunked processing (images per chunk)
DEFAULT_CHUNK_SIZE = 50


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


def _expand_tiles_pandas(df: pd.DataFrame, tiling_cfg: dict) -> pd.DataFrame:
    """Expand DataFrame rows into tile rows where tiling is needed.

    For images that exceed the maximum dimension, tiles are created and
    stored as numpy arrays in the ``image`` column. For images that do not
    need tiling, the image is loaded from disk and stored as-is.

    Args:
        df: DataFrame with ``image_path`` column.
        tiling_cfg: Dict with keys: enabled, tile_size, overlap, max_dimension.

    Returns:
        DataFrame with ``image`` (numpy array) and ``_tile_info`` columns added.
    """
    if np is None:
        raise RuntimeError("NumPy is required for OCR tiling")
    if not _PIL_AVAILABLE:
        raise RuntimeError("PIL is required for OCR image loading")

    enabled = tiling_cfg.get("enabled", True)
    tile_size = tiling_cfg.get("tile_size", 1024)
    overlap = tiling_cfg.get("overlap", 64)
    max_dimension = tiling_cfg.get("max_dimension", 2048)

    expanded_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        image_path = row_dict.get("image_path")
        if not image_path or not os.path.isfile(str(image_path)):
            continue

        # Load image
        img = PILImage.open(str(image_path)).convert("RGB")
        img_np = np.asarray(img)

        if not enabled or not needs_tiling(img_np, max_dimension):
            # No tiling needed — keep the full image
            row_dict["image"] = img_np
            row_dict["_tile_info"] = None
            expanded_rows.append(row_dict)
        else:
            # Tile the image
            tiles = tile_image(img_np, tile_size=tile_size, overlap=overlap)
            del img_np  # Free the full image

            for tile_array, tile_info in tiles:
                tile_row = {}
                # Copy metadata from original row
                for key in ["sample_id", "image_path", "location_group", "location_id", "face"]:
                    if key in row_dict:
                        tile_row[key] = row_dict[key]

                tile_row["image"] = tile_array
                tile_row["_tile_info"] = tile_info.to_dict()
                tile_row["tile_idx"] = tile_info.tile_idx
                tile_row["tile_row"] = tile_info.row
                tile_row["tile_col"] = tile_info.col
                expanded_rows.append(tile_row)

    if not expanded_rows:
        return pd.DataFrame()

    return pd.DataFrame(expanded_rows)


def _make_ocr_preprocess(cfg: DictConfig):
    """Build OCR preprocess callback for run_vllm_inference."""
    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        _suppress_multiprocessing_warnings(cfg)
        _maybe_silence_vllm_logs()
        return ocr_preprocess(row, cfg)
    return _pre


def _make_ocr_postprocess(cfg: DictConfig):
    """Build OCR postprocess callback with tile coordinate transformation."""
    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
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

            bbox_max_x, bbox_max_y = get_global_bbox_range(tile_info)
            result["bbox_max_x"] = bbox_max_x
            result["bbox_max_y"] = bbox_max_y
        else:
            result["n_tiles_x"] = 1
            result["n_tiles_y"] = 1
            result["bbox_max_x"] = 999
            result["bbox_max_y"] = 999

        return result
    return _post


def _flatten_detections_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten detections from inference results into individual rows."""
    results_list: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        detections = row_dict.get("_detections", [])

        metadata = {}
        for key in ["sample_id", "image_path", "location_group", "location_id", "face",
                     "ts_processed", "tile_idx", "tile_row", "tile_col",
                     "n_tiles_x", "n_tiles_y", "bbox_max_x", "bbox_max_y"]:
            if key in row_dict:
                metadata[key] = row_dict[key]

        flat_rows = flatten_detections(detections, metadata)
        results_list.extend(flat_rows)

    if results_list:
        return pd.DataFrame(results_list)
    return pd.DataFrame(columns=[
        "sample_id", "image_path", "text",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "confidence", "confidence_numeric", "text_type", "detection_count"
    ])


def run_ocr_stage(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run OCR inference using direct vLLM with chunked processing.

    Args:
        df: DataFrame with image_path column (from data handler)
        cfg: Configuration object

    Returns:
        DataFrame with flat text detections (one row per detection)
    """
    from dagspaces.common.vllm_inference import run_vllm_inference

    _suppress_multiprocessing_warnings(cfg)

    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "sample_id", "image_path", "text",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "confidence", "confidence_numeric", "text_type", "detection_count"
        ])

    # Convert non-DataFrame inputs
    if hasattr(df, "to_pandas") and not isinstance(df, pd.DataFrame):
        df = df.to_pandas()

    if np is None:
        raise RuntimeError("NumPy is required for OCR stage")

    # Get tiling configuration
    print(f"[ocr_stage] Loading tiling configuration...", flush=True)
    tiling_cfg = get_tiling_config(cfg)
    print(f"[ocr_stage] Tiling: enabled={tiling_cfg.get('enabled', True)}, "
          f"tile_size={tiling_cfg.get('tile_size', 1024)}, "
          f"overlap={tiling_cfg.get('overlap', 64)}, "
          f"max_dim={tiling_cfg.get('max_dimension', 2048)}", flush=True)

    # Build preprocess/postprocess callbacks
    preprocess = _make_ocr_preprocess(cfg)
    postprocess = _make_ocr_postprocess(cfg)

    # Process in chunks to manage memory (tiling can expand images significantly)
    chunk_size = int(getattr(getattr(cfg, "runtime", None), "ocr_chunk_size", DEFAULT_CHUNK_SIZE) or DEFAULT_CHUNK_SIZE)
    total_images = len(df)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, total_images, chunk_size)]

    print(f"[ocr_stage] Processing {total_images} images in {len(chunks)} chunk(s) "
          f"(chunk_size={chunk_size})", flush=True)

    all_results: List[pd.DataFrame] = []

    for chunk_idx, chunk_df in enumerate(chunks):
        print(f"[ocr_stage] Chunk {chunk_idx + 1}/{len(chunks)}: "
              f"tiling {len(chunk_df)} images...", flush=True)

        # Expand tiles for this chunk
        tiled_df = _expand_tiles_pandas(chunk_df, tiling_cfg)
        if tiled_df.empty:
            print(f"[ocr_stage] Chunk {chunk_idx + 1}: no valid images, skipping", flush=True)
            continue

        print(f"[ocr_stage] Chunk {chunk_idx + 1}: {len(chunk_df)} images -> "
              f"{len(tiled_df)} tiles/images, running inference...", flush=True)

        # Run inference on this chunk
        inferred = run_vllm_inference(
            df=tiled_df,
            cfg=cfg,
            preprocess=preprocess,
            postprocess=postprocess,
            stage_name="urbanocr_ocr",
        )

        # Flatten detections
        flattened = _flatten_detections_df(inferred)
        all_results.append(flattened)

        print(f"[ocr_stage] Chunk {chunk_idx + 1}: {len(flattened)} detections", flush=True)

    # Concat all chunk results
    if all_results:
        df_results = pd.concat(all_results, ignore_index=True)
    else:
        df_results = pd.DataFrame(columns=[
            "sample_id", "image_path", "text",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "confidence", "confidence_numeric", "text_type", "detection_count"
        ])

    # Deduplicate: remove duplicate detections per image
    rows_before_dedup = len(df_results)
    dedup_cols = ["image_path", "text", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    has_detections = df_results["text"].notna()
    df_with_text = df_results[has_detections].drop_duplicates(subset=dedup_cols, keep="first")
    df_no_text = df_results[~has_detections]
    df_results = pd.concat([df_with_text, df_no_text], ignore_index=True)

    rows_after_dedup = len(df_results)
    duplicates_removed = rows_before_dedup - rows_after_dedup

    if duplicates_removed > 0:
        print(f"[ocr_stage] Deduplication: removed {duplicates_removed} duplicate detections "
              f"({100 * duplicates_removed / rows_before_dedup:.1f}%)", flush=True)

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
