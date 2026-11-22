"""Combine two rasters into derived products.

Takes two single-band GeoTIFFs (e.g., relevance maps for different queries)
and produces derived rasters that capture the relationship between them.

Operations:
  - ``signed_diff``: A - B (positive = A dominates, negative = B dominates)
  - ``abs_diff``: |A - B| (classification confidence — high when one dominates)
  - ``max``: max(A, B) (either-or detection)
  - ``sum``: A + B (total evidence)

Artifacts produced:
  - <name>_signed_diff.tif   — Signed difference raster
  - <name>_abs_diff.tif      — Absolute difference raster
  - <name>_metadata.json     — Sidecar with input paths, operations, stats
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

TAG = "[combine]"


def run_combine_stage(cfg: DictConfig) -> str:
    """Combine two rasters into derived products.

    Args:
        cfg: Hydra config with ``combine`` section.

    Returns:
        Absolute path to the output directory.
    """
    import rasterio

    cc = cfg.combine

    raster_a_path = str(getattr(cc, "raster_a_path", "") or "")
    raster_b_path = str(getattr(cc, "raster_b_path", "") or "")
    if not raster_a_path or not raster_b_path:
        raise ValueError(
            "combine.raster_a_path and combine.raster_b_path must be set"
        )

    label_a = str(getattr(cc, "label_a", "A"))
    label_b = str(getattr(cc, "label_b", "B"))
    operations = list(getattr(cc, "operations", ["signed_diff", "abs_diff"]))
    output_name = str(getattr(cc, "output_name", "combined"))

    # Output directory
    output_path = str(getattr(cfg.runtime, "output_path", "") or "")
    if not output_path:
        output_path = "outputs/raster_combine"
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    print(
        f"{TAG} Combining rasters:\n"
        f"  A ({label_a}): {raster_a_path}\n"
        f"  B ({label_b}): {raster_b_path}\n"
        f"  Operations: {operations}",
        flush=True,
    )

    # Resolve paths — if directory, find the .tif inside
    raster_a_path = _resolve_tif_path(raster_a_path)
    raster_b_path = _resolve_tif_path(raster_b_path)

    # Load rasters
    with rasterio.open(raster_a_path) as src_a:
        data_a = src_a.read(1)
        profile_a = src_a.profile.copy()
        bounds_a = src_a.bounds
        tags_a = src_a.tags()

    with rasterio.open(raster_b_path) as src_b:
        data_b = src_b.read(1)
        bounds_b = src_b.bounds

    # Validate compatibility
    if data_a.shape != data_b.shape:
        raise ValueError(
            f"Raster shapes do not match: A={data_a.shape}, B={data_b.shape}. "
            "Both rasters must have the same grid dimensions."
        )
    if bounds_a != bounds_b:
        print(
            f"{TAG} WARNING: Bounds differ slightly. A={bounds_a}, B={bounds_b}. "
            "Proceeding with A's georeference.",
            flush=True,
        )

    print(
        f"{TAG} Raster shape: {data_a.shape}, "
        f"A valid: {np.sum(np.isfinite(data_a))}, "
        f"B valid: {np.sum(np.isfinite(data_b))}",
        flush=True,
    )

    # Compute products
    output_paths: Dict[str, str] = {}
    output_stats: Dict[str, Dict] = {}

    for op in operations:
        if op == "signed_diff":
            result = data_a - data_b
        elif op == "abs_diff":
            result = np.abs(data_a - data_b)
        elif op == "max":
            result = np.maximum(data_a, data_b)
        elif op == "sum":
            result = data_a + data_b
        else:
            print(f"{TAG} Unknown operation '{op}', skipping", flush=True)
            continue

        # Write output
        out_file = os.path.join(output_path, f"{output_name}_{op}.tif")
        _write_band(result, profile_a, out_file, tags={
            "operation": op,
            "label_a": label_a,
            "label_b": label_b,
            "source_a": raster_a_path,
            "source_b": raster_b_path,
        })
        output_paths[op] = out_file

        valid = result[np.isfinite(result)]
        output_stats[op] = {
            "min": float(valid.min()) if len(valid) > 0 else None,
            "max": float(valid.max()) if len(valid) > 0 else None,
            "mean": float(valid.mean()) if len(valid) > 0 else None,
            "std": float(valid.std()) if len(valid) > 0 else None,
        }
        print(
            f"{TAG} {op}: range=[{output_stats[op]['min']:.4f}, {output_stats[op]['max']:.4f}], "
            f"mean={output_stats[op]['mean']:.4f}",
            flush=True,
        )

    # Write metadata sidecar
    metadata = {
        "raster_a": raster_a_path,
        "raster_b": raster_b_path,
        "label_a": label_a,
        "label_b": label_b,
        "operations": operations,
        "grid_shape": list(data_a.shape),
        "output_paths": output_paths,
        "stats": output_stats,
    }
    meta_path = os.path.join(output_path, f"{output_name}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"{TAG} Wrote metadata: {meta_path}", flush=True)

    print(f"{TAG} Done. Output: {output_path}", flush=True)
    return output_path


def _resolve_tif_path(path: str) -> str:
    """If path is a directory, find the single .tif inside it."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        tifs = [f for f in os.listdir(path) if f.endswith(".tif")]
        if len(tifs) == 1:
            return os.path.join(path, tifs[0])
        elif len(tifs) == 0:
            raise FileNotFoundError(f"No .tif files found in {path}")
        else:
            raise ValueError(
                f"Multiple .tif files found in {path}: {tifs}. "
                "Set combine.raster_a_path / raster_b_path to the specific file."
            )
    return path


def _write_band(
    data: np.ndarray,
    profile: dict,
    output_path: str,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Write a single-band float32 GeoTIFF using an existing profile."""
    import rasterio

    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)
        if tags:
            dst.update_tags(**tags)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"{TAG} Wrote {output_path} ({size_mb:.1f} MB)", flush=True)
