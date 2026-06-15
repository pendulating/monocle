"""Build browser-searchable index from embedding parquet output.

Reads embedding parquet parts from a prior embed stage, fits PCA for
dimensionality reduction, quantizes to uint8, and exports all artifacts
needed by the viz/embedding_search/ web app.

Artifacts produced:
  - index.bin         — Flat uint8 quantized embeddings (N × dim bytes)
  - index_meta.json   — Dimensions, count, per-dim min/max for dequantization
  - pca_components.bin — PCA projection matrix (dim × orig_dim, float32)
  - pca_mean.bin      — PCA mean vector (orig_dim, float32)
  - image_manifest.json — Row-aligned metadata (image_path, recording_id, lat/lon, etc.)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _load_embedding_parquet(path: str) -> pd.DataFrame:
    """Load embeddings from a single parquet file or a directory of parts."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        parts = sorted(Path(path).glob("part-*.parquet"))
        if not parts:
            parts = sorted(Path(path).glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet files found in {path}")
        print(f"[build_browser_index] Loading {len(parts)} parquet parts from {path}", flush=True)
        dfs = [pd.read_parquet(p) for p in tqdm(parts, desc="Reading parquet")]
        df = pd.concat(dfs, ignore_index=True)
    else:
        print(f"[build_browser_index] Loading parquet from {path}", flush=True)
        df = pd.read_parquet(path)
    print(f"[build_browser_index] Loaded {len(df)} rows", flush=True)
    return df


def _fit_pca(
    embeddings: np.ndarray, n_components: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit PCA and return (reduced, components, mean, explained_variance_sum)."""
    print(
        f"[build_browser_index] Fitting PCA: {embeddings.shape[1]} -> {n_components}",
        flush=True,
    )
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    variance_explained = float(pca.explained_variance_ratio_.sum())
    print(
        f"[build_browser_index] PCA explained variance: {variance_explained:.4f}",
        flush=True,
    )
    return reduced, pca.components_, pca.mean_, variance_explained


def _quantize_uint8(
    reduced: np.ndarray,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Per-dimension min/max quantization to uint8."""
    dim_mins = reduced.min(axis=0)
    dim_maxs = reduced.max(axis=0)

    ranges = dim_maxs - dim_mins
    ranges[ranges == 0] = 1.0

    normalized = (reduced - dim_mins) / ranges
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return quantized, dim_mins.tolist(), dim_maxs.tolist()


def _build_manifest(
    metadata: pd.DataFrame, image_root: Optional[str]
) -> list[dict]:
    """Build image manifest with optional path prefix stripping."""
    manifest = []
    for _, row in metadata.iterrows():
        entry: Dict[str, Any] = {}

        # Prefer image_path_original (serveable path) over image_path (scratch)
        image_path = ""
        for col in ("image_path_original", "image_path"):
            if col in row.index and pd.notna(row[col]):
                image_path = str(row[col])
                break
        if image_root and image_path.startswith(image_root):
            image_path = image_path[len(image_root):].lstrip("/")
        entry["image_path"] = image_path

        for col in ("recording_id", "face", "sample_id"):
            if col in row.index:
                val = row[col]
                entry[col] = str(val) if pd.notna(val) else None

        # lat/lon: prefer latitude/longitude, fall back to lat/lon
        for canonical, fallback in (("latitude", "lat"), ("longitude", "lon")):
            val = None
            if canonical in row.index and pd.notna(row[canonical]):
                val = float(row[canonical])
            elif fallback in row.index and pd.notna(row[fallback]):
                val = float(row[fallback])
            entry[canonical] = val

        if "yawDegrees" in row.index:
            val = row["yawDegrees"]
            entry["yawDegrees"] = float(val) if pd.notna(val) else None

        manifest.append(entry)

    return manifest


def _export_artifacts(
    output_dir: str,
    quantized: np.ndarray,
    dim_mins: list[float],
    dim_maxs: list[float],
    components: np.ndarray,
    mean: np.ndarray,
    manifest: list[dict],
) -> Dict[str, str]:
    """Write all browser search artifacts and return paths dict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # Quantized index
    index_path = out / "index.bin"
    with open(index_path, "wb") as f:
        f.write(quantized.tobytes())
    paths["index"] = str(index_path)
    print(f"[build_browser_index] Wrote {index_path} ({index_path.stat().st_size:,} bytes)", flush=True)

    # Index metadata
    meta = {
        "dim": int(quantized.shape[1]),
        "count": int(quantized.shape[0]),
        "mins": dim_mins,
        "maxs": dim_maxs,
    }
    meta_path = out / "index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    paths["index_meta"] = str(meta_path)

    # PCA components
    pca_comp_path = out / "pca_components.bin"
    with open(pca_comp_path, "wb") as f:
        f.write(components.astype(np.float32).tobytes())
    paths["pca_components"] = str(pca_comp_path)
    print(f"[build_browser_index] Wrote {pca_comp_path} ({pca_comp_path.stat().st_size:,} bytes)", flush=True)

    # PCA mean
    pca_mean_path = out / "pca_mean.bin"
    with open(pca_mean_path, "wb") as f:
        f.write(mean.astype(np.float32).tobytes())
    paths["pca_mean"] = str(pca_mean_path)

    # Image manifest
    manifest_path = out / "image_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    paths["manifest"] = str(manifest_path)
    print(f"[build_browser_index] Wrote {manifest_path} ({len(manifest)} entries)", flush=True)

    return paths


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def run_build_browser_index_stage(cfg: DictConfig) -> str:
    """Build browser search index from embedding parquet output.

    Args:
        cfg: Hydra config with ``browser_index`` and ``runtime`` sections.

    Returns:
        Absolute path to the output directory containing all artifacts.
    """
    # Resolve input path (from chained pipeline or config)
    embeddings_input = str(
        getattr(cfg.browser_index, "embeddings_input_path", "") or ""
    )
    if not embeddings_input:
        raise ValueError(
            "browser_index.embeddings_input_path must be set — "
            "either via pipeline chaining or CLI override"
        )

    # Resolve output path
    output_path = str(getattr(cfg.runtime, "output_path", "") or "")
    if not output_path:
        output_path = "outputs/browser_index"
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Config
    n_components = int(getattr(cfg.browser_index, "n_components", 256))
    image_root = str(getattr(cfg.browser_index, "image_root", "") or "")
    if hasattr(cfg, "data") and hasattr(cfg.data, "image_path") and not image_root:
        image_root = str(cfg.data.image_path)

    print(
        f"[build_browser_index] embeddings={embeddings_input}, "
        f"output={output_path}, n_components={n_components}, "
        f"image_root={image_root or '(none)'}",
        flush=True,
    )

    # Load embeddings
    df = _load_embedding_parquet(embeddings_input)

    embeddings = np.stack(df["embedding"].values).astype(np.float32)
    print(f"[build_browser_index] Embedding matrix: {embeddings.shape}", flush=True)

    meta_cols = [c for c in df.columns if c != "embedding"]
    metadata = df[meta_cols].copy()
    del df  # free memory

    # PCA
    reduced, components, mean, variance = _fit_pca(embeddings, n_components)
    del embeddings

    # Quantize
    quantized, dim_mins, dim_maxs = _quantize_uint8(reduced)
    del reduced

    # Manifest
    manifest = _build_manifest(metadata, image_root or None)

    # Export
    artifact_paths = _export_artifacts(
        output_path, quantized, dim_mins, dim_maxs, components, mean, manifest
    )

    # Write a summary JSON for pipeline metadata
    summary = {
        "n_vectors": int(quantized.shape[0]),
        "n_components": n_components,
        "pca_explained_variance": round(variance, 4),
        "artifacts": artifact_paths,
    }
    summary_path = os.path.join(output_path, "build_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[build_browser_index] Done. {quantized.shape[0]} vectors indexed.", flush=True)
    return output_path
