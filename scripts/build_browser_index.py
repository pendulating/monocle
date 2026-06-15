#!/usr/bin/env python3
"""Build browser-searchable index from embedding parquet output.

Reads parquet parts from the urbanembed pipeline, fits PCA for
dimensionality reduction, quantizes to uint8, and exports all
artifacts needed by the viz/embedding_search/ web app.

Requirements: numpy, pandas, pyarrow, scikit-learn, tqdm

Usage:
    python scripts/build_browser_index.py \
        --embed-dir outputs/embed/cyclomedia_20260407_060223 \
        --output-dir viz/embedding_search/public/data \
        --image-root /share/ju/cyclomedia/raw/manhattan_2025_1k
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_embeddings(
    embed_dir: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all parquet parts and return (embeddings matrix, metadata df)."""
    embed_path = Path(embed_dir)
    parts = sorted(embed_path.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet files found in {embed_dir}")

    logger.info("Loading %d parquet parts from %s", len(parts), embed_dir)
    frames = []
    for p in tqdm(parts, desc="Reading parquet"):
        frames.append(pd.read_parquet(p))

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows", len(df))

    # Stack embedding column into (N, D) matrix
    embeddings = np.stack(df["embedding"].values).astype(np.float32)
    logger.info("Embedding matrix shape: %s", embeddings.shape)

    # Extract metadata columns (drop embedding)
    meta_cols = [c for c in df.columns if c not in ("embedding",)]
    metadata = df[meta_cols].copy()

    return embeddings, metadata


def fit_pca(
    embeddings: np.ndarray, n_components: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA and return (reduced, components, mean)."""
    logger.info(
        "Fitting PCA: %d -> %d components", embeddings.shape[1], n_components
    )
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info("PCA explained variance: %.4f", variance_explained)
    return reduced, pca.components_, pca.mean_


def quantize_uint8(
    reduced: np.ndarray,
) -> tuple[np.ndarray, list[float], list[float]]:
    """Per-dimension min/max quantization to uint8."""
    dim_mins = reduced.min(axis=0)
    dim_maxs = reduced.max(axis=0)

    # Avoid division by zero for constant dimensions
    ranges = dim_maxs - dim_mins
    ranges[ranges == 0] = 1.0

    normalized = (reduced - dim_mins) / ranges  # [0, 1]
    quantized = np.clip(normalized * 255, 0, 255).astype(np.uint8)

    return quantized, dim_mins.tolist(), dim_maxs.tolist()


def build_manifest(
    metadata: pd.DataFrame, image_root: str | None
) -> list[dict]:
    """Build image manifest with optional path prefix stripping."""
    manifest = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Building manifest"):
        entry: dict = {}

        # Image path — prefer image_path_original (serveable) over image_path (scratch)
        image_path = ""
        for col in ("image_path_original", "image_path"):
            if col in row.index and pd.notna(row[col]):
                image_path = str(row[col])
                break
        if image_root and image_path.startswith(image_root):
            image_path = image_path[len(image_root) :].lstrip("/")
        entry["image_path"] = image_path

        # Standard metadata columns
        for col in ("recording_id", "face", "sample_id"):
            if col in row.index:
                val = row[col]
                entry[col] = str(val) if pd.notna(val) else None

        # Geo columns — prefer latitude/longitude, fall back to lat/lon
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


def export_artifacts(
    output_dir: str,
    quantized: np.ndarray,
    dim_mins: list[float],
    dim_maxs: list[float],
    components: np.ndarray,
    mean: np.ndarray,
    manifest: list[dict],
) -> None:
    """Write all browser search artifacts to output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Quantized index (flat uint8, row-major)
    index_path = out / "index.bin"
    quantized.tobytes()
    with open(index_path, "wb") as f:
        f.write(quantized.tobytes())
    logger.info("Wrote %s (%d bytes)", index_path, index_path.stat().st_size)

    # 2. Index metadata
    meta = {
        "dim": quantized.shape[1],
        "count": quantized.shape[0],
        "mins": dim_mins,
        "maxs": dim_maxs,
    }
    meta_path = out / "index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    logger.info("Wrote %s", meta_path)

    # 3. PCA components (D_out x D_in, float32)
    pca_comp_path = out / "pca_components.bin"
    with open(pca_comp_path, "wb") as f:
        f.write(components.astype(np.float32).tobytes())
    logger.info("Wrote %s (%d bytes)", pca_comp_path, pca_comp_path.stat().st_size)

    # 4. PCA mean (D_in, float32)
    pca_mean_path = out / "pca_mean.bin"
    with open(pca_mean_path, "wb") as f:
        f.write(mean.astype(np.float32).tobytes())
    logger.info("Wrote %s (%d bytes)", pca_mean_path, pca_mean_path.stat().st_size)

    # 5. Image manifest
    manifest_path = out / "image_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, separators=(",", ":"))
    logger.info("Wrote %s (%d entries)", manifest_path, len(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build browser search index from embedding parquet output"
    )
    parser.add_argument(
        "--embed-dir",
        required=True,
        help="Directory containing embedding parquet part files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for browser artifacts",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=256,
        help="PCA target dimensionality (default: 256)",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Image path prefix to strip for relative paths in manifest",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Load
    embeddings, metadata = load_embeddings(args.embed_dir)

    # PCA
    reduced, components, mean = fit_pca(embeddings, args.n_components)

    # Quantize
    quantized, dim_mins, dim_maxs = quantize_uint8(reduced)

    # Manifest
    manifest = build_manifest(metadata, args.image_root)

    # Export
    export_artifacts(
        args.output_dir,
        quantized,
        dim_mins,
        dim_maxs,
        components,
        mean,
        manifest,
    )

    logger.info("Done. Artifacts written to %s", args.output_dir)


if __name__ == "__main__":
    main()
