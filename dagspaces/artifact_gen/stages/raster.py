"""Generate a GeoTIFF raster of text-query relevance over geolocated image embeddings.

Takes pre-computed embeddings (from urbanembed) with lat/lon coordinates and a text
query string, computes cosine similarity between the query and all image embeddings,
then interpolates the scores onto a regular geographic grid.

Two interpolation modes:
  - ``idw``: Isotropic inverse distance weighting (baseline).
  - ``rays``: Directional ray accumulation — each face image casts a ray in its
    bearing direction with linear distance decay. Converging rays amplify scores,
    encoding spatial confidence.

Output: a single-band float32 GeoTIFF where each pixel value is the estimated
relevance of the query at that geographic location, normalized to [0, 1].

Artifacts produced:
  - <query>.tif            — GeoTIFF raster (float32, single band, with CRS)
  - <query>_metadata.json  — Sidecar with query, bbox, resolution, score stats
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

logger = logging.getLogger(__name__)

TAG = "[raster]"


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------


def _load_and_filter_embeddings(
    path: str,
    bbox: list[float],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load embedding parquet(s) and filter to bounding box.

    Args:
        path: Path to a single parquet or directory of part-*.parquet files.
        bbox: [west, south, east, north] in EPSG:4326.

    Returns:
        coords: (N, 2) float64 array of [longitude, latitude].
        embeddings: (N, D) float32 array.
        metadata: DataFrame with remaining columns (face, yawDegrees, etc.).
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        parts = sorted(Path(path).glob("part-*.parquet"))
        if not parts:
            parts = sorted(Path(path).glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet files found in {path}")
        print(f"{TAG} Loading {len(parts)} parquet parts from {path}", flush=True)
    else:
        parts = [Path(path)]
        print(f"{TAG} Loading parquet from {path}", flush=True)

    # Detect lat/lon column names from the first file's schema
    import pyarrow.parquet as pq
    first_cols = set(pq.read_schema(parts[0]).names)
    lat_col, lon_col = None, None
    # Peek at first file to find columns with actual data
    coord_cols = [c for c in ["latitude", "longitude", "lat", "lon"] if c in first_cols]
    peek_table = pq.read_table(parts[0], columns=coord_cols)
    peek = peek_table.slice(0, min(1000, len(peek_table))).to_pandas()
    del peek_table
    for candidate_lat, candidate_lon in [("latitude", "longitude"), ("lat", "lon")]:
        if candidate_lat in peek.columns and candidate_lon in peek.columns:
            if peek[candidate_lat].notna().any():
                lat_col, lon_col = candidate_lat, candidate_lon
                break
    del peek
    if lat_col is None:
        if "lat" in first_cols:
            lat_col, lon_col = "lat", "lon"
        elif "latitude" in first_cols:
            lat_col, lon_col = "latitude", "longitude"
        else:
            raise ValueError(
                f"Parquet must contain latitude/longitude (or lat/lon) columns. "
                f"Found: {sorted(first_cols)}"
            )
    print(f"{TAG} Using coordinate columns: {lat_col}, {lon_col}", flush=True)

    # Load part-by-part with bbox filter BEFORE concat to limit memory
    west, south, east, north = bbox
    filtered_parts = []
    total_loaded = 0
    for p in tqdm(parts, desc="Reading parquet"):
        df_part = pd.read_parquet(p)
        total_loaded += len(df_part)
        # Filter: valid coords + bbox
        mask = (
            df_part[lat_col].notna()
            & df_part[lon_col].notna()
            & df_part["embedding"].notna()
            & (df_part[lon_col] >= west)
            & (df_part[lon_col] <= east)
            & (df_part[lat_col] >= south)
            & (df_part[lat_col] <= north)
        )
        kept = df_part[mask]
        if len(kept) > 0:
            filtered_parts.append(kept)
        del df_part

    if not filtered_parts:
        raise ValueError(
            f"No geolocated embeddings found within bbox {bbox}. "
            f"Loaded {total_loaded} total rows. "
            "Check embeddings_input_path and bbox settings."
        )

    df = pd.concat(filtered_parts, ignore_index=True)
    del filtered_parts
    print(
        f"{TAG} {len(df)} points after bbox filter (from {total_loaded} total)",
        flush=True,
    )

    coords = np.column_stack([
        df[lon_col].values.astype(np.float64),
        df[lat_col].values.astype(np.float64),
    ])
    embeddings = np.stack(df["embedding"].values).astype(np.float32)

    # Return metadata columns (everything except embedding)
    meta_cols = [c for c in df.columns if c != "embedding"]
    metadata = df[meta_cols].copy()

    print(f"{TAG} Embedding matrix: {embeddings.shape}", flush=True)
    return coords, embeddings, metadata


# ---------------------------------------------------------------------------
# Query encoding
# ---------------------------------------------------------------------------


def _load_pca_artifacts(
    pca_dir: str, pca_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load PCA components and mean from binary files."""
    pca_dir = os.path.abspath(pca_dir)
    components_raw = np.frombuffer(
        Path(pca_dir, "pca_components.bin").read_bytes(), dtype=np.float32
    )
    components = components_raw.reshape(pca_dim, -1)
    mean = np.frombuffer(
        Path(pca_dir, "pca_mean.bin").read_bytes(), dtype=np.float32
    )
    print(f"{TAG} PCA components: {components.shape}, mean: {mean.shape}", flush=True)
    return components, mean


def _reduce_embeddings_pca(
    embeddings: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Apply pre-fitted PCA: reduced = (emb - mean) @ components.T"""
    reduced = (embeddings - mean) @ components.T
    return reduced.astype(np.float32)


def _encode_query_bge(
    query_text: str,
    bge_model: str,
    w_proj_path: str,
    pca_dim: int,
) -> np.ndarray:
    """Encode query text via BGE-small + learned projection to PCA space.

    Returns L2-normalized (pca_dim,) float32 vector.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"{TAG} Encoding query with {bge_model}: '{query_text}'", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(bge_model)
    model = AutoModel.from_pretrained(bge_model)
    model.eval()

    with torch.no_grad():
        encoded = tokenizer(
            [query_text], padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        outputs = model(**encoded)
        bge_emb = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)

    # L2-normalize BGE embedding
    norm = np.linalg.norm(bge_emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    bge_emb = bge_emb / norm
    bge_emb = bge_emb[0]  # (384,)

    # Load projection matrix: (pca_dim, bge_dim)
    w_proj_path = os.path.abspath(w_proj_path)
    # W_proj.bin may be in the directory itself
    if os.path.isdir(w_proj_path):
        w_proj_file = os.path.join(w_proj_path, "W_proj.bin")
    else:
        w_proj_file = w_proj_path

    w_proj_raw = np.frombuffer(
        Path(w_proj_file).read_bytes(), dtype=np.float32
    )
    bge_dim = bge_emb.shape[0]
    w_proj = w_proj_raw.reshape(pca_dim, bge_dim)
    print(f"{TAG} W_proj: {w_proj.shape}", flush=True)

    # Project: q_pca = W_proj @ bge_emb
    q_pca = w_proj @ bge_emb

    # L2-normalize
    q_norm = np.linalg.norm(q_pca)
    if q_norm > 0:
        q_pca = q_pca / q_norm

    return q_pca.astype(np.float32)


def _encode_query_qwen(
    query_text: str,
    model_path: str,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
) -> np.ndarray:
    """Encode query text directly with Qwen3-VL-Embedding + PCA.

    Requires GPU. Returns L2-normalized (pca_dim,) float32 vector.
    """
    from vllm import LLM

    print(f"{TAG} Encoding query with Qwen ({model_path}): '{query_text}'", flush=True)
    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=512,
    )

    outputs = llm.embed([query_text])
    emb = np.array(outputs[0].outputs.embedding, dtype=np.float32)

    # PCA reduce
    q_pca = (emb - pca_mean) @ pca_components.T

    # L2-normalize
    q_norm = np.linalg.norm(q_pca)
    if q_norm > 0:
        q_pca = q_pca / q_norm

    # Free GPU
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return q_pca.astype(np.float32)


def _encode_query_qwen_raw(
    query_text: str,
    model_path: str,
) -> np.ndarray:
    """Encode query text with Qwen3-VL-Embedding in native embedding space.

    No PCA reduction — returns the raw embedding vector.
    Requires GPU. Returns L2-normalized (D,) float32 vector.
    """
    from vllm import LLM

    print(f"{TAG} Encoding query with Qwen raw ({model_path}): '{query_text}'", flush=True)
    llm = LLM(
        model=model_path,
        runner="pooling",
        trust_remote_code=True,
        dtype="float16",
        max_model_len=512,
    )

    outputs = llm.embed([query_text])
    emb = np.array(outputs[0].outputs.embedding, dtype=np.float32)
    print(f"{TAG} Raw query embedding: {emb.shape}", flush=True)

    # L2-normalize
    q_norm = np.linalg.norm(emb)
    if q_norm > 0:
        emb = emb / q_norm

    # Free GPU
    del llm
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _compute_similarity(
    query_vec: np.ndarray, embeddings: np.ndarray
) -> np.ndarray:
    """Cosine similarity between query and all embeddings (both L2-normalized)."""
    # L2-normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_normed = embeddings / norms

    scores = emb_normed @ query_vec  # (N,)
    return scores


# ---------------------------------------------------------------------------
# Grid construction & interpolation
# ---------------------------------------------------------------------------


def _build_grid(
    bbox: list[float],
    resolution_m: float,
    working_crs: str,
) -> Tuple[np.ndarray, np.ndarray, int, int, dict]:
    """Build a regular grid in projected CRS.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.
        resolution_m: Grid cell size in meters.
        working_crs: Projected CRS for meter-accurate grid (e.g. EPSG:32618).

    Returns:
        grid_points_proj: (H*W, 2) array of grid cell centers in projected CRS.
        grid_points_geo: (H*W, 2) array of grid cell centers in EPSG:4326.
        width: Number of columns.
        height: Number of rows.
        grid_info: Dict with transform parameters for GeoTIFF output.
    """
    from pyproj import Transformer

    west, south, east, north = bbox

    # Transform bbox to working CRS
    to_proj = Transformer.from_crs("EPSG:4326", working_crs, always_xy=True)
    to_geo = Transformer.from_crs(working_crs, "EPSG:4326", always_xy=True)

    px_west, px_south = to_proj.transform(west, south)
    px_east, px_north = to_proj.transform(east, north)

    # Build grid in projected space
    width = int(np.ceil((px_east - px_west) / resolution_m))
    height = int(np.ceil((px_north - px_south) / resolution_m))

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid grid dimensions ({width}x{height}). "
            f"Check bbox and resolution_m."
        )

    print(
        f"{TAG} Grid: {width}x{height} pixels at {resolution_m}m resolution "
        f"(projected bbox: [{px_west:.0f}, {px_south:.0f}, {px_east:.0f}, {px_north:.0f}])",
        flush=True,
    )

    # Cell centers
    xs = np.linspace(px_west + resolution_m / 2, px_east - resolution_m / 2, width)
    ys = np.linspace(px_north - resolution_m / 2, px_south + resolution_m / 2, height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_points_proj = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Transform grid points back to geographic for GeoTIFF metadata
    geo_x, geo_y = to_geo.transform(grid_points_proj[:, 0], grid_points_proj[:, 1])
    grid_points_geo = np.column_stack([geo_x, geo_y])

    grid_info = {
        "proj_bbox": [px_west, px_south, px_east, px_north],
        "geo_bbox": bbox,
        "working_crs": working_crs,
        "width": width,
        "height": height,
    }

    return grid_points_proj, grid_points_geo, width, height, grid_info


def _interpolate_idw(
    point_coords: np.ndarray,
    scores: np.ndarray,
    grid_points: np.ndarray,
    power: float = 2.0,
    max_neighbors: int = 50,
    max_distance: float = 100.0,
) -> np.ndarray:
    """Inverse distance weighting interpolation.

    Args:
        point_coords: (N, 2) array of data point positions (projected CRS).
        scores: (N,) similarity scores.
        grid_points: (M, 2) array of grid cell centers (projected CRS).
        power: IDW power parameter.
        max_neighbors: Maximum neighbors per grid cell.
        max_distance: Maximum search distance in meters.

    Returns:
        (M,) array of interpolated values, NaN where no neighbors found.
    """
    from scipy.spatial import cKDTree

    print(
        f"{TAG} IDW interpolation: {len(point_coords)} points -> {len(grid_points)} grid cells "
        f"(power={power}, k={max_neighbors}, max_dist={max_distance}m)",
        flush=True,
    )

    tree = cKDTree(point_coords)

    # Query all grid points at once
    distances, indices = tree.query(
        grid_points,
        k=min(max_neighbors, len(point_coords)),
        distance_upper_bound=max_distance,
        workers=-1,
    )

    # Handle single-neighbor case (query returns 1D arrays)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    result = np.full(len(grid_points), np.nan, dtype=np.float32)

    # Mask for valid (within distance) neighbors
    valid = np.isfinite(distances) & (indices < len(scores))

    # Process in vectorized fashion
    for i in range(len(grid_points)):
        neighbor_mask = valid[i]
        if not np.any(neighbor_mask):
            continue

        d = distances[i, neighbor_mask]
        idx = indices[i, neighbor_mask]

        # Handle exact coincidence (distance = 0)
        zero_dist = d == 0
        if np.any(zero_dist):
            result[i] = np.mean(scores[idx[zero_dist]])
            continue

        weights = 1.0 / np.power(d, power)
        result[i] = np.sum(weights * scores[idx]) / np.sum(weights)

    n_valid = np.sum(np.isfinite(result))
    n_total = len(result)
    print(
        f"{TAG} IDW complete: {n_valid}/{n_total} grid cells have data "
        f"({n_total - n_valid} NoData)",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# Ray accumulation interpolation
# ---------------------------------------------------------------------------

# Face bearing offsets (degrees relative to recording yaw).
# Matches dagspaces/urbanroamvqa/graph/street_graph.py:FACE_BEARING_DEG
_FACE_BEARING_DEG: Dict[str, float] = {
    "F": 0.0,
    "R": 90.0,
    "B": 180.0,
    "L": 270.0,
}


def _compute_face_bearings(
    metadata: pd.DataFrame,
) -> np.ndarray:
    """Compute absolute bearing for each image from yawDegrees + face offset.

    Args:
        metadata: DataFrame with ``face`` and ``yawDegrees``/``recorderDirection`` columns.

    Returns:
        (N,) float64 array of absolute bearings in degrees [0, 360).
    """
    # Resolve yaw column
    yaw_col = None
    for col in ("recorderDirection", "yawDegrees", "orientation"):
        if col in metadata.columns:
            yaw_col = col
            break
    if yaw_col is None:
        raise ValueError(
            "Ray interpolation requires yawDegrees or recorderDirection column "
            "in the embeddings parquet. Ensure the dataset was created with "
            "catalog enrichment (--catalog_csv)."
        )
    if "face" not in metadata.columns:
        raise ValueError(
            "Ray interpolation requires a 'face' column in the embeddings parquet."
        )

    yaw_values = metadata[yaw_col].values.astype(np.float64)

    # Auto-detect radians (values < 7 are likely radians)
    if np.nanmedian(np.abs(yaw_values)) < 7.0:
        print(f"{TAG} Detected radians in {yaw_col}, converting to degrees", flush=True)
        yaw_values = np.degrees(yaw_values)

    # Compute face offsets
    face_offsets = metadata["face"].map(_FACE_BEARING_DEG).values.astype(np.float64)

    # Rows with unknown faces (U, D, or missing) get NaN
    nan_mask = np.isnan(face_offsets)
    if np.any(nan_mask):
        n_bad = int(np.sum(nan_mask))
        print(f"{TAG} {n_bad} images have non-horizontal faces (U/D), will be skipped", flush=True)

    bearings = (yaw_values + face_offsets) % 360.0
    return bearings


def _bresenham_line(
    r0: int, c0: int, r1: int, c1: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bresenham's line algorithm returning (rows, cols) arrays.

    Generates all integer grid cells along the line from (r0, c0) to (r1, c1).
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1

    n = max(dr, dc) + 1
    rows = np.empty(n, dtype=np.int64)
    cols = np.empty(n, dtype=np.int64)

    r, c = r0, c0
    err = dr - dc
    for i in range(n):
        rows[i] = r
        cols[i] = c
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc

    return rows, cols


def _interpolate_rays(
    point_coords_proj: np.ndarray,
    scores: np.ndarray,
    bearings_deg: np.ndarray,
    width: int,
    height: int,
    proj_bbox: list[float],
    resolution_m: float,
    ray_length_m: float = 30.0,
    ray_decay: str = "linear",
    normalize_by_count: bool = False,
) -> np.ndarray:
    """Directional ray accumulation interpolation.

    For each image, casts a ray of ``ray_length_m`` in the face's absolute bearing
    direction. Each cell the ray passes through accumulates the image's score,
    weighted by distance decay along the ray.

    Args:
        point_coords_proj: (N, 2) image positions in projected CRS (meters).
        scores: (N,) similarity scores per image.
        bearings_deg: (N,) absolute bearing of each face image (degrees, 0=north).
        width: Grid width in pixels.
        height: Grid height in pixels.
        proj_bbox: [west, south, east, north] in projected CRS.
        resolution_m: Cell size in meters.
        ray_length_m: Length of each ray in meters.
        ray_decay: Decay function along the ray: "linear" or "none".
        normalize_by_count: If True, divide accumulated score by ray count per cell.

    Returns:
        (H*W,) flat array of interpolated values, NaN where no rays reached.
    """
    px_west, px_south, px_east, px_north = proj_bbox

    accumulator = np.zeros((height, width), dtype=np.float64)
    ray_count = np.zeros((height, width), dtype=np.int32)

    # Convert bearing from geographic (0=N, CW) to math angle (0=E, CCW)
    # dx = sin(bearing), dy = cos(bearing) in geographic convention
    bearings_rad = np.radians(bearings_deg)
    dx = np.sin(bearings_rad)  # east component
    dy = np.cos(bearings_rad)  # north component

    # Valid mask: finite bearing and score
    valid = np.isfinite(bearings_deg) & np.isfinite(scores)
    n_valid = int(np.sum(valid))
    n_total = len(scores)
    print(
        f"{TAG} Ray accumulation: {n_valid}/{n_total} images with valid bearings, "
        f"ray_length={ray_length_m}m, decay={ray_decay}, "
        f"normalize_by_count={normalize_by_count}",
        flush=True,
    )

    ray_length_px = ray_length_m / resolution_m

    n_rays_cast = 0
    for i in range(n_total):
        if not valid[i]:
            continue

        score = scores[i]
        x0 = point_coords_proj[i, 0]
        y0 = point_coords_proj[i, 1]

        # Ray endpoint in projected CRS
        x1 = x0 + dx[i] * ray_length_m
        y1 = y0 + dy[i] * ray_length_m

        # Convert to grid coordinates (col, row)
        # Col increases east (x), row increases downward (north to south)
        c0 = int((x0 - px_west) / resolution_m)
        r0 = int((px_north - y0) / resolution_m)
        c1 = int((x1 - px_west) / resolution_m)
        r1 = int((px_north - y1) / resolution_m)

        # Clip endpoints to grid bounds for Bresenham
        # (Bresenham will generate points between them; we clip after)
        if (c0 < 0 and c1 < 0) or (c0 >= width and c1 >= width):
            continue
        if (r0 < 0 and r1 < 0) or (r0 >= height and r1 >= height):
            continue

        rows, cols = _bresenham_line(r0, c0, r1, c1)

        # Clip to grid bounds
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows = rows[in_bounds]
        cols = cols[in_bounds]

        if len(rows) == 0:
            continue

        n_rays_cast += 1

        if ray_decay == "linear":
            # Distance from origin along ray, normalized to [0, 1]
            pixel_dists = np.sqrt(
                ((cols - c0) * resolution_m) ** 2
                + ((rows - r0) * resolution_m) ** 2
            )
            t = pixel_dists / ray_length_m
            t = np.clip(t, 0.0, 1.0)
            weights = score * (1.0 - t)
        else:
            # No decay: uniform score along ray
            weights = np.full(len(rows), score, dtype=np.float64)

        # Accumulate
        np.add.at(accumulator, (rows, cols), weights)
        np.add.at(ray_count, (rows, cols), 1)

    print(f"{TAG} Cast {n_rays_cast} rays", flush=True)

    # Build result
    result = accumulator.ravel().astype(np.float32)
    count_flat = ray_count.ravel()

    if normalize_by_count:
        # Divide by ray count where count > 0
        has_rays = count_flat > 0
        result[has_rays] = result[has_rays] / count_flat[has_rays].astype(np.float32)
        result[~has_rays] = np.nan
    else:
        # Raw accumulation: set cells with no rays to NaN
        result[count_flat == 0] = np.nan

    n_covered = int(np.sum(count_flat > 0))
    n_cells = width * height
    print(
        f"{TAG} Ray accumulation complete: {n_covered}/{n_cells} cells covered "
        f"({100 * n_covered / n_cells:.1f}%)",
        flush=True,
    )

    return result


# ---------------------------------------------------------------------------
# GeoTIFF output
# ---------------------------------------------------------------------------


def _write_geotiff(
    grid_values: np.ndarray,
    width: int,
    height: int,
    bbox: list[float],
    crs: str,
    output_path: str,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Write a single-band float32 GeoTIFF.

    Args:
        grid_values: (H*W,) flat array of pixel values.
        width: Raster width in pixels.
        height: Raster height in pixels.
        bbox: [west, south, east, north] in the output CRS.
        crs: Output CRS string (e.g. 'EPSG:4326').
        output_path: Path to write the GeoTIFF.
        tags: Optional metadata tags to embed in the GeoTIFF.

    Returns:
        Absolute path to the written file.
    """
    import rasterio
    from rasterio.transform import from_bounds

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    west, south, east, north = bbox
    transform = from_bounds(west, south, east, north, width, height)

    grid_2d = grid_values.reshape(height, width)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="deflate",
    ) as dst:
        dst.write(grid_2d, 1)
        if tags:
            dst.update_tags(**tags)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"{TAG} Wrote GeoTIFF: {output_path} ({size_mb:.1f} MB)", flush=True)
    return output_path


# ---------------------------------------------------------------------------
# Stage entrypoint
# ---------------------------------------------------------------------------


def run_raster_stage(cfg: DictConfig) -> str:
    """Generate a GeoTIFF raster of query relevance from geolocated embeddings.

    Args:
        cfg: Hydra config with ``raster`` and ``runtime`` sections.

    Returns:
        Absolute path to the output directory containing the GeoTIFF and metadata.
    """
    from pyproj import Transformer

    rc = cfg.raster

    # Resolve paths
    embeddings_path = str(getattr(rc, "embeddings_input_path", "") or "")
    if not embeddings_path:
        raise ValueError(
            "raster.embeddings_input_path must be set — "
            "either via pipeline chaining or CLI override"
        )

    pca_dir = str(getattr(rc, "pca_artifacts_path", "") or "")

    query_text = str(getattr(rc, "query_text", "") or "")
    if not query_text:
        raise ValueError("raster.query_text must be set")

    # Output directory
    output_path = str(getattr(cfg.runtime, "output_path", "") or "")
    if not output_path:
        output_path = "outputs/raster"
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    # Config values
    query_mode = str(getattr(rc, "query_mode", "bge_projection"))
    bbox = list(rc.bbox)
    resolution_m = float(rc.resolution_m)
    working_crs = str(rc.working_crs)
    output_crs = str(rc.crs)
    idw_power = float(rc.idw_power)
    max_neighbors = int(rc.max_neighbors)
    max_distance_m = float(rc.max_distance_m)
    normalization = str(getattr(rc, "normalization", "minmax"))
    pca_dim = int(getattr(rc, "pca_dim", 256))

    interpolation = str(getattr(rc, "interpolation", "idw"))

    print(
        f"{TAG} query='{query_text}', mode={query_mode}, interpolation={interpolation}, "
        f"bbox={bbox}, res={resolution_m}m, crs={output_crs}",
        flush=True,
    )

    # Step 1: Load and filter embeddings
    coords_geo, embeddings, row_metadata = _load_and_filter_embeddings(
        embeddings_path, bbox
    )

    # Step 2 & 3: Encode query and compute similarity
    # qwen_raw skips PCA entirely; other modes reduce embeddings first
    if query_mode == "qwen_raw":
        model_path = str(cfg.model.model_source)
        query_vec = _encode_query_qwen_raw(query_text, model_path)
        # Cosine similarity directly in native embedding space
        scores = _compute_similarity(query_vec, embeddings)
        del embeddings
    else:
        if not pca_dir:
            raise ValueError(
                "raster.pca_artifacts_path must be set for bge_projection/qwen_direct modes"
            )
        pca_components, pca_mean = _load_pca_artifacts(pca_dir, pca_dim)
        emb_reduced = _reduce_embeddings_pca(embeddings, pca_components, pca_mean)
        del embeddings

        if query_mode == "bge_projection":
            proj_path = str(getattr(rc, "projection_matrix_path", "") or "")
            if not proj_path:
                raise ValueError(
                    "raster.projection_matrix_path must be set for bge_projection mode"
                )
            bge_model = str(getattr(rc, "bge_model", "BAAI/bge-small-en-v1.5"))
            query_vec = _encode_query_bge(query_text, bge_model, proj_path, pca_dim)
        elif query_mode == "qwen_direct":
            model_path = str(cfg.model.model_source)
            query_vec = _encode_query_qwen(
                query_text, model_path, pca_components, pca_mean
            )
        else:
            raise ValueError(f"Unknown query_mode: {query_mode}")

        scores = _compute_similarity(query_vec, emb_reduced)
        del emb_reduced

    print(
        f"{TAG} Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
        f"mean={scores.mean():.4f}, std={scores.std():.4f}",
        flush=True,
    )

    # Step 5: Build grid
    grid_points_proj, grid_points_geo, width, height, grid_info = _build_grid(
        bbox, resolution_m, working_crs
    )
    proj_bbox = grid_info["proj_bbox"]

    # Project data point coordinates to working CRS
    to_proj = Transformer.from_crs("EPSG:4326", working_crs, always_xy=True)
    proj_x, proj_y = to_proj.transform(coords_geo[:, 0], coords_geo[:, 1])
    point_coords_proj = np.column_stack([proj_x, proj_y])

    # Step 6: Interpolation
    if interpolation == "idw":
        grid_values = _interpolate_idw(
            point_coords_proj, scores, grid_points_proj,
            power=idw_power,
            max_neighbors=max_neighbors,
            max_distance=max_distance_m,
        )
        interp_metadata = {
            "method": "idw",
            "power": idw_power,
            "max_neighbors": max_neighbors,
            "max_distance_m": max_distance_m,
        }
    elif interpolation == "rays":
        # Compute face bearings from metadata
        bearings = _compute_face_bearings(row_metadata)

        ray_length_m = float(getattr(rc, "ray_length_m", 30.0))
        ray_decay = str(getattr(rc, "ray_decay", "linear"))
        ray_normalize = bool(getattr(rc, "ray_normalize_by_count", False))

        grid_values = _interpolate_rays(
            point_coords_proj, scores, bearings,
            width=width,
            height=height,
            proj_bbox=proj_bbox,
            resolution_m=resolution_m,
            ray_length_m=ray_length_m,
            ray_decay=ray_decay,
            normalize_by_count=ray_normalize,
        )
        interp_metadata = {
            "method": "rays",
            "ray_length_m": ray_length_m,
            "ray_decay": ray_decay,
            "normalize_by_count": ray_normalize,
        }
    else:
        raise ValueError(
            f"Unknown interpolation method: {interpolation}. "
            "Use 'idw' or 'rays'."
        )

    # Step 7: Normalize to [0, 1]
    valid_mask = np.isfinite(grid_values)
    if np.any(valid_mask):
        valid_vals = grid_values[valid_mask]
        if normalization == "minmax":
            vmin, vmax = valid_vals.min(), valid_vals.max()
            if vmax > vmin:
                grid_values[valid_mask] = (valid_vals - vmin) / (vmax - vmin)
            else:
                grid_values[valid_mask] = 0.5
        elif normalization == "quantile":
            from scipy.stats import rankdata
            ranks = rankdata(valid_vals, method="average")
            grid_values[valid_mask] = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else 0.5
        print(
            f"{TAG} Normalized ({normalization}): "
            f"min={grid_values[valid_mask].min():.4f}, max={grid_values[valid_mask].max():.4f}",
            flush=True,
        )

    # Step 8: Write GeoTIFF
    safe_query = "".join(c if c.isalnum() or c in " _-" else "_" for c in query_text)
    safe_query = safe_query.replace(" ", "_")[:80]
    tif_path = os.path.join(output_path, f"{safe_query}.tif")

    tags = {
        "query": query_text,
        "interpolation": interpolation,
        "resolution_m": str(resolution_m),
        "normalization": normalization,
        "n_points": str(len(scores)),
    }
    if interpolation == "idw":
        tags["idw_power"] = str(idw_power)
        tags["max_distance_m"] = str(max_distance_m)
        tags["max_neighbors"] = str(max_neighbors)
    elif interpolation == "rays":
        tags["ray_length_m"] = str(interp_metadata["ray_length_m"])
        tags["ray_decay"] = str(interp_metadata["ray_decay"])
        tags["ray_normalize_by_count"] = str(interp_metadata["normalize_by_count"])

    _write_geotiff(
        grid_values, width, height, bbox, output_crs, tif_path, tags=tags
    )

    # Step 9: Write metadata sidecar
    output_metadata = {
        "query_text": query_text,
        "query_mode": query_mode,
        "bbox": bbox,
        "crs": output_crs,
        "working_crs": working_crs,
        "resolution_m": resolution_m,
        "grid_width": width,
        "grid_height": height,
        "n_input_points": int(len(scores)),
        "interpolation": interp_metadata,
        "normalization": normalization,
        "score_stats": {
            "raw_min": float(scores.min()),
            "raw_max": float(scores.max()),
            "raw_mean": float(scores.mean()),
            "raw_std": float(scores.std()),
        },
        "output_tif": tif_path,
    }
    meta_path = os.path.join(output_path, f"{safe_query}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(output_metadata, f, indent=2)
    print(f"{TAG} Wrote metadata: {meta_path}", flush=True)

    print(f"{TAG} Done. Output: {output_path}", flush=True)
    return output_path
