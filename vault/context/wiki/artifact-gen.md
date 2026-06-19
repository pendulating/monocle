---
title: Artifact Gen
category: dagspace
created: 2026-04-08
updated: 2026-04-08
tags: [dagspace, raster, geotiff, interpolation, embedding-search]
---

# Artifact Gen

Dagspace for generating geospatial data products from embedding similarity scores. Converts the browser-based heatmap concept ([[guide-browser-search]]) into exportable GeoTIFF rasters.

## Stages

| Stage | Runner | Purpose | GPU |
|-------|--------|---------|-----|
| `raster` | `RasterRunner` | Text query → IDW-interpolated GeoTIFF | No (CPU default) |

## Raster Stage

**Input**: Pre-computed embeddings with lat/lon (from [[urban-embed]]) + text query string
**Output**: Single-band float32 GeoTIFF where pixel value = query relevance in [0, 1]

### Pipeline

```
Embeddings parquet (urbanembed)
  + PCA artifacts (build_browser_index)
  + W_proj.bin (train_query_projection)
  + text query
       ↓
[1] Load & bbox-filter embeddings
[2] PCA-reduce image embeddings
[3] Encode query (BGE + W_proj, or Qwen direct)
[4] Cosine similarity: scores = emb_normed @ query_normed
[5] Project coords to UTM, build regular grid
[6] IDW interpolation via cKDTree
[7] Min-max normalize to [0, 1]
[8] Write GeoTIFF with rasterio
```

### Query Encoding Modes

| Mode | Config value | Requires | Speed |
|------|-------------|----------|-------|
| BGE + projection | `bge_projection` (default) | CPU only | Fast (~1s) |
| Qwen direct | `qwen_direct` | GPU + vLLM | Slower (~30s) |

BGE mode uses the same pipeline as the browser search: encode with bge-small-en-v1.5 (384-dim), project via `W_proj.bin` to PCA-reduced Qwen space (256-dim).

### Interpolation Methods

Two interpolation modes are available, selectable via `raster.interpolation`:

#### IDW (Inverse Distance Weighting)

Isotropic point-source interpolation. Each image is treated as a point radiating its score in all directions. Good as a baseline.

| Parameter | Config key | Default |
|-----------|-----------|---------|
| IDW power | `raster.idw_power` | 2.0 |
| Max neighbors | `raster.max_neighbors` | 50 |
| Max distance | `raster.max_distance_m` | 100m |

Pixels beyond `max_distance_m` from any data point are set to NoData (NaN).

#### Ray Accumulation (Flow Vectors)

Directional evidence accumulation. Each face image casts a ray of length K from the camera position in the face's absolute bearing direction. Cells along the ray accumulate score, weighted by linear distance decay. Where rays from multiple images converge, scores amplify — encoding confidence that the query target is physically at that location.

**How it works:**

```
For each image (face-level):
  1. Absolute bearing = FACE_BEARING[face]
     where FACE_BEARING = {F: 0° (N), R: 90° (E), B: 180° (S), L: 270° (W)}
     NOTE: Cyclomedia's NYC cube faces are rendered in a globally-oriented
     frame — do NOT add yawDegrees / recorderDirection. See [[cyclomedia-catalog]].
  2. Cast a ray of length K meters from camera (lat/lon) in bearing direction
  3. Rasterize ray onto grid via Bresenham's line algorithm
  4. For each cell the ray passes through:
       cell_accumulator += score × (1 - d/K)    # linear decay
       cell_ray_count += 1
```

> **Known bug:** `artifact_gen/stages/raster.py::_compute_face_bearings` currently adds `yawDegrees` (or `recorderDirection`) to the face offset. That treats the cube as vehicle-relative — wrong. Scheduled fix matches the catalog-side correction in [[cyclomedia-catalog]]. Until then, ray accumulation deposits evidence in the wrong compass direction for any recording where the vehicle wasn't driving due north.

After all images are processed:
- **Raw mode** (`normalize_by_count: false`): pixel = accumulated score sum. Areas with more converging rays get higher values. Good for detection (total evidence).
- **Normalized mode** (`normalize_by_count: true`): pixel = accumulated sum / ray count. Controls for unequal image density across neighborhoods. Good for comparison.

**Why directional?** A face image with high scaffolding score deposits signal *in front of the camera* where the scaffolding actually is, not behind it. When two cameras at different positions both see scaffolding, their rays converge at its physical location — triangulation emerges naturally.

**Required columns:** `yawDegrees` (or `recorderDirection`) and `face` must be present in the embeddings parquet. These come from Cyclomedia catalog enrichment (`--catalog_csv` flag in `create_cyclomedia_dataset.py`).

| Parameter | Config key | Default |
|-----------|-----------|---------|
| Ray length | `raster.ray_length_m` | 30.0 |
| Distance decay | `raster.ray_decay` | linear |
| Normalize by count | `raster.ray_normalize_by_count` | false |

**Design notes:**
- A single center-line ray (not a cone) is used per face. With ~5m spacing between recordings and 4 faces per panorama, adjacent rays tile the space naturally.
- Linear decay `weight = score × (1 - d/K)` means full score at the camera, zero at the ray tip. This prevents over-depositing at range where angular imprecision (up to ±45° at face edges) would mislocate targets.
- Default K=30m covers a typical NYC street width (18–30m), reaching across to building facades where scaffolding/objects are located.

### Output Artifacts

| File | Format | Content |
|------|--------|---------|
| `<query>.tif` | GeoTIFF (float32, deflate) | Relevance raster with CRS + tags |
| `<query>_metadata.json` | JSON | Query, bbox, resolution, score stats, interpolation params |

### Configuration

Key config group: `raster` in `dagspaces/artifact_gen/conf/config.yaml`.

```yaml
raster:
  query_text: "construction scaffolding"
  query_mode: bge_projection
  embeddings_input_path: /path/to/embed/output
  pca_artifacts_path: /path/to/browser_index/output
  projection_matrix_path: /path/to/query_projection/output
  bbox: [-74.02, 40.70, -73.93, 40.82]
  resolution_m: 10.0
  crs: "EPSG:4326"
  working_crs: "EPSG:32618"
```

### Usage

```bash
python -m dagspaces.artifact_gen.cli -m \
  pipeline=raster_from_embeddings \
  raster.query_text="construction scaffolding" \
  raster.embeddings_input_path=/path/to/embed/output \
  raster.pca_artifacts_path=/path/to/browser_index/output \
  raster.projection_matrix_path=/path/to/query_projection/output
```

## Upstream Dependencies

Consumes artifacts from three [[urban-embed]] stages:

| Source | Stage | Artifacts |
|--------|-------|-----------|
| Embeddings | `embed` | `part-*.parquet` with `embedding`, `latitude`, `longitude` |
| PCA artifacts | `build_browser_index` | `pca_components.bin`, `pca_mean.bin` |
| Projection | `train_query_projection` | `W_proj.bin` |

## File Map

```
dagspaces/artifact_gen/
├── cli.py                          # Hydra entry point
├── orchestrator.py                 # RasterRunner + stage registry
├── stages/raster.py                # Core raster generation logic
└── conf/
    ├── config.yaml                 # Root config with raster params
    └── pipeline/
        └── raster_from_embeddings.yaml  # Pipeline consuming external sources
```

## Memory & Performance

- 1M embeddings × 3072-dim = ~12 GB; bbox-filtered typically 100K–300K (~1–4 GB)
- After PCA to 256-dim: ~100 MB for 100K points
- Manhattan at 10m: ~900×1300 grid = 5 MB
- IDW with cKDTree: seconds for typical grid sizes
- Total: fits comfortably in 64 GB `slurm_cpu_beefy` allocation
