---
title: "Image Tiling"
category: concept
created: 2026-04-06
updated: 2026-04-06
tags:
  - concept
  - ocr
  - tiling
  - images
---

# Image Tiling

How UrbanOCR handles large images that exceed vision-language model input limits.

## Problem

Cyclomedia cyclorama faces are 8192x8192 pixels. Vision-language models have context limits that cannot accept images of this size directly. Passing such large images would either fail or cause severe quality degradation through aggressive downsampling.

## Solution

The framework applies automatic tiling with configurable overlap and coordinate remapping. Large images are split into a grid of smaller tiles, each tile is processed independently through the OCR model, and tile-local bounding box coordinates are transformed back to the original full-resolution image coordinate space.

## Key File

`dagspaces/urbanocr/tiling.py`

## Core Data Structure

### TileInfo

A dataclass carrying metadata for a single tile:

| Field | Type | Description |
|-------|------|-------------|
| `tile_idx` | `int` | Sequential tile index (0-based) |
| `row` | `int` | Row in the tile grid (0-indexed) |
| `col` | `int` | Column in the tile grid (0-indexed) |
| `x_offset` | `int` | Pixel offset from the left edge of the original image |
| `y_offset` | `int` | Pixel offset from the top edge of the original image |
| `tile_width` | `int` | Actual width of this tile (may be smaller at edges) |
| `tile_height` | `int` | Actual height of this tile (may be smaller at edges) |
| `original_width` | `int` | Width of the original image |
| `original_height` | `int` | Height of the original image |
| `n_rows` | `int` | Total rows in the tile grid |
| `n_cols` | `int` | Total columns in the tile grid |

`TileInfo` supports serialization via `to_dict()` and `from_dict()` for parquet storage.

## Functions

### `calculate_tile_grid(image_width, image_height, tile_size=1024, overlap=64)`

Computes the tile grid layout for a given image size. Returns a list of `TileInfo` objects.

The stride between tiles is `tile_size - overlap`. The function calculates:
- `n_cols = ceil((width - overlap) / stride)`
- `n_rows = ceil((height - overlap) / stride)`

Edge tiles are clamped to image bounds, so the last tile in each dimension may overlap more than the configured overlap value.

For an 8192x8192 image with tile_size=1024 and overlap=64, this produces a 9x9 grid of 81 tiles.

### `tile_image(image, tile_size=1024, overlap=64)`

Splits a numpy array (H, W, C) or (H, W) into tiles. Returns a list of `(tile_array, TileInfo)` tuples.

### `extract_tile(image, tile_info)`

Extracts a single tile from a numpy array using the offsets in a `TileInfo`.

### `needs_tiling(image, max_dimension=2048)`

Checks whether an image exceeds `max_dimension` in either width or height. Returns `True` if tiling is needed.

### `transform_bbox_to_original(bbox, tile_info, tile_normalize=999)`

The critical coordinate remapping function. OCR models return bounding boxes normalized to 0-999 within each tile. This function transforms those tile-local coordinates to a high-resolution global coordinate space.

The transformation pipeline:

1. **Tile-local normalized (0-999)** -- model output coordinates within the tile
2. **Tile pixel coordinates** -- multiply by `tile_width / 999` (or `tile_height / 999`)
3. **Original image pixel coordinates** -- add `x_offset` and `y_offset` from `TileInfo`
4. **Global normalized coordinates** -- scale to `0` through `(n_tiles * 999)`

For a 9x9 tile grid, the global coordinate range is 0-8991 (9 * 999), preserving the full detection resolution rather than compressing back to 0-999.

### `get_global_bbox_range(tile_info, tile_normalize=999)`

Returns the maximum `(max_x, max_y)` for the global coordinate space, computed as `(n_cols * 999, n_rows * 999)`.

### `get_tiling_config(cfg)`

Extracts tiling configuration from the Hydra config object, with defaults:

```python
{
    "enabled": True,
    "tile_size": 1024,
    "overlap": 64,
    "max_dimension": 2048,
}
```

Looks for tiling settings in `cfg.tiling` or `cfg.data.tiling`.

## Configuration

Tiling is controlled through the Hydra config:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tiling.enabled` | `True` | Whether to apply tiling |
| `tiling.tile_size` | `1024` | Width and height of each tile in pixels |
| `tiling.overlap` | `64` | Overlap between adjacent tiles in pixels |
| `tiling.max_dimension` | `2048` | Threshold above which tiling is triggered |

The overlap ensures text at tile boundaries is captured by at least one tile. The default 64px overlap is sufficient for most text sizes encountered in street-level imagery.

## Coordinate Space Example

For an 8192x8192 image with tile_size=1024, overlap=64:

```
Tile grid: 9 rows x 9 cols = 81 tiles
Stride: 1024 - 64 = 960 pixels
Global coordinate range: 0-8991 (x), 0-8991 (y)

Tile (0,0): x_offset=0,    y_offset=0    -> global x: 0-999
Tile (0,1): x_offset=960,  y_offset=0    -> global x: ~999-1998
Tile (4,4): x_offset=3840, y_offset=3840 -> global x: ~4495-5494
```

## See Also

- [[urban-ocr]] -- the UrbanOCR dagspace that uses tiling
