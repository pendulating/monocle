---
title: "UrbanOCR — Text Spotting"
category: dagspace
created: 2026-04-06
tags:
  - dagspace
  - ocr
  - text-spotting
  - tiling
  - bounding-box
---

# UrbanOCR — Text Spotting

UrbanOCR is the dagspace for **OCR and text detection with bounding boxes** on street-view imagery. It detects text in images using Qwen3-VL's text spotting capability (`bbox_2d`), with automatic tiling for large images (e.g., 8192x8192 Cyclomedia panoramic faces) and a pluggable data handler system for different image sources.

## Purpose

- Detect and localize text in urban imagery with bounding box coordinates
- Handle very large images (8192x8192 Cyclomedia cyclorama faces) via transparent tiling
- Produce flat parquet output with one row per text detection (not one per image)
- Support multiple data sources via pluggable handler pattern

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanocr/cli.py` | Hydra CLI entry point |
| `dagspaces/urbanocr/orchestrator.py` | DAG execution engine; defines `OCRRunner(StageRunner)` and `get_stage_registry()` |
| `dagspaces/urbanocr/stages/ocr.py` | Core OCR stage: `run_ocr_stage()`, Ray Data + vLLM pipeline |
| `dagspaces/urbanocr/tiling.py` | Image tiling utilities: `TileInfo`, `tile_image`, `transform_bbox_to_original` |
| `dagspaces/urbanocr/data_handlers/base.py` | `OCRDataHandler` ABC with factory `get_handler()` |
| `dagspaces/urbanocr/data_handlers/cyclomedia.py` | `CyclomediaHandler` -- Cyclomedia-specific metadata extraction |
| `dagspaces/urbanocr/data_handlers/generic.py` | `GenericImageHandler` -- generic image directory loading |
| `dagspaces/urbanocr/prompts/ocr_preprocessing.py` | System/user prompts, `OCR_OUTPUT_SCHEMA`, pre/post-processing |

## Pluggable Data Handlers

UrbanOCR uses an abstract base class pattern for data loading. The handler is selected via `data.handler` in the Hydra config.

```
OCRDataHandler (ABC)
  -> load_dataset(cfg) -> Ray Dataset
  -> extract_metadata(path) -> Dict
  -> get_handler(name) -> OCRDataHandler  [class factory method]
```

| Handler | Config Name | Description |
|---------|-------------|-------------|
| `CyclomediaHandler` | `"cyclomedia"` | Loads Cyclomedia panoramic face images; extracts recording_id, face, location metadata from path structure |
| `GenericImageHandler` | `"generic"` | Loads any directory of images via `ray.data.read_images()` |

**Standardized output schema** from all handlers:

| Column | Type | Description |
|--------|------|-------------|
| `image` | numpy array | Raw image data from `ray.data.read_images()` |
| `image_path` | string | Full path to the image file |
| `sample_id` | string | Unique identifier for the image |
| + metadata | varies | Handler-specific metadata (e.g., `recording_id`, `face`, `location_id` for Cyclomedia) |

## Automatic Tiling

Cyclomedia cyclorama faces are 8192x8192 pixels, far exceeding what vision-language models can process in a single pass. The tiling system in `dagspaces/urbanocr/tiling.py` transparently splits large images into overlapping tiles and remaps detection coordinates back to the original image space.

### TileInfo Dataclass

```python
@dataclass
class TileInfo:
    tile_idx: int           # Sequential tile index
    row: int                # Row in tile grid (0-indexed)
    col: int                # Column in tile grid (0-indexed)
    x_offset: int           # Pixel offset from left edge of original image
    y_offset: int           # Pixel offset from top edge of original image
    tile_width: int         # Actual width of this tile (may be smaller at edges)
    tile_height: int        # Actual height of this tile
    original_width: int     # Width of original image
    original_height: int    # Height of original image
    n_rows: int             # Total rows in tile grid
    n_cols: int             # Total columns in tile grid
```

### Tiling Functions

| Function | Description |
|----------|-------------|
| `needs_tiling(image, max_dimension=2048)` | Returns True if either dimension exceeds `max_dimension` |
| `calculate_tile_grid(width, height, tile_size, overlap)` | Computes tile positions with stride = tile_size - overlap |
| `tile_image(image, tile_size=1024, overlap=64)` | Splits image into `(tile_array, TileInfo)` tuples |
| `extract_tile(image, tile_info)` | Crops a single tile from the original image array |
| `transform_bbox_to_original(bbox, tile_info)` | Remaps tile-local [x1,y1,x2,y2] (0-999) to global coordinates |
| `get_global_bbox_range(tile_info)` | Returns `(max_x, max_y)` for the global coordinate space |
| `get_tiling_config(cfg)` | Extracts tiling settings from Hydra config |

### Coordinate Transformation

The OCR model returns bounding boxes normalized to 0-999 within each tile. `transform_bbox_to_original` converts these to a high-resolution global coordinate space:

1. Convert normalized tile coords (0-999) to pixel coordinates within the tile
2. Add tile offset to get pixel position in the original image
3. Scale to global space: 0 to (N_tiles * 999) per dimension

For a 9x9 tile grid over an 8192x8192 image, global coordinates range from 0 to 8991 (9 * 999), preserving full detection resolution.

See [[concept-tiling]] for the full algorithm and edge cases.

## Data Flow

```
Data Handler (Cyclomedia/Generic)
  -> Ray Dataset (image arrays + metadata)
  -> needs_tiling() check per image
     -> If large: tile_image() -> multiple tile rows
     -> If small: pass through as single row
  -> ocr_preprocess() -- build chat messages with system/user prompts
  -> Ray Data LLM (vLLMEngineProcessorConfig, sampling_params_ocr)
  -> ocr_postprocess() -- parse_ocr_response() + JSON extraction
  -> transform_bbox_to_original() for tiled images
  -> flatten_detections() -- one row per text detection
  -> Output Parquet
```

## Prompts

Prompts are defined in `dagspaces/urbanocr/prompts/ocr_preprocessing.py`:

| Function | Description |
|----------|-------------|
| `get_system_prompt()` | System prompt for text spotting mode |
| `get_user_prompt()` | User prompt requesting text detection with bounding boxes |
| `OCR_OUTPUT_SCHEMA` | JSON schema for structured OCR output |
| `get_sampling_params()` | Default sampling params (`max_tokens=4096`) |
| `ocr_preprocess(row)` | Build chat messages from a data row |
| `ocr_postprocess(row)` | Parse model response into structured detections |
| `parse_ocr_response(text)` | Extract text + bbox from raw model output |
| `flatten_detections(df)` | Explode one-per-image rows into one-per-detection |

## Output Format

The output parquet has **one row per text detection**, not one per image. Key columns:

| Column | Description |
|--------|-------------|
| `sample_id` | Source image identifier |
| `image_path` | Path to the source image |
| `text` | Detected text string |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | Bounding box coordinates (global space for tiled images) |
| `confidence` | Detection confidence score |
| `text_type` | Classification of text type |
| `location_id` | Location identifier (Cyclomedia handler) |
| `face` | Panoramic face identifier (Cyclomedia handler) |

## Configuration

### Tiling Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tiling.enabled` | `true` | Enable automatic tiling |
| `tiling.tile_size` | `1024` | Tile dimensions in pixels (square) |
| `tiling.overlap` | `64` | Overlap between adjacent tiles in pixels |
| `tiling.max_dimension` | `2048` | Tile images larger than this threshold |

### Data Configs (`dagspaces/urbanocr/conf/data/`)

| Config | Description |
|--------|-------------|
| `cyclomedia.yaml` | Base Cyclomedia panoramic data |
| `cyclomedia_manhattan.yaml` | Manhattan Cyclomedia dataset |
| `cyclomedia_manhattan_small.yaml` | Small Manhattan subset |
| `cyclomedia_manhattan_tiny.yaml` | Tiny test subset |
| `cyclomedia_test_w0etz.yaml` | Specific recording test |
| `generic_images.yaml` | Generic image directory |

### Pipeline Configs (`dagspaces/urbanocr/conf/pipeline/`)

| Config | Description |
|--------|-------------|
| `ocr_manhattan.yaml` | Full Manhattan OCR run |
| `ocr_manhattan_small.yaml` | Small Manhattan OCR |
| `ocr_manhattan_tiny.yaml` | Tiny test run |
| `ocr_batch.yaml` | Batch OCR pipeline |
| `ocr_test_tiny.yaml` | Minimal test pipeline |
| `ocr_test_w0etz.yaml` | Single recording test |

### Prompt Config

| Config | Description |
|--------|-------------|
| `conf/prompt/text_spotting.yaml` | Text spotting prompt configuration |

### SLURM Launcher

UrbanOCR has its own launcher override at `dagspaces/urbanocr/conf/hydra/launcher/` for OCR-specific GPU and memory requirements.

## Related Pages

- [[architecture]] -- overall pipeline architecture
- [[concept-tiling]] -- detailed tiling algorithm and coordinate math
- [[shared-infrastructure]] -- common modules (vLLM inference, Ray Data, W&B)
- [[urban-vqa]] -- core VQA dagspace (OCR reuses the vLLM engine)
- [[urban-pair-vqa]] -- pairwise comparison dagspace
- [[urban-roam-vqa]] -- street traversal dagspace
- [[urban-embed]] -- embedding dagspace
