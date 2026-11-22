"""Image tiling utilities for processing large images.

Cyclomedia cyclorama faces are 8192x8192 pixels, which exceeds what
vision-language models can process. This module provides tiling to
split large images into smaller tiles for OCR, then transforms
coordinates back to the original image space.

Key features:
- Configurable tile size (default 1024x1024)
- Configurable overlap (default 64px) to catch text at boundaries
- Coordinate transformation from tile-local to full-image space
- Tile metadata tracking (row, col, offsets)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


@dataclass
class TileInfo:
    """Metadata for a single tile."""
    tile_idx: int           # Sequential tile index
    row: int                # Row in tile grid (0-indexed)
    col: int                # Column in tile grid (0-indexed)
    x_offset: int           # Pixel offset from left edge of original image
    y_offset: int           # Pixel offset from top edge of original image
    tile_width: int         # Actual width of this tile (may be smaller at edges)
    tile_height: int        # Actual height of this tile
    original_width: int     # Width of original image
    original_height: int    # Height of original image
    n_rows: int = 1         # Total rows in tile grid
    n_cols: int = 1         # Total columns in tile grid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tile_idx": self.tile_idx,
            "row": self.row,
            "col": self.col,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "tile_width": self.tile_width,
            "tile_height": self.tile_height,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TileInfo":
        """Create from dictionary."""
        # Handle backwards compatibility
        d = dict(d)
        d.setdefault("n_rows", 1)
        d.setdefault("n_cols", 1)
        return cls(**d)


def calculate_tile_grid(
    image_width: int,
    image_height: int,
    tile_size: int = 1024,
    overlap: int = 64,
) -> List[TileInfo]:
    """Calculate tile grid for an image.
    
    Args:
        image_width: Width of the original image
        image_height: Height of the original image
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles in pixels
        
    Returns:
        List of TileInfo objects describing each tile
    """
    stride = tile_size - overlap
    
    # Calculate number of tiles needed in each dimension
    n_cols = max(1, math.ceil((image_width - overlap) / stride))
    n_rows = max(1, math.ceil((image_height - overlap) / stride))
    
    tiles = []
    tile_idx = 0
    
    for row in range(n_rows):
        for col in range(n_cols):
            x_offset = col * stride
            y_offset = row * stride
            
            # Clamp to image bounds
            x_offset = min(x_offset, max(0, image_width - tile_size))
            y_offset = min(y_offset, max(0, image_height - tile_size))
            
            # Calculate actual tile dimensions (may be smaller at edges)
            tile_width = min(tile_size, image_width - x_offset)
            tile_height = min(tile_size, image_height - y_offset)
            
            tiles.append(TileInfo(
                tile_idx=tile_idx,
                row=row,
                col=col,
                x_offset=x_offset,
                y_offset=y_offset,
                tile_width=tile_width,
                tile_height=tile_height,
                original_width=image_width,
                original_height=image_height,
                n_rows=n_rows,
                n_cols=n_cols,
            ))
            tile_idx += 1
    
    return tiles


def extract_tile(
    image: "np.ndarray",
    tile_info: TileInfo,
) -> "np.ndarray":
    """Extract a tile from an image.
    
    Args:
        image: Original image as numpy array (H, W, C) or (H, W)
        tile_info: Tile metadata
        
    Returns:
        Tile as numpy array
    """
    if np is None:
        raise RuntimeError("NumPy is required for tiling")
    
    x = tile_info.x_offset
    y = tile_info.y_offset
    w = tile_info.tile_width
    h = tile_info.tile_height
    
    if image.ndim == 3:
        return image[y:y+h, x:x+w, :].copy()
    else:
        return image[y:y+h, x:x+w].copy()


def tile_image(
    image: "np.ndarray",
    tile_size: int = 1024,
    overlap: int = 64,
) -> List[Tuple["np.ndarray", TileInfo]]:
    """Tile an image into smaller pieces.
    
    Args:
        image: Original image as numpy array (H, W, C) or (H, W)
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Returns:
        List of (tile_array, tile_info) tuples
    """
    if np is None:
        raise RuntimeError("NumPy is required for tiling")
    
    if image.ndim == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    tile_infos = calculate_tile_grid(width, height, tile_size, overlap)
    
    tiles = []
    for tile_info in tile_infos:
        tile_array = extract_tile(image, tile_info)
        tiles.append((tile_array, tile_info))
    
    return tiles


def transform_bbox_to_original(
    bbox: List[int],
    tile_info: TileInfo,
    tile_normalize: int = 999,
) -> List[int]:
    """Transform bounding box from tile-local to full-resolution global coordinates.
    
    The OCR model returns bboxes normalized to 0-999 within the tile.
    To preserve resolution, this function converts to 0-(N*999) in the global space,
    where N is the number of tiles in that dimension.
    
    For a 9x9 tile grid:
      - Tile-local: 0-999 (model output)
      - Global: 0-8991 (9 * 999)
    
    This preserves the full detection resolution instead of compressing back to 0-999.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2] normalized to 0-999 in tile space
        tile_info: Tile metadata including grid dimensions
        tile_normalize: Normalization range for tile-local coords (default 999)
        
    Returns:
        Bounding box [x1, y1, x2, y2] in global space 0-(N*999)
    """
    if len(bbox) != 4:
        return bbox
    
    x1, y1, x2, y2 = bbox
    
    # Global coordinate space: 0 to (N * tile_normalize)
    # Each tile occupies tile_normalize units in global space
    # Position = tile_index * tile_normalize + local_coord
    
    # For x: col * 999 + local_x  (but account for overlap)
    # For y: row * 999 + local_y
    
    # The key insight: we want each tile's 0-999 to map to a distinct
    # range in global space, preserving resolution.
    # Tile (0,0): 0-999, Tile (0,1): 999-1998, etc.
    # But with overlap, coordinates should smoothly transition.
    
    # Simpler approach: scale by tile position in the grid
    # Global = (tile_row/col * (N-1) + local / 999) * global_max / (N)
    
    # Actually, the cleanest is: global = col * 999 + local_x (for non-overlapping)
    # For overlapping tiles, we need to account for the offset proportion
    
    # Most intuitive: 
    # - Convert local (0-999) to pixel position within tile
    # - Add tile offset to get pixel in original image
    # - Scale to global normalized space where max = n_tiles * 999
    
    # Convert from normalized (0-999) to tile pixel coordinates
    tile_x1 = (x1 / tile_normalize) * tile_info.tile_width
    tile_y1 = (y1 / tile_normalize) * tile_info.tile_height
    tile_x2 = (x2 / tile_normalize) * tile_info.tile_width
    tile_y2 = (y2 / tile_normalize) * tile_info.tile_height
    
    # Add tile offset to get original image pixel coordinates
    orig_x1 = tile_x1 + tile_info.x_offset
    orig_y1 = tile_y1 + tile_info.y_offset
    orig_x2 = tile_x2 + tile_info.x_offset
    orig_y2 = tile_y2 + tile_info.y_offset
    
    # Scale to high-resolution global space: 0 to (N * 999)
    # where N is the number of tiles in that dimension
    global_max_x = tile_info.n_cols * tile_normalize
    global_max_y = tile_info.n_rows * tile_normalize
    
    norm_x1 = int((orig_x1 / tile_info.original_width) * global_max_x)
    norm_y1 = int((orig_y1 / tile_info.original_height) * global_max_y)
    norm_x2 = int((orig_x2 / tile_info.original_width) * global_max_x)
    norm_y2 = int((orig_y2 / tile_info.original_height) * global_max_y)
    
    # Clamp to valid range
    norm_x1 = max(0, min(global_max_x, norm_x1))
    norm_y1 = max(0, min(global_max_y, norm_y1))
    norm_x2 = max(0, min(global_max_x, norm_x2))
    norm_y2 = max(0, min(global_max_y, norm_y2))
    
    return [norm_x1, norm_y1, norm_x2, norm_y2]


def get_global_bbox_range(tile_info: TileInfo, tile_normalize: int = 999) -> Tuple[int, int]:
    """Get the maximum coordinate range for global bounding boxes.
    
    Args:
        tile_info: Tile metadata
        tile_normalize: Normalization range for tile-local coords
        
    Returns:
        Tuple of (max_x, max_y) for global coordinate space
    """
    return (
        tile_info.n_cols * tile_normalize,
        tile_info.n_rows * tile_normalize,
    )


def needs_tiling(
    image: "np.ndarray",
    max_dimension: int = 2048,
) -> bool:
    """Check if an image needs tiling.
    
    Args:
        image: Image as numpy array
        max_dimension: Maximum dimension before tiling is needed
        
    Returns:
        True if image exceeds max_dimension in either direction
    """
    if np is None:
        return False
    
    if image.ndim == 3:
        height, width = image.shape[:2]
    else:
        height, width = image.shape
    
    return width > max_dimension or height > max_dimension


def get_tiling_config(cfg) -> Dict[str, Any]:
    """Extract tiling configuration from config.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        Dictionary with tiling settings
    """
    defaults = {
        "enabled": True,
        "tile_size": 1024,
        "overlap": 64,
        "max_dimension": 2048,  # Tile images larger than this
    }
    
    try:
        tiling_cfg = getattr(cfg, "tiling", None)
        if tiling_cfg is None:
            # Check in data config
            tiling_cfg = getattr(getattr(cfg, "data", None), "tiling", None)
        
        if tiling_cfg is not None:
            defaults["enabled"] = bool(getattr(tiling_cfg, "enabled", True))
            defaults["tile_size"] = int(getattr(tiling_cfg, "tile_size", 1024))
            defaults["overlap"] = int(getattr(tiling_cfg, "overlap", 64))
            defaults["max_dimension"] = int(getattr(tiling_cfg, "max_dimension", 2048))
    except Exception:
        pass
    
    return defaults

