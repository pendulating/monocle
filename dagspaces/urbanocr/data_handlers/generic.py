"""Generic image directory data handler for OCR pipeline.

Handles any directory of images without specialized metadata extraction.
This is the fallback handler for data sources without dedicated handlers.
"""

import os
import re
import json
from datetime import datetime
from typing import Any, Dict

from omegaconf import DictConfig

from .base import OCRDataHandler

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    ray = None
    _RAY_AVAILABLE = False


class GenericImageHandler(OCRDataHandler):
    """Generic data handler for image directories.
    
    Works with any directory containing images. Extracts minimal metadata:
    - sample_id: Derived from filename (without extension)
    - image_path: Full path to the image
    """
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
    
    def load_dataset(self, cfg: DictConfig) -> Any:
        """Load images from a generic directory into a Ray Dataset.
        
        Expects cfg.data to contain:
        - image_path: Path to directory containing images
        - recursive: Whether to search subdirectories (default: True)
        - extensions: List of file extensions to include (default: all supported)
        
        Returns:
            Ray Dataset with columns: image, image_path, sample_id
        """
        if not _RAY_AVAILABLE:
            raise RuntimeError("Ray is required for image data loading")
        
        data_cfg = getattr(cfg, "data", None)
        if data_cfg is None:
            raise ValueError("Configuration missing 'data' section")
        
        image_path = getattr(data_cfg, "image_path", None)
        if not image_path:
            raise ValueError("data.image_path must be specified")
        
        image_path = os.path.abspath(os.path.expanduser(str(image_path)))
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image path does not exist: {image_path}")
        
        # Check if it's a single file or directory
        if os.path.isfile(image_path):
            # Single file mode
            print(f"[generic] Loading single image: {image_path}", flush=True)
            pattern = image_path
        else:
            # Directory mode
            recursive = getattr(data_cfg, "recursive", True)
            extensions = getattr(data_cfg, "extensions", None)
            
            if extensions:
                ext_list = list(extensions)
            else:
                ext_list = list(self.SUPPORTED_EXTENSIONS)
            
            # Build pattern
            if recursive:
                # Use ** for recursive glob
                pattern = os.path.join(image_path, "**", "*")
            else:
                pattern = os.path.join(image_path, "*")
            
            print(f"[generic] Loading images from: {image_path} (recursive={recursive})", flush=True)
        
        # Read images with paths
        ds = ray.data.read_images(pattern, include_paths=True)
        
        # Filter by extension if specified (ray.data.read_images may include non-images)
        if os.path.isdir(image_path):
            extensions_lower = {ext.lower() for ext in ext_list}
            
            def _filter_by_extension(row: Dict[str, Any]) -> bool:
                path = row.get("path", "")
                if not path:
                    return False
                ext = os.path.splitext(path)[1].lower()
                return ext in extensions_lower
            
            ds = ds.filter(_filter_by_extension)
        
        # Add metadata columns
        def _enrich_generic_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
            """Add basic metadata columns."""
            row_out = dict(row)
            
            path_val = row_out.get("path")
            if path_val:
                path_str = str(path_val)
                row_out["image_path"] = path_str
                
                # Extract metadata
                metadata = self.extract_metadata(path_str)
                row_out.update(metadata)
            else:
                row_out["image_path"] = None
                row_out["sample_id"] = None
            
            return row_out
        
        ds = ds.map(_enrich_generic_metadata)
        
        # Log dataset info
        try:
            count = ds.count()
            print(
                json.dumps({
                    "generic_handler": {
                        "event": "dataset_loaded",
                        "count": count,
                        "path": image_path,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                }),
                flush=True,
            )
        except Exception:
            print(f"[generic] Dataset loaded from {image_path}", flush=True)
        
        # Apply sample limit if configured
        sample_n = getattr(getattr(cfg, "runtime", None), "sample_n", None)
        if isinstance(sample_n, int) and sample_n > 0:
            ds = ds.limit(sample_n)
            print(f"[generic] Applied sample limit: {sample_n}", flush=True)
        
        return ds
    
    def extract_metadata(self, path: str) -> Dict[str, Any]:
        """Extract basic metadata from image path.
        
        Args:
            path: Full path to the image file
            
        Returns:
            Dictionary with sample_id derived from filename
        """
        metadata = {
            "sample_id": None,
        }
        
        if not path:
            return metadata
        
        try:
            # Get filename without extension
            basename = os.path.basename(path)
            stem = os.path.splitext(basename)[0]
            
            # Sanitize to create valid sample_id
            sample_id = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
            metadata["sample_id"] = sample_id
            
        except Exception:
            pass
        
        return metadata

