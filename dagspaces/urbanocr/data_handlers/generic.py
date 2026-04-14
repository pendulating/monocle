"""Generic image directory data handler for OCR pipeline.

Handles any directory of images without specialized metadata extraction.
This is the fallback handler for data sources without dedicated handlers.
"""

import os
import re
import json
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig

from .base import OCRDataHandler


class GenericImageHandler(OCRDataHandler):
    """Generic data handler for image directories.

    Works with any directory containing images. Extracts minimal metadata:
    - sample_id: Derived from filename (without extension)
    - image_path: Full path to the image
    """

    # Supported image extensions
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}

    def load_dataset(self, cfg: DictConfig) -> pd.DataFrame:
        """Load image paths from a generic directory into a DataFrame.

        Expects cfg.data to contain:
        - image_path: Path to directory containing images
        - recursive: Whether to search subdirectories (default: True)
        - extensions: List of file extensions to include (default: all supported)

        Returns:
            pd.DataFrame with columns: image_path, sample_id
        """
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
            print(f"[generic] Loading single image: {image_path}", flush=True)
            metadata = self.extract_metadata(image_path)
            rows = [{"image_path": image_path, **metadata}]
        else:
            recursive = getattr(data_cfg, "recursive", True)
            extensions = getattr(data_cfg, "extensions", None)

            if extensions:
                ext_set = {ext.lower() for ext in extensions}
            else:
                ext_set = self.SUPPORTED_EXTENSIONS

            print(f"[generic] Loading images from: {image_path} (recursive={recursive})", flush=True)

            rows = []
            if recursive:
                for dirpath, _, filenames in os.walk(image_path):
                    for fname in filenames:
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in ext_set:
                            full_path = os.path.join(dirpath, fname)
                            metadata = self.extract_metadata(full_path)
                            rows.append({"image_path": full_path, **metadata})
            else:
                for fname in os.listdir(image_path):
                    full_path = os.path.join(image_path, fname)
                    if os.path.isfile(full_path):
                        ext = os.path.splitext(fname)[1].lower()
                        if ext in ext_set:
                            metadata = self.extract_metadata(full_path)
                            rows.append({"image_path": full_path, **metadata})

        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["image_path", "sample_id"])

        print(
            json.dumps({
                "generic_handler": {
                    "event": "dataset_loaded",
                    "count": len(df),
                    "path": image_path,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            }),
            flush=True,
        )

        # Apply sample limit if configured
        sample_n = getattr(getattr(cfg, "runtime", None), "sample_n", None)
        if isinstance(sample_n, int) and sample_n > 0:
            df = df.head(sample_n)
            print(f"[generic] Applied sample limit: {sample_n}", flush=True)

        return df

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
