"""Base class for OCR data handlers.

Data handlers are pluggable adapters that produce pandas DataFrames
compatible with the OCR stage.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig


class OCRDataHandler(ABC):
    """Abstract base class for OCR data handlers.

    Data handlers are responsible for:
    1. Discovering images from a data source
    2. Extracting source-specific metadata from image paths
    3. Returning a DataFrame with image_path and sample_id columns

    Standardized Output Schema:
    - image_path: string - full path to the image
    - sample_id: string - unique identifier for the image
    - Additional metadata columns specific to the handler
    """

    @abstractmethod
    def load_dataset(self, cfg: DictConfig) -> pd.DataFrame:
        """Discover images and return a DataFrame with paths and metadata.

        Args:
            cfg: Hydra configuration object with data source settings

        Returns:
            pd.DataFrame with columns: image_path, sample_id, and handler-specific metadata
        """
        pass

    @abstractmethod
    def extract_metadata(self, path: str) -> Dict[str, Any]:
        """Extract source-specific metadata from an image path.

        Args:
            path: Full path to the image file

        Returns:
            Dictionary of metadata extracted from the path
        """
        pass

    @classmethod
    def get_handler(cls, handler_name: str) -> "OCRDataHandler":
        """Factory method to get a handler by name.

        Args:
            handler_name: Name of the handler (e.g., "cyclomedia", "generic")

        Returns:
            Instance of the requested handler
        """
        handlers = {
            "cyclomedia": "CyclomediaHandler",
            "generic": "GenericImageHandler",
        }

        if handler_name not in handlers:
            raise ValueError(f"Unknown handler: {handler_name}. Available: {list(handlers.keys())}")

        # Import handler class dynamically
        if handler_name == "cyclomedia":
            from .cyclomedia import CyclomediaHandler
            return CyclomediaHandler()
        elif handler_name == "generic":
            from .generic import GenericImageHandler
            return GenericImageHandler()
        else:
            raise ValueError(f"Handler not implemented: {handler_name}")
