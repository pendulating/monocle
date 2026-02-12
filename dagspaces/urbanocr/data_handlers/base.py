"""Base class for OCR data handlers.

Data handlers are pluggable adapters that produce Ray Datasets
compatible with the OCR stage.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from omegaconf import DictConfig

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    ray = None
    _RAY_AVAILABLE = False


class OCRDataHandler(ABC):
    """Abstract base class for OCR data handlers.
    
    Data handlers are responsible for:
    1. Loading images from a data source into a Ray Dataset
    2. Extracting source-specific metadata from image paths
    3. Providing a standardized schema for the OCR stage
    
    Standardized Output Schema:
    - image: numpy array (from ray.data.read_images())
    - image_path: string - full path to the image
    - sample_id: string - unique identifier for the image
    - Additional metadata columns specific to the handler
    """
    
    @abstractmethod
    def load_dataset(self, cfg: DictConfig) -> Any:
        """Load images into a Ray Dataset.
        
        Args:
            cfg: Hydra configuration object with data source settings
            
        Returns:
            Ray Dataset with standardized schema (image, image_path, sample_id, metadata)
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

