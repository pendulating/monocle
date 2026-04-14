"""Data handlers for urbanocr dagspace.

Data handlers are pluggable adapters that produce pandas DataFrames
with image paths for the OCR stage.
"""

from .base import OCRDataHandler
from .cyclomedia import CyclomediaHandler
from .generic import GenericImageHandler

__all__ = ["OCRDataHandler", "CyclomediaHandler", "GenericImageHandler"]

