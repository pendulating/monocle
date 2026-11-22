"""Data handlers for ocr dagspace.

Data handlers are pluggable adapters that produce pandas DataFrames
with image paths for the OCR stage.
"""

from .base import OCRDataHandler
from .generic import GenericImageHandler

__all__ = ["OCRDataHandler", "GenericImageHandler"]

