"""Old path. The code is in ``dagspaces.sources.cyclomedia.validation``; this wrapper adds the NYC box."""

from __future__ import annotations

from typing import Optional

import polars as pl

from dagspaces.sources.cyclomedia import validation as _validation
from dagspaces.sources.cyclomedia.schema import BBox
from dagspaces.sources.cyclomedia.validation import ValidationError

from .nyc import NYC_BBOX

__all__ = ["run_validation", "ValidationError"]


def run_validation(df: pl.DataFrame, output_root: str, raw_root: str,
                   bbox: Optional[BBox] = NYC_BBOX) -> str:
    """Run the checks. The check 6 uses the NYC box."""
    return _validation.run_validation(df, output_root, raw_root, bbox=bbox)
