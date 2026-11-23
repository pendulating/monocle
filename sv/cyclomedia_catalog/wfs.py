"""Old path. The code is in ``dagspaces.sources.cyclomedia.wfs``; this wrapper adds the NYC globs and time zone."""

from __future__ import annotations

from typing import Iterable

import polars as pl

from dagspaces.sources.cyclomedia import wfs as _wfs

from .nyc import TZ, WFS_GLOBS

__all__ = ["load_wfs_catalog", "DEFAULT_CATALOG_GLOB"]

DEFAULT_CATALOG_GLOB = WFS_GLOBS


def load_wfs_catalog(patterns: Iterable[str] = DEFAULT_CATALOG_GLOB, tz: str = TZ) -> pl.DataFrame:
    """Read the WFS CSVs. recordedAt is in the time zone of New York."""
    return _wfs.load_wfs_catalog(patterns, tz=tz)
