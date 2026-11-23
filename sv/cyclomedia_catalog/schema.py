"""Old path. The schema is in ``dagspaces.sources.cyclomedia.schema``; the NYC values are in ``.nyc``."""

from __future__ import annotations

from dagspaces.sources.cyclomedia.schema import (
    ALL_FACES,
    BOROUGH_UNKNOWN,
    CATALOG_COLUMNS,
    FACE_BEARING_DEG,
    HORIZONTAL_FACES,
    SCHEMA_VERSION,
    WFS_JOIN_COLUMNS,
)

from .nyc import NYC_BBOX, dataset_to_borough

__all__ = [
    "ALL_FACES",
    "HORIZONTAL_FACES",
    "FACE_BEARING_DEG",
    "BOROUGH_UNKNOWN",
    "dataset_to_borough",
    "NYC_BBOX",
    "SCHEMA_VERSION",
    "CATALOG_COLUMNS",
    "WFS_JOIN_COLUMNS",
]
