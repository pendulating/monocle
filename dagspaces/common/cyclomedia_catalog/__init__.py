"""Cyclomedia catalog — spatial/temporal index over `/share/ju/cyclomedia/raw/`.

One canonical parquet catalog, one Polars + polars-st query API, one CLI for
building / validating / querying. Replaces the per-run NFS walk in
`scripts/create_cyclomedia_dataset.py`.

See `docs/plans/cyclomedia-catalog.md` for the full design.
"""

from __future__ import annotations

from .catalog import CyclomediaCatalog, DEFAULT_CATALOG_ROOT
from .indexer import BuildResult, build_catalog
from .schema import (
    ALL_FACES,
    CATALOG_COLUMNS,
    FACE_BEARING_DEG,
    HORIZONTAL_FACES,
    SCHEMA_VERSION,
    dataset_to_borough,
)
from .validation import ValidationError, run_validation

__all__ = [
    "CyclomediaCatalog",
    "DEFAULT_CATALOG_ROOT",
    "BuildResult",
    "build_catalog",
    "ValidationError",
    "run_validation",
    "ALL_FACES",
    "HORIZONTAL_FACES",
    "FACE_BEARING_DEG",
    "CATALOG_COLUMNS",
    "SCHEMA_VERSION",
    "dataset_to_borough",
]
