"""The Cyclomedia catalog of NYC: the settings of this project over the framework code.

The catalog code (walk, manifests, WFS join, validation, query, CLI) is in
image-forward: ``dagspaces.sources.cyclomedia`` (see its
``docs/sources-cyclomedia.md``). This package adds the NYC settings (``.nyc``):
the folders on the cluster, the borough of each pull, the NYC box and the
time zone of New York. The submodules keep the old import paths.
"""

from __future__ import annotations

from .catalog import DEFAULT_CATALOG_ROOT, CyclomediaCatalog
from .indexer import BuildResult, build_catalog, rejoin_wfs
from .nyc import NYC_BBOX, dataset_to_borough
from .schema import (
    ALL_FACES,
    CATALOG_COLUMNS,
    FACE_BEARING_DEG,
    HORIZONTAL_FACES,
    SCHEMA_VERSION,
)
from .validation import ValidationError, run_validation

__all__ = [
    "CyclomediaCatalog",
    "DEFAULT_CATALOG_ROOT",
    "BuildResult",
    "build_catalog",
    "rejoin_wfs",
    "ValidationError",
    "run_validation",
    "ALL_FACES",
    "HORIZONTAL_FACES",
    "FACE_BEARING_DEG",
    "CATALOG_COLUMNS",
    "SCHEMA_VERSION",
    "NYC_BBOX",
    "dataset_to_borough",
]
