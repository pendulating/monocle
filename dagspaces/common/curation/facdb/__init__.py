"""FacDB (City Planning Facilities Database) curation.

Pulls NYC facility POIs from Socrata ``ji82-xba5`` filtered at any level of
the 4-tier hierarchy (``facdomain`` > ``facgroup`` > ``facsubgrp`` > ``factype``),
joins each to a building polygon via BIN (with nearest-building fallback),
buffers by N feet, and writes a curation sub-dataset with the same
contract as :mod:`dagspaces.common.curation.permits`.

Entry points:
- :func:`.facdb_facilities.build` — programmatic
- ``python -m dagspaces.common.curation facdb-facilities ...`` — CLI
"""

from .categorization import (
    HIERARCHY_LEVELS,
    UnknownCategoryError,
    load_categorization,
    validate_filter_values,
)
from .facdb_facilities import FacdbBuildResult, build

__all__ = [
    "build",
    "FacdbBuildResult",
    "HIERARCHY_LEVELS",
    "UnknownCategoryError",
    "load_categorization",
    "validate_filter_values",
]
