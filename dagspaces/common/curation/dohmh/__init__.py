"""DOHMH restaurant inspection curation.

Pulls NYC food-service establishments from the DOHMH Restaurant Inspection
Results Socrata dataset (``43nn-pn8j``), used here as a proxy for "every
restaurant in NYC". The raw dataset has one row per (inspection, violation);
this module dedupes to **one row per restaurant** (CAMIS) carrying the
most-recent inspection's metadata, joins each to a building polygon via
BIN (with nearest-building fallback), buffers by N feet, and writes a
curation sub-dataset with the same contract as
:mod:`dagspaces.common.curation.permits` and :mod:`.facdb`.

Entry points:
- :func:`.dohmh_restaurants.build` — programmatic
- ``python -m dagspaces.common.curation dohmh-restaurants ...`` — CLI
"""

from __future__ import annotations

from .aggregate import AggregateResult, aggregate_restaurants
from .cuisines import (
    UnknownCuisineError,
    load_cuisines,
    validate_cuisines,
)
from .dohmh_restaurants import DohmhBuildResult, build

__all__ = [
    "build",
    "DohmhBuildResult",
    "aggregate_restaurants",
    "AggregateResult",
    "UnknownCuisineError",
    "load_cuisines",
    "validate_cuisines",
]
