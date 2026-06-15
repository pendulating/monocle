"""Geometry attachment for permit curation — thin wrapper around shared utilities.

The 3-stage BIN → nearest → point matcher lives in
:mod:`dagspaces.common.curation.geom` (shared across curation sub-datasets).
This module re-exports it with permit-specific column-name defaults for
backward compatibility with existing call sites.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import polars as pl

from ..geom import (
    DEFAULT_BUILDINGS_PATH,
    DEFAULT_NEAREST_MAX_FT,
    NYC_SP_CRS,
    WGS84_CRS,
    attach_geometry as _shared_attach_geometry,
    load_buildings as _shared_load_buildings,
)

__all__ = [
    "attach_geometry",
    "DEFAULT_BUILDINGS_PATH",
    "DEFAULT_NEAREST_MAX_FT",
    "NYC_SP_CRS",
    "WGS84_CRS",
]

# Re-export under the permits namespace for tests that import privately.
_load_buildings = _shared_load_buildings


def attach_geometry(
    permits: pl.DataFrame,
    buffer_ft: float = 80.0,
    buildings_path: str = DEFAULT_BUILDINGS_PATH,
    buildings_gdf: Optional[gpd.GeoDataFrame] = None,
    nearest_max_ft: float = DEFAULT_NEAREST_MAX_FT,
) -> gpd.GeoDataFrame:
    """Permit-specific wrapper: uses ``permit_id``, ``bin``, ``raw_latitude``,
    ``raw_longitude`` column names. See :func:`curation.geom.attach_geometry`
    for the full contract."""
    return _shared_attach_geometry(
        permits,
        buffer_ft=buffer_ft,
        buildings_path=buildings_path,
        buildings_gdf=buildings_gdf,
        nearest_max_ft=nearest_max_ft,
        id_col="permit_id",
        bin_col="bin",
        lat_col="raw_latitude",
        lon_col="raw_longitude",
    )
