"""Schema constants for the Cyclomedia catalog.

Centralizes face definitions, per-face bearing math, dataset→borough mapping,
and the canonical column list so every module agrees on the wire format.
"""

from __future__ import annotations

from typing import Optional

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

SCHEMA_VERSION = "v1"

ALL_FACES: tuple[str, ...] = ("F", "B", "L", "R", "U", "D")
HORIZONTAL_FACES: tuple[str, ...] = ("F", "B", "L", "R")

# Absolute compass bearing (deg, 0-360) of each cube face. Empirically
# verified (2026-04-22): Cyclomedia's NYC cube faces are rendered in a
# globally-oriented frame (F=N=0°, R=E=90°, B=S=180°, L=W=270°) — NOT
# rotated by vehicle heading. The `orientation` column (camera yaw) is ~0
# across 100% of NYC rows, confirming F points north. U/D have no
# horizontal direction → None.
FACE_BEARING_DEG: dict[str, Optional[float]] = {
    "F": 0.0,
    "R": 90.0,
    "B": 180.0,
    "L": 270.0,
    "U": None,
    "D": None,
}

BOROUGH_UNKNOWN = "unknown"

# Rough NYC bounding box for coord sanity checks.
NYC_BBOX = {"lat_min": 40.4, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.6}


def dataset_to_borough(dataset: str) -> str:
    """Map a dataset name to a borough.

    The pull pipeline names each dataset by the borough it targets, but uses a
    lat/lon bounding rectangle, so edge recordings may physically sit in a
    neighboring borough. The returned borough reflects the pull batch, not
    polygon-accurate geography. Callers that need strict geographic truth
    should reverse-geocode against CD polygons at query time.
    """
    d = dataset.lower()
    for b in ("manhattan", "brooklyn", "queens", "bronx"):
        if b in d:
            return b
    if "si_" in d or d.startswith("si_") or "staten" in d:
        return "staten_island"
    return BOROUGH_UNKNOWN


# Columns we pull from the WFS catalog CSV. The first entry is the join key
# (renamed from imageId → recording_id at load time).
WFS_JOIN_COLUMNS: tuple[str, ...] = (
    "recording_id",
    "recordedAt",
    "recorderDirection",
    "yawDegrees",
    "orientation",
    "orientationPrecision",
    "yawPrecisionDegrees",
    "statePlaneX",
    "statePlaneY",
    "locationSRS",
    "height",
    "heightSystem",
    "groundLevelOffset",
    "latitudePrecision",
    "longitudePrecision",
    "heightPrecision",
    "year",
    "panoramaTileSchema",
    "tileSchema",
    "hasDepthMap",
    "isAuthorized",
    "productType",
)


# Canonical catalog column order (written to parquet). Kept here so writer,
# reader, and validator agree. Type resolution happens downstream.
CATALOG_COLUMNS: tuple[str, ...] = (
    # identity / path
    "sample_id",
    "recording_id",
    "face",
    "image_path",
    "dataset",
    "group",
    "borough",
    # spatial
    "latitude",
    "longitude",
    "geom_wkb",
    "statePlaneX",
    "statePlaneY",
    "locationSRS",
    "latitudePrecision",
    "longitudePrecision",
    "height",
    "heightPrecision",
    "groundLevelOffset",
    "heightSystem",
    # temporal / orientation
    "recordedAt",
    "year",
    "recorderDirection",
    "yawDegrees",
    "yawPrecisionDegrees",
    "orientation",
    "orientationPrecision",
    "bearing",
    # product metadata
    "productType",
    "panoramaTileSchema",
    "tileSchema",
    "hasDepthMap",
    "isAuthorized",
    # manifest-derived
    "manifest_zoom",
    "manifest_tile_px",
    "manifest_tile_schema",
    "manifest_name_version",
    "manifest_mode",
    "manifest_checkpoint",
    "manifest_no_tiles",
    "face_elapsed_s",
    "face_used_render",
    "depthmap_present",
    "depthmap_used_render",
    "depthmap_render_size",
    "depthmap_rgb_render_size",
    "depthmap_downsample_factor",
    "depthmap_stitched",
    # index provenance
    "file_size",
    "file_mtime",
    "manifest_ok",
    "catalog_hit",
    "indexed_at",
)
