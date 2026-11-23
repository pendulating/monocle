"""The NYC and site settings of the Cyclomedia catalog.

The catalog code is in image-forward (``dagspaces.sources.cyclomedia``). This
module holds only the values of this project: the folders on the cluster,
the borough of each pull, the NYC box and the time zone.
"""

from __future__ import annotations

__all__ = [
    "RAW_ROOT",
    "CATALOG_ROOT",
    "WFS_GLOBS",
    "FD_PATH",
    "TZ",
    "NYC_BBOX",
    "dataset_to_borough",
]

RAW_ROOT = "/share/ju/cyclomedia/raw"
CATALOG_ROOT = "/share/ju/cyclomedia/catalog/v1"

# The WFS CSVs of the NYC pulls in /share/ju/cyclomedia/pull/:
#   - a folder of chunks for each borough (manhattan, manhattan_latter, queens)
#   - flat part files (bronx parts 1 to 4)
#   - out_catalog/recordings_{borough}_2025.csv (brooklyn and staten island are
#     only here; the other boroughs are copies, and the loader removes them)
WFS_GLOBS: tuple[str, ...] = (
    "/share/ju/cyclomedia/pull/recordings_*_chunks/*.csv",
    "/share/ju/cyclomedia/pull/recordings_*_part*.csv",
    "/share/ju/cyclomedia/pull/out_catalog/recordings_*.csv",
)

# The fd binary of this cluster (not on the PATH of the venv).
FD_PATH = "/share/ju/matt/.cargo/bin/fd"

# The catalog keeps recordedAt in the local time of New York.
TZ = "America/New_York"

# An approximate NYC box for the validation check 6.
NYC_BBOX = {"lat_min": 40.4, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.6}


def dataset_to_borough(dataset: str) -> str:
    """The borough of a pull, from its dataset name ("unknown" when no borough name is in it).

    The pull tool names each dataset by its target borough, but it pulls a
    lat/lon box. Thus a recording at the edge can be in the next borough. The
    value tells the pull, not the true borough. For the true borough, test the
    point against the borough polygons.
    """
    d = dataset.lower()
    for b in ("manhattan", "brooklyn", "queens", "bronx"):
        if b in d:
            return b
    if "si_" in d or "staten" in d:
        return "staten_island"
    return "unknown"
