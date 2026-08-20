"""DuckDB access layer for the materialised Cyclomedia catalog.

The catalog (``/share/ju/cyclomedia/catalog/v1``) is a hive-partitioned parquet
dataset -- 31.5M rows, one per cube face, over 5.26M recordings across all five
boroughs. DuckDB scans the whole thing in a few seconds, so it backs the
browser directly; nothing needs to be loaded into pandas up front.

Two levels of granularity:

* **face level** -- the catalog as stored: one row per (recording, face), with
  ``image_path``, ``bearing``, ``file_size``, ``depthmap_present``, ...
* **recording level** -- one row per physical recording, which is what a map
  wants. Built once by :func:`build_recording_index` and cached to parquet;
  5.26M rows, loads in about a second.

**Never walk the image tree.** Directory listing on this NFS mount is glacial
(a ``find`` over one dataset times out at two minutes). Every path the browser
touches is constructed from the catalog instead -- see :func:`recording_dir`.
"""

from __future__ import annotations

import os
from typing import Optional

import duckdb
import pandas as pd

__all__ = [
    "CATALOG_ROOT",
    "RAW_ROOT",
    "FACE_GLOB",
    "DEFAULT_INDEX_PATH",
    "connect",
    "recording_dir",
    "build_recording_index",
    "load_recording_index",
    "nearest_recordings",
    "recordings_in_bbox",
    "overview_grid",
    "faces_for_recording",
]

CATALOG_ROOT = "/share/ju/cyclomedia/catalog/v1"
RAW_ROOT = "/share/ju/cyclomedia/raw"
FACE_GLOB = os.path.join(CATALOG_ROOT, "by_dataset", "**", "*.parquet")

# Cached recording-level index. Lives under the project's data dir, not $HOME
# (which is a tight NFS quota -- see server.env).
DEFAULT_INDEX_PATH = "/share/pierson/matt/mllmsci/data/cyclomedia/browser/recordings_v1.parquet"

# Metres per degree at NYC's latitude. Good to ~0.1% over the city, which is
# far tighter than the ~1 m spacing between recordings.
_M_PER_DEG_LAT = 110_540.0
_M_PER_DEG_LON = 84_300.0  # 111_320 * cos(40.7 deg)


def connect(threads: Optional[int] = None) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection tuned for this catalog."""
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    con.execute("SET enable_progress_bar = false")
    return con


def recording_dir(dataset: str, group: str, recording_id: str) -> str:
    """Path to a recording's directory, built from catalog fields (no globbing).

    Layout: ``{RAW_ROOT}/{dataset}/{group}/{recording_id}/`` containing
    ``faces/`` and ``depthmaps_faces/``.
    """
    return os.path.join(RAW_ROOT, dataset, group, recording_id)


def build_recording_index(
    con: duckdb.DuckDBPyConnection,
    out_path: str = DEFAULT_INDEX_PATH,
    overwrite: bool = False,
) -> str:
    """Collapse the face-level catalog to one row per recording and cache it.

    ~5.26M rows. Takes tens of seconds; the result is cached to parquet and
    reused thereafter.

    A recording_id can appear under more than one dataset (the catalog's own
    validation flags 128k such (recording_id, face) pairs, from overlapping
    borough pulls). We keep one row per physical recording, choosing the
    alphabetically-first dataset so the pick is deterministic.
    """
    if os.path.exists(out_path) and not overwrite:
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    con.execute(
        f"""
        COPY (
            WITH per_rec AS (
                SELECT
                    recording_id,
                    dataset,
                    any_value("group")          AS "group",
                    any_value(borough)          AS borough,
                    avg(latitude)               AS latitude,
                    avg(longitude)              AS longitude,
                    any_value(recordedAt)       AS recordedAt,
                    any_value(height)           AS height,
                    any_value(groundLevelOffset) AS groundLevelOffset,
                    any_value(recorderDirection) AS recorderDirection,
                    count(*)                    AS n_faces,
                    sum(depthmap_present::INT)  AS n_depth_faces,
                    sum(file_size)              AS bytes_total
                FROM read_parquet('{FACE_GLOB}', hive_partitioning = 1)
                GROUP BY recording_id, dataset
            ),
            ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY recording_id ORDER BY dataset
                ) AS rn
                FROM per_rec
            )
            SELECT
                recording_id, dataset, "group", borough,
                latitude, longitude, recordedAt,
                height, groundLevelOffset, recorderDirection,
                n_faces, n_depth_faces, bytes_total,
                (n_depth_faces > 0) AS has_depth,
                year(recordedAt)    AS year,
                month(recordedAt)   AS month
            FROM ranked
            WHERE rn = 1
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return out_path


def load_recording_index(
    con: duckdb.DuckDBPyConnection,
    index_path: str = DEFAULT_INDEX_PATH,
    table: str = "recordings",
) -> int:
    """Register the cached recording index as an in-memory DuckDB table.

    Materialising it (rather than querying the parquet each time) keeps the
    map's click-to-query round trip well under a second.
    """
    con.execute(
        f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{index_path}')"
    )
    return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _dist_expr(lat: float, lon: float) -> str:
    """SQL for the equirectangular distance in metres from (lat, lon)."""
    return (
        f"sqrt( pow((longitude - {lon}) * {_M_PER_DEG_LON}, 2)"
        f"    + pow((latitude  - {lat}) * {_M_PER_DEG_LAT}, 2) )"
    )


def nearest_recordings(
    con: duckdb.DuckDBPyConnection,
    lat: float,
    lon: float,
    k: int = 25,
    radius_m: float = 200.0,
    table: str = "recordings",
    where: Optional[str] = None,
) -> pd.DataFrame:
    """The ``k`` recordings nearest to (lat, lon), within ``radius_m``.

    This is what a click on the map resolves to. A bounding-box prefilter (on
    raw lat/lon, so it stays a cheap range scan) narrows the 5.26M rows before
    distances are computed.
    """
    dlat = radius_m / _M_PER_DEG_LAT
    dlon = radius_m / _M_PER_DEG_LON
    extra = f"AND ({where})" if where else ""
    return con.execute(
        f"""
        SELECT * FROM (
            SELECT *, {_dist_expr(lat, lon)} AS dist_m
            FROM {table}
            WHERE latitude  BETWEEN {lat - dlat} AND {lat + dlat}
              AND longitude BETWEEN {lon - dlon} AND {lon + dlon}
              {extra}
        )
        WHERE dist_m <= {radius_m}
        ORDER BY dist_m
        LIMIT {int(k)}
        """
    ).fetchdf()


def recordings_in_bbox(
    con: duckdb.DuckDBPyConnection,
    lat0: float,
    lat1: float,
    lon0: float,
    lon1: float,
    limit: int = 5000,
    table: str = "recordings",
    where: Optional[str] = None,
) -> pd.DataFrame:
    """Recordings inside a lat/lon box, capped at ``limit``.

    The cap is a rendering guard: a dense box can hold hundreds of thousands of
    recordings and no map wants them all. When the cap bites, rows are sampled
    uniformly (``USING SAMPLE``) rather than truncated, so the result stays
    spatially representative instead of collapsing into whichever corner sorted
    first. :func:`count_in_bbox` reports the true total.
    """
    extra = f"AND ({where})" if where else ""
    total = count_in_bbox(con, lat0, lat1, lon0, lon1, table=table, where=where)
    # The sample must wrap the *filtered* set: DuckDB applies USING SAMPLE right
    # after FROM, before WHERE, so sampling in the same SELECT would draw from
    # all 5.2M rows and leave only the handful that happen to fall in the box.
    sample = f"USING SAMPLE {int(limit)} ROWS" if total > limit else ""
    df = con.execute(
        f"""
        SELECT * FROM (
            SELECT * FROM {table}
            WHERE latitude BETWEEN {min(lat0, lat1)} AND {max(lat0, lat1)}
              AND longitude BETWEEN {min(lon0, lon1)} AND {max(lon0, lon1)}
              {extra}
        ) {sample}
        """
    ).fetchdf()
    df.attrs["total_in_bbox"] = total
    df.attrs["sampled"] = total > limit
    return df


def count_in_bbox(
    con: duckdb.DuckDBPyConnection,
    lat0: float,
    lat1: float,
    lon0: float,
    lon1: float,
    table: str = "recordings",
    where: Optional[str] = None,
) -> int:
    """True number of recordings in a box, ignoring any render cap."""
    extra = f"AND ({where})" if where else ""
    return int(
        con.execute(
            f"""
            SELECT count(*) FROM {table}
            WHERE latitude BETWEEN {min(lat0, lat1)} AND {max(lat0, lat1)}
              AND longitude BETWEEN {min(lon0, lon1)} AND {max(lon0, lon1)}
              {extra}
            """
        ).fetchone()[0]
    )


def overview_grid(
    con: duckdb.DuckDBPyConnection,
    cell_m: float = 250.0,
    table: str = "recordings",
    where: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate recordings into a lat/lon grid for the citywide view.

    5.26M points will not render; a 250 m grid collapses the city to ~15k cells,
    which any map handles happily. Returns one row per non-empty cell with its
    centre and count.
    """
    dlat = cell_m / _M_PER_DEG_LAT
    dlon = cell_m / _M_PER_DEG_LON
    extra = f"WHERE {where}" if where else ""
    return con.execute(
        f"""
        SELECT
            (floor(latitude  / {dlat}) + 0.5) * {dlat} AS latitude,
            (floor(longitude / {dlon}) + 0.5) * {dlon} AS longitude,
            count(*)                 AS n,
            any_value(borough)       AS borough
        FROM {table}
        {extra}
        GROUP BY 1, 2
        ORDER BY n DESC
        """
    ).fetchdf()


def faces_for_recording(
    con: duckdb.DuckDBPyConnection,
    recording_id: str,
    dataset: Optional[str] = None,
) -> pd.DataFrame:
    """Face-level catalog rows for one recording (paths, bearings, sizes).

    Reads the face-level parquet directly. The hive partition on ``dataset``
    means passing it turns a full scan into a partition scan.
    """
    filters = [f"recording_id = '{recording_id}'"]
    if dataset:
        filters.append(f"dataset = '{dataset}'")
    return con.execute(
        f"""
        SELECT sample_id, recording_id, dataset, face, bearing, image_path,
               depthmap_present, file_size, latitude, longitude, recordedAt,
               height, groundLevelOffset, recorderDirection, yawDegrees
        FROM read_parquet('{FACE_GLOB}', hive_partitioning = 1)
        WHERE {' AND '.join(filters)}
        ORDER BY face
        """
    ).fetchdf()
