"""DuckDB access to the NYC Cyclomedia catalog: the folders of this cluster over the framework code.

The code is in ``dagspaces.sources.cyclomedia.recordings``. This module adds
the NYC folders as defaults:

* ``CATALOG_ROOT`` -- the face catalog (``/share/ju/cyclomedia/catalog/v1``).
* ``RAW_ROOT`` -- the pull tree.
* ``DEFAULT_INDEX_PATH`` -- the recording index of the browser (in ``data/`` of
  the monocle checkout, not in ``$HOME``, which has a small NFS quota).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from dagspaces.sources.cyclomedia import recordings as _rec
from dagspaces.sources.cyclomedia.recordings import (
    connect,
    count_in_bbox,
    nearest_recordings,
    overview_grid,
    recordings_in_bbox,
)

from sv.cyclomedia_catalog.nyc import CATALOG_ROOT, RAW_ROOT

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
    "count_in_bbox",
    "overview_grid",
    "faces_for_recording",
]

FACE_GLOB = os.path.join(CATALOG_ROOT, "by_dataset", "**", "*.parquet")
# The monocle checkout that holds this file (sv/cyclomedia/catalog.py), thus the
# path does not depend on the current folder.
_REPO = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = str(_REPO / "data" / "cyclomedia" / "browser" / "recordings_v1.parquet")


def recording_dir(dataset: str, group: str, recording_id: str) -> str:
    """The folder of a recording under ``RAW_ROOT``, from the catalog fields (no glob)."""
    return _rec.recording_dir(dataset, group, recording_id, raw_root=RAW_ROOT)


def build_recording_index(con: duckdb.DuckDBPyConnection, out_path: str = DEFAULT_INDEX_PATH,
                          overwrite: bool = False) -> str:
    """Make the recording index of the NYC catalog (once) and give its path."""
    return _rec.build_recording_index(con, out_path, overwrite=overwrite, catalog_root=CATALOG_ROOT)


def load_recording_index(con: duckdb.DuckDBPyConnection, index_path: str = DEFAULT_INDEX_PATH,
                         table: str = "recordings") -> int:
    """Load the recording index into a DuckDB table. Give the row count."""
    return _rec.load_recording_index(con, index_path, table=table)


def faces_for_recording(con: duckdb.DuckDBPyConnection, recording_id: str,
                        dataset: Optional[str] = None) -> pd.DataFrame:
    """The face rows of 1 recording of the NYC catalog."""
    return _rec.faces_for_recording(con, recording_id, dataset=dataset, catalog_root=CATALOG_ROOT)
