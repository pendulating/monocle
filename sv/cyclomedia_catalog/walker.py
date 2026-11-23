"""Old path. The code is in ``dagspaces.sources.cyclomedia.walker``; this wrapper adds the fd of the cluster."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import polars as pl

from dagspaces.sources.cyclomedia import walker as _walker
from dagspaces.sources.cyclomedia.walker import WalkResult

from .nyc import FD_PATH

__all__ = ["walk_dataset", "walk_datasets", "FD_DEFAULT_PATH", "WalkResult"]

FD_DEFAULT_PATH = FD_PATH


def _fd(fd_path: Optional[str]) -> Optional[str]:
    if fd_path and os.path.isfile(fd_path) and os.access(fd_path, os.X_OK):
        return fd_path
    return FD_DEFAULT_PATH


def walk_dataset(raw_root: str, dataset: str, fd_path: Optional[str] = None) -> WalkResult:
    """Walk 1 dataset. Without a good ``fd_path``, use the fd of the cluster."""
    return _walker.walk_dataset(raw_root, dataset, fd_path=_fd(fd_path))


def walk_datasets(raw_root: str, datasets: Iterable[str], fd_path: Optional[str] = None) -> pl.DataFrame:
    """Walk some datasets and join the rows."""
    return _walker.walk_datasets(raw_root, datasets, fd_path=_fd(fd_path))
