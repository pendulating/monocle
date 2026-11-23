"""Old path. The code is in ``dagspaces.sources.cyclomedia.indexer``; these wrappers add the NYC settings.

The NYC settings: the borough of each pull, the NYC box of the validation,
the time zone of New York, the fd of the cluster and the folders of
``.nyc``.
"""

from __future__ import annotations

from typing import Iterable, Optional

from dagspaces.sources.cyclomedia import indexer as _indexer
from dagspaces.sources.cyclomedia.indexer import BuildResult

from . import nyc

__all__ = ["build_catalog", "rejoin_wfs", "BuildResult"]


def build_catalog(
    raw_root: str,
    output_root: str,
    datasets: Optional[Iterable[str]] = None,
    catalog_globs: Iterable[str] = nyc.WFS_GLOBS,
    fd_path: Optional[str] = None,
    write_validation: bool = True,
) -> BuildResult:
    """Build the catalog with the NYC settings."""
    return _indexer.build_catalog(
        raw_root, output_root, datasets, catalog_globs, fd_path or nyc.FD_PATH, write_validation,
        region_of=nyc.dataset_to_borough, bbox=nyc.NYC_BBOX, tz=nyc.TZ,
    )


def rejoin_wfs(
    output_root: str,
    datasets: Iterable[str],
    catalog_globs: Iterable[str] = nyc.WFS_GLOBS,
    raw_root: str = nyc.RAW_ROOT,
    write_validation: bool = True,
) -> BuildResult:
    """Do the WFS join again with the NYC settings."""
    return _indexer.rejoin_wfs(
        output_root, datasets, catalog_globs, raw_root, write_validation,
        region_of=nyc.dataset_to_borough, bbox=nyc.NYC_BBOX, tz=nyc.TZ,
    )
