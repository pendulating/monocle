"""Old path. The code is in ``dagspaces.sources.cyclomedia.catalog``; this class adds the NYC catalog folder.

    from sv.cyclomedia_catalog import CyclomediaCatalog

    cat = CyclomediaCatalog()          # /share/ju/cyclomedia/catalog/v1
    df = cat.query(faces={"F"}, datasets=["manhattan_2025_1k"])
"""

from __future__ import annotations

from typing import Optional

from dagspaces.sources.cyclomedia import catalog as _catalog

from .nyc import CATALOG_ROOT

__all__ = ["CyclomediaCatalog", "DEFAULT_CATALOG_ROOT"]

DEFAULT_CATALOG_ROOT = CATALOG_ROOT


class CyclomediaCatalog(_catalog.CyclomediaCatalog):
    """The framework catalog. The default folder is the NYC catalog of this cluster."""

    def __init__(self, root: Optional[str] = DEFAULT_CATALOG_ROOT) -> None:
        super().__init__(root or DEFAULT_CATALOG_ROOT)
