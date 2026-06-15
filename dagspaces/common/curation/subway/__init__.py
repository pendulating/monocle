"""NYC subway station entrance/exit curation.

Pulls the NY State Open Data dataset ``i9wp-a4ja`` (MTA Subway — Permanent
Station Entrances/Exits, ~2,120 rows / 485 stations / 13 entrance types)
and produces a curation sub-dataset of buffered **points** at each
entrance's lat/lon.

Unlike :mod:`..facdb` and :mod:`..permits`, subway entrances are mostly
sidewalk stairs and other street furniture, not building features — so
this build skips the BIN-match + nearest-building stages of
:mod:`..geom.attach_geometry` and buffers the entrance point directly.

Entry points:
- :func:`.subway_entrances.build` — programmatic
- ``python -m dagspaces.common.curation subway-entrances ...`` — CLI
"""

from __future__ import annotations

from .entrance_types import (
    UnknownEntranceTypeError,
    load_entrance_types,
    validate_entrance_types,
)
from .subway_entrances import SubwayBuildResult, build

__all__ = [
    "build",
    "SubwayBuildResult",
    "UnknownEntranceTypeError",
    "load_entrance_types",
    "validate_entrance_types",
]
