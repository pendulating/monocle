"""NYC Open Restaurants / Dining Out NYC outdoor-dining curation.

Pulls the DCWP licensing dataset ``fpeh-f7ci`` (restaurants licensed for
``Sidewalk`` or ``Roadway`` outdoor dining), joins each license to a building
polygon via BIN (with nearest-building + point fallback), buffers by N feet,
and writes a curation sub-dataset with the same contract as
:mod:`dagspaces.common.curation.facdb`.

The dataset has no native primary key, so a stable ``uid`` is synthesized per
license in :mod:`.normalize`.

Entry points:
- :func:`.open_restaurants.build` — programmatic
- ``python -m dagspaces.common.curation open-restaurants ...`` — CLI
"""

from __future__ import annotations

from .license_types import (
    LICENSE_TYPES,
    UnknownLicenseTypeError,
    validate_license_types,
)
from .open_restaurants import OpenRestaurantsBuildResult, build

__all__ = [
    "build",
    "OpenRestaurantsBuildResult",
    "LICENSE_TYPES",
    "UnknownLicenseTypeError",
    "validate_license_types",
]
