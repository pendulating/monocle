"""Scaffold/shed permit curation.

Pipeline (see :func:`.scaffolding_permits.build`):

    fetch BIS + DOB NOW → normalize → buffer polygons/points → validate → write

Entry points:
- :func:`.scaffolding_permits.build` — programmatic
- ``python -m dagspaces.common.curation scaffolding-permits ...`` — CLI
"""

from .scaffolding_permits import build, PermitBuildResult

__all__ = ["build", "PermitBuildResult"]
