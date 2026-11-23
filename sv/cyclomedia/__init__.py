"""Access to the Cyclomedia NYC street-view data, for the browser notebook.

The code is in image-forward (``dagspaces.sources.cyclomedia``). These
modules keep the old import paths for ``research/notebooks/cyclomedia/browser.py``
and the research scripts:

* :mod:`.catalog` -- DuckDB over the face catalog, with the NYC folders.
* :mod:`.cubemap` -- the cube geometry (a copy of the names of the framework module).
* :mod:`.depth`   -- the depth-map decode (the same).
"""

from . import catalog, cubemap, depth

__all__ = ["catalog", "cubemap", "depth"]
