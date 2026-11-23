"""Access to the Cyclomedia NYC street-view data, for the browser notebook.

The code is in image-forward (``dagspaces.sources.cyclomedia``). This
package adds the NYC folders for ``research/notebooks/cyclomedia/browser.py``
and the research scripts:

* :mod:`.catalog` -- DuckDB over the face catalog, with the NYC folders.

Import the cube geometry and the depth decode from the framework:
``dagspaces.sources.cyclomedia.cubemap`` and ``dagspaces.sources.cyclomedia.depth``.
"""

from . import catalog

__all__ = ["catalog"]
