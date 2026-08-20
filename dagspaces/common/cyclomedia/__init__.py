"""Interactive access to the materialised Cyclomedia NYC street-view database.

Backs ``notebooks/cyclomedia/browser.py``. Three layers:

* :mod:`.catalog` -- DuckDB over the 31.5M-row face catalog; spatial queries.
* :mod:`.cubemap` -- cube-face geometry; unfolded cross and equirectangular panorama.
* :mod:`.depth`   -- depth-map decoding (relative only; see the module docstring).
"""

from . import catalog, cubemap, depth

__all__ = ["catalog", "cubemap", "depth"]
