"""The catalog CLI with the NYC settings. The code is in ``dagspaces.sources.cyclomedia.cli``.

    python -m sv.cyclomedia_catalog.cli -v build --datasets manhattan_2025_1k
    python -m sv.cyclomedia_catalog.cli validate
    python -m sv.cyclomedia_catalog.cli sample -73.99,40.755,-73.98,40.762 -n 20 --out out/s

The defaults: the folders, the WFS globs and the fd of this cluster, the
borough of each pull, the NYC box and the time zone of New York.
"""

from __future__ import annotations

import sys
from typing import Optional

from dagspaces.sources.cyclomedia import cli as _cli

from . import nyc


def main(argv: Optional[list[str]] = None) -> int:
    return _cli.main(
        argv,
        prog="python -m sv.cyclomedia_catalog.cli",
        region_of=nyc.dataset_to_borough,
        bbox=nyc.NYC_BBOX,
        tz=nyc.TZ,
        catalog_globs=list(nyc.WFS_GLOBS),
        fd_path=nyc.FD_PATH,
        arg_defaults={"raw_root": nyc.RAW_ROOT, "output": nyc.CATALOG_ROOT, "catalog": nyc.CATALOG_ROOT},
    )


if __name__ == "__main__":
    sys.exit(main())
