"""Add the project config roots to the Hydra search path of each dagspace.

The framework (`dagspaces/`) does not know about the project code. This plugin
lets a command such as `python -m dagspaces.vqa.cli -m pipeline=vqa_bayflood`
find the configs in `site/conf/`, `sv/conf/`, `sv/nyc/conf/` and `research/conf/`.

Hydra finds this plugin through the `hydra_plugins` namespace package. The repo
root must be on `sys.path`: run the commands from the repo root.

A project config root has this layout:

| Folder | Search path of |
|--------|----------------|
| `<root>/<dagspace>/` | the dagspace `<dagspace>` only (for example `vqa/pipeline/`) |
| `<root>/common/` | all dagspaces (shared groups, for example `data/`) |
"""

import re
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

REPO = Path(__file__).resolve().parents[2]

#: The project config roots, in search order.
PROJECT_ROOTS = (
    # The cluster settings of this site (the launchers). A launcher here takes
    # the place of the generic framework launcher with the same name.
    REPO / "site" / "conf",
    REPO / "sv" / "conf",
    REPO / "sv" / "nyc" / "conf",
    REPO / "research" / "conf",
)

#: The main config path of a dagspace: `.../dagspaces/<name>/conf` or `pkg://...dagspaces.<name>.conf`.
_DAGSPACE = re.compile(r"dagspaces[/.](?P<name>[A-Za-z0-9_]+)[/.]conf/?$")


def dagspace_of(search_path: ConfigSearchPath):
    """Return the dagspace name of the main config path, or None."""
    for element in search_path.get_path():
        if element.provider != "main" or not element.path:
            continue
        m = _DAGSPACE.search(element.path)
        if m and m.group("name") != "common":
            return m.group("name")
    return None


class MllmsciSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        name = dagspace_of(search_path)
        if name is None:
            return
        # A dagspace folder comes before the common folders. Thus a dagspace-local
        # config wins over a shared config with the same name, as in dagspaces/.
        for sub in (name, "common"):
            for root in PROJECT_ROOTS:
                folder = root / sub
                if folder.is_dir():
                    search_path.append(provider="mllmsci-searchpath", path=f"file://{folder}")
