"""The `dagspace` console script: one entry point for each dagspace CLI.

Usage::

    dagspace vqa -m pipeline=vqa_cyclomedia_scaffolding runtime.debug=true
    dagspace research.dagspaces.urbanroamvqa -m pipeline=...

``dagspace <name> ARGS`` is the same as ``python -m dagspaces.<name>.cli ARGS``.
A name with a dot is a module path: the script adds ``.cli`` when the path does
not end with it. All the arguments after the name go to Hydra.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
from typing import List, Optional


def _framework_dagspaces() -> List[str]:
    """Return the names of the dagspaces in ``dagspaces/`` that have a ``cli`` module."""
    import dagspaces

    names = []
    for info in pkgutil.iter_modules(dagspaces.__path__):
        if not info.ispkg or info.name.startswith("_") or info.name == "common":
            continue
        try:
            if importlib.util.find_spec(f"dagspaces.{info.name}.cli") is not None:
                names.append(info.name)
        except ImportError:
            continue
    return sorted(names)


def _module_for(name: str) -> str:
    """Return the CLI module path for a dagspace name or a module path."""
    if "." not in name:
        return f"dagspaces.{name}.cli"
    return name if name.endswith(".cli") else f"{name}.cli"


def _usage() -> str:
    names = ", ".join(_framework_dagspaces())
    return (
        "usage: dagspace <name> [hydra arguments]\n"
        f"  <name>: {names}, or a module path such as research.dagspaces.urbanroamvqa\n"
        "  example: dagspace vqa -m pipeline=vqa_cyclomedia_scaffolding runtime.debug=true"
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Run the CLI of the dagspace that the first argument names."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        sys.exit(0 if args else 2)

    module_name = _module_for(args[0])
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and module_name.startswith(exc.name):
            print(f"dagspace: no CLI module {module_name}\n\n{_usage()}", file=sys.stderr)
            sys.exit(2)
        raise
    entry = getattr(module, "main", None)
    if entry is None:
        print(f"dagspace: {module_name} has no main()", file=sys.stderr)
        sys.exit(2)

    # Hydra parses sys.argv. Give it the arguments after the dagspace name.
    sys.argv = [module.__file__ or module_name] + args[1:]
    entry()


if __name__ == "__main__":
    main()
