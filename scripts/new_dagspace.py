"""Make a new dagspace from the template in dagspaces/_template/.

Usage:
    python scripts/new_dagspace.py NAME                 # dagspaces/NAME and tests/test_NAME_dagspace.py
    python scripts/new_dagspace.py NAME --dest DIR      # DIR/NAME; DIR must be on sys.path or in the repo

Then run the example pipeline of the new dagspace:
    dagspace NAME -m hydra/launcher=local pipeline=describe_images

For a dagspace outside dagspaces/, give the module path:
    dagspace research.dagspaces.NAME -m hydra/launcher=local pipeline=describe_images
"""

from __future__ import annotations

import argparse
import keyword
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(REPO, "dagspaces", "_template")
TEMPLATE_TEST = os.path.join(REPO, "tests", "test_template_dagspace.py")
TEXT_EXT = (".py", ".yaml", ".yml", ".md", ".jinja", ".json", ".txt")


def replacements(name: str, module: str) -> Dict[str, str]:
    """Return the text changes from the template to the new dagspace, in order."""
    return {
        "dagspaces._template": module,
        "dagspaces/_template": module.replace(".", "/"),
        "dagspace _template": f"dagspace {module if module != f'dagspaces.{name}' else name}",
        'name="template"': f'name="{name}"',
        "TEMPLATE": name.upper(),
        "the template dagspace": f"the {name} dagspace",
        "The template dagspace": f"The {name} dagspace",
        "test_template_dagspace": f"test_{name}_dagspace",
    }


def render(text: str, changes: Dict[str, str]) -> str:
    for old, new in changes.items():
        text = text.replace(old, new)
    return text


def _module_for(dest_dir: str) -> str:
    rel = os.path.relpath(os.path.abspath(dest_dir), REPO)
    if rel.startswith(".."):
        return os.path.basename(os.path.normpath(dest_dir))
    return rel.replace(os.sep, ".")


def new_dagspace(name: str, dest: Optional[str] = None, *, with_test: bool = True) -> List[str]:
    """Copy the template to ``dest/name``. Return the paths of the new files."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name):
        raise ValueError(f"the name must be a lowercase Python identifier, not {name!r}")
    parent = os.path.abspath(dest) if dest else os.path.join(REPO, "dagspaces")
    target = os.path.join(parent, name)
    if os.path.exists(target):
        raise FileExistsError(f"{target} exists")
    module = _module_for(target)
    changes = replacements(name, module)

    written: List[str] = []
    shutil.copytree(TEMPLATE_DIR, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for root, _dirs, files in os.walk(target):
        for fname in files:
            path = os.path.join(root, fname)
            if fname.endswith(TEXT_EXT):
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(render(text, changes))
            written.append(path)

    if with_test and os.path.isfile(TEMPLATE_TEST):
        test_dir = os.path.join(REPO, "tests") if dest is None else target
        test_path = os.path.join(test_dir, f"test_{name}_dagspace.py")
        with open(TEMPLATE_TEST, encoding="utf-8") as fh:
            text = render(fh.read(), changes)
        with open(test_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(test_path)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("name", help="The dagspace name: a lowercase Python identifier")
    parser.add_argument("--dest", default=None, help="The parent folder (default: dagspaces/)")
    parser.add_argument("--no-test", action="store_true", help="Do not write a test file")
    args = parser.parse_args(argv)
    try:
        files = new_dagspace(args.name, args.dest, with_test=not args.no_test)
    except (ValueError, FileExistsError) as exc:
        print(f"new_dagspace: {exc}", file=sys.stderr)
        return 1
    for path in files:
        print(os.path.relpath(path, REPO) if not os.path.relpath(path, REPO).startswith("..") else path)
    print(f"\nNext: dagspace {args.name if args.dest is None else _module_for(os.path.join(args.dest, args.name))} "
          "-m hydra/launcher=local pipeline=describe_images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
