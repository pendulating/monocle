"""Write a parquet table of the images in a folder. The dagspaces read this table.

Usage:
    python scripts/make_image_table.py FOLDER OUT.parquet [--prompt TEXT] [--relative]

Columns:
    sample_id   The path below FOLDER, without the extension ("/" becomes "__").
    image_path  The absolute path, or with --relative the path below FOLDER.
                Set data.image_root to FOLDER to read a relative table.
    prompt      With --prompt: the same text in each row.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def image_table(
    folder: str, *, prompt: Optional[str] = None, relative: bool = False, recursive: bool = True
) -> pd.DataFrame:
    """Return one row for each image file in ``folder``, sorted by path."""
    folder = os.path.abspath(folder)
    paths: List[str] = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for name in sorted(files):
            if name.lower().endswith(EXTENSIONS):
                paths.append(os.path.join(root, name))
        if not recursive:
            break
    rows = []
    for path in paths:
        rel = os.path.relpath(path, folder)
        row = {
            "sample_id": os.path.splitext(rel)[0].replace(os.sep, "__"),
            "image_path": rel if relative else path,
        }
        if prompt is not None:
            row["prompt"] = prompt
        rows.append(row)
    columns = ["sample_id", "image_path"] + (["prompt"] if prompt is not None else [])
    return pd.DataFrame(rows).reindex(columns=pd.Index(columns))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("out")
    parser.add_argument("--prompt", default=None, help="The same prompt text in each row")
    parser.add_argument("--relative", action="store_true", help="Write paths below FOLDER")
    parser.add_argument("--no-recursive", action="store_true", help="Do not read the subfolders")
    args = parser.parse_args(argv)
    df = image_table(
        args.folder, prompt=args.prompt, relative=args.relative, recursive=not args.no_recursive
    )
    if df.empty:
        print(f"make_image_table: no images in {args.folder}", file=sys.stderr)
        return 1
    df.to_parquet(args.out, index=False)
    print(f"make_image_table: {len(df)} images -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
