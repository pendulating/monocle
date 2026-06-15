#!/usr/bin/env python3
"""Export rerank results as a folder of symlinks to retrieved images.

Each symlink is named ``{rank:04d}_{score:.4f}_{original_stem}{ext}`` so
that a simple ``ls`` shows results in ranked order.

Usage:
    python scripts/export_rerank_symlinks.py /path/to/rerank.parquet
    python scripts/export_rerank_symlinks.py /path/to/rerank.parquet -o /tmp/results -n 50
    python scripts/export_rerank_symlinks.py /path/to/rerank.parquet --remap /scratch/mwf62/cyclomedia:/share/ju/cyclomedia/raw
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Known remapping: flat face files → nested recording/faces/ layout.
# /scratch/mwf62/cyclomedia/manhattan_2025_1k/W0EHO/W0EHO3ME_F.jpg
#   → /share/ju/cyclomedia/raw/manhattan_2025_1k/W0EHO/W0EHO3ME/faces/F.jpg
_FACE_PATTERN = re.compile(r"^(.+)_([BDFLLRU])\.jpg$")


def _remap_path(image_path: Path, remap_from: str, remap_to: str) -> Path:
    """Remap an image path, handling the flat→nested face layout."""
    s = str(image_path)
    if not s.startswith(remap_from):
        return image_path

    remapped = s.replace(remap_from, remap_to, 1)

    # Try direct replacement first
    direct = Path(remapped)
    if direct.exists():
        return direct

    # Try nested recording/faces/FACE.jpg layout
    m = _FACE_PATTERN.match(direct.name)
    if m:
        recording_id, face = m.group(1), m.group(2)
        nested = direct.parent / recording_id / "faces" / f"{face}.jpg"
        if nested.exists():
            return nested

    return direct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", help="Path to rerank output parquet")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: <parquet_stem>/images/)")
    parser.add_argument("-n", "--top-n", type=int, default=None,
                        help="Limit to top N results (default: all)")
    parser.add_argument("--remap", default=None, metavar="FROM:TO",
                        help="Remap image path prefix (e.g. /scratch/mwf62/cyclomedia:/share/ju/cyclomedia/raw)")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"Error: {parquet_path} not found", file=sys.stderr)
        sys.exit(1)

    remap_from, remap_to = None, None
    if args.remap:
        parts = args.remap.split(":", 1)
        if len(parts) != 2:
            print("Error: --remap must be FROM:TO", file=sys.stderr)
            sys.exit(1)
        remap_from, remap_to = parts

    df = pd.read_parquet(parquet_path)

    if "rerank_rank" not in df.columns or "rerank_score" not in df.columns:
        print("Error: parquet missing rerank_rank/rerank_score columns", file=sys.stderr)
        sys.exit(1)

    df = df.sort_values("rerank_rank")
    if args.top_n:
        df = df.head(args.top_n)

    output_dir = Path(args.output_dir) if args.output_dir else parquet_path.with_suffix("") / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for _, row in df.iterrows():
        rank = int(row["rerank_rank"])
        score = float(row["rerank_score"])
        image_path = Path(str(row["image_path"]))

        if remap_from and remap_to:
            image_path = _remap_path(image_path, remap_from, remap_to)

        if not image_path.exists():
            if skipped < 5:
                print(f"  skip rank {rank}: {image_path} not found")
            skipped += 1
            continue

        original_name = Path(str(row["image_path"])).name
        link_name = f"{rank:04d}_{score:.4f}_{original_name}"
        link_path = output_dir / link_name
        link_path.unlink(missing_ok=True)
        link_path.symlink_to(image_path.resolve())
        created += 1

    if skipped > 5:
        print(f"  ... and {skipped - 5} more skipped")
    print(f"Created {created} symlinks in {output_dir}" +
          (f" ({skipped} skipped)" if skipped else ""))


if __name__ == "__main__":
    main()
