"""Contact sheets: the images closest to and furthest from each pole vector.

    .venv-3.12/bin/python scripts/pole_sample_figure.py --case restaurants --layer 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path("/share/ju/matt/mllmsci/outputs/_monocle/pole_sample")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="restaurants")
    ap.add_argument("--layer", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root) / args.case
    s = json.loads((root / "summary.json").read_text())
    w_pos, w_neg = s["words"]
    df = pd.read_parquet(root / "projections.parquet")
    df = df[df.layer == args.layer].reset_index(drop=True)
    k = args.k
    panels = []
    for w in (w_neg, w_pos):
        col = f"proj_{w}"
        top = df.sort_values(col, ascending=False).head(k)
        bot = df.sort_values(col, ascending=True).head(k)
        panels.append((f"closest to '{w}' (highest projection)", top, col))
        panels.append((f"furthest from '{w}' (lowest projection)", bot, col))
    fig, axes = plt.subplots(len(panels), k, figsize=(3.2 * k, 3.0 * len(panels)))
    for r, (title, sub, col) in enumerate(panels):
        for c, (_, row) in enumerate(sub.iterrows()):
            ax = axes[r, c]
            ax.imshow(Image.open(row["image_path"]).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"proj {row[col]:+.1f}   c {row['c']:+.1f}", fontsize=9)
            if c == 0:
                ax.set_ylabel(title, fontsize=9)
    fig.suptitle(f"{args.case}, block {args.layer}: answer residual projected on each pole vector", fontsize=11)
    fig.tight_layout()
    out = root / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / f"poles_L{args.layer}.png", dpi=110, facecolor="white")
    print("wrote", out / f"poles_L{args.layer}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
