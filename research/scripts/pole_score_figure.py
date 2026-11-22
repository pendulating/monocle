"""Forest plot: Spearman rho vs the unit proxy, per folder, for the pairwise
rating and the single-image reads on the same units.

    .venv-3.12/bin/python research/scripts/pole_score_figure.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path("/share/ju/matt/mllmsci/outputs/_monocle/pole_score")
READS = ["pairwise", "c", "contrast_L42", "contrast_L45"]
LABELS = {"pairwise": "pairwise rating (CVPR)", "c": "single-image gap c",
          "contrast_L42": "contrast projection, block 42",
          "contrast_L45": "contrast projection, block 45"}
COLORS = {"pairwise": "#222222", "c": "#d1495b", "contrast_L42": "#1f6f8b", "contrast_L45": "#6a9fb5"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    df = pd.read_parquet(root / "proxy_correlations.parquet")
    df = df[df["read"].isin(READS)]
    folders = list(dict.fromkeys(df["folder"]))
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(folders) + 2.2))
    yt, yl = [], []
    for fi, folder in enumerate(folders):
        sub = df[df.folder == folder].set_index("read")
        n = int(sub.loc["pairwise", "n_units"])
        for ri, read in enumerate(READS):
            if read not in sub.index:
                continue
            y = fi * (len(READS) + 1) + ri
            r = sub.loc[read]
            ax.errorbar(r.rho, y, xerr=[[r.rho - r.rho_lo], [r.rho_hi - r.rho]], fmt="o",
                        color=COLORS[read], capsize=3, label=LABELS[read] if fi == 0 else None)
        yt.append(fi * (len(READS) + 1) + 1.5)
        yl.append(f"{folder}\n(n = {n:,})")
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(yt); ax.set_yticklabels(yl)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman rho vs the unit proxy (95% bootstrap over units)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = root / "research" / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / "proxy_forest.png", dpi=150, facecolor="white")
    print("wrote", out / "proxy_forest.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
