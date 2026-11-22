"""The control that separates a judgment reader from a picture reader.

Each arm gets 2 bars: the right question and the wrong question. An instrument
that reads the judgment must FALL to chance when the question changes.

Usage:
    python research/scripts/plot_cross_question.py --out outputs/figures/cross_question.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEAL = "#498573"
CORAL = "#e57264"
NICE = {"gxa": "grad x activation", "cheap": "centered grad x act",
        "meangp_own": "mean gradient, per position",
        "meang_own": "mean gradient", "lens_own": "Jacobian lens"}


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="outputs/figures/cross_question.png")
    args = ap.parse_args(argv)
    data = json.loads(Path(args.json).read_text())
    cases = list(data)
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": True, "figure.facecolor": "#ffffff",
    })
    fig, axes = plt.subplots(1, len(cases), figsize=(3.1 * len(cases), 2.8),
                             sharey=True, sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        d = data[case]
        arms = list(d)
        y = np.arange(len(arms), dtype=float)
        right = [d[a]["right"] for a in arms]
        wrong = [d[a]["wrong"] for a in arms]
        ax.barh(y + 0.19, right, height=0.34, color=TEAL, zorder=2,
                label="the right question")
        ax.barh(y - 0.19, wrong, height=0.34, color=CORAL, zorder=2,
                label="the wrong question")
        ax.axvline(d[arms[0]]["chance"], color="#1a1a1a", lw=0.9, ls=":",
                   zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([NICE.get(a, a) for a in arms])
        ax.text(0.02, 1.02, case.replace("_", " "), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, color="#1a1a1a")
        if ax is axes[len(cases) // 2]:
            ax.set_xlabel("the measured best cell is named first")
    axes[-1].legend(frameon=False, fontsize=7.4, loc="lower right")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
