"""Can an instrument NAME the cell that matters most? One bar for each arm.

The 3 reference lines are the controls the claim must beat:

| Line | What it is |
|---|---|
| chance | 1 of 64 |
| position only | the mean measured map of the OTHER images |
| best saliency | the best of 6 model-free features, sign given free |

Usage:
    python research/scripts/plot_top_cell.py --json <top1.json> --out <png>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEAL = "#498573"
GREY = "#7a6f63"
CORAL = "#e57264"
NICE = {"lens_own": "Jacobian lens", "gxa": "grad x activation",
        "cheap": "centered grad x act", "meang_own": "mean gradient",
        "meangp_own": "mean gradient, per position"}


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="outputs/figures/top_cell.png")
    args = ap.parse_args(argv)
    data = json.loads(Path(args.json).read_text())
    cases = list(data)
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": True, "figure.facecolor": "#ffffff",
    })
    # A shared x axis, or the eye compares 3 different scales.
    fig, axes = plt.subplots(1, len(cases), figsize=(3.1 * len(cases), 2.7),
                             sharey=True, sharex=True)
    if len(cases) == 1:
        axes = [axes]
    for ax, case in zip(axes, cases):
        d = data[case]
        arms = list(d["arms"])
        vals = [d["arms"][a]["top1"] for a in arms]
        sig = [d["arms"][a]["p_vs_prior"] < 0.05
               and d["arms"][a]["p_chance"] < 0.05 for a in arms]
        y = range(len(arms))
        ax.barh(list(y), vals, color=[TEAL if s else GREY for s in sig],
                height=0.62, zorder=2)
        ax.axvline(d["chance"], color="#1a1a1a", lw=0.9, ls=":", zorder=3)
        ax.axvline(d["prior"], color=CORAL, lw=1.1, zorder=3)
        ax.axvline(d["best_saliency"], color=CORAL, lw=1.1, ls="--", zorder=3)
        ax.set_yticks(list(y))
        ax.set_yticklabels([NICE.get(a, a) for a in arms])
        if ax is axes[len(cases) // 2]:
            ax.set_xlabel("the measured best cell is named first")
        ax.text(0.02, 1.02, case.replace("_", " "), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, color="#1a1a1a")
    # The 3 reference lines, named once.
    from matplotlib.lines import Line2D
    axes[-1].legend(handles=[
        Line2D([], [], color="#1a1a1a", ls=":", label="chance, 1 of 64"),
        Line2D([], [], color=CORAL, label="position only"),
        Line2D([], [], color=CORAL, ls="--", label="best saliency"),
    ], frameon=False, fontsize=7.2, loc="lower right")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
