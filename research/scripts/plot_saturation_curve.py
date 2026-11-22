"""How the judgment answers as more of the photograph goes away.

The curve says whether patch causality accumulates. A straight line means the
cells act on their own and add up. A curve that flattens at once means the
judgment saturates, and no additive instrument can follow it.

Usage:
    python research/scripts/plot_saturation_curve.py --out outputs/figures/saturation.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("outputs/_monocle/deletion_curve")
TEAL = "#498573"
CORAL = "#e57264"
GREY = "#7a6f63"

RUNS = {
    "subway safety": "subway_safety_g2_self_ablate",
    "libraries": "libraries_g2_ceil_ablate",
    "restaurants": "restaurants_g2_ceil_ablate",
}


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/figures/saturation.png")
    ap.add_argument("--runs", nargs="+", default=None)
    args = ap.parse_args(argv)

    runs = RUNS if args.runs is None else {
        Path(r).name: Path(r).name for r in args.runs}
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": True, "figure.facecolor": "#ffffff",
    })
    fig, axes = plt.subplots(1, len(runs), figsize=(2.9 * len(runs), 2.5),
                             sharey=False)
    if len(runs) == 1:
        axes = [axes]
    for ax, (nice, tag) in zip(axes, runs.items()):
        p = D / tag / "summary.parquet"
        if not p.exists():
            ax.set_axis_off()
            continue
        d = pd.read_parquet(p)
        d = d[~d["keep_only"]]
        g = d.groupby("k")
        ks = np.array(sorted(d.k.unique()), dtype=float)
        safe = np.array([g.get_group(k)["dc_safe"].median() for k in ks])
        rnd = np.array([g.get_group(k)["dc_random"].median() for k in ks])
        dang = np.array([g.get_group(k)["dc_danger"].median() for k in ks])
        ax.plot(ks, safe, "o-", color=TEAL, ms=3.5, lw=1.4,
                label="the strongest cells go")
        ax.plot(ks, rnd, "o-", color=GREY, ms=3.5, lw=1.4,
                label="random cells go")
        ax.plot(ks, dang, "o-", color=CORAL, ms=3.5, lw=1.4,
                label="the weakest cells go")
        # A straight line from the origin through k=1: what a map that ADDS up
        # would give.
        ax.plot(ks, safe[0] * ks, "--", color=TEAL, lw=0.9, alpha=0.55,
                label="if the cells added up")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(int(k)) for k in ks])
        ax.set_ylim(min(safe.min() * 1.15, safe[0] * 8), max(dang.max() * 1.4,
                                                             1.0))
        ax.set_xlabel("cells removed, of 64")
        ax.text(0.02, 1.02, nice, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9, color="#1a1a1a")
    axes[0].set_ylabel("change in the judgment")
    axes[-1].legend(frameon=False, fontsize=7.2, loc="lower left")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
