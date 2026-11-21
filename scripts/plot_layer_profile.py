"""Where in depth the instrument reads the judgment.

One line for each case: the gain of the instrument over the same instrument
under another case's question, layer by layer. A band, not a spike, is what a
real effect looks like.

Usage:
    python scripts/plot_layer_profile.py --json outputs/figures/k1_layers.json \
        --out outputs/figures/layer_profile.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEAL = "#498573"
CORAL = "#e57264"
GREY = "#7a6f63"
INK = "#1a1a1a"
COLOR = {"restaurants": TEAL, "libraries": CORAL, "subway_safety": GREY}
NICE = {"gxa": "grad x activation", "cheap": "centered grad x act",
        "meang_own": "mean gradient", "meangp_own": "mean gradient, per position",
        "lens_own": "Jacobian lens"}


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", required=True)
    ap.add_argument("--arms", nargs="+", default=["gxa", "cheap"])
    ap.add_argument("--out", default="outputs/figures/layer_profile.png")
    a = ap.parse_args(argv)
    data = json.loads(Path(a.json).read_text())
    cases = [c for c in ("subway_safety", "libraries", "restaurants")
             if c in data]

    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.transparent": True, "figure.facecolor": "#ffffff",
    })
    fig, axes = plt.subplots(1, len(a.arms), figsize=(3.6 * len(a.arms), 2.9),
                             sharey=True, sharex=True)
    if len(a.arms) == 1:
        axes = [axes]
    for ax, arm in zip(axes, a.arms):
        for case in cases:
            rec = data[case]["arms"].get(arm, {})
            ls = sorted(int(k) for k in rec if "gain" in rec[k])
            if not ls:
                continue
            g = np.array([rec[str(L)]["gain"] if str(L) in rec
                          else rec[L]["gain"] for L in ls])
            lo = np.array([(rec[str(L)] if str(L) in rec else rec[L])["lo"]
                           for L in ls])
            hi = np.array([(rec[str(L)] if str(L) in rec else rec[L])["hi"]
                           for L in ls])
            col = COLOR.get(case, INK)
            ax.fill_between(ls, lo, hi, color=col, alpha=0.14, lw=0)
            ax.plot(ls, g, color=col, lw=1.7, marker="o", ms=3.4,
                    label=case.replace("_", " "))
        ax.axhline(0.0, color=INK, lw=0.8)
        ax.set_xlabel("block")
        ax.text(0.0, 1.03, NICE.get(arm, arm), transform=ax.transAxes,
                ha="left", va="bottom", color=INK)
    axes[0].set_ylabel("gain over the wrong question")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(a.out)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
