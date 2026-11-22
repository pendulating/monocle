"""How much the 2 poles share their words, by layer, on every case.

The back-to-back panels of `plot_pole_tokens.py` show 1 case. This puts all 7
in 1 figure. For each case and layer it takes the top-N words of each pole and
gives the Jaccard of the 2 lists.

The reference line is the pole against a RANDOM patch of the same photograph.
The poles do differ from a random patch, so that line sits lower. The gap is
what the 2 poles SHARE and a random patch does not. When the pole-pole line
sits far above it, the 2 poles carry the same words.

Usage:
    python research/scripts/plot_pole_overlap.py \\
        --roots outputs/_monocle/causal_words/rlens8s \\
                outputs/_monocle/causal_words/jlens8s \\
        --labels R-lens J-lens --out outputs/figures/pole_overlap.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

TEAL = "#498573"
CORAL = "#e57264"
INK = "#1a1a1a"
FAINT = "#b8b0a6"
JUNK = re.compile(r"^[\s\W_]*$")


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def case_curve(path: Path, n_words: int, top_rank: int):
    df = pd.read_parquet(path)
    df["word"] = df["word"].str.strip().str.lower()
    df = df[(df["rank"] < top_rank)
            & (~df["word"].map(lambda w: bool(JUNK.match(w))))]
    layers, pn, pr = [], [], []
    for layer in sorted(df.layer.unique()):
        g = df[df.layer == layer]
        top = {r: set(g[g.role == r].word.value_counts().index[:n_words])
               for r in ("pos", "neg", "random")}
        layers.append(int(layer))
        pn.append(jaccard(top["pos"], top["neg"]))
        pr.append(0.5 * (jaccard(top["pos"], top["random"])
                         + jaccard(top["neg"], top["random"])))
    return layers, np.array(pn), np.array(pr)


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--n-words", type=int, default=12)
    ap.add_argument("--top-rank", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    roots = [Path(r) for r in a.roots]
    labels = a.labels or [r.name for r in roots]
    fig, axes = plt.subplots(1, len(roots), figsize=(4.5 * len(roots), 3.5),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, root, label in zip(axes, roots, labels):
        cases = sorted(p.name for p in root.iterdir() if p.is_dir())
        allpn, allpr, layers = [], [], None
        for case in cases:
            f = root / case / "words.parquet"
            if not f.exists():
                continue
            layers, pn, pr = case_curve(f, a.n_words, a.top_rank)
            ax.plot(layers, pn, color=TEAL, lw=0.9, alpha=0.32, zorder=2)
            ax.plot(layers, pr, color=FAINT, lw=0.9, alpha=0.42, zorder=1)
            allpn.append(pn)
            allpr.append(pr)
        if layers is None:
            continue
        ax.plot(layers, np.nanmean(allpn, 0), color=TEAL, lw=2.6, zorder=4,
                marker="o", ms=4.2, label="the 2 poles against each other")
        ax.plot(layers, np.nanmean(allpr, 0), color=CORAL, lw=2.0, zorder=3,
                ls="--", marker="s", ms=3.6,
                label="a pole against a random patch")
        ax.set_xticks(layers)
        ax.set_xlabel("lens layer", fontsize=9, color="#4a4a4a")
        ax.tick_params(labelsize=8, colors="#6b6b6b", length=2)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=FAINT, lw=0.5, alpha=0.35, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color(FAINT)
        ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", va="top", color=INK)

    axes[0].set_ylabel(f"top-{a.n_words} word overlap (Jaccard)",
                       fontsize=9, color="#4a4a4a")
    axes[0].legend(frameon=False, fontsize=8.2, loc="lower left",
                   bbox_to_anchor=(0.0, 0.03))
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, transparent=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
