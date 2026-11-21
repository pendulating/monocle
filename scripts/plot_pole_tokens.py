"""The words at the 2 poles of the causal map, side by side, 1 panel each layer.

`monocle.causal_words` reads the lens at the patches that hold a judgment up
and at the patches that push it down. This draws the 2 word distributions back
to back, so a reader sees whether the 2 poles say different things.

Each panel is 1 layer. A word runs left for the negative pole and right for the
positive pole. The bars of a word that both poles say are the same length, so a
symmetric panel means the poles carry the same words.

Usage:
    python scripts/plot_pole_tokens.py --root outputs/_monocle/causal_words/rlens8s \\
        --case restaurants --out outputs/figures/pole_tokens_restaurants.png
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


def clean(w: str) -> str:
    return str(w).strip().lower()


def load(root: Path, case: str, top_rank: int) -> pd.DataFrame:
    df = pd.read_parquet(root / case / "words.parquet")
    df["word"] = df["word"].map(clean)
    df = df[(df["rank"] < top_rank)
            & (~df["word"].map(lambda w: bool(JUNK.match(w))))]
    return df[df.role.isin(["pos", "neg"])]


def panel(ax, g: pd.DataFrame, n_words: int, layer: int, *,
          share: bool = True, lim: float | None = None) -> tuple[float, int]:
    """Draw 1 layer. Returns the Jaccard of the 2 top lists and the shared count.

    With `share` each pole becomes a share of its OWN word events. The 2 sides
    are then distributions, and a symmetric panel means the poles say the same
    words in the same proportion.
    """
    pos = g[g.role == "pos"].word.value_counts()
    neg = g[g.role == "neg"].word.value_counts()
    if share:
        pos = 100.0 * pos / max(pos.sum(), 1)
        neg = 100.0 * neg / max(neg.sum(), 1)
    both = (pos.add(neg, fill_value=0)).sort_values(ascending=False)
    words = list(both.index[:n_words])[::-1]
    y = np.arange(len(words))
    pv = np.array([pos.get(w, 0) for w in words], dtype=float)
    nv = np.array([neg.get(w, 0) for w in words], dtype=float)

    ax.barh(y, pv, color=TEAL, height=0.74, zorder=2)
    ax.barh(y, -nv, color=CORAL, height=0.74, zorder=2)
    if lim is None:
        lim = max(pv.max(), nv.max(), 1.0) * 1.62
    for i, w in enumerate(words):
        ax.text(0, i, f" {w} ", ha="center", va="center", fontsize=7.1,
                color=INK, zorder=4,
                bbox=dict(boxstyle="round,pad=0.14", fc="white",
                          ec="none", alpha=0.86))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.8, len(words) - 0.2)
    ax.axvline(0, color=FAINT, lw=0.7, zorder=3)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6.5, colors="#6b6b6b", length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(FAINT)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.text(0.02, 0.975, f"L{layer:02d}", transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", va="top", ha="left", color=INK)

    ptop, ntop = set(pos.index[:n_words]), set(neg.index[:n_words])
    u = len(ptop | ntop)
    jac = len(ptop & ntop) / u if u else float("nan")
    ax.text(0.98, 0.975, f"shared {len(ptop & ntop)}/{n_words}",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            color="#6b6b6b")
    return jac, len(ptop & ntop)


def global_limit(df: pd.DataFrame, layers, n_words: int,
                 share: bool) -> float:
    """The largest bar over every panel, so all the panels share 1 scale."""
    hi = 0.0
    for layer in layers:
        g = df[df.layer == layer]
        for role in ("pos", "neg"):
            v = g[g.role == role].word.value_counts()
            if v.empty:
                continue
            if share:
                v = 100.0 * v / max(v.sum(), 1)
            hi = max(hi, float(v.iloc[:n_words].max()))
    return max(hi, 1.0) * 1.62


def main(argv: list[str] | None = None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--layers", nargs="+", type=int, default=None)
    ap.add_argument("--n-words", type=int, default=12)
    ap.add_argument("--top-rank", type=int, default=5)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--counts", action="store_true",
                    help="draw raw counts; the default draws a share of a pole")
    ap.add_argument("--shared-x", action="store_true",
                    help="1 x range for every panel; the default fits each one")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    df = load(Path(a.root), a.case, a.top_rank)
    layers = a.layers or sorted(df.layer.unique())
    ncols = min(a.ncols, len(layers))
    nrows = -(-len(layers) // ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.05 * ncols, 0.30 * a.n_words * nrows
                                      + 1.05))
    axes = np.atleast_1d(axes).ravel()
    share = not a.counts
    lim = global_limit(df, layers, a.n_words, share) if a.shared_x else None
    jac = []
    for ax, layer in zip(axes, layers):
        j, _ = panel(ax, df[df.layer == layer], a.n_words, int(layer),
                     share=share, lim=lim)
        jac.append(j)
    for ax in axes[len(layers):]:
        ax.axis("off")

    fig.legend(handles=[
        mp.Patch(color=TEAL, label="the patch holds the judgment up"),
        mp.Patch(color=CORAL, label="the patch pushes the judgment down")],
        loc="lower center", ncol=2, frameon=False, fontsize=8.5,
        bbox_to_anchor=(0.5, 0.0))
    xlab = ("what share of a pole's words this word is, in percent"
            if share else "how many times the word is in the top 5 of a pole")
    fig.supxlabel(xlab, fontsize=8, color="#6b6b6b", y=0.052)
    fig.tight_layout(rect=(0, 0.075, 1, 1))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, transparent=True)
    print(f"wrote {out}")
    print(f"mean top-{a.n_words} overlap across layers: {np.nanmean(jac):.3f}")
    for layer, j in zip(layers, jac):
        print(f"  L{int(layer):02d}  Jaccard {j:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
