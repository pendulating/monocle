"""F10.21: separate the embedded conception from the change that an image causes.

Each point is 1 store word at 1 token and layer. One axis gives the rank in the same text
with no image, and the other gives the rank in the mean read over 591 images.

| Output in `research/figures/f10/mechanism/` | Content |
|------------------------------------|---------|
| `rank_cells.csv` | `model`, `token`, `layer`, `word`, `rank_image`, `rank_text`, `kind` |
| `mech_ranks_{model}.png`, `.json` | 1 panel for each token |

| Kind | Rule |
|------|------|
| `embedded` | Rank of at most TOP in both reads |
| `raised` | Rank of at most TOP with an image and at least FAR with no image |
| `lowered` | Rank of at most TOP with no image and at least FAR with an image |

The ranks match `research/figures/f10/stage/stage_words.csv`.

Use:
    python research/scripts/f10/plot_ranks.py          # reads the store, then plots
    python research/scripts/f10/plot_ranks.py --plot   # plots from rank_cells.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
OUT = ROOT / "research/figures/f10/mechanism"
TOP, FAR = 100, 1000
CAP = 30.0
# Each cell holds the paper words of that token. Qwen3.5-9B has fewer layers, and its cells are the best layers of F10.17.
CELLS = {"gemma-4-31b": [(" place", 40), (" cool", 36), ("?", 40), ("model", 44)], "qwen3.5-9b": [(" place", 16), (" cool", 24), ("?", 20), ("assistant", 24)]}
SHOW = {" place": "place", " cool": "cool", "?": "?", "model": "model", "assistant": "assistant"}
PREFER = {"location", "nearby", "surroundings", "vibe", "idea", "opinion", "funky", "subjective", "judging", "debatable", "attractiveness", "aesthetics", "inexpensive"}
# Axes positions of the word lists and of the count of each kind.
PLACE = {"embedded": (.98, .50, "right", "top"), "raised": (.02, .50, "left", "top"), "lowered": (.98, .02, "right", "bottom")}
COUNT = {"embedded": (.70, .985, "right"), "raised": (.02, .985, "left"), "lowered": (.98, .26, "right")}
# Function words stay as points but take no label.
SKIP = {"own", "que", "est", "anymore", "both", "which", "either", "whichever"}
NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B"}


def compute(model: str) -> pd.DataFrame:
    from analyze_logit import load, shard_reads

    d = load(model)
    head = d["head"]
    tail = d["store"].manifest["tail_tokens"]
    words = d["store"].vocab["token"].str.strip().to_numpy()
    files = shard_reads(d)
    text = np.load(d["store"].path / "text_reads.npy", mmap_mode="r")
    rows = []
    for token, layer in CELLS[model]:
        a = len(tail) - 1 - tail[::-1].index(token)
        b = head["layers"].index(layer)
        total, n = 0.0, 0
        for f in files:
            x = np.asarray(np.load(f, mmap_mode="r")[:, a, b], dtype=np.float64)
            total = total + ((x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True)).sum(0)
            n += len(x)
        mean = total / n
        t = np.asarray(text[a, b], dtype=np.float64)
        if head["softcap"]:
            t = CAP * np.arctanh((t / CAP).clip(-1 + 1e-7, 1 - 1e-7))
        t = (t - t.mean()) / t.std()
        order_i, order_t = (-mean).argsort().argsort() + 1, (-t).argsort().argsort() + 1
        kind = np.select([(order_i <= TOP) & (order_t <= TOP), (order_i <= TOP) & (order_t >= FAR), (order_t <= TOP) & (order_i >= FAR)], ["embedded", "raised", "lowered"], "")
        rows.append(pd.DataFrame({"model": model, "token": token, "layer": layer, "word": words, "rank_image": order_i, "rank_text": order_t, "kind": kind}))
        print(f"[f10.21] {model} {token!r} L{layer}: {pd.Series(kind).value_counts().to_dict()}", flush=True)
    return pd.concat(rows, ignore_index=True)


def picks(g: pd.DataFrame, kind: str, k: int) -> pd.DataFrame:
    """Give the words to label: the words of the paper first, then the strongest."""
    s = g[(g["kind"] == kind) & ~g["word"].isin(SKIP)].assign(prefer=lambda t: t["word"].isin(PREFER))
    if kind == "embedded":
        s = s.assign(key=s["rank_image"] + s["rank_text"])
    elif kind == "raised":
        s = s.assign(key=s["rank_image"] - s["rank_text"] / 1000)
    else:
        s = s.assign(key=s["rank_text"] - s["rank_image"] / 1000)
    return s.sort_values(["prefer", "key"], ascending=[False, True]).head(k)


def plot(t: pd.DataFrame, model: str) -> None:
    import _style as S

    S.apply_house_style()
    colors = {"embedded": S.PAL["blue"], "raised": S.PAL["coral"], "lowered": S.PAL["warmgrey"]}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0.07})
    for ax, (token, layer) in zip(axes, CELLS[model]):
        g = t[(t["token"] == token) & (t["layer"] == layer)]
        ax.hexbin(g["rank_text"], g["rank_image"], gridsize=34, xscale="log", yscale="log", bins="log", cmap="Greys", mincnt=1, linewidths=0, alpha=.45, extent=(0, 4.8, 0, 4.8))
        ax.plot([1, 6e4], [1, 6e4], color=S.EDGE, lw=.4, ls=(0, (3, 3)))
        for kind, k in (("embedded", 4), ("raised", 5), ("lowered", 3)):
            sel = g[g["kind"] == kind]
            ax.scatter(sel["rank_text"], sel["rank_image"], s=2.2, color=colors[kind], lw=0, zorder=3, alpha=.85)
            names = picks(g, kind, k)["word"].tolist()
            if names:
                x, y, ha, va = PLACE[kind]
                ax.text(x, y, "\n".join(names), transform=ax.transAxes, ha=ha, va=va, fontsize=6, color=colors[kind], linespacing=1.05, zorder=4, path_effects=[withStroke(linewidth=1.6, foreground="white")])
            if len(sel):
                x, y, ha = COUNT[kind]
                ax.text(x, y, str(len(sel)), transform=ax.transAxes, ha=ha, va="top", fontsize=6.5, color=colors[kind], fontweight="bold", zorder=4, path_effects=[withStroke(linewidth=1.6, foreground="white")])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(7e4, .7)
        ax.set_ylim(7e4, .7)
        ax.set_xticks([1, 100, 10000], ["1", "100", "10k"])
        ax.set_yticks([1, 100, 10000], ["1", "100", "10k"])
        ax.minorticks_off()
        ax.tick_params(labelsize=6.5, length=2, width=.4)
        ax.grid(False)
        ax.set_title(f"{SHOW[token]}, L{layer}", fontsize=7.5, family="monospace", pad=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("rank with image", fontsize=7)
    fig.supxlabel("rank with no image", fontsize=7, y=-.055)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=2.8, color=colors[k], label=n) for k, n in (("embedded", "embedded"), ("raised", "raised by image"), ("lowered", "lowered by image"))]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=6.5, frameon=False, handletextpad=.1, columnspacing=1.0, bbox_to_anchor=(.5, 1.07))
    path = OUT / f"mech_ranks_{model}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.02)
    plt.close(fig)
    meta = {"plot-title": "Embedded and image-caused words", "plot-caption": f"Rank of each word in the {NAMES[model]} read with and without an image. Bold values count the words of each kind in the top 100.",
            "plot-latex-label": "fig:ranks", "plot-tags": ["F10.21", "rank_added", model]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = OUT / "rank_cells.csv"
    if not args.plot:
        pd.concat([compute(m) for m in CELLS], ignore_index=True).to_csv(path, index=False)
    t = pd.read_csv(path, keep_default_na=False)
    for m in CELLS:
        plot(t[t["model"] == m], m)
        print(pd.concat([picks(g, k, 6) for _, g in t[t["model"] == m].groupby("token", sort=False) for k in ("embedded", "raised", "lowered")])[["token", "kind", "word", "rank_image", "rank_text"]].to_string(index=False))


if __name__ == "__main__":
    main()
