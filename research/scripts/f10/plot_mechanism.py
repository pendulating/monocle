"""F10 summary figures for the mechanism report. Each figure gets a JSON sidecar."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/notebooks/cvpr"), str(ROOT / "research/scripts/f10")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import _style as S  # noqa: E402

OUT = ROOT / "research/figures/f10/mechanism"
REASON = ROOT / "research/figures/f10/reason"
MODELS = ("gemma-4-31b", "qwen3.5-9b")
NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B"}
VALUES = ["beauty", "nature", "comfort", "order", "heritage", "modernity", "character", "community", "vitality", "status", "edge", "utility"]


def save(fig, name: str, title: str, caption: str, tags: list[str]) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (OUT / f"{name}.json").write_text(json.dumps({"plot-title": title, "plot-caption": caption, "plot-latex-label": f"fig:{name.replace('_', '-')}", "plot-tags": tags}, indent=2) + "\n")


def values() -> None:
    d = pd.read_csv(REASON / "reason_labels.csv")
    d = d[d["field"] == "reasoning"].set_index(["model", "label"])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), constrained_layout=True, sharex=True)
    order = list(reversed(VALUES))
    y = np.arange(len(order))
    for ax, model in zip(axes, MODELS):
        for stance, dy, color in [("for", 0.19, S.PAL["teal"]), ("against", -0.19, S.PAL["coral"])]:
            r = [d["r_margin"].get((model, f"criterion:{v}:{stance}"), np.nan) for v in order]
            q = [d["p_margin_holm"].get((model, f"criterion:{v}:{stance}"), 1.0) for v in order]
            ax.barh(y + dy, r, height=0.36, color=color, alpha=1.0, edgecolor=S.EDGE, linewidth=S.EDGE_LW, label=stance)
            for yy, rr, qq in zip(y + dy, r, q):
                if qq < 0.05:
                    ax.plot(rr + np.sign(rr) * 0.03, yy, marker="o", ms=1.8, color=S.EDGE)
        ax.axvline(0, color=S.EDGE, linewidth=0.6)
        ax.set_yticks(y, order, fontsize=7)
        ax.set_xlabel("r with the cool margin, over 591 images")
        ax.set_title(NAMES[model], fontsize=8.5)
    axes[1].legend(loc="lower right", fontsize=7, frameon=False, title="stance", title_fontsize=7)
    save(fig, "mech_values", "Stated values and the cool margin",
         "Correlation over 591 images between the share of reasoning texts that use a value, for or against cool, and the cool margin of the read. A dot marks a Holm-adjusted p below 0.05.", ["F10.11", "value"])


def objects() -> None:
    d = pd.read_csv(REASON / "argument_cells.csv")
    d = d[d["n"] >= 10]
    keep = pd.read_csv(REASON / "argument_flips.csv").groupby("object")["n"].sum().sort_values(ascending=False).index[:16]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        g = d[(d["model"] == model)].pivot(index="object", columns="value", values="for_share").reindex(index=list(keep), columns=VALUES)
        im = ax.imshow(g.to_numpy(), aspect="auto", cmap=S.CMAP_DIV, vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor(S.PAL["cream"])
        ax.grid(False)
        ax.set_xticks(range(len(VALUES)), VALUES, rotation=55, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(keep)), list(keep), fontsize=6.5)
        ax.set_title(NAMES[model], fontsize=8.5)
    fig.colorbar(im, ax=axes, shrink=0.6, label="share of spans for cool")
    save(fig, "mech_objects", "One object, opposite signs",
         "Share of criterion spans that argue for cool, among spans for or against cool, for each cited object under each stated value. Blank cells hold fewer than 10 spans.", ["F10.16", "observation"])


def label(t: str) -> str:
    return t.strip() if t.strip() else t.replace("\n", "\\n")


def gain() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.0), constrained_layout=True, sharey=True)
    for ax, model in zip(axes, MODELS):
        d = pd.read_csv(ROOT / f"data/f10/hop_free/{model}/decoder_grid1.csv")
        d["pass"] = (d["p_holm"] < 0.05) & (d["gain"] > 0)
        best = d.sort_values("gain", ascending=False).groupby("index", sort=True).first()
        passed = d.groupby("index")["pass"].any()
        ax.plot(best.index, best["gain"], color=S.PAL["blue"], linewidth=1.1, zorder=2)
        ax.scatter(best.index[passed], best["gain"][passed], s=16, color=S.PAL["blue"], edgecolor=S.EDGE, linewidth=S.EDGE_LW, zorder=3, label="a layer passes Holm")
        ax.scatter(best.index[~passed], best["gain"][~passed], s=16, facecolor="white", edgecolor=S.EDGE, linewidth=S.EDGE_LW, zorder=3, label="no layer passes Holm")
        ax.axhline(0, color=S.EDGE, linewidth=0.6)
        ax.set_xticks(best.index, [label(t) for t in best["token"]], rotation=55, ha="right", fontsize=6, family="monospace")
        ax.set_ylabel("gain in R2")
        ax.set_title(NAMES[model], fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    save(fig, "mech_gain", "Margin information beyond the mean patch",
         "Largest gain in out-of-fold R2 over the layers of each token, when a read of the cell joins a ridge decoder on the mean patch of the image at all layers. Folds follow the NTA. Holm covers the cells of 1 model.", ["F10.9", "decoder"])


BOROUGHS = ["Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island"]
BOROUGH_COLOR = {"Manhattan": "coral", "Brooklyn": "amber", "Bronx": "teal", "Queens": "blue", "Staten Island": "periwinkle"}


def sense() -> None:
    d = pd.read_csv(ROOT / "research/figures/f10/sense/sense_images.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2), layout="constrained")
    for i, model in enumerate(MODELS):
        g = d[d["model"] == model]
        for ax, y, name in zip(axes[i], ["margin", "temperature"], ["cool margin", "traces with temperature words"]):
            for k in BOROUGHS:
                h = g[g["borough"] == k]
                ax.scatter(h["score"], h[y], s=6, color=S.PAL[BOROUGH_COLOR[k]], alpha=0.75, linewidth=0, label=k)
            if y == "temperature":
                bins = pd.qcut(g["score"], 10, labels=False)
                mean = g.groupby(bins)[["score", "temperature"]].mean()
                ax.plot(mean["score"], mean["temperature"], color=S.EDGE, marker="s", ms=3, linewidth=1.0, label="mean of 1 decile")
            ax.set_ylabel(name, fontsize=7.5)
            ax.set_title(NAMES[model], fontsize=8.5)
        axes[i, 0].set_xlabel("sense score at the word cool, layer 24")
        axes[i, 1].set_xlabel("sense score at the word cool, layer 24")
    fig.legend(*axes[0, 1].get_legend_handles_labels(), fontsize=7, frameon=False, markerscale=2, loc="outside lower center", ncol=6)
    save(fig, "mech_sense", "The sense of cool and the verdict",
         "Each point is 1 of 591 images. The sense score is the mean read of social words minus the mean read of temperature words. Left: the score does not follow the cool margin. Right: the score follows the share of separate reasoning traces that name temperature.", ["F10.17", "sense"])


def borough() -> None:
    probe = pd.read_csv(ROOT / "research/figures/f10/probe/probe_borough.csv")
    cells = pd.read_csv(ROOT / "research/figures/f10/sense/sense_cells.csv")
    splits = pd.read_csv(ROOT / "research/figures/f10/sense/sense_splits.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1]})
    show = [("cool", "coral", "-"), ("trendy", "amber", "-"), ("urban", "blue", "-"), ("green", "teal", "-")]
    for i, model in enumerate(MODELS):
        ax = axes[i, 0]
        x = np.arange(5)
        for w, c, ls in show:
            r = probe[(probe["model"] == model) & (probe["word"] == w)].iloc[0]
            ax.plot(x, [r[k] for k in BOROUGHS], color=S.PAL[c], marker="o", ms=3, linewidth=1.1, linestyle=ls, label=f"probe: {w}")
        g = cells[(cells["model"] == model) & (cells["token"] == " cool") & (cells["layer"] == 24)].iloc[0]
        ax.plot(x, [g[k] for k in BOROUGHS], color=S.EDGE, marker="s", ms=3, linewidth=1.1, linestyle="--", label="sense score")
        ax.axhline(0, color=S.EDGE, linewidth=0.5)
        ax.set_xticks(x, ["Manh", "Bklyn", "Bronx", "Queens", "SI"], fontsize=7)
        ax.set_ylabel("borough mean, in NTA SD", fontsize=7.5)
        ax.set_title(NAMES[model], fontsize=8.5)
        ax = axes[i, 1]
        for dx, k, c, name in [(-0.15, "k0", "blue", "all axes"), (0.15, "k1", "coral", "axis 1 removed")]:
            g = splits[splits["model"] == model].groupby(["split", "borough"])[k].mean().unstack()[BOROUGHS]
            ax.errorbar(x + dx, g.mean(), yerr=[g.mean() - g.min(), g.max() - g.mean()], fmt="o", ms=3, color=S.PAL[c], ecolor=S.PAL[c], elinewidth=0.8, label=name)
        ax.axhline(0, color=S.EDGE, linewidth=0.5)
        ax.set_xticks(x, ["Manh", "Bklyn", "Bronx", "Queens", "SI"], fontsize=7)
        ax.set_ylabel("split-half r of word t", fontsize=7.5)
        ax.set_ylim(-0.3, 1)
        ax.set_title(NAMES[model], fontsize=8.5)
    axes[0, 0].legend(fontsize=6, frameon=False, ncol=2, loc="upper right")
    axes[0, 1].legend(fontsize=6.5, frameon=False, loc="lower left")
    save(fig, "mech_borough", "The borough axis and the stability of lists",
         "Left: borough means of 4 probe margins and of the sense score at the word cool, in SD of the 197 NTA means. Right: correlation of word t values between 2 random halves of the NTAs, as the mean over 3 cells for each of 40 splits. Bars give the lowest and the highest split.", ["F10.14", "F10.17", "borough"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    S.apply_house_style()
    which = sys.argv[1:] or ["values", "objects"]
    for name in which:
        globals()[name]()


if __name__ == "__main__":
    main()
