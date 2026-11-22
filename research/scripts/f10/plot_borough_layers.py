"""F10.22: the top words of each borough at the `cool` token, through the layers.

The contrast is C3 of the paper: the mean read of 1 borough minus the mean read of the other
4, with the capped reads. Each cell draws from the top 12 words of `by_raw` in
`research/figures/f10/borough/borough_words.csv`. The words that the paper cites for that borough and
token (`borough_claims.csv`) come first, then the other words by rank. Thus a cell is a
selection from the top 12, not the top N. The words in SKIP take no place.

Use:
    python research/scripts/f10/plot_borough_layers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/notebooks/cvpr")]
OUT = ROOT / "research/figures/f10/mechanism/mech_borough_layers.png"
MODEL = "gemma-4-31b"
TOKENS = (" cool",)
LAYERS = {" cool": (8, 12, 16, 20, 24, 28, 32, 40)}
BOROUGHS = ("Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island")
COLOR = {"Manhattan": "coral", "Brooklyn": "amber", "Bronx": "teal", "Queens": "blue", "Staten Island": "periwinkle"}
SKIP = {"own", "and", "que", "por"}
# These cells cite more words than a cell holds. The words that the prose cites come first.
PIN = {("Manhattan", 20): ("lingerie", "femme", "couture", "housewife", "salon"), ("Bronx", 24): ("porn", "anarch", "illegal", "vandalism", "gangs"), ("Brooklyn", 24): ("slums", "prostit", "traff", "marijuana", "bistro")}
TOP, N_WORDS = 12, 6
ROW, HEAD, GAP, LEFT = 0.54, 0.2, 0.0, 0.86


def main() -> None:
    import _style as S

    S.apply_house_style()
    t = pd.read_csv(ROOT / "research/figures/f10/borough/borough_words.csv")
    t = t[(t["model"] == MODEL) & t["token"].isin(TOKENS)]
    t = {(r.token, r.borough, r.layer): r.by_raw for r in t.itertuples()}
    c = pd.read_csv(ROOT / "research/figures/f10/borough/borough_claims.csv")
    c = c[c["model"] == MODEL]
    cited = {k: set(g["word"]) for k, g in c.groupby(["borough", "token"])}
    shown, wanted = set(), {(b, t, w) for (b, t), v in cited.items() for w in v if t in TOKENS}
    width, col = 5.9, (5.9 - LEFT) / len(LAYERS[TOKENS[0]])
    height = len(TOKENS) * (HEAD + ROW * len(BOROUGHS) + GAP)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    y = 0.0
    for token in TOKENS:
        ax.text(0, y + HEAD - .05, token.strip(), family="monospace", fontsize=7.5, va="bottom", fontweight="bold")
        for j, layer in enumerate(LAYERS[token]):
            ax.text(LEFT + j * col + .03, y + HEAD - .05, f"L{layer}", fontsize=6.5, va="bottom", color=S.EDGE)
        y += HEAD
        ax.plot([0, width], [y, y], color=S.EDGE, lw=.5)
        if token == " cool":
            a, b = LEFT + LAYERS[token].index(12) * col, LEFT + (LAYERS[token].index(32) + 1) * col
            ax.add_patch(plt.Rectangle((a, y), b - a, ROW * len(BOROUGHS), color=S.PAL["coral"], alpha=.09, lw=0, zorder=0))
        for name in BOROUGHS:
            ax.add_patch(plt.Rectangle((0, y + .045), .05, ROW - .09, color=S.PAL[COLOR[name]], lw=0))
            ax.text(.09, y + ROW / 2, name, fontsize=6.5, va="center")
            for j, layer in enumerate(LAYERS[token]):
                pool = [w for w in t[(token, name, layer)].split(", ") if w not in SKIP][:TOP]
                mine = cited.get((name, token), set())
                pin = PIN.get((name, layer), ())
                words = sorted(pool, key=lambda w: (w not in pin, w not in mine))[:N_WORDS]
                shown |= {(name, token, w) for w in words if w in mine}
                ax.text(LEFT + j * col + .03, y + ROW / 2, "\n".join(words), fontsize=5.8, va="center", linespacing=1.02)
            y += ROW
            ax.plot([0, width], [y, y], color="#dddddd", lw=.3)
        y += GAP
    print(f"[f10.22] {len(shown)} of {len(wanted)} cited words shown; missing {sorted(w for _, _, w in wanted - shown)}", flush=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=.01, transparent=False, facecolor="white")
    plt.close(fig)
    OUT.with_suffix(".json").write_text(json.dumps({"plot-title": "Borough words through the layers", "plot-caption": "Words of top-12 contrast C3 at the cool token through the layers of Gemma-4-31B.",
                                                    "plot-latex-label": "fig:borough-layers", "plot-tags": ["F10.22", "C3", MODEL]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
