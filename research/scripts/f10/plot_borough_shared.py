"""F10.30: the borough words that all 3 models share, at the `cool` token, by relative depth.

The contrast is C3 of the paper, in the form of a Welch t over NTAs: the 197 NTA mean readouts of
1 borough against those of the other 4, for each word and saved layer. The t of a word is the mean
over the saved layers in a band of relative depth (layer / number of layers). The 3 models have
different vocabularies, thus we join them on the word string. We keep the lowercase words of 4 or
more letters that all 3 vocabularies hold.

In each model, we z-score the t over the shared words. The score of a word is its lowest z of the
3 models. Thus a word ranks high only when it ranks high in all 3 models. The figure gives the top
words of each borough and band. The script also gives the Spearman rho of the full t vectors
between each pair of models.

Use:
    python research/scripts/f10/plot_borough_shared.py
"""
from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__version__ = "1.0.0"
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/notebooks/cvpr")]
DATA = ROOT / "data/f10/prompts"
OUT = ROOT / "research/figures/f10/mechanism/mech_borough_shared"
# The number of layers of each model. Relative depth = layer / this number.
MODELS = {"gemma-4-31b": 60, "qwen3.5-9b": 32, "qwen3.6-27b": 64}
NAMES = {"gemma-4-31b": "Gemma-4-31B", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
POSITION = "after_place"  # the `cool` token
BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))
BOROUGHS = ("Manhattan", "Brooklyn", "Bronx", "Queens", "Staten Island")
COLOR = {"Manhattan": "coral", "Brooklyn": "amber", "Bronx": "teal", "Queens": "blue", "Staten Island": "periwinkle"}
WORD = re.compile(r"[a-z]{4,}")
N_WORDS, N_CSV = 5, 20
ROW, HEAD, LEFT, WIDTH = 0.46, 0.2, 0.86, 5.9


def borough_t(model: str) -> tuple[dict, np.ndarray]:
    """Give the band-mean Welch t of each borough against the rest, and the vocabulary of the model."""
    fr = pd.read_parquet(DATA / model / "frame.parquet")
    order = pd.factorize(fr["nta2020"])[1]
    borough = fr.drop_duplicates("nta2020").set_index("nta2020")["borough"].reindex(order).to_numpy()
    layers = np.array(json.loads((DATA / model / "meta.json").read_text())["layers"])
    x = np.load(DATA / model / f"nta_{POSITION}.npy", mmap_mode="r")
    if x.shape[0] != len(order) or x.shape[1] != len(layers):
        raise ValueError(f"{model}: shape {x.shape} does not match the frame and the layers")
    vocab = pd.read_parquet(ROOT / f"data/f10/hop_all/{model}/vocab.parquet")["token"].str.strip().to_numpy().astype(str)
    out = {}
    for b in BOROUGHS:
        g = borough == b
        for k, (lo, hi) in enumerate(BANDS):
            rel = layers / MODELS[model]
            ts = []
            for li in np.flatnonzero((rel >= lo) & (rel < hi)):
                a, c = np.asarray(x[g, li], dtype=np.float64), np.asarray(x[~g, li], dtype=np.float64)
                ts.append(stats.ttest_ind(a, c, axis=0, equal_var=False).statistic)
            out[(b, k)] = np.nanmean(ts, axis=0)
    return out, vocab


def shared_words() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the top words of each borough and band, and the rho between models."""
    t, index = {}, {}
    for m in MODELS:
        t[m], vocab = borough_t(m)
        s = pd.Series(np.arange(len(vocab)), index=vocab)
        index[m] = s[~s.index.duplicated()]
    common = sorted(set.intersection(*(set(index[m].index) for m in MODELS)))
    common = [w for w in common if WORD.fullmatch(w)]
    rows, agree = [], []
    for b in BOROUGHS:
        for k, (lo, hi) in enumerate(BANDS):
            T = np.stack([t[m][(b, k)][index[m][common].to_numpy()] for m in MODELS])
            Z = (T - np.nanmean(T, 1, keepdims=True)) / np.nanstd(T, 1, keepdims=True)
            low = np.nan_to_num(np.nanmin(Z, 0), nan=-np.inf)
            for rank, i in enumerate(np.argsort(-low)[:N_CSV]):
                rows.append({"borough": b, "band": k, "depth": f"{lo:.2f}-{min(hi, 1):.2f}", "rank": rank + 1,
                             "word": common[i], "z_min": float(low[i]), **{f"t_{m}": float(T[j, i]) for j, m in enumerate(MODELS)}})
            for (i, a), (j, c) in combinations(enumerate(MODELS), 2):
                ok = ~np.isnan(T[i]) & ~np.isnan(T[j])
                rho = stats.spearmanr(T[i, ok], T[j, ok]).statistic
                agree.append({"borough": b, "band": k, "pair": f"{a}|{c}", "rho": float(rho), "n_words": int(ok.sum())})
    print(f"[f10.30] {len(common)} shared words", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(agree)


def figure(words: pd.DataFrame) -> None:
    import _style as S

    S.apply_house_style()
    col = (WIDTH - LEFT) / len(BANDS)
    height = HEAD + ROW * len(BOROUGHS)
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.text(0, HEAD - .05, "cool", family="monospace", fontsize=7.5, va="bottom", fontweight="bold")
    for k, (lo, hi) in enumerate(BANDS):
        ax.text(LEFT + k * col + .03, HEAD - .05, f"depth {lo:.2f}–{min(hi, 1):.2f}", fontsize=6.5, va="bottom", color=S.EDGE)
    ax.plot([0, WIDTH], [HEAD, HEAD], color=S.EDGE, lw=.5)
    y = HEAD
    for b in BOROUGHS:
        ax.add_patch(plt.Rectangle((0, y + .045), .05, ROW - .09, color=S.PAL[COLOR[b]], lw=0))
        ax.text(.09, y + ROW / 2, b, fontsize=6.5, va="center")
        for k in range(len(BANDS)):
            w = words[(words.borough == b) & (words.band == k)].sort_values("rank").word.head(N_WORDS)
            ax.text(LEFT + k * col + .03, y + ROW / 2, "\n".join(w), fontsize=5.8, va="center", linespacing=1.02)
        y += ROW
        ax.plot([0, WIDTH], [y, y], color="#dddddd", lw=.3)
    for ext in ("png", "pdf"):
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {"dpi": 300}
        fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=.01, facecolor="white", **kw)
    plt.close(fig)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    words, agree = shared_words()
    words.to_csv(OUT.with_name("borough_shared_words.csv"), index=False)
    agree.to_csv(OUT.with_name("borough_shared_agreement.csv"), index=False)
    figure(words)
    summary = agree.groupby("band").rho.agg(["min", "median", "max"]).round(3)
    print(summary.to_string(), flush=True)
    OUT.with_suffix(".json").write_text(json.dumps({
        "plot-title": "Borough words that three models share",
        "plot-caption": ("The top words of contrast C3 at the cool token, by relative depth, for Gemma-4-31B, Qwen3.5-9B "
                         "and Qwen3.6-27B. C3 is a Welch t over NTAs of one borough against the rest. Words are ranked by "
                         "their lowest z-score over the three models, thus a word ranks high only when all three models "
                         "raise it."),
        "plot-latex-label": "fig:borough-layers", "plot-tags": ["F10.30", "C3", *MODELS], "version": __version__,
        "rho_between_models_by_band": summary.reset_index().to_dict(orient="records")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
