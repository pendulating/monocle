"""F10.31: each aesthetic standard in the J-space, for 3 models (Figure 2 of the paper).

One row for each of the 7 adjectives of F10.26, with the noun `place`. Three columns of words:

1. Conception (no image). The no-image readout at the adjective token, minus the mean no-image
   readout of the 7 adjectives. Thus the words are what the model holds about this standard
   before it sees an image, and not what all 7 prompts hold.
2. Toward yes, and 3. toward no. The readout at the `place` token, before the model reads the
   adjective. The 7 prompts have the same text up to this token, thus the same activations. Only
   the answer differs. For each word, r over the 197 NTAs of the NTA mean readout with the NTA
   mean expected answer E = P(yes) - P(no) of that prompt.

Each value is the mean over the saved layers at relative depth .25 to .75. The 3 vocabularies
are joined on the word string (lowercase words of 4 or more letters). In each model, the values
are z-scored over these words. The score of a word is its lowest z of the 3 models, thus a word
ranks high only when all 3 models rank it high (the rule of F10.30).

The figure leaves out the adjective and its forms (words that start with its first 4 letters).

Use:
    python research/scripts/f10/plot_standards.py
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
sys.path[:0] = [str(ROOT / "research/notebooks/cvpr"), str(ROOT / "research/scripts/f10")]
from hop_hierarchy import answer_store  # noqa: E402

DATA = ROOT / "data/f10/prompts"
OUT = ROOT / "research/figures/f10/mechanism/mech_standards"
MODELS = {"gemma-4-31b": 60, "qwen3.5-9b": 32, "qwen3.6-27b": 64}  # number of layers
ADJECTIVES = ("cool", "nice", "pretty", "fancy", "ordinary", "urban", "ugly")
YES = ["lp_Yes", "lp_yes", "lp_ Yes", "lp_ yes"]
NO = ["lp_No", "lp_no", "lp_ No", "lp_ no"]
DEPTH = (0.25, 0.75)
WORD = re.compile(r"[a-z]{4,}")
N_WORDS, N_CSV = 6, 20
COLUMNS = (("concept", 1, "Conception, no image", "at the adjective"),
           ("scene", 1, "Toward yes", "in the scene, at 'place'"),
           ("scene", -1, "Toward no", "in the scene, at 'place'"))


def store(model: str, adjective: str) -> str:
    return model if adjective == "cool" else f"{model}-{adjective}"


def model_values(model: str) -> tuple[dict, np.ndarray]:
    """Give the conception and the scene r of each adjective, over the full vocabulary of 1 model."""
    vocab = pd.read_parquet(ROOT / f"data/f10/hop_all/{model}/vocab.parquet")["token"].str.strip().to_numpy().astype(str)
    text, scene = {}, {}
    for a in ADJECTIVES:
        d = DATA / store(model, a)
        meta = json.loads((d / "meta.json").read_text())
        if not meta.get("complete"):
            raise ValueError(f"{d} is not complete")
        layers = np.array(meta["layers"])
        mid = np.flatnonzero((layers / MODELS[model] >= DEPTH[0]) & (layers / MODELS[model] < DEPTH[1]))
        text[a] = np.load(d / "text_reads.npy", mmap_mode="r")[meta["positions"].index("after_place")][mid].astype(np.float64)
        fr = pd.read_parquet(d / "frame.parquet")
        order = pd.factorize(fr["nta2020"])[1]
        ans = pd.read_parquet(ROOT / "data/f10/hop_logit" / answer_store(model, a, "place") / "answers.parquet")
        e_img = pd.Series(np.exp(ans[YES]).sum(1).to_numpy() - np.exp(ans[NO]).sum(1).to_numpy(), index=ans["image_idx"].to_numpy())
        e = fr.assign(E=fr["image_idx"].map(e_img)).groupby("nta2020")["E"].mean().reindex(order).to_numpy()
        if np.isnan(e).any():
            raise ValueError(f"{model} {a}: NTA without E")
        x = np.load(d / "nta_place.npy", mmap_mode="r")
        ec = e - e.mean()
        rs = []
        for li in mid:
            X = np.asarray(x[:, li], dtype=np.float64)
            X = X - X.mean(0)
            rs.append((ec @ X) / np.sqrt((ec ** 2).sum() * (X ** 2).sum(0) + 1e-30))
        scene[a] = np.mean(rs, 0)
    base = np.mean([text[a] for a in ADJECTIVES], 0)
    out = {}
    for a in ADJECTIVES:
        out[("concept", a)] = (text[a] - base).mean(0)
        out[("scene", a)] = scene[a]
    return out, vocab


def shared() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give the top words of each cell, and the Spearman rho of the full vectors between models."""
    vals, index = {}, {}
    for m in MODELS:
        vals[m], vocab = model_values(m)
        s = pd.Series(np.arange(len(vocab)), index=vocab)
        index[m] = s[~s.index.duplicated()]
        print(f"[f10.31] {m} done", flush=True)
    common = [w for w in sorted(set.intersection(*(set(index[m].index) for m in MODELS))) if WORD.fullmatch(w)]
    rows, agree = [], []
    for a in ADJECTIVES:
        for kind in ("concept", "scene"):
            T = np.stack([vals[m][(kind, a)][index[m][common].to_numpy()] for m in MODELS])
            for (i, p), (j, q) in combinations(enumerate(MODELS), 2):
                agree.append({"adjective": a, "kind": kind, "pair": f"{p}|{q}", "rho": float(stats.spearmanr(T[i], T[j]).statistic)})
            for sign in ((1,) if kind == "concept" else (1, -1)):
                Z = (sign * T - (sign * T).mean(1, keepdims=True)) / T.std(1, keepdims=True)
                low = Z.min(0)
                for rank, k in enumerate(np.argsort(-low)[:N_CSV]):
                    rows.append({"adjective": a, "kind": kind, "sign": sign, "rank": rank + 1, "word": common[k], "z_min": float(low[k]),
                                 **{f"value_{m}": float(T[j, k]) for j, m in enumerate(MODELS)}})
    print(f"[f10.31] {len(common)} shared words", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(agree)


def figure(words: pd.DataFrame, embedded: pd.DataFrame) -> None:
    import _style as S

    S.apply_house_style()
    width, left, row, head = 6.0, 0.95, 0.30, 0.34
    col = (width - left) / len(COLUMNS)
    ink = [S.EDGE, S.PAL["teal"], S.PAL["coral"]]
    height = head + row * len(ADJECTIVES)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    for c, (_, _, title, sub) in enumerate(COLUMNS):
        x = left + c * col + .04
        ax.text(x, head - .17, title, fontsize=7, fontweight="bold", color=ink[c], va="bottom")
        ax.text(x, head - .05, sub, fontsize=5.8, color="#777777", va="bottom")
    ax.text(0, head - .05, "embedded / 100", fontsize=5.8, color="#777777", va="bottom")
    ax.plot([0, width], [head, head], color=S.EDGE, lw=.5)
    y = head
    for a in ADJECTIVES:
        ax.text(0, y + row * .40, a, family="monospace", fontsize=7.5, fontweight="bold", va="center")
        e = embedded.loc[a]
        ax.text(0, y + row * .78, " · ".join(str(int(e[m])) for m in MODELS), fontsize=5.6, color="#777777", va="center")
        for c, (kind, sign, _, _) in enumerate(COLUMNS):
            w = words[(words.adjective == a) & (words.kind == kind) & (words.sign == sign)].sort_values("rank").word
            w = [x for x in w if not x.startswith(a[:4])][:N_WORDS]  # the adjective and its forms say nothing new
            half = (len(w) + 1) // 2
            ax.text(left + c * col + .04, y + row / 2, ", ".join(w[:half]) + ",\n" + ", ".join(w[half:]), fontsize=6.3,
                    color=ink[c], va="center", linespacing=1.15)
        y += row
        ax.plot([0, width], [y, y], color="#dddddd", lw=.3)
    for ext in ("png", "pdf"):
        kw = {"metadata": {"CreationDate": None}} if ext == "pdf" else {"dpi": 300}
        fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=.01, facecolor="white", **kw)
    plt.close(fig)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    words, agree = shared()
    words.to_csv(OUT.with_name("standards_words.csv"), index=False)
    agree.to_csv(OUT.with_name("standards_agreement.csv"), index=False)
    g = pd.read_csv(ROOT / "research/figures/f10/prompts/t2_grid.csv")
    embedded = g[g["noun"] == "place"].pivot(index="adjective", columns="model", values="embedded_mid")
    figure(words, embedded)
    summary = agree.groupby("kind").rho.agg(["min", "median", "max"]).round(3)
    print(summary.to_string(), flush=True)
    OUT.with_suffix(".json").write_text(json.dumps({
        "plot-title": "Each standard in the J-space",
        "plot-caption": ("For each adjective asked of a 'place': the words that the three models hold about it with no image "
                         "(the no-image readout at the adjective, minus the mean of the seven adjectives), and the words of the "
                         "scene that go with a yes or a no (r over 197 NTAs of the readout at 'place' with the expected answer). "
                         "Each word is ranked by its lowest z-score over Gemma-4-31B, Qwen3.5-9B and Qwen3.6-27B, at relative "
                         "depth .25 to .75. The adjective and its forms are left out. Embedded: the top-100 words at the adjective that are also in the no-image top 100."),
        "plot-latex-label": "fig:standards", "plot-tags": ["F10.31", *MODELS], "version": __version__,
        "rho_between_models": summary.reset_index().to_dict(orient="records")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
