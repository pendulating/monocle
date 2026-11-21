#!/usr/bin/env python
"""F10.13: the stages of the judgment, with general valence removed. CPU only.

2 questions decide how to read the word sets.

| Output in `figures/f10/stage/` | Question | Method |
|--------------------------------|----------|--------|
| `stage_partial.csv` | Does a property or a value move with the margin beyond general valence | Partial r of the set and the margin, given the valence pair at the same cell, and given the borough. NTA-clustered t, Holm over set and cell |
| `stage_onset.csv` | Where does each set first hold beyond valence | The first token, in prompt order, with a cell that passes Holm |
| `stage_words.csv` | Do the words of the paper come from the image or from the prompt | For each word of the mean read: rank in the mean read, in the text with no image, and in the mean read minus the text (`rank_added`), and the gap on NTA means, df 196, Holm |

The words of `stage_words.csv` come from a hand read of the mean read, thus the test asks
only if the image adds them, and not if they are present.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_stage.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from analyze_logit import MODELS, cluster_r, holm, load, shard_reads  # noqa: E402

__version__ = "1.0.0"

SETS = ROOT / "data/f10/hop_sets"
OUT = ROOT / "figures/f10/stage"
CAP = 30.0
# The token of each paper position, on each model. The last 3 tokens differ by name.
TOKENS = {"gemma-4-31b": {"place": " place", "cool": " cool", "?": "?", "model": "model", "channel": "<|channel>", "thought": "thought", "end": "<channel|>"},
          "qwen3.5-9b": {"place": " place", "cool": " cool", "?": "?", "model": "assistant", "channel": "<think>", "end": "</think>"}}
TOKENS["qwen3.6-27b"] = TOKENS["qwen3.5-9b"]
CLAIMS = [("place", (8, 12, 16, 20, 24), "metropolitan beachfront"), ("place", (24, 28, 32, 36, 40, 44, 48, 52, 56), "nearby relocated surroundings location"),
          ("cool", (0, 4, 8, 12, 16), "cookware kiln canyon"), ("cool", (16, 20, 24, 28, 32), "amazing awesome funky"), ("cool", (32, 36, 40, 44, 48, 52, 56), "vibe idea opinion"),
          ("?", (40, 44, 48, 52, 56), "questionable judging debatable subjective attractiveness"), ("model", (32,), "renovated fashionable decorated"), ("model", (40,), "ugly handsome favorite"),
          ("model", (44,), "inexpensive aesthetics"), ("model", (48,), "believable"), ("model", (52,), "authenticity austerity enjoyment"),
          ("channel", None, "older modernized deteriorating ordinary vacant humble quaint utilitarian european grungy cleanliness cheap neat sanitary austere stereotypical"),
          ("thought", None, "older modernized deteriorating ordinary vacant humble quaint utilitarian european grungy cleanliness cheap neat sanitary austere stereotypical"),
          ("end", (36, 40, 44, 48, 52, 56), "inconclusive cannot subjective idk jury")]


def residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Give the residual of y on x, with an intercept. x has 1 row for each image."""
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def partial_table(d: dict, model: str) -> pd.DataFrame:
    z = np.load(SETS / model / "scores.npz")
    names, score = z["names"].tolist(), z["score"]
    im, head = d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    y, cl = im["margin"].to_numpy(), im["nta2020"].to_numpy()
    dummies = pd.get_dummies(im["borough"]).to_numpy(float)
    valence = score[names.index("positive")] - score[names.index("negative")]
    rows = []
    for s, name in enumerate(names):
        if name in ("positive", "negative"):
            continue
        for a, pos in enumerate(head["positions"]):
            for b, layer in enumerate(head["layers"]):
                x = np.c_[dummies, stats.zscore(valence[:, a, b])]
                r, t, p = cluster_r(residual(score[s, :, a, b], x), residual(y, x), cl)
                rows.append({"model": model, "set": name, "position": pos, "token": tokens[a], "index": a, "layer": layer, "r_partial": r, "t": t, "p": p,
                             "r_raw": cluster_r(score[s, :, a, b], y, cl)[0], "r_valence": cluster_r(score[s, :, a, b], valence[:, a, b], cl)[0]})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].to_numpy())
    return t


def word_table(d: dict, model: str) -> pd.DataFrame:
    head, im = d["head"], d["images"]
    tail = d["store"].manifest["tail_tokens"]
    words = d["store"].vocab["token"].str.strip().to_numpy()
    at = {w: i for i, w in enumerate(words)}
    x = np.concatenate([(lambda r: (r - r.mean(-1, keepdims=True)) / r.std(-1, keepdims=True))(np.load(f)) for f in shard_reads(d)])
    text = np.load(d["store"].path / "text_reads.npy")
    if head["softcap"]:
        text = CAP * np.arctanh((text.astype(np.float64) / CAP).clip(-1 + 1e-7, 1 - 1e-7))
    text = (text - np.nanmean(text, -1, keepdims=True)) / np.nanstd(text, -1, keepdims=True)
    codes = pd.factorize(im["nta2020"])[0]
    rows = []
    for name, layers, listed in CLAIMS:
        if name not in TOKENS[model]:
            continue
        a = len(tail) - 1 - tail[::-1].index(TOKENS[model][name])
        use = [head["layers"].index(v) for v in layers or () if v in head["layers"]] or list(range(len(head["layers"])))
        mean = x[:, a].mean(0)
        added = mean - np.nan_to_num(text[a])
        for w in listed.split():
            if w not in at:
                rows.append({"model": model, "position": name, "word": w, "in_store": False})
                continue
            i = at[w]
            b = use[int(np.argmax(mean[use, i]))]
            gap = np.bincount(codes, x[:, a, b, i] - text[a, b, i]) / np.bincount(codes)
            t, p = stats.ttest_1samp(gap, 0.0)
            rows.append({"model": model, "position": name, "token": tail[a], "word": w, "in_store": True, "layer": head["layers"][b], "rank_image": int((mean[b] > mean[b, i]).sum()) + 1,
                         "rank_text": int((text[a, b] > text[a, b, i]).sum()) + 1, "rank_added": int((added[b] > added[b, i]).sum()) + 1, "read": float(mean[b, i]), "text": float(text[a, b, i]), "gap": float(gap.mean()), "t": t, "p": p})
    t = pd.DataFrame(rows)
    ok = t["in_store"]
    t.loc[ok, "p_holm"] = holm(t.loc[ok, "p"].to_numpy())
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--words-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    part, words = [], []
    for model in args.models:
        d = load(model)
        if not args.words_only:
            part.append(partial_table(d, model))
        words.append(word_table(d, model))
    if args.words_only:
        pd.concat(words, ignore_index=True).to_csv(OUT / "stage_words.csv", index=False)
        return
    part = pd.concat(part, ignore_index=True)
    part.to_csv(OUT / "stage_partial.csv", index=False)
    ok = part[part["p_holm"] < 0.05].sort_values("index")
    onset = ok.groupby(["model", "set"], as_index=False).first()[["model", "set", "token", "index", "layer", "r_partial"]]
    best = ok.loc[ok.groupby(["model", "set"])["r_partial"].apply(lambda v: v.abs().idxmax())][["model", "set", "token", "layer", "r_partial"]]
    onset.merge(best, on=["model", "set"], suffixes=("", "_best")).assign(cells=lambda t: t.set_index(["model", "set"]).index.map(ok.groupby(["model", "set"]).size())).to_csv(OUT / "stage_onset.csv", index=False)
    pd.concat(words, ignore_index=True).to_csv(OUT / "stage_words.csv", index=False)
    print("[f10.13] finished", flush=True)


if __name__ == "__main__":
    main()
