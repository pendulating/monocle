"""F10.32: the words of the mean readout along the prompt that all 3 models share.

For the `cool` prompt ("Is this place cool?"), at the tokens `place`, `?` and the last token, and
in each quarter of relative depth. Two scores for each word:

1. mean: the mean readout over the 591 images.
2. image: the mean readout minus the no-image readout. Thus a word scores high when the image
   raises it.

The 3 vocabularies are joined on the word string (lowercase words of 4 or more letters). In each
model, the score is z-scored over these words. A word ranks by its lowest z over the 3 models
(the rule of F10.30). The CSV also gives the rank of the word in the no-image readout of each
model (1 = top), at the same token and band.

Use:
    python research/scripts/f10/prompt_words_shared.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "1.0.0"
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/f10/prompts"
OUT = ROOT / "research/figures/f10/mechanism/prompt_shared_words.csv"
MODELS = {"gemma-4-31b": 60, "qwen3.5-9b": 32, "qwen3.6-27b": 64}  # number of layers
POSITIONS = {"place": "place", "after_cool": "?", "final": "last"}
BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))
WORD = re.compile(r"[a-z]{4,}")
N_CSV = 25


def load(model: str) -> tuple[dict, pd.Series]:
    """Give the band means of the 2 scores and of the no-image readout, for 1 model."""
    meta = json.loads((DATA / model / "meta.json").read_text())
    layers = np.array(meta["layers"]) / MODELS[model]
    full = np.load(DATA / model / "full_mean.npy", mmap_mode="r")
    text = np.load(DATA / model / "text_reads.npy", mmap_mode="r")
    vocab = pd.read_parquet(ROOT / f"data/f10/hop_all/{model}/vocab.parquet")["token"].str.strip().to_numpy().astype(str)
    s = pd.Series(np.arange(len(vocab)), index=vocab)
    out = {}
    for pos in POSITIONS:
        p = meta["positions"].index(pos)
        for k, (lo, hi) in enumerate(BANDS):
            li = np.flatnonzero((layers >= lo) & (layers < hi))
            f = np.asarray(full[p, li], dtype=np.float64).mean(0)
            t = np.asarray(text[p, li], dtype=np.float64).mean(0)
            out[(pos, k, "mean")] = f
            out[(pos, k, "image")] = f - t
            out[(pos, k, "noimage")] = t
    return out, s[~s.index.duplicated()]


def main() -> None:
    vals, index = {}, {}
    for m in MODELS:
        vals[m], index[m] = load(m)
        print(f"[f10.32] {m} done", flush=True)
    common = [w for w in sorted(set.intersection(*(set(index[m].index) for m in MODELS))) if WORD.fullmatch(w)]
    print(f"[f10.32] {len(common)} shared words", flush=True)
    rows = []
    for pos, name in POSITIONS.items():
        for k, (lo, hi) in enumerate(BANDS):
            # The rank of each word in the no-image readout of each model, over the full vocabulary.
            ranks = {}
            for m in MODELS:
                t = vals[m][(pos, k, "noimage")]
                r = np.empty(len(t), dtype=np.int64)
                r[np.argsort(-t)] = np.arange(1, len(t) + 1)
                ranks[m] = r[index[m][common].to_numpy()]
            for score in ("mean", "image"):
                T = np.stack([vals[m][(pos, k, score)][index[m][common].to_numpy()] for m in MODELS])
                Z = (T - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)
                low = Z.min(0)
                for rank, i in enumerate(np.argsort(-low)[:N_CSV]):
                    rows.append({"token": name, "band": k, "depth": f"{lo:.2f}-{min(hi, 1):.2f}", "score": score,
                                 "rank": rank + 1, "word": common[i], "z_min": round(float(low[i]), 3),
                                 **{f"noimage_rank_{m}": int(ranks[m][i]) for m in MODELS}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[f10.32] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
