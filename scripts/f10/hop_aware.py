#!/usr/bin/env python
"""F10.29, item 4: does the model show that it knows the task is an aesthetic judgment? CPU only.

Gemma-4-31B gives no yes or no on `ugly` (571 and 508 images), `nice` (295, 58) and `pretty` (125, 2). The first
token is then `Subject` (Subjective), `Beauty`, `Unable`, `I` or `Whether`. The 2 Qwen models always answer.

The script measures, in the NTA mean readouts at the last token (F10.26 summaries):
1. The r over NTAs of each lens word with the NTA share of `unable` answers, at the last saved layer.
   The 20 words with the largest r name what the readout says where the model refuses.
2. The readout of a fixed list of refusal words (`subjective`, `opinion`, `personal`, `depends`, `matter`, `taste`,
   `unable`, `cannot`, `beauty`) by layer: r with the unable share, for the cells with 5% or more `unable`.
3. For each of the 42 cells: the share of `unable` answers, overall and by borough.

Output: `figures/f10/aware/words.csv`, `refusal_words_by_layer.csv`, `unable_share.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_aware.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10")]
from hop_prompts import MODELS, PROMPTS, Summary, store_name  # noqa: E402
from hop_shared import corr_cols  # noqa: E402

__version__ = "1.0.0"
OUT = ROOT / "figures/f10/aware"
REFUSAL = ("subjective", "opinion", "personal", "depends", "matter", "taste", "unable", "cannot", "beauty")
TOP = 20


def vocab_of(model: str) -> np.ndarray:
    return pd.read_parquet(ROOT / f"data/f10/hop_all/{store_name(model, 'cool', 'place')}/vocab.parquet")["token"].str.strip().to_numpy().astype(str)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shares, words, by_layer = [], [], []
    for m in MODELS:
        vocab = vocab_of(m)
        low = pd.Series(vocab).str.lower()
        idx = {w: int(np.flatnonzero(low == w)[0]) for w in REFUSAL if (low == w).any()}
        for a, n in PROMPTS:
            S = Summary(m, a, n)
            f = S.frame
            un = (f.outcome == "unable").astype(float)
            row = {"model": m, "noun": n, "adjective": a, "n_unable": int(un.sum()), "share_unable": float(un.mean())}
            row |= {b: float(v) for b, v in un.groupby(f.borough).mean().items()}
            shares.append(row)
            if un.mean() < 0.05:
                continue
            order = pd.factorize(f["nta2020"])[1]
            y = un.groupby(f["nta2020"]).mean().reindex(order).to_numpy()
            x = S.nta("final")
            for b, layer in enumerate(S.layers):
                r = corr_cols(x[:, b, :].astype(np.float64), y)
                if layer == S.layers[-1]:
                    top = np.argsort(-r)[:TOP]
                    bottom = np.argsort(r)[:TOP]
                    words.append({"model": m, "noun": n, "adjective": a, "layer": layer, "n_nta": len(y), "share_unable": float(un.mean()),
                                  "up (with the unable share)": " ".join(f"{vocab[i]} {r[i]:.2f}" for i in top),
                                  "down (with the verdict)": " ".join(f"{vocab[i]} {r[i]:.2f}" for i in bottom)})
                by_layer.append({"model": m, "noun": n, "adjective": a, "layer": layer} | {w: float(r[i]) for w, i in idx.items()})
            print(f"[f10.29] {m} {n} {a}: unable {un.mean():.2f}", flush=True)
    pd.DataFrame(shares).to_csv(OUT / "unable_share.csv", index=False)
    pd.DataFrame(words).to_csv(OUT / "words.csv", index=False)
    pd.DataFrame(by_layer).to_csv(OUT / "refusal_words_by_layer.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 300)
    s = pd.DataFrame(shares)
    print(s[s.share_unable > 0].round(3).to_string(index=False))
    for r in words:
        print(f"\n== {r['model']} {r['noun']} {r['adjective']} L{r['layer']} (unable {r['share_unable']:.2f})\n up:   {r['up (with the unable share)']}\n down: {r['down (with the verdict)']}")
    bl = pd.DataFrame(by_layer)
    print("\nr of the refusal words with the unable share, at the last layer:")
    print(bl[bl.layer == bl.groupby("model").layer.transform("max")].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
