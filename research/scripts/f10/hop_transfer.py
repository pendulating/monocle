#!/usr/bin/env python
"""F10.29, item 2: the shared aesthetic direction of 2 models reads out the third. CPU only.

The F10.27 direction of a model is a vector over its lens words (`research/figures/f10/shared/{model}_place/direction.npz`).
The 2 Qwen models share a lens vocab, and Gemma-4-31B shares 36,490 words with them. The script takes the
direction of 2 source models on the common words, averages the 2 unit vectors, and scores the NTA mean readouts
of the third model at its noun token. The test is r over the 197 NTAs with the margin of each of the 14 prompts
of the third model, at its own F10.27 test layer (chosen on its own fit prompts, with no use of the sources).
Bonferroni over the 14 prompts. The ceiling is the leave-1-prompt-out r of the model's own direction (F10.27, `main`).

Variants of the source direction: `test_layer` (the direction at the source's test layer) and `layer_mean`
(the mean over layers of the z-scored direction, as the F10.28 words use).

Output: `research/figures/f10/transfer/transfer.csv` (all layers), `transfer_test_layer.csv`, `sources.csv` (1 source at a time).

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_transfer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
from research.scripts.f10.hop_prompts import ADJECTIVES, MODELS, NOUNS, store_name  # noqa: E402

__version__ = "1.0.0"
DATA = ROOT / "data/f10/prompts"
SHARED = ROOT / "research/figures/f10/shared"
OUT = ROOT / "research/figures/f10/transfer"
PROMPTS = [(a, n) for n in NOUNS for a in ADJECTIVES]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def direction(model: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Give the vocab and the 2 variants of the direction of a model, each as a unit vector over its vocab."""
    d = np.load(SHARED / f"{model}_place/direction.npz", allow_pickle=True)
    s = json.loads((SHARED / f"{model}_place/summary.json").read_text())
    layers, dirs, vocab = d["layers"].tolist(), d["direction"].astype(np.float64), d["vocab"].astype(str)
    z = (dirs - dirs.mean(1, keepdims=True)) / dirs.std(1, keepdims=True)
    return vocab, s["test_layer"], {"test_layer": unit(dirs[layers.index(s["test_layer"])]), "layer_mean": unit(z.mean(0))}


def target(model: str) -> tuple[np.ndarray, list[int], np.ndarray, dict[tuple, np.ndarray]]:
    """Give the NTA mean readouts at the noun token (NTA, layer, word), the layers, the vocab, and the 14 NTA mean margins."""
    base = DATA / store_name(model, "cool", "place")
    x = np.load(base / "nta_place.npy").astype(np.float64)
    meta = json.loads((base / "meta.json").read_text())
    vocab = pd.read_parquet(ROOT / f"data/f10/hop_all/{store_name(model, 'cool', 'place')}/vocab.parquet")["token"].str.strip().to_numpy().astype(str)
    fr = pd.read_parquet(base / "frame.parquet")
    order = pd.factorize(fr["nta2020"])[1].tolist()
    margins = {}
    for a, n in PROMPTS:
        f = pd.read_parquet(DATA / store_name(model, a, n) / "frame.parquet")
        if pd.factorize(f["nta2020"])[1].tolist() != order:
            raise ValueError(f"NTA order differs in {model} {a} {n}")
        margins[(a, n)] = f.groupby("nta2020")["margin"].mean().reindex(order).to_numpy(float)
    return x - x.mean(0, keepdims=True), meta["layers"], vocab, margins


def r_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    res = stats.pearsonr(a[ok], b[ok])
    return float(res.statistic), float(res.pvalue), int(ok.sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dirs = {m: direction(m) for m in MODELS}
    ceiling = {}
    for m in MODELS:
        h = pd.read_csv(SHARED / f"{m}_place/heldout.csv")
        h = h[(h.variant == "main") & (h.layer == dirs[m][1])]
        ceiling[m] = dict(zip(h.prompt, h.r))
    rows = []
    for t in MODELS:
        x, layers, vocab_t, margins = target(t)
        sources = [m for m in MODELS if m != t]
        common = sorted(set(vocab_t) & set.intersection(*(set(dirs[m][0]) for m in sources)))
        idx_t = pd.Series(np.arange(len(vocab_t)), index=vocab_t).reindex(common).to_numpy()
        for variant in ("test_layer", "layer_mean"):
            vec = {}
            for m in sources:
                v, _, d = dirs[m]
                vec[m] = unit(d[variant][pd.Series(np.arange(len(v)), index=v).reindex(common).to_numpy()])
            vec["both"] = unit(vec[sources[0]] + vec[sources[1]])
            for name, u in vec.items():
                for b, layer in enumerate(layers):
                    score = x[:, b, idx_t] @ u
                    for p in PROMPTS:
                        r, pv, n = r_test(score, margins[p])
                        rows.append({"target": t, "source": name if name == "both" else name, "variant": variant, "layer": layer, "prompt": f"{p[1]}-{p[0]}",
                                     "adjective": p[0], "noun": p[1], "r": r, "p": pv, "n_nta": n, "n_words": len(common),
                                     "test_layer": layer == dirs[t][1], "ceiling_r": ceiling[t].get(f"{p[1]}-{p[0]}", np.nan)})
        print(f"[transfer] {t}: {len(common)} common words, test layer L{dirs[t][1]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "transfer.csv", index=False)
    at = df[df.test_layer].copy()
    at["p_bonferroni"] = (at.p * len(PROMPTS)).clip(upper=1.0)
    at.to_csv(OUT / "transfer_test_layer.csv", index=False)
    both = at[at.source == "both"]
    pd.set_option("display.width", 250)
    print(both.pivot_table(index=["target", "prompt"], columns="variant", values="r").round(2).join(both[both.variant == "test_layer"].set_index(["target", "prompt"])["ceiling_r"].round(2)).to_string())
    print("\nBonferroni passes at the test layer (of 14), source = both:")
    print(both.groupby(["target", "variant"]).apply(lambda g: int((g.p_bonferroni < .05).sum()), include_groups=False).to_string())
    one = at[at.source != "both"]
    print("\nmean r over the 8 aesthetic prompts (cool, nice, fancy, pretty x 2 nouns), 1 source at a time:")
    aes = one[one.adjective.isin(["cool", "nice", "fancy", "pretty"])]
    print(aes.groupby(["target", "source", "variant"])["r"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
