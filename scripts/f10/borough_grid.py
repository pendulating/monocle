"""F10.22: the words of each borough at the same token, through the layers.

A contrast is the two-sample t of 1 borough against the other 4, on NTA means. We repeat
this in 40 random splits of the NTAs, inside each borough.

| Column of `figures/f10/borough/borough_grid.csv` | Content |
|--------------------------------------------------|---------|
| `stability` | Mean r over 40 splits of the t vectors of the 2 halves |
| `stability_sd` | SD of that r over the splits |
| `n_stable` | Words with t above 0 in both halves of at least 38 of 40 splits, and full t of at least T_MIN |
| `words` | The stable words, by full t |

Use:
    python scripts/f10/borough_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10")]
from analyze_logit import load, shard_reads  # noqa: E402
from hop_borough import BOROUGHS, split_within, two_sample_t  # noqa: E402

OUT = ROOT / "figures/f10/borough/borough_grid.csv"
MODEL = "gemma-4-31b"
TOKENS = (" place", " cool", "?")
LAYERS = (8, 16, 24, 32, 40, 48, 56)
SPLITS, NEED, T_MIN = 40, 38, 3.0
SKIP = {"own", "and", "que", "est", "the", "not", "por", "para"}


def main() -> None:
    d = load(MODEL)
    im, head = d["images"], d["head"]
    tail = d["store"].manifest["tail_tokens"]
    words = d["store"].vocab["token"].str.strip().to_numpy()
    codes = pd.factorize(im["nta2020"])[0]
    borough = im.groupby(codes)["borough"].first().to_numpy()
    groups = codes.max() + 1
    at = [tail.index(t) for t in TOKENS]
    bs = [head["layers"].index(v) for v in LAYERS]
    m = np.zeros((groups, len(at), len(bs), len(words)), np.float32)
    start = 0
    for f in shard_reads(d):
        x = np.load(f, mmap_mode="r")
        rows = np.arange(start, start + len(x))
        for i, a in enumerate(at):
            for j, b in enumerate(bs):
                z = np.asarray(x[:, a, b], dtype=np.float32)
                z = (z - z.mean(-1, keepdims=True)) / z.std(-1, keepdims=True)
                for g in np.unique(codes[rows]):
                    m[g, i, j] += z[codes[rows] == g].sum(0)
        start += len(x)
        print(f"[f10.22] {f.name}", flush=True)
    m /= np.bincount(codes)[:, None, None, None]
    keep = ~np.isin(words, list(SKIP))
    rows = []
    for name in BOROUGHS:
        mask = borough == name
        for i, token in enumerate(TOKENS):
            for j, layer in enumerate(LAYERS):
                cell = m[:, i, j]
                full = two_sample_t(cell, mask)
                hits, r = np.zeros(len(words), int), []
                for s in range(SPLITS):
                    h = split_within(borough, s)
                    ta, tb = two_sample_t(cell[h], mask[h]), two_sample_t(cell[~h], mask[~h])
                    hits += (ta > 0) & (tb > 0)
                    r.append(np.corrcoef(ta, tb)[0, 1])
                ok = (hits >= NEED) & (full >= T_MIN) & keep
                order = np.flatnonzero(ok)[np.argsort(-full[ok])]
                rows.append({"model": MODEL, "borough": name, "token": token, "layer": layer, "stability": np.mean(r), "stability_sd": np.std(r), "n_stable": int(ok.sum()),
                             "max_t": float(full.max()), "words": ", ".join(words[order[:12]])})
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"[f10.22] {name}", flush=True)


if __name__ == "__main__":
    main()
