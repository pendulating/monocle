#!/usr/bin/env python
"""F10.9b: how much of the margin the question-free patches hold, at finer grids. CPU only.

The scene test of `hop_free.py` pools the patches to a 4 x 4 grid. A gain beyond that block
can come from the pool and not from the question. This script gives the scene block more
room, and the 2 question-free reads are the check.

| Block | Features |
|-------|----------|
| `pool1` | the mean of all patches at each layer |
| `grid4`, `grid8` | a spatial grid at each layer |
| `full` | all patches at each layer, with no pool |
| `prefix` | the image-end token, which sees the image and no question |
| `grid8_rbf` | `grid8` with an RBF kernel at the median distance |

For each block: out-of-fold R2 alone, and the gain of 6 cells over the block. The cells are
the tokens around the question at 2 depths.

Output: `research/figures/f10/free/scene_blocks.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_scene.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
from research.scripts.f10.analyze_logit import MODELS, SEED, holm, load  # noqa: E402
from research.scripts.f10.hop_free import REPEATS, folds, grid_pool, kernel, load_tail, oof_mixed, oof_predict, r2  # noqa: E402

__version__ = "1.0.0"
OUT = ROOT / "research/figures/f10/free"
TOKENS = (0, " this", " place", " cool", "?", " yes", -1)


def blocks(d: dict) -> dict[str, np.ndarray]:
    files = sorted((d["store"].path / "patches").glob("residuals-*.npy"))
    side = int(round(np.sqrt(d["store"].manifest["patches"])))
    shards = {k: [] for k in ("pool1", "grid4", "grid8")}
    layers = []
    for f in files:
        p = np.load(f, mmap_mode="r")
        arr = np.asarray(p, dtype=np.float32)
        shards["pool1"].append(arr.mean(1, keepdims=True).transpose(0, 2, 1, 3))
        shards["grid4"].append(grid_pool(arr, 4))
        shards["grid8"].append(grid_pool(arr, min(8, side)))
        layers.append(arr.transpose(0, 2, 1, 3))
        print(f"[f10.9b] {f.name}", flush=True)
    full = np.concatenate(layers)
    del layers
    got = {k: np.concatenate(v) for k, v in shards.items()}
    n, nl = full.shape[:2]
    out = {k: sum(kernel(v[:, b].reshape(n, -1)) for b in range(nl)) / nl for k, v in got.items()}
    out["full"] = sum(kernel(full[:, b].reshape(n, -1)) for b in range(nl)) / nl
    x = got["grid8"].reshape(n, nl, -1).reshape(n, -1)
    x = (x - x.mean(0)).astype(np.float32)
    g = x @ x.T
    q = np.diag(g)[:, None] + np.diag(g)[None, :] - 2 * g
    out["grid8_rbf"] = np.exp(-q / np.median(q[np.triu_indices(n, 1)]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        d = load(model)
        im = d["images"]
        y = im["margin"].to_numpy(dtype=np.float64)
        groups = im["nta2020"].to_numpy()
        codes = pd.factorize(groups)[0]
        reps = [folds(groups, seed=SEED + i) for i in range(REPEATS)]
        mean_of = lambda fn: np.mean([fn(s) for s in reps], axis=0)
        x = load_tail(d)
        tokens = d["store"].manifest["tail_tokens"]
        layers = d["head"]["layers"]
        kern = blocks(d)
        kern["prefix"] = sum(kernel(x[:, 0, b].astype(np.float64)) for b in range(len(layers))) / len(layers)
        for name, K in kern.items():
            base = mean_of(lambda s: oof_predict(K, y, s))
            rows.append({"model": model, "block": name, "cell": "alone", "r2": r2(y, base, reps[0])})
            print(f"[f10.9b] {model} {name}: R2 {rows[-1]['r2']:.3f}", flush=True)
            for t in TOKENS:
                a = t if isinstance(t, int) and t >= 0 else (len(tokens) - 1 if t == -1 else tokens.index(t))
                for b in (len(layers) // 2, len(layers) - 2):
                    C = kernel(x[:, a, b].astype(np.float64))
                    joint = mean_of(lambda s: oof_mixed(K, C, y, s))
                    gap = np.bincount(codes, (y - base) ** 2 - (y - joint) ** 2) / np.bincount(codes)
                    tv, p = stats.ttest_1samp(gap, 0.0)
                    rows.append({"model": model, "block": name, "cell": f"{tokens[a]}|L{layers[b]}", "r2": r2(y, joint, reps[0]), "gain": r2(y, joint, reps[0]) - r2(y, base, reps[0]), "t": tv, "p": p})
        pd.DataFrame(rows).to_csv(OUT / "scene_blocks.csv", index=False)
    t = pd.DataFrame(rows)
    t.loc[t["p"].notna(), "p_holm"] = holm(t.loc[t["p"].notna(), "p"].to_numpy())
    t.to_csv(OUT / "scene_blocks.csv", index=False)
    print("[f10.9b] finished", flush=True)


if __name__ == "__main__":
    main()
