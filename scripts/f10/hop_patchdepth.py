#!/usr/bin/env python
"""F10.9c: at which depth do the question-free patches hold each judgment? CPU only.

The mean patch of the image is the best scene block (`hop_scene.py`). This script gives the
out-of-fold R2 of the mean patch at each layer alone, for the cool margin and for each
adjective of F10.14. Patches see the image and no question. A deep patch can still hold
an appraisal that the model builds with no question.

Output: `figures/f10/free/patch_depth.csv`, `patch_words.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_patchdepth.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from analyze_logit import MODELS, SEED, load  # noqa: E402
from analyze_probe import GROUP_OF, margins  # noqa: E402
from hop_free import REPEATS, folds, kernel, oof_predict, r2  # noqa: E402

__version__ = "1.0.0"
OUT = ROOT / "figures/f10/free"


def main() -> None:
    depth, words = [], []
    for model in MODELS:
        d = load(model)
        im = d["images"]
        groups = im["nta2020"].to_numpy()
        reps = [folds(groups, seed=SEED + i) for i in range(REPEATS)]
        mean_patch = np.concatenate([np.asarray(np.load(f, mmap_mode="r"), dtype=np.float32).mean(1) for f in sorted((d["store"].path / "patches").glob("residuals-*.npy"))])
        m = margins(model).reindex(im["image_idx"])
        layers = d["head"]["layers"]
        kernels = [kernel(mean_patch[:, b].astype(np.float64)) for b in range(len(layers))]
        fit = lambda K, y: r2(y, np.mean([oof_predict(K, y, s) for s in reps], axis=0), reps[0])
        y = im["margin"].to_numpy(dtype=np.float64)
        for b, layer in enumerate(layers):
            depth.append({"model": model, "layer": layer, "r2": fit(kernels[b], y)})
        all_k = sum(kernels) / len(kernels)
        for w in m.columns:
            v = m[w].to_numpy(dtype=np.float64)
            r_each = [fit(K, v) for K in kernels]
            words.append({"model": model, "word": w, "group": GROUP_OF[w], "sd": v.std(), "r2_all": fit(all_k, v), "r2_best_layer": max(r_each), "best_layer": layers[int(np.argmax(r_each))]})
        print(f"[f10.9c] {model} finished", flush=True)
    pd.DataFrame(depth).to_csv(OUT / "patch_depth.csv", index=False)
    pd.DataFrame(words).to_csv(OUT / "patch_words.csv", index=False)


if __name__ == "__main__":
    main()
