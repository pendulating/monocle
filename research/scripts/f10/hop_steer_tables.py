#!/usr/bin/env python
"""F10.29, items 5 and 7: tables from the steering outputs of `hop_steer.py`. CPU only.

Output in `research/figures/f10/steer/`:
- `run.csv`: for each model, prompt, direction, where and alpha (the 197-image run): the mean change of the margin
  against alpha 0, its sd, the yes share, the share of answered images (first token yes or no) and the top first token.
- `dose.csv`: the same for the 40-image dose run, by layer.
- `flip.csv`: the dose at which the yes share crosses .5 from each side, by model, prompt, direction and layer.
- `generations.csv`: the free text of 10 images at alpha -6, 0, 6.
- `words_model.csv`: the margin and the last-token words of the text-only prompts with the direction added.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_steer_tables.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/f10/steer"
OUT = ROOT / "research/figures/f10/steer"
MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
__version__ = "1.0.0"


def summary(d: pd.DataFrame) -> pd.DataFrame:
    base = d[d.direction == "none"].set_index(["model", "prompt", "image_idx"])["margin"]
    d = d.copy()
    d["d_margin"] = d.margin - base.reindex(list(zip(d.model, d.prompt, d.image_idx))).to_numpy()
    d["yes"] = d.margin > 0
    d["answered"] = d.top.str.strip().str.lower().isin(["yes", "no"])
    keys = ["model", "prompt", "direction", "where", "layer", "alpha"]
    g = d.groupby(keys, dropna=False).agg(n=("margin", "size"), mean_margin=("margin", "mean"), mean_d_margin=("d_margin", "mean"), sd_d_margin=("d_margin", "std"),
                                           yes_share=("yes", "mean"), answered=("answered", "mean"), top=("top", lambda s: s.str.strip().value_counts().index[0])).reset_index()
    return g


def load(pattern: str) -> pd.DataFrame:
    parts = []
    for m in MODELS:
        for f in sorted(glob.glob(str(DATA / m / pattern))):
            x = pd.read_parquet(f)
            x.insert(0, "model", m)
            parts.append(x)
    return pd.concat(parts) if parts else pd.DataFrame()


def flips(g: pd.DataFrame) -> pd.DataFrame:
    """Give, for each model, prompt, direction and layer, the smallest abs alpha at which the yes share is at least .9 (+) or at most .1 (-)."""
    rows = []
    for (m, p, dr, l), x in g[g.direction != "none"].groupby(["model", "prompt", "direction", "layer"]):
        base = g[(g.model == m) & (g.prompt == p) & (g.direction == "none")].yes_share.iloc[0]
        up = x[(x.alpha > 0) & (x.yes_share >= 0.9)].alpha
        down = x[(x.alpha < 0) & (x.yes_share <= 0.1)].alpha
        rows.append({"model": m, "prompt": p, "direction": dr, "layer": l, "base_yes_share": base, "alpha_to_yes_90": float(up.min()) if len(up) else np.nan,
                     "alpha_to_no_90": float(down.max()) if len(down) else np.nan, "answered_min": float(x.answered.min())})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 300)
    run = load("margins-place-*.parquet")
    if len(run):
        g = summary(run)
        g.to_csv(OUT / "run.csv", index=False)
        print(g[g.alpha.isin([-6.0, 0.0, 6.0])].round(3).to_string(index=False))
    dose = load("dose-place-*.parquet")
    if len(dose):
        g = summary(dose)
        g.to_csv(OUT / "dose.csv", index=False)
        f = flips(g)
        f.to_csv(OUT / "flip.csv", index=False)
        print("\n", f.round(2).to_string(index=False))
    gen = load("generations.parquet")
    if len(gen):
        gen.to_csv(OUT / "generations.csv", index=False)
    wm = load("words_model.parquet")
    if len(wm):
        wm.to_csv(OUT / "words_model.csv", index=False)


if __name__ == "__main__":
    main()
