#!/usr/bin/env python
"""F10.15: the shared axes of the reads, and why a borough list can fail. CPU only.

A contrast of boroughs is unstable when 1 large axis runs through all images and a borough
sits near its middle: the sign of the axis then flips between 2 halves of the NTAs. This
script finds the axes and asks which borough lists survive when they go.

| Output in `figures/f10/axes/` | Question | Method |
|-------------------------------|----------|--------|
| `axes_cells.csv` | How much of the read variance sits on the first axes, and what they are | SVD of NTA means at each cell. Variance share, r with the margin, borough means |
| `axes_words.csv` | Which words load on each axis | Top and bottom words of 3 axes, at the cells that the paper reads |
| `axes_stability.csv` | Which boroughs keep a stable list | r of word t values between 2 halves of the NTAs, after k axes go. Each half fits its own axes |
| `axes_influence.csv` | Do a few images carry a raw list | The share of the raw contrast of the top 10 words that the top 3 images of the borough give |

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_axes.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from analyze_logit import MODELS, load, shard_reads  # noqa: E402
from hop_borough import BOROUGHS, balanced_means, read_all, split_within, two_sample_t  # noqa: E402

__version__ = "1.0.0"

OUT = ROOT / "figures/f10/axes"
KS = (0, 1, 3, 5)
WORD_CELLS = {" place": (24, 36, 44), " cool": (20, 24, 36), "?": (44,)}


def axes(m: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give the centred rows, the first k left axes scaled by their size, and the right axes. The eigenproblem of the small Gram matrix is fast."""
    c = m - m.mean(0, keepdims=True)
    w, u = np.linalg.eigh(c @ c.T)
    w, u = w[::-1][:k].clip(min=0), u[:, ::-1][:, :k]
    s = np.sqrt(w)
    return c, u * s, (u.T @ c) / s[:, None].clip(min=1e-9)


def stability(m: np.ndarray, members: dict[str, np.ndarray], half: np.ndarray) -> dict[str, dict[int, float]]:
    """Give r of word t values between 2 halves after k axes go. Each half fits its own axes."""
    a, ua, va = axes(m[half], max(KS))
    b, ub, vb = axes(m[~half], max(KS))
    out = {}
    for name, member in members.items():
        out[name] = {}
        for k in KS:
            ta = two_sample_t(a - ua[:, :k] @ va[:k], member[half])
            tb = two_sample_t(b - ub[:, :k] @ vb[:k], member[~half])
            out[name][k] = float(np.corrcoef(ta, tb)[0, 1])
    return out


def analyze(model: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = load(model)
    im, head = d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    words = d["store"].vocab["token"].str.strip().to_numpy()
    codes = pd.factorize(im["nta2020"])[0]
    borough = im.groupby(codes)["borough"].first().to_numpy()
    half = split_within(borough)
    members = {k: borough == k for k in BOROUGHS}
    x = read_all(d, shard_reads(d))
    m = balanced_means(x, codes)
    y = pd.Series(im["margin"].to_numpy()).groupby(codes).mean().to_numpy()
    cells, loads, stable, power = [], [], [], []
    for a, pos in enumerate(head["positions"]):
        for b, layer in enumerate(head["layers"]):
            c = m[:, a, b] - m[:, a, b].mean(0, keepdims=True)
            w, u = np.linalg.eigh(c @ c.T)
            w, u = w[::-1].clip(min=0), u[:, ::-1]
            s = np.sqrt(w)
            vt = (u[:, :3].T @ c) / s[:3, None]
            share = w / w.sum()
            row = {"model": model, "position": pos, "token": tokens[a], "index": a, "layer": layer}
            for i in range(3):
                z = u[:, i] / u[:, i].std()
                row |= {f"share{i + 1}": share[i], f"r_margin{i + 1}": np.corrcoef(z, y)[0, 1]} | {f"pc{i + 1}_{k}": z[borough == k].mean() for k in BOROUGHS}
            cells.append(row)
            if layer in WORD_CELLS.get(tokens[a], ()):
                for i in range(3):
                    order = np.argsort(vt[i])
                    loads.append({"model": model, "token": tokens[a], "layer": layer, "axis": i + 1, "share": share[i], "low": ", ".join(words[order[:14]]), "high": ", ".join(words[order[-14:]][::-1])})
            for name, v in stability(m[:, a, b], members, half).items():
                stable.append({"model": model, "borough": name, "token": tokens[a], "index": a, "layer": layer} | {f"k{k}": v[k] for k in KS})
        print(f"[f10.15] {model}: {a + 1}/{len(tokens)}", flush=True)
    for name in BOROUGHS:
        member = (im["borough"] == name).to_numpy()
        for a, b in [(tokens.index(t), head["layers"].index(v)) for t, ls in WORD_CELLS.items() for v in ls if v in head["layers"]]:
            cell = x[:, a, b]
            gap = cell[member].mean(0) - cell[~member].mean(0)
            top = np.argsort(-gap)[:10]
            each = cell[member][:, top].mean(1) - cell[~member][:, top].mean()
            order = np.argsort(-each)
            power.append({"model": model, "borough": name, "token": tokens[a], "layer": head["layers"][b], "top_words": ", ".join(words[top]), "share_top3": float(each[order[:3]].sum() / each.sum()),
                          "share_top10": float(each[order[:10]].sum() / each.sum()), "images": int(member.sum()), "top_images": ", ".join(map(str, im.loc[member, "image_idx"].to_numpy()[order[:3]]))})
    return pd.DataFrame(cells), pd.DataFrame(loads), pd.DataFrame(stable), pd.DataFrame(power)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    got = [analyze(m) for m in args.models]
    for i, name in enumerate(("cells", "words", "stability", "influence")):
        pd.concat([g[i] for g in got], ignore_index=True).to_csv(OUT / f"axes_{name}.csv", index=False)
    print("[f10.15] finished", flush=True)


if __name__ == "__main__":
    main()
