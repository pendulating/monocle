#!/usr/bin/env python
"""F10.17: which sense of "cool" do the reads hold, and is a borough list stable over many splits? CPU only.

The first axis of the reads at the word `cool` separates temperature words from social words
(F10.15). This script measures that axis with 2 fixed lists, which name the 2 dictionary
senses of the word. The lists share words with the axis. Thus the axis is the evidence, and
the score is a readable measure of it.

| Output in `figures/f10/sense/` | Question | Method |
|--------------------------------|----------|--------|
| `sense_cells.csv` | Where does the sense score vary, and does it follow the margin | At each cell: mean, the score with no image, r with the margin and with the `urban` probe, borough ANOVA on NTA means, r with axis 1 |
| `sense_images.csv` | Which images take which sense | Score at the word `cool`, margin and trace rates of each image |
| `sense_links.csv` | Does the argument use the same sense | NTA-clustered r between the score and the share of traces with temperature words or social words |
| `sense_splits.csv` | Is the stability of a borough a property of 1 split | r of word t values between 2 halves over 40 random splits, with k axes removed |

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_sense.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10"), str(ROOT / "notebooks/cvpr")]
from analyze_logit import MODELS, SEED, cluster_r, load, shard_reads  # noqa: E402
from analyze_probe import margins  # noqa: E402
from hop_axes import axes  # noqa: E402
from hop_borough import BOROUGHS, balanced_means, read_all, split_within, two_sample_t  # noqa: E402

OUT = ROOT / "figures/f10/sense"
CAP = 30.0
TEMPERATURE = ("temperature temperatures cold chilly cooling cooled cooler chilled breeze weather thermostat refrigerator humidity coolant climate").split()
SOCIAL = ("trendy hip hipster stylish fashionable edgy funky vintage retro boutique bohemian nightlife artsy awesome swagger").split()
KEY = {"gemma-4-31b": [(" place", 36), (" cool", 24), ("?", 44)], "qwen3.5-9b": [(" place", 16), (" cool", 24), ("?", 20)]}
COOL_LAYER = 24
TEXT_TEMPERATURE = re.compile(r"\b(temperature|temperatures|chilly|cold|breeze|breezy|refreshing)\b", re.I)
TEXT_SOCIAL = re.compile(r"\b(trendy|hip|hipster|stylish|fashionable|edgy|funky|artsy|vintage|retro)\b", re.I)
SPLITS = 40


def trace_rates(model: str) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data/f4_traces/extract_input/f4_cool_x2.parquet")
    d = d[(d["model"] == model) & (d["field"] == "reasoning")]
    return d.assign(temperature=[TEXT_TEMPERATURE.search(t) is not None for t in d["text"]], social=[TEXT_SOCIAL.search(t) is not None for t in d["text"]]).groupby("image_idx")[["temperature", "social"]].mean()


def analyze(model: str) -> dict[str, pd.DataFrame]:
    d = load(model)
    im, head = d["images"], d["head"]
    tokens = d["store"].manifest["tail_tokens"]
    words = d["store"].vocab["token"].str.strip().str.lower().to_numpy()
    hot = np.flatnonzero(np.isin(words, TEMPERATURE))
    soc = np.flatnonzero(np.isin(words, SOCIAL))
    print(f"[f10.17] {model}: temperature {sorted(words[hot])}; social {sorted(words[soc])}", flush=True)
    x = read_all(d, shard_reads(d))
    score = x[..., soc].mean(-1) - x[..., hot].mean(-1)
    text = np.load(d["store"].path / "text_reads.npy")
    if head["softcap"]:
        text = CAP * np.arctanh((text.astype(np.float64) / CAP).clip(-1 + 1e-7, 1 - 1e-7))
    text = (text - np.nanmean(text, -1, keepdims=True)) / np.nanstd(text, -1, keepdims=True)
    base = text[..., soc].mean(-1) - text[..., hot].mean(-1)
    codes = pd.factorize(im["nta2020"])[0]
    borough = im.groupby(codes)["borough"].first().to_numpy()
    cl, y = im["nta2020"].to_numpy(), im["margin"].to_numpy()
    probe = margins(model).reindex(im["image_idx"].to_numpy())
    m = balanced_means(x, codes)
    rows = []
    for a in range(len(tokens)):
        for b, layer in enumerate(head["layers"]):
            v = score[:, a, b]
            nta = np.bincount(codes, v) / np.bincount(codes)
            f, p = stats.f_oneway(*[nta[borough == k] for k in BOROUGHS])
            c = m[:, a, b] - m[:, a, b].mean(0, keepdims=True)
            w, u = np.linalg.eigh(c @ c.T)
            rows.append({"model": model, "token": tokens[a], "index": a, "layer": layer, "mean": v.mean(), "sd": v.std(), "no_image": base[a, b], "r_margin": cluster_r(v, y, cl)[0],
                         "r_urban": cluster_r(v, probe["urban"].to_numpy(), cl)[0], "F": f, "p": p, "r_axis1": np.corrcoef(nta, u[:, -1])[0, 1], "share1": w[-1] / w.sum()}
                        | {k: (nta[borough == k].mean() - nta.mean()) / nta.std() for k in BOROUGHS})
    a = tokens.index(" cool")
    b = head["layers"].index(COOL_LAYER)
    rates = trace_rates(model).reindex(im["image_idx"].to_numpy())
    table = pd.DataFrame({"model": model, "image_idx": im["image_idx"].to_numpy(), "borough": im["borough"].to_numpy(), "nta": cl, "score": score[:, a, b], "margin": y, "urban": probe["urban"].to_numpy(),
                          "cool_probe": probe["cool"].to_numpy(), "temperature": rates["temperature"].to_numpy(), "social": rates["social"].to_numpy()})
    links = []
    for name in ("temperature", "social"):
        keep = table[name].notna().to_numpy()
        r, t, p = cluster_r(table["score"].to_numpy()[keep], table[name].to_numpy()[keep], cl[keep])
        links.append({"model": model, "x": "score", "y": name, "r": r, "t": t, "p": p, "share": table[name].mean()})
    for name in ("margin", "urban"):
        r, t, p = cluster_r(table["score"].to_numpy(), table[name].to_numpy(), cl)
        links.append({"model": model, "x": "score", "y": name, "r": r, "t": t, "p": p, "share": np.nan})

    splits = []
    members = {k: borough == k for k in BOROUGHS}
    for token, layer in KEY[model]:
        a, b = tokens.index(token), head["layers"].index(layer)
        cell = m[:, a, b]
        for s in range(SPLITS):
            half = split_within(borough, SEED + s)
            ca, ua, va = axes(cell[half], 1)
            cb, ub, vb = axes(cell[~half], 1)
            for name, member in members.items():
                row = {"model": model, "token": token, "layer": layer, "split": s, "borough": name}
                row["k0"] = np.corrcoef(two_sample_t(ca, member[half]), two_sample_t(cb, member[~half]))[0, 1]
                row["k1"] = np.corrcoef(two_sample_t(ca - ua @ va, member[half]), two_sample_t(cb - ub @ vb, member[~half]))[0, 1]
                splits.append(row)
    return {"cells": pd.DataFrame(rows), "images": table, "links": pd.DataFrame(links), "splits": pd.DataFrame(splits)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    got = [analyze(m) for m in MODELS]
    for k in got[0]:
        pd.concat([g[k] for g in got], ignore_index=True).to_csv(OUT / f"sense_{k}.csv", index=False)
    print("[f10.17] finished", flush=True)


if __name__ == "__main__":
    main()
