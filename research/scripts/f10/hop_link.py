#!/usr/bin/env python
"""F10.12: does a read hold the words of the later argument of the same image? CPU only.

The reads come from the one-word prompt with no thinking. The arguments come from separate
runs of the same image, with thinking on. Thus a link is not a copy: the read comes from
another pass of the model.

For a cell, centre each word across the images. Then the score of image i is the mean
centred read of i over the words of the argument of i. The words of 1 list are:

| List | Words |
|------|-------|
| `text` | all store words in the traces and the answers of the image |
| `evidence` | store words in `visual_evidence` spans |
| `inference` | store words in `inference` spans |
| `criterion` | store words in `criterion` spans |
| `observation` | store words in the `evidence` attribute of `criterion` spans |

A word enters when it occurs in 1% to 50% of the images. Each image uses all 6 answers.

| Column | Meaning |
|--------|---------|
| `score` | NTA mean of the score above. The null is 0: the read of an average image |
| `within` | the score minus the mean score of the other images of the same borough, on the same words |

Test: one-sample t on NTA means, df 196, Holm over the cells of 1 model and 1 list.

The image holds the words of both the read and the argument. A link thus shows that the 2
share content, and not that the read causes the argument.

Output: `research/figures/f10/link/link_cells.csv`, `link_{list}.png`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_link.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10"), str(ROOT / "research/notebooks/cvpr")]
from analyze_logit import MODELS, NAMES, holm, load, plot_grid, shard_reads  # noqa: E402

__version__ = "1.0.0"

REASON = ROOT / "data/f10/hop_reason"
OUT = ROOT / "research/figures/f10/link"
WORD = re.compile(r"[a-zà-ÿ][a-zà-ÿ'-]+")


def tokens(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def presence(rows: pd.DataFrame, images: pd.Series, words: np.ndarray) -> np.ndarray:
    """Give [images, words] True where the word occurs in the text of the image, in 1% to 50% of the images."""
    at = {w: i for i, w in enumerate(words)}
    pos = {v: i for i, v in enumerate(images)}
    p = np.zeros((len(images), len(words)), bool)
    for idx, text in zip(rows["image_idx"], rows["text"]):
        for w in tokens(text):
            if w in at:
                p[pos[idx], at[w]] = True
    share = p.mean(0)
    p[:, (share < 0.01) | (share > 0.5)] = False
    return p


def link_scores(x: np.ndarray, p: np.ndarray, borough: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give the score of each image and the score minus the mean of the other images of the borough."""
    c = x - x.mean(0, keepdims=True)
    w = p / p.sum(1, keepdims=True).clip(min=1)
    s = c @ w.T
    same = borough[:, None] == borough[None, :]
    np.fill_diagonal(same, False)
    return np.diag(s).copy(), np.diag(s) - (s * same).sum(0) / same.sum(0)


def text_lists(model: str) -> dict[str, pd.DataFrame]:
    t = pd.read_parquet(REASON / "traces.parquet")
    t = t[t["model"] == model]
    both = pd.concat([t[["image_idx", "reasoning"]].rename(columns={"reasoning": "text"}), t[["image_idx", "response"]].rename(columns={"response": "text"})]).dropna()
    got = {"text": both}
    spans = REASON / "spans.parquet"
    if spans.exists():
        d = pd.read_parquet(spans)
        d = d[d["model"] == model]
        got["evidence"] = d[d["extraction_class"] == "visual_evidence"].rename(columns={"extraction_text": "text"})[["image_idx", "text"]]
        got["inference"] = d[d["extraction_class"] == "inference"].rename(columns={"extraction_text": "text"})[["image_idx", "text"]]
        c = d[d["extraction_class"] == "criterion"]
        got["criterion"] = c.rename(columns={"extraction_text": "text"})[["image_idx", "text"]]
        got["observation"] = pd.DataFrame({"image_idx": c["image_idx"], "text": [json.loads(a or "{}").get("evidence", "") for a in c["attributes_json"]]})
    return got


def analyze(model: str) -> pd.DataFrame:
    d = load(model)
    im, head = d["images"], d["head"]
    words = d["store"].vocab["token"].str.strip().to_numpy()
    tail = d["store"].manifest["tail_tokens"]
    codes = pd.factorize(im["nta2020"])[0]
    borough = im["borough"].to_numpy()
    x = np.concatenate([(lambda r: (r - r.mean(-1, keepdims=True)) / r.std(-1, keepdims=True))(np.load(f)) for f in shard_reads(d)])
    rows = []
    for name, part in text_lists(model).items():
        p = presence(part, im["image_idx"], words).astype(np.float32)
        print(f"[f10.12] {model}/{name}: {int(p.any(0).sum())} words, mean {p.sum(1).mean():.1f} for each image", flush=True)
        block = []
        for a, pos in enumerate(head["positions"]):
            for b, layer in enumerate(head["layers"]):
                s, w = link_scores(x[:, a, b], p, borough)
                nta, nta_w = np.bincount(codes, s) / np.bincount(codes), np.bincount(codes, w) / np.bincount(codes)
                t, q = stats.ttest_1samp(nta, 0.0)
                t2, q2 = stats.ttest_1samp(nta_w, 0.0)
                block.append({"model": model, "list": name, "position": pos, "token": tail[a], "index": a, "layer": layer, "score": nta.mean(), "t": t, "p": q,
                              "d": nta.mean() / nta.std(ddof=1), "within": nta_w.mean(), "t_within": t2, "p_within": q2})
        block = pd.DataFrame(block)
        block["p_holm"], block["p_within_holm"] = holm(block["p"].to_numpy()), holm(block["p_within"].to_numpy())
        rows.append(block)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    got = pd.concat([analyze(m) for m in args.models], ignore_index=True)
    got.to_csv(OUT / "link_cells.csv", index=False)
    for name in got["list"].unique():
        part = got[got["list"] == name]
        plot_grid({NAMES[m]: part[part["model"] == m] for m in args.models}, "within", OUT / f"link_{name}.png", f"Reads and the later argument: {name}",
                  f"Mean centred read over the {name} words of the argument of the same image, minus the same value for the other images of the borough, at each token and layer. "
                  "The arguments come from separate runs with thinking on.", ["F10.12", "arguments"], 0.0, float(part["within"].max()), "image-specific link", False)
    print("[f10.12] finished", flush=True)


if __name__ == "__main__":
    main()
