#!/usr/bin/env python
"""F10.4: the words that the image adds at each text position (pilot, no preset test).

For each prompt and position, the script compares the mean read over the images with
the read of the same text with no image. It also compares Manhattan images with Bronx
images, and it gives the top words of the images with the highest and lowest `yes_no`.

| File in `research/figures/f10/hops/` | Content |
|-----------------------------|---------|
| `hops_added.csv` | the top words that the image adds, for each model, case, prompt and position |
| `hops_borough.csv` | the top words of Manhattan images minus Bronx images, and of the reverse |
| `hops_depth.csv` | the words that recur in the added top lists, at each layer |
| `hops_yes_no.csv` | the Pearson r of the `yes_no` value of the prompt with the presence of a word in the top lists |
| `hops_replication.csv` | the Manhattan - Bronx gap of the pilot against the held-out images, and a random split as a control |
| `hops_words.csv` | the words with the largest gap in both runs. A list selected on both runs is an illustration, not a test |
| `hops_layers.csv` | the gap at each layer: its replication, and the top words of the pilot with the share that keep their sign on new images |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/analyze_hops.py [--runs data/f10/hops]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

__version__ = "1.0.0"

MODELS = ("gemma-4-31b", "qwen3.5-9b")
DIRS = {"gemma-4-31b": "/share/ju/matt/zoo/models/Gemma-4-31B-it", "qwen3.5-9b": "/share/ju/matt/zoo/models/Qwen3.5-9B"}
CASES = ("restaurant", "street")
TOP = 15
OUT = Path("research/figures/f10/hops")


def is_word(w: str) -> bool:
    """Keep the English word tokens: a table of hops cannot use code tokens or fragments."""
    return bool(re.fullmatch(r" ?[A-Za-z]{3,}", w))


def top_words(x: torch.Tensor, tok, k: int = TOP) -> list[str]:
    out = []
    for i in x.topk(4000).indices.tolist():
        w = tok.decode([i])
        if is_word(w) and w.strip().lower() not in {o.lower() for o in out}:
            out.append(w.strip())
        if len(out) == k:
            break
    return out


def replication(pilot: Path, held: Path, tok, model: str) -> tuple[list, list]:
    """Compare the gaps of the pilot and the held-out images, on the lower-case English word tokens."""
    p, h = (torch.load(d / model / "means.pt", weights_only=False) for d in (pilot, held))
    n = next(iter(p.values())).numel()
    words = [tok.decode([i]) for i in range(n)]
    keep = np.array([i for i, w in enumerate(words) if re.fullmatch(r" [a-z]{3,}", w)])
    rows, lists = [], []
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    for k in sorted({k[:2] for k in p}):
        old = {c: (p[(*k, c, "Manhattan")] - p[(*k, c, "Bronx")]).numpy()[keep] for c in CASES}
        cell = lambda c, b, f: h[(*k, c, b, f)].numpy()[keep]
        new = {c: (cell(c, "Manhattan", 0) + cell(c, "Manhattan", 1) - cell(c, "Bronx", 0) - cell(c, "Bronx", 1)) / 2 for c in CASES}
        null = {c: (cell(c, "Manhattan", 0) + cell(c, "Bronx", 0) - cell(c, "Manhattan", 1) - cell(c, "Bronx", 1)) / 2 for c in CASES}
        a, b, z = ((x["restaurant"] + x["street"]) / 2 for x in (old, new, null))
        top, bot = np.argsort(-a)[:50], np.argsort(a)[:50]
        rows.append({"model": model, "prompt": k[0], "position": k[1], "pilot_new_restaurant": r(old["restaurant"], new["restaurant"]),
                     "pilot_new_street": r(old["street"], new["street"]), "new_across_cases": r(new["restaurant"], new["street"]),
                     "split_across_cases": r(null["restaurant"], null["street"]), "sd_gap": b.std(), "sd_split": z.std(),
                     "top50_positive": (b[top] > 0).mean(), "bottom50_negative": (b[bot] < 0).mean(),
                     "top50_mean": b[top].mean(), "bottom50_mean": b[bot].mean()})
        names = lambda ix: ", ".join(words[keep[i]].strip() for i in ix[:14])
        lists.append({"model": model, "prompt": k[0], "position": k[1], "manhattan": names(np.argsort(-np.minimum(a, b))),
                      "bronx": names(np.argsort(np.maximum(a, b)))})
    return rows, lists


def layers(pilot: Path, held: Path, tok, model: str) -> list:
    """Give the gap at each layer. The words come from the pilot only, thus the share that holds is a fair test."""
    p, h = (torch.load(d / model / "means_depth.pt", weights_only=False) for d in (pilot, held))
    n = next(iter(p.values())).numel()
    words = [tok.decode([i]) for i in range(n)]
    keep = np.array([i for i, w in enumerate(words) if re.fullmatch(r" [a-z]{3,}", w)])
    gap = lambda m, k, c, l: (m[k[0], k[1], c, "Manhattan", l] - m[k[0], k[1], c, "Bronx", l]).numpy()[keep]
    rows = []
    for k in sorted({(a, b, l) for a, b, _, _, l in p}):
        old, new = ({c: gap(m, k, c, k[2]) for c in CASES} for m in (p, h))
        a, b = ((x["restaurant"] + x["street"]) / 2 for x in (old, new))
        top, bot = np.argsort(-a)[:50], np.argsort(a)[:50]
        name = lambda ix: ", ".join(words[keep[i]].strip() for i in ix[:12])
        rows.append({"model": model, "prompt": k[0], "position": k[1], "layer": k[2], "pilot_new": float(np.corrcoef(a, b)[0, 1]),
                     "across_cases": float(np.corrcoef(new["restaurant"], new["street"])[0, 1]), "sd": float(b.std()),
                     "top50_held": float((b[top] > 0).mean()), "bottom50_held": float((b[bot] < 0).mean()),
                     "manhattan": name(top), "bronx": name(bot)})
    return rows


def main() -> None:
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="data/f10/hops")
    ap.add_argument("--heldout", default="data/f10/hops-heldout")
    ap.add_argument("--depth", nargs=2, default=None, metavar=("PILOT", "HELDOUT"), help="write only `hops_layers.csv`, from 2 runs with `means_depth.pt`")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    root, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.depth:
        rows = []
        for m in MODELS:
            rows += layers(Path(args.depth[0]), Path(args.depth[1]), AutoTokenizer.from_pretrained(DIRS[m]), m)
        pd.DataFrame(rows).to_csv(out / "hops_layers.csv", index=False)
        return
    added, borough, depth, yn, rep, lists = [], [], [], [], [], []
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(DIRS[m])
        got = replication(root, Path(args.heldout), tok, m)
        rep += got[0]
        lists += got[1]
        means, text = (torch.load(root / m / f, weights_only=False) for f in ("means.pt", "text.pt"))
        primary = json.loads((root / m / "meta.json").read_text())["primary"]
        for (prompt, pos, case) in sorted({k[:3] for k in means}):
            pair = {b: means[prompt, pos, case, b] for b in ("Manhattan", "Bronx")}
            mean, t = (pair["Manhattan"] + pair["Bronx"]) / 2, text[prompt, case, pos, primary]
            added.append({"model": m, "case": case, "prompt": prompt, "position": pos, "layer": primary,
                          "added": ", ".join(top_words(mean - t, tok)), "raw": ", ".join(top_words(mean, tok))})
            gap = pair["Manhattan"] - pair["Bronx"]
            borough.append({"model": m, "case": case, "prompt": prompt, "position": pos, "manhattan": ", ".join(top_words(gap, tok)),
                            "bronx": ", ".join(top_words(-gap, tok)), "sd": float(gap.std()), "max": float(gap.abs().max())})
        d = pd.concat(pd.read_parquet(p) for p in sorted((root / m).glob("part-*.parquet")))
        for (case, prompt, pos, l), g in d.groupby(["case", "prompt", "position", "layer"]):
            words = pd.Series([tok.decode([i]) for ids in g["added_id"] for i in ids[:10]])
            words = words[words.map(is_word)].str.strip().str.lower().value_counts()
            depth.append({"model": m, "case": case, "prompt": prompt, "position": pos, "layer": l, "images": len(g),
                          "recurring": ", ".join(f"{w} {c}" for w, c in words.head(12).items())})
            hit = {w: np.array([w in {tok.decode([i]).strip().lower() for i in ids[:30]} for ids in g["added_id"]], float)
                   for w in words.head(40).index}
            for w, v in hit.items():
                if 10 <= v.sum() <= len(v) - 10:
                    yn.append({"model": m, "case": case, "prompt": prompt, "position": pos, "layer": l, "word": w, "share": v.mean(),
                               "r": float(np.corrcoef(v, g["yes_no"])[0, 1])})
    pd.set_option("display.width", 240, "display.max_colwidth", 170)
    frames = {"hops_added": added, "hops_borough": borough, "hops_depth": depth, "hops_yes_no": yn, "hops_replication": rep, "hops_words": lists}
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    print(pd.DataFrame(rep).round(3).to_string())
    print(pd.DataFrame(added).drop(columns=["layer"]).to_string())
    print(pd.DataFrame(borough).to_string())
    print(pd.DataFrame(depth).query("prompt != 'f9' or position == 'final'").to_string())
    r = pd.DataFrame(yn)
    if len(r):
        print(r[r["r"].abs() >= 0.3].sort_values("r").round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
