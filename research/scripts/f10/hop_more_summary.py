#!/usr/bin/env python
"""F10.30: the F10.26 summary of 1 lean store, with k images for each NTA. CPU only.

The lean stores (`research/scripts/f10/hop_lean.py`, `data/f10/hop_more/lean/{store}/`) hold the residuals at 6 positions,
the lens readouts at 3 positions and the answers of the 10-image sample. This script writes the summary in the
layout of F10.26 stage 1, thus the F10.27 to F10.29 scripts read it with a change of root
(`F10_PROMPTS_ROOT=data/f10/prompts_k10`).

| File in `data/f10/prompts_k{k}/{store}/` | Shape | Content |
|------------------------------------------|-------|---------|
| `nta_{position}.npy` | [NTAs, layers, vocab] float32 | the NTA mean of the z-scored readout at `place`, `after_place`, `final`. Gemma readouts are uncapped first |
| `residual_{position}.npy` | [images, layers, d] float16 | the raw residual at the same 3 positions |
| `full_mean.npy` | [3, layers, vocab] | the mean readout over the images |
| `sense.npy` | [images, 3, layers] | the F10.17 score |
| `frame.parquet` | 1 row for each image | `image_idx`, `nta2020`, `borough`, `rank`, `top`, `outcome`, `margin`, `codes` |
| `meta.json` | | layers, positions, k, counts, `complete` |

The images are the rank 1 to k views of each NTA (`data/f10/hop_more/images_k10.parquet`; the 591 of F10.26
are ranks 1 to 3). The NTA order is the F10.26 order (first appearance in `data/f10/hop_all/images.parquet`), thus `nta_*.npy` align with `data/f10/prompts/`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_summary.py --model gemma-4-31b --adjective cool --noun place --k 10
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_more_summary.py --model gemma-4-31b --all --k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10")]
READ_POSITIONS = ("place", "after_place", "final")  # as hop_lean.py
from hop_outcomes import outcome  # noqa: E402
from hop_prompts import ADJECTIVES, NOUNS, softcap_of, store_name, uncap, zscore  # noqa: E402
from hop_sense import SOCIAL, TEMPERATURE  # noqa: E402

__version__ = "1.0.0"
LEAN = ROOT / "data/f10/hop_more/lean"
SAMPLE = ROOT / "data/f10/hop_more/images_k10.parquet"


def summary(model: str, adjective: str, noun: str, k: int, out_root: Path, lean: Path = LEAN, sample: Path = SAMPLE) -> None:
    name = store_name(model, adjective, noun)
    st = lean / name
    man = json.loads((st / "manifest.json").read_text())
    chunk, layers = int(man["chunk"]), list(man["layers"])
    positions = list(man["positions"])
    read_pos = list(man.get("read_positions", READ_POSITIONS))
    res_idx = [positions.index(p) for p in read_pos]
    images = pd.read_parquet(sample)
    if len(images) != int(man["n_images"]):
        raise ValueError(f"{name}: the sample has {len(images)} images, the store {man['n_images']}")
    keep = (images["rank"] <= k).to_numpy()
    fr = images.loc[keep, ["image_idx", "nta2020", "borough", "rank"]].reset_index(drop=True)
    # The NTA order of F10.26 (first appearance in `data/f10/hop_all/images.parquet`), thus the arrays align across roots.
    order = pd.read_parquet(ROOT / "data/f10/hop_all/images.parquet")["nta2020"].drop_duplicates().tolist()
    if set(fr["nta2020"]) != set(order):
        raise ValueError(f"{name}: the NTAs differ from the F10.26 NTAs")
    codes = fr["nta2020"].map({n: i for i, n in enumerate(order)}).to_numpy()
    n_nta = len(order)
    vocab = pd.read_parquet(st / "vocab.parquet")
    words = vocab["token"].str.strip().str.lower()
    hot = np.flatnonzero(words.isin(TEMPERATURE).to_numpy())
    soc = np.flatnonzero(words.isin(SOCIAL).to_numpy())
    cap = softcap_of(model)
    n_shards = (len(images) + chunk - 1) // chunk
    answers = []
    for s in range(n_shards):
        if not (st / "answers" / f"answers-{s:05d}.parquet").exists():
            raise FileNotFoundError(f"{name}: shard {s} is not done")
    n_pos, n_lay, n_voc, d = len(read_pos), len(layers), len(vocab), int(man["d"])
    full = np.zeros((n_pos, n_lay, n_voc), np.float64)
    nta = np.zeros((n_nta, n_pos, n_lay, n_voc), np.float32)
    sense = np.zeros((len(fr), n_pos, n_lay), np.float32)
    res = np.zeros((len(fr), n_pos, n_lay, d), np.float16)
    row = 0
    for s in range(n_shards):
        lo, hi = s * chunk, min((s + 1) * chunk, len(images))
        sel = np.flatnonzero(keep[lo:hi])
        a = pd.read_parquet(st / "answers" / f"answers-{s:05d}.parquet")
        if (a.image_idx.to_numpy() != images.image_idx.to_numpy()[lo:hi]).any():
            raise ValueError(f"{name}: shard {s} holds other images")
        answers.append(a.iloc[sel])
        if len(sel) == 0:
            continue
        x = np.load(st / "reads" / f"reads-{s:05d}.npy")[sel]  # [n, 3, layers, vocab] float16
        if x.shape[1] != n_pos:
            raise ValueError(f"{name}: reads hold {x.shape[1]} positions, expected {n_pos}")
        z = zscore(uncap(np.asarray(x, np.float64), cap))
        del x
        rows = np.arange(row, row + len(sel))
        full += z.sum(0)
        for c in np.unique(codes[rows]):
            nta[c] += z[codes[rows] == c].sum(0).astype(np.float32)
        sense[rows] = z[..., soc].mean(-1) - z[..., hot].mean(-1)
        del z
        r = np.load(st / "residuals" / f"residuals-{s:05d}.npy")[sel]  # [n, 6, layers, d] float16
        res[rows] = r[:, res_idx]
        del r
        row += len(sel)
    if row != len(fr):
        raise ValueError(f"{name}: read {row} images, expected {len(fr)}")
    ans = pd.concat(answers).reset_index(drop=True)
    fr = fr.merge(ans[["image_idx", "margin", "top"]], on="image_idx", validate="1:1")
    fr["outcome"] = fr["top"].map(outcome)
    fr.loc[fr["outcome"] == "unable", "margin"] = np.nan
    fr["codes"] = codes
    # Sort the images by NTA code and rank, thus the first appearance order of `nta2020` in the frame is the
    # F10.26 order, as the F10.27 to F10.29 scripts expect (they factorize the frame).
    perm = np.lexsort((fr["rank"].to_numpy(), codes))
    fr = fr.iloc[perm].reset_index(drop=True)
    codes, res, sense = codes[perm], res[perm], sense[perm]
    counts = np.bincount(codes, minlength=n_nta)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(read_pos):
        np.save(out / f"nta_{p}.npy", (nta[:, i] / counts[:, None, None]).astype(np.float32))
        np.save(out / f"residual_{p}.npy", res[:, i])
    np.save(out / "full_mean.npy", (full / max(row, 1)).astype(np.float32))
    np.save(out / "sense.npy", sense)
    fr.to_parquet(out / "frame.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"version": __version__, "store": name, "model": model, "adjective": adjective, "noun": noun, "k": k,
                                               "positions": read_pos, "layers": layers, "n_images": int(row), "n_nta": n_nta, "vocab": n_voc, "d": d,
                                               "softcap": cap, "source": str(st.relative_to(ROOT)), "complete": True}, indent=1) + "\n")
    print(f"[f10.30] {name}: {row} images, {n_nta} NTAs, k={k} -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adjective")
    ap.add_argument("--noun")
    ap.add_argument("--all", action="store_true", help="all 14 prompts of the model")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out-root", default=None, help="default data/f10/prompts_k{k}")
    args = ap.parse_args()
    out_root = Path(args.out_root) if args.out_root else ROOT / f"data/f10/prompts_k{args.k}"
    pairs = [(a, n) for n in NOUNS for a in ADJECTIVES] if args.all else [(args.adjective, args.noun)]
    for adj, noun in pairs:
        try:
            summary(args.model, adj, noun, args.k, out_root)
        except FileNotFoundError as e:
            print(f"[f10.30] skip: {e}", flush=True)


if __name__ == "__main__":
    main()
