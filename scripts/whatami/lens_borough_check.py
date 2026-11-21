#!/usr/bin/env python3
"""Does a latent borough token match the borough of the image?

For each answer and lens: is a borough token at top 1 at any position, at any
layer in the late third below the last layer (lens_report.bands)? The script
cross-tabulates that with the true borough of the image. A latent borough that
tracks the image is a signal. A latent borough at the same rate for every true
borough is a property of the lens or the model, not of the image.

Usage:
    python scripts/whatami/lens_borough_check.py --model gemma-4-12b
Writes multirun/whatami/lens_report/<model>/borough_check.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import lens_common as lc
from lens_report import REPLAY, REPORT, bands, single_token_ids

BOROUGH_TERMS = {"bronx": "Bronx", "brooklyn": "Brooklyn", "manhattan": "Manhattan",
                 "queens": "Queens", "staten_island": "Staten"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(lc.MODELS))
    ap.add_argument("--band", default="late", choices=["early", "middle", "late"])
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(lc.MODELS[args.model].path)
    ids = {b: single_token_ids(tok, w) for b, w in BOROUGH_TERMS.items()}
    sample = pd.read_parquet(lc.REPO / "data/whatami/sample_1k.parquet", columns=["image_idx", "borough"])
    borough_of = dict(zip(sample.image_idx, sample.borough))
    root = Path(REPLAY) / args.model
    rows = []
    for lens in sorted(p.name for p in root.iterdir() if p.is_dir()):
        hits = {b: set() for b in ids}
        keys = set()
        for part in sorted((root / lens).glob("part-*.parquet")):
            t = pq.read_table(part, columns=["image_idx", "repeat", "topk_ids", "target_rank"])
            n = t.num_rows
            n_layers = t.column("target_rank").combine_chunks().flatten().to_numpy().size // n
            top = t.column("topk_ids").combine_chunks().flatten().flatten().to_numpy().reshape(n, n_layers, -1)[:, :, 0]
            lo, hi = bands(n_layers)[args.band]
            sub = top[:, lo:hi]
            img = t.column("image_idx").to_numpy()
            rep = t.column("repeat").to_numpy()
            keys.update(zip(img.tolist(), rep.tolist()))
            for b, tids in ids.items():
                h = np.isin(sub, tids).any(axis=1)
                hits[b].update(zip(img[h].tolist(), rep[h].tolist()))
        by_true: dict[str, list] = {}
        for k in keys:
            by_true.setdefault(borough_of[k[0]], []).append(k)
        for true_b, ks in sorted(by_true.items()):
            row = {"lens": lens, "band": args.band, "true_borough": true_b, "answers": len(ks)}
            for b in ids:
                row[f"latent_{b}"] = sum(k in hits[b] for k in ks) / len(ks)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(REPORT) / args.model
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"borough_check_{args.band}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
