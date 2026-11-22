#!/usr/bin/env python
"""F4: draw the 1,000 restaurant images that every model answers (D88).

A uniform random draw of 1,000 restaurants from the 10,000 of F6
(`data/f6_restaurant_depiction/v4/facade_check_b300/drawn_10000.parquet`),
seed 20261023. The image of a restaurant is its rendered best view. The output
has the columns of `research/scripts/whatami/stage_inputs.sh`: `image_idx`,
`sample_id` (the restaurant uid), and `image_path`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/sample_images.py
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import polars as pl

ROOT = "/share/ju/matt/mllmsci"
DRAWN = os.path.join(ROOT, "data/f6_restaurant_depiction/v4/facade_check_b300/drawn_10000.parquet")
OUT = os.path.join(ROOT, "data/f4_traces/f4_restaurants_1k.parquet")
SEED = 20261023

__version__ = "1.0.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drawn", default=DRAWN)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    drawn = pl.read_parquet(args.drawn)
    s = (drawn.sort("uid").sample(n=args.n, seed=args.seed, shuffle=True)
         .with_row_index("image_idx")
         .select(pl.col("image_idx").cast(pl.Int64), pl.col("uid").alias("sample_id"),
                 pl.col("view_path").alias("image_path"), "uid", "recording_id", "camis", "dba", "borough"))
    missing = [p for p in s["image_path"].to_list() if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing, for example {missing[0]}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    s.write_parquet(args.out)
    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": os.path.relpath(args.drawn, ROOT), "n": s.height, "seed": args.seed,
        "distinct_images": s["image_path"].n_unique(),
        "by_borough": dict(s["borough"].value_counts().iter_rows()),
    }
    with open(args.out.replace(".parquet", "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
