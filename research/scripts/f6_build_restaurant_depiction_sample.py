#!/usr/bin/env python
"""Build the F6 sample: 100 restaurant images for a manual depiction check.

F6 asks whether an image that the facing filter attributes to a restaurant
truly shows a restaurant, in high definition, from close by. See
`research/docs/facct-runbook.md`, section F6.

The sample
----------
1. The pool: the images of `cyclomedia_near_restaurants_facing.parquet` whose
   `attribution_confidence` is at or above the 0.9 quantile over all images
   (the top 10%).
2. The restaurants: draw `--n` restaurants (`unit_uid`) at random from the
   restaurants that have at least 1 image in the pool.
3. The images: draw 1 image of each drawn restaurant at random, from its
   images in the pool.

The seed is fixed, thus the same call gives the same 100 images.

`attribution_confidence` is geometric, not a read of the pixels: the share of
the restaurant inside the face, times a cubic falloff in the distance
(`sv/nyc/curation/filter_facing.py`).

The outputs, in `--out-dir`
---------------------------
| File | Content |
|------|---------|
| `sample.parquet`, `sample.csv` | the 100 rows, with every column of the source |
| `images/<sample_id>.jpg` | a copy of each image, for the Label Studio local-files root |
| `tasks.json` | 1 Label Studio task for each image; the score is NOT in the task |
| `label_config.xml` | the labelling interface: 1 verdict (Pass or Fail) and a notes box |
| `manifest.json` | the cutoff, the seed, the counts, and the source path |

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f6_build_restaurant_depiction_sample.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE = os.path.join(
    ROOT, "curation/dohmh_restaurants_inspected_all/cyclomedia_near_restaurants_facing.parquet")
OUT_DIR = os.path.join(ROOT, "data/f6_restaurant_depiction")

__version__ = "1.0.0"

# The labelling interface. The verdict covers the 3 parts of the F6 criterion
# at once (decision of 2026-10-23: 1 pass/fail and notes).
LABEL_CONFIG = """<View>
  <Header value="$unit_name"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="false"/>
  <Choices name="verdict" toName="image" choice="single" required="true" showInline="true">
    <Choice value="Pass"/>
    <Choice value="Fail"/>
  </Choices>
  <TextArea name="notes" toName="image" placeholder="Notes" rows="3" editable="true" maxSubmissions="1"/>
</View>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--quantile", type=float, default=0.9,
                    help="the pool holds images at or above this quantile of the score")
    ap.add_argument("--n", type=int, default=100, help="restaurants to draw, 1 image each")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--force", action="store_true", help="write over an existing sample")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out_dir, "tasks.json")) and not args.force:
        raise SystemExit(f"a sample exists in {args.out_dir}; pass --force to replace it")

    df = pl.read_parquet(args.source)
    cutoff = float(df["attribution_confidence"].quantile(args.quantile, interpolation="linear"))
    pool = df.filter(pl.col("attribution_confidence") >= cutoff)
    units = pool["unit_uid"].unique().sort()
    if units.len() < args.n:
        raise SystemExit(f"the pool holds {units.len()} restaurants, fewer than {args.n}")

    drawn = units.sample(n=args.n, seed=args.seed, shuffle=True)
    # One image for each restaurant. A shuffle with 1 seed for the whole
    # frame, then the first row of each group: a per-group sample() would use
    # the same seed in each group (whatami-probe.md, warning).
    sample = (
        pool.filter(pl.col("unit_uid").is_in(drawn.implode()))
        .sort("sample_id")
        .sample(fraction=1.0, shuffle=True, seed=args.seed + 1)
        .group_by("unit_uid", maintain_order=True).first()
        .sort("unit_uid")
    )
    assert sample.height == args.n and sample["unit_uid"].n_unique() == args.n

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    tasks = []
    for row in sample.iter_rows(named=True):
        name = f"{row['sample_id']}.jpg"
        shutil.copyfile(row["image_path"], os.path.join(img_dir, name))
        tasks.append({"data": {
            "image": f"/data/local-files/?d=images/{name}",
            "sample_id": row["sample_id"],
            "unit_uid": row["unit_uid"],
            "unit_name": row["unit_name"],
        }})

    sample.write_parquet(os.path.join(args.out_dir, "sample.parquet"))
    # The CSV is for a quick look; nested and binary columns (the WKB
    # geometry) do not fit a CSV. The parquet keeps them.
    flat = [c for c, t in sample.schema.items() if not t.is_nested() and t != pl.Binary]
    sample.select(flat).write_csv(os.path.join(args.out_dir, "sample.csv"))
    with open(os.path.join(args.out_dir, "tasks.json"), "w") as fh:
        json.dump(tasks, fh, indent=1)
    with open(os.path.join(args.out_dir, "label_config.xml"), "w") as fh:
        fh.write(LABEL_CONFIG)

    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": args.source, "source_rows": df.height,
        "source_restaurants": df["unit_uid"].n_unique(),
        "quantile": args.quantile, "cutoff": cutoff,
        "pool_images": pool.height, "pool_restaurants": units.len(),
        "n": args.n, "seed": args.seed,
        "sample_score_min": float(sample["attribution_confidence"].min()),
        "sample_score_max": float(sample["attribution_confidence"].max()),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
