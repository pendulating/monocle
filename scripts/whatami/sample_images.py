#!/usr/bin/env python3
"""Draw the image sample for the "what am I looking at?" probe.

The frame is the 2025 Cyclomedia NYC coverage: the year=2025 partitions of the
five *_2025_1k datasets, horizontal cube faces only (F, B, L, R). The script
draws recordings uniformly at random, then one face for each recording. Thus
no two images show the same spot, and each borough gets weight in proportion
to its coverage.

Rows with manifest_ok false or a file under 50 KB are removed first. The catalog
check 8 flags these files as suspect renders.

Usage:
    python scripts/whatami/sample_images.py --n 1000 --seed 20261020 \
        --out data/whatami/sample_1k.parquet
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import polars as pl

CATALOG_GLOB = "/share/ju/cyclomedia/catalog/v1/by_dataset/dataset=*_2025_1k/year=2025/part-0.parquet"
FACES = ["F", "B", "L", "R"]
MIN_FILE_BYTES = 50_000
COLUMNS = [
    "sample_id", "recording_id", "face", "image_path", "group", "borough",
    "latitude", "longitude", "recordedAt", "bearing", "yawDegrees",
    "file_size", "manifest_ok",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", default="data/whatami/sample_1k.parquet")
    args = ap.parse_args()

    files = sorted(glob.glob(CATALOG_GLOB))
    if len(files) != 5:
        raise SystemExit(f"expected 5 year=2025 partitions, found {len(files)}: {files}")

    frames = []
    for f in files:
        dataset = f.split("dataset=")[1].split("/")[0]
        frames.append(
            pl.scan_parquet(f)
            .select(COLUMNS)
            .filter(pl.col("face").is_in(FACES) & pl.col("manifest_ok") & (pl.col("file_size") > MIN_FILE_BYTES))
            .with_columns(pl.lit(dataset).alias("dataset"))
        )
    faces = pl.concat(frames).collect()
    print(f"eligible faces: {faces.height:,}")

    # Uniform over recordings, then one face in each recording.
    recordings = faces.select("dataset", "recording_id").unique().sort("dataset", "recording_id")
    print(f"eligible recordings: {recordings.height:,}")
    picked = recordings.sample(n=args.n, seed=args.seed, shuffle=True)
    cand = faces.join(picked, on=["dataset", "recording_id"], how="inner").sort("sample_id")
    # One random key for each row from one RNG, then the smallest key wins in
    # each recording. Do not use sample() inside group_by().agg(): the seed is
    # the same in each group, thus it takes the same position (the L face) each time.
    rng = np.random.default_rng(args.seed + 1)
    sample = (
        cand.with_columns(pl.Series("_key", rng.random(cand.height)))
        .sort("_key")
        .unique(subset=["dataset", "recording_id"], keep="first", maintain_order=True)
        .drop("_key")
        .sort("sample_id")
        .with_row_index("image_idx")
    )
    assert sample.height == args.n, sample.height
    assert sample["sample_id"].n_unique() == args.n

    missing = [p for p in sample["image_path"] if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} image files are missing, e.g. {missing[:3]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(out)
    print(f"wrote {sample.height} rows to {out}")
    print(sample.group_by("borough").len().sort("borough"))
    print(sample.group_by("face").len().sort("face"))


if __name__ == "__main__":
    main()
