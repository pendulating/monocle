#!/usr/bin/env python
"""F10.5: draw the street images of the depth read, stratified by NTA.

The sample takes 3 images from each NTA of the 10,000 F9 street images, or all the
images of an NTA with fewer. Both models read the same file.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/build_depth_sample.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__version__ = "1.0.0"

SEED = 20261031
PER_NTA = 3
POOL = "data/f4_traces/subj/f4_street_10k.parquet"
OUT = Path("data/f10/hop_depth/images.parquet")


def stratified(pool: pd.DataFrame, per_area: int = PER_NTA, seed: int = SEED) -> pd.DataFrame:
    """Give up to per_area rows of each NTA, in a fixed order."""
    pool = pool.dropna(subset=["nta2020"]).sort_values("image_idx")
    got = pool.groupby("nta2020", group_keys=False).sample(frac=1, random_state=seed).groupby("nta2020").head(per_area)
    return got.sort_values(["nta2020", "image_idx"]).reset_index(drop=True)


def main() -> None:
    cols = ["image_idx", "sample_id", "image_path", "nta2020", "nta_name", "borough", "latitude", "longitude"]
    pool = pd.read_parquet(POOL, columns=cols)
    got = stratified(pool)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    got.to_parquet(OUT, index=False)
    size = pool.groupby("nta2020").size()
    print(f"{len(got)} images, {got['nta2020'].nunique()} NTAs of {pool['nta2020'].nunique()}; "
          f"{int((size < PER_NTA).sum())} NTAs have fewer than {PER_NTA}: {size[size < PER_NTA].to_dict()}")
    print(got.groupby("borough").agg(images=("image_idx", "size"), ntas=("nta2020", "nunique")).to_string())


if __name__ == "__main__":
    main()
