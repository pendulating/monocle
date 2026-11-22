#!/usr/bin/env python
"""F4: give each of the 1,000 F4 restaurants its 2020 NTA.

The F4 image table (`data/f4_traces/f4_restaurants_1k.parquet`) has no
geography beyond the borough. The facade point (`px`, `py`, EPSG:2263) of each
selected view is in the F6 table `data/f6_restaurant_depiction/v4/best.parquet`.
This script joins that point to the NTA polygons
(`data/geo/nynta2020_26b/nynta2020.shp`, EPSG:2263) with "within".

Output `data/f4_traces/f4_restaurants_nta.parquet`, 1 row for each image:
`image_idx`, `uid`, `camis`, `dba`, `borough`, `px`, `py`, `NTA2020`,
`NTAName`, `NTAType`, `CDTA2020`, `BoroName`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/restaurant_nta.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/restaurant_nta.py --images data/f4_traces/b2/f4_restaurants_b2.parquet \
        --out data/f4_traces/b2/f4_restaurants_b2_nta.parquet
"""
from __future__ import annotations

import argparse
import os

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
IMAGES = os.path.join(ROOT, "data/f4_traces/f4_restaurants_1k.parquet")
VIEWS = os.path.join(ROOT, "data/f6_restaurant_depiction/v4/best.parquet")
NTA = os.path.join(ROOT, "data/geo/nynta2020_26b/nynta2020.shp")
OUT = os.path.join(ROOT, "data/f4_traces/f4_restaurants_nta.parquet")

__version__ = "1.1.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=IMAGES, help="the image table (batch 2: data/f4_traces/b2/f4_restaurants_b2.parquet)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    images = pd.read_parquet(args.images, columns=["image_idx", "uid", "camis", "dba", "borough"])
    # Join on `uid` only: for 17 F6 restaurants the checked view is a neighbor
    # recording of the best one (batch 2, D132). The facade point is of the restaurant.
    views = pd.read_parquet(VIEWS, columns=["uid", "px", "py"])
    df = images.merge(views, on="uid", how="left", validate="many_to_one")
    if df.px.isna().any():
        raise SystemExit(f"{int(df.px.isna().sum())} images have no facade point in {VIEWS}")
    nta = gpd.read_file(NTA).to_crs(2263)
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.px, df.py), crs=2263)
    joined = gpd.sjoin(points, nta[["NTA2020", "NTAName", "NTAType", "CDTA2020", "BoroName", "geometry"]],
                       how="left", predicate="within")
    if joined.index.duplicated().any():
        raise SystemExit("a facade point falls in 2 NTAs")
    missing = int(joined.NTA2020.isna().sum())
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    out = out[["image_idx", "uid", "camis", "dba", "borough", "px", "py",
               "NTA2020", "NTAName", "NTAType", "CDTA2020", "BoroName"]]
    out.to_parquet(args.out, index=False)
    sizes = out.NTA2020.value_counts()
    print(f"[nta] {len(out):,} restaurants, {missing} outside every NTA, {sizes.size} NTAs; "
          f"restaurants per NTA p50 {sizes.median():.0f}, max {sizes.max()}")
    for k in (1, 3, 5, 10):
        print(f"[nta]   NTAs with >= {k:>2} restaurants: {(sizes >= k).sum():>3} ({sizes[sizes >= k].sum():,} restaurants)")
    print(f"[nta] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
