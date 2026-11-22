#!/usr/bin/env python
"""F4 batch 2: draw the restaurant images, stratified by NTA (D132).

The canonical 1,000 images (`sample_images.py`) are a uniform draw. Batch 2
adds images until the set holds `--total` images, as even over the NTAs as
the pool permits. Exactly `--total` images comes first; an equal count for
each NTA comes second.

| Item | Rule |
|------|------|
| Pool | the F6 restaurants with a "Yes" from the facade check (`facade_check_b300/part_*.parquet`), as `f6_vlm_facade_check.py draw`; the image is the rendered view |
| NTA | the facade point (`px`, `py` of `best.parquet`, joined on `uid`; the image keeps the recording of the checked view) in the 2020 NTA polygons, "within", EPSG:2263, as `restaurant_nta.py`. A restaurant outside every NTA stays out |
| Allocation | NTA `i` gets `clip(L, kept_i, capacity_i)`, with the largest level `L` that keeps the sum at `--total` or less. The rest goes 1 at a time to NTAs at the level with a spare restaurant, at random |
| Kept images | the images of `--keep` stay; they count in their NTA |
| Draw | in each NTA, the allocation minus the kept images, at random from the other restaurants of the NTA |
| Order | the new rows in a random order; `image_idx` starts at `--idx-offset` |

The generator is `numpy.random.default_rng(--seed)`: first the NTAs that get
1 more, then the draw in each NTA (NTAs in sorted order, restaurants in `uid`
order), then the order of the rows.

Output: the new rows only, with the columns of `sample_images.py` and
`NTA2020`, `NTAName`, `nta_capacity` (the pool of the NTA), `nta_n` (the
allocation of the NTA, kept images included). In NTA `i`, a restaurant of the
pool is in the set with the chance `nta_n / nta_capacity`. A manifest is
written next to the output.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/sample_images_nta.py --total 10000 \\
        --keep data/f4_traces/f4_restaurants_1k.parquet --seed 20261027 --idx-offset 1000 \\
        --out data/f4_traces/b2/f4_restaurants_b2.parquet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
F6 = os.path.join(ROOT, "data/f6_restaurant_depiction/v4")
CHECK = os.path.join(F6, "facade_check_b300")
NTA = os.path.join(ROOT, "data/geo/nynta2020_26b/nynta2020.shp")
COLS = ["image_idx", "sample_id", "image_path", "uid", "recording_id", "camis", "dba", "borough"]

__version__ = "1.0.0"


def pool() -> pd.DataFrame:
    """Give the "Yes" restaurants with their view, their DBA, and their NTA."""
    ans = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(CHECK, "part_*.parquet")))])
    # Join on `uid` only, as the F6 draw: for 17 restaurants the checked view is
    # a neighbor recording of the best one. The facade point is of the restaurant.
    best = pd.read_parquet(os.path.join(F6, "best.parquet"), columns=["uid", "camis", "dba", "borough", "px", "py"])
    yes = ans[ans.answer == "Yes"][["uid", "recording_id", "view_path"]]
    df = yes.merge(best, on="uid", how="left", validate="one_to_one")
    if df.px.isna().any():
        raise SystemExit(f"{int(df.px.isna().sum())} restaurants have no facade point")
    nta = gpd.read_file(NTA).to_crs(2263)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.px, df.py), crs=2263)
    j = gpd.sjoin(pts, nta[["NTA2020", "NTAName", "geometry"]], how="left", predicate="within")
    if j.index.duplicated().any():
        raise SystemExit("a facade point falls in 2 NTAs")
    return pd.DataFrame(j.drop(columns=["geometry", "index_right"]))


def allocate(cap: pd.Series, floor: pd.Series, total: int, rng: np.random.Generator) -> pd.Series:
    """Give the count of each NTA: as even as `cap` permits, never below `floor`, `total` in all."""
    if floor.sum() > total or cap.sum() < total:
        raise SystemExit(f"no allocation: kept {floor.sum()}, pool {cap.sum()}, total {total}")
    lo, hi = 0, int(cap.max())
    while lo < hi:
        m = (lo + hi + 1) // 2
        if np.minimum(np.maximum(m, floor), cap).sum() <= total:
            lo = m
        else:
            hi = m - 1
    n = np.minimum(np.maximum(lo, floor), cap)
    rest = total - int(n.sum())
    if rest:
        spare = sorted(n.index[(n == lo) & (cap > lo)])
        n.loc[rng.choice(spare, size=rest, replace=False)] += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, required=True, help="images in all, kept images included")
    ap.add_argument("--keep", required=True, help="the sample whose images stay (the canonical 1,000)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--idx-offset", type=int, required=True, help="the first image_idx of the new rows")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    p = pool()
    outside = int(p.NTA2020.isna().sum())
    keep = set(pd.read_parquet(args.keep, columns=["uid"]).uid)
    p["kept"] = p.uid.isin(keep)
    if p.kept.sum() != len(keep):
        raise SystemExit(f"{len(keep) - int(p.kept.sum())} kept images are not in the pool")
    if p[p.kept].NTA2020.isna().any():
        raise SystemExit("a kept image is outside every NTA")
    p = p.dropna(subset=["NTA2020"])
    cap = p.groupby("NTA2020").size()
    floor = p.groupby("NTA2020").kept.sum().astype(int)
    n = allocate(cap, floor, args.total, rng)

    parts = []
    for k in sorted(n.index):
        need = int(n[k] - floor[k])
        if need:
            cand = p[(p.NTA2020 == k) & ~p.kept].sort_values("uid")
            parts.append(cand.iloc[np.sort(rng.choice(len(cand), size=need, replace=False))])
    new = pd.concat(parts)
    new = new.iloc[rng.permutation(len(new))].reset_index(drop=True)
    new["image_idx"] = np.arange(args.idx_offset, args.idx_offset + len(new), dtype=np.int64)
    new["sample_id"] = new.uid
    new["image_path"] = new.view_path
    new["nta_capacity"] = new.NTA2020.map(cap).astype(int)
    new["nta_n"] = new.NTA2020.map(n).astype(int)
    missing = [x for x in new.image_path if not os.path.isfile(x)]
    if missing:
        raise SystemExit(f"{len(missing)} images are missing, for example {missing[0]}")
    out = new[COLS + ["NTA2020", "NTAName", "nta_capacity", "nta_n"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)

    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pool_yes": int(len(p) + outside), "pool_outside_nta": outside, "ntas": int(len(n)),
        "keep": os.path.relpath(os.path.abspath(args.keep), ROOT), "kept": len(keep),
        "total": args.total, "new": len(out), "seed": args.seed, "idx_offset": args.idx_offset,
        "ntas_whole_pool": int((n == cap).sum()),
        "per_nta": {k: round(float(v), 1) for k, v in n.describe().items()},
        "by_borough": {str(k): int(v) for k, v in out.borough.value_counts().items()},
    }
    with open(args.out.replace(".parquet", "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
