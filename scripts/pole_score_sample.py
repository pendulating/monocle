"""Draw the images to score with the pole projections, per proxy folder.

Units come from `_unit_validation.joined(folder, model, "canonical")`: the
units that hold BOTH a pairwise rating (5 comparisons or more) and a proxy
value. Each drawn unit gets up to `--images-per-unit` of its own images
from the canonical pairs table. The subway folder pools the entrances of a
station. The road folder is image-level: it draws rated images directly.
Street photography has no proxy and is not drawn.

Writes `outputs/_monocle/pole_score/<case>/sample.parquet` with
image_path, sample_id, unit_uid, proxy_key, proxy_value, mean_score,
n_comparisons, folder, case.

    .venv-3.12/bin/python scripts/pole_score_sample.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
CVPR = REPO / "notebooks/cvpr"
OUT = REPO / "outputs/_monocle/pole_score"
FOLDER_CASE = {"restaurants": "restaurants", "subway": "subway_safety",
               "libraries": "libraries", "schools": "schools",
               "parks": "parks_plazas", "plazas": "parks_plazas",
               "road": "road_quality"}
N_UNITS = {"restaurants": 1500, "subway": 10_000, "libraries": 10_000,
           "schools": 10_000, "parks": 10_000, "plazas": 10_000}
N_ROAD_IMAGES = 3000
MODEL = "gemma-4-12b"


def image_pool(run_dir: Path, image_level: bool) -> pd.DataFrame:
    cols = ["sample_id_a", "sample_id_b", "image_path_a", "image_path_b"]
    if not image_level:
        cols += ["unit_uid_a", "unit_uid_b"]
    p = pd.read_parquet(run_dir / "pairs.parquet", columns=cols)
    sides = []
    for s in "ab":
        d = {"sample_id": p[f"sample_id_{s}"], "image_path": p[f"image_path_{s}"]}
        if not image_level:
            d["unit_uid"] = p[f"unit_uid_{s}"].astype(str)
        sides.append(pd.DataFrame(d))
    return pd.concat(sides).drop_duplicates("image_path").reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folders", default=",".join(FOLDER_CASE))
    ap.add_argument("--images-per-unit", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(CVPR))
    import _canonical as C
    import _unit_validation as U

    rng = np.random.default_rng(args.seed)
    frames: dict[str, list] = {}
    for folder in args.folders.split(","):
        case = FOLDER_CASE[folder]
        run_dir = REPO / f"notebooks/cvpr/canonical_data/proxy/canonical/{case}__{MODEL}"
        j = U.joined(folder, MODEL, "canonical", None)
        if folder == "road":
            pool = image_pool(run_dir, image_level=True)
            j = j.drop_duplicates("sample_id")
            keep = pool.merge(j[["sample_id", "mean_score", "n_comparisons", "proxy_value"]],
                              on="sample_id")
            take = keep.sample(n=min(N_ROAD_IMAGES, len(keep)), random_state=args.seed)
            take["unit_uid"] = take["sample_id"]
            take["proxy_key"] = take["sample_id"]
        else:
            pool = image_pool(run_dir, image_level=False)
            if folder == "subway":
                import _proxies_subway as sub
                m = sub.entrance_station_map()[["unit_uid", "station_key"]]
                m["unit_uid"] = m["unit_uid"].astype(str)
                pool = pool.merge(m, on="unit_uid", how="inner")
                pool["proxy_key"] = pool["station_key"]
            else:
                pool["proxy_key"] = pool["unit_uid"]
            key = "unit_uid"  # joined() carries the rolled-up key as unit_uid
            j = j.drop_duplicates(key)
            j["proxy_key"] = j[key].astype(str)
            units = j["proxy_key"].unique()
            n = min(N_UNITS[folder], len(units))
            drawn = rng.choice(units, size=n, replace=False)
            keep = pool[pool["proxy_key"].isin(set(drawn))]
            take = (keep.groupby("proxy_key", group_keys=False)
                    .apply(lambda g: g.sample(n=min(args.images_per_unit, len(g)),
                                              random_state=int(rng.integers(1 << 31)))))
            take = take.merge(j[["proxy_key", "mean_score", "n_comparisons", "proxy_value"]],
                              on="proxy_key")
        take["folder"] = folder
        take["case"] = case
        print(f"[sample] {folder}: {take['proxy_key'].nunique():,} units, "
              f"{len(take):,} images", flush=True)
        frames.setdefault(case, []).append(take)

    for case, parts in frames.items():
        out = OUT / case
        out.mkdir(parents=True, exist_ok=True)
        df = pd.concat(parts).reset_index(drop=True)
        df = df[["image_path", "sample_id", "unit_uid", "proxy_key", "proxy_value",
                 "mean_score", "n_comparisons", "folder", "case"]]
        df.to_parquet(out / "sample.parquet", index=False)
        print(f"[sample] wrote {out / 'sample.parquet'} ({len(df):,} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
