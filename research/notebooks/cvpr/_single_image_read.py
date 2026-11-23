"""The single-image reads of Gemma at the 3 geography layers of the master table.

The white-box study scored 3 images for each unit that holds a pairwise score
and a proxy value (`research/scripts/pole_score_sample.py`, `research/monocle/pole_score.py`).
Each image gives 2 reads of the judgment:

| Read | What |
|------|------|
| `c` | The logit gap between the 2 polar words at the answer position. |
| `contrast_L42` | The centered projection of the block-42 answer residual on the lens contrast vector. |

This module puts those reads on the footing of the master table. It takes the
mean read of each unit, joins the unit to its census tract, NTA, and community
district with the same point layer and the same unit floor as the case
notebook, and reads Kendall tau against the area proxy of the case export.
The proxy side is the file the master table reads, thus a row here and a row
of the master table differ in the model score alone.

The units are the units the study scored. Restaurants holds 1,500 of them,
the road case holds 3,000 rated images, and the other cases hold every unit
of the join.

Run it from the canonical venv:

    .venv-mllmsci-vllm025cu129/bin/python research/notebooks/cvpr/_single_image_read.py

The frame goes to `master/outputs/single_image_read_v<version>_long.csv`, and
`_proxy_overview.py` draws it beside the pairwise markers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from research.notebooks.cvpr import _geography as geo  # noqa: E402
from research.notebooks.cvpr import _results_table as R  # noqa: E402
from research.notebooks.cvpr import _unit_validation as U  # noqa: E402

__version__ = "1.0.0"

REPO_ROOT = HERE.parent.parent.parent
SCORE_ROOT = REPO_ROOT / "outputs" / "_monocle" / "pole_score"
OUT_DIR = HERE / "master" / "outputs"

MODEL = "gemma-4-12b"
READS: List[str] = ["c", "contrast_L42"]
LAYERS = [("census_tract", "tract"), ("nta", "NTA"), ("community_district", "CD")]

#: The case directory of the score root for each folder of the master table.
CASE_OF_FOLDER: Dict[str, str] = {
    "subway": "subway_safety", "libraries": "libraries", "schools": "schools",
    "road": "road_quality", "parks": "parks_plazas", "plazas": "parks_plazas",
    "restaurants": "restaurants",
}

#: The polygon floor of each case notebook (`MIN_UNITS_PER_POLYGON`).
MIN_UNITS: Dict[str, int] = {
    "subway": 1, "libraries": 3, "schools": 3, "road": 1, "parks": 1,
    "plazas": 1, "restaurants": 3,
}


def image_reads(case: str) -> pd.DataFrame:
    """One row for each scored image: the sample columns and the centered reads."""
    shards = sorted((SCORE_ROOT / case).glob("scores_shard*.parquet"))
    if not shards:
        raise FileNotFoundError(f"{SCORE_ROOT / case}: no score shards")
    sc = pd.concat([pd.read_parquet(s) for s in shards])
    sample = pd.read_parquet(SCORE_ROOT / case / "sample.parquet")
    sc["contrast"] = sc["raw_contrast"] - sc.groupby("layer")["raw_contrast"].transform("mean")
    wide = sc.pivot(index="image_path", columns="layer", values="contrast")
    wide.columns = [f"contrast_L{int(b)}" for b in wide.columns]
    wide["c"] = sc.groupby("image_path")["c"].first()
    return sample.merge(wide, left_on="image_path", right_index=True, how="inner")


def unit_reads(img: pd.DataFrame, folder: str) -> pd.DataFrame:
    """The mean read of each unit of one folder. An image case keys on the image."""
    sub = img[img["folder"] == folder]
    key = "sample_id" if U._spec(folder)["mode"] == "image" else "unit_uid"
    out = (sub.groupby(key)[READS].mean().reset_index()
           .rename(columns={key: "place"}))
    out["n_images"] = sub.groupby(key).size().to_numpy()
    return out


def unit_points(folder: str, units: pd.DataFrame):
    """The point of each unit, in the working CRS, with a `place` column."""
    import geopandas as gpd

    if U._spec(folder)["mode"] == "image":
        scores = U.place_scores(folder, MODEL)
        pos = scores[["sample_id", "latitude", "longitude"]].drop_duplicates("sample_id")
        pts = gpd.GeoDataFrame(
            pos.rename(columns={"sample_id": "place"}),
            geometry=gpd.points_from_xy(pos["longitude"], pos["latitude"]),
            crs="EPSG:4326").to_crs(geo.WORKING_CRS)
    else:
        gdf = U._units_gdf(folder)
        pts = gdf[["unit_uid", "geometry"]].rename(columns={"unit_uid": "place"})
    pts["place"] = pts["place"].astype(str)
    units = units.assign(place=units["place"].astype(str))
    return pts.merge(units, on="place", how="inner")


def area_reads(pts, layer: str, min_units: int) -> pd.DataFrame:
    """The plain mean read of each polygon of one layer, over the unit floor."""
    import geopandas as gpd

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    hit = gpd.sjoin(pts[["place", "geometry"] + READS], poly[[key, "geometry"]],
                    how="inner", predicate="within")
    out = hit.groupby(key).agg(n_units=("place", "nunique"),
                               **{r: (r, "mean") for r in READS}).reset_index()
    out = out[out["n_units"] >= min_units]
    drop = geo.excluded_keys(layer)
    if drop:
        out = out[~out[key].astype(str).isin(drop)]
    return out.rename(columns={key: "unit"})


def area_proxy(folder: str, layer: str) -> Optional[pd.DataFrame]:
    """The area proxy values of the case export, the file the master table reads."""
    case = next(c for c in R.CASES if c["folder"] == folder)
    path = R._newest(str(HERE / folder / "outputs" / f"{case['prefix']}_proxy_v*.parquet"))
    if not path:
        return None
    key = R.LAYER_KEY[layer]
    prx = pd.read_parquet(path)
    prx = prx[prx["layer"] == layer]
    if "proxy" not in prx.columns:
        prx = prx.assign(proxy=case.get("single_proxy", "proxy"))
    prx = prx[["proxy", key, "proxy_mean"]].dropna(subset=[key])
    drop = geo.excluded_keys(layer)
    if drop:
        prx = prx[~prx[key].astype(str).isin(drop)]
    return prx.rename(columns={key: "unit"})


def build(folders: Sequence[str] = tuple(CASE_OF_FOLDER)) -> pd.DataFrame:
    """The long frame: 1 row for each (folder, proxy, read, layer)."""
    rows = []
    cache: Dict[str, pd.DataFrame] = {}
    for folder in folders:
        case_dir = CASE_OF_FOLDER[folder]
        if case_dir not in cache:
            cache[case_dir] = image_reads(case_dir)
        units = unit_reads(cache[case_dir], folder)
        pts = unit_points(folder, units)
        label = next(c["label"] for c in R.CASES if c["folder"] == folder)
        order = [c["folder"] for c in R.CASES].index(folder)
        print(f"[single] {folder}: {len(units):,} units scored, {len(pts):,} with a point",
              flush=True)
        for layer, short in LAYERS:
            areas = area_reads(pts, layer, MIN_UNITS[folder])
            prx = area_proxy(folder, layer)
            if prx is None or areas.empty:
                continue
            areas["unit"] = areas["unit"].astype(str)
            prx["unit"] = prx["unit"].astype(str)
            for proxy, gp in prx.groupby("proxy"):
                m = areas.merge(gp, on="unit", how="inner")
                for read in READS:
                    met = R.metrics(m[read], m["proxy_mean"])
                    rows.append({
                        "case": label, "case_order": order, "folder": folder,
                        "proxy_key": proxy, "model": R.MODEL_LABEL.get(MODEL, MODEL),
                        "read": read, "layer": short, "n_units_scored": len(units),
                        **met,
                    })
    return pd.DataFrame(rows).sort_values(["case_order", "proxy_key", "read", "layer"]).reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", default="all")
    args = ap.parse_args(argv)
    folders = tuple(CASE_OF_FOLDER) if args.folders == "all" else tuple(args.folders.split(","))
    df = build(folders)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"single_image_read_v{__version__}_long.csv"
    df.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    show = df.pivot_table(index=["case", "proxy_key", "read"], columns="layer", values="tau")
    print(show.round(2).to_string())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
