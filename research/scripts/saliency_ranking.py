"""A model-free ranking of image cells, and a screen against any gate map.

The control this answers
------------------------
If a ranking built from raw image statistics separates a deletion test as well
as a lens does, then the lens says nothing about THIS model's internals. Every
claim of the form "the lens finds what the model uses" needs this control to
fail.

Two modes:

| Mode | What it does |
|---|---|
| `--screen` | correlate every gate arm with every image statistic |
| `--rank` | write a ranking parquet for `deletion_curve --rank-from` |

The statistics come from `monocle.cell_statistics.cell_stats`: luminance,
contrast, edge density, saturation, and the row and column of the cell.

Warning: a raw feature has no sign. `--rank` writes the feature as given, so
choose the sign deliberately with `--negate` and say which you chose.

Usage:
    python research/scripts/saliency_ranking.py --screen \
        --maps outputs/_monocle/input_gate/subway_safety/maps.parquet \
        --case subway_safety
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("luminance", "contrast", "edge", "saturation", "row", "col")


def stats_for(image_ids, case: str, n_cells: int,
              seed: int = 777) -> dict[str, pd.DataFrame]:
    """Per-cell statistics for each image, keyed by image_id."""
    from PIL import Image
    from research.monocle import cell_statistics as cs
    from research.monocle import single as sg

    want = set(map(str, image_ids))
    out = {}
    # The gate reads a deterministic prefix, so a generous draw covers it.
    for r in sg.load_single_rows(max(64, len(want) * 4), case,
                                 seed=seed):
        if str(r["image_id"]) not in want:
            continue
        img = Image.open(r["path"]).convert("RGB")
        out[str(r["image_id"])] = cs.cell_stats(img, n_cells)
        if len(out) == len(want):
            break
    return out


def screen(maps: Path, case: str, arms: list[str],
           seed: int = 777) -> pd.DataFrame:
    from scipy.stats import pearsonr

    df = pd.read_parquet(maps)
    have = [a for a in arms if a in df.columns]
    n_cells = len(np.asarray(df.iloc[0]["causal"]).ravel())
    stats = stats_for(df.target.unique(), case, n_cells, seed=seed)
    rows = []
    for layer, sub in df.groupby("layer"):
        for arm in have:
            for feat in FEATURES:
                vals = []
                for _, r in sub.iterrows():
                    m = r[arm]
                    st = stats.get(str(r.target))
                    if m is None or np.ndim(m) == 0 or st is None:
                        continue
                    a = np.asarray(m, float).ravel()
                    b = st[feat].to_numpy(float)
                    if a.size != b.size or a.std() == 0 or b.std() == 0:
                        continue
                    vals.append(pearsonr(a, b)[0])
                if vals:
                    rows.append({"layer": int(layer), "arm": arm,
                                 "feature": feat, "r": float(np.mean(vals)),
                                 "n": len(vals)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--case", default="subway_safety")
    ap.add_argument("--arms", nargs="+", default=[
        "causal", "gxa", "cheap", "meang_own", "meangp_own", "lens_own"])
    ap.add_argument("--seed", type=int, default=777,
                    help="which images the maps come from. 777 is "
                         "the jlens ladder. 20260821 is causal_maps "
                         "and deletion_curve.")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--rank", default=None,
                    help="A feature name; writes a ranking parquet.")
    ap.add_argument("--negate", action="store_true")
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.screen:
        res = screen(Path(a.maps), a.case, a.arms, seed=a.seed)
        if res.empty:
            print("no overlap between the maps and the statistics")
            return 1
        piv = res.pivot_table(index=["layer", "arm"], columns="feature",
                              values="r")
        print("Pearson r between each gate arm and each image statistic")
        print("(a large value means the arm is reading the picture, not the "
              "model)")
        print(piv.round(3).to_string())
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            res.to_parquet(a.out, index=False)
            print(f"wrote {a.out}")
        return 0

    if a.rank:
        from research.scripts.make_ranking import pool

        df = pd.read_parquet(a.maps)
        # 2 layouts hold a map. The gate writes `target`/`causal`; the
        # causal_maps run writes `image_id`/`values`. Read either one.
        idc = "target" if "target" in df.columns else "image_id"
        vc = "causal" if "causal" in df.columns else "values"
        n_cells = len(np.asarray(df.iloc[0][vc]).ravel())
        stats = stats_for(df[idc].unique(), a.case, n_cells,
                          seed=a.seed)
        sign = -1.0 if a.negate else 1.0
        rows = []
        for image_id, st in stats.items():
            v = sign * st[a.rank].to_numpy(float)
            rows.append({"image_id": image_id,
                         "values": pool(v, a.group).tolist()})
        out = Path(a.out or f"outputs/_monocle/rankings/saliency_{a.rank}.parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"wrote {out}: {len(rows)} images, feature={a.rank} "
              f"sign={'-' if a.negate else '+'} group={a.group}")
        return 0

    raise SystemExit("give --screen or --rank FEATURE")


if __name__ == "__main__":
    raise SystemExit(main())
