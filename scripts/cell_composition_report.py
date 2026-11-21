"""How the 256 cells compose the whole-image sum: is it long-tailed?

Two per-cell quantities, per image:

| quantity | definition | sum over cells |
|---|---|---|
| first-order projection | <g_l[cell], v_l> with v_l the unit judgment axis at block l | the first-order response to a unit uniform injection |
| removal contribution | -delta of the single-cell removal at 280 tokens | not a sum of anything; the map itself |

Per image the script reads the distribution of the cell values, then takes
the median across images:

| column | meaning |
|---|---|
| top5, top10, top25 | share of sum abs(x) held by the largest k cells |
| gini | Gini of abs(x); 0 = uniform, 1 = one cell |
| pr | participation ratio (sum abs)^2 / (256 sum x^2); 1 = uniform |
| cancel | sum abs(x) / abs(sum x); 1 = one sign, large = the net is a residual |
| frac_pos | share of cells with the sign of the net sum |
| kurt | excess kurtosis of x |
| tail_slope | slope of log abs(x) vs log rank over ranks 1..50 (power-law exponent, negative) |

A random unit vector in place of v_l gives the floor of the projection rows.

Usage:
    .venv-3.12/bin/python scripts/cell_composition_report.py --cases all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path("/share/ju/matt/mllmsci")
GRADS = REPO / "outputs/_monocle/patch_grads"
VECS = REPO / "outputs/_monocle/patch_axis_mean"
REMOVAL = REPO / "outputs/_monocle/patch_budget"
OUT = REPO / "outputs/_monocle/cell_composition"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
LAYERS = [12, 16, 20, 22]


def shape_stats(x: np.ndarray) -> dict:
    a = np.abs(x)
    s = np.sort(a)[::-1]
    tot = s.sum() + 1e-12
    net = x.sum()
    n = len(x)
    cum = np.cumsum(s) / tot
    gini = float((2 * np.sum(np.arange(1, n + 1) * np.sort(a)) / (n * tot)) - (n + 1) / n)
    pr = float(tot ** 2 / (n * (x ** 2).sum() + 1e-12))
    ranks = np.arange(1, 51)
    slope = float(np.polyfit(np.log(ranks), np.log(s[:50] + 1e-12), 1)[0])
    return {
        "top5": float(cum[4]), "top10": float(cum[9]), "top25": float(cum[24]),
        "gini": gini, "pr": pr,
        "cancel": float(tot / (abs(net) + 1e-12)),
        "frac_pos": float((np.sign(x) == np.sign(net)).mean()),
        "kurt": float(sps.kurtosis(x)),
        "tail_slope": slope,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--grads-root", default=str(GRADS))
    ap.add_argument("--vecs-root", default=str(VECS))
    ap.add_argument("--removal-root", default=str(REMOVAL))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows, curves = [], []
    for case in cases:
        g_root = Path(args.grads_root) / case
        s = json.loads((g_root / "summary.json").read_text())
        layers = [int(l) for l in s["layers"]]
        idx = pd.read_parquet(g_root / "index.parquet")
        G = np.load(g_root / "grads.npy", mmap_mode="r")
        vecs = dict(np.load(Path(args.vecs_root) / case / "vectors.npz"))
        rem = pd.read_parquet(Path(args.removal_root) / case / "maps.parquet")
        rem = rem[rem["budget"] == 280].set_index("image_path")

        # removal contribution
        stats_r, curve_r = [], []
        for p in idx["image_path"]:
            if p not in rem.index:
                continue
            x = -np.asarray(rem.loc[p, "values"], dtype=np.float64)
            stats_r.append(shape_stats(x))
            a = np.sort(np.abs(x))[::-1]
            curve_r.append(a / a.sum())
        if stats_r:
            med = pd.DataFrame(stats_r).median().to_dict()
            rows.append({"case": case, "layer": -1, "quantity": "removal",
                         "n": len(stats_r), **med})
            curves.append({"case": case, "quantity": "removal", "layer": -1,
                           "share": np.median(np.stack(curve_r), 0).tolist()})

        for l in LAYERS:
            if l not in layers:
                continue
            j = layers.index(l)
            v = np.asarray(vecs[f"v_axis_L{l}"], dtype=np.float64)
            v = v / np.linalg.norm(v)
            r = rng.standard_normal(v.shape)
            r = r / np.linalg.norm(r)
            for name, vec in [("proj_axis", v), ("proj_random", r)]:
                st, cv = [], []
                for i in range(len(idx)):
                    x = np.asarray(G[i, j], dtype=np.float64) @ vec
                    st.append(shape_stats(x))
                    a = np.sort(np.abs(x))[::-1]
                    cv.append(a / a.sum())
                med = pd.DataFrame(st).median().to_dict()
                rows.append({"case": case, "layer": l, "quantity": name,
                             "n": len(st), **med})
                curves.append({"case": case, "quantity": name, "layer": l,
                               "share": np.median(np.stack(cv), 0).tolist()})
        print(f"[cell-composition] {case}: scored", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "composition.parquet", index=False)
    pd.DataFrame(curves).to_parquet(out / "rank_share.parquet", index=False)
    pd.set_option("display.width", 220)
    cols = ["case", "layer", "quantity", "top5", "top10", "top25", "gini",
            "pr", "cancel", "frac_pos", "kurt", "tail_slope"]
    print(df[cols].round(3).to_string(index=False))
    print(f"\nuniform floor: top5 {5/256:.3f} top10 {10/256:.3f} top25 "
          f"{25/256:.3f} gini 0 pr 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
