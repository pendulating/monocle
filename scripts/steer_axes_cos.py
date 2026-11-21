"""Step 3: one axis or seven, and is the axis a picture statistic.

Part A reads the 7 cases' judgment axes from the 4-arm sweeps and prints
the cosine matrix per block, with the pinv vectors' matrix beside it. Part
B tests the axis coordinate of each eval image against 4 plain picture
statistics (brightness, contrast, edge energy, saturation): the
correlation of each statistic with the coordinate and with the judgment,
and the partial correlation of the coordinate with the judgment once the
4 statistics are taken out.

Usage:
    python scripts/steer_axes_cos.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from k1_image_stats import cell_stats  # noqa: E402

AXIS_ROOT = REPO / "outputs/_monocle/single_steer_axis"
COORD_ROOT = REPO / "outputs/_monocle/steer_coord"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
STATS = ("lum", "contrast", "edge", "sat")


def log(m: str) -> None:
    print(f"[steer-axes-cos] {m}", flush=True)


def cos_matrix(vecs: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(vecs)
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = float(vecs[a] @ vecs[b])
    return pd.DataFrame(M, index=names, columns=names)


def part_a(layers: list[int]) -> None:
    for l in layers:
        axes, pinvs = {}, {}
        for case in CASES:
            f = AXIS_ROOT / case / "vectors.npz"
            if not f.exists():
                continue
            v = np.load(f)
            axes[case] = v[f"v_axis_L{l}"]
            pinvs[case] = v[f"v_signal_L{l}"]
        if len(axes) < 2:
            log(f"L{l}: fewer than 2 cases have vectors")
            continue
        print(f"\nAXIS cosine matrix, L{l}")
        print(cos_matrix(axes).round(3).to_string())
        print(f"\nPINV cosine matrix, L{l}")
        print(cos_matrix(pinvs).round(3).to_string())
        off = cos_matrix(axes).to_numpy()[np.triu_indices(len(axes), 1)]
        print(f"axis off-diagonal: median {np.median(off):.3f}, "
              f"min {off.min():.3f}, max {off.max():.3f}")


def image_stats(path: str, cache: dict) -> np.ndarray:
    if path not in cache:
        st = cell_stats(path, 16)
        cache[path] = np.array([float(np.mean(st[k])) for k in STATS])
    return cache[path]


def part_b(layer: int) -> pd.DataFrame:
    rows = []
    cache: dict = {}
    for case in CASES:
        vf = AXIS_ROOT / case / "vectors.npz"
        nf = COORD_ROOT / case / "natural.parquet"
        af = COORD_ROOT / case / "natural_acts.npy"
        if not (vf.exists() and nf.exists() and af.exists()):
            continue
        v = np.load(vf)
        layers = [int(x) for x in v["layers"]]
        j = layers.index(layer)
        axis = v[f"v_axis_L{layer}"]
        nat = pd.read_parquet(nf)
        acts = np.load(af).astype(np.float64)[:, j, :]
        s = (acts - acts.mean(0)) @ axis
        y = nat["logit_diff"].to_numpy(dtype=np.float64)
        X = np.stack([image_stats(p, cache) for p in nat["image_path"]])
        row = {"case": case, "layer": layer,
               "r_axis_judgment": float(sps.pearsonr(s, y)[0])}
        for k, name in enumerate(STATS):
            row[f"r_{name}_axis"] = float(sps.pearsonr(X[:, k], s)[0])
            row[f"r_{name}_judgment"] = float(sps.pearsonr(X[:, k], y)[0])
        # Take the 4 statistics out of both, then correlate the residuals.
        Xd = np.column_stack([np.ones(len(y)), X])
        beta_s, *_ = np.linalg.lstsq(Xd, s, rcond=None)
        beta_y, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        r_s, r_y = s - Xd @ beta_s, y - Xd @ beta_y
        row["partial_r_axis_judgment"] = float(sps.pearsonr(r_s, r_y)[0])
        row["r2_stats_on_judgment"] = float(
            1 - np.var(r_y) / np.var(y))
        rows.append(row)
        log(f"{case}: stats done")
    return pd.DataFrame(rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="41,42,44,45")
    ap.add_argument("--stats-layer", type=int, default=41)
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")]
    part_a(layers)
    df = part_b(args.stats_layer)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"\nPICTURE STATISTICS against the axis coordinate, L{args.stats_layer}")
    print(df.round(3).to_string(index=False))
    df.to_parquet(AXIS_ROOT / "axis_image_stats.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
