"""Step 1: the judgment axis against the recorded residuals (CPU only).

The axis is the unit regression direction of the judgment on the answer
residual. The GPU run recorded that residual for the 60 eval images only,
so this script fits the axis on one half of them and reads the other half,
both ways. Per case and per layer it reports:

| Column | Meaning |
|---|---|
| ``cos_axis_folds`` | the cosine of the 2 fold axes, the stability of the fit |
| ``cos_axis_v`` | the cosine of the axis with the pinv steering vector |
| ``natural_slope`` | held-out regression slope of the judgment on the axis coordinate, logits per unit |
| ``natural_r`` | its correlation |
| ``cell_slope`` | pooled slope of the c-map on the axis-map over the held-out cells |
| ``cell_r_median`` | the per-image correlation of the 2 maps, median |
| ``sd_axis`` | held-out spread of the images along the axis, in units |
| ``gain_v`` | the causal gain of the pinv vector from the sweep, for scale |

If the axis is the carrier, ``natural_slope`` and ``cell_slope`` agree,
and the injection gain along the axis in step 2 lands on the same number.

Usage:
    python research/scripts/steer_axis_cpu.py --cases all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle.steer_coord import contribution_sign, delta_map  # noqa: E402
from scripts.steer_coord_report import CASES, steering_gain  # noqa: E402

ROOT_DEFAULT = REPO / "outputs/_monocle/steer_coord"
SWEEP_DEFAULT = REPO / "outputs/_monocle/single_steer_fine"


def log(m: str) -> None:
    print(f"[steer-axis-cpu] {m}", flush=True)


def axis_of(h: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit regression direction, the same as `single_steer.judgment_axis`."""
    yc = y - y.mean()
    m = (yc[:, None] * (h - h.mean(0, keepdims=True))).sum(0)
    return m / np.linalg.norm(m)


def score_case(case_dir: Path, sweep: Optional[pd.DataFrame]) -> list[dict]:
    vecs = dict(np.load(case_dir / "vectors.npz"))
    layers = [int(l) for l in vecs["layers"]]
    nat = pd.read_parquet(case_dir / "natural.parquet")
    nat_acts = np.load(case_dir / "natural_acts.npy").astype(np.float64)
    maps = pd.read_parquet(case_dir / "maps.parquet")
    acts = np.load(case_dir / "answer_acts.npy", mmap_mode="r")
    y = nat["logit_diff"].to_numpy(dtype=np.float64)
    n = len(y)
    c_maps = np.stack([np.asarray(v, dtype=np.float64) for v in maps["values"]])
    half = np.arange(n) < n // 2
    folds = [(half, ~half), (~half, half)]
    rows = []
    for j, l in enumerate(layers):
        h = nat_acts[:, j, :]
        a = np.asarray(acts[:, :, j, :], dtype=np.float32)
        axes = [axis_of(h[fit], y[fit]) for fit, _ in folds]
        nat_slopes, nat_rs, sd_axis = [], [], []
        cell_slopes, cell_rs = [], []
        for (fit, held), m in zip(folds, axes):
            hc = h[held] - h[held].mean(0, keepdims=True)
            s = hc @ m
            nat_slopes.append(float(np.polyfit(s, y[held], 1)[0]))
            nat_rs.append(float(sps.pearsonr(s, y[held])[0]))
            sd_axis.append(float(s.std(ddof=1)))
            s_all, c_all = [], []
            for i in np.flatnonzero(held):
                s_map = contribution_sign(delta_map(a[i], m))
                cell_rs.append(float(sps.pearsonr(s_map, c_maps[i])[0]))
                s_all.append(s_map)
                c_all.append(c_maps[i])
            s_all = np.concatenate(s_all)
            c_all = np.concatenate(c_all)
            cell_slopes.append(float(np.polyfit(s_all, c_all, 1)[0]))
        v = vecs[f"v_signal_L{l}"]
        row = {
            "case": case_dir.name, "layer": l,
            "cos_axis_folds": float(axes[0] @ axes[1]),
            "cos_axis_v": float(np.mean([m @ v for m in axes])),
            "natural_slope": float(np.mean(nat_slopes)),
            "natural_r": float(np.mean(nat_rs)),
            "sd_axis": float(np.mean(sd_axis)),
            "cell_slope": float(np.mean(cell_slopes)),
            "cell_r_median": float(np.median(cell_rs)),
            "scale": float(vecs["scales"][j]),
        }
        if sweep is not None:
            row["gain_v"] = steering_gain(sweep, l, row["scale"])["signal"]["gain"]
        rows.append(row)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    ap.add_argument("--sweep-dir", default=str(SWEEP_DEFAULT))
    ap.add_argument("--cases", default="all")
    args = ap.parse_args(argv)
    root = Path(args.root)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rows = []
    for case in cases:
        case_dir = root / case
        if not (case_dir / "answer_acts.npy").exists():
            log(f"{case}: no run")
            continue
        sweep_path = Path(args.sweep_dir) / case / "steer_results.parquet"
        sweep = pd.read_parquet(sweep_path) if sweep_path.exists() else None
        rows += score_case(case_dir, sweep)
        log(f"{case}: scored")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nJUDGMENT AXIS, fitted on one half of the eval images, read on the other")
    print(df.round(3).to_string(index=False))
    df.to_parquet(root / "axis_cpu_report.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
