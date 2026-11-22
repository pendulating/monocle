"""Read Gate B: how well does each instrument predict the input ablation?

Gate B is the content gate. The truth is the measured effect of removing a
patch at the INPUT, so an instrument that wins here reads the picture.

4 statistics are reported, because they answer different questions.

| Statistic | The question |
|---|---|
| `pearson` | does the magnitude track, over all cells? |
| `spearman` | does the ORDER track, over all cells? |
| `top_pearson` | does the magnitude track on the 50 strongest cells? |
| `top_sign` | does the instrument get the direction right there? |

Usage:
    python research/scripts/gate_b_report.py --runs outputs/_monocle/gateB_J ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATS = ("pearson", "spearman", "top_pearson", "top_sign")
ARMS = ("gxa", "cheap", "meang_own", "meangp_own", "lens_own")


def boot_ci(v: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if v.size < 3:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    m = np.mean(v[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(path: Path, stat: str, n_boot: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path / "agreement.parquet")
    rng = np.random.default_rng(seed)
    rows = []
    for layer, g in df.groupby("layer"):
        row = {"layer": int(layer)}
        for arm in ARMS:
            col = f"{arm}_{stat}"
            if col not in g.columns:
                row[arm] = float("nan")
                continue
            v = g[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            row[arm] = float(np.mean(v)) if v.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def best_cell(path: Path, stat: str) -> tuple[str, int, float, tuple]:
    df = pd.read_parquet(path / "agreement.parquet")
    rng = np.random.default_rng(777)
    best = ("", -1, float("-inf"), (float("nan"), float("nan")))
    for arm in ARMS:
        col = f"{arm}_{stat}"
        if col not in df.columns:
            continue
        for layer, g in df.groupby("layer"):
            v = g[col].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size < 3:
                continue
            m = float(np.mean(v))
            if m > best[2]:
                best = (arm, int(layer), m, boot_ci(v, 20000, rng))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--case", default="subway_safety")
    ap.add_argument("--stats", nargs="+", default=list(STATS))
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args(argv)

    for run in args.runs:
        p = Path(run) / args.case
        if not (p / "agreement.parquet").exists():
            print(f"skip {p}: no agreement.parquet")
            continue
        print("=" * 72)
        print(run)
        for stat in args.stats:
            tab = report(p, stat, args.n_boot, args.seed)
            keep = [c for c in tab.columns
                    if c == "layer" or tab[c].notna().any()]
            n_img = int(pd.read_parquet(
                p / "agreement.parquet")["target"].nunique())
            print(f"\n-- {stat} (mean over {n_img} images) --")
            print(tab[keep].to_string(
                index=False, float_format=lambda v: f"{v:+.3f}"))
            arm, layer, m, (lo, hi) = best_cell(p, stat)
            print(f"   best: {arm} at L{layer} = {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
