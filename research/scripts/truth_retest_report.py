"""Read the map retest, and give the ceiling it implies.

The retest run before 2026-08-25 stored the raw removal delta, while
`causal_maps` stores the contribution, which is the same numbers with the
sign flipped. This report flips the sign when the correlation says it must,
so a run from either side reads the same.

The ceiling
-----------
`sep_z` here uses the fresh map as the instrument and the stored map as the
truth. No instrument can beat a second measurement of the thing itself, so
this number is the honest ceiling of the 1-cell separation.

Usage:
    python research/scripts/truth_retest_report.py --root outputs/_monocle/truth_retest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.scripts.k1_separation import boot_ci  # noqa: E402


def score(fresh: np.ndarray, stored: np.ndarray, flip: bool) -> dict:
    from scipy.stats import pearsonr, spearmanr
    a = -np.asarray(fresh, float) if flip else np.asarray(fresh, float)
    b = np.asarray(stored, float)
    sd = float(np.std(b, ddof=1))
    sep = float("nan")
    if np.isfinite(sd) and sd > 0:
        sep = float(b[int(np.argmax(a))] - b[int(np.argmin(a))]) / sd
    full = (float(b.max() - b.min()) / sd) if sd > 0 else float("nan")
    return {"pearson": float(pearsonr(a, b).statistic),
            "spearman": float(spearmanr(a, b).statistic),
            "top1_same": int(np.argmax(a)) == int(np.argmax(b)),
            "bottom1_same": int(np.argmin(a)) == int(np.argmin(b)),
            "sep_z": sep, "full_range_z": full}


def read(root: Path, n_boot: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for p in sorted(root.glob("*/retest.parquet")):
        df = pd.read_parquet(p)
        raw = [score(r["fresh"], r["stored"], False) for _, r in df.iterrows()]
        flip = float(np.mean([x["pearson"] for x in raw])) < 0
        s = [score(r["fresh"], r["stored"], flip) for _, r in df.iterrows()]
        pear = np.array([x["pearson"] for x in s])
        sep = np.array([x["sep_z"] for x in s])
        full = np.array([x["full_range_z"] for x in s])
        lo, hi = boot_ci(sep, n_boot, rng, "mean")
        rows.append({
            "case": p.parent.name, "n": int(len(df)), "sign_flipped": flip,
            "pearson": float(pear.mean()),
            "spearman": float(np.mean([x["spearman"] for x in s])),
            "top1_same": float(np.mean([x["top1_same"] for x in s])),
            "bottom1_same": float(np.mean([x["bottom1_same"] for x in s])),
            "sep_z": float(sep.mean()), "lo": lo, "hi": hi,
            "full_range_z": float(full.mean()),
            "share_of_full": float(sep.mean() / full.mean())})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    rows = read(Path(args.root), args.n_boot, args.seed)
    hdr = (f"{'case':>14} {'n':>3} {'r':>7} {'rho':>7} {'top1':>6} "
           f"{'bot1':>6} "
           f"{'ceiling':>8} {'95% CI':>18} {'full':>7} {'share':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['case']:>14} {r['n']:>3} {r['pearson']:>+7.3f} "
              f"{r['spearman']:>+7.3f} {r['top1_same']:>6.2f} "
              f"{r['bottom1_same']:>6.2f} {r['sep_z']:>+8.3f} "
              f"[{r['lo']:>+6.3f},{r['hi']:>+6.3f}] "
              f"{r['full_range_z']:>7.3f} {r['share_of_full']:>6.2f}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
