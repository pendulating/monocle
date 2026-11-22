"""Can an instrument NAME the cell that matters most?

Why this statistic
------------------
The judgment saturates: one cell of 64 gives much of what half the photograph
gives. So the question a reader of a lens actually asks is not "what is the
correlation", it is "show me the patch that matters most".

The script reports, for each arm:

| Column | Meaning |
|---|---|
| `top1` | how often the arm's best cell IS the measured best cell |
| `top1_in5` | how often the measured best cell is in the arm's best 5 |
| `rank` | the mean rank the arm gives the measured best cell |
| `p` | a binomial tail against the chance rate |

Chance for `top1` is 1/64. Chance for `rank` is 32.5.

Usage:
    python research/scripts/top_cell_hit.py --maps outputs/_monocle/gateB60/libraries/maps.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("lens_own", "cheap", "gxa", "meang_own", "meangp_own")


def main(argv: list[str] | None = None) -> int:
    from scipy.stats import binomtest

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.maps)
    if args.layer is not None:
        df = df[df.layer == args.layer]
    tc = "causal" if "causal" in df.columns else "values"
    n_cell = len(np.asarray(df.iloc[0][tc]).ravel())
    print(f"{args.label or args.maps}: {len(df)} images, {n_cell} cells")
    print(f"chance top1 = {1 / n_cell:.4f}, chance rank = "
          f"{(n_cell + 1) / 2:.1f}\n")
    print(f"{'arm':>12} {'n':>4} {'top1':>7} {'p':>8} "
          f"{'top' + str(args.top):>7} {'rank':>7} {'best|worst':>11}")
    for arm in args.arms:
        if arm not in df.columns:
            continue
        hits, in_k, ranks, both = 0, 0, [], 0
        n = 0
        for _, r in df.iterrows():
            t = np.asarray(r[tc], float).ravel()
            v = r[arm]
            if v is None:
                continue
            v = np.asarray(v, float).ravel()
            if v.size != t.size or not np.isfinite(v).all():
                continue
            n += 1
            t_best = int(np.argmax(t))
            order = np.argsort(-v)
            hits += int(order[0] == t_best)
            in_k += int(t_best in set(order[:args.top].tolist()))
            ranks.append(int(np.where(order == t_best)[0][0]) + 1)
            # Does it also put the measured WORST cell last?
            both += int(int(np.argsort(v)[0]) == int(np.argmin(t)))
        if n == 0:
            continue
        p = binomtest(hits, n, 1 / n_cell, alternative="greater").pvalue
        print(f"{arm:>12} {n:>4} {hits / n:>7.3f} {p:>8.3f} "
              f"{in_k / n:>7.3f} {np.mean(ranks):>7.1f} "
              f"{both / n:>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
