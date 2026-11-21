"""Do the negative poles of 7 DIFFERENT questions land on the same patch?

`causal_maps --question-case` measures the 256-cell map of ONE photograph
under each of the 7 questions. The residual is question-blind, so any
difference between the maps comes from the gradient alone.

The hypothesis: 1 question-general negative feature. It predicts that the
NEGATIVE TAILS of the 7 maps coincide MORE than the whole maps correlate.

| Statistic | What it is |
|---|---|
| observed | where question B ranks the cell that question A calls worst |
| bulk null | the same, for 2 maps that share only the whole-map correlation |
| scene null | the same, when B is a DIFFERENT photograph |

The bulk null is the one that matters. Two maps that correlate at r=0.3 will
agree about the worst cell more often than chance, with no shared feature at
all. Only an observed value BELOW the bulk null is evidence of a shared tail.

Usage:
    python scripts/xq_pole_coincidence.py --root outputs/_monocle/xq30
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def load(root: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for f in sorted(glob.glob(str(root / "*" / "*_absent" / "maps.parquet"))):
        q = Path(f).parents[1].name
        df = pd.read_parquet(f)
        col = "values" if "values" in df.columns else "causal"
        out[q] = {str(r.image_id): np.asarray(r[df.columns.get_loc(col) + 1],
                                              dtype=float).ravel()
                  for r in df.itertuples()}
    return out


def pole_pct(a: np.ndarray, b: np.ndarray) -> float:
    """Where b ranks the cell a calls MOST negative. 0 = b agrees, 0.5 = chance."""
    cell = int(np.argmin(a))
    return float((rankdata(b)[cell] - 1) / (b.size - 1))


def bulk_null(r: float, n: int, draws: int, rng) -> float:
    """The same statistic for 2 maps whose ONLY tie is a correlation of r."""
    r = float(np.clip(r, -0.999, 0.999))
    vals = np.empty(draws)
    for i in range(draws):
        x = rng.standard_normal(n)
        y = r * x + np.sqrt(1 - r * r) * rng.standard_normal(n)
        vals[i] = pole_pct(x, y)
    return float(vals.mean())


def copula_null(A: np.ndarray, B: np.ndarray, draws: int, rng) -> float:
    """A fairer null: keep BOTH maps' own values, and only their correlation.

    A causal map has a heavy tail, and the Gaussian null does not. Two heavy
    tails that correlate agree about the worst cell more often than 2 normal
    ones do. So this null draws a correlated Gaussian pair, and then puts the
    REAL values of each map back by rank. The marginal of each surrogate is
    thus exactly the marginal of the map it stands for, and the only tie left
    between them is the rank correlation.
    """
    n = A.size
    r = float(np.clip(np.corrcoef(rankdata(A), rankdata(B))[0, 1],
                      -0.999, 0.999))
    a_sorted, b_sorted = np.sort(A), np.sort(B)
    vals = np.empty(draws)
    for i in range(draws):
        x = rng.standard_normal(n)
        y = r * x + np.sqrt(1 - r * r) * rng.standard_normal(n)
        xa = a_sorted[np.argsort(np.argsort(x))]
        yb = b_sorted[np.argsort(np.argsort(y))]
        vals[i] = pole_pct(xa, yb)
    return float(vals.mean())


def tail_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """How many of the k most-negative cells the 2 maps share, of k."""
    return len(set(np.argsort(a)[:k]) & set(np.argsort(b)[:k])) / k


def tail_null(A: np.ndarray, B: np.ndarray, k: int, draws: int,
              rng) -> float:
    """The copula surrogate, scored on the whole tail and not on 1 cell."""
    n = A.size
    r = float(np.clip(np.corrcoef(rankdata(A), rankdata(B))[0, 1],
                      -0.999, 0.999))
    a_sorted, b_sorted = np.sort(A), np.sort(B)
    vals = np.empty(draws)
    for i in range(draws):
        x = rng.standard_normal(n)
        y = r * x + np.sqrt(1 - r * r) * rng.standard_normal(n)
        vals[i] = tail_overlap(a_sorted[np.argsort(np.argsort(x))],
                               b_sorted[np.argsort(np.argsort(y))], k)
    return float(vals.mean())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/_monocle/xq30")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--tail-k", nargs="+", type=int, default=[4, 10, 25],
                    help="the size of the negative tail to compare")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="outputs/_monocle/xq30/coincidence.csv")
    a = ap.parse_args(argv)

    qs = load(Path(a.root))
    names = sorted(qs)
    keys = sorted(set.intersection(*[set(v) for v in qs.values()]))
    print(f"{len(names)} questions, {len(keys)} photographs: {names}\n")
    rng = np.random.default_rng(a.seed)

    rows = []
    for i, qa in enumerate(names):
        for j, qb in enumerate(names):
            if i == j:
                continue
            for k in keys:
                A, B = qs[qa][k], qs[qb][k]
                r = float(np.corrcoef(A, B)[0, 1])
                alt = keys[int(rng.integers(len(keys)))]
                rows.append({
                    "qa": qa, "qb": qb, "image_id": k, "r": r,
                    "obs": pole_pct(A, B),
                    "bulk": bulk_null(r, A.size, a.draws // 40, rng),
                    "copula": copula_null(A, B, a.draws // 40, rng),
                    "scene": (pole_pct(A, qs[qb][alt]) if alt != k else np.nan),
                    "same_cell": float(np.argmin(A) == np.argmin(B)),
                    **{f"tail{k}": tail_overlap(A, B, k) for k in a.tail_k},
                    **{f"tail{k}_null": tail_null(A, B, k, a.draws // 40, rng)
                       for k in a.tail_k},
                })
    d = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)

    print("Where the OTHER question ranks this question's most-negative cell")
    print("(0.0 = it agrees exactly, 0.5 = chance)\n")
    print(f"  observed                  {d.obs.mean():.4f}")
    print(f"  bulk-correlation null     {d.bulk.mean():.4f}   (gaussian)")
    print(f"  copula null               {d.copula.mean():.4f}   "
          f"<- the control that matters")
    print(f"  different-photograph null {d.scene.mean():.4f}")
    print(f"\n  observed minus bulk       {d.obs.mean() - d.bulk.mean():+.4f}")
    print(f"  observed minus copula     {d.obs.mean() - d.copula.mean():+.4f}")

    # Paired over photographs, which is the cluster unit.
    per = d.groupby("image_id").apply(
        lambda g: g.obs.mean() - g.copula.mean(), include_groups=False)
    rng2 = np.random.default_rng(a.seed + 1)
    boot = [per.sample(len(per), replace=True, random_state=int(
        rng2.integers(1 << 31))).mean() for _ in range(4000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  95% interval over photographs, against the copula null "
          f"[{lo:+.4f}, {hi:+.4f}]")
    print(f"\n  exact same worst cell     {d.same_cell.mean():.4f}  "
          f"(chance 1/256 = 0.0039)")

    print("\nThe WHOLE negative tail, which has far more power than 1 cell")
    print(f"  {'k':>4} {'shared':>8} {'copula null':>12} {'excess':>9} "
          f"{'95% interval':>22}")
    for k in a.tail_k:
        obs_k = d[f"tail{k}"]
        nul_k = d[f"tail{k}_null"]
        per_k = d.assign(_d=obs_k - nul_k).groupby("image_id")["_d"].mean()
        rng3 = np.random.default_rng(a.seed + 7)
        bk = [per_k.sample(len(per_k), replace=True,
                           random_state=int(rng3.integers(1 << 31))).mean()
              for _ in range(3000)]
        l2, h2 = np.percentile(bk, [2.5, 97.5])
        print(f"  {k:>4} {obs_k.mean():>8.4f} {nul_k.mean():>12.4f} "
              f"{obs_k.mean() - nul_k.mean():>+9.4f} "
              f"  [{l2:+.4f}, {h2:+.4f}]")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
