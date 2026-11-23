"""Three cheap reads of the k=1 reader, from data already on disk.

1. The sign split. The separation adds a high half and a low half. We
   measure each half against the average cell, so we can see which end
   carries the gain.
2. The flip rate. Delete the cell the reader names and ask if the model
   changes its answer. That is the decision the judgment makes.
3. Two readers as one. `gxa` and `cheap` barely overlap, so their mean
   rank may beat both.

    python research/scripts/k1_sign_and_flip.py \
        --root outputs/_monocle/gateB60_g1L \
        --xq-root outputs/_monocle/gateB60_g1L_xq \
        --cases restaurants libraries subway_safety
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.scripts.k1_separation import bh, boot_ci  # noqa: E402

ARMS = ("gxa", "cheap")


def pooled(path: Path, arms) -> dict[str, dict]:
    """The mean rank over every layer, for each arm and each photograph."""
    df = pd.read_parquet(path)
    out: dict[str, dict] = {}
    for key, g in df.groupby("target"):
        t = np.asarray(g.iloc[0]["causal"], dtype=float).ravel()
        rec: dict = {"truth": t}
        for arm in arms:
            rk = []
            for _, r in g.iterrows():
                v = r.get(arm)
                if v is None:
                    continue
                v = np.asarray(v, dtype=float).ravel()
                if v.size == t.size and np.isfinite(v).all():
                    rk.append(rankdata(v))
            if rk:
                rec[arm] = np.mean(rk, axis=0)
        out[str(key)] = rec
    return out


def halves(values: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """The high half and the low half of the separation, each against
    the average cell of the same photograph."""
    sd = float(np.std(truth, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan"), float("nan")
    mu = float(truth.mean())
    hi = (float(truth[int(np.argmax(values))]) - mu) / sd
    lo = (mu - float(truth[int(np.argmin(values))])) / sd
    return hi, lo


def flip(values: np.ndarray, truth: np.ndarray, c0: float) -> bool:
    """True when the deletion of the named cell changes the answer.

    `truth` holds a contribution, so a deletion moves c by -truth[p].
    The cell that holds the answer up is the top cell when c0 is
    positive, and the bottom cell when c0 is negative.
    """
    p = int(np.argmax(values)) if c0 > 0 else int(np.argmin(values))
    return (c0 - float(truth[p])) * c0 < 0


def rand_flip(truth: np.ndarray, c0: float, rng, n: int) -> float:
    idx = rng.integers(0, truth.size, size=n)
    return float(np.mean((c0 - truth[idx]) * c0 < 0))


def c0_of(root: Path, case: str) -> dict[str, float]:
    p = root / case / "agreement.parquet"
    d = pd.read_parquet(p)
    d = d.dropna(subset=["c0"])
    return {str(k): float(g["c0"].iloc[0]) for k, g in d.groupby("target")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--xq-root", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="outputs/_monocle/k1_sign_and_flip.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root, xq = Path(args.root), Path(args.xq_root)
    sign_rows, flip_rows = [], []

    for case in args.cases:
        a, b = root / case / "maps.parquet", xq / case / "maps.parquet"
        if not (a.exists() and b.exists()):
            print(f"skip {case}: no maps")
            continue
        A, B = pooled(a, ARMS + ("both",)), pooled(b, ARMS + ("both",))
        keys = sorted(set(A) & set(B))
        c0 = c0_of(root, case)

        for rec in (A, B):
            for k in rec:
                if all(x in rec[k] for x in ARMS):
                    rec[k]["both"] = 0.5 * (rec[k]["gxa"] + rec[k]["cheap"])

        for arm in ARMS + ("both",):
            ks = [k for k in keys if arm in A[k] and arm in B[k]]
            if not ks:
                continue
            for name, pick in (("high", 0), ("low", 1)):
                dr = np.array([halves(A[k][arm], A[k]["truth"])[pick]
                               for k in ks])
                dw = np.array([halves(B[k][arm], B[k]["truth"])[pick]
                               for k in ks])
                d = dr - dw
                if not np.isfinite(d).all() or np.allclose(d, 0):
                    continue
                lo, hi = boot_ci(d, args.n_boot, rng, "mean")
                sign_rows.append({
                    "case": case, "arm": arm, "half": name, "n": int(d.size),
                    "real": float(dr.mean()), "wrong": float(dw.mean()),
                    "gain": float(d.mean()), "lo": lo, "hi": hi,
                    "p": float(wilcoxon(d).pvalue)})

            ks2 = [k for k in ks if k in c0]
            if ks2:
                got = np.array([flip(A[k][arm], A[k]["truth"], c0[k])
                                for k in ks2], dtype=float)
                base = np.array([rand_flip(A[k]["truth"], c0[k], rng, 256)
                                 for k in ks2])
                d = got - base
                lo, hi = boot_ci(d, args.n_boot, rng, "mean")
                flip_rows.append({
                    "case": case, "arm": arm, "n": int(d.size),
                    "reader": float(got.mean()), "chance": float(base.mean()),
                    "gain": float(d.mean()), "lo": lo, "hi": hi,
                    "p": float(wilcoxon(d).pvalue)})

    for rows in (sign_rows, flip_rows):
        if rows:
            q = bh(np.array([r["p"] for r in rows]))
            for r, v in zip(rows, q):
                r["p_bh"] = float(v)

    print("=== the sign split: which half carries the gain ===")
    print(f"{'case':>18} {'arm':>6} {'half':>5} {'own':>7} {'wrong':>7} "
          f"{'gain':>7} {'95% CI':>18} {'q':>7}")
    for r in sign_rows:
        print(f"{r['case']:>18} {r['arm']:>6} {r['half']:>5} "
              f"{r['real']:+7.3f} {r['wrong']:+7.3f} {r['gain']:+7.3f} "
              f"[{r['lo']:+6.3f},{r['hi']:+6.3f}] {r['p_bh']:7.4f}")

    print("\n=== the flip rate: does the named cell change the answer ===")
    print(f"{'case':>18} {'arm':>6} {'reader':>7} {'chance':>7} {'gain':>7} "
          f"{'95% CI':>18} {'q':>7}")
    for r in flip_rows:
        print(f"{r['case']:>18} {r['arm']:>6} {r['reader']:7.3f} "
              f"{r['chance']:7.3f} {r['gain']:+7.3f} "
              f"[{r['lo']:+6.3f},{r['hi']:+6.3f}] {r['p_bh']:7.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"sign": sign_rows, "flip": flip_rows}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
