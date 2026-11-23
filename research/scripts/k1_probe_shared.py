"""One shared causal patch reader, and how much data a fit needs.

Two questions that the per-case probe cannot answer.

1. Is there 1 reader or 1 for each judgment? We pool the photographs of
   every case into 1 ridge and leave out 1 photograph at a time. The
   pooled reader never sees the target photograph, and it never sees a
   case-specific weight either. If it reads each case, the reader is
   shared and only the gradient carries the judgment.

2. How many photographs does the fit need? The analytic reader costs 1
   backward pass and nothing else. A fit that needs 60 photographs is a
   different tool from a fit that needs 5. We draw random subsets of a
   size, fit once, and score every photograph outside the subset.

    python research/scripts/k1_probe_shared.py --cases restaurants libraries \\
        subway_safety schools street_photography --layer 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.scripts.k1_separation import bh, boot_ci, sep_z  # noqa: E402
from research.scripts import k1_fitted_probe as fp  # noqa: E402


def score(w, A: dict, B: dict, keys: list[str]) -> tuple:
    """Own separation and wrong-question separation over the keys."""
    zr = np.array([sep_z(fp.predict(A[k][0], w), A[k][1]) for k in keys])
    zw = np.array([sep_z(fp.predict(B[k][0], w), A[k][1]) for k in keys])
    return zr, zw


def pack(name: str, case: str, n_fit: int, zr, zw, rng, n_boot) -> dict:
    d = zr - zw
    lo, hi = boot_ci(d, n_boot, rng, "mean")
    try:
        pv = float(wilcoxon(d).pvalue)
    except ValueError:
        pv = 1.0
    return {"block": name, "case": case, "n_fit": n_fit, "n": int(d.size),
            "real": float(zr.mean()), "wrong": float(zw.mean()),
            "gain": float(d.mean()), "lo": lo, "hi": hi, "p": pv}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--xq", default="outputs/_monocle/acts60_xq")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--feature", default="gh")
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[5, 15, 30, 59])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="outputs/_monocle/k1_probe_shared.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    A, B = {}, {}
    for c in args.cases:
        a = fp.load_acts(Path(args.own), c, args.layer, args.feature)
        b = fp.load_acts(Path(args.xq), c, args.layer, args.feature)
        if not a or not b:
            print(f"skip {c}: no activations", flush=True)
            continue
        for k in sorted(set(a) & set(b)):
            A[f"{c}::{k}"] = a[k]
            B[f"{c}::{k}"] = b[k]
        print(f"loaded {c}: {len(set(a) & set(b))} photographs", flush=True)

    keys = sorted(A)
    rows = []

    # The pooled ridge serves every block. `fit(drop=...)` takes a list,
    # so a whole case leaves the fit with no rebuild.
    alpha = fp.pick_alpha(A, keys, rng)
    print(f"pooled alpha {alpha:.0f} over {len(keys)} photographs",
          flush=True)
    G = fp.Gram(A, keys)

    # 1. leave 1 CASE out: fit on the other judgments, read this one.
    # Nothing about the target case is in the fit. Not its photographs,
    # and not its judgment.
    for c in args.cases:
        ck = [k for k in keys if k.startswith(f"{c}::")]
        rest = [k for k in keys if not k.startswith(f"{c}::")]
        if not ck or not rest:
            continue
        w = G.fit(alpha, drop=ck)
        zr, zw = score(w, A, B, ck)
        rows.append(pack("leave 1 CASE out", c, len(rest), zr, zw, rng,
                         args.n_boot))
        print(f"  {c:>18} case held out  "
              f"gain {rows[-1]['gain']:+.3f}", flush=True)

    # 2. the pooled reader: 1 ridge over every case, 1 photograph out
    per = {c: ([], []) for c in args.cases}
    for k in keys:
        w = G.fit(alpha, drop=k)
        c = k.split("::")[0]
        zr, zw = score(w, A, B, [k])
        per[c][0].append(zr[0])
        per[c][1].append(zw[0])
    for c in args.cases:
        if not per[c][0]:
            continue
        rows.append(pack("pooled, leave 1 out", c, len(keys) - 1,
                         np.array(per[c][0]), np.array(per[c][1]),
                         rng, args.n_boot))
    allr = np.concatenate([np.array(per[c][0]) for c in args.cases
                           if per[c][0]])
    allw = np.concatenate([np.array(per[c][1]) for c in args.cases
                           if per[c][1]])
    rows.append(pack("pooled, leave 1 out", "ALL", len(keys) - 1,
                     allr, allw, rng, args.n_boot))

    # 3. the learning curve, inside each case
    for c in args.cases:
        ck = [k for k in keys if k.startswith(f"{c}::")]
        if not ck:
            continue
        for n_fit in args.sizes:
            if n_fit >= len(ck):
                n_fit = len(ck) - 1
            zr, zw = [], []
            for _ in range(args.draws):
                pick = list(rng.choice(ck, size=n_fit, replace=False))
                rest = [k for k in ck if k not in set(pick)]
                Gs = fp.Gram(A, pick)
                w = Gs.fit(alpha)
                r, wq = score(w, A, B, rest)
                zr.append(r)
                zw.append(wq)
            rows.append(pack("learning curve", c, n_fit,
                             np.concatenate(zr), np.concatenate(zw),
                             rng, args.n_boot))
            print(f"  {c:>18} fit on {n_fit:>3} "
                  f"gain {rows[-1]['gain']:+.3f}", flush=True)

    q = bh(np.array([r["p"] for r in rows]))
    for r, v in zip(rows, q):
        r["p_bh"] = float(v)

    print(f"\n{'block':>22} {'case':>18} {'fit n':>6} {'own':>7} "
          f"{'wrong':>7} {'gain':>7} {'q':>8}")
    for r in rows:
        print(f"{r['block']:>22} {r['case']:>18} {r['n_fit']:>6} "
              f"{r['real']:+7.3f} {r['wrong']:+7.3f} {r['gain']:+7.3f} "
              f"{r['p_bh']:8.4f}")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
