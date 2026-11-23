r"""What does the fitted weight do that the plain sum does not?

The fitted reader is `sum_d w_d * g_d * h_d`, which is the analytic
reader `gxa` with the residual basis rescaled by `w`. So the fit does
not add a new mechanism. It only says which of the 3840 dimensions to
listen to.

That makes 3 questions worth a measurement.

| Question | The test |
|---|---|
| how many dimensions does it need? | keep the top m of `\|w\|`, drop the rest |
| does a plain trim do the same work? | drop the m loudest dimensions from `gxa` |
| is `w` the same weight for each case? | the cosine between the cases |

If a small top-m reader keeps the gain, the reader is compact and we can
name its dimensions. If a plain trim of the loudest dimensions recovers
`gxa`, then no fit is needed at all.

    python research/scripts/k1_reader_anatomy.py --cases subway_safety \\
        restaurants libraries schools street_photography --layer 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from research.scripts.k1_separation import bh, boot_ci, sep_z  # noqa: E402
from research.scripts import k1_fitted_probe as fp  # noqa: E402

GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 3840]
TRIM = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]


def pack(block: str, case: str, m: int, zr, zw, rng, n_boot) -> dict:
    d = np.asarray(zr) - np.asarray(zw)
    lo, hi = boot_ci(d, n_boot, rng, "mean")
    try:
        pv = float(wilcoxon(d).pvalue)
    except ValueError:
        pv = 1.0
    return {"block": block, "case": case, "m": int(m), "n": int(d.size),
            "real": float(np.mean(zr)), "wrong": float(np.mean(zw)),
            "gain": float(d.mean()), "lo": lo, "hi": hi, "p": pv}


def sweep(block: str, case: str, grid, weight_of, A, B, keys, rng,
          n_boot) -> list[dict]:
    """Score 1 weight vector for each entry of the grid."""
    out = []
    for m in grid:
        w = weight_of(m)
        zr = [sep_z(fp.predict(A[k][0], w), A[k][1]) for k in keys]
        zw = [sep_z(fp.predict(B[k][0], w), A[k][1]) for k in keys]
        out.append(pack(block, case, m, zr, zw, rng, n_boot))
        print(f"  {case:>18} {block:>18} m={m:>5} "
              f"gain {out[-1]['gain']:+.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--xq", default="outputs/_monocle/acts60_xq")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--alpha-from", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=1e5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out",
                    default="outputs/_monocle/k1_reader_anatomy.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    alpha = {c: args.alpha for c in args.cases}
    for f in args.alpha_from:
        for r in json.load(open(f)):
            if (r["feature"] == "gh" and r["layer"] == args.layer
                    and r["split"] == "leave 1 out"):
                alpha[r["case"]] = float(r["alpha"])

    rows, W, D = [], {}, {}
    for c in args.cases:
        A = fp.load_acts(Path(args.own), c, args.layer, "gh")
        B = fp.load_acts(Path(args.xq), c, args.layer, "gh")
        keys = sorted(set(A) & set(B))
        if not keys:
            print(f"skip {c}: no activations", flush=True)
            continue
        w = fp.Gram(A, keys).fit(alpha[c])
        W[c] = w
        d = w.size

        # the per-dimension loudness of `g * h`, pooled over photographs
        sq = np.zeros(d)
        n = 0
        for k in keys:
            X = fp.prep(A[k][0])
            sq += (X * X).sum(0)
            n += X.shape[0]
        sd = np.sqrt(sq / max(n - 1, 1))
        D[c] = sd

        # 1. keep the top m dimensions of the fitted weight
        order = np.argsort(-np.abs(w * sd))

        def keep(m, w=w, order=order, d=d):
            v = np.zeros(d)
            v[order[:m]] = w[order[:m]]
            return v

        rows += sweep("top m of w", c, GRID, keep, A, B, keys, rng,
                      args.n_boot)

        # 2. the plain sum, with the m loudest dimensions dropped
        loud = np.argsort(-sd)

        def trim(m, d=d, loud=loud):
            v = np.ones(d)
            v[loud[:m]] = 0.0
            return v

        rows += sweep("gxa, m loudest out", c, TRIM, trim, A, B, keys,
                      rng, args.n_boot)

    # 3. is the weight the same for each case?
    cos = []
    cs = sorted(W)
    for i, a in enumerate(cs):
        for b in cs[i + 1:]:
            u, v = W[a], W[b]
            cos.append({"a": a, "b": b,
                        "cosine": float(u @ v / (np.linalg.norm(u)
                                                 * np.linalg.norm(v))),
                        "spearman": float(spearmanr(u, v).statistic)})

    for c in W:
        w2 = W[c] ** 2
        part = float(w2.sum() ** 2 / (w2 ** 2).sum())
        cos.append({"a": c, "b": c, "participation": part})

    q = bh(np.array([r["p"] for r in rows]))
    for r, v in zip(rows, q):
        r["p_bh"] = float(v)
    out = {"rows": rows, "weights": cos, "layer": args.layer,
           "alpha": alpha}
    Path(args.out).write_text(json.dumps(out, indent=1))

    print("\nthe cosine between the fitted weights")
    for r in cos:
        if "cosine" in r:
            print(f"  {r['a']:>18} {r['b']:>18} "
                  f"cos {r['cosine']:+.3f} rho {r['spearman']:+.3f}")
        else:
            print(f"  {r['a']:>18} participation ratio "
                  f"{r['participation']:.1f} of {W[r['a']].size}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
