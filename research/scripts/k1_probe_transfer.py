"""Is there 1 causal patch reader, or 1 for each judgment?

The fitted probe learns 3840 weights on the `g * h` of 1 case. This
asks whether those weights read ANOTHER case. If they do, the reader is
generic and a judgment only supplies the gradient. If they do not, the
weights hold something about the judgment itself.

Every test uses the source case's TRAIN photographs for the fit and the
target case's photographs for the score, so no photograph is ever in
both. The control is the target's own wrong-question features.

    python research/scripts/k1_probe_transfer.py --layer 20 \
        --cases restaurants libraries subway_safety
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--xq", default="outputs/_monocle/acts60_xq")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--feature", default="gh")
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--alpha-from", nargs="*", default=[],
                    help="probe files that give each case its own alpha")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out",
                    default="outputs/_monocle/k1_probe_transfer.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Each case keeps the ridge strength that its own probe chose. One
    # shared alpha would make a case transfer badly for the wrong reason.
    alpha = {c: args.alpha for c in args.cases}
    for f in args.alpha_from:
        for r in json.load(open(f)):
            if r["feature"] == args.feature and r["layer"] == args.layer \
                    and r["split"] == "leave 1 out":
                alpha[r["case"]] = float(r["alpha"])

    A, B, W = {}, {}, {}
    for c in args.cases:
        a = fp.load_acts(Path(args.own), c, args.layer, args.feature)
        b = fp.load_acts(Path(args.xq), c, args.layer, args.feature)
        if not a or not b:
            print(f"skip {c}: no activations")
            continue
        A[c], B[c] = a, b
        W[c] = fp.Gram(a, sorted(a)).fit(alpha[c])
        print(f"fitted on {c}: {len(a)} photographs, "
              f"alpha {alpha[c]:.0f}")

    rows = []
    for src in W:
        for tgt in A:
            keys = sorted(set(A[tgt]) & set(B[tgt]))
            if src == tgt:
                # leave 1 out, so the fit never sees the photograph
                G = fp.Gram(A[tgt], keys)
                zr = np.array([sep_z(fp.predict(A[tgt][k][0],
                                                G.fit(alpha[tgt], drop=k)),
                                     A[tgt][k][1]) for k in keys])
                zw = np.array([sep_z(fp.predict(B[tgt][k][0],
                                                G.fit(alpha[tgt], drop=k)),
                                     A[tgt][k][1]) for k in keys])
            else:
                w = W[src]
                zr = np.array([sep_z(fp.predict(A[tgt][k][0], w),
                                     A[tgt][k][1]) for k in keys])
                zw = np.array([sep_z(fp.predict(B[tgt][k][0], w),
                                     A[tgt][k][1]) for k in keys])
            d = zr - zw
            lo, hi = boot_ci(d, args.n_boot, rng, "mean")
            try:
                pv = float(wilcoxon(d).pvalue)
            except ValueError:
                pv = 1.0
            rows.append({"fit_on": src, "score_on": tgt,
                         "alpha": alpha[src], "n": int(d.size),
                         "real": float(zr.mean()), "wrong": float(zw.mean()),
                         "gain": float(d.mean()), "lo": lo, "hi": hi,
                         "p": pv})
    q = bh(np.array([r["p"] for r in rows]))
    for r, v in zip(rows, q):
        r["p_bh"] = float(v)

    print(f"\n{'fit on':>16} {'score on':>16} {'own':>7} {'wrong':>7} "
          f"{'gain':>7} {'95% CI':>18} {'q':>8}")
    for r in rows:
        mark = "  SELF" if r["fit_on"] == r["score_on"] else ""
        print(f"{r['fit_on']:>16} {r['score_on']:>16} {r['real']:+7.3f} "
              f"{r['wrong']:+7.3f} {r['gain']:+7.3f} "
              f"[{r['lo']:+6.3f},{r['hi']:+6.3f}] {r['p_bh']:8.4f}{mark}")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
