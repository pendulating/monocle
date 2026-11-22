"""Does the reader find the content map beyond what the hole map shares?

The absent map and the substitute map correlate at only about 0.25, and
`gxa` beats its control against BOTH. Two readings are open. The reader
may read only the part the 2 maps share, or it may read each map on its
own terms.

The test takes the hole map out of the content map, cell by cell, with 1
regression for each photograph. Then it scores the reader again.

    python research/scripts/k1_hole_content_residual.py \
        --root outputs/_monocle/gateB60_g1L_sub \
        --xq-root outputs/_monocle/gateB60_g1L_sub_xq \
        --other outputs/_monocle/causal_maps60 \
        --cases restaurants libraries
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
from k1_separation import bh, boot_ci, sep_z  # noqa: E402


def pooled(path: Path, arm: str) -> dict[str, tuple]:
    df = pd.read_parquet(path)
    out = {}
    for key, g in df.groupby("target"):
        t = np.asarray(g.iloc[0]["causal"], dtype=float).ravel()
        rk = []
        for _, r in g.iterrows():
            v = r.get(arm)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            if v.size == t.size and np.isfinite(v).all():
                rk.append(rankdata(v))
        if rk:
            out[str(key)] = (np.mean(rk, axis=0), t)
    return out


def other_maps(root: Path, case: str) -> dict[str, np.ndarray]:
    hits = sorted(root.glob(f"{case}_*/maps.parquet"))
    if not hits:
        return {}
    df = pd.read_parquet(hits[0])
    return {str(r["image_id"]): np.asarray(r["values"], dtype=float).ravel()
            for _, r in df.iterrows()}


def resid(t: np.ndarray, o: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(t.size), o])
    beta, *_ = np.linalg.lstsq(A, t, rcond=None)
    return t - A @ beta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--xq-root", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--arm", default="gxa")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out",
                    default="outputs/_monocle/k1_hole_content_residual.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for case in args.cases:
        A = pooled(Path(args.root) / case / "maps.parquet", args.arm)
        B = pooled(Path(args.xq_root) / case / "maps.parquet", args.arm)
        O = other_maps(Path(args.other), case)
        keys = sorted(set(A) & set(B) & set(O))
        if not keys:
            print(f"skip {case}")
            continue
        for name in ("as measured", "the hole map removed"):
            dr, dw = [], []
            for k in keys:
                t = A[k][1]
                if name != "as measured":
                    t = resid(t, O[k])
                dr.append(sep_z(A[k][0], t))
                dw.append(sep_z(B[k][0], t))
            d = np.array(dr) - np.array(dw)
            lo, hi = boot_ci(d, args.n_boot, rng, "mean")
            rows.append({"case": case, "variant": name, "n": int(d.size),
                         "real": float(np.mean(dr)),
                         "wrong": float(np.mean(dw)),
                         "gain": float(d.mean()), "lo": lo, "hi": hi,
                         "p": float(wilcoxon(d).pvalue)})
    q = bh(np.array([r["p"] for r in rows]))
    for r, v in zip(rows, q):
        r["p_bh"] = float(v)

    print(f"{'case':>14} {'variant':>22} {'own':>7} {'wrong':>7} {'gain':>7} "
          f"{'95% CI':>18} {'q':>8}")
    for r in rows:
        print(f"{r['case']:>14} {r['variant']:>22} {r['real']:+7.3f} "
              f"{r['wrong']:+7.3f} {r['gain']:+7.3f} "
              f"[{r['lo']:+6.3f},{r['hi']:+6.3f}] {r['p_bh']:8.4f}")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
