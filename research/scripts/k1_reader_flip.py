"""Does the FITTED reader name a cell whose deletion changes the answer?

A separation is a shift of a logit. A judgment is an ANSWER. The
analytic reader failed this test: it named a cell whose deletion flipped
the answer on 0 to 8 per cent of photographs, and it did not beat a
random cell of the same photograph.

The fitted reader names other cells, so the test must run again. We add
the ORACLE, which is the best a 1-cell reader could ever do: the share
of photographs where SOME single cell flips the answer. Without it a low
rate looks like a weak reader when it may be a hard ceiling.

    python research/scripts/k1_reader_flip.py --cases restaurants libraries \\
        subway_safety schools street_photography parks_plazas road_quality
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.scripts.k1_separation import bh, boot_ci  # noqa: E402
from research.scripts import k1_fitted_probe as fp  # noqa: E402


def flips(values: np.ndarray, truth: np.ndarray, c0: float) -> bool:
    p = int(np.argmax(values)) if c0 > 0 else int(np.argmin(values))
    return bool((c0 - float(truth[p])) * c0 < 0)


def oracle(truth: np.ndarray, c0: float) -> bool:
    return bool(np.any((c0 - truth) * c0 < 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--cells", default="outputs/_monocle/reader_cells")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["self", "shared"])
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n-rand", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="outputs/_monocle/k1_reader_flip.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for case in args.cases:
        A = fp.load_acts(Path(args.own), case, args.layer, "gh")
        if not A:
            print(f"skip {case}: no activations", flush=True)
            continue
        ag = pd.read_parquet(Path(args.own) / case / "agreement.parquet")
        ag = ag.dropna(subset=["c0"])
        idc = "target" if "target" in ag.columns else "image_id"
        c0 = {str(k): float(g["c0"].iloc[0]) for k, g in ag.groupby(idc)}

        for mode in args.modes:
            path = Path(args.cells) / f"{case}_{mode}" / "maps.parquet"
            if not path.exists():
                continue
            mp = pd.read_parquet(path)
            got, rnd, orc = [], [], []
            for _, r in mp.iterrows():
                k = str(r["image_id"])
                if k not in A or k not in c0:
                    continue
                v = np.asarray(r["causal"], dtype=float).ravel()
                t = np.asarray(A[k][1], dtype=float).ravel()
                c = c0[k]
                got.append(float(flips(v, t, c)))
                idx = rng.integers(0, t.size, size=args.n_rand)
                rnd.append(float(np.mean((c - t[idx]) * c < 0)))
                orc.append(float(oracle(t, c)))
            if not got:
                continue
            d = np.asarray(got) - np.asarray(rnd)
            lo, hi = boot_ci(d, args.n_boot, rng, "mean")
            try:
                pv = float(wilcoxon(d).pvalue)
            except ValueError:
                pv = 1.0
            rows.append({"case": case, "mode": mode, "n": len(got),
                         "reader": float(np.mean(got)),
                         "random": float(np.mean(rnd)),
                         "oracle": float(np.mean(orc)),
                         "gain": float(d.mean()), "lo": lo, "hi": hi,
                         "p": pv})
            print(f"{case:>18} {mode:>7} reader {rows[-1]['reader']:.3f} "
                  f"random {rows[-1]['random']:.3f} "
                  f"oracle {rows[-1]['oracle']:.3f}", flush=True)

    if rows:
        q = bh(np.array([r["p"] for r in rows]))
        for r, v in zip(rows, q):
            r["p_bh"] = float(v)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
