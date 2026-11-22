"""The photograph-permutation null, for the GAIN and not for `real`.

`k1_fitted_probe.py` reports `null_mean`, which is a null for the `real`
column alone. That column has a positive null wherever a case has a
strong mean map: a probe fitted on a shuffled truth learns the average
map, and the average map separates any truth map that looks like it.

The statistic under test is the GAIN, which is `real` minus `wrong`. The
own features and the wrong-question features both carry the average map,
so the gain under the null must be 0. This script measures it for a run
that is already on disk, and it does not repeat the fits of that run.

    python research/scripts/k1_null_gain.py --json outputs/_monocle/k1_fitted_probe_subway.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_fitted_probe import Gram, load_acts, predict  # noqa: E402
from k1_separation import sep_z  # noqa: E402


def null_gain(case: str, layer: int, feat: str, alpha: float, own: Path,
              xq: Path, rng, n_null: int, n_sub: int) -> dict:
    """Shuffle which truth map goes with which photograph, then rescore."""
    A = load_acts(own, case, layer, feat)
    B = load_acts(xq, case, layer, feat)
    keys = sorted(set(A) & set(B))
    sub = keys[:n_sub]
    nreal, ngain = [], []
    for _ in range(n_null):
        sh = list(keys)
        rng.shuffle(sh)
        swap = {k: A[v][1] for k, v in zip(keys, sh)}
        fake = {k: (A[k][0], swap[k]) for k in keys}
        G = Gram(fake, keys)
        gr, gg = [], []
        for k in sub:
            w = G.fit(alpha, drop=k)
            zr = sep_z(predict(fake[k][0], w), swap[k])
            zw = sep_z(predict(B[k][0], w), swap[k])
            gr.append(zr)
            gg.append(zr - zw)
        nreal.append(float(np.nanmean(gr)))
        ngain.append(float(np.nanmean(gg)))
    return {
        "case": case, "layer": layer, "feature": feat, "alpha": alpha,
        "n_null": n_null, "n_photographs": len(sub),
        "null_real": float(np.mean(nreal)),
        "null_gain": float(np.mean(ngain)),
        "null_gain_sd": float(np.std(ngain, ddof=1)) if n_null > 1 else 0.0,
        "draws": ngain,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True,
                    help="the probe files whose rows give case, layer, alpha")
    ap.add_argument("--own", default="outputs/_monocle/acts60")
    ap.add_argument("--xq", default="outputs/_monocle/acts60_xq")
    ap.add_argument("--feature", default="gh")
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--n-sub", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="outputs/_monocle/k1_null_gain.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    want = {}
    for f in args.json:
        for r in json.load(open(f)):
            if r["feature"] != args.feature \
                    or r["split"] != "leave 1 out":
                continue
            want[(r["case"], r["layer"])] = float(r["alpha"])

    out = []
    for (case, layer), alpha in sorted(want.items()):
        row = null_gain(case, layer, args.feature, alpha, Path(args.own),
                        Path(args.xq), rng, args.n_null, args.n_sub)
        out.append(row)
        print(f"{case:>18} L{layer:<3} alpha {alpha:>10.0f}  "
              f"null real {row['null_real']:+.3f}  "
              f"null gain {row['null_gain']:+.3f} "
              f"+- {row['null_gain_sd']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
