"""Average the reader over several wordings of one judgment.

Why
---
`gxa` keeps 88 to 90 per cent of its separation when the question is
reworded. The missing tenth is wording noise. An average over several
wordings of the SAME judgment must remove some of it.

The average is over RANKS, so no wording sets the scale.

The control
-----------
The wrong-judgment side gets the SAME number of wordings, so a gain cannot
come from the count of reads. Both sides grow together, 1 wording at a time,
and the report shows the whole curve.

Usage:
    python research/scripts/k1_wording_ensemble.py --cases restaurants libraries
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_layer_profile import pooled_ranks  # noqa: E402
from k1_separation import bh, boot_ci, sep_z  # noqa: E402

ROOT = Path("outputs/_monocle")
OWN = ["gateB60_g1L", "gateB60_g1L_par", "gateB60_g1L_p2", "gateB60_g1L_p3"]
XQ = ["gateB60_g1L_xq", "gateB60_g1L_xq_p1", "gateB60_g1L_xq_p2",
      "gateB60_g1L_xq_p3"]


def stack(dirs: list[str], case: str, arm: str) -> tuple[list, dict]:
    """The pooled rank map of each wording, and the truth, for each image."""
    got, truth = [], {}
    for d in dirs:
        p = ROOT / d / case / "maps.parquet"
        if not p.exists():
            continue
        m = pooled_ranks(p, arm)
        got.append({k: v[0] for k, v in m.items()})
        for k, v in m.items():
            truth.setdefault(k, v[1])
    return got, truth


def curve(case: str, arm: str, n_boot: int, rng) -> list[dict]:
    from scipy.stats import wilcoxon
    own, truth = stack(OWN, case, arm)
    xq, _ = stack(XQ, case, arm)
    n = min(len(own), len(xq))
    if n == 0:
        return []
    keys = sorted(set(truth))
    for m in own[:n] + xq[:n]:
        keys = [k for k in keys if k in m]
    rows = []
    for w in range(1, n + 1):
        zr = np.array([sep_z(np.mean([m[k] for m in own[:w]], axis=0),
                             truth[k]) for k in keys])
        zw = np.array([sep_z(np.mean([m[k] for m in xq[:w]], axis=0),
                             truth[k]) for k in keys])
        d = zr - zw
        lo, hi = boot_ci(d, n_boot, rng, "mean")
        rows.append({"case": case, "arm": arm, "wordings": w,
                     "n": int(len(keys)), "real": float(zr.mean()),
                     "wrong": float(zw.mean()), "gain": float(d.mean()),
                     "lo": lo, "hi": hi,
                     "p": float(wilcoxon(d, alternative="greater").pvalue)})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+",
                    default=["restaurants", "libraries", "subway_safety"])
    ap.add_argument("--arm", default="gxa")
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    rows = [r for c in args.cases for r in curve(c, args.arm, args.n_boot, rng)]
    if rows:
        q = bh(np.asarray([r["p"] for r in rows], dtype=float))
        for r, qq in zip(rows, q):
            r["p_bh"] = float(qq)
    hdr = (f"{'case':>14} {'wordings':>9} {'n':>4} {'own':>7} {'other':>7} "
           f"{'gain':>7} {'95% CI':>18} {'q':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = " *" if r.get("p_bh", 1) < 0.05 and r["lo"] > 0 else ""
        print(f"{r['case']:>14} {r['wordings']:>9} {r['n']:>4} "
              f"{r['real']:>+7.3f} {r['wrong']:>+7.3f} {r['gain']:>+7.3f} "
              f"[{r['lo']:>+6.3f},{r['hi']:>+6.3f}] "
              f"{r.get('p_bh', float('nan')):>8.4f}{mark}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
