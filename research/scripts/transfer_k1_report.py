"""Read the k=1 transfer battery.

One row for each case and each map source. The statistic is the paired
`separation = dc_pos - dc_neg` inside a (target, source) pair.

Two things the report holds fixed:

| Column | What it answers |
|---|---|
| `gain_gap` | the separation after the pixel gap is taken out |
| `random_sd` | how far an arbitrary cell moves the judgment |

The pixel gap is the distance between the pasted square and the square it
replaces. A cell can move `c` because it is UNLIKE its neighbourhood. The
report fits `separation` on the gap difference and reads the intercept, which
is the separation at an equal gap.

Usage:
    python research/scripts/transfer_k1_report.py --root outputs/_monocle/transfer_k1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.scripts.k1_separation import bh, boot_ci  # noqa: E402


def gap_adjusted(s: pd.DataFrame) -> tuple[float, float]:
    """The separation at an equal pixel gap, and the slope on the gap."""
    d = (s.gap_pos - s.gap_neg).to_numpy(float)
    y = s.separation.to_numpy(float)
    ok = np.isfinite(d) & np.isfinite(y)
    if ok.sum() < 5 or np.allclose(d[ok], d[ok][0]):
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(d[ok], y[ok], 1)
    return float(intercept), float(slope)


def read(root: Path, n_boot: int, seed: int) -> list[dict]:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for case_dir in sorted(root.glob("*/summary.parquet")):
        s = pd.read_parquet(case_dir)
        name = case_dir.parent.name
        case, _, pick = name.partition("_")
        if case == "subway":
            case, _, pick = name.partition("_")
        sep = s.separation.to_numpy(float)
        ok = np.isfinite(sep)
        if ok.sum() < 5:
            continue
        lo, hi = boot_ci(sep[ok], n_boot, rng, "mean")
        adj, slope = gap_adjusted(s)
        rows.append({
            "run": name, "n": int(ok.sum()),
            "dc_pos": float(s.dc_pos.mean()),
            "dc_neg": float(s.dc_neg.mean()),
            "dc_random": float(s.dc_random.mean()),
            "random_sd": float(s.random_sd.mean()),
            "separation": float(sep[ok].mean()),
            "lo": lo, "hi": hi,
            "share_positive": float((sep[ok] > 0).mean()),
            "gap_adjusted": adj, "gap_slope": slope,
            "p": float(wilcoxon(sep[ok], alternative="greater").pvalue)})
    if rows:
        q = bh(np.asarray([r["p"] for r in rows], dtype=float))
        for r, qq in zip(rows, q):
            r["p_bh"] = float(qq)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = read(Path(args.root), args.n_boot, args.seed)
    hdr = (f"{'run':>26} {'n':>4} {'dc pos':>7} {'dc rnd':>7} {'dc neg':>7} "
           f"{'sep':>7} {'95% CI':>18} {'+ve':>5} {'gap adj':>8} {'q':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = " *" if r.get("p_bh", 1) < 0.05 and r["lo"] > 0 else ""
        print(f"{r['run']:>26} {r['n']:>4} {r['dc_pos']:>+7.3f} "
              f"{r['dc_random']:>+7.3f} {r['dc_neg']:>+7.3f} "
              f"{r['separation']:>+7.3f} "
              f"[{r['lo']:>+6.3f},{r['hi']:>+6.3f}] "
              f"{r['share_positive']:>5.2f} {r['gap_adjusted']:>+8.3f} "
              f"{r.get('p_bh', float('nan')):>8.4f}{mark}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
