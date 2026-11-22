"""Does the instrument rank the WHOLE map, or only the 2 poles?

Why this test
-------------
The 1-cell separation reads 2 cells of 256. A heat map that a person can use
must agree over the whole grid, not at the ends alone. This script measures
the rank correlation between the instrument and the measured map, cell by
cell, inside each photograph.

The wrong-question control runs beside it. A picture reader keeps its
correlation when the question changes. A judgment reader loses it.

Two statistics, because they fail in different ways:

| Name | What it reads |
|---|---|
| `spearman` | the whole ranking of 256 cells |
| `top16_recall` | how many of the 16 strongest cells the instrument names |

Usage:
    python research/scripts/k1_rank_agreement.py --root outputs/_monocle/gateB60_g1L \
        --xq-root outputs/_monocle/gateB60_g1L_xq
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k1_layer_profile import ARMS, pooled_ranks  # noqa: E402
from k1_separation import bh, boot_ci  # noqa: E402


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def top_recall(values: np.ndarray, truth: np.ndarray, k: int) -> float:
    """The share of the truth's k strongest cells that the instrument names.

    Both ends count. A map of causality must find the cells that hold the
    judgment up AND the cells that push it down, so the set is the k highest
    and the k lowest of each.
    """
    def ends(x: np.ndarray) -> set:
        o = np.argsort(x)
        return set(o[:k].tolist()) | set(o[-k:].tolist())
    want = ends(truth)
    got = ends(values)
    return len(want & got) / len(want)


def scores(A: dict[str, tuple], keys: list[str], k: int) -> dict:
    sp = np.array([spearman(A[key][0], A[key][1]) for key in keys])
    tr = np.array([top_recall(A[key][0], A[key][1], k) for key in keys])
    return {"spearman": sp, "top_recall": tr}


def report(root: Path, xq_root: Path, cases: list[str], k: int, n_boot: int,
           seed: int) -> list[dict]:
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for case in cases:
        a = root / case / "maps.parquet"
        b = xq_root / case / "maps.parquet"
        if not (a.exists() and b.exists()):
            continue
        for arm in ARMS:
            A, B = pooled_ranks(a, arm), pooled_ranks(b, arm)
            common = sorted(set(A) & set(B))
            if len(common) < 5:
                continue
            sa, sb = scores(A, common, k), scores(B, common, k)
            for stat in ("spearman", "top_recall"):
                zr, zw = sa[stat], sb[stat]
                d = zr - zw
                ok = np.isfinite(d)
                if ok.sum() < 5 or np.allclose(d[ok], 0):
                    continue
                lo, hi = boot_ci(d[ok], n_boot, rng, "mean")
                rlo, rhi = boot_ci(zr[ok], n_boot, rng, "mean")
                rows.append({
                    "case": case, "arm": arm, "stat": stat,
                    "n": int(ok.sum()), "real": float(zr[ok].mean()),
                    "real_lo": rlo, "real_hi": rhi,
                    "wrong": float(zw[ok].mean()), "gain": float(d[ok].mean()),
                    "lo": lo, "hi": hi,
                    "p": float(wilcoxon(d[ok],
                                        alternative="greater").pvalue)})
    if rows:
        q = bh(np.asarray([r["p"] for r in rows], dtype=float))
        for r, qq in zip(rows, q):
            r["p_bh"] = float(qq)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--xq-root", required=True)
    ap.add_argument("--cases", nargs="+",
                    default=["restaurants", "libraries", "subway_safety"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = report(Path(args.root), Path(args.xq_root), args.cases, args.k,
                  args.n_boot, args.seed)
    hdr = (f"{'case':>14} {'arm':>11} {'stat':>11} {'own q':>7} "
           f"{'own 95% CI':>18} {'other q':>8} {'gain':>7} {'q':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = " *" if r.get("p_bh", 1) < 0.05 and r["lo"] > 0 else ""
        print(f"{r['case']:>14} {r['arm']:>11} {r['stat']:>11} "
              f"{r['real']:>+7.3f} "
              f"[{r['real_lo']:>+6.3f},{r['real_hi']:>+6.3f}] "
              f"{r['wrong']:>+8.3f} {r['gain']:>+7.3f} "
              f"{r.get('p_bh', float('nan')):>8.4f}{mark}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
