"""Does the 1-cell reading survive when the layout goes?

The worry
---------
Some cells of a 16x16 grid matter more than others in EVERY photograph. The
bottom rows hold the pavement, the top rows hold the sky. An instrument that
learns only that shape would name the poles of an average photograph, and the
separation would look like a read of the picture.

The control
-----------
The position prior of a photograph is the mean measured map of the OTHER 59
photographs, cell by cell. Subtract it, and what is left is the part of the
truth that belongs to this photograph alone.

| Variant | Truth | Instrument |
|---|---|---|
| `raw` | as measured | as measured |
| `truth_resid` | prior removed | as measured |
| `both_resid` | prior removed | own prior removed |

The wrong-question control runs beside every variant, so each row still
reports a gain, not a bare number.

Usage:
    python research/scripts/k1_position_control.py \
        --root outputs/_monocle/gateB60_g1L \
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
from k1_separation import bh, boot_ci, sep_z  # noqa: E402


def loo_prior(stack: np.ndarray) -> np.ndarray:
    """The mean of the OTHER rows, for each row.

    A prior that used the row itself would remove part of the signal it is
    supposed to leave alone.
    """
    n = stack.shape[0]
    if n < 2:
        return np.zeros_like(stack)
    total = stack.sum(axis=0, keepdims=True)
    return (total - stack) / (n - 1)


def variants(A: dict[str, tuple], keys: list[str]) -> dict[str, np.ndarray]:
    """The per-image separation under each variant."""
    inst = np.stack([A[k][0] for k in keys]).astype(float)
    truth = np.stack([A[k][1] for k in keys]).astype(float)
    tr = truth - loo_prior(truth)
    ir = inst - loo_prior(inst)
    return {
        "raw": np.array([sep_z(inst[i], truth[i]) for i in range(len(keys))]),
        "truth_resid": np.array([sep_z(inst[i], tr[i])
                                 for i in range(len(keys))]),
        "both_resid": np.array([sep_z(ir[i], tr[i])
                                for i in range(len(keys))]),
    }


def report(root: Path, xq_root: Path, cases: list[str], n_boot: int,
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
            keys = sorted(set(A) & set(B))
            if len(keys) < 5:
                continue
            va, vb = variants(A, keys), variants(B, keys)
            for name in ("raw", "truth_resid", "both_resid"):
                zr, zw = va[name], vb[name]
                d = zr - zw
                ok = np.isfinite(d)
                if ok.sum() < 5 or np.allclose(d[ok], 0):
                    continue
                lo, hi = boot_ci(d[ok], n_boot, rng, "mean")
                rows.append({
                    "case": case, "arm": arm, "variant": name,
                    "n": int(ok.sum()), "real": float(zr[ok].mean()),
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
    ap.add_argument("--n-boot", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = report(Path(args.root), Path(args.xq_root), args.cases,
                  args.n_boot, args.seed)
    hdr = (f"{'case':>14} {'arm':>11} {'variant':>12} {'own q':>7} "
           f"{'other q':>8} {'gain':>7} {'95% CI':>18} {'q':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mark = " *" if r.get("p_bh", 1) < 0.05 and r["lo"] > 0 else ""
        print(f"{r['case']:>14} {r['arm']:>11} {r['variant']:>12} "
              f"{r['real']:>+7.3f} {r['wrong']:>+8.3f} {r['gain']:>+7.3f} "
              f"[{r['lo']:>+6.3f},{r['hi']:>+6.3f}] "
              f"{r.get('p_bh', float('nan')):>8.4f}{mark}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
