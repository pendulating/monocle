"""Score a lens against the causal ground truth, in the gate's own metric.

One number decides the corpus-faithfulness ladder: how much of a patch's
causal effect the fitted transport predicts. This module computes it, and it
computes the same number for the LOCAL instrument, so every rung reports both
the score and the distance to the local truth.

The metric
----------
Take the top ``k`` patches by absolute causal effect. Score Pearson ``r``
between the causal map and a candidate map on those patches. Take the mean
over the donor-target pairs. This is the metric of the stage-4 table in
`research/docs/r-lens.md`, and ``--verify`` reproduces that table.

The maps
--------
| Column | What it is | Local or fitted |
|---|---|---|
| `causal` | the measured effect of a patch swap | ground truth |
| `taylor` | `<g_p, h_donor - h_p>`, the local Jacobian | local |
| `lens_delta` | `readout(J h_donor) - readout(J h)` | fitted |
| `lens_own` | `readout(J h)`, the map we would render | fitted |

The bias
--------
    bias = r( taylor , lens_delta )   on the same top-k patches

The taylor map is the local truth. Thus `bias` asks how much of it the
constant matrix keeps. A corpus that narrows should raise it.

Usage:
    python -m research.monocle.lens_bias --runs outputs/_monocle/patch_map_causal_J2
    python -m research.monocle.lens_bias --verify
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

#: The top-k of the stage-4 table. Do not change it without re-baselining.
DEFAULT_K = 50

#: Candidate maps, and whether each one reads a local or a fitted instrument.
CANDIDATES = {"taylor": "local", "lens_delta": "fitted", "lens_own": "fitted",
              "meang": "constant", "meangp": "constant"}

#: The stage-4 table, for --verify. See research/docs/r-lens.md.
STAGE4 = {
    "_J2": {12: (0.607, +0.198, 0.520, -0.073),
            18: (0.587, +0.348, 0.477, -0.073),
            24: (0.637, +0.299, 0.537, +0.040)},
    "_R2": {12: (0.607, +0.198, 0.530, +0.098),
            18: (0.587, +0.348, 0.547, +0.076),
            24: (0.637, +0.299, 0.503, -0.013)},
}


def log(m: str) -> None:
    print(f"[lens-bias] {m}", flush=True)


def as_map(value: Any) -> Optional[np.ndarray]:
    """A stored map as a 1-D float array, or None when the column is absent.

    A row whose lens columns are null holds a scalar, not an array. That
    happens at a layer the lens is not fitted at, and it is not an error.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).ravel()
    return None if arr.size < 2 else arr


def top_k_score(
    causal: np.ndarray, candidate: Optional[np.ndarray], k: int = DEFAULT_K,
) -> tuple[float, float]:
    """``(sign agreement, Pearson r)`` on the top ``k`` patches by |causal|."""
    from scipy.stats import pearsonr

    nan = (float("nan"), float("nan"))
    if candidate is None:
        return nan
    c = np.asarray(causal, float).ravel()
    m = np.asarray(candidate, float).ravel()
    if c.size != m.size:
        return nan
    ok = np.isfinite(c) & np.isfinite(m)
    c, m = c[ok], m[ok]
    if c.size < k:
        return nan
    idx = np.argsort(-np.abs(c))[:k]
    c, m = c[idx], m[idx]
    if c.std() == 0 or m.std() == 0:
        return nan
    return float(np.mean(np.sign(c) == np.sign(m))), float(pearsonr(c, m)[0])


def nanmean(values: list[float]) -> float:
    """The mean of the finite values, or NaN when none is finite.

    ``np.nanmean`` warns on an all-NaN slice. That case is normal here: a
    layer the lens is not fitted at has no fitted score.
    """
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def score_maps(df: pd.DataFrame, k: int = DEFAULT_K) -> pd.DataFrame:
    """Per-layer scores for every candidate map, plus the bias to the local."""
    rows = []
    for layer, sub in df.groupby("layer"):
        row: dict[str, Any] = {"layer": int(layer), "n_maps": len(sub)}
        for name in CANDIDATES:
            if name not in sub.columns:
                row[f"{name}_sign"] = row[f"{name}_r"] = float("nan")
                continue
            scores = [top_k_score(r.causal, as_map(getattr(r, name)), k)
                      for _, r in sub.iterrows()]
            row[f"{name}_sign"] = nanmean([s[0] for s in scores])
            row[f"{name}_r"] = nanmean([s[1] for s in scores])
        # The bias: how much of the LOCAL map the fitted transport keeps.
        if {"taylor", "lens_delta"} <= set(sub.columns):
            # top_k_score returns NaN when lens_delta is absent, so a layer
            # the lens does not reach scores NaN rather than a wrong number.
            row["bias_r"] = nanmean(
                [top_k_score(r.taylor, as_map(r.lens_delta), k)[1]
                 for _, r in sub.iterrows()])
        else:
            row["bias_r"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def cell(sign: float, r: float) -> str:
    return "not fitted" if np.isnan(r) else f"{sign:.3f} / {r:+.3f}"


def print_table(name: str, scored: pd.DataFrame) -> None:
    log(f"=== {name} ===")
    cols = [c for c in CANDIDATES if f"{c}_r" in scored.columns
            and not scored[f"{c}_r"].isna().all()]
    log(f"{'layer':>5} {'n':>3}  " + " ".join(f"{c:>15}" for c in cols)
        + f" {'bias':>8}")
    for _, r in scored.iterrows():
        bias = "     --" if np.isnan(r.bias_r) else f"{r.bias_r:+.3f}"
        cells = " ".join(f"{cell(r[f'{c}_sign'], r[f'{c}_r']):>15}"
                         for c in cols)
        log(f"{int(r.layer):>5} {int(r.n_maps):>3}  {cells} {bias:>8}")


def per_pair_r(df: pd.DataFrame, arm: str, k: int = DEFAULT_K) -> pd.DataFrame:
    """``(layer, target, r)`` for one arm, one row per donor-target pair."""
    rows = []
    for _, r in df.iterrows():
        cand = as_map(getattr(r, arm)) if arm in df.columns else None
        rows.append({"layer": int(r.layer), "target": str(r.target),
                     "r": top_k_score(r.causal, cand, k)[1]})
    return pd.DataFrame(rows)


def paired_compare(
    a: pd.DataFrame, b: pd.DataFrame, arm: str, *,
    k: int = DEFAULT_K, n_boot: int = 20000, seed: int = 777,
) -> pd.DataFrame:
    """Compare 2 runs on the SAME pairs, per layer.

    Every rung reads the same causal maps over the same donor-target pairs.
    So the per-pair difference is far tighter than the gap between 2
    intervals, and 2 overlapping intervals do NOT mean the rungs agree. This
    is the test the preregistration commits to.
    """
    rng = np.random.default_rng(seed)
    ra = per_pair_r(a, arm, k).rename(columns={"r": "r_a"})
    rb = per_pair_r(b, arm, k).rename(columns={"r": "r_b"})
    merged = ra.merge(rb, on=["layer", "target"], how="inner")
    out = []
    for layer, sub in merged.groupby("layer"):
        d = (sub.r_b - sub.r_a).to_numpy(float)
        d = d[np.isfinite(d)]
        row = {"layer": int(layer), "n_pairs": int(d.size),
               "mean_a": nanmean(sub.r_a.tolist()),
               "mean_b": nanmean(sub.r_b.tolist())}
        if d.size < 2:
            row.update({"delta": float("nan"), "lo": float("nan"),
                        "hi": float("nan"), "n_up": 0})
        else:
            bs = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row.update({"delta": float(d.mean()), "lo": float(lo),
                        "hi": float(hi), "n_up": int((d > 0).sum())})
        out.append(row)
    return pd.DataFrame(out).sort_values("layer").reset_index(drop=True)


def print_compare(name_a: str, name_b: str, arm: str,
                  cmp: pd.DataFrame) -> None:
    log(f"=== paired: {name_b} against {name_a}, arm={arm} ===")
    log(f"{'layer':>5} {'n':>3} {'A':>8} {'B':>8} {'delta':>8} "
        f"{'95% CI':>18} {'up':>6}")
    for _, r in cmp.iterrows():
        if np.isnan(r.delta):
            log(f"{int(r.layer):>5} {int(r.n_pairs):>3}   not comparable")
            continue
        star = " *" if (r.lo > 0 or r.hi < 0) else ""
        log(f"{int(r.layer):>5} {int(r.n_pairs):>3} {r.mean_a:>+8.3f} "
            f"{r.mean_b:>+8.3f} {r.delta:>+8.3f} "
            f"[{r.lo:>+6.3f},{r.hi:>+6.3f}] {int(r.n_up)}/{int(r.n_pairs)}{star}")
    log("  * = the paired 95% interval excludes zero")


def load_run(run: str, case: str) -> pd.DataFrame:
    path = Path(run)
    if path.is_dir():
        path = path / case / "maps.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def cmd_verify(args: argparse.Namespace) -> int:
    """Reproduce the stage-4 table. The tool is wrong if this fails."""
    bad = 0
    for tag, want in STAGE4.items():
        run = REPO / f"outputs/_monocle/patch_map_causal{tag}"
        if not (run / args.case / "maps.parquet").exists():
            log(f"{tag}: absent, skipped")
            continue
        scored = score_maps(load_run(str(run), args.case), args.k)
        print_table(f"patch_map_causal{tag}", scored)
        for layer, (t_s, t_r, l_s, l_r) in want.items():
            got = scored[scored.layer == layer]
            if got.empty:
                log(f"  MISS L{layer}: absent"); bad += 1; continue
            g = got.iloc[0]
            for label, a, b in (("taylor sign", g.taylor_sign, t_s),
                                ("taylor r", g.taylor_r, t_r),
                                ("lens_delta sign", g.lens_delta_sign, l_s),
                                ("lens_delta r", g.lens_delta_r, l_r)):
                if abs(a - b) > 0.001:
                    log(f"  MISMATCH L{layer} {label}: {a:+.3f} != {b:+.3f}")
                    bad += 1
    log("verify FAILED" if bad else "verify ok -- the stage-4 table reproduces")
    return 1 if bad else 0


def cmd_score(args: argparse.Namespace) -> int:
    out: dict[str, Any] = {"k": args.k, "case": args.case, "runs": {}}
    for run in args.runs:
        scored = score_maps(load_run(run, args.case), args.k)
        print_table(Path(run).name, scored)
        out["runs"][Path(run).name] = scored.to_dict(orient="records")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        log(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=[],
                    help="patch_map_causal run dirs, or maps.parquet paths.")
    ap.add_argument("--case", default="subway_safety")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "RUNG"),
                    default=None,
                    help="Paired comparison of 2 runs over the same pairs.")
    ap.add_argument("--arm", default="lens_delta", choices=sorted(CANDIDATES),
                    help="Which map the paired comparison reads.")
    ap.add_argument("--verify", action="store_true",
                    help="Reproduce the stage-4 table and check every cell.")
    ap.add_argument("--out", default=None)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        return cmd_verify(args)
    if args.compare:
        base, rung = args.compare
        cmp = paired_compare(load_run(base, args.case),
                             load_run(rung, args.case), args.arm, k=args.k)
        print_compare(Path(base).name, Path(rung).name, args.arm, cmp)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(
                {"base": base, "rung": rung, "arm": args.arm, "k": args.k,
                 "rows": cmp.to_dict(orient="records")}, indent=1))
            log(f"wrote {args.out}")
        return 0
    if not args.runs:
        raise SystemExit("give --runs, --compare, or --verify")
    return cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
