#!/usr/bin/env python
"""Paired statistics for the workspace depth read.

What this answers
-----------------
`monocle/safety_workspace.py` writes one row for each
(pair, condition, lens, layer, position). The depth claim reads the
`label` position: the production prompt puts the judgment in the readable
space at a depth, and the neutral prompt does not. That claim needs a
PAIRED test, because the same photograph pair goes through both arms.

This script does 4 things, for each (lens, layer) at `pos == "label"`:

| Item | The quantity |
|---|---|
| 1 | prod minus neutral in `p_prod_label`, paired by `pair_id` |
| 2 | prod minus neutral in `argmax_correct`, paired by `pair_id` |
| 3 | the neutral arm alone: mean `p_prod_label` and its interval |
| 4 | the difference that this many pairs can detect at 80% power |

Each difference carries a bootstrap interval over the pairs and a Wilcoxon
signed-rank p. A Benjamini-Hochberg step runs over the layers inside one
lens, and the table marks the layers that survive it.

Item 3 is the number behind a sentence such as "never under neutral". Read
the interval, not the point.

Item 4 is the floor of the instrument. A layer with a small difference and a
large floor says nothing. A layer with a small difference and a small floor
says the effect is absent.

Run
---
    .venv-3.12/bin/python scripts/answer_depth_stats.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/share/ju/matt/mllmsci")
PARQUET_DEFAULT = REPO / "outputs/_monocle/safety_workspace/answer_depth.parquet"
OUT_DEFAULT = REPO / "outputs/_monocle/safety_workspace/answer_depth_stats.json"

#: The condition that carries the production prompt.
COND_TEST = "prod"
#: The condition that carries the neutral prompt.
COND_REF = "neutral"
#: The position that holds the label token.
POS = "label"


def log(m: str) -> None:
    print(f"[depth-stats] {m}", flush=True)


def boot_mean_ci(
    values: np.ndarray, *, n_boot: int, seed: int, alpha: float,
) -> tuple[float, float]:
    """A percentile interval for the mean. The resample unit is the pair."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1 or float(v.std()) == 0.0:
        return (float(v.mean()), float(v.mean()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def wilcoxon_p(diff: np.ndarray) -> float:
    """A two-sided signed-rank p. All-zero differences give 1.0."""
    d = np.asarray(diff, dtype=float)
    if d.size == 0 or not np.any(d != 0.0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def bh_step(pvals: list[float], alpha: float) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg. Gives the adjusted p and the survivor flag."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return ([], [])
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out_adj = np.empty(m, dtype=float)
    out_adj[order] = adj
    return (out_adj.tolist(), (out_adj <= alpha).tolist())


def mde(sd: float, n: int, alpha: float) -> float:
    """The difference detectable at 80% power, normal approximation."""
    if n <= 1 or not np.isfinite(sd):
        return float("nan")
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(0.80))
    return (z_a + z_b) * float(sd) / float(np.sqrt(n))


def paired_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """One row per pair, with the test arm, the reference arm, and the gap."""
    wide = df.pivot_table(
        index="pair_id", columns="cond", values=column, aggfunc="first")
    for cond in (COND_TEST, COND_REF):
        if cond not in wide.columns:
            return pd.DataFrame(columns=["test", "ref", "diff"])
    wide = wide[[COND_TEST, COND_REF]].dropna()
    wide.columns = ["test", "ref"]
    wide["diff"] = wide["test"] - wide["ref"]
    return wide


def analyse(
    df: pd.DataFrame, *, n_boot: int, seed: int, alpha: float,
) -> list[dict]:
    """One record for each (lens, layer)."""
    rows: list[dict] = []
    for (lens, layer), sub in df.groupby(["lens", "layer"], sort=True):
        rec: dict = {"lens": str(lens), "layer": int(layer)}
        rec["n_rows"] = int(len(sub))
        rec["conds_present"] = sorted(sub["cond"].unique().tolist())

        for name, column in (("p", "p_prod_label"), ("acc", "argmax_correct")):
            w = paired_frame(sub.assign(**{
                column: sub[column].astype(float)}), column)
            d = w["diff"].to_numpy(dtype=float)
            lo, hi = boot_mean_ci(d, n_boot=n_boot, seed=seed, alpha=alpha)
            sd = float(d.std(ddof=1)) if d.size > 1 else float("nan")
            rec[f"{name}_n_pairs"] = int(d.size)
            rec[f"{name}_mean_test"] = float(w["test"].mean()) if d.size else float("nan")
            rec[f"{name}_mean_ref"] = float(w["ref"].mean()) if d.size else float("nan")
            rec[f"{name}_mean_diff"] = float(d.mean()) if d.size else float("nan")
            rec[f"{name}_ci_lo"] = lo
            rec[f"{name}_ci_hi"] = hi
            rec[f"{name}_sd_diff"] = sd
            rec[f"{name}_n_positive"] = int((d > 0).sum())
            rec[f"{name}_n_zero"] = int((d == 0).sum())
            rec[f"{name}_p_wilcoxon"] = wilcoxon_p(d)
            rec[f"{name}_mde80"] = mde(sd, int(d.size), alpha)

        ref = sub[sub["cond"] == COND_REF]
        vals = ref["p_prod_label"].to_numpy(dtype=float)
        lo, hi = boot_mean_ci(vals, n_boot=n_boot, seed=seed, alpha=alpha)
        rec["neutral_n"] = int(vals.size)
        rec["neutral_mean"] = float(vals.mean()) if vals.size else float("nan")
        rec["neutral_median"] = float(np.median(vals)) if vals.size else float("nan")
        rec["neutral_ci_lo"] = lo
        rec["neutral_ci_hi"] = hi
        rec["neutral_argmax_correct"] = (
            float(ref["argmax_correct"].astype(float).mean())
            if len(ref) else float("nan"))
        rows.append(rec)

    for name in ("p", "acc"):
        for lens in sorted({r["lens"] for r in rows}):
            block = [r for r in rows if r["lens"] == lens]
            block.sort(key=lambda r: r["layer"])
            adj, keep = bh_step([r[f"{name}_p_wilcoxon"] for r in block], alpha)
            for r, a, k in zip(block, adj, keep):
                r[f"{name}_q_bh"] = float(a)
                r[f"{name}_survives_bh"] = bool(k)
    return rows


def print_tables(rows: list[dict], alpha: float) -> None:
    """3 compact tables, 1 for each question."""
    lenses = sorted({r["lens"] for r in rows})

    log("")
    log("1. p_prod_label, prod minus neutral, paired by pair")
    log(f"{'lens':>9} {'layer':>5} {'n':>4} {'prod':>10} {'neutral':>10} "
        f"{'diff':>11} {'ci_lo':>11} {'ci_hi':>11} {'wilcoxon':>10} "
        f"{'q_BH':>9} {'keep':>5}")
    for lens in lenses:
        for r in sorted([x for x in rows if x["lens"] == lens],
                        key=lambda x: x["layer"]):
            log(f"{lens:>9} {r['layer']:>5} {r['p_n_pairs']:>4} "
                f"{r['p_mean_test']:>10.3e} {r['p_mean_ref']:>10.3e} "
                f"{r['p_mean_diff']:>+11.3e} {r['p_ci_lo']:>+11.3e} "
                f"{r['p_ci_hi']:>+11.3e} {r['p_p_wilcoxon']:>10.3e} "
                f"{r['p_q_bh']:>9.3e} {'yes' if r['p_survives_bh'] else 'no':>5}")

    log("")
    log("2. argmax_correct, prod minus neutral, paired by pair")
    log(f"{'lens':>9} {'layer':>5} {'n':>4} {'prod':>8} {'neutral':>8} "
        f"{'diff':>9} {'ci_lo':>9} {'ci_hi':>9} {'wilcoxon':>10} "
        f"{'q_BH':>9} {'keep':>5}")
    for lens in lenses:
        for r in sorted([x for x in rows if x["lens"] == lens],
                        key=lambda x: x["layer"]):
            log(f"{lens:>9} {r['layer']:>5} {r['acc_n_pairs']:>4} "
                f"{r['acc_mean_test']:>8.3f} {r['acc_mean_ref']:>8.3f} "
                f"{r['acc_mean_diff']:>+9.3f} {r['acc_ci_lo']:>+9.3f} "
                f"{r['acc_ci_hi']:>+9.3f} {r['acc_p_wilcoxon']:>10.3e} "
                f"{r['acc_q_bh']:>9.3e} "
                f"{'yes' if r['acc_survives_bh'] else 'no':>5}")

    log("")
    log("3. the neutral arm alone, and the detectable difference "
        f"(80% power, alpha {alpha})")
    log(f"{'lens':>9} {'layer':>5} {'n':>4} {'mean_p':>10} {'ci_lo':>10} "
        f"{'ci_hi':>10} {'median_p':>10} {'acc':>6} {'mde_p':>10} "
        f"{'mde_acc':>8}")
    for lens in lenses:
        for r in sorted([x for x in rows if x["lens"] == lens],
                        key=lambda x: x["layer"]):
            log(f"{lens:>9} {r['layer']:>5} {r['neutral_n']:>4} "
                f"{r['neutral_mean']:>10.3e} {r['neutral_ci_lo']:>10.3e} "
                f"{r['neutral_ci_hi']:>10.3e} {r['neutral_median']:>10.3e} "
                f"{r['neutral_argmax_correct']:>6.3f} {r['p_mde80']:>10.3e} "
                f"{r['acc_mde80']:>8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=PARQUET_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    log(f"{args.parquet}: {len(df)} rows")
    log(f"conds {sorted(df['cond'].unique())}, "
        f"lenses {sorted(df['lens'].unique())}, "
        f"positions {sorted(df['pos'].unique())}")

    at_label = df[df["pos"] == POS].copy()
    log(f"{POS}: {len(at_label)} rows, {at_label['pair_id'].nunique()} pairs, "
        f"layers {sorted(at_label['layer'].unique())}")

    rows = analyse(
        at_label, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)
    print_tables(rows, args.alpha)

    payload = {
        "parquet": str(args.parquet),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "position": POS,
        "cond_test": COND_TEST,
        "cond_ref": COND_REF,
        "conds_in_file": sorted(df["cond"].unique().tolist()),
        "lenses_in_file": sorted(df["lens"].unique().tolist()),
        "n_pairs_in_file": int(df["pair_id"].nunique()),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "alpha": float(args.alpha),
        "power": 0.80,
        "bh_family": "the layers inside 1 lens, 1 family for each metric",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    log("")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
