"""Step 2: the judgment axis against the pinv vector, as steering arms.

Reads a `monocle.single_steer` run with the arms ``signal`` (pinv), ``control``
(its pinv random), ``axis`` (the judgment axis from the calibration images)
and ``control_raw`` (a raw random unit direction). Per case, per layer, per
arm it reports:

| Column | Meaning |
|---|---|
| ``window`` | the contiguous dose range around 0 where the arm holds the answer contract at 0.90 or more |
| ``gain`` | logits per unit residual norm, from the paired doses ``+-0.5`` |
| ``gain_025`` | the same from ``+-0.25`` |
| ``neg_at_lo`` / ``pos_at_hi`` | the fraction of images at the negative pole at the low window edge, and at the positive pole at the high edge |
| ``effect_lo`` / ``effect_hi`` | the paired effect against the arm's own control at those edges, with a bootstrap interval |

The step 1 prediction (`research/scripts/steer_axis_cpu.py`) is the natural slope of
the judgment on the axis. The ``axis`` gain lands on it if the axis is the
carrier.

Usage:
    python research/scripts/steer_axis_report.py --cases all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT_DEFAULT = REPO / "outputs/_monocle/single_steer_axis"
AXIS_CPU_DEFAULT = REPO / "outputs/_monocle/steer_coord/axis_cpu_report.parquet"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
#: Each arm's null. The axis gets 2: the raw random direction at the same
#: scale, and the pinv random of the signal arm. The raw random breaks the
#: answer contract at the first dose, so its paired effect is read against a
#: broken output; the pinv random holds the contract and is the cleaner null.
CONTROL_OF = {"signal": ["control"], "axis": ["control_raw", "control"]}
CONTRACT = 0.90


def log(m: str) -> None:
    print(f"[steer-axis-report] {m}", flush=True)


def window(cells: list[dict], layer: int, arm: str,
           thresh: float = CONTRACT) -> Optional[tuple[float, float]]:
    rows = sorted(((c["alpha"], c["frac_on_contract"]) for c in cells
                   if c["layer"] == layer and c["direction"] == arm),
                  key=lambda r: r[0])
    alphas = [a for a, _ in rows]
    frac = dict(rows)
    if 0.0 not in frac or frac[0.0] < thresh:
        return None
    i = alphas.index(0.0)
    j = k = i
    while j - 1 >= 0 and frac[alphas[j - 1]] >= thresh:
        j -= 1
    while k + 1 < len(alphas) and frac[alphas[k + 1]] >= thresh:
        k += 1
    return alphas[j], alphas[k]


def gain(df: pd.DataFrame, layer: int, arm: str, a: float) -> float:
    g = df[(df["layer"] == layer) & (df["direction"] == arm)]
    lo = g[g["alpha"] == -a].set_index("image_path")["logit_diff"]
    hi = g[g["alpha"] == a].set_index("image_path")["logit_diff"]
    idx = lo.index.intersection(hi.index)
    if len(idx) == 0:
        return float("nan")
    scale = float(g["scale_l"].iloc[0])
    return float(((hi.loc[idx] - lo.loc[idx]) / (2 * a * scale)).mean())


def paired_effect(df: pd.DataFrame, layer: int, arm: str, ctl: str,
                  alpha: float, n: int = 10000, seed: int = 0) -> tuple:
    sel = (df["layer"] == layer) & (df["alpha"] == alpha)
    s = df[sel & (df["direction"] == arm)].set_index("image_path")["logit_diff"]
    c = df[sel & (df["direction"] == ctl)].set_index("image_path")["logit_diff"]
    eff = (s - c).dropna().to_numpy()
    if len(eff) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = eff[rng.integers(0, len(eff), size=(n, len(eff)))].mean(1)
    return (float(eff.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def cell(cells: list[dict], layer: int, arm: str, alpha: float) -> dict:
    for c in cells:
        if (c["layer"] == layer and c["direction"] == arm
                and c["alpha"] == alpha):
            return c
    return {}


def score_case(case_dir: Path, axis_cpu: Optional[pd.DataFrame]) -> list[dict]:
    s = json.loads((case_dir / "summary.json").read_text())
    df = pd.read_parquet(case_dir / "steer_results.parquet")
    cells = list(s["per_cell"].values())
    rows = []
    for layer in s["layers"]:
        base = cell(cells, layer, "signal", 0.0)
        for arm in s.get("arms", ["signal", "control"]):
            w = window(cells, layer, arm)
            row = {"case": case_dir.name, "layer": layer, "arm": arm,
                   "base_pos": base.get("frac_top_pos"),
                   "window_lo": w[0] if w else float("nan"),
                   "window_hi": w[1] if w else float("nan"),
                   "gain": gain(df, layer, arm, 0.5),
                   "gain_025": gain(df, layer, arm, 0.25)}
            if w:
                lo_c = cell(cells, layer, arm, w[0])
                hi_c = cell(cells, layer, arm, w[1])
                row["neg_at_lo"] = lo_c.get("frac_top_neg")
                row["pos_at_hi"] = hi_c.get("frac_top_pos")
                for ctl in CONTROL_OF.get(arm, []):
                    suffix = "" if ctl == CONTROL_OF[arm][0] else "_vs_pinv"
                    for tag, a in (("lo", w[0]), ("hi", w[1])):
                        if a == 0:
                            continue
                        m, l, u = paired_effect(df, layer, arm, ctl, a)
                        row[f"effect_{tag}{suffix}"] = m
                        row[f"effect_{tag}{suffix}_ci"] = f"[{l:+.1f}, {u:+.1f}]"
            if axis_cpu is not None and arm == "axis":
                q = axis_cpu[(axis_cpu["case"] == case_dir.name)
                             & (axis_cpu["layer"] == layer)]
                if len(q):
                    row["predicted_gain"] = float(q["natural_slope"].iloc[0])
            rows.append(row)
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT_DEFAULT))
    ap.add_argument("--axis-cpu", default=str(AXIS_CPU_DEFAULT))
    ap.add_argument("--cases", default="all")
    args = ap.parse_args(argv)
    root = Path(args.root)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    axis_cpu = (pd.read_parquet(args.axis_cpu)
                if Path(args.axis_cpu).exists() else None)
    rows = []
    for case in cases:
        case_dir = root / case
        if not (case_dir / "summary.json").exists():
            log(f"{case}: no run yet")
            continue
        rows += score_case(case_dir, axis_cpu)
        log(f"{case}: scored")
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = ["case", "layer", "arm", "base_pos", "window_lo", "window_hi",
            "gain", "gain_025", "predicted_gain", "neg_at_lo", "pos_at_hi",
            "effect_lo", "effect_lo_ci", "effect_hi", "effect_hi_ci",
            "effect_lo_vs_pinv", "effect_hi_vs_pinv"]
    cols = [c for c in cols if c in df.columns]
    print("\nSTEP 2: the 4 arms, per case and layer")
    print(df[cols].round(3).to_string(index=False))
    df.to_parquet(root / "axis_report.parquet", index=False)
    # The headline: the axis gain against the pinv gain, at the best layer
    # of each arm (the widest window).
    print("\nHEADLINE: gain (logits per unit norm) at each arm's widest window")
    for case in cases:
        d = df[df["case"] == case]
        if not len(d):
            continue
        line = [case]
        for arm in ("axis", "signal"):
            a = d[d["arm"] == arm].copy()
            if not len(a):
                continue
            a["width"] = a["window_hi"] - a["window_lo"]
            best = a.sort_values("width", ascending=False).iloc[0]
            line.append(
                f"{arm}: L{int(best['layer'])} window [{best['window_lo']:+g}, "
                f"{best['window_hi']:+g}] gain {best['gain']:+.2f}"
                + (f" (predicted {best['predicted_gain']:+.2f})"
                   if arm == "axis" and "predicted_gain" in best
                   and pd.notna(best["predicted_gain"]) else ""))
        print("  " + " | ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
