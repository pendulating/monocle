"""Compare the corpus-faithfulness rungs against the input-ablation truth.

Gate B (`monocle/input_gate.py`) scores each candidate map against the map
that 256 single-cell input ablations measured. This script puts the rungs of
`monocle/jlens_fit_ladder.py` side by side, one row for each fitted layer.

The rungs differ in the cotangent target and in the prompt:

| Rung | Prompt | Target positions |
|---|---|---|
| R0 | "Describe this scene." | every position from the first patch |
| R1 | "Describe this scene." | the last position |
| R2 | `rating_2point`, forced prefix | the answer token |
| R3 | `rating_2point`, held-out images | the answer token |

`lens_own` is the arm that changes with the rung. `gxa` is the local
instrument, and it must not move between rungs: the same photographs and the
same seed give the same gradient. The script prints it as a control.

Usage:
    python scripts/ladder_gate_b_report.py --case subway_safety
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIT_LAYERS = (6, 12, 18, 24, 30, 36, 42, 46)
RUNGS = ("R0", "R1", "R2", "R3")
#: The names each rung carries in a figure or a table.
RUNG_LABEL = {
    "R0": "R0 span/describe",
    "R1": "R1 last/describe",
    "R2": "R2 answer/rating",
    "R3": "R3 answer/held-out",
}


def load_rung(
    root: Path, rung: str, case: str, prefix: str = "gateB_ladder"
) -> pd.DataFrame | None:
    """Read the per-photograph frame of one rung, or None when it is absent."""
    path = root / f"{prefix}{rung}" / case / "agreement.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["rung"] = rung
    return df


def paired_gain(
    a: pd.DataFrame, b: pd.DataFrame, layer: int, column: str
) -> tuple[float, float, int]:
    """The per-photograph gain of `b` over `a`, and its t statistic.

    Both rungs read the SAME photographs under the same seed, so the paired
    difference is far tighter than the gap between two means.

    Returns:
        (mean gain, two-sided p value, the number of photographs)
    """
    from scipy import stats

    ja = a[a.layer == layer].set_index("target")[column]
    jb = b[b.layer == layer].set_index("target")[column]
    common = sorted(set(ja.index) & set(jb.index))
    if len(common) < 3:
        return float("nan"), float("nan"), len(common)
    d = jb.loc[common].to_numpy() - ja.loc[common].to_numpy()
    ok = np.isfinite(d)
    if int(ok.sum()) < 3:
        return float("nan"), float("nan"), int(ok.sum())
    t = stats.ttest_1samp(d[ok], 0.0)
    return float(np.mean(d[ok])), float(t.pvalue), int(ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="subway_safety")
    ap.add_argument("--root", default=str(REPO / "outputs/_monocle"))
    ap.add_argument("--column", default="lens_own_top_pearson",
                    help="The score column. Top-50 Pearson by default.")
    ap.add_argument("--baseline", default="R0",
                    help="The rung that every gain is measured against.")
    ap.add_argument("--prefix", default="gateB_ladder",
                    help="The directory prefix of the gate runs.")
    ap.add_argument("--out", default=None, help="Write the table as JSON.")
    args = ap.parse_args()

    root = Path(args.root)
    frames = {r: load_rung(root, r, args.case, args.prefix) for r in RUNGS}
    have = [r for r in RUNGS if frames[r] is not None]
    if not have:
        print(f"no gate-B output under {root} for case {args.case}")
        return 2
    print(f"case {args.case}, column {args.column}")
    print(f"rungs on disk: {', '.join(RUNG_LABEL[r] for r in have)}\n")

    rows = []
    for layer in FIT_LAYERS:
        row: dict[str, object] = {"layer": layer}
        for r in have:
            df = frames[r]
            sel = df[df.layer == layer][args.column]
            row[r] = float(sel.mean()) if len(sel) else float("nan")
        sel = frames[have[0]]
        gxa = sel[sel.layer == layer]["gxa_top_pearson"]
        row["gxa"] = float(gxa.mean()) if len(gxa) else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows).set_index("layer")
    print(table.round(4).to_string())

    base = args.baseline
    if base in have:
        print(f"\npaired gain over {RUNG_LABEL[base]}, by photograph")
        for r in have:
            if r == base:
                continue
            print(f"  {RUNG_LABEL[r]}")
            for layer in FIT_LAYERS:
                g, p, n = paired_gain(
                    frames[base], frames[r], layer, args.column)
                star = " *" if np.isfinite(p) and p < 0.05 else ""
                print(f"    L{layer:<3d} gain {g:+.4f}  p {p:.4f}  n {n}{star}")

    # The control. The local map reads the same photographs under every rung,
    # so a drift here means the rungs did not read the same set.
    print("\ncontrol: the local `gxa` map must not move between rungs")
    for layer in (18, 24):
        vals = {r: float(frames[r][frames[r].layer == layer]
                         ["gxa_top_pearson"].mean()) for r in have}
        spread = max(vals.values()) - min(vals.values())
        print(f"  L{layer}: " + "  ".join(f"{r} {v:+.4f}" for r, v in
                                          vals.items())
              + f"   spread {spread:.2e}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"case": args.case, "column": args.column,
             "table": table.reset_index().to_dict(orient="records")}, indent=1))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
