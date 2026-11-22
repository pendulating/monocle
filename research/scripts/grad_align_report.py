"""Step 2 report, CPU part: the gradient against the axis and the pinv vector.

Per case and block: the median gradient norm (the first-order gain of a
unit step along it), how consistent the gradient is across images, and
its cosine with the axis and with the pinv vector.

Usage:
    python research/scripts/grad_align_report.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/_monocle/grad_align"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    rows = []
    for case in CASES:
        f = root / case / "cosines.parquet"
        if not f.exists():
            print(f"[grad-align-report] {case}: no run yet")
            continue
        df = pd.read_parquet(f)
        for l, g in df.groupby("layer"):
            rows.append({
                "case": case, "layer": int(l),
                "grad_norm_median": float(g["grad_norm"].median()),
                "cos_g_meangrad": float(g["cos_mean_grad"].mean()),
                "cos_g_axis": float(g["cos_axis"].mean()),
                "abs_cos_g_axis_max": float(g["cos_axis"].abs().max()),
                "cos_g_pinv": float(g["cos_pinv"].mean()),
                "abs_cos_g_pinv_max": float(g["cos_pinv"].abs().max()),
            })
    if not rows:
        return 1
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\nGRADIENT at the answer position, per case and block")
    print(out.round(3).to_string(index=False))
    out.to_parquet(root / "grad_align_report.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
