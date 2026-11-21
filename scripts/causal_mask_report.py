"""The 12B under a causal image mask against its bidirectional self.

Per case: the judgment (contrast, positive share, agreement with the
bidirectional answer on the same images), and the removal map at 280
tokens against the bidirectional removal map, cell for cell. Both runs
come from `patch_budget` (deltas at budget 280).

`--root` names another masked run (the self-only mask in
`outputs/_monocle/self12b`) and `--label` names its columns.

Usage:
    python scripts/causal_mask_report.py --cases all
    python scripts/causal_mask_report.py --root outputs/_monocle/self12b --label self
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path("/share/ju/matt/mllmsci")
BI = REPO / "outputs/_monocle/patch_budget"
CA = REPO / "outputs/_monocle/causal12b/patch_budget"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]


def load(root: Path, case: str) -> pd.DataFrame:
    df = pd.read_parquet(root / case / "maps.parquet")
    return df[df["budget"] == 280].set_index("image_path")


def main(argv: Optional[list[str]] = None) -> int:
    global CA
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--root", default=str(CA.parent),
                    help="run root that holds patch_budget/<case>")
    ap.add_argument("--label", default="causal",
                    help="column label of the masked run")
    args = ap.parse_args(argv)
    CA = Path(args.root) / "patch_budget"
    lab = args.label
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rows = []
    for case in cases:
        if not (CA / case / "maps.parquet").exists():
            print(f"[causal-mask-report] {case}: no run yet")
            continue
        b, c = load(BI, case), load(CA, case)
        shared = b.index.intersection(c.index)
        zb = b.loc[shared, "zero"].astype(float)
        zc = c.loc[shared, "zero"].astype(float)
        rs, sd_b, sd_c = [], [], []
        for p in shared:
            mb = np.asarray(b.loc[p, "values"], dtype=np.float64)
            mc = np.asarray(c.loc[p, "values"], dtype=np.float64)
            rs.append(sps.pearsonr(mb, mc)[0])
            sd_b.append(mb.std())
            sd_c.append(mc.std())
        rows.append({
            "case": case, "n": len(shared),
            "mean_c_bidir": float(zb.mean()), f"mean_c_{lab}": float(zc.mean()),
            "frac_pos_bidir": float((zb >= 0).mean()),
            f"frac_pos_{lab}": float((zc >= 0).mean()),
            "agree": float(((zb >= 0) == (zc >= 0)).mean()),
            "r_contrast": float(np.corrcoef(zb, zc)[0, 1]),
            "map_r_median": float(np.median(rs)),
            "map_frac_r_pos": float(np.mean(np.array(rs) > 0)),
            "cell_sd_bidir": float(np.mean(sd_b)),
            f"cell_sd_{lab}": float(np.mean(sd_c)),
        })
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\n{lab.upper()} image mask vs bidirectional, gemma-4-12b, 280 tokens")
    print(df.round(3).to_string(index=False))
    df.to_parquet(CA.parent / f"{lab}_mask_report.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
