"""Report of the grouped removal at the production budget.

Three comparisons per case and group size ``k``:

| Comparison | Question |
|---|---|
| block map against the single-cell map pooled ``k`` x ``k`` | do single cells add up to a block? |
| block map against the coarse-budget map of the same grid (70 for k=2) | does the coarse picture find the same blocks? |
| block sum against the sum of pooled cells | how far from additive is a block? |

Usage:
    python research/scripts/patch_group_report.py --cases all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.patch_budget_report import compare, pool, resample, square  # noqa: E402

ROOT = REPO / "outputs/_monocle/patch_group"
BUDGET = REPO / "outputs/_monocle/patch_budget"
COORD = REPO / "outputs/_monocle/steer_coord"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
#: The coarse budget whose grid a block size meets: 2 x 2 blocks of the
#: 16 grid are the 8 x 8 grid of the 70 budget.
BUDGET_OF_GROUP = {2: 70}


def load_single(case: str) -> dict[str, np.ndarray]:
    """The single-cell removal deltas at 280 (the `steer_coord` values are
    contributions, so the sign turns back)."""
    b = BUDGET / case / "maps.parquet"
    if b.exists():
        df = pd.read_parquet(b)
        df = df[df["budget"] == 280]
        return {p: square(v) for p, v in zip(df["image_path"], df["values"])}
    df = pd.read_parquet(COORD / case / "maps.parquet")
    return {p: -square(v) for p, v in zip(df["image_path"], df["values"])}


def load_budget(case: str, budget: int) -> dict[str, np.ndarray]:
    b = BUDGET / case / "maps.parquet"
    if not b.exists():
        return {}
    df = pd.read_parquet(b)
    df = df[df["budget"] == budget]
    return {p: square(v) for p, v in zip(df["image_path"], df["values"])}


def score_case(case: str) -> list[dict]:
    df = pd.read_parquet(ROOT / case / "maps.parquet")
    single = load_single(case)
    rows = []
    for k, g in df.groupby("group"):
        k = int(k)
        blocks = {p: square(v) for p, v in zip(g["image_path"], g["values"])}
        coarse = load_budget(case, BUDGET_OF_GROUP.get(k, -1))
        r_pool, t_pool, r_coarse, t_coarse, add, sd = [], [], [], [], [], []
        for p, m in blocks.items():
            if p in single:
                pooled = pool(single[p], k)
                r, t = compare(m, pooled)
                r_pool.append(r)
                t_pool.append(t)
                add.append(sps.pearsonr(m.ravel(), pooled.ravel())[0]
                           if False else float(np.sum(m) / (np.sum(pooled) + 1e-9)))
            if p in coarse:
                r, t = compare(m, coarse[p])
                r_coarse.append(r)
                t_coarse.append(t)
            sd.append(m.std())
        rows.append({
            "case": case, "group": k, "n_blocks": int(m.size), "n": len(blocks),
            "block_sd": float(np.mean(sd)),
            "r_vs_pooled_cells": float(np.median(r_pool)) if r_pool else np.nan,
            "top_vs_pooled_cells": float(np.median(t_pool)) if t_pool else np.nan,
            "sum_ratio_vs_pooled": float(np.median(add)) if add else np.nan,
            "r_vs_coarse_budget": float(np.median(r_coarse)) if r_coarse else np.nan,
            "top_vs_coarse_budget": float(np.median(t_coarse)) if t_coarse else np.nan,
        })
    return rows


def figure(case: str, out: Path, n_show: int = 6) -> None:
    from PIL import Image, ImageDraw
    from research.monocle import causal_maps as cm
    from research.monocle.single_steer import CASE_WORDS
    pos, neg = CASE_WORDS[case]
    df = pd.read_parquet(ROOT / case / "maps.parquet")
    single = load_single(case)
    coarse = load_budget(case, 70)
    groups = sorted(int(k) for k in df["group"].unique())
    blocks = {(p, int(k)): -square(v) for p, k, v in
              zip(df["image_path"], df["group"], df["values"])}
    paths = [p for p in single if all((p, k) in blocks for k in groups)]
    zero = dict(zip(df["image_path"], df["zero"]))
    order = np.argsort([zero[p] for p in paths])
    pick = [paths[i] for i in list(order[: n_show // 2])
            + list(order[-(n_show - n_show // 2):])]
    cols = [("cells", lambda p: -single[p]),
            ("cells pooled 2x2", lambda p: -pool(single[p], 2))]
    for k in groups:
        cols.append((f"{k}x{k} blocks", lambda p, k=k: blocks[(p, k)]))
    if coarse:
        cols.append(("70 tokens", lambda p: -coarse[p] if p in coarse
                     else np.zeros((8, 8))))
    vmax = {name: float(np.percentile(np.abs(np.concatenate(
        [f(p).ravel() for p in pick])), 99)) for name, f in cols}
    cell, lab = 220, 18
    W = (1 + len(cols)) * cell
    H = len(pick) * (cell + lab) + 44
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, p in enumerate(pick):
        img = Image.open(p).convert("RGB")
        y = r * (cell + lab)
        sheet.paste(img.resize((cell - 6, cell - 6), Image.LANCZOS), (3, y + 3))
        draw.text((4, y + cell - 2), f"c={zero[p]:+.1f}", fill=(40, 40, 40))
        for ci, (name, f) in enumerate(cols, start=1):
            m = f(p)
            pic = cm.overlay(img, m.ravel(), vmax[name])
            x = ci * cell
            sheet.paste(pic.resize((cell - 6, cell - 6), Image.LANCZOS),
                        (x + 3, y + 3))
            draw.text((x + 4, y + cell - 2), f"{name}  |max|={vmax[name]:.1f}",
                      fill=(40, 40, 40))
    sheet.paste(cm.colour_bar(360, 34, 1.0, teal_label=f"{pos} (scale per panel)",
                              coral_label=neg), (20, H - 38))
    sheet.save(out / f"group_{case}.png")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--out-dir", default=str(ROOT / "research" / "figures"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rows = []
    for case in cases:
        if not (ROOT / case / "summary.json").exists():
            print(f"[patch-group-report] {case}: no run yet")
            continue
        rows += score_case(case)
        if not args.no_figures:
            figure(case, out)
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\nGROUPED REMOVAL at 280: blocks against pooled cells and against the coarse budget")
    print(df.round(3).to_string(index=False))
    df.to_parquet(ROOT / "patch_group_report.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
