"""Report of the removal maps across the visual token budgets.

``hole_offset`` is the mean removal delta over an image's cells: what one
missing slot does on its own, before any cell's content counts.
``cell_sd`` is the spread of the deltas inside an image, the map's signal.

Per case: the judgment at each budget against the production budget,
and the map agreement. The 280 map (16 x 16) pools 2 x 2 to 8 x 8 to
meet the 70 map cell for cell; the 140 map (11 x 11) is resampled to
16 x 16 to meet the 280 map. Figures draw one image at the 3 budgets on
one colour bar, in the `causal_maps` style.

Usage:
    python research/scripts/patch_budget_report.py --cases all
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

ROOT = REPO / "outputs/_monocle/patch_budget"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
TOP_SHARE = 0.10


def square(v: np.ndarray) -> np.ndarray:
    side = int(round(v.size ** 0.5))
    return np.asarray(v, dtype=np.float64).reshape(side, side)


def pool(m: np.ndarray, k: int) -> np.ndarray:
    """Sum k x k blocks: the removal effect of the block's cells, added."""
    s = m.shape[0] // k
    return m[: s * k, : s * k].reshape(s, k, s, k).sum(axis=(1, 3))


def resample(m: np.ndarray, side: int) -> np.ndarray:
    """Bilinear resample of a square map to ``side`` x ``side``, mass kept."""
    from PIL import Image
    im = Image.fromarray(m.astype(np.float32), mode="F")
    out = np.asarray(im.resize((side, side), Image.BILINEAR), dtype=np.float64)
    return out * (m.sum() / out.sum()) if abs(out.sum()) > 1e-9 else out


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    r = sps.pearsonr(a.ravel(), b.ravel())[0]
    k = max(1, round(TOP_SHARE * a.size))
    ta = set(np.argsort(np.abs(a.ravel()))[-k:])
    tb = set(np.argsort(np.abs(b.ravel()))[-k:])
    return float(r), len(ta & tb) / k


def score_case(case: str) -> tuple[list[dict], list[dict]]:
    df = pd.read_parquet(ROOT / case / "maps.parquet")
    s = json.loads((ROOT / case / "summary.json").read_text())
    budgets = sorted(df["budget"].unique())
    ref_b = max(budgets)
    by = {(p, int(b)): square(v) for p, b, v in
          zip(df["image_path"], df["budget"], df["values"])}
    zero = {(p, int(b)): z for p, b, z in
            zip(df["image_path"], df["budget"], df["zero"])}
    paths = sorted({p for p, b in by if b == ref_b})
    judg, maps = [], []
    for b in budgets:
        pb = s["per_budget"][str(int(b))]
        judg.append({"case": case, "budget": int(b), **pb})
        if b == ref_b:
            offs = np.array([by[(p, b)].mean() for p in paths])
            sds = np.array([by[(p, b)].std() for p in paths])
            maps.append({"case": case, "budget": int(b), "ref_budget": int(ref_b),
                         "n": len(paths), "cells": int(by[(paths[0], b)].size),
                         "hole_offset": float(offs.mean()),
                         "cell_sd": float(sds.mean())})
            continue
        rs, tops, sums = [], [], []
        for p in paths:
            if (p, b) not in by:
                continue
            m, ref = by[(p, b)], by[(p, ref_b)]
            if ref.shape[0] % m.shape[0] == 0:
                r, t = compare(m, pool(ref, ref.shape[0] // m.shape[0]))
            else:
                r, t = compare(resample(m, ref.shape[0]), ref)
            rs.append(r)
            tops.append(t)
            sums.append((m.sum(), ref.sum()))
        sums = np.array(sums)
        offs = np.array([by[(p, b)].mean() for p in paths if (p, b) in by])
        sds = np.array([by[(p, b)].std() for p in paths if (p, b) in by])
        maps.append({
            "case": case, "budget": int(b), "ref_budget": int(ref_b),
            "n": len(rs), "cells": int(m.size),
            "hole_offset": float(offs.mean()),
            "cell_sd": float(sds.mean()),
            "median_r": float(np.median(rs)),
            "frac_r_pos": float(np.mean(np.array(rs) > 0)),
            "median_top_overlap": float(np.median(tops)),
            "sum_ratio": float(np.median(sums[:, 0] / (sums[:, 1] + 1e-9))),
            "r_map_sums": float(np.corrcoef(sums[:, 0], sums[:, 1])[0, 1]),
        })
    return judg, maps


def figure(case: str, out: Path, n_show: int = 6) -> None:
    from PIL import Image, ImageDraw
    from research.monocle import causal_maps as cm
    from research.monocle.single_steer import CASE_WORDS
    pos, neg = CASE_WORDS[case]
    df = pd.read_parquet(ROOT / case / "maps.parquet")
    budgets = sorted(df["budget"].unique())
    # The stored values are removal deltas; the picture paints the
    # contribution, their negative, as `causal_maps` does.
    piv = {(p, int(b)): (-np.asarray(v, dtype=np.float64), z) for p, b, v, z in
           zip(df["image_path"], df["budget"], df["values"], df["zero"])}
    paths = sorted({p for p, b in piv if all((p, bb) in piv for bb in budgets)})
    ref = max(budgets)
    order = np.argsort([piv[(p, ref)][1] for p in paths])
    pick = [paths[i] for i in list(order[: n_show // 2])
            + list(order[-(n_show - n_show // 2):])]
    # One colour bar per budget: a coarse cell removes more pixels.
    vmax = {b: float(np.percentile(np.abs(np.concatenate(
        [piv[(p, b)][0] for p in pick])), 99)) for b in budgets}
    cell, lab = 240, 18
    W = (1 + len(budgets)) * cell
    H = len(pick) * (cell + lab) + 44
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, p in enumerate(pick):
        img = Image.open(p).convert("RGB")
        y = r * (cell + lab)
        sheet.paste(img.resize((cell - 6, cell - 6), Image.LANCZOS), (3, y + 3))
        for ci, b in enumerate(budgets, start=1):
            v, z = piv[(p, b)]
            pic = cm.overlay(img, v, vmax[b])
            x = ci * cell
            sheet.paste(pic.resize((cell - 6, cell - 6), Image.LANCZOS),
                        (x + 3, y + 3))
            draw.text((x + 4, y + cell - 2), f"{b} tokens  c={z:+.1f}",
                      fill=(40, 40, 40))
    bw = min(W // len(budgets) - 40, 360)
    for ci, b in enumerate(budgets):
        sheet.paste(cm.colour_bar(bw, 34, vmax[b], teal_label=pos,
                                  coral_label=neg),
                    (20 + ci * (W // len(budgets)), H - 38))
    sheet.save(out / f"budget_{case}.png")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--out-dir", default=str(ROOT / "research" / "figures"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    judg, maps = [], []
    for case in cases:
        if not (ROOT / case / "summary.json").exists():
            print(f"[patch-budget-report] {case}: no run yet")
            continue
        j, m = score_case(case)
        judg += j
        maps += m
        if not args.no_figures:
            figure(case, out)
    if not judg:
        return 1
    pd.set_option("display.width", 220)
    jd, md = pd.DataFrame(judg), pd.DataFrame(maps)
    print("\nJUDGMENT at each budget (agreement and r are against the largest budget)")
    print(jd.round(3).to_string(index=False))
    print("\nMAP AGREEMENT with the largest budget (pooled or resampled to one grid)")
    print(md.round(3).to_string(index=False))
    jd.to_parquet(ROOT / "budget_judgment.parquet", index=False)
    md.to_parquet(ROOT / "budget_maps.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
