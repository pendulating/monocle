"""Step 5b report: the per-cell injection map against the removal map.

For each image, block and arm the injection map is the cell's response to
the patch axis: ``(delta at +2 - delta at -2) / 2``, in logits per 2 scale
units. The report compares it, cell by cell, with the measured removal map
of `steer_coord` (the c-map, block-free) and with the raw random control,
and draws both maps in the `causal_maps` style.

Usage:
    python research/scripts/patch_steer_report.py --cases all
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

ROOT = REPO / "outputs/_monocle/patch_steer"
COORD = REPO / "outputs/_monocle/steer_coord"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
TOP_SHARE = 0.05


def log(m: str) -> None:
    print(f"[patch-steer-report] {m}", flush=True)


def sensitivity(maps: pd.DataFrame, arm: str, layer: int) -> dict[str, np.ndarray]:
    """Per image: the symmetric response ``(+dose - -dose) / 2`` when both
    doses exist, else the single dose's delta."""
    g = maps[(maps["arm"] == arm) & (maps["layer"] == layer)]
    out = {}
    for path, d in g.groupby("image_path"):
        by = {float(r["dose"]): np.asarray(r["values"], dtype=np.float64)
              for _, r in d.iterrows()}
        doses = sorted(by)
        if len(doses) >= 2 and doses[0] < 0 < doses[-1]:
            out[path] = (by[doses[-1]] - by[doses[0]]) / 2.0
        else:
            out[path] = by[doses[-1]] / abs(doses[-1])
    return out


def score_case(case: str) -> tuple[list[dict], dict]:
    maps = pd.read_parquet(ROOT / case / "maps.parquet")
    s = json.loads((ROOT / case / "summary.json").read_text())
    cmap = pd.read_parquet(COORD / case / "maps.parquet")
    c_of = {p: np.asarray(v, dtype=np.float64)
            for p, v in zip(cmap["image_path"], cmap["values"])}
    rows = []
    sens_all = {}
    for layer in s["layers"]:
        arms = sorted(set(maps[maps["layer"] == layer]["arm"]))
        sens = {arm: sensitivity(maps, arm, layer) for arm in arms}
        sens_all[layer] = sens
        for arm in arms:
            rs, tops, mags, ctrl_rs = [], [], [], []
            for path, m in sens[arm].items():
                if path not in c_of:
                    continue
                c = c_of[path]
                k = max(1, round(TOP_SHARE * c.size))
                rs.append(sps.pearsonr(m, c)[0])
                tc = set(np.argsort(np.abs(c))[-k:])
                tm = set(np.argsort(np.abs(m))[-k:])
                tops.append(len(tc & tm) / k)
                mags.append(np.abs(m).mean())
            g = maps[(maps["arm"] == arm) & (maps["layer"] == layer)]
            rows.append({
                "case": case, "layer": int(layer), "arm": arm,
                "n": len(rs),
                "mean_cell_response": float(np.mean(
                    [m.mean() for m in sens[arm].values()])),
                "mean_abs_response": float(np.mean(mags)),
                "median_r_with_cmap": float(np.median(rs)) if rs else np.nan,
                "frac_r_pos": float(np.mean(np.array(rs) > 0)) if rs else np.nan,
                "median_top_overlap": float(np.median(tops)) if tops else np.nan,
                "frac_on_contract": float(g["frac_on_contract"].mean()),
            })
    return rows, {"c_of": c_of, "sens": sens_all, "words": s["words"]}


def figure(case: str, data: dict, layer: int, out: Path, n_show: int = 6) -> None:
    from PIL import Image, ImageDraw
    from research.monocle import causal_maps as cm
    pos, neg = data["words"]
    sens = data["sens"][layer]
    arms = [a for a in ("axis", "control_raw") if a in sens]
    paths = [p for p in sens[arms[0]] if p in data["c_of"]]
    order = np.argsort([data["c_of"][p].sum() for p in paths])
    pick = [paths[i] for i in list(order[: n_show // 2])
            + list(order[-(n_show - n_show // 2):])]
    cols = ["photo", "removal c-map"] + [f"inject {a}" for a in arms]
    vmax_c = float(np.percentile(np.abs(np.concatenate(
        [data["c_of"][p] for p in pick])), 99))
    vmax_i = float(np.percentile(np.abs(np.concatenate(
        [sens["axis"][p] for p in pick])), 99)) if "axis" in sens else vmax_c
    cell, lab = 240, 18
    W = len(cols) * cell
    H = len(pick) * (cell + lab) + 44
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, p in enumerate(pick):
        img = Image.open(p).convert("RGB")
        y = r * (cell + lab)
        sheet.paste(img.resize((cell - 6, cell - 6), Image.LANCZOS), (3, y + 3))
        draw.text((4, y + cell - 2), f"sum c={data['c_of'][p].sum():+.1f}",
                  fill=(40, 40, 40))
        panels = [("removal", data["c_of"][p], vmax_c)]
        for a in arms:
            panels.append((f"inject {a}", sens[a][p], vmax_i))
        for ci, (label, m, vmax) in enumerate(panels, start=1):
            pic = cm.overlay(img, m, vmax)
            x = ci * cell
            sheet.paste(pic.resize((cell - 6, cell - 6), Image.LANCZOS),
                        (x + 3, y + 3))
            txt = label if ci == 1 else (
                f"{label}  r={sps.pearsonr(m, data['c_of'][p])[0]:+.2f}")
            draw.text((x + 4, y + cell - 2), txt, fill=(40, 40, 40))
    sheet.paste(cm.colour_bar(min(W // 2 - 40, 400), 34, vmax_c,
                              teal_label=pos, coral_label=neg), (20, H - 38))
    sheet.paste(cm.colour_bar(min(W // 2 - 40, 400), 34, vmax_i,
                              teal_label=pos, coral_label=neg),
                (W // 2 + 20, H - 38))
    sheet.save(out / f"inject_{case}_L{layer}.png")


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
            log(f"{case}: no run yet")
            continue
        r, data = score_case(case)
        rows += r
        if not args.no_figures:
            for layer in data["sens"]:
                figure(case, data, layer, out)
        log(f"{case}: scored")
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("\nPER-CELL INJECTION of the patch axis, against the removal map")
    print(df.round(3).to_string(index=False))
    df.to_parquet(ROOT / "patch_steer_report.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
