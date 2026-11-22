"""Patch maps of the steering work, in the causal-maps figure style.

The `causal_maps` figures paint each image cell coral or teal by what the
cell contributes to the judgment, in logits, on one shared colour bar.
This script draws the same figure for the 4 maps this work can read from
one photograph, all at one block:

| Map | Source | Units |
|---|---|---|
| ``c`` | the measured change in the judgment when the cell goes away | logits |
| ``axis`` | the change of the answer residual along the judgment axis, times the axis gain | logits |
| ``pinv`` | the same along the pinv vector, times the pinv gain | logits |
| ``grad`` | the same along the image's own gradient, times its norm | logits, first order |

The residual changes come from the recorded answer residuals of
`monocle.steer_coord`, the axis and the pinv vector from the 4-arm sweep,
the gains from `research/scripts/steer_axis_report.py`, and the gradients from
`monocle.grad_align`. No GPU is needed.

Outputs, under ``outputs/_monocle/steer_coord/figures/patch/``:

| File | Content |
|---|---|
| ``contact_<case>_<map>_L<l>.png`` | the 60-image contact sheet of one map, the `causal_maps` layout |
| ``compare_<case>_L<l>.png`` | 6 images: photograph, then the 4 maps, one colour bar |
| ``map_agreement.parquet`` | per case and map: median r with the c-map, top-cell overlap, scale |

Usage:
    python research/scripts/steer_patch_figures.py --cases all
    python research/scripts/steer_patch_figures.py --cases restaurants --layer 41
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats as sps

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import causal_maps as cm  # noqa: E402
from research.monocle.single_steer import CASE_WORDS  # noqa: E402
from research.monocle.steer_coord import contribution_sign, delta_map  # noqa: E402

COORD = REPO / "outputs/_monocle/steer_coord"
AXIS = REPO / "outputs/_monocle/single_steer_axis"
GRAD = REPO / "outputs/_monocle/grad_align"
OUT_DEFAULT = COORD / "figures/patch"
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
MAPS = ("c", "axis", "pinv", "grad")
TOP_SHARE = 0.05


def log(m: str) -> None:
    print(f"[steer-patch-figures] {m}", flush=True)


def gains_of(case: str, layer: int) -> dict[str, float]:
    rep = pd.read_parquet(AXIS / "axis_report.parquet")
    q = rep[(rep["case"] == case) & (rep["layer"] == layer)]
    out = {}
    for arm, name in (("axis", "axis"), ("signal", "pinv")):
        r = q[q["arm"] == arm]
        if not len(r):
            raise ValueError(f"{case}: no {arm} gain at L{layer}")
        out[name] = float(r["gain"].iloc[0])
    return out


def load_case(case: str, layer: int) -> Optional[dict]:
    """Every map of every eval image, in logits. None when a store is
    missing."""
    need = [COORD / case / "answer_acts.npy", COORD / case / "maps.parquet",
            AXIS / case / "vectors.npz", GRAD / case / "eval_grads.npy",
            GRAD / case / "cosines.parquet"]
    if not all(p.exists() for p in need):
        return None
    maps = pd.read_parquet(COORD / case / "maps.parquet")
    stats = pd.read_parquet(COORD / case / "map_stats.parquet")
    coord_vecs = dict(np.load(COORD / case / "vectors.npz"))
    axis_vecs = dict(np.load(AXIS / case / "vectors.npz"))
    layers = [int(x) for x in coord_vecs["layers"]]
    j = layers.index(layer)
    acts = np.load(COORD / case / "answer_acts.npy", mmap_mode="r")
    grads = np.load(GRAD / case / "eval_grads.npy")
    cos = pd.read_parquet(GRAD / case / "cosines.parquet")
    cos = cos[cos["layer"] == layer].reset_index(drop=True)
    n = min(len(maps), acts.shape[0], grads.shape[0])
    if cos["image_path"].tolist()[:n] != maps["image_path"].tolist()[:n]:
        raise SystemExit(f"{case}: the gradient store and the map store "
                         "hold different images")
    gj = [int(x) for x in np.load(GRAD / case / "grad_dirs.npz")["layers"]]
    gains = gains_of(case, layer)
    v_axis = axis_vecs[f"v_axis_L{layer}"]
    v_pinv = coord_vecs[f"v_signal_L{layer}"]
    items = []
    for i in range(n):
        a = np.asarray(acts[i, :, j, :], dtype=np.float32)
        g = grads[i, gj.index(layer)].astype(np.float64)
        g /= np.linalg.norm(g)
        norm = float(cos["grad_norm"].iloc[i])
        items.append({
            "image_path": maps["image_path"].iloc[i],
            "c": float(stats["c"].iloc[i]),
            "maps": {
                "c": np.asarray(maps["values"].iloc[i], dtype=np.float64),
                "axis": contribution_sign(delta_map(a, v_axis)) * gains["axis"],
                "pinv": contribution_sign(delta_map(a, v_pinv)) * gains["pinv"],
                "grad": contribution_sign(delta_map(a, g)) * norm,
            },
        })
    return {"case": case, "layer": layer, "items": items, "gains": gains}


def agreement(data: dict) -> list[dict]:
    items = data["items"]
    k = max(1, round(TOP_SHARE * items[0]["maps"]["c"].size))
    rows = []
    for name in MAPS[1:]:
        rs, tops, scale = [], [], []
        for it in items:
            c, m = it["maps"]["c"], it["maps"][name]
            rs.append(sps.pearsonr(m, c)[0])
            tc = set(np.argsort(np.abs(c))[-k:])
            tm = set(np.argsort(np.abs(m))[-k:])
            tops.append(len(tc & tm) / k)
            scale.append(np.median(np.abs(m)) / (np.median(np.abs(c)) + 1e-9))
        rows.append({"case": data["case"], "layer": data["layer"],
                     "map": name, "n": len(items),
                     "median_r": float(np.median(rs)),
                     "frac_r_pos": float(np.mean(np.array(rs) > 0)),
                     "median_top_overlap": float(np.median(tops)),
                     "median_scale_over_c": float(np.median(scale))})
    return rows


def contact_sheets(data: dict, out: Path) -> None:
    """One `causal_maps` sheet per map, the shared scale per sheet."""
    pos, neg = CASE_WORDS[data["case"]]
    for name in MAPS:
        vals = [it["maps"][name] for it in data["items"]]
        vmax = float(np.percentile(np.abs(np.concatenate(vals)), 99))
        panels = []
        for it, v in zip(data["items"], vals):
            img = Image.open(it["image_path"]).convert("RGB")
            panels.append((f"c={it['c']:+.1f}", cm.overlay(img, v, vmax)))
        sheet = cm.contact_sheet(panels, vmax, cols=6, teal_label=pos,
                                 coral_label=neg)
        sheet.save(out / f"contact_{data['case']}_{name}_L{data['layer']}.png")


def compare_sheet(data: dict, out: Path, n_show: int = 6) -> None:
    """Photograph, then the 4 maps, on ONE logit scale from the c-map."""
    pos, neg = CASE_WORDS[data["case"]]
    items = data["items"]
    order = np.argsort([it["c"] for it in items])
    pick = list(order[: n_show // 2]) + list(order[-(n_show - n_show // 2):])
    vmax = float(np.percentile(np.abs(np.concatenate(
        [items[i]["maps"]["c"] for i in pick])), 99))
    cell, lab = 240, 18
    cols = 1 + len(MAPS)
    W = cols * cell
    H = len(pick) * (cell + lab) + 44
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, i in enumerate(pick):
        it = items[i]
        img = Image.open(it["image_path"]).convert("RGB")
        y = r * (cell + lab)
        sheet.paste(img.resize((cell - 6, cell - 6), Image.LANCZOS), (3, y + 3))
        draw.text((4, y + cell - 2), f"c={it['c']:+.1f}", fill=(40, 40, 40))
        for cidx, name in enumerate(MAPS, start=1):
            m = it["maps"][name]
            pic = cm.overlay(img, m, vmax)
            x = cidx * cell
            sheet.paste(pic.resize((cell - 6, cell - 6), Image.LANCZOS),
                        (x + 3, y + 3))
            label = name if name == "c" else (
                f"{name}  r={sps.pearsonr(m, it['maps']['c'])[0]:+.2f}")
            draw.text((x + 4, y + cell - 2), label, fill=(40, 40, 40))
    sheet.paste(cm.colour_bar(min(W - 40, 480), 34, vmax, teal_label=pos,
                              coral_label=neg), (20, H - 38))
    sheet.save(out / f"compare_{data['case']}_L{data['layer']}.png")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--layer", type=int, default=41)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--no-contact", action="store_true",
                    help="skip the 60-image sheets")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rows = []
    for case in cases:
        data = load_case(case, args.layer)
        if data is None:
            log(f"{case}: a store is missing, skipped")
            continue
        rows += agreement(data)
        compare_sheet(data, out)
        if not args.no_contact:
            contact_sheets(data, out)
        log(f"{case}: drawn (gains axis {data['gains']['axis']:.2f}, "
            f"pinv {data['gains']['pinv']:.2f})")
    if rows:
        df = pd.DataFrame(rows)
        pd.set_option("display.width", 200)
        print("\nMAP AGREEMENT with the measured c-map, per case")
        print(df.round(3).to_string(index=False))
        df.to_parquet(out / "map_agreement.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
