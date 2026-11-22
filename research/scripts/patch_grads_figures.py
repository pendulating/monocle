"""Per-cell maps from the patch gradients, in the `causal_maps` style.

Per case, one comparison sheet at one block: the photograph, the measured
removal map, the gradient-times-input map ``gxh``, and the first-order
sensitivity map ``<g[cell], v>`` for the patch axis and for the pinv
vector. The removal and ``gxh`` maps share one logit scale; each
sensitivity map has its own, because they are in logits per unit dose
and the two vectors have gains 100 times apart.

Usage:
    python research/scripts/patch_grads_figures.py --cases all --layer 22
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import causal_maps as cm  # noqa: E402
from research.monocle.single_steer import CASE_WORDS  # noqa: E402
from scripts.patch_grads_report import CASES, COORD, GRADS, VECS  # noqa: E402

OUT = GRADS / "figures"


def maps_of(case: str, layer: int) -> list[dict]:
    idx = pd.read_parquet(GRADS / case / "index.parquet")
    import json
    s = json.loads((GRADS / case / "summary.json").read_text())
    layers = [int(l) for l in s["layers"]]
    j = layers.index(layer)
    G = np.load(GRADS / case / "grads.npy", mmap_mode="r")
    R = np.load(GRADS / case / "resid.npy", mmap_mode="r")
    vecs = dict(np.load(VECS / case / "vectors.npz"))
    v_axis = np.asarray(vecs[f"v_axis_L{layer}"], dtype=np.float64)
    v_pinv = np.asarray(vecs[f"v_signal_L{layer}"], dtype=np.float64)
    rem = pd.read_parquet(COORD / case / "maps.parquet")
    removal = {p: np.asarray(v, dtype=np.float64)
               for p, v in zip(rem["image_path"], rem["values"])}
    out = []
    for i, (p, c) in enumerate(zip(idx["image_path"], idx["c"])):
        if p not in removal:
            continue
        g = np.asarray(G[i, j], dtype=np.float64)
        h = np.asarray(R[i, j], dtype=np.float64)
        hc = h - h.mean(0, keepdims=True)
        out.append({"image_path": p, "c": float(c),
                    "removal": removal[p],
                    "gxh": np.einsum("cd,cd->c", g, hc),
                    "sens_axis": g @ v_axis,
                    "sens_pinv": g @ v_pinv})
    return out


def sheet(case: str, items: list[dict], layer: int, n_show: int = 6) -> None:
    from PIL import Image, ImageDraw
    pos, neg = CASE_WORDS[case]
    order = np.argsort([it["c"] for it in items])
    pick = [items[i] for i in list(order[: n_show // 2])
            + list(order[-(n_show - n_show // 2):])]
    cols = ["removal", "gxh", "sens_axis", "sens_pinv"]
    v_shared = float(np.percentile(np.abs(np.concatenate(
        [it["removal"] for it in pick] + [it["gxh"] for it in pick])), 99))
    vmax = {"removal": v_shared, "gxh": v_shared}
    for k in ("sens_axis", "sens_pinv"):
        vmax[k] = float(np.percentile(np.abs(np.concatenate(
            [it[k] for it in pick])), 99))
    cell, lab = 230, 18
    W = (1 + len(cols)) * cell
    H = len(pick) * (cell + lab) + 44
    im = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    for r, it in enumerate(pick):
        img = Image.open(it["image_path"]).convert("RGB")
        y = r * (cell + lab)
        im.paste(img.resize((cell - 6, cell - 6), Image.LANCZOS), (3, y + 3))
        draw.text((4, y + cell - 2), f"c={it['c']:+.1f}", fill=(40, 40, 40))
        for ci, k in enumerate(cols, start=1):
            pic = cm.overlay(img, it[k], vmax[k])
            x = ci * cell
            im.paste(pic.resize((cell - 6, cell - 6), Image.LANCZOS), (x + 3, y + 3))
            txt = k if k == "removal" else (
                f"{k}  r={sps.pearsonr(it[k], it['removal'])[0]:+.2f}")
            draw.text((x + 4, y + cell - 2), txt, fill=(40, 40, 40))
    im.paste(cm.colour_bar(300, 34, v_shared, teal_label=pos, coral_label=neg),
             (20, H - 38))
    draw.text((340, H - 30), f"sens_axis |max| {vmax['sens_axis']:.2f}  "
              f"sens_pinv |max| {vmax['sens_pinv']:.2f}  (logits per unit dose)",
              fill=(40, 40, 40))
    im.save(OUT / f"cells_{case}_L{layer}.png")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--layer", type=int, default=22)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    for case in cases:
        if not (GRADS / case / "summary.json").exists():
            continue
        sheet(case, maps_of(case, args.layer), args.layer)
        print(f"[patch-grads-figures] {case}: drawn", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
