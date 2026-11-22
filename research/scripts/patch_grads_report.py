"""Per-cell alignment of the steering vectors, from the patch gradients.

For each image, block and vector ``v`` two per-cell maps come from the
recorded patch gradients ``g[cell]`` and residuals ``h[cell]``:

| Map | Formula | Reads |
|---|---|---|
| sensitivity | ``<g[cell], v>`` | how much the answer moves when ``v`` enters this cell (first order) |
| coordinate | ``<h[cell] - mean_cells h, v>`` | how far this cell already sits along ``v`` |

and one vector-free map, ``gxh = <g[cell], h[cell] - mean_cells h>``, the
first-order estimate of the cell's own removal effect.

The report scores each map against the measured removal map (the
contribution, `steer_coord`) and, where the per-cell injection runs exist
(`patch_steer`, `patch_steer_pinv`), the sensitivity map against the
measured per-cell injection response. It also reports how concentrated a
sensitivity map is (the share of its mass in the top 5 percent of cells).

Vectors: the patch-fitted judgment axis, the pinv vector, the pinv random
control and the raw random control, all from the patch-position sweep's
``vectors.npz`` (directions only; the dose scale is reported separately).

Usage:
    python research/scripts/patch_grads_report.py --cases all
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

GRADS = REPO / "outputs/_monocle/patch_grads"
VECS = REPO / "outputs/_monocle/patch_axis"
COORD = REPO / "outputs/_monocle/steer_coord"
STEER = {"axis": REPO / "outputs/_monocle/patch_steer",
         "signal": REPO / "outputs/_monocle/patch_steer_pinv"}
CASES = ["subway_safety", "libraries", "schools", "road_quality",
         "parks_plazas", "restaurants", "street_photography"]
VECTORS = ("axis", "signal", "control", "control_raw")
TOP_SHARE = 0.05


def log(m: str) -> None:
    print(f"[patch-grads-report] {m}", flush=True)


def top_mass(m: np.ndarray) -> float:
    a = np.abs(m)
    k = max(1, round(TOP_SHARE * a.size))
    return float(np.sort(a)[-k:].sum() / (a.sum() + 1e-12))


def r_of(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(sps.pearsonr(a, b)[0])


def load_injection(case: str, arm: str, layer: int) -> dict[str, np.ndarray]:
    """The measured per-cell response ``(+dose - -dose) / 2`` per image."""
    f = STEER[arm] / case / "maps.parquet"
    if not f.exists():
        return {}
    df = pd.read_parquet(f)
    df = df[(df["arm"] == arm) & (df["layer"] == layer)]
    out = {}
    for p, d in df.groupby("image_path"):
        by = {float(r["dose"]): np.asarray(r["values"], dtype=np.float64)
              for _, r in d.iterrows()}
        doses = sorted(by)
        if len(doses) >= 2 and doses[0] < 0 < doses[-1]:
            out[p] = (by[doses[-1]] - by[doses[0]]) / 2.0
    return out


def load_removal(case: str, root: Path, kind: str) -> dict[str, np.ndarray]:
    """The removal CONTRIBUTION per image. ``kind`` names what the store
    holds: ``contribution`` (`steer_coord`) or ``delta`` at 280 tokens
    (`patch_budget`, whose negative is the contribution)."""
    if str(root) == "none":
        # A model with no removal maps (E4B): the removal columns stay NaN.
        return {}
    cf = root / case / "maps.parquet"
    if not cf.exists():
        return {}
    df = pd.read_parquet(cf)
    if kind == "delta":
        if "budget" in df.columns:
            df = df[df["budget"] == 280]
        return {p: -np.asarray(v, dtype=np.float64)
                for p, v in zip(df["image_path"], df["values"])}
    return {p: np.asarray(v, dtype=np.float64)
            for p, v in zip(df["image_path"], df["values"])}


def score_case(case: str, grads_root: Path = GRADS, vecs_root: Path = VECS,
               removal_root: Path = COORD,
               removal_kind: str = "contribution") -> list[dict]:
    s = json.loads((grads_root / case / "summary.json").read_text())
    layers = [int(l) for l in s["layers"]]
    idx = pd.read_parquet(grads_root / case / "index.parquet")
    G = np.load(grads_root / case / "grads.npy", mmap_mode="r")
    R = np.load(grads_root / case / "resid.npy", mmap_mode="r")
    vf = vecs_root / case / "vectors.npz"
    if vf.exists():
        vecs = dict(np.load(vf))
        v_layers = [int(x) for x in vecs["layers"]]
    else:
        # No sweep for this case: the gxh rows still score.
        vecs, v_layers = {}, []
    removal = load_removal(case, removal_root, removal_kind)
    rows = []
    for j, l in enumerate(layers):
        inj = {arm: load_injection(case, arm, l) for arm in STEER}
        scale = float(vecs["scales"][v_layers.index(l)]) if l in v_layers else np.nan
        per_vec = {name: {"r_removal": [], "r_coord_removal": [], "r_inj": [],
                          "slope_inj": [], "top_mass": [], "cos_med": []}
                   for name in VECTORS}
        gxh_r = []
        for i, p in enumerate(idx["image_path"]):
            g = np.asarray(G[i, j], dtype=np.float64)
            h = np.asarray(R[i, j], dtype=np.float64)
            hc = h - h.mean(0, keepdims=True)
            gxh = np.einsum("cd,cd->c", g, hc)
            if p in removal:
                # removal is a contribution (-delta); gxh predicts -delta too:
                # the cell's own content, projected on the cell's gradient.
                gxh_r.append(r_of(gxh, removal[p]))
            gn = np.linalg.norm(g, axis=1) + 1e-12
            for name in VECTORS:
                key = f"v_{name}_L{l}"
                if key not in vecs:
                    continue
                v = np.asarray(vecs[key], dtype=np.float64)
                sens = g @ v
                coord = hc @ v
                d = per_vec[name]
                d["top_mass"].append(top_mass(sens))
                d["cos_med"].append(float(np.median(sens / gn)))
                if p in removal:
                    d["r_removal"].append(r_of(sens, removal[p]))
                    d["r_coord_removal"].append(r_of(coord, removal[p]))
                m = inj.get(name, {}).get(p)
                if m is not None:
                    pred = sens * scale  # per unit dose
                    d["r_inj"].append(r_of(pred, m))
                    d["slope_inj"].append(float(np.polyfit(pred, m, 1)[0])
                                          if pred.std() > 0 else np.nan)
        for name, d in per_vec.items():
            if not d["top_mass"]:
                continue
            med = lambda k: float(np.nanmedian(d[k])) if d[k] else np.nan  # noqa: E731
            rows.append({
                "case": case, "layer": l, "vector": name, "n": len(d["top_mass"]),
                "sens_top5_mass": med("top_mass"),
                "cos_g_v_median": med("cos_med"),
                "r_sens_vs_removal": med("r_removal"),
                "r_coord_vs_removal": med("r_coord_removal"),
                "n_inj": len(d["r_inj"]),
                "r_sens_vs_injection": med("r_inj"),
                "slope_injection_over_firstorder": med("slope_inj"),
            })
        rows.append({"case": case, "layer": l, "vector": "gxh (no vector)",
                     "n": len(gxh_r),
                     "r_sens_vs_removal": float(np.nanmedian(gxh_r)) if gxh_r else np.nan})
    return rows


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    ap.add_argument("--grads-root", default=str(GRADS))
    ap.add_argument("--vecs-root", default=str(VECS),
                    help="a sweep root with <case>/vectors.npz")
    ap.add_argument("--removal-root", default=str(COORD))
    ap.add_argument("--removal-kind", choices=["contribution", "delta"],
                    default="contribution")
    args = ap.parse_args(argv)
    grads_root = Path(args.grads_root)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rows = []
    for case in cases:
        if not (grads_root / case / "summary.json").exists():
            log(f"{case}: no run yet")
            continue
        rows += score_case(case, grads_root, Path(args.vecs_root),
                           Path(args.removal_root), args.removal_kind)
        log(f"{case}: scored")
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print("\nPER-CELL ALIGNMENT from the patch gradients")
    print(df.round(3).to_string(index=False))
    df.to_parquet(grads_root / "patch_grads_report.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
