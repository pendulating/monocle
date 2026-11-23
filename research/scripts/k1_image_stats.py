"""Is the reader only edge energy, brightness or colour?

A photograph gives a 16x16 grid of cells. Each cell has plain image
statistics that no model computes: brightness, contrast, edge energy,
colour. Each statistic is a candidate reader, and each is blind to the
question. So the test runs in 2 parts.

1. How well does each statistic separate the poles of the measured map?
   That is the question-blind ceiling.
2. Take every statistic out of the truth map, cell by cell, with 1
   regression for each photograph. Then score `gxa` again. If the gain
   holds, the reader does not read the image statistics.

    python research/scripts/k1_image_stats.py \
        --root outputs/_monocle/gateB60_g1L \
        --xq-root outputs/_monocle/gateB60_g1L_xq \
        --stats-cache outputs/_monocle/k1_image_stats_cache.npz \
        --cases restaurants libraries subway_safety
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from research.scripts.k1_separation import bh, boot_ci, sep_z  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SIZE = 896
STATS = ("lum", "contrast", "edge", "sat", "row", "col", "centre")


def cell_stats(path: str, side: int) -> dict[str, np.ndarray]:
    """One value for each cell of the 16x16 grid, for each statistic.

    The picture is made square at 896 pixels first, which is the same
    step the crop and the model take. So a cell here is the cell the
    model sees.
    """
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    lum = a.mean(axis=2)
    mx, mn = a.max(axis=2), a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gy, gx = np.gradient(lum)
    grad = np.hypot(gx, gy)

    step = SIZE // side
    out = {k: np.zeros(side * side, dtype=float) for k in STATS}
    for r in range(side):
        for c in range(side):
            sl = (slice(r * step, (r + 1) * step),
                  slice(c * step, (c + 1) * step))
            i = r * side + c
            out["lum"][i] = float(lum[sl].mean())
            out["contrast"][i] = float(lum[sl].std())
            out["edge"][i] = float(grad[sl].mean())
            out["sat"][i] = float(sat[sl].mean())
            out["row"][i] = float(r)
            out["col"][i] = float(c)
            out["centre"][i] = -float(
                (r - (side - 1) / 2) ** 2 + (c - (side - 1) / 2) ** 2)
    return out


def pooled_gxa(path: Path, arm: str) -> dict[str, np.ndarray]:
    df = pd.read_parquet(path)
    out = {}
    for key, g in df.groupby("target"):
        t = np.asarray(g.iloc[0]["causal"], dtype=float).ravel()
        rk = []
        for _, r in g.iterrows():
            v = r.get(arm)
            if v is None:
                continue
            v = np.asarray(v, dtype=float).ravel()
            if v.size == t.size and np.isfinite(v).all():
                rk.append(rankdata(v))
        if rk:
            out[str(key)] = (np.mean(rk, axis=0), t)
    return out


def residualise(t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Take the image statistics out of the truth of 1 photograph."""
    A = np.column_stack([np.ones(t.size), X])
    beta, *_ = np.linalg.lstsq(A, t, rcond=None)
    return t - A @ beta


def paths_of(case: str, n_images: int, seed: int) -> dict[str, str]:
    from research.monocle import single as sg
    return {str(r["image_id"]): str(r["path"])
            for r in sg.load_single_rows(n_images, case, seed=seed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--xq-root", required=True)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--arm", default="gxa")
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--boot-seed", type=int, default=20260825)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default="outputs/_monocle/k1_image_stats.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.boot_seed)
    root, xq = Path(args.root), Path(args.xq_root)
    stat_rows, resid_rows = [], []

    for case in args.cases:
        a, b = root / case / "maps.parquet", xq / case / "maps.parquet"
        if not (a.exists() and b.exists()):
            print(f"skip {case}")
            continue
        A, B = pooled_gxa(a, args.arm), pooled_gxa(b, args.arm)
        keys = sorted(set(A) & set(B))
        pmap = paths_of(case, args.n_images, args.seed)
        side = int(round(float(A[keys[0]][1].size) ** 0.5))

        S: dict[str, dict[str, np.ndarray]] = {}
        for k in keys:
            if k in pmap:
                S[k] = cell_stats(pmap[k], side)
        ks = [k for k in keys if k in S]
        print(f"{case}: {len(ks)} photographs with pixels")

        # 1. every statistic as its own reader, against the measured map
        for st in STATS:
            z = np.array([sep_z(S[k][st], A[k][1]) for k in ks])
            zn = np.array([sep_z(-S[k][st], A[k][1]) for k in ks])
            best = z if abs(z.mean()) >= abs(zn.mean()) else zn
            sign = "+" if best is z else "-"
            lo, hi = boot_ci(best, args.n_boot, rng, "mean")
            stat_rows.append({"case": case, "stat": f"{sign}{st}",
                              "n": int(best.size), "sep": float(best.mean()),
                              "lo": lo, "hi": hi,
                              "p": float(wilcoxon(best).pvalue)})
        zg = np.array([sep_z(A[k][0], A[k][1]) for k in ks])
        stat_rows.append({"case": case, "stat": args.arm, "n": int(zg.size),
                          "sep": float(zg.mean()),
                          "lo": boot_ci(zg, args.n_boot, rng, "mean")[0],
                          "hi": boot_ci(zg, args.n_boot, rng, "mean")[1],
                          "p": float(wilcoxon(zg).pvalue)})

        # 2. the gain after every statistic leaves the truth map
        for name, cols in (("raw", ()),
                           ("no_pixels", ("lum", "contrast", "edge", "sat")),
                           ("no_layout", ("row", "col", "centre")),
                           ("no_both", STATS)):
            dr, dw = [], []
            for k in ks:
                t = A[k][1]
                if cols:
                    X = np.column_stack([S[k][c] for c in cols])
                    t = residualise(t, X)
                dr.append(sep_z(A[k][0], t))
                dw.append(sep_z(B[k][0], t))
            d = np.array(dr) - np.array(dw)
            lo, hi = boot_ci(d, args.n_boot, rng, "mean")
            resid_rows.append({"case": case, "variant": name,
                               "n": int(d.size),
                               "real": float(np.mean(dr)),
                               "wrong": float(np.mean(dw)),
                               "gain": float(d.mean()), "lo": lo, "hi": hi,
                               "p": float(wilcoxon(d).pvalue)})

    for rows in (stat_rows, resid_rows):
        if rows:
            q = bh(np.array([r["p"] for r in rows]))
            for r, v in zip(rows, q):
                r["p_bh"] = float(v)

    print("\n=== every statistic as a reader, against the measured map ===")
    print(f"{'case':>18} {'stat':>10} {'separation':>11} {'95% CI':>18} {'q':>8}")
    for r in stat_rows:
        print(f"{r['case']:>18} {r['stat']:>10} {r['sep']:+11.3f} "
              f"[{r['lo']:+6.3f},{r['hi']:+6.3f}] {r['p_bh']:8.4f}")

    print("\n=== the gain after the statistics leave the truth map ===")
    print(f"{'case':>18} {'variant':>10} {'own':>7} {'wrong':>7} {'gain':>7} "
          f"{'95% CI':>18} {'q':>8}")
    for r in resid_rows:
        print(f"{r['case']:>18} {r['variant']:>10} {r['real']:+7.3f} "
              f"{r['wrong']:+7.3f} {r['gain']:+7.3f} "
              f"[{r['lo']:+6.3f},{r['hi']:+6.3f}] {r['p_bh']:8.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"stats": stat_rows, "resid": resid_rows}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
