"""A causal map fitted on SETS, not on single cells.

Why leave-one-out is the wrong ground truth
--------------------------------------------
The measured map deletes 1 cell at a time. That is a MARGINAL effect. The
deletion test removes a SET, which is a JOINT effect. Measured on this model
the 2 do not agree: a 2x2 block moves `c` about 5 times less than the sum of
its 4 patches, and at 8x8 the sum carries no information at all.

So a map fitted on singletons cannot rank sets, even when it is exact.

What this module fits
---------------------
For each image it deletes M RANDOM sets of cells and records the change in
`c`. Then it regresses that change on the 64 cell indicators:

    dc(S) = sum over c in S of w_c + b

The fitted `w` is the best ADDITIVE summary of the joint effect, in the least
squares sense over the sampled sets. It is a Banzhaf-style value. The map it
gives is calibrated for the operation the deletion test does.

The sign
--------
`dc` falls when a cell that holds up the positive label goes away, so the
CONTRIBUTION of a cell is `-w_c`. The module writes the contribution, which
matches the sign of the measured map.

The set sizes
-------------
A design that includes each cell with probability one half puts all its mass
on 32-cell sets, and the test reads sets of 1 to 8. So the sizes are drawn
from the same grid the test uses, which keeps the fit near the operating
point.

Usage:
    python -m monocle.subset_map --cases subway_safety --n-images 20 \
        --group 2 --n-sets 320 --tag subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import deletion_curve as dcv  # noqa: E402
from monocle import patch_ablate as pab  # noqa: E402
from monocle import patch_attribution as pa  # noqa: E402
from monocle import patch_map_causal as pmc  # noqa: E402
from monocle import single as sg  # noqa: E402

DEFAULT_SIZES = (1, 2, 4, 8, 16)


def log(m: str) -> None:
    print(m, flush=True)


def draw_sets(n_cell: int, n_sets: int, sizes: tuple,
              rng: np.random.Generator) -> list[list[int]]:
    """`n_sets` random cell sets, with the sizes spread over `sizes`."""
    out = []
    for i in range(n_sets):
        k = int(sizes[i % len(sizes)])
        out.append(sorted(rng.choice(n_cell, size=k, replace=False).tolist()))
    return out


def fit_values(sets: list[list[int]], dc: np.ndarray, n_cell: int,
               alpha: float) -> np.ndarray:
    """Ridge from the cell indicators to the change in `c`.

    The design holds an intercept, so a set of any size is centered.
    """
    x = np.zeros((len(sets), n_cell + 1), dtype=np.float64)
    for i, s in enumerate(sets):
        x[i, s] = 1.0
    x[:, n_cell] = 1.0
    a = x.T @ x + alpha * np.eye(n_cell + 1)
    a[n_cell, n_cell] -= alpha          # never shrink the intercept
    w = np.linalg.solve(a, x.T @ dc)
    return w[:n_cell]


def holdout_r2(sets: list[list[int]], dc: np.ndarray, n_cell: int,
               alpha: float, rng: np.random.Generator) -> float:
    """How much of the set effect an ADDITIVE map can explain, held out."""
    idx = rng.permutation(len(sets))
    cut = len(sets) // 5
    te, tr = idx[:cut], idx[cut:]
    w = fit_values([sets[i] for i in tr], dc[tr], n_cell, alpha)
    pred = np.array([w[sets[i]].sum() for i in te])
    y = dc[te]
    pred = pred - pred.mean()
    y0 = y - y.mean()
    ss = float((y0 ** 2).sum())
    if ss == 0:
        return float("nan")
    # A scale-free score: the fit may miss the slope but keep the order.
    beta = float((pred @ y0) / max((pred @ pred), 1e-12))
    return 1.0 - float(((y0 - beta * pred) ** 2).sum()) / ss


def run_case(
    case: str, *, proc: Any, model: Any, lens_model: Any, tmpl: str,
    n_images: int, group: int, device: str, out_dir: Path, seed: int,
    n_sets: int, sizes: tuple, alpha: float, batch: int,
) -> pd.DataFrame:
    from PIL import Image

    tok = proc.tokenizer
    pos_id, neg_id = sg.contrast_ids(tok, case)
    image_token_id = int(model.config.image_token_id)
    rows, maps = [], []
    t0 = time.time()
    for i, r in enumerate(sg.load_single_rows(n_images, case, seed=seed)):
        rng = np.random.default_rng(seed + 1000 * i)
        img = Image.open(r["path"]).convert("RGB")
        inputs = sg.build_single_inputs(proc, tmpl, img, case, device=device)
        slots = pab.slots_of(inputs, image_token_id)
        blocks = (pmc.patch_groups(len(slots), group) if group > 1
                  else [[j] for j in range(len(slots))])
        n_cell = len(blocks)
        c0 = pmc.batched_baseline(model, lens_model, inputs, pos_id, neg_id,
                                  batch=batch)
        sets = draw_sets(n_cell, n_sets, sizes, rng)
        flat = [dcv.cells_to_patches(s, blocks) for s in sets]
        # `ablation_batch` refuses a mixed size, because an ablation shortens
        # the sequence and 2 lengths cannot share a batch. So we group the
        # sets by their patch count and put the answers back in order.
        by_size: dict[int, list[int]] = {}
        for i, f in enumerate(flat):
            by_size.setdefault(len(f), []).append(i)
        dc = np.zeros(len(flat), dtype=np.float64)
        for size, idx in sorted(by_size.items()):
            part = dcv.measure_dc(
                model, lens_model, inputs, slots, [flat[i] for i in idx],
                pos_id, neg_id, image_token_id, batch, "ablate", None)
            for j, i in enumerate(idx):
                dc[i] = float(part[j])
        w = fit_values(sets, dc, n_cell, alpha)
        r2 = holdout_r2(sets, dc, n_cell, alpha, rng)
        # The CONTRIBUTION: a cell that holds up the positive label makes `c`
        # fall when it goes, so we negate the fitted weight.
        values = -w
        maps.append({"image_id": str(r["image_id"]),
                     "values": values.astype(float).tolist()})
        rows.append({"case": case, "image_id": str(r["image_id"]),
                     "c0": c0, "n_sets": len(sets), "n_cells": n_cell,
                     "additive_r2": r2,
                     "dc_sd": float(dc.std()),
                     "w_absmax": float(np.abs(w).max())})
        log(f"  {i + 1}/{n_images} {r['image_id']} c0={c0:+.2f} "
            f"additive R2={r2:+.3f} dc_sd={dc.std():.2f} "
            f"{time.time() - t0:.0f}s")
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(case_dir / "summary.parquet", index=False)
    pd.DataFrame(maps).to_parquet(case_dir / "maps.parquet", index=False)
    pd.DataFrame(maps).to_parquet(case_dir / "ranking.parquet", index=False)
    log(f"wrote {case_dir}; mean held-out additive R2 = "
        f"{df['additive_r2'].mean():+.3f}")
    return df


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["subway_safety"])
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--n-sets", type=int, default=320)
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=list(DEFAULT_SIZES))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--tag", default="subset")
    ap.add_argument("--model-dir", default=pa.MODEL_DIR)
    ap.add_argument("--device", default="cuda:0")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from monocle import attn_route as ar
    from monocle import jlens_read

    proc, model, tmpl = ar.load_eager(args.model_dir, args.device)
    lens_model = jlens_read.wrap_for_unembed(model, proc.tokenizer)
    out_dir = Path("outputs/_monocle/subset_map") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    for case in args.cases:
        run_case(case, proc=proc, model=model, lens_model=lens_model,
                 tmpl=tmpl, n_images=args.n_images, group=args.group,
                 device=args.device, out_dir=out_dir, seed=args.seed,
                 n_sets=args.n_sets, sizes=tuple(args.sizes),
                 alpha=args.alpha, batch=args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
