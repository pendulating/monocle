#!/usr/bin/env python
"""Cross-lens agreement at the answer position alone.

Why a second number
-------------------
`monocle/judgment_trace.py` prints one agreement per layer. That number
correlates the 2 lenses over EVERY position of 1 image, about 3000 positions,
and `AGREE_FLOOR = 0.8` gates it. It asks a wide question: do the 2 lenses
paint the same picture of the whole prompt?

The depth claim is narrow. It reads 2 label rows at 1 position, the answer
position, and it compares photographs. The correct agreement for that claim
correlates the 2 lenses ACROSS IMAGES at that 1 position. This script
measures it.

| Read | Unit of the correlation | n for each layer |
|---|---|---|
| answer | 1 answer position for each image | the images |
| image | every image patch, all images pooled | images x patches |

The `image` block is the regime the wide floor lives in. It is here as the
contrast, not as the answer.

Scale
-----
The answer read uses the RAW `s`. One position for each image gives no
positions to centre over, so `s_centered` is not defined for this unit. The
image block reports both, because the wide floor uses the centred column.

Run
---
    .venv-3.12/bin/python scripts/lens_agreement_answer.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/share/ju/matt/mllmsci")
TRACE_DEFAULT = (
    REPO / "outputs/_monocle/judgment_trace/subway_safety/trace.parquet")
OUT_DEFAULT = (
    REPO / "outputs/_monocle/judgment_trace/subway_safety/agreement_answer.json")

#: The floor that `judgment_trace` applies to the wide read.
AGREE_FLOOR = 0.8


def log(m: str) -> None:
    print(f"[agree] {m}", flush=True)


def pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pearson r and its two-sided p. A flat input gives nan."""
    if x.size < 3 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return (float("nan"), float("nan"))
    r, p = stats.pearsonr(x, y)
    return (float(r), float(p))


def sign_agree(x: np.ndarray, y: np.ndarray) -> float:
    """The fraction of units where the 2 lenses give the same sign."""
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.sign(x) == np.sign(y)))


def pair_lenses(
    sub: pd.DataFrame, lens_a: str, lens_b: str, column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the 2 lenses on (image_id, pos) and give the 2 vectors."""
    wide = sub.pivot_table(
        index=["image_id", "pos"], columns="lens", values=column,
        aggfunc="first")
    if lens_a not in wide.columns or lens_b not in wide.columns:
        return (np.array([]), np.array([]))
    wide = wide[[lens_a, lens_b]].dropna().sort_index()
    return (wide[lens_a].to_numpy(float), wide[lens_b].to_numpy(float))


def block(
    df: pd.DataFrame, group: str, lens_a: str, lens_b: str,
) -> list[dict]:
    """One record for each layer, for 1 position group."""
    rows: list[dict] = []
    sel = df[df["group"] == group]
    for layer, sub in sel.groupby("layer", sort=True):
        rec: dict = {"group": group, "layer": int(layer)}
        rec["n_images"] = int(sub["image_id"].nunique())
        for column, tag in (("s", "raw"), ("s_centered", "centered")):
            x, y = pair_lenses(sub, lens_a, lens_b, column)
            r, p = pearson(x, y)
            rec[f"n_{tag}"] = int(x.size)
            rec[f"pearson_{tag}"] = r
            rec[f"p_{tag}"] = p
            rec[f"sign_agree_{tag}"] = sign_agree(x, y)
        rec["trustworthy_raw"] = bool(
            np.isfinite(rec["pearson_raw"])
            and rec["pearson_raw"] >= AGREE_FLOOR)
        rows.append(rec)
    return rows


def within_image_wide(
    df: pd.DataFrame, lens_a: str, lens_b: str,
) -> list[dict]:
    """The wide read, as `judgment_trace` computes it: all positions, 1 image.

    The median over the images is the number the trace log prints.
    """
    rows: list[dict] = []
    for layer, sub in df.groupby("layer", sort=True):
        per_image = []
        for _, one in sub.groupby("image_id"):
            x, y = pair_lenses(one, lens_a, lens_b, "s_centered")
            r, _ = pearson(x, y)
            per_image.append(r)
        arr = np.asarray(per_image, dtype=float)
        rows.append({
            "layer": int(layer),
            "n_images": int(arr.size),
            "pearson_median": float(np.nanmedian(arr)),
            "pearson_min": float(np.nanmin(arr)),
            "pearson_max": float(np.nanmax(arr)),
        })
    return rows


def print_tables(
    answer: list[dict], image: list[dict], wide: list[dict],
    lens_a: str, lens_b: str,
) -> None:
    log("")
    log(f"answer position, raw `s`, correlated across images "
        f"({lens_a} against {lens_b})")
    log(f"{'layer':>6} {'n_img':>6} {'pearson':>9} {'p':>10} {'sign':>6} "
        f"{'>=0.8':>6}")
    for r in answer:
        log(f"{r['layer']:>6} {r['n_raw']:>6} {r['pearson_raw']:>9.3f} "
            f"{r['p_raw']:>10.3e} {r['sign_agree_raw']:>6.2f} "
            f"{'yes' if r['trustworthy_raw'] else 'no':>6}")

    log("")
    log("image patches, pooled over the images (the wide floor's regime)")
    log(f"{'layer':>6} {'n':>7} {'raw_r':>9} {'raw_sign':>9} "
        f"{'cent_r':>9} {'cent_sign':>10}")
    for r in image:
        log(f"{r['layer']:>6} {r['n_raw']:>7} {r['pearson_raw']:>9.3f} "
            f"{r['sign_agree_raw']:>9.2f} {r['pearson_centered']:>9.3f} "
            f"{r['sign_agree_centered']:>10.2f}")

    log("")
    log("for reference: the wide read of `judgment_trace`, all positions, "
        "1 image, centred")
    log(f"{'layer':>6} {'median_r':>9} {'min':>9} {'max':>9}")
    for r in wide:
        log(f"{r['layer']:>6} {r['pearson_median']:>9.3f} "
            f"{r['pearson_min']:>9.3f} {r['pearson_max']:>9.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, default=TRACE_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    df = pd.read_parquet(args.trace)
    lenses = sorted(df["lens"].unique().tolist())
    if len(lenses) < 2:
        raise SystemExit(f"{args.trace} holds {lenses}; 2 lenses are needed")
    lens_a, lens_b = lenses[0], lenses[1]
    log(f"{args.trace}: {len(df)} rows, {df['image_id'].nunique()} images, "
        f"lenses {lenses}")
    log(f"groups {sorted(df['group'].unique())}")

    answer = block(df, "answer", lens_a, lens_b)
    image = block(df, "image", lens_a, lens_b)
    wide = within_image_wide(df, lens_a, lens_b)
    print_tables(answer, image, wide, lens_a, lens_b)

    payload = {
        "trace": str(args.trace),
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lens_a": lens_a,
        "lens_b": lens_b,
        "agree_floor": AGREE_FLOOR,
        "n_images": int(df["image_id"].nunique()),
        "cases": sorted(df["case"].unique().tolist()),
        "answer_position": answer,
        "image_positions_pooled": image,
        "wide_within_image": wide,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    log("")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
