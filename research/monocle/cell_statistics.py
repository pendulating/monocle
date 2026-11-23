"""What is IN the squares the map calls dangerous?

The gap this fills
------------------
The deletion and substitution tests show that the map's order is real and that
it is about content. Neither says WHAT the content is. A reader still cannot
check the caption "the model finds this square dangerous" against the picture.

There are no human labels here, so the anchor is measurable image statistics.
For every cell of every map, compute a few properties a person can name, then
ask which of them the map value tracks:

| Property | Why it is worth asking |
|---|---|
| luminance | is a dark square read as dangerous? |
| contrast | is clutter read as dangerous? |
| edge density | is structure read as safe, and blank read as dangerous? |
| saturation | is a grey scene read as dangerous? |
| row (height in frame) | is the sky safe and the ground dangerous? |
| column | is there a left-right bias, which would be an artefact |

A statement like "the model reads dark, low-edge squares as dangerous" is
checkable by eye and is the first thing here that a reader can falsify.

What a null would mean
----------------------
If no simple property predicts the map, the model keys on something these
statistics cannot see. That is a real result too, and it bounds how far a
heat map alone can take a reader.

The `column` row is a control. A left-right effect has no meaning in a street
photograph, so a strong column effect would say the map carries a positional
artefact rather than content.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]

OUT_DEFAULT = REPO / "outputs/_monocle/cell_statistics"
#: The model sees a 768 px square, whatever the file holds.
MODEL_SIDE = 768


def log(m: str) -> None:
    print(f"[cellstats] {m}", flush=True)


def cell_boxes(n_cells: int, side_px: int = MODEL_SIDE) -> list[tuple]:
    """(left, top, right, bottom) for each cell, in row-major order."""
    side = int(round(n_cells ** 0.5))
    if side * side != n_cells:
        raise ValueError(f"{n_cells} cells is not a square grid")
    step = side_px / side
    out = []
    for r in range(side):
        for c in range(side):
            out.append((int(c * step), int(r * step),
                        int((c + 1) * step), int((r + 1) * step)))
    return out


def cell_stats(image: Image.Image, n_cells: int) -> pd.DataFrame:
    """1 row for each cell: the properties a person can name.

    The photograph is resized to what the model actually reads, so a cell here
    covers the same pixels the model's soft token covered.
    """
    side = int(round(n_cells ** 0.5))
    img = image.convert("RGB").resize((MODEL_SIDE, MODEL_SIDE),
                                      Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    # A cheap gradient magnitude: no scipy, and the units do not matter
    # because every column is standardized before the fit.
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    edge = gx + gy
    mx, mn = a.max(-1), a.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    rows = []
    for i, (l, t, r, b) in enumerate(cell_boxes(n_cells)):
        sl = (slice(t, b), slice(l, r))
        rows.append({
            "cell": i, "row": i // side, "col": i % side,
            "luminance": float(lum[sl].mean()),
            "contrast": float(lum[sl].std()),
            "edge": float(edge[sl].mean()),
            "saturation": float(sat[sl].mean()),
        })
    return pd.DataFrame(rows)


def standardize(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def fit_within_image(
    df: pd.DataFrame, props: list[str], target: str = "value",
) -> pd.DataFrame:
    """Correlate each property with the map value, INSIDE each image.

    Within-image is the right unit. A between-image fit would mostly recover
    that a dark photograph reads dangerous overall, which the map never
    claimed. The question is whether a dark SQUARE reads dangerous next to its
    own neighbours.
    """
    rows = []
    for image_id, sub in df.groupby("image_id"):
        y = standardize(sub[target].to_numpy())
        for p in props:
            x = standardize(sub[p].to_numpy())
            r = float(np.mean(x * y)) if x.std() and y.std() else np.nan
            rows.append({"image_id": image_id, "prop": p, "r": r})
    return pd.DataFrame(rows)


def multi_fit(df: pd.DataFrame, props: list[str]) -> pd.DataFrame:
    """A least-squares fit of the map on ALL properties, inside each image.

    A single correlation cannot separate 2 properties that move together, and
    dark squares are often also low-edge squares. The joint fit gives each
    property the part that only it explains.
    """
    rows = []
    for image_id, sub in df.groupby("image_id"):
        y = standardize(sub["value"].to_numpy())
        X = np.column_stack([standardize(sub[p].to_numpy()) for p in props]
                            + [np.ones(len(sub))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        r2 = 1.0 - float(((y - pred) ** 2).sum() / max((y ** 2).sum(), 1e-9))
        row = {"image_id": image_id, "r2": r2}
        row.update({f"beta_{p}": float(b) for p, b in zip(props, beta[:-1])})
        rows.append(row)
    return pd.DataFrame(rows)


def build(
    maps_path: Path, stats_path: Path, out_dir: Path,
) -> pd.DataFrame:
    maps = pd.read_parquet(maps_path)
    st = pd.read_parquet(stats_path).set_index("image_id")
    rows = []
    for i, image_id in enumerate(maps["image_id"]):
        v = np.array(maps["values"].iloc[i], dtype=np.float64)
        path = st.loc[str(image_id), "path"]
        cs = cell_stats(Image.open(path), v.size)
        cs["image_id"] = str(image_id)
        cs["value"] = v
        cs["worth_ok"] = bool(st.loc[str(image_id), "worth_ok"]) \
            if "worth_ok" in st.columns else True
        rows.append(cs)
    df = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "cells.parquet", index=False)
    return df


def report(df: pd.DataFrame, out_dir: Path) -> None:
    props = ["luminance", "contrast", "edge", "saturation", "row", "col"]
    strong = df[df.worth_ok] if "worth_ok" in df.columns else df
    per = fit_within_image(strong, props)
    joint = multi_fit(strong, props)
    per.to_parquet(out_dir / "per_property.parquet", index=False)
    joint.to_parquet(out_dir / "joint_fit.parquet", index=False)

    n = strong.image_id.nunique()
    log(f"{n} images, {len(strong)} cells. A positive r means the property "
        "goes with the POSITIVE label.")
    log(f"{'property':>12} {'median r':>10} {'images with r>0':>17} "
        f"{'median beta':>12}")
    for p in props:
        sub = per[per.prop == p]
        pos = int((sub.r > 0).sum())
        b = joint[f"beta_{p}"].median()
        log(f"{p:>12} {sub.r.median():>10.3f} {f'{pos}/{len(sub)}':>17} "
            f"{b:>12.3f}")
    log(f"the joint fit explains a median {joint.r2.median():.3f} of the map")
    (out_dir / "summary.json").write_text(json.dumps({
        "n_images": int(n),
        "median_r": {p: float(per[per.prop == p].r.median()) for p in props},
        "median_beta": {p: float(joint[f"beta_{p}"].median()) for p in props},
        "median_r2": float(joint.r2.median()),
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", required=True, help="a maps.parquet")
    ap.add_argument("--stats", required=True, help="its map_stats.parquet")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    df = build(Path(args.maps), Path(args.stats), out)
    report(df, out)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
