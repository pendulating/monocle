"""Turn one arm of a gate `maps.parquet` into a ranking for deletion_curve.

Why
---
`monocle/deletion_curve.py --rank-from` needs a parquet of `image_id` and
`values`. This writes one from any arm of a gate run, so a LENS can supply the
ranking and an INPUT deletion can test it.

That pairing is what makes the test non-circular. The lens reads a residual and
never touches the input; the deletion edits the input and never reads a lens.
Agreement between them cannot come from shared machinery.

Usage:
    python scripts/make_ranking.py \
        --maps outputs/_monocle/input_gate/subway_safety/maps.parquet \
        --arm lens_own --layer 20 --group 2 \
        --out outputs/_monocle/rankings/lens_own_L20_g2.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pool(values: np.ndarray, group: int) -> np.ndarray:
    """Sum a 256-cell map into `group` x `group` blocks, row-major.

    The map is a contribution in logit units, so blocks SUM. A mean would
    change the scale between grids and make 2 runs incomparable.
    """
    if group == 1:
        return values
    side = int(round(np.sqrt(values.size)))
    if side * side != values.size:
        raise SystemExit(f"{values.size} cells is not a square grid")
    if side % group:
        raise SystemExit(f"a {side}x{side} grid does not divide by {group}")
    grid = values.reshape(side, side)
    n = side // group
    return grid.reshape(n, group, n, group).sum(axis=(1, 3)).ravel()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--group", type=int, default=2)
    ap.add_argument("--shuffle", type=int, default=0,
                    metavar="SEED",
                    help="give each image ANOTHER image's map. "
                         "This is the null that asks whether the "
                         "map tracks THIS photograph. A seed above "
                         "0 turns it on.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_parquet(a.maps)
    # `causal_maps` writes image_id/values; the gates write target/<arm>. The
    # measured input map is the deletion ceiling, so both layouts are read.
    if "image_id" in df.columns and "values" in df.columns \
            and a.arm not in df.columns:
        df = df.rename(columns={"image_id": "target", "values": a.arm})
        df["layer"] = a.layer
    if a.arm not in df.columns:
        raise SystemExit(f"{a.maps} holds no {a.arm!r}; has {list(df.columns)}")
    sub = df[df.layer == a.layer]
    if sub.empty:
        raise SystemExit(f"no rows at layer {a.layer}; "
                         f"have {sorted(df.layer.unique())}")

    rows = []
    for _, r in sub.iterrows():
        v = r[a.arm]
        if v is None or np.ndim(v) == 0:
            print(f"  skip {r.target}: {a.arm} is empty at L{a.layer}")
            continue
        rows.append({"image_id": str(r.target),
                     "values": pool(np.asarray(v, float).ravel(),
                                    a.group).tolist()})
    if not rows:
        raise SystemExit(f"{a.arm} is empty at layer {a.layer}")
    if a.shuffle > 0:
        # A derangement: no image keeps its own map.
        rng = np.random.default_rng(a.shuffle)
        n = len(rows)
        if n < 2:
            raise SystemExit("a shuffle needs 2 images or more")
        order = np.arange(n)
        while True:
            rng.shuffle(order)
            if not np.any(order == np.arange(n)):
                break
        vals = [rows[i]["values"] for i in order]
        for row, v in zip(rows, vals):
            row["values"] = v
        print(f"  shuffled: no image keeps its own map (seed {a.shuffle})")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out}: {len(rows)} images, {len(rows[0]['values'])} cells, "
          f"arm={a.arm} layer={a.layer} group={a.group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
