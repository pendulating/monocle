"""Predict the k=1 gain of a case from its truth map alone.

The locality claim says the reader works where the cause sits in few
patches. The top-1 share measures that, and it needs no instrument. So a
line fitted on the 3 first cases predicts a new case BEFORE any gate runs.

    python scripts/k1_locality_predict.py --fit-only
    python scripts/k1_locality_predict.py \
        --new-maps outputs/_monocle/causal_maps60_new

Warning: run `--fit-only` and record the line before the new maps land.
A line fitted after you see the answer is not a prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The 3 cases that the line is fitted on, with the pooled gain from
# outputs/_monocle/k1_pool_train.json (arm `gxa`, 9 layers, mean rank).
FIT_MAPS = Path("outputs/_monocle/causal_maps60")
TRAIN_JSON = Path("outputs/_monocle/k1_pool_train.json")
FIT_CASES = ("restaurants", "libraries", "subway_safety")


def top1_share(values: np.ndarray) -> float:
    """The share of the total absolute effect that sits in 1 cell."""
    a = np.abs(values)
    tot = a.sum()
    if tot <= 0:
        return float("nan")
    return float(a.max() / tot)


def case_shares(maps_dir: Path, case: str) -> np.ndarray:
    """One top-1 share for each photograph of a case."""
    hits = sorted(maps_dir.glob(f"{case}_*/maps.parquet"))
    if not hits:
        raise FileNotFoundError(f"no maps for {case} under {maps_dir}")
    df = pd.read_parquet(hits[0])
    out = []
    for _, row in df.iterrows():
        out.append(top1_share(np.asarray(row["values"], dtype=float)))
    return np.array(out, dtype=float)


def train_gains(path: Path, arm: str = "gxa") -> dict[str, float]:
    rows = json.loads(path.read_text())
    return {r["case"]: r["gain"] for r in rows if r["arm"] == arm}


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-maps", default=str(FIT_MAPS))
    ap.add_argument("--train-json", default=str(TRAIN_JSON))
    ap.add_argument("--new-maps", default=None)
    ap.add_argument("--arm", default="gxa")
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument("--out", default="outputs/_monocle/k1_locality_predict.json")
    args = ap.parse_args()

    gains = train_gains(Path(args.train_json), args.arm)
    xs, ys, names = [], [], []
    for c in FIT_CASES:
        s = case_shares(Path(args.fit_maps), c)
        xs.append(float(np.median(s)))
        ys.append(gains[c])
        names.append(c)
    x = np.array(xs)
    y = np.array(ys)
    a, b = fit_line(x, y)
    r = float(np.corrcoef(x, y)[0, 1])

    print(f"fit on {len(x)} cases, arm {args.arm}")
    print(f"  gain = {a:+.3f} {b:+.2f} * top1_share   (r = {r:+.3f})")
    for c, xi, yi in zip(names, x, y):
        print(f"  {c:>20}  top1={xi:.5f}  gain={yi:+.3f}  "
              f"fit={a + b * xi:+.3f}")

    rec = {"arm": args.arm, "intercept": a, "slope": b, "r": r,
           "fit": [{"case": c, "top1": float(xi), "gain": float(yi)}
                   for c, xi, yi in zip(names, x, y)],
           "predict": []}

    if args.new_maps and not args.fit_only:
        root = Path(args.new_maps)
        for d in sorted(root.glob("*_absent")):
            case = d.name[: -len("_absent")]
            s = case_shares(root, case)
            xi = float(np.median(s))
            lo, hi = np.percentile(s, [25, 75])
            pred = a + b * xi
            print(f"  PREDICT {case:>20}  top1={xi:.5f} "
                  f"[{lo:.5f},{hi:.5f}]  gain_hat={pred:+.3f}")
            rec["predict"].append({"case": case, "top1": xi,
                                   "q25": float(lo), "q75": float(hi),
                                   "gain_hat": float(pred),
                                   "n_images": int(len(s))})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
