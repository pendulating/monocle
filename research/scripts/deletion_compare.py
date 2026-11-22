"""Compare deletion arms, paired on the image.

The statistic
-------------
`separation_z` is per image: the gap between the 2 directed deletions, divided
by the spread of the random draws on that same image. It already removes the
"a hole moves `c`" effect, so it is comparable across arms.

Why paired
----------
The arms read the SAME images. An unpaired interval carries the spread between
photographs, which is large and which cancels. A paired test asks the question
we want: on this photograph, does arm A rank better than arm B?

The ceiling
-----------
The measured input map is the best ranking that a per-cell method can give,
because it IS the measurement. Every model arm reports its share of that
ceiling, so a reader sees how much of the truth the cheap instrument keeps.

Usage:
    python research/scripts/deletion_compare.py \
        --runs outputs/_monocle/deletion_curve/subway_safety_g2_self_ablate \
               outputs/_monocle/deletion_curve/subway_safety_g2_donor \
        --ceiling subway_safety_g2_self_ablate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STAT = "separation_z"


def load(run: Path) -> pd.DataFrame:
    df = pd.read_parquet(run / "summary.parquet")
    df = df[~df["keep_only"]] if "keep_only" in df.columns else df
    df["arm"] = run.name
    return df[["arm", "image_id", "k", "separation", STAT]]


def boot_median(values: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    if values.size < 3:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    meds = np.median(values[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def table(frames: list[pd.DataFrame], ceiling: str | None,
          n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.concat(frames, ignore_index=True)
    rows = []
    ceil = {}
    if ceiling:
        sub = df[df["arm"] == ceiling]
        ceil = sub.groupby("k")[STAT].median().to_dict()
    for (arm, k), g in df.groupby(["arm", "k"]):
        v = g[STAT].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        lo, hi = boot_median(v, n_boot, rng)
        med = float(np.median(v)) if v.size else float("nan")
        rows.append({"arm": arm, "k": int(k), "n": int(v.size),
                     "median_z": med, "lo": lo, "hi": hi,
                     "share_of_ceiling": (med / ceil[k]
                                          if ceil.get(k) else float("nan"))})
    return pd.DataFrame(rows).sort_values(["k", "median_z"],
                                          ascending=[True, False])


def paired(frames: list[pd.DataFrame], base: str, other: str,
           n_boot: int, seed: int) -> pd.DataFrame:
    """The per-image gain of `other` over `base`, for each k."""
    rng = np.random.default_rng(seed)
    df = pd.concat(frames, ignore_index=True)
    a = df[df["arm"] == base].set_index(["image_id", "k"])[STAT]
    b = df[df["arm"] == other].set_index(["image_id", "k"])[STAT]
    join = pd.concat({"base": a, "other": b}, axis=1).dropna()
    rows = []
    for k, g in join.groupby(level="k"):
        d = (g["other"] - g["base"]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if d.size < 3:
            continue
        idx = rng.integers(0, d.size, size=(n_boot, d.size))
        meds = np.median(d[idx], axis=1)
        rows.append({"k": int(k), "n": int(d.size),
                     "median_gain": float(np.median(d)),
                     "lo": float(np.percentile(meds, 2.5)),
                     "hi": float(np.percentile(meds, 97.5)),
                     "wins": int((d > 0).sum())})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ceiling", default=None,
                    help="the arm name that sets the ceiling")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "OTHER"),
                    default=None)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    frames = []
    for r in args.runs:
        p = Path(r)
        if not (p / "summary.parquet").exists():
            print(f"skip {p}: no summary.parquet")
            continue
        frames.append(load(p))
    if not frames:
        print("no run holds a summary")
        return 2

    tab = table(frames, args.ceiling, args.n_boot, args.seed)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    if args.compare:
        print(f"\npaired: {args.compare[1]} minus {args.compare[0]}")
        print(paired(frames, args.compare[0], args.compare[1],
                     args.n_boot, args.seed).to_string(
                         index=False, float_format=lambda v: f"{v:+.3f}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tab.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
