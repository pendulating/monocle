"""The block-45 pole lists against the per-unit ratings of the canonical run.

Per-unit: TrueSkill mu over the canonical pairwise results (the report's
recipe: 1v1 updates in row order, draws allowed), plus the unit's mean
relative score. Per-image: the mean sign-adjusted relative score of the
image's own pairs. The 60 eval images of `pole_sample` join by image path.

    .venv-3.12/bin/python scripts/pole_sample_per_unit.py --case restaurants --layer 45
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path("/share/ju/matt/mllmsci")
CANON = REPO / "notebooks/cvpr/canonical_data/proxy/canonical"
POLE = REPO / "outputs/_monocle/pole_sample"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="restaurants")
    ap.add_argument("--model", default="gemma-4-12b")
    ap.add_argument("--layers", default="42,45")
    ap.add_argument("--draw-prob", type=float, default=0.1)
    args = ap.parse_args(argv)
    run = CANON / f"{args.case}__{args.model}"
    cache = POLE / args.case / "ratings"
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "units.parquet").exists():
        units = pd.read_parquet(cache / "units.parquet")
        img = pd.read_parquet(cache / "images.parquet")
        print(f"[per-unit] {args.case}: cached ratings, {len(units):,} units, {len(img):,} images")
        return join(args, units, img)
    res = pd.read_parquet(run / "results.parquet",
                          columns=["pair_id", "repeat_idx", "sample_id_a", "sample_id_b",
                                   "image_path_a", "image_path_b", "relative_score", "relative_label"])
    pairs = pd.read_parquet(run / "pairs.parquet",
                            columns=["pair_id", "repeat_idx", "unit_uid_a", "unit_uid_b"])
    df = res.merge(pairs, on=["pair_id", "repeat_idx"], how="left")
    df["relative_score"] = pd.to_numeric(df["relative_score"], errors="coerce")
    n_abst = int((df["relative_label"] == "NotSure").sum())
    work = df.dropna(subset=["relative_score", "unit_uid_a", "unit_uid_b"])
    work = work[work["unit_uid_a"] != work["unit_uid_b"]]
    print(f"[per-unit] {len(df):,} rows, {n_abst:,} NotSure, {len(work):,} rated rows "
          f"({work['unit_uid_a'].nunique():,} units)", flush=True)

    # TrueSkill per unit, the report's recipe.
    import trueskill
    env = trueskill.TrueSkill(draw_probability=args.draw_prob)
    ratings = defaultdict(env.create_rating)
    for a, b, s in zip(work["unit_uid_a"].values, work["unit_uid_b"].values,
                       work["relative_score"].values):
        ra, rb = ratings[a], ratings[b]
        if s > 0:
            ra, rb = env.rate_1vs1(ra, rb)
        elif s < 0:
            rb, ra = env.rate_1vs1(rb, ra)
        else:
            ra, rb = env.rate_1vs1(ra, rb, drawn=True)
        ratings[a], ratings[b] = ra, rb
    units = pd.DataFrame([{"unit_uid": u, "mu": r.mu, "sigma": r.sigma}
                          for u, r in ratings.items()])
    # Mean relative score per unit and per image (sign-adjusted for side B).
    long = pd.concat([
        pd.DataFrame({"unit_uid": work["unit_uid_a"], "sample_id": work["sample_id_a"],
                      "image_path": work["image_path_a"], "score": work["relative_score"]}),
        pd.DataFrame({"unit_uid": work["unit_uid_b"], "sample_id": work["sample_id_b"],
                      "image_path": work["image_path_b"], "score": -work["relative_score"]}),
    ])
    unit_mean = long.groupby("unit_uid")["score"].agg(["mean", "count"]).rename(
        columns={"mean": "unit_mean_score", "count": "unit_n"})
    img = long.groupby("image_path").agg(sample_id=("sample_id", "first"),
                                          unit_uid=("unit_uid", "first"),
                                          image_mean_score=("score", "mean"),
                                          image_n=("score", "size"))
    units = units.set_index("unit_uid").join(unit_mean)
    units["mu_pct"] = units["mu"].rank(pct=True)
    print(f"[per-unit] mu: mean {units.mu.mean():.2f} sd {units.mu.std():.2f}, "
          f"sigma median {units.sigma.median():.2f}", flush=True)
    units.to_parquet(cache / "units.parquet")
    img.to_parquet(cache / "images.parquet")
    return join(args, units, img)


def join(args, units: pd.DataFrame, img: pd.DataFrame) -> int:
    for layer in [int(x) for x in args.layers.split(",")]:
        join_layer(args, units, img, layer)
    return 0


def join_layer(args, units: pd.DataFrame, img: pd.DataFrame, layer: int) -> None:
    print(f"\n=== {args.case} block {layer} ===")
    pole = pd.read_parquet(POLE / args.case / "projections.parquet")
    pole = pole[pole.layer == layer].copy()
    words = [c[5:] for c in pole.columns if c.startswith("proj_") and c != "proj_contrast"]
    j = pole.merge(img, left_on="image_path", right_index=True, how="left")
    j = j.merge(units, left_on="unit_uid", right_index=True, how="left")
    out = POLE / args.case / f"per_unit_join_L{layer}.parquet"
    j.to_parquet(out, index=False)
    print(f"[per-unit] eval images joined: {j.mu.notna().sum()} of {len(j)}; "
          f"pairs per image median {j.image_n.median():.0f}, per unit median {j.unit_n.median():.0f}")

    rows = []
    for x in [f"proj_{w}" for w in words] + ["proj_contrast", "c"]:
        for y in ["mu", "unit_mean_score", "image_mean_score"]:
            m = j[[x, y]].dropna()
            rows.append({"x": x, "y": y, "n": len(m),
                         "spearman": sps.spearmanr(m[x], m[y])[0],
                         "pearson": np.corrcoef(m[x], m[y])[0, 1]})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # The lists: percentile of the unit's mu for the top/bottom 5 of each pole.
    for w in words:
        col = f"proj_{w}"
        top = j.sort_values(col, ascending=False).head(5)
        bot = j.sort_values(col, ascending=True).head(5)
        print(f"\nclosest to '{w}':  mu pct {np.round(top.mu_pct.values, 2).tolist()}  "
              f"image score {np.round(top.image_mean_score.values, 2).tolist()}")
        print(f"furthest from '{w}': mu pct {np.round(bot.mu_pct.values, 2).tolist()}  "
              f"image score {np.round(bot.image_mean_score.values, 2).tolist()}")


if __name__ == "__main__":
    raise SystemExit(main())
