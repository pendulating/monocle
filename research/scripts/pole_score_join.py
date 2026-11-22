"""The pole projections against the unit proxies, on the pairwise ratings' own footing.

Per case: concatenate the score shards, center each raw projection over the
scored images, aggregate per proxy key (mean over the unit's drawn images),
join to the CVPR unit frame (`_unit_validation.joined`: pairwise mean_score
+ proxy_value on the same units), and report Spearman rho and Kendall tau
with a 400-resample bootstrap over units for every read:

| read | what |
|---|---|
| pairwise | the CVPR pairwise mean_score of the unit (the published number) |
| c | the single-image logit gap, mean over the unit's images |
| contrast_L<l> | the centered projection on the contrast vector at block l |
| pos_L<l>, neg_L<l> | the same on each pole's own vector |

    .venv-3.12/bin/python research/scripts/pole_score_join.py --cases all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "outputs/_monocle/pole_score"
CASES = ["restaurants", "subway_safety", "libraries", "schools", "parks_plazas", "road_quality"]
BOOT_N, BOOT_SEED = 400, 20260831


def rho_tau(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan
    rho = sps.spearmanr(x, y)[0]
    tau, p = sps.kendalltau(x, y)
    return rho, tau, p


def boot(x, y, rng):
    n = len(x)
    rs = []
    for _ in range(BOOT_N):
        i = rng.integers(0, n, n)
        rs.append(sps.spearmanr(x[i], y[i])[0])
    return np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="all")
    args = ap.parse_args(argv)
    cases = CASES if args.cases == "all" else args.cases.split(",")
    rows, unit_frames = [], []
    for case in cases:
        shards = sorted((ROOT / case).glob("scores_shard*.parquet"))
        if not shards:
            print(f"[join] {case}: no shards yet")
            continue
        sc = pd.concat([pd.read_parquet(s) for s in shards])
        sample = pd.read_parquet(ROOT / case / "sample.parquet")
        n_scored = sc["image_path"].nunique()
        for k in ("pos", "neg", "contrast"):
            sc[k] = sc[f"raw_{k}"] - sc.groupby("layer")[f"raw_{k}"].transform("mean")
        wide = sc.pivot(index="image_path", columns="layer", values=["pos", "neg", "contrast"])
        wide.columns = [f"{a}_L{b}" for a, b in wide.columns]
        wide["c"] = sc.groupby("image_path")["c"].first()
        img = sample.merge(wide, left_on="image_path", right_index=True, how="inner")
        reads = ["c"] + [col for col in wide.columns if col != "c"]
        unit = img.groupby(["folder", "proxy_key"]).agg(
            {**{r: "mean" for r in reads}, "proxy_value": "first",
             "mean_score": "first", "n_comparisons": "first",
             "image_path": "size"}).rename(columns={"image_path": "n_images"}).reset_index()
        unit["case"] = case
        unit_frames.append(unit)
        for folder, u in unit.groupby("folder"):
            rng = np.random.default_rng(BOOT_SEED)
            y = u["proxy_value"].to_numpy(dtype=float)
            for read in ["mean_score"] + reads:
                x = u[read].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                rho, tau, p = rho_tau(x[ok], y[ok])
                lo, hi = boot(x[ok], y[ok], rng) if ok.sum() >= 3 else (np.nan, np.nan)
                rows.append({"case": case, "folder": folder, "read": "pairwise" if read == "mean_score" else read,
                             "n_units": int(ok.sum()), "rho": rho, "rho_lo": lo, "rho_hi": hi,
                             "tau": tau, "tau_p": p})
            # and how the reads relate to the pairwise rating itself
            for read in reads:
                x = u[read].to_numpy(dtype=float); z = u["mean_score"].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(z)
                rows.append({"case": case, "folder": folder, "read": f"{read} vs pairwise",
                             "n_units": int(ok.sum()), "rho": sps.spearmanr(x[ok], z[ok])[0],
                             "rho_lo": np.nan, "rho_hi": np.nan, "tau": np.nan, "tau_p": np.nan})
        print(f"[join] {case}: {n_scored:,} images scored of {len(sample):,}, "
              f"{unit['proxy_key'].nunique():,} units", flush=True)
    if not rows:
        return 1
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / "proxy_correlations.parquet", index=False)
    pd.concat(unit_frames).to_parquet(ROOT / "unit_scores.parquet", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 500)
    main_reads = ["pairwise", "c", "contrast_L42", "contrast_L45", "pos_L45", "neg_L45"]
    t = df[df["read"].isin(main_reads)].copy()
    t["ci"] = t.apply(lambda r: f"[{r.rho_lo:+.2f}, {r.rho_hi:+.2f}]", axis=1)
    print("\nSPEARMAN RHO vs the unit proxy (bootstrap CI over units)")
    print(t[["folder", "read", "n_units", "rho", "ci", "tau", "tau_p"]].round(3).to_string(index=False))
    print("\nEACH READ vs the pairwise mean_score of the same units (Spearman)")
    v = df[df["read"].str.endswith("vs pairwise")]
    print(v.pivot_table(index="folder", columns="read", values="rho").round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
