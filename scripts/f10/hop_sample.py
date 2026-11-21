#!/usr/bin/env python
"""F10.29, item 3: is a sample of 3 images for each NTA enough? CPU only, reads the F10.26 frames.

For each of the 42 cells (3 models, 7 adjectives, 2 nouns) the script gives:
- ICC(1): the share of the margin variance that lies between NTAs (one-way ANOVA on images).
- ICC(1,3): the reliability of the mean of 3 images, by the Spearman-Brown rule.
- Split r: r over NTAs between the margin of 1 image and the mean margin of the other 2 (mean over the 3 choices).
- The same tests as F10.26 and F10.28 with 1 image for each NTA, for each of the 3 choices:
  Spearman rho of the NTA value with median household income, and the Welch t of Manhattan against the rest.

Output: `figures/f10/sample/cells.csv`, `figures/f10/sample/summary.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/hop_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/f10")]
from hop_prompts import ADJECTIVES as ADJS, MODELS, NOUNS, store_name  # noqa: E402

__version__ = "1.0.0"
OUT = ROOT / "figures/f10/sample"
ALPHA = 0.05 / 42


def icc1(x: pd.Series, g: pd.Series) -> float:
    """ICC(1) of a one-way design with k = 3 images in each group. Groups with 1 image count in the within term only."""
    d = pd.DataFrame({"x": x, "g": g}).dropna()
    n = d.groupby("g")["x"].size()
    k = n.mean()
    grand = d.x.mean()
    means = d.groupby("g")["x"].mean()
    msb = (n * (means - grand) ** 2).sum() / (len(n) - 1)
    msw = ((d.x - means.reindex(d.g).to_numpy()) ** 2).sum() / (len(d) - len(n))
    return float((msb - msw) / (msb + (k - 1) * msw))


def cell(model: str, adj: str, noun: str, acs: pd.Series) -> dict:
    f = pd.read_parquet(ROOT / "data/f10/prompts" / store_name(model, adj, noun) / "frame.parquet")
    f = f.dropna(subset=["margin"])
    f["rank"] = f.groupby("nta2020").cumcount()
    full = f.groupby("nta2020")["margin"].mean()
    row = {"model": model, "noun": noun, "adjective": adj, "n_images": len(f), "n_nta": f.nta2020.nunique(), "icc1": icc1(f.margin, f.nta2020)}
    row["icc1_3"] = 3 * row["icc1"] / (1 + 2 * row["icc1"])  # the reliability of the mean of 3 images (Spearman-Brown)
    rs, rho1, t1 = [], [], []
    for r in range(3):
        one = f[f["rank"] == r].set_index("nta2020")["margin"]
        rest = f[f["rank"] != r].groupby("nta2020")["margin"].mean()
        both = pd.concat([one, rest], axis=1, keys=["one", "rest"]).dropna()
        if len(both) > 10:
            rs.append(stats.pearsonr(both.one, both.rest)[0])
        a = acs.reindex(one.index)
        ok = a.notna()
        rho1.append(stats.spearmanr(one[ok], a[ok]))
        b = f[f["rank"] == r]
        t1.append(stats.ttest_ind(b.margin[b.borough == "Manhattan"], b.margin[b.borough != "Manhattan"], equal_var=False))
    a = acs.reindex(full.index)
    ok = a.notna()
    rho3 = stats.spearmanr(full[ok], a[ok])
    bor = f.groupby("nta2020")["borough"].first().reindex(full.index)
    t3 = stats.ttest_ind(full[bor == "Manhattan"], full[bor != "Manhattan"], equal_var=False)
    row |= {"split_r": float(np.mean(rs)) if rs else np.nan, "rho_income_3": rho3[0], "p_income_3": rho3[1],
            "rho_income_1_min": min(x[0] for x in rho1), "rho_income_1_max": max(x[0] for x in rho1), "pass_income_1": sum(x[1] < ALPHA for x in rho1),
            "t_manhattan_3": t3[0], "p_manhattan_3": t3[1], "t_manhattan_1_min": min(x[0] for x in t1), "t_manhattan_1_max": max(x[0] for x in t1), "pass_manhattan_1": sum(x[1] < ALPHA for x in t1)}
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acs = pd.read_parquet(ROOT / "data/f4_traces/nta_acs.parquet").set_index("nta2020")["median_household_income"]
    rows = [cell(m, a, n, acs) for m in MODELS for n in NOUNS for a in ADJS]
    c = pd.DataFrame(rows)
    c.to_csv(OUT / "cells.csv", index=False)
    full = c[c.n_images >= 500]
    s = pd.DataFrame([{
        "cells_with_500_images_or_more": len(full),
        "icc1_min": full.icc1.min(), "icc1_median": full.icc1.median(), "icc1_max": full.icc1.max(),
        "icc1_3_min": full.icc1_3.min(), "icc1_3_median": full.icc1_3.median(), "icc1_3_max": full.icc1_3.max(),
        "split_r_min": full.split_r.min(), "split_r_median": full.split_r.median(), "split_r_max": full.split_r.max(),
        "income_pass_3_images": int((full.p_income_3 < ALPHA).sum()), "income_pass_1_image_all_3_choices": int((full.pass_income_1 == 3).sum()), "income_pass_1_image_any_choice": int((full.pass_income_1 >= 1).sum()),
        "manhattan_pass_3_images": int((full.p_manhattan_3 < ALPHA).sum()), "manhattan_pass_1_image_all_3_choices": int((full.pass_manhattan_1 == 3).sum()), "manhattan_pass_1_image_any_choice": int((full.pass_manhattan_1 >= 1).sum()),
    }])
    s.T.to_csv(OUT / "summary.csv", header=False)
    print(c.round(3).to_string(index=False))
    print(s.T.round(3).to_string(header=False))


if __name__ == "__main__":
    main()
