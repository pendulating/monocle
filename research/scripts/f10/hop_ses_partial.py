#!/usr/bin/env python
"""F10.29, item 1: is the Black-share gradient of the answers an income gradient? CPU only.

For `ugly`, `nice` and `cool` (3 models, 2 nouns, 18 cells) the script gives the Spearman rho over the 197 NTAs
of the expected answer E = P(yes) - P(no) (F10.28) with the Black share, and the partial Spearman rho of E with
the Black share given the median household income, and of E with income given the Black share. The partial rho
is the Pearson r of the rank residuals. Bonferroni over 18 tests in each family.

Output: `research/figures/f10/hierarchy/ses_partial.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f10/hop_ses_partial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "research/scripts/f10")]
from hop_prompts import MODELS, NOUNS, answer_store  # noqa: E402

__version__ = "1.0.0"
ADJS = ("ugly", "nice", "cool")


def expected_answer(model: str, adj: str, noun: str, nta: pd.Series) -> pd.Series:
    """Give the NTA mean of E = P(yes) - P(no) over the 8 answer forms."""
    a = pd.read_parquet(ROOT / "data/f10/hop_logit" / answer_store(model, adj, noun) / "answers.parquet").set_index("image_idx")
    yes = [c for c in a.columns if c.startswith("lp_") and c[3:].strip().lower() == "yes"]
    no = [c for c in a.columns if c.startswith("lp_") and c[3:].strip().lower() == "no"]
    e = np.exp(a[yes]).sum(1) - np.exp(a[no]).sum(1)
    return e.groupby(nta.reindex(e.index)).mean()


def partial_spearman(x: pd.Series, y: pd.Series, z: pd.Series) -> tuple[float, float, int]:
    d = pd.concat([x, y, z], axis=1).dropna().rank()
    rx = d.iloc[:, 0] - np.polyval(np.polyfit(d.iloc[:, 2], d.iloc[:, 0], 1), d.iloc[:, 2])
    ry = d.iloc[:, 1] - np.polyval(np.polyfit(d.iloc[:, 2], d.iloc[:, 1], 1), d.iloc[:, 2])
    r = stats.pearsonr(rx, ry)
    return float(r[0]), float(r[1]), len(d)


def main() -> None:
    acs = pd.read_parquet(ROOT / "data/f4_traces/nta_acs.parquet").set_index("nta2020")
    nta = pd.read_parquet(ROOT / "data/f10/hop_all/images.parquet").set_index("image_idx")["nta2020"]
    rows = []
    for adj in ADJS:
        for m in MODELS:
            for n in NOUNS:
                e = expected_answer(m, adj, n, nta)
                a = acs.reindex(e.index)
                rb = stats.spearmanr(e, a.pct_black_nh, nan_policy="omit")
                r, p, k = partial_spearman(e, a.pct_black_nh, a.median_household_income)
                ri, pi, _ = partial_spearman(e, a.median_household_income, a.pct_black_nh)
                rows.append({"adjective": adj, "model": m, "noun": n, "n_nta": k, "rho_black": float(rb[0]), "p_black": float(rb[1]),
                             "rho_black_given_income": r, "p_black_given_income": p, "rho_income_given_black": ri, "p_income_given_black": pi})
    d = pd.DataFrame(rows)
    for c in ("p_black", "p_black_given_income", "p_income_given_black"):
        d[c + "_bonferroni"] = (d[c] * len(d)).clip(upper=1.0)
    out = ROOT / "research/figures/f10/hierarchy/ses_partial.csv"
    d.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(d.round(3).to_string(index=False))
    print(f"\nBlack share alone passes {int((d.p_black_bonferroni < .05).sum())} of {len(d)}; given income {int((d.p_black_given_income_bonferroni < .05).sum())}; income given the Black share {int((d.p_income_given_black_bonferroni < .05).sum())}")


if __name__ == "__main__":
    main()
