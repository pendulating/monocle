"""F10.23: the outcome of each answer: yes, no or unable.

The outcome comes from the first token of the answer. A form of yes or no is a verdict. Any
other first token is `unable`: the model does not give a verdict (for example `Subjective`,
`Unable to determine.`). A refusal is a third outcome, not a low margin. The margin of a refused
image compares 2 answers that the model did not give, thus the `margin` column is empty there.

| File in `research/figures/f10/outcomes/` | Content |
|---------------------------------|---------|
| `outcomes.parquet` | 1 row for each model, adjective and image: `top`, `outcome`, `margin` |
| `outcome_rates.csv` | the share of each outcome, by model and adjective, and by borough |
| `outcome_tests.csv` | chi-square of borough against outcome, Holm over the rows |

Use:
    python research/scripts/f10/hop_outcomes.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research/figures/f10/outcomes"
MODELS = ("gemma-4-31b", "qwen3.5-9b", "qwen3.6-27b")
ADJECTIVES = ("cool", "nice", "ordinary")
OUTCOMES = ("yes", "no", "unable")


def outcome(top: str) -> str:
    """Give the outcome of a first token. Case and spaces do not change the outcome."""
    word = top.strip().lower()
    return word if word in ("yes", "no") else "unable"


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adjusted = (p[order] * (len(p) - pd.Series(range(len(p)), index=order))).clip(upper=1).cummax()
    return adjusted[p.index]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = pd.read_parquet(ROOT / "data/f10/hop_all/images.parquet")[["image_idx", "borough"]]
    frames = []
    for model in MODELS:
        for adj in ADJECTIVES:
            store = model if adj == "cool" else f"{model}-{adj}"
            a = pd.concat(map(pd.read_parquet, sorted(glob.glob(str(ROOT / f"data/f10/hop_all/{store}/answers/*.parquet")))))
            frames.append(a[["image_idx", "top"]].assign(model=model, adjective=adj))
    t = pd.concat(frames, ignore_index=True).merge(images, on="image_idx", validate="m:1")
    t["outcome"] = t["top"].map(outcome)
    margins = []
    for model in MODELS:
        for adj in ADJECTIVES:
            if adj == "cool":
                m = pd.read_parquet(ROOT / f"data/f10/hop_logit/{model}/answers.parquet")[["image_idx", "margin"]]
            else:
                m = pd.concat(map(pd.read_parquet, sorted(glob.glob(str(ROOT / f"data/f10/hop_all/{model}-{adj}/answers/*.parquet")))))[["image_idx", "margin"]]
            margins.append(m.assign(model=model, adjective=adj))
    t = t.merge(pd.concat(margins, ignore_index=True), on=["image_idx", "model", "adjective"], validate="1:1")
    t.loc[t["outcome"] == "unable", "margin"] = float("nan")
    t.to_parquet(OUT / "outcomes.parquet", index=False)

    keys = ["model", "adjective"]
    rates = pd.concat([t.groupby(keys)["outcome"].value_counts(normalize=True).unstack(fill_value=0).assign(borough="all").reset_index(),
                       t.groupby(keys + ["borough"])["outcome"].value_counts(normalize=True).unstack(fill_value=0).reset_index()], ignore_index=True)
    rates.reindex(columns=keys + ["borough", *OUTCOMES], fill_value=0.0).to_csv(OUT / "outcome_rates.csv", index=False)

    rows = []
    for (model, adj), g in t.groupby(keys):
        table = pd.crosstab(g["borough"], g["outcome"])
        table = table.loc[:, table.sum() > 0]
        chi, p, dof, _ = stats.chi2_contingency(table)
        rows.append({"model": model, "adjective": adj, "chi2": chi, "dof": dof, "p": p, "unable": int((g["outcome"] == "unable").sum())})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm(tests["p"])
    tests.to_csv(OUT / "outcome_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
