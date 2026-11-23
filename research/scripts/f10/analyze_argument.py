"""F10.16: How 1 visual object enters the argument for or against "cool".

The `criteria` arm gives each criterion span a value, a stance and the observation that
supports it. This script asks if the object or the value gives the sign of the argument.

| Output | Question | Method |
|--------|----------|--------|
| `argument_objects.csv` | Which side does each object support | Share of for, against and conditional spans |
| `argument_cells.csv` | What is the for share of each object under each value | Counts of for and against spans |
| `argument_flips.csv` | Does the sign of 1 object change with the value | Chi-square of value by stance inside the object, Holm |
| `argument_sign.csv` | Which label predicts the sign | Chi-square with Cramer V; out-of-fold AUC of a logistic fit, folds by image |
| `argument_repeat.csv` | Do the repeats of 1 image agree | One-way ICC over 3 repeats of the presence of a value or an object; `recur`, the share of the other repeats that also hold it when 1 repeat does; `share`, its base rate |
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from research.scripts.f10.analyze_cool import OUT, load_spans  # noqa: E402
from research.scripts.f10.analyze_logit import MODELS, SEED, holm  # noqa: E402

# The object list comes from the most frequent observations. A pattern is fixed before the tests.
OBJECTS = {
    "brick": r"\bbrick", "brownstone or row house": r"brownstone|row ?hou|townhouse", "fire escape": r"fire escape", "tree or greenery": r"\btrees?\b|greenery|leafy|foliage",
    "lawn or yard": r"\blawns?\b|\byards?\b|landscap|hedge|shrub|bushes", "graffiti": r"graffiti|tag(s|ging)?\b", "mural": r"\bmural", "scaffolding": r"scaffold|sidewalk shed",
    "power lines": r"power line|overhead wire|\bwires?\b|utility pole", "chain-link fence": r"chain.?link|\bfence", "parking lot": r"parking lot|\basphalt lot",
    "construction": r"construction", "trash": r"trash|garbage|litter|dumpster", "ac units": r"\bac unit|air condition", "concrete": r"concrete", "siding": r"\bsiding",
    "glass": r"\bglass", "skyscraper": r"skyscraper|high.?rise|skyline|tall building", "storefront": r"storefront|\bshops?\b|bodega|grocery|\bcafe|restaurant",
    "people": r"\bpeople\b|pedestrian|\bcrowd", "cars": r"\bcars?\b|\btraffic\b|\bsuv\b|\btruck", "bike lane": r"bike lane|bicycle", "basketball or playground": r"basketball|playground",
    "sunny": r"\bsunny|sunlight|sunshine|golden hour", "overcast": r"overcast|gr[ae]y sky|cloudy|gloomy", "blue sky": r"blue sky", "shade": r"\bshade|dappled",
}
PATTERN = {k: re.compile(v) for k, v in OBJECTS.items()}
MIN_CELL = 10


def tag(text: str) -> list[str]:
    return [k for k, p in PATTERN.items() if p.search(text)]


def cramer(tab: np.ndarray) -> tuple[float, float, float]:
    tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    chi, p, _, _ = stats.chi2_contingency(tab)
    return chi, p, float(np.sqrt(chi / (tab.sum() * (min(tab.shape) - 1))))


def icc(table: np.ndarray) -> float:
    """Give the one-way ICC of an image-by-repeat table."""
    k = table.shape[1]
    grand, row = table.mean(), table.mean(1)
    between = k * ((row - grand) ** 2).sum() / (len(table) - 1)
    within = ((table - row[:, None]) ** 2).sum() / (len(table) * (k - 1))
    return float((between - within) / (between + (k - 1) * within)) if between + within > 0 else np.nan


def recurrence(table: np.ndarray) -> float:
    """Give the share of the other repeats that hold an item when 1 repeat of the image holds it."""
    k = table.shape[1]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            if i != j:
                den += table[:, i].sum()
                num += (table[:, i] * table[:, j]).sum()
    return float(num / den) if den > 0 else np.nan


def auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    score = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(x, y, np.random.default_rng(SEED).permutation(pd.factorize(groups)[0].max() + 1)[pd.factorize(groups)[0]]):
        score[te] = LogisticRegression(max_iter=2000, C=1.0).fit(x[tr], y[tr]).decision_function(x[te])
    return float(roc_auc_score(y, score))


def analyze(model: str, s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c = s[(s["model"] == model) & (s["extraction_class"] == "criterion")].copy()
    c["value"] = [a.get("value") for a in c["attr"]]
    c["stance"] = [a.get("stance") for a in c["attr"]]
    c["observation"] = [str(a.get("evidence") or "").lower().strip() for a in c["attr"]]
    c = c[(c["observation"] != "") & c["value"].notna() & c["stance"].isin(["for", "against", "conditional"])]
    c["objects"] = [tag(o) for o in c["observation"]]
    long = c.explode("objects").dropna(subset=["objects"]).rename(columns={"objects": "object"}).reset_index(drop=True)

    rows, flips = [], []
    for name, g in long.groupby("object"):
        side = g["stance"].value_counts()
        n = len(g)
        for_top, against_top = (g[g["stance"] == k]["value"].value_counts() for k in ("for", "against"))
        rows.append({"model": model, "object": name, "n": n, "images": g["image_idx"].nunique(), "for": side.get("for", 0) / n, "against": side.get("against", 0) / n,
                     "conditional": side.get("conditional", 0) / n, "for_values": ", ".join(f"{k} {v}" for k, v in for_top.head(3).items()),
                     "against_values": ", ".join(f"{k} {v}" for k, v in against_top.head(3).items())})
        t = pd.crosstab(g["value"], g["stance"]).reindex(columns=["for", "against"], fill_value=0)
        t = t[t.sum(1) >= MIN_CELL]
        if len(t) >= 2 and (t.sum(0) > 0).all():
            chi, p, v = cramer(t.to_numpy())
            share = (t["for"] / t.sum(1)).sort_values()
            flips.append({"model": model, "object": name, "values": len(t), "n": int(t.to_numpy().sum()), "chi2": chi, "p": p, "v": v, "lowest": f"{share.index[0]} {share.iloc[0]:.2f}", "highest": f"{share.index[-1]} {share.iloc[-1]:.2f}"})
    cells = pd.crosstab([long["object"], long["value"]], long["stance"]).reindex(columns=["for", "against"], fill_value=0).reset_index()
    cells = cells.assign(model=model, n=cells["for"] + cells["against"], for_share=lambda d: d["for"] / d["n"].clip(lower=1))
    flips = pd.DataFrame(flips)
    flips["p_holm"] = holm(flips["p"].to_numpy())

    sided = c[c["stance"].isin(["for", "against"]) & c["objects"].map(bool)].reset_index(drop=True)
    flat = sided.explode("objects").reset_index(drop=True)
    y = (sided["stance"] == "for").to_numpy().astype(int)
    v_x = pd.get_dummies(sided["value"]).to_numpy(dtype=float)
    o_x = np.array([[k in o for k in OBJECTS] for o in sided["objects"]], dtype=float)
    sign = []
    for name, x, tab in [("value", v_x, pd.crosstab(sided["value"], sided["stance"])), ("object", o_x, pd.crosstab(flat["objects"], flat["stance"])), ("value + object", np.hstack([v_x, o_x]), None)]:
        chi, p, v = cramer(tab.to_numpy()) if tab is not None else (np.nan, np.nan, np.nan)
        sign.append({"model": model, "predictor": name, "n": len(y), "for_share": y.mean(), "chi2": chi, "p": p, "v": v, "auc": auc(x, y, sided["image_idx"].to_numpy())})

    every = s[s["model"] == model]
    images = sorted(every["image_idx"].unique())
    key = pd.MultiIndex.from_product([images, range(3)], names=["image_idx", "repeat"])
    every = every.assign(repeat=every["image"].str.split(".").str[1].astype(int))
    crit = every[every["extraction_class"] == "criterion"].assign(value=lambda d: [a.get("value") for a in d["attr"]])
    crit = crit.assign(observation=[str(a.get("evidence") or "").lower() for a in crit["attr"]])
    repeats = []
    for run, g in crit.groupby("run"):
        for name in sorted(g["value"].dropna().unique()):
            m = g[g["value"] == name].groupby(["image_idx", "repeat"]).size().reindex(key, fill_value=0).gt(0).unstack().to_numpy(dtype=float)
            repeats.append({"model": model, "run": run, "kind": "value", "name": name, "share": m.mean(), "icc": icc(m), "recur": recurrence(m)})
        for name, pat in PATTERN.items():
            m = g[[pat.search(o) is not None for o in g["observation"]]].groupby(["image_idx", "repeat"]).size().reindex(key, fill_value=0).gt(0).unstack().to_numpy(dtype=float)
            repeats.append({"model": model, "run": run, "kind": "object", "name": name, "share": m.mean(), "icc": icc(m), "recur": recurrence(m)})
    return {"objects": pd.DataFrame(rows).sort_values("n", ascending=False), "flips": flips.sort_values("chi2", ascending=False), "sign": pd.DataFrame(sign), "repeat": pd.DataFrame(repeats), "cells": cells}


def main() -> None:
    s = load_spans()
    got = [analyze(m, s) for m in MODELS]
    for k in got[0]:
        pd.concat([g[k] for g in got], ignore_index=True).to_csv(OUT / f"argument_{k}.csv", index=False)
    print("[f10.16] finished", flush=True)


if __name__ == "__main__":
    main()
