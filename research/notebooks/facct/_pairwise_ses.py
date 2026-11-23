"""The pairwise results of 3 cases against the socioeconomic features of the area.

The CVPR figure `research/notebooks/cvpr/_proxy_overview.py` read each pairwise case
against outside measurements (crime, inspections, ridership), with the
white-box lens read beside it. The FAccT paper reads 3 of those cases against
the 5 ACS features that the F4, F8 and F9 figures use, with no white-box read:

| Case | Question | Area score |
|------|----------|------------|
| Subway safety | which station entrance looks safer | `subway/outputs/subway_aggregated_v1.0.0.parquet` (the unit score, mean over the area) |
| Street photography | which block is a more appealing location for a street photography photoshoot | `_maps.image_scores_by_layer` (the image score, mean over the area, 5 images minimum) |
| Restaurants | which restaurant would you rather eat at | `restaurants/outputs/restaurants_aggregated_v1.0.0.parquet` |

The scores come from the registered 1,000,000-pair runs of Gemma-4-12B and
Qwen3.5-9B (the canonical registry, `research/notebooks/cvpr/README.md`). A higher
score is the side the model picks. The features come from
`research/scripts/f4/nta_acs.py` (ACS 2024 5-year), at the census tract, the NTA and
the community district, by the same method at each layer.

The test of 1 cell is the Kendall tau-b of the area score with 1 feature over
the areas with residents, at 1 layer. A cell needs 20 areas
(`_results_table.MIN_AREAS`). Bonferroni runs over all the cells of the
figure. A positive tau says that the model picks the images of areas with
more of the feature.

The restaurant check (`restaurant_f6_cells`, `--f6`) asks if the restaurant
gradients need a visible storefront. The pairwise runs show raw cube faces, and
the bottom of the ranking holds many restaurants that fail the F6 pixel check.
The check runs the same notebook steps (`score_units`, `_geography.aggregate`,
3 units for each area) in 3 variants:

| Variant | Restaurants | Comparisons |
|---------|-------------|-------------|
| `all` | all; this must give the values of the figure | all |
| `yes` | F6 `Yes` only | all, thus the opponents can be any restaurant |
| `yes_pairs` | F6 `Yes` only | only the pairs with 2 `Yes` restaurants |

The pixel check read the F6 view, not the raw faces. Thus a `Yes` restaurant
can still have raw faces with no storefront, and this check is a lower bound
on the effect. `share_yes` tests the path itself: the Kendall tau of the share
of `Yes` restaurants in an area with each feature.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/notebooks/facct/_pairwise_ses.py
    .venv-mllmsci-vllm025cu129/bin/python research/notebooks/facct/_pairwise_ses.py --f6
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
_CVPR = REPO_ROOT / "research" / "notebooks" / "cvpr"

from research.notebooks.cvpr import _style as S  # noqa: E402

__version__ = "1.1.0"

OUT = REPO_ROOT / "research" / "notebooks" / "facct" / "figures" / "pairwise"
CACHE = REPO_ROOT / "data" / "facct_pairwise"
F6_CHECK = REPO_ROOT / "data" / "f6_restaurant_depiction" / "v4" / "facade_check_b300"
ACS = REPO_ROOT / "data" / "f4_traces"

# case key -> (label, CVPR folder, export prefix or None for an image-mode case)
CASES: Dict[str, Tuple[str, str, Optional[str]]] = {
    "subway_safety": ("Subway safety", "subway", "subway"),
    "street_photography": ("Street photography", "street_photography", None),
    "restaurants": ("Restaurants", "restaurants", "restaurants"),
}
FEATURES = ("pct_white_nh", "pct_black_nh", "pct_asian_nh", "pct_bachelors", "median_household_income")
FEATURE_LABELS = {"pct_white_nh": "% white", "pct_black_nh": "% Black", "pct_asian_nh": "% Asian",
                  "pct_bachelors": "% bachelor's", "median_household_income": "Median income"}
# CVPR layer -> (short label, the key of the CVPR export, the ACS file, the key of the ACS file)
LAYERS: Dict[str, Tuple[str, str, str, str]] = {
    "census_tract": ("tract", "boroct2020", "tract_acs.parquet", "boroct2020"),
    "nta": ("NTA", "NTA2020", "nta_acs.parquet", "nta2020"),
    "community_district": ("CD", "boro_cd", "cd_acs.parquet", "boro_cd"),
}
MODELS = ("gemma-4-12b", "qwen3.5-9b")
MODEL_SHORT = {"gemma-4-12b": "Gemma-4-12B", "qwen3.5-9b": "Qwen3.5-9B"}
# The colours of the CVPR figure: blue and amber pass the colour-vision checks as a pair.
MODEL_COLOUR = {"gemma-4-12b": S.PAL["blue"], "qwen3.5-9b": S.PAL["amber"]}
MIN_AREAS = 20
ALPHA = 0.05


def _model(m: str) -> str:
    return m.split("/")[0]


def area_scores(case: str, layer: str) -> pd.DataFrame:
    """Give the score of each area for each model: columns `model`, `key`, `mean_score`.

    The areas with no residents (`_geography.excluded_keys`) drop out. The
    image-mode case is slow (1,100,000 pairs for each model), thus its
    result goes to a cache in `data/facct_pairwise/`.
    """
    from research.notebooks.cvpr import _geography as geo

    _, folder, prefix = CASES[case]
    short, key, _, _ = LAYERS[layer]
    if prefix is not None:
        agg = pd.read_parquet(_CVPR / folder / "outputs" / f"{prefix}_aggregated_v1.0.0.parquet")
        agg = agg[agg["layer"] == layer]
        out = pd.DataFrame({"model": agg["model"].map(_model), "key": agg[key].astype(str),
                            "mean_score": agg["mean_score"]})
    else:
        path = CACHE / f"{case}_{layer}.parquet"
        if path.exists():
            out = pd.read_parquet(path)
        else:
            from research.notebooks.cvpr import _maps as M

            sc = M.image_scores_by_layer(case, layer)
            out = pd.DataFrame({"model": sc["model"].map(_model), "key": sc[key].astype(str),
                                "mean_score": sc["mean_score"], "n_images": sc["n_images"]})
            CACHE.mkdir(parents=True, exist_ok=True)
            out.to_parquet(path, index=False)
    drop = geo.excluded_keys(layer)
    return out[~out["key"].isin(drop)].dropna(subset=["key", "mean_score"]).reset_index(drop=True)


def features(layer: str) -> pd.DataFrame:
    """Give the 5 features of each area with residents, keyed by `key`."""
    _, _, name, key = LAYERS[layer]
    acs = pd.read_parquet(ACS / name)
    acs = acs[acs["population"] > 0]
    return acs.rename(columns={key: "key"}).assign(key=lambda d: d["key"].astype(str))[["key", *FEATURES]]


def cells() -> pd.DataFrame:
    """Give 1 row for each (case, feature, model, layer): n, tau, p, and the Bonferroni p over all the rows that run."""
    from scipy.stats import kendalltau

    rows = []
    for layer in LAYERS:
        feats = features(layer)
        for case in CASES:
            sc = area_scores(case, layer).merge(feats, on="key", how="inner")
            for model in MODELS:
                part = sc[sc["model"] == model]
                for f in FEATURES:
                    ok = part[["mean_score", f]].dropna()
                    row = {"case": case, "feature": f, "model": model, "layer": LAYERS[layer][0], "n": len(ok),
                           "tau": np.nan, "p": np.nan}
                    if len(ok) >= MIN_AREAS:
                        t = kendalltau(ok["mean_score"], ok[f])
                        row.update(tau=float(t.statistic), p=float(t.pvalue))
                    rows.append(row)
    out = pd.DataFrame(rows)
    m = int(out["p"].notna().sum())
    out["p_bonf"] = (out["p"] * m).clip(upper=1.0)
    out["significant"] = out["p_bonf"] < ALPHA
    out.attrs["tests"] = m
    return out


def f6_answers() -> pd.Series:
    """Give the answer of the F6 pixel check (`Yes`, `No`) for each restaurant id (`camis`, as a string)."""
    import glob

    parts = sorted(glob.glob(str(F6_CHECK / "part_*.parquet")))
    check = pd.concat([pd.read_parquet(p, columns=["uid", "answer"]) for p in parts]).drop_duplicates("uid")
    return check.assign(uid=check["uid"].astype(str)).set_index("uid")["answer"]


def _restaurant_units():
    """Give the restaurant point layer, with `unit_uid` as a string."""
    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _proxies_restaurants as pr

    units = geo.load_units(pr.CURATION_ROOT, filename=pr.UNIT_TABLE)
    return units.assign(unit_uid=units["unit_uid"].astype(str))


def _areas(scores: pd.DataFrame, units, layer: str) -> pd.DataFrame:
    """Give the mean score and the features of each area with residents and 3 or more units."""
    from research.notebooks.cvpr import _geography as geo

    key = geo.LAYERS[layer]["key"]
    area = geo.aggregate(scores, units, layer, min_units=3).assign(key=lambda d: d[key].astype(str))
    return area[~area["key"].isin(geo.excluded_keys(layer))].merge(features(layer), on="key")


def restaurant_f6_cells(answers: Optional[pd.Series] = None) -> pd.DataFrame:
    """Give the Kendall tau of each restaurant cell in each variant.

    Each row also holds `tau_all_same`: the `all` value over the same areas,
    thus a change does not come from a change of areas. The column
    `significant` uses the Bonferroni count of the figure (90), thus it reads
    as a filled or hollow marker of the figure.
    """
    from research.notebooks.cvpr import _provenance as prov
    from scipy.stats import kendalltau

    answers = f6_answers() if answers is None else answers
    yes = set(answers.index[answers == "Yes"])
    units = _restaurant_units()
    rows = []
    for rec in prov.discover_runs("restaurants"):
        model = _model(rec.model)
        run = prov.load_run(rec)
        a, b = run["unit_uid_a"].astype("string"), run["unit_uid_b"].astype("string")
        run = run.assign(unit_uid_a=a, unit_uid_b=b)
        full = prov.score_units(run)
        full = full.assign(unit_uid=full["unit_uid"].astype(str))
        both = prov.score_units(run[a.isin(yes) & b.isin(yes)])
        both = both.assign(unit_uid=both["unit_uid"].astype(str))
        scores = {"all": full, "yes": full[full["unit_uid"].isin(yes)], "yes_pairs": both}
        for layer, (short, _, _, _) in LAYERS.items():
            area = {v: _areas(sc, units, layer) for v, sc in scores.items()}
            for variant, part in area.items():
                same = area["all"][area["all"]["key"].isin(part["key"])]
                for f in FEATURES:
                    ok, base = part[["mean_score", f]].dropna(), same[["mean_score", f]].dropna()
                    t = kendalltau(ok["mean_score"], ok[f])
                    rows.append({"variant": variant, "model": model, "layer": short, "feature": f, "n": len(ok),
                                 "tau": float(t.statistic), "p": float(t.pvalue),
                                 "tau_all_same": float(kendalltau(base["mean_score"], base[f]).statistic),
                                 "units_scored": len(scores[variant])})
    out = pd.DataFrame(rows)
    out["p_bonf"] = (out["p"] * 90).clip(upper=1.0)
    out["significant"] = out["p_bonf"] < ALPHA
    return out


def restaurant_answer_share(answers: Optional[pd.Series] = None) -> pd.DataFrame:
    """Give the Kendall tau of the share of `Yes` and of `No` restaurants of an area with each feature.

    The denominator is all the restaurants of the area, thus a restaurant with
    no F6 view counts in the share. The floor of 3 units is the same.
    """
    from scipy.stats import kendalltau

    answers = f6_answers() if answers is None else answers
    units = _restaurant_units()
    got = units["unit_uid"].map(answers)
    rows = []
    for answer in ("Yes", "No"):
        flag = pd.DataFrame({"unit_uid": units["unit_uid"], "mean_score": got.eq(answer).astype(float),
                             "n_comparisons": 1, "abstention_rate": 0.0})
        for layer, (short, _, _, _) in LAYERS.items():
            part = _areas(flag, units, layer)
            for f in FEATURES:
                ok = part[["mean_score", f]].dropna()
                t = kendalltau(ok["mean_score"], ok[f])
                rows.append({"answer": answer, "layer": short, "feature": f, "n": len(ok), "tau": float(t.statistic),
                             "p": float(t.pvalue), "mean_share": float(ok["mean_score"].mean())})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, width: float = 3.33):
    """Draw the Kendall tau of each (case, feature) at each layer, 1 row for each feature (1 column wide).

    The marker size grows with the layer (tract, NTA, community district); a
    line joins the 3 layers of 1 model, and a hollow marker is not
    significant after Bonferroni.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    size = {"tract": 9, "NTA": 18, "CD": 32}
    off = {MODELS[0]: -0.18, MODELS[1]: 0.18}
    ys, labels, heads = [], [], []
    y = 0.0
    for i, case in enumerate(CASES):
        if i:
            y += 0.35
        heads.append((y, CASES[case][0]))
        y += 1
        for f in FEATURES:
            ys.append((case, f, y))
            labels.append(FEATURE_LABELS[f])
            y += 1
    row_h = 0.15
    fig, ax = plt.subplots(figsize=(width, row_h * y + 0.75))
    for case, f, yy in ys:
        for m in MODELS:
            part = df[(df["case"] == case) & (df["feature"] == f) & (df["model"] == m) & df["tau"].notna()]
            xs = []
            for short in ("tract", "NTA", "CD"):
                r = part[part["layer"] == short]
                if r.empty:
                    continue
                r = r.iloc[0]
                xs.append(r["tau"])
                ax.scatter(r["tau"], yy + off[m], s=size[short],
                           facecolors=MODEL_COLOUR[m] if r["significant"] else "white",
                           edgecolors=S.EDGE if r["significant"] else MODEL_COLOUR[m],
                           linewidths=0.5 if r["significant"] else 0.9, zorder=4)
            if len(xs) >= 2:
                ax.plot([min(xs), max(xs)], [yy + off[m]] * 2, color=MODEL_COLOUR[m], lw=0.8, alpha=0.7, zorder=3)
    ax.set_yticks([yy for *_, yy in ys] + [h for h, _ in heads])
    ax.set_yticklabels(labels + [t for _, t in heads], fontsize=6.5)
    for tick, (_, t) in zip(ax.get_yticklabels()[len(labels):], heads):
        tick.set_fontweight("bold")
    for h, _ in heads[1:]:
        ax.axhline(h - 0.68, color=S.EDGE, lw=0.4)
    ax.axvline(0, color=S.EDGE, lw=0.8, zorder=2)
    ax.set_ylim(y - 0.4, -0.6)
    lo, hi = min(0.0, float(np.nanmin(df["tau"]))), max(0.0, float(np.nanmax(df["tau"])))
    pad = 0.06 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.4, alpha=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"Kendall $\tau$ of the area score with the feature", fontsize=7)
    grey = "grey"
    handles = [Line2D([], [], marker="o", ls="", ms=3, color=MODEL_COLOUR[m], label=MODEL_SHORT[m]) for m in MODELS]
    handles += [Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec=grey, label="not significant"),
                Line2D([], [], marker="o", ls="", ms=2.0, mfc=grey, mec=grey, label="tract"),
                Line2D([], [], marker="o", ls="", ms=3.0, mfc=grey, mec=grey, label="NTA"),
                Line2D([], [], marker="o", ls="", ms=4.1, mfc=grey, mec=grey, label="community district")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.42, -0.1 * 4.6 / (row_h * y + 0.75)),
              ncol=3, frameon=False, fontsize=6, handletextpad=0.3, columnspacing=0.8)
    return fig


def caption(df: pd.DataFrame) -> str:
    n = {(r.case, r.layer): int(r.n) for r in df[df.model == MODELS[0]].drop_duplicates(["case", "layer"]).itertuples()}
    areas = "; ".join(f"{CASES[c][0].lower()} {n.get((c, 'tract'), 0)}, {n.get((c, 'NTA'), 0)} and "
                      f"{n.get((c, 'CD'), 0)}" for c in CASES)
    return ("The pairwise judgments of 3 cases against the socioeconomic features of the area. Each row is the "
            "Kendall tau-b of the area score of 1 model with 1 ACS 2024 feature, over the areas with residents. "
            "The area score is the mean score of the units (subway entrances, restaurants) or images (street "
            "photography) in the area, from 1,000,000 image pairs for each model. A positive value says that the "
            "model picks the area with more of the feature. The marker size grows with the layer: census tract, "
            "NTA and community district; a line joins the 3 layers of 1 model. A hollow marker is not significant "
            f"after Bonferroni over the {df.attrs.get('tests', int(df.p.notna().sum()))} tests of the figure "
            f"(alpha {ALPHA}). Areas for each layer (tract, NTA, CD): {areas}.")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    from research.notebooks.cvpr import _figmeta as FM

    S.apply_house_style()
    OUT.mkdir(parents=True, exist_ok=True)
    if "--f6" in sys.argv:
        answers = f6_answers()
        f6 = restaurant_f6_cells(answers)
        f6.to_csv(OUT / "pairwise_ses_restaurants_f6.csv", index=False)
        share = restaurant_answer_share(answers)
        share.to_csv(OUT / "pairwise_ses_restaurants_f6_share.csv", index=False)
        print(f6.pivot_table(index=["layer", "feature"], columns=["model", "variant"], values="tau").round(2).to_string())
        print(f6.groupby(["model", "variant", "layer"])["n"].first().unstack().to_string())
        print(f6.groupby(["model", "variant"])["units_scored"].first().to_string())
        print(f6.groupby(["model", "variant", "layer"])["significant"].sum().unstack().to_string())
        print(share.round(3).to_string())
        return 0
    df = cells()
    df.to_csv(OUT / "pairwise_ses_cells.csv", index=False)
    fig = plot(df)
    FM.save_figure(fig, OUT / "pairwise_ses.png", dpi=300, title="The pairwise judgments and the area features",
                   caption=caption(df), label="fig:pairwise-ses",
                   tags=["pairwise", "subway_safety", "street_photography", "restaurants", "acs", "kendall"],
                   provenance={"script": "research/notebooks/facct/_pairwise_ses.py", "version": __version__})
    fig.savefig(OUT / "pairwise_ses.pdf", bbox_inches="tight")
    print(df.pivot_table(index=["case", "feature"], columns=["model", "layer"], values="tau").round(2).to_string())
    print("significant", int(df.significant.sum()), "of", df.attrs["tests"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
