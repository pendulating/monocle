"""F8.1 method B: the NTA effects of the F4 texts, with image-content controls.

The F4 results correlate the output of a model with the ACS features of the
NTA of the image. The images also differ by NTA (the cuisine, the
storefront). Method B asks: does the NTA effect stay when a regression
holds the image content fixed?

The unit is 1 text: 1 (model, case, field, image). For each (model, case,
field, feature), the module fits 2 OLS models:

| Model | Formula |
|-------|---------|
| 0 | outcome ~ 1 + z(feature) |
| 1, restaurant | model 0 + cuisine FE + DOHMH grade FE + borough FE |
| 1, street | model 0 + borough FE |

The standard errors are clustered by NTA (CR1). The test is the t test of
the coefficient of z(feature), with G - 1 degrees of freedom (G = the count
of NTAs). Bonferroni goes over all rows of 1 run.

The CR1 formula (Cameron and Miller 2015, eq. 11 and the small-sample
factor of Stata and statsmodels):

    V = c * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1
    c = G / (G - 1) * (N - 1) / (N - K)

N is the count of rows, K the count of columns of X, u the OLS residuals.

The default outcome is the net valence of section 4.12 of the traces
notebook: for 1 text, (good - bad) / spans over its `visual_evidence` spans
with a valence. Another outcome (for example the R3 readout `c` of F8.3)
can come from a parquet file (`--outcome-parquet`).

See `research/docs/f8-method-b.md`.

Run (the default valence outcome)::

    python research/scripts/f8/method_b.py

Run (an external outcome)::

    python research/scripts/f8/method_b.py --outcome-parquet PATH --outcome-col c_yes --tag r3_c_yes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

__version__ = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "research" / "figures" / "f8" / "method_b"
ACS_PATH = REPO_ROOT / "data" / "f4_traces" / "nta_acs.parquet"
INSPECTIONS_PATH = REPO_ROOT / "curation" / "dohmh_restaurants_inspected_all" / "restaurants.parquet"
F6_BEST = REPO_ROOT / "data" / "f6_restaurant_depiction" / "v4" / "best.parquet"
F6_CANDIDATES = REPO_ROOT / "data" / "f6_restaurant_depiction" / "v4" / "candidates.parquet"

CASES = ("restaurant", "street")
FIELDS = ("response", "reasoning")
# The predictors. `log_median_household_income` is the natural log of the
# ACS median household income (top-coded at 200,000 dollars).
FEATURES = ("pct_white_nh", "pct_black_nh", "pct_asian_nh", "pct_bachelors", "log_median_household_income")
# The fixed effects of model 1, for each case.
CONTROLS: Dict[str, tuple] = {
    "restaurant": ("cuisine", "grade", "borough"),
    "street": ("borough",),
}
# The DOHMH grades. N, Z and P are the "not yet graded" and "grade pending"
# codes; they go into 1 level.
GRADE_LEVELS = {"A": "A", "B": "B", "C": "C", "N": "pending", "Z": "pending", "P": "pending"}
GRADE_MISSING = "missing"


def _f4():
    """Import `research/notebooks/facct/_f4.py`. The import is late, thus the OLS code needs no data."""
    from research.notebooks.facct import _f4  # noqa: E402

    return _f4


# ---------------------------------------------------------------------------
# OLS with standard errors clustered by group (CR1)
# ---------------------------------------------------------------------------


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit OLS and give the CR1 cluster-robust standard errors.

    Args:
        y: the outcome, shape (N,).
        X: the design matrix with the intercept column, shape (N, K). It must
            have full column rank.
        groups: the cluster label of each row, shape (N,).

    Returns:
        A dict: `beta`, `se`, `t`, `p` (2-sided, t with G - 1 degrees of
        freedom), and the scalars `n`, `k`, `g`, `rank`.

    Raises:
        ValueError: when X does not have full column rank, or when there are
            fewer than 2 clusters.
    """
    from scipy.stats import t as t_dist

    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    rank = int(np.linalg.matrix_rank(X))
    if rank < k:
        raise ValueError(f"the design matrix has rank {rank} < {k} columns")
    codes, _ = pd.factorize(pd.Series(groups), sort=False)
    g = int(codes.max()) + 1
    if g < 2:
        raise ValueError("cluster-robust standard errors need 2 clusters or more")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    u = y - X @ beta
    # The score of each cluster: sum over its rows of x_i * u_i.
    scores = np.zeros((g, k))
    np.add.at(scores, codes, X * u[:, None])
    meat = scores.T @ scores
    c = g / (g - 1) * (n - 1) / (n - k)
    V = c * xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(V))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * t_dist.sf(np.abs(t), df=g - 1)
    return {"beta": beta, "se": se, "t": t, "p": p, "n": n, "k": k, "g": g, "rank": rank}


def design(frame: pd.DataFrame, predictor: str, fixed_effects: Sequence[str] = ()) -> np.ndarray:
    """Build the design matrix: the intercept, the predictor, then 1 dummy for each FE level.

    Each FE drops its most frequent level (the reference level). A level
    that occurs in all rows adds no column.
    """
    cols = [np.ones(len(frame)), frame[predictor].to_numpy(float)]
    for fe in fixed_effects:
        values = frame[fe].astype(str)
        levels = values.value_counts().index.tolist()
        for level in levels[1:]:
            cols.append((values == level).to_numpy(float))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------


def nta_features(acs_path: Path = ACS_PATH) -> pd.DataFrame:
    """Give the z-scored ACS features of each NTA with residents: `nta2020` and 1 column for each feature.

    The z-score goes over all NTAs of the ACS table with population above
    0 and a value (sample SD, ddof = 1). Thus 1 unit is 1 SD over the NTAs
    of New York City, the same for both cases and all models.
    """
    acs = pd.read_parquet(acs_path)
    acs = acs[acs.population > 0].copy()
    acs["log_median_household_income"] = np.log(acs.median_household_income)
    out = acs[["nta2020"]].copy()
    for f in FEATURES:
        x = acs[f]
        out[f] = (x - x.mean()) / x.std(ddof=1)
    return out.reset_index(drop=True)


def image_capture_dates() -> pd.DataFrame:
    """Give the capture time of each pooled restaurant image: `image_idx`, `captured`.

    The F4 file name is `<camis>_<recording_id>.jpg`. The F6 v4 `best`
    table gives `recordedAt` for (camis, recording_id); the F6 `candidates`
    table fills the rows that `best` does not hold.
    """
    F = _f4()
    t = pd.read_parquet(F.TRACES_PATH, columns=["image_idx", "case", "camis", "image_path"])
    t = t[t.case == "restaurant"].drop_duplicates("image_idx")
    t["recording_id"] = t.image_path.str.extract(r"_([A-Z0-9]+)\.jpg$")[0]
    t["camis"] = t.camis.astype(str)
    best = pd.read_parquet(F6_BEST, columns=["camis", "recording_id", "recordedAt"])
    best["camis"] = best.camis.astype(str)
    best = best.drop_duplicates(["camis", "recording_id"])
    out = t.merge(best, on=["camis", "recording_id"], how="left")
    miss = out.recordedAt.isna()
    if miss.any():
        cand = pd.read_parquet(F6_CANDIDATES, columns=["uid", "recording_id", "recordedAt"])
        cand = cand.rename(columns={"uid": "camis"}).assign(camis=lambda d: d.camis.astype(str))
        cand = cand.drop_duplicates(["camis", "recording_id"]).set_index(["camis", "recording_id"]).recordedAt
        key = pd.MultiIndex.from_frame(out.loc[miss, ["camis", "recording_id"]])
        out.loc[miss, "recordedAt"] = cand.reindex(key).to_numpy()
    out["captured"] = pd.to_datetime(out.recordedAt, utc=True).dt.tz_localize(None)
    return out[["image_idx", "camis", "captured"]]


def image_grades() -> pd.DataFrame:
    """Give the DOHMH grade of each pooled restaurant image: `image_idx`, `grade`, `grade_rule`, `grade_gap_days`.

    The grade is the graded inspection of the restaurant (`camis`) with the
    `grade_date` closest to the capture time of the image, before or after.
    An image with no capture time gets the latest grade. A restaurant with
    no graded inspection gets the level `missing`. The grades N, Z and P go
    into 1 level `pending`.
    """
    img = image_capture_dates()
    ins = pd.read_parquet(INSPECTIONS_PATH, columns=["camis", "inspection_date", "grade", "grade_date"])
    ins = ins[ins.grade.isin(GRADE_LEVELS)].copy()
    ins["camis"] = ins.camis.astype(str)
    ins["date"] = ins.grade_date.fillna(ins.inspection_date)
    m = img.merge(ins[["camis", "grade", "date"]], on="camis", how="left")
    m["gap"] = (m.date - m.captured).abs()
    dated = m[m.captured.notna() & m.date.notna()].sort_values(["image_idx", "gap", "date"])
    closest = dated.drop_duplicates("image_idx").assign(grade_rule="closest to capture")
    undated = m[m.captured.isna() & m.date.notna()].sort_values(["image_idx", "date"])
    latest = undated.drop_duplicates("image_idx", keep="last").assign(grade_rule="latest")
    out = img[["image_idx"]].merge(pd.concat([closest, latest])[["image_idx", "grade", "grade_rule", "gap"]],
                                   on="image_idx", how="left")
    out["grade"] = out.grade.map(GRADE_LEVELS).fillna(GRADE_MISSING)
    out["grade_rule"] = out.grade_rule.fillna("no graded inspection")
    out["grade_gap_days"] = out.gap.dt.days
    return out.drop(columns="gap")


def image_controls(case: str, min_cuisine: int = 100) -> pd.DataFrame:
    """Give the NTA and the controls of each image of `case`.

    | Column | Rule |
    |--------|------|
    | `nta2020` | `_f4.image_ntas` |
    | `borough` | the first 2 letters of the NTA code (MN, BX, BK, QN, SI) |
    | `cuisine` | restaurant only: `_f4.image_cuisines(min_cuisine)`; small cuisines are `other (small)` |
    | `grade` | restaurant only: `image_grades` |
    """
    F = _f4()
    n = F.image_ntas()
    out = n[n.case == case][["image_idx", "nta2020"]].copy()
    out["borough"] = out.nta2020.str[:2]
    if case == "restaurant":
        out = out.merge(F.image_cuisines(min_cuisine), on="image_idx", how="left", validate="one_to_one")
        out = out.merge(image_grades()[["image_idx", "grade"]], on="image_idx", how="left", validate="one_to_one")
        out["cuisine"] = out.cuisine.fillna("unknown")
        out["grade"] = out.grade.fillna(GRADE_MISSING)
    return out.reset_index(drop=True)


def load_spans():
    """Read the x2 spans as the traces notebook does: (`spans`, `raw`)."""
    F = _f4()
    raw = F.load_extractions()
    return F.spans(raw), raw


def valence_outcome(ext: pd.DataFrame, raw: pd.DataFrame, cases: Sequence[str] = CASES,
                    fields: Sequence[str] = FIELDS) -> pd.DataFrame:
    """Give the net valence of each text of the complete images: the outcome of section 4.12 at the text level.

    The texts are `_f4._complete_texts` (the complete images of section
    4.8). The spans are the `visual_evidence` spans with a valence (good,
    bad, neutral), as `_f4.valence_by_stratum`. `outcome` = (good - bad) /
    spans. A text with no such span has no outcome (NaN); `fit_all` drops it.
    """
    F = _f4()
    frames = []
    for case in cases:
        for field in fields:
            texts = F._complete_texts(raw, case, field)
            ve = ext[(ext.case == case) & (ext.field == field) & (ext.extraction_class == "visual_evidence")
                     & ext.valence.isin(F.VALENCES)]
            n = ve.groupby(["model", "image_idx", "valence"]).size().unstack("valence", fill_value=0)
            n = n.reindex(columns=list(F.VALENCES), fill_value=0)
            t = texts.merge(n.reset_index(), on=["model", "image_idx"], how="left").fillna(
                {v: 0 for v in F.VALENCES})
            t["spans"] = t[list(F.VALENCES)].sum(axis=1)
            t["outcome"] = np.where(t.spans > 0, (t.good - t.bad) / t.spans.where(t.spans > 0), np.nan)
            frames.append(t.assign(case=case, field=field))
    return pd.concat(frames, ignore_index=True)


def external_outcome(path: Path, column: str, default_field: str = "answer") -> pd.DataFrame:
    """Read a per-text outcome from a parquet file.

    The file needs the columns `model`, `case`, the outcome `column`, and
    `image_idx` or `sample_id`. A `field` column is optional; without it
    every row gets `default_field`. `sample_id` maps to `image_idx` with
    the F4 traces of the pool.

    Raises:
        ValueError: when a required column is missing, or when a sample_id
            has no image.
    """
    F = _f4()
    d = pd.read_parquet(path)
    need = {"model", "case", column}
    if not need <= set(d.columns) or not ({"image_idx", "sample_id"} & set(d.columns)):
        raise ValueError(f"{path}: needs {sorted(need)} and image_idx or sample_id; has {list(d.columns)}")
    if "image_idx" not in d.columns:
        keys = []
        for p in (F.TRACES_PATH, F.STREET_TRACES_PATH):
            keys.append(pd.read_parquet(p, columns=["case", "image_idx", "sample_id"]).drop_duplicates())
        keys = pd.concat(keys, ignore_index=True).assign(sample_id=lambda k: k.sample_id.astype(str))
        d = d.assign(sample_id=d.sample_id.astype(str)).merge(keys, on=["case", "sample_id"], how="left")
        if d.image_idx.isna().any():
            raise ValueError(f"{int(d.image_idx.isna().sum())} rows of {path} have no F4 image")
        d["image_idx"] = d.image_idx.astype(int)
    if "field" not in d.columns:
        d["field"] = default_field
    return d[["model", "case", "field", "image_idx"]].assign(outcome=d[column].astype(float))


def build_table(outcome: pd.DataFrame, min_cuisine: int = 100) -> pd.DataFrame:
    """Join the outcome to the NTA, the controls and the z-scored ACS features of each image."""
    feats = nta_features()
    frames = []
    for case, g in outcome.groupby("case", sort=False):
        if case not in CONTROLS:
            raise ValueError(f"case {case!r} has no controls; use one of {sorted(CONTROLS)}")
        c = image_controls(case, min_cuisine)
        frames.append(g.merge(c, on="image_idx", how="inner").merge(feats, on="nta2020", how="left"))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# The fits
# ---------------------------------------------------------------------------


def fit_all(table: pd.DataFrame, features: Sequence[str] = FEATURES,
            controls: Optional[Dict[str, Sequence[str]]] = None, alpha: float = 0.05) -> pd.DataFrame:
    """Fit model 0 and model 1 for each (model, case, field, feature) of `table`.

    `table` needs `model`, `case`, `field`, `image_idx`, `nta2020`,
    `outcome`, the feature columns, and the control columns of each case.
    The 2 models of 1 row use the same texts: the texts with an outcome and
    a value of the feature.

    Returns 1 row for each (model, case, field, feature): `n`, `n_nta`,
    `beta0`, `se0`, `p0`, `beta1`, `se1`, `p1`, `share_kept` = beta1 /
    beta0, `k1` (the columns of model 1), and the Bonferroni results
    `p0_bonf`, `p1_bonf`, `sig0`, `sig1` over all rows.
    """
    controls = CONTROLS if controls is None else controls
    rows = []
    for (model, case, field), g in table.groupby(["model", "case", "field"], sort=False):
        fe = list(controls.get(case, ()))
        for f in features:
            d = g.dropna(subset=["outcome", f])
            if len(d) < 3 or d.nta2020.nunique() < 2:
                continue
            r0 = ols_cluster(d.outcome, design(d, f), d.nta2020)
            r1 = ols_cluster(d.outcome, design(d, f, fe), d.nta2020)
            rows.append({
                "model": model, "case": case, "field": field, "feature": f,
                "n": len(d), "n_nta": int(d.nta2020.nunique()), "outcome_mean": float(d.outcome.mean()),
                "outcome_sd": float(d.outcome.std(ddof=1)),
                "beta0": r0["beta"][1], "se0": r0["se"][1], "p0": r0["p"][1],
                "beta1": r1["beta"][1], "se1": r1["se"][1], "p1": r1["p"][1],
                "k1": r1["k"], "controls": "+".join(fe),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["share_kept"] = out.beta1 / out.beta0
    m = len(out)
    out["p0_bonf"] = (out.p0 * m).clip(upper=1.0)
    out["p1_bonf"] = (out.p1 * m).clip(upper=1.0)
    out["sig0"] = out.p0_bonf < alpha
    out["sig1"] = out.p1_bonf < alpha
    out["tests"] = m
    return out


def summarize(fits: pd.DataFrame) -> pd.DataFrame:
    """Give 1 row for each (model, case, field): the significant features in model 0 and model 1, and the share kept.

    | Column | Meaning |
    |--------|---------|
    | `sig0`, `sig1` | the count of features significant in model 0, in model 1 (Bonferroni over the run) |
    | `sig0_and_sig1` | significant in both, with the same sign |
    | `median_share_kept` | the median of beta1 / beta0 over all features |
    | `median_share_kept_sig0` | the same, over the features significant in model 0 only |
    """
    rows = []
    for (model, case, field), g in fits.groupby(["model", "case", "field"], sort=False):
        s0 = g[g.sig0]
        rows.append({
            "model": model, "case": case, "field": field, "features": len(g), "n": int(g.n.max()),
            "sig0": int(g.sig0.sum()), "sig1": int(g.sig1.sum()),
            "sig0_and_sig1": int((g.sig0 & g.sig1 & (np.sign(g.beta0) == np.sign(g.beta1))).sum()),
            "median_share_kept": float(g.share_kept.median()),
            "median_share_kept_sig0": float(s0.share_kept.median()) if len(s0) else np.nan,
        })
    return pd.DataFrame(rows)


def case_summary(fits: pd.DataFrame) -> pd.DataFrame:
    """Give the headline numbers for each case: the same counts as `summarize`, over all models and fields."""
    rows = []
    for case, g in fits.groupby("case", sort=False):
        s0 = g[g.sig0]
        rows.append({
            "case": case, "rows": len(g), "sig0": int(g.sig0.sum()), "sig1": int(g.sig1.sum()),
            "sig0_and_sig1": int((g.sig0 & g.sig1 & (np.sign(g.beta0) == np.sign(g.beta1))).sum()),
            "median_share_kept": float(g.share_kept.median()),
            "median_share_kept_sig0": float(s0.share_kept.median()) if len(s0) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def plot_betas(fits: pd.DataFrame, title: str = "Net valence"):
    """Plot beta1 against beta0, 1 panel for each case and field, with the identity line.

    The marker fill is the model colour of `_f4`; a hollow marker is a row
    that is not significant in model 1 (Bonferroni).
    """
    import matplotlib.pyplot as plt

    F = _f4()
    F.S.apply_house_style()
    panels = [(c, f) for c in CASES for f in FIELDS if ((fits.case == c) & (fits.field == f)).any()]
    panels += [(c, f) for (c, f) in fits[["case", "field"]].drop_duplicates().itertuples(index=False)
               if (c, f) not in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(1.9 * len(panels) + 0.4, 2.3), squeeze=False)
    lim = float(np.nanmax(np.abs(fits[["beta0", "beta1"]].to_numpy()))) * 1.1
    for ax, (case, field) in zip(axes[0], panels):
        g = fits[(fits.case == case) & (fits.field == field)]
        ax.axhline(0, color="#888888", lw=0.5, zorder=1)
        ax.axvline(0, color="#888888", lw=0.5, zorder=1)
        ax.plot([-lim, lim], [-lim, lim], color="#1a1a1a", lw=0.6, ls="--", zorder=1)
        for model in [m for m in F.MODELS if m in set(g.model)]:
            h = g[g.model == model]
            color = F.MODEL_COLORS[model]
            for sig, part in h.groupby("sig1"):
                ax.scatter(part.beta0, part.beta1, s=16, marker=F.MODEL_MARKERS[model],
                           facecolors=color if sig else "none", edgecolors=color if not sig else "#1a1a1a",
                           linewidths=0.8 if not sig else 0.4, zorder=3,
                           label=model if sig or not h.sig1.any() else None)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{case}, {field}")
        ax.set_xlabel("beta, model 0 (no controls)")
    axes[0][0].set_ylabel("beta, model 1 (fixed effects)")
    handles, labels = [], []
    for ax in axes[0]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    order = [labels.index(m) for m in F.MODELS if m in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center",
               ncol=len(order), frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{title}: the NTA coefficient (per 1 SD) without and with the controls. "
                 "Hollow: not significant in model 1", fontsize=7.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# The sanity checks
# ---------------------------------------------------------------------------


def nta_level_check(ext: pd.DataFrame, raw: pd.DataFrame, model: str = "gemma-4-12b",
                    case: str = "restaurant", field: str = "response", feature: str = "pct_white_nh",
                    min_images: int = 20) -> dict:
    """Compute the 4.12 NTA result again with `_f4`, and the NTA-level OLS slope on z(feature).

    Returns the Spearman rho of the 4.12 table (the NTA `net` of category
    `any`) and the OLS slope of the NTA `net` on the z-scored feature over
    the same NTAs, to compare with beta0 of the text level.
    """
    from scipy.stats import spearmanr

    F = _f4()
    v = F.valence_by_stratum(ext, raw, case, field, "nta", min_images)
    v = v[(v.model == model) & (v.category == "any")][["group", "net", "spans"]]
    feats = nta_features().set_index("nta2020")
    v = v[v.group.isin(feats.index)]
    x = feats.loc[v.group, feature].to_numpy()
    ok = ~np.isnan(x)
    rho, p = spearmanr(v.net.to_numpy()[ok], x[ok])
    slope = np.polyfit(x[ok], v.net.to_numpy()[ok], 1)[0]
    return {"model": model, "case": case, "field": field, "feature": feature, "ntas": int(ok.sum()),
            "spearman_rho_nta": float(rho), "p": float(p), "ols_slope_nta_per_sd": float(slope)}


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outcome-parquet", type=Path, default=None,
                    help="a per-text outcome: model, case, image_idx or sample_id, the outcome column, field (optional)")
    ap.add_argument("--outcome-col", default="outcome", help="the outcome column of --outcome-parquet")
    ap.add_argument("--tag", default=None, help="the prefix of the output files (default: valence, or the outcome column)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--fields", nargs="+", default=list(FIELDS), help="valence outcome only")
    ap.add_argument("--min-cuisine", type=int, default=100)
    ap.add_argument("--min-nta-images", type=int, default=1,
                    help="keep the NTAs with this many images or more of the case (default 1: all)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ext = raw = None
    if args.outcome_parquet is None:
        tag = args.tag or "valence"
        print("read the x2 spans ...", flush=True)
        ext, raw = load_spans()
        outcome = valence_outcome(ext, raw, args.cases, args.fields)
        title = "Net valence"
    else:
        tag = args.tag or args.outcome_col
        outcome = external_outcome(args.outcome_parquet, args.outcome_col)
        outcome = outcome[outcome.case.isin(args.cases)]
        title = args.outcome_col
    table = build_table(outcome, args.min_cuisine)
    if args.min_nta_images > 1:
        size = table.drop_duplicates(["case", "image_idx"]).groupby(["case", "nta2020"]).size()
        keep = size[size >= args.min_nta_images].reset_index()[["case", "nta2020"]]
        table = table.merge(keep, on=["case", "nta2020"])
        tag += f"_nta{args.min_nta_images}"

    fits = fit_all(table, alpha=args.alpha)
    summary = summarize(fits)
    cases = case_summary(fits)
    fits.to_csv(args.out_dir / f"{tag}_fits.csv", index=False)
    summary.to_csv(args.out_dir / f"{tag}_summary.csv", index=False)
    cases.to_csv(args.out_dir / f"{tag}_cases.csv", index=False)
    fig = plot_betas(fits, title)
    fig.savefig(args.out_dir / f"{tag}_betas.png")

    # The counts of the texts: what entered, and what had no outcome.
    counts = (table.assign(has_outcome=table.outcome.notna())
              .groupby(["case", "field", "model"])
              .agg(texts=("image_idx", "size"), images=("image_idx", "nunique"),
                   with_outcome=("has_outcome", "sum"), ntas=("nta2020", "nunique"))
              .reset_index())
    counts.to_csv(args.out_dir / f"{tag}_counts.csv", index=False)
    meta = {"script": "research/scripts/f8/method_b.py", "version": __version__, "tag": tag,
            "outcome_parquet": str(args.outcome_parquet) if args.outcome_parquet else None,
            "outcome_col": args.outcome_col if args.outcome_parquet else "net valence (section 4.12)",
            "cases": args.cases, "min_cuisine": args.min_cuisine, "min_nta_images": args.min_nta_images,
            "alpha": args.alpha, "tests": int(len(fits)), "seconds": round(time.time() - t0, 1)}
    if "restaurant" in args.cases:
        gr = image_grades()
        meta["grade_levels"] = gr.grade.value_counts().to_dict()
        meta["grade_rules"] = gr.grade_rule.value_counts().to_dict()
        meta["grade_gap_days_median"] = float(gr.grade_gap_days.median())
        cu = image_controls("restaurant", args.min_cuisine).cuisine
        meta["cuisine_levels"] = int(cu.nunique())
    if ext is not None:
        checks = [nta_level_check(ext, raw, feature=f) for f in ("pct_white_nh", "pct_bachelors")]
        checks.append(nta_level_check(ext, raw, model="qwen3.8-27b", feature="pct_white_nh"))
        meta["nta_level_checks"] = checks
    (args.out_dir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(cases.to_string(index=False))
    print(f"wrote {tag}_* to {args.out_dir} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
