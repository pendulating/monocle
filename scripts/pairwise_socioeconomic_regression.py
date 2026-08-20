#!/usr/bin/env python
"""Tract-level socioeconomic regression for urbanpairvqa layers.

Regresses model-derived latent scores for the **street-photography appeal** and
**subway-entrance safety** layers against two census-tract covariates:

  * median household income (ACS 5-Year 2024, table S1901, on disk)
  * a severity-weighted NYPD complaint density (FELONY=3 / MISDEMEANOR=2 /
    VIOLATION=1, summed per tract and divided by tract land area in km^2)

Per layer the recipe mirrors ``notebooks/css/wealth.ipynb`` (the canonical
image -> tract TrueSkill -> OLS-vs-covariate pattern), generalised to two
covariates and a multi-model normalized ensemble:

  result parquet (per model, repeat_idx==0)
    -> merge sample_id -> lat/lon (source manifest)
    -> point-in-polygon each side to a 2020 census tract
    -> relabel unit_uid = tract geoid, drop intra-tract pairs
    -> scripts.pairwise_vqa_report._compute_trueskill  (per-tract mu)
    -> z-score mu within model, average across models = normalized ensemble
    -> join tract income + severity-weighted crime density
    -> OLS  mu ~ income | crime | income+crime   (HC3 robust SE)
    -> scatter + choropleth plots, markdown report
    -> mirror_to_wandb  (metrics + table + report artifacts)

Run:
    .venv/bin/python scripts/pairwise_socioeconomic_regression.py
    .venv/bin/python scripts/pairwise_socioeconomic_regression.py --no-wandb
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/share/pierson/matt/mllmsci")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import geopandas as gpd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats as sps  # noqa: E402

from pairwise_vqa_report import _compute_trueskill  # noqa: E402

# ---------------------------------------------------------------------------
# Static inputs
# ---------------------------------------------------------------------------
TRACT_GEOJSON = REPO / "data/geo/2020_Census_Tracts_20260304.geojson"
INCOME_CSV = REPO / "data/demo/ct/ACSST5Y2024.S1901-Data.csv"
CRIME_CSV = REPO / "data/external/nypd_complaints_ytd/nypd_complaint_ytd_2026_pulled20260629.csv"
# ACS 2024 5-year B01003 total population by tract (via Census Reporter, keyless).
POP_PARQUET = REPO / "data/external/acs_tract_population/acs_b01003_nyc_tract.parquet"

SEVERITY_WEIGHTS = {"FELONY": 3.0, "MISDEMEANOR": 2.0, "VIOLATION": 1.0}
SQFT_PER_KM2 = 10_763_910.41671  # 1 km^2 in US-survey sq feet (EPSG:2263)
DRAW_PROB = 0.05  # wealth.ipynb canon
# Sweep model index -> name. phi-4-mm (idx 2) is intentionally omitted: it
# produces degenerate, non-discriminative pairvqa output (100% NotSure on these
# two layers; 88% "Same" elsewhere — see memory feedback_drop_phi4_from_sweeps /
# project_phi4_mm_degenerate) and crashed on these layers anyway, so it never had
# usable data. Keeping it out of MODEL_NAMES makes the exclusion explicit.
MODEL_NAMES = {
    0: "gemma-4-e2b",
    1: "gemma-4-e4b",
    3: "qwen3.5-2b",
    4: "qwen3.5-4b",
    5: "qwen3.5-9b",
}

# Extra single-model runs folded in as additional raters. gemma-4-12b (the
# encoder-free gemma4_unified 12B) ran as its own klara_1x sweep on 2026-06-29;
# both layers' result parquets landed in the same stage dir (same-second launch
# collision — see memory project_hydra_sweep_dir_stage_collision), but the
# layer-specific ``result_glob`` disambiguates which one each layer picks up.
_GEMMA12B_RUN = REPO / "multirun/2026-06-29_URBANPAIRVQA/14-37-17/0"

LAYERS = {
    "subway": {
        "title": "Subway-entrance safety",
        "base": REPO / "multirun/2026-06-25_URBANPAIRVQA/22-11-26",
        "manifest": REPO / "curation/subway_entrances_all/cyclomedia_near_subway_facing.parquet",
        "result_glob": "outputs/pairwise/subway_safety_mvp_*.parquet",
        "extra_runs": {"gemma-4-12b": _GEMMA12B_RUN},
    },
    "street": {
        "title": "Street-photography appeal",
        "base": REPO / "multirun/2026-06-25_URBANPAIRVQA/22-11-31",
        "manifest": REPO / "data/cyclomedia/cyclomedia_all_2025_citywide_500k.parquet",
        "result_glob": "outputs/pairwise/street_photography_mvp_*.parquet",
        "extra_runs": {"gemma-4-12b": _GEMMA12B_RUN},
    },
}

# ---------------------------------------------------------------------------
# Covariate builders (computed once, shared across layers)
# ---------------------------------------------------------------------------

def load_tracts() -> gpd.GeoDataFrame:
    tr = gpd.read_file(TRACT_GEOJSON)[
        ["geoid", "boroname", "ntaname", "geometry"]
    ].copy()
    tr["geoid"] = tr["geoid"].astype(str)
    tr["area_km2"] = tr.to_crs(2263).area / SQFT_PER_KM2
    return tr


def load_income() -> pd.DataFrame:
    inc = pd.read_csv(INCOME_CSV, header=1, dtype=str, low_memory=False)
    geoid = inc["Geography"].str.extract(r"US(\d{11})$")[0]
    med = pd.to_numeric(
        inc["Estimate!!Households!!Median income (dollars)"], errors="coerce"
    )
    mean = pd.to_numeric(
        inc.get("Estimate!!Households!!Mean income (dollars)"), errors="coerce"
    )
    out = pd.DataFrame(
        {"geoid": geoid, "median_income": med.values, "mean_income": mean.values}
    ).dropna(subset=["geoid"])
    return out


def load_population() -> pd.DataFrame:
    pop = pd.read_parquet(POP_PARQUET)
    pop["geoid"] = pop["geoid"].astype(str)
    return pop[["geoid", "population"]]


def load_crime(tracts: gpd.GeoDataFrame, population: pd.DataFrame) -> pd.DataFrame:
    """Per-tract severity-weighted crime, expressed as both a density (per km^2)
    and a per-capita rate (per 1,000 residents)."""
    cr = pd.read_csv(
        CRIME_CSV,
        usecols=["law_cat_cd", "latitude", "longitude"],
    ).dropna(subset=["latitude", "longitude"])
    cr["w"] = cr["law_cat_cd"].map(SEVERITY_WEIGHTS).fillna(0.0)
    pts = gpd.GeoDataFrame(
        cr, geometry=gpd.points_from_xy(cr["longitude"], cr["latitude"]), crs=4326
    )
    j = gpd.sjoin(pts, tracts[["geoid", "geometry"]], how="inner", predicate="within")
    agg = j.groupby("geoid").agg(
        crime_count=("w", "size"),
        crime_weighted=("w", "sum"),
        felony_count=("law_cat_cd", lambda s: (s == "FELONY").sum()),
    )
    agg = agg.merge(
        tracts[["geoid", "area_km2"]].set_index("geoid"),
        left_index=True, right_index=True, how="left",
    )
    agg["crime_density"] = agg["crime_weighted"] / agg["area_km2"]
    agg["felony_density"] = agg["felony_count"] / agg["area_km2"]
    agg = agg.reset_index().merge(population, on="geoid", how="left")
    # Per-capita rate per 1,000 residents; tracts with ~0 population (parks,
    # cemeteries) are undefined -> NaN (dropped downstream).
    pop_ok = agg["population"].where(agg["population"] >= 50)
    agg["crime_per_capita"] = agg["crime_weighted"] / pop_ok * 1000.0
    agg["felony_per_capita"] = agg["felony_count"] / pop_ok * 1000.0
    return agg


# ---------------------------------------------------------------------------
# Per-layer rating
# ---------------------------------------------------------------------------

def discover_models(layer: dict) -> Dict[str, Path]:
    """Map model name -> result parquet for the task dirs that produced one.

    Includes the indexed sweep models (``layer['base']/<idx>/``) plus any
    standalone ``extra_runs`` (e.g. the gemma-4-12b klara_1x run). The
    layer-specific ``result_glob`` selects the right parquet in each dir.
    """
    found: Dict[str, Path] = {}
    for idx, name in MODEL_NAMES.items():
        hits = glob.glob(str(layer["base"] / str(idx) / layer["result_glob"]))
        if hits:
            found[name] = Path(sorted(hits)[-1])
    for name, run_dir in layer.get("extra_runs", {}).items():
        hits = glob.glob(str(Path(run_dir) / layer["result_glob"]))
        if hits:
            found[name] = Path(sorted(hits)[-1])
    return found


def sampleid_to_tract(manifest: Path, tracts: gpd.GeoDataFrame,
                      needed: set) -> Dict[str, str]:
    man = pd.read_parquet(manifest, columns=["sample_id", "latitude", "longitude"])
    man["sample_id"] = man["sample_id"].astype(str)
    man = man[man["sample_id"].isin(needed)].drop_duplicates("sample_id")
    man = man.dropna(subset=["latitude", "longitude"])
    pts = gpd.GeoDataFrame(
        man, geometry=gpd.points_from_xy(man["longitude"], man["latitude"]), crs=4326
    )
    j = gpd.sjoin(pts, tracts[["geoid", "geometry"]], how="left", predicate="within")
    j = j.dropna(subset=["geoid"])
    return dict(zip(j["sample_id"].astype(str), j["geoid"].astype(str)))


def tract_ratings(result_pq: Path, s2t: Dict[str, str]) -> pd.DataFrame:
    df = pd.read_parquet(
        result_pq,
        columns=["repeat_idx", "sample_id_a", "sample_id_b", "relative_score"],
    )
    df = df[df["repeat_idx"] == 0]                       # one obs / canonical pair
    df = df.dropna(subset=["relative_score"])            # drop NotSure abstentions
    df["unit_uid_a"] = df["sample_id_a"].astype(str).map(s2t)
    df["unit_uid_b"] = df["sample_id_b"].astype(str).map(s2t)
    df = df.dropna(subset=["unit_uid_a", "unit_uid_b"])
    df = df[df["unit_uid_a"] != df["unit_uid_b"]]        # drop intra-tract pairs
    ratings = _compute_trueskill(df, DRAW_PROB)
    return ratings.rename(columns={"unit_uid": "geoid"})[
        ["geoid", "mu", "sigma", "n_comparisons"]
    ]


def normalized_ensemble(per_model: Dict[str, pd.DataFrame], min_comp: int,
                        min_models: int) -> pd.DataFrame:
    frames = []
    for name, r in per_model.items():
        r = r[r["n_comparisons"] >= min_comp].copy()
        r["mu_z"] = (r["mu"] - r["mu"].mean()) / r["mu"].std(ddof=0)
        frames.append(r[["geoid", "mu_z"]].assign(model=name))
    allz = pd.concat(frames, ignore_index=True)
    ens = allz.groupby("geoid").agg(
        mu=("mu_z", "mean"), n_models=("mu_z", "size")
    ).reset_index()
    ens = ens[ens["n_models"] >= min_models]
    ens["sigma"] = np.nan
    ens["n_comparisons"] = ens["n_models"]
    return ens


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _z(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std(ddof=0)


_CRIME_METRICS = {"density": "crime_density", "percapita": "crime_per_capita"}


def regress(df: pd.DataFrame, ycol: str) -> Dict[str, float]:
    """Standardized OLS with HC3 SEs. Income-only model, plus crime-only and
    income+crime joint models for BOTH crime metrics (density per km^2 and
    per-capita per 1k residents). Predictors standardized; crime log1p'd first.
    Flat scalar dict; crime keys suffixed _density / _percapita.
    """
    need = [ycol, "median_income", *_CRIME_METRICS.values()]
    d = df.dropna(subset=need).copy()
    if len(d) < 10:
        return {"n": float(len(d))}
    d["y"] = _z(d[ycol])
    d["income_z"] = _z(d["median_income"])

    out: Dict[str, float] = {"n": float(len(d))}
    out["pearson_income"] = float(sps.pearsonr(d["y"], d["income_z"])[0])
    out["spearman_income"] = float(sps.spearmanr(d[ycol], d["median_income"])[0])
    fi = sm.OLS(d["y"], sm.add_constant(d[["income_z"]])).fit(cov_type="HC3")
    out["r2_income"] = float(fi.rsquared)
    out["beta_income"] = float(fi.params["income_z"])
    out["p_income"] = float(fi.pvalues["income_z"])

    for tag, col in _CRIME_METRICS.items():
        cz = f"crime_{tag}_z"
        d[cz] = _z(np.log1p(d[col]))
        out[f"pearson_crime_{tag}"] = float(sps.pearsonr(d["y"], d[cz])[0])
        out[f"spearman_crime_{tag}"] = float(sps.spearmanr(d[ycol], d[col])[0])
        fc = sm.OLS(d["y"], sm.add_constant(d[[cz]])).fit(cov_type="HC3")
        out[f"r2_crime_{tag}"] = float(fc.rsquared)
        out[f"beta_crime_{tag}"] = float(fc.params[cz])
        out[f"p_crime_{tag}"] = float(fc.pvalues[cz])
        fj = sm.OLS(d["y"], sm.add_constant(d[["income_z", cz]])).fit(cov_type="HC3")
        out[f"r2_joint_{tag}"] = float(fj.rsquared)
        out[f"beta_jinc_{tag}"] = float(fj.params["income_z"])
        out[f"p_jinc_{tag}"] = float(fj.pvalues["income_z"])
        out[f"beta_jcrime_{tag}"] = float(fj.params[cz])
        out[f"p_jcrime_{tag}"] = float(fj.pvalues[cz])
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def scatter(df: pd.DataFrame, ycol: str, xcol: str, xlabel: str, title: str,
            path: Path, logx: bool = False) -> None:
    d = df.dropna(subset=[ycol, xcol])
    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.log1p(d[xcol]) if logx else d[xcol]
    ax.scatter(x, d[ycol], s=8, alpha=0.35, edgecolor="none")
    if len(d) >= 3:
        b1, b0 = np.polyfit(x, d[ycol], 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b0 + b1 * xs, color="crimson", lw=2)
        r = sps.pearsonr(x, d[ycol])[0]
        ax.set_title(f"{title}\nPearson r = {r:.3f}  (n={len(d)})", fontsize=10)
    ax.set_xlabel(("log1p " if logx else "") + xlabel)
    ax.set_ylabel(ycol)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def choropleth(tracts: gpd.GeoDataFrame, df: pd.DataFrame, col: str, title: str,
               path: Path) -> None:
    g = tracts.merge(df[["geoid", col]], on="geoid", how="left")
    fig, ax = plt.subplots(figsize=(7, 7))
    g.plot(column=col, cmap="viridis", legend=True, ax=ax,
           missing_kwds={"color": "lightgrey"}, linewidth=0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def run_layer(name: str, layer: dict, tracts: gpd.GeoDataFrame,
              income: pd.DataFrame, crime: pd.DataFrame, outdir: Path,
              min_comp: int, min_models: int) -> dict:
    print(f"\n===== LAYER {name}: {layer['title']} =====")
    models = discover_models(layer)
    print(f"  models with results: {list(models)}")

    # unique sample ids across all model parquets -> tract lookup (once)
    needed: set = set()
    for pq in models.values():
        s = pd.read_parquet(pq, columns=["repeat_idx", "sample_id_a", "sample_id_b"])
        s = s[s["repeat_idx"] == 0]
        needed |= set(s["sample_id_a"].astype(str)) | set(s["sample_id_b"].astype(str))
    print(f"  unique sample_ids: {len(needed)}")
    s2t = sampleid_to_tract(layer["manifest"], tracts, needed)
    print(f"  sample_ids assigned to a tract: {len(s2t)}")

    per_model = {nm: tract_ratings(pq, s2t) for nm, pq in models.items()}
    for nm, r in per_model.items():
        print(f"    {nm:12s} tracts={len(r):4d}  (>= {min_comp} cmp: "
              f"{(r['n_comparisons'] >= min_comp).sum()})")

    ens = normalized_ensemble(per_model, min_comp, min_models)
    print(f"  ensemble tracts (>= {min_models} models): {len(ens)}")

    cov = income.merge(crime, on="geoid", how="outer").merge(
        tracts[["geoid", "boroname", "ntaname", "area_km2"]], on="geoid", how="left"
    )
    if "population" not in cov.columns:
        cov = cov.merge(load_population(), on="geoid", how="left")

    # Build per-spec analysis frames and regress
    specs = {nm: per_model[nm][per_model[nm]["n_comparisons"] >= min_comp]
             for nm in per_model}
    specs["ensemble"] = ens

    results: Dict[str, dict] = {}
    analysis_rows = []
    for spec_name, rdf in specs.items():
        m = rdf.merge(cov, on="geoid", how="left")
        res = regress(m, "mu")
        results[spec_name] = res
        a = m[["geoid", "mu", "n_comparisons", "median_income", "population",
               "crime_density", "crime_per_capita", "felony_density",
               "boroname", "ntaname"]].copy()
        a.insert(0, "layer", name)
        a.insert(1, "spec", spec_name)
        analysis_rows.append(a)
        if "r2_joint_density" in res:
            print(f"    [{spec_name:12s}] n={int(res['n']):4d}  r(inc)={fmt(res['pearson_income'])}"
                  f"  DENSITY joint: bINC={fmt(res['beta_jinc_density'])} "
                  f"bCRIME={fmt(res['beta_jcrime_density'])}(p={fmt(res['p_jcrime_density'])}) "
                  f"R2={fmt(res['r2_joint_density'])}"
                  f"  | PERCAPITA joint: bINC={fmt(res['beta_jinc_percapita'])} "
                  f"bCRIME={fmt(res['beta_jcrime_percapita'])}(p={fmt(res['p_jcrime_percapita'])}) "
                  f"R2={fmt(res['r2_joint_percapita'])}")

    analysis = pd.concat(analysis_rows, ignore_index=True)
    apath = outdir / f"{name}_tract_analysis.parquet"
    analysis.to_parquet(apath)

    # Plots for the ensemble (headline) + covariate maps
    plots: List[Path] = []
    ens_m = ens.merge(cov, on="geoid", how="left")
    p1 = outdir / f"{name}_ensemble_vs_income.png"
    scatter(ens_m, "mu", "median_income", "median household income ($)",
            f"{layer['title']}: ensemble z-score vs income", p1)
    p2 = outdir / f"{name}_ensemble_vs_crime_density.png"
    scatter(ens_m, "mu", "crime_density", "severity-weighted crime / km^2",
            f"{layer['title']}: ensemble z-score vs crime density", p2, logx=True)
    p2b = outdir / f"{name}_ensemble_vs_crime_percapita.png"
    scatter(ens_m, "mu", "crime_per_capita", "severity-weighted crime / 1k residents",
            f"{layer['title']}: ensemble z-score vs crime per-capita", p2b, logx=True)
    p3 = outdir / f"{name}_choropleth_score.png"
    choropleth(tracts, ens, "mu", f"{layer['title']}: ensemble tract score", p3)
    plots += [p1, p2, p2b, p3]

    return {"results": results, "analysis_path": apath, "plots": plots,
            "n_models": len(models), "models": list(models)}


def write_report(layer_out: Dict[str, dict], outdir: Path) -> Path:
    lines = ["# Pairwise socioeconomic regression",
             f"_generated {datetime.now():%Y-%m-%d %H:%M}_", "",
             "Outcome: tract-level latent score (TrueSkill mu; ensemble = "
             "per-model z-scored mean). Predictors standardized; income = "
             "z(median household income); crime entered two ways for comparison "
             "— **density** = z(log1p(severity-weighted complaints / km^2)) and "
             "**per-capita** = z(log1p(severity-weighted complaints / 1k "
             "residents)). Severity weights: felony 3 / misdemeanor 2 / violation "
             "1. OLS with HC3 robust SEs. Each joint model is income + that crime "
             "metric.", ""]
    for name, info in layer_out.items():
        lines.append(f"## {LAYERS[name]['title']}  ({info['n_models']} models)")
        lines.append("")
        lines.append("Joint model = income + crime. β are standardized; **bold** "
                     "= p<0.05.")
        lines.append("")
        lines.append("| spec | n | r(inc) | β_inc | "
                     "β_crime ᴰᵉⁿˢ | R²ᴰᵉⁿˢ | β_crime ᴾᶜ | R²ᴾᶜ |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
        for spec, r in info["results"].items():
            if "r2_joint_density" not in r:
                continue

            def b(beta_key, p_key):
                v = r[beta_key]
                return f"**{v:.3f}**" if r[p_key] < 0.05 else f"{v:.3f}"
            lines.append(
                f"| {spec} | {int(r['n'])} | {r['pearson_income']:.3f} | "
                f"{b('beta_jinc_density','p_jinc_density')} | "
                f"{b('beta_jcrime_density','p_jcrime_density')} | "
                f"{r['r2_joint_density']:.3f} | "
                f"{b('beta_jcrime_percapita','p_jcrime_percapita')} | "
                f"{r['r2_joint_percapita']:.3f} |"
            )
        lines.append("")
    path = outdir / "REPORT.md"
    path.write_text("\n".join(lines))
    return path


def maybe_wandb(name: str, info: dict, report: Path, project: str,
                entity: Optional[str]) -> Optional[str]:
    from pairwise_analysis_common import mirror_to_wandb
    flat: Dict[str, float] = {}
    for spec, r in info["results"].items():
        for k, v in r.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                flat[f"{spec}/{k}"] = float(v)
    record = {
        "experiment_id": f"socioecon-{name}-{datetime.now():%Y%m%d_%H%M%S}",
        "mode": "tract_regression",
        "layer": name,
        "models": info["models"],
        "covariates": ["median_income", "severity_weighted_crime_density"],
        "results": flat,
    }
    results_df = pd.read_parquet(info["analysis_path"])
    return mirror_to_wandb(
        record=record,
        results=results_df,
        artifact_paths=[report, info["analysis_path"], *info["plots"]],
        project=project,
        entity=entity,
        run_label=f"socioecon_{name}",
        stage="socioecon_regression",
        extra_tags=[f"layer:{name}", "covariate:income", "covariate:crime"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=list(LAYERS),
                    choices=list(LAYERS))
    ap.add_argument("--min-comparisons", type=int, default=20)
    ap.add_argument("--min-models", type=int, default=3)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--project", default="URBANPAIRVQA")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (
        REPO / f"outputs/pairwise_socioeconomic/{datetime.now():%Y%m%d_%H%M%S}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"outdir: {outdir}")

    print("loading tracts / income / population / crime ...")
    tracts = load_tracts()
    income = load_income()
    population = load_population()
    crime = load_crime(tracts, population)
    print(f"  tracts={len(tracts)}  income_tracts={income['median_income'].notna().sum()}"
          f"  pop_tracts={population['population'].gt(0).sum()}  crime_tracts={len(crime)}"
          f"  per_capita_defined={crime['crime_per_capita'].notna().sum()}")

    layer_out: Dict[str, dict] = {}
    for name in args.layers:
        layer_out[name] = run_layer(name, LAYERS[name], tracts, income, crime,
                                    outdir, args.min_comparisons, args.min_models)

    report = write_report(layer_out, outdir)
    print(f"\nreport: {report}")

    if not args.no_wandb:
        for name, info in layer_out.items():
            url = maybe_wandb(name, info, report, args.project, args.entity)
            print(f"  [wandb {name}] {url}")

    print("\nDONE")


if __name__ == "__main__":
    main()
