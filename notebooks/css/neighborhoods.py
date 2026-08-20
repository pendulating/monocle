"""Neighborhood (NTA) aggregation of pairwise street-view rankings.

Marimo notebook that turns the 100k-pair urbanpairvqa outputs (restaurants:
"which would you rather eat at"; schools: "which would you rather send your
child to") into per-neighborhood rankings over NYC's 2020 Neighborhood
Tabulation Areas (NTAs).

Pipeline:
    1. Join the pairwise output parquet (relative_score, already de-swapped to
       the canonical A-vs-B comparison) with its sibling ``pairs.parquet``
       (unit_uid, unit_name, lat/lon per side).
    2. Locate each unit (one point = mean of its recording lat/lon) and assign
       it to a containing NTA via point-in-polygon (EPSG:2263).
    3. Fit unit-level TrueSkill (canonical recipe: relative_score>0 -> A wins,
       <0 -> B wins, 0 -> draw; magnitude discarded).
    4. Produce TWO NTA-level rankings, shown side by side:
         (A) Direct zone TrueSkill -- relabel every comparison by each side's
             NTA, drop within-NTA pairs, fit TrueSkill on NTA-vs-NTA.
         (B) Mean of unit ratings -- average unit mu within each NTA.
    5. Choropleths, a method-agreement scatter, a borough breakdown, and
       persisted per-NTA score tables.

Run with:
    marimo edit notebooks/css/neighborhoods.py
"""

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _intro():
    import marimo as mo

    mo.md(
        """
        # Pairwise rankings → NYC neighborhoods (NTA 2020)

        Aggregates the 100k-pair **urbanpairvqa** comparisons to the
        **Neighborhood Tabulation Area** level. The 5-label ordinal answers are
        collapsed to win/loss/draw and fit with **TrueSkill** (the canonical
        `wealth.ipynb` recipe).

        Two NTA-level rankings are shown **side by side**:

        - **(A) Direct zone TrueSkill** — each comparison is relabelled by the
          NTA of each side; TrueSkill is fit directly on NTA-vs-NTA matches.
        - **(B) Mean of unit ratings** — unit-level TrueSkill, then average
          `mu` within each NTA.

        Use the controls below to switch dataset and the minimum-comparisons
        reliability filter.
        """
    )
    return (mo,)


@app.cell
def _imports():
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import trueskill
    import matplotlib.pyplot as plt

    return Path, defaultdict, gpd, pd, plt, trueskill


@app.cell
def _config(Path):
    REPO = Path("/share/pierson/matt/mllmsci")

    # Each dataset: the consolidated pairwise output parquet + its sibling
    # pairs.parquet (geo + unit identity) live in the same run directory.
    DATASETS = {
        "restaurants": REPO / "multirun/2026-06-08_URBANPAIRVQA/12-21-06/0/outputs/pairwise/restaurants_mvp_20260608_122106.parquet",
        "schools": REPO / "multirun/2026-06-08_URBANPAIRVQA/12-21-15/0/outputs/pairwise/schools_mvp_20260608_122123.parquet",
    }

    NTA_SHP = REPO / "data/geo/nynta2020_26b/nynta2020.shp"

    # TrueSkill draw probability — matches the canonical wealth.ipynb (≈ the
    # observed "Same" rate). NTA shapefile is in NY State Plane (feet).
    DRAW_PROB = 0.05
    NTA_CRS = 2263

    OUT_DIR = REPO / "notebooks/css/results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return DATASETS, DRAW_PROB, NTA_CRS, NTA_SHP, OUT_DIR


@app.cell
def _helpers(DRAW_PROB, NTA_CRS, defaultdict, gpd, pd, trueskill):
    def load_merged(output_path):
        """Join pairwise output (scores) with sibling pairs.parquet (geo/ids)."""
        out = pd.read_parquet(output_path)
        pairs_path = output_path.parent / "pairs.parquet"
        pairs = pd.read_parquet(pairs_path)
        geo_cols = [
            "pair_id", "unit_uid_a", "unit_uid_b", "unit_name_a", "unit_name_b",
            "latitude_a", "longitude_a", "latitude_b", "longitude_b",
        ]
        geo_cols = [c for c in geo_cols if c in pairs.columns]
        merged = out.merge(pairs[geo_cols], on="pair_id", how="inner")
        return merged

    def build_unit_locations(merged):
        """One point per unit_uid = mean of its recording lat/lon (across all
        rows where it appears as side A or B)."""
        a = merged[["unit_uid_a", "unit_name_a", "latitude_a", "longitude_a"]].rename(
            columns={"unit_uid_a": "unit_uid", "unit_name_a": "unit_name",
                     "latitude_a": "lat", "longitude_a": "lon"})
        b = merged[["unit_uid_b", "unit_name_b", "latitude_b", "longitude_b"]].rename(
            columns={"unit_uid_b": "unit_uid", "unit_name_b": "unit_name",
                     "latitude_b": "lat", "longitude_b": "lon"})
        both = pd.concat([a, b], ignore_index=True).dropna(subset=["unit_uid"])
        units = both.groupby("unit_uid").agg(
            unit_name=("unit_name", "first"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        ).reset_index()
        return units

    def assign_nta(units, nta_gdf):
        """Point-in-polygon: unit -> containing NTA (NTA2020/NTAName/BoroName)."""
        u = units.dropna(subset=["lat", "lon"]).copy()
        pts = gpd.GeoDataFrame(
            u,
            geometry=gpd.points_from_xy(u["lon"], u["lat"]),
            crs=4326,
        ).to_crs(NTA_CRS)
        keep = nta_gdf[["NTA2020", "NTAName", "BoroName", "geometry"]]
        joined = gpd.sjoin(pts, keep, how="left", predicate="within")
        cols = ["unit_uid", "unit_name", "lat", "lon", "NTA2020", "NTAName", "BoroName"]
        return joined[cols].drop_duplicates(subset="unit_uid")

    def _fit_trueskill(matches_a, matches_b, scores):
        """Online 1v1 TrueSkill over canonical A-vs-B scores. Returns dict
        entity -> Rating and a comparison-count Series."""
        env = trueskill.TrueSkill(draw_probability=DRAW_PROB)
        ratings = defaultdict(env.create_rating)
        for a, b, score in zip(matches_a, matches_b, scores):
            if a == b:
                continue
            ra, rb = ratings[a], ratings[b]
            if score > 0:        # A wins
                ra, rb = env.rate_1vs1(ra, rb, drawn=False)
            elif score < 0:      # B wins (rate_1vs1 is winner-first)
                rb, ra = env.rate_1vs1(rb, ra, drawn=False)
            else:                # draw
                ra, rb = env.rate_1vs1(ra, rb, drawn=True)
            ratings[a], ratings[b] = ra, rb
        counts = pd.Series(list(matches_a) + list(matches_b)).value_counts()
        return ratings, counts

    def fit_unit_trueskill(merged):
        """Per-unit ratings from the raw comparisons."""
        work = merged.dropna(subset=["unit_uid_a", "unit_uid_b", "relative_score"]).copy()
        work["unit_uid_a"] = work["unit_uid_a"].astype(str)
        work["unit_uid_b"] = work["unit_uid_b"].astype(str)
        ratings, counts = _fit_trueskill(
            work["unit_uid_a"].to_numpy(),
            work["unit_uid_b"].to_numpy(),
            work["relative_score"].astype(int).to_numpy(),
        )
        rows = [{
            "unit_uid": uid, "mu": r.mu, "sigma": r.sigma,
            "ts_conservative": r.mu - 3.0 * r.sigma,
            "n_comparisons": int(counts.get(uid, 0)),
        } for uid, r in ratings.items()]
        return pd.DataFrame(rows)

    def aggregate_nta_mean(unit_scores_nta, min_unit_comparisons=1):
        """Method (B): average unit mu within each NTA."""
        s = unit_scores_nta[unit_scores_nta["n_comparisons"] >= min_unit_comparisons]
        s = s.dropna(subset=["NTA2020"])
        agg = s.groupby(["NTA2020", "NTAName", "BoroName"]).agg(
            mu=("mu", "mean"),
            mu_std=("mu", "std"),
            n_units=("unit_uid", "nunique"),
            n_comparisons=("n_comparisons", "sum"),
        ).reset_index()
        return agg.sort_values("mu", ascending=False).reset_index(drop=True)

    def fit_nta_zone_trueskill(merged, unit_to_nta):
        """Method (A): relabel each comparison by NTA, drop within-NTA pairs,
        fit TrueSkill on NTA-vs-NTA matches."""
        u2n = unit_to_nta.set_index("unit_uid")["NTA2020"].to_dict()
        work = merged.dropna(subset=["unit_uid_a", "unit_uid_b", "relative_score"]).copy()
        work["nta_a"] = work["unit_uid_a"].astype(str).map(u2n)
        work["nta_b"] = work["unit_uid_b"].astype(str).map(u2n)
        work = work.dropna(subset=["nta_a", "nta_b"])
        work = work[work["nta_a"] != work["nta_b"]]
        ratings, counts = _fit_trueskill(
            work["nta_a"].to_numpy(),
            work["nta_b"].to_numpy(),
            work["relative_score"].astype(int).to_numpy(),
        )
        rows = [{
            "NTA2020": nta, "mu": r.mu, "sigma": r.sigma,
            "ts_conservative": r.mu - 3.0 * r.sigma,
            "n_comparisons": int(counts.get(nta, 0)),
        } for nta, r in ratings.items()]
        out = pd.DataFrame(rows)
        meta = unit_to_nta[["NTA2020", "NTAName", "BoroName"]].drop_duplicates("NTA2020")
        out = out.merge(meta, on="NTA2020", how="left")
        return out.sort_values("mu", ascending=False).reset_index(drop=True)

    return (
        aggregate_nta_mean,
        assign_nta,
        build_unit_locations,
        fit_nta_zone_trueskill,
        fit_unit_trueskill,
        load_merged,
    )


@app.cell
def _load_nta(NTA_SHP, gpd):
    nta_gdf = gpd.read_file(NTA_SHP)
    # Drop non-residential pseudo-NTAs (parks, cemeteries, airports: NTAType != 0)
    # only for ranking; keep full geometry for the map backdrop.
    nta_gdf_full = nta_gdf.copy()
    return nta_gdf, nta_gdf_full


@app.cell
def _controls(DATASETS, mo):
    dataset_dd = mo.ui.dropdown(
        options=list(DATASETS.keys()), value="restaurants", label="Dataset")
    min_comp = mo.ui.slider(
        start=1, stop=300, value=30, step=1,
        label="Min comparisons per NTA (reliability filter)", show_value=True)
    mo.vstack([dataset_dd, min_comp])
    return dataset_dd, min_comp


@app.cell
def _compute(
    DATASETS,
    aggregate_nta_mean,
    assign_nta,
    build_unit_locations,
    dataset_dd,
    fit_nta_zone_trueskill,
    fit_unit_trueskill,
    load_merged,
    nta_gdf,
):
    _path = DATASETS[dataset_dd.value]
    merged = load_merged(_path)
    units = build_unit_locations(merged)
    unit_to_nta = assign_nta(units, nta_gdf)

    unit_scores = fit_unit_trueskill(merged)
    unit_scores_nta = unit_scores.merge(unit_to_nta, on="unit_uid", how="left")

    nta_zone = fit_nta_zone_trueskill(merged, unit_to_nta)      # method A
    nta_mean = aggregate_nta_mean(unit_scores_nta)              # method B
    return merged, nta_mean, nta_zone, unit_scores_nta, unit_to_nta


@app.cell
def _summary(dataset_dd, merged, mo, nta_zone, unit_scores_nta, unit_to_nta):
    n_assigned = unit_to_nta["NTA2020"].notna().sum()
    mo.md(
        f"""
        ### `{dataset_dd.value}` — {len(merged):,} comparisons
        - **{unit_to_nta['unit_uid'].nunique():,}** units; **{n_assigned:,}**
          assigned to an NTA ({n_assigned / max(len(unit_to_nta), 1):.1%}).
        - **{nta_zone['NTA2020'].nunique()}** NTAs with ≥1 cross-NTA comparison.
        - Unit-level ratings computed for **{len(unit_scores_nta):,}** units.
        """
    )
    return


@app.cell
def _tables(min_comp, mo, nta_mean, nta_zone):
    thr_t = min_comp.value
    zone = nta_zone[nta_zone["n_comparisons"] >= thr_t].copy()
    mean = nta_mean[nta_mean["n_comparisons"] >= thr_t].copy()

    def _fmt(df, score_col):
        d = df[["NTAName", "BoroName", score_col, "n_comparisons"]].copy()
        d[score_col] = d[score_col].round(3)
        d.insert(0, "rank", range(1, len(d) + 1))
        return d

    left = mo.vstack([
        mo.md(f"#### (A) Direct zone TrueSkill — top/bottom ({len(zone)} NTAs)"),
        mo.ui.table(_fmt(zone, "mu").head(15), selection=None),
        mo.md("…"),
        mo.ui.table(_fmt(zone, "mu").tail(10), selection=None),
    ])
    right = mo.vstack([
        mo.md(f"#### (B) Mean of unit μ — top/bottom ({len(mean)} NTAs)"),
        mo.ui.table(_fmt(mean, "mu").head(15), selection=None),
        mo.md("…"),
        mo.ui.table(_fmt(mean, "mu").tail(10), selection=None),
    ])
    mo.hstack([left, right], widths=[1, 1])
    return


@app.cell
def _maps(dataset_dd, min_comp, nta_gdf_full, nta_mean, nta_zone, plt):
    thr_m = min_comp.value

    def _choropleth(ax, nta_scores, title):
        score = nta_scores[nta_scores["n_comparisons"] >= thr_m][["NTA2020", "mu"]]
        g = nta_gdf_full.merge(score, on="NTA2020", how="left")
        # Backdrop: all NTAs in light gray (incl. below-threshold / unscored).
        g.plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.2)
        g.dropna(subset=["mu"]).plot(
            ax=ax, column="mu", cmap="RdYlBu", legend=True,
            edgecolor="white", linewidth=0.2,
            legend_kwds={"shrink": 0.5, "label": "TrueSkill μ"})
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

    fig_m, axes = plt.subplots(1, 2, figsize=(15, 8))
    _choropleth(axes[0], nta_zone, f"{dataset_dd.value} · (A) zone TrueSkill")
    _choropleth(axes[1], nta_mean, f"{dataset_dd.value} · (B) mean of unit μ")
    fig_m.suptitle(
        f"NTA ranking · {dataset_dd.value} · ≥{thr_m} comparisons", fontsize=13)
    fig_m.tight_layout()
    fig_m
    return


@app.cell
def _agreement(dataset_dd, min_comp, nta_mean, nta_zone, plt):
    thr_a = min_comp.value
    za = nta_zone[nta_zone["n_comparisons"] >= thr_a][["NTA2020", "NTAName", "mu"]]
    mb = nta_mean[nta_mean["n_comparisons"] >= thr_a][["NTA2020", "mu"]]
    cmp = za.merge(mb, on="NTA2020", suffixes=("_zone", "_mean")).dropna()

    fig_a, ax = plt.subplots(figsize=(6, 6))
    if len(cmp) >= 2:
        # Scales differ between methods; compare via rank correlation.
        rho = cmp["mu_zone"].rank().corr(cmp["mu_mean"].rank())
        ax.scatter(cmp["mu_zone"], cmp["mu_mean"], s=14, alpha=0.6)
        ax.set_xlabel("(A) zone TrueSkill μ")
        ax.set_ylabel("(B) mean of unit μ")
        ax.set_title(
            f"{dataset_dd.value}: method agreement\n"
            f"Spearman ρ = {rho:.3f}  (n = {len(cmp)} NTAs)")
    else:
        ax.text(0.5, 0.5, "Not enough NTAs above threshold", ha="center")
    fig_a.tight_layout()
    fig_a
    return


@app.cell
def _borough(dataset_dd, min_comp, mo, nta_zone):
    thr_b = min_comp.value
    z = nta_zone[nta_zone["n_comparisons"] >= thr_b]
    boro = z.groupby("BoroName").agg(
        mean_mu=("mu", "mean"),
        n_ntas=("NTA2020", "nunique"),
        n_comparisons=("n_comparisons", "sum"),
    ).reset_index().sort_values("mean_mu", ascending=False)
    boro["mean_mu"] = boro["mean_mu"].round(3)
    mo.vstack([
        mo.md(f"#### Borough breakdown — (A) zone TrueSkill · `{dataset_dd.value}`"),
        mo.ui.table(boro, selection=None),
    ])
    return


@app.cell
def _persist(OUT_DIR, dataset_dd, mo, nta_mean, nta_zone, unit_scores_nta):
    tag = dataset_dd.value
    p_zone = OUT_DIR / f"nta_{tag}_zone_trueskill.parquet"
    p_mean = OUT_DIR / f"nta_{tag}_mean_of_units.parquet"
    p_unit = OUT_DIR / f"unit_{tag}_trueskill.parquet"
    nta_zone.to_parquet(p_zone, index=False)
    nta_mean.to_parquet(p_mean, index=False)
    unit_scores_nta.to_parquet(p_unit, index=False)
    mo.md(
        f"""
        ### Persisted (`{tag}`)
        - `{p_zone.name}` — {len(nta_zone)} NTAs (method A)
        - `{p_mean.name}` — {len(nta_mean)} NTAs (method B)
        - `{p_unit.name}` — {len(unit_scores_nta):,} units
        ↳ `{OUT_DIR}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
