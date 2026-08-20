"""Validation via statistical proxy — the subway safety case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
urbanpairvqa run of the subway prompt, scores each station entrance, and
compares the scores against 2 proxies in 3 NYC geographies:

- median household income (ACS 5-Year 2024, table S1901)
- the NYPD complaint record, severity weighted

Both proxies describe the AREA around an entrance, not the entrance itself.
Read the warning in section 4 before you read a number.

The notebook discovers its runs from W&B. It does not name run directories, so
it picks up a model as soon as that model finishes.

Run it:
    .venv-nightly/bin/marimo edit notebooks/cvpr/subway_safety_validation.py

Warning: use `.venv-nightly`. It has marimo and geopandas.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    # The shared modules sit in the parent, `notebooks/cvpr/`, because every
    # prompt folder uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import _geography as geo
    import _provenance as prov
    import _proxies as px

    return Path, geo, mo, pd, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed.
    NOTEBOOK_VERSION = "1.0.0"
    CASE = "subway"
    CURATION_ROOT = "curation/subway_entrances_all"
    # The subway curation writes `entrances.parquet`, not `facilities.parquet`.
    UNIT_TABLE = "entrances.parquet"
    # No filter. 1 keeps every polygon that holds an entrance.
    MIN_UNITS_PER_POLYGON = 1
    return (
        CASE,
        CURATION_ROOT,
        MIN_UNITS_PER_POLYGON,
        NOTEBOOK_VERSION,
        UNIT_TABLE,
    )


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, prov, px):
    mo.md(
        f"""
    # Subway safety — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__}

    The model answers "which station entrance is safer". The proxies are median
    household income and the NYPD complaint record.
    """
    )
    return


@app.cell
def _(CASE, mo, prov):
    mo.md("## 1. Run discovery")

    # Warning: this cell reads the network. It runs one time.
    records = prov.discover_runs(CASE, only_finished=False, only_canonical=True)
    usable = [r for r in records if r.is_usable and r.state == "finished"]
    pending = [r for r in records if r not in usable]
    return pending, records, usable


@app.cell
def _(mo, pending, prov, records, usable):
    table = prov.provenance_table(records)
    mo.md(
        f"""
    ### Provenance

    - **{len(usable)}** run(s) are finished and readable.
    - **{len(pending)}** run(s) are not ready yet. This notebook adds them when
      they finish. Do not edit the notebook for that.
    """
    )
    return (table,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(mo, prov, usable):
    mo.md("## 2. Unit scores")

    per_model = {}
    for _rec in usable:
        per_model[_rec.model] = prov.score_units(prov.load_run(_rec))
    return (per_model,)


@app.cell
def _(mo, pd, per_model):
    if per_model:
        summary = pd.DataFrame(
            [
                {
                    "model": m,
                    "entrances_scored": len(s),
                    "mean_score": round(s.mean_score.mean(), 4),
                    "mean_abstention": round(s.abstention_rate.mean(), 4),
                    "median_comparisons": int(s.n_comparisons.median()),
                }
                for m, s in per_model.items()
            ]
        )
    else:
        summary = pd.DataFrame()

    mo.md(
        """
    ### Warning: read the abstention rate before you trust a model

    The subway case drew a high abstention rate in the persona probe. A model
    that declines most pairs gives each entrance few comparisons, so its mean
    is weak evidence.
    """
    )
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.cell
def _(CURATION_ROOT, UNIT_TABLE, geo, mo):
    mo.md("## 3. Geographic aggregation")

    units_gdf = geo.load_units(CURATION_ROOT, filename=UNIT_TABLE)
    return (units_gdf,)


@app.cell
def _(MIN_UNITS_PER_POLYGON, geo, pd, per_model, units_gdf):
    _frames = []
    for _model, _scores in per_model.items():
        for _layer in geo.LAYERS:
            _out = geo.aggregate(
                _scores, units_gdf, _layer, min_units=MIN_UNITS_PER_POLYGON
            )
            if not _out.empty:
                _out.insert(0, "model", _model)
                _frames.append(_out)
    aggregated = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return (aggregated,)


@app.cell
def _(CASE, mo, px):
    mo.md("## 4. The 2 proxies")

    recipe = px.load_recipe(CASE)
    mo.md(
        f"""
    `{CASE}/recipe.json` names {len(recipe)} sources. All of them sit on disk,
    so this notebook makes no network call for a proxy.

    ### Warning: both proxies describe the area, not the entrance

    Neither proxy measures a station entrance. Income describes the households
    around it. The complaint record describes events near it. Thus a correlation
    supports only this claim: the look of an entrance tracks the wealth or the
    crime of its surroundings. It cannot say the model reads danger at the
    entrance.

    ### The 2 proxies do not aggregate the same way

    | Proxy | Tract | NTA and community district |
    |-------|-------|----------------------------|
    | Income | True ACS median | **Population-weighted mean of tract medians** |
    | Crime | True value | **True value** |

    Crime is point data, so a count and an area aggregate exactly. Income is a
    median, and you cannot average medians. Name the income value at NTA and
    community district a population-weighted mean, never a median.

    ### Orientation

    Both proxies are oriented so that higher is better. Income rises with
    wealth. Crime carries a sign of -1, so a low crime count gives a high value.
    A **positive** correlation therefore always means the model agrees with the
    proxy. Do not flip a sign again.
    """
    )
    return (recipe,)


@app.cell
def _(mo, px):
    crime_metric_ui = mo.ui.dropdown(
        options=list(px.CRIME_METRICS),
        value="crime_density",
        label="Crime metric",
    )
    crime_metric_ui
    return (crime_metric_ui,)


@app.cell
def _(crime_metric_ui, geo, pd, px):
    _frames = []
    for _layer in geo.LAYERS:
        _inc = px.income_by_layer(_layer)
        if not _inc.empty:
            _inc = _inc.copy()
            _inc["proxy"] = "median_household_income"
            _frames.append(_inc)
        _cri = px.crime_by_layer(_layer, metric=crime_metric_ui.value)
        if not _cri.empty:
            _cri = _cri.copy()
            _cri["proxy"] = crime_metric_ui.value
            _frames.append(_cri)
    proxy_agg = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return (proxy_agg,)


@app.cell
def _(proxy_agg):
    proxy_agg.groupby(["proxy", "layer"]).agg(
        polygons=("layer", "size"),
        mean_value=("proxy_mean", "mean"),
    ).reset_index()
    return


@app.cell
def _(aggregated, geo, mo, pd, proxy_agg, px):
    mo.md("## 5. Model score against each proxy")

    _rows = []
    if not aggregated.empty and not proxy_agg.empty:
        for _model in aggregated.model.unique():
            _m = aggregated[aggregated.model == _model]
            for _proxy in proxy_agg["proxy"].unique():
                _p = proxy_agg[proxy_agg["proxy"] == _proxy]
                for _layer in geo.LAYERS:
                    _r = px.correlate(_m, _p, _layer)
                    _r["model"] = _model
                    _r["proxy"] = _proxy
                    _rows.append(_r)
    correlations = pd.DataFrame(_rows)
    return (correlations,)


@app.cell
def _(correlations, mo):
    mo.md(
        """
    Read the Spearman value first, and read `n` beside it.

    A positive value against income means the model calls an entrance safer in a
    wealthier area. A positive value against crime means the model calls an
    entrance safer where the complaint record is thinner.
    """
    )
    correlations
    return


@app.cell
def _(mo):
    mo.md(
        """
## 6. Export

The next cell writes the tables to `notebooks/cvpr/outputs/`. Give the paper
these files together. The numbers mean nothing without the provenance.
"""
    )
    return


@app.cell
def _(CASE, NOTEBOOK_VERSION, Path, aggregated, correlations, mo, proxy_agg, table):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    written = []
    for _name, _frame in [
        ("aggregated", aggregated),
        ("proxy", proxy_agg),
        ("correlations", correlations),
    ]:
        if _frame is not None and not _frame.empty:
            _p = out_dir / f"{CASE}_{_name}_v{NOTEBOOK_VERSION}.parquet"
            _frame.to_parquet(_p, index=False)
            written.append(str(_p))
    if not table.empty:
        _p = out_dir / f"{CASE}_provenance_v{NOTEBOOK_VERSION}.csv"
        table.to_csv(_p, index=False)
        written.append(str(_p))

    mo.md(
        "**Written:**\n\n" + "\n".join(f"- `{w}`" for w in written)
        if written
        else "**Nothing to write yet.** No finished run exists."
    )
    return


if __name__ == "__main__":
    app.run()
