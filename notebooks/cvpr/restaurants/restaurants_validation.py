"""Validation via statistical proxy — the restaurants case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
urbanpairvqa run of the restaurants prompt, scores each restaurant, and compares
the scores against the DOHMH inspection record in 3 NYC geographies.

The notebook discovers its runs from W&B. It does not name run directories, so
it picks up a model as soon as that model finishes.

This case is simpler than the schools case. The curation tooling in
`dagspaces/common/curation/dohmh` already wrote `restaurants_aggregated.parquet`,
which holds the inspection history AND the position. Thus the notebook makes no
network call for the proxy, and it needs no name join: `uid` matches `unit_uid`
in the pairs manifest exactly.

Run it:
    .venv-nightly/bin/marimo edit notebooks/cvpr/restaurants_validation.py

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
    # Raise NOTEBOOK_VERSION when you change how a number is computed. The
    # export file names carry it.
    NOTEBOOK_VERSION = "1.0.0"
    CASE = "restaurants"
    CURATION_ROOT = "curation/dohmh_restaurants_inspected_all"
    UNIT_TABLE = "restaurants_aggregated.parquet"
    MIN_UNITS_PER_POLYGON = 3
    MIN_INSPECTIONS = 2
    return (
        CASE,
        CURATION_ROOT,
        MIN_INSPECTIONS,
        MIN_UNITS_PER_POLYGON,
        NOTEBOOK_VERSION,
        UNIT_TABLE,
    )


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, prov, px):
    mo.md(
        f"""
    # Restaurants — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__}

    The model answers "which restaurant would you rather eat at". The proxy is
    the DOHMH inspection record.
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

    # `score_units` drops an abstention instead of scoring it 0.
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
                    "restaurants_scored": len(s),
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

    A high abstention rate means the model declined most pairs, so each score
    rests on few comparisons.
    """
    )
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.cell
def _(CURATION_ROOT, UNIT_TABLE, geo, mo):
    mo.md("## 3. Units and the proxy")

    # 1 table gives the position AND the proxy. The DOHMH curation tooling made
    # it, so the notebook makes no network call here.
    units = geo.load_facilities(CURATION_ROOT, filename=UNIT_TABLE)
    units_gdf = geo.load_units(CURATION_ROOT, filename=UNIT_TABLE)
    return units, units_gdf


@app.cell
def _(mo, px):
    metric_ui = mo.ui.dropdown(
        options=list(px.RESTAURANT_METRICS),
        value="inspection_score",
        label="Proxy metric",
    )
    metric_ui
    return (metric_ui,)


@app.cell
def _(MIN_INSPECTIONS, metric_ui, mo, px, units):
    proxy_units = px.restaurant_proxy(
        units, metric=metric_ui.value, min_inspections=MIN_INSPECTIONS
    )
    proxy_points = px.points_from_latlon(proxy_units)
    _spec = px.RESTAURANT_METRICS[metric_ui.value]

    mo.md(
        f"""
    ### {_spec['label']}

    {_spec['note']}

    | Quantity | Value |
    |---|---|
    | Restaurants in the unit table | {len(units)} |
    | With a proxy value and {MIN_INSPECTIONS}+ inspections | {len(proxy_units)} |
    | Placed on the map | {len(proxy_points)} |

    **Warning: the raw DOHMH score counts violation points, so a LOW raw score
    is a CLEAN restaurant.** The loader multiplies it by
    {_spec['sign']}, so the value here is oriented as "higher is better". A
    POSITIVE correlation below therefore always means the model and the
    inspector agree. Do not flip the sign again.
    """
    )
    return proxy_points, proxy_units


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
def _(MIN_UNITS_PER_POLYGON, geo, pd, proxy_points, px):
    _frames = []
    for _layer in geo.LAYERS:
        _out = px.aggregate_proxy(
            proxy_points, _layer, min_units=MIN_UNITS_PER_POLYGON
        )
        if not _out.empty:
            _frames.append(_out)
    proxy_agg = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return (proxy_agg,)


@app.cell
def _(aggregated, geo, metric_ui, mo, pd, proxy_agg, px):
    mo.md("## 4. Model score against the proxy")

    _rows = []
    if not aggregated.empty and not proxy_agg.empty:
        for _model in aggregated.model.unique():
            _m = aggregated[aggregated.model == _model]
            for _layer in geo.LAYERS:
                _r = px.correlate(_m, proxy_agg, _layer)
                _r["model"] = _model
                _r["metric"] = metric_ui.value
                _rows.append(_r)
    correlations = pd.DataFrame(_rows)
    return (correlations,)


@app.cell
def _(correlations, mo):
    mo.md(
        """
    Read the Spearman value first. The model score is an ordinal mean, so a
    monotone agreement matters more than a linear one.

    **Warning:** a correlation here is not a causal claim. The model sees a
    storefront. The inspector measures food safety inside. A weak value can mean
    that the facade carries little signal about the kitchen, which is itself a
    result worth reporting.
    """
    )
    correlations
    return


@app.cell
def _(mo):
    mo.md(
        """
## 5. Export

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
