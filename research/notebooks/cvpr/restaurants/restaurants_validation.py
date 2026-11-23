"""Validation via statistical proxy — the restaurants case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
pairwisevqa run of the restaurants prompt, scores each restaurant, and compares
the scores against the DOHMH inspection record in 3 NYC geographies.

The notebook discovers its runs from W&B. It does not name run directories, so
it picks up a model as soon as that model finishes. The UNIT section reads the
canonical registry instead, through `_unit_validation.place_scores`.

This case is simpler than the schools case. The curation tooling in
`sv/nyc/curation/dohmh` already wrote `restaurants_aggregated.parquet`
and the `restaurants_inspection_history.parquet` sidecar beside it. Both hold
the inspection record AND the position. Thus the notebook makes no network call
for a proxy, and it needs no name join: `uid` matches `unit_uid` in the pairs
manifest exactly.

What changed at version 1.2.0
-----------------------------
The headline proxy is no longer the single most recent inspection. It is the
MEAN score over the cycle initial inspections of each restaurant. The export
now carries 4 proxies at once, and the outdoor-dining channel joins them as a
CONFOUNDER of the preference judgment. See
`research/docs/restaurant-proxies.md`.

Run it:
    .venv-nightly/bin/marimo edit research/notebooks/cvpr/restaurants/restaurants_validation.py

Warning: use `.venv-nightly`. It has marimo and geopandas.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _provenance as prov
    from research.notebooks.cvpr import _proxies as px
    from research.notebooks.cvpr import _proxies_restaurants as pr

    return Path, geo, mo, pd, pr, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed. The
    # export file names carry it.
    #
    # 1.1.0: the proxy export carries a `proxy` column and the 3 provenance
    # columns. Version 1.0.0 wrote no proxy name at all, thus a reader could
    # not tell an inspection score from a grade rate.
    #
    # 1.2.0: the export carries ALL 4 proxies, not the 1 that a dropdown
    # picked. The headline is `inspection_score_history`, the mean score over
    # the cycle initial inspections. `inspection_score`, the score of the last
    # inspection, stays as context and its rows do not change.
    NOTEBOOK_VERSION = "1.2.0"

    # The model-score export did NOT change at 1.2.0. Its version stays at
    # 1.0.0, and the export cell writes it only when the file is absent.
    # Warning: never overwrite that file. The paper table reads it.
    MODEL_EXPORT_VERSION = "1.0.0"

    CASE = "restaurants"
    MIN_UNITS_PER_POLYGON = 3
    return CASE, MIN_UNITS_PER_POLYGON, MODEL_EXPORT_VERSION, NOTEBOOK_VERSION


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, pr, prov, px):
    mo.md(
        f"""
    # Restaurants — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__} ·
    restaurant proxies {pr.__version__}

    The model answers "which restaurant would you rather eat at". The proxies
    are the DOHMH inspection record and the outdoor-dining license record.

    **Warning: the prompt asks for a PREFERENCE, and the inspection record
    measures FOOD SAFETY.** The 2 constructs are not the same thing. Read
    `research/docs/restaurant-proxies.md` before you call a low
    correlation a failure of the model.
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
def _(CASE, geo, mo, pr, px):
    mo.md("## 3. Units and the proxies")

    # 2 curation files give the position AND every proxy. The notebook makes no
    # network call here.
    units = pr.load_units()
    units_gdf = geo.load_units(pr.CURATION_ROOT, filename=pr.UNIT_TABLE)

    # `recipe.json` names each source. The export copies the id and the window
    # of the source that each metric reads, so a reader of the parquet never
    # has to open this notebook.
    recipe = px.load_recipe(CASE)
    return recipe, units, units_gdf


@app.cell
def _(mo):
    mo.md(
        """
    ### The provenance contract

    Every row of the proxy export carries 4 columns:

    | Column | Holds |
    |--------|-------|
    | `proxy` | The metric name. For this case it IS the metric. |
    | `proxy_metric` | The same name, for the readers that expect it |
    | `proxy_source` | The dataset id, or the file stem, from `recipe.json` |
    | `proxy_window` | The time window of that source |

    **Warning: never write a fixed proxy name beside a metric that a reader
    chooses.** A fixed name lets a run export one metric under the name of
    another. `_proxies_restaurants.stamp` derives the name from the metric.
    """
    )
    return


@app.cell
def _(mo, pr):
    metric_ui = mo.ui.dropdown(
        options=list(pr.METRIC_ORDER),
        value="inspection_score_history",
        label="Proxy metric to describe below",
    )

    # Warning: the dropdown drives the DESCRIPTION only. Since version 1.2.0
    # the export writes every metric of `pr.METRIC_ORDER` in one pass, thus a
    # reader of the parquet never depends on what this dropdown held.
    metric_ui
    return (metric_ui,)


@app.cell
def _(metric_ui, mo, pr, units):
    proxy_units = pr.restaurant_proxy(metric_ui.value, units=units)
    _spec = pr.METRICS[metric_ui.value]

    mo.md(
        f"""
    ### {_spec['label']} — role: {_spec['role']}

    {_spec['note']}

    | Quantity | Value |
    |---|---|
    | Restaurants in the unit table | {len(units)} |
    | With a value for this metric | {len(proxy_units)} |
    | Window | {pr.METRIC_WINDOW[metric_ui.value]} |

    **Warning: the raw DOHMH score counts violation points, so a LOW raw score
    is a CLEAN restaurant.** `_proxies_restaurants` multiplies it by
    {_spec['sign']}, so the value here is oriented as "higher is better". A
    POSITIVE correlation below therefore always means the model and the record
    agree. Do not flip the sign again.
    """
    )
    return (proxy_units,)


@app.cell
def _(mo, pr):
    mo.md(
        """
    ### What each metric keeps

    The history mean costs coverage: it reads 1 inspection type and it needs 2
    such inspections. The outdoor-dining flag keeps every restaurant, because
    the 0 rows carry the information and an area mean over them IS the share of
    the restaurants of that area.
    """
    )
    coverage = pr.coverage_table()
    return (coverage,)


@app.cell
def _(coverage):
    coverage
    return


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
def _(MIN_UNITS_PER_POLYGON, pr, recipe, units):
    # 1 call writes every metric, each one stamped with its own name, source,
    # and window. The export name IS the metric name.
    proxy_agg = pr.area_export(
        recipe, min_units=MIN_UNITS_PER_POLYGON, units=units
    )
    return (proxy_agg,)


@app.cell
def _(aggregated, geo, mo, pd, pr, proxy_agg, px):
    mo.md("## 4. Model score against the proxies, by AREA")

    _rows = []
    if not aggregated.empty and not proxy_agg.empty:
        for _model in aggregated.model.unique():
            _m = aggregated[aggregated.model == _model]
            for _metric in pr.METRIC_ORDER:
                _p = proxy_agg[proxy_agg.proxy == _metric]
                for _layer in geo.LAYERS:
                    _r = px.correlate(_m, _p, _layer)
                    _r["model"] = _model
                    _r["metric"] = _metric
                    _r["role"] = pr.METRICS[_metric]["role"]
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

    **Warning: `outdoor_dining` is a confounder, not a validation.** Outdoor
    seating stands on the street, thus the model can see it. A high value on
    that row says that one visible amenity explains much of the judgment.
    """
    )
    correlations
    return


@app.cell
def _(mo, pr):
    mo.md(
        """
    ## 5. Model score against the proxies, by UNIT

    An area correlation can appear where no unit correlation exists, because
    the aggregation itself makes agreement. This case needs no join work:
    `uid` IS `unit_uid`. Thus the unit number is the honest one.

    The place scores come from the canonical registry, through
    `_unit_validation.place_scores`. This cell reads that module and writes
    nothing to it.

    **Warning: this cell reads 2 run parquets of 1,100,000 rows each. Allow
    minutes.**
    """
    )
    unit_correlations = pr.unit_rows()

    # The same table again, on the units that the history mean keeps. Every row
    # then holds the same restaurants, thus a reader compares 2 rulers and not
    # 2 samples.
    unit_correlations_common = pr.unit_rows(
        restrict_to="inspection_score_history"
    )
    return unit_correlations, unit_correlations_common


@app.cell
def _(unit_correlations):
    unit_correlations
    return


@app.cell
def _(unit_correlations_common):
    unit_correlations_common
    return


@app.cell
def _(mo):
    mo.md(
        """
## 6. Export

The next cell writes the tables to `restaurants/outputs/`. Give the paper these
files together. The numbers mean nothing without the provenance.

The proxy table carries `proxy`, `proxy_metric`, `proxy_source`, and
`proxy_window`. Read them before you name a proxy in the paper.

**Warning: the model-score export stays at version 1.0.0.** Its arithmetic did
not change at 1.2.0, and the paper table reads that file. The cell writes it
only when it is absent.
"""
    )
    return


@app.cell
def _(
    CASE,
    MODEL_EXPORT_VERSION,
    NOTEBOOK_VERSION,
    Path,
    aggregated,
    correlations,
    mo,
    proxy_agg,
    unit_correlations,
    unit_correlations_common,
):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    written = []
    skipped = []

    _agg_path = out_dir / f"{CASE}_aggregated_v{MODEL_EXPORT_VERSION}.parquet"
    if aggregated is not None and not aggregated.empty:
        if _agg_path.exists():
            skipped.append(str(_agg_path))
        else:
            aggregated.to_parquet(_agg_path, index=False)
            written.append(str(_agg_path))

    for _name, _frame in [
        ("proxy", proxy_agg),
        ("correlations", correlations),
        ("unit_correlations", unit_correlations),
        ("unit_correlations_common", unit_correlations_common),
    ]:
        if _frame is not None and not _frame.empty:
            _p = out_dir / f"{CASE}_{_name}_v{NOTEBOOK_VERSION}.parquet"
            _frame.to_parquet(_p, index=False)
            written.append(str(_p))

    mo.md(
        "**Written:**\n\n"
        + "\n".join(f"- `{w}`" for w in written)
        + ("\n\n**Kept as it was:**\n\n"
           + "\n".join(f"- `{s}`" for s in skipped) if skipped else "")
        if written or skipped
        else "**Nothing to write yet.** No finished run exists."
    )
    return


if __name__ == "__main__":
    app.run()
