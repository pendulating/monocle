"""Validation via statistical proxy — the schools case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
urbanpairvqa run of the schools prompt, scores each school, and aggregates the
scores into 3 NYC geographies.

The notebook discovers its runs from W&B. It does not name run directories. Thus
it picks up the qwen3.5-9b run as soon as that run finishes, with no edit.

Run it:
    .venv-nightly/bin/marimo edit notebooks/cvpr/schools_validation.py

Warning: use `.venv-nightly`. It has marimo and geopandas. `.venv-3.12` has no
marimo.
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

    # The notebook lives beside its helper modules.
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
    CASE = "schools"
    CURATION_ROOT = "curation/facdb_schools_k_12"
    MIN_UNITS_PER_POLYGON = 3
    return CASE, CURATION_ROOT, MIN_UNITS_PER_POLYGON, NOTEBOOK_VERSION


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, prov):
    mo.md(
        f"""
    # Schools — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** ·
    provenance module {prov.__version__} · geography module {geo.__version__}

    This notebook reads every canonical run of the schools prompt, scores each
    school from the pairwise labels, and aggregates the scores into
    neighborhood tabulation areas, community districts, and census tracts.
    """
    )
    return


@app.cell
def _(CASE, mo, prov):
    mo.md("## 1. Run discovery")

    # Warning: this cell reads the network. It runs one time.
    # `only_finished=False` also lists the runs that are still in progress, so
    # you can see what the notebook will pick up later.
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

    The paper cites this table. Each row names the model, the git commit, and
    the W&B run that made the numbers below.

    - **{len(usable)}** run(s) are finished and readable.
    - **{len(pending)}** run(s) are not ready yet. This notebook adds them when
      they finish. Do not edit the notebook for that.

    A run counts as canonical only when the model matches the canonical set
    exactly, the run starts on or after the consolidation date, and the run
    comes from a canonical sweep. See `README.md`.
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

    # `score_units` drops an abstention instead of scoring it 0. A `NotSure` is
    # not a judgment of "equal", so it must not pull a mean toward 0.
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
                    "schools_scored": len(s),
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

    A high abstention rate means the model declined most pairs. The score of a
    school then rests on few comparisons, and the mean is weak evidence.
    """
    )
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.cell
def _(CURATION_ROOT, geo, mo):
    mo.md("## 3. Geographic aggregation")

    # The unit position comes from FacDB, not from the pairs manifest. The pairs
    # manifest holds the camera position, which can fall in the neighbor polygon.
    units_gdf = geo.load_units(CURATION_ROOT)
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
def _(MIN_UNITS_PER_POLYGON, aggregated, mo, pd):
    if aggregated.empty:
        coverage = pd.DataFrame()
    else:
        coverage = (
            aggregated.groupby(["model", "layer"])
            .agg(
                polygons=("layer", "size"),
                schools=("n_units", "sum"),
                mean_score=("mean_score", "mean"),
            )
            .reset_index()
        )

    mo.md(
        f"""
    ### Coverage

    `polygons` counts the areas that hold at least {MIN_UNITS_PER_POLYGON}
    scored schools. An area with fewer schools gives a mean that is only 1 or 2
    schools, so the aggregation drops it.
    """
    )
    return (coverage,)


@app.cell
def _(coverage):
    coverage
    return


@app.cell
def _(aggregated):
    aggregated
    return


@app.cell
def _(CASE, mo, px):
    mo.md("## 4. Statistical proxy")

    # `recipe.json` in the case directory names the proxy sources. Add a source
    # there. Do not name an endpoint in this notebook.
    recipe = px.load_recipe(CASE)
    proxy_entry = next(e for e in recipe if e.get("role", "proxy") == "proxy")
    geocode_entry = next((e for e in recipe if e.get("role") == "geocode"), None)

    mo.md(
        f"""
    The proxy comes from `{CASE}/recipe.json`. It names
    **{proxy_entry['name']}** (`{proxy_entry['resource_id']}`), and
    **{geocode_entry['name'] if geocode_entry else 'no geocode source'}** to put
    each school on the map.

    Credentials present: **{px.has_credentials()}**. The notebook reads the
    SODA key from the environment variables `NYC_API_ID` and `NYC_API_SKEY`.
    Those values live in `.env`, which git ignores. Never write a key into this
    repository, and never print one.
    """
    )
    return geocode_entry, proxy_entry, recipe


@app.cell
def _(DEFAULT_METRIC, PROXY_YEAR, mo, proxy_entry, px):
    # `list_metrics` groups on the server, so it moves a small table.
    metric_choices = px.list_metrics(proxy_entry, school_year=PROXY_YEAR, top=40)
    metric_ui = mo.ui.dropdown(
        options=list(metric_choices.metric_display_name),
        value=DEFAULT_METRIC
        if DEFAULT_METRIC in list(metric_choices.metric_display_name)
        else list(metric_choices.metric_display_name)[0],
        label="Proxy metric",
    )
    metric_ui
    return metric_choices, metric_ui


@app.cell
def _():
    # The proxy year and the default metric. Change them here, not in a cell
    # that also computes something.
    PROXY_YEAR = "2024"
    DEFAULT_METRIC = "Average Student Attendance"
    return DEFAULT_METRIC, PROXY_YEAR


@app.cell
def _(CASE, PROXY_YEAR, geocode_entry, metric_ui, mo, proxy_entry, px):
    # Warning: fetch 1 metric and 1 year. The whole source holds more than
    # 1,000,000 rows, and a whole-table fetch returns a truncated answer.
    proxy_wide = px.school_quality_wide(
        px.fetch_metric(proxy_entry, metric_ui.value, PROXY_YEAR),
        metric_ui.value,
        school_year=PROXY_YEAR,
    )

    # Put each school on the map with an exact DBN join. The `ATS` column of
    # the geocode file is the DBN. A key join cannot make a false match; a name
    # join can, because 2 schools can share a name inside a borough.
    school_points = px.load_geocode(geocode_entry, CASE)
    join_result = px.join_proxy_points(proxy_wide, school_points)

    mo.md(
        f"""
    ### Put the proxy on the map

    | Quantity | Value |
    |---|---|
    | Proxy schools with a value | {join_result['n_proxy']} |
    | Points in the geocode file | {join_result['n_points']} |
    | Matched by DBN | {join_result['n_matched']} |
    | Match rate | {join_result['match_rate']:.1%} |

    **Warning: full coverage is not possible, and that is expected.** The proxy
    covers public schools. The model unit set comes from FacDB, which also holds
    private, charter, and postgraduate schools. Those have no DBN. The 2 sets
    describe different school populations. Thus compare them at the geography
    level, not school by school.
    """
    )
    return join_result, proxy_wide, school_points




@app.cell
def _(CURATION_ROOT, MIN_UNITS_PER_POLYGON, geo, join_result, pd, px):
    # The building of a school IS the unit, thus FacDB gives the year by key
    # and no spatial join is necessary.
    facilities = geo.load_facilities(CURATION_ROOT)
    vintage = px.building_vintage(facilities)

    _frames = []
    for _layer in geo.LAYERS:
        _out = px.aggregate_proxy(
            join_result["joined"], _layer, min_units=MIN_UNITS_PER_POLYGON
        )
        if not _out.empty:
            _out = _out.copy()
            _out["proxy"] = "school_report_card"
            _frames.append(_out)
        _vin = px.vintage_by_layer(vintage, _layer,
                                   min_units=MIN_UNITS_PER_POLYGON)
        if not _vin.empty:
            _vin = _vin.copy()
            _vin["proxy"] = "construction_year"
            _frames.append(_vin)
    proxy_agg = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return facilities, proxy_agg, vintage


@app.cell(hide_code=True)
def _(facilities, mo, vintage):
    mo.md(
        f"""
    ### The second proxy: the year the building went up

    `data/geo/nyc_buildings.parquet` is the local copy of the city BUILDING
    footprints. It gives each BIN a `construction_year`, and FacDB gives each
    school a BIN, thus the join needs no geometry.

    | Measure | Value |
    |---------|-------|
    | Units | {len(facilities)} |
    | Units with a year | {len(vintage)} ({len(vintage) / len(facilities):.1%}) |
    | Median year | {int(vintage.vintage_year.median())} |

    **Warning: a year is not a quality measure.** The report card points one
    way, and a positive number means agreement. This one does not. A positive
    number here says that the model calls a NEWER school better, which is a
    finding about the model and not a test of it.

    **Warning: the report card covers public schools only.** The construction
    year covers every school with a BIN, thus the 2 proxies do not describe the
    same unit set. Read the `n` of each row.
    """
    )
    return


@app.cell
def _(aggregated, geo, mo, pd, proxy_agg, px):
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

    mo.md(
        """
    ### Model score against the proxy

    Read the Spearman value first. The model score is an ordinal mean, so a
    monotone agreement matters more than a linear one.

    **Warning:** a correlation here is not a causal claim. The model sees a
    facade. The proxy measures an outcome. A weak value can mean that the
    facade carries little signal, or that the abstention rate cut the sample.
    """
    )
    return (correlations,)


@app.cell
def _(correlations):
    correlations
    return


@app.cell
def _(mo):
    mo.md(
        """
## 5. Export

The next cell writes the aggregated table, the proxy table, the correlations,
and the provenance table to `notebooks/cvpr/outputs/`. Give the paper these
files together. The numbers mean nothing without the provenance.
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
