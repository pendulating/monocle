"""Validation via statistical proxy — the parks case (DPR property only).

This notebook covers the PARK half of the FacDB parks-and-plazas group. The
plaza half lives in `plazas_validation.py`. Warning: do not merge them. The
Parks Inspection Program rates only DPR property, so a mixed table lets the
proxy look complete when it covers 79% of the units.

The Parks Inspection Program rates the same object the model rates, so this
notebook compares them UNIT BY UNIT — 1 park, 1 model score, 1 inspector
rating. It also reports the area comparison, because the 2 disagree, and that
disagreement is the result of this case.

Run it:
    .venv-nightly/bin/marimo edit notebooks/cvpr/parks_validation.py

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
    # CASE drives W&B discovery and must match the pipeline name.
    CASE = "parks"
    OUTPUT_PREFIX = "parks"
    RECIPE_DIR = "parks"
    UNIT_KIND = "park"
    CURATION_ROOT = "curation/facdb_parks_plazas"
    # No filter. 1 keeps every polygon that holds a park.
    MIN_UNITS_PER_POLYGON = 1
    PIP_SINCE_YEAR = 2023
    return (
        CASE,
        OUTPUT_PREFIX,
        RECIPE_DIR,
        CURATION_ROOT,
        MIN_UNITS_PER_POLYGON,
        NOTEBOOK_VERSION,
        PIP_SINCE_YEAR,
        UNIT_KIND,
    )


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, prov, px):
    mo.md(
        f"""
    # Parks — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__}

    The model answers "which one is better maintained". The Parks Inspection
    Program answers the same question about the same parks, so this notebook
    compares them unit by unit. Income and crime give area context.

    This notebook holds the PARK half only. The plaza half lives in
    `plazas_validation.py`.
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
                    "parks_scored": len(s),
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
    ### Warning: check the unit attribution for this case

    Most park rows in FacDB carry no BIN, so the facing filter could not test
    the line of sight to the park's own building. It tested a nearby building
    instead. `attribution_confidence` is low but it varies: mean 0.097, median
    0.061. Treat a park score as noisier than a school or a restaurant score.
    """
    )
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.cell
def _(CURATION_ROOT, UNIT_KIND, geo, mo, px):
    mo.md("## 3. Units — parks only")

    # Split first. The plaza half has no maintenance proxy, so a mixed table
    # would let PIP look complete when it covers 79% of the group.
    facilities = geo.load_facilities(CURATION_ROOT)
    kept = px.split_parks_plazas(facilities, UNIT_KIND)
    units_gdf = geo.load_units(CURATION_ROOT)
    units_gdf = units_gdf[units_gdf.unit_uid.isin(kept.uid)]

    mo.md(
        f"""
    | Source | Units |
    |--------|-------|
    | Kept for this notebook ({UNIT_KIND}) | {len(kept)} |
    | Sent to `plazas_validation.py` | {len(facilities) - len(kept)} |
    """
    )
    return facilities, kept, units_gdf


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
def _(RECIPE_DIR, mo, px):
    mo.md("## 4. The 3 proxies")

    recipe = px.load_recipe(RECIPE_DIR)
    mo.md(
        f"""
    `{RECIPE_DIR}/recipe.json` names {len(recipe)} sources.

    | Proxy | Measures | Supports |
    |-------|----------|----------|
    | Parks Inspection Program | The park itself | A claim about park upkeep |
    | Median household income | The area | A claim about the surroundings only |
    | NYPD complaints | The area | A claim about the surroundings only |

    **Read the Parks Inspection Program as the result of this case.** It is the
    only proxy that rates the same object the model rates. The other 2 give
    context, and they cannot say whether the model reads park upkeep.

    ### How PIP reaches a park

    PIP inspects a ZONE inside a park and names it `B007-01`. The DPR layer
    holds only the parent site `B007`, so the loader cuts the suffix and rolls
    the zones up to the parent.

    **Warning: the FacDB geometry for a park is a nearby BUILDING, not the
    park.** Its median area is 0.0038 km^2, while Central Park covers 3.41
    km^2. Thus the code never uses the FacDB polygon. It takes the unit point
    and joins it to the true DPR polygon: 82.7% of park units fall inside a
    park, and 335 of the remaining 352 sit within 500 ft, so 98.0% get a
    property id.

    That id is what lifts this case from an area comparison to a **unit**
    comparison: 1 park, 1 model score, 1 inspector rating.

    ### Orientation

    All 3 proxies leave `_proxies` oriented as "higher is better". A PIP rate is
    a share of acceptable inspections. Income rises with wealth. Crime carries a
    sign of -1. Thus a **positive** correlation always means agreement.
    """
    )
    return (recipe,)


@app.cell
def _(mo, px):
    pip_metric_ui = mo.ui.dropdown(
        options=list(px.PIP_METRICS),
        value="cleanliness_acceptable_rate",
        label="PIP metric",
    )
    crime_metric_ui = mo.ui.dropdown(
        options=list(px.CRIME_METRICS),
        value="crime_density",
        label="Crime metric",
    )
    mo.hstack([pip_metric_ui, crime_metric_ui])
    return crime_metric_ui, pip_metric_ui


@app.cell
def _(PIP_SINCE_YEAR, mo, pip_metric_ui, px):
    # Warning: filter the year on the server. The source holds 151,484 rows.
    pip_sites = px.parks_pip_proxy(
        metric=pip_metric_ui.value, since_year=PIP_SINCE_YEAR
    )
    parks_shapes = px.load_parks_properties()
    pip_points = parks_shapes.merge(pip_sites, on="proxy_key", how="inner")

    mo.md(
        f"""
    | Quantity | Value |
    |---|---|
    | PIP parent sites since {PIP_SINCE_YEAR} | {len(pip_sites)} |
    | Parks Properties shapes | {len(parks_shapes)} |
    | Placed on the map | {len(pip_points)} |
    | Mean acceptable rate | {pip_points.proxy_value.mean():.3f} |
    """
    )
    return pip_points, pip_sites


@app.cell
def _(kept, mo, px, units_gdf):
    # A park is land, not a building, thus it carries no construction year.
    # The DPR acquisition date is the vintage that fits: the year the land
    # became a park. FacDB holds no property id, so this takes the same
    # spatial join that the inspection proxy takes.
    vintage = px.park_vintage(kept, units_gdf)
    mo.md(
        f"""
    ### The vintage proxy: the year the city took the land

    | Measure | Value |
    |---------|-------|
    | Park units | {len(kept)} |
    | Units with a year | {len(vintage)} ({len(vintage) / len(kept):.1%}) |
    | Median year | {int(vintage.vintage_year.median())} |

    **Warning: acquisition is not construction.** A park that the city acquired
    in 1936 can hold a playground of 2015. This row describes the age of the
    SITE, and it does not date what stands on it.

    **Warning: a year is not a quality measure.** The inspection rate points
    one way, and a positive number means agreement. This one does not. A
    positive number here says that the model calls a NEWER park better.

    Building construction year does not work here. Only 33% of the park units
    carry a BIN, and that BIN belongs to a comfort station or a recreation
    building, never to the park.
    """
    )
    return (vintage,)


@app.cell
def _(MIN_UNITS_PER_POLYGON, crime_metric_ui, geo, pd, pip_points, px, vintage):
    _frames = []
    for _layer in geo.LAYERS:
        _pip = px.aggregate_proxy(pip_points, _layer, min_units=MIN_UNITS_PER_POLYGON)
        if not _pip.empty:
            _pip = _pip.copy()
            _pip["proxy"] = "pip_acceptable_rate"
            _frames.append(_pip)
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
        _vin = px.vintage_by_layer(vintage, _layer,
                                   min_units=MIN_UNITS_PER_POLYGON)
        if not _vin.empty:
            _vin = _vin.copy()
            _vin["proxy"] = "acquisition_year"
            _frames.append(_vin)
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
def _(mo, per_model, pip_sites, px, units_gdf):
    mo.md(
        """
    ## 5. Unit comparison — the result of this case

    This is the number to report. It compares the model against the inspector
    for the same park.
    """
    )

    unit_key = px.attach_park_property_id(units_gdf)
    unit_rows = []
    for _model, _scores in per_model.items():
        _r = px.correlate_units(_scores, unit_key, pip_sites)
        _r["model"] = _model
        _r["proxy"] = "pip_acceptable_rate"
        unit_rows.append(_r)
    unit_correlations = __import__("pandas").DataFrame(unit_rows)
    return unit_correlations, unit_key


@app.cell
def _(unit_correlations):
    unit_correlations
    return


@app.cell
def _(mo):
    mo.md(
        """
    **Warning: the area value and the unit value disagree, and the unit value
    is the honest one.**

    Measured 2026-08-12 with gemma-4-12b, cleanliness:

    | Scope | n | Spearman rho |
    |-------|---|--------------|
    | Community district | 60 | +0.227 |
    | NTA | 206 | +0.117 |
    | **Park (unit)** | **1,048** | **-0.005** |

    Aggregation into polygons creates agreement that does not exist between the
    objects. Report the unit value. Use the area value only to show the gap.

    The proxy is not degenerate, so this is not a ceiling effect. 35.8% of parks
    failed a cleanliness inspection and 49.6% failed a condition inspection.
    Among only the parks with a failure, rho stays near zero: +0.081 for
    cleanliness and +0.031 for condition.
    """
    )
    return


@app.cell
def _(aggregated, geo, mo, pd, proxy_agg, px):
    mo.md("## 6. Area comparison, for contrast")

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
    Read the `pip_acceptable_rate` rows first. They compare the model against an
    inspector who rated the same sites.

    **Warning: watch whether the 3 proxies agree.** The subway case showed the
    model agree with income and disagree with crime, which revealed that the
    model tracked density rather than safety. If PIP and the area proxies split
    here, report the split. Do not report only the proxy that agrees.
    """
    )
    correlations
    return


@app.cell
def _(mo):
    mo.md(
        """
## 7. Export

The next cell writes the tables to `notebooks/cvpr/outputs/`. Give the paper
these files together. The numbers mean nothing without the provenance.
"""
    )
    return


@app.cell
def _(NOTEBOOK_VERSION, OUTPUT_PREFIX, Path, aggregated, correlations, mo, proxy_agg, table, unit_correlations):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    written = []
    for _name, _frame in [
        ("aggregated", aggregated),
        ("proxy", proxy_agg),
        ("correlations", correlations),
        ("unit_correlations", unit_correlations),
    ]:
        if _frame is not None and not _frame.empty:
            _p = out_dir / f"{OUTPUT_PREFIX}_{_name}_v{NOTEBOOK_VERSION}.parquet"
            _frame.to_parquet(_p, index=False)
            written.append(str(_p))
    if not table.empty:
        _p = out_dir / f"{OUTPUT_PREFIX}_provenance_v{NOTEBOOK_VERSION}.csv"
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
