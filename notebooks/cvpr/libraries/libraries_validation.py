"""Validation via statistical proxy — the libraries case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
urbanpairvqa run of the libraries prompt, scores each library, and compares the
scores against median household income in 3 NYC geographies.

The notebook discovers its runs from W&B. It does not name run directories, so
it picks up a model as soon as that model finishes.

The proxy differs from the other cases. A library has no measurement of its own,
so the proxy describes the AREA around it, not the building. Read the warning in
section 4 before you read a number.

Run it:
    .venv-nightly/bin/marimo edit notebooks/cvpr/libraries_validation.py

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
    CASE = "libraries"
    CURATION_ROOT = "curation/facdb_libraries"
    # No filter. 1 keeps every polygon that holds a library.
    # NYC has only 236 libraries, and 112 NTAs hold exactly 1. A threshold of 3
    # dropped 152 of the 164 covered NTAs, so it cost most of the map to buy a
    # little smoothing. See the warning in section 2.
    MIN_UNITS_PER_POLYGON = 1
    return CASE, CURATION_ROOT, MIN_UNITS_PER_POLYGON, NOTEBOOK_VERSION


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, prov, px):
    mo.md(
        f"""
    # Libraries — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__}

    The model answers "which library building is better maintained". The proxy
    is median household income from the ACS 5-Year 2024 survey.
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
                    "libraries_scored": len(s),
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
    ### Warning: most polygons hold exactly 1 library

    NYC holds 236 public libraries, far fewer than the 2,000+ units of the
    schools and restaurants cases. More pairs cannot change that. The run makes
    110,000 pairs from those same 236 buildings, which buys about 693
    comparisons for each library — high precision for each unit, and no more
    map coverage.

    Libraries for each NTA, measured 2026-08-12:

    | Libraries in the NTA | Number of NTAs |
    |---|---|
    | 1 | 112 |
    | 2 | 40 |
    | 3 | 6 |
    | 4 | 5 |
    | 6 | 1 |

    `MIN_UNITS_PER_POLYGON = 1`, so no polygon is dropped. **Thus most polygon
    values are 1 library, not an area mean.** The `n_units` column states the
    count for each row. A threshold of 3 kept only 12 of the 164 covered NTAs,
    which cost most of the map, so this case takes the coverage instead.
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
def _(CASE, mo, px):
    mo.md("## 4. Statistical proxy — median household income")

    recipe = px.load_recipe(CASE)
    mo.md(
        f"""
    `{CASE}/recipe.json` names {len(recipe)} sources. The income table and the
    population weights sit on disk, so this cell makes no network call.

    ### Warning: this proxy describes the area, not the building

    The other cases measure the unit itself — an inspection grades a restaurant,
    a report card grades a school. **Income grades neither the library nor its
    upkeep.** It describes the households around it.

    Thus a correlation here supports a weaker claim. It can only say that the
    look of a library building tracks the wealth of its neighborhood. It cannot
    say that the model reads library quality.

    ### Warning: 2 of the 3 layers report a mean, not a median

    | Layer | Method | Is it a median? |
    |-------|--------|-----------------|
    | Census tract | Direct join | Yes, the true ACS median |
    | NTA | Population-weighted mean of tract medians | **No** |
    | Community district | Population-weighted mean of tract medians | **No** |

    You cannot average medians. No median of medians is a median of the whole
    area. Name the NTA and community-district value a population-weighted mean
    in the paper. Use the tract layer when a true median matters.

    The tract-to-area step needs no NYC atomic polygons (`wgbs-damt`), because
    the geographies nest. Every one of the 2,325 tracts puts its interior point
    in exactly 1 NTA and exactly 1 community district, and no tract falls in 2
    polygons. A strict `within` test matches only 681 tracts, because a shared
    border defeats it, so the code uses an interior point.
    """
    )
    return (recipe,)


@app.cell
def _(CURATION_ROOT, MIN_UNITS_PER_POLYGON, geo, pd, px):
    # The building of a library IS the unit, thus FacDB gives the year by key
    # and no spatial join is necessary.
    facilities = geo.load_facilities(CURATION_ROOT)
    vintage = px.building_vintage(facilities)

    _frames = []
    for _layer in geo.LAYERS:
        _out = px.income_by_layer(_layer)
        if not _out.empty:
            _out = _out.copy()
            _out["proxy"] = "median_household_income"
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
def _(facilities, mo, px, vintage):
    mo.md(
        f"""
    ### The second proxy: the year the building went up

    `data/geo/nyc_buildings.parquet` is the local copy of the city BUILDING
    footprints. It gives each BIN a `construction_year`, and FacDB gives each
    library a BIN, thus the join needs no geometry.

    | Measure | Value |
    |---------|-------|
    | Units | {len(facilities)} |
    | Units with a year | {len(vintage)} ({len(vintage) / len(facilities):.1%}) |
    | Median year | {int(vintage.vintage_year.median())} |

    **Warning: a year is not a quality measure.** The other proxies point one
    way, and a positive number means agreement. This one does not. A positive
    number here says that the model calls a NEWER library better, which is a
    finding about the model and not a test of it.
    """
    )
    return


@app.cell
def _(proxy_agg):
    proxy_agg.groupby("layer").agg(
        polygons=("layer", "size"),
        mean_income=("proxy_mean", "mean"),
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
    Read the Spearman value first, and read `n` beside it. A library case gives
    few polygons, so a correlation over a small `n` moves easily.

    A positive value means that the model calls a library better maintained in a
    wealthier area.
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
