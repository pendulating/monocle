"""Validation via statistical proxy — the parks case (DPR property only).

This notebook covers the PARK half of the FacDB parks-and-plazas group. The
plaza half lives in `plazas_validation.py`. Warning: do not merge them. The
Parks Inspection Program rates only DPR property, so a mixed table lets the
proxy look complete when it covers 79% of the units.

The Parks Inspection Program rates the same object the model rates, so this
notebook compares them UNIT BY UNIT — 1 park, 1 model score, 1 inspector
rating. It also reports the area comparison, because the 2 disagree, and that
disagreement is the result of this case.

Version 1.2.0 adds 2 proxies:

- `pip_condition_acceptable_rate`, the second PIP metric. Version 1.1.0 built
  it and never exported it, thus the paper saw the surface of a park and not
  its fabric.
- `park_acreage`, the size of the site. It carries NO orientation, and it is
  the strongest unit-level signal that this case holds.

Run it:
    .venv/bin/marimo edit notebooks/cvpr/parks/parks_validation.py

Warning: use `.venv`. It has marimo and geopandas.
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
    import _proxies_parks as pk

    return Path, geo, mo, pd, pk, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed.
    #
    # 1.1.0: the proxy export carries `proxy_metric`, `proxy_source`, and
    # `proxy_window`, and the PIP name follows the dropdown.
    # 1.2.0: the export carries BOTH PIP metrics and the park acreage.
    NOTEBOOK_VERSION = "1.2.0"
    # CASE drives W&B discovery and must match the pipeline name.
    CASE = "parks"
    OUTPUT_PREFIX = "parks"
    RECIPE_DIR = "parks"
    UNIT_KIND = "park"
    CURATION_ROOT = "curation/facdb_parks_plazas"

    # The area floor of this case. 1 keeps every polygon that holds a park.
    #
    # It matches `_proxy_arms.CASE_SPEC["parks"]["min_units"]`, which is also 1.
    # Measured 2026-08-31 at the community district layer, after the JIA
    # filter: floors 1, 2, 3, and 5 all keep the SAME 59 districts. No park
    # district holds 1 park alone, thus the floor cannot change a paper number
    # at the layer the master table reports. Keep the 2 constants equal.
    #
    # Warning: the floor DOES bite at the finer layers — the NTA layer keeps
    # 196 areas at 1 and 163 at 3, and the tract layer keeps 892 at 1 and 172
    # at 3. Read `n` on any row that leaves the community district layer.
    MIN_UNITS_PER_POLYGON = 1
    PIP_SINCE_YEAR = 2023

    # The model-score export does not change at 1.2.0. The floor stays where
    # 1.0.0 put it, thus `parks_aggregated_v1.0.0.parquet` still holds the
    # right rows and the master table reads it beside the new proxy file.
    #
    # Warning: set this to True when the floor moves, or the 2 files disagree.
    EXPORT_AGGREGATED = False
    return (
        CASE,
        EXPORT_AGGREGATED,
        OUTPUT_PREFIX,
        RECIPE_DIR,
        CURATION_ROOT,
        MIN_UNITS_PER_POLYGON,
        NOTEBOOK_VERSION,
        PIP_SINCE_YEAR,
        UNIT_KIND,
    )


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, pk, prov, px):
    mo.md(
        f"""
    # Parks — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__} ·
    parks proxies {pk.__version__}

    The model answers "which one is better maintained". The Parks Inspection
    Program answers the same question about the same parks, so this notebook
    compares them unit by unit. Income and crime give area context.

    This notebook holds the PARK half only. The plaza half lives in
    `plazas_validation.py`.

    ## What version 1.2.0 adds

    | Proxy | Measures | Orientation |
    |-------|----------|-------------|
    | `pip_condition_acceptable_rate` | The structure and the landscape | Higher is better |
    | `park_acreage` | The size of the site | **None** |

    PIP rates a park on 2 axes, and they are different visual constructs.
    `cleanliness` reads litter, glass, and graffiti — the surface today.
    `overall_condition` reads the structure and the planting — the fabric. A
    park can be clean and broken, thus 1 rate cannot stand for both.

    **Warning: acreage carries no orientation.** A positive number on that row
    says that the model calls a LARGER park better. State it as a finding.
    Never read it as agreement with an inspector.
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
    mo.md("## 4. The proxies")

    recipe = px.load_recipe(RECIPE_DIR)
    # 1 recipe entry for each proxy. The export copies the id and the window of
    # the entry, so a reader of the parquet never has to open this notebook.
    #
    # The 2 PIP metrics share 1 entry, because 1 source holds both columns.
    # The `proxy_metric` column of each row names the exact metric.
    pip_entry = next(e for e in recipe if e.get("key") == "prop_id")
    income_entry = next(e for e in recipe if "S1901" in e["name"])
    crime_entry = next(e for e in recipe if "NYPD" in e["name"])
    acquisition_entry = next(
        e for e in recipe if e.get("column") == "acquisitiondate"
    )
    acreage_entry = next(e for e in recipe if e.get("column") == "acres")

    mo.md(
        f"""
    `{RECIPE_DIR}/recipe.json` names {len(recipe)} sources.

    | Proxy | Measures | Supports |
    |-------|----------|----------|
    | PIP cleanliness | The park itself | A claim about litter and graffiti |
    | PIP overall condition | The park itself | A claim about the fabric |
    | Park acreage | The park itself | A statement about SIZE, not quality |
    | Park acquisition year | The park itself | A statement about AGE, not quality |
    | Median household income | The area | A claim about the surroundings only |
    | NYPD complaints | The area | A claim about the surroundings only |

    **Read the 2 Parks Inspection Program rows as the result of this case.**
    They are the only proxies that GRADE the same object the model rates.
    Acreage and the acquisition year describe the park too, but they measure
    size and age, so they say what the model reads and not whether it is right.
    Income and crime describe the surroundings alone.

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

    The PIP rates, income, and crime leave `_proxies` oriented as "higher is
    better". A PIP rate is a share of acceptable inspections. Income rises with
    wealth. Crime carries a sign of -1. Thus a **positive** correlation on
    those rows means agreement.

    **2 rows carry no orientation: `acquisition_year` and `park_acreage`.**
    One measures age and the other measures size. A positive number there is a
    statement about what the model reads, and it is never an agreement score.

    ### Warning: the PIP source serves 0 rows today

    On 2026-08-31 a plain request to `yg3y-7juh` returns the header and no
    data. The 2 PIP metrics of this notebook come from the disk cache that a
    2026-08-12 pull filled, which holds `prop_id`, the rating, and the year for
    every inspection from 2023 onward.

    **Warning: do not delete `notebooks/cvpr/.proxy_cache/`.** The PIP rows
    cannot be fetched again while the city serves the source empty.
    """
    )
    return (acquisition_entry, acreage_entry, crime_entry, income_entry,
            pip_entry, recipe)


@app.cell
def _(Path):
    # The provenance contract of the proxy export, from notebook version
    # 1.1.0. Every row of the export carries 3 more columns:
    #
    # | Column | Holds |
    # |--------|-------|
    # | `proxy_metric` | The metric or the column that this run selected |
    # | `proxy_source` | The dataset id, or the file stem, from `recipe.json` |
    # | `proxy_window` | The time window of the source |
    #
    # Warning: never write a fixed proxy name beside a dropdown. Derive the
    # name from the selection. A fixed name lets a run export one metric under
    # the name of another.
    def source_id(entry):
        """The dataset id of a recipe source, or the stem of its file."""
        return entry.get("resource_id") or Path(entry["file"]).stem

    def stamp(frame, proxy, metric, entry, window=None):
        """Add the proxy name and the 3 provenance columns to 1 frame."""
        out = frame.copy()
        out["proxy"] = proxy
        out["proxy_metric"] = metric
        out["proxy_source"] = source_id(entry)
        out["proxy_window"] = window or entry["window"]
        return out

    return source_id, stamp


@app.cell
def _(mo, px):
    crime_metric_ui = mo.ui.dropdown(
        options=list(px.CRIME_METRICS),
        value="crime_density",
        label="Crime metric",
    )

    # The export name of each PIP metric. PIP rates cleanliness and overall
    # condition, and the 2 ratings disagree, thus 1 name cannot serve both.
    #
    # Warning: the cleanliness rows keep the name `pip_acceptable_rate`,
    # because the master table and the canonical registry hold that key. The
    # `proxy_metric` column names the exact metric of every row.
    PIP_PROXY_NAME = {
        "cleanliness_acceptable_rate": "pip_acceptable_rate",
        "condition_acceptable_rate": "pip_condition_acceptable_rate",
    }

    def pip_proxy_name(metric: str) -> str:
        """Give a PIP metric its export name, or stop the run."""
        if metric not in PIP_PROXY_NAME:
            raise KeyError(
                f"no export name for PIP metric {metric!r}. Add it to "
                f"PIP_PROXY_NAME, or the export names the wrong metric."
            )
        return PIP_PROXY_NAME[metric]

    # Version 1.1.0 held a dropdown here and exported the 1 metric it showed.
    # A reader of the parquet then saw the surface of a park and never its
    # fabric. Version 1.2.0 exports EVERY metric of `px.PIP_METRICS`, and
    # `pip_proxy_name` still derives each name from its own metric.
    PIP_METRICS_EXPORTED = tuple(px.PIP_METRICS)

    crime_metric_ui
    return (PIP_METRICS_EXPORTED, PIP_PROXY_NAME, crime_metric_ui,
            pip_proxy_name)


@app.cell
def _(PIP_METRICS_EXPORTED, PIP_SINCE_YEAR, mo, pd, pip_proxy_name, px):
    # Warning: filter the year on the server. The source holds 151,484 rows.
    #
    # Warning: `yg3y-7juh` serves 0 rows on 2026-08-31. These 2 metrics come
    # from the disk cache of the 2026-08-12 pull. Do not delete the cache.
    parks_shapes = px.load_parks_properties()
    pip_sites = {}
    pip_points = {}
    for _metric in PIP_METRICS_EXPORTED:
        _sites = px.parks_pip_proxy(metric=_metric, since_year=PIP_SINCE_YEAR)
        pip_sites[_metric] = _sites
        pip_points[_metric] = parks_shapes.merge(
            _sites, on="proxy_key", how="inner")

    pip_summary = pd.DataFrame([
        {
            "metric": _m,
            "proxy": pip_proxy_name(_m),
            "sites": len(pip_sites[_m]),
            "on_the_map": len(pip_points[_m]),
            "mean_rate": round(float(pip_points[_m].proxy_value.mean()), 4),
            "failed_once": int((pip_points[_m].proxy_value < 1.0).sum()),
        }
        for _m in PIP_METRICS_EXPORTED
    ])

    mo.md(
        f"""
    ### Both PIP metrics, since {PIP_SINCE_YEAR}

    The Parks Properties layer holds {len(parks_shapes)} shapes. A site reaches
    the map through its `gispropnum`.

    The 2 rates differ, and that difference is the reason to export both. A
    park can pass a cleanliness inspection and fail a condition inspection.
    """
    )
    return parks_shapes, pip_points, pip_sites, pip_summary


@app.cell
def _(pip_summary):
    pip_summary
    return


@app.cell
def _(MIN_UNITS_PER_POLYGON, mo, pk):
    # The size of a park, from the same DPR layer that supplies the shape and
    # the acquisition year. All 2,059 properties carry a value.
    acreage = pk.park_acreage()
    acreage_points = pk.park_acreage_points()

    mo.md(
        f"""
    ### The acreage proxy: how big the park is

    | Measure | Value |
    |---------|-------|
    | Properties with acreage | {len(acreage)} |
    | Placed on the map | {len(acreage_points)} |
    | Median acres | {acreage.proxy_value.median():.2f} |
    | Largest | {acreage.proxy_value.max():,.0f} acres |

    **Warning: acreage is not a quality measure and it carries no
    orientation.** A positive number says that the model calls a LARGER park
    better. Read it as a statement about what the model sees.

    Why the row belongs here: a photograph of a large park shows a horizon, a
    lawn, and a tree line, and a photograph of a small one shows a fence and a
    building. The model can read that difference without reading any upkeep.
    The area floor is {MIN_UNITS_PER_POLYGON}, the same as every other row.
    """
    )
    return acreage, acreage_points


@app.cell
def _(mo, pk):
    # Why version 1.2.0 exports no feature-level rate. Print the reason, so a
    # reader sees the blocker and not a silent gap.
    feature_status = pk.pip_feature_status()
    mo.md(
        f"""
    ### The feature ratings: found, and not usable today

    `{feature_status['feature_resource']}` holds
    {feature_status['feature_rows']:,} per-feature ratings, and it rates the
    parts of a park that a photograph shows: benches, paved surfaces,
    sidewalks, play equipment, fences, and trails.

    {feature_status['reason']}

    The Inspections source returned
    {feature_status['inspection_rows']} rows on this run.
    """
    )
    return (feature_status,)


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
def _(
    MIN_UNITS_PER_POLYGON,
    PIP_METRICS_EXPORTED,
    acquisition_entry,
    acreage_entry,
    acreage_points,
    crime_entry,
    crime_metric_ui,
    geo,
    income_entry,
    pd,
    pip_entry,
    pip_points,
    pip_proxy_name,
    px,
    stamp,
    vintage,
):
    # Each PIP name follows its OWN metric. Thus a condition row cannot leave
    # this notebook under the name of a cleanliness row.
    _frames = []
    for _layer in geo.LAYERS:
        for _metric in PIP_METRICS_EXPORTED:
            _pip = px.aggregate_proxy(pip_points[_metric], _layer,
                                      min_units=MIN_UNITS_PER_POLYGON)
            if not _pip.empty:
                _frames.append(stamp(_pip, pip_proxy_name(_metric), _metric,
                                     pip_entry))
        _inc = px.income_by_layer(_layer)
        if not _inc.empty:
            _frames.append(stamp(_inc, "median_household_income",
                                 income_entry["column"], income_entry))
        _cri = px.crime_by_layer(_layer, metric=crime_metric_ui.value)
        if not _cri.empty:
            _frames.append(stamp(_cri, crime_metric_ui.value,
                                 crime_metric_ui.value, crime_entry))
        _vin = px.vintage_by_layer(vintage, _layer,
                                   min_units=MIN_UNITS_PER_POLYGON)
        if not _vin.empty:
            _frames.append(stamp(_vin, "acquisition_year",
                                 acquisition_entry["column"], acquisition_entry))
        # Acreage joins the map through the same DPR point layer that PIP
        # uses, thus a park lands in the polygon its rating lands in.
        _acr = px.aggregate_proxy(acreage_points, _layer,
                                  min_units=MIN_UNITS_PER_POLYGON)
        if not _acr.empty:
            _frames.append(stamp(_acr, "park_acreage",
                                 acreage_entry["column"], acreage_entry))
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
def _(
    PIP_METRICS_EXPORTED,
    acreage,
    mo,
    pd,
    per_model,
    pip_proxy_name,
    pip_sites,
    px,
    units_gdf,
):
    mo.md(
        """
    ## 5. Unit comparison — the result of this case

    This is the number to report. It compares the model against the inspector
    for the same park, on both PIP metrics. The acreage row sits beside them,
    and it says what the model reads when the inspector says nothing.
    """
    )

    unit_key = px.attach_park_property_id(units_gdf)
    # 1 entry for each unit-keyed proxy. `correlate_units` takes a frame of
    # `proxy_key` and `proxy_value`, thus every entry has the same shape.
    _unit_proxies = [
        (pip_proxy_name(_m), _m, pip_sites[_m][["proxy_key", "proxy_value"]])
        for _m in PIP_METRICS_EXPORTED
    ] + [("park_acreage", "acres", acreage)]

    unit_rows = []
    for _model, _scores in per_model.items():
        for _name, _metric, _prx in _unit_proxies:
            _r = px.correlate_units(_scores, unit_key, _prx)
            _r["model"] = _model
            _r["proxy"] = _name
            _r["proxy_metric"] = _metric
            unit_rows.append(_r)
    unit_correlations = pd.DataFrame(unit_rows)
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
    is the honest one.** Aggregation into polygons creates agreement that does
    not exist between the objects. Report the unit value. Use the area value
    only to show the gap.

    Measured 2026-08-31 on the canonical 1,100,000-pair runs, at the park unit:

    | Rater | Proxy | n | Spearman rho | p on tau |
    |-------|-------|---|--------------|----------|
    | gemma-4-12b | PIP cleanliness | 1,054 | -0.023 | 0.47 |
    | gemma-4-12b | PIP condition | 1,054 | **-0.068** | 0.028 |
    | gemma-4-12b | **Park acreage** | 1,573 | **+0.285** | 1e-30 |
    | qwen3.5-9b | PIP cleanliness | 1,054 | +0.062 | 0.044 |
    | qwen3.5-9b | PIP condition | 1,054 | +0.064 | 0.039 |
    | qwen3.5-9b | Park acreage | 1,573 | +0.077 | 0.002 |

    **Read the acreage row beside the 2 inspector rows.** Neither inspector
    rate reaches 0.07 on either rater, and gemma-4-12b runs the WRONG way on
    condition. Acreage reaches +0.285 on the same parks. The model separates a
    large park from a small one, and it does not separate a clean park from a
    dirty one.

    The proxy is not degenerate, so this is not a ceiling effect. 35.8% of parks
    failed a cleanliness inspection and 49.6% failed a condition inspection.
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
    Read the `pip_acceptable_rate` and `pip_condition_acceptable_rate` rows
    first. They compare the model against an inspector who rated the same
    sites.

    **Warning: watch whether the proxies agree.** The subway case showed the
    model agree with income and disagree with crime, which revealed that the
    model tracked density rather than safety. If PIP and the area proxies split
    here, report the split. Do not report only the proxy that agrees.

    **Warning: `park_acreage` and `acquisition_year` are not agreement rows.**
    A positive number on either one is a statement about what the model reads.
    Add both names to `_results_table.UNORIENTED_PROXIES`.
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

The proxy table carries `proxy_metric`, `proxy_source`, and `proxy_window`.
Read them before you name a proxy in the paper.
"""
    )
    return


@app.cell
def _(EXPORT_AGGREGATED, NOTEBOOK_VERSION, OUTPUT_PREFIX, Path, aggregated, correlations, mo, proxy_agg, table, unit_correlations):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # The model-score table keeps the file that 1.0.0 wrote. The floor did not
    # move at 1.2.0, thus a new file would repeat the old rows under a new
    # name, and `_results_table._newest` reads each pattern on its own.
    _tables = [
        ("proxy", proxy_agg),
        ("correlations", correlations),
        ("unit_correlations", unit_correlations),
    ]
    if EXPORT_AGGREGATED:
        _tables.insert(0, ("aggregated", aggregated))

    written = []
    for _name, _frame in _tables:
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
