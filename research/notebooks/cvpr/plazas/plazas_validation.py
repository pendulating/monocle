"""Validation via statistical proxy — the plazas case.

This notebook covers the PLAZA half of the FacDB parks-and-plazas group: a
privately owned public space, a DOT pedestrian plaza, or a special district
space. The park half lives in `parks_validation.py`.

**No NYC source RATES the condition of a plaza, and none exists.** The Parks
Inspection Program rates DPR property only and never rates a plaza. Neither
the POPS source nor the DOT plaza source carries a condition rating.

Version 1.2.0 does not solve that. It adds 3 proxies that measure the
MAINTENANCE CAPACITY of a plaza instead — who pays to keep it, and how much
trouble its owner is in:

| Proxy | Measures | Units reached |
|-------|----------|---------------|
| `bid_membership` | A business improvement district covers it | 504 of 504 |
| `dob_ecb_violations` | Open orders against its tax lot, negated | 419 of 504 |
| `plaza_bid_partner` | Its DOT maintenance partner is a district | 92 of 504 |

Capacity is not condition. A plaza with money can still be dirty. Say
"maintenance capacity" in the paper, never "plaza condition".

The 3 proxies key on the UNIT, thus this case gains its first unit-level
comparison. Section 6 reports it, and it is the number to read.

Run it:
    .venv/bin/marimo edit research/notebooks/cvpr/plazas/plazas_validation.py

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

    # The shared modules sit in the parent, `research/notebooks/cvpr/`, because every
    # prompt folder uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    from research.notebooks.cvpr import _geography as geo
    from research.notebooks.cvpr import _provenance as prov
    from research.notebooks.cvpr import _proxies as px
    from research.notebooks.cvpr import _proxies_parks as pk

    return Path, geo, mo, pd, pk, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed.
    #
    # 1.1.0: the proxy export carries `proxy_metric`, `proxy_source`, and
    # `proxy_window`.
    # 1.2.0: 3 maintenance-capacity proxies, and the first unit comparison.
    NOTEBOOK_VERSION = "1.2.0"
    # CASE drives W&B discovery, so it must match the pipeline name
    # `pairwise_parks_plazas_mvp`. OUTPUT_PREFIX keeps this notebook from
    # overwriting the files that `parks_validation.py` writes.
    CASE = "parks"
    OUTPUT_PREFIX = "plazas"
    RECIPE_DIR = "plazas"
    UNIT_KIND = "plaza"
    CURATION_ROOT = "curation/facdb_parks_plazas"
    UNIT_TABLE = "facilities.parquet"

    # The area floor of this case. 1 keeps every polygon that holds a plaza.
    #
    # WARNING: `_proxy_arms.CASE_SPEC["plazas"]["min_units"]` reads 3 today.
    # The 2 constants must agree, and 3 is the wrong one. The evidence, at the
    # community district layer after the JIA filter, measured 2026-08-31:
    #
    # | Floor | Model areas | Income, crime, BID | Violations | Partner |
    # |-------|-------------|--------------------|------------|---------|
    # | 1 | 43 | 43 | 27 | 36 |
    # | 3 | 16 | 16 | 11 | 10 |
    #
    # `_results_table.MIN_AREAS` is 20. At floor 3 EVERY plaza row falls under
    # it, thus the case leaves the paper. At floor 1 the 3 broad rows carry
    # more than twice the floor. The NTA layer does not rescue floor 3 either:
    # it keeps 21 areas on the broad rows, 1 above the floor, and 8 on the
    # partner row.
    #
    # The cost of floor 1 is a thin district, not a biased one: 19 of the 43
    # districts hold exactly 1 plaza, thus their area mean IS that plaza's
    # score. Section 6 answers that cost by reporting the unit layer, where a
    # single-plaza district cannot appear.
    #
    # Change `_proxy_arms.CASE_SPEC["plazas"]` to `min_units=1`.
    MIN_UNITS_PER_POLYGON = 1

    # The model-score export does not change at 1.2.0. The floor stays at 1,
    # where 1.0.0 put it, thus `plazas_aggregated_v1.0.0.parquet` still holds
    # the right rows.
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
        UNIT_KIND,
        UNIT_TABLE,
    )


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, pk, prov, px):
    mo.md(
        f"""
    # Plazas — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__} ·
    parks proxies {pk.__version__}

    The model answers "which one is better maintained". No NYC source RATES
    plaza upkeep, and none exists. Version 1.2.0 measures the maintenance
    CAPACITY of a plaza instead: who pays to keep it, and how much trouble the
    owner of its lot is in.

    | Proxy | Scope | Measures |
    |-------|-------|----------|
    | `bid_membership` | The plaza | A district raises money on its street |
    | `dob_ecb_violations` | The tax lot | Orders against the lot, negated |
    | `plaza_bid_partner` | The plaza | Its DOT partner is a district |
    | `year_completed` | The plaza | AGE. Not oriented. |
    | `median_household_income` | The area | The households around it |
    | `crime_density` | The area | The complaint record near it |

    **Warning: capacity is not condition.** A plaza with money can still be
    dirty, and a lot with violations can hold a clean plaza. Write
    "maintenance capacity" in the paper, never "plaza condition".
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
                    "plazas_scored": len(s),
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
    ### Warning: unit attribution is weak for this group

    Most park and plaza rows in FacDB carry no BIN, so the facing filter tested
    the line of sight to a nearby building, not to the plaza.
    `attribution_confidence` averages 0.097. Treat a plaza score as noisier
    than a school or a restaurant score.
    """
    )
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.cell
def _(CURATION_ROOT, UNIT_KIND, UNIT_TABLE, geo, mo, px):
    mo.md("## 3. Units — plazas only")

    # Split first. A plaza has no inspector, so it must not sit in the same
    # table as a DPR park, where PIP would make the proxy look complete.
    facilities = geo.load_facilities(CURATION_ROOT, filename=UNIT_TABLE)
    kept = px.split_parks_plazas(facilities, UNIT_KIND)
    units_gdf = geo.load_units(CURATION_ROOT, filename=UNIT_TABLE)
    units_gdf = units_gdf[units_gdf.unit_uid.isin(kept.uid)]

    mo.md(
        f"""
    | Group | Units |
    |-------|-------|
    | Kept for this notebook (plaza) | {len(kept)} |
    | Sent to `parks_validation.py` | {len(facilities) - len(kept)} |
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
    income_entry = next(e for e in recipe if "S1901" in e["name"])
    crime_entry = next(e for e in recipe if "NYPD" in e["name"])
    pops_year_entry = next(
        e for e in recipe if e.get("column") == "year_completed"
    )
    partner_entry = next(e for e in recipe if e.get("column") == "partner")
    bid_entry = next(e for e in recipe if e.get("key") == "bid_polygon")
    dob_entry = next(e for e in recipe if "DOB Violations" in e["name"])
    ecb_entry = next(e for e in recipe if "ECB Violations" in e["name"])

    mo.md(
        f"""
    `{RECIPE_DIR}/recipe.json` names {len(recipe)} sources.

    ### Warning: no source rates the condition of a plaza

    Income describes the households around a plaza. The complaint record
    describes events near it. The 3 new proxies describe the MONEY and the
    ORDERS that reach it. **No NYC source rates plaza upkeep**, thus this case
    still cannot test the model against an inspector, the way the parks,
    schools, and restaurants cases can.

    ### What the 3 capacity proxies measure

    | Proxy | The claim it supports |
    |-------|-----------------------|
    | `bid_membership` | A district raises an assessment on this street and spends it on cleaning |
    | `dob_ecb_violations` | The owner of this tax lot carries open orders |
    | `plaza_bid_partner` | A district, and not a city agency, maintains this DOT plaza |

    **Warning: a violation names the tax LOT, not the plaza surface.** A POPS
    plaza shares its lot with the tower that had to build it, thus most orders
    on that lot describe an elevator, a boiler, or a facade. A large tower
    collects more orders than a small one for that reason alone.

    ### The proxies do not aggregate the same way

    | Proxy | Tract | NTA and community district |
    |-------|-------|----------------------------|
    | Income | True ACS median | **Population-weighted mean of tract medians** |
    | Crime | True value | **True value** |
    | The 3 capacity proxies | Mean over the plazas | Mean over the plazas |

    Crime is point data, so a count and an area aggregate exactly. Income is a
    median, and you cannot average medians. Name the income value at NTA and
    community district a population-weighted mean, never a median. A 0 or 1
    capacity value aggregates to a SHARE of the plazas of that area.

    ### Orientation

    Every proxy except `year_completed` is oriented so that higher is better.
    Income rises with wealth. Crime carries a sign of -1. A violation count
    carries a sign of -1. BID membership and a BID partner both read as
    maintenance money. A **positive** correlation on those rows means the model
    agrees with the proxy. Do not flip a sign again.

    **`year_completed` carries no orientation.** A positive number there says
    that the model calls a NEWER plaza better.

    ### Warning: the crime source is 1 winter quarter

    The file name says year to date 2026, but the record stops on 2026-03-31.
    94.8% of the 133,114 complaints fall in January to March 2026. Thus the
    export writes `2026Q1` in `proxy_window`. Never call it a year.
    """
    )
    return (bid_entry, crime_entry, dob_entry, ecb_entry, income_entry,
            partner_entry, pops_year_entry, recipe)


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

    def stamp(frame, proxy, metric, entry, window=None, source=None):
        """Add the proxy name and the 3 provenance columns to 1 frame.

        `source` overrides the dataset id. The violation proxy reads 2
        sources, thus its rows must name both and no single entry can.
        """
        out = frame.copy()
        out["proxy"] = proxy
        out["proxy_metric"] = metric
        out["proxy_source"] = source or source_id(entry)
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
    crime_metric_ui
    return (crime_metric_ui,)


@app.cell
def _(kept, mo, px):
    # A POPS plaza holds the year the SPACE was completed, thus the vintage
    # describes the plaza and not the tower beside it. The 92 DOT pedestrian
    # plazas carry no year in any city source, so they drop out.
    vintage = px.pops_vintage(kept)
    coverage_vintage = px.vintage_coverage(kept, vintage)
    mo.md(
        f"""
    ### The third proxy: the year the space was completed

    | Measure | Value |
    |---------|-------|
    | Plaza units | {len(kept)} |
    | Units with a year | {len(vintage)} ({len(vintage) / len(kept):.1%}) |
    | Median year | {int(vintage.vintage_year.median())} |

    **Warning: 1 plaza source of 3 carries a year.** POPS holds
    `year_completed`. The DOT pedestrian plazas and the city-state parks hold
    nothing, in any NYC source, thus this proxy covers the POPS units and no
    other. The table below gives the rate for each type.

    **Warning: a year is not a quality measure.** A positive number here says
    that the model calls a NEWER plaza better. It is a finding about the model,
    not a test of it.
    """
    )
    return coverage_vintage, vintage


@app.cell
def _(coverage_vintage):
    coverage_vintage
    return


@app.cell
def _(kept, mo, pd, pk, units_gdf):
    mo.md("### The 3 maintenance-capacity proxies")

    # (a) The DOT maintenance partner. Only the 92 DOT plazas carry one, and
    # the code enforces that: without the filter the nearest-plaza join gives
    # 28 POPS units the partner of a DOT plaza beside them.
    partner_values = pk.plaza_partner_values(units_gdf, kept)

    # (b) BID membership. Every plaza unit takes a 0 or a 1, thus this is the
    # first plaza proxy that covers the whole case.
    bid_values = pk.plaza_bid_values(units_gdf)

    # (c) DOB and ECB orders against the tax lot, negated.
    violation_values = pk.plaza_violation_values(kept)

    capacity_coverage = pd.DataFrame([
        {k: v for k, v in pk.coverage(_v, kept, _n).items() if k != "by_source"}
        for _n, _v in [("plaza_bid_partner", partner_values),
                       ("bid_membership", bid_values),
                       ("dob_ecb_violations", violation_values)]
    ])
    return (bid_values, capacity_coverage, partner_values, violation_values)


@app.cell
def _(capacity_coverage):
    capacity_coverage
    return


@app.cell
def _(bid_values, mo, partner_values, pk, violation_values):
    mo.md(
        f"""
    #### What each capacity proxy reaches

    | Proxy | Units | Value |
    |-------|-------|-------|
    | `plaza_bid_partner` | {len(partner_values)} of 504 | {partner_values.proxy_value.mean():.3f} of the DOT plazas take a district partner |
    | `bid_membership` | {len(bid_values)} of 504 | {bid_values.proxy_value.mean():.3f} of the plazas sit in a district |
    | `dob_ecb_violations` | {len(violation_values)} of 504 | median {-violation_values.proxy_value.median():.0f} orders on the lot |

    **Warning: do not build "the share of plazas with a named partner".** All
    93 DOT plazas name one, thus that share is 1.0 inside the DOT set and 0.0
    outside it. It measures which source a plaza came from, and nothing about
    upkeep. The partner row reports the TYPE instead:
    {pk.PARTNER_TYPE_LABEL['bid']} against a city agency or a community group.

    **Warning: a BID boundary follows the sidewalk, and a DOT plaza sits in the
    roadbed.** A strict `within` test finds 5 of the 92 DOT plazas. The loader
    uses a {pk.BID_TOLERANCE_FT:.0f} ft tolerance, which finds 43. See
    `_proxies_parks.BID_TOLERANCE_FT` for the whole sensitivity table.

    **Warning: 85 of the 504 plaza units carry no tax lot.** A DOT plaza sits
    in the roadbed, which owns no lot, thus its absence is by construction and
    not by an error. The violation row describes the POPS plazas and the
    special district spaces.
    """
    )
    return


@app.cell
def _(
    MIN_UNITS_PER_POLYGON,
    bid_entry,
    bid_values,
    crime_entry,
    crime_metric_ui,
    dob_entry,
    ecb_entry,
    geo,
    income_entry,
    kept,
    partner_entry,
    partner_values,
    pd,
    pk,
    pops_year_entry,
    px,
    source_id,
    stamp,
    vintage,
    violation_values,
):
    # The crime name IS the dropdown value, so a felony run cannot leave this
    # notebook under the name of a density run.
    #
    # The violation row reads 2 sources, thus its `proxy_source` names both.
    _viol_source = f"{source_id(dob_entry)}+{source_id(ecb_entry)}"

    _unit_keyed = [
        ("bid_membership", bid_entry["column"], bid_entry, None,
         bid_values[["unit_uid", "proxy_value"]]),
        ("plaza_bid_partner", partner_entry["column"], partner_entry, None,
         partner_values[["unit_uid", "proxy_value"]]),
        ("dob_ecb_violations", "violation_count", dob_entry, _viol_source,
         violation_values[["unit_uid", "proxy_value"]]),
    ]

    _frames = []
    for _layer in geo.LAYERS:
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
            _frames.append(stamp(_vin, "year_completed",
                                 pops_year_entry["column"], pops_year_entry))
        for _name, _metric, _entry, _src, _vals in _unit_keyed:
            _out = pk.unit_values_by_layer(_vals, kept, _layer,
                                           min_units=MIN_UNITS_PER_POLYGON)
            if not _out.empty:
                _frames.append(stamp(_out, _name, _metric, _entry,
                                     source=_src))
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

    A positive value against income means the model calls a plaza better in a
    wealthier area. A positive value against crime means the model calls a
    plaza better where the complaint record is thinner. A positive value
    against `bid_membership` or `dob_ecb_violations` means the model calls a
    plaza better where the maintenance money is.

    **Warning: read `n` on every row.** The plaza set is small and it sits
    mostly in Manhattan. At the community district layer the broad rows carry
    43 areas, the violation row carries 27, and the vintage row carries 12.
    `_results_table.MIN_AREAS` is 20, thus the last 2 rows print as thin.
    """
    )
    correlations
    return


@app.cell
def _(
    bid_values,
    mo,
    pd,
    partner_values,
    per_model,
    px,
    violation_values,
    vintage,
):
    mo.md(
        """
    ## 6. Unit comparison — the number to read

    Version 1.1.0 could not build this section. The 3 capacity proxies key on
    the UNIT, thus the plazas case now compares the model against an outside
    record for the SAME plaza.

    An area correlation can appear where no unit correlation exists, because
    the aggregation into polygons makes agreement of its own. The parks case
    shows exactly that. Read the unit row first.
    """
    )

    # `correlate_units` takes a `proxy_key` frame and a unit_uid -> proxy_key
    # map. Here the unit IS the key, thus the map is the identity.
    _unit_proxies = [
        ("bid_membership", "bid_polygon", bid_values),
        ("plaza_bid_partner", "partner", partner_values),
        ("dob_ecb_violations", "violation_count", violation_values),
        ("year_completed", "year_completed",
         vintage.rename(columns={"vintage_year": "proxy_value"})),
    ]

    unit_rows = []
    for _model, _scores in per_model.items():
        for _name, _metric, _vals in _unit_proxies:
            _v = _vals[["unit_uid", "proxy_value"]].dropna()
            _key = pd.DataFrame({"unit_uid": _v["unit_uid"],
                                 "proxy_key": _v["unit_uid"]})
            _prx = _v.rename(columns={"unit_uid": "proxy_key"})
            _r = px.correlate_units(_scores, _key, _prx)
            _r["model"] = _model
            _r["proxy"] = _name
            _r["proxy_metric"] = _metric
            unit_rows.append(_r)
    unit_correlations = pd.DataFrame(unit_rows)
    return (unit_correlations,)


@app.cell
def _(unit_correlations):
    unit_correlations
    return


@app.cell
def _(mo):
    mo.md(
        """
    Measured 2026-08-31 on the canonical 1,100,000-pair runs, at the plaza
    unit:

    | Rater | Proxy | n | Spearman rho | p on tau |
    |-------|-------|---|--------------|----------|
    | gemma-4-12b | BID membership | 470 | +0.031 | 0.51 |
    | gemma-4-12b | **DOB and ECB violations** | 395 | **+0.158** | 0.0014 |
    | gemma-4-12b | BID partner | 81 | +0.191 | 0.088 |
    | gemma-4-12b | Completion year | 342 | +0.015 | 0.77 |
    | qwen3.5-9b | BID membership | 470 | +0.087 | 0.060 |
    | qwen3.5-9b | **DOB and ECB violations** | 395 | **+0.139** | 0.0062 |
    | qwen3.5-9b | BID partner | 81 | +0.245 | 0.028 |
    | qwen3.5-9b | Completion year | 342 | +0.014 | 0.77 |

    **Both raters agree with the violation record, and the sign is the one the
    prompt asks for.** A plaza whose lot carries fewer open orders takes a
    higher model score. The effect is small and it is not zero.

    Membership of a district alone reaches nothing. The partner TYPE reaches
    more than membership on both raters, and 81 units carry it, thus read that
    row as a direction to test and not as a result.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## 7. Export

The next cell writes the tables to `plazas/outputs/`. Give the paper these
files together. The numbers mean nothing without the provenance.

The proxy table carries `proxy_metric`, `proxy_source`, and `proxy_window`.
Read them before you name a proxy in the paper.
"""
    )
    return


@app.cell
def _(EXPORT_AGGREGATED, NOTEBOOK_VERSION, OUTPUT_PREFIX, Path, aggregated, correlations, mo, proxy_agg, table, unit_correlations):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # The model-score table keeps the file that 1.0.0 wrote. The floor stays
    # at 1, thus a new file would repeat the old rows under a new name, and
    # `_results_table._newest` reads each pattern on its own.
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
