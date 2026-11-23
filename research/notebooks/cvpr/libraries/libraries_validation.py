"""Validation via statistical proxy — the libraries case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
pairwisevqa run of the libraries prompt, scores each library, and compares the
scores against median household income in 3 NYC geographies.

The notebook discovers its runs from W&B. It does not name run directories, so
it picks up a model as soon as that model finishes.

From version 1.2.0 the case carries 3 proxies that measure the BUILDING. FacDB
gives 96.8% of the libraries a BIN, thus the city building record reaches each
unit by key. Income stays in the table as a context row, and it grades the
households near the library and not the library. Read section 4.

Run it:
    .venv-nightly/bin/marimo edit research/notebooks/cvpr/libraries_validation.py

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
    from research.notebooks.cvpr import _proxies_buildings as pb

    return Path, geo, mo, pb, pd, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed.
    #
    # 1.1.0: the proxy export carries `proxy_metric`, `proxy_source`, and
    # `proxy_window`.
    # 1.2.0: 3 building-condition proxies join by BIN, and the polygon floor
    # rises from 1 to 3. The area numbers MOVE at this version. See
    # `research/docs/building-condition-proxies.md`.
    NOTEBOOK_VERSION = "1.2.0"
    CASE = "libraries"
    CURATION_ROOT = "curation/facdb_libraries"
    # A polygon needs 3 scored libraries. Below that the polygon value is 1 or
    # 2 buildings, and the reader cannot tell an area effect from 1 building.
    # `_proxy_arms.CASE_SPEC` already reads 3, thus this line makes the 2
    # numbers agree. See the note in section 2.
    MIN_UNITS_PER_POLYGON = 3
    return CASE, CURATION_ROOT, MIN_UNITS_PER_POLYGON, NOTEBOOK_VERSION


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, pb, prov, px):
    mo.md(
        f"""
    # Libraries — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__} ·
    building proxies {pb.__version__}

    The model answers "which library building is better maintained". From
    version 1.2.0 the case reads 5 proxies:

    | Proxy | Source | What it grades |
    |-------|--------|----------------|
    | `building_violations_open` | DOB and ECB violations | The building |
    | `building_violations_total` | DOB and ECB violations | The building |
    | `alteration_permits` | DOB permit issuance | The building |
    | `median_household_income` | ACS 5-Year 2024 | The area |
    | `construction_year` | Building footprints | The age of the building |

    The first 3 rows answer the prompt. The last 2 stay for context.
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
    ### The polygon floor is 3, and it costs most of the map

    NYC holds 236 public libraries, far fewer than the 2,000+ units of the
    schools and restaurants cases. More pairs cannot change that. The run makes
    110,000 pairs from those same 236 buildings, which buys about 693
    comparisons for each library — high precision for each unit, and no more
    map coverage.

    Libraries for each polygon, measured 2026-08-31:

    | Layer | Polygons with 1+ | Polygons with 3+ |
    |-------|------------------|------------------|
    | NTA | 166 | 16 |
    | Community district | 59 | 49 |
    | Census tract | 231 | 3 |

    Version 1.1.0 read `MIN_UNITS_PER_POLYGON = 1`, thus most polygon values
    were 1 library and not an area mean. Version 1.2.0 reads 3, which is the
    floor that every other facility case reads and the floor that
    `_proxy_arms.CASE_SPEC` already applied to this case. A mean over 1
    building is that building, and no reader can tell it from an area effect.

    **Warning: the area numbers moved at version 1.2.0.** The NTA layer keeps
    16 of 166 polygons and the census tract layer keeps 3 of 231, thus read the
    community district layer for this case. The unit layer carries the case,
    and section 5 reports it.
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
def _(CASE, mo, pb, px):
    mo.md("## 4. Statistical proxies")

    recipe = px.load_recipe(CASE)
    # 1 recipe entry for each proxy. The export copies the id and the window of
    # the entry, so a reader of the parquet never has to open this notebook.
    income_entry = next(e for e in recipe if "S1901" in e["name"])
    building_entry = next(
        e for e in recipe if e.get("column") == "construction_year"
    )
    dob_entry = next(e for e in recipe
                     if e.get("resource_id") == pb.DOB_VIOLATIONS)
    ecb_entry = next(e for e in recipe
                     if e.get("resource_id") == pb.ECB_VIOLATIONS)
    permit_entry = next(e for e in recipe
                        if e.get("resource_id") == pb.PERMIT_ISSUANCE)

    # The recipe stays the authority on the id and the window. The module holds
    # the metric text, thus the 2 halves of the provenance never drift.
    building_entries = {
        "building_violations_open": {
            "resource_id": f"{dob_entry['resource_id']} + "
                           f"{ecb_entry['resource_id']}",
            "window": dob_entry["window"],
        },
        "building_violations_total": {
            "resource_id": f"{dob_entry['resource_id']} + "
                           f"{ecb_entry['resource_id']}",
            "window": dob_entry["window"],
        },
        "alteration_permits": {
            "resource_id": permit_entry["resource_id"],
            "window": permit_entry["window"],
        },
    }

    mo.md(
        f"""
    `{CASE}/recipe.json` names {len(recipe)} sources. The income table and the
    population weights sit on disk. The 3 building sources read the network,
    and each query filters on the BIN list of this case on the server.

    Credentials present: **{px.has_credentials()}**. The notebook reads the
    SODA key from the environment variables `NYC_API_ID` and `NYC_API_SKEY`.
    Those values live in `.env`, which git ignores. Never write a key into this
    repository, and never print one.

    ### Warning: income describes the area, not the building

    **Income grades neither the library nor its upkeep.** It describes the
    households around it. A correlation on that row can only say that the look
    of a library tracks the wealth of its neighborhood.

    The 3 building rows do not carry that limit. The city wrote each violation
    and each permit against the BIN of this library, thus they grade the object
    the model looked at.

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
    return (
        building_entries,
        building_entry,
        dob_entry,
        ecb_entry,
        income_entry,
        permit_entry,
        recipe,
    )


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
def _(CURATION_ROOT, geo, mo, pb):
    mo.md("### The building record")

    # The building of a library IS the unit, thus FacDB gives the BIN by key
    # and no spatial join is necessary. 1 pull serves all 3 building proxies.
    facilities = geo.load_facilities(CURATION_ROOT)
    building_bins = pb.facility_bins(facilities)
    record = pb.building_record(building_bins)
    record_coverage = pb.record_coverage(facilities, record)
    return building_bins, facilities, record, record_coverage


@app.cell(hide_code=True)
def _(mo, pb, record_coverage):
    _c = record_coverage.iloc[0]
    mo.md(
        f"""
    `_proxies_buildings` reads 3 city sources for the BINs of this case only.
    The window is **{pb.WINDOW_LABEL}**.

    | Measure | Value |
    |---------|-------|
    | Units | {int(_c["units"])} |
    | Units with a usable BIN | {int(_c["with_bin"])} ({_c["bin_rate"]:.1%}) |
    | Units the record reached | {int(_c["with_record"])} |
    | Units with 1+ violation | {int(_c["with_violation"])} ({_c["violation_rate"]:.1%}) |
    | Units with 1+ alteration job | {int(_c["with_permit"])} ({_c["permit_rate"]:.1%}) |

    **A count is a raw count for 1 building, and this notebook divides it by
    nothing.** The unit IS the building, thus the count already describes 1
    object. A count for each square foot would ask whether a building carries
    more faults than its size predicts, which is a different question.

    **Warning: a violation count carries a sign of -1.** Every value arrives
    oriented "higher is better", thus a positive correlation always means that
    the model and the city record agree.

    **Warning: the 2 building rows are not independent.** Over the library
    BINs the total violation count and the permit count agree at Spearman
    0.49, because both rise when work happens on a building.
    """
    )
    return


@app.cell
def _(
    MIN_UNITS_PER_POLYGON,
    building_entries,
    building_entry,
    facilities,
    geo,
    income_entry,
    pb,
    pd,
    px,
    record,
    stamp,
):
    vintage = px.building_vintage(facilities)

    _frames = []
    for _layer in geo.LAYERS:
        _out = px.income_by_layer(_layer)
        if not _out.empty:
            _frames.append(stamp(_out, "median_household_income",
                                 income_entry["column"], income_entry))
        _vin = px.vintage_by_layer(vintage, _layer,
                                   min_units=MIN_UNITS_PER_POLYGON)
        if not _vin.empty:
            _frames.append(stamp(_vin, "construction_year",
                                 building_entry["column"], building_entry))
        # The building proxies. The name IS the key of `BUILDING_METRICS`, thus
        # the export name follows the selection and never a fixed literal.
        for _name, _spec in pb.BUILDING_METRICS.items():
            _unit = pb.building_condition(facilities, _name, record=record)
            _bld = pb.condition_by_layer(_unit, _layer,
                                         min_units=MIN_UNITS_PER_POLYGON)
            if not _bld.empty:
                _frames.append(stamp(_bld, _name, _spec["metric"],
                                     building_entries[_name]))
    proxy_agg = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return proxy_agg, vintage


@app.cell(hide_code=True)
def _(facilities, mo, px, vintage):
    mo.md(
        f"""
    ### The context proxy: the year the building went up

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
    # 1 row for each proxy and layer. The units differ between the proxies —
    # dollars, a year, and a count — thus read `proxy_mean` down a row and
    # never across the rows.
    proxy_agg.groupby(["proxy", "layer"]).agg(
        polygons=("layer", "size"),
        proxy_mean=("proxy_mean", "mean"),
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
    few polygons, so a correlation over a small `n` moves easily. At the NTA
    layer 16 polygons hold 3 libraries, and at the census tract layer 3 do.

    | Proxy | What a POSITIVE value says |
    |-------|----------------------------|
    | `building_violations_open` | The model prefers a building with fewer open faults |
    | `building_violations_total` | The model prefers a building with fewer faults |
    | `alteration_permits` | The model prefers a building the city worked on |
    | `median_household_income` | The model prefers a library in a wealthier area |
    | `construction_year` | The model prefers a NEWER library |

    The last row is a finding and not a test. A year measures age, not quality.
    """
    )
    correlations
    return


@app.cell
def _(facilities, mo, pb, pd, per_model, record):
    mo.md("## 5b. Unit-level correlations")

    # The BIN proxies name a building, so the correlation can skip the area
    # layer. The unit layer is the headline: the area layer can make or
    # destroy agreement. The interval is a bootstrap over units.
    _rows = []
    for _model, _scores in per_model.items():
        for _metric in pb.BUILDING_METRICS:
            _unit = pb.building_condition(facilities, _metric, record=record)
            _r = pb.unit_correlation(_scores[["unit_uid", "mean_score"]], _unit)
            _r["model"] = _model
            _r["proxy"] = _metric
            _rows.append(_r)
    unit_correlations = pd.DataFrame(_rows)
    unit_correlations
    return (unit_correlations,)


@app.cell
def _(mo):
    mo.md(
        """
## 6. Export

The next cell writes the tables to `research/notebooks/cvpr/outputs/`. Give the paper
these files together. The numbers mean nothing without the provenance.

The proxy table carries `proxy_metric`, `proxy_source`, and `proxy_window`.
Read them before you name a proxy in the paper.
"""
    )
    return


@app.cell
def _(
    CASE,
    NOTEBOOK_VERSION,
    Path,
    aggregated,
    correlations,
    mo,
    proxy_agg,
    table,
    unit_correlations,
):
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
