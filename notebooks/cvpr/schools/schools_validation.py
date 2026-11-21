"""Validation via statistical proxy — the schools case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
urbanpairvqa run of the schools prompt, scores each school, and aggregates the
scores into 3 NYC geographies.

The notebook discovers its runs from W&B. It does not name run directories. Thus
it picks up the qwen3.5-9b run as soon as that run finishes, with no edit.

From version 1.2.0 the case carries the building record. FacDB gives 99.0% of
the schools a BIN, thus the DOB and ECB violations and the DOB permits reach
each school by key. A teacher survey item adds the only city measurement of
school cleanliness, and the Economic Need Index replaces tract income as the
socio-economic control. Read section 4.

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
    import _proxies_buildings as pb

    return Path, geo, mo, pb, pd, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed. The
    # export file names carry it.
    #
    # 1.1.0: the proxy export carries `proxy_metric`, `proxy_source`, and
    # `proxy_window`, and the attendance rows carry the name
    # `school_attendance`. Version 1.0.0 named them `school_report_card`.
    # 1.2.0: 3 building-condition proxies join by BIN, the teacher survey adds
    # `school_kept_clean`, and the Economic Need Index adds
    # `school_economic_need`. The attendance row stays for context. See
    # `vlm-narratives-docs/building-condition-proxies.md`.
    NOTEBOOK_VERSION = "1.2.0"
    CASE = "schools"
    CURATION_ROOT = "curation/facdb_schools_k_12"
    MIN_UNITS_PER_POLYGON = 3
    return CASE, CURATION_ROOT, MIN_UNITS_PER_POLYGON, NOTEBOOK_VERSION


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, pb, prov):
    mo.md(
        f"""
    # Schools — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** ·
    provenance module {prov.__version__} · geography module {geo.__version__} ·
    building proxies {pb.__version__}

    This notebook reads every canonical run of the schools prompt, scores each
    school from the pairwise labels, and aggregates the scores into
    neighborhood tabulation areas, community districts, and census tracts.

    | Proxy | Source | What it grades |
    |-------|--------|----------------|
    | `building_violations_open` | DOB and ECB violations | The building |
    | `building_violations_total` | DOB and ECB violations | The building |
    | `alteration_permits` | DOB permit issuance | The building |
    | `school_kept_clean` | 2017-18 teacher survey | The building |
    | `school_economic_need` | Demographic Snapshot | The student body |
    | `school_attendance` | School Quality Reports | The students |
    | `construction_year` | Building footprints | The age of the building |

    The first 4 rows answer the prompt, which asks which school BUILDING looks
    better kept. The last 3 stay for context.
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
def _(CASE, mo, pb, px):
    mo.md("## 4. Statistical proxies")

    # `recipe.json` in the case directory names the proxy sources. Add a source
    # there. Do not name an endpoint in this notebook.
    recipe = px.load_recipe(CASE)
    proxy_entry = next(e for e in recipe
                       if e.get("resource_id") == "dnpx-dfnc")
    geocode_entry = next((e for e in recipe if e.get("role") == "geocode"), None)
    # The context proxy. It is a different source, thus the export must name it
    # apart from the School Quality Reports.
    building_entry = next(
        e for e in recipe if e.get("column") == "construction_year"
    )
    dob_entry = next(e for e in recipe
                     if e.get("resource_id") == pb.DOB_VIOLATIONS)
    ecb_entry = next(e for e in recipe
                     if e.get("resource_id") == pb.ECB_VIOLATIONS)
    permit_entry = next(e for e in recipe
                        if e.get("resource_id") == pb.PERMIT_ISSUANCE)
    survey_entry = next(e for e in recipe
                        if e.get("resource_id") == pb.SURVEY_TEACHER_2018)
    eni_entry = next(e for e in recipe
                     if e.get("resource_id") == pb.DEMOGRAPHIC_SNAPSHOT)

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
    school_entries = {
        "school_kept_clean": survey_entry,
        "school_economic_need": eni_entry,
    }

    mo.md(
        f"""
    The proxies come from `{CASE}/recipe.json`, which names {len(recipe)}
    sources. **{geocode_entry['name'] if geocode_entry else 'No geocode source'}**
    puts each DBN-keyed proxy on the map.

    Credentials present: **{px.has_credentials()}**. The notebook reads the
    SODA key from the environment variables `NYC_API_ID` and `NYC_API_SKEY`.
    Those values live in `.env`, which git ignores. Never write a key into this
    repository, and never print one.
    """
    )
    return (
        building_entries,
        building_entry,
        dob_entry,
        ecb_entry,
        eni_entry,
        geocode_entry,
        permit_entry,
        proxy_entry,
        recipe,
        school_entries,
        survey_entry,
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
    # the name of another, which is the fault this version corrects.
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
def _(pb):
    # The proxy year and the default metric. Change them here, not in a cell
    # that also computes something.
    PROXY_YEAR = "2024"
    DEFAULT_METRIC = "Average Student Attendance"

    # The export name of each metric. The School Quality Reports hold 404
    # metrics for school year 2024, thus 1 fixed name cannot describe the
    # export.
    #
    # Warning: the name before version 1.1.0 was `school_report_card`, and it
    # named no metric at all. The paper thus called an attendance rate a
    # report card. Do not bring that name back.
    METRIC_PROXY_NAME = {
        "Average Student Attendance": "school_attendance",
    }

    # Version 1.2.0 extends the contract to the sources outside the School
    # Quality Reports. `_proxies_buildings` states the metric text of each
    # proxy it builds, thus the name of a building proxy also follows its
    # selection and never a literal in this notebook.
    OTHER_METRIC_NAME = {
        spec["metric"]: name
        for name, spec in {**pb.BUILDING_METRICS, **pb.SCHOOL_METRICS}.items()
    }

    def proxy_name_of_metric(metric: str) -> str:
        """Give a metric the export name that describes it.

        The function reads 3 tables, in order:

        1. `METRIC_PROXY_NAME`, the School Quality Reports metrics.
        2. `OTHER_METRIC_NAME`, the building and school metrics of
           `_proxies_buildings`.
        3. A slug of the metric text.

        A metric that no table names takes a name from its own text. Thus a new
        selection can never leave under an old name.
        """
        known = METRIC_PROXY_NAME.get(metric) or OTHER_METRIC_NAME.get(metric)
        if known:
            return known
        slug = "".join(c if c.isalnum() else "_" for c in metric.lower())
        while "__" in slug:
            slug = slug.replace("__", "_")
        return "school_" + slug.strip("_")

    return (
        DEFAULT_METRIC,
        METRIC_PROXY_NAME,
        OTHER_METRIC_NAME,
        PROXY_YEAR,
        proxy_name_of_metric,
    )


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
def _(CURATION_ROOT, geo, mo, pb):
    mo.md("### The building record")

    # The building of a school IS the unit, thus FacDB gives the BIN by key and
    # no spatial join is necessary. 1 pull serves all 3 building proxies.
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

    **These 3 proxies cover the whole unit set, and the DBN proxies do not.**
    Attendance, cleanliness, and economic need all need a DBN, which a private
    or a postgraduate school has not. A BIN reaches 99.0% of the case.

    **A count is a raw count for 1 building, and this notebook divides it by
    nothing.** The unit IS the building, thus the count already describes 1
    object.

    **Warning: a violation count carries a sign of -1.** Every value arrives
    oriented "higher is better", thus a positive correlation always means that
    the model and the city record agree.

    **Warning: a school campus can hold 2 units in 1 building.** Both units
    then take the same value, which is correct: the record grades the building.

    **Warning: the 2 building rows are not independent.** Over the school BINs
    the total violation count and the permit count agree at Spearman 0.47,
    because both rise when work happens on a building.
    """
    )
    return


@app.cell
def _(CASE, geocode_entry, mo, pb, px):
    mo.md("### The 2 school-level proxies")

    # `school_kept_clean` is the only city measurement of a school BUILDING
    # that names a school. `school_economic_need` is the control that belongs
    # beside it, and it beats tract income here: a school draws its students
    # from a zone and not from its own block.
    clean_wide = pb.school_cleanliness().rename(columns={"dbn": "proxy_key"})
    eni_wide = pb.school_economic_need().rename(columns={"dbn": "proxy_key"})

    school_points_2 = px.load_geocode(geocode_entry, CASE)
    clean_join = px.join_proxy_points(clean_wide, school_points_2,
                                      key="proxy_key")
    eni_join = px.join_proxy_points(eni_wide, school_points_2, key="proxy_key")

    school_joins = {"school_kept_clean": clean_join,
                    "school_economic_need": eni_join}

    mo.md(
        f"""
    | Proxy | Schools with a value | Matched by DBN | Match rate |
    |-------|---------------------|----------------|------------|
    | `school_kept_clean` | {clean_join['n_proxy']} | {clean_join['n_matched']} | {clean_join['match_rate']:.1%} |
    | `school_economic_need` | {eni_join['n_proxy']} | {eni_join['n_matched']} | {eni_join['match_rate']:.1%} |

    **`school_kept_clean` is the teacher answer to `6f. This school is kept
    clean.`** The value is the share of teachers who agree or strongly agree.
    This is the one item-level building measurement that a city school source
    holds.

    **Warning: the survey window is 2017-18.** The city stopped publishing the
    item-level survey after that year, and the 2019 file holds only the 6
    framework scores. Thus this proxy sits about 7 years before the run. State
    the year in the paper.

    **Warning: `school_economic_need` is a control, not a building measure.**
    The index rises with need, thus `_proxies_buildings` negates it and a
    positive correlation says that the model prefers a building in a wealthier
    school community.
    """
    )
    return clean_join, clean_wide, eni_join, eni_wide, school_joins


@app.cell
def _(
    MIN_UNITS_PER_POLYGON,
    building_entries,
    building_entry,
    facilities,
    geo,
    join_result,
    metric_ui,
    pb,
    pd,
    proxy_entry,
    proxy_name_of_metric,
    px,
    record,
    school_entries,
    school_joins,
    stamp,
):
    vintage = px.building_vintage(facilities)

    # The export name follows the dropdown. Thus a different metric cannot
    # leave this notebook under the name of the default metric.
    proxy_name = proxy_name_of_metric(metric_ui.value)

    _frames = []
    for _layer in geo.LAYERS:
        _out = px.aggregate_proxy(
            join_result["joined"], _layer, min_units=MIN_UNITS_PER_POLYGON
        )
        if not _out.empty:
            _frames.append(stamp(_out, proxy_name, metric_ui.value, proxy_entry))
        _vin = px.vintage_by_layer(vintage, _layer,
                                   min_units=MIN_UNITS_PER_POLYGON)
        if not _vin.empty:
            _frames.append(stamp(_vin, "construction_year",
                                 building_entry["column"], building_entry))
        # The building proxies. The name IS the key of `BUILDING_METRICS`, and
        # `proxy_name_of_metric` gives that same name back from the metric
        # text, thus the export name follows the selection.
        for _name, _spec in pb.BUILDING_METRICS.items():
            _unit = pb.building_condition(facilities, _name, record=record)
            _bld = pb.condition_by_layer(_unit, _layer,
                                         min_units=MIN_UNITS_PER_POLYGON)
            if not _bld.empty:
                _frames.append(stamp(_bld, proxy_name_of_metric(_spec["metric"]),
                                     _spec["metric"], building_entries[_name]))
        # The 2 school-level proxies. They reach a polygon through the DOE
        # point file, the same road the attendance row takes.
        for _name, _spec in pb.SCHOOL_METRICS.items():
            _sch = px.aggregate_proxy(school_joins[_name]["joined"], _layer,
                                      min_units=MIN_UNITS_PER_POLYGON)
            if not _sch.empty:
                _frames.append(stamp(_sch, proxy_name_of_metric(_spec["metric"]),
                                     _spec["metric"], school_entries[_name],
                                     window=_spec["window"]))
    proxy_agg = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return proxy_agg, proxy_name, vintage


@app.cell(hide_code=True)
def _(facilities, metric_ui, mo, proxy_name, vintage):
    mo.md(
        f"""
    ### The context proxy: the year the building went up

    `data/geo/nyc_buildings.parquet` is the local copy of the city BUILDING
    footprints. It gives each BIN a `construction_year`, and FacDB gives each
    school a BIN, thus the join needs no geometry.

    | Measure | Value |
    |---------|-------|
    | Units | {len(facilities)} |
    | Units with a year | {len(vintage)} ({len(vintage) / len(facilities):.1%}) |
    | Median year | {int(vintage.vintage_year.median())} |

    **Warning: a year is not a quality measure.** The building proxies point
    one way, and a positive number means agreement. This one does not. A
    positive number here says that the model calls a NEWER school better,
    which is a finding about the model and not a test of it.

    **Warning: the DBN proxies cover public schools only.** The BIN proxies
    cover every school with a BIN, thus the 2 groups do not describe the same
    unit set. Read the `n` of each row.

    ### What the attendance proxy measures

    The export names it `{proxy_name}`, and `proxy_metric` holds the exact
    School Quality Reports metric: **{metric_ui.value}**. That source holds 404
    metrics for school year 2024 and NONE of them measure a building — the
    metric names cover attendance, test proficiency, course pass rates, and
    Regents scores. Thus attendance stays as a CONTEXT row, and the building
    rows carry the case. Do not call this row a report card: that word names
    the whole source, and this row holds 1 metric of it.
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
    ### Model score against each proxy

    Read the Spearman value first. The model score is an ordinal mean, so a
    monotone agreement matters more than a linear one.

    | Proxy | What a POSITIVE value says |
    |-------|----------------------------|
    | `building_violations_open` | The model prefers a building with fewer open faults |
    | `building_violations_total` | The model prefers a building with fewer faults |
    | `alteration_permits` | The model prefers a building the city worked on |
    | `school_kept_clean` | The model prefers a school the teachers call clean |
    | `school_economic_need` | The model prefers a school of a wealthier community |
    | `school_attendance` | The model prefers a school with better attendance |
    | `construction_year` | The model prefers a NEWER school |

    **Warning:** a correlation here is not a causal claim. The model sees a
    facade. Most of these proxies measure an outcome. A weak value can mean
    that the facade carries little signal, or that the abstention rate cut the
    sample.
    """
    )
    return (correlations,)


@app.cell
def _(correlations):
    correlations
    return


@app.cell
def _(clean_wide, eni_wide, facilities, mo, pb, pd, per_model, record,
      school_points_2):
    mo.md("### Unit-level correlations")

    # The unit layer is the headline: the area layer can make or destroy
    # agreement. The BIN proxies join by key. The 2 DBN proxies join with
    # `pb.school_unit_key`, which maps each unit to the nearest DBN point
    # inside 200 ft. The interval is a bootstrap over units.
    _key = pb.school_unit_key(school_points_2, facilities)

    _rows = []
    for _model, _scores in per_model.items():
        _s = _scores[["unit_uid", "mean_score"]]
        for _metric in pb.BUILDING_METRICS:
            _unit = pb.building_condition(facilities, _metric, record=record)
            _r = pb.unit_correlation(_s, _unit)
            _r["model"], _r["proxy"] = _model, _metric
            _rows.append(_r)
        for _name, _wide in [
            ("school_kept_clean", clean_wide),
            ("school_economic_need", eni_wide),
        ]:
            _unit = _key.merge(_wide[["proxy_key", "proxy_value"]],
                               on="proxy_key", how="inner")
            _r = pb.unit_correlation(_s, _unit[["unit_uid", "proxy_value"]])
            _r["model"], _r["proxy"] = _model, _name
            _rows.append(_r)
    unit_correlations = pd.DataFrame(_rows)
    unit_correlations
    return (unit_correlations,)


@app.cell
def _(mo):
    mo.md(
        """
## 5. Export

The next cell writes the aggregated table, the proxy table, the correlations,
and the provenance table to `notebooks/cvpr/outputs/`. Give the paper these
files together. The numbers mean nothing without the provenance.

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
