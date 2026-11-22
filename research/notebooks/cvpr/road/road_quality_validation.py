"""Validation via statistical proxy — the road quality case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
pairwisevqa run of the road-quality prompt, scores each street-level image, and
compares the scores against 4 proxies.

**This case is IMAGE mode.** Road quality pairs random street-level shots, not
curated units, so `pairs.parquet` holds no `unit_uid` and there is no unit
table. The notebook scores each IMAGE and places it by its own coordinates.

The DOT pavement rating scores the same thing the model scores, so the notebook
compares them image by image, and also by area.

Run it:
    .venv-nightly/bin/marimo edit research/notebooks/cvpr/road_quality_validation.py

Warning: use `.venv-nightly`. It has marimo and geopandas.
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import pandas as pd

    # The shared modules sit in the parent, `research/notebooks/cvpr/`, because every
    # prompt folder uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import _geography as geo
    import _provenance as prov
    import _proxies as px
    import _proxies_road as pr

    return Path, geo, gpd, mo, pd, pr, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed.
    #
    # 1.1.0: the proxy export carries `proxy_metric`, `proxy_source`, and
    # `proxy_window`.
    # 1.2.0: the U5 upgrade. The rates get a roadway denominator, the pavement
    # mean gets a length weight, and the case gets the demand side. Section 6
    # holds the work, and `_proxies_road.py` holds the builders. The old rows
    # go into the export unchanged, so every published number reconciles.
    NOTEBOOK_VERSION = "1.2.0"

    # The version of the MODEL-SIDE exports. The U5 upgrade touched the proxy
    # side only, thus the model scores keep the version they shipped with and
    # the export cell never writes over them.
    MODEL_EXPORT_VERSION = "1.0.0"

    CASE = "road_quality"
    OUTPUT_PREFIX = "road_quality"
    RECIPE_DIR = "road"
    # No filter. 1 keeps every polygon that holds an image.
    MIN_UNITS_PER_POLYGON = 1
    PAVEMENT_SINCE_YEAR = 2024
    # A camera sits in the roadway, so a small cap is right. A large cap would
    # attach the rating of another street.
    MAX_SEGMENT_DISTANCE_FT = 100.0
    # A segment floor for an area mean of the pavement rating and of the
    # resurfacing age. 3 is the floor that every other case uses.
    MIN_SEGMENTS_PER_POLYGON = 3
    return (
        CASE,
        MAX_SEGMENT_DISTANCE_FT,
        MIN_SEGMENTS_PER_POLYGON,
        MIN_UNITS_PER_POLYGON,
        MODEL_EXPORT_VERSION,
        NOTEBOOK_VERSION,
        OUTPUT_PREFIX,
        RECIPE_DIR,
    )


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, pr, prov, px):
    mo.md(
        f"""
    # Road quality — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__} ·
    road proxies {pr.__version__}

    The model answers "which roadway is in better condition". DOT answers the
    same question about the same streets, so this notebook compares them image
    by image. Potholes, income, and crime give more context.

    Version 1.2.0 corrects the area side. Section 6 divides each pothole count
    by the centerline mile, weights the pavement mean by segment length, and
    adds the demand side that the case did not read.
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
      they finish.
    """
    )
    return (table,)


@app.cell
def _(table):
    table
    return


@app.cell
def _(mo, prov, usable):
    mo.md("## 2. Image scores")

    # Warning: image mode. Use `load_run_images` and `score_images`. The unit
    # functions fail here, because `pairs.parquet` holds no `unit_uid`.
    per_model = {}
    for _rec in usable:
        per_model[_rec.model] = prov.score_images(prov.load_run_images(_rec))
    return (per_model,)


@app.cell
def _(mo, pd, per_model):
    if per_model:
        summary = pd.DataFrame(
            [
                {
                    "model": m,
                    "images_scored": len(s),
                    "median_comparisons": float(s.n_comparisons.median()),
                    "mean_comparisons": round(s.n_comparisons.mean(), 2),
                    "mean_score": round(s.mean_score.mean(), 4),
                    "mean_abstention": round(s.abstention_rate.mean(), 4),
                }
                for m, s in per_model.items()
            ]
        )
    else:
        summary = pd.DataFrame()

    mo.md(
        """
    ### Warning: 1 image score is nearly 1 label

    An image carries far fewer comparisons than a curated unit. A library sat in
    about 693 comparisons. Here 110,000 pairs spread over a 500,000-image
    manifest, so the median image has **1** comparison and the mean has 1.33.

    Thus a single image score is close to a single ordinal label, and it is very
    noisy. The area step carries the precision, because a polygon holds many
    images. Never read 1 image score on its own.
    """
    )
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.cell
def _(RECIPE_DIR, mo, px):
    mo.md("## 3. The proxy sources")

    recipe = px.load_recipe(RECIPE_DIR)
    # 1 recipe entry for each proxy. The export copies the id and the window of
    # the entry, so a reader of the parquet never has to open this notebook.
    pavement_entry = next(e for e in recipe if "Pavement Ratings" in e["name"])
    pothole_entry = next(e for e in recipe if "Pothole" in e["name"])
    income_entry = next(e for e in recipe if "S1901" in e["name"])
    crime_entry = next(e for e in recipe if "NYPD" in e["name"])

    mo.md(
        f"""
    `{RECIPE_DIR}/recipe.json` names {len(recipe)} sources.

    | Proxy | Measures | Orientation |
    |-------|----------|-------------|
    | DOT pavement rating | The street itself | 0 to 10, high is good |
    | Pothole repairs | The city repair, the supply | Negated, so high is good |
    | 311 pothole complaints | The resident report, the demand | Negated, so high is good |
    | Years since resurfacing | The maintenance cycle | Negated, so high is good |
    | Median household income | The area | High is wealthy |
    | NYPD complaints | The area | Negated, so high is safe |

    The street centerline (CSCL) is not a proxy. It is the DENOMINATOR of the
    2 pothole rates. Section 6 explains why the rates need it.

    **Read the pavement rating as the result of this case.** It is the only
    proxy that scores the same object the model scores.

    ### Warning: a rating of 0.0 means NOT RATED

    72,768 of the 514,521 rows carry `systemrating = 0.0`. It does not mean the
    worst pavement. Scored as a zero it would drag whole neighborhoods down and
    look like a real signal. The zeros fall from 11,043 in 2021 to 1,358 in
    2026 while the rated rows hold steady, which is a backlog that clears, not
    pavement that improves. `pavement_segments` drops them.

    ### Warning: neither pothole source counts potholes

    The work orders count REPAIRS, so a borough that fixes potholes quickly
    looks worse than one that ignores them. The 311 rows count REPORTS, so a
    neighborhood that calls 311 often looks worse than a quiet one with the
    same pavement. Read the 2 together, and read both beside the pavement
    rating.

    ### Warning: 54% of the 311 rows carry no coordinate

    The filter finds 147,824 pothole complaints since 2024-01, and 68,108 of
    them carry a latitude. The located share differs by borough: Queens 42%,
    Manhattan 54%. The located rate and the whole-set rate rank the 59
    community districts at Spearman 0.92, thus the subset is safe for an ORDER
    and not for a LEVEL.

    ### Warning: the crime source is 1 winter quarter

    The file name says year to date 2026, but the record stops on 2026-03-31.
    94.8% of the 133,114 complaints fall in January to March 2026. Thus the
    export writes `2026Q1` in `proxy_window`. Never call it a year.
    """
    )
    return crime_entry, income_entry, pavement_entry, pothole_entry, recipe


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
def _(PAVEMENT_SINCE_YEAR, mo, px):
    # Warning: filter on the server. The source holds 514,521 rows.
    segments = px.pavement_segments(since_year=PAVEMENT_SINCE_YEAR)
    mo.md(
        f"""
    Rated segments since {PAVEMENT_SINCE_YEAR}: **{len(segments):,}** ·
    mean rating {segments.proxy_value.mean():.2f} ·
    standard deviation {segments.proxy_value.std():.2f}
    """
    )
    return (segments,)


@app.cell
def _(MAX_SEGMENT_DISTANCE_FT, gpd, mo, per_model, px, segments):
    mo.md(
        """
    ## 4. Image comparison — the result of this case

    Each image takes the rating of the nearest rated segment. This is the same
    move that overturned the area result in the parks case.
    """
    )

    image_points = {}
    unit_rows = []
    for _model, _s in per_model.items():
        _pts = gpd.GeoDataFrame(
            _s, geometry=gpd.points_from_xy(_s.longitude, _s.latitude), crs="EPSG:4326"
        )
        image_points[_model] = _pts
        _key = px.attach_nearest_segment(
            _pts, segments, max_distance_ft=MAX_SEGMENT_DISTANCE_FT
        )
        _j = _s.merge(_key, on="sample_id", how="inner")
        unit_rows.append({
            "model": _model,
            "proxy": "dot_pavement_rating",
            "scope": "image",
            "n": len(_j),
            "matched_pct": round(len(_j) / len(_s) * 100, 1),
            "pearson_r": round(_j.mean_score.corr(_j.proxy_value), 4),
            "spearman_rho": round(
                _j.mean_score.corr(_j.proxy_value, method="spearman"), 4
            ),
        })
    return image_points, unit_rows


@app.cell
def _(pd, unit_rows):
    unit_correlations = pd.DataFrame(unit_rows)
    unit_correlations
    return (unit_correlations,)


@app.cell
def _(mo):
    mo.md(
        """
    **Warning: read this beside the noise.** The median image has 1 comparison,
    so the model score is nearly a single label, and measurement error pulls any
    correlation toward zero.

    I checked whether the noise hides a signal. It does not. Measured 2026-08-12
    against the DOT rating, by the least number of comparisons for each image:

    | Comparisons | n (gemma) | rho (gemma) | rho (qwen) |
    |---|---|---|---|
    | 1 or more | 127,883 | +0.014 | +0.034 |
    | 2 or more | 33,674 | +0.020 | +0.038 |
    | 3 or more | 7,448 | +0.032 | +0.017 |

    The value stays near zero as the noise falls, so the null is not an
    artifact of thin sampling. Do not read the 5-or-more row: it holds 228
    images, and it swings negative.
    """
    )
    return


@app.cell
def _(MIN_UNITS_PER_POLYGON, geo, gpd, image_points, pd):
    _frames = []
    for _model, _pts in image_points.items():
        _p = _pts.to_crs(geo.WORKING_CRS)
        for _layer in geo.LAYERS:
            _poly = geo.load_layer(_layer)
            _key = geo.LAYERS[_layer]["key"]
            _hit = gpd.sjoin(
                _p[["sample_id", "mean_score", "n_comparisons", "geometry"]],
                _poly[[_key, "geometry"]], how="inner", predicate="within",
            )
            _out = _hit.groupby(_key).agg(
                n_units=("sample_id", "nunique"),
                mean_score=("mean_score", "mean"),
                sd_score=("mean_score", "std"),
                total_comparisons=("n_comparisons", "sum"),
            ).reset_index()
            _out = _out[_out.n_units >= MIN_UNITS_PER_POLYGON]
            _out.insert(0, "layer", _layer)
            _out.insert(0, "model", _model)
            _frames.append(_out)
    aggregated = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return (aggregated,)


@app.cell
def _(
    crime_entry,
    geo,
    income_entry,
    mo,
    pavement_entry,
    pd,
    pothole_entry,
    px,
    segments,
    stamp,
):
    mo.md("## 5. Area comparison")

    # This case holds no crime dropdown. The name still comes from the value
    # that the call uses, thus the 2 can never drift apart.
    CRIME_METRIC = "crime_density"

    _frames = []
    for _layer in geo.LAYERS:
        for _name, _metric, _entry, _fn in [
            ("dot_pavement_rating", pavement_entry["column"], pavement_entry,
             lambda L: px.pavement_by_layer(L, segments=segments)),
            ("pothole_repairs", "closed work orders for each km^2",
             pothole_entry, px.pothole_by_layer),
            ("median_household_income", income_entry["column"], income_entry,
             px.income_by_layer),
            (CRIME_METRIC, CRIME_METRIC, crime_entry,
             lambda L: px.crime_by_layer(L, metric=CRIME_METRIC)),
        ]:
            _out = _fn(_layer)
            if _out is not None and not _out.empty:
                _frames.append(stamp(_out, _name, _metric, _entry))
    proxy_agg = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return CRIME_METRIC, proxy_agg


@app.cell
def _(aggregated, geo, pd, proxy_agg, px):
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
def _(correlations):
    correlations
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## 6. The corrected proxies

An audit of 2026-08-31 found 3 problems in the area side of this case, and
`_proxies_road.py` holds the corrections. `_proxies.py` does not change, thus
every published number stays reproducible.

### The denominator

`pothole_repairs` divides closed work orders by the LAND area of a polygon. A
pothole happens on a roadway, not on land. A dense street grid holds more
roadway inside the same square kilometre, thus repairs for each km^2 rises with
street density and not only with bad pavement. The rate needs the roadway in
the denominator.

### The length weight

`dot_pavement_rating` takes a plain mean over street segments. A 60 ft block
and a 1 mi avenue each give 1 rating, thus the short block speaks as loudly as
the avenue. A driver meets the avenue for a mile. The corrected mean weights
each rating by the length of its own segment.

### The demand side

The case reads the city REPAIR and never the resident REPORT. 311 holds the
report. The 2 sides answer different questions, and the gap between them is
itself a measurement.
"""
    )
    return


@app.cell
def _(mo, pr):
    # Warning: this cell reads the network on the first run. It then reads the
    # parquet under `data/external/nyc_street_centerline/`.
    centerline = pr.centerline_lines()
    miles = pr.centerline_miles()
    _total = centerline["length_ft"].sum() / pr.FT_PER_MILE
    _by_layer = miles.groupby("layer")["centerline_miles"].agg(["size", "sum"])

    mo.md(
        f"""
    ### The centerline denominator

    CSCL (`{pr.CENTERLINE_RESOURCE}`), roadway types 1 (street) and 2
    (highway), status 2 (constructed): **{len(centerline):,}** segments and
    **{_total:,.1f}** centerline miles. NYC DOT publishes about 6,300
    centerline miles of street, thus this total is right.

    Each line is CUT at the polygon border, so a segment that crosses 2 areas
    gives each one its true part.

    | Layer | Areas | Centerline miles |
    |-------|-------|------------------|
    | Community district | {_by_layer.loc['community_district', 'size']:.0f} | {_by_layer.loc['community_district', 'sum']:,.1f} |
    | NTA | {_by_layer.loc['nta', 'size']:.0f} | {_by_layer.loc['nta', 'sum']:,.1f} |
    | Census tract | {_by_layer.loc['census_tract', 'size']:.0f} | {_by_layer.loc['census_tract', 'sum']:,.1f} |

    The community district and the NTA totals sit under the city total because
    `_geography` drops the areas that hold no residents.
    """
    )
    return centerline, miles


@app.cell
def _(MIN_SEGMENTS_PER_POLYGON, mo, pd, pr, recipe, segments):
    # The corrected proxies of every layer, with the 4 provenance columns.
    complaints = pr.pothole_complaints()
    blocks = pr.resurfacing_blocks()
    new_proxy = pr.area_export(
        recipe,
        min_segments=MIN_SEGMENTS_PER_POLYGON,
        segments=segments,
        complaints=complaints,
        blocks=blocks,
    )
    _seen = (new_proxy[["proxy", "proxy_metric", "proxy_source", "proxy_window"]]
             .drop_duplicates())
    mo.md(
        f"""
    ### What the corrected builders made

    311 pothole complaints with a coordinate: **{len(complaints):,}** ·
    resurfaced blocks: **{len(blocks):,}** ·
    new proxy rows: **{len(new_proxy):,}**
    """
    )
    return blocks, complaints, new_proxy


@app.cell
def _(new_proxy):
    new_proxy[["proxy", "proxy_metric", "proxy_source",
               "proxy_window"]].drop_duplicates().reset_index(drop=True)
    return


@app.cell
def _(MODEL_EXPORT_VERSION, OUTPUT_PREFIX, Path, aggregated, pd):
    # The model score of each polygon. A live run gives it; when no run is
    # readable, the shipped export gives it, so the comparison below still
    # works with no network.
    if aggregated is not None and not aggregated.empty:
        model_scores = aggregated
        model_scores_from = "this run"
    else:
        _p = (Path(__file__).resolve().parent / "outputs"
              / f"{OUTPUT_PREFIX}_aggregated_v{MODEL_EXPORT_VERSION}.parquet")
        model_scores = pd.read_parquet(_p) if _p.exists() else pd.DataFrame()
        model_scores_from = str(_p) if _p.exists() else "nothing"
    return model_scores, model_scores_from


@app.cell
def _(blocks, complaints, geo, model_scores, pd, pr, px, segments):
    # Old against new, on the same areas and the same model scores. Street
    # density carries no sign: it is the denominator, and it is here to show
    # how much of the old rate it explains.
    _agg_for_compare = model_scores
    _rows = []
    if not _agg_for_compare.empty:
        for _layer in geo.LAYERS:
            _frames = {
                "dot_pavement_rating":
                    px.pavement_by_layer(_layer, segments=segments),
                "dot_pavement_rating_weighted":
                    pr.pavement_by_layer_weighted(_layer, segments=segments),
                "pothole_repairs": px.pothole_by_layer(_layer),
                "pothole_repairs_per_mile":
                    pr.pothole_repairs_per_mile(_layer),
                "pothole_complaints_per_mile":
                    pr.pothole_complaints_per_mile(_layer, complaints=complaints),
                "resurfacing_age":
                    pr.resurfacing_age_by_layer(_layer, blocks=blocks),
                "street_density": pr.street_density(_layer),
            }
            _t = pr.compare(_agg_for_compare, _frames, layer=_layer)
            _t["transform"] = _t["proxy"].map({
                "dot_pavement_rating": "old",
                "pothole_repairs": "old",
                "dot_pavement_rating_weighted": "new",
                "pothole_repairs_per_mile": "new",
                "pothole_complaints_per_mile": "new",
                "resurfacing_age": "new",
                "street_density": "denominator",
            })
            _rows.append(_t)
    denominator_comparison = (pd.concat(_rows, ignore_index=True) if _rows
                              else pd.DataFrame())
    return (denominator_comparison,)


@app.cell
def _(denominator_comparison):
    denominator_comparison[
        denominator_comparison.layer == "community_district"
    ].reset_index(drop=True) if not denominator_comparison.empty else None
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### How to read the comparison

    Measured 2026-08-31 at the community district layer, over the 59 areas
    that hold residents. Kendall tau-b, the measure the master table prints.

    | Proxy | Transform | gemma-4-12b | qwen3.5-9b |
    |-------|-----------|-------------|------------|
    | Pothole repairs | Old, per km^2 | +0.276 | -0.002 |
    | Pothole repairs | New, per mile | +0.232 | -0.020 |
    | DOT pavement rating | Old, plain mean | -0.107 | +0.089 |
    | DOT pavement rating | New, length-weighted | -0.091 | +0.057 |
    | 311 pothole complaints | New, per mile | +0.248 | +0.066 |
    | Years since resurfacing | New, length-weighted | -0.032 | +0.094 |
    | Street density | The denominator alone | -0.230 | -0.010 |

    **The audit was right about the denominator.** The old rate correlates with
    street density at Spearman -0.60, and the new rate at -0.34. Street density
    ALONE reaches tau -0.230 against the gemma-4-12b score, which is nearly the
    whole size of the old pothole row. The correction cuts the gemma tau from
    +0.276 to +0.232 at this layer, and the same fall appears at all 3 layers.

    **The length weight moves the pavement rating very little.** The old and
    the new area means agree at Spearman 0.973, and the mean area moves 0.11
    rating points. The correction is right, but it changes no reading.

    **The demand side tracks the model a little better than the supply side.**
    311 complaints for each mile reach tau +0.248 with gemma-4-12b and +0.066
    with qwen3.5-9b, above the repair rate on both raters. The 2 pothole rates
    agree with each other at Spearman 0.819, thus they are not the same
    measurement.

    **No pothole rate tracks the pavement rating.** The corrected repair rate
    and the length-weighted rating agree at Spearman -0.05, and the complaint
    rate at +0.12. A pothole count and a pavement score measure different
    things at the area level, so the paper must not call them 1 proxy.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The image level does not move

    The length weight enters an AREA mean only. Section 4 gives each image the
    rating of its nearest segment, and that value carries no weight, thus the
    image numbers of this case stay exactly as they were.

    I checked it on 2026-08-31. Section 4 was run again with `_proxies_road`
    loaded, and it returned the shipped table exactly: gemma-4-12b n = 382,176,
    r = +0.0828, rho = +0.0837; qwen3.5-9b n = 382,235, r = +0.0519,
    rho = +0.0454. Both match `road_quality_unit_correlations_v1.0.0.parquet`
    to 4 decimal places.
    """
    )
    return


@app.cell
def _(mo, pr):
    mo.md(
        "### The labels that `_results_table.PROXY_LABEL` needs\n\n"
        "```python\n"
        + "\n".join(f'    "{k}": "{v}",' for k, v in pr.PROXY_LABEL.items())
        + "\n```\n\n"
        "`_results_table.py` belongs to the master table, thus this notebook "
        "prints the lines and a maintainer copies them. Without them the "
        "table prints the raw proxy name."
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## 7. Export

The next cell writes the tables to `road/outputs/`. Give the paper these files
together. The numbers mean nothing without the provenance.

The proxy table carries `proxy_metric`, `proxy_source`, and `proxy_window`.
Read them before you name a proxy in the paper.

Version 1.2.0 of the proxy export holds the whole of version 1.1.0, row for
row, and the corrected rows under it. Thus a reader can reconcile every
published number, and `pothole_repairs` stays as the context of
`pothole_repairs_per_mile`.

Warning: this upgrade touched the proxy side only. The export cell writes the
model-side tables at `MODEL_EXPORT_VERSION` and it SKIPS a file that already
exists, so a run of this notebook cannot quietly replace a shipped model score.
"""
    )
    return


@app.cell
def _(
    MODEL_EXPORT_VERSION,
    NOTEBOOK_VERSION,
    OUTPUT_PREFIX,
    Path,
    aggregated,
    correlations,
    denominator_comparison,
    mo,
    new_proxy,
    pd,
    pr,
    proxy_agg,
    table,
    unit_correlations,
):
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    # The proxy export of version 1.2.0: the whole of version 1.1.0, then the
    # corrected rows. `merge_export` raises an error when a proxy name appears
    # on both sides, because that would give a reader 2 values for 1 polygon.
    _prev = out_dir / f"{OUTPUT_PREFIX}_proxy_v1.1.0.parquet"
    if not new_proxy.empty and _prev.exists():
        proxy_export = pr.merge_export(pd.read_parquet(_prev), new_proxy)
    elif not new_proxy.empty and not proxy_agg.empty:
        proxy_export = pr.merge_export(proxy_agg, new_proxy)
    else:
        proxy_export = proxy_agg

    written = []
    skipped = []
    # `proxy` and `denominator_comparison` carry the U5 work, thus they take
    # NOTEBOOK_VERSION. The 3 model-side tables keep the version they shipped
    # with, and an existing file stays as it is.
    for _name, _frame, _ver, _guard in [
        ("proxy", proxy_export, NOTEBOOK_VERSION, False),
        ("denominator_comparison", denominator_comparison,
         NOTEBOOK_VERSION, False),
        ("aggregated", aggregated, MODEL_EXPORT_VERSION, True),
        ("correlations", correlations, MODEL_EXPORT_VERSION, True),
        ("unit_correlations", unit_correlations, MODEL_EXPORT_VERSION, True),
    ]:
        if _frame is None or _frame.empty:
            continue
        _p = out_dir / f"{OUTPUT_PREFIX}_{_name}_v{_ver}.parquet"
        if _guard and _p.exists():
            skipped.append(str(_p))
            continue
        _frame.to_parquet(_p, index=False)
        written.append(str(_p))
    if not table.empty:
        _p = (out_dir /
              f"{OUTPUT_PREFIX}_provenance_v{MODEL_EXPORT_VERSION}.csv")
        if _p.exists():
            skipped.append(str(_p))
        else:
            table.to_csv(_p, index=False)
            written.append(str(_p))

    mo.md(
        ("**Written:**\n\n" + "\n".join(f"- `{w}`" for w in written)
         if written else "**Nothing to write yet.** No frame is ready.")
        + ("\n\n**Kept as it is** (a model-side export never changes here):\n\n"
           + "\n".join(f"- `{s}`" for s in skipped) if skipped else "")
    )
    return (proxy_export,)


if __name__ == "__main__":
    app.run()
