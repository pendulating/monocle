"""Validation via statistical proxy — the road quality case.

This notebook is load-bearing for the CVPR paper. It reads every canonical
urbanpairvqa run of the road-quality prompt, scores each street-level image, and
compares the scores against 4 proxies.

**This case is IMAGE mode.** Road quality pairs random street-level shots, not
curated units, so `pairs.parquet` holds no `unit_uid` and there is no unit
table. The notebook scores each IMAGE and places it by its own coordinates.

The DOT pavement rating scores the same thing the model scores, so the notebook
compares them image by image, and also by area.

Run it:
    .venv-nightly/bin/marimo edit notebooks/cvpr/road_quality_validation.py

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

    # The shared modules sit in the parent, `notebooks/cvpr/`, because every
    # prompt folder uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import _geography as geo
    import _provenance as prov
    import _proxies as px

    return Path, geo, gpd, mo, pd, prov, px


@app.cell
def _():
    # Raise NOTEBOOK_VERSION when you change how a number is computed.
    NOTEBOOK_VERSION = "1.0.0"
    CASE = "road_quality"
    OUTPUT_PREFIX = "road_quality"
    RECIPE_DIR = "road"
    # No filter. 1 keeps every polygon that holds an image.
    MIN_UNITS_PER_POLYGON = 1
    PAVEMENT_SINCE_YEAR = 2024
    # A camera sits in the roadway, so a small cap is right. A large cap would
    # attach the rating of another street.
    MAX_SEGMENT_DISTANCE_FT = 100.0
    return (
        CASE,
        MAX_SEGMENT_DISTANCE_FT,
        MIN_UNITS_PER_POLYGON,
        NOTEBOOK_VERSION,
        OUTPUT_PREFIX,
        RECIPE_DIR,
    )


@app.cell
def _(NOTEBOOK_VERSION, geo, mo, prov, px):
    mo.md(
        f"""
    # Road quality — validation via statistical proxy

    **Notebook version {NOTEBOOK_VERSION}** · provenance {prov.__version__} ·
    geography {geo.__version__} · proxies {px.__version__}

    The model answers "which roadway is in better condition". DOT answers the
    same question about the same streets, so this notebook compares them image
    by image. Potholes, income, and crime give more context.
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
    mo.md("## 3. The 4 proxies")

    recipe = px.load_recipe(RECIPE_DIR)
    mo.md(
        f"""
    `{RECIPE_DIR}/recipe.json` names {len(recipe)} sources.

    | Proxy | Measures | Orientation |
    |-------|----------|-------------|
    | DOT pavement rating | The street itself | 0 to 10, high is good |
    | Pothole repairs | The street, indirectly | Negated, so high is good |
    | Median household income | The area | High is wealthy |
    | NYPD complaints | The area | Negated, so high is safe |

    **Read the pavement rating as the result of this case.** It is the only
    proxy that scores the same object the model scores.

    ### Warning: a rating of 0.0 means NOT RATED

    72,768 of the 514,521 rows carry `systemrating = 0.0`. It does not mean the
    worst pavement. Scored as a zero it would drag whole neighborhoods down and
    look like a real signal. The zeros fall from 11,043 in 2021 to 1,358 in
    2026 while the rated rows hold steady, which is a backlog that clears, not
    pavement that improves. `pavement_segments` drops them.

    ### Warning: the pothole proxy counts REPAIRS, not defects

    A borough that fixes potholes quickly looks worse here than one that ignores
    them. Read it beside the pavement rating, never alone.
    """
    )
    return (recipe,)


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
def _(geo, mo, pd, px, segments):
    mo.md("## 5. Area comparison")

    _frames = []
    for _layer in geo.LAYERS:
        for _name, _fn in [
            ("dot_pavement_rating", lambda L: px.pavement_by_layer(L, segments=segments)),
            ("pothole_repairs", px.pothole_by_layer),
            ("median_household_income", px.income_by_layer),
            ("crime_density", px.crime_by_layer),
        ]:
            _out = _fn(_layer)
            if _out is not None and not _out.empty:
                _out = _out.copy()
                _out["proxy"] = _name
                _frames.append(_out)
    proxy_agg = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame()
    return (proxy_agg,)


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
        """
## 6. Export

The next cell writes the tables to `notebooks/cvpr/outputs/`. Give the paper
these files together. The numbers mean nothing without the provenance.
"""
    )
    return


@app.cell
def _(
    NOTEBOOK_VERSION,
    OUTPUT_PREFIX,
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
