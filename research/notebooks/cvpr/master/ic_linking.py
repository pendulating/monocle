import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Does more reasoning make the model more right?

    The IC report says how complex each trace is. The validation notebooks say
    whether the model agrees with an outside measurement. This notebook joins
    the 2 at the level of a single PAIR.

    ## Where the join happens

    There is no per-unit proxy. `_proxies` puts each proxy on the map and
    aggregates it to a polygon, and `_geography` does the same for the model
    units. The 2 sides meet at the **polygon**, thus the ground truth of a pair
    is:

    > the polygon of unit A holds a higher proxy value than the polygon of unit B

    ```
    proxy_gap = proxy(polygon of A) - proxy(polygon of B)
    model_dir = sign of relative_score, which already states A against B
    agrees    = the 2 signs match
    ```

    A pair whose 2 units sit in 1 polygon carries no difference and drops out,
    and so does an abstention or a `Same`. About half of the traces survive,
    and the coverage table below says exactly how many.

    **Warning: the target is coarse.** A unit can differ from its polygon, thus
    a pair can be marked wrong when the model read the 2 buildings correctly
    and the 2 neighbourhoods disagree. Read an agreement rate as a floor.

    ## The confound you must not skip

    A model reasons longest about a HARD pair, and a hard pair is one where the
    2 areas sit close together. Complexity and difficulty move together, thus a
    raw comparison of "code 5 against code 3" can measure the difficulty alone.

    Every table here splits the pairs into bands by the SIZE of the proxy gap
    and compares the codes INSIDE a band. The banded table is the answer; the
    table by code is context.

    **The code and the judgment come from the same run.** This shows an
    association and never a cause.

    ## Which cases

    Only the 5 unit-mode cases. Road quality and street photography pair random
    images, thus a pair holds no unit to place in a polygon.
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import pandas as pd

    import _canonical as C
    import _ic
    import _ic_link as LK
    import _style as style

    # The gate: the corpus must come from the registered thinking runs.
    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    FIG_DIR = _here / "figures"
    return C, FIG_DIR, LK, NOTEBOOK_VERSION, _ic, mo, pd, style


@app.cell(hide_code=True)
def _(C, mo):
    mo.md(f"**Canonical registry:** {C.summary()}")
    return


@app.cell
def _(_ic, mo):
    try:
        raw = _ic.load()
        load_error = ""
    except (FileNotFoundError, RuntimeError) as _exc:
        raw, load_error = None, str(_exc)

    mo.md(
        f"**{len(raw):,} ingredient rows over {raw.doc_id.nunique():,} pair ids**"
        if load_error == ""
        else f"**No usable corpus.**\n\n```\n{load_error}\n```"
    )
    return load_error, raw


@app.cell
def _(_ic, load_error, raw):
    codes = None if load_error else _ic.codes(raw)
    return (codes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. The 3 controls

    | Control | What to look for |
    |---------|------------------|
    | `layer` | The polygon the join runs on. A finer layer holds fewer cross-polygon pairs |
    | `split_code` | The rung that counts as complex. 5 is the first with a justified weighing |
    | `bands` | How many difficulty bands. More bands control the gap harder and thin each cell |
    """
    )
    return


@app.cell
def _(LK, mo):
    layer_pick = mo.ui.radio(options=["community_district", "nta", "census_tract"],
                             value=LK.DEFAULT_LAYER, label="Geography layer")
    split_pick = mo.ui.slider(3, 6, value=5, step=1, label="Complex means code >=",
                              show_value=True)
    band_pick = mo.ui.slider(2, 6, value=4, step=1, label="Difficulty bands",
                             show_value=True)
    mo.hstack([layer_pick, split_pick, band_pick])
    return band_pick, layer_pick, split_pick


@app.cell
def _(LK, band_pick, codes, layer_pick, load_error, split_pick):
    if load_error:
        summary, tables = None, {}
    else:
        summary, tables = LK.link_all(
            codes, layer=layer_pick.value,
            split_code=int(split_pick.value), quantiles=int(band_pick.value),
        )
    return summary, tables


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Coverage and the headline

    `kept` is the share of traces that reach the test. `agreement` is the share
    of those pairs the model put on the same side as the proxy; chance is 0.50.
    `pooled_difference` is the answer: the agreement of the complex traces minus
    the agreement of the simple ones, measured inside a difficulty band and
    pooled over the bands.

    A `pooled_difference` near 0 means the reasoning structure does not predict
    whether the model matches the outside measurement.
    """
    )
    return


@app.cell
def _(summary):
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Inside 1 case

    The band table holds the interval, so a difference can be read against the
    noise. A Wilson interval is used, because a cell can be small.
    """
    )
    return


@app.cell
def _(mo, tables):
    case_pick = mo.ui.dropdown(options=sorted(tables), label="Case and proxy",
                               value=sorted(tables)[0] if tables else None)
    case_pick
    return (case_pick,)


@app.cell
def _(LK, band_pick, case_pick, split_pick, tables):
    picked = tables.get(case_pick.value) if case_pick.value else None
    bands = (LK.agreement_by_gap(picked, quantiles=int(band_pick.value),
                                 split_code=int(split_pick.value))
             if picked is not None else None)
    bands
    return bands, picked


@app.cell
def _(LK, band_pick, picked, split_pick):
    fig = (LK.plot_agreement_by_gap(picked, quantiles=int(band_pick.value),
                                    split_code=int(split_pick.value))
           if picked is not None and not picked.empty else None)
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. The table by code, for context only

    Read the median gap in this table. When it rises with the code, the model
    reasoned more about the harder pairs, and a raw comparison across codes
    would measure the difficulty.
    """
    )
    return


@app.cell
def _(LK, picked):
    by_code = LK.agreement_by_code(picked) if picked is not None else None
    by_code
    return (by_code,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Export

    The next cell writes the summary, the bands of every case, and 1 figure for
    each case and proxy.
    """
    )
    return


@app.cell
def _(FIG_DIR, LK, band_pick, mo, split_pick, summary, tables):
    written = ([] if summary is None else
               LK.export(summary, tables, FIG_DIR,
                         split_code=int(split_pick.value),
                         quantiles=int(band_pick.value)))
    mo.md("**Written:**\n\n" + "\n".join(f"- `{FIG_DIR / w}`" for w in written)
          if written else "**Nothing to write.**")
    return (written,)


if __name__ == "__main__":
    app.run()
