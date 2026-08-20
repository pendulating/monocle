import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Integrative Complexity over the reasoning traces

    How complex is the reasoning the model wrote while it compared 2
    photographs? This notebook reads the **ingredient corpus** and gives each
    trace a code.

    ## The 2 steps, and why they are apart

    | Step | Where | Cost of a change |
    |------|-------|------------------|
    | ingredients | `ic_extract` on the GPU | 59 GPU-hours |
    | codes | `dagspaces/common/ic_codes.py` | seconds |

    The extractor never states a code. It returns SPANS: the dimensions the
    trace names, the perspectives it develops, the weighings it makes, and the
    justification under each one. A code is derived from those spans here, thus
    a threshold moves without a second run.

    ## The scale

    | Code | What the trace shows |
    |------|----------------------|
    | 1 | 1 view. It names cues and decides |
    | 2 | a hedge or a set-aside alternative, but no second view |
    | 3 | 2 or more views, held apart |
    | 4 | the views meet, but nothing justifies the link |
    | 5 | a justified weighing relates them |
    | 6 | the weighing holds under a named condition |

    **Warning: read this as 1 to 6.** Code 7 of the codebook needs an
    organizing principle above the integrations, and the schema holds no
    ingredient for it. "No trace reached 7" is a fact about the schema, not
    about the model.

    ## What counts

    Only a **located** span counts. A quote that no search finds in the trace is
    a defect of the extractor, thus it lifts no code. A justification counts
    only when its own sub-quote is located as well.

    A trace whose answer the token cap CUT is dropped from every rate: an
    absence inside a cut answer is not a zero.
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `notebooks/cvpr/`. This notebook
    # lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import pandas as pd

    import _canonical as C
    import _ic
    import _style as style

    # The gate. The corpus is 1 step downstream of the thinking runs, thus it
    # must come from the runs the registry names.
    C.verify_or_raise()

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    OUT_DIR = _here / "outputs"
    FIG_DIR = _here / "figures"
    return C, FIG_DIR, NOTEBOOK_VERSION, OUT_DIR, _ic, mo, pd, style


@app.cell(hide_code=True)
def _(C, mo):
    mo.md(f"**Canonical registry:** {C.summary()}")
    return


@app.cell
def _(_ic, mo):
    # A stale corpus stops the notebook here, with the file names in the
    # message. It does NOT fall back to an older corpus in silence.
    try:
        raw = _ic.load()
        load_error = ""
    except (FileNotFoundError, RuntimeError) as _exc:
        raw, load_error = None, str(_exc)

    mo.md(
        f"**{len(raw):,} ingredient rows over {raw.doc_id.nunique():,} traces**, "
        f"cases: {', '.join(sorted(raw.case.unique()))}"
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
    ## 1. Is the corpus sound?

    Read this table before any number below it. `quote_found` is the share of
    spans a search really finds in the trace. `answers_cut` is the share of
    traces the token cap cut; those traces are dropped from every rate.
    """
    )
    return


@app.cell
def _(_ic, codes, load_error, raw):
    quality = None if load_error else _ic.quality_table(raw, codes)
    quality
    return (quality,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. The code of each case

    `differentiated`, `integrated`, and `context_sensitive` are the components
    the code rests on. Read them beside the mean: 2 cases can share a mean and
    reach it by different rungs.
    """
    )
    return


@app.cell
def _(_ic, codes, load_error):
    by_case = None if load_error else _ic.case_table(codes)
    by_case
    return (by_case,)


@app.cell
def _(_ic, codes, load_error):
    mix = None if load_error else _ic.plot_code_mix(codes)
    mix
    return (mix,)


@app.cell
def _(_ic, codes, load_error):
    components = None if load_error else _ic.plot_components(codes)
    components
    return (components,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Does the code move with the judgment?

    This is the first test of the whole pipeline. A `Same` and a `NotSure` are
    the hard pairs, thus they should cost more reasoning than a `MuchMore`. A
    flat line means the code measures nothing that the judgment knows about.

    A cell under 20 traces is left out: a mean over 2 traces is noise.
    """
    )
    return


@app.cell
def _(_ic, codes, load_error):
    by_label = None if load_error else _ic.label_table(codes)
    by_label
    return (by_label,)


@app.cell
def _(_ic, codes, load_error):
    label_fig = None if load_error else _ic.plot_code_by_label(codes)
    label_fig
    return (label_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3b. What each part of the reasoning sounds like

    A word block over the located SPANS, not over whole traces. The weights use
    the same `distinctive` score as the trace clouds: a word grows when this
    block uses it and the other blocks do not.

    A weighing should read as a weighing ("usually", "better", "difference",
    "considered", "wins"), and a reconsideration as a turn back ("examine",
    "closer", "check", "double"). When 2 blocks read alike, the schema is not
    separating them, and the codes above rest on that separation.
    """
    )
    return


@app.cell
def _(load_error, raw):
    import _ic_words as W
    import _traces as T

    if load_error:
        block_counts, block_sizes = {}, None
    else:
        block_counts, block_sizes = W.counts_by_group(
            raw, T.default_stopwords(), group_by="type")
    block_sizes
    return T, W, block_counts, block_sizes


@app.cell
def _(W, block_counts, mo):
    block_pick = mo.ui.dropdown(
        options=W.ordered_blocks(block_counts, "type") if block_counts else [],
        value="weighing" if "weighing" in block_counts else None,
        label="Ingredient type",
    )
    block_pick
    return (block_pick,)


@app.cell
def _(W, block_counts, block_pick):
    block_fig = (W.plot_block(block_counts, block_pick.value)
                 if block_counts and block_pick.value else None)
    block_fig
    return (block_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. The thresholds

    Every number a code depends on. Change one in `ic_codes.Thresholds` and run
    this notebook again; the GPU is not involved.
    """
    )
    return


@app.cell
def _(load_error, pd):
    from dagspaces.common import ic_codes as IC

    thresholds = None if load_error else pd.DataFrame([IC.thresholds_used()])
    thresholds
    return IC, thresholds


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Export

    The next cell writes the tables, the per-trace codes, and the figures. Give
    the paper all of them together: a mean code means nothing without the
    quality table beside it.
    """
    )
    return


@app.cell
def _(FIG_DIR, _ic, codes, load_error, mo, raw):
    written = [] if load_error else (
        _ic.export(raw, codes, FIG_DIR) + W.export(raw, FIG_DIR))
    mo.md(
        "**Written:**\n\n" + "\n".join(f"- `{FIG_DIR / w}`" for w in written)
        if written
        else "**Nothing to write.** No usable corpus."
    )
    return (written,)


if __name__ == "__main__":
    app.run()
