import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The proxy overview — 1 figure and 2 tables

    The paper has little space, thus this notebook folds the validation by
    proxy, the geography layer, the prompt ablations, and the cross-model
    agreement into 1 full-width figure and 2 one-column tables.

    | Question | Where it sits |
    |----------|---------------|
    | Does the model agree with the proxy? | Panel A, 1 marker for each layer |
    | Does the geography layer change that? | Panel A, the line between the markers |
    | Does an ablation align the model more or less with the proxy? | Panel B, the heatmap |
    | Do the 2 models agree with each other? | The cross-model table |
    | Which value does an ablation move? | The family table |

    This notebook is **downstream**. It reads the exports of
    `results_table.py` (the 3 layers) and `ablations.py` (the proxy re-run
    of the arms) in `master/outputs/`. Run those first. The raw cross-model
    rows read the run parquets of the canonical registry, never W&B.

    **The export writes into the paper repo.** The figure goes to
    `figures/proxy_overview.{pdf,png}` and the tables to
    `tables/cross-model.tex` and `tables/ablation-by-family.tex`, thus the
    paper reads them in place.

    ## The value families

    A proxy stands for an attitude, and the ablation question is "does the
    prompt change align the model more with that attitude?". `FAMILY` in
    `_proxy_overview.py` names the family of each proxy. Edit it there.

    | Glyph | Family | Proxies |
    |-------|--------|---------|
    | D | Disorder (broken windows) | crime, transit crime, building violations, potholes |
    | A | Affluence and investment | income, alteration permits, BID membership, economic need |
    | R | Official condition rating | DOT pavement rating, PIP, teacher survey, DOHMH score |
    | V | Vintage | construction year, acquisition year, POPS completion year |
    | c | Context | ridership, park acreage, attendance, outdoor dining |

    A context row describes the place and makes no validation claim, thus
    the family table leaves it out.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `research/notebooks/cvpr/`, because every
    # folder here uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo

    import _canonical as C
    import _proxy_overview as O
    import _style as style

    # The gate. The raw rows read the run parquets of the registry. Stop when
    # the registry no longer matches the disk.
    C.verify_or_raise()

    style.apply_house_style()
    return O, mo


@app.cell(hide_code=True)
def _(O, mo):
    arm_layer = mo.ui.radio(
        options=[k for k, _ in O.LAYERS], value="community_district",
        label="Layer of the ablation panel (needs a proxy_arms export at that layer)",
    )
    mo.md(f"## 1. Controls\n\n{arm_layer}")
    return (arm_layer,)


@app.cell
def _(O, arm_layer):
    long = O.load_long()
    arms = O.load_arms(layer=arm_layer.value)
    single = O.load_single()
    rows = O.table_rows(long)
    return arms, long, rows, single


@app.cell(hide_code=True)
def _(mo, rows):
    mo.md(
        f"""
    ## 2. The figure

    **{rows['case'].nunique()}** cases and **{len(rows)}** proxy rows, the
    rows of the master table. Panel A reads the 3 layers of
    `results_table.py`, and the single-image reads of Gemma from
    `_single_image_read.py` when its export exists. Panel B reads the
    proxy re-run of the arms.
    """
    )
    return


@app.cell
def _(O, arm_layer, arms, long, single):
    fig = O.plot_overview(long, arms, single=single,
                          arm_layer_label=O.LAYER_SHORT[arm_layer.value])
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The cross-model table

    The raw columns read the label of each comparison from the canonical
    registry, for all 7 cases. Schools stays, with a dash on the side
    kappa, because qwen3.5-9b answers too few pairs there. Street
    photography has no proxy join, thus it has raw columns alone.

    The layer columns take the cross-model tau of the master table at each
    layer, from the proxy with the most areas. Parks and plazas share 1
    run and 2 folders, thus the raw cells print once and the layer cells
    print on 2 lines.

    **Warning: this reads the 1,100,000-row run parquets.** It takes about
    a minute.
    """)
    return


@app.cell
def _(O, long):
    xm = O.cross_model_table(long)
    xm
    return (xm,)


@app.cell
def _(O, mo, xm):
    xm_tex = O.cross_model_to_latex(xm)
    mo.md(f"```latex\n{xm_tex}\n```")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The family table

    The mean paired change of tau at the layer of panel B, by value family,
    model, and arm. A positive value aligns the model MORE with the
    family. `n_sig` counts the cells whose bootstrap interval excludes 0.
    """)
    return


@app.cell
def _(O, arms):
    fam = O.family_table(arms)
    fam
    return (fam,)


@app.cell
def _(O, fam, mo):
    fam_tex = O.family_to_latex(fam)
    mo.md(f"```latex\n{fam_tex}\n```")
    return


@app.cell(hide_code=True)
def _(O, mo):
    mo.md(
        "## 5. The figure float\n\nPaste this into a section, or `\\input` "
        "the file the export writes.\n\n```latex\n" + O.figure_include() + "\n```"
    )
    return


@app.cell(hide_code=True)
def _(O, mo):
    save = mo.ui.run_button(label="Write the files")
    mo.md(
        f"## 6. Export\n\n{save}\n\nWrites to `master/outputs/` and to the "
        f"paper repo at `{O.PAPER_DIR}`."
    )
    return (save,)


@app.cell
def _(O, arms, fam, fig, long, mo, save, single, xm):
    if save.value:
        _written = O.export(long, arms, fig, xm, fam, single=single)
        _msg = mo.md("Wrote:\n\n" + "\n".join(f"- `{w}`" for w in _written))
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
