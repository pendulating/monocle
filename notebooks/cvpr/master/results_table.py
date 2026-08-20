import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Master results table — validation by proxy

    1 camera-ready table over every prompt. The case sits at the first indent
    and its proxies indent under it.

    This notebook is **downstream**. Sections 1 and 2 read what each prompt
    notebook already exported. Section 3 reads the run parquets of the
    canonical registry, because a raw agreement needs the label of each
    comparison. Neither section touches W&B. Run the prompt notebooks first,
    then this one.

    ## The 3 columns

    | Column | Meaning | Chance |
    |--------|---------|--------|
    | agr. | Share of units on the same side of BOTH medians | 0.50 |
    | $r$ | Pearson correlation | 0 |
    | $\tau$ | Kendall tau-b, the order-sensitive measure | 0 |

    **agr. and $\tau$ answer different questions.** Agreement splits each series
    at its own median and asks about a side, so it needs no shared scale and it
    survives any monotone distortion — but it throws away the size of a gap.
    Kendall $\tau$ counts concordant pairs, so it reads the whole ordering:
    $\tau = 0.2$ means a random pair of units is ordered the same way 60% of the
    time. A high agreement beside a low $\tau$ means the model finds the good
    half but cannot rank inside it.

    **Read $r$ beside $\tau$, never alone.** Pearson assumes a linear relation,
    and income is right-skewed, so the two part company on the income rows.

    ## Warning: every proxy already points the same way

    `_proxies` applies the sign BEFORE it exports. Crime density and pothole
    repairs are negated in the parquet, and the restaurant inspection score is
    flipped. Thus a **positive number always means agreement**, on every row,
    and this notebook applies no sign of its own.

    A negative crime row is therefore a real disagreement, not a sign artefact.

    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `notebooks/cvpr/`, because every
    # folder here uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import pandas as pd

    import _canonical as C
    import _results_table as R
    import _style as style

    # The gate. This notebook reads what the case notebooks exported, and those
    # exports came from the canonical registry. Stop when the registry no
    # longer matches the disk, because then the exports describe runs that
    # moved or changed.
    C.verify_or_raise()

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    OUT_DIR = _here / "outputs"
    return C, NOTEBOOK_VERSION, OUT_DIR, R, mo, pd, style


@app.cell(hide_code=True)
def _(R, mo):
    layer_pick = mo.ui.radio(
        options=list(R.LAYER_KEY), value=R.DEFAULT_LAYER, label="Geography layer"
    )
    star_p = mo.ui.slider(
        0.01, 0.10, value=0.05, step=0.01, label="Star Kendall tau below p",
        show_value=True,
    )
    full_width = mo.ui.checkbox(
        value=True, label="Full width (table*, spans the 2 columns)"
    )
    label_in = mo.ui.text(value=R.DEFAULT_LABEL, label="LaTeX label", full_width=True)
    caption_in = mo.ui.text_area(
        value="", label="Caption (empty = the default in _results_table.CAPTION)",
        full_width=True,
    )
    mo.md(
        f"""
    ## 1. Controls

    {layer_pick}
    {star_p}
    {full_width}
    {label_in}
    {caption_in}

    **Community district is the default for a reason.** It is the only layer
    with a usable $n$ for every prompt. The libraries case keeps 48 community
    districts but just 1 census tract, because 3 libraries almost never share a
    tract.
    """
    )
    return caption_in, full_width, label_in, layer_pick, star_p


@app.cell
def _(R, layer_pick):
    results = R.build(layer=layer_pick.value)
    missing = [
        c["label"] for c in R.CASES
        if R.load_case(c, layer=layer_pick.value) is None
    ]
    return missing, results


@app.cell(hide_code=True)
def _(layer_pick, missing, mo, results):
    mo.md(
        f"""
    ## 2. The table

    **{results['case'].nunique()}** prompts, **{len(results) // 2}** proxy rows,
    at the `{layer_pick.value}` layer.
    """
        + (
            f"\n\n**Warning: {len(missing)} prompt(s) exported nothing at this "
            f"layer and are absent: {', '.join(missing)}.** Run their "
            "validation notebook, or read a layer they cover."
            if missing else ""
        )
        if not results.empty
        else "**Nothing to show.** No prompt has exported at this layer. Run the "
             "validation notebooks first."
    )
    return


@app.cell
def _(R, results):
    table = R.to_display(results)
    table
    return (table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Raw cross-model agreement

    The `x-model` block above compares the 2 models AFTER the aggregation into
    polygons. That step averages hundreds of comparisons into 1 polygon score,
    and an average of noise moves toward the middle. Thus 2 raters that
    disagree on almost every comparison can still track each other across
    areas, and the block reads higher than the raters deserve.

    This section removes that help. It reads the label of each comparison from
    the canonical registry and asks how often the 2 models write the same label
    on the same pair. The pair id carries the presentation order, thus the 2
    models saw the identical pair in the identical order.

    | Column | Meaning | Chance |
    |--------|---------|--------|
    | pairs | Comparisons that both models answered or declined | -- |
    | abstain | Share of pairs where the model wrote `NotSure` | -- |
    | label agr | Same label of the 6, over every pair | the marginals |
    | label kappa | The same, corrected for chance | 0 |
    | both answer | Pairs where neither model abstained | -- |
    | dir agr | Same side, over the pairs that both answered | the marginals |
    | dir kappa | The same, corrected for chance | 0 |

    **Read the kappa, not the agreement.** A model that abstains on most pairs
    agrees with another abstainer by accident, so the raw share rewards
    silence. Cohen $\kappa$ takes that away.

    **The direction columns carry a floor.** They need pairs that both models
    answered, and a mean over a handful of them describes the selection and
    not the case. A row under `R.MIN_BOTH` answered pairs thus shows a dash.
    The schools row is the one: qwen3.5-9b abstains on 99.7% of the pairs, so
    277 pairs of 110,000 remain. The label columns take no floor, because they
    run over every pair.

    **Warning: this section reads the run parquets.** The rest of the notebook
    reads only the exports of the case notebooks. The read goes through the
    canonical registry, never through W&B, and the gate above already checked
    it.

    **Warning: the $n$ here is not the $n$ above.** A row counts every pair of
    the run, and the table above keeps an area only where the proxy exists.
    Parks and plazas come from 1 run, thus they share 1 row.
    """
    )
    return


@app.cell
def _(R):
    raw = R.raw_cross_model()
    raw_table = R.raw_to_display(raw)
    raw_table
    return raw, raw_table


@app.cell
def _(R, mo, raw):
    # The raw table carries its own caption and label, in `_results_table.py`.
    # It needs no control here: it does not move with the geography layer,
    # because a comparison sits in no polygon.
    raw_latex = R.raw_to_latex(raw)
    mo.md(f"```latex\n{raw_latex}\n```")
    return (raw_latex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. LaTeX

    Paste this straight into the paper. It regenerates, so do not edit it by
    hand — change `_results_table.py` instead.
    """
    )
    return


@app.cell
def _(R, caption_in, full_width, label_in, layer_pick, mo, results, star_p):
    latex = R.to_latex(
        results, layer=layer_pick.value, star_p=float(star_p.value),
        caption=(caption_in.value.strip() or None),
        label=label_in.value.strip() or R.DEFAULT_LABEL,
        full_width=bool(full_width.value),
    )
    mo.md(f"```latex\n{latex}\n```")
    return (latex)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the table files")
    mo.md(f"## 5. Export\n\n{save}")
    return (save,)


@app.cell
def _(NOTEBOOK_VERSION, OUT_DIR, R, latex, layer_pick, mo, raw, raw_latex,
      results, save, table):
    if save.value:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _stem = f"results_table_{layer_pick.value}_v{NOTEBOOK_VERSION}"
        (OUT_DIR / f"{_stem}.tex").write_text(latex)
        table.to_csv(OUT_DIR / f"{_stem}.csv", index=False)
        # The long frame keeps tau_p and the proxy keys, which the display
        # table drops. Give the paper both.
        results.to_csv(OUT_DIR / f"{_stem}_long.csv", index=False)
        R.wide(results).to_csv(OUT_DIR / f"{_stem}_wide.csv")
        # The raw rows carry no layer, thus their stem drops it.
        _raw_stem = f"raw_cross_model_v{NOTEBOOK_VERSION}"
        (OUT_DIR / f"{_raw_stem}.tex").write_text(raw_latex)
        raw.to_csv(OUT_DIR / f"{_raw_stem}.csv", index=False)
        _msg = mo.md(
            f"Wrote 4 files with the stem `{_stem}` and 2 files with the stem "
            f"`{_raw_stem}` to `master/outputs/`."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
