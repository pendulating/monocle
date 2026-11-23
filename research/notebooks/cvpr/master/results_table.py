import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Master results table — validation by proxy

    1 camera-ready table over every prompt. The case sits at the first indent
    and its proxies indent under it.

    This notebook is **downstream**. Sections 1 and 2 read what each prompt
    notebook already exported. Section 3 reads the run parquets of the
    canonical registry, because a raw agreement needs the label of each
    comparison. Neither section touches W&B. Run the prompt notebooks first,
    then this one.

    **This notebook holds the canonical arm alone.** The prompt ablations
    (`noabstain`, `nocity`, `textfirst`) sit in `ablations.py` beside it,
    so the paper can present them apart from the master table.

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

    ## Warning: 2 floors run before the table prints

    | Floor | Constant | What it removes |
    |-------|----------|-----------------|
    | Areas | `R.MIN_AREAS` (20) | A row whose join keeps too few areas |
    | Coverage | `R.MIN_COVERAGE` (0.5) | A rater that keeps too few units |

    The long frame keeps every row and marks it `thin_areas`. The table drops
    a proxy row where no rater reaches the area floor, and it dashes the
    cell of a rater that does not. The POPS completion year row of the
    plazas case reaches 12 areas, thus it leaves the table.

    ## Warning: the areas with no residents are gone

    `_geography` drops the 12 joint interest areas of the community district
    layer, and the 65 non-residential NTAs. A park, an airport, or Rikers
    Island holds no household income and almost no resident crime record,
    thus a proxy value there is a boundary artifact. `load_case` drops the
    same keys from an older export.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    _here = Path(__file__).resolve().parent

    import marimo as mo
    import pandas as pd

    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _results_table as R
    from research.notebooks.cvpr import _style as style
    from research.notebooks.cvpr import _unit_validation as U

    # The gate. This notebook reads what the case notebooks exported, and those
    # exports came from the canonical registry. Stop when the registry no
    # longer matches the disk, because then the exports describe runs that
    # moved or changed.
    C.verify_or_raise()

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    OUT_DIR = _here / "outputs"
    return NOTEBOOK_VERSION, OUT_DIR, R, U, mo


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
    mo.md(r"""
    ## 1b. The same validation at the UNIT layer

    Section 1 reads the community district. That layer is a choice, and the
    choice decides the result. A district holds hundreds of facilities, thus a
    facility measure is nearly flat across districts: the `cv areas` column
    below sits at 0.02 to 0.07, while the socio-economic proxies reach 0.4 to
    0.95 over the same areas.

    This section reads the same question 1 level down. A unit is the object the
    model looked at AND the object the record grades.

    | Case | The unit | How the proxy reaches it |
    |------|----------|--------------------------|
    | Subway safety | 1 station | the entrance rolls up to its station |
    | Libraries | 1 library | FacDB carries the BIN; an exact key join |
    | Restaurants | 1 restaurant | `uid` is `unit_uid`; an exact key join |
    | Schools | 1 school | borough and school name; a name join |
    | Parks | 1 park | the unit point in the DPR polygon |
    | Plazas | 1 plaza | the capacity proxies key on the unit |
    | Road quality | 1 image | the nearest rated segment, 100 ft |

    A case can carry more than 1 proxy. `U.CASE_SPEC` names the headline in
    `proxy` and the rest in `alt_proxies`, and each proxy gets its own row.

    | Column | Meaning |
    |--------|---------|
    | `units` | Units that hold a model score AND a proxy value |
    | `rho` | Spearman, with a 95% bootstrap interval over units |
    | `tau` | Kendall tau-b, the measure section 1 reports |
    | `area tau` | The tau-b of the SAME cell at the area layer |
    | `gap tau` | The area value minus the unit value |
    | `cv units`, `cv areas` | The spread of the proxy on each layer |

    A large positive `gap tau` says that the aggregation made the agreement.

    **Note on the subway row:** the case pairs station ENTRANCES and the
    record grades a STATION, thus `U.roll_up_to_station` turns the entrance
    scores into 1 score for each station. `U.SUBWAY_SEAM` states the rule.

    **Warning: a thin rater stays here.** The master table drops qwen3.5-9b
    from Schools. This table keeps it and marks it, because the unit layer is
    where a reader asks what a heavy abstainer left behind.

    **Warning: this reads the 1,100,000-row run parquets, and road runs a
    spatial join over about 490,000 image points.** It takes about 2.5 minutes.
    """)
    return


@app.cell
def _(U, layer_pick):
    units_long = U.build()
    units_paired = U.with_area(units_long, layer=layer_pick.value)
    U.to_display(units_paired)
    return units_long, units_paired


@app.cell
def _(U, mo, units_long):
    units_rec = U.reconcile(units_long)
    mo.vstack([
        mo.md("**Does this reproduce what the case notebooks published?** "
              "Six cases export a unit correlation. A `True` in "
              "`agrees` says the count and the rho both match."),
        units_rec,
    ])
    return (units_rec,)


@app.cell
def _(U, layer_pick, mo, units_paired):
    unit_latex = U.to_latex(units_paired, layer=layer_pick.value)
    mo.md(f"```latex\n{unit_latex}\n```")
    return (unit_latex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    """)
    return


@app.cell
def _(R):
    raw = R.raw_cross_model()
    raw_table = R.raw_to_display(raw)
    raw_table
    return (raw,)


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
    mo.md(r"""
    ## 4. LaTeX

    Paste this straight into the paper. It regenerates, so do not edit it by
    hand — change `_results_table.py` instead.
    """)
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
    return (latex,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the table files")
    mo.md(f"## 5. Export\n\n{save}")
    return (save,)


@app.cell
def _(
    NOTEBOOK_VERSION,
    OUT_DIR,
    R,
    U,
    latex,
    layer_pick,
    mo,
    raw,
    raw_latex,
    results,
    save,
    table,
    units_long,
    units_paired,
    units_rec,
):
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
        # The ablation files come from `ablations.py`, not from here.
        # The unit-layer table. The unit rows carry no layer, thus their stem
        # drops it; the paired rows carry one, thus their stem holds it.
        _u = U.export(units_long, units_paired, OUT_DIR,
                      version=NOTEBOOK_VERSION, layer=layer_pick.value,
                      reconciliation=units_rec)
        _msg = mo.md(
            f"Wrote 4 files with the stem `{_stem}` and 2 files with the "
            f"stem `{_raw_stem}` to `master/outputs/`. "
            f"The unit layer wrote {len(_u)}: `{'`, `'.join(_u)}`."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
