import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Perspectives and the vote

    Version 3 of the extraction schema puts the spans on the Integrative
    Complexity (IC) codebook. A `perspective` span is the evaluative standard
    the model applies ("maintenance", "standard", "prestige"). It carries the
    image it counts FOR (`favors`), and the extractor says whether the
    decision follows it (`wins`).

    This notebook asks: which perspectives go with a win of the image they
    count for, and which go with a loss? It asks the same of the `inference`
    spans (by name and `kind`) and the `dimension` spans (by name and
    `valence`).

    ## The arithmetic

    The test is the directional test of `_cues_across.keyed_lift`:

    | Class | The image the span counts for | The key |
    |-------|-------------------------------|---------|
    | perspective | `favors` | the name |
    | inference | `image` | `name \| kind` |
    | dimension | `image` | `name \| valence` |

    1 vote for each (trace, key, image). A key on both images in 1 trace is
    dropped. `Same` and `NotSure` do not vote. The win rate is shrunk toward
    the base rate of the case by 25 pseudo votes, and the **lift** is the
    shrunk rate less the base rate. The base rate is the win rate of every
    vote of the class in the case. The interval is the Wilson interval of
    the raw rate.

    **Warning: the base rate of a perspective is high, 0.79 to 0.95 on the
    primary gemma arm.** The model names a standard for the image it then
    picks. Thus a lift near 0 is a perspective that goes with the winner as
    often as any other perspective, and a negative lift is a standard the
    model names for an image and then overrules.

    **Warning: the names are the model's own phrases.** `_ic_v3` joins the
    spelling variants only (case, hyphen, article, degree adverb, plural).
    `maintenance` and `upkeep` stay 2 keys. A key needs the count floor.

    **Warning: this is an association, not a cause.**
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `research/notebooks/cvpr/`.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import pandas as pd

    import _canonical as C
    import _extractions as X
    import _ic_v3 as I
    import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.1.0"
    FIG_DIR = _here / "figures"
    # The LaTeX tables go to `outputs/`, never to `figures/`.
    OUT_DIR = _here / "outputs"
    return C, FIG_DIR, I, NOTEBOOK_VERSION, OUT_DIR, X, mo


@app.cell(hide_code=True)
def _(C, I, X, mo):
    allow_stale = mo.ui.checkbox(
        value=False,
        label="Read a corpus the registry does not name (a look only, "
              "never for the paper)",
    )
    _arms = I.present_arms()
    _default = X.PRIMARY_ARMS["gemma-4-12b"]
    arm_pick = mo.ui.dropdown(
        options={I.arm_label(a): a for a in _arms},
        value=I.arm_label(_default if _default in _arms else _arms[0]),
        label="Arm",
    )
    mo.md(
        f"""
    ## 1. The gate and the corpus

    **Canonical registry:** {C.summary()}

    The primary arm of each judge is the default: the gemma-4-12b traces read
    by Qwen3.8-27B, and the qwen3.5-9b traces read by Gemma-4-31B.

    {arm_pick}

    {allow_stale}
    """
    )
    return allow_stale, arm_pick


@app.cell
def _(I, allow_stale, arm_pick, mo):
    arm = arm_pick.value
    try:
        corpus = I.load_arm(arm, require_canonical=not allow_stale.value)
        gate_ok, gate_error = not allow_stale.value, ""
    except (RuntimeError, FileNotFoundError) as _exc:
        corpus, gate_ok, gate_error = None, False, str(_exc)
    if corpus is None:
        _msg = f"**No usable corpus.**\n\n```\n{gate_error}\n```"
    else:
        _msg = (f"**{I.arm_label(arm)}.** {len(corpus):,} spans, "
                f"{corpus['tid'].nunique():,} traces."
                + ("" if gate_ok else " **The gate is OFF**: every file "
                   "takes a `_stale` suffix."))
    mo.md(_msg)
    return arm, corpus, gate_ok


@app.cell(hide_code=True)
def _(I, corpus, mo):
    (mo.vstack([mo.md("### What the corpus holds"), I.coverage(corpus)])
     if corpus is not None else mo.md("_No corpus._"))
    return


@app.cell(hide_code=True)
def _(mo):
    min_count = mo.ui.slider(10, 200, value=25, step=5, show_value=True,
                             label="Fewest decided votes for a key")
    n_ends = mo.ui.slider(3, 10, value=5, step=1, show_value=True,
                          label="Keys at each end of a case")
    min_cases = mo.ui.slider(2, 7, value=3, step=1, show_value=True,
                             label="Fewest cases for a key that holds its sign")
    mo.hstack([min_count, n_ends, min_cases], gap=2)
    return min_cases, min_count, n_ends


@app.cell
def _(I, arm, corpus, min_count):
    bundle = ({} if corpus is None else
              I.arm_bundle(corpus, arm, min_count=int(min_count.value),
                           questions=("perspectives",)))
    return (bundle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The perspectives

    1 panel for each case: the keys of highest and of lowest lift. The count
    in brackets is the decided votes. `wins_yes_share` in the table is the
    share of the rows where the extractor wrote `wins = yes`.
    """)
    return


@app.cell
def _(I, arm, bundle, mo, n_ends):
    persp_ends = I.case_ends(bundle.get("lift_perspective", I.pd.DataFrame()),
                             int(n_ends.value))
    (mo.vstack([I.captioned(I.plot_case_ends(persp_ends, "perspective", arm=arm)),
                mo.accordion({"The table": persp_ends})])
     if not persp_ends.empty else mo.md("_No key over the floor._"))
    return


@app.cell
def _(I, arm, bundle, min_cases, mo):
    _tables = bundle.get("lift_perspective", I.pd.DataFrame())
    persp_held = I.consistent_keys(_tables, int(min_cases.value))
    _rate = I.sign_hold_rate(_tables, int(min_cases.value))
    _view = mo.vstack([
        mo.md(f"### The perspectives that hold 1 sign over the cases\n\n"
              f"Of the {_rate['n_keys']} perspectives in "
              f"{int(min_cases.value)} cases or more, {_rate['n_hold']} hold "
              "1 sign in every case."),
        I.captioned(I.plot_consistent(_tables, persp_held, "perspective",
                                              arm=arm)),
        persp_held,
    ]) if not persp_held.empty else mo.md("_No perspective holds its sign._")
    _view
    return


@app.cell(hide_code=True)
def _(bundle, mo):
    _check = bundle.get("wins_check")
    (mo.vstack([
        mo.md(
            "### Does the `wins` attribute agree with the vote?\n\n"
            "`favored_wins` is the share of the perspective rows whose "
            "favored image won, under a decided label. A `wins = yes` row "
            "should read about 1.0, because the attribute says the decision "
            "follows the perspective. On the primary gemma arm it reads "
            "0.997 to 1.000, thus the extractor reads `wins` from the "
            "decision. A `wins = no` row that still wins is a perspective "
            "the model names for the winner and does not credit."),
        _check,
    ]) if _check is not None and not _check.empty else None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The inferences

    The key is `name | kind`. **Warning: an inference has no valence.**
    `maintenance | upkeep` is the same key for "well maintained" and for
    "poorly maintained", thus its lift pools the 2 directions. A key with a
    strong lift is a name that the model mostly uses in 1 direction.
    """)
    return


@app.cell
def _(I, arm, bundle, mo, n_ends):
    infer_ends = I.case_ends(bundle.get("lift_inference", I.pd.DataFrame()),
                             int(n_ends.value))
    (mo.vstack([I.captioned(I.plot_case_ends(infer_ends, "inference", arm=arm)),
                mo.accordion({"The table": infer_ends})])
     if not infer_ends.empty else mo.md("_No key over the floor._"))
    return


@app.cell
def _(I, arm, bundle, min_cases, mo):
    _tables = bundle.get("lift_inference", I.pd.DataFrame())
    infer_held = I.consistent_keys(_tables, int(min_cases.value))
    (mo.vstack([mo.md("### The inferences that hold 1 sign over the cases"),
                I.captioned(I.plot_consistent(_tables, infer_held, "inference",
                                              arm=arm)),
                infer_held])
     if not infer_held.empty else mo.md("_No inference holds its sign._"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The dimensions

    The key is `name | valence`. A `bad` dimension on the image that loses
    gives a negative lift, thus the sign carries the valence first. Read the
    size of the lift for the strength, and read a `neutral` key for the
    surprise: a neutral key far from 0 is a property the model calls neutral
    and that still goes with the vote.
    """)
    return


@app.cell
def _(I, arm, bundle, mo, n_ends):
    dim_ends = I.case_ends(bundle.get("lift_dimension", I.pd.DataFrame()),
                           int(n_ends.value))
    (mo.vstack([I.captioned(I.plot_case_ends(dim_ends, "dimension", arm=arm)),
                mo.accordion({"The table": dim_ends})])
     if not dim_ends.empty else mo.md("_No key over the floor._"))
    return


@app.cell
def _(I, arm, bundle, min_cases, mo):
    _tables = bundle.get("lift_dimension", I.pd.DataFrame())
    dim_held = I.consistent_keys(_tables, int(min_cases.value))
    (mo.vstack([mo.md("### The dimensions that hold 1 sign over the cases"),
                I.captioned(I.plot_consistent(_tables, dim_held, "dimension",
                                              arm=arm)),
                dim_held])
     if not dim_held.empty else mo.md("_No dimension holds its sign._"))
    return


@app.cell(hide_code=True)
def _(I, arm, mo):
    _sib = I.sibling_arm(arm)
    sibling_load = mo.ui.checkbox(
        value=False,
        label=(f"Read the other extractor of this judge: "
               f"{I.arm_label(_sib)} (10 to 15 seconds)") if _sib else
              "No other extractor for this judge",
    )
    mo.md(
        f"""
    ## 5. The same keys under the other extractor

    The 2 extractors of 1 judge read the same traces. A lift that holds
    under both comes from the traces. A lift that moves comes from the
    extractor. The Qwen3.8-27B extractor marks about 2x the perspectives of
    the Gemma-4-31B extractor, thus the 2 arms do not hold the same keys.

    {sibling_load}
    """
    )
    return (sibling_load,)


@app.cell
def _(I, allow_stale, arm, min_count, sibling_load):
    _sib = I.sibling_arm(arm)
    if sibling_load.value and _sib:
        _df = I.load_arm(_sib, require_canonical=not allow_stale.value)
        sibling_bundle = I.arm_bundle(_df, _sib, min_count=int(min_count.value),
                                      questions=("perspectives",))
        del _df
    else:
        sibling_bundle = None
    return (sibling_bundle,)


@app.cell
def _(I, X, arm, bundle, mo, sibling_bundle):
    if sibling_bundle is None:
        agreement = I.pd.DataFrame()
        _view = mo.md("_Tick the box to read the other extractor._")
    else:
        _labels = (X.ARMS[arm]["extractor"],
                   X.ARMS[I.sibling_arm(arm)]["extractor"])
        _parts, _figs, _tabs = [], [], []
        for _cls in I.LIFT_CLASSES:
            _merged, _agree = I.extractor_agreement(
                bundle.get(f"lift_{_cls}", I.pd.DataFrame()),
                sibling_bundle.get(f"lift_{_cls}", I.pd.DataFrame()))
            _tabs.append(_agree)
            if not _merged.empty:
                _figs.append(I.captioned(I.plot_extractor_agreement(
                    _merged, _cls, _labels, arm=arm)))
        agreement = I.pd.concat(_tabs, ignore_index=True)
        _view = mo.vstack([
            mo.md("`spearman` is the rank correlation of the lift over the "
                  "keys that clear the floor in BOTH arms. `same_sign` is the "
                  "share of those keys whose sign agrees."),
            mo.hstack(_figs, wrap=True),
            agreement.round(3),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG, CSV, and TeX files")
    mo.md(
        f"""
    ## 6. Export

    {save}

    The files land in `master/figures/` under the tag
    `ic_perspectives_<arm>_`. Each PNG gets a `.json` sidecar with its
    title, caption, LaTeX label, and tags. The 2 LaTeX tables (`ends`, `signs`) land in
    `master/outputs/`. `research/scripts/export_ic_v3_figures.py` writes the same
    files for the 2 primary arms without marimo.
    """
    )
    return (save,)


@app.cell
def _(
    FIG_DIR,
    I,
    NOTEBOOK_VERSION,
    OUT_DIR,
    arm,
    bundle,
    gate_ok,
    min_cases,
    mo,
    n_ends,
    save,
    sibling_bundle,
):
    if save.value and bundle:
        _written = I.export_perspectives(
            bundle, FIG_DIR, arm, gate_ok=gate_ok, n_ends=int(n_ends.value),
            min_cases=int(min_cases.value), sibling=sibling_bundle)
        _written += I.export_perspectives_latex(
            bundle, OUT_DIR, arm, NOTEBOOK_VERSION, gate_ok=gate_ok,
            min_cases=int(min_cases.value))
        _msg = mo.md(f"Wrote {len(_written)} files: "
                     + ", ".join(f"`{n}`" for n in _written)
                     + f"\n\nNotebook {NOTEBOOK_VERSION}, `_ic_v3` "
                     + f"{I.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
