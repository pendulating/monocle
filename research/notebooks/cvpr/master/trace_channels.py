import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The 4 other channels of the reasoning trace

    `master/cue_hierarchy.py` reads 1 extraction class, `visual_evidence`.
    That class is 46% of the corpus. This notebook reads the other 4 classes
    that carry a measurable signal.

    | Channel | Class | Question |
    |---------|-------|----------|
    | Axes | `comparison` | Which axis does the model decide ON? |
    | Abstention | `uncertainty` | Where does a `NotSure` come from? |
    | Inference | `inference` | What does the model read INTO a photograph? |
    | Artifact | `image_artifact` | Does the camera move the judgment? |

    ## The axes

    A `comparison` span carries a free phrase, `dimension`, and a `direction`
    that names the winning image. Thus the model writes the axis of the
    comparison in its own words: `smoothness`, `standardness`, `vibe`,
    `visual interest`. There are about 61,000 distinct phrases, and they are a
    second vocabulary beside the cue words. The cue says WHAT the model saw;
    the axis says what it compared on.

    **Warning: `agree` measures polarity first, and binding second.**
    `direction` names the image with MORE of the quality, not the image that
    wins. Thus `wear` reads 0.09 and `maintenance` reads 0.89, and both of
    them bind the vote hard. Read `bind` for the strength and `polarity` for
    the sign. An axis near chance is the interesting one: the model names it
    and then does not decide on it.

    ## The abstention

    The COUNT of the uncertainty spans in a trace separates an abstention
    from a decision. On the version 3 primary arms (2026-09-30) the AUC is
    0.62 to 0.93 for the gemma-4-12b judge (5 cases) and 0.79 to 0.85 for the
    qwen3.5-9b judge (2 cases). The result reproduces the version 1 range
    (0.74 to 0.94) in the strong cases. The `cannot_see` and `explicit_doubt`
    counts separate better than the total. The new `reconsideration` marker
    ("Wait, let me look again") separates worse, and it dilutes the total.
    The dose curve is flat below about 20 spans, and it climbs above that.

    ## The inference and the artifact

    Both classes name an image, thus both take the same directional test as a
    cue: 1 vote for each comparison, a key on both images is dropped, and the
    rate is shrunk toward the base rate of the case.

    **Warning: this is an association, not a cause.** The model narrates while
    it decides, thus a span can follow the judgment rather than drive it.
    """
    )
    return


@app.cell
def _():
    from pathlib import Path

    _here = Path(__file__).resolve().parent

    import marimo as mo
    import pandas as pd

    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _extractions as X
    from research.notebooks.cvpr import _style as style
    from research.notebooks.cvpr import _trace_channels as TC

    # The gate on the judgment runs. The gate on the EXTRACTION corpus is a
    # second test, and the next cell holds it.
    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.3.0"
    FIG_DIR = _here / "figures"
    # The LaTeX tables go to `outputs/`, never to `figures/`.
    TEX_DIR = _here / "outputs"
    return C, FIG_DIR, NOTEBOOK_VERSION, TC, TEX_DIR, X, mo, pd, style


@app.cell(hide_code=True)
def _(C, mo):
    allow_stale = mo.ui.checkbox(
        value=False,
        label="Read a corpus the registry does not name (a look only, "
              "never for the paper)",
    )
    mo.md(
        f"""
    ## 1. The 2 gates

    **Canonical registry:** {C.summary()}

    Every figure here is 2 steps downstream of a judgment run: the judge wrote
    the trace, and the extractor read it. Thus it must pass 2 gates.

    {allow_stale}
    """
    )
    return (allow_stale,)


@app.cell
def _(TC, allow_stale, mo):
    try:
        corpus = TC.load(require_canonical=not allow_stale.value)
        gate_ok, gate_error = True, ""
    except (RuntimeError, FileNotFoundError) as _exc:
        corpus, gate_ok, gate_error = None, False, str(_exc)

    if corpus is not None and not allow_stale.value:
        _msg = f"**The corpus passes the gate.** {len(corpus):,} extractions."
    elif corpus is not None:
        _msg = (
            "**The gate is OFF.** Every exported file takes a `_stale` "
            f"suffix.\n\n{len(corpus):,} extractions."
        )
    else:
        _msg = (
            "**No usable corpus.** Run the `extract_traces` pipeline on the "
            "registered trace runs, then `research/scripts/merge_trace_extractions.py`."
            f"\n\n```\n{gate_error}\n```"
        )
    mo.md(_msg)
    return corpus, gate_ok


@app.cell(hide_code=True)
def _(TC, corpus, mo):
    if corpus is None:
        _view = mo.md("_No corpus._")
    else:
        _view = mo.vstack([
            mo.md("### What each channel holds\n\n"
                  "Read this before a figure. A channel that a case never "
                  "wrote is absent, and an absent row is not a null result."),
            TC.coverage(corpus),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ### The same channel under the other extractor

    Every rate here is 2 steps downstream of the judge: the judge wrote the
    trace, and an extractor read it. The 2 x 2 of `_extractions.ARMS` reads
    the same traces with the other extractor. A rate that holds along a row
    of the 2 x 2 comes from the traces. A rate that moves comes from the
    extractor, and the paper must report it from both extractors or not at
    all.

    Measured on 2026-09-20 on the version 1 corpora, after the duplicate
    rows and the unanswered traces leave: `visual_evidence`, `comparison`
    and `decision` hold in every case. `uncertainty` differs 1.6x to 3.8x in
    the median case, `inference` about 2x, and `image_artifact` up to 27x.
    Thus the abstention channel below is a fact about the extractor pair,
    not about the judge.
    """
    )
    return


@app.cell
def _(FIG_DIR, X, corpus, gate_ok, mo, pd):
    if corpus is None:
        stability, stability_table = pd.DataFrame(), pd.DataFrame()
        _view = mo.md("_No corpus._")
    else:
        # The judge of the loaded corpus, and every arm that read its traces.
        _judge = str(corpus.judge_model.iloc[0]).split("/")[0] if len(corpus) else ""
        _arms = [name for name, spec in X.arm_roots().items()
                 if spec["judge"] == _judge]
        if len(_arms) < 2:
            stability, stability_table = pd.DataFrame(), pd.DataFrame()
            _view = mo.md(
                f"_1 arm only for judge `{_judge}`. The other extractor's corpus "
                "is not on disk, thus no rate here has an error bar._"
            )
        else:
            _long = X.compare_arms(X.class_rates, arms=_arms,
                                   require_canonical=gate_ok)
            stability = X.rate_stability(_long)
            stability_table = X.stability_summary(stability)
            _suffix = "" if gate_ok else "_stale"
            stability.to_csv(FIG_DIR / f"trace_channels_stability{_suffix}.csv",
                             index=False)
            _view = mo.vstack([
                mo.md(f"Judge `{_judge}`, arms: `{'`, `'.join(_arms)}`. "
                      "A row `holds` when the 2 extractors' rates stay within "
                      f"{X.STABLE_RATIO}x in every case."),
                stability_table,
                mo.accordion({"Every case": stability}),
            ])
    _view
    return stability, stability_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. The axes the model compares on

    `agree` is the share of the decided comparisons where the final label
    names the image the axis named. The band around chance holds the axes the
    model names and then does not decide on.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    axis_floor = mo.ui.slider(
        start=100, stop=2000, step=100, value=400,
        label="Fewest decided comparisons an axis needs",
    )
    axis_floor
    return (axis_floor,)


@app.cell
def _(TC, axis_floor, corpus, pd):
    axes = (pd.DataFrame() if corpus is None
            else TC.axis_table(corpus, min_count=int(axis_floor.value)))
    hedge = {} if corpus is None else TC.hedge_summary(corpus)
    return axes, hedge


@app.cell(hide_code=True)
def _(axes, hedge, mo):
    if not hedge:
        _view = mo.md("_No corpus._")
    else:
        _bind = axes[axes["polarity"] != "none"] if not axes.empty else axes
        _view = mo.md(f"""
    ### The headline

    | Number | Value |
    |--------|-------|
    | Comparisons that name an axis and a winner | {int(hedge['n_comparisons']):,} |
    | Distinct axis phrases | {int(hedge['n_axes']):,} |
    | Agreement with the final label | {hedge['agree_all']:.3f} |
    | Agreement, the model did NOT hedge | {hedge['agree_plain']:.3f} |
    | Agreement, the model hedged | {hedge['agree_hedged']:.3f} |
    | Share of the comparisons the model hedged | {hedge['hedge_rate']:.3f} |
    | Axes over the count floor | {len(axes)} |
    | Of those, axes that bind the vote | {len(_bind)} |
    """)
    _view
    return


@app.cell
def _(TC, axes, mo):
    _fig = None if axes.empty else TC.plot_axes(axes)
    # The caption goes under the figure, never on it.
    (mo.vstack([mo.md("### Which axis binds the vote"), _fig,
                mo.md(TC.caption_md(_fig))]) if _fig else None)
    return


@app.cell
def _(TC, axes, hedge, mo):
    _fig = None if axes.empty else TC.plot_axis_hedge(axes, hedge)
    (mo.vstack([mo.md("### What the model's own hedge marker costs"), _fig,
                mo.md(TC.caption_md(_fig))])
     if _fig else None)
    return


@app.cell(hide_code=True)
def _(TC, corpus, mo):
    if corpus is None:
        _view = mo.md("_No corpus._")
    else:
        _view = mo.vstack([
            mo.md("### Each case, over every axis\n\n"
                  "`hedge_cost` is `agree_plain - agree_hedged`. It is "
                  "positive in 6 of the 7 cases. Road quality is the "
                  "exception, thus the marker is not universal."),
            TC.axis_by_case(corpus),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Where an abstention comes from

    The count of the uncertainty spans in a trace is a score. The AUC asks
    whether it separates an abstention from a decision, and the dose curve
    asks how the rate climbs.

    Version 3 holds 1,000 traces for each case, and it marks about 10x the
    uncertainty spans of version 1: the median deciding trace holds 11 spans
    on the primary gemma arm. Thus the dose bins are wider than the version 1
    bins (`TC.DOSE_EDGES`). A case with fewer than 25 abstentions has no AUC.
    Road quality has 6 and street photography has 0.

    **Warning: the left end of the curve is a small group.** 28 traces abstain
    with 0 to 2 uncertainty spans. They are short traces (a median of 14
    spans in all, against 26 for a deciding trace with so few uncertainty
    spans). The judge answers `NotSure` and gives no long reason. Read the
    rise of the curve above 20 spans, not the dip at 3 to 10.
    """
    )
    return


@app.cell
def _(TC, corpus, pd):
    traces = pd.DataFrame() if corpus is None else TC.trace_frame(corpus)
    abstain = pd.DataFrame() if traces.empty else TC.abstention_table(traces)
    dose = pd.DataFrame() if traces.empty else TC.abstention_dose(traces)
    return abstain, dose, traces


@app.cell
def _(TC, abstain, dose, mo):
    _fig = (None if abstain.empty or not abstain["auc"].notna().any()
            else TC.plot_abstention(dose, abstain))
    (mo.vstack([mo.md("### The dose curve"), _fig,
                mo.md(TC.caption_md(_fig)), abstain])
     if _fig else None)
    return


@app.cell(hide_code=True)
def _(TC, abstain, mo):
    if abstain.empty:
        marker_auc = abstain
        _view = mo.md("_No corpus._")
    else:
        _cols = (["case_label", "n_abstain", "abstain_rate", "auc", "auc_core"]
                 + [f"auc_marker_{_m}" for _m in TC.MARKER_ORDER]
                 + [f"auc_{_r}" for _r in TC.REASON_ORDER])
        marker_auc = (abstain[[_c for _c in _cols if _c in abstain.columns]]
                      .dropna(subset=["auc"]).round(3))
        _view = mo.vstack([
            mo.md(
                "### Which kind of doubt separates an abstention\n\n"
                "Each AUC reads 1 count alone: the spans of 1 marker or of 1 "
                "reason. `auc_core` reads the total less the "
                "`reconsideration` spans, which version 1 and version 2 "
                "mostly did not mark. Thus `auc_core` is the number to "
                "compare with the earlier versions.\n\n"
                "An AUC below 0.5 says the count goes with a DECISION. "
                "On the primary gemma arm `looks_equal` reads 0.33 to 0.44: "
                "the judge that says the 2 photographs look equal answers "
                "`Same`, not `NotSure`."),
            marker_auc,
        ])
    _view
    return (marker_auc,)


@app.cell
def _(TC, corpus, mo):
    _mix = None if corpus is None else TC.reason_mix(corpus)
    _fig = (TC.plot_reasons(_mix) if _mix is not None and not _mix.empty
            else None)
    (mo.vstack([
        mo.md("### What the model says it cannot do\n\n"
              "`other` is the reason the extractor could not type, and it "
              "takes 38% to 85% of the spans. Read the 3 named reasons "
              "against each other, never against the whole."),
        _fig, mo.md(TC.caption_md(_fig)), _mix,
    ]) if _fig is not None else None)
    return


@app.cell(hide_code=True)
def _(mo):
    arms_load = mo.ui.checkbox(
        value=False, label="Read the 4 version 3 arms (about a minute)")
    arm_floor = mo.ui.slider(
        start=5, stop=50, step=5, value=25,
        label="Fewest abstentions a case needs for an AUC")
    mo.md(
        f"""
    ### The abstention in the 2 x 2 of judge and extractor

    The same traces of each judge, read by 2 extractors. The AUC that holds
    along a judge comes from the traces. The AUC that moves comes from the
    extractor. The x axis of the figure is the quantile of the count INSIDE
    the arm, because the 2 extractors of 1 judge mark up to 2.5x different
    counts.

    **Warning: the box reads all 4 corpora.** That takes about a minute and
    about 3.5 GB. The qwen-trace arms hold up to 1.1 M rows each.

    **Warning: the qwen3.5-9b judge abstains seldom.** At the floor of 25
    it has an AUC in 2 cases only (schools, restaurants). A lower floor
    shows more cases, but an AUC on 14 abstentions is noisy.

    {arms_load}

    {arm_floor}
    """
    )
    return arm_floor, arms_load


@app.cell
def _(TC, arms_load, gate_ok):
    arm_trace_frames = (TC.arm_traces(require_canonical=gate_ok)
                        if arms_load.value else {})
    return (arm_trace_frames,)


@app.cell
def _(TC, arm_floor, arm_trace_frames, pd):
    if arm_trace_frames:
        arm_stats, arm_dose = TC.abstention_from_traces(
            arm_trace_frames, min_abstain=int(arm_floor.value))
    else:
        arm_stats, arm_dose = pd.DataFrame(), pd.DataFrame()
    return arm_dose, arm_stats


@app.cell(hide_code=True)
def _(TC, arm_dose, arm_stats, mo):
    if arm_stats.empty:
        _view = mo.md("_Tick the box to read the 4 arms._")
    else:
        _fig = (TC.plot_abstention_arms(arm_dose, arm_stats)
                if arm_stats["auc"].notna().any() else None)
        _view = mo.vstack([
            mo.md("**AUC of the total count, case x arm.** The primary arm "
                  "of each judge comes first."),
            TC.auc_by_arm(arm_stats).round(3),
            mo.md("**AUC of the count less `reconsideration`.**"),
            TC.auc_by_arm(arm_stats, "auc_core").round(3),
            _fig,
            mo.md(TC.caption_md(_fig)) if _fig is not None else mo.md(""),
            mo.accordion({
                "Every AUC of every arm": TC.auc_long(arm_stats).round(3),
                "The dose curve of every arm": arm_dose.round(3),
            }),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. What the model reads INTO a photograph

    An `inference` span is the model past the pixels. `kind` types it, and
    4 of the types carry the risk result: wealth, class, crime, and
    demographic.

    **`kind` names the TOPIC, never the direction.** `wealth` is the same key
    whether the model wrote `upscale` or `lower-income area`, thus a lift on
    `kind` alone cannot say which way the inference runs. This section adds 2
    readings that can:

    | Reading | What it gives |
    |---------|---------------|
    | `pole` | `high` or `low`, from 2 short word lists |
    | `word` | the phrase itself, with a price |

    Both first drop a **restatement**: a span that names the other photograph
    or the label restates the vote and infers nothing. `much less rather eat
    at A than B` is the 2nd most frequent `wealth` span of the corpus.
    """
    )
    return


@app.cell
def _(TC, corpus, pd):
    infer = (pd.DataFrame() if corpus is None
             else TC.kind_lift(corpus, TC.INFERENCE_CLASS))
    infer_keys = (pd.DataFrame() if infer.empty
                  else TC.kind_summary(infer, order=TC.INFERENCE_ORDER))
    poles = pd.DataFrame() if corpus is None else TC.pole_lift(corpus)
    if poles.empty:
        pole_keys = pd.DataFrame()
    else:
        pole_keys = TC.kind_summary(poles, order=None)
        pole_keys = (pole_keys.set_index("key")
                     .loc[TC.pole_key_order(pole_keys)].reset_index())
    words = pd.DataFrame() if corpus is None else TC.inference_words(corpus)
    return infer, infer_keys, pole_keys, poles, words


@app.cell(hide_code=True)
def _(TC, corpus, mo):
    if corpus is None:
        _view = mo.md("_No corpus._")
    else:
        _view = mo.vstack([
            mo.md(
                "### Where does each kind survive?\n\n"
                "A rare kind can pass this whole notebook without a row and "
                "read as absent. `crime` is 454 spans of 302,784: it holds 1 "
                "dot on the ladder below, and it needs its own word floor to "
                "appear at all. Read `word_floor` beside `word_keys`."),
            TC.inference_coverage(corpus),
        ])
    _view
    return


@app.cell
def _(TC, infer, infer_keys, mo):
    _fig = (None if infer.empty else TC.plot_kind_ladder(
        infer, infer_keys, name="inference"))
    (mo.vstack([mo.md("### The price of an inference, by topic"), _fig,
                mo.md(TC.caption_md(_fig)), infer_keys])
     if _fig else None)
    return


@app.cell(hide_code=True)
def _(TC, corpus, mo):
    if corpus is None:
        _view = mo.md("_No corpus._")
    else:
        _view = mo.vstack([
            mo.md(
                "### Which way does each kind run?\n\n"
                "`share_high` reads only the spans the word lists DETERMINE. "
                "Read it beside `share_determined`: a kind that leaves most "
                "of its spans undetermined cannot carry the number far.\n\n"
                "**Warning: `public school` sits on the LOW list** because "
                "the model uses it as its own contrast to `private school`. "
                "It is the model's class vocabulary, not a claim about a "
                "school."),
            TC.pole_mix(corpus),
        ])
    _view
    return


@app.cell
def _(TC, mo, pole_keys, poles):
    _fig = (None if poles.empty else TC.plot_kind_ladder(
        poles, pole_keys, figsize=(6.4, 5.0), legend_loc="below",
        name="inference_pole"))
    (mo.vstack([mo.md("### The price of an inference, by DIRECTION"), _fig,
                mo.md(TC.caption_md(_fig)), pole_keys])
     if _fig else None)
    return


@app.cell
def _(TC, mo, words):
    _fig = None if words.empty else TC.plot_inference_words(words)
    (mo.vstack([
        mo.md("### The words themselves\n\n"
              "The lift is pooled over the cases, because an inference span "
              "is a clause and almost no phrase clears a per-case floor. "
              "Each vote carries the base rate of its OWN case."),
        _fig,
        mo.md(TC.caption_md(_fig)),
    ]) if _fig else None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### The sentences behind the numbers

    A lift is a number. These are the sentences the model wrote, quoted from
    the traces themselves. The rows of 1 cell are chosen, never sampled at
    random: the phrases whose price is furthest from 0 take the cell, half
    from the paying end and half from the costing end.

    **Warning: an excerpt needs the source run on disk.** A case whose run is
    gone is absent here. That is not a null result.
    """
    )
    return


@app.cell
def _(TC, corpus, pd):
    quoted = pd.DataFrame() if corpus is None else TC.excerpts(corpus)
    return (quoted,)


@app.cell(hide_code=True)
def _(mo, quoted):
    case_pick = mo.ui.dropdown(
        options=sorted(quoted["case_label"].unique()) if not quoted.empty
        else ["none"],
        value=(sorted(quoted["case_label"].unique())[0] if not quoted.empty
               else "none"),
        label="Case",
    )
    case_pick
    return (case_pick,)


@app.cell
def _(TC, case_pick, mo, quoted):
    if quoted.empty:
        _view = mo.md("_No excerpt. The source runs may not be on disk._")
    else:
        _rows = quoted[quoted["case_label"] == case_pick.value]
        _view = mo.md(TC.format_excerpts(_rows))
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Does the camera move the judgment?

    An `image_artifact` span names a flaw of the photograph, not of the place.
    A lift far from 0 says the judgment tracks how the photograph was taken.
    """
    )
    return


@app.cell
def _(TC, corpus, pd):
    artifact = (pd.DataFrame() if corpus is None
                else TC.kind_lift(corpus, TC.ARTIFACT_CLASS))
    artifact_keys = (pd.DataFrame() if artifact.empty
                     else TC.kind_summary(artifact, order=TC.ARTIFACT_ORDER))
    return artifact, artifact_keys


@app.cell
def _(TC, artifact, artifact_keys, mo):
    _fig = (None if artifact.empty else TC.plot_kind_ladder(
        artifact, artifact_keys, name="artifact"))
    (mo.vstack([mo.md("### The price of a photographic flaw"), _fig,
                mo.md(TC.caption_md(_fig)), artifact_keys])
     if _fig else None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 6. The numbers behind the figures")
    return


@app.cell
def _(axes):
    axis_keys = axes
    axis_keys
    return


@app.cell
def _(traces):
    trace_rows = traces
    trace_rows
    return


@app.cell
def _(words):
    inference_word_keys = words
    inference_word_keys
    return


@app.cell
def _(quoted):
    excerpt_rows = quoted
    excerpt_rows
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG and CSV files")
    mo.md(
        f"""
    ## 7. Export

    {save}

    The files land in `master/figures/` under the `trace_channels` tag. A
    corpus that fails the gate writes every name with a `_stale` suffix.
    Each PNG gets a `.json` sidecar with its title, caption, LaTeX label,
    and tags. The figure itself carries no caption text.
    The 2 LaTeX tables of the 2 x 2 abstention land in `master/outputs/`:
    `trace_channels_abstention_auc_v<version>.tex` and
    `trace_channels_abstention_markers_v<version>.tex`.

    **Warning: the export reads the 4 version 3 arms** for the 2 x 2 files,
    unless the box of section 3 read them first. That adds about a minute.
    """
    )
    return (save,)


@app.cell
def _(
    FIG_DIR,
    NOTEBOOK_VERSION,
    TC,
    TEX_DIR,
    arm_trace_frames,
    axis_floor,
    corpus,
    gate_ok,
    mo,
    save,
):
    if save.value and corpus is not None:
        # The export reads the 4 arms when the box above did not. The files
        # take the paper floor of 25 abstentions, whatever the slider says.
        _written = TC.export(
            corpus, FIG_DIR, axis_min_count=int(axis_floor.value),
            version=NOTEBOOK_VERSION, gate_ok=gate_ok,
            arms_tables=(TC.abstention_from_traces(arm_trace_frames)
                         if arm_trace_frames else None),
            tex_dir=TEX_DIR,
        )
        _msg = mo.md(
            f"Wrote {len(_written)} files to `master/figures/` (the `.tex` "
            "tables to `master/outputs/`): "
            + ", ".join(f"`{n}`" for n in _written)
            + f"\n\nNotebook {NOTEBOOK_VERSION}, `_trace_channels` "
            + f"{TC.__version__}."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
