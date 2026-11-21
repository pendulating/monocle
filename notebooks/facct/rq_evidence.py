import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The evidence for the 3 research questions

    | RQ | Question | What the paper can claim | What it cannot claim |
    |----|----------|--------------------------|----------------------|
    | RQ1 | Do VLMs make aesthetic inferences that observable cues do not back up? | From 1 storefront photo, every model infers the business, the location, the atmosphere and the popularity of the place. The thinking trace holds more claims about people than the answer | "Ungrounded" in the strict sense: no experiment ties an inference to a missing cue |
    | RQ2 | Do the interpretations of the pixels correlate with the socioeconomic features of the area? | In restaurants, the valence of what the model sees and its "is this a cool place" judgment rise with % white and % bachelor's, in all models; the image-content controls remove about 10% | That the model treats people differently (the NTA is an ecological unit); the same for street photography (near null) |
    | RQ3 | Are the inferences and biases unavoidable? | The shift is present under every elicitation we tried (6 models, 2 batches, 2 extraction rounds, paraphrase, persona), and at fixed pixels a neighbourhood name alone changes the judgment | "Unavoidable": the direction of the name effect depends on the model and the case, and any name moves the judgment more than the rich-poor gap |

    The review that sorts the results is `vlm-narratives-docs/facct-rq-evidence.md`.
    This notebook does no test. It reads the result files of F4, F8 and the
    parent audit. **Bold** or a filled marker: significant after the
    Bonferroni correction of its family of tests.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import marimo as mo
    import matplotlib.pyplot as plt

    import _f8 as F8
    import _rq as Q
    import _style as style

    style.apply_house_style()
    R = Q.load()
    NOTEBOOK_VERSION = "1.5.0"
    return F8, NOTEBOOK_VERSION, Q, R, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RQ1. Inferences beyond the pixels

    **Verdict.** The models infer a lot from 1 photo. Popularity claims ("a
    neighbourhood favorite") are in 8% to 43% of the restaurant answers, but
    a still photo of a storefront shows no popularity. Social inferences
    (wealth, class, demographic, crime) are rare: under 11% of the answers.
    The F4 schema does not tie an inference to the cue that supports it.
    Thus the counts show that the models infer; they do not show that a
    given inference has no cue.
    """)
    return


@app.cell
def _(Q, R):
    fig_kinds = Q.plot_kinds(R["model_diff"])
    fig_kinds
    return


@app.cell(hide_code=True)
def _(F8, Q, R, mo):
    mo.md(
        "**Table 1.** The % of the restaurant answers (F4, about 9,800 for each model) with a claim of each type.\n\n"
        + F8.md_table(Q.social_rates(R["model_diff"]))
        + "\n\nThe range over the models is large: uncertainty 7% to 80%, popularity 8% to 43%. "
        "Each row differs by model (chi-square, Bonferroni over 316 rows)."
    )
    return


@app.cell
def _(Q, R):
    fig_trace = Q.plot_trace_gap(R["trace"])
    fig_trace
    return


@app.cell(hide_code=True)
def _(F8, Q, R, mo):
    _p = R["parent_inf"].set_index("key").loc[["wealth", "class"],
                                              ["n_cases", "n", "lift_mean", "lift_min", "lift_max", "n_pays", "n_costs"]]
    mo.md(
        "**Table 2.** The thinking trace against the final answer (4 thinking models).\n\n"
        + F8.md_table(Q.trace_gap(R["trace"]))
        + "\n\nThe caption traces think about the people in the photo (26% to 54%); the captions "
        "almost never say it (0.5% to 13%).\n\n"
        "**Table 3.** The parent audit (gemma-4-12b, 2 images, about 77,000 traces): the win-rate lift of the "
        "image whose trace holds the inference.\n\n" + F8.md_table(_p.round(3))
        + "\n\nA wealth inference helps its image win in all 6 cases. **Caution:** the `class` lift "
        "changed sign between the v2 and the v3 extraction schema (M9 of the review). Report wealth only."
    )
    return


@app.cell
def _(Q, mo):
    try:
        P = Q.load_patch_words()
    except FileNotFoundError:
        P = None
    if P is None:
        _out = mo.md("### What the patches carry\n\n_Not written yet._ Run `scripts/rq1/patch_words_sample.py`, "
                     "then `monocle.patch_word_gif --sample` (see `_rq.load_patch_words`).")
    else:
        _out = mo.vstack([
            mo.md("### What the patches carry\n\nThe top R-lens token of each 4x4 block of image patches (gemma-4-12b, "
                  "layer 42), as the raw readout (word pieces and other scripts stay), and 2 inference spans of the final answer to the same image. Most patches name "
                  "the scene. A few carry social words from the pixels alone (Norwood: `illegal`, "
                  "`inhabitants`, `homelessness`). The lens names what a patch holds; it does not show which "
                  "patches move the answer."),
            Q.plot_patch_words(P),
        ])
    _out
    return (P,)


@app.cell
def _(Q, mo):
    try:
        PL = Q.load_patch_layers()
    except FileNotFoundError:
        PL = None
    if PL is None:
        _out = mo.md("### The patches, layer by layer (appendix)\n\n_Not written yet._ See the cell above.")
    else:
        _out = mo.vstack([
            mo.md("### The patches, layer by layer (appendix)\n\nThe first image of each case and income group, "
                  "with 1 row for each of 6 layers (6, 18, 30, 36, 42, 46) and the raw top token of each block. "
                  "Below layer 30 (grey labels) the lens does not read the content of a token: word pieces "
                  "(`ed`, `ured`) and the same words repeat (`French`, `pastel`). From layer 30, most tokens "
                  "name the scene."),
            Q.plot_patch_layers(PL),
        ])
    _out
    return (PL,)


@app.cell
def _(F8, PL, Q, mo):
    try:
        PT = Q.load_patch_layers(tag="rlens_text_first")
    except FileNotFoundError:
        PT = None
    if PT is None or PL is None:
        _out = mo.md("### The patches with the question first (appendix)\n\n_Not written yet._ Run "
                     "`monocle.patch_word_gif --sample ... --layout text_first --tag rlens_text_first`.")
    else:
        _ch = Q.patch_changes(PT, PL)
        _t = (_ch.mean(axis=1) * 100).round(0).astype(int).rename("blocks that change (%)").to_frame()
        _out = mo.vstack([
            mo.md("### The patches with the question first (appendix)\n\nThe same lens, but the question "
                  "comes before the image (`text_first`). A patch reads only the tokens before it, so now "
                  "the question can change the patch tokens. Coral: the top token differs from the run with "
                  "the image first. Words of the question enter the patches (`How`, `Good`, `appeal`, "
                  "`picturesque`), and the Norwood words (`illegal`, `homelessness`) go away. **Caution:** "
                  "the layout also moves the patches to later positions. The pair `neutral_first` and "
                  "`question_first` of `monocle.single` holds the positions fixed."),
            Q.plot_patch_layers(PT, base=PL),
            mo.md(F8.md_table(_t)),
        ])
    _out
    return (PT,)


@app.cell
def _(F8, PL, PT, Q, mo):
    try:
        PQ = Q.load_patch_layers(tag="rlens_question_first")
        PN = Q.load_patch_layers(tag="rlens_neutral_first")
    except FileNotFoundError:
        PQ = PN = None
    if PQ is None or PL is None or PT is None:
        _out = mo.md("### The question effect, with the patch positions fixed (appendix)\n\n_Not written yet._ "
                     "Run `monocle.patch_word_gif` with `--layout question_first` and `--layout neutral_first`.")
    else:
        _pairs = {"question: (question, image, filler) vs (filler, image, question)": (PQ, PN),
                  "filler and position: (filler, image, question) vs image first": (PN, PL),
                  "question first vs image first": (PT, PL)}
        _t = Q.patch_change_table(_pairs)
        _out = mo.vstack([
            mo.md("### The question effect, with the patch positions fixed (appendix)\n\nThe question first "
                  "also moves the patches. This pair holds the positions fixed: (question, image, filler) "
                  "against (filler, image, question). The filler repeats \"This is a photograph.\" to the "
                  "token count of the question, so the patches start at the same position and the sequences "
                  "have the same length. Coral: the top token differs between the 2 arms, thus the question "
                  "alone changes it. **Caution:** the filler alone also changes most top tokens (second "
                  "column). The top token changes with any text before the image."),
            Q.plot_patch_layers(PQ, base=PN),
            mo.md("Blocks whose top token changes (%), mean of the 8 images:\n\n" + F8.md_table(_t)),
        ])
    _out
    return PN, PQ


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The limits of RQ1.**

    - "Ungrounded" is not a class of the schema. The border between
      `inference` and `visual_evidence` is the border that the extractor
      draws worst: against 10 human-read traces, `inference` recall is 0.26
      (precision 0.86). Every inference rate is a floor.
    - The class rates moved 2x to 10x when the extractor changed. Compare F4
      rates with F4 rates only.
    - The gap that closes RQ1: tie each inference span to a cue on a sample
      of 300 answers, by hand. The data are on disk; no GPU run is needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RQ2. The reading of the pixels follows the neighbourhood

    **Verdict.** Strong, small, and in restaurants only. 2 outcomes rise
    with the share of white residents and of residents with a bachelor's
    degree: the net valence of what the model says it sees (F8.1), and the
    judgment "is this a cool place to eat at?" after the answer (F8.3). The
    cuisine, the DOHMH grade and the borough remove about 10% of the
    valence effect. The effect is 0.05 to 0.32 SD of the outcome for each SD
    of the feature.
    """)
    return


@app.cell
def _(Q, R):
    fig_rq2 = Q.plot_rq2(R)
    fig_rq2
    return


@app.cell(hide_code=True)
def _(F8, Q, R, mo):
    mo.md(
        "**Table 4.** Method B: OLS of the outcome on 1 z-scored NTA feature, without and with the "
        "image-content controls (restaurant: cuisine, grade, borough; street: borough). SE clustered by NTA; "
        "Bonferroni over the rows of each outcome.\n\n" + F8.md_table(Q.method_b_counts(R))
        + "\n\n% Asian is never significant for the valence."
    )
    return


@app.cell
def _(F8, R):
    fig_nta = F8.plot_nta_scatter("restaurant", "c_cool_i", "pct_bachelors", R["spearman"])
    fig_nta
    return


@app.cell(hide_code=True)
def _(F8, Q, R, mo):
    _s = Q.sign_agreement(R["nta_cor"])
    _t = "\n".join(f"| {k} | {v:,} |" for k, v in _s.items())
    mo.md(
        "**Table 5.** The F4 cue rates by NTA: the Spearman tests of each (case, field, category) rate with "
        "each ACS feature, for each model (Bonferroni over all tests).\n\n| Count | Value |\n|---|---|\n" + _t
        + "\n\nThe models disagree on how often a cue appears (RQ3), but not on the direction of its NTA "
        "association.\n\n**The limits of RQ2.** The NTA is an ecological unit: write \"the valence shifts "
        "with the resident mix of the area\". The controls are coarse: they do not hold the storefront, the "
        "light or the season fixed. The street case is near null (valence 4 of 50 rows with the controls)."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RQ3. Is the bias unavoidable?

    **Verdict.** "Present under every elicitation we tried, and not removed
    by our controls" is what the data support. "Unavoidable" is not. At
    fixed pixels, a neighbourhood name in the prompt changes the judgment of
    3 of 4 models (F8.2). But the direction depends on the model and the
    case: for street photography, the gemma models rate a place as cooler
    with a poor name. And any name moves the judgment away from the no-cue
    answer more than the gap between a rich and a poor name.
    """)
    return


@app.cell
def _(F8, R):
    fig_cue = F8.plot_primary(R["primary"], ("c_yes_i", "c_cool_i"))
    fig_cue
    return


@app.cell(hide_code=True)
def _(R, mo):
    _p = R["primary"]
    _i = _p[_p["readout"].isin(["c_yes_i", "c_cool_i"])]
    _w = R["words"]
    _up = _w[(_w["list"] == "upscale") & _w["significant"]]
    mo.md(
        f"The same image, \"I am in {{rich name}}\" against \"I am in {{poor name}}\": "
        f"{int(_p['significant'].sum())} of {len(_p)} readout tests are significant "
        f"({int(_i['significant'].sum())} of {len(_i)} for the 2 instructed readouts above). "
        f"With the rich name, {_up['model'].nunique()} of 4 models write more \"upscale\" words "
        f"({len(_up)} significant rows of the word tests). **Caution:** in 9 of 10 name pairs the rich "
        "name is also the whiter area, thus F8.2 does not separate class from race. F8.5 below does."
    )
    return


@app.cell
def _(F8, R):
    fig_base = F8.plot_baseline(R["primary"], "c_yes_i")
    fig_base
    return


@app.cell(hide_code=True)
def _(F8, Q, R, mo):
    _md = R["model_diff"]
    _rob = Q.parent_income(R["robust"])
    mo.md(
        "**Table 6.** The NTA shift under different elicitations.\n\n"
        "| Change | Result | Source |\n|---|---|---|\n"
        f"| 6 models, 3 families | {int(_md['significant'].sum())} of {len(_md)} cue categories differ by model, "
        f"{int((_md['range_pp'] >= 10).sum())} by 10 points or more; but no NTA row has 2 significant models "
        "with opposite signs (Table 5) | F4 4.8, 4.10 |\n"
        "| 2 image batches | 1 of 115 cue rates differs between the batches (Bonferroni) | F4 C1 |\n"
        "| image-content controls | 37 of 40 restaurant valence effects stay; 16 of 16 `c_cool_i` rows | F8.1, F8.3 |\n"
        "| paraphrase, persona (parent audit) | the tract r with median income stays positive (table below) | robustness battery |\n\n"
        "The parent audit, the tract-level r of the rating with median income:\n\n" + F8.md_table(_rob)
        + "\n\nThe parent audit reads income only, at the area layer, and the layer inflates the "
        "correlation. The persona moves the 2 judges in opposite directions."
    )
    return


@app.cell(hide_code=True)
def _(F8, Q, R, mo):
    _dc = R["dc_primary"]
    if _dc.empty:
        _view = mo.md(
            "### F8.5: class against race (D136)\n\n_Running._ The decoupled cue uses names where the "
            "whiter name is the poorer name: Borough Park against East New York (the same income) and "
            "Canarsie (richer) in Brooklyn; Maspeth against Hollis (the same income) and Laurelton (richer) "
            "in Queens. Run `scripts/f8/analyze_decoupled.py` after the jobs end, then run this cell again."
        )
    else:
        _t = _dc.assign(cell=[F8.mark(d, s) for d, s in zip(_dc["diff"], _dc["significant"])])
        _t = _t.pivot_table(index=["case", "contrast", "readout"], columns="model", values="cell", aggfunc="first")
        _t = _t[[m for m in F8.CUE_MODELS if m in _t.columns]]
        _t.index = [f"{c}, {k}, `{r}`" for c, k, r in _t.index]
        _view = mo.vstack([
            mo.md("### F8.5: class against race (D136)\n\n`race`: the whiter name minus a non-white name with "
                  "the same income and education. `reversed`: the whiter name minus a richer non-white name. "
                  "Positive: the whiter name gets the higher judgment."),
            Q.plot_decoupled(_dc),
            mo.md(F8.md_table(_t) + f"\n\n{int(_dc['significant'].sum())} of {len(_dc)} tests are significant. "
                  "Only 1 significant test is positive (qwen3.5-9b, street, +0.17). gemma-4-12b follows class: "
                  "the richer name wins in all 4 `reversed` tests. The gemma-4-31b street effect comes from 1 "
                  "name: East New York also beats Canarsie (`class` -5.7 in Brooklyn), and Queens gives -0.7 "
                  "and -1.4. No word-list row is significant (0 of 128). See the runbook, F8.5."),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The headline results

    | # | Result | Source | Caveat |
    |---|--------|--------|--------|
    | H1 | After the model describes a restaurant, its answer to "Is this a cool place to eat at?" rises with % white (rho +0.45 to +0.74) and % bachelor's (+0.40 to +0.80) of the area, in all 4 thinking models; 16 of 16 rows stay with the controls | F8.3; `r3_nta_spearman.csv`, `method_b_r3/c_cool_i_*` | an NTA correlation; coarse controls; 0.07 to 0.32 SD for each SD |
    | H2 | The valence of what the model says it sees shifts with the resident mix in all 6 models: 37 of 40 restaurant effects survive the controls, median share kept 0.89 | F8.1; `valence_cases.csv` | small (0.05 to 0.18 SD); ecological unit |
    | H3 | The same image with a rich or a poor neighbourhood name gets a different judgment: 16 of 32 tests; all 4 models write more "upscale" words with the rich name | F8.2; `cue_primary.csv`, `cue_words.csv` | the direction differs by model and case; class and race mixed (F8.5) |
    | H4 | The model changes what the text holds (291 of 316 categories differ, up to 87 points), but not the direction of the NTA association | F4 4.8, 4.10 | the texts differ in length; the extractor reads each style differently |
    | H5 | The parent audit: the area-income association holds under paraphrase, persona, order swap and no city name (tract r 0.22 to 0.47) | robustness battery | area layer only; income only |
    """)
    return


@app.cell(hide_code=True)
def _(Q, mo):
    write_btn = mo.ui.run_button(label="Write the figures")
    mo.vstack([mo.md(f"## The figures\n\nThe button writes each figure as PDF and PNG, with a JSON file "
                     f"(title, caption, LaTeX label, tags), to `{Q.OUT.relative_to(Q.REPO_ROOT)}/`."), write_btn])
    return (write_btn,)


@app.cell
def _(F8, NOTEBOOK_VERSION, P, PL, PN, PQ, PT, Q, R, mo, plt, write_btn):
    if write_btn.value:
        _figs = {
            "rq1_inference_kinds": lambda: Q.plot_kinds(R["model_diff"]),
            "rq1_trace_against_answer": lambda: Q.plot_trace_gap(R["trace"]),
            "rq2_method_b": lambda: Q.plot_rq2(R),
            "rq2_nta_c_cool_i_bachelors": lambda: F8.plot_nta_scatter("restaurant", "c_cool_i", "pct_bachelors",
                                                                      R["spearman"]),
            "rq3_cue_primary": lambda: F8.plot_primary(R["primary"], ("c_yes_i", "c_cool_i")),
            "rq3_cue_baseline": lambda: F8.plot_baseline(R["primary"], "c_yes_i"),
        }
        if P is not None:
            _figs["rq1_patch_words"] = lambda: Q.plot_patch_words(P)
        if PL is not None:
            _figs["rq1_patch_layers"] = lambda: Q.plot_patch_layers(PL)
        if PL is not None and PT is not None:
            _figs["rq1_patch_layers_text_first"] = lambda: Q.plot_patch_layers(PT)
            _figs["rq1_patch_layers_text_first_diff"] = lambda: Q.plot_patch_layers(PT, base=PL)
        if PQ is not None and PN is not None:
            _figs["rq1_patch_layers_question_first"] = lambda: Q.plot_patch_layers(PQ)
            _figs["rq1_patch_layers_question_first_diff"] = lambda: Q.plot_patch_layers(PQ, base=PN)
            _figs["rq1_patch_layers_neutral_first"] = lambda: Q.plot_patch_layers(PN)
        if not R["dc_primary"].empty:
            _figs["rq3_decoupled"] = lambda: Q.plot_decoupled(R["dc_primary"])
        _written = []
        for _name, _make in _figs.items():
            _fig = _make()
            _written += Q.save(_fig, _name)
            plt.close(_fig)
        _msg = mo.md(f"Wrote {len(_written)} files to `{Q.OUT.relative_to(Q.REPO_ROOT)}/`. "
                     f"Notebook {NOTEBOOK_VERSION}, `_rq` {Q.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
