import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F4 — reasoning traces of 6 models on 1 image

    F4 shows 1 image to a model at a time. 6 models read each image, with 3
    prompts (the 3 cases):

    | Case | Prompt | Images |
    |------|--------|--------|
    | `restaurant` | Look and tell me more about this restaurant. | restaurant storefronts |
    | `caption` | Look and write an Instagram caption for what's in front of me. | the same storefronts |
    | `street` | Look and tell me a good angle to take a picture from. | street views, about the same count in each residential NTA |

    Section 0 gives the data that this notebook reads, with the counts. The
    code comes from the trace notebook of the CVPR restaurant corpus
    (`research/notebooks/cvpr/restaurants/restaurants_traces.py`, "the parent"). In
    the parent, 1 judge reads 2 images in 1 prompt.

    Each answer has 2 texts: the trace (`reasoning`) and the answer
    (`response`). The 2 Llama models write no trace (D87). The **scope** below
    selects the text:

    | Scope | 1 unit is | Models |
    |-------|-----------|--------|
    | `reasoning` | 1 trace | the 4 thinking models |
    | `response` | 1 final answer | all 6 |
    | `all` | 1 answer: its trace and its answer together | all 6 |

    ## Words, then claims

    Sections 2 and 3 count WORDS. Section 4 counts CLAIMS: typed spans from the
    DeepSeek-V4-Flash extraction of the x2 round (D130), which F4 uses as its
    result (D131): schema `single-v2` for restaurant and caption, `street-v3`
    for street.

    **Warning: compare F4 rates with F4 rates only.** F4 changes the
    extractor, the schema, the prompt, the input (1 image, not 2), and the
    images at the same time as the parent corpus.
    """)
    return


@app.cell
def _():
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    from research.notebooks.facct import _f4 as F
    from research.notebooks.cvpr import _style as style
    from research.notebooks.cvpr import _traces as T

    # The paper's house palette and camera-ready rcParams. Call it once, before
    # any figure is drawn.
    style.apply_house_style()

    NOTEBOOK_VERSION = "2.9.0"
    return F, NOTEBOOK_VERSION, Path, T, mo, plt, re


@app.cell(hide_code=True)
def _(F, mo):
    # Section 0 states, from the data, which data this notebook reads.
    traces = F.load_traces()
    if F.DATA == "pooled":
        _name = "the pool of 2 batches (D132): the canonical batch and batch 2 together"
    elif not F.DATA:
        _name = "the canonical batch only, for history (`F4_DATA=canonical`)"
    else:
        _name = f"`F4_DATA={F.DATA}`"
    _answers = (traces.groupby(["model", "case"]).size().unstack("case")
                .reindex(index=list(F.MODELS), columns=list(F.CASES)).fillna(0).astype(int))
    _ntas = F.image_ntas()
    _ntas = _ntas[_ntas.case.isin(["restaurant", "street"])]
    _per = _ntas.groupby(["case", "nta2020"]).size()
    _images = "\n".join(
        f"| `{c}` | {traces[traces.case == c].image_idx.nunique():,} | "
        f"{_per[c].size} | {_per[c].median():g} | {_per[c].min()} to {_per[c].max()} |"
        for c in ("restaurant", "street")
    )
    if F.WEIGHTED:
        _weights = (
            "**The rates of sections 4.1, 4.2, 4.3 and 4.7 are weighted.** The pooled "
            "restaurant images are a sample stratified by NTA. Thus each restaurant and "
            "caption text gets the weight of its image (`nta_capacity / nta_n`), and a "
            "rate divides by the sum of the weights. A weighted rate is an estimate for "
            "all the restaurants of the F6 pool. Each street text gets the weight 1. "
            "The other sections do not use the weights."
        )
    else:
        _weights = "**No rate is weighted.** Each text counts 1."
    mo.vstack([
        mo.md(
            f"## 0. The data\n\nThis notebook reads {_name}, at "
            f"`{F.F4.relative_to(F.REPO_ROOT)}`. Figures go to "
            f"`research/notebooks/facct/figures/{F.FIGURES}/`. The answers for each "
            f"(model, case):"
        ),
        _answers,
        mo.md(
            "The images, and the images in 1 NTA. Restaurant and caption use the "
            "same images.\n\n| Case | Images | NTAs | Median images in 1 NTA | "
            f"Range |\n|------|--------|------|------------------------|-------|\n{_images}\n\n"
            + _weights
        ),
    ])
    return (traces,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The corpus

    1 row for each (case, model). `cut` counts the answers that stopped at
    `max_tokens`. The `source_run` column names the runs and the reruns (see
    the runbook, "Results of the F4 run"). The paper cites this table.
    """)
    return


@app.cell
def _(F, traces):
    corpus = F.corpus_table(traces)
    corpus
    return


@app.cell(hide_code=True)
def _(F, mo):
    scope_pick = mo.ui.radio(
        options=list(F.SCOPES), value="reasoning", label="Scope", inline=True
    )
    mo.md(f"### The scope of every section below\n\n{scope_pick}")
    return (scope_pick,)


@app.cell(hide_code=True)
def _(F, mo):
    by_pick = mo.ui.radio(
        options={"models within a case": "models", "cases within a model": "cases"},
        value="models within a case", label="Contrast",
    )
    case_pick = mo.ui.dropdown(options=list(F.CASES), value="restaurant", label="Case")
    model_pick = mo.ui.dropdown(options=list(F.MODELS), value="gemma-4-12b", label="Model")
    mode_pick = mo.ui.radio(
        options=["distinctive", "frequency"], value="distinctive", label="Mode"
    )
    ngram_pick = mo.ui.radio(
        options={"single words": 1, "word pairs": 2}, value="single words",
        label="Token",
    )
    max_words = mo.ui.slider(
        30, 250, value=120, step=10, label="Words in a cloud", show_value=True
    )
    extra_stop = mo.ui.text_area(
        value=" ".join(sorted(F.PROMPT_ECHO)),
        label="More stopwords (space or comma separated). The default is the prompt echo.",
    )
    mo.md(
        f"""
    ## 2. The clouds

    | Contrast | 1 cloud | Its background |
    |----------|---------|----------------|
    | models within a case | 1 model of the case | the other models, same case |
    | cases within a model | 1 case of the model | the other 2 cases, same model |

    The first asks what 1 model writes that the other models do not, on the
    same images and the same prompt. The second asks what 1 prompt draws out
    of 1 model. The model picker applies to the second contrast only, and to
    sections 4.4 and 4.6.

    Use `distinctive`, as in the parent notebook: the log-odds ratio with an
    informative Dirichlet prior (Monroe, Colaresi, and Quinn, 2008).

    {mo.hstack([mo.vstack([by_pick, case_pick, model_pick]), mo.vstack([mode_pick, ngram_pick]), mo.vstack([max_words, extra_stop])], gap=2)}
    """
    )
    return (
        by_pick,
        case_pick,
        extra_stop,
        max_words,
        mode_pick,
        model_pick,
        ngram_pick,
    )


@app.cell
def _(
    F,
    T,
    by_pick,
    case_pick,
    extra_stop,
    model_pick,
    ngram_pick,
    re,
    scope_pick,
    traces,
):
    stopwords = T.default_stopwords(re.split(r"[,\s]+", extra_stop.value or ""))
    counts = F.word_counts(traces, scope_pick.value, stopwords, ngram=int(ngram_pick.value))
    groups = F.contrast(counts, by_pick.value, case_pick.value, model_pick.value)
    return (groups,)


@app.cell(hide_code=True)
def _(F, groups, mo):
    _body = "\n".join(
        f"- `{g}`: {sum(groups[g].values()):,} words" for g in F.ordered(groups)
    ) or "_Nothing to draw. The Llama models have no trace: pick another scope._"
    mo.md(_body)
    return


@app.cell
def _(
    F,
    T,
    by_pick,
    case_pick,
    groups,
    max_words,
    mo,
    mode_pick,
    model_pick,
    plt,
    scope_pick,
):
    def _draw(group):
        w = T.cloud_weights(groups, group, mode=mode_pick.value, max_words=int(max_words.value))
        if not w:
            return mo.md(f"**{group}**: no word passes the filter.")
        _where = case_pick.value if by_pick.value == "models" else model_pick.value
        fig = F.plot_cloud(w, f"{group}, {_where}, {scope_pick.value}  ({mode_pick.value})",
                           model=group if by_pick.value == "models" else None)
        plt.close(fig)
        return mo.as_html(fig)

    clouds = mo.vstack([_draw(g) for g in F.ordered(groups)])
    clouds
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. The words behind the cloud
    """)
    return


@app.cell
def _(F, groups, max_words, mode_pick):
    words = F.word_table(groups, mode=mode_pick.value, max_words=int(max_words.value))
    words
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Structured extractions

    DeepSeek-V4-Flash-0731 read every text of both batches in chunks of 6,000
    characters (D130). `single-v2` (`urban_cues_single_v2.yaml`) has 5 classes, no
    `image` attribute, no `comparison`, and no `decision`. `street-v3` adds
    `photo_advice`, the camera advice of the street answers. A cue span can
    lie inside another one (a seen detail inside an advice sentence); both
    count. A text part with no cue gives a `none` item, which no count uses.
    The texts that got a copy of a prompt example in both passes are dropped
    (the line under this cell gives the count).

    **Every rate divides by the texts of 1 (case, model)**, also the texts
    with no span. A rate is "spans for each 100 texts". Under the scope `all`
    the unit is 1 answer. When section 0 says that the rates are weighted, a
    rate divides by the sum of the text weights, not by the count.

    **Warning: count a quotable span only.** A `match_lesser` span is a
    sentence that the extractor composed, and its offsets point at a fragment.
    The merge finds a misplaced span in the text again (`alignment_rescue`).
    `_f4.spans` drops the rows that stay unquotable. It also drops a later
    copy of the same class and text inside 1 text (an extractor loop).
    """)
    return


@app.cell
def _(F, mo):
    import traceback as _tb

    # Warning: show the traceback, never only the message. A bare message cannot
    # say WHICH line raised.
    try:
        raw = F.load_extractions()
        ext = F.spans(raw)
        ext_error = ""
    except FileNotFoundError as _exc:
        raw, ext, ext_error = None, None, str(_exc)
    except Exception:
        raw, ext, ext_error = None, None, _tb.format_exc()

    if not ext_error:
        _msg = (
            f"**{len(ext):,}** counted spans, of **{len(raw):,}** rows over "
            f"**{raw.doc_id.nunique():,}** texts. Copy suspects dropped: "
            f"{raw.attrs.get('copy_suspects', {})}."
        )
    elif "no extraction parquet" in ext_error:
        _msg = f"**No extraction data.** {ext_error}"
    else:
        # marimo keeps `auto_reload` off. Restart the kernel after a module edit.
        _msg = (
            "**The extraction module raised.** This is a code fault, not "
            "missing data. If `_f4.py` or `_extractions.py` changed after you "
            "opened this notebook, restart the kernel.\n\n```\n" + ext_error + "\n```"
        )
    mo.md(_msg)
    return ext, ext_error, raw


@app.cell
def _(F, ext_error, raw):
    coverage = F.coverage(raw) if not ext_error else None
    coverage
    return


@app.cell
def _(F, ext_error, raw, scope_pick):
    totals = None if ext_error else F.text_totals(raw, scope_pick.value)
    return (totals,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.1 What the model says, by case and model
    """)
    return


@app.cell
def _(F, ext, ext_error, mo, plt, scope_pick, totals):
    if ext_error:
        rates_view = mo.md("_No data._")
    else:
        _table = F.class_rates(ext, totals, scope_pick.value)
        _fig = F.plot_rates(_table, title=f"classes, {scope_pick.value}")
        plt.close(_fig)
        rates_view = mo.vstack([mo.as_html(_fig), _table])
    rates_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 Claims the photograph cannot support

    The rows of the parent risk panel: the model infers wealth, class, a
    demographic, or crime; the model writes about the people in the
    photograph; the model names a fault of the photograph.
    """)
    return


@app.cell
def _(F, ext, ext_error, mo, plt, scope_pick, totals):
    if ext_error:
        risk_view = mo.md("_No data._")
    else:
        _table = F.risk_panel(ext, totals, scope_pick.value)
        _fig = F.plot_rates(_table, title=f"risk panel, {scope_pick.value}")
        plt.close(_fig)
        risk_view = mo.vstack([mo.as_html(_fig), _table])
    risk_view
    return


@app.cell(hide_code=True)
def _(F, mo):
    attr_pick = mo.ui.dropdown(
        options={f"{c}.{a}": (c, a) for c, a in F.ATTRIBUTES},
        value="inference.kind", label="Attribute",
    )
    mo.md(
        f"""
    ### 4.3 The values of 1 attribute

    {attr_pick}

    The rate of each declared value, for each 100 texts. `inference.kind`
    holds the values that the single-image schemas add: `location`, `time`,
    `brand_knowledge`, `institution_knowledge`, and in x2 `atmosphere`,
    `business_type`, `neighborhood_character`, `popularity`.
    `uncertainty.reason` gains `illegible`, `identity`, and `knowledge` in x2.
    `photo_advice.kind` exists for street only.
    """
    )
    return (attr_pick,)


@app.cell
def _(F, attr_pick, ext, ext_error, mo, plt, scope_pick, totals):
    if ext_error:
        attr_view = mo.md("_No data._")
    else:
        _cls, _attr = attr_pick.value
        _table = F.attribute_rates(ext, totals, scope_pick.value, _cls, _attr)
        if _table.empty:
            attr_view = mo.md("_No span carries this attribute in this scope._")
        else:
            _fig = F.plot_rates(_table, title=f"{_cls}.{_attr}, {scope_pick.value}")
            plt.close(_fig)
            attr_view = mo.vstack([mo.as_html(_fig), _table])
    attr_view
    return


@app.cell(hide_code=True)
def _(by_pick, case_pick, mo, model_pick):
    unit_pick = mo.ui.radio(
        options={
            "a thing the model named": "class_text",
            "a class and its attributes": "class_attr",
            "a class": "class",
        },
        value="a thing the model named",
        label="Unit of the distinctive score",
    )
    _target = (
        f"`{model_pick.value}` against the other models of `{case_pick.value}`"
        if by_pick.value == "models"
        else f"`{case_pick.value}` against the other cases, for `{model_pick.value}`"
    )
    mo.md(
        f"""
    ### 4.4 What 1 group names that the others do not

    {unit_pick}

    The contrast of section 2, with a claim as the unit instead of a word.
    Now: {_target}. The x2 schemas have no `decision` class, thus no class is
    left out.
    """
    )
    return (unit_pick,)


@app.cell
def _(
    F,
    by_pick,
    case_pick,
    ext,
    ext_error,
    model_pick,
    scope_pick,
    unit_pick,
):
    distinctive = (
        None if ext_error
        else F.distinctive(ext, scope_pick.value, by_pick.value, case_pick.value,
                           model_pick.value, unit=unit_pick.value, max_rows=60)
    )
    distinctive
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.5 The vocabulary report

    What `normalize` moved into `other`. The extraction used guided decoding
    on the enums of the x2 schemas, thus this table is usually empty. A value in
    it means the shared `VOCABULARY` of `_extractions.py` lacks a value of the
    schema.
    """)
    return


@app.cell
def _(F, ext, ext_error):
    vocabulary = F.vocabulary_report(ext) if not ext_error else None
    vocabulary
    return


@app.cell(hide_code=True)
def _(F, mo):
    quote_class = mo.ui.dropdown(options=list(F.CLASSES), value="inference", label="Class")
    mo.md(
        f"""
    ### 4.6 Quotes

    {quote_class}

    15 verbatim spans of the case and the model picked in section 2. The text
    at the offsets IS the span, thus a reader can find it in the text that
    `doc_id` names.
    """
    )
    return (quote_class,)


@app.cell
def _(F, case_pick, ext, ext_error, model_pick, quote_class, scope_pick):
    quotes = (
        None if ext_error
        else F.quotes(ext, scope_pick.value, case_pick.value, model_pick.value,
                      quote_class.value, n=15)
    )
    quotes
    return


@app.cell(hide_code=True)
def _(F, mo):
    tva_pick = mo.ui.dropdown(
        options=["inference: any of the 4", *[f"inference: {k}" for k in F.RISK_KINDS],
                 "person_reference: any"],
        value="person_reference: any", label="Claim",
    )
    mo.md(
        f"""
    ### 4.7 The trace against the answer

    {tva_pick}

    The 4 thinking models write a trace and a final answer for the same
    image. The unit is 1 answer whose 2 texts were both read; the scope picker
    has no effect here. The weights of section 0 apply here too. For each
    100 answers:

    | Row | Answers whose ... |
    |-----|-------------------|
    | `trace` | trace holds the claim |
    | `answer` | final answer holds the claim |
    | `trace_only` | trace holds the claim and final answer does not |

    The table also gives `kept`: among the answers whose trace holds the
    claim, the share whose final answer holds it too. "Any of the 4" is an
    inference of wealth, class, a demographic, or crime.

    **Warning: the 2 texts are 2 extractions.** The extractor reads the trace
    and the answer apart. A claim that the answer words in another way
    still counts in `answer`; a claim that the extractor misses in 1 text
    shows as a difference. Read the quotes (section 4.6, scope by field)
    before you cite a gap.
    """
    )
    return (tva_pick,)


@app.cell
def _(F, ext, ext_error, mo, plt, raw, tva_pick):
    if ext_error:
        tva = None
        tva_view = mo.md("_No data._")
    else:
        tva = F.trace_against_answer(ext, raw)
        _fig = F.plot_rates(F.trace_answer_rows(tva, tva_pick.value), unit="answers",
                            of="answers", title=tva_pick.value)
        plt.close(_fig)
        tva_view = mo.vstack([mo.as_html(_fig), tva[tva.claim == tva_pick.value].reset_index(drop=True)])
    tva_view
    return


@app.cell(hide_code=True)
def _(F, mo):
    md_case = mo.ui.dropdown(options=list(F.CASES), value="restaurant", label="Case")
    md_field = mo.ui.dropdown(options=list(F.FIELDS), value="response", label="Field")
    md_all = mo.ui.checkbox(value=False, label="Show the categories that do not differ too")
    mo.md(
        f"""
    ### 4.8 The model as a hyperparameter

    {md_case} {md_field} {md_all}

    Does the choice of the model change what a text holds? For each category
    (each class, each value of each attribute, and the 4 risk kinds
    together), the outcome of 1 text is 0 or 1: the text holds 1 or more spans
    of the category. The test is the **Pearson chi-square test** of the 2 x k
    table (holds it or not, by model), 1 test for each (case, field,
    category), with Bonferroni over every test of the table (all cases and
    fields). All the models read the same images, thus the samples are
    paired; the chi-square test ignores this and is mostly conservative
    here. The `response` field compares the 6 models, and the `reasoning`
    field compares the 4 thinking models. A test uses only the images where
    every model has a text that was read. The scope picker has no effect here.
    The rates are not weighted: each model reads the same images, thus a
    weight does not change a comparison of the models.

    `range_pp` is the largest rate minus the smallest rate, in percentage
    points, with a 95% interval from a bootstrap over the images.

    **Warning: the interval of `range_pp` is not a test.** The range of 6
    rates is above 0 even when the models are equal, because each rate has
    sampling noise. Read `p_bonf` for the test. A noise floor (the same model
    run 2 times) gives the range to expect with no model difference.

    **Warning: the texts have different lengths.** The thinking models write
    long answers and the Llama models write short ones. A longer text has
    more chances to hold a category. This test stays the primary test: the
    unit is 1 completion, and the length of a completion is a property of
    the model. Section 4.9 is a secondary check inside length bands.
    """
    )
    return md_all, md_case, md_field


@app.cell
def _(F, ext, ext_error, raw):
    # Slow: each test of the table uses 1,000 bootstrap draws over the images.
    model_diffs = None if ext_error else F.model_differences(ext, raw)
    return (model_diffs,)


@app.cell
def _(F, md_all, md_case, md_field, mo, model_diffs, plt):
    if model_diffs is None:
        md_view = mo.md("_No data._")
    else:
        _part = model_diffs[(model_diffs.case == md_case.value) & (model_diffs.field == md_field.value)]
        if not md_all.value:
            _part = _part[_part.significant]
        _part = _part.sort_values("range_pp", ascending=False).dropna(axis=1, how="all")
        _items = [F.model_difference_summary(model_diffs)]
        try:
            _fig = F.plot_model_differences(model_diffs, md_case.value, md_field.value,
                                            title=f"{md_case.value}, {md_field.value}: the 20 largest ranges")
            plt.close(_fig)
            _items.append(mo.as_html(_fig))
        except ValueError as _exc:
            _items.append(mo.md(f"_{_exc}._"))
        _items.append(_part.reset_index(drop=True))
        md_view = mo.vstack(_items)
    md_view
    return


@app.cell(hide_code=True)
def _(F, mo):
    ml_case = mo.ui.dropdown(options=list(F.CASES), value="restaurant", label="Case")
    ml_field = mo.ui.dropdown(options=list(F.FIELDS), value="response", label="Field")
    mo.md(
        f"""
    ### 4.9 Secondary check: the model differences inside length bands

    {ml_case} {ml_field}

    **This is not the primary result.** Section 4.8 compares 1 completion of
    each model, and a model that writes long texts does so as part of the
    model. This check asks a narrower question: do the models still differ
    when their texts have about the same length?

    The texts of 1 (case, field) go into 4 bands, by the quartiles of the
    word count of all models together. Inside 1 band, the test is the same
    Pearson chi-square test of section 4.8. A model enters the test of a band
    when it has 30 texts or more in the band. Bonferroni covers every test of
    every band. The texts of 1 band come from different images, thus this
    test is not paired.

    **Warning: a band often holds only 2 to 4 models.** A short band holds
    few texts of the thinking models, and a long band holds few texts of the
    Llama models. Read the `models` column before you compare 2 bands.
    """
    )
    return ml_case, ml_field


@app.cell
def _(F, ext, ext_error, raw, traces):
    by_length = None if ext_error else F.model_differences_by_length(ext, raw, traces)
    return (by_length,)


@app.cell
def _(F, by_length, ml_case, ml_field, mo, model_diffs):
    if by_length is None or model_diffs is None:
        ml_view = mo.md("_No data._")
    else:
        _sum = F.length_summary(by_length, model_diffs)
        _part = by_length[(by_length.case == ml_case.value) & (by_length.field == ml_field.value)
                          & by_length.significant]
        ml_view = mo.vstack([
            _sum[(_sum.case == ml_case.value) & (_sum.field == ml_field.value)].reset_index(drop=True),
            _part.sort_values(["band", "range_pp"], ascending=[True, False]).dropna(axis=1, how="all")
            .reset_index(drop=True),
        ])
    ml_view
    return


@app.cell(hide_code=True)
def _(F, mo):
    nt_case = mo.ui.dropdown(options=list(F.CASES), value="restaurant", label="Case")
    nt_field = mo.ui.dropdown(options=list(F.FIELDS), value="response", label="Field")
    nt_min = mo.ui.number(start=2, stop=200, step=1, value=F.nta_min_images(),
                          label="Minimum images in 1 NTA")
    _per = F.image_ntas().groupby(["case", "nta2020"]).size().groupby("case").median()
    _size = (
        f"This data has a median of {_per['restaurant']:g} restaurant images and "
        f"{_per['street']:g} street views in 1 NTA. The default minimum is "
        f"{F.nta_min_images()} images."
    )
    if min(_per["restaurant"], _per["street"]) < 20:
        _size += (
            "\n\n**Warning: this sample is too small for these tests.** A half of 1 NTA "
            "holds only a few images. Thus `within` is near 0, and the tests have almost "
            "no power. The pooled data (the default, with no `F4_DATA`) gives the real tests."
        )
    mo.md(
        f"""
    ### 4.10 The model and the NTA findings

    {nt_case} {nt_field} {nt_min}

    Section 4.8 asks if the model changes what a text holds. This section asks
    if the model changes a finding about places. The rate of a category in
    1 NTA is the share of texts of the NTA that hold it. The texts are the
    same complete images as in section 4.8. The rate is not weighted.

    **Test 1: the NTA ranks.** Do 2 models put the NTAs in the same order? The
    images of each NTA go at random into 2 halves, 50 times. `within` is the
    Spearman correlation of the NTA rates of 1 model on half A with the same
    model on half B: the agreement that the noise of the images permits.
    `between` is the same with 2 different models. `gap` = `within` -
    `between`. A gap above 0 shows that the model changes the NTA ranks. Both
    correlations use halves of the same size, thus they have the same noise.

    **Test 2: the NTA features.** For each model, the Spearman correlation
    over the NTAs between the rate of a category and 1 ACS 2024 feature of
    the NTA: % Black non-Hispanic, % white non-Hispanic, % Asian
    non-Hispanic, % with a bachelor's degree or higher, and the median
    household income. Bonferroni covers every test of all cases and fields.
    The sign table gives +1, -1, or 0 (not significant) for each model.

    {_size}

    **Warning: a sign table row with 1 significant model is not a test of a
    difference.** A correlation that is significant for 1 model and not for
    another can come from noise. Compare the `rho` values, not only the signs.
    """
    )
    return nt_case, nt_field, nt_min


@app.cell
def _(F, ext, ext_error, nt_min, raw):
    # Slow: test 1 uses 50 random splits for each (case, field).
    nta_agree, nta_cor = (None, None) if ext_error else F.nta_tests(ext, raw, min_images=int(nt_min.value))
    return nta_agree, nta_cor


@app.cell
def _(F, mo, nt_case, nt_field, nta_agree, nta_cor):
    if nta_agree is None or not len(nta_agree):
        nt_view = mo.md("_No data._")
    else:
        _ag = nta_agree[(nta_agree.case == nt_case.value) & (nta_agree.field == nt_field.value)]
        _cor = nta_cor[(nta_cor.case == nt_case.value) & (nta_cor.field == nt_field.value)]
        _signs = F.feature_disagreement(nta_cor) if len(nta_cor) else nta_cor
        _signs = _signs[(_signs.case == nt_case.value) & (_signs.field == nt_field.value)
                        & (_signs.significant_models > 0)]
        nt_view = mo.vstack([
            mo.md(f"**Test 1.** NTAs: {int(_ag.ntas.max()) if len(_ag) else 0}. "
                  f"Median `within` {_ag.within.median():.3f}, median `between` {_ag.between.median():.3f}."),
            _ag.drop(columns=["case", "field"]).sort_values("gap", ascending=False).reset_index(drop=True),
            mo.md(f"**Test 2.** Tests (all cases and fields): {len(nta_cor)}. "
                  f"Significant: {int((nta_cor.sign != 0).sum())}. The rows with 1 significant model or more:"),
            _signs.drop(columns=["case", "field"]).reset_index(drop=True),
            mo.md("The correlations of this case and field, by p:"),
            _cor.drop(columns=["case", "field"]).sort_values("p").reset_index(drop=True),
        ])
    nt_view
    return


@app.cell(hide_code=True)
def _(F, ext, ext_error, mo, nt_case, nt_field):
    _cats = [] if ext_error else list(F.difference_categories(ext.head(1)))
    qn_cat = mo.ui.dropdown(options=_cats, searchable=True, label="Category",
                            value="inference: kind = popularity" if "inference: kind = popularity" in _cats else None)
    qn_feature = mo.ui.dropdown(options=list(F.NTA_FEATURES), value="median_household_income", label="ACS feature")
    qn_k = mo.ui.slider(start=1, stop=30, step=1, value=5, label="k NTAs on each side", show_value=True)
    qn_model = mo.ui.dropdown(options=["all models", *F.MODELS], value="all models", label="Model")
    qn_n = mo.ui.number(start=1, stop=200, step=1, value=15, label="Quotes on each side")
    qn_seed = mo.ui.number(start=0, stop=10**6, step=1, value=777, label="Seed")
    mo.md(
        f"""
    #### 4.10.1 Quotes from the top-k and the bottom-k NTAs

    {qn_cat} {qn_feature} {qn_k}

    {qn_model} {qn_n} {qn_seed}

    This viewer uses the case, the field and the minimum images of section 4.10
    (now: {nt_case.value}, {nt_field.value}). It ranks the NTAs with residents
    by the ACS feature, and takes the `k` NTAs with the highest value and the `k`
    NTAs with the lowest value. Then it gives a random sample of the spans of
    the category in the texts of each set. The `context` column shows the text
    around each span, with the span in [brackets].

    The "texts" rate is the share of the texts of the set that hold the
    category 1 time or more. A difference in this rate between the sets is
    not a test. Section 4.10 gives the test (the Spearman correlation over all
    the NTAs).

    **Warning: the quotes are a random sample, not typical examples.** Use
    them to read what the category holds. Do not count them.
    """
    )
    return qn_cat, qn_feature, qn_k, qn_model, qn_n, qn_seed


@app.cell
def _(F, ext, ext_error, mo, nt_case, nt_field, nt_min, qn_cat, qn_feature, qn_k, qn_model, qn_n, qn_seed, traces):
    if ext_error or not qn_cat.value:
        qn_view = mo.md("_No data._")
    else:
        _sets = F.extreme_ntas(nt_case.value, qn_feature.value, int(qn_k.value), min_images=int(nt_min.value))
        _parts = []
        for _side, _label in (("top", "highest"), ("bottom", "lowest")):
            _ntas = _sets[_side]
            _q = F.nta_quotes(ext, nt_case.value, nt_field.value, qn_cat.value, _ntas.nta2020,
                              model=qn_model.value, n=int(qn_n.value), seed=int(qn_seed.value), traces=traces)
            _texts = _q.attrs.get("texts", 0)
            _with = _q.attrs.get("texts_with", 0)
            _rate = f"{100 * _with / _texts:.1f}%" if _texts else "n/a"
            _parts += [
                mo.md(f"**The {len(_ntas)} NTAs with the {_label} `{qn_feature.value}`.** "
                      f"Spans: {_q.attrs.get('spans', 0):,}. Texts with the category: "
                      f"{_with:,} of {_texts:,} ({_rate})."),
                _ntas,
                _q.dropna(axis=1, how="all") if len(_q) else mo.md("_No span of this category in these NTAs._"),
            ]
        qn_view = mo.vstack(_parts)
    qn_view
    return


@app.cell(hide_code=True)
def _(F, mo):
    cc_class = mo.ui.dropdown(options=["all", *F.CROSS_CLASSES], value="all", label="Class")
    mo.md(
        f"""
    ### 4.11 Findings in both cases, and findings in 1 case only

    {cc_class}

    Which findings occur in the restaurant case and in the street case? Which
    findings occur in 1 case only? This section compares only `restaurant`
    and `street`. It uses 2 types of finding:

    | Finding | Estimate in 1 case | Significant in 1 case | Difference between the 2 cases |
    |---------|--------------------|-----------------------|--------------------------------|
    | `rates` (section 4.8) | the rate of 1 model minus the rate of the other models, in percentage points | the Pearson chi-square test of the 2 x 2 table (the model or the other models; holds the category or not), Bonferroni over the tests of the case | `_f4.difference_test` on the 2 estimates and their standard errors |
    | `nta_features` (section 4.10, test 2) | the Spearman `rho` of 1 model, 1 category, and 1 ACS feature | the sign table of section 4.10, with its Bonferroni | the Fisher z form of `difference_test`: `arctanh(rho)`, with the standard error `1 / sqrt(ntas - 3)` |

    The `rates` findings use the unit of section 4.8: 1 text of 1 complete
    image, not weighted. For `reasoning`, the other models are the other 3
    thinking models. For `response`, they are the other 5 models. The
    standard error is the standard error of 2 independent proportions. The
    model and the other models read the same images, thus the 2 groups are
    not truly independent. The tests ignore this, as in section 4.8.

    Each table has 1 row for each finding that exists in both cases.
    Bonferroni covers every difference test of 1 table (`diff_p_bonf`). The
    class of 1 finding:

    | Class | Rule |
    |-------|------|
    | `both, same sign` | significant in each case, with the same sign |
    | `both, opposite sign` | significant in each case, with opposite signs |
    | `restaurant only`, `street only` | significant in that case only, and the difference test is significant |
    | `1 case only, not different` | significant in 1 case only, but the difference test is not significant |
    | `neither` | significant in no case |

    **Warning: a finding that is significant in 1 case and not in the
    other is not evidence of a difference between the cases.** Only the
    difference test can show that difference.

    The figure shows the `rates` findings, 1 panel for each field and 1 dot
    for each (category, model). The x axis is the restaurant estimate, and
    the y axis is the street estimate. A dot on the dashed line has the same
    estimate in both cases. A dot in the top-left or the bottom-right
    quarter has opposite signs in the 2 cases.

    The last table puts test 1 of section 4.10 (`within`, `between`, `gap`)
    of the 2 cases side by side. It is a description only, with no test. The
    `nta_features` and test 1 tables follow the NTA minimum of section 4.10.
    """
    )
    return (cc_class,)


@app.cell
def _(F, ext, ext_error, nta_agree, nta_cor, raw):
    if ext_error:
        cc_rates, cc_cor, cc_ranks = None, None, None
    else:
        cc_rates = F.crosscase_rates(F.model_contrasts(ext, raw))
        _has_cor = nta_cor is not None and len(nta_cor)
        _has_ag = nta_agree is not None and len(nta_agree)
        cc_cor = F.crosscase_correlations(nta_cor) if _has_cor else None
        cc_ranks = F.crosscase_ranks(nta_agree) if _has_ag else None
    return cc_cor, cc_ranks, cc_rates


@app.cell
def _(F, cc_class, cc_cor, cc_ranks, cc_rates, mo, plt):
    def _pick(table):
        part = table if cc_class.value == "all" else table[table["class"] == cc_class.value]
        return part.sort_values("diff_p").reset_index(drop=True)

    if cc_rates is None:
        cc_view = mo.md("_No data._")
    else:
        _only = cc_rates.attrs.get("one_case_only", {})
        _fig = F.plot_crosscase(cc_rates)
        plt.close(_fig)
        _items = [
            mo.md(
                f"**Summary.** `rates`: {len(cc_rates)} findings in both cases. "
                f"`nta_features`: {0 if cc_cor is None else len(cc_cor)} findings in both cases. "
                f"The `rates` findings that exist in 1 case only stay out of the tables: "
                + ", ".join(f"{c} {n}" for c, n in _only.items()) + "."
            ),
            F.crosscase_summary(cc_rates, cc_cor),
            mo.as_html(_fig),
            mo.md("**The `rates` findings**, by `diff_p`:"),
            _pick(cc_rates),
        ]
        if cc_cor is not None:
            _items += [mo.md("**The `nta_features` findings**, by `diff_p`:"), _pick(cc_cor)]
        if cc_ranks is not None:
            _items += [mo.md("**Test 1 of section 4.10, side by side** (a description only):"), cc_ranks]
        cc_view = mo.vstack(_items)
    cc_view
    return


@app.cell(hide_code=True)
def _(F, mo):
    st_case = mo.ui.dropdown(options=["restaurant", "caption", "street"], value="restaurant", label="Case")
    st_field = mo.ui.dropdown(options=list(F.FIELDS), value="response", label="Field")
    st_by = mo.ui.radio(options={"NTA": "nta", "cuisine": "cuisine"}, value="cuisine", label="Strata", inline=True)
    st_min = mo.ui.number(start=3, stop=200, step=1, value=F.nta_min_images(), label="Images in 1 NTA, minimum")
    st_cuisine = mo.ui.number(start=10, stop=1000, step=10, value=100, label="Images of 1 cuisine, minimum")
    mo.md(
        f"""
    ### 4.12 Strata: the NTA and the cuisine

    Sections 4.12 and 4.13 put the texts of the complete images (section
    4.8) into strata: the NTA of the image, or the cuisine of the
    restaurant. The cuisine is the DOHMH `cuisine_description` of the
    restaurant (`camis`); the street views have no cuisine. A cuisine with
    fewer images than the minimum below is not a stratum. The rates are not
    weighted.

    {mo.hstack([st_case, st_field, st_by], justify="start", gap=2)}
    {mo.hstack([st_min, st_cuisine], justify="start", gap=2)}

    **Warning: the cuisine and the NTA features go together.** The last table
    of this section gives, for each cuisine, the Spearman correlation over
    the NTAs of its share of the images with each ACS feature. When a
    cuisine goes with a feature, a cuisine difference and an NTA feature
    correlation can be 1 effect, not 2. The tests here do not separate them.
    """
    )
    return st_by, st_case, st_cuisine, st_field, st_min


@app.cell
def _(F, ext, ext_error, raw, st_by, st_case, st_cuisine, st_field, st_min):
    if ext_error or (st_by.value == "cuisine" and st_case.value == "street"):
        strata_valence = strata_uncertainty = None
    else:
        _a = (ext, raw, st_case.value, st_field.value, st_by.value, int(st_min.value), int(st_cuisine.value))
        strata_valence = F.valence_by_stratum(*_a)
        strata_uncertainty = F.uncertainty_by_stratum(*_a)
    return strata_uncertainty, strata_valence


@app.cell(hide_code=True)
def _(F, mo, st_case, st_min):
    sv_model = mo.ui.dropdown(options=["all models", *F.MODELS], value="all models", label="Model")
    sv_value = mo.ui.radio(options={"net valence": "net", "% bad": "pct_bad", "% good": "pct_good",
                                    "spans": "spans"}, value="net valence", label="Cell", inline=True)
    mo.md(
        f"""
    ### 4.12.1 The valence of the visual evidence, by category and stratum

    The unit is 1 `visual_evidence` span. `net` = (good - bad) / spans, from
    -1 (all bad) to +1 (all good). The category `any` holds every category.
    The table gives 1 row for each stratum and 1 column for each category.
    The inference spans have no valence in the x2 schemas; only
    `visual_evidence` has one.

    **The tests.**

    - Cuisine: the Pearson chi-square test of the cuisines x (good, bad,
      neutral) span counts, 1 test for each (model, category), Bonferroni.
      `cochran_ok` is false when the expected counts are too small.
      **Warning:** 1 text can hold many spans, and these spans are not
      independent. Thus the p values are too small.
    - NTA: the Spearman correlation over the NTAs of the NTA `net` with each
      ACS feature, as test 2 of section 4.10, Bonferroni. The NTA is the
      unit, thus the spans of 1 text do not make this test too strong.

    `all models` is shown but not tested: the models read the same images.

    {mo.hstack([sv_model, sv_value], justify="start", gap=2)}
    """
    )
    return sv_model, sv_value


@app.cell
def _(F, mo, st_by, st_case, strata_valence, sv_model, sv_value):
    if strata_valence is None:
        _view = mo.md("_No data: the street views have no cuisine._" if st_case.value == "street" else "_No data._")
    else:
        _t = strata_valence[strata_valence.model == sv_model.value]
        _wide = _t.pivot(index="group", columns="category", values=sv_value.value)
        _n = _t[_t.category == "any"].set_index("group").spans
        _wide.insert(0, "spans (any)", _n)
        _wide = _wide.sort_values("spans (any)", ascending=False)
        if st_by.value == "nta":
            _names = F.pd.read_parquet(F.ACS_PATH).set_index("nta2020").ntaname
            _wide.insert(0, "name", _names.reindex(_wide.index))
            _tests = F.valence_nta_features(strata_valence)
            _head = (f"**NTA features:** {len(_tests)} tests, "
                     f"{int((_tests.sign != 0).sum()) if len(_tests) else 0} significant.")
            _tests = _tests.sort_values("p") if len(_tests) else _tests
        else:
            _tests = F.valence_tests(strata_valence, "cuisine")
            _head = (f"**Chi-square over the cuisines:** {len(_tests)} tests, "
                     f"{int(_tests.significant.sum())} significant, {int(_tests.cochran_ok.sum())} meet "
                     "Cochran's rule.")
            _tests = _tests.sort_values("p")
        _view = mo.vstack([_wide.reset_index(), mo.md(_head), _tests.reset_index(drop=True)])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.12.2 Inference kinds by stratum

    The x2 schemas give an inference span a `kind` but no valence. The
    share of texts that hold each inference kind, by stratum, is the
    uncertainty table of section 4.13 with `inference: kind = ...` in place
    of `uncertainty`. A valence for inferences needs a decision (see the
    runbook).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.13 Uncertainty, the NTA features, the cuisine, and the NTA

    The unit is 1 text of a complete image. The outcome is 1 when the text
    holds an `uncertainty` span (`any`), or an uncertainty span of 1 reason
    (`illegible`, `out_of_view`, `identity`, `knowledge`, `other`). The
    strata come from the controls of section 4.12.

    **The tests.** Each model gives 1 text for each image, thus the texts of
    1 model are independent. `all models` is shown but not tested.

    - Across the strata (NTA or cuisine): the Pearson chi-square test of the
      strata x (holds, does not hold) text counts, 1 test for each (model,
      outcome), Bonferroni. `cochran_ok` is false when the expected counts
      are too small (often for 1 reason over many NTAs).
    - With the NTA features (NTA strata only): the Spearman correlation over
      the NTAs of the NTA share with each ACS feature, Bonferroni.
    - The cuisine and the NTA features: the last table gives the Spearman
      correlation over the NTAs of the share of each cuisine with each ACS
      feature. Read it before you give a cuisine or an NTA feature as the
      cause of a difference.
    """
    )
    return


@app.cell
def _(F, mo, st_by, st_case, st_cuisine, st_min, strata_uncertainty):
    if strata_uncertainty is None:
        _view = mo.md("_No data: the street views have no cuisine._" if st_case.value == "street" else "_No data._")
    else:
        _wide = strata_uncertainty.pivot_table(index=["group", "measure"], columns="model", values="pct")
        _wide = _wide.reindex(columns=["all models", *[m for m in F.MODELS if m in _wide.columns]])
        _texts = strata_uncertainty[(strata_uncertainty.model == "all models")
                                    & (strata_uncertainty.measure == "any")].set_index("group").texts
        _any = _wide.xs("any", level="measure").assign(texts=_texts).sort_values("all models")
        _tests = F.uncertainty_tests(strata_uncertainty, st_by.value).sort_values("p").reset_index(drop=True)
        _parts = [
            mo.md("**The share of texts with an uncertainty span (`any`), in percent, by stratum:**"),
            _any.reset_index(),
            mo.md(f"**Chi-square over the strata:** {len(_tests)} tests, {int(_tests.significant.sum())} "
                  f"significant, {int(_tests.cochran_ok.sum())} meet Cochran's rule."),
            _tests,
        ]
        if st_by.value == "nta":
            _cor = F.uncertainty_nta_features(strata_uncertainty)
            _parts += [mo.md(f"**NTA features:** {len(_cor)} tests, "
                             f"{int((_cor.sign != 0).sum()) if len(_cor) else 0} significant."),
                       _cor.sort_values("p").reset_index(drop=True) if len(_cor) else _cor]
        if st_case.value in ("restaurant", "caption"):
            _mix = F.cuisine_nta_mix(st_case.value, int(st_min.value), int(st_cuisine.value))
            _parts += [mo.md(f"**The cuisine and the NTA features:** {len(_mix)} tests, "
                             f"{int((_mix.sign != 0).sum())} significant. The significant rows:"),
                       _mix[_mix.sign != 0].drop(columns="model").rename(columns={"measure": "cuisine"})
                       .sort_values("p").reset_index(drop=True)]
        _view = mo.vstack(_parts)
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parent sections that do not port

    | Parent section | Why it is absent |
    |----------------|------------------|
    | 4.6 Which cues go with a win | It needs 2 images and a winner. F4 has 1 image and no judgment |
    | 4.7 The model against itself | It compares the cue valence with the winner. Same reason |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG and CSV files")
    mo.md(
        f"""
    ## 5. Export

    {save}

    The export writes every scope, every case, and every model, not only the
    current selection. The clouds use the `models within a case` contrast. The
    mode, the token, the stopwords, and the unit come from the controls above.
    """
    )
    return (save,)


@app.cell
def _(
    F,
    NOTEBOOK_VERSION,
    Path,
    T,
    extra_stop,
    mo,
    mode_pick,
    ngram_pick,
    raw,
    re,
    save,
    traces,
    unit_pick,
):
    if save.value:
        _dir = Path(__file__).resolve().parent / "figures" / F.FIGURES
        _stop = T.default_stopwords(re.split(r"[,\s]+", extra_stop.value or ""))
        _w = F.export(_dir, traces, raw, _stop, mode=mode_pick.value,
                      ngram=int(ngram_pick.value), unit=unit_pick.value)
        _msg = mo.md(
            f"Wrote {len(_w)} files to `research/notebooks/facct/figures/{F.FIGURES}/`.\n\n"
            f"Notebook {NOTEBOOK_VERSION}, `_f4` {F.__version__}."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
