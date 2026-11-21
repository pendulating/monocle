import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F4 — the aesthetic descriptors of 6 models

    F4 asks 6 models about 1 image at a time, in 3 cases. Each answer has a
    reasoning trace (`reasoning`, the 4 thinking models only) and a response
    (`response`). This notebook reads the words with which each model
    characterizes a place: "no-frills", "welcoming", "gritty", "cinematic".
    DeepSeek-V4-Flash extracts them with LangExtract. Only the `restaurant`
    and the `street` cases have descriptors; `caption` has none.

    The notebook reads the x2 extraction round (D130, D131). The data root
    comes from `_f4`: the pool of the 2 batches (D132) is the default, and
    `F4_DATA=canonical` reads the canonical batch only. The first round
    (D109 to D129) exists for the canonical batch only; read it with
    `F4_DATA=canonical F4DESC_ROUND=v1`, for history. The line under this
    cell names the data, the round, and the number of texts. 1 span is 1
    descriptor, with these attributes:

    | Attribute | Values |
    |-----------|--------|
    | `head` | the plain form of the descriptor ("hole-in-the-wall") |
    | `vibe` | yes when it names a feel, a mood, or an atmosphere |
    | `slot` | the frame noun ("feel", "vibe", "sense"), or none |
    | `aspect` | appearance, atmosphere, social_register, business_type, figurative, other; x2 adds evaluation (praise), size, composition (a quality of a frame) |
    | `negated` | yes when the text denies it ("not upscale") |
    | `target` | x2: `place` (true of the thing in the street), `current_photo` (true only of the photo the model got), `planned_photo` (the picture to take); round 1: street only, `place` or `photo` |

    | Case | Prompt | Images | Schema, x2 | Schema, round 1 |
    |------|--------|--------|------------|-----------------|
    | `restaurant` | Look and tell me more about this restaurant. | restaurant storefronts (F6) | `descriptors-v6-restaurant`, cues `single-v2` | `descriptors-v3` (+ `v3r`), cues `single-v1` |
    | `street` | Look and tell me a good angle to take a picture from. | street views of the residential NTAs | `descriptors-v6-street`, cues `street-v3` | `descriptors-v5`, cues `street-v2` |

    The line under this cell gives the number of images and NTAs of each case.

    ### The 2 tracks

    | Track | Spans | Question |
    |-------|-------|----------|
    | C | the descriptors inside the `inference` and `visual_evidence` cue spans | Which words does each model use to characterize what it sees? |
    | V | the descriptors with `vibe` yes, in the full text | Which feel does each model give a place? |

    **Warnings:**

    - Compare models inside 1 (case, field) only. The 2 cases use different
      schemas, prompts, and images. Section 11 compares the 2 cases only
      through the contrast of each model against the other models.
    - The street cue spans come from their own schema (`street-v3` in x2),
      with a class for camera advice. Street Track C is a conservative set:
      a manual check of the x2 smoke run (D129, D130) found that about half
      of the place words outside a Track C cue span are cue misses. The
      line under section 1 gives the share inside a cue span for these
      data. Section 3.2 checks the street signatures with a second filter.
      With no street cue file, Track C on street reads all descriptors.
    - x2: a cue span can lie inside another one; the innermost span gives
      the class. The texts that got a copy of a prompt example even after
      the retry pass are dropped (`copy_suspect`).
    - The Llama answers on street are short and often hold no descriptor
      (the line under section 1 gives the count). Their signatures rest on
      fewer texts.
    - A head counts 1 time for each text that holds it.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # `_f4desc` sits beside this notebook and puts `notebooks/cvpr/` on the path.
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import marimo as mo
    import matplotlib.pyplot as plt

    import _f4desc as D
    import _style as style

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.8.0"
    return D, NOTEBOOK_VERSION, Path, mo


@app.cell(hide_code=True)
def _(D, mo, raw, totals):
    import pandas as _pd

    _src = {
        "pooled": "the pool of the canonical batch and batch 2 (D132, `data/f4_traces/pooled/`)",
        "": "the canonical batch only (`data/f4_traces/`)",
    }.get(D.DATA, f"`{D.F4}`")
    _img = []
    for _c in D.CASES:
        _n = raw.loc[raw.case == _c, "image_idx"].nunique()
        _nta = _pd.read_parquet(D.NTA[_c])
        _per = _nta.groupby("nta2020" if "nta2020" in _nta else "NTA2020").size()
        _img.append(f"{_c}: {_n:,} images in {len(_per)} NTAs ({_per.min()} to {_per.max()} for each NTA)")
    _t = totals.unstack("model")
    _t = _t[[m for m in D.MODELS if m in _t.columns]].fillna(0).astype(int)
    mo.vstack([
        mo.md(
            f"**The data.** The notebook reads {_src}, extraction round `{D.ROUND}`. "
            f"{'; '.join(_img)}. The table gives the texts of each (case, field) and model, "
            "after the drop of the copy suspects. A 0 is a field that the model does not write."
        ),
        _t,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Coverage

    1 row for each (case, field, model). `empty_texts` counts the texts with
    no descriptor; the rates below keep them in the denominator.
    `photo_share` is the share of spans whose target is not `place`; it is
    0 for the restaurant case of round 1, which has no `target`.
    """)
    return


@app.cell
def _(D):
    raw = D.load_raw()
    spans = D.mark_cue_spans(D.spans(raw))
    totals = D.text_totals(raw)
    coverage = D.coverage(raw, spans)
    coverage
    return coverage, raw, spans, totals


@app.cell(hide_code=True)
def _(D, coverage, mo, raw, spans):
    _lines = []
    for _c in D.CASES:
        _r = spans[spans.case == _c]
        if D.has_cues(spans, _c):
            _lines.append(f"- {_c}: {_r.in_cue.astype(float).mean():.1%} of the descriptors lie inside an "
                          f"`inference` or a `visual_evidence` cue span.")
        else:
            _lines.append(f"- {_c}: no cue file. Track C keeps every descriptor.")
    _street = D.select(spans, "street", "response", "C", "place", cue_only=False)
    if D.has_cues(spans, "street") and len(_street):
        _lines.append(f"- street, response, target place: {_street.in_cue.astype(float).mean():.1%} of the "
                      "descriptors lie inside a cue span of Track C.")
    _cov = coverage.loc[("street", "response")]
    for _m in [m for m in D.MODELS if m.startswith("llama") and m in _cov.index]:
        _lines.append(f"- street, response, {_m}: {_cov.loc[_m, 'empty_texts']:,} of "
                      f"{_cov.loc[_m, 'texts']:,} texts hold no descriptor.")
    _drop = raw.attrs.get("copy_suspects", {})
    _lines.append(f"- Round `{D.ROUND}`. Texts dropped as copy suspects: {_drop or 'none'}.")
    mo.md("Track C keeps the descriptors inside the cue spans:\n\n" + "\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(D, mo, spans):
    _aspects = sorted(spans.aspect.dropna().unique())
    aspect_pick = mo.ui.multiselect(options=_aspects, value=_aspects, label="Aspects")
    case_pick = mo.ui.radio(options=list(D.CASES), value="restaurant", label="Case", inline=True)
    field_pick = mo.ui.radio(options=list(D.FIELDS), value="response", label="Field", inline=True)
    track_pick = mo.ui.radio(options={"C: characterization": "C", "V: vibe": "V"},
                             value="C: characterization", label="Track", inline=True)
    target_pick = mo.ui.radio(options=list(D.TARGETS), value="place", label="Target (street)", inline=True)
    cue_pick = mo.ui.checkbox(value=True, label="Track C: cue spans only")
    n_pick = mo.ui.slider(start=5, stop=30, step=1, value=12, label="Heads per model")
    mo.md(
        f"""
    ## 2. Controls

    {case_pick}

    {field_pick}

    {track_pick}

    {target_pick}

    {cue_pick}

    {aspect_pick}

    {n_pick}

    The `reasoning` field holds the 4 thinking models only. The target filter
    has no effect on the restaurant case. The target `photo` keeps both
    photo targets. The aspect filter keeps the chosen aspects; by default it
    keeps all of them.
    """
    )
    return (
        aspect_pick,
        case_pick,
        cue_pick,
        field_pick,
        n_pick,
        target_pick,
        track_pick,
    )


@app.cell
def _(
    D,
    aspect_pick,
    case_pick,
    cue_pick,
    field_pick,
    spans,
    target_pick,
    track_pick,
):
    part = D.select(spans, case_pick.value, field_pick.value, track_pick.value,
                    target_pick.value if case_pick.value == "street" else "both", cue_pick.value)
    part = part[part.aspect.isin(aspect_pick.value)]
    tag = f"{case_pick.value} / {field_pick.value} / track {track_pick.value}" + (
        f" / {target_pick.value}" if case_pick.value == "street" else "")
    return part, tag


@app.cell(hide_code=True)
def _(mo, part, tag):
    mo.md(f"""
    ## 3. Signature vocabulary

    **{tag}**: {len(part):,} head uses.

    The log-odds z of each head for 1 model against the other models of the
    same (case, field), with an informative Dirichlet prior. `share` is the
    share of the model's texts that hold the head. `ntas` and `nta_share` tell
    if the head spreads over the city or sits in a few neighborhoods. A head
    needs 5 texts over all the models together.
    """)
    return


@app.cell
def _(D, part, totals):
    signature = D.signatures(part, totals, max_rows=40)
    signature
    return (signature,)


@app.cell
def _(D, n_pick, signature, tag):
    D.plot_signatures(signature, n=n_pick.value, title=tag) if not signature.empty else None
    return


@app.cell(hide_code=True)
def _(mo):
    strict_pick = mo.ui.checkbox(value=True, label="Only words that no other model writes anywhere in its texts")
    mo.md(
        f"""
    ### 3.1 Words that only 1 model uses

    A signature word (section 3) is over-used, not unique. This table gives
    the words that only 1 model uses as a descriptor, in the case and field
    above, with a test.

    **1 word, many spellings.** The unit is the spelling key
    (`_f4desc.head_key`). The key removes the accents, puts British spelling
    in American spelling (a table of 1,730 words from breame 0.1.2), and
    removes the spaces, hyphens, slashes and a plural "s". Thus "closed-down"
    and "closed down", or "single-storey" and "single story", are 1 word.
    `head` gives the most frequent form; `forms` lists all the forms of the
    model. The key does not join different words with near spellings
    (carved, curved), and it does not join synonyms (flat profile, low
    profile).

    **The test (exact binomial).** All models read the same images with the
    same prompt. If a word has no link to a model, each text that holds it
    comes from model M with the probability `share_M`, the share of M among
    the texts (about 1/6 for the answers, 1/4 for the traces). The chance
    that all `k` texts come from M is `share_M ** k`. `p_bonf` multiplies
    this by the number of candidate words (Bonferroni); the table keeps
    `p_bonf` < 0.05. The line under the checkbox gives the number of texts
    that a word needs to pass.

    **The text check.** The candidate test counts the other models over all
    their descriptors, not only the selection. `others_in_text` counts the
    texts of the other models that hold the word in any of its spellings:
    the extractor can miss a word, a search cannot. A word with `others_in_text` 0 is unique in
    the text, and the checkbox keeps only those.

    The selection (track, target, aspects) comes from the controls.

    {strict_pick}
    """
    )
    return (strict_pick,)


@app.cell
def _(D, mo, part, spans, strict_pick, totals):
    import math as _math

    _e = D.exclusive_heads(part, spans, totals)
    _n = _e.attrs.get("candidates", 0)
    _need = ""
    if _n and len(part):
        _t = totals.xs((part.case.iloc[0], part.field.iloc[0]), level=("case", "field"))
        _k = [_math.floor(_math.log(0.05 / _n) / _math.log(v / _t.sum())) + 1 for v in _t]
        _need = f" A word needs {min(_k)} to {max(_k)} texts to pass (it depends on the share of the model)."
    if not _e.empty and strict_pick.value:
        _e = _e[_e.others_in_text == 0].reset_index(drop=True)
    mo.vstack([
        mo.md(f"{_n:,} candidate words (used by 1 model only); {len(_e)} pass.{_need}"),
        _e if not _e.empty else mo.md("_No word passes._"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Street: the check of D129

    The street cue run misses some of the place words (D129, D130). This table
    compares the street place signatures of track C with a second filter:
    the descriptors with target place that are not inside a `photo_advice`
    effect span or an `image_artifact` span. The field comes from the
    controls. If `shared_top` is high and `spearman` is near 1, the misses do
    not change the result. `only_c2` names the top heads that only the
    second filter gives.
    """)
    return


@app.cell
def _(D, field_pick, spans, totals):
    D.robustness(spans, totals, field_pick.value) if D.has_cues(spans, "street") else None
    return


@app.cell(hide_code=True)
def _(D, mo, part):
    cloud_model = mo.ui.dropdown(options=[m for m in D.MODELS if m in set(part.model)],
                                 value=next(m for m in D.MODELS if m in set(part.model)),
                                 label="Model")
    mo.md(
        f"""
    ## 4. Clouds

    {cloud_model}

    The word size is the log-odds z of section 3.
    """
    )
    return (cloud_model,)


@app.cell
def _(D, cloud_model, signature, tag):
    _s = signature[signature.model == cloud_model.value] if not signature.empty else signature
    D.plot_cloud(dict(zip(_s["head"], _s.z)), f"{cloud_model.value}: {tag}", model=cloud_model.value) if len(_s) else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 The most frequent heads

    The raw counts, not the contrast. A head that all models share ("urban",
    "small") is frequent here and absent from section 3.
    """)
    return


@app.cell
def _(D, part, totals):
    D.top_heads(part, totals, n=15)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Concentration

    How monolithic is the vocabulary of each model? A low entropy and a high
    top-10 share mean that the model gives most places the same few words.

    The plug-in entropy grows with the sample, and the models give different
    numbers of spans. `entropy_rare` draws the same number of uses (`k`, the
    smallest model) 200 times. Compare models on `entropy_rare`.

    The second table gives the mean share of NTAs in which the 20 most
    frequent heads of a model appear. A high value means the model writes the
    same words about every neighborhood.
    """)
    return


@app.cell
def _(D, part):
    concentration = D.concentration(part)
    concentration
    return (concentration,)


@app.cell
def _(D, concentration, tag):
    D.plot_concentration(concentration, title=tag) if not concentration.empty else None
    return


@app.cell
def _(D, part):
    D.nta_overlap(part)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Vibe and frames

    ### 6.1 Vibe descriptors for each 100 texts

    All cases, fields, and models. The silent texts count in the denominator.
    The rates are not weighted: each model reads the same images.
    """)
    return


@app.cell
def _(D, spans, totals):
    D.vibe_rates(spans, totals)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.2 The frame nouns

    The `slot` of the vibe spans of the current selection: "a cozy **feel**",
    "a sense of **grandeur**". Set the track to V to read the vibe spans only.
    """)
    return


@app.cell
def _(D, part):
    D.frames(part, n=8)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6.3 Aspects

    The share of each `aspect` among the first uses, for each (case, field,
    model). All descriptors, not only the current track.
    """)
    return


@app.cell
def _(D, spans):
    D.aspect_shares(spans)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Where the descriptors sit

    For the case and the field above: the share of the descriptors (first
    uses) of each model in each innermost cue span. `none` is no cue span;
    `inference: <kind>` is an inference of that kind. The texts that the cue
    file lacks are left out. The columns of 1 row add to 1.
    """)
    return


@app.cell
def _(D, case_pick, field_pick, spans):
    D.cue_kind_shares(spans, case_pick.value, field_pick.value)
    return


@app.cell(hide_code=True)
def _(D, mo):
    social_pick = mo.ui.multiselect(options=list(D.SOCIAL_KINDS), value=list(D.SOCIAL_KINDS),
                                    label="Inference kinds")
    mo.md(
        f"""
    ## 8. Descriptors inside social inferences

    {social_pick}

    The descriptors that each model puts inside an inference about wealth,
    class, a demographic, crime, or safety: the words that carry a social
    judgment of the place. `texts` counts the texts of the model that hold
    the word there; `per_1000` divides by all texts of the model in the
    (case, field). The case and the field come from the controls above; the
    track and the target do not apply.

    **Warning:** a count under 5 texts comes from fewer than 5 images. Read
    the quotes (section 10) before you cite a word.
    """
    )
    return (social_pick,)


@app.cell
def _(D, case_pick, field_pick, social_pick, spans, totals):
    D.social_heads(spans, totals, case_pick.value, field_pick.value, kinds=social_pick.value, n=12)
    return


@app.cell(hide_code=True)
def _(coverage, mo):
    _s = coverage.loc["street", "photo_share"].reset_index()
    mo.md(
        f"""
    ## 9. Street: the place against the photo

    The street prompt asks for a camera angle. Thus the answers describe the
    scene (`place`), the photo the model got (`current_photo`, "cluttered"),
    and the picture to take (`planned_photo`, "a moody shot", "a cinematic
    look"). The share of spans that are not `place` goes from
    {_s.photo_share.min():.0%} to {_s.photo_share.max():.0%}. Set the case
    to street and the target to `place` to compare the models on the scene
    only; set it to `planned_photo` for the style of the planned picture.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, signature):
    _heads = sorted(set(signature["head"])) if not signature.empty else ["none"]
    quote_model = mo.ui.dropdown(options=sorted(set(signature.model)) if not signature.empty else ["none"],
                                 value=signature.model.iloc[0] if not signature.empty else "none",
                                 label="Model")
    quote_head = mo.ui.dropdown(options=_heads, value=signature["head"].iloc[0] if not signature.empty else "none",
                                label="Head")
    mo.md(
        f"""
    ## 10. Quotes

    {quote_model} {quote_head}

    12 spans of the head in their text, for the case and the field above. The
    span is in brackets. All tracks: the quotes do not use the track filter.
    """
    )
    return quote_head, quote_model


@app.cell
def _(D, case_pick, field_pick, quote_head, quote_model, spans):
    D.quotes(spans, case_pick.value, field_pick.value, quote_model.value, quote_head.value, n=12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Findings in both cases, and findings in 1 case only

    This section compares the `restaurant` case with the `street` case, for
    each field. It does not read `caption`. The 2 cases use different
    prompts, schemas, and images. Thus the section compares only the
    contrast of each model against the other models of the same case.

    **The selection.** Findings 1 and 2 read the default selection of
    section 3 in each case: track C, all aspects, target `place` on street,
    and all targets on restaurant. The cue filter comes from the checkbox of
    section 2. The other controls do not apply. Finding 3 reads all the
    descriptors, as section 6 does.

    **The 3 findings.**

    1. The signature words of section 3. The unit is the spelling key of
       section 3.1, thus a z can differ a little from the z of section 3.
       For each model and key, the estimate is the log-odds of the model
       against the other models, and `se` is its standard error (z is the
       estimate over `se`). The table holds both signs.
    2. The exclusive heads of section 3.1. For each model, the keys that pass
       the test in both cases, and the keys that pass in 1 case only. A key
       of 1 case gets 1 of 3 classes: another model uses it in the other
       case, only this model uses it there, or no model uses it there. This
       finding has no more test.
    3. The vibe rate (vibe descriptors for each 100 texts) and the aspect
       shares of section 6. The estimate is the rate of the model minus the
       rate of the other models of the same case, with its standard error.
       The rates are not weighted, as in section 6: each model reads the
       same images.

    **The tests.** In each case, a finding is significant when its 2-sided
    p, times the number of tests of that case in the table (Bonferroni), is
    less than 0.05. The difference test of Altman and Bland
    (`_f4.difference_test`) compares the 2 estimates of 1 finding, with
    Bonferroni over all the difference tests of the table.

    **The classes of findings 1 and 3.**

    | Class | Rule |
    |-------|------|
    | `both, same sign` | significant in each case, with the same sign |
    | `both, opposite sign` | significant in each case, with opposite signs |
    | `restaurant only`, `street only` | significant in 1 case, and the difference test is significant |
    | `1 case only, not different` | significant in 1 case, but the difference test is not significant |
    | `neither` | significant in no case |
    | `vocabulary of 1 case` | finding 1 only: a signature word (significant, positive z) of 1 case whose key is not a descriptor of the other case |
    | `too rare in the other case` | finding 1 only: a signature word of 1 case whose key has fewer than 5 texts in the selection of the other case |

    A finding that is significant in 1 case and not significant in the
    other case does not show a difference between the cases; only a
    significant difference test shows it.

    The summary tables count the classes for each field and model. The full
    tables and the figure follow the field of section 2. The figure gives
    the 2 log-odds of each key with an estimate in both cases; the dashed
    line is y = x, and the labels name the 5 keys with the largest absolute
    `z_diff`.
    """)
    return


@app.cell
def _(D, cue_pick, raw, spans, totals):
    cases = {_f: D.case_tables(raw, spans, totals, _f, cue_only=cue_pick.value) for _f in D.FIELDS}
    return (cases,)


@app.cell(hide_code=True)
def _(D, cases, mo):
    _w = "weighted (each restaurant text gets the weight of its image; the se is the weighted form)" \
        if cases[D.FIELDS[0]]["rates"].attrs.get("weighted") else "not weighted"
    mo.vstack([
        mo.md("**Summary, finding 1: the signature words.**"),
        D.case_summaries(cases, "signatures"),
        mo.md("**Summary, finding 2: the exclusive heads.**"),
        D.case_summaries(cases, "exclusive"),
        mo.md("**Summary, finding 3: the vibe rate and the aspect shares.** "
              f"The rates of this table are {_w}."),
        D.case_summaries(cases, "rates"),
    ])
    return


@app.cell
def _(cases, field_pick):
    cases[field_pick.value]["signatures"]
    return


@app.cell
def _(D, cases, field_pick):
    D.plot_case_signatures(cases[field_pick.value]["signatures"],
                           title=f"restaurant against street / {field_pick.value}") \
        if not cases[field_pick.value]["signatures"].empty else None
    return


@app.cell
def _(cases, field_pick):
    cases[field_pick.value]["exclusive"]
    return


@app.cell
def _(cases, field_pick):
    cases[field_pick.value]["rates"]
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG and CSV files")
    mo.md(
        f"""
    ## 12. Export

    {save}

    The export writes every case, field, track, and target, not only the
    current selection. The Track C scope comes from the checkbox above. The
    exclusive heads of the export (`exclusive_heads_*.csv`) read all the
    descriptors of each (case, field): no cue filter, all targets. Thus they
    differ from the default selection of section 3.1. The export also
    writes the tables and the figures of section 11 (`cases_*`).
    """
    )
    return (save,)


@app.cell
def _(D, NOTEBOOK_VERSION, Path, cue_pick, mo, raw, save, spans):
    if save.value:
        _name = "f4_descriptors" if D.ROUND == "v1" else f"f4_descriptors_{D.ROUND}"
        _name += f"_{D.DATA}" if D.DATA else ""
        _dir = Path(__file__).resolve().parent / "figures" / _name
        _w = D.export(_dir, raw, spans, cue_only=cue_pick.value)
        _msg = mo.md(
            f"Wrote {len(_w)} files to `notebooks/facct/figures/{_name}/`.\n\n"
            f"Notebook {NOTEBOOK_VERSION}, `_f4desc` {D.__version__}."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
