import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Aesthetic hierarchies and contradictions in the span attributes

    The version 3 extraction gives each span of a reasoning trace a class and
    a set of attributes. The IC notebooks read the class, the free `name`,
    and the vote. This notebook reads the other attributes: `valence` and
    `image` of a dimension, `kind` of an inference, `source` and `favors` of
    a perspective, `direction` and `hedged` of a comparison.

    It asks 2 questions:

    1. **What does the model prize and penalise?** Which dimension names take
       good valence and which take bad, and does the sign hold over the 7
       cases? Which inference kinds go with the winner? How do the
       world-knowledge and the task-convention perspectives win? Is there a
       position effect (Image A against Image B)?
    2. **Where does the model contradict itself?** Across the cases, inside
       1 case, for THE SAME IMAGE in 2 pairs, and inside 1 trace.

    ## The image join

    Image A of a trace is `presented_left_path`. The pair table stores the
    canonical pair, and a swapped pair shows `image_path_b` on the left.
    `_ic_metadata.pair_meta` maps each presented side to a `sample_id` (and
    to a `unit_uid` for a facility case) through the paths.

    **Warning: the sample is 1,000 traces of each case out of 11,000 pairs.**
    Few images appear in 2 traces of 1 case. The unit tables (1 facility, many
    images) give more repeats. Road quality and street photography read the
    SAME 1,000 pairs under 2 prompts, which is a natural test.

    **Warning: a perspective and an inference count are facts about the
    extractor.** Read each finding on both extractors of the judge (section
    4).

    **Warning: this is an association, not a cause.**
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `notebooks/cvpr/`.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import pandas as pd

    import _canonical as C
    import _extractions as X
    import _ic_metadata as M
    import _ic_v3 as I
    import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.1.0"
    OUT_DIR = _here / "outputs" / "explore"
    return C, I, M, NOTEBOOK_VERSION, OUT_DIR, X, mo, pd


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
    ## 0. The gate and the corpus

    **Canonical registry:** {C.summary()}

    The default is the primary arm of the gemma judge: the gemma-4-12b
    traces read by Qwen3.8-27B.

    {arm_pick}

    {allow_stale}
    """
    )
    return allow_stale, arm_pick


@app.cell
def _(I, M, allow_stale, arm_pick, mo):
    arm = arm_pick.value
    try:
        _df = M.load_arm(arm, require_canonical=not allow_stale.value)
        meta = M.pair_meta(M.judge_of(arm))
        corpus = M.attach_images(_df, meta)
        del _df
        gate_ok, _err = not allow_stale.value, ""
    except (RuntimeError, FileNotFoundError) as _exc:
        corpus, meta, gate_ok, _err = None, None, False, str(_exc)
    if corpus is None:
        _msg = f"**No usable corpus.**\n\n```\n{_err}\n```"
    else:
        _miss = float(corpus["sample_A"].isna().mean())
        _msg = (f"**{I.arm_label(arm)}.** {len(corpus):,} spans, "
                f"{corpus['tid'].nunique():,} traces. Spans with no pair row "
                f"in the results parquet: {_miss:.1%}."
                + ("" if gate_ok else " **The gate is OFF.**"))
    mo.md(_msg)
    return arm, corpus, gate_ok, meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. What the model prizes and penalises

    ### 1a. Valence is the language of the vote

    Of the dimension spans on A or B in a decided trace, the share on the
    winner, by valence. 1 vote for each (trace, name, side, valence).
    """
    )
    return


@app.cell
def _(M, corpus, mo):
    vote = M.valence_vote(corpus) if corpus is not None else None
    (mo.vstack([vote.pivot(index="case", columns="valence", values="rate").round(3),
                mo.accordion({"Counts and intervals": vote.round(3)})])
     if vote is not None else mo.md("_No corpus._"))
    return (vote,)


@app.cell(hide_code=True)
def _(mo):
    min_count = mo.ui.slider(10, 200, value=25, step=5, show_value=True,
                             label="Fewest good or bad rows for a name in a case")
    mo.md(
        f"""
    ### 1b. The valence of each dimension name, over the cases

    `net` = (good - bad) / (good + bad). A name is settled good (bad) in a
    case when the Wilson interval of its good share clears 0.5 above
    (below). A name **flips** when it is settled good in 1 case and settled
    bad in another. `axis` marks a name that names an axis ("facade
    condition"), which takes good or bad by the state it sees.

    {min_count}
    """
    )
    return (min_count,)


@app.cell
def _(M, arm, corpus, min_count, mo):
    if corpus is None:
        val_table, val_summary = None, None
        _view = mo.md("_No corpus._")
    else:
        val_table = M.valence_table(corpus, int(min_count.value))
        val_summary = M.valence_summary(val_table, 3)
        _many = val_summary[val_summary["n_cases"] >= 3]
        _prop = _many[~_many["axis"]]
        _flip = val_summary[(val_summary["n_cases"] >= 2) & ~val_summary["axis"]]
        _view = mo.vstack([
            mo.md(
                f"Of the {len(_prop)} property names (not axes) in 3 cases or "
                f"more, **{int(_prop['holds'].sum())} hold 1 sign** in every "
                f"case. Of the {len(_flip)} property names in 2 cases or more, "
                f"**{int(_flip['flips'].sum())} flip**. Inside 1 case, "
                f"{val_table['settled'].mean():.0%} of the (case, name) cells "
                f"are settled, and {val_table.loc[~val_table['axis'], 'mixed'].mean():.0%} "
                f"of the property cells are mixed (good share 0.3 to 0.7)."),
            M.captioned(M.plot_valence_grid(val_table, arm=arm)),
            mo.accordion({"Summary over the cases": val_summary,
                          "Each case": val_table.round(3)}),
        ])
    _view
    return val_summary, val_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1c. Which inference kinds count for the winner

    The directional lift of `inference.kind` (`_cues_across.keyed_lift`: 1
    vote for each trace, kind, and side; a kind on both sides is dropped;
    the rate is shrunk toward the base by 25 pseudo votes). An inference has
    no valence, thus a positive lift says the model names the kind for the
    image it then picks.
    """
    )
    return


@app.cell
def _(M, corpus, mo):
    kinds = M.keyed_win(corpus, "inference", "kind", 25) if corpus is not None else None
    (mo.vstack([kinds.pivot(index="key", columns="case", values="lift").round(3),
                mo.accordion({"Counts and intervals": kinds.round(3)})])
     if kinds is not None and not kinds.empty else mo.md("_No kind over the floor._"))
    return (kinds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1d. World knowledge against task convention

    `source` says where the standard of a perspective comes from. `rate` is
    the share of the rows that favor A or B whose favored image wins.
    `neither_share` is the share of the rows that favor no image. The
    conflict table counts the traces where the 2 sources favor opposite
    images.
    """
    )
    return


@app.cell
def _(M, corpus, mo):
    sources = M.source_table(corpus) if corpus is not None else None
    conflict = M.source_conflict(corpus) if corpus is not None else None
    (mo.vstack([sources.round(3), mo.md("**Conflict traces**"), conflict])
     if sources is not None else mo.md("_No corpus._"))
    return conflict, sources


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1e. Hedged comparisons and the vote

    `direction` names the image with MORE of the quality. `rate` is the
    share of the comparisons with direction A or B that point at the winner.
    The second table is the share of hedged comparisons in a trace, by the
    strength of the label.
    """
    )
    return


@app.cell
def _(M, corpus, mo):
    hedges = M.hedge_table(corpus) if corpus is not None else None
    hedge_strength = M.hedge_by_strength(corpus) if corpus is not None else None
    (mo.vstack([hedges.round(3),
                hedge_strength.pivot(index="case", columns="strength",
                                     values="hedged_share").round(3)])
     if hedges is not None else mo.md("_No corpus._"))
    return hedge_strength, hedges


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1f. The position effect

    The share that falls on Image A. The order of the pair is random, thus
    0.5 is no effect. `decision_A` is a fact about the judge; the span
    shares are facts about the judge and the extractor.
    """
    )
    return


@app.cell
def _(M, corpus, mo):
    position = M.position_table(corpus) if corpus is not None else None
    (mo.vstack([position.pivot(index="case", columns="measure", values="rate").round(3),
                mo.accordion({"Counts and intervals": position.round(3)})])
     if position is not None else mo.md("_No corpus._"))
    return (position,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Where the model contradicts itself

    ### 2a. Across the cases: perspectives that pay in 1 case and cost in another

    The lift is `_ic_v3.lift_tables`. A perspective **flips** when its
    interval clears the base rate upward in 1 case and downward in another.
    """
    )
    return


@app.cell
def _(M, corpus, mo):
    persp_flips = M.perspective_flips(corpus) if corpus is not None else None
    (mo.vstack([mo.md(f"{int(persp_flips['flips'].sum())} of {len(persp_flips)} "
                      "perspectives in 2 cases or more flip."),
                persp_flips[persp_flips["flips"]],
                mo.accordion({"Every perspective": persp_flips.round(3)})])
     if persp_flips is not None and not persp_flips.empty else mo.md("_No corpus._"))
    return (persp_flips,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2b. The same image in 2 pairs

    A group is 1 image (or 1 facility), 1 name, and 2 or more traces that
    give it a valence. `pair_agree` is the share of the pairs of traces in a
    group that give the same valence. `chance_agree` is the agreement of 2
    traces on DIFFERENT images, from the share of the name in the case.
    For a perspective, the valence is `favors` (the perspective favors this
    image) or `against`.

    The partner test then takes the pairs of traces where the image wins in
    1 and loses in the other. If the image did not change but its valence
    did, where does the good valence sit?
    """
    )
    return


@app.cell
def _(M, arm, corpus, mo, pd):
    if corpus is None:
        same_image, partner, examples = None, None, None
        _view = mo.md("_No corpus._")
    else:
        _rows, _prows = [], []
        for _ent, _lab in (("entity_sample", "image"), ("entity_unit", "facility")):
            _v = M.trace_valence(corpus, _ent)
            _f = M.trace_favor(corpus, _ent)
            for _cls, _votes in (("dimension", _v), ("perspective", _f)):
                _c = M.consistency(_votes)
                if not _c.empty:
                    _rows.append(_c.assign(entity=_lab, extraction_class=_cls))
                _p = M.partner_effect(_votes, corpus)
                if not _p.empty:
                    _prows.append(_p.assign(entity=_lab, extraction_class=_cls,
                                            label=f"{_cls}, same {_lab}"))
        same_image = pd.concat(_rows, ignore_index=True) if _rows else pd.DataFrame()
        partner = pd.concat(_prows, ignore_index=True) if _prows else pd.DataFrame()
        examples = M.contradictions(M.trace_valence(corpus, "entity_sample"), corpus, 15)
        _view = mo.vstack([
            mo.md("**Consistency by case.**"), same_image.round(3),
            mo.md("**The partner test.** `rate` is the share of the discordant "
                  "win-loss pairs where the good valence sits in the win."),
            partner.round(3),
            (M.captioned(M.plot_partner_effect(partner, arm=arm))
             if not partner.empty else mo.md("")),
            mo.accordion({"The clearest contradictions (1 image, 1 name, 2 valences)": examples}),
        ])
    _view
    return examples, partner, same_image


@app.cell
def _(M, corpus, mo):
    if corpus is None:
        flips_img, flips_unit = None, None
        _view = mo.md("_No corpus._")
    else:
        flips_img = M.flip_summary(M.outcome_flips(corpus, "entity_sample"))
        flips_unit = M.flip_summary(M.outcome_flips(corpus, "entity_unit"))
        _view = mo.vstack([
            mo.md("### 2c. Win 1 pair, lose another, with the same reason\n\n"
                  "Of the images (top) and the facilities (bottom) in 2 or more "
                  "decided traces: the share that win 1 and lose 1, and the "
                  "count where the same good dimension or the same favoring "
                  "perspective appears in a win AND a loss."),
            flips_img.round(3), flips_unit.round(3)])
    _view
    return flips_img, flips_unit


@app.cell
def _(M, corpus, meta, mo):
    if corpus is None:
        cross_val, cross_win, repeats = None, None, None
        _view = mo.md("_No corpus._")
    else:
        cross_val = M.cross_prompt_valence(corpus)
        cross_win = M.cross_prompt_winner(corpus)
        repeats = M.repeat_pairs(corpus, meta)
        _view = mo.vstack([
            mo.md(
                "### 2d. The same pair under 2 prompts\n\n"
                "Road quality and street photography read the same pairs in "
                "the same order. **The same image wins under both prompts in "
                f"{cross_win['same_rate']:.3f} of {cross_win['same_n']} pairs** "
                f"[{cross_win['same_lo']:.3f}, {cross_win['same_hi']:.3f}]; "
                f"2 independent votes give {cross_win['expected_same']:.3f} "
                f"(phi {cross_win['phi']:.3f}). The table gives the same name on "
                "the same image under the 2 prompts."),
            cross_val.round(3),
            mo.md(f"**Repeated pairs** (the same canonical pair 2 times in the "
                  f"sample): {len(repeats)} pairs, same winner in "
                  f"{repeats['same_winner'].mean():.2f}."
                  if len(repeats) else "No repeated pair."),
        ])
    _view
    return cross_val, cross_win, repeats


@app.cell
def _(M, corpus, mo):
    within = M.within_trace(corpus) if corpus is not None else None
    (mo.vstack([
        mo.md("### 2e. Inside 1 trace\n\n`both_valence`: a name with good AND "
              "bad on the same image. `favors_loser`: at least 1 perspective "
              "favors the image that loses. `majority_loser`: most of the "
              "perspectives favor the loser."),
        within.round(3)]) if within is not None else mo.md("_No corpus._"))
    return (within,)


@app.cell(hide_code=True)
def _(I, arm, mo):
    _sib = I.sibling_arm(arm)
    sibling_load = mo.ui.checkbox(
        value=False,
        label=(f"Read the other extractor of this judge: {I.arm_label(_sib)} "
               "(about 30 seconds)") if _sib else "No other extractor",
    )
    mo.md(
        f"""
    ## 3. The key numbers beside the other extractor

    The 2 extractors of 1 judge read the same traces. A number that holds
    under both comes from the traces.

    {sibling_load}
    """
    )
    return (sibling_load,)


@app.cell
def _(I, M, allow_stale, arm, corpus, meta, mo, sibling_load):
    if corpus is None:
        headline = None
        _view = mo.md("_No corpus._")
    else:
        headline = M.headline_q12(corpus).rename(
            columns={"value": arm, "lo": f"{arm}_lo", "hi": f"{arm}_hi", "n": f"{arm}_n"})
        _sib = I.sibling_arm(arm)
        if sibling_load.value and _sib:
            _d = M.attach_images(M.load_arm(_sib, require_canonical=not allow_stale.value), meta)
            _h = M.headline_q12(_d).rename(
                columns={"value": _sib, "lo": f"{_sib}_lo", "hi": f"{_sib}_hi", "n": f"{_sib}_n"})
            del _d
            headline = headline.merge(_h, on="measure", how="outer")
        _view = headline.round(3)
    _view
    return (headline,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the CSV, PNG, and LaTeX files")
    mo.md(
        f"""
    ## 4. Export

    {save}

    The files land in `master/outputs/explore/` under the tag
    `aesthetic_consistency_<arm>_`. The 3 LaTeX tables are
    `<tag>_valence_flip_v<version>.tex`, `<tag>_same_image_v<version>.tex`,
    and `<tag>_content_hierarchy_v<version>.tex`.
    """
    )
    return (save,)


@app.cell
def _(M, NOTEBOOK_VERSION, OUT_DIR, arm, conflict, cross_val, examples,
      flips_img, flips_unit, gate_ok, headline, hedge_strength, hedges, kinds,
      mo, partner, persp_flips, position, repeats, same_image, save, sources,
      val_summary, val_table, vote, within):
    if save.value and val_table is not None:
        _prefix = f"aesthetic_consistency_{arm}" + ("" if gate_ok else "_stale")
        _written = M.export({
            "valence_vote": vote, "valence_table": val_table,
            "valence_summary": val_summary, "inference_kind_lift": kinds,
            "source": sources, "source_conflict": conflict, "hedge": hedges,
            "hedge_strength": hedge_strength, "position": position,
            "perspective_flips": persp_flips, "same_image": same_image,
            "partner_effect": partner, "contradictions": examples,
            "flips_image": flips_img, "flips_unit": flips_unit,
            "cross_prompt_valence": cross_val, "repeat_pairs": repeats,
            "within_trace": within, "headline": headline,
        }, OUT_DIR, _prefix)
        # The paper tables of this arm. A driver passes the 2 primary arms
        # to the same functions for the 2-arm tables.
        _written += M.write_latex({
            "valence_flip": M.latex_valence_flip({arm: partner}),
            "same_image": M.latex_same_image({arm: same_image}),
            "content_hierarchy": M.latex_content_hierarchy(
                {arm: val_summary}, {arm: persp_flips}),
        }, OUT_DIR, _prefix, NOTEBOOK_VERSION)
        # Each PNG gets a `.json` sidecar with its caption.
        _n_figs = 1
        M.save_figure(M.plot_valence_grid(val_table, arm=arm),
                      OUT_DIR / f"{_prefix}_valence_grid.png", dpi=200,
                      bbox_inches="tight",
                      provenance={"notebook_version": NOTEBOOK_VERSION})
        if not partner.empty:
            _n_figs += 1
            M.save_figure(M.plot_partner_effect(partner, arm=arm),
                          OUT_DIR / f"{_prefix}_partner_effect.png", dpi=200,
                          bbox_inches="tight",
                          provenance={"notebook_version": NOTEBOOK_VERSION})
        _msg = mo.md(f"Wrote {len(_written) + 2 * _n_figs} files to `{OUT_DIR}`. "
                     f"Notebook {NOTEBOOK_VERSION}, `_ic_metadata` {M.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
