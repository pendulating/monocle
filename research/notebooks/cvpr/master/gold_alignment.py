import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # How well do the extractors line up with a human?

    The gold set is a Label Studio project of 30 reasoning traces
    (`research/scripts/build_gold_set.py`). A human marks the spans of each trace
    blind. Each extractor arm of the 2 x 2 (and the small 9B and 12B arms)
    holds spans for the same trace. This notebook puts them side by side.

    ## 2 facts about the human spans

    **The human marks nested spans.** An `inference` often USES a piece of
    `visual_evidence` to make its claim, and the annotator marks both: the
    evidence inside the inference. A region can also carry 2 labels. Thus a
    character can belong to 2 classes at one time, and a span-to-span match
    cannot read that.

    **The human marks a clause; the extractor quotes a phrase.** On the first
    trace the human wrote 124 spans, the primary extractor 35, and the
    control 18. The extractor span most often sits INSIDE the human span.

    Thus the main measure here reads the characters, not the spans. A
    character counts in each class that covers it, on both sides.

    | Measure | Question |
    |---------|----------|
    | Character P/R/F1 | Of the characters the human put in a class, how many did the extractor put there? And the reverse. |
    | Ledger | For each extractor span: exact, inside, contains, partial, other class, or none? |
    | Coverage | For each human span: did a same-class extractor span cover half of it, touch it, or miss it? |
    | Confusion | Under an extractor span, what class did the human give the innermost region? |
    | Nesting | For each nested human pair, did the extractor take the outer, the inner, both, or neither? |
    | Decisions | The human marks every candidate decision. The paper keeps the last one. Does the extractor? |

    **Warning: an IC `perspective` is an `inference` here.** The Integrative
    Complexity codebook names a perspective, the standard by which a
    dimension counts. The schema has no such class, and the annotation puts
    it in `inference`. See `ic-ingredient-extraction.md`.

    **Warning: this reads 1 annotator on the traces done so far.** The
    numbers move with each new task. The 2-annotator agreement, the ceiling,
    appears when a task holds 2 annotations.
    """
    )
    return


@app.cell
def _():
    from pathlib import Path

    _here = Path(__file__).resolve().parent

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _extractions as X
    from research.notebooks.cvpr import _gold_alignment as G
    from research.notebooks.cvpr import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.4.0"
    OUT_DIR = _here / "outputs"
    return C, G, NOTEBOOK_VERSION, OUT_DIR, X, mo, np, pd, plt


@app.cell(hide_code=True)
def _(G, mo):
    _exports = G.find_exports()
    export_pick = mo.ui.dropdown(
        options={p.name: str(p) for p in _exports},
        value=_exports[-1].name if _exports else None,
        label="Label Studio export",
    )
    last_decision = mo.ui.checkbox(
        value=True,
        label="Keep 1 extractor decision for each trace, the last one (the paper's reading)",
    )
    mo.md(
        f"""
    ## 1. The export and the arms

    Exports live in `research/notebooks/cvpr/cues_annotations/`. Export the project as
    JSON from Label Studio and drop the file there; the newest is the default.

    {export_pick}

    {last_decision}
    """
    )
    return export_pick, last_decision


@app.cell
def _(G, export_pick, mo):
    mo.stop(export_pick.value is None, mo.md("**No export found.** Export the project as JSON first."))
    tasks, human = G.load_export(export_pick.value)
    _annotated = tasks[tasks.n_annotations > 0]
    _judges = sorted(_annotated.judge.dropna().unique())
    # The export's class names say which schema the annotator used: v3 names
    # `dimension`, v2 names `visual_evidence`. The arms of that schema are the
    # default; the others stay on the list for a look.
    _export_schema = "v3" if (human.cls == "dimension").any() else "v2"
    arms_all = G.arms_for(_judges)
    _default = [a for a in arms_all if arms_all[a]["schema_version"] == _export_schema] or list(arms_all)
    arm_pick = mo.ui.multiselect(
        options=list(arms_all), value=_default, label=f"Extractor arms (default: schema {_export_schema})",
    )
    mo.md(
        f"""
    **{len(_annotated)} of {len(tasks)} tasks hold an annotation**, over
    {human.annotation_id.nunique()} annotations and {len(human):,} human spans.
    The annotated judges: {", ".join(_judges) or "none"}.

    {arm_pick}
    """
    )
    return arm_pick, arms_all, human, tasks


@app.cell
def _(G, arm_pick, arms_all, human, last_decision, mo, tasks):
    _annotated = tasks[tasks.n_annotations > 0]
    _arms = {k: arms_all[k] for k in arm_pick.value}
    pred = G.load_arm_spans(_annotated, arms=_arms, last_decision_only=last_decision.value)
    R = G.compare(_annotated, human, pred)
    _t = (
        human.groupby("task_id").agg(human_spans=("cls", "size"), annotations=("annotation_id", "nunique"),
                                     minutes=("lead_time_s", lambda s: round(s.iloc[0] / 60, 1) if len(s) else None))
        .join(_annotated.set_index("task_id")[["case", "judge", "chars", "presented_label", "shows_a", "shows_b",
                                                "conf_min", "conf_quantile"]])
        .reset_index()
    )
    _checked = _t[_t.shows_a.notna() | _t.shows_b.notna()]
    _bad = _checked[(_checked.shows_a != "yes") | (_checked.shows_b != "yes")]
    _check_msg = (
        "No image check yet: the set of 2026-09-23 on asks, for each image, whether it shows the item."
        if _checked.empty else
        f"**Image check:** {len(_checked)} traces answered; {len(_bad)} with an image that does NOT show "
        f"the item, or is unclear. A trace whose images fail the check tests the extractor on a judge "
        f"that argued with its own prompt."
    )
    _low = _t[_t.conf_min.notna() & (_t.conf_quantile < 0.75)]
    _conf_msg = (
        "`conf_min` is the facing-filter confidence of the weaker image, `conf_quantile` its rank in the "
        f"case. The set of 2026-09-23 on draws from the top quarter. **{len(_low)} annotated traces sit "
        "below that floor** (pinned from an earlier set); drop them to read the confident pairs alone."
        if _t.conf_min.notna().any() else
        "This export carries no image confidence (a set from before 2026-09-23)."
    )
    mo.vstack([
        mo.md("## 2. The annotated traces"),
        mo.ui.table(_t, selection=None),
        mo.md(_conf_msg),
        mo.md(_check_msg),
        mo.md(f"The arms hold **{len(pred):,} extractor spans** over these traces."),
    ])
    return R, pred


@app.cell(hide_code=True)
def _(R, mo):
    mo.vstack([
        mo.md(
            r"""
    ## 3. The summary, one row for each arm

    | Column | Reads |
    |--------|-------|
    | `granularity` | extractor spans for each human span |
    | `pred_same_class` | share of extractor spans that overlap a human span of the same class (exact, inside, contains, or partial) |
    | `pred_inside` | share that sit inside a human span of the same class |
    | `pred_other_class` | share whose innermost human class is another class |
    | `pred_none` | share that no human span touches |
    | `human_hit` | share of human spans that a same-class extractor span covers by half or more |
    | `human_touched` | share of human spans that a same-class extractor span touches at all |
    | `char_*` | character precision, recall, and F1, micro over the classes, mean over the tasks |
    | `decision_last_matches` | share of tasks where the extractor's decision is the human's LAST decision |
    """
        ),
        mo.ui.table(R["summary"], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(G, R, mo):
    _f1 = G.class_table(R["char_prf"], "f1")
    _p = G.class_table(R["char_prf"], "precision")
    _r = G.class_table(R["char_prf"], "recall")
    mo.vstack([
        mo.md(
            r"""
    ## 4. Character agreement, by class

    Precision reads from the extractor: of the characters it put in a class,
    the share the human put there too. Recall reads from the human. A low
    recall with a high precision says the extractor quotes a piece of what the
    human marked, and marks less of the trace.
    """
        ),
        mo.ui.tabs({
            "F1": mo.ui.table(_f1.reset_index(), selection=None),
            "Precision": mo.ui.table(_p.reset_index(), selection=None),
            "Recall": mo.ui.table(_r.reset_index(), selection=None),
        }),
    ])
    return


@app.cell
def _(G, R, mo):
    _prf = R["char_prf"]
    mo.stop(_prf.empty, mo.md("No overlap to plot."))
    # The caption goes under the figure, never on it.
    fig_f1 = G.plot_char_prf(_prf, list(R["summary"].arm))
    mo.vstack([fig_f1, mo.md(G.caption_md(fig_f1))])
    return


@app.cell(hide_code=True)
def _(mo):
    min_share = mo.ui.slider(start=0.2, stop=0.9, step=0.1, value=0.5,
                             label="Share of the shorter span's content a match must cover")
    mo.vstack([mo.md(
        r"""
    ## 4b. Positional alignment: the relaxed reading

    The exact bounds differ between the human and the extractor, and so does
    the scaffold the judge prints at the head of a row: `Image A`, `Image B`,
    a bold header (`**Vibe:**`), a list marker. This reading masks that
    scaffold on both sides. Then a span matches a span of the same class on
    the other side when their shared content characters cover at least the
    share below of the SHORTER span's content. A phrase inside a clause
    counts, and a clause around a phrase counts. The match is 1 to 1.

    | Row | Reads |
    |-----|-------|
    | a class | the spans of that class alone |
    | `all` | every span, the classes must agree |
    | `any_class` | every span, the class ignored: the boundary alone |

    The gap between `all` and `any_class` is the class disagreement. The
    gap under `any_class` is what the extractor did not mark at all.
    """
    ), min_share])
    return (min_share,)


@app.cell
def _(G, human, min_share, mo, pred, tasks):
    _annotated = tasks[tasks.n_annotations > 0]
    P = G.compare_positional(_annotated, human, pred, min_share=min_share.value)
    mo.stop(P["by_arm"].empty, mo.md("No span to match."))
    _ext = P["by_extractor"][P["by_extractor"].cls.isin(["all", "any_class"])][
        ["extractor", "cls", "n_human", "n_pred", "matched", "tasks", "precision", "recall", "f1"]]
    mo.vstack([
        mo.md("**By extractor, both judges pooled.** Micro over the tasks: the counts sum, then P/R/F1."),
        mo.ui.table(_ext, selection=None),
        mo.md("**By arm and class, F1.**"),
        mo.ui.table(G.positional_table(P["by_arm"], "arm", "f1").reset_index(), selection=None),
        mo.ui.tabs({
            "Precision": mo.ui.table(G.positional_table(P["by_arm"], "arm", "precision").reset_index(), selection=None),
            "Recall": mo.ui.table(G.positional_table(P["by_arm"], "arm", "recall").reset_index(), selection=None),
            "Content characters, F1": mo.ui.table(G.positional_table(P["content_chars"], "arm", "f1").reset_index(),
                                                  selection=None),
        }),
    ])
    return (P,)


@app.cell(hide_code=True)
def _(G, R, mo, pd):
    _led = R["ledger"]
    mo.stop(_led.empty, mo.md("No extractor span to place."))
    _status = pd.crosstab(_led.arm, _led.status, normalize="index").reindex(
        columns=[s for s in G.LEDGER_ORDER if s in set(_led.status)]).round(3)
    confusion_arm = mo.ui.dropdown(options=sorted(_led.arm.unique()),
                                   value=R["summary"].arm.iloc[0], label="Arm for the confusion table")
    mo.vstack([
        mo.md(
            r"""
    ## 5. Where each extractor span lands

    `inside` is the normal case: the extractor quotes a phrase out of the
    human's clause. `other_class` is the disagreement to read: the extractor
    gave a class, and the innermost human region under the span carries a
    different one.
    """
        ),
        mo.ui.table(_status.reset_index(), selection=None),
        confusion_arm,
    ])
    return (confusion_arm,)


@app.cell(hide_code=True)
def _(G, R, confusion_arm, mo):
    _led = R["ledger"]
    _part = _led[_led.arm == confusion_arm.value]
    _conf = G.confusion(_part)
    _other = _part[_part.status.isin(["other_class", "none"])][["cls", "text", "human_cls", "human_text"]]
    mo.vstack([
        mo.md(f"**Confusion for `{confusion_arm.value}`.** Rows: the innermost human class. "
              "Columns: the extractor's class. The diagonal is agreement."),
        mo.ui.table(_conf.reset_index(), selection=None),
        mo.md("The spans in another class, with the human's region:"),
        mo.ui.table(_other, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(R, mo, pd):
    _cov = R["coverage"]
    mo.stop(_cov.empty, mo.md("No human span to cover."))
    _t = pd.crosstab([_cov.arm, _cov.cls], _cov.status, normalize="index").round(3)
    _n = _cov.groupby(["arm", "cls"]).size().rename("human_spans")
    mo.vstack([
        mo.md(
            r"""
    ## 6. Which human spans the extractors reach

    For each human span: `hit` when a same-class extractor span covers half
    of it or more, `touched` when one overlaps it at all, `missed` when none
    does. A class with a high `missed` share is a class the extractor does
    not read the way the human does.
    """
        ),
        mo.ui.table(_t.join(_n).reset_index(), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(R, mo, pd):
    _pairs = R["nested_pairs"]
    _nest = R["nesting"]
    _pair_table = (_pairs.groupby(["outer_cls", "inner_cls", "co_labelled"]).size().rename("pairs").reset_index()
                   if not _pairs.empty else pd.DataFrame())
    _took = (pd.crosstab(_nest.arm, _nest.took) if not _nest.empty else pd.DataFrame())
    mo.vstack([
        mo.md(
            r"""
    ## 7. The nested spans

    A nested pair is a human span inside another human span of the same
    annotation. The usual shape: an `inference` that holds the
    `visual_evidence` it rests on. A co-labelled pair shares the bounds and
    carries 2 labels. For each pair and each arm: which side did the
    extractor take? A side counts when a same-class extractor span covers half
    of it.
    """
        ),
        mo.ui.table(_pair_table, selection=None),
        mo.ui.table(_took.reset_index() if not _took.empty else _took, selection=None),
        mo.ui.table(_pairs[["task_id", "outer_cls", "outer_text", "inner_cls", "inner_text"]]
                    if not _pairs.empty else _pairs, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(R, mo):
    mo.vstack([
        mo.md(
            r"""
    ## 8. The decisions

    The annotator marks every sentence that states an answer. The paper keeps
    the last extractor decision only (`last_decision_only`). `last_matches`
    says the extractor's decision overlaps the human's last one;
    `earlier_matched` counts the human's earlier decisions the extractor
    took instead. An extractor that stops at an earlier candidate reads a
    decision the model then revised.
    """
        ),
        mo.ui.table(R["decisions"], selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(R, mo):
    _attrs = R["attributes"]
    _msg = ("**No attribute to compare.** The annotator set no attribute on the matched "
            "spans so far. The panel below the text in Label Studio holds them."
            if _attrs.empty else
            "On the exact, inside, and contains matches: the share of attribute values the extractor "
            "and the human share.")
    mo.vstack([mo.md(f"## 9. The attributes\n\n{_msg}"),
               mo.ui.table(_attrs, selection=None) if not _attrs.empty else mo.md("")])
    return


@app.cell(hide_code=True)
def _(R, mo, pred, tasks):
    _annotated = tasks[tasks.n_annotations > 0]
    trace_pick = mo.ui.dropdown(options=list(_annotated.task_id), value=_annotated.task_id.iloc[0],
                                label="Trace")
    view_arm = mo.ui.dropdown(options=sorted(pred.arm.unique()) if len(pred) else [],
                              value=R["summary"].arm.iloc[0] if not R["summary"].empty else None,
                              label="Extractor arm to underline")
    mo.vstack([mo.md("## 10. One trace, both readings"), mo.hstack([trace_pick, view_arm])])
    return trace_pick, view_arm


@app.cell
def _(G, human, mo, pred, tasks, trace_pick, view_arm):
    _text = tasks.set_index("task_id").loc[trace_pick.value, "text"]
    _h = human[human.task_id == trace_pick.value]
    _p = pred[(pred.task_id == trace_pick.value) & (pred.arm == view_arm.value)]
    mo.vstack([
        mo.Html(G.render_trace(_text, _h, _p)),
        mo.md(f"**The spans in text order.** {len(_h)} human, {len(_p)} from `{view_arm.value}`."),
        mo.ui.table(G.span_ledger_view(_h, _p), selection=None, page_size=25),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    traces_per_case = mo.ui.slider(start=5, stop=100, step=5, value=20,
                                   label="Shared traces drawn for each case and judge")
    mo.vstack([mo.md(
        r"""
    ## 11. The 2 x 2: the two extractors of each judge against each other

    Every arm of a judge reads the same 1,000 traces of each case, thus the
    two extractors can be read against each other on a seeded draw of those
    traces with the same positional rule as above. The primary arm of the
    judge (the other family's strong model) is the reference; the same-family
    control is the other side. A class where the two agree carries a rate
    that is about the traces; a class where they part carries a rate that is
    about the extractor, and the paper reports it from both or not at all.

    **Warning: this reads 4 corpora.** The first run takes about a minute.
    """
    ), traces_per_case])
    return (traces_per_case,)


@app.cell
def _(G, X, mo, pd, tasks, traces_per_case):
    _pairs = {}
    for _judge, _primary in X.PRIMARY_ARMS.items():
        _others = [a for a, s in X.arm_roots(present_only=True).items()
                   if s["judge"] == _judge and s["schema_version"] == X.ARMS[_primary]["schema_version"] and a != _primary]
        if _others and _primary in X.arm_roots(present_only=True):
            _pairs[_judge] = (_primary, _others[0])
    mo.stop(not _pairs, mo.md("No judge holds 2 arms of the current schema on disk."))
    _texts = {(r.case, r.pair_id): r.text for r in tasks.itertuples()} if len(tasks) else None
    A = G.compare_arm_pairs(_pairs, traces_per_case=traces_per_case.value, texts=_texts)
    profile = G.arm_profile([a for pair in _pairs.values() for a in pair])
    _show = [c for c in ["arm", "judge", "extractor", "traces", "spans_per_trace", "decision_share",
                         "perspectives_per_trace", "verbatim", "share_dimension", "share_inference",
                         "share_perspective", "share_uncertainty"] if c in profile.columns]
    mo.vstack([
        mo.md("**The rungs.** What each arm's corpus holds after the paper filters."),
        mo.ui.table(profile[_show], selection=None),
        mo.md(f"**Agreement, by class.** Reference = primary arm; traces drawn: {A['traces']}. "
              f"Pairs: {', '.join(f'{j}: {p[0]} vs {p[1]}' for j, p in _pairs.items())}."),
        mo.ui.tabs({
            "Span F1": mo.ui.table(G.agreement_table(A["by_judge"], "f1").reset_index(), selection=None),
            "Precision of the control": mo.ui.table(G.agreement_table(A["by_judge"], "precision").reset_index(), selection=None),
            "Recall of the reference": mo.ui.table(G.agreement_table(A["by_judge"], "recall").reset_index(), selection=None),
            "Content characters, F1": mo.ui.table(G.agreement_table(A["chars"], "f1").reset_index(), selection=None),
            "By case": mo.ui.table(A["by_case"], selection=None, page_size=20),
        }),
    ])
    return A, profile


@app.cell(hide_code=True)
def _(mo):
    export_button = mo.ui.run_button(label="Write the tables (CSV and LaTeX) to master/outputs/")
    mo.md(f"## 12. Export\n\n{export_button}")
    return (export_button,)


@app.cell
def _(A, G, NOTEBOOK_VERSION, OUT_DIR, P, R, export_button, mo, profile):
    mo.stop(not export_button.value, mo.md("Press the button to write the CSV and the LaTeX files."))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _written = []
    _tables = {**{k: R[k] for k in ("summary", "char_prf", "ledger", "coverage", "nested_pairs", "nesting",
                                    "decisions", "attributes")},
               **{f"positional_{k}": P[k] for k in ("by_arm", "by_extractor", "by_task", "content_chars")},
               "arm_profile": profile,
               **{f"arm_pairs_{k}": A[k] for k in ("by_case", "by_judge", "chars")}}
    for _name, _df in _tables.items():
        if _df is None or _df.empty:
            continue
        _path = OUT_DIR / f"gold_alignment_{_name}_v{NOTEBOOK_VERSION}.csv"
        _df.to_csv(_path, index=False)
        _written.append(_path.name)
    # The 2 paper tables go next to the CSV files. A table with no data
    # writes a comment line alone.
    _tex = {
        "positional_alignment": G.latex_positional_alignment(P["by_arm"]),
        "extractor_agreement": G.latex_extractor_agreement(A["by_judge"], traces=A.get("traces")),
    }
    for _name, _body in _tex.items():
        _path = OUT_DIR / f"gold_alignment_{_name}_v{NOTEBOOK_VERSION}.tex"
        _path.write_text(_body)
        _written.append(_path.name)
    mo.md("Written:\n\n" + "\n".join(f"- `{w}`" for w in _written))
    return


if __name__ == "__main__":
    app.run()
