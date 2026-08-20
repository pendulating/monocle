import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Subway safety — reasoning traces

    1 notebook for each prompt, like the validation-by-proxy notebooks. This one
    reads the **subway_safety** thinking runs and draws the words the model writes
    while it decides.

    Only runs from **2026-08-11 or later** enter this notebook. That is the
    consolidation date: the battery changed to 7 cases, a minimal prompt with no
    persona, and abstention always on. An earlier run repeats the persona and
    the cue list of the old prompt, thus its cloud describes the prompt, not the
    model.

    ## How to read a cloud

    | Mode | What it shows |
    |------|---------------|
    | `distinctive` | Words this prompt uses more than the OTHER prompts |
    | `frequency` | The most common words |

    Use `distinctive`. Every trace names the 2 images, lists cues, then picks a
    label, so a frequency cloud looks nearly the same for each prompt. The
    `distinctive` score divides out that shared scaffold with the log-odds ratio
    and an informative Dirichlet prior (Monroe, Colaresi, and Quinn, 2008).

    Thus this notebook counts **every** case and draws only **subway_safety**: the
    other cases are the background that the comparison needs.

    ## Words, then claims

    Sections 1 to 3 count WORDS. Section 4 counts CLAIMS: typed spans with
    attributes and character offsets, from the extraction stage. A word count
    cannot say which image holds a cue, whether the cue is good or bad, or
    whether the model went past the pixels. Section 4 can.
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `notebooks/cvpr/`, because every
    # prompt folder uses them. This notebook lives one level down.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    import _provenance as prov
    import _style as style
    import _traces as T
    import _trace_notebook as TN

    # The paper's house palette and camera-ready rcParams. Call it once, before
    # any figure is drawn.
    style.apply_house_style()

    CASE = "subway_safety"
    NOTEBOOK_VERSION = "1.1.0"
    return CASE, NOTEBOOK_VERSION, T, TN, mo, pd, plt, prov, style


@app.cell(hide_code=True)
def _(mo, prov):
    refresh = mo.ui.run_button(label="Refresh from W&B")
    all_dates = mo.ui.checkbox(
        value=False,
        label=f"Include runs from before {prov.CONSOLIDATION_DATE} (not comparable)",
    )
    mo.md(
        f"""
    ## 1. Discovery

    {refresh}
    {all_dates}

    A refresh reads the network and costs about 20 minutes, but the scan is
    shared: every notebook here reads the same cached result.
    """
    )
    return all_dates, refresh


@app.cell
def _(CASE, TN, all_dates, prov, refresh):
    runs = TN.discover(
        min_date=None if all_dates.value else prov.CONSOLIDATION_DATE,
        refresh=bool(refresh.value),
    )
    case_runs = [r for r in runs if r.case == CASE]
    return case_runs, runs


@app.cell(hide_code=True)
def _(CASE, case_runs, mo, runs):
    mo.md(
        f"""
    ### Provenance

    **{len(case_runs)}** {CASE} runs, against **{len(runs) - len(case_runs)}**
    runs of other cases that form the background. The paper cites this table.
    """
        if case_runs
        else f"**No {CASE} run passes the filters.** A thinking run of this "
             "prompt may not exist yet, or it may start before the "
             "consolidation date."
    )
    return


@app.cell
def _(T, case_runs):
    provenance = T.runs_table(case_runs)
    provenance
    return (provenance,)


@app.cell(hide_code=True)
def _(T, mo):
    mode_pick = mo.ui.radio(
        options=["distinctive", "frequency"], value="distinctive", label="Mode"
    )
    ngram_pick = mo.ui.radio(
        options={"single words": 1, "word pairs": 2}, value="single words",
        label="Token"
    )
    label_pick = mo.ui.multiselect(
        options=list(T.LABELS), value=[], label="Keep these labels only (empty = all)"
    )
    max_words = mo.ui.slider(
        30, 250, value=120, step=10, label="Words in a cloud", show_value=True
    )
    extra_stop = mo.ui.text_area(
        value="", label="More stopwords (space or comma separated)"
    )
    mo.hstack(
        [mo.vstack([mode_pick, ngram_pick]),
         mo.vstack([label_pick, max_words, extra_stop])],
        gap=2,
    )
    return extra_stop, label_pick, max_words, mode_pick, ngram_pick


@app.cell
def _(CASE, T, TN, extra_stop, label_pick, ngram_pick, runs):
    import re as _re

    stopwords = T.default_stopwords(_re.split(r"[,\s]+", extra_stop.value or ""))
    counts, pairs, n_runs, mixed = T.counts_by_group(
        runs, stopwords, ngram=int(ngram_pick.value),
        labels=list(label_pick.value) or None,
    )
    groups = TN.groups_for_case(counts, CASE)
    return counts, groups, mixed, n_runs, pairs, stopwords


@app.cell(hide_code=True)
def _(CASE, counts, groups, mixed, mo, n_runs, pairs):
    _warn = ""
    if CASE in mixed:
        _qs = "\n".join(f'- "{q}"' for q in mixed[CASE])
        _warn = (
            f"\n**Warning: this case asked more than 1 question.**\n\n{_qs}\n\n"
            "Each one gets its own cloud. Do not pool them.\n"
        )
    _body = "\n".join(
        f"- `{g}` — {pairs[g]:,} pairs, {n_runs[g]} run(s), "
        f"{sum(counts[g].values()):,} words"
        for g in groups
    ) or "_Nothing to draw._"
    mo.md(f"## 2. The clouds\n\n{_body}\n{_warn}")
    return


@app.cell
def _(T, counts, groups, max_words, mo, mode_pick, plt, style):
    def _draw(group):
        w = T.cloud_weights(counts, group, mode=mode_pick.value,
                            max_words=int(max_words.value))
        if not w:
            return mo.md(f"**{group}** — no word passes the filter.")
        wc = T.make_cloud(w)
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(style.PAPER)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{group}  ({mode_pick.value})",
                     fontsize=9, color=style.EDGE)
        fig.tight_layout()
        plt.close(fig)
        return mo.as_html(fig)

    clouds = mo.vstack([_draw(g) for g in groups])
    clouds
    return (clouds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 3. The words behind the cloud")
    return


@app.cell
def _(TN, counts, groups, max_words, mode_pick):
    words = TN.word_table(counts, groups, mode=mode_pick.value,
                          max_words=int(max_words.value))
    words
    return (words,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Structured extractions

    A word count cannot say which image holds a cue, whether the cue is good or
    bad, or whether the model went past the pixels. The extraction stage turns
    each trace into typed spans with offsets, and this section reads them.

    **Warning: count a quotable span only.** A `match_lesser` span is a sentence
    the model COMPOSED out of the trace plus its own words, and its offsets
    point at a fragment. `_extractions.load` drops those rows.

    **Warning: an attribute value is a free string, not an enum.** The schema
    fixes the class names and the attribute names; the model still writes a
    value the prompt never lists. Read the vocabulary report below before you
    quote a rate.
    """
    )
    return


@app.cell
def _(mo):
    import traceback as _tb

    # Warning: show the traceback, never only the message. A bare message such
    # as "'float' object has no attribute 'lower'" cannot say WHICH line
    # raised, and a reader then blames the data when the fault is in the code.
    try:
        import _extractions as X

        ext = X.load()
        ext_error = ""
    except FileNotFoundError as _exc:
        X, ext, ext_error = None, None, str(_exc)
    except Exception:
        X, ext, ext_error = None, None, _tb.format_exc()

    if not ext_error:
        _msg = (
            f"**{len(ext):,}** quotable extractions over "
            f"**{int(X.trace_totals().sum()):,}** traces."
        )
    elif "no extraction parquet" in ext_error:
        _msg = (
            f"**No extraction data.** {ext_error}\n\nRun the `extract_traces` "
            "pipeline, then `scripts/merge_trace_extractions.py`."
        )
    elif "does not come from the canonical" in ext_error:
        # Not a code fault and not missing data: the corpus belongs to an
        # older battery, thus its panels would answer an older question.
        _msg = (
            "**The extraction corpus is stale.** It comes from trace runs "
            "that the canonical registry does not name, thus the panels "
            "below stay empty and the figures hold the words alone."
            "\n\nRun the `extract_traces` pipeline on the registered "
            "runs, then `scripts/merge_trace_extractions.py`."
            "\n\n```\n" + ext_error + "\n```"
        )
    else:
        # marimo keeps `auto_reload` off, thus a module edited after this
        # session started stays in memory as it was. Restart the kernel first.
        _msg = (
            "**The extraction module raised.** This is a code fault, not "
            "missing data.\n\nIf `_extractions.py` changed after you opened "
            "this notebook, restart the kernel: marimo does not reload a "
            "module on its own.\n\n```\n" + ext_error + "\n```"
        )
    mo.md(_msg)
    return X, ext, ext_error


@app.cell
def _(X, ext, ext_error):
    coverage = X.coverage(ext) if not ext_error else None
    coverage
    return (coverage,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("### 4.1 What the model says, by case")
    return


@app.cell
def _(X, ext, ext_error, mo, plt):
    if ext_error:
        rates_view = mo.md("_No data._")
    else:
        _fig = X.plot_class_rates(ext)
        plt.close(_fig)
        rates_view = mo.vstack([mo.as_html(_fig), X.class_rates(ext)])
    rates_view
    return (rates_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.2 Claims the photograph cannot support

    2 things a word count cannot see, because their words are ordinary: the
    model infers wealth, class, a demographic, or crime from a street, and the
    model reasons about the people in the photograph.
    """
    )
    return


@app.cell
def _(X, ext, ext_error, mo, plt):
    if ext_error:
        risk_view = mo.md("_No data._")
    else:
        _fig = X.plot_risk_panel(ext)
        plt.close(_fig)
        risk_view = mo.vstack([mo.as_html(_fig), X.risk_panel(ext)])
    risk_view
    return (risk_view,)


@app.cell(hide_code=True)
def _(mo):
    unit_pick = mo.ui.radio(
        options={
            "a thing the model named": "class_text",
            "a class and its attributes": "class_attr",
            "a class": "class",
        },
        value="a thing the model named",
        label="Unit of the distinctive score",
    )
    mo.md(
        f"""
    ### 4.3 What subway_safety names that the other cases do not

    {unit_pick}

    The same log-odds ratio with an informative Dirichlet prior that section 2
    applies to words. Only the unit changes: a claim instead of a word. The
    `decision` class is left out — every trace ends with one, so it ranks high
    everywhere and says nothing.
    """
    )
    return (unit_pick,)


@app.cell
def _(CASE, X, ext, ext_error, unit_pick):
    distinctive = (
        None if ext_error
        else X.distinctive(ext, CASE, unit=unit_pick.value, max_rows=60)
    )
    distinctive
    return (distinctive,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.4 The vocabulary report

    What `normalize` moved into `other`. A large share means the vocabulary
    under-covers the case, NOT that the model said nothing. Raise the schema
    version when you change the lists.
    """
    )
    return


@app.cell
def _(X, ext, ext_error):
    vocabulary = X.vocabulary_report(ext) if not ext_error else None
    vocabulary
    return (vocabulary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.5 Quotes

    Every row passed the alignment test, thus the text is the model's own and a
    reader can find it in the trace with the offsets.
    """
    )
    return


@app.cell
def _(CASE, X, ext, ext_error):
    quotes = X.quotes(ext, CASE, "inference", n=15) if not ext_error else None
    quotes
    return (quotes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.6 Which cues go with a win

    The extraction names the image a cue sits on, thus the question can be
    DIRECTIONAL: when the model attaches a cue to image A, does A win? A test
    of presence cannot ask this, because both images live in one trace.

    The block sets the cues in ORDER, the best win rate first, and the colour
    repeats that order. Reading order carries the meaning, so no reader has to
    compare 2 word sizes. The bars below give the same numbers with a Wilson
    interval, which is what the paper should quote.

    The photograph rows show the UNITS the model ranks highest and lowest —
    the places themselves, not the words. That is a second statistic of the
    same run, and its middle differs: a unit wins exactly half of its decided
    comparisons by construction, while a cue sits on the winner about 60% of
    the time. Each frame carries the colour of its own scale.

    **Warning: the colour diverges around the BASE RATE, not around 50%.** A
    cue sits on the winning image about 60% of the time in every case, because
    the model lists more cues for the image it prefers. Centre the scale at
    0.5 and almost every word turns green, which says nothing.

    **Warning: this is an association, not a cause.** The model narrates while
    it decides, so a cue may follow the judgment rather than drive it. Read it
    as "the cues that accompany a win".

    3 rules keep the number honest: 1 vote for each comparison, a cue on BOTH
    images is dropped, and the rate is shrunk toward the base rate so a cue
    seen 4 times cannot take the top of the scale.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    win_class = mo.ui.radio(
        options={"cues the model sees": "visual_evidence",
                 "claims beyond the pixels": "inference"},
        value="cues the model sees", label="Class",
    )
    win_min = mo.ui.slider(
        10, 200, value=25, step=5, label="Fewest comparisons for a cue",
        show_value=True,
    )
    win_photos = mo.ui.slider(
        0, 8, value=6, step=1, show_value=True,
        label="Unit photographs above and below (0 = none)",
    )
    mo.hstack([win_class, win_min, win_photos], gap=2)
    return win_class, win_min, win_photos


@app.cell
def _(CASE, X, ext, ext_error, mo, plt, win_class, win_min, win_photos):
    if ext_error:
        win_view = mo.md("_No data._")
    else:
        try:
            _fig, _table = X.plot_win_block(
                ext, CASE, classes=(win_class.value,), min_count=int(win_min.value),
                unit_photos=int(win_photos.value),
            )
            _bars = X.plot_win_bars(
                ext, CASE, classes=(win_class.value,), min_count=int(win_min.value)
            )
            plt.close(_fig)
            plt.close(_bars)
            win_view = mo.vstack([mo.as_html(_fig), mo.as_html(_bars), _table])
        except ValueError as _exc:
            win_view = mo.md(f"_{_exc}_")
    win_view
    return (win_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.7 The model against itself

    The model labels each cue `good`, `bad`, or `neutral`, and it also picks a
    winner. Those 2 statements should agree. Where they do not, the trace and
    the judgment disagree, and that gap is the interesting part.
    """
    )
    return


@app.cell
def _(CASE, X, ext, ext_error):
    consistency = X.valence_consistency(ext, CASE) if not ext_error else None
    consistency
    return (consistency,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG and CSV files")
    mo.md(f"## 5. Export\n\n{save}")
    return (save,)


@app.cell
def _(CASE, NOTEBOOK_VERSION, TN, X, counts, ext, ext_error, groups, mo,
      mode_pick, runs, save, unit_pick):
    if save.value:
        _dir = TN.figures_dir(__file__)
        _w = TN.export(CASE, counts, groups, runs, _dir, mode=mode_pick.value)
        if not ext_error:
            _w += X.export(CASE, ext, _dir, unit=unit_pick.value)
        _msg = mo.md(
            f"Wrote {len(_w)} files to `{CASE}/figures/`: "
            + ", ".join(f"`{n}`" for n in _w)
            + f"\n\nNotebook {NOTEBOOK_VERSION}, `_trace_notebook` "
            + f"{TN.__version__}, `_extractions` "
            + (X.__version__ if not ext_error else "absent")
            + "."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
