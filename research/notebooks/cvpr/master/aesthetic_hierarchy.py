import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The aesthetic hierarchy across the cases

    A `<case>_traces.py` notebook draws 1 word cloud for each prompt, and its
    `distinctive` score answers "which words does THIS case use and the others
    do not". This notebook asks the opposite question over all 7 cases at once.

    **Does the model carry 1 aesthetic hierarchy from prompt to prompt?**

    If it does, the model writes `clean`, `maintained`, and `modern` at about
    the same rate for a school, a park, and a subway entrance, and it puts them
    in the same order. If each prompt has a hierarchy of its own, the rates and
    the ranks separate.

    ## How to read the scatter

    1 point is 1 token: a word, or a phrase of 2 or 3 words. The token set is
    the union of the top tokens of each case, by rate.

    | Axis | Meaning |
    |------|---------|
    | x | the mean rate of the token, per 10,000 counted tokens |
    | y | the evenness of that rate over the 7 cases, from 0 to 1 |

    | Where a token sits | What it is |
    |--------------------|------------|
    | top right | common and even — a rung that every case keeps |
    | bottom right | common and uneven — a case noun, such as `library` |
    | top left | rare and even — a token of the shared background |
    | bottom left | rare and uneven — a detail of 1 case |

    Evenness is the Shannon entropy of the rate shares, divided by the entropy
    of a flat share. 1.0 says the model writes the word at the same rate in
    every case.

    ## The reasoning scaffold

    The model writes `looks`, `like`, `maybe`, and `prompt` in every case, at
    the same rate. Those words take the whole even band, and the aesthetic
    words fall off the top of the picture. `_words_across.SCAFFOLD` drops them,
    and the switch below puts them back.

    ## Words, or phrases

    The **Token** control reads 1 word, 2 words, or 3. A phrase says what a
    word cannot: `well lit`, `less safe`, `not maintained`, and `well
    maintained` are the judgment, and each one is invisible to a word count.

    **Warning: this is NOT `_traces.count_words(ngram=2)`.** That function
    removes the stopwords BEFORE it joins the tokens, thus it reports a phrase
    that nobody wrote — `the building is clean` gives `building clean` — and it
    can never report `well maintained`, because `well` is a stopword. The
    counter here joins ADJACENT tokens of the raw text, and then it applies 4
    rules:

    | Rule | It drops |
    |------|----------|
    | The phrase stays inside 1 sentence | `entrance the entrance` |
    | No token is barred | `roadway looks better` |
    | The first and the last token carry meaning | `look the`, `the entrance` |
    | 1 token or more is a real content word | `very well`, `much more` |

    A plain stopword glues a phrase, thus it is barred at the edges alone:
    `cars are parked` is a phrase and `cars parked` is not. The prompt echo and
    the reasoning voice are barred everywhere, because `looks` and `image` say
    nothing wherever they sit.

    `_words_across.MODIFIER` holds the degree and the polarity words —
    `very`, `well`, `more`, `less`, `not`. Every stopword list drops them, and
    for 1 word that is right. In a phrase they carry the meaning, thus they are
    exempt from the bar, they pass the third rule, and they fail the fourth.

    **The city name comes back at 2 words.** `new` sits in `CITY_ECHO` because
    1 word cannot separate `New York` from `new building`. A phrase can, thus a
    phrase figure shows the 2 as separate points.

    ## What the figure cannot say

    A rate is a count of words. It cannot say which photograph a word describes,
    and it cannot say that a word moved the judgment. Read section 4 for the
    directional statistic, and the `<case>_traces.py` notebooks for the cues
    that go with a win.
    """
    )
    return


@app.cell
def _():
    from pathlib import Path

    _here = Path(__file__).resolve().parent

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _style as style
    from research.notebooks.cvpr import _trace_notebook as TN
    from research.notebooks.cvpr import _traces as T
    from research.notebooks.cvpr import _words_across as WA

    # The gate. Every figure comes from the registered trace runs, thus the
    # notebook stops here when the registry does not match the disk.
    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    FIG_DIR = _here / "figures"
    OUT_DIR = _here / "outputs"
    return C, FIG_DIR, NOTEBOOK_VERSION, OUT_DIR, T, TN, WA, mo, pd, plt, style


@app.cell(hide_code=True)
def _(C, mo):
    mo.md(f"**Canonical registry:** {C.summary()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. The runs and the corpus

    The registry names 1 thinking run for each case and each rater. This
    notebook reads all of them.
    """
    )
    return


@app.cell
def _(TN):
    runs = TN.discover()
    return (runs,)


@app.cell
def _(T, runs):
    provenance = T.runs_table(runs)
    provenance
    return (provenance,)


@app.cell(hide_code=True)
def _(WA, mo):
    rater_pick = mo.ui.radio(
        options=["both raters", "gemma-4-12b", "qwen3.5-9b"],
        value="both raters",
        label="Rater",
    )
    scaffold_drop = mo.ui.checkbox(
        value=True,
        label=f"Drop the reasoning scaffold ({len(WA.SCAFFOLD)} words)",
    )
    ngram_pick = mo.ui.radio(
        options={"single words": 1, "word pairs": 2, "3-word phrases": 3},
        value="single words", label="Token",
    )
    top_per_case = mo.ui.slider(
        20, 150, value=60, step=10, show_value=True,
        label="Tokens each case brings to the set",
    )
    extra_stop = mo.ui.text_area(
        value="", label="More stopwords (space or comma separated)"
    )
    mo.hstack(
        [mo.vstack([rater_pick, ngram_pick, scaffold_drop]),
         mo.vstack([top_per_case, extra_stop])],
        gap=2,
    )
    return extra_stop, ngram_pick, rater_pick, scaffold_drop, top_per_case


@app.cell(hide_code=True)
def _(WA, mo, ngram_pick):
    # The floor follows the token length, thus it resets when you change the
    # length. A bigram corpus is about 20 times sparser than a word corpus, so
    # 1 fixed floor either buries the phrases or floods the words.
    _n = int(ngram_pick.value)
    _default = WA.min_count_for(_n)
    min_count = mo.ui.slider(
        max(10, _default // 10), _default * 10, value=_default,
        step=max(10, _default // 10), show_value=True,
        label=f"Fewest times the corpus writes a token ({_n}-word)",
    )
    min_count
    return (min_count,)


@app.cell
def _(WA, extra_stop, ngram_pick, runs, scaffold_drop):
    import re as _re

    _extra = [w for w in _re.split(r"[,\s]+", extra_stop.value or "") if w]
    stopwords = WA.stopwords(_extra, drop_scaffold=bool(scaffold_drop.value),
                             ngram=int(ngram_pick.value))
    # The tokens a phrase refuses even in the middle. It does nothing at 1 word.
    barred = WA.barred_words(drop_scaffold=bool(scaffold_drop.value),
                             extra=_extra)
    # The first pass costs some minutes, and a phrase pass costs about 3.
    # Both keep every count on disk, thus a later change to a slider is
    # immediate.
    _counts, _totals = WA.run_counts(runs, stopwords, barred=barred,
                                     ngram=int(ngram_pick.value))
    rates = WA.rates_frame(_counts, _totals)
    return barred, rates, stopwords


@app.cell
def _(WA, rates):
    corpus = WA.corpus_sizes(rates)
    corpus
    return (corpus,)


@app.cell(hide_code=True)
def _(corpus, mo, rater_pick):
    mo.md(
        f"**{corpus.corpus_words.sum():,} trace words** over "
        f"{corpus.case.nunique()} cases and {corpus.model.nunique()} raters. "
        f"The figures below use: **{rater_pick.value}**.\n\n"
        "**Warning: the 2 raters do not write the same amount.** qwen3.5-9b "
        "writes about 3 times the words of gemma-4-12b. A rate divides that "
        "out. A count does not, thus a pooled group weights qwen more."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. The scatter

    The word set is the union of the top words of each case. A point that sits
    high and to the right is a word the model writes often, and about as often
    in every case.
    """
    )
    return


@app.cell
def _(WA, min_count, rater_pick, rates, top_per_case):
    models = None if rater_pick.value == "both raters" else [rater_pick.value]
    matrix, tops = WA.rate_matrix(
        rates, models=models,
        top_per_case=int(top_per_case.value),
        min_count=int(min_count.value),
    )
    summary = WA.summarise(matrix, tops)
    return matrix, models, summary, tops


@app.cell(hide_code=True)
def _(mo, summary):
    colour_pick = mo.ui.radio(
        options={"the case that writes it most": "peak_case",
                 "the commitment lift": "commitment"},
        value="the case that writes it most", label="Colour",
    )
    n_core = mo.ui.slider(
        6, 40, value=22, step=2, show_value=True,
        label="Words to name in the shared band",
    )
    n_contrast = mo.ui.slider(
        0, 20, value=8, step=1, show_value=True,
        label="Case nouns to name, for contrast",
    )
    mo.hstack([colour_pick, n_core, n_contrast], gap=2) if not summary.empty \
        else mo.md("**The word set is empty.** Lower the count filter.")
    return colour_pick, n_contrast, n_core


@app.cell
def _(WA, barred, models, ngram_pick, runs, stopwords, summary):
    # 1 pass over the label-conditioned counts. The colour of section 2, the
    # table of section 4, and the export all read this.
    lift = (
        None if summary.empty
        else WA.commitment_lift(runs, stopwords, summary["word"], models=models,
                                ngram=int(ngram_pick.value), barred=barred)
    )
    return (lift,)


@app.cell
def _(WA, colour_pick, lift, mo, n_contrast, n_core, plt, summary):
    if summary.empty:
        scatter = mo.md("_No words._")
    else:
        _fig = WA.plot_hierarchy(
            summary, colour_by=colour_pick.value, lift=lift,
            n_core=int(n_core.value), n_contrast=int(n_contrast.value),
        )
        plt.close(_fig)
        scatter = mo.as_html(_fig)
    scatter
    return (scatter,)


@app.cell(hide_code=True)
def _(mo, summary):
    _n = 0 if summary.empty else int(summary["n_groups"].iloc[0])
    mo.md(
        f"**{len(summary)} words**, over {_n} cases.\n\n"
        "The table below holds the numbers behind every point, so a word can "
        "be quoted with its rate and its evenness. `floor_rate` is the strict "
        "test: the rate of the case that writes the word LEAST."
    )
    return


@app.cell
def _(summary):
    words = summary
    words
    return (words,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Does the order hold?

    Evenness asks about the rate. A hierarchy is about the ORDER. Section 3
    tests the order directly.

    Each panel plots 1 case against the mean of the other 6. A tight diagonal
    says the case keeps the shared order. A vertical stripe says the case
    writes its own words, which no other case writes.

    The block after it gives the Spearman rho of every case pair. Read the 2
    together: rho over the whole word set answers "do the vocabularies match",
    and rho over the shared words alone answers "where the vocabulary is
    shared, is the order shared".
    """
    )
    return


@app.cell(hide_code=True)
def _(matrix, mo):
    breadth_pick = mo.ui.slider(
        1, max(1, len(matrix.columns)), value=1, step=1, show_value=True,
        label="Cases that must rank the word (1 = every word)",
    )
    breadth_pick
    return (breadth_pick,)


@app.cell
def _(WA, breadth_pick, matrix, mo, plt, summary):
    if summary.empty:
        pairs_view = mo.md("_No words._")
    else:
        _words = WA.shared_words(summary, min_breadth=int(breadth_pick.value))
        try:
            _fig = WA.plot_case_pairs(matrix, words=_words)
            plt.close(_fig)
            pairs_view = mo.vstack([
                mo.md(f"**{len(_words)} words** in these panels."),
                mo.as_html(_fig),
            ])
        except ValueError as _exc:
            pairs_view = mo.md(f"_{_exc}_")
    pairs_view
    return (pairs_view,)


@app.cell
def _(WA, matrix, mo, plt, summary):
    if summary.empty or matrix.shape[1] < 2:
        agree_view = mo.md("_No words._")
    else:
        _fig = WA.plot_agreement(matrix)
        plt.close(_fig)
        _pairs = WA.agreement(matrix)
        agree_view = mo.vstack([
            mo.as_html(_fig),
            mo.md(f"**Median rho {_pairs.rho.median():.2f}**, "
                  f"from {_pairs.rho.min():.2f} to {_pairs.rho.max():.2f}, "
                  f"over {len(_pairs)} case pairs."),
            _pairs,
        ])
    agree_view
    return (agree_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Which words go with a decision

    A shared vocabulary is not yet a hierarchy. A hierarchy also RANKS, so the
    figure needs a second statistic: does the model write the word when it
    picks a winner, or when it cannot separate the 2 photographs?

    The commitment lift is the log-odds ratio with an informative Dirichlet
    prior, the same statistic as the trace clouds. The target is the traces
    that named a winner. The background is the traces that answered `Same` or
    `NotSure`.

    The score is computed INSIDE each case and then averaged. A pooled score
    would follow the abstention rate instead: the street photography runs
    abstain on about 1% of the pairs, and the schools runs on more than half.

    **Warning: this is an association, not a cause.** The model narrates while
    it decides, thus a word can follow the judgment rather than drive it.
    """
    )
    return


@app.cell
def _(lift, mo, summary):
    if summary.empty or lift is None or lift.empty:
        lift_table = mo.md("_No words._")
    else:
        lift_table = (
            summary[["word", "mean_rate", "evenness", "breadth", "peak_case"]]
            .assign(commitment_lift=lambda d: d["word"].map(lift))
            .sort_values("commitment_lift", ascending=False)
            .round(3)
        )
    lift_table
    return (lift_table,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG and CSV files")
    mo.md(
        f"""
    ## 5. Export

    {save}

    The files land in `master/figures/`. A cross-case figure belongs to no 1
    prompt, thus it must not go in a prompt folder.
    """
    )
    return (save,)


@app.cell
def _(FIG_DIR, NOTEBOOK_VERSION, WA, lift, matrix, mo, ngram_pick, save,
      summary):
    if save.value and not summary.empty:
        # The tag carries the token length, so a phrase set never replaces the
        # word set.
        _n = int(ngram_pick.value)
        _tag = "aesthetic_hierarchy" if _n == 1 else f"aesthetic_hierarchy_{_n}gram"
        _written = WA.export(summary, matrix, FIG_DIR, tag=_tag, lift=lift)
        _msg = mo.md(
            f"Wrote {len(_written)} files to `master/figures/`: "
            + ", ".join(f"`{n}`" for n in _written)
            + f"\n\nNotebook {NOTEBOOK_VERSION}, `_words_across` "
            + f"{WA.__version__}."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
