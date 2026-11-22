import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The aesthetic hierarchy over the urban cues

    Does the model carry 1 aesthetic standard from prompt to prompt? Does a
    chain-link fence cost a school what it costs a subway entrance?

    ## Why this notebook, and not the word one

    `master/aesthetic_hierarchy.py` counts words and phrases. It answers a
    NARROWER question — does the model use the same vocabulary — and the answer
    is a weak yes: the top of a word list is mostly what the case is ABOUT. A
    word count cannot separate the subject from the judgment, thus the
    cross-case agreement of the word rates sits near 0.35.

    This notebook counts CUES. The extraction stage turns each trace into typed
    spans, and a `visual_evidence` span carries 3 things a word never can:

    | Field | What it gives |
    |-------|---------------|
    | `image` | WHICH photograph the cue sits on |
    | `valence` | good, bad, or neutral, in the model's own reading |
    | `category` | 1 of 11 urban attribute types |

    ## The price of a cue

    Because a cue names its image, the question turns DIRECTIONAL. A cue on
    image A wins when the label says A. Thus each cue carries a price: how
    often the photograph that holds it wins.

    That price is the hierarchy. The cross-case question is then sharp: does a
    cue keep its price when the question changes?

    ## The lift, never the raw rate

    A raw win rate cannot be compared across cases. The base rate differs — a
    cue sits on the winning image 59% of the time in the schools case and 70%
    in the libraries case, because the model names more cues for the photograph
    it prefers. Every number here is a LIFT over the base rate of its own case.

    ## The unit

    | `unit` | 1 key | Shared by every case? |
    |--------|-------|-----------------------|
    | `category_valence` | `upkeep \| good` | yes, up to 33 keys |
    | `category` | `upkeep` | yes, 11 keys |
    | `cue` | `chain-link fence` | the head only |
    | `cue_valence` | `chain-link fence \| bad` | the head only |

    **`category_valence` is the unit that answers the question.** `category`
    alone pools good greenery with bad greenery, and those 2 carry opposite
    prices, so the pooled key reads near zero and the hierarchy disappears.

    **Warning: this is an association, not a cause.** The model narrates while
    it decides, thus a cue can follow the judgment rather than drive it. Read a
    price as "the cues that go with a win".
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # The shared modules sit in the parent, `research/notebooks/cvpr/`. This notebook
    # lives one level down, in `master/`, because it reads every prompt.
    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    import _canonical as C
    import _cues_across as CA
    import _style as style
    import _words_across as WA

    # The gate on the judgment runs. The gate on the EXTRACTION corpus is a
    # second test, and the next cell holds it.
    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    FIG_DIR = _here / "figures"
    return C, CA, FIG_DIR, NOTEBOOK_VERSION, WA, mo, pd, plt, style


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

    A cue figure is 2 steps downstream of a judgment run: the judge wrote the
    trace, and the extractor read it. Thus it must pass 2 gates. The registry
    above accepts the judgment runs. `_extractions.load` tests whether the
    extraction corpus comes from those same runs.

    {allow_stale}
    """
    )
    return (allow_stale,)


@app.cell
def _(CA, allow_stale, mo):
    try:
        corpus = CA.load(require_canonical=not allow_stale.value)
        gate_ok, gate_error = True, ""
    except RuntimeError as _exc:
        corpus, gate_ok, gate_error = None, False, str(_exc)
    except FileNotFoundError as _exc:
        corpus, gate_ok, gate_error = None, False, str(_exc)

    if corpus is not None and not allow_stale.value:
        _msg = f"**The corpus passes the gate.** {len(corpus):,} extractions."
    elif corpus is not None:
        _msg = (
            "**The gate is OFF.** The figures below are a look at a corpus "
            "that the registry does not name, and every exported file takes a "
            "`_stale` suffix so it can never be read as a paper figure."
            f"\n\n{len(corpus):,} extractions."
        )
    else:
        _msg = (
            "**No usable corpus.** Run the `extract_traces` pipeline on the "
            "registered trace runs, then `research/scripts/merge_trace_extractions.py`."
            f"\n\n```\n{gate_error}\n```"
        )
    mo.md(_msg)
    return corpus, gate_error, gate_ok


@app.cell(hide_code=True)
def _(CA, corpus, mo):
    if corpus is None:
        _view = mo.md("_No corpus._")
    else:
        _missing = CA.missing_cases(corpus)
        _warn = (
            "\n\n**Warning: the corpus does not cover the whole battery.** "
            f"It holds no `{'`, `'.join(_missing)}`. An absent case is not a "
            "null result: the extraction stage has not run on it. Every "
            "number below describes the cases that ARE present."
            if _missing else ""
        )
        _view = mo.vstack([
            mo.md("### What the corpus holds" + _warn),
            CA.coverage(corpus),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(CA, mo):
    unit_pick = mo.ui.radio(
        options={
            "category and valence": "category_valence",
            "category": "category",
            "the cue text": "cue",
            "the cue text and valence": "cue_valence",
        },
        value="category and valence", label="Unit",
    )
    class_pick = mo.ui.radio(
        options={"the cues the model sees": "visual_evidence",
                 "claims beyond the pixels": "inference"},
        value="the cues the model sees", label="Class",
    )
    min_count = mo.ui.slider(
        10, 300, value=25, step=5, show_value=True,
        label="Fewest decided comparisons for a key",
    )
    max_keys = mo.ui.slider(
        10, 60, value=30, step=5, show_value=True,
        label="Rungs to draw in the ladder",
    )
    mo.hstack([mo.vstack([unit_pick, class_pick]),
               mo.vstack([min_count, max_keys])], gap=2)
    return class_pick, max_keys, min_count, unit_pick


@app.cell
def _(CA, class_pick, corpus, min_count, unit_pick):
    if corpus is None:
        tables, matrix, summary = None, None, None
    else:
        tables = CA.lift_tables(
            corpus, unit=unit_pick.value, classes=(class_pick.value,),
            min_count=int(min_count.value),
        )
        matrix = CA.lift_matrix(tables)
        summary = CA.lift_summary(matrix, tables)
    return matrix, summary, tables


@app.cell(hide_code=True)
def _(matrix, mo, summary):
    if summary is None or summary.empty:
        _msg = "**No key passes the filter.** Lower the comparison floor."
    else:
        _n = int(summary["n_groups"].iloc[0])
        _same = int(summary["same_sign"].sum())
        _msg = (
            f"## 2. The hierarchy\n\n**{len(matrix)} keys** that all {_n} "
            f"cases scored. **{_same} of {len(summary)}** sit on the same side "
            "of the base rate in every case.\n\n"
            "A short row says every case gave the key the same price. A long "
            "row says the price moved with the question."
        )
    mo.md(_msg)
    return


@app.cell
def _(CA, matrix, max_keys, mo, plt, summary):
    if summary is None or summary.empty:
        ladder = mo.md("_No keys._")
    else:
        _fig = CA.plot_ladder(matrix, summary, max_keys=int(max_keys.value))
        plt.close(_fig)
        ladder = mo.as_html(_fig)
    ladder
    return (ladder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. The price against how much it moves

    1 point is 1 key. The x axis is the price. The y axis is the spread of that
    price between the cases, thus a point near the BOTTOM keeps its price
    whatever the question asks.

    The colour is the model's own reading of the cue, so the figure needs no
    outside judgment about which cues should be good. A heavy outline marks a
    key that every case put on the same side of its base rate.
    """
    )
    return


@app.cell
def _(CA, mo, plt, summary):
    if summary is None or summary.empty:
        price = mo.md("_No keys._")
    else:
        _fig = CA.plot_price(summary)
        plt.close(_fig)
        price = mo.as_html(_fig)
    price
    return (price,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Does the order hold?

    Each panel plots 1 case against the mean of the others. A tight diagonal
    says the case prices the cues the way the rest of the battery does. The
    block after it gives the Spearman rho of every case pair.
    """
    )
    return


@app.cell
def _(CA, WA, matrix, mo, plt, summary):
    if summary is None or summary.empty or matrix.shape[1] < 2:
        order_view = mo.md("_2 cases or more are needed._")
    else:
        _pairs = CA.plot_price_pairs(matrix)
        _block = WA.plot_agreement(matrix)
        plt.close(_pairs)
        plt.close(_block)
        _rho = CA.agreement(matrix)
        order_view = mo.vstack([
            mo.as_html(_pairs),
            mo.as_html(_block),
            mo.md(f"**Median rho {_rho.rho.median():.2f}**, from "
                  f"{_rho.rho.min():.2f} to {_rho.rho.max():.2f}, over "
                  f"{len(_rho)} case pairs."),
            _rho,
        ])
    order_view
    return (order_view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. How often the model names a cue

    Section 2 asks what a cue is WORTH. This section asks how often the model
    names it, and whether it names it as often in every case. The axes are the
    ones the word notebook uses, so the 2 reports read the same way. Only the
    token changes: a typed cue instead of a word.
    """
    )
    return


@app.cell
def _(CA, WA, corpus, class_pick, mo, plt, unit_pick):
    if corpus is None:
        rate_view, rate_summary, rate_matrix = mo.md("_No corpus._"), None, None
    else:
        _rates = CA.rates_frame(corpus, unit=unit_pick.value,
                                classes=(class_pick.value,))
        rate_matrix, _tops = WA.rate_matrix(_rates, top_per_case=60,
                                            min_count=1)
        rate_summary = WA.summarise(rate_matrix, _tops)
        if rate_summary.empty:
            rate_view = mo.md("_No key._")
        else:
            _fig = WA.plot_hierarchy(rate_summary, n_core=18, n_contrast=6)
            plt.close(_fig)
            rate_view = mo.as_html(_fig)
    rate_view
    return rate_matrix, rate_summary, rate_view


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5b. The same question, in the words the model wrote

    Every key above carries OUR vocabulary: 11 categories and 3 valences, and
    `good upkeep` is a word we chose. This section asks the question again
    with the model's own words as the key.

    ### Why the ladder above cannot use them

    A key must recur in every case to take a row of a cross-case ladder, and a
    raw cue span does not recur. The corpus holds **254,361 distinct spans**,
    and rules that join the spelling variants (`normalise_cue`) leave 225,975.
    Only **17** clear 50 comparisons in all 7 cases, and 13 of the 17 name a
    car. So the cross-case ladder must abstract, and `category_valence` is
    that abstraction.

    ### Where the words DO work

    | Figure | Cases | Keys | The question |
    |--------|-------|------|--------------|
    | per case | 7 | 78 to 315 each | What does each case value? |
    | facility ladder | 5 | about 50 | Do the words price the same? |

    Road quality and street photography pair random street images and name no
    facility, thus they share almost no word with the rest. The shared ladder
    runs over the 5 facility cases, where 50 words recur and the cross-case
    rho is 0.85.

    **`normalise_cue` joins spelling variants only.** `clean`, `well
    maintained` and `tidy` stay 3 keys, because they are 3 words the model
    chose. The rules are deterministic and read no corpus.
    """)
    return


@app.cell
def _(CA, corpus):
    words_per_case = CA.lift_tables(corpus, unit="cue_norm", min_count=50)
    words_per_case
    return (words_per_case,)


@app.cell
def _(CA, plt, words_per_case):
    _f = CA.plot_case_ladder(words_per_case)
    plt.gca()
    _f
    return


@app.cell
def _(CA, corpus):
    _fac = corpus[corpus["case"].isin(CA.FACILITY_CASES)]
    words_shared = CA.lift_tables(_fac, unit="cue_norm", min_count=25)
    words_matrix = CA.lift_matrix(words_shared)
    words_summary = CA.lift_summary(words_matrix, words_shared)
    return words_matrix, words_shared, words_summary


@app.cell
def _(CA, plt, words_matrix, words_summary):
    _f = CA.plot_ladder(words_matrix, words_summary, max_keys=44,
                        figsize=(6.6, 8.6))
    plt.gca()
    _f
    return


@app.cell
def _(CA, mo, words_matrix):
    _a = CA.agreement(words_matrix)
    mo.vstack([
        mo.md(f"**Cross-case Spearman over the facility cases:** median "
              f"`{_a['rho'].median():.2f}`, from `{_a['rho'].min():.2f}` to "
              f"`{_a['rho'].max():.2f}`, over {len(words_matrix)} shared "
              f"words."),
        _a,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why a row of the ladder is wide

    A wide row says the price of that word moved with the question. This
    figure keeps the row and writes the descriptor along it.

    | Mark | What it is |
    |------|------------|
    | grey line | the spread of the bare word over the cases |
    | large dot | the bare word in 1 case |
    | faint dot | 1 phrase that ends in the word, in 1 case |
    | labelled dot | the same, with the word the model ADDED |

    **The modifier explains the spread.** The bare `window` moves over 0.33,
    but `large glass` reaches +0.29 and `smaller` falls to -0.39, and both sit
    outside the grey line. The model does not price a window; it prices a
    LARGE one.

    | Word | Pays | Costs |
    |------|------|-------|
    | window | clear, intact, large glass | small, smaller, standard |
    | building | well maintained, modern looking school | older brick, older |
    | tree | healthy looking, lush green | bare, sparse, dense |
    | sidewalk | clean, clear, wide | narrow, narrower |
    | scaffolding | **no** scaffolding | extensive, heavy |

    **`tree` prices grooming, not quantity:** `dense` AND `sparse` both cost.
    **`car` is the control:** its descriptors are `black`, `grey`, `blue`,
    `silver`, `police` — colours and types, no evaluative word at all.

    **Warning: the head is the last word, which is a rule and not a parser.**
    `spread_heads` therefore takes only a 1-word key, where the head IS the
    key. A modifier that holds a preposition is dropped, because
    `brick building with large` does not describe a window.
    """)
    return


@app.cell
def _(CA, plt, words_shared, words_summary):
    _f = CA.plot_ladder_modifiers(words_shared, words_summary)
    plt.gca()
    _f
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5c. The outliers

    Which cues should a reader look at? This table isolates them, and it
    reads EVERY case: a cue reverses because 1 case disagrees, thus dropping
    a case would hide the row that matters.

    The scale is the log odds ratio of each case against the base rate of its
    OWN case, because a raw win rate cannot be compared when the base rate
    runs from 0.47 to 0.60.

    | `flag` | The test |
    |--------|----------|
    | `reverses` | 1 case pays and another costs, each clearing 0 alone |
    | `neutral_but_decisive` | label is `neutral`, but the cue moves the judgment |
    | `strong_and_consistent` | a `good` or `bad` cue that carries every case |

    **Warning: do not select on heterogeneity.** The median `i2` runs 0.66 to
    0.87, thus almost every cue is heterogeneous here and Cochran's Q
    separates nothing. `reverses` needs both sides to clear 0 alone.

    **Warning: do not read `theta` on a row that reverses.** It pools by
    precision, thus the largest case drives it. Read `theta_min`,
    `theta_max`, `cases_pay` and `cases_cost`.
    """)
    return


@app.cell
def _(CA, corpus):
    odd = CA.outliers(corpus)
    odd[[c for c in CA.OUTLIER_COLUMNS if c in odd.columns]]
    return (odd,)


@app.cell
def _(mo, odd):
    _m = odd[odd["minority_case"] != ""]["minority_case"].value_counts()
    mo.vstack([
        mo.md("**Which case stands alone?** `minority_case` names the case "
              "that disagrees with the rest of the corpus on a cue."),
        _m,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 6. The numbers behind the figures")
    return


@app.cell
def _(summary):
    keys = summary
    keys
    return (keys,)


@app.cell
def _(tables):
    per_case = tables
    per_case
    return (per_case,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG and CSV files")
    mo.md(
        f"""
    ## 7. Export

    {save}

    The files land in `master/figures/`. A corpus that fails the gate writes
    every name with a `_stale` suffix, thus a stale figure can never be read as
    a paper figure.
    """
    )
    return (save,)


@app.cell
def _(CA, FIG_DIR, NOTEBOOK_VERSION, corpus, gate_ok, matrix, max_keys, mo,
      rate_summary, save, summary, tables):
    if save.value and summary is not None and not summary.empty:
        _written = CA.export(
            matrix, summary, tables, rate_summary, FIG_DIR,
            max_keys=int(max_keys.value), gate_ok=gate_ok,
            provenance=CA.provenance_table(corpus, gate_ok),
        )
        # The word figures answer a different question, thus they take their
        # own tag and their own floors. See `export_words`.
        _written += CA.export_words(corpus, FIG_DIR, gate_ok=gate_ok)
        _msg = mo.md(
            f"Wrote {len(_written)} files to `master/figures/`: "
            + ", ".join(f"`{n}`" for n in _written)
            + f"\n\nNotebook {NOTEBOOK_VERSION}, `_cues_across` "
            + f"{CA.__version__}."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
