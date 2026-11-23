import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The vocabulary of each judge

    **The question.** Which keys does 1 judge use that the other judge does
    not, in each case? The 2 judges are Gemma-4-12B and Qwen3.5-9B. The
    classes are the 3 named classes of the version 3 schema: `dimension`
    (the attribute observed), `inference` (the attribute evaluated), and
    `perspective` (the mode of valuation). A **key** is the normalised free
    `name` of a span.

    ## The confound

    **Warning: the extractor writes the name, not the judge.** An extractor
    model reads the trace of the judge and writes the short phrase. The 2
    extractors (Qwen3.8-27B, Gemma-4-31B) differ in vocabulary and in
    volume: Qwen3.8-27B writes about 1.5x the spans. Thus a comparison of
    `gemma_by_qwen27_v3` with `qwen_by_gemma31_v3` (the 2 primary arms)
    mixes the judge and the extractor.

    This notebook holds the extractor fixed:

    | Extractor | Gemma-4-12B traces | Qwen3.5-9B traces |
    |-----------|--------------------|-------------------|
    | Qwen3.8-27B | `gemma_by_qwen27_v3` | `qwen_by_qwen27_v3` |
    | Gemma-4-31B | `gemma_by_gemma31_v3` | `qwen_by_gemma31_v3` |

    It compares the 2 judges in each row, then asks which contrasts hold in
    both rows. **A key that is distinctive of 1 judge under BOTH extractors
    is a fact about the judge. A key that is distinctive under 1 extractor
    only is not.**

    ## The method

    - The unit is the trace. The count of a key is the number of traces
      that use it 1 time or more (the document count). The qwen traces are
      about 2.5x longer, and the Gemma-4-31B extractor read them in
      6,000-character chunks, thus a raw span count is not usable.
    - The contrast is the weighted log-odds ratio of Monroe, Colaresi and
      Quinn (2008), with an informative Dirichlet prior from the pooled
      counts of both judges in the case and extractor. The z is positive
      for a Gemma-4-12B key and negative for a Qwen3.5-9B key.
    - A key is **robust** (judge-distinctive) when its z has 1 sign and
      |z| >= 1.96 under both extractors, and it has the floor of traces
      under both.

    **Warning: the z compares shares of each judge's vocabulary, not
    per-trace rates.** The qwen traces hold 1.4x to 2.1x the distinct keys
    of the gemma traces. A key that both judges use in almost every trace
    can thus be a larger share of the gemma vocabulary and take a positive
    z, while its per-trace rate is lower for gemma. A dagger marks such a
    key. Every robust qwen key also has the higher per-trace rate.

    **Warning: the z has no correction for the number of keys.**
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

    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _figmeta as F
    from research.notebooks.cvpr import _judge_vocab as J
    from research.notebooks.cvpr import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    # The paper outputs go to `outputs/judge_vocab/`, never to `figures/`.
    OUT_DIR = _here / "outputs" / "judge_vocab"
    return C, F, J, NOTEBOOK_VERSION, OUT_DIR, mo, pd


@app.cell(hide_code=True)
def _(C, mo):
    allow_stale = mo.ui.checkbox(
        value=False,
        label="Read a corpus the registry does not name (a look only, "
              "never for the paper)",
    )
    mo.md(
        f"""
    ## 1. The gate and the corpus

    **Canonical registry:** {C.summary()}

    The notebook reads the 4 v3 arms with the paper filters (about 50
    seconds).

    {allow_stale}
    """
    )
    return (allow_stale,)


@app.cell
def _(J, allow_stale, mo):
    try:
        docs = J.load_grid(require_canonical=not allow_stale.value)
        gate_ok, gate_error = not allow_stale.value, ""
    except (RuntimeError, FileNotFoundError) as _exc:
        docs, gate_ok, gate_error = None, False, str(_exc)
    mo.stop(docs is None, mo.md(f"**No usable corpus.**\n\n```\n{gate_error}\n```"))
    mo.md(f"**{docs['arm'].nunique()} arms read.** "
          + ("" if gate_ok else "**The gate is OFF**: every file takes a "
             "`_stale` suffix."))
    return docs, gate_ok


@app.cell(hide_code=True)
def _(J, docs, mo):
    _cov = J.coverage(docs)
    _br = J.breadth(docs)
    mo.vstack([
        mo.md("### What the corpus holds\n\n`traces` is the denominator: "
              "every answered trace. `keys_<class>` is the distinct keys."),
        _cov,
        mo.md("### The distinct keys in 1 trace\n\nThis is the `n` of the "
              "log-odds, divided by the traces. The qwen judge spreads its "
              "traces over more keys under both extractors."),
        _br.round(2),
    ])
    return


@app.cell(hide_code=True)
def _(J, mo):
    case_pick = mo.ui.dropdown(options=list(J.CASES), value=J.CASES[0],
                               label="Case")
    class_pick = mo.ui.dropdown(options=list(J.CLASSES), value="perspective",
                                label="Class")
    min_traces = mo.ui.slider(3, 50, value=10, step=1, show_value=True,
                              label="Fewest traces for a key (min_traces)")
    top_k = mo.ui.slider(3, 10, value=5, step=1, show_value=True,
                         label="Keys of each judge (top k)")
    min_cases = mo.ui.slider(2, 7, value=3, step=1, show_value=True,
                             label="Fewest cases for a general key")
    mo.vstack([mo.md("## 2. The controls"),
               mo.hstack([case_pick, class_pick], gap=2),
               mo.hstack([min_traces, top_k, min_cases], gap=2)])
    return case_pick, class_pick, min_cases, min_traces, top_k


@app.cell
def _(J, docs, min_cases, min_traces):
    bundle = J.build_bundle(docs, min_traces=int(min_traces.value),
                            min_cases=int(min_cases.value))
    return (bundle,)


@app.cell(hide_code=True)
def _(bundle, mo):
    mo.vstack([
        mo.md(r"""
    ## 3. The headline: how much of the contrast holds under both extractors

    1 row for each class. `sig_q27` and `sig_g31` are the (case, key) pairs
    with |z| >= 1.96 under 1 extractor. `robust_<judge>` holds under both.
    `robust_rate_<judge>` is the part of those whose per-trace rate is also
    higher for that judge. `general_<judge>` is the robust keys in
    `min_cases` cases or more. `survive` is the robust pairs over the pairs
    significant under 1 extractor or more. The Spearman is over the cases.
    """),
        bundle["summary"].round(3),
    ])
    return


@app.cell(hide_code=True)
def _(bundle, mo):
    mo.vstack([
        mo.md(r"""
    ### The agreement of the 2 extractors, by case and class

    `spearman` is the rank correlation of the 2 extractors' z over the
    `n_both` keys that both extractors see (`min_traces` or more). A high
    value means that the 2 extractors rank the judge contrast in the same
    order. `same_sign` is the share of those keys whose 2 z have 1 sign.
    """),
        bundle["agreement"].round(3),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The keys of each judge, by class

    The figure for the class picked above. The caption is under the figure.
    """)
    return


@app.cell
def _(F, J, bundle, class_pick, min_traces, mo, top_k):
    _fig = J.plot_class(bundle["robust"], class_pick.value, k=int(top_k.value),
                        min_traces=int(min_traces.value))
    mo.vstack([_fig, mo.md(F.caption_md(_fig))])
    return


@app.cell
def _(J, bundle, case_pick, class_pick, min_traces, mo, top_k):
    _r = bundle["robust"]
    _cols = ["key", "side", "robust", "rate_agrees", "z_q27", "z_g31", "z_mean",
             "df_gemma_q27", "df_qwen_q27", "df_gemma_g31", "df_qwen_g31"]
    _top = J.top_keys(_r, class_pick.value, k=int(top_k.value),
                      min_traces=int(min_traces.value))
    case_table = _top[_top["case"] == case_pick.value][_cols]
    mo.vstack([mo.md(f"### {case_pick.value}, {class_pick.value}: the top "
                     "keys of each judge"),
               case_table.round(3)])
    return


@app.cell(hide_code=True)
def _(bundle, case_pick, class_pick, mo):
    _u = bundle["unique"]
    _u = _u[(_u["case"] == case_pick.value)
            & (_u["extraction_class"] == class_pick.value)]
    mo.vstack([
        mo.md(r"""
    ### The strictly unique keys

    A key that 1 judge uses in `min_traces` traces or more and the other
    judge in none, under 1 extractor. `both_extractors` is True when it is
    unique to the same judge under the 2 extractors. **Warning: strict
    uniqueness picks rare keys.** Read the z ranking above first.
    """),
        _u.round(3),
    ])
    return


@app.cell(hide_code=True)
def _(bundle, class_pick, mo):
    _c = bundle["cross_case"]
    _c = _c[_c["extraction_class"] == class_pick.value]
    mo.vstack([
        mo.md(r"""
    ## 5. Across the cases: a general vocabulary or 1 case only

    The robust keys of each judge, with the number of cases where they hold.
    `scope` is `general` at `min_cases` cases or more, `case-specific` in 1
    case. `n_rate_agrees` is the cases where the per-trace rate also agrees.
    """),
        _c.round(3),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG, PDF, JSON, TeX, and CSV files")
    mo.md(
        f"""
    ## 6. Export

    {save}

    The files land in `master/outputs/judge_vocab/`: 1 figure for each class
    (`judge_vocab_<class>.png` and `.pdf`, with the `.json` caption), 1 key
    table for each class (`judge_vocab_<class>_keys_v<version>.tex`), the
    agreement table (`judge_vocab_agreement_v<version>.tex`), and the CSVs.
    `research/scripts/export_judge_vocab.py` writes the same files without marimo.
    """
    )
    return (save,)


@app.cell
def _(J, NOTEBOOK_VERSION, OUT_DIR, bundle, gate_ok, min_traces, mo, save,
      top_k):
    if save.value:
        _written = J.export(bundle, OUT_DIR, version=NOTEBOOK_VERSION,
                            gate_ok=gate_ok, k=int(top_k.value), n_tex=5,
                            min_traces=int(min_traces.value))
        _msg = mo.md(f"Wrote {len(_written)} files: "
                     + ", ".join(f"`{n}`" for n in _written)
                     + f"\n\nNotebook {NOTEBOOK_VERSION}, `_judge_vocab` "
                     + f"{J.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
