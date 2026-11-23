import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # What grounds a perspective

    A `perspective` span names an evaluative standard ("modern", "standard",
    "maintenance"). The IC codebook asks what the standard rests on: which
    observations (`dimension`) and which evaluative claims (`inference`) go
    with it. This notebook reads 2 links:

    | Link | Rule | Unit |
    |------|------|------|
    | Co-occurrence | the 2 spans sit in the same trace | the trace |
    | Nesting | 1 span lies inside the other, by character offsets | the trace |

    The nesting has 3 relations:

    | Relation | Meaning |
    |----------|---------|
    | `contains_perspective` | the perspective word lies inside the inference or dimension clause: the mode word of the schema ("a **modern** glass facade") |
    | `inside_perspective` | the inference or dimension lies inside the perspective clause |
    | `same_span` | 1 clause carries the 2 classes |

    ## The arithmetic

    `lift` is the co-occurrence against independence INSIDE each case:
    `expected` sums, over the cases, the traces of the case that name the
    perspective times the share of the case's traces that name the partner.
    Thus a road perspective and a road dimension do not link only because
    both live in the road case. `nest_rate` is the share of the traces that
    name the perspective where the partner nests with it.

    **Warning: co-occurrence is weak.** A gemma trace by Qwen27 holds about
    60 spans, thus most partners sit in most traces. On the primary gemma
    arm 90% of the pairs over the floor have a lift of 0.83 to 1.75. Nesting is the strong link: on the primary gemma
    arm 68% to 88% of the perspective spans sit inside an inference clause.

    **Warning: the names are the model's own phrases**, joined by spelling
    only. The partner key is `name | kind` for an inference and
    `name | valence` for a dimension, as in `master/ic_perspectives.py`.
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
    from research.notebooks.cvpr import _ic_v3 as I
    from research.notebooks.cvpr import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.1.0"
    FIG_DIR = _here / "figures"
    # The LaTeX tables go to `outputs/`, never to `figures/`.
    OUT_DIR = _here / "outputs"
    return C, FIG_DIR, I, NOTEBOOK_VERSION, OUT_DIR, X, mo, pd


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
    ## 1. The gate and the corpus

    **Canonical registry:** {C.summary()}

    {arm_pick}

    {allow_stale}
    """
    )
    return allow_stale, arm_pick


@app.cell
def _(I, allow_stale, arm_pick, mo):
    arm = arm_pick.value
    try:
        corpus = I.load_arm(arm, require_canonical=not allow_stale.value)
        gate_ok, gate_error = not allow_stale.value, ""
    except (RuntimeError, FileNotFoundError) as _exc:
        corpus, gate_ok, gate_error = None, False, str(_exc)
    if corpus is None:
        _msg = f"**No usable corpus.**\n\n```\n{gate_error}\n```"
    else:
        _msg = (f"**{I.arm_label(arm)}.** {len(corpus):,} spans, "
                f"{corpus['tid'].nunique():,} traces."
                + ("" if gate_ok else " **The gate is OFF**: every file "
                   "takes a `_stale` suffix."))
    mo.md(_msg)
    return arm, corpus, gate_ok


@app.cell(hide_code=True)
def _(mo):
    min_pair = mo.ui.slider(10, 200, value=25, step=5, show_value=True,
                            label="Fewest traces for a pair")
    n_heads = mo.ui.slider(5, 30, value=15, step=1, show_value=True,
                           label="Perspectives to show")
    mo.hstack([min_pair, n_heads], gap=2)
    return min_pair, n_heads


@app.cell
def _(I, arm, corpus, min_pair):
    bundle = ({} if corpus is None else
              I.arm_bundle(corpus, arm, min_count=int(min_pair.value),
                           questions=("grounding",)))
    grounding = bundle.get("grounding", I.pd.DataFrame())
    return bundle, grounding


@app.cell(hide_code=True)
def _(bundle, mo):
    _over = bundle.get("nest_overview")
    if _over is None or _over.empty:
        _view = mo.md("_No corpus._")
    else:
        _wide = (_over.pivot_table(index=["case", "case_label"],
                                   columns=["partner_class", "relation"],
                                   values="share_spans").round(3))
        _wide.columns = [f"{c} {r}" for c, r in _wide.columns]
        _view = mo.vstack([
            mo.md("## 2. How often a perspective nests\n\n"
                  "The share of the perspective spans with at least 1 "
                  "partner in each relation."),
            _wide.reset_index(),
        ])
    _view
    return


@app.cell
def _(I, arm, bundle, mo):
    _over = bundle.get("nest_overview")
    (I.captioned(I.plot_nesting(_over, arm=arm)) if _over is not None and not _over.empty
     else mo.md("_No nesting._"))
    return


@app.cell(hide_code=True)
def _(mo):
    rank_pick = mo.ui.radio(
        options={"the nesting rate": "nest_rate",
                 "the co-occurrence lift": "lift",
                 "the count of shared traces": "n_both"},
        value="the nesting rate", label="Rank the partners by",
    )
    mo.md(
        f"""
    ## 3. The partners of each perspective

    For the most named perspectives: the 5 inference keys and the 5
    dimension keys of highest rank. A rank by the count favors the partners
    that sit in every trace (`architecture | neutral`). A rank by the lift
    favors the small pairs near the floor. The nesting rate names the clause
    the perspective word lives in.

    {rank_pick}
    """
    )
    return (rank_pick,)


@app.cell
def _(I, grounding, n_heads, rank_pick):
    partners = I.top_partners(grounding, n_perspectives=int(n_heads.value),
                              n=5, by=rank_pick.value)
    _cols = ["perspective", "partner_class", "partner",
             "n_perspective_traces", "n_both", "share", "lift", "n_nested",
             "nest_rate", "n_contains", "n_inside", "n_same"]
    (partners[[c for c in _cols if c in partners.columns]].round(3)
     if not partners.empty else None)
    return (partners,)


@app.cell
def _(I, arm, grounding, mo, n_heads):
    if grounding.empty:
        _view = mo.md("_No pair over the floor._")
    else:
        _figs = []
        for _cls in I.PARTNER_CLASSES:
            for _value in ("nest_rate", "log2_lift"):
                try:
                    _figs.append(I.captioned(I.plot_grounding(
                        grounding, _cls, _value,
                        n_perspectives=min(int(n_heads.value), 15), arm=arm)))
                except I.EmptyFigure:
                    pass
        _view = mo.vstack([
            mo.md("### The heat maps\n\nThe nesting rate first, then the "
                  "co-occurrence lift, for the inference spans and then the "
                  "dimension spans."),
            *_figs,
        ])
    _view
    return


@app.cell(hide_code=True)
def _(I, bundle, mo):
    _by_key = bundle.get("nest_by_key", I.pd.DataFrame())
    if _by_key.empty:
        _view = mo.md("_No nesting._")
    else:
        _view = mo.vstack([
            mo.md("## 4. The pairs that nest most often\n\n"
                  "`n_traces` is the traces that nest the pair in the "
                  "relation; `n_nested` is the span pairs."),
            mo.ui.tabs({
                _rel.replace("_", " "): I.nested_partners(_by_key, _rel)
                for _rel in I.RELATIONS
            }),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(I, arm, mo):
    _sib = I.sibling_arm(arm)
    sibling_load = mo.ui.checkbox(
        value=False,
        label=(f"Read the other extractor of this judge: {I.arm_label(_sib)}"
               if _sib else "No other extractor for this judge"),
    )
    mo.md(
        f"""
    ## 5. The nesting under the other extractor

    The nesting of a perspective is a choice of the extractor as much as a
    fact of the trace: the schema TEACHES the mode word inside an inference.
    The other extractor of the same judge shows how much of the share holds.

    {sibling_load}
    """
    )
    return (sibling_load,)


@app.cell
def _(I, allow_stale, arm, bundle, mo, sibling_load):
    _sib = I.sibling_arm(arm)
    if sibling_load.value and _sib:
        _df = I.load_arm(_sib, require_canonical=not allow_stale.value)
        _, sibling_overview = I.nesting(_df)
        del _df
        sibling_overview.insert(0, "arm", _sib)
        _both = I.pd.concat([bundle.get("nest_overview", I.pd.DataFrame()),
                             sibling_overview], ignore_index=True)
        _wide = (_both.pivot_table(index=["case_label", "partner_class",
                                          "relation"],
                                   columns="arm", values="share_spans")
                 .round(3).reset_index())
        _view = _wide
    else:
        sibling_overview = I.pd.DataFrame()
        _view = mo.md("_Tick the box to read the other extractor._")
    _view
    return (sibling_overview,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG, CSV, and TeX files")
    mo.md(
        f"""
    ## 6. Export

    {save}

    The files land in `master/figures/` under the tag
    `ic_grounding_<arm>_`. Each PNG gets a `.json` sidecar with its
    title, caption, LaTeX label, and tags. The 2 LaTeX tables (`nesting`,
    `strongest_nests`) land in `master/outputs/`.
    """
    )
    return (save,)


@app.cell
def _(FIG_DIR, I, NOTEBOOK_VERSION, OUT_DIR, arm, bundle, gate_ok, mo, save):
    if save.value and bundle:
        _written = I.export_grounding(bundle, FIG_DIR, arm, gate_ok=gate_ok)
        _written += I.export_grounding_latex(bundle, OUT_DIR, arm,
                                             NOTEBOOK_VERSION, gate_ok=gate_ok)
        _msg = mo.md(f"Wrote {len(_written)} files: "
                     + ", ".join(f"`{n}`" for n in _written)
                     + f"\n\nNotebook {NOTEBOOK_VERSION}, `_ic_v3` "
                     + f"{I.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
