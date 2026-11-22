import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Pseudo-differentiation and what the model discards

    2 questions of the IC codebook, category 3.

    ## Pseudo-differentiation

    The model states an answer (a `verdict`), then doubts it ("Wait, let me
    look again": an `uncertainty` span with marker `reconsideration`), and
    then writes the SAME answer as its `decision`. The doubt looks like a
    second view, but it changes nothing. For each trace:

    | Flag | Rule |
    |------|------|
    | `has_verdict` | at least 1 verdict before the decision |
    | `reconsiders_after` | at least 1 reconsideration after the FIRST verdict and before the decision |
    | `decision_unchanged` | the decision label equals the LAST verdict before it |
    | `pseudo` | has_verdict, reconsiders_after, decision_unchanged |
    | `revised` | has_verdict, reconsiders_after, NOT decision_unchanged |

    The variants: `*_any` reads any uncertainty span in place of a
    reconsideration; `*_first` reads the FIRST verdict in place of the last;
    `revised_direction` asks whether the winner (A, B, or undecided) moved,
    not only the strength of the label.

    **Warning: a span after the decision does not count.** It cannot change
    the decision. The loader keeps the last decision of a trace, and it
    makes the last verdict the decision when the extractor wrote none.

    **Warning: the uncertainty rate is a fact about the extractor.** The
    Qwen3.8-27B extractor marks about 2x the uncertainty spans of the
    Gemma-4-31B extractor on the same traces. Read section 4 before a share.

    **Warning: the Gemma-4-31B qwen arm reads a trace in chunks of 6,000
    characters.** A chunk that states an answer writes a decision, thus an
    earlier chunk's decision can be a verdict in fact, and the loader keeps
    only the last decision.

    ## What the model discards

    A `dismissal` span names a thing and sets it aside. `about` says what:
    a `cue`, a `perspective`, or a `premise` of the prompt ("I don't see a
    library"). `target` is a free short phrase. A dismissal MATCHES a named
    span of the same trace when the span's normalised name equals the
    normalised target, or when the span text holds the target.
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    _shared = _here.parent
    if str(_shared) not in sys.path:
        sys.path.insert(0, str(_shared))

    import marimo as mo
    import pandas as pd

    import _canonical as C
    import _extractions as X
    import _ic_v3 as I
    import _style as style

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


@app.cell
def _(I, arm, corpus):
    bundle = ({} if corpus is None else
              I.arm_bundle(corpus, arm, questions=("differentiation",)))
    return (bundle,)


@app.cell(hide_code=True)
def _(I, bundle, mo):
    if not bundle:
        _view = mo.md("_No corpus._")
    else:
        _p = I.pooled_differentiation(bundle["flags"].drop(columns="arm"))
        _view = mo.md(f"""
    ## 2. Pseudo-differentiation

    ### The headline, over every case

    | Number | Value |
    |--------|-------|
    | Traces with a decision | {int(_p['n_traces']):,} |
    | Share with a verdict before the decision | {_p['share_verdict']:.3f} |
    | Share with a reconsideration after the first verdict | {_p['share_recon_after']:.3f} |
    | Share pseudo (verdict, reconsideration, decision unchanged) | {_p['share_pseudo']:.3f} |
    | Share revised (verdict, reconsideration, decision changed) | {_p['share_revised']:.3f} |
    | Of the traces that reconsider: the decision changes | {_p['revised_given_recon']:.3f} |
    | ... against the FIRST verdict | {_p['revised_first_given_recon']:.3f} |
    | ... the WINNER changes (A, B, undecided) | {_p['revised_direction_given_recon']:.3f} |
    | Of the traces with any doubt after a verdict: the decision changes | {_p['revised_any_given_doubt']:.3f} |
    | Mean reconsiderations, pseudo traces | {_p['mean_recon_pseudo']:.2f} |
    | Mean reconsiderations, revised traces | {_p['mean_recon_revised']:.2f} |
    """)
    _view
    return


@app.cell
def _(I, arm, bundle, mo):
    _diff = bundle.get("diff_case", I.pd.DataFrame())
    (mo.vstack([
        mo.md("### By case"),
        I.captioned(I.plot_differentiation(_diff, I.arm_label(arm),
                                           arm=arm)),
        _diff.drop(columns="arm").round(3),
    ]) if not _diff.empty else None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. What the model discards

    The share of each `about`, by case, and the most frequent targets. The
    colour of a target bar is its most frequent `about`.
    """
    )
    return


@app.cell
def _(I, arm, bundle, mo):
    _about = bundle.get("about_case", I.pd.DataFrame())
    _rate = bundle.get("rate_case", I.pd.DataFrame())
    (mo.vstack([
        I.captioned(I.plot_about(_about, I.arm_label(arm), arm=arm)),
        _about.drop(columns="arm").round(3),
        mo.md("Dismissals for each 100 traces:"),
        _rate.drop(columns="arm").round(3),
    ]) if not _about.empty else mo.md("_No dismissal._"))
    return


@app.cell(hide_code=True)
def _(mo):
    n_targets = mo.ui.slider(3, 20, value=10, step=1, show_value=True,
                             label="Targets for each case")
    n_targets
    return (n_targets,)


@app.cell
def _(I, arm, bundle, mo, n_targets):
    _dm = bundle.get("dismissals", I.pd.DataFrame())
    targets = (I.top_targets(_dm.drop(columns="arm", errors="ignore"),
                             n=int(n_targets.value))
               if not _dm.empty else I.pd.DataFrame())
    (mo.vstack([I.captioned(I.plot_top_targets(
                    targets, n=min(int(n_targets.value), 8), arm=arm)),
                targets.round(3)])
     if not targets.empty else mo.md("_No target over the floor._"))
    return (targets,)


@app.cell(hide_code=True)
def _(I, bundle, mo):
    _match = bundle.get("match_case", I.pd.DataFrame())
    (mo.vstack([
        mo.md(
            "### Did the model name the target before it set it aside?\n\n"
            "`named_<class>` matches by the normalised name only. "
            "`matched_<class>` adds the text rule. `matched_elsewhere` "
            "leaves out a span that overlaps the dismissal clause, which is "
            "often the same sentence marked twice. `named_before` keeps only "
            "a span that ENDS before the dismissal starts: the model named "
            "it, then set it aside.\n\n"
            "**Warning: a `premise` target is the facility of the prompt** "
            "(\"library building\", \"park or plaza\"). A match then says "
            "the model also described the facility, not that it reasoned "
            "about it."),
        _match.drop(columns="arm").round(3),
    ]) if not _match.empty else None)
    return


@app.cell(hide_code=True)
def _(mo):
    arms_load = mo.ui.checkbox(
        value=False, label="Read the 4 version 3 arms (about a minute)")
    mo.md(
        f"""
    ## 4. The 2 x 2 of judge and extractor

    The same traces of each judge, read by 2 extractors. A share that holds
    along a judge comes from the traces. A share that moves comes from the
    extractor.

    **Warning: the box reads all 4 corpora**, 1 at a time. That takes about
    a minute and up to about 2 GB at the peak.

    {arms_load}
    """
    )
    return (arms_load,)


@app.cell
def _(I, allow_stale, arms_load):
    arm_bundles = (I.across_arms(require_canonical=not allow_stale.value,
                                 questions=("differentiation",))
                   if arms_load.value else {})
    return (arm_bundles,)


@app.cell
def _(I, arm_bundles, mo):
    if not arm_bundles:
        _view = mo.md("_Tick the box to read the 4 arms._")
    else:
        _long = I.stack(arm_bundles, "diff_case")
        _view = mo.vstack([
            mo.md("### Pseudo-differentiation, each arm over every case"),
            I.differentiation_by_arm(arm_bundles).round(3),
            I.captioned(I.plot_differentiation_arms(_long)),
            I.captioned(I.plot_differentiation_arms(_long, "share_pseudo")),
            mo.md("### Dismissals, each arm over every case"),
            I.dismissals_by_arm(arm_bundles).round(3),
            mo.accordion({
                "Pseudo-differentiation, every case of every arm":
                    _long.round(3),
                "Dismissal `about`, every case of every arm":
                    I.stack(arm_bundles, "about_case").round(3),
            }),
        ])
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the PNG, CSV, and TeX files")
    mo.md(
        f"""
    ## 5. Export

    {save}

    The files land in `master/figures/` under the tag
    `ic_differentiation_<arm>_`. When section 4 read the 4 arms, the export
    also writes the `ic_differentiation_arms_` files. Each PNG gets a
    `.json` sidecar with its title, caption, LaTeX label, and tags. The LaTeX tables
    (`pseudo_by_case`, `dismissals`, and `pseudo` for the 4 arms) land in
    `master/outputs/`.
    """
    )
    return (save,)


@app.cell
def _(FIG_DIR, I, NOTEBOOK_VERSION, OUT_DIR, arm, arm_bundles, bundle,
      gate_ok, mo, save):
    if save.value and bundle:
        _written = I.export_differentiation(bundle, FIG_DIR, arm,
                                            gate_ok=gate_ok)
        _written += I.export_differentiation_latex(
            bundle, OUT_DIR, arm, NOTEBOOK_VERSION, gate_ok=gate_ok)
        if arm_bundles:
            _written += I.export_across_arms(arm_bundles, FIG_DIR,
                                             gate_ok=gate_ok)
            _written += I.export_across_arms_latex(
                arm_bundles, OUT_DIR, NOTEBOOK_VERSION, gate_ok=gate_ok)
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
