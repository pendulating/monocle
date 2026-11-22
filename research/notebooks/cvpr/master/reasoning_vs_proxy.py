import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The reasoning against the validation by proxy

    The paper validates the judgments against outside measurements (the proxy
    arm). This notebook asks, at the level of 1 pair: do the spans of a
    reasoning trace mark the traces that agree with the proxy?

    ## The join

    The join of `_ic_link`, in 2 modes:

    | Mode | The value of each side | Cases |
    |------|------------------------|-------|
    | `unit` | the headline unit proxy of `_unit_validation`: the station, the building, the school, the DPR property, the restaurant, or the nearest rated street segment | 6 cases |
    | `area` | the community district polygon that holds the unit (or the camera): median household income, crime density | 4 cases |

        proxy_gap = value(canonical a) - value(canonical b)
        agrees    = sign(proxy_gap) == sign(relative_score)

    Every proxy value is "higher is better", thus crime density is negated.
    Street photography has no outside measurement.

    ## The control for difficulty

    A model reasons longest about a hard pair, and a hard pair is 2 units
    with close values. Thus each cell (case, mode, proxy) splits into 4 bands
    by `abs(proxy_gap)`, and the contrast compares the traces with a feature
    to the traces without it INSIDE a band. The pooled value is the
    Mantel-Haenszel risk difference over the (cell, band) strata, with a
    normal 95% interval.

    **Warning: the sample holds 1,000 traces of each case, and 100 to 900
    of them reach a cell.** Read the interval, not the point.

    **Warning: many names are tested in section 4.** Read a name only when
    its interval clears 0 under both extractors of the judge.

    **Warning: this is an association, not a cause.** The spans and the
    judgment come from the same run.
    """
    )
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
    bands = mo.ui.slider(2, 6, value=4, step=1, show_value=True,
                         label="Gap bands inside each cell")
    mo.md(
        f"""
    ## 0. The gate and the corpus

    **Canonical registry:** {C.summary()}

    {arm_pick}

    {bands}

    {allow_stale}
    """
    )
    return allow_stale, arm_pick, bands


@app.cell
def _(I, M, allow_stale, arm_pick, mo):
    arm = arm_pick.value
    try:
        corpus = M.load_arm(arm, require_canonical=not allow_stale.value)
        meta = M.pair_meta(M.judge_of(arm))
        gate_ok, _err = not allow_stale.value, ""
    except (RuntimeError, FileNotFoundError) as _exc:
        corpus, meta, gate_ok, _err = None, None, False, str(_exc)
    mo.md(f"**No usable corpus.**\n\n```\n{_err}\n```" if corpus is None else
          f"**{I.arm_label(arm)}.** {len(corpus):,} spans, "
          f"{corpus['tid'].nunique():,} traces."
          + ("" if gate_ok else " **The gate is OFF.**"))
    return arm, corpus, gate_ok, meta


@app.cell
def _(M, bands, corpus, meta, mo):
    if corpus is None:
        pairs, proxy_cov = None, None
        _view = mo.md("_No corpus._")
    else:
        pairs, proxy_cov = M.all_proxy_pairs(meta, tids=set(corpus["tid"]),
                                             quantiles=int(bands.value))
        _view = mo.vstack([
            mo.md("## 1. The cells and their agreement\n\n`traces` is the "
                  "sampled traces of the case, `pairs` the pairs that reach "
                  "the cell, and `agreement` the share whose vote has the sign "
                  "of the gap (Wilson interval)."),
            proxy_cov.round(3)])
    _view
    return pairs, proxy_cov


@app.cell
def _(M, corpus, mo, pairs):
    if corpus is None or pairs is None or pairs.empty:
        costly, features, frame = None, None, None
        _view = mo.md("_No pair._")
    else:
        costly = M.costly_perspectives(corpus)
        features = M.trace_features(corpus, costly)
        frame = M.proxy_frame(pairs, features)
        frame["family"] = M.proxy_family(frame["proxy"])
        _share = features[list(M.BOOL_FEATURES)].mean().rename("share of traces")
        _view = mo.vstack([
            mo.md("## 2. The trace features\n\nThe rules are in "
                  "`_ic_metadata.trace_features`. `costly` uses the "
                  f"{len(costly)} (case, perspective) cells whose win rate sits "
                  "under the base rate of the case."),
            _share.round(3).to_frame()])
    _view
    return costly, features, frame


@app.cell
def _(M, arm, frame, mo):
    if frame is None:
        contrasts, contrasts_cell = None, None
        _view = mo.md("_No pair._")
    else:
        contrasts = M.contrast_table(frame, by="family")
        contrasts_cell = M.contrast_table(
            frame, features=["wealth_inference", "risk_inference", "good_margin_hi",
                             "premise_dismissal", "costly", "favors_loser"],
            by="cell")
        _key = contrasts[contrasts["feature"].isin(
            ["wealth_inference", "risk_inference", "good_margin_hi",
             "premise_dismissal", "costly", "favors_loser", "tc_for_winner",
             "hedged_any"])]
        _view = mo.vstack([
            mo.md("## 3. The band-controlled contrasts\n\n`diff` is the "
                  "agreement of the traces with the feature less the agreement "
                  "of the traces without it, inside the strata. `n1` and `n0` "
                  "are the pairs in the kept strata. The families: `income` "
                  "(median household income of the polygon), `crime_area` "
                  "(crime density of the polygon, negated), `unit` (the "
                  "headline unit proxy)."),
            M.captioned(M.plot_contrasts(_key, arm=arm)),
            contrasts.round(3),
            mo.accordion({"Each cell": contrasts_cell.round(3)}),
        ])
    _view
    return contrasts, contrasts_cell


@app.cell(hide_code=True)
def _(mo):
    min_traces = mo.ui.slider(25, 200, value=40, step=5, show_value=True,
                              label="Fewest traces for a name in a family")
    mo.md(
        f"""
    ## 4. Which names mark the traces that agree

    For each name (a perspective, an inference, a dimension with its
    valence): the band-controlled contrast of the traces that hold the name
    against the traces that do not, in the `unit` and the `income`
    families.

    {min_traces}
    """
    )
    return (min_traces,)


@app.cell
def _(M, corpus, frame, min_traces, mo, pd):
    if frame is None:
        markers = None
        _view = mo.md("_No pair._")
    else:
        _rows = []
        for _cls, _wv in (("perspective", False), ("inference", False), ("dimension", True)):
            for _fam in ("unit", "income"):
                _nm = M.name_markers(frame[frame["family"].eq(_fam)], corpus, _cls,
                                     min_traces=int(min_traces.value), with_valence=_wv)
                if not _nm.empty:
                    _rows.append(_nm.assign(extraction_class=_cls, family=_fam))
        markers = pd.concat(_rows, ignore_index=True) if _rows else pd.DataFrame()
        _clear = markers[(markers["lo"] > 0) | (markers["hi"] < 0)]
        _view = mo.vstack([
            mo.md(f"{len(_clear)} of {len(markers)} (name, family) cells have "
                  "an interval that clears 0."),
            _clear[["extraction_class", "family", "feature", "traces", "diff",
                    "lo", "hi", "rate1", "rate0"]].round(3),
            mo.accordion({"Every name": markers.round(3)}),
        ])
    _view
    return (markers,)


@app.cell(hide_code=True)
def _(I, arm, mo):
    _sib = I.sibling_arm(arm)
    sibling_load = mo.ui.checkbox(
        value=False,
        label=(f"Read the other extractor of this judge: {I.arm_label(_sib)} "
               "(about 40 seconds)") if _sib else "No other extractor",
    )
    mo.md(
        f"""
    ## 5. The same contrasts under the other extractor

    The 2 arms of 1 judge share the traces, the pairs, and the proxy. Only
    the spans differ. A contrast that holds under both comes from the
    traces.

    {sibling_load}
    """
    )
    return (sibling_load,)


@app.cell
def _(I, M, allow_stale, arm, contrasts, markers, min_traces, mo, pairs, pd,
      sibling_load):
    _sib = I.sibling_arm(arm)
    if contrasts is None or not (sibling_load.value and _sib):
        replication = None
        _view = mo.md("_Tick the box to read the other extractor._")
    else:
        _d = M.load_arm(_sib, require_canonical=not allow_stale.value)
        _f = M.proxy_frame(pairs, M.trace_features(_d, M.costly_perspectives(_d)))
        _f["family"] = M.proxy_family(_f["proxy"])
        _c = M.contrast_table(_f, by="family")
        _rows = []
        for _cls, _wv in (("perspective", False), ("inference", False), ("dimension", True)):
            for _fam in ("unit", "income"):
                _nm = M.name_markers(_f[_f["family"].eq(_fam)], _d, _cls,
                                     min_traces=int(min_traces.value), with_valence=_wv)
                if not _nm.empty:
                    _rows.append(_nm.assign(extraction_class=_cls, family=_fam))
        del _d
        _sm = pd.concat(_rows, ignore_index=True) if _rows else pd.DataFrame()
        _feat = contrasts.merge(_c, on=["group", "feature"], suffixes=("", "_sib"))
        _names = markers.merge(_sm, on=["extraction_class", "family", "feature"],
                               suffixes=("", "_sib"))
        _both = _names[((_names["lo"] > 0) & (_names["lo_sib"] > 0))
                       | ((_names["hi"] < 0) & (_names["hi_sib"] < 0))]
        replication = {"features": _feat, "names": _names, "names_both": _both}
        _view = mo.vstack([
            _feat[["group", "feature", "diff", "lo", "hi", "diff_sib", "lo_sib",
                   "hi_sib"]].round(3),
            mo.md(f"**{len(_both)} names clear 0 with the same sign under both "
                  "extractors.**"),
            _both[["extraction_class", "family", "feature", "traces", "diff", "lo",
                   "hi", "traces_sib", "diff_sib", "lo_sib", "hi_sib"]].round(3),
        ])
    _view
    return (replication,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the CSV, PNG, and LaTeX files")
    mo.md(
        f"""
    ## 6. Export

    {save}

    The files land in `master/outputs/explore/` under the tag
    `reasoning_vs_proxy_<arm>_`. The LaTeX tables are
    `<tag>_proxy_contrasts_v<version>.tex`, and
    `<tag>_name_markers_v<version>.tex` when section 5 reads the other
    extractor.
    """
    )
    return (save,)


@app.cell
def _(M, NOTEBOOK_VERSION, OUT_DIR, arm, contrasts, contrasts_cell, gate_ok,
      markers, mo, proxy_cov, replication, save):
    if save.value and contrasts is not None:
        _prefix = f"reasoning_vs_proxy_{arm}" + ("" if gate_ok else "_stale")
        _tables = {"coverage": proxy_cov, "contrasts": contrasts,
                   "contrasts_cell": contrasts_cell, "name_markers": markers}
        if replication is not None:
            _tables.update({"replication_features": replication["features"],
                            "replication_names": replication["names_both"]})
        _written = M.export(_tables, OUT_DIR, _prefix)
        # The paper tables of this arm. The name table needs the other
        # extractor (section 5).
        _tex = {"proxy_contrasts": M.latex_proxy_contrasts({arm: contrasts})}
        if replication is not None:
            _tex["name_markers"] = M.latex_name_markers({arm: replication["names_both"]})
        _written += M.write_latex(_tex, OUT_DIR, _prefix, NOTEBOOK_VERSION)
        # The PNG gets a `.json` sidecar with its caption.
        M.save_figure(
            M.plot_contrasts(contrasts[contrasts["feature"].isin(
                ["wealth_inference", "risk_inference", "good_margin_hi",
                 "premise_dismissal", "costly", "favors_loser"])], arm=arm),
            OUT_DIR / f"{_prefix}_contrasts.png", dpi=200, bbox_inches="tight",
            provenance={"notebook_version": NOTEBOOK_VERSION})
        _msg = mo.md(f"Wrote {len(_written) + 2} files to `{OUT_DIR}`. "
                     f"Notebook {NOTEBOOK_VERSION}, `_ic_metadata` {M.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
