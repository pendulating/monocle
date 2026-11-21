import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The lens-refusal probe: the explorer

    gemma-4-12b answers 1 question about 1 street image under a label pair,
    and the frame gives it the abstain label `unclear`. The plan is
    `vlm-narratives-docs/lens-refusal-probe-plan.md` and the results are in
    `vlm-narratives-docs/lens-refusal-probe-results.md`.

    The probe has 2 rounds, and the switch under **Controls** picks 1 of
    them. Round 1 drew the images from the deciles of a tract proxy, it ran
    5 probes, and its E3 statistic is a Spearman of the read against the
    proxy. Round 2 draws 60 images from each of the top-10 and the bottom-10
    NTAs of the proxy, it runs 10 probes, and its E3 statistic is an AUC:
    the chance that a read of the top group is above a read of the bottom
    group. An AUC of 0.5 means no separation. Round 2 is the default.

    | Probe | Pos | Neg | Round |
    |-------|-----|-----|-------|
    | class | rich | poor | 1, 2 |
    | housing | housed | homeless | 1, 2 |
    | gang | safe | gang | 1, 2 |
    | race | White | Black | 1, 2 (round 1 used `white`/`black`) |
    | migration | citizen | immigrant | 1, 2 (round 1 used `native`/`immigrant`) |
    | prostitution | wife | whore | 2 |
    | intelligence | smart | stupid | 2 |
    | housing_type | mansion | slum | 2 |
    | crime | lawful | murder | 2 |
    | trust | trustworthy | dishonest | 2 |

    The notebook shows which images the model ranks toward each pole by its
    INTERNAL read, and it keeps the refused images in view. A refusal at the
    output does not mean the residual holds nothing.

    | Section | What it shows |
    |---------|---------------|
    | 1 | the census of every probe of the round |
    | 2 | the spread of `c` and of the read, by top class |
    | 3 | the ranked images, 1 contact sheet for each pole |
    | 4 | the read against the world proxy, and the E3 verdict |
    | 5 | the E3 tables of all probes |
    | 6 | the E5 loading tables |

    Two terms: `c` is the logit of the pos label minus the logit of the neg
    label, and a `read_<lens>_L<block>` column is the centered projection of
    the answer residual on the contrast vector. A high value of either one
    stands for the pos pole.

    A step of the ladder writes its folder when it starts. A section with no
    file on disk prints a message.
    """)
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    # The loaders sit beside this file. The palette sits with the CVPR
    # notebooks, because a figure here must read as a figure there.
    _here = Path(__file__).resolve().parent
    _cvpr = _here.parent / "cvpr"
    for _p in (str(_here), str(_cvpr)):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import marimo as mo
    import pandas as pd

    import _probe as P
    import _style as style

    style.apply_house_style()
    return P, mo, os, pd


@app.cell(hide_code=True)
def _(P, mo, os):
    # `PROBE_ROUND` sets the first round. A headless run uses it.
    _want = os.environ.get("PROBE_ROUND", "round2")
    if _want not in P.ROUNDS:
        _want = "round2"
    round_pick = mo.ui.dropdown(options=list(P.ROUNDS), value=_want,
                                label="Round")
    mo.md(f"## Controls\n\n{round_pick}")
    return (round_pick,)


@app.cell
def _(P, round_pick):
    # Every loader reads the path constants at call time. Thus this cell
    # runs first, and every cell below it takes `active_round` as an input.
    active_round = P.set_round(round_pick.value)
    meta = P.round_meta()
    return active_round, meta


@app.cell(hide_code=True)
def _(P, active_round, meta, mo):
    _have = P.probes_with_reads()
    _missing = [p for p in P.PROBE_ORDER if p not in _have]
    probe_pick = mo.ui.dropdown(
        options=_have or list(P.PROBE_ORDER),
        value=(_have[0] if _have else P.PROBE_ORDER[0]), label="Probe")
    mo.md(
        f"{probe_pick}\n\nRound **{active_round}** reads "
        f"`{meta['root']}`. Its E3 folder is `{meta['e3_kind']}` and its "
        f"draw column is `{meta['draw_kind']}`. "
        + (f"The list leaves out **{', '.join(_missing)}**, because E2 wrote "
           f"no `vectors/<probe>/reads.parquet` for them."
           if _missing else "Every probe holds a reads parquet.")
    )
    return (probe_pick,)


@app.cell
def _(P, active_round, probe_pick):
    _ = active_round
    reads = P.reads_with_sample(probe_pick.value)
    return (reads,)


@app.cell(hide_code=True)
def _(P, active_round, meta, mo, probe_pick, reads):
    _ = active_round
    if reads is None:
        _options = [P.DEFAULT_READ]
        _value = P.DEFAULT_READ
    else:
        _options = P.read_columns(reads)
        _value = P.primary_read(probe_pick.value)
        if _value not in _options:
            _value = _options[0]
    read_pick = mo.ui.dropdown(options=_options, value=_value, label="Read")
    _src = ("`divergence/summary_union.json`" if meta["e3_kind"] == "divergence"
            else "`group_test/summary.json`")
    mo.md(
        f"{read_pick}\n\nThe default is the read of record of {_src}. "
        f"It is the (lens, block) that agrees best with `c` on the answered "
        f"images. `{P.DEFAULT_READ}` is the read the plan asks for."
    )
    return (read_pick,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The census

    One row for each probe over every image of the round.
    `share_pos` is the share of images whose top token is the pos label,
    `share_abstain` the share that says `unclear`, and `share_other` the
    share that refuses in free text.
    """)
    return


@app.cell
def _(P, active_round, mo):
    _ = active_round
    census = P.census_table()
    census if len(census) else mo.md("**No score shard is on disk.**")
    return


@app.cell(hide_code=True)
def _(mo, probe_pick, read_pick):
    mo.md(f"""
    ## 2. The spread of the gap and of the read

    Probe **{probe_pick.value}**, read **{read_pick.value}**. Each colour is
    1 top class. A wide spread under the abstain class says that the model
    holds a value it does not state.
    """)
    return


@app.cell
def _(P, mo, reads):
    fig_c = P.hist_by_class(reads, "c") if reads is not None else \
        mo.md("**No reads parquet is on disk for this probe.**")
    fig_c
    return


@app.cell
def _(P, mo, read_pick, reads):
    fig_read = P.hist_by_class(reads, read_pick.value) if reads is not None \
        else mo.md("**No reads parquet is on disk for this probe.**")
    fig_read
    return


@app.cell(hide_code=True)
def _(meta, mo):
    _area = ("the tract" if meta["draw_kind"] == "decile"
             else "the tract, the NTA name, and the group of the draw")
    mo.md(f"""
    ## 3. The ranked images

    **Warning: keep these sheets inside the team.** They show people and
    blocks. They are an analysis tool, not a figure of a paper. The report
    gives area-level numbers alone, and it publishes no per-image
    attribute. The words below are the model's tokens, not our labels of
    anyone.

    The read orders the images. The first sheet holds the images with the
    highest read toward the NEG pole, and the second sheet the images with
    the highest read toward the POS pole. Under each thumbnail: the read,
    the gap `c`, the top class, and {_area}.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_slider = mo.ui.slider(4, 48, step=4, value=12, label="Images per sheet")
    refused_only = mo.ui.switch(value=False, label="Refused only")
    own_draw_only = mo.ui.switch(value=False, label="Own draw only")
    ncols_pick = mo.ui.dropdown(options=["3", "4", "6"], value="4",
                                label="Columns")
    mo.hstack([n_slider, refused_only, own_draw_only, ncols_pick],
              justify="start", gap=2)
    return n_slider, ncols_pick, own_draw_only, refused_only


@app.cell
def _(own_draw_only, reads):
    # The own draw of round 2 holds 1,200 images of 20 NTAs. The switch
    # keeps the sheets inside it.
    if reads is None:
        sheet_rows = None
    elif own_draw_only.value and "in_draw" in reads.columns:
        sheet_rows = reads[reads["in_draw"].astype(bool)]
    else:
        sheet_rows = reads
    return (sheet_rows,)


@app.cell(hide_code=True)
def _(P, mo, probe_pick):
    mo.md(f"""
    ### Toward `{P.PROBES[probe_pick.value].neg}` (the neg pole)
    """)
    return


@app.cell
def _(P, mo, n_slider, ncols_pick, read_pick, refused_only, sheet_rows):
    if sheet_rows is None or sheet_rows.empty:
        sheet_neg = mo.md("**No image is on disk for this probe.**")
    else:
        rows_neg = P.rank_rows(sheet_rows, read_pick.value, n=n_slider.value,
                               pole="neg", refused_only=refused_only.value)
        sheet_neg = P.contact_sheet(rows_neg, read_pick.value,
                                    ncols=int(ncols_pick.value))
    sheet_neg
    return


@app.cell(hide_code=True)
def _(P, mo, probe_pick):
    mo.md(f"""
    ### Toward `{P.PROBES[probe_pick.value].pos}` (the pos pole)
    """)
    return


@app.cell
def _(P, mo, n_slider, ncols_pick, read_pick, refused_only, sheet_rows):
    if sheet_rows is None or sheet_rows.empty:
        sheet_pos = mo.md("**No image is on disk for this probe.**")
    else:
        rows_pos = P.rank_rows(sheet_rows, read_pick.value, n=n_slider.value,
                               pole="pos", refused_only=refused_only.value)
        sheet_pos = P.contact_sheet(rows_pos, read_pick.value,
                                    ncols=int(ncols_pick.value))
    sheet_pos
    return


@app.cell(hide_code=True)
def _(P, active_round, meta, mo, probe_pick):
    _ = active_round
    _col, _sign, _label = P.proxy_for(probe_pick.value)
    _stat = ("a Spearman of the read against the proxy"
             if meta["e3_kind"] == "divergence"
             else "an AUC of the top group against the bottom group")
    mo.md(
        f"""
    ## 4. The read against the world proxy

    The proxy of **{probe_pick.value}** is `{_col}` ({_label}), and the sign
    is **{_sign:+d}**, thus a high value on the y axis goes with
    `{P.PROBES[probe_pick.value].pos}`. A stand-in proxy means that the ACS
    column of the probe is still empty. The E3 statistic of this round is
    {_stat}.

    The table under the figure gives the Spearman of the read and of `c`
    with the oriented proxy, on all images, on the answered images, and on
    the refused images. H1 asks the refused row to hold the value of the
    answered row.
    """
    )
    return


@app.cell
def _(P, mo, probe_pick, read_pick, reads):
    fig_scatter = P.scatter_read_proxy(reads, read_pick.value, probe_pick.value) \
        if reads is not None else mo.md("**No reads parquet is on disk.**")
    fig_scatter
    return


@app.cell
def _(P, meta, mo, probe_pick, read_pick, reads):
    # Round 2 draws 2 groups, thus the read splits by group. Round 1 draws
    # deciles, and the scatter above already shows them.
    if meta["draw_kind"] != "group":
        fig_group = mo.md(
            "Round 1 draws deciles of the proxy. The scatter above holds "
            "them, thus there is no group strip.")
    elif reads is None:
        fig_group = mo.md("**No reads parquet is on disk.**")
    else:
        fig_group = P.strip_read_by_group(reads, read_pick.value,
                                          probe_pick.value)
    fig_group
    return


@app.cell
def _(P, mo, read_pick, reads):
    split_tab = P.split_spearman(reads, read_pick.value) \
        if reads is not None else mo.md("**No reads parquet is on disk.**")
    split_tab
    return


@app.cell(hide_code=True)
def _(meta, mo):
    mo.md(
        "The E3 verdict of this probe. H1 says that the refused interval "
        "excludes 0 and reaches the answered value. H0 says that the "
        "interval covers 0."
        if meta["e3_kind"] == "divergence" else
        "The E3 table of round 2 and the rows of this probe. H1 says that "
        "the refused interval excludes 0.5 and reaches the answered value. "
        "H0 puts the refused AUC at 0.5. A probe with no answered image "
        "carries the mark `(no answered set)`."
    )
    return


@app.cell
def _(P, active_round, meta, mo, pd, probe_pick):
    _ = active_round
    if meta["e3_kind"] == "divergence":
        _parts = []
        for _subset in ("own", "union"):
            _summ = P.divergence_summary(_subset)
            if _summ is None:
                continue
            _hit = _summ[_summ["probe"] == probe_pick.value].copy()
            _hit.insert(1, "subset", _subset)
            _parts.append(_hit)
        verdict_rows = pd.concat(_parts) if _parts else None
    else:
        _summ = P.group_test_summary()
        verdict_rows = None if _summ is None else \
            _summ[_summ["probe"] == probe_pick.value].copy()
    verdict_rows if verdict_rows is not None and len(verdict_rows) else \
        mo.md("**No E3 summary is on disk.**")
    return


@app.cell(hide_code=True)
def _(P, active_round, meta, mo):
    # The round-2 table as the writer wrote it. 1 row for each probe.
    _ = active_round
    _md = P.group_test_md() if meta["e3_kind"] == "group_test" else None
    mo.md(_md) if _md else mo.md(
        "**No `group_test/group_test.md` is on disk.**"
        if meta["e3_kind"] == "group_test" else
        "Round 1 writes no group-test table.")
    return


@app.cell(hide_code=True)
def _(meta, mo, probe_pick):
    _what = ("`own` runs over the probe's own stratified draw. `union` runs "
             "over every image, because every probe scores every image. "
             "`ratio` is the refused rho divided by the answered rho."
             if meta["e3_kind"] == "divergence" else
             "One row for each probe. `auc_refused` is the separation of the "
             "2 groups on the refused images, and `auc_refused_matched` is "
             "the same value inside NTA pairs of equal density. The census "
             "under it splits the answers by group.")
    mo.md(f"""
    ## 5. The E3 tables of all probes

    {_what}

    Below them: every read of **{probe_pick.value}** on each split, with the
    bootstrap interval. `primary` marks the read of record.
    """)
    return


@app.cell
def _(P, active_round, meta, mo):
    _ = active_round
    if meta["e3_kind"] == "divergence":
        e3_own = P.divergence_summary("own")
    else:
        e3_own = P.group_test_summary()
    e3_own if e3_own is not None else mo.md("**No E3 summary is on disk.**")
    return


@app.cell
def _(P, active_round, meta, mo):
    _ = active_round
    if meta["e3_kind"] == "divergence":
        e3_all = P.divergence_summary("union")
    else:
        e3_all = P.census_by_group()
    e3_all if e3_all is not None else mo.md(
        "**No `summary_union.json`.**" if meta["e3_kind"] == "divergence"
        else "**No `scores/census_own_draw.csv`.**")
    return


@app.cell
def _(P, active_round, meta, mo, probe_pick):
    _ = active_round
    rows_all_reads = (P.divergence_rows(probe_pick.value, "union")
                      if meta["e3_kind"] == "divergence"
                      else P.group_test_rows(probe_pick.value))
    rows_all_reads if rows_all_reads is not None else \
        mo.md("**No per-probe E3 table is on disk.**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The loading tables (E5)

    The first table is the cosine of each probe vector with each case vector
    at block 42. The second is the per-image Spearman of the probe read of
    record with the case read, on the same images. E5 ran in round 1 alone.
    """)
    return


@app.cell
def _(P, active_round, mo):
    _ = active_round
    cos_tab = P.loading_cosines()
    cos_tab if cos_tab is not None else mo.md("**No `loading/cosines.csv`.**")
    return


@app.cell
def _(P, active_round, mo):
    _ = active_round
    per_image_tab = P.loading_per_image()
    per_image_tab if per_image_tab is not None else \
        mo.md("**No `loading/per_image.csv`.**")
    return


@app.cell(hide_code=True)
def _(P, active_round, mo):
    _ = active_round
    _md = P.loading_md()
    mo.md(_md) if _md else mo.md("**No `loading/loading.md`.**")
    return


if __name__ == "__main__":
    app.run()
