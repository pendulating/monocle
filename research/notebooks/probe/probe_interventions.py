import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The lens-refusal probe: the interventions

    Three steps of the ladder of
    `research/docs/lens-refusal-probe-plan.md` act on the model
    instead of the ranks:

    | Step | Question | Folder |
    |------|----------|--------|
    | E4 | Does the coordinate cause the answer? | `steer/<probe>/<run>/` |
    | E1b, E5c | Which words does the lens read on the way? | `scrape/<probe>/` |
    | E5d | Where does a label token enter the top of the list? | `ranks/` |

    The switch under **Controls** picks the round. Round 1 sits under
    `outputs/_monocle/probe/` and round 2 under
    `outputs/_monocle/probe_nta/`. Round 2 runs 10 probes and draws its
    images from the top-10 and the bottom-10 NTAs of each proxy.

    E4 adds `alpha` times the contrast vector at the answer position of a
    refused image. The `signal` arm uses the vector of the probe, and the
    `control` arm uses a random direction of the same length. The go rule
    asks for a dose where the signal turns more than 50% of the refusals
    into a pole and the control turns under 20%. A run whose name ends with
    `_wide` holds a wider dose grid.

    The scrape reads the top words of 3 position groups: the patches
    (`image`), the question tokens (`text_after`), and the answer position
    (`answer`). Each group has the lens whose estimand matches it: `mm` at
    the patches, `C` (`mm_textafter`) at the text, and `B` (`wikitext1k`)
    as the text-only check.

    A step that is still on the queue prints a message here.
    """)
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

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
def _(active_round, meta, mo):
    mo.md(f"""
    ## 1. E4, the dose response

    Round **{active_round}** reads `{meta['root']}`. The table holds every
    run folder of its `steer/`. `has_parquet` False means the run holds a
    folder alone, thus it is still on the queue.
    """)
    return


@app.cell
def _(P, active_round):
    _ = active_round
    steer_tab = P.steer_runs()
    return (steer_tab,)


@app.cell
def _(mo, steer_tab):
    steer_tab if len(steer_tab) else mo.md("**No `steer/` folder is on disk.**")
    return


@app.cell(hide_code=True)
def _(mo, steer_tab):
    # The picker lists every run of the active round, `_wide` included.
    # Round 2 holds 20 runs, thus the plots draw the picked runs alone.
    _ready = steer_tab[steer_tab["has_parquet"]] if len(steer_tab) else steer_tab
    _keys = [f"{r['probe']}/{r['run']}" for _, r in _ready.iterrows()]
    run_pick = mo.ui.multiselect(
        options=_keys, value=_keys[:2], label="Steering runs to draw")
    run_pick if _keys else mo.md("**No `steer.parquet` is on disk.**")
    return (run_pick,)


@app.cell
def _(P, pd, run_pick, steer_tab):
    # One dose table for each picked run.
    _frames = []
    steer_data = {}
    for _, _r in steer_tab.iterrows():
        _key = f"{_r['probe']}/{_r['run']}"
        if not _r["has_parquet"] or _key not in run_pick.value:
            continue
        _df = P.load_steer(_r["path"])
        if _df is None or _df.empty:
            continue
        steer_data[_key] = _df
        _tab = P.dose_table(_df)
        _tab.insert(0, "run", _r["run"])
        _tab.insert(0, "probe", _r["probe"])
        _frames.append(_tab)
    dose_all = pd.concat(_frames, ignore_index=True) if _frames else None
    return dose_all, steer_data


@app.cell
def _(dose_all, mo):
    dose_all if dose_all is not None else mo.md(
        "**No dose grid to show. Pick a run above.**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The summary of each run of the round. `scale` is the length of the
    vector at dose 1, `cos_signal_control` proves that the control points
    elsewhere, and `go_doses` names the doses that pass the rule.
    """)
    return


@app.cell
def _(P, mo, pd, steer_tab):
    _rows = []
    for _, _r in steer_tab[steer_tab["has_summary"]].iterrows():
        _s = P.load_steer_summary(_r["path"])
        if _s is None:
            continue
        _rows.append({
            "probe": _s.get("probe", _r["probe"]),
            "run": _r["run"],
            "split": _s.get("split"),
            "lens": _s.get("lens"),
            "layer": _s.get("layer"),
            "n_images": _s.get("n_images"),
            "scale": _s.get("scale"),
            "cos_signal_control": _s.get("cos_signal_control"),
            "go": _s.get("go"),
            "go_doses": ", ".join(str(v) for v in _s.get("go_doses", [])),
        })
    steer_summary = pd.DataFrame(_rows)
    steer_summary if len(steer_summary) else mo.md(
        "**No steering summary is on disk.**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The left panel of each pair holds the share of images that answer a pole
    against the dose. The right panel splits that share into the 3 answer
    classes: a solid line is the signal arm and a dashed line the control
    arm.
    """)
    return


@app.cell
def _(P, mo, steer_data):
    _blocks = []
    for _key, _df in steer_data.items():
        _blocks.append(mo.md(f"**{_key}**"))
        _blocks.append(mo.as_html(P.plot_dose(P.dose_table(_df))))
    mo.vstack(_blocks) if _blocks else mo.md("**No dose grid to draw.**")
    return


@app.cell(hide_code=True)
def _(P, active_round, mo):
    _ = active_round
    _have = [p for p in P.PROBE_ORDER if P.scrape_dirs(p)]
    probe_pick = mo.ui.dropdown(
        options=_have or list(P.PROBE_ORDER),
        value=(_have[0] if _have else P.PROBE_ORDER[0]), label="Probe")
    mo.md(f"## 2. The scrape\n\nPick the probe and the run folder.\n\n"
          f"{probe_pick}")
    return (probe_pick,)


@app.cell
def _(P, active_round, probe_pick):
    _ = active_round
    scrape_options = [str(d) for d in P.scrape_dirs(probe_pick.value)]
    return (scrape_options,)


@app.cell(hide_code=True)
def _(mo, scrape_options):
    dir_pick = mo.ui.dropdown(
        options=scrape_options or ["none"],
        value=(scrape_options[0] if scrape_options else "none"),
        label="Scrape folder")
    top_k = mo.ui.slider(1, 10, step=1, value=5, label="Top k per position")
    hide_special = mo.ui.switch(value=True, label="Hide control tokens")
    mo.hstack([dir_pick, top_k, hide_special], justify="start", gap=2)
    return dir_pick, hide_special, top_k


@app.cell
def _(P, dir_pick, scrape_options):
    if scrape_options:
        tokens = P.load_tokens(dir_pick.value)
        label_ranks = P.load_label_ranks(dir_pick.value)
    else:
        tokens = None
        label_ranks = None
    return label_ranks, tokens


@app.cell(hide_code=True)
def _(mo, tokens):
    mo.md(
        "The scrape folder of this probe holds no token file."
        if tokens is None else
        f"The folder holds **{len(tokens):,}** token rows over "
        f"**{tokens['image_path'].nunique()}** images."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a. The patch words

    The plan asks for block 22 under the lens `mm`. The scrape holds the
    blocks below, and the control picks the block nearest to 22. The patch
    residual is question-blind, because the image tokens come before the
    question. Thus these words describe the scene, not the inference.
    """)
    return


@app.cell(hide_code=True)
def _(P, mo, tokens):
    _layers = P.layers_of(tokens, "image", "mm") if tokens is not None else []
    _default = P.nearest_layer(_layers, 22)
    patch_layer = mo.ui.dropdown(
        options=[str(v) for v in _layers] or ["none"],
        value=(str(_default) if _default is not None else "none"),
        label="Patch block (lens mm)")
    patch_layer
    return (patch_layer,)


@app.cell
def _(P, hide_special, mo, patch_layer, tokens, top_k):
    if tokens is None or patch_layer.value == "none":
        patch_refused = mo.md("**No patch scrape is on disk.**")
    else:
        patch_refused = P.word_counts(
            tokens, "image", "mm", int(patch_layer.value), "refused",
            top_k=top_k.value, hide_special=hide_special.value)
    patch_refused
    return


@app.cell
def _(P, hide_special, mo, patch_layer, tokens, top_k):
    if tokens is None or patch_layer.value == "none":
        patch_answered = mo.md("**No patch scrape is on disk.**")
    else:
        patch_answered = P.word_counts(
            tokens, "image", "mm", int(patch_layer.value), "answered",
            top_k=top_k.value, hide_special=hide_special.value)
    patch_answered
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2b. The question-token words

    Block 38 under the lens C, with the words that lens B also holds in its
    top 5. Lens B is the text-only check. It does not hold every block, thus
    the message below names the block the mark comes from. The token
    identities of a scrape depend on the estimand under block 42, so a word
    that both lenses give is the safer claim.
    """)
    return


@app.cell(hide_code=True)
def _(P, mo, tokens):
    _layers = P.layers_of(tokens, "text_after", "C") if tokens is not None else []
    _default = P.nearest_layer(_layers, 38)
    question_layer = mo.ui.dropdown(
        options=[str(v) for v in _layers] or ["none"],
        value=(str(_default) if _default is not None else "none"),
        label="Question block (lens C)")
    question_layer
    return (question_layer,)


@app.cell
def _(P, hide_special, question_layer, tokens, top_k):
    if tokens is None or question_layer.value == "none":
        q_refused = None
        q_answered = None
        b_layer = None
    else:
        q_refused, b_layer = P.question_words(
            tokens, int(question_layer.value), "refused",
            top_k=top_k.value, hide_special=hide_special.value)
        q_answered, _ = P.question_words(
            tokens, int(question_layer.value), "answered",
            top_k=top_k.value, hide_special=hide_special.value)
    return b_layer, q_answered, q_refused


@app.cell(hide_code=True)
def _(P, b_layer, mo, question_layer, tokens):
    _b_layers = P.layers_of(tokens, "text_after", "B") if tokens is not None else []
    if b_layer is None:
        _msg = "**No question scrape is on disk.**"
    elif str(b_layer) == question_layer.value:
        _msg = f"The mark comes from lens B at block {b_layer}."
    else:
        _msg = (f"**Lens B holds no block {question_layer.value}.** It holds "
                f"{_b_layers}. The mark comes from block {b_layer}.")
    mo.md(_msg)
    return


@app.cell
def _(mo, q_refused):
    q_refused if q_refused is not None else mo.md("**No question scrape.**")
    return


@app.cell
def _(mo, q_answered):
    q_answered if q_answered is not None else mo.md("**No question scrape.**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2c. The label ranks at the answer position

    The median rank of the 3 label tokens in the lens vocabulary, by block
    and by split. Rank 0 is the top of the distribution. A pole that falls
    while the abstain label stays at the top says that the residual carries
    the judgment under the refusal.
    """)
    return


@app.cell(hide_code=True)
def _(label_ranks, mo):
    _lenses = sorted(label_ranks[label_ranks["group"] == "answer"]["lens"]
                     .unique()) if label_ranks is not None else []
    rank_lens = mo.ui.dropdown(
        options=[str(v) for v in _lenses] or ["none"],
        value=("C" if "C" in _lenses else (str(_lenses[0]) if _lenses else "none")),
        label="Lens of the answer read")
    rank_lens
    return (rank_lens,)


@app.cell
def _(P, label_ranks, mo, probe_pick, rank_lens):
    if label_ranks is None or rank_lens.value == "none":
        rank_tab = None
    else:
        rank_tab = P.label_rank_table(label_ranks, probe_pick.value,
                                      group="answer", lens=rank_lens.value)
    rank_tab if rank_tab is not None and len(rank_tab) else \
        mo.md("**No label rank file is on disk.**")
    return (rank_tab,)


@app.cell
def _(P, mo, probe_pick, rank_tab):
    P.plot_label_ranks(rank_tab, probe_pick.value) \
        if rank_tab is not None and len(rank_tab) else \
        mo.md("**No label rank file to draw.**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2d. The scrape summary as the writer wrote it

    The keys of the summary file of the chosen folder.
    """)
    return


@app.cell
def _(P, dir_pick, mo, scrape_options):
    _summ = P.load_scrape_summary(dir_pick.value) if scrape_options else None
    mo.md("**No scrape summary is on disk.**" if _summ is None else
          "Keys: " + ", ".join(f"`{k}`" for k in _summ))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. The rank trajectory (E5d)

    The median rank of the 3 label tokens along the blocks, on the answered
    images and on the refused images. The step writes `ranks/trajectory.csv`
    after every shard lands.
    """)
    return


@app.cell
def _(P, active_round, mo):
    _ = active_round
    traj = P.ranks_trajectory()
    traj if traj is not None else mo.md(
        f"**No `ranks/trajectory.csv` under `{P.RANKS_DIR}`.** The rank "
        "step of this round is still on the queue.")
    return


@app.cell(hide_code=True)
def _(P, active_round, mo):
    _ = active_round
    _md = P.ranks_md()
    mo.md(_md) if _md else mo.md("**No `ranks/trajectory.md`.**")
    return


if __name__ == "__main__":
    app.run()
