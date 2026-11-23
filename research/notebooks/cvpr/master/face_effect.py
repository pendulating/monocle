import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Does the camera direction move the judgment?

    Every photograph in the battery is 1 face of a Cyclomedia cube, and the
    cube is fixed to the **compass**, not to the car:

    | Face | The camera looks toward | The sun sits |
    |------|-------------------------|--------------|
    | `F` | north | behind the camera, all day |
    | `R` | east | in front in the morning, behind in the afternoon |
    | `B` | south | in front, near noon |
    | `L` | west | behind in the morning, in front in the afternoon |

    The face is a **camera decision**. A library does not become a better
    library when the car photographs it from the south. Thus a model that reads
    the place must give the same answer whichever face it gets, and every face
    coefficient below must be 0.

    ## The design the question asks for

    Isolate the comparisons where the SAME 2 places meet again, with a
    DIFFERENT pair of faces. Nothing then moves but the direction of the
    camera.

    Only the **libraries** case carries that design. It holds 236 places, thus
    its 1,000,000 draws fall on 27,730 place pairs and each pair repeats about
    39 times. 99.7% of its rows sit in a pair that came up again with a new
    face pair.

    The other facility cases hold 2,000 to 18,000 places, thus nearly every
    draw is a new place pair. They take the weaker design: a fixed effect on
    the place of each side, which removes every property of the place but not
    the pairing.

    | `fe` | What it holds fixed | Which cases |
    |------|---------------------|-------------|
    | `dyad` | the same 2 places | libraries |
    | `place` | the place of each side | the other 6 |

    **The 2 designs agree.** On the libraries case they give the same 4
    coefficients to 4 decimals. They differ only in the standard error, and
    section 3 says why.
    """)
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
    import pandas as pd

    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _faces as FX
    from research.notebooks.cvpr import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    FIG_DIR = _here / "figures"
    OUT_DIR = _here / "outputs"
    return C, FIG_DIR, FX, NOTEBOOK_VERSION, OUT_DIR, mo, pd


@app.cell(hide_code=True)
def _(C, mo):
    mo.md(f"""
    ## 1. The gate

    **Canonical registry:** {C.summary()}

    Every number below reads a registered `proxy` run of the `canonical` arm.
    """)
    return


@app.cell(hide_code=True)
def _(FX, mo):
    case_pick = mo.ui.dropdown(
        options=list(FX.CASES), value="libraries", label="Case")
    model_pick = mo.ui.dropdown(
        options=list(FX.MODELS), value="qwen3.5-9b", label="Rater")
    mo.md(
        f"""
    ## 2. One cell, in detail

    {case_pick} {model_pick}
    """
    )
    return case_pick, model_pick


@app.cell
def _(FX, case_pick, model_pick):
    frame = FX.load_face_frame(case_pick.value, model_pick.value)
    replication = FX.dyad_replication(frame) if frame is not None else None
    return frame, replication


@app.cell(hide_code=True)
def _(FX, case_pick, frame, mo, replication):
    if frame is None:
        _body = "The registry holds no such cell."
    else:
        _fe = FX.default_fe(case_pick.value)
        _body = f"""
    | Quantity | Value |
    |----------|-------|
    | Comparisons | {len(frame):,} |
    | With a usable label | {int(frame['y'].notna().sum()):,} |
    | Abstentions | {100 * frame['abstain'].mean():.1f}% |
    | Place pairs | {int(replication['n_dyads']):,} |
    | Median repeats of a place pair | {replication['median_repeats']:.0f} |
    | Pairs that repeat with a NEW face pair | {int(replication['dyads_with_2plus_face_combos']):,} |
    | Share of rows in those pairs | {100 * replication['share_of_rows']:.1f}% |
    | Design this case can carry | `{_fe}` |
    """
    mo.md(_body)
    return


@app.cell
def _(FX, case_pick, frame):
    effect = (None if frame is None else
              FX.fit(frame, fe=FX.default_fe(case_pick.value)))
    return (effect,)


@app.cell(hide_code=True)
def _(FX, effect, mo):
    if effect is None:
        _body = "No estimate. The cell holds too few usable comparisons."
    else:
        _rows = "\n".join(
            f"| {FX.FACE_LABEL[f]} | {effect.beta[f]:+.4f} | "
            f"{effect.se[f]:.4f} | {effect.t[f]:+.1f} | {effect.p[f]:.1e} |"
            for f in FX.FACES)
        _body = f"""
    ### The 4 coefficients

    A value is the shift of the ordinal label, from -2 to +2. The 4 sit on a
    mean of 0, thus a positive value says the model prefers a photograph that
    looks that way.

    | Face | Effect | SE | t | p |
    |------|--------|----|---|---|
    {_rows}

    **Swing (best minus worst): {effect.swing:.4f}**, SE {effect.swing_se:.4f},
    p {effect.swing_p:.1e}. Design `{effect.fe}`, clustered on `{effect.cluster}`,
    n = {effect.n:,}.

    {'; '.join(effect.notes) if effect.notes else ''}
    """
    mo.md(_body)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Read the standard error, not only the estimate

    A comparison touches 2 places, thus 2 comparisons that share a place carry
    correlated errors. The module offers 2 answers, and they answer 2
    questions:

    | Cluster | The question | Libraries / qwen |
    |---------|--------------|------------------|
    | `dyad` | Does the face move the judgment for THESE places? | t = 17 |
    | `dyadic` | Would it move it for a NEW library? | t = 2.3 |

    The second is conservative on purpose: the libraries case holds 236 places,
    thus 236 clusters carry the whole interval. Report it.

    **The grid settles the question that a single cell cannot.** 7 questions
    and 2 raters give 14 cells, and section 4 asks whether the same face wins
    in all of them.
    """)
    return


@app.cell(hide_code=True)
def _(FX, effect, frame, mo):
    if effect is None or frame is None:
        _fig1 = mo.md("")
    else:
        _fig1 = FX.plot_head_to_head(
            FX.head_to_head(frame, fe=effect.fe),
            title_value=f", {FX.CASE_LABEL[frame.attrs['case']]}")
    mo.vstack([
        mo.md(
            r"""
    ### The matrix, with no model

    Cell (f, g) is the mean label when side A carried face `f` and side B
    carried `g`, after the fixed effect. The matrix assumes no model, thus it
    also shows an interaction that a single coefficient for each face hides.
    It is antisymmetric by construction: the side order is arbitrary.
    """),
        _fig1,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The grid: 7 questions, 2 raters

    A single cell carries a wide interval. The grid does not. If the face were
    noise, the winner would move from cell to cell.

    **Warning: this cell reads 14 runs of 1,100,000 rows. It needs about 4
    minutes.**
    """)
    return


@app.cell
def _(FX):
    grid = FX.build()
    return (grid,)


@app.cell(hide_code=True)
def _(FX, grid, mo):
    mo.vstack([
        mo.md(f"**{FX.summary(grid)}**"),
        mo.ui.table(FX.to_display(grid), selection=None),
        mo.md(
            r"""
    | Column | Meaning | Value under no effect |
    |--------|---------|-----------------------|
    | the 4 face columns | the shift of the label, centred on 0 | 0 |
    | swing | the best face minus the worst | 0 |
    | swing / sd | the swing against the real spread of the place scores | 0 |
    | pctile | how far the swing carries a median place up the ranking | 0 |
    | place var. | the share of the between-place score variance the face mix explains | 0 |

    An image case gives no `swing / sd`, no `pctile` and no `place var.`. Its
    "place" is 1 recording, which sits in about 4 comparisons, thus a score
    over those 4 is nearly all sampling noise. The paper aggregates such a case
    to a census tract instead.

    The **schools / qwen3.5-9b** cell gives no estimate at all. That rater
    abstained on 99.7% of the schools comparisons, thus 3,384 usable rows
    remain out of 1,100,000.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(FX, grid, mo):
    mo.vstack([mo.md("### The coefficient figure"), FX.plot_grid(grid)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. The mechanism: the sun test

    A face effect alone does not name a cause. The cube is fixed to the
    compass, thus a face is also a **light condition**, and the light turns
    during the day while the street does not.

    | Window, local time | The sun sits | Backlit face | Front-lit face |
    |--------------------|--------------|--------------|----------------|
    | before 11 | east | `R` | `L` |
    | 11 to 14 | south | `B` | `F` |
    | after 14 | west | `L` | `R` |

    That gives a prediction a reader can falsify:

    - `R - L` must be **negative** in the morning, near **0** at midday, and
      **positive** in the afternoon.
    - `F - B` must stay **positive** all day. A north-looking camera never
      faces the sun.
    - If the model reads the CITY and not the light, then nothing moves with
      the hour. A street does not change direction at lunchtime.

    Only the 2 image cases carry `recordedAt`, thus only they can run the test.
    Each window keeps the pairs whose 2 photographs both fall inside it.
    """)
    return


@app.cell
def _(FX):
    hours = {}
    for _case in FX.IMAGE_CASES:
        for _model in FX.MODELS:
            _f = FX.load_face_frame(_case, _model)
            if _f is None:
                continue
            _h = FX.by_hour(_f)
            if not _h.empty:
                hours[(_case, _model)] = _h
    return (hours,)


@app.cell(hide_code=True)
def _(FX, hours, mo):
    _cols = ["window", "n", "beta_F", "beta_R", "beta_B", "beta_L",
             "R_minus_L", "F_minus_B"]
    mo.vstack([
        mo.vstack([
            mo.md(f"### {FX.CASE_LABEL[_key[0]]} — {_key[1]}"),
            mo.ui.table(_h[_cols].round(4), selection=None),
            FX.plot_by_hour(_h),
        ])
        for _key, _h in hours.items()
    ]) if hours else mo.md("No image case gave a timestamp.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. What it does to a published place score

    A single judgment is not the paper's output. A place score is, and it is a
    mean over many comparisons. The battery draws the face at random, thus most
    of the face effect cancels inside that mean.

    It does not cancel all the way. A place does not see the 4 faces evenly: a
    library on the north side of an east-west street is only visible from a
    camera that looks north, thus its pool is nearly all `F`. The **face mix**
    of a place is the channel by which the camera reaches its score.

    `place var.` in the grid measures that channel: the share of the
    between-place score variance that the face mix alone explains. The table
    below says the same thing as a rank shift.
    """)
    return


@app.cell
def _(FX, effect, frame, pd):
    if frame is None or effect is None:
        shift = pd.DataFrame()
    else:
        shift = FX.corrected_place_scores(frame, effect)
    return (shift,)


@app.cell(hide_code=True)
def _(case_pick, mo, model_pick, shift):
    if shift.empty:
        _body = "No place score for this cell."
    else:
        _s = shift["rank_shift"].abs()
        _body = f"""
    **{case_pick.value} — {model_pick.value}**, {len(shift):,} places.

    | Quantity | Value |
    |----------|-------|
    | Median absolute rank shift | {100 * _s.median():.2f} percentile points |
    | 90th percentile | {100 * _s.quantile(0.9):.2f} |
    | Largest | {100 * _s.max():.2f} |
    | Places that move more than 5 points | {100 * (_s > 0.05).mean():.1f}% |
    | Spearman, raw against corrected | {shift['mean_score'].corr(shift['corrected_score'], method='spearman'):.4f} |

    The correction subtracts the part of the score that the face mix explains,
    thus it gives the score a place would carry if its pool held the 4 faces
    evenly.
    """
    mo.md(_body)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. What this does not say

    - **The face is not only the light.** A camera that looks north also sees
      the south wall of the building, which is a different wall. The estimate
      holds the PLACE fixed, not the surface. The sun test in section 5
      separates the two: a wall does not turn during the day, and the `R - L`
      contrast does.
    - **It is a lower bound on the camera.** The hour, the season and the
      weather move a photograph as well, and no term here holds them.
    - **The place-level number is small on purpose.** The battery draws the
      face at random, thus it already does most of the work. The number would
      be far larger for a study that gave each place 1 photograph.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the tables and the figures")
    mo.md(
        f"""
    ## 8. Export

    {save}

    The button runs the same calls as
    `python research/scripts/export_face_effect.py`, which is the reproducible path.
    Prefer the script for a paper file.
    """
    )
    return (save,)


@app.cell
def _(FIG_DIR, FX, NOTEBOOK_VERSION, OUT_DIR, grid, hours, mo, pd, save):
    if save.value and not grid.empty:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        _v = NOTEBOOK_VERSION
        grid.to_parquet(OUT_DIR / f"face_effect_grid_v{_v}.parquet", index=False)
        (OUT_DIR / f"face_effect_table_v{_v}.tex").write_text(FX.to_latex(grid))
        if hours:
            _h = pd.concat(
                [v.assign(case=k[0], model=k[1]) for k, v in hours.items()],
                ignore_index=True)
            _h.to_parquet(OUT_DIR / f"face_by_hour_v{_v}.parquet", index=False)
        FX.plot_grid(grid).savefig(FIG_DIR / "face_effect_grid.png", dpi=300,
                                   bbox_inches="tight")
        for _k, _v2 in hours.items():
            FX.plot_by_hour(_v2).savefig(
                FIG_DIR / f"face_by_hour_{_k[0]}_{_k[1]}.png", dpi=300,
                bbox_inches="tight")
        _msg = mo.md(f"Wrote the grid, the table and the figures. "
                     f"Notebook {NOTEBOOK_VERSION}, `_faces` {FX.__version__}.")
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
