import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The prompt ablations

    The ablation results, apart from the master table. `results_table.py`
    holds the canonical arm alone: the validation by proxy, the unit layer,
    and the raw cross-model agreement. This notebook holds every arm that
    changed ONE thing in the prompt.

    This notebook is **downstream**. Every section reads the run parquets of
    the canonical registry, never W&B. Section 4 also reads the proxy
    parquets that the case notebooks exported, thus run those first.

    Three batteries judged the same UNIT pairs with ONE thing changed. The
    registry holds them as arms beside the canonical prompt, thus this
    notebook chooses no run: it asks for an arm and joins.

    **Warning: a matched `pair_id` names the same 2 units, and in `nocity`
    and `textfirst` it does NOT name the same 2 photographs.** Section 4
    gives the measured share for each case.

    | Arm | What changed | Pairs for each case |
    |-----|--------------|---------------------|
    | `noabstain` | the decoder cannot emit `NotSure` | 1,000,000 |
    | `nocity` | the prompt does not name the city | 100,000 |
    | `textfirst` | the prompt text comes BEFORE the images | 100,000 |

    The `textfirst` arm gives each rater ONE change against ITS OWN production
    layout: qwen3.5-9b goes `images_then_text` to `text_first`, and
    gemma-4-12b goes `interleaved_labels` to `text_first_interleaved`. That
    arch does not take plain `text_first`, which puts the 2 images next to
    each other and makes the model read them as 1 image.

    **Every row holds the 1 change its arm names.** The gemma halves of
    `nocity` and `textfirst` re-ran on 2026-09-02 and the `noabstain` gemma
    half on 2026-09-04, all on the layout of the canonical gemma control. The
    `layout conf.` column stays and is empty; it marks a future arm that runs
    another layout.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    _here = Path(__file__).resolve().parent

    import marimo as mo

    from research.notebooks.cvpr import _ablations as A
    from research.notebooks.cvpr import _canonical as C
    from research.notebooks.cvpr import _proxy_arms as PA
    from research.notebooks.cvpr import _quantile_shift as Q
    from research.notebooks.cvpr import _results_table as R
    from research.notebooks.cvpr import _style as style

    # The gate. Every section reads a run of the canonical registry. Stop
    # when the registry no longer matches the disk, because then a table
    # describes runs that moved or changed.
    C.verify_or_raise()

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    OUT_DIR = _here / "outputs"
    return A, NOTEBOOK_VERSION, OUT_DIR, PA, Q, R, mo


@app.cell(hide_code=True)
def _(R, mo):
    layer_pick = mo.ui.radio(
        options=list(R.LAYER_KEY), value=R.DEFAULT_LAYER, label="Geography layer"
    )
    mo.md(
        f"""
    ## Controls

    {layer_pick}

    The layer drives section 4 alone. Sections 1 to 3 read the comparisons
    and the places, and a comparison sits in no polygon.

    **Community district is the default for a reason.** It is the only layer
    with a usable $n$ for every prompt.
    """
    )
    return (layer_pick,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The ablation table

    **The join key is `pair_id`, not `canonical_pair_id`.** `pair_id` carries
    the presentation order. The 100,000-pair draw is a subset of the 1,000,000
    -pair draw, but about 8% of the shared pairs came up in the OPPOSITE A/B
    order. A join on the canonical id would compare a pair with its own mirror
    image and read the order effect as a prompt effect. The `match` column
    reports the share of the arm's rows that the join keeps.

    | Column | Meaning | No effect |
    |--------|---------|-----------|
    | n | Pairs both arms judged, in the same order | -- |
    | match | Share of the arm's rows the join kept | 1.00 |
    | abstain canon / arm | Share of `NotSure` in each arm | -- |
    | label agr | Same label of the 6 | the marginals |
    | label kappa | The same, corrected for chance | 0 |
    | both answer | Pairs where neither arm abstained | -- |
    | dir kappa | Same side, corrected for chance | 0 |
    | delta | Mean ordinal score, arm minus canonical, PAIRED | 0 |
    | se | Standard error of that paired mean | -- |

    **`label agr` is not a quality measure.** It answers "how far did the
    prompt change move the labels?". A low value is a large effect.

    **The standard error is the paired one.** Both arms judged the same pair,
    thus `se` uses the standard deviation of the DIFFERENCE. It is much
    smaller than an unpaired error, and the test is much sharper.

    **A row under `R.MIN_BOTH` answered pairs shows no direction and no
    score.** A score needs a pair that BOTH arms answered. The qwen3.5-9b
    schools row leaves 26 such pairs of 100,938, thus it shows dashes.
    """)
    return


@app.cell
def _(A):
    ablations = A.all_arms()
    ablation_table = A.to_display(ablations)
    ablation_table
    return (ablations,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Where the held-back judgments went

    This is the question the forced-choice battery exists to answer. The rows
    are the pairs the CANONICAL arm answered `NotSure`. The columns are the
    share of them that the model gave to each label when it could not abstain.

    `lean` is the mean ordinal score of those forced labels. It is 0 when the
    forced choices split evenly around `Same`, and it moves away from 0 when
    the model calls one image better.

    **Read `lean` against 0.** A `lean` near 0 says the abstention hid no
    direction: the model had no held-back opinion to give. A `lean` far from 0
    says the model held an opinion and declined to write it.
    """)
    return


@app.cell
def _(A):
    flow = A.abstention_flow_table()
    flow.drop(columns=["canon_case"])
    return (flow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. How far the abstention moved the score of a PLACE

    `delta` in section 1 is a paired mean, and a mean cannot see this arm. A
    pairwise score is antisymmetric — each comparison adds $+s$ to one place
    and $-s$ to the other — thus the mean over places stays near 0 whatever
    the prompt does. A prompt that squeezes the scores toward the middle moves
    NO mean.

    So read the change rank by rank instead. Sort the place scores of each
    arm. `d(tau)` is the difference of the 2 sorted lists at the same rank.

    | Column | Definition | Reading |
    |--------|------------|---------|
    | `spread` | the ratio of the standard deviations, arm over canonical | 1.00 = no change; 0.73 = a 27% compression |
    | `rank tau` | the Kendall tau-b between the 2 arms' place scores | 1.00 = the order of the places held |
    | `d(0.05)` | the arm minus the canonical at the 5th percentile | How far the bottom of the scale moved |
    | `d(0.25)` | the same at the first quartile | |
    | `d(0.5)` | the same at the median | The middle of the scale |
    | `d(0.75)` | the same at the third quartile | |
    | `d(0.95)` | the same at the 95th percentile | How far the top of the scale moved |
    | `layout conf.` | the image layout differs too | Read that row with care |

    A positive `d(0.05)` beside a negative `d(0.95)` says the arm pulled the 2
    ends toward the middle, and `spread` says the same in one number.

    `d(tau)` sorts each arm on its own, thus it is blind to the identity of a
    place. `rank tau` reads that other axis: whether a place kept its position
    in the order. An arm can compress the scale and keep the order, or hold
    the scale and shuffle the places. The 2 columns separate those effects.

    `spread` is the statistic that the `spread` column of section 4 reports.
    This table reads it over the PLACES, and section 4 reads it over the
    areas of a proxy join, thus the 2 tables can be read against each other.

    **The variable is a PLACE, not a comparison.** A comparison carries 1 of 5
    ordinal labels, and a quantile of 5 values is a step function. The
    continuous variable sits one level up: the mean score of a place over
    every comparison it took part in. This is the same quantity that the
    master table reads after the geography join.

    **Warning: 2 cases have no continuous score.** Road quality and street
    photography pair random street-level images, thus a place is 1 image and
    it takes about 4 comparisons. `Q.MIN_COMPARISONS` drops such a place, and
    those cases then hold no quantile. The skipped table below says so.

    Brackets hold a 95% bootstrap interval that resamples PLACES and reads
    both arms at the same place, because the 2 arms scored the same places.
    """)
    return


@app.cell
def _(Q):
    qshift = Q.shift_table()
    Q.to_display(qshift)
    return (qshift,)


@app.cell
def _(Q, mo):
    _skip = Q.skipped()
    mo.vstack([
        mo.md("**The cells the quantile table drops, and why.**"),
        _skip.drop(columns=["canon_case"]),
    ])
    return


@app.cell
def _(Q):
    qcurves = Q.delta_curves()
    Q.plot_delta_curves(qcurves)
    return (qcurves,)


@app.cell
def _(Q, qshift):
    Q.plot_headline(qshift)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Validation by proxy, run again on every ablation arm

    The master table reads the canonical arm alone. This section runs the
    SAME road from the run parquet to the 3 metrics for EACH prompt ablation,
    so every arm can be read against the same outside measurements.

    | Arm | What it removes | The question it asks |
    |-----|-----------------|----------------------|
    | no abstention | the `NotSure` label | Does the model hide its harder calls? |
    | no city name | "New York City" in the prompt | Does the place prior carry the agreement? |
    | text first | the images-first reading order | Does the reading order carry it? |

    An arm can move the agreement in either direction, and each direction
    says something different:

    | What happens | What it says |
    |--------------|--------------|
    | The arm agrees LESS | The removed thing carried real signal. |
    | The arm agrees MORE | The removed thing carried a bias. |
    | No change | The removed thing carried nothing either way. |

    **Warning: a fall does not prove the removed thing carried signal.** A
    narrower spread lowers a correlation with no change of ranking at all
    (section 3 reports the same `spread` ratio over the places, thus the 2
    tables read against each other). The `spread` column is the ratio of the
    standard deviation of the area scores, arm over canonical. Under 1 means
    the arm compressed them. **Read `d tau`
    against `spread`, never alone.**

    ### Each arm gets its OWN control

    `nocity` and `textfirst` drew 110,000 pairs for each case, and the
    canonical battery drew 1,100,000. A place thus takes about 780
    comparisons in those arms against 8,246 in the canonical run, and a
    noisier place score attenuates a correlation on its own. So the control
    of an arm is rebuilt on that arm's OWN pair set. About 92% of the pairs
    of each small arm match.

    The join key is `pair_id`, never `canonical_pair_id`. `pair_id` carries
    the presentation order, and about 8% of the shared pairs came up in the
    opposite order.

    **Warning: the match holds the UNIT pair and the A/B order. It does not
    hold the photographs.** A facility case draws a fresh image for each side
    when it builds the pairs, thus a matched `pair_id` in `nocity` or
    `textfirst` names the same 2 facilities and 2 other pictures of them.
    Measured 2026-08-31, both photographs match on 0.2% of the shared pairs
    (libraries, schools, parks and plazas), 0.6% (subway safety) and 1.7%
    (restaurants).

    | Arm | Do the photographs match? |
    |-----|---------------------------|
    | `noabstain` | Yes, 100%. It reads the same 1,100,000-pair draw. |
    | `nocity` | Almost never, outside road quality. |
    | `textfirst` | Almost never, outside road quality. |

    Road quality is an image case: there the place IS the photograph, thus it
    matches by construction. The mismatch adds NOISE to `d tau` in the 2
    small arms and it adds no direction, because the image draw does not know
    which arm it serves. Read a small `d tau` there as "no effect that
    survives the image noise", never as "no effect".

    **`d tau 95%` is a paired bootstrap interval.** It resamples AREAS and
    reads both arms at the same area, the same pattern as section 3. The
    band that `reads` uses (0.02) is a practical size, not a standard error:
    the marginal error of a Kendall tau-b over 50 areas is near 0.09, but a
    paired difference cancels most of it. Read the interval for
    significance, and read the band for size.

    **Every arm goes through `_proxy_arms.py`, never through the case
    export.** A case notebook can set its own floor, and comparing a
    recomputed arm with an exported control would read that difference as a
    prompt effect. The recomputed canonical rows reproduce all 34 exported
    numbers exactly.

    The proxy itself is an outside measurement, thus it does not move with
    the arm. Every arm joins the same proxy parquet.

    **Every gemma-4-12b row reads cleanly since 2026-09-04.** The 3 gemma
    arm halves re-ran on the layout of the canonical gemma control
    (`interleaved_labels`; `text_first_interleaved` for the text-first arm),
    thus a row holds the named change alone. The `layout conf.` column stays
    and is empty. The qwen3.5-9b half ran `images_then_text` in the control
    and `text_first` in the arm, which differ in the prompt position alone.
    """)
    return


@app.cell
def _(PA, layer_pick):
    arms_long = PA.build(layer=layer_pick.value)
    arms_drift = PA.drift(arms_long)
    PA.to_display(arms_drift)
    return arms_drift, arms_long


@app.cell
def _(PA, arms_drift, mo):
    mo.vstack([
        mo.md("**Which way did each arm go?** `d tau` over the band that "
              "counts as no change. A rater block sits under its arm."),
        PA.summary(arms_drift),
    ])
    return


@app.cell
def _(PA, layer_pick, mo):
    weighted = PA.weighted_tau_table(layer=layer_pick.value)
    mo.vstack([
        mo.md(
            "**Does the abstention enter the area score?** The area score is "
            "a plain mean over its places, thus a place that abstained often "
            "pulls the area as hard as a place that answered every time. "
            "`tau weighted` gives each place the weight of its comparison "
            "count, and a small `d tau` says the abstention does not carry "
            "the agreement."
        ),
        PA.weighted_to_display(weighted),
    ])
    return (weighted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The drift, arm by arm

    One row for each cell, one colour for each arm. A cell where the arms
    move together points at the geography or the proxy. A cell where 1 arm
    moves alone points at the prompt.
    """)
    return


@app.cell
def _(PA, arms_drift):
    PA.plot_drift(arms_drift)
    return


@app.cell
def _(PA, arms_drift):
    PA.plot_arm_slopes(arms_drift)
    return


@app.cell
def _(PA, arms_drift):
    PA.plot_spread_vs_delta(arms_drift)
    return


@app.cell
def _(PA, arms_drift, mo):
    mo.vstack([
        mo.md("**Do the arms agree with each other?** 1 row for each cell, "
              "1 column for each arm."),
        PA.drift_wide(arms_drift),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The held pairs alone

    Section 3 shows that the no-abstention arm moves places in the order
    (`rank tau` under 1), but the proxy metrics above barely move. This
    section asks the direct question: do the FORCED judgments carry the
    proxy signal?

    Split the arm's pairs by what the canonical arm did on the same pair:

    - `held` — the canonical arm wrote `NotSure`; the arm had to choose.
    - `answered` — the canonical arm gave a label.

    Both halves go down the SAME road to the proxy, with the same rater and
    the same prompt. A subset with fewer pairs gives each place a noisier
    score, and noise attenuates a correlation on its own, thus both halves
    take the size of the smaller one, with a seed-fixed draw.

    | Reading | Meaning |
    |---------|---------|
    | `tau held` ≈ `tau answered` | The abstention hid judgments with the same signal. |
    | `tau held` ≈ 0 | The forced choices are noise. The abstention was honest. |
    | In between | The held calls carry a weaker form of the signal. |

    **Warning: the 2 sides split the comparisons of a place, not the
    places.** A place with a high abstention rate puts most of its
    comparisons on the held side, thus the 2 sides can keep different
    places and different areas. The `areas` columns say what each side
    kept.
    """)
    return


@app.cell
def _(PA, layer_pick):
    held = PA.held_table(layer=layer_pick.value)
    PA.held_to_display(held)
    return (held,)


@app.cell(hide_code=True)
def _(mo):
    save = mo.ui.run_button(label="Write the ablation files")
    mo.md(f"## 5. Export\n\n{save}")
    return (save,)


@app.cell
def _(
    NOTEBOOK_VERSION,
    OUT_DIR,
    PA,
    Q,
    ablations,
    arms_drift,
    arms_long,
    flow,
    held,
    layer_pick,
    mo,
    qcurves,
    qshift,
    save,
    weighted,
):
    if save.value:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        # The ablation rows carry no layer, thus they take a stem with no
        # layer. They come from the run parquets, not from a case export.
        _files = [f"ablations_v{NOTEBOOK_VERSION}.csv",
                  f"ablation_abstention_flow_v{NOTEBOOK_VERSION}.csv"]
        ablations.to_csv(OUT_DIR / _files[0], index=False)
        flow.to_csv(OUT_DIR / _files[1], index=False)
        # The quantile shift and the proxy re-run each write their own frames,
        # LaTeX and figures. The quantile rows carry no layer; the proxy rows
        # do, thus their stem holds it.
        _q = Q.export(qshift, qcurves, OUT_DIR, version=NOTEBOOK_VERSION,
                      skipped_rows=Q.skipped())
        _a = PA.export(arms_long, arms_drift, OUT_DIR,
                       version=NOTEBOOK_VERSION, layer=layer_pick.value)
        # The robustness frame of section 4: the plain area mean beside the
        # comparison-count-weighted one, and the held split.
        _w = f"proxy_weighted_tau_{layer_pick.value}_v{NOTEBOOK_VERSION}.csv"
        weighted.to_csv(OUT_DIR / _w, index=False)
        _h = f"held_pairs_{layer_pick.value}_v{NOTEBOOK_VERSION}.csv"
        held.to_csv(OUT_DIR / _h, index=False)
        _msg = mo.md(
            f"Wrote `{'`, `'.join(_files)}`, `{_w}` and `{_h}` to "
            f"`master/outputs/`. "
            f"The quantile shift wrote {len(_q)}: `{'`, `'.join(_q)}`. "
            f"The proxy re-run wrote {len(_a)}: `{'`, `'.join(_a)}`."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
