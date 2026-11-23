import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F8 — do the pixels explain the bias results?

    The F4 bias results are correlations between the model output and the
    NTA features (% white, income, and others). The images differ by NTA, thus
    the pixels can explain each result. F8 tests this in 4 parts (D134, D135):

    | Part | Question | Method | Verdict of the advisor |
    |------|----------|--------|------------------------|
    | F8.1 | Does the NTA effect stay when we control for the image content? | method B: OLS of the valence on 1 z-scored NTA feature, with cuisine, DOHMH grade and borough FE; SE clustered by NTA | solid |
    | F8.2 | Does a neighbourhood name change the answer when the pixels stay fixed? | the counterfactual cue: the same image 2 times, with a rich or a poor name; paired t test | usable with caveats |
    | F8.3 | Does the judgment after the answer follow the NTA? | the R3 readout (D18) after each F4 answer: Spearman over the NTAs, and method B | solid |
    | F8.4 | Where in the model do the class words come up? | the plain logit lens, gemma-4-31b, 200 answers | a figure only |

    This notebook does no test. The scripts in `research/scripts/f8/` do the tests,
    and the notebook reads their CSV files in `research/figures/f8/`. It reads the
    raw answers only for the example of 1 image (section 2.6) and for the
    NTA scatter (section 3.2). The record of each part is in
    `research/docs/`: `f8-method-b.md`, `f8-counterfactual-cue.md`,
    `f8-lens-figure.md`, and the F8 section of `facct-runbook.md`.

    **A value in bold is significant after the Bonferroni correction**
    (alpha 0.05). The correction covers all the rows of 1 family of tests.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    # `_f8` sits beside this notebook. It puts `research/notebooks/cvpr/` on the path
    # itself, thus `_style` imports after it.
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    from research.notebooks.facct import _f8 as F
    from research.notebooks.cvpr import _style as style

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    return F, NOTEBOOK_VERSION, mo, pd, plt


@app.cell
def _(F):
    R = F.load_results()
    return (R,)


@app.cell(hide_code=True)
def _(F, R, mo):
    _files = "\n".join(f"| `{k}` | {len(v):,} |" for k, v in R.items())
    mo.vstack([
        mo.md(
            "## 0. The data\n\n"
            f"The result files are in `{F.FIG.relative_to(F.REPO_ROOT)}/`, and the raw "
            f"answers and readouts in `{F.DATA.relative_to(F.REPO_ROOT)}/`. All GPU jobs "
            "ran on 1 B300 between 04:10 and 11:05 UTC on 2026-10-29. To make the "
            "F8.2 and F8.3 files again, run `bash research/scripts/f8/finalize.sh`.\n\n"
            "| Table | Rows |\n|-------|------|\n" + _files
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. F8.1 — method B: the NTA effect with image-content controls

    The unit is 1 text: 1 (model, case, field, image). The outcome is the net
    valence of the text, (good - bad) / spans, over its `visual_evidence`
    spans (the 4.12 rule). The predictor is 1 ACS feature at a time,
    z-scored over the NTAs of New York City.

    | Fit | Columns |
    |-----|---------|
    | model 0 | 1 + z(feature) |
    | model 1, restaurant | model 0 + cuisine FE + DOHMH grade FE + borough FE |
    | model 1, street | model 0 + borough FE |

    The SE is clustered by NTA (CR1). The test is the t test of the
    z(feature) coefficient with G - 1 degrees of freedom. Bonferroni covers
    100 rows (20 (model, case, field) x 5 features). `share_kept` is
    beta1 / beta0.
    """)
    return


@app.cell(hide_code=True)
def _(F, R, mo):
    _c = R["mb_cases"].set_index("case")
    _c20 = R["mb20_cases"].set_index("case")
    _t = _c[["rows", "sig0", "sig1", "sig0_and_sig1", "median_share_kept_sig0"]].round(2)
    _t.columns = ["rows", "significant, model 0", "significant, model 1", "in both, same sign",
                  "median share kept (model-0 rows)"]
    _t20 = _c20[["sig0", "sig1", "median_share_kept_sig0"]].round(2)
    _t20.columns = ["significant, model 0", "significant, model 1", "median share kept"]
    mo.md(
        "### 1.1 The totals\n\n" + F.md_table(_t) +
        "\n\nThe check with only the NTAs with 20 images or more (`valence_nta20_*`):\n\n" + F.md_table(_t20) +
        "\n\n**In restaurants, the controls remove only about 10% of the NTA effect.** 37 of the 40 "
        "model-0 effects stay significant, all with the same sign. The 3 that do not stay are Llama "
        "responses. In the street case the effects are weak before the controls (9 of 50).\n\n"
        "**Caution:** the controls are coarse. Cuisine, grade and borough do not hold the pixels "
        "fixed (the storefront, the light, the season stay free). The NTA is an ecological unit: "
        "write \"the valence shifts with the resident mix of the area\", not \"the model treats "
        "people differently\". Section 2 holds the pixels fixed."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mb_case = mo.ui.dropdown(["restaurant", "street"], value="restaurant", label="case")
    mb_field = mo.ui.dropdown(["response", "reasoning"], value="response", label="field")
    mo.vstack([
        mo.md(
            "### 1.2 The coefficients\n\nThe size is in SD of the outcome for each SD of the "
            "feature. The bars are 95% CIs (t, G - 1 df). The Llama models write no trace, thus "
            "the reasoning field has 4 models."
        ),
        mo.hstack([mb_case, mb_field], justify="start"),
    ])
    return mb_case, mb_field


@app.cell
def _(F, R, mb_case, mb_field):
    fig_mb = F.plot_method_b(R["mb_fits"], mb_case.value, mb_field.value)
    fig_mb
    return


@app.cell
def _(F, R, mb_case, mb_field, mo):
    _f = F.fits_view(R["mb_fits"])
    mo.ui.table(_f[(_f["case"] == mb_case.value) & (_f["field"] == mb_field.value)], selection=None,
                page_size=30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. F8.2 — the counterfactual cue: the same pixels, a different name

    Each image goes to the model 2 times. The prompt starts with
    `I am in {name}, {borough}. `, then the F4 prompt of the case. The 2 names
    are a rich and a poor NTA in the borough of the image. When the image's own
    NTA is in the drawn pair, the image gets the other pair of its borough.

    | Item | Value |
    |------|-------|
    | Images | the canonical F4 batch: 1,000 restaurant images, 985 street images |
    | Models | gemma-4-12b, qwen3.5-9b, gemma-4-31b, qwen3.8-27b |
    | Sampling | the F4 values, thinking on, the same seed for both arms; token limit 8,192 |
    | Readout | R3 after the answer: 4 readouts (below) |
    | Baseline | R3 after the F4 answer (no cue) of the same image |
    | Test | paired t test, rich minus poor, for each (model, case, readout); Bonferroni over 32 rows |

    The 4 readouts. Each is a difference of 2 raw log-probs (nats) at the
    first token of the follow-up turn: "Is this a cool place to eat at?"
    (restaurant) or "... to take a photo in?" (street).

    | Readout | Value |
    |---------|-------|
    | `c_yes` | logp(`Yes`) - logp(`No`) |
    | `c_cool` | logp(` cool`) - logp(` lame`), after the prefix "This place is" |
    | `c_yes_i` | `c_yes`, with " Answer with one word: yes or no." after the question |
    | `c_cool_i` | `c_cool`, with ' Answer with "This place is cool." or "This place is lame."' |

    **Use the instructed readouts (`_i`) for the main text.** With the
    verbatim question, `Yes` is at rank 12 to 47 and ` lame` at rank
    20,000 to 80,000, thus `c_yes` and `c_cool` compare 2 improbable
    tokens. Claude added the instructed form after the smoke job and before
    any full result.
    """)
    return


@app.cell(hide_code=True)
def _(F, mo, pd):
    _p = pd.concat([F.cue_design(c).assign(case=c) for c in F.CASES])
    _p = (_p.groupby(["boro", "rich_name", "poor_name", "case"]).size().unstack("case", fill_value=0)
          .reset_index().rename(columns={"boro": "borough", "rich_name": "rich name", "poor_name": "poor name"}))
    mo.md(
        "### 2.1 The name pairs and the images in each pair\n\n" + F.md_table(_p.set_index("borough")) +
        "\n\n**Caution:** in 9 of 10 pairs the rich name is also the whiter NTA. Some names also "
        "carry fame (Tribeca, Long Island City). Thus F8.2 cannot separate class from race or from fame."
    )
    return


@app.cell(hide_code=True)
def _(F, R, mo):
    _cut = R["cut"].assign(arm_case=lambda d: d["case"] + ", " + d["arm"])
    _cut = _cut.pivot_table(index="model", columns="arm_case", values="cut").astype(int)
    _echo = R["echo"].assign(col=lambda d: d["case"] + ", " + d["field"] + ", " + d["arm"])
    _echo = _echo.pivot_table(index="model", columns="col", values="says_cue_name").map(lambda v: f"{v:.0%}")
    _n = R["primary"].pivot_table(index="model", columns="case", values="n", aggfunc="first")
    mo.md(
        "### 2.2 The checks\n\nThe answers that stopped at 8,192 tokens. A paired test drops an "
        "image when 1 of its 2 answers stops:\n\n" + F.md_table(_cut) +
        "\n\nThe pairs that enter the tests:\n\n" + F.md_table(_n) +
        "\n\nEach model stops more rich-arm restaurant answers. Thus the pairs are not a random "
        "subset (strongest for qwen3.5-9b). The advisor check: an unpaired Welch test on all the "
        "answers that did not stop gives the same signs.\n\n"
        "The share of answers that say the cue name:\n\n" + F.md_table(_echo)
    )
    return


@app.cell(hide_code=True)
def _(F, R, mo):
    mo.md(
        "### 2.3 The primary tests (32, Bonferroni)\n\nThe mean of rich minus poor over the image "
        "pairs, in nats.\n\n" + F.md_table(F.primary_wide(R["primary"])) +
        "\n\n16 of 32 tests are significant (restaurant 7, street 9).\n\n"
        "- **At fixed pixels, the name changes the answer and the readout after it.** 3 of 4 models "
        "change a readout. qwen3.5-9b changes almost nothing.\n"
        "- **The restaurant direction differs by model.** Up for gemma-4-12b and qwen3.8-27b, down "
        "for gemma-4-31b (`c_cool_i` -1.39), null for qwen3.5-9b.\n"
        "- **Street photography: a poor name makes the place \"cooler\" for the 2 gemma models.** "
        "gemma-4-31b `c_cool_i` -5.27; the rich arm is higher on only 30% of the images.\n"
        "- The estimand is the effect of the name on the answer and the readout together: each arm "
        "gets a new answer."
    )
    return


@app.cell
def _(F, R):
    fig_primary = F.plot_primary(R["primary"])
    fig_primary
    return


@app.cell(hide_code=True)
def _(F, mo):
    base_readout = mo.ui.dropdown(list(F.READOUTS), value="c_yes_i", label="readout")
    mo.vstack([
        mo.md(
            "### 2.4 Any name moves the readout\n\nThe no-cue mean (the F4 answer of the same images), "
            "the poor-arm mean and the rich-arm mean. **The shift from no cue to any name can be "
            "larger than the gap between rich and poor.** Example: gemma-4-31b restaurant "
            "`c_yes_i`: no cue -0.92, poor +4.18, rich +4.61. The F4 answer used a 4,096-token "
            "limit, thus this comparison is not a test."
        ),
        base_readout,
    ])
    return (base_readout,)


@app.cell
def _(F, R, base_readout):
    fig_base = F.plot_baseline(R["primary"], base_readout.value)
    fig_base
    return


@app.cell(hide_code=True)
def _(F, mo):
    pair_case = mo.ui.dropdown(list(F.CASES), value="restaurant", label="case")
    pair_readout = mo.ui.dropdown(list(F.READOUTS), value="c_cool_i", label="readout")
    mo.vstack([
        mo.md(
            "### 2.5 The effect for each name pair (no test)\n\nThe per-pair means differ in sign. "
            "For gemma-4-12b restaurants, Park Slope against Brownsville gives +2.65 (`c_cool_i`), "
            "and Douglaston against Jamaica gives -1.09. A pair holds 40 to 160 images."
        ),
        mo.hstack([pair_case, pair_readout], justify="start"),
    ])
    return pair_case, pair_readout


@app.cell
def _(F, R, pair_case, pair_readout):
    fig_pairs = F.plot_pairs(R["by_pair"], pair_case.value, pair_readout.value)
    fig_pairs
    return


@app.cell(hide_code=True)
def _(F, mo):
    ex_model = mo.ui.dropdown(list(F.CUE_MODELS), value="gemma-4-31b", label="model")
    ex_case = mo.ui.dropdown(list(F.CASES), value="street", label="case")
    ex_readout = mo.ui.dropdown(list(F.READOUTS), value="c_cool_i", label="readout")
    ex_image = mo.ui.number(start=0, stop=999, value=970, label="image_idx")
    mo.vstack([
        mo.md(
            "### 2.6 The 2 answers of 1 image\n\nThe default is the example of the record: gemma-4-31b, "
            "street image 970. With \"Brooklyn Heights\" the answer calls the scene \"fairly "
            "cluttered\" (`c_cool_i` -20.5); with \"East New York\" it is an \"Urban Oasis\" "
            "(`c_cool_i` +31.3). The table under the answers lists the images with the largest gap."
        ),
        mo.hstack([ex_model, ex_case, ex_readout, ex_image], justify="start"),
    ])
    return ex_case, ex_image, ex_model, ex_readout


@app.cell
def _(F, ex_case, ex_image, ex_model, ex_readout, mo):
    def _arm(row, arm, name):
        _v = row[ex_readout.value]
        _val = "no readout (the answer stopped)" if _v != _v else f"{_v:+.2f}"
        return mo.vstack([
            mo.md(f"#### {arm} arm: \"{name}\"\n\n`{ex_readout.value}` = **{_val}**, "
                  f"finish `{row['finish_reason']}`, {int(row['completion_tokens']):,} tokens\n\n"
                  f"> {row['prompt']}"),
            mo.accordion({"reasoning": mo.md(str(row["reasoning"] or ""))}),
            mo.md(str(row["response"] or "")),
        ])

    try:
        _ex = F.cue_example(ex_model.value, ex_case.value, int(ex_image.value))
    except KeyError:
        _view = mo.md(f"_Image {ex_image.value} is not in the {ex_case.value} sample._")
    else:
        _d = _ex["design"]
        _none = _ex["rich"][f"none_{ex_readout.value}"]
        _view = mo.vstack([
            mo.hstack([
                mo.image(src=str(_d["image_path"]), width=360),
                mo.md(f"**{ex_case.value} image {ex_image.value}** (`{_d['sample_id']}`)\n\n"
                      f"Own NTA: {_d['nta_name']}, {_d['boro']}\n\n"
                      f"No-cue `{ex_readout.value}` (F4 answer): {_none:+.2f}"),
            ], justify="start"),
            mo.hstack([_arm(_ex["rich"], "rich", _d["rich_name"]), _arm(_ex["poor"], "poor", _d["poor_name"])],
                      widths="equal", align="start"),
        ])
    _view
    return


@app.cell
def _(F, ex_case, ex_model, ex_readout, mo):
    mo.ui.table(F.cue_gaps(ex_model.value, ex_case.value, ex_readout.value).round(2), selection=None,
                page_size=10, label="The images with the largest gap, rich minus poor")
    return


@app.cell(hide_code=True)
def _(F, mo):
    words_field = mo.ui.dropdown(["response", "reasoning"], value="response", label="field")
    mo.vstack([
        mo.md(
            "### 2.7 The word lists (64 tests, Bonferroni)\n\nThe share of answers with 1 word of "
            "the list or more. The test is the exact McNemar test on the discordant pairs.\n\n"
            "| List | Words |\n|------|-------|\n"
            "| `upscale` | upscale, high-end, upmarket, luxury, luxurious, elegant, sophisticated, refined, "
            "chic, pricey, expensive, exclusive, posh, fancy, swanky |\n"
            "| `cheap` | cheap, affordable, budget, inexpensive, bargain, no-frills, low-cost, "
            "hole-in-the-wall, cheap eats |\n"
            "| `danger` | unsafe, dangerous, sketchy, rough, gritty, seedy, crime, run-down, rundown, "
            "dilapidated, neglected |\n"
            "| `safe` | safe, secure, peaceful, quiet, tranquil |\n\n"
            "- **With a rich name, all 4 models use more `upscale` words on the same images.** About "
            "one third of these hits (21% to 37%) sit within 60 characters of an area word, thus "
            "they describe the named area, not the image.\n"
            "- With a poor name, 2 models write the `danger` list more often in the street reasoning. "
            "\"gritty\" drives it, then \"rough\". These are also photography words. **Do not call "
            "this result \"danger\".**\n"
            "- No `cheap` or `safe` row is significant."
        ),
        words_field,
    ])
    return (words_field,)


@app.cell
def _(F, R, words_field):
    fig_words = F.plot_words(R["words"], words_field.value)
    fig_words
    return


@app.cell
def _(R, mo, words_field):
    _w = R["words"][R["words"]["field"] == words_field.value].round(4)
    mo.ui.table(_w, selection=None, page_size=32)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. F8.3 — the R3 readout on the F4 pool

    The same 4 readouts, after each F4 answer of the pool (the canonical batch
    and batch 2; no cue). The pool has about 20,000 images for each model:
    10,000 restaurant and 10,000 street. 2 tests:

    | Test | Unit | Correction |
    |------|------|------------|
    | Spearman of the NTA mean of the readout with each ACS feature | 1 NTA with 20 images or more | Bonferroni over 160 rows |
    | method B (section 1) with the readout as the outcome | 1 image | Bonferroni over the 40 rows of each readout |
    """)
    return


@app.cell(hide_code=True)
def _(F, R, mo):
    _sp = R["spearman"]
    _cnt = _sp.groupby(["case", "readout"])["significant"].sum().unstack("readout")[list(F.READOUTS)]
    mo.md(
        "### 3.1 The NTA test\n\nThe significant rows of 20 for each (case, readout):\n\n" + F.md_table(_cnt) +
        "\n\nThe NTA mean of `c_cool_i` against the features, restaurants:\n\n" +
        F.md_table(F.spearman_wide(_sp, "restaurant", "c_cool_i")) +
        "\n\n**In restaurants, the judgment rises with % white and % bachelor's in all 4 models** "
        "(rho +0.40 to +0.80). The street rows are weak. The verbatim `c_cool` behaves differently "
        "in some rows (for example qwen3.5-9b: rho -0.62 with % white); section 2 gives the reason."
    )
    return


@app.cell(hide_code=True)
def _(F, mo):
    sp_case = mo.ui.dropdown(list(F.CASES), value="restaurant", label="case")
    sp_case
    return (sp_case,)


@app.cell
def _(F, R, sp_case):
    fig_sp = F.plot_spearman(R["spearman"], sp_case.value)
    fig_sp
    return


@app.cell(hide_code=True)
def _(F, mo):
    sc_case = mo.ui.dropdown(list(F.CASES), value="restaurant", label="case")
    sc_readout = mo.ui.dropdown(list(F.READOUTS), value="c_cool_i", label="readout")
    sc_feature = mo.ui.dropdown(list(F.FEATURES), value="pct_white_nh", label="feature")
    mo.vstack([
        mo.md("### 3.2 The NTA means\n\n1 dot is 1 NTA. The rho in the title comes from "
              "`r3_nta_spearman.csv`."),
        mo.hstack([sc_case, sc_readout, sc_feature], justify="start"),
    ])
    return sc_case, sc_feature, sc_readout


@app.cell
def _(F, R, sc_case, sc_feature, sc_readout):
    fig_sc = F.plot_nta_scatter(sc_case.value, sc_readout.value, sc_feature.value, R["spearman"])
    fig_sc
    return


@app.cell(hide_code=True)
def _(F, R, mo):
    _f = R["r3_fits"]
    _t = _f.groupby("readout")[["sig0", "sig1"]].sum().reindex(list(F.READOUTS)).astype(int)
    _t.columns = ["significant, model 0 (of 40)", "significant, model 1 (of 40)"]
    _r = _f[(_f["readout"] == "c_cool_i") & (_f["case"] == "restaurant") & (_f["feature"] != "pct_asian_nh")]
    _r = _r.assign(size=_r["beta1"] / _r["outcome_sd"])
    mo.vstack([
        mo.md(
            "### 3.3 Method B with the readout as the outcome\n\nThe controls of section 1 "
            "(restaurant: cuisine, grade, borough; street: borough).\n\n" + F.md_table(_t) +
            f"\n\n**For `c_cool_i` in restaurants, {int(_r['sig1'].sum())} of {len(_r)} rows of % white, "
            "% Black, % bachelor's and income stay significant with the controls, in all 4 models.** "
            f"The size in model 1 is {_r['size'].abs().min():.2f} to {_r['size'].abs().max():.2f} SD of the "
            "readout for each SD of the feature. These effects are larger than the valence effects of "
            "F8.1 (0.04 to 0.18)."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(F, mo):
    r3_readout = mo.ui.dropdown(list(F.READOUTS), value="c_cool_i", label="readout")
    r3_case = mo.ui.dropdown(list(F.CASES), value="restaurant", label="case")
    mo.hstack([r3_readout, r3_case], justify="start")
    return r3_case, r3_readout


@app.cell
def _(F, R, r3_case, r3_readout):
    _f = R["r3_fits"][R["r3_fits"]["readout"] == r3_readout.value]
    fig_r3 = F.plot_method_b(_f, r3_case.value, "answer", F.CUE_MODELS)
    fig_r3
    return


@app.cell(hide_code=True)
def _(F, R, mo):
    _p = R["pearson"].assign(pair=lambda d: d["model_a"] + " / " + d["model_b"])
    _p = _p.pivot_table(index="pair", columns=["case", "readout"], values="r").round(2)
    _p.columns = [f"{c}, {r}" for c, r in _p.columns]
    mo.md(
        "### 3.4 The models agree\n\nThe Pearson r of each readout between 2 models, over the same "
        "images. The instructed readouts agree: 0.45 to 0.72 (`c_yes_i`) and 0.50 to 0.74 "
        "(`c_cool_i`). The verbatim `c_cool` agrees less (-0.23 to 0.36).\n\n" + F.md_table(_p)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. F8.4 — the logit lens (a figure only)

    **Caution:** this is a figure, not a harm claim (D135). The plain logit
    lens shows which tokens a layer puts near the top. It does not show why.

    gemma-4-31b replays 200 stored F4 restaurant answers: 100 from the 2
    lowest-income NTAs (Mott Haven-Port Morris, Belmont) and 100 from 2 NTAs
    at the income topcode (Tribeca-Civic Center, Upper East Side-Carnegie
    Hill). For each position and each of the 60 layers, the lens gives the
    top 5 tokens. A rate is the share of positions with a token of the list
    in the top 5. `cities` (6 cities of the whatami study) is the control.

    What the figure shows:

    1. In layers 0 to 38 the list words are almost never in the top 5. The
       plain lens does not read these layers well for this model.
    2. The list words come up at about layer 50 (and a short peak at 39-41).
    3. In the late layers, `upscale` is more frequent in the high group and
       `cheap` in the low group, **the same direction as the written text**.
       The latent rates follow the said rates: the lens adds little beyond
       the text.
    4. The control is not 0: the high-group answers say Paris and London.

    Each group holds only 2 NTAs, thus the figure does not separate the
    pixels from the neighbourhood.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lens_measure = mo.ui.dropdown(["rate_top5", "rate_top1", "rate_top5_not_said"], value="rate_top5",
                                  label="measure")
    lens_first = mo.ui.checkbox(value=False, label="add the first tokens of the 9 multi-token words")
    mo.hstack([lens_measure, lens_first], justify="start")
    return lens_first, lens_measure


@app.cell
def _(F, R, lens_first, lens_measure):
    fig_lens = F.plot_lens(R["lens_rates"], measure=lens_measure.value, suffix="+first" if lens_first.value else "")
    fig_lens
    return


@app.cell(hide_code=True)
def _(F, R, mo):
    _b = R["lens_bands"]
    _b = _b[(_b["band"] == "late") & _b["list"].isin(["upscale", "cheap", "danger", "safe", "cities"])]
    _b = _b.assign(top5=lambda d: (d["rate_top5"] * 1000).map("{:.2f}".format) + " ("
                   + (d["ci_lo"] * 1000).map("{:.2f}".format) + "-" + (d["ci_hi"] * 1000).map("{:.2f}".format) + ")",
                   said=lambda d: (d["said"] * 1000).map("{:.2f}".format))
    _t = _b.pivot_table(index=["field", "list"], columns="group", values=["top5", "said"], aggfunc="first")
    _t.columns = [f"{g}: {m}" for m, g in _t.columns]
    _t = _t[["high: top5", "high: said", "low: top5", "low: said"]]
    _t.index = [f"{f}, {l}" for f, l in _t.index]
    mo.md("The late band (layers 39-58), per 1,000 positions, with the 95% bootstrap interval:\n\n"
          + F.md_table(_t))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What F8 shows, and what it does not show

    | Claim | Status |
    |-------|--------|
    | The restaurant valence follows the NTA after the image-content controls (F8.1) | shown: 37 of 40, about 90% of each effect stays; the effects are small (0.04 to 0.18 SD) |
    | A neighbourhood name changes the description and the judgment of the same image (F8.2) | shown for 4 of 4 models (words) and 3 of 4 (readouts) |
    | A rich name gives more "upscale" words (F8.2) | shown, all 4 models; about one third describe the area |
    | A poor name gives more "gritty" and "rough" in the street reasoning (F8.2) | shown, 2 models; these are also photography words |
    | The direction of the judgment is "poor is worse" (F8.2) | **not** shown; for street photography the gemma models rate a poor-name place as cooler |
    | The restaurant judgment rises with % white and % bachelor's (F8.3) | shown, all 4 models, with the controls (0.07 to 0.32 SD) |
    | The model holds a hidden class judgment (F8.4) | **not** shown; the lens follows the text |

    The caveats for the paper:

    - The cue pairs differ in more than income: in 9 of 10 pairs the rich name
      is also the whiter NTA, and some names carry fame.
    - Any name moves the readout from the no-cue baseline, and that shift can
      be larger than the rich-poor gap.
    - More rich-arm answers stop at the token limit; the advisor check shows
      that this does not change the signs.
    - F8.1 and F8.3 are correlations with an ecological unit. The p values
      can be too small because neighbour NTAs are alike.
    - The engine is not fully deterministic: the same token differs by an SD
      of 0.005 to 0.11 nats between 2 requests. The smallest significant rows
      are at this scale.
    """)
    return


@app.cell(hide_code=True)
def _(F, mo):
    write_btn = mo.ui.run_button(label="Write the figures")
    mo.vstack([
        mo.md(f"## 6. The figures\n\nThe button writes the default view of each figure as PDF and PNG, with a JSON file "
              f"(title, caption, LaTeX label, tags) to `{F.OUT.relative_to(F.REPO_ROOT)}/`."),
        write_btn,
    ])
    return (write_btn,)


@app.cell
def _(F, NOTEBOOK_VERSION, R, mo, plt, write_btn):
    if write_btn.value:
        _figs = {
            "f8_method_b_restaurant_response": lambda: F.plot_method_b(R["mb_fits"], "restaurant", "response"),
            "f8_method_b_street_response": lambda: F.plot_method_b(R["mb_fits"], "street", "response"),
            "f8_cue_primary": lambda: F.plot_primary(R["primary"]),
            "f8_cue_primary_instructed": lambda: F.plot_primary(R["primary"], ("c_yes_i", "c_cool_i")),
            "f8_cue_baseline_c_yes_i": lambda: F.plot_baseline(R["primary"], "c_yes_i"),
            "f8_cue_pairs_restaurant_c_cool_i": lambda: F.plot_pairs(R["by_pair"], "restaurant", "c_cool_i"),
            "f8_cue_pairs_street_c_cool_i": lambda: F.plot_pairs(R["by_pair"], "street", "c_cool_i"),
            "f8_cue_words_response": lambda: F.plot_words(R["words"], "response"),
            "f8_cue_words_reasoning": lambda: F.plot_words(R["words"], "reasoning"),
            "f8_r3_spearman_restaurant": lambda: F.plot_spearman(R["spearman"], "restaurant"),
            "f8_r3_spearman_street": lambda: F.plot_spearman(R["spearman"], "street"),
            "f8_r3_nta_c_cool_i_white": lambda: F.plot_nta_scatter("restaurant", "c_cool_i", "pct_white_nh",
                                                                  R["spearman"]),
            "f8_r3_method_b_c_cool_i": lambda: F.plot_method_b(
                R["r3_fits"][R["r3_fits"]["readout"] == "c_cool_i"], "restaurant", "answer", F.CUE_MODELS),
            "f8_lens_rates": lambda: F.plot_lens(R["lens_rates"]),
        }
        _written = []
        for _name, _make in _figs.items():
            _fig = _make()
            _written += F.save(_fig, _name)
            plt.close(_fig)
        _msg = mo.md(
            f"Wrote {len(_written)} files to `{F.OUT.relative_to(F.REPO_ROOT)}/`. "
            f"Notebook {NOTEBOOK_VERSION}, `_f8` {F.__version__}."
        )
    else:
        _msg = mo.md("_Press the button to write the files._")
    _msg
    return


if __name__ == "__main__":
    app.run()
