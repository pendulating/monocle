import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # F11 — does a model judge a cue the same in the 2 paradigms?

    The pairwise battery asks `Which restaurant would you rather eat at?` over
    2 images. F9 asks `Does this look like a nice restaurant to eat at?` over
    1 image. If a model applies 1 standard, then it gives a cue the same
    valence in both paradigms (H11; `facct-runbook.md`, F11, D150, D151).

    | Arm | Texts | Extractor |
    |-----|-------|-----------|
    | pairwise | the thinking traces of the canonical trace runs, 10,000 pairs | cross-family: Qwen3.8-27B reads gemma, Gemma-4-31B reads qwen |
    | F9 | the `reasoning` field, 10,000 images | DeepSeek-V4-Flash, the F9 schema |

    The measure is the Jensen-Shannon divergence (JS) above a permutation
    null. A value of 0 says that the 2 arms differ no more than 2 random
    splits of the pooled texts. A value alone has no meaning, thus each
    panel shows the anchors:

    | Anchor | What changes | Same texts? |
    |--------|--------------|-------------|
    | R1 | the extractor, inside the pairwise schema | yes |
    | R4 | the extractor, DeepSeek-V4-Flash against the primary (when its job is done) | yes |
    | R2 | the prompt: F4 description against F9 judgment, 1 extractor | same images |
    | R3 | nothing: the same pair again | same pairs |

    The computation runs in `_paradigm_kl.py` (about 10 min on the CPU):
    `python research/notebooks/facct/_paradigm_kl.py`. This notebook reads its output
    in `data/f11/`.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import marimo as mo
    import pandas as pd

    from research.notebooks.facct import _paradigm_kl as K
    from research.notebooks.cvpr import _style as style

    style.apply_house_style()

    NOTEBOOK_VERSION = "1.0.0"
    return K, NOTEBOOK_VERSION, mo, pd


@app.cell
def _(K, pd):
    div = pd.read_parquet(K.RESULTS / "f11_divergence.parquet")
    cues = pd.read_parquet(K.RESULTS / "f11_cues.parquet")
    arms = pd.read_parquet(K.RESULTS / "f11_arms.parquet")
    places = {m: pd.read_parquet(K.RESULTS / f"f11_places_{s}.parquet") for m, s in K.MODELS.items()}
    return arms, cues, div, places


@app.cell(hide_code=True)
def _(arms, mo):
    mo.vstack([
        mo.md(
            "## 0. The arms\n\n"
            "1 pairwise text is 1 side of 1 pair: the spans that the extractor gives to 1 image. "
            "`clusters` is the bootstrap unit (the pair or the image). A pairwise span is a phrase "
            "and an F9 span is a clause, thus the unit of U3 is 1 word or 2 adjacent words."
        ),
        mo.ui.table(arms.round(3), selection=None, page_size=30),
    ])
    return


@app.cell(hide_code=True)
def _(K, div, mo):
    fig_div = K.plot_divergence(div)
    mo.vstack([
        mo.md(
            "## 1. The divergence and its anchors\n\n"
            "The filled dot is the test. H11 holds for a cell when its interval overlaps the "
            "extractor anchor (R4 when it exists, else R1)."
        ),
        fig_div,
    ])
    return (fig_div,)


@app.cell(hide_code=True)
def _(div, mo):
    _t = div[~div.comparison.str.startswith("R0")].copy()
    _t = _t[["model", "case", "unit", "comparison", "raked", "js", "js_null", "js_excess",
             "js_excess_lo", "js_excess_hi", "kl_ab", "kl_ba", "rho", "rho_lo", "rho_hi",
             "clusters_a", "clusters_b", "keys"]].round(4)
    mo.vstack([
        mo.md(
            "### The table\n\n"
            "`kl_ab` is KL(a || b) and `kl_ba` is KL(b || a), in bits; for the paradigm row a is "
            "the pairwise arm. `raked` sets the valence margins of the 2 arms equal first; it "
            "separates the level of the valence from its distribution over the keys. `rho` is the "
            "Spearman correlation of the net valence of the U3 cues."
        ),
        mo.ui.table(_t, selection=None, page_size=40),
    ])
    return


@app.cell(hide_code=True)
def _(K, cues, mo):
    fig_cues = K.plot_cues(cues)
    mo.vstack([
        mo.md(
            "## 2. The cues\n\n"
            "The order of the cues holds (rho 0.7 to 0.9), but the level moves. Read the words "
            "off the diagonal: a pairwise trace uses a word to compare (`cleaner`, `darker`), "
            "and the F9 trace uses it to describe."
        ),
        fig_cues,
    ])
    return (fig_cues,)


@app.cell(hide_code=True)
def _(cues, mo):
    mo.vstack([
        mo.md("### The cues with the largest JS, and the cues that change sign"),
        mo.ui.table(cues.round(3), selection=None, page_size=25),
    ])
    return


@app.cell(hide_code=True)
def _(K, mo, pd, places):
    fig_places = K.plot_places(places)
    _rows = []
    for _m, _t in places.items():
        for _x, _y in [("battery_score", "f9_score"), ("battery_score", "net_f9"),
                       ("net_pairwise", "net_f9"), ("trace_score", "f9_score")]:
            _rows.append({"model": _m, **K.place_rho(_t, _x, _y, n_boot=1000)})
    mo.vstack([
        mo.md(
            "## 3. The same restaurant (no KL)\n\n"
            "The trace run has too few comparisons for 1 restaurant: 31 restaurants have 5 "
            "comparisons or more. Thus the pairwise score is the 1,000,000-pair battery (no "
            "thinking). **Caution:** the 2 paradigms show different views of the restaurant."
        ),
        fig_places,
        mo.ui.table(pd.DataFrame(_rows).round(3), selection=None),
    ])
    return (fig_places,)


@app.cell
def _(mo):
    save = mo.ui.run_button(label="Write the figures to figures/f11/")
    save
    return (save,)


@app.cell
def _(K, fig_cues, fig_div, fig_places, mo, save):
    if save.value or "write" in mo.cli_args():
        _paths = []
        for _fig, _name in [(fig_div, "f11_divergence"), (fig_cues, "f11_cues"),
                            (fig_places, "f11_places")]:
            _paths += K.save(_fig, _name)
        print("\n".join(str(p) for p in _paths))
    return


if __name__ == "__main__":
    app.run()
