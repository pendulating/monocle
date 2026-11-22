import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Street photography, against income, across the city

    2 maps of the same census tracts. **Left:** the ACS median household income.
    **Right:** the mean street-photography score the model gave the images in
    that tract.

    The pair is the point. A reader sees the shared pattern before any
    correlation is quoted, and sees where the 2 maps disagree.

    ## How a tract gets its score

    Street photography is an IMAGE-mode case: a pair is 2 random citywide shots,
    and there is no unit. The camera position places each shot, which here is
    the true position, and a tract takes the mean of the images inside it.

    A tract needs at least 5 scored images. A tract under that floor keeps the
    light ground, thus a gap on the map is missing data and not a low score.

    ## Reading the colours

    | Panel | Scale | Zero |
    |-------|-------|------|
    | income | sequential, low to high | none; the bar starts at the 5% quantile |
    | score | diverging, centred on 0 | 0 means the model had no preference |

    Each panel keeps its own bar: dollars and an ordinal mean share no scale.
    Both bars are clipped at the 5% and 95% quantile, so a handful of extreme
    tracts cannot take the whole ramp.
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
    import _maps as M
    import _proxies as px
    import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    CASE = "street_photography"
    FIG_DIR = _here / "figures"
    return C, CASE, FIG_DIR, M, mo, pd, px, style


@app.cell(hide_code=True)
def _(C, mo):
    mo.md(f"**Canonical registry:** {C.summary()}")
    return


@app.cell
def _(M, mo):
    layer_pick = mo.ui.radio(options=["census_tract", "nta", "community_district"],
                            value="census_tract", label="Geography layer")
    model_pick = mo.ui.dropdown(options=["gemma-4-12b", "qwen3.5-9b"],
                               value="gemma-4-12b", label="Rater on the right panel")
    min_images = mo.ui.slider(1, 30, value=M.MIN_IMAGES, step=1,
                              label="Images a polygon needs", show_value=True)
    mo.hstack([layer_pick, model_pick, min_images])
    return layer_pick, min_images, model_pick


@app.cell
def _(CASE, M, layer_pick, min_images):
    scores = M.image_scores_by_layer(CASE, layer_pick.value, int(min_images.value))
    return (scores,)


@app.cell
def _(layer_pick, px):
    income = px.income_by_layer(layer_pick.value)
    return (income,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Coverage

    How much of the city each panel really shows. Read it before the map: a
    thin coverage makes a pattern look sharper than the data is.
    """
    )
    return


@app.cell
def _(pd, scores):
    coverage = (scores.groupby("model")
                .agg(polygons=("mean_score", "size"),
                     images=("n_images", "sum"),
                     comparisons=("comparisons", "sum"),
                     mean_score=("mean_score", "mean"))
                .round(3).reset_index())
    coverage
    return (coverage,)


@app.cell
def _(M, income, layer_pick, model_pick, scores):
    fig = M.small_multiples(
        income, "proxy_mean", "median household income (USD)",
        scores[scores.model == model_pick.value], "mean_score",
        f"mean street photography score ({model_pick.value})",
        layer=layer_pick.value,
    )
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Warning: the income panel is a median only at the tract layer

    ACS gives a median for each tract. An NTA or a community district holds
    several tracts, and no average of medians is a median. `income_by_layer`
    returns a population-weighted mean of the tract medians there, which is the
    usual approximation. Call it that, and prefer the tract layer.
    """
    )
    return


@app.cell
def _(FIG_DIR, M, layer_pick, min_images, mo, model_pick):
    written = M.export_street_photography(FIG_DIR, layer_pick.value,
                                          int(min_images.value), model_pick.value)
    mo.md("**Written:**\n\n" + "\n".join(f"- `{FIG_DIR / w}`" for w in written))
    return (written,)


if __name__ == "__main__":
    app.run()
