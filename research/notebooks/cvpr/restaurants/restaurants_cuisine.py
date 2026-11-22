import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Which cuisines does the model choose?

    The restaurants case asks which of 2 storefronts the model would rather eat
    at. Every restaurant carries a DOHMH `cuisine_description`, thus a unit
    score groups by cuisine.

    **Warning: a cuisine is a property of the business, and the model sees only
    the storefront.** A high mean says the facades of that cuisine look, to this
    model, like places it would rather eat at. It says nothing about the food.

    A cuisine enters the ranking only when BOTH raters scored at least 20 of its
    restaurants. The count sits beside each name.
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
    import _cuisine as CU
    import _style as style

    C.verify_or_raise()
    style.apply_house_style()

    FIG_DIR = _here / "figures"
    return C, CU, FIG_DIR, mo, pd, style


@app.cell(hide_code=True)
def _(C, mo):
    mo.md(f"**Canonical registry:** {C.summary()}")
    return


@app.cell
def _(CU, mo):
    min_units = mo.ui.slider(5, 100, value=CU.MIN_UNITS, step=5,
                             label="Restaurants a cuisine needs, from EACH rater",
                             show_value=True)
    band_size = mo.ui.slider(3, 10, value=CU.BAND_SIZE, step=1,
                             label="Cuisines in each band", show_value=True)
    mo.hstack([min_units, band_size])
    return band_size, min_units


@app.cell
def _(CU, min_units):
    table = CU.cuisine_table(int(min_units.value))
    return (table,)


@app.cell(hide_code=True)
def _(mo, table):
    mo.md(f"**{len(table)} cuisines** pass the floor.")
    return


@app.cell
def _(CU, band_size, table):
    picked = CU.bands(table, int(band_size.value))
    picked
    return (picked,)


@app.cell
def _(CU, band_size, table):
    fig = CU.plot_bands(table, int(band_size.value))
    fig
    return (fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The whole ranking

    The bands cut 15 rows out of an ordered list. This is the list.
    """
    )
    return


@app.cell
def _(table):
    table
    return


@app.cell
def _(CU, FIG_DIR, band_size, min_units, mo):
    written = CU.export(FIG_DIR, int(min_units.value), int(band_size.value))
    mo.md("**Written:**\n\n" + "\n".join(f"- `{FIG_DIR / w}`" for w in written))
    return (written,)


if __name__ == "__main__":
    app.run()
