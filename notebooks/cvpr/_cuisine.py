"""Which cuisines does the model choose?

The restaurants case asks which of 2 storefronts the model would rather eat at.
Every restaurant carries a DOHMH `cuisine_description`, thus a unit score can be
grouped by cuisine, and the question becomes: does the preference for a facade
line up with a kind of food?

Warning: a cuisine is a property of the BUSINESS, and the model sees only the
storefront. A high mean is not a statement about the food. It says the facades
of that cuisine look, to this model, like places it would rather eat at.

Warning: a small cuisine is noise. `MIN_UNITS` keeps a cuisine out unless both
raters scored that many restaurants of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _provenance as prov
import _style as S

__version__ = "1.0.0"

CASE = "restaurants"
UNIT_TABLE = ("/share/pierson/matt/mllmsci/curation/dohmh_restaurants_inspected_all"
              "/restaurants_aggregated.parquet")

# A cuisine needs this many scored restaurants from EACH rater. Below it a mean
# rests on a handful of storefronts.
MIN_UNITS = 20

# How many cuisines each band holds.
BAND_SIZE = 5

MODEL_LABEL = {"gemma-4-12b/instruct": "gemma-4-12b",
               "qwen3.5-9b/instruct": "qwen3.5-9b"}


def unit_cuisines(path: str = UNIT_TABLE) -> pd.DataFrame:
    """Read `unit_uid` and its cuisine from the DOHMH curation table."""
    df = pd.read_parquet(path, columns=["uid", "cuisine_description"])
    return (df.rename(columns={"uid": "unit_uid", "cuisine_description": "cuisine"})
              .dropna(subset=["cuisine"])
              .drop_duplicates("unit_uid"))


def cuisine_table(min_units: int = MIN_UNITS,
                  records: Optional[Sequence] = None) -> pd.DataFrame:
    """Give each cuisine a mean score for each canonical rater.

    Returns 1 row for each cuisine: the units, the mean of each model, and the
    pooled mean that the ranking uses.
    """
    records = list(records or prov.discover_runs(CASE))
    if not records:
        raise FileNotFoundError("no canonical restaurants run in the registry")

    frames = []
    for rec in records:
        scores = prov.score_units(prov.load_run(rec))
        scores["model"] = MODEL_LABEL.get(rec.model, rec.model)
        frames.append(scores)
    scored = pd.concat(frames, ignore_index=True)

    joined = scored.merge(unit_cuisines(), on="unit_uid", how="inner")
    long = (joined.groupby(["cuisine", "model"])
                  .agg(units=("unit_uid", "nunique"),
                       mean_score=("mean_score", "mean"))
                  .reset_index())

    wide = long.pivot(index="cuisine", columns="model", values=["units", "mean_score"])
    models = sorted({m for m in long["model"]})
    out = pd.DataFrame(index=wide.index)
    for m in models:
        out[f"units_{m}"] = wide[("units", m)]
        out[f"score_{m}"] = wide[("mean_score", m)]
    # The floor applies to the SMALLER rater, so no row rests on 1 model alone.
    out["units"] = out[[f"units_{m}" for m in models]].min(axis=1)
    out["pooled"] = out[[f"score_{m}" for m in models]].mean(axis=1)
    out = out[out["units"] >= min_units]
    return out.sort_values("pooled", ascending=False).reset_index()


def bands(table: pd.DataFrame, size: int = BAND_SIZE) -> pd.DataFrame:
    """Take the top, the middle, and the bottom `size` cuisines.

    The middle band comes from the CENTRE of the ranking, not from a sample, so
    the 3 bands read as 1 ordered list with the rest cut out.
    """
    n = len(table)
    if n < 3 * size:
        raise ValueError(f"only {n} cuisines pass the floor; need {3 * size}")
    mid = (n - size) // 2
    out = pd.concat([
        table.iloc[:size].assign(band="top"),
        table.iloc[mid:mid + size].assign(band="middle"),
        table.iloc[-size:].assign(band="bottom"),
    ], ignore_index=True)
    out["rank"] = [*range(1, size + 1), *range(mid + 1, mid + size + 1),
                   *range(n - size + 1, n + 1)]
    return out


def plot_bands(table: pd.DataFrame, size: int = BAND_SIZE):
    """3 panels, 1 for each band, sharing the score axis.

    The bar is the pooled mean. The 2 dots are the raters, so a reader can see
    at once whether the 2 agree about a cuisine.
    """
    import matplotlib.pyplot as plt

    picked = bands(table, size)
    models = [c[len("score_"):] for c in table.columns if c.startswith("score_")]
    colours = {m: S.PAL[c] for m, c in zip(models, ("blue", "coral"))}
    band_label = {"top": f"top {size}", "middle": f"middle {size}",
                  "bottom": f"bottom {size}"}

    # No tight_layout here: a 3-panel figure with long y labels and a figure
    # legend makes it warn and guess. The margins are set once, by hand.
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 5.4), sharex=True)
    for ax, band in zip(axes, ("top", "middle", "bottom")):
        part = picked[picked["band"] == band].iloc[::-1]
        y = np.arange(len(part))
        ax.barh(y, part["pooled"], color=S.PAL["slate"], height=0.62, zorder=2)
        for m in models:
            ax.scatter(part[f"score_{m}"], y, s=14, zorder=3, color=colours[m],
                       label=m, edgecolor=S.EDGE, linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  ({int(n)})" for c, n in
                            zip(part["cuisine"], part["units"])])
        ax.axvline(0, color=S.EDGE, linewidth=0.7, zorder=1)
        ax.grid(axis="y", visible=False)
        # The band name sits on the axis, not in a title: the caption of the
        # paper owns the title.
        ax.set_ylabel(band_label[band])
    axes[-1].set_xlabel("mean ordinal score of the cuisine")
    handles, labels = axes[0].get_legend_handles_labels()
    # tight_layout first, then reserve the strip the legend sits in. Passing a
    # rect AND a figure legend makes matplotlib warn that the axes are not
    # compatible with tight_layout.
    fig.subplots_adjust(left=0.38, right=0.98, top=0.98, bottom=0.14, hspace=0.30)
    fig.legend(handles[:len(models)], labels[:len(models)], ncol=len(models),
               loc="lower center", frameon=False)
    return fig


def export(out_dir: Path, min_units: int = MIN_UNITS, size: int = BAND_SIZE
           ) -> List[str]:
    """Write the ranking, the bands, and the figure."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = cuisine_table(min_units)
    written = []
    table.round(4).to_csv(out_dir / "restaurants_cuisine_ranking.csv", index=False)
    written.append("restaurants_cuisine_ranking.csv")
    bands(table, size).round(4).to_csv(out_dir / "restaurants_cuisine_bands.csv",
                                       index=False)
    written.append("restaurants_cuisine_bands.csv")
    fig = plot_bands(table, size)
    fig.savefig(out_dir / "restaurants_cuisine_bands.png", dpi=200)
    plt.close(fig)
    written.append("restaurants_cuisine_bands.png")
    return written
