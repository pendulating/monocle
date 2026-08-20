"""Choropleth maps of a case, on 1 geography layer.

The validation notebooks correlate a model score with a proxy. A map answers a
different question: WHERE does the model score high, and does that map look
like the map of the proxy?

The small-multiples pair is the point. 2 maps of the same polygons, side by
side, let a reader see the shared pattern before any correlation is quoted.

Warning: a model score and a proxy carry different units and different scales.
Each panel therefore keeps its own colour bar, and the bars are labelled. A
shared bar would be meaningless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _geography as geo
import _provenance as prov
import _style as S

__version__ = "1.0.0"

# A polygon needs this many scored images before it gets a colour. An image-mode
# run spreads 110,000 pairs over a 500,000-image manifest, thus a tract with 2
# images carries almost no evidence.
MIN_IMAGES = 5

MODEL_LABEL = {"gemma-4-12b/instruct": "gemma-4-12b",
               "qwen3.5-9b/instruct": "qwen3.5-9b"}


def image_scores_by_layer(case: str, layer: str = "census_tract",
                          min_images: int = MIN_IMAGES,
                          models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Aggregate an IMAGE-mode case into polygons.

    The camera position places each shot, which for an image-mode case is the
    true position: there is no unit that the camera looks at from a distance.

    Returns 1 row for each model and polygon: the key, the mean score, and the
    number of images behind it.
    """
    import geopandas as gpd

    records = [r for r in prov.discover_runs(case)
               if models is None or r.model in set(models)]
    if not records:
        raise FileNotFoundError(f"no canonical run for {case}")

    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    frames = []
    for rec in records:
        scores = prov.score_images(prov.load_run_images(rec))
        pts = gpd.GeoDataFrame(
            scores,
            geometry=gpd.points_from_xy(scores["longitude"], scores["latitude"]),
            crs="EPSG:4326",
        ).to_crs(geo.WORKING_CRS)
        hit = gpd.sjoin(pts[["sample_id", "mean_score", "n_comparisons", "geometry"]],
                        poly[[key, "geometry"]], how="inner", predicate="within")
        out = hit.groupby(key).agg(
            n_images=("sample_id", "nunique"),
            mean_score=("mean_score", "mean"),
            comparisons=("n_comparisons", "sum"),
        ).reset_index()
        out = out[out["n_images"] >= min_images]
        out.insert(0, "model", MODEL_LABEL.get(rec.model, rec.model))
        out.insert(0, "layer", layer)
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def join_geometry(values: pd.DataFrame, layer: str, value_col: str):
    """Put a table of polygon values back onto the polygons, for drawing."""
    poly = geo.load_layer(layer)
    key = geo.LAYERS[layer]["key"]
    left = poly[[key, "geometry"]].copy()
    left[key] = left[key].astype(str)
    right = values.copy()
    right[key] = right[key].astype(str)
    return left.merge(right[[key, value_col]], on=key, how="left")


def _panel(ax, gdf, value_col: str, cmap, label: str,
           diverging: bool = False, quantile_clip: float = 0.05):
    """Draw 1 choropleth, with a colour bar under it.

    A few extreme polygons otherwise take the whole ramp, thus the scale runs
    between the 5% and the 95% quantile. The polygons past the clip keep the end
    colour, which is honest: the map says "at least this", not "exactly this".
    """
    import matplotlib as mpl

    values = gdf[value_col].astype(float)
    ok = values.notna()
    lo, hi = values[ok].quantile([quantile_clip, 1 - quantile_clip])
    if diverging:
        span = max(abs(lo), abs(hi))
        lo, hi = -span, span
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi)

    # The polygons with no value keep a light ground, so a reader can see the
    # city outline and knows the gap is missing data and not a low value.
    gdf[~ok].plot(ax=ax, color="#e9e6e1", edgecolor="none", zorder=1)
    gdf[ok].plot(ax=ax, column=value_col, cmap=cmap, norm=norm, edgecolor="none",
                 zorder=2)
    ax.set_axis_off()
    ax.set_aspect("equal")

    cb = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            orientation="horizontal", fraction=0.045, pad=0.02,
                            shrink=0.86)
    cb.set_label(label)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6, length=2)
    return ax


def small_multiples(left: pd.DataFrame, left_col: str, left_label: str,
                    right: pd.DataFrame, right_col: str, right_label: str,
                    layer: str = "census_tract",
                    left_cmap=None, right_cmap=None,
                    right_diverging: bool = True,
                    quantile_clip: float = 0.05):
    """2 maps of the same polygons, side by side.

    Args:
        left, right: Tables of polygon values, keyed by the layer key.
        *_label: The colour-bar label of each panel. The panels carry no title:
            the caption of the paper owns that text.
    """
    import matplotlib.pyplot as plt

    gl = join_geometry(left, layer, left_col)
    gr = join_geometry(right, layer, right_col)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.9))
    _panel(axes[0], gl, left_col, left_cmap or S.CMAP_SEQ, left_label,
           quantile_clip=quantile_clip)
    _panel(axes[1], gr, right_col, right_cmap or S.CMAP_DIV, right_label,
           diverging=right_diverging, quantile_clip=quantile_clip)
    fig.tight_layout(w_pad=0.6)
    return fig


def coverage(values: pd.DataFrame, layer: str, value_col: str) -> Dict[str, object]:
    """How much of the city the map really shows."""
    poly = geo.load_layer(layer)
    have = values[value_col].notna().sum()
    return {"layer": layer, "polygons": int(len(poly)), "coloured": int(have),
            "share": round(have / max(1, len(poly)), 3)}


def export_street_photography(out_dir: Path, layer: str = "census_tract",
                              min_images: int = MIN_IMAGES,
                              model: str = "gemma-4-12b") -> List[str]:
    """Write the income-against-score map, and the table behind it."""
    import matplotlib.pyplot as plt

    import _proxies as px

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = image_scores_by_layer("street_photography", layer, min_images)
    income = px.income_by_layer(layer)
    key = geo.LAYERS[layer]["key"]

    written: List[str] = []
    joined = scores.merge(income[[key, "proxy_mean"]]
                          .rename(columns={"proxy_mean": "median_income"}),
                          on=key, how="left")
    joined.round(4).to_csv(out_dir / f"street_photography_{layer}_scores.csv",
                           index=False)
    written.append(f"street_photography_{layer}_scores.csv")

    part = scores[scores["model"] == model]
    fig = small_multiples(income, "proxy_mean", "median household income (USD)",
                          part, "mean_score", "mean street photography score",
                          layer=layer)
    name = f"street_photography_income_map_{layer}.png"
    fig.savefig(out_dir / name, dpi=220)
    plt.close(fig)
    written.append(name)
    return written
