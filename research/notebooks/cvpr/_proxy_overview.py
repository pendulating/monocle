"""The proxy overview: 1 figure and 2 tables for the validation by proxy.

The paper has little space, thus this module folds 3 questions into 1 figure
and 1 small table:

| Question | Where it sits |
|----------|---------------|
| Does the model agree with the proxy? | Panel A, 1 marker for each layer |
| Does the white-box read of Gemma agree as well? | Panel A, the green diamond |
| Does the geography layer change that? | Panel A, the line between the markers |
| Does an ablation align the model more or less with the proxy? | Panel B, the heatmap |
| Do the 2 models agree with each other? | The cross-model table |
| Which value family does an ablation move? | The family table |

This module is **downstream**. It reads the exports of `master/results_table.py`
and `master/ablations.py` in `master/outputs/`, and the run parquets of the
canonical registry for the raw cross-model rows. It never touches W&B.

Run it from the canonical venv:

    .venv-mllmsci-vllm025cu129/bin/python research/notebooks/cvpr/_proxy_overview.py

The files go to `master/outputs/` and to the paper repo (`PAPER_DIR`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _results_table as R  # noqa: E402
import _style as S  # noqa: E402

__version__ = "1.1.0"

REPO_ROOT = HERE.parent.parent.parent
OUT_DIR = HERE / "master" / "outputs"
PAPER_DIR = REPO_ROOT / "research" / "papers" / "cvpr27_machine-perception-public-spaces"
EXPORT_VERSION = "1.0.0"   # the version stem of the master exports it reads
SINGLE_VERSION = "1.0.0"   # the version stem of the single-image export

# The 3 geography layers, coarse last. The short name is what the figure shows.
LAYERS: List[Tuple[str, str]] = [
    ("census_tract", "tract"), ("nta", "NTA"), ("community_district", "CD"),
]
LAYER_SHORT = dict(LAYERS)
MODELS = ["gemma-4-12b", "qwen3.5-9b"]
MODEL_SHORT = {"gemma-4-12b": "Gemma-4-12B", "qwen3.5-9b": "Qwen3.5-9B"}
# Blue and amber pass the colour-vision checks as a pair. The coral and teal
# of the house ramp do not (protan Delta E 3.9), thus the ramp keeps the
# polarity job and these 2 keep the identity job.
MODEL_COLOUR = {"gemma-4-12b": S.PAL["blue"], "qwen3.5-9b": S.PAL["amber"]}
# The single-image reads of Gemma (`_single_image_read.py`): the read key,
# the legend label, the marker, and the colour. They are reads of the SAME
# rater as the blue markers, thus they take a marker shape of their own and
# not a second blue.
SINGLE_READS: List[Tuple[str, str, str, str]] = [
    ("contrast_L42", "Gemma, lens read (block 42)", "D", S.PAL["green"]),
    ("c", "Gemma, single-image gap $c$", "^", S.PAL["warmgrey"]),
]
# What the figure draws by default: the lens read alone, at the community
# district alone, 1 diamond for each row. Three layers of 2 more series made
# the panel too dense to read (2026-09-06). The lens read is the white-box
# read, thus it is the one the figure shows; the gap stays in the paper
# table and in the export frame.
DEFAULT_SINGLE_READS: Tuple[str, ...] = ("contrast_L42",)
DEFAULT_SINGLE_LAYERS: Tuple[str, ...] = ("CD",)
ARMS: List[Tuple[str, str]] = [
    ("noabstain", "no abstain"), ("nocity", "no city"), ("textfirst", "text first"),
]
ARM_TEX = {"noabstain": "no abst.", "nocity": "no city", "textfirst": "text first"}

# The 7 cases of the raw cross-model table, in the order of the paper. Parks
# and plazas come from 1 run, thus they share 1 raw row. Street photography
# holds no proxy, thus it has raw columns alone.
RAW_CASES: List[Tuple[str, str]] = [
    ("subway_safety", "Subway safety"),
    ("libraries", "Libraries"),
    ("schools", "Schools"),
    ("road_quality", "Road quality"),
    ("parks_plazas", "Parks / Plazas"),
    ("restaurants", "Restaurants"),
    ("street_photography", "Street photography"),
]
# The folder of the master table that carries the layer columns of a raw row.
# Parks and plazas share 1 run and 2 folders, thus the raw row shows the layer
# values of both, on 2 lines.
LAYER_FOLDERS = {
    "subway_safety": ["Subway safety"], "libraries": ["Libraries"],
    "schools": ["Schools"], "road_quality": ["Road quality"],
    "parks_plazas": ["Parks", "Plazas"], "restaurants": ["Restaurants"],
    "street_photography": [],
}

# The value family of each proxy. A family names the attitude the proxy stands
# for, thus panel B can be read as "does the ablation align the model more
# with this attitude?". Edit this map when a proxy changes its meaning.
FAMILY: Dict[str, str] = {
    "crime_density": "disorder",
    "transit_crime_per_rider": "disorder",
    "transit_felony_per_rider": "disorder",
    "building_violations_total": "disorder",
    "building_violations_open": "disorder",
    "dob_ecb_violations": "disorder",
    "pothole_complaints_per_mile": "disorder",
    "pothole_repairs_per_mile": "disorder",
    "pothole_repairs": "disorder",
    "median_household_income": "affluence",
    "alteration_permits": "affluence",
    "bid_membership": "affluence",
    "plaza_bid_partner": "affluence",
    "school_economic_need": "affluence",
    "dot_pavement_rating_weighted": "official rating",
    "dot_pavement_rating": "official rating",
    "resurfacing_age": "official rating",
    "pip_acceptable_rate": "official rating",
    "pip_condition_acceptable_rate": "official rating",
    "school_kept_clean": "official rating",
    "inspection_score_history": "official rating",
    "inspection_score": "official rating",
    "grade_a_rate": "official rating",
    "construction_year": "vintage",
    "acquisition_year": "vintage",
    "year_completed": "vintage",
    "subway_ridership": "context",
    "park_acreage": "context",
    "school_attendance": "context",
    "outdoor_dining": "context",
}
FAMILY_ORDER = ["disorder", "affluence", "official rating", "vintage", "context"]
FAMILY_GLYPH = {"disorder": "D", "affluence": "A", "official rating": "R",
                "vintage": "V", "context": "c"}
FAMILY_LABEL = {
    "disorder": "Disorder (broken windows)",
    "affluence": "Affluence and investment",
    "official rating": "Official condition rating",
    "vintage": "Vintage",
    "context": "Context",
}

MIN_AREAS = R.MIN_AREAS
MIN_BOTH = R.MIN_BOTH


def _fmt(v: float, digits: int = 2) -> str:
    """The table format of `_results_table`, and a zero never carries a sign."""
    if v is not None and isinstance(v, float) and np.isfinite(v) and abs(v) < 0.5 * 10 ** -digits:
        v = 0.0
    return R._fmt(v, digits)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
def load_long(version: str = EXPORT_VERSION) -> pd.DataFrame:
    """Read the long frame of the master table at the 3 layers.

    The frame keeps the rows of the table (`in_table`, not `superseded`). A
    cell under the area floor loses its statistics but keeps its row, thus a
    layer with too few areas prints as a gap and not as a missing row.
    """
    frames = []
    for key, short in LAYERS:
        path = OUT_DIR / f"results_table_{key}_v{version}_long.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Export the master table at the {key} "
                "layer first (master/results_table.py, or "
                "research/scripts/export_cvpr_results_table.py).")
        d = pd.read_csv(path)
        d["layer"] = short
        frames.append(d)
    long = pd.concat(frames, ignore_index=True)
    long = long[long["in_table"] & ~long["superseded"]].copy()
    thin = long["n"] < MIN_AREAS
    long.loc[thin, ["tau", "tau_p", "r", "agreement"]] = np.nan
    long["family"] = long["proxy_key"].map(FAMILY)
    return long


def load_single(version: str = SINGLE_VERSION) -> Optional[pd.DataFrame]:
    """Read the single-image reads at the 3 layers, or None when absent.

    The frame takes the same area floor as `load_long`: a cell under
    `MIN_AREAS` loses its statistics and prints as a gap.
    """
    path = OUT_DIR / f"single_image_read_v{version}_long.csv"
    if not path.exists():
        print(f"[overview] {path} is missing; the figure shows no single-image read")
        return None
    d = pd.read_csv(path)
    thin = d["n"] < MIN_AREAS
    d.loc[thin, ["tau", "tau_p", "r", "agreement"]] = np.nan
    return d


def load_arms(layer: str = "community_district",
              version: str = EXPORT_VERSION) -> pd.DataFrame:
    """Read the proxy re-run of the ablation arms at 1 layer.

    `master/ablations.py` writes it for the layer its control picks. The
    community district is the one it has written so far.
    """
    path = OUT_DIR / f"proxy_arms_{layer}_v{version}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Export master/ablations.py at the {layer} "
            "layer first.")
    arms = pd.read_csv(path)
    arms["family"] = arms["proxy"].map(FAMILY)
    return arms


def table_rows(long: pd.DataFrame) -> pd.DataFrame:
    """The rows of the figure: 1 for each (case, proxy) of the master table.

    The row order is the order of the master table. A row must reach the
    area floor at the community district for at least 1 rater, the same
    rule the LaTeX table applies.
    """
    cd = long[long["layer"] == "CD"]
    keep = (cd.groupby(["case", "proxy_key"])["tau"]
            .apply(lambda s: s.notna().any()))
    keep = keep[keep].reset_index()[["case", "proxy_key"]]
    rows = (cd.drop_duplicates(["case", "proxy_key"])
            .merge(keep, on=["case", "proxy_key"])
            .sort_values(["case_order", "proxy"])
            [["case", "case_order", "folder", "proxy_key", "proxy", "context", "family"]]
            .reset_index(drop=True))
    return rows


# The row name of each proxy, with its orientation in words. Every proxy
# arrives "higher is better", thus a bad-is-high measure was negated, and
# its row opens with "fewer" or "lower": a positive tau then says that the
# model's better-scored area holds fewer violations, less crime, and so on.
# A vintage or context row carries no orientation, thus its tag states what
# a positive value means. A key that is absent here prints the master-table
# label without its tags.
ROW_LABEL: Dict[str, str] = {
    "crime_density": "Lower crime density",
    "transit_crime_per_rider": "Lower transit crime per 100k riders",
    "transit_felony_per_rider": "Lower transit felonies per 100k riders",
    "building_violations_total": "Fewer building violations",
    "building_violations_open": "Fewer open building violations",
    "dob_ecb_violations": "Fewer DOB and ECB violations",
    "pothole_complaints_per_mile": "Fewer pothole complaints per mile",
    "pothole_repairs_per_mile": "Fewer pothole repairs per mile",
    "pothole_repairs": "Fewer pothole repairs",
    "resurfacing_age": "Fewer years since resurfacing",
    "school_economic_need": "Lower economic need index",
    "inspection_score_history": "Fewer DOHMH inspection points (mean)",
    "inspection_score": "Fewer DOHMH inspection points (last)",
    "construction_year": "Building construction year (newer = +)",
    "acquisition_year": "Park acquisition year (newer = +)",
    "year_completed": "POPS completion year (newer = +)",
    "park_acreage": "Park acreage (larger = +)",
    "subway_ridership": "Rider entries (busier = +)",
}


def row_label(proxy: str, family: Optional[str], proxy_key: str = "") -> str:
    """The y label of a row: the oriented proxy name and its family glyph."""
    lab = ROW_LABEL.get(proxy_key)
    if lab is None:
        lab = proxy
        for tag in (" (negated)", " (context)", " (oriented)"):
            lab = lab.replace(tag, "")
    glyph = FAMILY_GLYPH.get(family or "", "?")
    return f"{lab}  [{glyph}]"


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------
def _cell(long: pd.DataFrame, case: str, proxy_key: str, model: str, layer: str
          ) -> Tuple[float, float]:
    sub = long[(long["case"] == case) & (long["proxy_key"] == proxy_key)
               & (long["model"] == model) & (long["layer"] == layer)]
    if sub.empty:
        return float("nan"), float("nan")
    return float(sub["tau"].iloc[0]), float(sub["tau_p"].iloc[0])


def _single_cell(single: pd.DataFrame, case: str, proxy_key: str, read: str,
                 layer: str) -> Tuple[float, float]:
    sub = single[(single["case"] == case) & (single["proxy_key"] == proxy_key)
                 & (single["read"] == read) & (single["layer"] == layer)]
    if sub.empty:
        return float("nan"), float("nan")
    return float(sub["tau"].iloc[0]), float(sub["tau_p"].iloc[0])


def plot_overview(long: pd.DataFrame, arms: pd.DataFrame,
                  single: Optional[pd.DataFrame] = None,
                  single_reads: Sequence[str] = DEFAULT_SINGLE_READS,
                  single_layers: Sequence[str] = DEFAULT_SINGLE_LAYERS,
                  width: float = 6.875, vmax: float = 0.12,
                  arm_layer_label: str = "CD"):
    """Draw the 2-panel overview figure and return it.

    Panel A: the Kendall tau of each row against the model score, 1 marker
    for each layer, the size grows with the layer. A hollow marker has
    p at or above 0.05. A line joins the layers of 1 rater. When `single`
    holds the single-image reads of Gemma, each row gets 1 more series for
    each read of `single_reads`, with the marker of `SINGLE_READS`, at the
    layers of `single_layers`, between the 2 raters. A series with 1 layer
    draws no line.

    Panel B: the paired change of tau under each ablation arm, as a
    heatmap. Teal aligns the model MORE with the proxy, coral LESS. A
    black outline marks a bootstrap interval that excludes 0.
    """
    rows = table_rows(long)
    n = len(rows)

    ys: List[float] = []
    labels: List[str] = []
    case_breaks: List[Tuple[float, str]] = []
    header_y: List[float] = []
    y = 0.0
    last_case = None
    for _, r in rows.iterrows():
        if r["case"] != last_case:
            if last_case is not None:
                y += 0.35
            case_breaks.append((y - 0.5, r["case"]))
            header_y.append(y)
            y += 1
            last_case = r["case"]
        ys.append(y)
        labels.append(row_label(r["proxy"], r["family"], r["proxy_key"]))
        y += 1
    rows = rows.assign(y=ys)
    y_end = y

    reads = [r for r in SINGLE_READS if r[0] in single_reads] if single is not None else []
    row_h = 0.165 if not reads else 0.165 + 0.02 * len(reads)
    fig = plt.figure(figsize=(width, row_h * (n + len(case_breaks)) + 1.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1.6], wspace=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

    if not reads:
        off = {MODELS[0]: -0.2, MODELS[1]: 0.2}
        single_off: Dict[str, float] = {}
    else:
        # The raters sit at the edges, the single-image reads between them.
        edge = 0.2 + 0.05 * len(reads)
        off = {MODELS[0]: -edge, MODELS[1]: edge}
        step = 2 * edge / (len(reads) + 1)
        single_off = {r: -edge + step * (k + 1) for k, (r, _, _, _) in enumerate(reads)}
    size = {"tract": 11, "NTA": 22, "CD": 38}

    # ---- panel A
    for _, r in rows.iterrows():
        for m in MODELS:
            yy = r["y"] + off[m]
            xs = []
            for _, short in LAYERS:
                tau, p = _cell(long, r["case"], r["proxy_key"], m, short)
                if np.isnan(tau):
                    continue
                xs.append(tau)
                filled = bool(p < 0.05) if not np.isnan(p) else False
                ax_a.scatter(
                    tau, yy, s=size[short],
                    facecolors=MODEL_COLOUR[m] if filled else "white",
                    edgecolors=S.EDGE if filled else MODEL_COLOUR[m],
                    linewidths=0.6 if filled else 1.0, zorder=4)
            if len(xs) >= 2:
                ax_a.plot([min(xs), max(xs)], [yy, yy], color=MODEL_COLOUR[m],
                          lw=0.9, alpha=0.7, zorder=3)
        for read, _, marker, colour in reads:
            yy = r["y"] + single_off[read]
            xs = []
            for _, short in LAYERS:
                if short not in single_layers:
                    continue
                tau, p = _single_cell(single, r["case"], r["proxy_key"], read, short)
                if np.isnan(tau):
                    continue
                xs.append(tau)
                filled = bool(p < 0.05) if not np.isnan(p) else False
                ax_a.scatter(
                    tau, yy, s=size[short], marker=marker,
                    facecolors=colour if filled else "white",
                    edgecolors=S.EDGE if filled else colour,
                    linewidths=0.6 if filled else 1.0, zorder=4)
            if len(xs) >= 2:
                ax_a.plot([min(xs), max(xs)], [yy, yy], color=colour,
                          lw=0.9, alpha=0.7, zorder=3)

    ax_a.axvline(0, color=S.EDGE, lw=0.8, zorder=2)
    ax_a.set_yticks(list(rows["y"]) + header_y)
    ax_a.set_yticklabels(labels + [c for _, c in case_breaks])
    for t in ax_a.get_yticklabels()[n:]:
        t.set_fontweight("bold")
    ax_a.set_ylim(y_end - 0.4, -0.7)
    ax_a.set_xlim(-0.5, 0.65)
    ax_a.set_xlabel(r"Kendall $\tau$ with the proxy (positive = agrees)")
    ax_a.grid(axis="y", visible=False)
    for yb, _ in case_breaks:
        ax_a.axhline(yb, color=S.EDGE, lw=0.4, alpha=0.5)
        ax_b.axhline(yb, color=S.EDGE, lw=0.4, alpha=0.5)

    # ---- panel B
    cols = [(m, a, al) for m in MODELS for a, al in ARMS]
    cmap = S.CMAP_DIV
    for j, (m, a, _) in enumerate(cols):
        for _, r in rows.iterrows():
            cell = arms[(arms["case"] == r["case"]) & (arms["proxy"] == r["proxy_key"])
                        & (arms["model"] == m) & (arms["arm"] == a)]
            if cell.empty:
                continue
            d = float(cell["d_tau_paired"].iloc[0])
            if np.isnan(d):
                continue
            lo = float(cell["d_tau_lo"].iloc[0])
            hi = float(cell["d_tau_hi"].iloc[0])
            sig = bool(lo > 0 or hi < 0)
            colour = cmap((np.clip(d, -vmax, vmax) + vmax) / (2 * vmax))
            ax_b.add_patch(mpatches.Rectangle(
                (j - 0.45, r["y"] - 0.42), 0.9, 0.84, facecolor=colour,
                edgecolor=S.EDGE if sig else "none",
                linewidth=1.1 if sig else 0, zorder=3))
            ax_b.text(j, r["y"], f"{d:+.2f}".replace("0.", "."), ha="center",
                      va="center", fontsize=5.6, color=S.text_on(colour),
                      fontweight="bold" if sig else "normal", zorder=4)
    ax_b.set_xlim(-0.6, len(cols) - 0.4)
    ax_b.set_xticks(range(len(cols)))
    ax_b.set_xticklabels([al for _, _, al in cols], rotation=60, ha="right")
    for k, m in enumerate(MODELS):
        ax_b.text(k * 3 + 1, -0.9, MODEL_SHORT[m], ha="center", va="bottom",
                  fontsize=7.5, color=MODEL_COLOUR[m], fontweight="bold")
    ax_b.axvline(2.5, color=S.EDGE, lw=0.5)
    ax_b.grid(visible=False)
    ax_b.tick_params(axis="y", left=False, labelleft=False)
    ax_b.set_xlabel(rf"$\Delta\tau$ under the ablation, {arm_layer_label} layer")
    for sp in ("left", "bottom"):
        ax_b.spines[sp].set_visible(False)

    # ---- legend
    grey = "grey"
    handles = [
        Line2D([], [], marker="o", ls="", ms=3, color=MODEL_COLOUR[MODELS[0]],
               label=MODEL_SHORT[MODELS[0]]),
        Line2D([], [], marker="o", ls="", ms=3, color=MODEL_COLOUR[MODELS[1]],
               label=MODEL_SHORT[MODELS[1]]),
    ]
    if reads:
        layer_tag = (f" ({', '.join(single_layers)})"
                     if len(single_layers) < len(LAYERS) else "")
        handles += [
            Line2D([], [], marker=marker, ls="", ms=3.4, color=colour,
                   label=label + layer_tag)
            for _, label, marker, colour in reads
        ]
    handles += [
        Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=grey,
               label=r"$p \geq 0.05$"),
        Line2D([], [], marker="o", ls="", ms=2.2, mfc=grey, mec=grey, label="tract"),
        Line2D([], [], marker="o", ls="", ms=3.4, mfc=grey, mec=grey, label="NTA"),
        Line2D([], [], marker="o", ls="", ms=4.6, mfc=grey, mec=grey,
               label="community district"),
    ]
    if not reads:
        ax_a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07),
                    ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.9)
    else:
        # More entries than fit under panel A alone, thus the legend spans
        # the figure, under both panels.
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=4, frameon=False, handletextpad=0.3, columnspacing=0.9,
                   fontsize=6.5)
    return fig


FIGURE_CAPTION = (
    "\\textbf{Validation by proxy, the geography layer, and the prompt "
    "ablations.} Each row compares the score of one model with one outside "
    "measurement. \\textbf{Left:} the Kendall $\\tau$ at the census tract, "
    "the NTA, and the community district; the marker size grows with the "
    "layer, a line joins the 3 layers of one rater, and a hollow marker has "
    "$p \\geq 0.05$. Every proxy points the same way, thus a positive value "
    "is agreement: a row that opens with \\emph{fewer} or \\emph{lower} is a "
    "bad-is-high measure that we negated, and a positive value there says that "
    "the better-scored area holds less of it. A vintage or context row "
    "carries no orientation, and its tag states what a positive value means. "
    "The green diamond adds the white-box read of Gemma "
    "(\\S\\ref{sec:methods-whitebox}): the block-42 residual of a single "
    "image projected on the lens contrast vector, at the community district "
    "over the units the white-box study scored. The letter after a proxy names its value family: D "
    "disorder (broken windows), A affluence and investment, R official "
    "condition rating, V vintage, c context. \\textbf{Right:} the paired "
    "change of $\\tau$ at the community district when the prompt loses the "
    "abstention, the city name, or the images-first order. Teal aligns the "
    "model more with the proxy, coral less. A black outline marks a 95\\% "
    "bootstrap interval over areas that excludes 0."
)
FIGURE_LABEL = "fig:proxy-overview"


def figure_include(path: str = "figures/proxy_overview.pdf",
                   caption: str = FIGURE_CAPTION, label: str = FIGURE_LABEL) -> str:
    """The `figure*` float that shows the overview. Paste it into a section."""
    return "\n".join([
        "% Generated by _proxy_overview.py — do not edit by hand.",
        "\\begin{figure*}[t]",
        "\\centering",
        f"\\includegraphics[width=\\textwidth]{{{path}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{figure*}",
    ]) + "\n"


# --------------------------------------------------------------------------
# The cross-model table
# --------------------------------------------------------------------------
def raw_rows(cases: Sequence[Tuple[str, str]] = tuple(RAW_CASES),
             models: Sequence[str] = tuple(MODELS)) -> pd.DataFrame:
    """The raw agreement of the 2 models, pair by pair, for each case.

    Unlike `R.raw_cross_model`, this keeps a case whose rater is thin, and
    it keeps street photography, which has no proxy. The label columns run
    over every pair, and the direction columns take the `MIN_BOTH` floor
    when they print.
    """
    rows = []
    for i, (canon, label) in enumerate(cases):
        labels = [R.load_raw_labels(canon, m) for m in models]
        if any(s is None for s in labels):
            print(f"[overview] no canonical run for {canon}; the raw row is absent")
            continue
        joined = pd.concat(
            [labels[0].rename("a"), labels[1].rename("b")], axis=1, join="inner")
        rows.append({
            "case": label, "case_order": i, "canon_case": canon,
            "model_a": models[0], "model_b": models[1],
            **R.raw_agreement(joined["a"], joined["b"]),
        })
    return pd.DataFrame(rows)


def layer_rows(long: pd.DataFrame) -> pd.DataFrame:
    """The cross-model tau of each case at each layer, from the master table.

    A case carries 1 cross-model value for each proxy, because each proxy
    keeps its own areas. The row takes the proxy with the MOST areas, and it
    reports that count.
    """
    out = []
    for (case, layer), sub in long.groupby(["case", "layer"]):
        sub = sub[sub["xm_n"] >= MIN_AREAS]
        if sub.empty:
            out.append({"case": case, "layer": layer, "xm_n": np.nan,
                        "xm_tau": np.nan, "xm_agreement": np.nan})
            continue
        best = sub.sort_values("xm_n", ascending=False).iloc[0]
        out.append({"case": case, "layer": layer, "xm_n": int(best["xm_n"]),
                    "xm_tau": float(best["xm_tau"]),
                    "xm_agreement": float(best["xm_agreement"])})
    return pd.DataFrame(out)


def cross_model_table(long: pd.DataFrame, raw: Optional[pd.DataFrame] = None
                      ) -> pd.DataFrame:
    """Join the raw rows to the layer rows: 1 line for each case.

    A line carries `kind`: `both` holds the raw and the layer cells, `raw`
    holds the raw cells alone (a header, or street photography), and
    `layer` holds the layer cells alone. Parks and plazas come from 1 run
    and 2 proxy folders, thus they print as 1 `raw` header and 2 `layer`
    lines under it.
    """
    raw = raw_rows() if raw is None else raw
    lay = layer_rows(long)

    def layer_cells(folder: Optional[str]) -> Dict[str, float]:
        cells: Dict[str, float] = {}
        for _, short in LAYERS:
            hit = lay[(lay["case"] == folder) & (lay["layer"] == short)]
            ok = len(hit) and pd.notna(hit["xm_n"].iloc[0])
            cells[f"n_{short}"] = int(hit["xm_n"].iloc[0]) if ok else np.nan
            cells[f"tau_{short}"] = float(hit["xm_tau"].iloc[0]) if ok else np.nan
        return cells

    raw_keys = ["n_pairs", "abstain_a", "abstain_b", "label_agreement",
                "label_kappa", "n_both", "direction_agreement", "direction_kappa"]
    out = []
    for _, r in raw.iterrows():
        base = {"case": r["case"], "case_order": r["case_order"],
                "canon_case": r["canon_case"], "model_a": r["model_a"],
                "model_b": r["model_b"]}
        raw_cells = {k: r[k] for k in raw_keys}
        folders = LAYER_FOLDERS.get(r["canon_case"], [])
        if len(folders) == 1:
            out.append({**base, "line": r["case"], "kind": "both", **raw_cells,
                        **layer_cells(folders[0])})
        else:
            out.append({**base, "line": r["case"], "kind": "raw", **raw_cells,
                        **layer_cells(None)})
            for folder in folders:
                out.append({**base, "line": folder, "kind": "layer",
                            **{k: np.nan for k in raw_keys},
                            **layer_cells(folder)})
    return pd.DataFrame(out).reset_index(drop=True)


CROSS_MODEL_CAPTION = (
    "\\textbf{Cross-model agreement.} Gemma-4-12B and Qwen3.5-9B judged the "
    "same 1{,}100{,}000 pairs of each case in the same order. The raw columns "
    "compare the 2 labels pair by pair: Cohen $\\kappa$ over the 6 labels, "
    "and over the side alone on the pairs that both models answered ($n$). "
    "The layer columns compare the 2 area scores after the mean over each "
    "census tract, NTA, or community district, as Kendall $\\tau$ over the "
    "areas of the largest proxy join. A mean over hundreds of comparisons "
    "removes noise, thus every layer reads higher than the raw pair."
)
CROSS_MODEL_LABEL = "tab:cross-model"


def _int(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{int(v):,}".replace(",", "{,}")


def cross_model_to_latex(df: pd.DataFrame, caption: str = CROSS_MODEL_CAPTION,
                         label: str = CROSS_MODEL_LABEL,
                         min_both: int = MIN_BOTH) -> str:
    """Write the cross-model table as a 1-column LaTeX float."""
    fmt = _fmt
    lines = [
        "% Generated by _proxy_overview.py — do not edit by hand.",
        f"% Written by research/notebooks/cvpr/_proxy_overview.py v{__version__}.",
        "% Needs \\usepackage{booktabs} and \\usepackage{graphicx}.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{4}{c}{raw, pair by pair} "
        "& \\multicolumn{3}{c}{area score, $\\tau$} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}",
        "Case & abst. & $\\kappa_{\\text{label}}$ & $n$ & "
        "$\\kappa_{\\text{side}}$ & tract & NTA & CD \\\\",
        "\\midrule",
    ]
    thin = False
    for _, r in df.iterrows():
        if r["kind"] == "layer":
            name = f"\\quad {r['line']}"
            raw_cells = ["", "", "", ""]
        else:
            name = r["line"]
            void = int(r["n_both"]) < min_both
            thin = thin or void
            raw_cells = [
                f"{fmt(r['abstain_a'])}/{fmt(r['abstain_b'])}",
                fmt(r["label_kappa"]),
                _int(r["n_both"]),
                "--" if void else fmt(r["direction_kappa"]),
            ]
        if r["kind"] == "raw":
            layer_cells = ["", "", ""] if r["canon_case"] == "parks_plazas" \
                else ["--", "--", "--"]
        else:
            layer_cells = ["--" if pd.isna(r[f"tau_{short}"]) else fmt(r[f"tau_{short}"])
                           for _, short in LAYERS]
        lines.append(" & ".join([name] + raw_cells + layer_cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}"]
    notes = [
        "abst.\\ is the abstention share of each model, Gemma/Qwen.",
        "A layer cell needs $\\geq 20$ areas, and a dash marks fewer "
        "or no proxy join.",
    ]
    if thin:
        notes.append(f"A dash under $\\kappa_{{\\text{{side}}}}$ marks "
                     f"$n < {min_both:,}$".replace(",", "{,}")
                     + ", where too few pairs carry a side.")
    if (df["canon_case"] == "parks_plazas").any():
        notes.append("Parks and plazas come from one run, thus they share "
                     "the raw cells.")
    lines += ["\\par\\smallskip", "{\\footnotesize " + " ".join(notes) + "}",
              "\\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The family table
# --------------------------------------------------------------------------
def family_table(arms: pd.DataFrame) -> pd.DataFrame:
    """The mean paired change of tau by value family, model, and arm.

    A context row describes the place and makes no validation claim, thus
    it stays out. A thin cell stays out, the same as in the drift table.
    """
    a = arms[~arms["thin"] & (arms["family"] != "context")
             & arms["family"].notna()].copy()
    g = (a.groupby(["family", "model", "arm"])
         .agg(d_tau=("d_tau_paired", "mean"),
              n_cells=("d_tau_paired", "size"),
              n_sig=("d_tau_lo", lambda s: int(((s > 0) | (a.loc[s.index, "d_tau_hi"] < 0)).sum())))
         .reset_index())
    g["family_order"] = g["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return g.sort_values(["family_order", "model", "arm"]).reset_index(drop=True)


FAMILY_CAPTION = (
    "\\textbf{Which value does an ablation move?} The mean paired change of "
    "Kendall $\\tau$ at the community district, over the proxies of each "
    "value family. A positive value aligns the model more with the family, "
    "a negative value less. \\emph{rows} counts the (case, proxy) cells of the "
    "family."
)
FAMILY_TEX_LABEL = "tab:ablation-by-family"


def family_to_latex(df: pd.DataFrame, caption: str = FAMILY_CAPTION,
                    label: str = FAMILY_TEX_LABEL) -> str:
    """Write the family table as a 1-column LaTeX float."""
    fmt = _fmt
    lines = [
        "% Generated by _proxy_overview.py — do not edit by hand.",
        f"% Written by research/notebooks/cvpr/_proxy_overview.py v{__version__}.",
        "% Needs \\usepackage{booktabs} and \\usepackage{graphicx}.",
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\small",
        # 8 columns do not fit 1 column at the body size, thus the box
        # scales the table to the width it gets.
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        f" & & \\multicolumn{{3}}{{c}}{{{MODEL_SHORT[MODELS[0]]}}} "
        f"& \\multicolumn{{3}}{{c}}{{{MODEL_SHORT[MODELS[1]]}}} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "Value family & rows & " + " & ".join(ARM_TEX[a] for a, _ in ARMS) + " & "
        + " & ".join(ARM_TEX[a] for a, _ in ARMS) + " \\\\",
        "\\midrule",
    ]
    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam]
        if sub.empty:
            continue
        n_cells = int(sub["n_cells"].max())
        cells = []
        for m in MODELS:
            for a, _ in ARMS:
                hit = sub[(sub["model"] == m) & (sub["arm"] == a)]
                cells.append(fmt(float(hit["d_tau"].iloc[0])) if len(hit) else "--")
        lines.append(f"{FAMILY_LABEL[fam]} & {n_cells} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export(long: pd.DataFrame, arms: pd.DataFrame, fig, xm: pd.DataFrame,
           fam: pd.DataFrame, out_dir: Path = OUT_DIR,
           paper_dir: Optional[Path] = PAPER_DIR,
           version: str = __version__,
           single: Optional[pd.DataFrame] = None) -> List[str]:
    """Write the figure, the 2 tables, and the frames behind them.

    `out_dir` gets every file. `paper_dir` gets the figure under `figures/`
    and the tables under `tables/`, thus the paper reads them in place.
    """
    written: List[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"proxy_overview_v{version}"

    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    written += [f"{stem}.pdf", f"{stem}.png"]

    xm_tex = cross_model_to_latex(xm)
    fam_tex = family_to_latex(fam)
    (out_dir / f"cross_model_v{version}.tex").write_text(xm_tex)
    (out_dir / f"ablation_by_family_v{version}.tex").write_text(fam_tex)
    (out_dir / f"{stem}_figure.tex").write_text(figure_include())
    xm.to_csv(out_dir / f"cross_model_v{version}.csv", index=False)
    fam.to_csv(out_dir / f"ablation_by_family_v{version}.csv", index=False)
    table_rows(long).to_csv(out_dir / f"{stem}_rows.csv", index=False)
    if single is not None:
        single.to_csv(out_dir / f"{stem}_single.csv", index=False)
        written.append(f"{stem}_single.csv")
    written += [f"cross_model_v{version}.tex", f"cross_model_v{version}.csv",
                f"ablation_by_family_v{version}.tex",
                f"ablation_by_family_v{version}.csv",
                f"{stem}_figure.tex", f"{stem}_rows.csv"]

    if paper_dir is not None:
        figs = paper_dir / "figures"
        tabs = paper_dir / "tables"
        figs.mkdir(parents=True, exist_ok=True)
        tabs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / "proxy_overview.pdf")
        fig.savefig(figs / "proxy_overview.png")
        (figs / "proxy_overview.tex").write_text(figure_include())
        (tabs / "cross-model.tex").write_text(xm_tex)
        (tabs / "ablation-by-family.tex").write_text(fam_tex)
        written += [str(figs / "proxy_overview.pdf"), str(figs / "proxy_overview.png"),
                    str(figs / "proxy_overview.tex"),
                    str(tabs / "cross-model.tex"), str(tabs / "ablation-by-family.tex")]
    return written


def build_all(arm_layer: str = "community_district", with_single: bool = True,
              single_reads: Sequence[str] = DEFAULT_SINGLE_READS,
              single_layers: Sequence[str] = DEFAULT_SINGLE_LAYERS):
    """Load, draw, and tabulate. Returns (long, arms, fig, xm, fam, single)."""
    long = load_long()
    arms = load_arms(layer=arm_layer)
    single = load_single() if with_single else None
    fig = plot_overview(long, arms, single=single, single_reads=single_reads,
                        single_layers=single_layers,
                        arm_layer_label=LAYER_SHORT.get(arm_layer, arm_layer))
    xm = cross_model_table(long)
    fam = family_table(arms)
    return long, arms, fig, xm, fam, single


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-layer", default="community_district",
                    help="the layer of the proxy_arms export that panel B reads")
    ap.add_argument("--paper-dir", default=str(PAPER_DIR),
                    help="the paper repo; pass '' to skip it")
    ap.add_argument("--no-single", action="store_true",
                    help="draw the figure without the single-image reads")
    ap.add_argument("--single-reads", default=",".join(DEFAULT_SINGLE_READS),
                    help="the single-image reads to draw, comma-separated "
                         "(c, contrast_L42)")
    ap.add_argument("--single-layers", default=",".join(DEFAULT_SINGLE_LAYERS),
                    help="the layers of the single-image reads, comma-separated "
                         "(tract, NTA, CD)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="do not check the canonical registry against the disk")
    args = ap.parse_args(argv)

    if not args.skip_verify:
        import _canonical as C
        C.verify_or_raise()
    matplotlib.use("Agg")
    S.apply_house_style()
    long, arms, fig, xm, fam, single = build_all(
        arm_layer=args.arm_layer, with_single=not args.no_single,
        single_reads=tuple(args.single_reads.split(",")),
        single_layers=tuple(args.single_layers.split(",")))
    paper = Path(args.paper_dir) if args.paper_dir else None
    written = export(long, arms, fig, xm, fam, paper_dir=paper, single=single)
    print("wrote:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
