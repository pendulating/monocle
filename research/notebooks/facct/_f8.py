"""F8: the helpers of the evidence notebook (`f8_evidence.py`).

F8 asks if the pixels explain the F4 bias results (D134). It has 4 parts:

| Part | Test | Scripts | Results |
|------|------|---------|---------|
| F8.1 | method B: OLS of the valence on 1 NTA feature, with image-content controls | `research/scripts/f8/method_b.py` | `research/figures/f8/method_b/` |
| F8.2 | the counterfactual cue: the same image, a rich or a poor neighbourhood name | `research/scripts/f8/build_cue_sample.py`, `cue_job.sbatch` | `research/figures/f8/cue/cue_*` |
| F8.3 | the R3 readout ("Is this a cool place ...?") after each answer, against the NTA | `research/scripts/f8/r3_readout.py`, `analyze_cue.py` | `research/figures/f8/cue/r3_*`, `research/figures/f8/method_b_r3/` |
| F8.4 | the plain logit lens on 200 gemma-4-31b answers, a figure only (D135) | `research/scripts/f8/lens_replay_f4.py`, `lens_figure.py` | `research/figures/f8/lens/` |

This module does no test. The scripts do the tests, and this module reads
their CSV files. It reads the raw answers and readouts only for the views
that the CSV files do not hold: the example of 1 image, and the NTA scatter.
`bash research/scripts/f8/finalize.sh` makes the CSV files of F8.2 and F8.3 again.

The record of each part is in `research/docs/`: `f8-method-b.md`,
`f8-counterfactual-cue.md`, `f8-lens-figure.md`, and the F8 section of
`facct-runbook.md`.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
_CVPR = REPO_ROOT / "research" / "notebooks" / "cvpr"
if str(_CVPR) not in sys.path:
    sys.path.insert(0, str(_CVPR))

import _style as S  # noqa: E402

S.apply_house_style()  # the same font and sizes in a script as in the notebooks

__version__ = "1.1.0"

FIG = REPO_ROOT / "research" / "figures" / "f8"
DATA = REPO_ROOT / "data" / "f8"
ACS = REPO_ROOT / "data" / "f4_traces" / "nta_acs.parquet"
OUT = REPO_ROOT / "research" / "notebooks" / "facct" / "figures" / "f8"

# The 4 models of F8.2 and F8.3. F8.1 also has the 2 Llama models (F4).
CUE_MODELS = ("gemma-4-12b", "gemma-4-31b", "qwen3.5-9b", "qwen3.8-27b")
ALL_MODELS = CUE_MODELS + ("llama-3.2-11b", "llama-3.2-90b")
CASES = ("restaurant", "street")
READOUTS = ("c_yes", "c_cool", "c_yes_i", "c_cool_i")  # _i: the instructed form
FEATURES = ("pct_white_nh", "pct_black_nh", "pct_asian_nh", "pct_bachelors", "median_household_income")
WORD_LISTS = ("upscale", "cheap", "danger", "safe")
MIN_NTA_IMAGES = 20  # F8.3, as analyze_cue.py

# The same colours and markers as `_f4.py`: 1 hue for each family, light for
# the small model and dark for the large model.
MODEL_COLORS: Dict[str, str] = {
    "gemma-4-12b": S.PAL["periwinkle"], "gemma-4-31b": S.PAL["blue"],
    "qwen3.5-9b": S.PAL["mint"], "qwen3.8-27b": S.PAL["teal"],
    "llama-3.2-11b": S.PAL["amber"], "llama-3.2-90b": S.PAL["coral"],
}
MODEL_MARKERS: Dict[str, str] = {m: {"gemma": "o", "qwen": "s", "llama": "^"}[m.split("-")[0].rstrip("0123456789.")]
                                 for m in ALL_MODELS}

READOUT_LABELS = {
    "c_yes": "c_yes: logp(Yes) - logp(No)",
    "c_cool": "c_cool: logp( cool) - logp( lame), after 'This place is'",
    "c_yes_i": "c_yes_i: c_yes, the question names the answer form",
    "c_cool_i": "c_cool_i: c_cool, the question names the answer form",
}
FEATURE_LABELS = {
    "pct_white_nh": "% white", "pct_black_nh": "% Black", "pct_asian_nh": "% Asian",
    "pct_bachelors": "% bachelor's", "median_household_income": "income",
    "log_median_household_income": "log income",
}


# ── reading ───────────────────────────────────────────────────────────
def _csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(FIG / rel)


def load_results() -> Dict[str, pd.DataFrame]:
    """Give each result CSV of F8 as a DataFrame, keyed by a short name."""
    out = {
        "mb_fits": _csv("method_b/valence_fits.csv"),
        "mb_summary": _csv("method_b/valence_summary.csv"),
        "mb_cases": _csv("method_b/valence_cases.csv"),
        "mb_counts": _csv("method_b/valence_counts.csv"),
        "mb20_cases": _csv("method_b/valence_nta20_cases.csv"),
        "primary": _csv("cue/cue_primary.csv"),
        "by_pair": _csv("cue/cue_by_pair.csv"),
        "words": _csv("cue/cue_words.csv"),
        "cut": _csv("cue/cue_cut.csv"),
        "echo": _csv("cue/cue_name_echo.csv"),
        "spearman": _csv("cue/r3_nta_spearman.csv"),
        "pearson": _csv("cue/r3_model_pearson.csv"),
        "lens_rates": _csv("lens/lens_rates.csv"),
        "lens_bands": _csv("lens/lens_bands.csv"),
        "lens_top1": _csv("lens/lens_top1_bands.csv"),
        "lens_said": _csv("lens/said_answers.csv"),
        "lens_tokens": _csv("lens/list_tokens.csv"),
    }
    out["r3_fits"] = pd.concat([_csv(f"method_b_r3/{rd}_fits.csv").assign(readout=rd) for rd in READOUTS],
                               ignore_index=True)
    return out


def _read_parts(d: Path) -> pd.DataFrame:
    parts = sorted(d.glob("part-*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True) if parts else pd.DataFrame()


@lru_cache(maxsize=None)
def r3_pool() -> pd.DataFrame:
    """Give the R3 readouts of the F4 pool: 1 row for each (model, case, image)."""
    return pd.read_parquet(DATA / "r3_pool.parquet")


@lru_cache(maxsize=None)
def acs() -> pd.DataFrame:
    return pd.read_parquet(ACS).set_index("nta2020")


@lru_cache(maxsize=None)
def cue_design(case: str) -> pd.DataFrame:
    """Give the cue design of 1 case: 1 row for each image, with the 2 names."""
    return pd.read_parquet(DATA / f"cue_{case}_rich.parquet",
                           columns=["image_idx", "sample_id", "image_path", "boro", "nta_name", "rich_name", "poor_name"])


@lru_cache(maxsize=None)
def cue_answers(model: str, case: str) -> pd.DataFrame:
    """Give the 2 cue answers of each image (1 model, 1 case), with the R3 readouts.

    The readouts come from `data/f8/r3/<model>/cue`, matched on `item_id`
    (`cue|<case>|<arm>|<image_idx>`). An answer that stopped at the token
    limit has no readout. The column `none_<readout>` is the readout on the
    F4 answer (no cue) of the same image.
    """
    rows = []
    for arm in ("rich", "poor"):
        d = pd.read_parquet(DATA / "runs" / model / f"{case}_{arm}" / "responses.parquet",
                            columns=["image_idx", "prompt", "reasoning", "response", "finish_reason",
                                     "completion_tokens", "error"])
        rows.append(d.assign(arm=arm))
    ans = pd.concat(rows, ignore_index=True)
    ans["item_id"] = "cue|" + case + "|" + ans["arm"] + "|" + ans["image_idx"].astype(str)
    r3 = _read_parts(DATA / "r3" / model / "cue")[["item_id", *READOUTS]]
    ans = ans.merge(r3, on="item_id", how="left", validate="1:1")
    f4 = _read_parts(DATA / "r3" / model / "f4")[["item_id", *READOUTS]]
    f4 = f4[f4["item_id"].str.startswith(f"f4|{case}|")]
    f4 = f4.assign(image_idx=f4["item_id"].str.rsplit("|", n=1).str[1].astype(int)).drop(columns="item_id")
    ans = ans.merge(f4.rename(columns={rd: f"none_{rd}" for rd in READOUTS}), on="image_idx", how="left")
    return ans


def cue_example(model: str, case: str, image_idx: int) -> Dict[str, object]:
    """Give the 2 arms of 1 image: the design row and 1 answer row for each arm."""
    des = cue_design(case).set_index("image_idx").loc[image_idx]
    ans = cue_answers(model, case)
    ans = ans[ans["image_idx"] == image_idx].set_index("arm")
    return {"design": des, "rich": ans.loc["rich"], "poor": ans.loc["poor"]}


def cue_gaps(model: str, case: str, readout: str) -> pd.DataFrame:
    """Give rich minus poor of 1 readout for each image with 2 readouts, largest gap first."""
    a = cue_answers(model, case).pivot_table(index="image_idx", columns="arm", values=readout).dropna()
    a["diff"] = a["rich"] - a["poor"]
    a = a.join(cue_design(case).set_index("image_idx")[["boro", "nta_name", "rich_name", "poor_name"]])
    return a.reindex(a["diff"].abs().sort_values(ascending=False).index).reset_index()


def nta_means(case: str, readout: str) -> pd.DataFrame:
    """Give the NTA mean of 1 readout for each model, NTAs with 20 images or more, with the ACS features."""
    p = r3_pool()
    p = p[p["case"] == case]
    out = []
    for m, g in p.groupby("model"):
        cnt = g.groupby("nta2020").size()
        keep = cnt[cnt >= MIN_NTA_IMAGES].index
        x = g[g["nta2020"].isin(keep)].groupby("nta2020")[readout].agg(["mean", "size"])
        out.append(x.assign(model=m))
    return pd.concat(out).join(acs()[list(FEATURES) + ["ntaname"]]).reset_index()


# ── tables ────────────────────────────────────────────────────────────
def mark(value: float, significant: bool, fmt: str = "{:+.2f}") -> str:
    """Give a value as text, in bold when it is significant (Markdown)."""
    s = fmt.format(value)
    return f"**{s}**" if significant else s


def md_table(df: pd.DataFrame) -> str:
    """Give a DataFrame as a Markdown table (the index becomes the first column)."""
    df = df.reset_index()
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def primary_wide(primary: pd.DataFrame) -> pd.DataFrame:
    """Give the 32 primary tests as 1 row for each (case, readout), 1 column for each model."""
    p = primary.assign(cell=[mark(d, s) for d, s in zip(primary["diff"], primary["significant"])])
    w = p.pivot_table(index=["case", "readout"], columns="model", values="cell", aggfunc="first")
    w = w.reindex(columns=list(CUE_MODELS))
    w = w.reindex(pd.MultiIndex.from_product([CASES, READOUTS], names=["case", "readout"]))
    w.index = [f"{c}, `{r}`" for c, r in w.index]
    w.index.name = "case, readout"
    return w


def spearman_wide(sp: pd.DataFrame, case: str, readout: str) -> pd.DataFrame:
    """Give the NTA Spearman rho of 1 (case, readout): 1 row for each model, 1 column for each feature."""
    g = sp[(sp["case"] == case) & (sp["readout"] == readout)]
    g = g.assign(cell=[mark(r, s) for r, s in zip(g["rho"], g["significant"])])
    w = g.pivot_table(index="model", columns="feature", values="cell", aggfunc="first")
    w = w.reindex(index=list(CUE_MODELS), columns=list(FEATURES))
    w.columns = [FEATURE_LABELS[f] for f in w.columns]
    return w


def fits_view(fits: pd.DataFrame) -> pd.DataFrame:
    """Give the method B fits with the size in SD of the outcome, for 1 table."""
    f = fits.copy()
    f["std_beta0"] = f["beta0"] / f["outcome_sd"]
    f["std_beta1"] = f["beta1"] / f["outcome_sd"]
    cols = ["model", "case", "field", "feature", "n", "n_nta", "std_beta0", "std_beta1", "share_kept",
            "p0_bonf", "p1_bonf", "sig0", "sig1"]
    return f[[c for c in cols if c in f]].round(4)


# ── figures ───────────────────────────────────────────────────────────
# The figures follow ~/style/proper-plotting.md: no caption text in the
# image. Each plot function attaches its title, caption and tags with
# `describe`, and `save` writes them to a .json file next to the image.

# Plain names of the readouts, for the titles and the axis labels.
READOUT_NAMES = {
    "c_yes": "yes over no", "c_cool": "cool over lame",
    "c_yes_i": "yes over no, instructed", "c_cool_i": "cool over lame, instructed",
}
LENS_MEASURES = {
    "rate_top5": "share in the top 5 tokens", "rate_top1": "share as the top token",
    "rate_top5_not_said": "share in the top 5, not said",
}
SIG_TEXT = ("A filled marker is significant (Bonferroni p < 0.05); a hollow marker is not significant. "
            "The bars are 95% confidence intervals.")


def readout_name(rd: str) -> str:
    return READOUT_NAMES.get(rd, rd)


def describe(fig, title: str, caption: str, tags) -> None:
    """Attach the caption data of 1 figure. `save` writes it to the .json file."""
    fig._plot_meta = {"plot-title": title, "plot-caption": caption, "plot-tags": list(tags)}


def _model_rows(ax, models, groups, n_models):
    ticks = [j * (n_models + 1) + (n_models - 1) / 2 for j in range(len(groups))]
    ax.set_yticks(ticks, groups)
    ax.invert_yaxis()


def _model_handles(models):
    """Give 1 filled legend marker for each model."""
    from matplotlib.lines import Line2D

    return [Line2D([], [], ls="", marker=MODEL_MARKERS[m], markersize=5, markerfacecolor=MODEL_COLORS[m],
                   markeredgecolor=S.EDGE, markeredgewidth=S.EDGE_LW, label=m) for m in models]


def _fill_handles(filled: str, hollow: str):
    """Give 2 neutral legend markers: filled and hollow."""
    from matplotlib.lines import Line2D

    kw = dict(ls="", marker="o", markersize=5, markeredgecolor=S.EDGE, markeredgewidth=S.EDGE_LW)
    return [Line2D([], [], markerfacecolor=S.PAL["warmgrey"], label=filled, **kw),
            Line2D([], [], markerfacecolor="white", label=hollow, **kw)]


def _top_legend(fig, handles=None, ncol: int = 4, xlabel: Optional[str] = None,
                ylabel: Optional[str] = None) -> None:
    """Lay out the panels, then put the legend on the top edge, and 1 shared x or y label (if any)."""
    import matplotlib.pyplot as plt

    fig.tight_layout(rect=(0.04 if ylabel else 0, 0.07 if xlabel else 0, 1, 1))
    top = max(ax.get_tightbbox().transformed(fig.transFigure.inverted()).y1 for ax in fig.axes)
    kw = {"handles": handles} if handles is not None else {}
    fig.legend(loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, top), borderaxespad=0.2, **kw)
    if xlabel:
        fig.supxlabel(xlabel, y=0.0, va="bottom", fontsize=plt.rcParams["axes.labelsize"])
    if ylabel:
        fig.supylabel(ylabel, x=0.0, ha="left", fontsize=plt.rcParams["axes.labelsize"])


def plot_primary(primary: pd.DataFrame, readouts=READOUTS):
    """Draw the primary tests: rich minus poor with the 95% CI, 1 panel for each readout."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(readouts), figsize=(1.9 * len(readouts) + 0.6, 2.4), sharey=True,
                             squeeze=False)
    for ax, rd in zip(axes[0], readouts):
        g = primary[primary["readout"] == rd]
        for j, case in enumerate(CASES):
            for i, m in enumerate(CUE_MODELS):
                r = g[(g["case"] == case) & (g["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j * (len(CUE_MODELS) + 1) + i
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]], fmt="none",
                            ecolor=S.EDGE, elinewidth=0.7, capsize=1.5)
                ax.scatter(r["diff"], y, s=26, marker=MODEL_MARKERS[m], linewidth=S.EDGE_LW, zorder=3,
                           color=MODEL_COLORS[m] if r["significant"] else "white", edgecolor=S.EDGE)
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.set_title(readout_name(rd).replace(", ", ",\n"))
        ax.set_xlabel("rich name minus\npoor name (nats)")
    _model_rows(axes[0][0], CUE_MODELS, CASES, len(CUE_MODELS))
    _top_legend(fig, _model_handles(CUE_MODELS), ncol=4 if len(readouts) > 2 else 2)
    n = len(primary)
    describe(fig, "A rich or a poor neighbourhood name changes the judgment of the same image",
             "The same image gets 2 prompts that differ only in the neighbourhood name: a rich name and a poor "
             "name. Each marker is the paired mean of the rich-name readout minus the poor-name readout, in nats "
             "of log-odds. 'Instructed' readouts name the answer form in the question. " + SIG_TEXT +
             f" Paired t tests, Bonferroni over {n} tests.",
             ["F8.2", "counterfactual cue", "RQ3"])
    return fig


def plot_baseline(primary: pd.DataFrame, readout: str):
    """Draw the no-cue, poor and rich means of 1 readout: any name moves the readout."""
    import matplotlib.pyplot as plt

    arms = (("mean_none", "no name", S.PAL["warmgrey"], "D", -0.18),
            ("mean_poor", "poor name", S.PAL["coral"], "o", 0.0),
            ("mean_rich", "rich name", S.PAL["teal"], "o", 0.18))
    fig, axes = plt.subplots(1, len(CASES), figsize=(6.2, 2.2), sharey=True)
    g = primary[primary["readout"] == readout].set_index(["case", "model"])
    for ax, case in zip(axes, CASES):
        for i, m in enumerate(CUE_MODELS):
            r = g.loc[(case, m)]
            vals = [r[c] for c, *_ in arms]
            for (c, lab, col, mk, dy) in arms:  # dy: a small offset, so that equal values stay visible
                ax.scatter(r[c], i + dy, s=24, color=col, marker=mk, edgecolor=S.EDGE, linewidth=S.EDGE_LW,
                           zorder=3, label=lab if (case == CASES[0] and i == 0) else None)
        ax.set_title(case)
        ax.set_xlabel(f"mean log-odds (nats),\n{readout_name(readout)}")
    axes[0].set_yticks(range(len(CUE_MODELS)), CUE_MODELS)
    axes[0].invert_yaxis()
    _top_legend(fig, ncol=3)
    describe(fig, "Any neighbourhood name moves the readout away from the no-name answer",
             f"The mean readout ({readout_name(readout)}) for each model and case, over the paired images of "
             "each cue test: with no name (the F4 answer), with the poor name, and with the rich name. The "
             "no-name answers used a 4,096-token limit. The 3 markers of a model have a small vertical offset.",
             ["F8.2", "counterfactual cue", "RQ3"])
    return fig


def plot_pairs(by_pair: pd.DataFrame, case: str, readout: str):
    """Draw the mean of rich minus poor for each name pair (no test), 1 column of dots for each model."""
    import matplotlib.pyplot as plt

    g = by_pair[(by_pair["case"] == case) & (by_pair["readout"] == readout)].copy()
    g["pair"] = g["rich_name"] + " / " + g["poor_name"]
    order = g.groupby("pair")["diff"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(order) + 0.8))
    for i, pair in enumerate(order):
        for k, m in enumerate(CUE_MODELS):
            r = g[(g["pair"] == pair) & (g["model"] == m)]
            if r.empty:
                continue
            ax.scatter(r["diff"].iloc[0], i + (k - 1.5) * 0.12, s=10 + r["n"].iloc[0] / 4, color=MODEL_COLORS[m],
                       marker=MODEL_MARKERS[m], edgecolor=S.EDGE, linewidth=S.EDGE_LW, zorder=3,
                       label=m if i == 0 else None)
    ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel(f"rich name minus poor name (nats), {readout_name(readout)}")
    ax.set_ylabel("rich name / poor name")
    ax.legend(loc="lower right", frameon=False)
    describe(fig, f"The cue effect for each name pair ({case})",
             f"The mean of the rich-name minus the poor-name readout ({readout_name(readout)}) for each name "
             "pair and model. The marker area grows with the count of images of the pair. This is a "
             "description: there is no test for each pair.",
             ["F8.2", "counterfactual cue", "name pairs"])
    return fig


def plot_words(words: pd.DataFrame, field: str):
    """Draw the share of answers with a word of each list, poor arm (hollow) to rich arm (filled)."""
    import matplotlib.pyplot as plt

    g = words[words["field"] == field]
    fig, axes = plt.subplots(1, len(WORD_LISTS), figsize=(7.0, 2.3), sharey=True)
    for ax, k in zip(axes, WORD_LISTS):
        h = g[g["list"] == k]
        for j, case in enumerate(CASES):
            for i, m in enumerate(CUE_MODELS):
                r = h[(h["case"] == case) & (h["model"] == m)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = j * (len(CUE_MODELS) + 1) + i
                ax.plot([r["rate_poor"], r["rate_rich"]], [y, y], color=S.EDGE, lw=0.6)
                ax.scatter(r["rate_poor"], y, s=20, facecolor="white", edgecolor=S.EDGE, marker=MODEL_MARKERS[m],
                           zorder=3)
                ax.scatter(r["rate_rich"], y, s=20, color=MODEL_COLORS[m], edgecolor=S.EDGE,
                           marker=MODEL_MARKERS[m], linewidth=S.EDGE_LW, zorder=3)
                if r["significant"]:
                    ax.annotate("*", (max(r["rate_poor"], r["rate_rich"]), y), xytext=(3, -3),
                                textcoords="offset points")
        ax.set_title(f"\u201c{k}\u201d words")
        ax.set_xlabel("share of answers")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    _model_rows(axes[0], CUE_MODELS, CASES, len(CUE_MODELS))
    _top_legend(fig, _model_handles(CUE_MODELS) + _fill_handles("rich name", "poor name"), ncol=3)
    describe(fig, f"The words of the {field} change with the neighbourhood name",
             f"The share of {field} texts with 1 word or more of each word list, with the poor name (hollow) "
             "and with the rich name (filled, model colour). * marks a significant change (exact McNemar test, "
             f"Bonferroni over {len(words)} tests, p < 0.05).",
             ["F8.2", "counterfactual cue", "word lists", "RQ3"])
    return fig


def plot_method_b(fits: pd.DataFrame, case: str, field: str, models=ALL_MODELS):
    """Draw the NTA coefficient in SD of the outcome, without (model 0) and with the controls (model 1).

    1 panel for each feature. The bars are 95% CIs with the t quantile of
    G - 1 degrees of freedom (G = the NTAs), as the test of `method_b.py`.
    A filled marker is significant after the Bonferroni correction.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy import stats

    g = fits[(fits["case"] == case) & (fits["field"] == field)]
    feats = [f for f in g["feature"].unique()]
    models = [m for m in models if m in set(g["model"])]
    fig, axes = plt.subplots(1, len(feats), figsize=(1.45 * len(feats) + 0.9, 0.32 * len(models) + 1.0),
                             sharey=True, squeeze=False)
    for ax, f in zip(axes[0], feats):
        for i, m in enumerate(models):
            r = g[(g["feature"] == f) & (g["model"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            q = stats.t.ppf(0.975, r["n_nta"] - 1)
            for k, dy, sig in ((0, -0.17, r["sig0"]), (1, 0.17, r["sig1"])):
                b, se = r[f"beta{k}"] / r["outcome_sd"], r[f"se{k}"] / r["outcome_sd"]
                ax.errorbar(b, i + dy, xerr=q * se, fmt="none", ecolor=S.EDGE, elinewidth=0.6, capsize=1)
                if k == 0:
                    ax.scatter(b, i + dy, s=18, marker="|", color=S.EDGE, linewidth=1.0, zorder=3)
                else:
                    ax.scatter(b, i + dy, s=18, marker=MODEL_MARKERS[m], zorder=3, edgecolor=S.EDGE,
                               color=MODEL_COLORS[m] if sig else "white", linewidth=S.EDGE_LW)
        ax.axvline(0, color=S.EDGE, lw=0.6, ls="--")
        ax.set_title(FEATURE_LABELS.get(f, f))
    axes[0][0].set_yticks(range(len(models)), models)
    axes[0][0].invert_yaxis()
    _top_legend(fig, [Line2D([], [], ls="", marker="|", color=S.EDGE, markersize=6, label="no controls"),
                      Line2D([], [], ls="", marker="o", markerfacecolor=S.PAL["warmgrey"], markeredgecolor=S.EDGE,
                             markeredgewidth=S.EDGE_LW, markersize=5, label="with the controls")], ncol=2,
                xlabel="SD of the outcome for each SD of the feature")
    outcome = "net valence" if field != "answer" else "the readout"
    describe(fig, f"The NTA association of {outcome} stays with the image-content controls ({case})",
             f"The OLS coefficient of 1 NTA feature on {outcome} ({field} field), in SD of the outcome for each "
             "SD of the feature. The upper bar (|) has no controls. The lower marker adds the image-content "
             "controls (cuisine, DOHMH grade, borough). " + SIG_TEXT.replace("A filled", "A filled lower") +
             " Standard errors clustered by NTA; t with G - 1 degrees of freedom (G = the NTAs); Bonferroni over "
             "the tests of the run.",
             ["F8.1", "method B", "RQ2"])
    return fig


def plot_spearman(sp: pd.DataFrame, case: str):
    """Draw the NTA Spearman rho as a grid: rows (model, readout), columns the features."""
    import matplotlib.pyplot as plt

    g = sp[sp["case"] == case]
    rows = [(m, rd) for m in CUE_MODELS for rd in READOUTS]
    grid = np.full((len(rows), len(FEATURES)), np.nan)
    sig = np.zeros_like(grid, dtype=bool)
    for i, (m, rd) in enumerate(rows):
        for j, f in enumerate(FEATURES):
            r = g[(g["model"] == m) & (g["readout"] == rd) & (g["feature"] == f)]
            if not r.empty:
                grid[i, j], sig[i, j] = r["rho"].iloc[0], r["significant"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.6, 0.2 * len(rows) + 0.9))
    im = ax.imshow(grid, cmap=S.CMAP_DIV, vmin=-0.85, vmax=0.85, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(FEATURES)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:+.2f}" + ("*" if sig[i, j] else ""), ha="center", va="center",
                        fontsize=5.5, color=S.text_on(mcolor(im, grid[i, j])))
    ax.set_xticks(range(len(FEATURES)), [FEATURE_LABELS[f] for f in FEATURES], rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [f"{m}, {readout_name(rd)}" for m, rd in rows])
    for k in range(1, len(CUE_MODELS)):
        ax.axhline(k * len(READOUTS) - 0.5, color="white", lw=1.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.6, label=r"Spearman $\rho$")
    describe(fig, f"The readout follows the NTA demographics ({case})",
             f"Spearman rho over the NTAs ({MIN_NTA_IMAGES} images or more) between the NTA mean of each readout "
             "and each ACS feature. 'Instructed' readouts name the answer form in the question. * marks "
             f"Bonferroni p < 0.05 over {len(sp)} tests.",
             ["F8.3", "R3 readout", "RQ2"])
    return fig


def mcolor(im, v: float) -> str:
    import matplotlib.colors as mcolors

    return mcolors.to_hex(im.cmap(im.norm(v)))


def plot_nta_scatter(case: str, readout: str, feature: str, sp: Optional[pd.DataFrame] = None):
    """Draw the NTA mean of 1 readout against 1 ACS feature, 1 panel for each model."""
    import matplotlib.pyplot as plt

    d = nta_means(case, readout)
    fig, axes = plt.subplots(1, len(CUE_MODELS), figsize=(7.2, 1.9), sharex=True,
                             gridspec_kw={"wspace": 0.45})  # room for the y tick labels of each panel
    for ax, m in zip(axes, CUE_MODELS):
        x = d[d["model"] == m].dropna(subset=[feature, "mean"])
        ax.scatter(x[feature], x["mean"], s=8, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                   edgecolor=S.EDGE, linewidth=0.3)
        title = m
        if sp is not None:
            r = sp[(sp["model"] == m) & (sp["case"] == case) & (sp["readout"] == readout) & (sp["feature"] == feature)]
            if not r.empty:
                title += f"\n$\\rho$ = {r['rho'].iloc[0]:+.2f}" + (" *" if r["significant"].iloc[0] else "")
        ax.set_title(title)
        ax.set_xlabel(FEATURE_LABELS[feature])
    axes[0].set_ylabel(f"NTA mean log-odds (nats),\n{readout_name(readout)}")
    n_tests = f" over {len(sp)} tests" if sp is not None else ""
    describe(fig, f"The NTA mean readout against {FEATURE_LABELS[feature]} ({case})",
             f"Each dot is 1 NTA with {MIN_NTA_IMAGES} images or more: the mean readout "
             f"({readout_name(readout)}) of its images against the ACS value ({FEATURE_LABELS[feature]}). "
             f"The title gives Spearman rho; * marks Bonferroni p < 0.05{n_tests}. The y scales differ by model.",
             ["F8.3", "R3 readout", "RQ2"])
    return fig


def plot_lens(rates: pd.DataFrame, lists=("upscale", "cheap", "danger", "safe", "cities"),
              measure: str = "rate_top5", suffix: str = ""):
    """Draw the lens rate of each list against the layer: rows the fields, columns the income groups."""
    import matplotlib.pyplot as plt

    colors = {"upscale": S.PAL["teal"], "cheap": S.PAL["amber"], "danger": S.PAL["coral"],
              "safe": S.PAL["blue"], "cities": S.PAL["warmgrey"]}
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.8), sharex=True, sharey=True)
    for i, field in enumerate(("reasoning", "response")):
        for j, group in enumerate(("low", "high")):
            ax = axes[i][j]
            for lst in lists:
                name = lst + suffix if lst != "cities" else lst
                g = rates[(rates["field"] == field) & (rates["group"] == group) & (rates["list"] == name)]
                g = g.sort_values("layer")
                ax.plot(g["layer"], np.maximum(g[measure], 1e-5), color=colors[lst], lw=1.0,
                        ls="--" if lst == "cities" else "-", label=lst if (i == 0 and j == 0) else None)
                ax.scatter([61.5], [max(g["said"].iloc[0], 1e-5)], s=12, color=colors[lst], edgecolor=S.EDGE,
                           linewidth=S.EDGE_LW, zorder=3, clip_on=False)
            ax.set_yscale("log")
            ax.set_ylim(8e-6, 2e-2)
            ax.set_title(f"{field}, {group}-income NTAs")
            if i == 1:
                ax.set_xlabel("layer")
    _top_legend(fig, ncol=5, ylabel=LENS_MEASURES.get(measure, measure))
    describe(fig, "The logit lens: the word lists across the layers of gemma-4-31b",
             "The share of token positions where a word of each list is in the lens prediction, by layer, for "
             "the thinking trace (reasoning) and the final answer (response) of 100 answers from low-income and "
             "100 from high-income NTAs. The dot at the right edge is the share of the said tokens. A value of 0 "
             "shows at the floor 1e-5. A description: there is no test.",
             ["F8.4", "logit lens", "RQ3"])
    return fig


def save(fig, name: str, out: Optional[Path] = None) -> List[Path]:
    """Write 1 figure as PDF and PNG, and its caption data as JSON, to `out` (default `research/figures/f8/`)."""
    import json

    out = out or OUT
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    for p in paths:
        fig.savefig(p, bbox_inches="tight", dpi=300)  # the notebook view also uses a tight box
    meta = getattr(fig, "_plot_meta", {"plot-title": name, "plot-caption": "", "plot-tags": []})
    meta = {"plot-title": meta["plot-title"], "plot-caption": meta["plot-caption"],
            "plot-latex-label": "fig:" + name.replace("_", "-"), "plot-tags": meta["plot-tags"]}
    paths.append(out / f"{name}.json")
    paths[-1].write_text(json.dumps(meta, indent=2) + "\n")
    return paths
