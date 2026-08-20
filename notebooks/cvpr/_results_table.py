"""The camera-ready table of validation-by-proxy results.

This reads what each prompt notebook already exported and builds 1 table over
every prompt. It does NOT read W&B: a prompt notebook is the only thing that
turns runs into unit scores, thus this module stays downstream of them.

The 1 exception is the raw cross-model section at the end. That section reads
the run parquets through the canonical registry, because a raw agreement needs
the label of each comparison, and no export holds it.

Run the prompt notebooks first. This module reads:

    <prompt>/outputs/<case>_aggregated_v*.parquet   the model score for each polygon
    <prompt>/outputs/<case>_proxy_v*.parquet        the proxy value for each polygon

The 3 columns
-------------
| Column | Meaning | Chance |
|--------|---------|--------|
| `agreement` | Share of polygons on the same side of both medians | 0.50 |
| `r` | Pearson correlation | 0 |
| `tau` | Kendall tau-b, the order-sensitive measure | 0 |

`agreement` and `tau` answer different questions. `agreement` splits at the
median and asks about a side, so it survives a monotone distortion of either
scale but throws away the size of a gap. `tau` counts concordant pairs, so it
reads the whole ordering. A high `agreement` with a low `tau` means the model
finds the good half but cannot rank inside it.

Warning: Pearson `r` assumes a linear relation. Read it beside `tau`, never
alone. Income is right-skewed, so `r` and `tau` part company on the income rows.

Orientation
-----------
**Every proxy arrives oriented "higher is better".** `_proxies` applies the
sign before it exports: crime density and pothole repairs are already NEGATED
in the parquet, and the restaurant inspection score is already flipped. Thus a
POSITIVE number always means agreement, and this module applies no sign of its
own.

Do not flip a sign here. The negative crime rows are a real disagreement — the
model calls dense Manhattan entrances safe while the record counts more crime
there — and a flip would hide the paper's most interesting result.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__version__ = "1.0.0"

HERE = Path(__file__).resolve().parent

# The join key of each geography layer.
LAYER_KEY = {
    "nta": "NTA2020",
    "community_district": "boro_cd",
    "census_tract": "boroct2020",
}

# The default layer of the table. Community district is the only layer with a
# usable n for every prompt: the libraries case keeps 48 community districts but
# only 1 census tract, because 3 libraries almost never share a tract.
DEFAULT_LAYER = "community_district"

# How a layer is named in the caption. The caption must read as prose, thus it
# gets the full name and not the column key.
LAYER_LABEL = {
    "nta": "neighborhood tabulation area",
    "community_district": "community district",
    "census_tract": "census tract",
}

DEFAULT_LABEL = "tab:validation-by-proxy"

# The caption. It states what a row holds and what each column measures, and it
# says nothing about what a number means. A reader draws that.
#
# Write this in ASD-STE100, the same as the rest of the project: the active
# voice, the simple tenses, 1 topic, and 6 sentences maximum.
CAPTION = (
    "Validation by proxy at the {layer} layer. "
    "Each row compares the model score of one area with an outside measurement "
    "of the same area. "
    "The columns give the number of areas $n$, the agreement, the Pearson "
    "correlation $r$, and the Kendall tau-b $\\tau$. "
    "The agreement is the share of areas on the same side of the two medians, "
    "and chance is 0.50. "
    "The last block compares the two models with each other on the same areas, "
    "and it uses no proxy. "
    "Every proxy points in the same direction, thus a positive value shows "
    "agreement."
)

# prompt folder -> (display name, output prefix, proxy name when the export
# holds no `proxy` column). The umbrella order comes first, then the standalone
# cases, which is the order the paper uses.
CASES: List[Dict[str, str]] = [
    {"folder": "subway", "label": "Subway safety", "prefix": "subway",
     "canon": "subway_safety"},
    {"folder": "libraries", "label": "Libraries", "prefix": "libraries",
     "single_proxy": "median_household_income", "canon": "libraries"},
    {"folder": "schools", "label": "Schools", "prefix": "schools",
     "single_proxy": "school_report_card", "canon": "schools"},
    {"folder": "road", "label": "Road quality", "prefix": "road_quality",
     "canon": "road_quality"},
    {"folder": "parks", "label": "Parks", "prefix": "parks",
     "canon": "parks_plazas"},
    {"folder": "plazas", "label": "Plazas", "prefix": "plazas",
     "canon": "parks_plazas"},
    {"folder": "restaurants", "label": "Restaurants", "prefix": "restaurants",
     "single_proxy": "inspection_score", "canon": "restaurants"},
]

# How a raw row is named. Parks and plazas come from 1 run, thus the 2 folders
# share 1 raw row and it needs a name of its own.
RAW_CASE_LABEL = {
    "subway_safety": "Subway safety",
    "libraries": "Libraries",
    "schools": "Schools",
    "road_quality": "Road quality",
    "parks_plazas": "Parks / Plazas",
    "restaurants": "Restaurants",
    "street_photography": "Street photography",
}

# How a proxy is named in the table. The "(negated)" note is not decoration: it
# tells a reader why a positive number means agreement on a bad-is-high measure.
PROXY_LABEL = {
    "median_household_income": "Median household income",
    "crime_density": "Crime density (negated)",
    "dot_pavement_rating": "DOT pavement rating",
    "pothole_repairs": "Pothole repairs (negated)",
    "pip_acceptable_rate": "PIP acceptable rate",
    "inspection_score": "DOHMH inspection score (oriented)",
    "school_report_card": "School report card",
    # The vintage rows. Each case gets the field that describes ITS unit, thus
    # the 3 keys stay separate and the label names the source.
    "construction_year": "Building construction year",
    "acquisition_year": "Park acquisition year",
    "year_completed": "POPS completion year",
}

# The proxies that carry NO orientation. Every other proxy arrives "higher is
# better", thus a positive value means agreement. A year does not: it measures
# age, not quality. A positive value on one of these rows says that the model
# calls a NEWER unit better, which the paper must state as a finding and never
# read as agreement.
UNORIENTED_PROXIES = {"construction_year", "acquisition_year", "year_completed"}

CANONICAL_MODELS = ("gemma-4-12b/instruct", "qwen3.5-9b/instruct")
MODEL_LABEL = {"gemma-4-12b/instruct": "gemma-4-12b", "qwen3.5-9b/instruct": "qwen3.5-9b"}


# --------------------------------------------------------------------------
# The 3 metrics
# --------------------------------------------------------------------------

def directional_agreement(x: Sequence[float], y: Sequence[float]) -> float:
    """Share of points on the same side of their OWN median.

    Each series splits at its own median, so the measure needs no shared scale
    and no calibration: the model score is an arbitrary ordinal mean, and the
    proxy is dollars, or complaints, or a 0-10 rating.

    Chance is 0.50. A value under 0.50 means the model puts units on the wrong
    side more often than a coin would.

    Warning: a point that sits exactly ON a median counts as "below", because
    the test is `> median`. With an odd n one point always does. This costs at
    most 1/n and it keeps the function deterministic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    hx = x > np.median(x)
    hy = y > np.median(y)
    return float((hx == hy).mean())


def metrics(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Return n, agreement, Pearson r, and Kendall tau-b for 1 pair of series."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {"n": int(len(x)), "agreement": float("nan"),
           "r": float("nan"), "tau": float("nan"), "tau_p": float("nan")}
    if len(x) < 3:
        return out
    out["agreement"] = directional_agreement(x, y)
    # A constant series has no correlation, and scipy warns rather than fails.
    if np.std(x) > 0 and np.std(y) > 0:
        out["r"] = float(stats.pearsonr(x, y)[0])
    tau = stats.kendalltau(x, y, variant="b")
    out["tau"] = float(tau.statistic)
    out["tau_p"] = float(tau.pvalue)
    return out


# --------------------------------------------------------------------------
# Reading what the prompt notebooks exported
# --------------------------------------------------------------------------

def cross_model(group: pd.DataFrame,
                models: Sequence[str] = CANONICAL_MODELS) -> Dict[str, float]:
    """Compare the 2 models with EACH OTHER on the areas of 1 row.

    The proxy plays no part here. This asks how far the 2 canonical raters
    agree, which sets a ceiling: no model can track an outside measurement more
    closely than the 2 models track one another, unless 1 of them is right by
    accident.

    Warning: the comparison uses the areas of THIS row, which are the areas
    where both models scored AND the proxy exists. Thus the number moves a
    little between the proxies of 1 case, and its `n` can be under the `n` of
    either model. The other choice — every area a model scored, whatever the
    proxy — gives 1 number for each case, but then its `n` does not match the
    row beside it.
    """
    if len(models) < 2:
        return {"n": float("nan"), "agreement": float("nan"),
                "r": float("nan"), "tau": float("nan"), "tau_p": float("nan")}
    a, b = models[0], models[1]
    ga = group[group["model"] == a][["unit", "mean_score"]]
    gb = group[group["model"] == b][["unit", "mean_score"]]
    both = ga.merge(gb, on="unit", suffixes=("_a", "_b"))
    return metrics(both["mean_score_a"], both["mean_score_b"])


def _newest(pattern: str) -> Optional[str]:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def load_case(case: Dict[str, str], layer: str = DEFAULT_LAYER
              ) -> Optional[pd.DataFrame]:
    """Join 1 prompt's model scores to its proxy values on 1 geography layer.

    Returns a long frame of (model, proxy, key, mean_score, proxy_mean), or
    `None` when the prompt has not exported yet.
    """
    folder, prefix = case["folder"], case["prefix"]
    agg_path = _newest(str(HERE / folder / "outputs" / f"{prefix}_aggregated_v*.parquet"))
    prx_path = _newest(str(HERE / folder / "outputs" / f"{prefix}_proxy_v*.parquet"))
    if not agg_path or not prx_path:
        return None

    key = LAYER_KEY[layer]
    agg = pd.read_parquet(agg_path)
    prx = pd.read_parquet(prx_path)
    agg = agg[agg["layer"] == layer]
    prx = prx[prx["layer"] == layer]
    if agg.empty or prx.empty:
        return None

    # A single-proxy export holds no `proxy` column, so name it here.
    if "proxy" not in prx.columns:
        prx = prx.assign(proxy=case.get("single_proxy", "proxy"))

    agg = agg[["model", key, "mean_score"]].dropna(subset=[key])
    prx = prx[["proxy", key, "proxy_mean"]].dropna(subset=[key])
    merged = agg.merge(prx, on=key, how="inner")
    return merged.rename(columns={key: "unit"})


def build(layer: str = DEFAULT_LAYER,
          models: Sequence[str] = CANONICAL_MODELS,
          cases: Sequence[Dict[str, str]] = tuple(CASES)) -> pd.DataFrame:
    """Build the long results frame: 1 row for each case, proxy, and model."""
    rows = []
    for case in cases:
        merged = load_case(case, layer=layer)
        if merged is None:
            continue
        for proxy, gp in merged.groupby("proxy"):
            for model in models:
                gm = gp[gp["model"] == model]
                if gm.empty:
                    continue
                m = metrics(gm["mean_score"], gm["proxy_mean"])
                rows.append({
                    "case": case["label"],
                    "case_order": [c["folder"] for c in CASES].index(case["folder"]),
                    "proxy_key": proxy,
                    "proxy": PROXY_LABEL.get(proxy, proxy),
                    "model": MODEL_LABEL.get(model, model),
                    **m,
                    **{f"xm_{k}": v for k, v in cross_model(gp, models).items()},
                })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["case_order", "proxy"]).reset_index(drop=True)


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long frame into the table shape: models across the columns."""
    if df.empty:
        return df
    w = df.pivot_table(
        index=["case_order", "case", "proxy"],
        columns="model", values=["n", "agreement", "r", "tau"], aggfunc="first",
    )
    # Put the model on the outside, so each model owns a block of columns.
    w = w.swaplevel(0, 1, axis=1)
    order = [m for m in MODEL_LABEL.values() if m in w.columns.get_level_values(0)]
    w = w.reindex(columns=pd.MultiIndex.from_product(
        [order, ["n", "agreement", "r", "tau"]]))
    # sort_index first, then flatten: dropping a column from a non-lexsorted
    # MultiIndex warns and is slow.
    w = w.sort_index(level="case_order")
    w = w.reset_index()
    w = w.drop(columns=[c for c in w.columns if c[0] == "case_order"])
    if "xm_n" in df.columns:
        # The cross-model values repeat on each model row, so take the first of
        # each (case, proxy) pair.
        x = (df.groupby(["case", "proxy"], as_index=False)
               [["xm_n", "xm_agreement", "xm_r", "xm_tau"]].first())
        label = f"{list(MODEL_LABEL.values())[0]} vs {list(MODEL_LABEL.values())[1]}"
        x.columns = pd.MultiIndex.from_tuples(
            [("case", ""), ("proxy", "")]
            + [(label, c.replace("xm_", "")) for c in
               ["xm_n", "xm_agreement", "xm_r", "xm_tau"]])
        w = w.merge(x, on=[("case", ""), ("proxy", "")], how="left")
    return w


# --------------------------------------------------------------------------
# Camera-ready output
# --------------------------------------------------------------------------

def _fmt(v: float, digits: int = 2) -> str:
    """Format a coefficient without a leading zero, the way a table wants it."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:.{digits}f}"
    return s.replace("0.", ".", 1) if s.startswith("0.") else \
        s.replace("-0.", "-.", 1) if s.startswith("-0.") else s


def to_latex(df: pd.DataFrame, layer: str = DEFAULT_LAYER,
             star_p: float = 0.05, caption: Optional[str] = None,
             label: str = DEFAULT_LABEL, full_width: bool = True,
             cross: bool = True) -> str:
    """Write the nested table as a full LaTeX float.

    The case is a row that spans the table, and its proxies indent under it.
    A star marks a Kendall tau with p < `star_p`. `tabular*` with
    `\\extracolsep{\\fill}` makes the table span the full line, whatever the
    number of models.

    Args:
        caption: The caption text. `None` builds the default from `CAPTION`.
        label: The `\\label` key that a `\\ref` points to.
        full_width: `True` writes `table*`, which spans the 2 columns of the
            paper. The table holds 13 columns, thus it does not fit 1 column.
            `False` writes a plain `table`.
        cross: Add the block that compares the 2 models with each other.

    Warning: the table needs `\\usepackage{booktabs}` for the rules.
    """
    if df.empty:
        return "% no results"
    models = [m for m in MODEL_LABEL.values() if m in set(df["model"])]
    n_mod = len(models)
    show_cross = cross and n_mod >= 2 and "xm_n" in df.columns
    n_blocks = n_mod + (1 if show_cross else 0)
    env = "table*" if full_width else "table"
    text = caption if caption is not None else CAPTION.format(
        layer=LAYER_LABEL.get(layer, layer), star_p=star_p,
    )
    heads = [f"\\multicolumn{{4}}{{c}}{{{m}}}" for m in models]
    if show_cross:
        heads.append(
            f"\\multicolumn{{4}}{{c}}{{{models[0]} vs.\\ {models[1]}}}")

    lines = [
        "% Generated by _results_table.py — do not edit by hand.",
        "% Needs \\usepackage{booktabs}.",
        f"\\begin{{{env}}}[t]",
        "\\centering",
        f"\\caption{{{text}}}",
        f"\\label{{{label}}}",
        # A 13-column table needs a tighter gap than the default 6pt. The
        # rest of the fit belongs to `tabular*`: `\extracolsep{\fill}` puts
        # the slack into the gaps between the columns, thus the table spans
        # exactly the line and the type stays at the size of the paper.
        "\\setlength{\\tabcolsep}{4pt}",
        "\\small",
        # `\linewidth` reads the width of the environment around it, thus the
        # same code fills a 1-column `table` and a 2-column `table*`.
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l"
        + "rrrr" * n_blocks + "}",
        "\\toprule",
        " & " + " & ".join(heads) + " \\\\",
        "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}"
                for i in range(n_blocks)),
        "Case / proxy & " + " & ".join(
            ["$n$ & agr. & $r$ & $\\tau$"] * n_blocks) + " \\\\",
        "\\midrule",
    ]
    for case in df.sort_values("case_order")["case"].unique():
        sub = df[df["case"] == case]
        lines.append(f"\\textbf{{{case}}}" + " & " * (4 * n_blocks) + " \\\\")
        for proxy in sub.sort_values("proxy")["proxy"].unique():
            cells = []
            for m in models:
                row = sub[(sub["proxy"] == proxy) & (sub["model"] == m)]
                if row.empty:
                    cells += ["--"] * 4
                    continue
                rec = row.iloc[0]
                star = "*" if np.isfinite(rec["tau_p"]) and rec["tau_p"] < star_p else ""
                cells += [str(int(rec["n"])), _fmt(rec["agreement"]),
                          _fmt(rec["r"]), _fmt(rec["tau"]) + star]
            if show_cross:
                # The cross-model values repeat on each model row of a proxy,
                # so read the first one.
                xr = sub[sub["proxy"] == proxy].iloc[0]
                xstar = ("*" if np.isfinite(xr["xm_tau_p"])
                         and xr["xm_tau_p"] < star_p else "")
                # n = 0 means the 2 models share no area on this row, which
                # happens when 1 of them abstained everywhere. Print a dash: a
                # "0" reads as a measured count.
                _xn = xr["xm_n"]
                cells += ["--" if (not np.isfinite(_xn) or int(_xn) == 0)
                          else str(int(_xn)),
                          _fmt(xr["xm_agreement"]), _fmt(xr["xm_r"]),
                          _fmt(xr["xm_tau"]) + xstar]
            lines.append(f"\\quad {proxy} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular*}",
        # The star note lives here, not in the caption: ASD-STE100 holds a
        # paragraph to 6 sentences, and the caption already uses all 6.
        "\\par\\smallskip",
        f"{{\\footnotesize A star marks $p < {star_p}$.}}",
        f"\\end{{{env}}}",
    ]
    return "\n".join(lines)


def to_display(df: pd.DataFrame) -> pd.DataFrame:
    """A plain nested frame for reading on screen, with the case shown once."""
    if df.empty:
        return df
    out = []
    for case in df.sort_values("case_order")["case"].unique():
        sub = df[df["case"] == case].sort_values("proxy")
        first = True
        for proxy in sub["proxy"].unique():
            row = {"case": case if first else "", "proxy": proxy}
            for m in MODEL_LABEL.values():
                r = sub[(sub["proxy"] == proxy) & (sub["model"] == m)]
                if r.empty:
                    row[f"{m} n"] = row[f"{m} agr"] = row[f"{m} r"] = row[f"{m} tau"] = "--"
                else:
                    rec = r.iloc[0]
                    row[f"{m} n"] = int(rec["n"])
                    row[f"{m} agr"] = _fmt(rec["agreement"])
                    row[f"{m} r"] = _fmt(rec["r"])
                    row[f"{m} tau"] = _fmt(rec["tau"])
            if "xm_n" in sub.columns:
                xr = sub[sub["proxy"] == proxy].iloc[0]
                row["x-model n"] = ("--" if (not np.isfinite(xr["xm_n"])
                                             or int(xr["xm_n"]) == 0)
                                    else int(xr["xm_n"]))
                row["x-model agr"] = _fmt(xr["xm_agreement"])
                row["x-model r"] = _fmt(xr["xm_r"])
                row["x-model tau"] = _fmt(xr["xm_tau"])
            out.append(row)
            first = False
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Raw cross-model agreement, at the level of 1 comparison
# --------------------------------------------------------------------------
#
# The `xm_` columns above compare the 2 models AFTER the aggregation into
# polygons. That step averages hundreds of comparisons into 1 polygon score,
# and an average of noise moves toward the middle, thus 2 raters that disagree
# on each comparison can still track each other across polygons.
#
# This section removes that help. It reads the label of each comparison from
# the canonical registry and asks how often the 2 models write the SAME label
# on the SAME pair. The pair id carries the presentation order, so the 2 models
# saw the identical image pair in the identical order.
#
# Warning: this section reads the run parquets, which the rest of the module
# does not. It reads them through the canonical registry, never from W&B.

# The score of each ordinal label. A `NotSure` maps to nothing and drops out,
# the same as in `_provenance.score_units`.
ORDINAL_SCORE = {"MuchLess": -2, "Less": -1, "Same": 0, "More": 1, "MuchMore": 2}
NOT_SURE = "NotSure"
# The direction columns need pairs that BOTH models answered. When one model
# abstains almost everywhere, the few pairs it answers are a selection and not
# a sample, thus a mean over them describes nothing.
#
# The schools case shows it: qwen3.5-9b abstains on 99.7% of the pairs, thus
# 277 pairs of 110,000 carry the direction columns, and they read .83 and .71 —
# the highest of the table. The renderers void that below this floor. The long
# frame keeps the number, so nothing is lost.
#
# Warning: the label columns take NO floor. They run over every pair, and they
# report the same case at chance, which is the true reading.
MIN_BOTH = 1000


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """The chance-corrected agreement of 2 raters over the same items.

    Raw agreement rewards a rater that writes 1 label everywhere: 2 models that
    abstain on 90% of the pairs agree on 81% of them by accident. Kappa removes
    what the marginals alone give. 0 is chance and 1 is a perfect match.

    Warning: kappa is undefined when both raters use exactly 1 label, because
    then chance agreement is 1. This returns NaN there.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) < 2:
        return float("nan")
    cats = pd.Index(sorted(set(a.tolist()) | set(b.tolist())))
    m = pd.crosstab(pd.Categorical(a, cats), pd.Categorical(b, cats),
                    dropna=False).to_numpy(dtype=float)
    n = m.sum()
    if n == 0:
        return float("nan")
    p_o = float(np.trace(m) / n)
    p_e = float((m.sum(axis=0) / n) @ (m.sum(axis=1) / n))
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def raw_agreement(a: pd.Series, b: pd.Series) -> Dict[str, float]:
    """Compare 2 label series that hold the same pairs, in the same order.

    Returns 6 numbers plus the counts:

    | Key | Meaning | Chance |
    |-----|---------|--------|
    | `label_agreement` | Same label of the 6, every pair | the marginals |
    | `label_kappa` | The same, corrected for chance | 0 |
    | `direction_agreement` | Same side, where both models answer | the marginals |
    | `direction_kappa` | The same, corrected for chance | 0 |

    The direction columns map each label to `-1`, `0`, or `+1` and drop a pair
    where 1 model abstains. They ask a weaker question than the label columns:
    a `More` beside a `MuchMore` counts as a match.
    """
    a = a.astype(str)
    b = b.astype(str)
    out = {
        "n_pairs": int(len(a)),
        "abstain_a": float(a.eq(NOT_SURE).mean()) if len(a) else float("nan"),
        "abstain_b": float(b.eq(NOT_SURE).mean()) if len(b) else float("nan"),
        "label_agreement": float("nan"), "label_kappa": float("nan"),
        "n_both": 0,
        "direction_agreement": float("nan"), "direction_kappa": float("nan"),
    }
    if len(a) < 2:
        return out
    out["label_agreement"] = float((a.to_numpy() == b.to_numpy()).mean())
    out["label_kappa"] = cohen_kappa(a.to_numpy(), b.to_numpy())

    sa = a.map(ORDINAL_SCORE)
    sb = b.map(ORDINAL_SCORE)
    both = sa.notna() & sb.notna()
    out["n_both"] = int(both.sum())
    if out["n_both"] >= 2:
        da = np.sign(sa[both].to_numpy(dtype=float)).astype(int)
        db = np.sign(sb[both].to_numpy(dtype=float)).astype(int)
        out["direction_agreement"] = float((da == db).mean())
        out["direction_kappa"] = cohen_kappa(da, db)
    return out


def load_raw_labels(canon_case: str, model: str, kind: str = "proxy"
                    ) -> Optional[pd.Series]:
    """Read the label of each comparison of 1 run, keyed by the pair id.

    The model is the SHORT name, such as `gemma-4-12b`. The read goes through
    the canonical registry, thus the notebook and the paper name the same run.
    """
    import _canonical as C

    hits = [r for r in C.runs(kind=kind, case=canon_case, model=model)
            if r.case == canon_case]
    if not hits:
        return None
    df = pd.read_parquet(hits[0].results_link,
                         columns=["pair_id", "relative_label"])
    return df.set_index("pair_id")["relative_label"].astype(str)


def raw_cross_model(models: Sequence[str] = CANONICAL_MODELS,
                    cases: Sequence[Dict[str, str]] = tuple(CASES),
                    kind: str = "proxy") -> pd.DataFrame:
    """Build 1 raw-agreement row for each case of the table.

    A case appears one time, whatever the number of proxies it carries, and
    the parks and plazas folders collapse into their 1 shared run.

    Warning: the row counts every pair of the run, and not the pairs behind the
    table above. The table above keeps a polygon only when the proxy exists
    there, thus the 2 `n` values do not match, and they answer 2 questions.
    """
    if len(models) < 2:
        return pd.DataFrame()
    short = [MODEL_LABEL.get(m, m) for m in models[:2]]

    seen: List[str] = []
    order = {}
    for i, case in enumerate(cases):
        canon = case.get("canon")
        if canon and canon not in seen:
            seen.append(canon)
            order[canon] = i

    rows = []
    for canon in seen:
        labels = [load_raw_labels(canon, m, kind=kind) for m in short]
        if any(s is None for s in labels):
            continue
        joined = pd.concat(
            [labels[0].rename("a"), labels[1].rename("b")], axis=1, join="inner")
        if joined.empty:
            continue
        rows.append({
            "case": RAW_CASE_LABEL.get(canon, canon),
            "case_order": order[canon],
            "canon_case": canon,
            "model_a": short[0],
            "model_b": short[1],
            **raw_agreement(joined["a"], joined["b"]),
        })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("case_order")
            .reset_index(drop=True))


def raw_to_display(df: pd.DataFrame, min_both: int = MIN_BOTH) -> pd.DataFrame:
    """A plain frame of the raw rows, for reading on screen.

    A row with fewer than `min_both` answered pairs shows no direction. See
    `MIN_BOTH` for why.
    """
    if df.empty:
        return df
    a, b = df["model_a"].iloc[0], df["model_b"].iloc[0]
    thin = df["n_both"] < min_both
    # `n_pairs` is the same on each row, thus it stays in the long frame and
    # out of the table. The caption of the LaTeX float states it.
    return pd.DataFrame({
        "case": df["case"],
        f"abstain {a}": df["abstain_a"].map(_fmt),
        f"abstain {b}": df["abstain_b"].map(_fmt),
        "label agr": df["label_agreement"].map(_fmt),
        "label kappa": df["label_kappa"].map(_fmt),
        "both answer": df["n_both"].astype(int),
        "dir agr": df["direction_agreement"].mask(thin).map(_fmt),
        "dir kappa": df["direction_kappa"].mask(thin).map(_fmt),
    })


RAW_CAPTION = (
    "Raw cross-model agreement, over the {pairs} comparisons of each case. "
    "The two models saw the identical image pairs in the identical order, thus "
    "a row compares their labels pair by pair and uses no proxy. "
    "The label columns ask for the same label of the six, over every pair. "
    "The direction columns map each label to a side and drop a pair where one "
    "model abstains. "
    "Cohen $\\kappa$ removes the agreement that the abstention rates give by "
    "accident, and 0 is chance. "
    "Read this table beside the aggregated agreement, which the average over "
    "each area lifts."
)

RAW_LABEL = "tab:raw-cross-model"



def raw_to_latex(df: pd.DataFrame, caption: Optional[str] = None,
                 label: str = RAW_LABEL, full_width: bool = False,
                 min_both: int = MIN_BOTH) -> str:
    """Write the raw-agreement rows as a full LaTeX float.

    The count of the comparisons is the same on each row, thus it is a caption
    and not a column. The table keeps `n`, which is not: `n` counts the pairs
    that BOTH models answered, and an abstention rate moves it.

    Args:
        min_both: The floor of the direction columns. A row under it prints a
            dash there, because a mean over a handful of answered pairs
            describes the selection and not the case. See `MIN_BOTH`.
        full_width: `False` writes a plain `table`, which sits in 1 column.
            The table holds 8 columns, thus `\\resizebox` scales it down to
            the column. `True` writes `table*`, which spans the 2 columns.

    Warning: the table needs `\\usepackage{booktabs}` for the rules and
    `\\usepackage{graphicx}` for the box.
    """
    if df.empty:
        return "% no raw cross-model results"
    a, b = df["model_a"].iloc[0], df["model_b"].iloc[0]
    env = "table*" if full_width else "table"
    pairs = sorted(set(int(v) for v in df["n_pairs"]))
    # The caption states 1 number, thus it needs the count to be the same on
    # each row. It is not when a case runs a different number of pairs.
    pairs_txt = (f"{pairs[0]:,}".replace(",", "{,}") if len(pairs) == 1
                 else "110{,}000")
    text = caption if caption is not None else RAW_CAPTION.format(pairs=pairs_txt)
    lines = [
        "% Generated by _results_table.py — do not edit by hand.",
        "% Needs \\usepackage{booktabs} and \\usepackage{graphicx}.",
        f"\\begin{{{env}}}[t]",
        "\\centering",
        f"\\caption{{{text}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\small",
        # 8 columns do not fit 1 column of a paper at the type size of the
        # body. `\resizebox` scales the box to the width it gets, thus the
        # table fits whatever `full_width` chooses.
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        " & \\multicolumn{2}{c}{abstention} & \\multicolumn{2}{c}{label} "
        "& \\multicolumn{3}{c}{direction} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}",
        f"Case & {a} & {b} & agr. & $\\kappa$ & $n$ & agr. & $\\kappa$ \\\\",
        "\\midrule",
    ]
    thin = False
    for _, r in df.iterrows():
        # Void the direction columns of a row that too few answered pairs
        # carry. The `n` beside the dash states how few.
        void = int(r["n_both"]) < min_both
        thin = thin or void
        lines.append(" & ".join([
            r["case"],
            _fmt(r["abstain_a"]), _fmt(r["abstain_b"]),
            _fmt(r["label_agreement"]), _fmt(r["label_kappa"]),
            f"{int(r['n_both']):,}".replace(",", "{,}"),
            "--" if void else _fmt(r["direction_agreement"]),
            "--" if void else _fmt(r["direction_kappa"]),
        ]) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
    ]
    # The dash needs a legend, or a reader counts it as a missing measurement.
    # It sits under the rules, the same as the star note of the table above.
    if thin:
        lines += [
            "\\par\\smallskip",
            f"{{\\footnotesize A dash marks $n < {min_both:,}$".replace(",", "{,}")
            + ", where too few pairs carry a direction.}",
        ]
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines)
