"""The vocabulary of each judge, with the extractor held fixed.

The question: which keys does 1 judge use that the other judge does not, in
each case, for the 3 named classes of the version 3 schema (`dimension`,
`inference`, `perspective`)? A **key** is the normalised `name` attribute
of a span.

**Warning: the extractor writes the name, not the judge.** The extractor
reads the trace of the judge and writes a short phrase. The 2 extractors
(Qwen3.8-27B, Gemma-4-31B) differ in vocabulary and in volume. Thus this
module never compares 2 arms with 2 different extractors. It compares the 2
judges under 1 extractor, then does the same under the other extractor,
then asks which contrasts hold under both:

| Extractor | Gemma-4-12B traces | Qwen3.5-9B traces |
|-----------|--------------------|-------------------|
| Qwen3.8-27B | `gemma_by_qwen27_v3` | `qwen_by_qwen27_v3` |
| Gemma-4-31B | `gemma_by_gemma31_v3` | `qwen_by_gemma31_v3` |

A key that is distinctive of 1 judge under BOTH extractors is a fact about
the judge. A key that is distinctive under 1 extractor only is not.

The unit of count is the trace (a document frequency). A key that 1 trace
uses 5 times counts 1 time. The qwen traces are about 2.5x longer than the
gemma traces, and the Gemma-4-31B extractor read the qwen traces in
6,000-character chunks, thus a raw span count favours the qwen judge.

The tables, from `build_bundle`:

| Key | Holds |
|-----|-------|
| `scores` | 1 row for each (case, class, extractor, key): the document count and the document frequency of each judge, their ratio, and the log-odds z |
| `robust` | 1 row for each (case, class, key): the z under each extractor, the floor flags, the `robust` flag, and the `side` |
| `unique` | the strictly unique keys of each extractor: >= `min_traces` traces of 1 judge, 0 of the other |
| `agreement` | 1 row for each (case, class): the Spearman of the 2 extractors' z over the keys both see, and the counts of significant and robust keys |
| `cross_case` | 1 row for each (class, judge, key) that is robust in 1 case or more: the number of cases |
| `summary` | 1 row for each class: the robust keys of each judge, and the median Spearman |
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _extractions as X
import _figmeta as F
import _ic_v3 as I
import _latex as L
import _proxy_overview as PO
import _style as S
import _words_across as WA

__version__ = "1.0.0"
_MOD = "_judge_vocab"

CLASSES: Tuple[str, ...] = ("dimension", "inference", "perspective")
JUDGES: Tuple[str, ...] = ("gemma-4-12b", "qwen3.5-9b")
EXTRACTORS: Tuple[str, ...] = ("qwen3.8-27b", "gemma-4-31b")
CASES: Tuple[str, ...] = tuple(WA.CASE_ORDER)

#: The judge colours of the project (`_proxy_overview.MODEL_COLOUR`): blue for
#: Gemma-4-12B, amber for Qwen3.5-9B. They pass the colour-vision checks as a
#: pair.
JUDGE_COLOUR: Dict[str, str] = dict(PO.MODEL_COLOUR)
JUDGE_SHORT: Dict[str, str] = {"gemma-4-12b": "Gemma-4-12B",
                               "qwen3.5-9b": "Qwen3.5-9B"}
EXTRACTOR_SHORT: Dict[str, str] = {"qwen3.8-27b": "Qwen3.8-27B",
                                   "gemma-4-31b": "Gemma-4-31B"}
#: The short column tag of each extractor.
EX_TAG: Dict[str, str] = {"qwen3.8-27b": "q27", "gemma-4-31b": "g31"}
#: The marker of each extractor in the figure.
EX_MARKER: Dict[str, str] = {"qwen3.8-27b": "o", "gemma-4-31b": "s"}

#: |z| at or above this is significant at 5%, 2-sided, for 1 key.
Z_CRIT = 1.96


# --------------------------------------------------------------------------
# The arms and the load
# --------------------------------------------------------------------------

def arm_grid() -> Dict[Tuple[str, str], str]:
    """The v3 arm of each (extractor, judge) cell, from `X.ARMS`."""
    out: Dict[Tuple[str, str], str] = {}
    for name, spec in X.ARMS.items():
        if spec.get("schema_version") == "v3":
            out[(spec["extractor"], spec["judge"])] = name
    return out


def normalise_keys(values: pd.Series) -> pd.Series:
    """Normalise the free names. The rule is `_ic_v3.normalise_names`, thus
    `_cues_across.normalise_cue`: lower case, punctuation and a hyphen
    become a space, an article and a degree adverb go, each word becomes
    singular. `Well-Maintained` and `well maintained` give 1 key.

    **Warning: this joins spelling variants only.** `maintenance` and
    `upkeep` stay 2 keys. The module uses no synonym list.
    """
    return I.normalise_names(values)


def slim_spans(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the rows of the 3 classes and the 4 columns that the count
    needs: `case`, `pair_id`, `extraction_class`, `key`."""
    part = df.loc[df["extraction_class"].isin(CLASSES),
                  ["case", "pair_id", "extraction_class", "name"]].copy()
    part["key"] = normalise_keys(part["name"])
    part = part.drop(columns="name").dropna(subset=["key"])
    for col in ("case", "extraction_class"):
        part[col] = part[col].astype(str)
    return part.reset_index(drop=True)


def doc_counts(spans: pd.DataFrame, totals: pd.Series) -> pd.DataFrame:
    """The document count of each key: the traces that use it 1 time or more.

    Args:
        spans: The output of `slim_spans` for 1 arm.
        totals: The traces of each case (the denominator). A trace with no
            span of the class still counts here.

    Returns:
        1 row for each (case, class, key): `docs`, `n_traces`, `df`
        (= docs / n_traces).
    """
    out = (spans.drop_duplicates(["case", "pair_id", "extraction_class", "key"])
           .groupby(["case", "extraction_class", "key"]).size()
           .rename("docs").reset_index())
    out["n_traces"] = out["case"].map(totals).astype(float)
    out["df"] = out["docs"] / out["n_traces"]
    return out


def load_arm_docs(arm: str, require_canonical: bool = True) -> pd.DataFrame:
    """Read 1 v3 arm with the paper filters and give its document counts.

    The read goes through `X.load(root=..., require_canonical=...)`. The
    denominator is `X.trace_counts`: every answered trace of the case.

    Raises:
        RuntimeError: the gate fails (the corpus is not canonical).
        FileNotFoundError: the corpus is not on disk.
    """
    spec = X.arm_roots([arm], present_only=False)[arm]
    df = X.load(root=spec["root"], require_canonical=require_canonical)
    totals = X.trace_counts(df, spec["root"])
    spans = slim_spans(df)
    del df
    out = doc_counts(spans, totals)
    out["judge"] = spec["judge"]
    out["extractor"] = spec["extractor"]
    out["arm"] = arm
    return out


def load_grid(require_canonical: bool = True) -> pd.DataFrame:
    """Read the 4 v3 arms and give 1 long frame of document counts.

    Raises:
        FileNotFoundError: an arm of the 2 x 2 is not on disk. The contrast
            needs all 4.
    """
    grid = arm_grid()
    present = X.arm_roots(list(grid.values()), present_only=True)
    missing = [a for a in grid.values() if a not in present]
    if missing:
        raise FileNotFoundError(f"the v3 arms {missing} are not on disk")
    return pd.concat([load_arm_docs(a, require_canonical) for a in grid.values()],
                     ignore_index=True)


# --------------------------------------------------------------------------
# The log-odds with an informative Dirichlet prior
# --------------------------------------------------------------------------

def log_odds_z(y_a, y_b, prior_scale: float = 1.0):
    """The weighted log-odds ratio of Monroe, Colaresi and Quinn (2008).

    The prior is informative: `alpha_k = prior_scale * (y_a + y_b)_k`, the
    pooled count of the 2 groups. With `prior_scale = 1` this is the common
    "background = pooled corpus" form.

    `delta_k = log((y_a + a) / (n_a + a0 - y_a - a))
             - log((y_b + a) / (n_b + a0 - y_b - a))`,
    `var_k = 1 / (y_a + a) + 1 / (y_b + a)`, `z_k = delta_k / sqrt(var_k)`.
    `n` is the sum of the counts of a group over the keys, and `a0` the sum
    of `alpha`.

    A positive z: group A uses the key more, as a share of its counts. Equal
    shares give z = 0 exactly, because the prior has the pooled share.

    **Warning: the share is of the key-trace pairs, not of the traces.** A
    judge whose traces hold more distinct keys has a larger `n`. Thus the z
    asks which keys take a larger part of a judge's vocabulary, and a long
    trace does not win every key. Read the document frequency for the plain
    rate.

    Args:
        y_a, y_b: The counts of each key for the 2 groups, aligned arrays.
        prior_scale: The weight of the prior relative to the pooled counts.

    Returns:
        `(delta, var, z)` as arrays. A key that neither group uses gives NaN.
    """
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    alpha = prior_scale * (y_a + y_b)
    a0, n_a, n_b = alpha.sum(), y_a.sum(), y_b.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = (np.log((y_a + alpha) / (n_a + a0 - y_a - alpha))
                 - np.log((y_b + alpha) / (n_b + a0 - y_b - alpha)))
        var = 1.0 / (y_a + alpha) + 1.0 / (y_b + alpha)
        z = delta / np.sqrt(var)
    bad = (y_a + y_b) <= 0
    delta[bad], var[bad], z[bad] = np.nan, np.nan, np.nan
    return delta, var, z


def score_table(docs: pd.DataFrame, prior_scale: float = 1.0) -> pd.DataFrame:
    """The judge contrast inside each (case, class, extractor).

    Group A is Gemma-4-12B and group B is Qwen3.5-9B, thus a positive `z` is
    a key of the gemma judge. The keys are the union of the 2 judges in the
    cell, with 0 for a key that 1 judge never uses.

    Returns:
        1 row for each (case, class, extractor, key): `docs_gemma`,
        `docs_qwen`, `df_gemma`, `df_qwen`, `df_ratio` (gemma / qwen, inf
        when qwen is 0), `docs_pooled`, `delta`, `z`.
    """
    ga, qw = JUDGES
    rows = []
    for (case, cls, ex), part in docs.groupby(["case", "extraction_class",
                                               "extractor"], sort=False):
        wide = part.pivot_table(index="key", columns="judge", values="docs",
                                aggfunc="sum", fill_value=0)
        for j in JUDGES:
            if j not in wide.columns:
                wide[j] = 0
        n_tr = part.groupby("judge")["n_traces"].first()
        delta, _, z = log_odds_z(wide[ga].to_numpy(), wide[qw].to_numpy(),
                                 prior_scale)
        out = pd.DataFrame({
            "case": case, "extraction_class": cls, "extractor": ex,
            "key": wide.index.astype(str),
            "docs_gemma": wide[ga].to_numpy().astype(int),
            "docs_qwen": wide[qw].to_numpy().astype(int),
            "n_traces_gemma": n_tr.get(ga, np.nan),
            "n_traces_qwen": n_tr.get(qw, np.nan),
            "delta": delta, "z": z,
        })
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    sc = pd.concat(rows, ignore_index=True)
    sc["df_gemma"] = sc["docs_gemma"] / sc["n_traces_gemma"]
    sc["df_qwen"] = sc["docs_qwen"] / sc["n_traces_qwen"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sc["df_ratio"] = sc["df_gemma"] / sc["df_qwen"]
    sc["docs_pooled"] = sc["docs_gemma"] + sc["docs_qwen"]
    return sc


# --------------------------------------------------------------------------
# Strict uniqueness, robustness, and agreement
# --------------------------------------------------------------------------

def strict_unique(scores: pd.DataFrame, min_traces: int = 10) -> pd.DataFrame:
    """The keys that 1 judge uses in `min_traces` traces or more and the
    other judge in none, under 1 extractor.

    `both_extractors` is True when the key is strictly unique to the same
    judge under the 2 extractors.

    **Warning: strict uniqueness picks rare keys.** A key in 12 traces of 1
    judge and 0 of the other passes; a key in 400 traces against 3 does not.
    Rank by the z of `robust_table`, and read this list as a side note.
    """
    s = scores
    g = (s["docs_gemma"] >= min_traces) & (s["docs_qwen"] == 0)
    q = (s["docs_qwen"] >= min_traces) & (s["docs_gemma"] == 0)
    out = s[g | q].copy()
    out["judge"] = np.where(g[g | q], JUDGES[0], JUDGES[1])
    out["docs_judge"] = np.where(out["judge"] == JUDGES[0], out["docs_gemma"],
                                 out["docs_qwen"])
    n_ex = (out.groupby(["case", "extraction_class", "key", "judge"])
            ["extractor"].transform("nunique"))
    out["both_extractors"] = n_ex == len(EXTRACTORS)
    cols = ["case", "extraction_class", "extractor", "judge", "key",
            "docs_judge", "df_gemma", "df_qwen", "z", "both_extractors"]
    return (out[cols].sort_values(["extraction_class", "case", "extractor",
                                   "judge", "docs_judge"],
                                  ascending=[True, True, True, True, False])
            .reset_index(drop=True))


def robust_table(scores: pd.DataFrame, min_traces: int = 10,
                 z_crit: float = Z_CRIT) -> pd.DataFrame:
    """Put the 2 extractors side by side for each (case, class, key).

    Columns, with `<x>` = `q27` (Qwen3.8-27B) or `g31` (Gemma-4-31B):

    - `z_<x>`, `df_gemma_<x>`, `df_qwen_<x>`, `docs_pooled_<x>`.
    - `seen_<x>`: the 2 judges together use the key in `min_traces` traces
      or more under that extractor (the floor).
    - `sig_<x>`: seen, and |z| >= `z_crit`.
    - `robust`: seen under both, the 2 z have the same sign, and both
      |z| >= `z_crit`. This is the judge-distinctive flag.
    - `side`: the judge that the mean z points to.
    - `z_mean`: the mean of the 2 z, with a missing z as 0. The figure and
      the tables rank by it, thus a key of 1 extractor only ranks low.
    - `z_min`: the z of smaller size when the 2 signs agree, else 0.
    - `df_gemma`, `df_qwen`: the mean document frequency over the 2
      extractors (over the extractors that see the key).
    - `rate_agrees`: the `side` judge also has the higher document
      frequency under both extractors.

    **Warning: a robust key can have the LOWER per-trace rate.** The z
    compares shares of each judge's key-trace pairs. The qwen traces hold
    1.4x to 2.1x the distinct keys of the gemma traces, thus a key that both
    judges use in almost every trace (`maintenance`, 88% against 96%) is a
    larger share of the gemma vocabulary and takes a positive z. Read
    `rate_agrees` before you call a key "used more" by a judge.

    **Warning: the flag is per key, with no correction for the number of
    keys.** Under the null, about 5% of the keys pass 1 extractor. To pass
    both with the same sign is rarer, but not a family-wise test.
    """
    keep = ["z", "df_gemma", "df_qwen", "docs_pooled", "docs_gemma", "docs_qwen"]
    parts = []
    for ex in EXTRACTORS:
        p = scores[scores["extractor"] == ex].set_index(
            ["case", "extraction_class", "key"])[keep]
        p.columns = [f"{c}_{EX_TAG[ex]}" for c in p.columns]
        parts.append(p)
    w = parts[0].join(parts[1], how="outer").reset_index()
    a, b = (EX_TAG[e] for e in EXTRACTORS)
    for t in (a, b):
        for c in ("docs_pooled", "docs_gemma", "docs_qwen"):
            w[f"{c}_{t}"] = w[f"{c}_{t}"].fillna(0).astype(int)
        w[f"seen_{t}"] = w[f"docs_pooled_{t}"] >= min_traces
        w[f"sig_{t}"] = w[f"seen_{t}"] & (w[f"z_{t}"].abs() >= z_crit)
    za, zb = w[f"z_{a}"], w[f"z_{b}"]
    same = np.sign(za) == np.sign(zb)
    w["robust"] = w[f"sig_{a}"] & w[f"sig_{b}"] & same
    w["z_mean"] = (za.fillna(0) + zb.fillna(0)) / 2.0
    w["z_min"] = np.where(same & za.notna() & zb.notna(),
                          np.sign(za) * np.minimum(za.abs(), zb.abs()), 0.0)
    w["side"] = np.where(w["z_mean"] > 0, JUDGES[0],
                         np.where(w["z_mean"] < 0, JUDGES[1], ""))
    w["df_gemma"] = w[[f"df_gemma_{a}", f"df_gemma_{b}"]].mean(axis=1)
    w["df_qwen"] = w[[f"df_qwen_{a}", f"df_qwen_{b}"]].mean(axis=1)
    # The per-trace rate of the side judge is higher under both extractors.
    g_more = ((w[f"df_gemma_{a}"].fillna(0) > w[f"df_qwen_{a}"].fillna(0))
              & (w[f"df_gemma_{b}"].fillna(0) > w[f"df_qwen_{b}"].fillna(0)))
    q_more = ((w[f"df_qwen_{a}"].fillna(0) > w[f"df_gemma_{a}"].fillna(0))
              & (w[f"df_qwen_{b}"].fillna(0) > w[f"df_gemma_{b}"].fillna(0)))
    w["rate_agrees"] = np.where(w["side"] == JUDGES[0], g_more,
                                np.where(w["side"] == JUDGES[1], q_more, False))
    return w


def breadth(docs: pd.DataFrame) -> pd.DataFrame:
    """The distinct keys of 1 class in 1 trace, on average, for each class,
    extractor, and judge (the mean over the cases).

    This is the `n` of the log-odds, divided by the traces. A judge with a
    larger value spreads its document counts over more keys, thus a key of
    the same per-trace rate takes a smaller share of its vocabulary.
    """
    per = (docs.groupby(["extraction_class", "extractor", "judge", "case"])
           .agg(docs=("docs", "sum"), n_traces=("n_traces", "first")))
    per["keys_per_trace"] = per["docs"] / per["n_traces"]
    out = (per["keys_per_trace"].groupby(["extraction_class", "extractor", "judge"])
           .mean().unstack("judge").reset_index())
    out["ratio_qwen_gemma"] = out[JUDGES[1]] / out[JUDGES[0]]
    return out


def agreement(robust: pd.DataFrame) -> pd.DataFrame:
    """How much of the judge contrast survives the extractor swap.

    1 row for each (case, class):

    - `n_both`: the keys over the floor under both extractors.
    - `spearman`: the rank correlation of the 2 z over those keys.
    - `same_sign`: the share of those keys whose 2 z have 1 sign.
    - `sig_q27`, `sig_g31`: the significant keys under each extractor.
    - `robust_gemma`, `robust_qwen`: the robust keys of each judge.
    - `survive`: robust keys / keys significant under 1 extractor or more.
    """
    a, b = (EX_TAG[e] for e in EXTRACTORS)
    rows = []
    for (case, cls), p in robust.groupby(["case", "extraction_class"]):
        both = p[p[f"seen_{a}"] & p[f"seen_{b}"]]
        rho = (both[f"z_{a}"].corr(both[f"z_{b}"], method="spearman")
               if len(both) >= 3 else np.nan)
        same = (np.sign(both[f"z_{a}"]) == np.sign(both[f"z_{b}"])).mean() \
            if len(both) else np.nan
        sig_any = int((p[f"sig_{a}"] | p[f"sig_{b}"]).sum())
        rob = p[p["robust"]]
        rows.append({
            "case": case, "extraction_class": cls, "n_both": len(both),
            "spearman": rho, "same_sign": same,
            f"sig_{a}": int(p[f"sig_{a}"].sum()), f"sig_{b}": int(p[f"sig_{b}"].sum()),
            "robust_gemma": int((rob["side"] == JUDGES[0]).sum()),
            "robust_qwen": int((rob["side"] == JUDGES[1]).sum()),
            "survive": len(rob) / sig_any if sig_any else np.nan,
        })
    out = pd.DataFrame(rows)
    return _case_sort(out)


def cross_case(robust: pd.DataFrame, min_cases: int = 3) -> pd.DataFrame:
    """The robust keys of each judge, counted over the cases.

    1 row for each (class, judge, key) that is robust in 1 case or more:
    `n_cases`, `cases` (in case order), `n_rate_agrees` (the cases where
    the per-trace rate agrees too), `z_mean` (the mean over those
    cases), `df_judge` and `df_other` (the mean document frequencies), and
    `scope`: `general` at `min_cases` cases or more, `case-specific` in 1
    case, else `few`.
    """
    rob = robust[robust["robust"]].copy()
    if rob.empty:
        return pd.DataFrame(columns=["extraction_class", "judge", "key",
                                     "n_cases", "cases", "z_mean", "df_judge",
                                     "df_other", "scope"])
    rob["judge"] = rob["side"]
    rob["df_judge"] = np.where(rob["judge"] == JUDGES[0], rob["df_gemma"], rob["df_qwen"])
    rob["df_other"] = np.where(rob["judge"] == JUDGES[0], rob["df_qwen"], rob["df_gemma"])
    order = {c: i for i, c in enumerate(CASES)}
    g = rob.groupby(["extraction_class", "judge", "key"])
    out = g.agg(n_cases=("case", "nunique"), z_mean=("z_mean", "mean"),
                n_rate_agrees=("rate_agrees", "sum"),
                df_judge=("df_judge", "mean"), df_other=("df_other", "mean"))
    out["cases"] = g["case"].apply(
        lambda s: ", ".join(sorted(set(s), key=lambda c: order.get(c, 99))))
    out = out.reset_index()
    out["scope"] = np.where(out["n_cases"] >= min_cases, "general",
                            np.where(out["n_cases"] == 1, "case-specific", "few"))
    out["abs_z"] = out["z_mean"].abs()
    out = out.sort_values(["extraction_class", "judge", "n_cases", "abs_z"],
                          ascending=[True, True, False, False])
    return out.drop(columns="abs_z").reset_index(drop=True)


def summary(robust: pd.DataFrame, agree: pd.DataFrame,
            cross: pd.DataFrame, min_cases: int = 3) -> pd.DataFrame:
    """1 row for each class: the robust (case, key) pairs of each judge
    (`robust_<judge>`), those whose per-trace rate agrees
    (`robust_rate_<judge>`), the keys that are general (`min_cases` cases or more), the significant
    pairs under each extractor, and the median Spearman over the cases."""
    a, b = (EX_TAG[e] for e in EXTRACTORS)
    rows = []
    for cls in CLASSES:
        r = robust[robust["extraction_class"] == cls]
        ag = agree[agree["extraction_class"] == cls]
        cc = cross[cross["extraction_class"] == cls]
        row = {"extraction_class": cls,
               f"sig_{a}": int(r[f"sig_{a}"].sum()), f"sig_{b}": int(r[f"sig_{b}"].sum())}
        for j, tag in zip(JUDGES, ("gemma", "qwen")):
            row[f"robust_{tag}"] = int((r["robust"] & (r["side"] == j)).sum())
            row[f"robust_rate_{tag}"] = int((r["robust"] & (r["side"] == j)
                                             & r["rate_agrees"]).sum())
            row[f"general_{tag}"] = int(((cc["judge"] == j)
                                         & (cc["n_cases"] >= min_cases)).sum())
        row["survive"] = (row["robust_gemma"] + row["robust_qwen"]) / max(
            1, int((r[f"sig_{a}"] | r[f"sig_{b}"]).sum()))
        row["spearman_median"] = ag["spearman"].median()
        row["spearman_min"] = ag["spearman"].min()
        row["spearman_max"] = ag["spearman"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def top_keys(robust: pd.DataFrame, cls: str, k: int = 5,
             robust_only: bool = False, min_traces: int = 10) -> pd.DataFrame:
    """The top `k` keys of each judge in each case of 1 class, by `z_mean`.

    A key needs the floor under 1 extractor or more. `robust_only` keeps
    the judge-distinctive keys only (the LaTeX table).
    """
    a, b = (EX_TAG[e] for e in EXTRACTORS)
    r = robust[(robust["extraction_class"] == cls)
               & (robust[f"seen_{a}"] | robust[f"seen_{b}"])]
    if robust_only:
        r = r[r["robust"]]
    out = []
    for case, p in r.groupby("case"):
        out.append(p[p["z_mean"] > 0].nlargest(k, "z_mean"))
        out.append(p[p["z_mean"] < 0].nsmallest(k, "z_mean"))
    if not out:
        return r.iloc[0:0]
    return _case_sort(pd.concat(out, ignore_index=True), extra=["side"])


def _case_sort(df: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    if df.empty:
        return df
    order = {c: i for i, c in enumerate(CASES)}
    cls_order = {c: i for i, c in enumerate(CLASSES)}
    d = df.assign(_c=df["case"].map(order), _k=df["extraction_class"].map(cls_order))
    return (d.sort_values(["_k", "_c", *extra], kind="stable")
            .drop(columns=["_c", "_k"]).reset_index(drop=True))


def build_bundle(docs: pd.DataFrame, min_traces: int = 10, min_cases: int = 3,
                 prior_scale: float = 1.0) -> Dict[str, pd.DataFrame]:
    """Run every step on the document counts of `load_grid`."""
    scores = score_table(docs, prior_scale)
    rob = robust_table(scores, min_traces)
    agree = agreement(rob)
    cross = cross_case(rob, min_cases)
    return {"docs": docs, "scores": scores, "robust": rob,
            "unique": strict_unique(scores, min_traces), "agreement": agree,
            "breadth": breadth(docs),
            "cross_case": cross, "summary": summary(rob, agree, cross, min_cases)}


def coverage(docs: pd.DataFrame) -> pd.DataFrame:
    """The traces and the distinct keys of each arm and class."""
    tr = (docs.groupby(["arm", "judge", "extractor", "case"])["n_traces"].first()
          .groupby(["arm", "judge", "extractor"]).sum().rename("traces"))
    keys = (docs.groupby(["arm", "judge", "extractor", "extraction_class"])["key"]
            .nunique().unstack())
    keys.columns = [f"keys_{c}" for c in keys.columns]
    return pd.concat([tr, keys], axis=1).reset_index()


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------

_LABEL_CHARS = 26


def _short(key: str, dagger: bool = False) -> str:
    """Cut a long key with an ellipsis, and add a dagger when asked."""
    out = key if len(key) <= _LABEL_CHARS else key[:_LABEL_CHARS - 1] + "…"
    return out + (" †" if dagger else "")


def plot_class(robust: pd.DataFrame, cls: str, k: int = 5,
               min_traces: int = 10):
    """1 class: a dot plot of the top `k` keys of each judge in each case.

    7 panels (1 for each case) and a legend panel. A key is 1 row. The 2
    marks of a row are the z under the 2 extractors (circle: Qwen3.8-27B,
    square: Gemma-4-31B), joined by a grey line. The colour is the judge
    that the mean z points to. A filled mark is a robust key (|z| >= 1.96
    with 1 sign under both extractors); an open mark is not. The x scale is
    common to the 7 panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    a, b = (EX_TAG[e] for e in EXTRACTORS)
    top = top_keys(robust, cls, k=k, min_traces=min_traces)
    cases = [c for c in CASES if c in set(top["case"])]
    zs = pd.concat([top[f"z_{a}"], top[f"z_{b}"]]).dropna()
    lim = max(3.0, float(zs.abs().max()) * 1.08) if len(zs) else 3.0
    n_rows = 2 * k
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 0.30 + 0.16 * n_rows * 2 + 0.6),
                             sharex=True)
    axes = axes.ravel()
    for ax, case in zip(axes, cases):
        p = top[top["case"] == case].sort_values("z_mean")
        y = np.arange(len(p))
        for yi, (_, r) in zip(y, p.iterrows()):
            col = JUDGE_COLOUR.get(r["side"], S.PAL["warmgrey"])
            vals = [r[f"z_{t}"] for t in (a, b)]
            ok = [v for v in vals if pd.notna(v)]
            if len(ok) == 2:
                ax.plot(ok, [yi, yi], color="#b5b0a8", lw=0.6, zorder=1)
            for ex, v in zip(EXTRACTORS, vals):
                if pd.isna(v):
                    continue
                ax.scatter([v], [yi], marker=EX_MARKER[ex], s=13, zorder=3,
                           facecolors=col if r["robust"] else "white",
                           edgecolors=S.EDGE if r["robust"] else col,
                           linewidths=0.4 if r["robust"] else 0.9)
        ax.axvline(0, color=S.EDGE, lw=0.6, zorder=0)
        for s in (-Z_CRIT, Z_CRIT):
            ax.axvline(s, color="#9c948a", lw=0.5, ls=(0, (2, 2)), zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([_short(t, not ra) for t, ra in
                            zip(p["key"], p["rate_agrees"])], fontsize=5.6)
        ax.set_ylim(-0.7, len(p) - 0.3)
        ax.set_xlim(-lim, lim)
        ax.set_title(WA.case_label(case), fontsize=7.5, pad=3)
        ax.grid(visible=False)
        for _sp in ("left", "bottom"):
            ax.spines[_sp].set_linewidth(0.6)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.tick_params(axis="x", labelsize=6, width=0.6, length=2.5)
    for ax in axes[len(cases):]:
        ax.axis("off")
    leg_ax = axes[len(cases)] if len(cases) < 8 else None
    handles = [
        Line2D([], [], marker="o", ls="", ms=4, markerfacecolor=JUDGE_COLOUR[JUDGES[0]],
               markeredgecolor=S.EDGE, markeredgewidth=0.4,
               label=f"{JUDGE_SHORT[JUDGES[0]]} key"),
        Line2D([], [], marker="o", ls="", ms=4, markerfacecolor=JUDGE_COLOUR[JUDGES[1]],
               markeredgecolor=S.EDGE, markeredgewidth=0.4,
               label=f"{JUDGE_SHORT[JUDGES[1]]} key"),
        Line2D([], [], marker="o", ls="", ms=4, markerfacecolor="#7a6f63",
               markeredgecolor=S.EDGE, markeredgewidth=0.4,
               label="filled: holds under both extractors"),
        Line2D([], [], marker="o", ls="", ms=4, markerfacecolor="white",
               markeredgecolor="#7a6f63", markeredgewidth=0.9,
               label="open: does not hold"),
        Line2D([], [], marker=EX_MARKER[EXTRACTORS[0]], ls="", ms=4,
               markerfacecolor="white", markeredgecolor=S.EDGE, markeredgewidth=0.6,
               label=f"extractor {EXTRACTOR_SHORT[EXTRACTORS[0]]}"),
        Line2D([], [], marker=EX_MARKER[EXTRACTORS[1]], ls="", ms=4,
               markerfacecolor="white", markeredgecolor=S.EDGE, markeredgewidth=0.6,
               label=f"extractor {EXTRACTOR_SHORT[EXTRACTORS[1]]}"),
        Line2D([], [], color="#9c948a", lw=0.6, ls=(0, (2, 2)),
               label="|z| = 1.96"),
    ]
    if leg_ax is not None:
        leg_ax.legend(handles=handles, loc="center", frameon=False,
                      fontsize=6, handletextpad=0.4, borderaxespad=0)
    fig.supxlabel("log-odds z of the judge contrast, the extractor held fixed"
                  "   (← Qwen3.5-9B key  |  Gemma-4-12B key →)",
                  fontsize=7)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    n_rob = int(top["robust"].sum())
    caption = (
        f"The {cls} keys that most separate the 2 judges in each case. A key is "
        f"the normalised free name of a {cls} span. 1 row is 1 key: the top "
        f"{k} keys of each judge by the mean log-odds z over the 2 extractors "
        "(Monroe et al. 2008, informative Dirichlet prior from the pooled "
        "counts of both judges in the case and extractor), on trace-level "
        "document counts. The 2 marks of a row are the z under the "
        "Qwen3.8-27B extractor (circle) and the Gemma-4-31B extractor "
        "(square); each z compares the 2 judges with the extractor held "
        "fixed. A positive z is a Gemma-4-12B key (blue), a negative z a "
        "Qwen3.5-9B key (amber). A filled mark is a key with |z| >= 1.96 and "
        "1 sign under both extractors: a fact about the judge. An open mark "
        "holds under 1 extractor or less. The dashed lines are |z| = 1.96. "
        "The z compares shares of each judge's vocabulary; a dagger marks a "
        "key whose per-trace rate is NOT higher for that judge under both "
        "extractors (the qwen traces hold more distinct keys, so a key common "
        "to both judges can be a larger share of the gemma vocabulary). "
        f"A key needs {min_traces} traces of the 2 judges together under 1 "
        f"extractor or more. {n_rob} of the {len(top)} rows shown are filled. "
        "About 1,000 traces for each case and judge, schema version 3, the "
        "canonical runs. A long label is cut with an ellipsis; the CSV holds "
        "the full key. The z has no correction for the number of keys.")
    meta = F.figure_meta(
        title=f"Judge-distinctive {cls} keys",
        caption=caption,
        label=F.fig_label("judge-vocab", cls),
        tags=["judge_vocabulary", "v3", cls, "log-odds z",
              "gemma-4-12b", "qwen3.5-9b", "extractor held fixed", *cases],
        module=_MOD, version=__version__)
    return F.attach(fig, meta)


# --------------------------------------------------------------------------
# The LaTeX tables
# --------------------------------------------------------------------------

def _key_cell(p: pd.DataFrame, own: str, other: str) -> str:
    if p.empty:
        return L.DASH
    return "; ".join(
        f"{L.escape(r['key'])}{'' if r['rate_agrees'] else '$^\\dagger$'} "
        f"({L.fmt_pct(r[own])}/{L.fmt_pct(r[other])})"
        for _, r in p.iterrows())


def latex_keys(robust: pd.DataFrame, cls: str, n: int = 5,
               min_traces: int = 10) -> str:
    """Table (a) for 1 class: the top `n` robust keys of each judge in each
    case, with the document frequencies (own / other judge, percent of the
    traces, the mean over the 2 extractors)."""
    top = top_keys(robust, cls, k=n, robust_only=True, min_traces=min_traces)
    body = []
    for case in CASES:
        p = top[top["case"] == case]
        g = p[p["side"] == JUDGES[0]].sort_values("z_mean", ascending=False)
        q = p[p["side"] == JUDGES[1]].sort_values("z_mean")
        body.append(L.row([L.case_label(case), _key_cell(g, "df_gemma", "df_qwen"),
                           _key_cell(q, "df_qwen", "df_gemma")]))
    head = [L.row(["Case", f"{JUDGE_SHORT[JUDGES[0]]} keys",
                   f"{JUDGE_SHORT[JUDGES[1]]} keys"])]
    caption = (
        rf"\textbf{{Which {cls} keys does 1 judge use that the other does not?}} "
        f"The top {n} keys of each judge in each case that hold under both "
        "extractors: the log-odds z (informative Dirichlet prior, trace-level "
        "document counts) has 1 sign and $|z| \\geq 1.96$ under Qwen3.8-27B "
        "and under Gemma-4-31B, each comparing the 2 judges with the "
        "extractor held fixed. Keys in the order of the mean z. In brackets: "
        "the percent of the traces of the judge that use the key, then the "
        "percent for the other judge (the mean over the 2 extractors). "
        "$^\\dagger$: the per-trace rate of the judge is not higher under "
        "both extractors; the key is a larger share of the judge's vocabulary "
        "only, because the Qwen3.5-9B traces hold more distinct keys. "
        f"A dash: no key holds. A key needs {min_traces} traces under both "
        "extractors.")
    return L.table(_MOD, __version__, caption, f"tab:judge-vocab-{cls}",
                   r"p{0.12\linewidth}p{0.40\linewidth}p{0.40\linewidth}",
                   head, body, env="table*")


def latex_agreement(agree: pd.DataFrame, min_traces: int = 10) -> str:
    """Table (b): the extractor agreement of the judge contrast, per case and
    class: the Spearman of the 2 extractors' z, and the robust keys of each
    judge (gemma / qwen)."""
    groups = [(c.capitalize(), 3) for c in CLASSES]
    head = L.group_header(groups) + [
        L.row(["Case"] + ["$n$", r"$\rho$", "G/Q"] * len(CLASSES))]
    body = []
    for case in CASES:
        cells = [L.case_label(case)]
        for cls in CLASSES:
            r = agree[(agree["case"] == case) & (agree["extraction_class"] == cls)]
            if r.empty:
                cells += [L.DASH] * 3
                continue
            r = r.iloc[0]
            cells += [L.fmt_int(r["n_both"]), L.fmt(r["spearman"]),
                      f"{int(r['robust_gemma'])}/{int(r['robust_qwen'])}"]
        body.append(L.row(cells))
    body.append("\\midrule")
    cells = ["Median"]
    for cls in CLASSES:
        r = agree[agree["extraction_class"] == cls]
        cells += [L.fmt_int(r["n_both"].median()), L.fmt(r["spearman"].median()),
                  f"{int(r['robust_gemma'].sum())}/{int(r['robust_qwen'].sum())}"]
    body.append(L.row(cells))
    caption = (
        r"\textbf{Does the judge contrast survive the swap of the extractor?} "
        r"$\rho$: the Spearman correlation, over the $n$ keys that both "
        f"extractors see ({min_traces} traces or more), of the log-odds z of "
        "Gemma-4-12B against Qwen3.5-9B under Qwen3.8-27B and under "
        "Gemma-4-31B. G/Q: the keys that hold under both extractors for "
        "Gemma-4-12B and for Qwen3.5-9B. The last row gives the median $n$ "
        r"and $\rho$, and the sum of G/Q.")
    return L.table(_MOD, __version__, caption, "tab:judge-vocab-agreement",
                   "l" + "rrr" * len(CLASSES), head, body, resize=True)


# --------------------------------------------------------------------------
# The export
# --------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent / "master" / "outputs" / "judge_vocab"
TAG = "judge_vocab"


def export(bundle: Dict[str, pd.DataFrame], out_dir: Path = OUT_DIR,
           version: str = "1.0.0", gate_ok: bool = True, k: int = 5,
           n_tex: int = 5, min_traces: int = 10) -> List[str]:
    """Write the figures (PNG and PDF, each with its `.json` sidecar), the
    LaTeX tables, and the CSVs. Give the written file names.

    A run with the gate off writes every file with a `_stale` suffix.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if gate_ok else "_stale"
    written: List[str] = []
    for cls in CLASSES:
        fig = plot_class(bundle["robust"], cls, k=k, min_traces=min_traces)
        for ext in ("png", "pdf"):
            path = out_dir / f"{TAG}_{cls}{sfx}.{ext}"
            # The PNG and the PDF share 1 stem, thus 1 sidecar.
            F.save_figure(fig, path, provenance={"notebook_version": version,
                                                 "gate_ok": gate_ok})
            written.append(path.name)
        written.append(F.sidecar_path(out_dir / f"{TAG}_{cls}{sfx}.png").name)
        plt.close(fig)
        name = f"{TAG}_{cls}_keys{sfx}_v{version}.tex"
        (out_dir / name).write_text(latex_keys(bundle["robust"], cls, n=n_tex,
                                               min_traces=min_traces))
        written.append(name)
    name = f"{TAG}_agreement{sfx}_v{version}.tex"
    (out_dir / name).write_text(latex_agreement(bundle["agreement"],
                                                       min_traces))
    written.append(name)
    for key in ("scores", "robust", "unique", "agreement", "cross_case",
                "summary", "breadth"):
        name = f"{TAG}_{key}{sfx}_v{version}.csv"
        bundle[key].to_csv(out_dir / name, index=False)
        written.append(name)
    return written
