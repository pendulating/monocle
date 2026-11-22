"""The 4 channels of the trace that the cue figures do not read.

`_cues_across` reads 1 extraction class, `visual_evidence`. That class is 46%
of the corpus. This module reads the other 4 classes that carry a measurable
signal, and each one answers a question the cue class cannot.

| Channel | Class | Question |
|---------|-------|----------|
| Axes | `comparison` | Which axis does the model decide ON? |
| Abstention | `uncertainty` | Where does a `NotSure` come from? |
| Inference | `inference` | What does the model read INTO a photograph? |
| Artifact | `image_artifact` | Does the camera move the judgment? |

The axes
--------
A `comparison` span carries a free phrase, `dimension`, and a `direction` that
names the winning image. Thus the model writes the axis of the comparison in
its own words: `smoothness`, `standardness`, `vibe`, `visual interest`. There
are 64,943 distinct phrases, and they are a second vocabulary beside the cue
words. The cue says WHAT the model saw; the axis says what it compared on.

`agree` is the share of the decided comparisons where the final label names
the image the axis named.

**Warning: `agree` measures polarity first, and binding second.** `direction`
names the image with MORE of the quality, not the image that wins. Thus a
negative axis word such as `wear` reads 0.09 and a positive one such as
`maintenance` reads 0.89, and both of them bind the vote hard. Read `bind`,
which is `|agree - 0.5| * 2`, for the strength, and `polarity` for the sign.
An axis near 0.5 is the interesting one: the model names it and then does not
decide on it.

`hedged` is a second free marker. A hedged comparison agrees with the final
label 0.478 of the time against 0.615 for a plain one, thus the model's own
hedge predicts that THIS comparison will not carry the decision.

**Warning: the marker works within an axis, not between axes.** The Spearman
of `hedge` against `bind` over the axes is +0.18, thus a heavily hedged axis
is not a weak axis. Read `hedge_cost`, which is the drop inside 1 axis.

The abstention
--------------
`uncertainty` names a reason (`cannot_see`, `out_of_view`, `looks_equal`,
`other`) and, under version 3, a marker (`qualifier`, `contrastive`,
`explicit_doubt`, `reconsideration`). The COUNT of the uncertainty spans in a
trace separates an abstention from a decision. On the version 3 primary arms
(2026-09-30) the AUC is 0.62 to 0.93 for the gemma judge and 0.79 to 0.85
for the qwen judge, in the cases with 25 abstentions or more. The
`cannot_see` and `explicit_doubt` counts separate better than the total, and
the `reconsideration` count separates worse. `abstention_across_arms` reads
the 4 arms of the 2 x 2.

The inference and the artifact
------------------------------
Both classes name an image, thus both take the same directional test as a cue.
`keyed_lift` in `_cues_across` holds that test, and this module calls it.

Warning: this is an association, not a cause
--------------------------------------------
The model narrates while it decides, thus a span can follow the judgment
rather than drive it. Read a lift as "what goes with a win".

See `research/docs/trace-channels.md`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _cues_across as CA
import _extractions as X
import _figmeta as F
import _latex as L
import _style as S
import _words_across as WA
# Re-exported, thus a notebook can call `TC.caption_md(fig)`.
from _figmeta import caption_md, captioned, save_figure  # noqa: F401

__version__ = "1.3.0"

AXIS_CLASS = "comparison"
UNCERTAINTY_CLASS = "uncertainty"
INFERENCE_CLASS = "inference"
ARTIFACT_CLASS = "image_artifact"

#: The inference kinds that carry the risk result. The model goes past the
#: pixels when it writes one of these.
RISK_KINDS = ("wealth", "class", "crime", "demographic")

#: The row order of the inference ladder. The risk kinds come first.
INFERENCE_ORDER = ("wealth", "class", "crime", "demographic",
                   "quality", "safety", "upkeep", "other")

ARTIFACT_ORDER = ("occlusion", "angle", "blur", "weather", "time_of_day",
                  "other")

#: The kinds too small to clear the word floor. `crime` holds 454 spans of
#: the 302,784 in the class, and its phrasing splits over `high crime area`,
#: `higher crime rate`, and `higher crime`. A floor tuned for `other`, which
#: carries 135 keys, erases it.
#:
#: The shrink does the work the floor was doing. A key at 8 votes is pulled
#: most of the way to 0 by `prior_strength`, thus a low floor here adds rows
#: without adding claims.
RARE_KINDS = ("crime", "demographic")
RARE_MIN_COUNT = 8

REASON_ORDER = ("cannot_see", "out_of_view", "looks_equal", "other")

#: The version 3 markers of an uncertainty span: what kind of doubt it
#: carries. `reconsideration` is the self-check ("Wait, let me look again"),
#: which version 1 and version 2 mostly did not mark.
MARKER_ORDER = ("qualifier", "contrastive", "explicit_doubt",
                "reconsideration")

#: The columns that name the arm in a long frame of `abstention_across_arms`.
ARM_COLUMNS = ("arm", "role", "judge", "extractor", "schema_version")

#: The 2 poles of an inference, in the model's own words.
#:
#: `kind` names the TOPIC of an inference, never its direction. `wealth` is
#: the same key whether the model wrote `upscale` or `lower-income area`, thus
#: `kind` alone cannot say WHICH way the inference runs. These 2 lists read
#: the direction off the span text.
#:
#: Both lists are short and hand-built. They do not try to cover the corpus:
#: they cover the words the model repeats. A span that matches neither takes
#: `none`, and a span that matches both takes `mixed`. Read `pole_mix` for
#: how much each kind leaves undetermined.
#:
#: Warning: `public school` is on the LOW list because the model uses it as
#: its own contrast to `private school`, and 4 of the 5 most frequent
#: low-pole `class` spans are `standard public school` and its variants. It
#: is the model's class vocabulary, not a claim about a school.
_INFERENCE_HIGH = re.compile(
    r"\b(high end|highend|higher end|upscale|upmarket|affluent|wealthy"
    r"|wealthier|rich|expensive|prestigious|elite|exclusive|premium|luxury"
    r"|luxurious|well funded|better funded|well resourced|more resource"
    r"|private school|charter school|prosperous|posh|gentrified"
    r"|nicer neighborhood|desirable|majestic|impressive)\b")

_INFERENCE_LOW = re.compile(
    r"\b(low income|lower income|poor|poorer|poverty|underfunded"
    r"|under funded|deprived|disadvantaged|working class|humble|struggling"
    r"|neglected|run down|rundown|cheap|modest|blighted|gritty|industrial"
    r"|standard public|typical public|public school|ordinary|utilitarian"
    r"|no frill|crime|criminal|vandalism|mugging|pickpocket|robbery"
    r"|loitering|drug)\b")

#: A word that turns a pole word around. `less upscale` is not upscale, and
#: `no visible crime` is not crime.
#:
#: Warning: the flip fires only when a negator sits in the 2 tokens BEFORE
#: the pole word. A blanket search over the whole span would break
#: `lower income area`, where `lower` is part of the pole word itself, and
#: `less safe due to poor visibility`, where `less` belongs to `safe` and not
#: to `poor`. The narrow window misses a long-range negation such as
#: `less likely to have crime`; those stay on their plain pole.
_NEGATORS = frozenset((
    "no", "not", "never", "less", "lower", "little", "without", "free",
    "fewer", "reduced", "lack", "lacking", "absent", "minimal", "away",
))

#: How many tokens before a pole word the negator may sit.
_NEGATION_WINDOW = 2

POLE_ORDER = ("high", "low", "mixed", "none")

POLE_COLOUR = {"high": S.PAL["teal"], "low": S.PAL["coral"],
               "mixed": S.PAL["amber"], "none": S.PAL["warmgrey"]}

#: A span whose whole text is 1 of these restates the vote. It infers nothing.
_LABEL_WORDS = frozenset((
    "less", "more", "same", "muchless", "muchmore", "notsure", "a", "b",
))

#: A span that names the other photograph, or the label itself, restates the
#: vote. `much less rather eat at A than B` is the 2nd most frequent `wealth`
#: span of the corpus, and it is not an inference.
_RESTATEMENT = re.compile(
    r"\brelative to\b|\bthan [ab]\b|\bmuchless\b|\bmuchmore\b"
    r"|\bnotsure\b")

REASON_LABEL = {
    "cannot_see": "cannot see it",
    "out_of_view": "out of view",
    "looks_equal": "the 2 look equal",
    "other": "other",
}

REASON_COLOUR = {
    "cannot_see": S.PAL["blue"],
    "out_of_view": S.PAL["periwinkle"],
    "looks_equal": S.PAL["amber"],
    "other": S.PAL["warmgrey"],
}

#: Values of `dimension` that name no axis. The model wrote each of them.
_AXIS_STOP = frozenset((
    "", "none", "nan", "n/a", "na", "null", "other", "unclear", "unknown",
))

#: The bins of the dose curve. A count of uncertainty spans is heavily
#: skewed, thus a linear axis would put 90% of the traces in 1 column.
#: Version 3 marks about 10x the spans of version 1 (the median deciding
#: trace holds 6 to 31 spans, by arm), thus the bins are wider than the
#: version 1 bins, which stay in `DOSE_EDGES_V1`.
DOSE_EDGES = (-0.5, 2.5, 5.5, 10.5, 20.5, 40.5, np.inf)
DOSE_LABELS = ("0-2", "3-5", "6-10", "11-20", "21-40", "41+")
DOSE_EDGES_V1 = (-0.5, 0.5, 1.5, 2.5, 4.5, 9.5, np.inf)
DOSE_LABELS_V1 = ("0", "1", "2", "3-4", "5-9", "10+")

#: The bins of the dose curve across arms: the quantiles of the count inside
#: 1 arm. The 2 extractors of a judge mark up to 2.5x different counts, thus
#: a fixed bin is not the same dose in 2 arms. A quantile is.
DOSE_QUANTILES = 5


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load(require_canonical: bool = True, root: Optional[str | Path] = None
         ) -> pd.DataFrame:
    """Read the extraction corpus. `_extractions.load` holds the gate."""
    return CA.load(require_canonical=require_canonical, root=root)


def trace_id(df: pd.DataFrame) -> pd.Series:
    """A key that names 1 trace.

    Warning: `pair_id` alone is not that key. The 7 cases draw from 1 pair
    table, thus the same `pair_id` appears once in each case, and a group by
    `pair_id` pools 7 traces into 1. Every count here groups by `case` and
    `pair_id` together.
    """
    return df["case"].astype(str) + "|" + df["pair_id"].astype(str)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """What each channel holds: spans, spans for each trace, and the cases."""
    if df.empty:
        return pd.DataFrame()
    ids = trace_id(df)
    traces = ids.groupby(df["case"]).nunique()
    rows = []
    for name, cls in (("axes", AXIS_CLASS), ("abstention", UNCERTAINTY_CLASS),
                      ("inference", INFERENCE_CLASS),
                      ("artifact", ARTIFACT_CLASS)):
        part = df[df["extraction_class"] == cls]
        spans = part.groupby("case").size().reindex(traces.index).fillna(0)
        rows.append(pd.DataFrame({
            "channel": name,
            "class": cls,
            "case": traces.index,
            "case_label": [WA.case_label(c) for c in traces.index],
            "traces": traces.to_numpy(),
            "spans": spans.to_numpy().astype(int),
            "spans_per_trace": (spans.to_numpy() / traces.to_numpy()).round(2),
        }))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# 1. The axes the model compares on
# --------------------------------------------------------------------------

def axis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The comparison rows that name an axis AND a winning image.

    The same 2 rules as a cue vote apply. A trace that names 1 axis 3 times
    still describes 1 comparison, thus the rows are made distinct on
    (trace, axis, direction). An axis that the model gave to BOTH images of a
    trace separates nothing, thus it is dropped.
    """
    part = df[df["extraction_class"] == AXIS_CLASS].copy()
    if part.empty or "dimension" not in part.columns:
        return pd.DataFrame()
    axis = (part["dimension"].astype("string").str.lower().str.strip()
            .str.replace(r"\s+", " ", regex=True))
    part["axis"] = axis
    part = part[part["axis"].notna()]
    part = part[~part["axis"].isin(_AXIS_STOP)]
    part = part[part["direction"].isin(["A", "B"])]
    if part.empty:
        return pd.DataFrame()

    part = part.drop_duplicates(["case", "pair_id", "axis", "direction"])
    sides = (part.groupby(["case", "pair_id", "axis"])["direction"]
             .transform("nunique"))
    part = part[sides == 1].copy()

    label = part["presented_label"]
    part["decided"] = ~label.isin(CA.UNDECIDED)
    part["agrees"] = (
        ((part["direction"].eq("A") & label.isin(CA.WIN_FOR_A))
         | (part["direction"].eq("B") & label.isin(CA.WIN_FOR_B)))
        & part["decided"]
    )
    part["is_hedged"] = part["hedged"].eq("yes")
    return part.reset_index(drop=True)


def axis_table(df: pd.DataFrame, min_count: int = 400) -> pd.DataFrame:
    """1 row for each axis, pooled over the cases.

    Args:
        min_count: Drop an axis with fewer decided comparisons than this.

    Returns:
        `axis`, `n`, `n_named`, `n_cases`, `agree`, its Wilson interval,
        `bind`, `polarity`, `hedge`, `agree_plain`, `agree_hedged`, and
        `hedge_cost`.
    """
    part = axis_frame(df)
    if part.empty:
        return pd.DataFrame()
    decided = part[part["decided"]]
    grouped = decided.groupby("axis")
    table = pd.DataFrame({
        "n": grouped.size(),
        "agree": grouped["agrees"].mean(),
        "n_cases": grouped["case"].nunique(),
    })
    named = part.groupby("axis")["is_hedged"].agg(["size", "mean"])
    table["n_named"] = named["size"]
    table["hedge"] = named["mean"]
    split = decided.groupby(["axis", "is_hedged"])["agrees"].mean().unstack()
    table["agree_plain"] = split.get(False)
    table["agree_hedged"] = split.get(True)
    table["hedge_cost"] = table["agree_plain"] - table["agree_hedged"]

    table = table[table["n"] >= min_count]
    if table.empty:
        return pd.DataFrame()
    low, high = zip(*(X._wilson(int(round(a * n)), int(n))
                      for a, n in zip(table["agree"], table["n"])))
    table["ci_low"], table["ci_high"] = low, high
    # `bind` is the strength, and it does not care about the polarity of the
    # axis word. See the warning at the top of the module.
    table["bind"] = (table["agree"] - 0.5).abs() * 2
    # An axis whose interval covers chance has NO polarity. A bare `>= 0.5`
    # would name a side for `appearance`, which sits at 0.492 with an interval
    # from 0.452 to 0.532.
    table["polarity"] = np.where(
        (table["ci_low"] <= 0.5) & (table["ci_high"] >= 0.5), "none",
        np.where(table["agree"] >= 0.5, "positive", "negative"))
    return table.reset_index().sort_values("agree").reset_index(drop=True)


def axis_by_case(df: pd.DataFrame) -> pd.DataFrame:
    """The agreement and the hedge rate of each case, over every axis."""
    part = axis_frame(df)
    if part.empty:
        return pd.DataFrame()
    decided = part[part["decided"]]
    out = pd.DataFrame({
        "n_named": part.groupby("case").size(),
        "n": decided.groupby("case").size(),
        "agree": decided.groupby("case")["agrees"].mean(),
        "hedge": part.groupby("case")["is_hedged"].mean(),
        "axes": part.groupby("case")["axis"].nunique(),
    })
    split = decided.groupby(["case", "is_hedged"])["agrees"].mean().unstack()
    out["agree_plain"] = split.get(False)
    out["agree_hedged"] = split.get(True)
    out["hedge_cost"] = out["agree_plain"] - out["agree_hedged"]
    out["case_label"] = [WA.case_label(c) for c in out.index]
    return out.reset_index()


def hedge_summary(df: pd.DataFrame) -> Dict[str, float]:
    """The 3 headline numbers of the hedge marker."""
    part = axis_frame(df)
    if part.empty:
        return {}
    decided = part[part["decided"]]
    plain = decided[~decided["is_hedged"]]["agrees"]
    hedged = decided[decided["is_hedged"]]["agrees"]
    return {
        "n_comparisons": float(len(part)),
        "n_decided": float(len(decided)),
        "n_axes": float(part["axis"].nunique()),
        "agree_all": float(decided["agrees"].mean()),
        "agree_plain": float(plain.mean()) if len(plain) else float("nan"),
        "agree_hedged": float(hedged.mean()) if len(hedged) else float("nan"),
        "hedge_rate": float(part["is_hedged"].mean()),
    }


def hedge_bind_corr(table: pd.DataFrame) -> float:
    """Does the model hedge on the axes that do not bind the vote?

    A Spearman correlation between `hedge` and `bind`. It measures +0.18 on
    this corpus, thus the answer is no: a heavily hedged axis is not a weak
    axis. The marker earns its keep INSIDE an axis, and `hedge_cost` holds
    that number.
    """
    if table.empty or len(table) < 4:
        return float("nan")
    return float(table["hedge"].rank().corr(table["bind"].rank()))


# --------------------------------------------------------------------------
# 2. Where an abstention comes from
# --------------------------------------------------------------------------

def trace_frame(df: pd.DataFrame) -> pd.DataFrame:
    """1 row for each trace, with the label and the uncertainty counts.

    The columns:

    - `n_uncertain`: all the uncertainty spans of the trace.
    - `n_<reason>`: the spans of each reason in `REASON_ORDER`.
    - `n_marker_<marker>`: the spans of each marker in `MARKER_ORDER`. Only a
      version 3 corpus has the `marker` column, thus these columns are absent
      for version 1 and version 2.
    - `n_core`: `n_uncertain` less the `reconsideration` spans. Version 1 and
      version 2 mostly did not mark a "let me look again" span, thus this
      count is the one to compare with them. Without a `marker` column it is
      `n_uncertain`.
    - `abstains`: the judge answered `NotSure`.
    """
    if df.empty:
        return pd.DataFrame()
    part = df.assign(tid=trace_id(df))
    base = (part.drop_duplicates("tid").set_index("tid")
            [["case", "presented_label"]].copy())
    unc = part[part["extraction_class"] == UNCERTAINTY_CLASS]

    def _count(rows: pd.DataFrame) -> pd.Series:
        return (rows.groupby("tid").size().reindex(base.index).fillna(0)
                .astype(int))

    base["n_uncertain"] = _count(unc)
    for reason in REASON_ORDER:
        base[f"n_{reason}"] = (_count(unc[unc["reason"] == reason])
                               if "reason" in unc.columns else 0)
    if "marker" in unc.columns:
        for marker in MARKER_ORDER:
            base[f"n_marker_{marker}"] = _count(unc[unc["marker"] == marker])
        base["n_core"] = (base["n_uncertain"]
                          - base["n_marker_reconsideration"])
    else:
        base["n_core"] = base["n_uncertain"]
    base["abstains"] = base["presented_label"].eq("NotSure")
    return base.reset_index()


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """The rank AUC of a score, with no scipy.

    A tie takes the mean rank, thus a heavily tied score reads 0.5 and not 1.
    """
    n_pos, n_neg = len(positive), len(negative)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([positive, negative])).rank()
    total = float(ranks.iloc[:n_pos].sum())
    return (total - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_columns() -> Dict[str, str]:
    """Give the AUC column name for each count column of a trace frame.

    The total count gives `auc`. Each other count gives `auc_<suffix>`, for
    example `auc_cannot_see`, `auc_marker_qualifier`, and `auc_core`.
    """
    out = {"n_uncertain": "auc", "n_core": "auc_core"}
    for reason in REASON_ORDER:
        out[f"n_{reason}"] = f"auc_{reason}"
    for marker in MARKER_ORDER:
        out[f"n_marker_{marker}"] = f"auc_marker_{marker}"
    return out


def abstention_table(traces: pd.DataFrame, min_abstain: int = 25
                     ) -> pd.DataFrame:
    """How well the uncertainty count separates an abstention, in each case.

    `auc` reads the total count. Each `auc_<reason>` and `auc_marker_<marker>`
    reads the count of 1 reason or 1 marker alone, and `auc_core` reads the
    total less the `reconsideration` spans. A count column that the trace
    frame does not hold gives `NaN`. `abstain_rate` is the share of the
    traces that answer `NotSure`.

    A case with fewer than `min_abstain` abstentions gives every AUC as
    `NaN`. Street photography holds NO abstention at all, thus it cannot
    enter this table as a result. Read the empty row as "the case never
    abstained".
    """
    if traces.empty:
        return pd.DataFrame()
    scores = score_columns()
    rows = []
    for case, grp in traces.groupby("case"):
        mask = grp["abstains"].to_numpy(bool)
        pos = grp.loc[mask, "n_uncertain"].to_numpy(float)
        neg = grp.loc[~mask, "n_uncertain"].to_numpy(float)
        enough = len(pos) >= min_abstain
        row = {
            "case": case,
            "case_label": WA.case_label(case),
            "traces": len(grp),
            "n_abstain": int(len(pos)),
            "abstain_rate": len(pos) / len(grp) if len(grp) else np.nan,
            "spans_per_trace": grp["n_uncertain"].mean(),
            "median_abstain": np.median(pos) if len(pos) else np.nan,
            "median_decide": np.median(neg) if len(neg) else np.nan,
        }
        for column, name in scores.items():
            if enough and column in grp.columns:
                values = grp[column].to_numpy(float)
                row[name] = _auc(values[mask], values[~mask])
            else:
                row[name] = np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values("auc", ascending=False)
            .reset_index(drop=True))


def auc_long(stats: pd.DataFrame) -> pd.DataFrame:
    """Give 1 row for each case and each score of an abstention table.

    `score` is the count the AUC reads: `all`, `core`, a reason, or
    `marker_<marker>`. The arm columns stay when the table holds them.
    """
    if stats.empty:
        return pd.DataFrame()
    auc_cols = [c for c in stats.columns if c == "auc" or c.startswith("auc_")]
    keep = [c for c in ARM_COLUMNS + ("case", "case_label", "n_abstain",
                                      "abstain_rate")
            if c in stats.columns]
    wide = stats.rename(columns={"auc": "auc_all"})
    auc_cols = ["auc_all" if c == "auc" else c for c in auc_cols]
    long = wide.melt(id_vars=keep, value_vars=auc_cols, var_name="score",
                     value_name="auc")
    long["score"] = long["score"].str.replace(r"^auc_", "", regex=True)
    return long


def abstention_dose(traces: pd.DataFrame, min_bin: int = 30,
                    quantiles: Optional[int] = None) -> pd.DataFrame:
    """The abstention rate at each count of uncertainty spans.

    This is the dose curve. It answers a question the AUC cannot: the AUC says
    the counts separate, and the curve says HOW the rate climbs.

    Args:
        min_bin: A bin with fewer traces gives `abstain_rate` as `NaN`.
        quantiles: `None` uses the fixed bins `DOSE_EDGES`. A number `q` cuts
            the count at its `q` quantiles over the WHOLE frame (all cases
            together), and names the bins `Q1` to `Qq`. `bin_range` then
            gives the counts of each bin. A tied count never splits, thus a
            frame with heavy ties can give fewer than `q` bins.
    """
    if traces.empty:
        return pd.DataFrame()
    out = traces.copy()
    if quantiles:
        codes, edges = pd.qcut(out["n_uncertain"], q=quantiles, labels=False,
                               retbins=True, duplicates="drop")
        labels = [f"Q{i + 1}" for i in range(len(edges) - 1)]
        out["bin"] = pd.Categorical.from_codes(
            codes.fillna(0).astype(int), categories=labels)
        lows = [int(np.floor(edges[0]))] + [int(np.floor(e)) + 1
                                            for e in edges[1:-1]]
        highs = [int(np.floor(e)) for e in edges[1:]]
        ranges = dict(zip(labels, [f"{lo}-{hi}" for lo, hi in zip(lows, highs)]))
    else:
        out["bin"] = pd.cut(out["n_uncertain"], bins=list(DOSE_EDGES),
                            labels=list(DOSE_LABELS))
        ranges = {label: label for label in DOSE_LABELS}
    grouped = out.groupby(["case", "bin"], observed=False)
    table = pd.DataFrame({
        "traces": grouped.size(),
        "abstain": grouped["abstains"].sum(),
    }).reset_index()
    table["bin"] = table["bin"].astype(str)
    table["bin_range"] = table["bin"].map(ranges)
    table["abstain_rate"] = table["abstain"] / table["traces"]
    table.loc[table["traces"] < min_bin, "abstain_rate"] = np.nan
    table["case_label"] = [WA.case_label(c) for c in table["case"]]
    return table


def v3_arms() -> List[str]:
    """The arms of `X.ARMS` with `schema_version == "v3"`, in their order."""
    return [name for name, spec in X.ARMS.items()
            if spec.get("schema_version") == "v3"]


def arm_role(name: str) -> str:
    """`primary` when `X.PRIMARY_ARMS` names the arm for its judge, else
    `control`."""
    return "primary" if name in set(X.PRIMARY_ARMS.values()) else "control"


def arm_traces(arms: Optional[Sequence[str]] = None,
               require_canonical: bool = True
               ) -> Dict[str, Tuple[Dict[str, str], pd.DataFrame]]:
    """Read each arm and keep only its trace frame.

    The function reads the arms of `X.arm_roots(arms, present_only=True)`
    with their own roots, 1 arm at a time, and drops each span frame before
    it reads the next. The default is the 4 version 3 arms (`v3_arms`).

    Returns:
        `{arm: (spec, traces)}`. `spec` is the entry of `X.arm_roots`.

    Warning: this reads every corpus named. The 4 version 3 arms take about
    a minute and about 3.5 GB at the peak, because the qwen-trace arms hold
    up to 1.1 M rows each.
    """
    names = list(arms) if arms else v3_arms()
    out: Dict[str, Tuple[Dict[str, str], pd.DataFrame]] = {}
    for name, spec in X.arm_roots(names, present_only=True).items():
        df = X.load(root=spec["root"], require_canonical=require_canonical)
        out[name] = (spec, trace_frame(df))
        del df
    return out


def abstention_from_traces(
    by_arm: Dict[str, Tuple[Dict[str, str], pd.DataFrame]],
    min_abstain: int = 25, min_bin: int = 30,
    quantiles: Optional[int] = DOSE_QUANTILES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Give the abstention table and the dose curve of each arm.

    Returns:
        `(stats, dose)`. Both are long frames. They start with the columns
        `arm`, `role`, `judge`, `extractor`, `schema_version`, and then hold
        the columns of `abstention_table` and of `abstention_dose`. The dose
        bins are the `quantiles` of the count inside each arm.
    """
    stats_parts: List[pd.DataFrame] = []
    dose_parts: List[pd.DataFrame] = []
    for name, (spec, traces) in by_arm.items():
        if traces.empty:
            continue
        labels = {"arm": name, "role": arm_role(name),
                  "judge": spec["judge"], "extractor": spec["extractor"],
                  "schema_version": spec["schema_version"]}
        for frame, parts in ((abstention_table(traces, min_abstain), stats_parts),
                             (abstention_dose(traces, min_bin, quantiles),
                              dose_parts)):
            if frame.empty:
                continue
            for i, (key, value) in enumerate(labels.items()):
                frame.insert(i, key, value)
            parts.append(frame)
    stats = (pd.concat(stats_parts, ignore_index=True) if stats_parts
             else pd.DataFrame(columns=list(ARM_COLUMNS)))
    dose = (pd.concat(dose_parts, ignore_index=True) if dose_parts
            else pd.DataFrame(columns=list(ARM_COLUMNS)))
    return stats, dose


def abstention_across_arms(arms: Optional[Sequence[str]] = None,
                           min_abstain: int = 25,
                           require_canonical: bool = True,
                           min_bin: int = 30,
                           quantiles: Optional[int] = DOSE_QUANTILES
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read each arm, and give its abstention table and its dose curve.

    This is `arm_traces` and then `abstention_from_traces`. The default is
    the 4 version 3 arms.

    Returns:
        `(stats, dose)`, 2 long frames that start with the columns `arm`,
        `role`, `judge`, `extractor`, `schema_version`.

    Warning: this reads every corpus named. See `arm_traces`.
    """
    return abstention_from_traces(
        arm_traces(arms, require_canonical=require_canonical),
        min_abstain=min_abstain, min_bin=min_bin, quantiles=quantiles)


def auc_by_arm(stats: pd.DataFrame, score: str = "auc") -> pd.DataFrame:
    """Give 1 row for each case and 1 column for each arm, for 1 AUC column.

    The primary arm of each judge comes before its control.
    """
    if stats.empty or score not in stats.columns:
        return pd.DataFrame()
    order = (stats.drop_duplicates("arm")
             .sort_values(["judge", "role"], ascending=[True, False])["arm"]
             .tolist())
    table = stats.pivot_table(index="case_label", columns="arm",
                              values=score, aggfunc="first", dropna=False)
    table = table.reindex(columns=order)
    table.columns.name = None
    return table.sort_values(order[0], ascending=False,
                             na_position="last").reset_index()


def reason_mix(df: pd.DataFrame) -> pd.DataFrame:
    """The share of each reason, and the spans for each trace, in each case."""
    part = df[df["extraction_class"] == UNCERTAINTY_CLASS]
    if part.empty:
        return pd.DataFrame()
    share = (pd.crosstab(part["case"], part["reason"], normalize="index")
             .reindex(columns=list(REASON_ORDER)).fillna(0.0))
    traces = trace_id(df).groupby(df["case"]).nunique()
    share["spans_per_trace"] = (part.groupby("case").size() / traces)
    share["with_any"] = (trace_id(part).groupby(part["case"]).nunique()
                         / traces)
    share["case_label"] = [WA.case_label(c) for c in share.index]
    return share.reset_index()


# --------------------------------------------------------------------------
# 3 and 4. The price of an inference, and of a photographic flaw
# --------------------------------------------------------------------------

def inference_key(text) -> Optional[str]:
    """The words of an inference span, reduced to 1 key.

    `_cues_across.normalise_cue` does the spelling work. This function adds
    2 rules that only an inference span needs:

    1. **Strip the image reference.** The model writes `well maintained image
       B`. The image already lives in its own field, thus the words `image B`
       add nothing and they split 1 phrase into 2 keys.
    2. **Drop a restatement.** A span that names the other photograph or the
       label itself infers nothing. It is 0.0% to 4.6% of a kind.

    Returns `None` for a span that survives neither rule.
    """
    key = CA.normalise_cue(text)
    if key is None or _RESTATEMENT.search(key):
        return None
    words, out, skip = key.split(), [], False
    for i, word in enumerate(words):
        if skip:
            skip = False
            continue
        if word == "image":
            # `image a` and `image b` go together. A bare `image` goes alone.
            skip = i + 1 < len(words) and words[i + 1] in ("a", "b")
            continue
        out.append(word)
    cleaned = " ".join(out).strip()
    if not cleaned or cleaned in _LABEL_WORDS:
        return None
    return cleaned


def inference_pole(key: Optional[str]) -> str:
    """Which way an inference runs: `high`, `low`, `mixed`, or `none`.

    A pole word can be turned around by the word in front of it. `less
    upscale` counts LOW and `no visible crime` counts HIGH. See `_NEGATORS`
    for the window and its limit.
    """
    if not key:
        return "none"
    high = _pole_votes(key, _INFERENCE_HIGH, _INFERENCE_LOW)
    low = _pole_votes(key, _INFERENCE_LOW, _INFERENCE_HIGH)
    if high and low:
        return "mixed"
    if high:
        return "high"
    return "low" if low else "none"


def _pole_votes(key: str, plain: "re.Pattern", flipped: "re.Pattern") -> int:
    """Matches that land on 1 pole: `plain` unnegated, plus `flipped` negated."""
    return (sum(1 for m in plain.finditer(key) if not _negated(key, m.start()))
            + sum(1 for m in flipped.finditer(key) if _negated(key, m.start())))


def _negated(key: str, position: int) -> bool:
    """Does a negator sit in the tokens just before `position`?"""
    before = key[:position].split()
    return any(word in _NEGATORS for word in before[-_NEGATION_WINDOW:])


def inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The inference rows, with the word key, the pole, and `kind | pole`."""
    part = df[df["extraction_class"] == INFERENCE_CLASS].copy()
    if part.empty:
        return pd.DataFrame()
    part["word"] = [inference_key(t) for t in part["extraction_text"]]
    part = part[part["word"].notna()].copy()
    part["pole"] = [inference_pole(k) for k in part["word"]]
    part["kind_pole"] = (part["kind"].astype("string") + " | "
                         + part["pole"].astype("string"))
    return part


def pole_mix(df: pd.DataFrame) -> pd.DataFrame:
    """How much of each kind takes a pole, and which way it runs.

    `share_high` reads only the spans the lists DETERMINE. Read it beside
    `n_none`: a kind that leaves most of its spans undetermined cannot carry
    the number far.
    """
    part = inference_frame(df)
    if part.empty:
        return pd.DataFrame()
    counts = (pd.crosstab(part["kind"], part["pole"])
              .reindex(columns=list(POLE_ORDER)).fillna(0).astype(int))
    counts.columns = [f"n_{c}" for c in counts.columns]
    counts["n"] = counts.sum(axis=1)
    decided = counts["n_high"] + counts["n_low"]
    counts["share_determined"] = (decided / counts["n"]).round(3)
    counts["share_high"] = (counts["n_high"] / decided).round(3)
    order = {k: i for i, k in enumerate(INFERENCE_ORDER)}
    counts = counts.reset_index()
    counts["row"] = counts["kind"].map(order).fillna(len(order))
    return (counts.sort_values("row").drop(columns="row")
            .reset_index(drop=True))


def inference_coverage(df: pd.DataFrame, kind_min_count: int = 25,
                       min_count: int = 20,
                       rare_min_count: int = RARE_MIN_COUNT) -> pd.DataFrame:
    """Where each inference kind survives, and where a floor removes it.

    A rare kind can pass the whole notebook without a row and read as absent.
    `crime` is 454 spans of 302,784, thus it holds 1 dot on the kind ladder
    and it needed its own floor to reach the word view at all. This table
    says so.
    """
    part = inference_frame(df)
    if part.empty:
        return pd.DataFrame()
    ladder = kind_lift(df, INFERENCE_CLASS, min_count=kind_min_count)
    poles = pole_lift(df, min_count=kind_min_count)
    words = inference_words(df, min_count=min_count,
                            rare_min_count=rare_min_count)
    rows = []
    for kind in INFERENCE_ORDER:
        spans = part[part["kind"] == kind]
        if spans.empty:
            continue
        rows.append({
            "kind": kind,
            "is_risk": kind in RISK_KINDS,
            "spans": len(spans),
            "cases_present": spans["case"].nunique(),
            "cases_on_ladder": (0 if ladder.empty else
                                int((ladder["key"] == kind).sum())),
            "pole_rows": (0 if poles.empty else
                          int(poles["key"].str.startswith(f"{kind} |").sum())),
            "word_keys": (0 if words.empty else
                          int((words["kind"] == kind).sum())),
            "word_floor": (rare_min_count if kind in RARE_KINDS
                           else min_count),
        })
    return pd.DataFrame(rows)


def pole_lift(df: pd.DataFrame, min_count: int = 25,
              prior_strength: float = 25.0) -> pd.DataFrame:
    """The directional lift of each `kind | pole` key, for each case.

    `kind_lift` asks what a wealth inference costs. This asks what a wealth
    inference that runs UP costs, against one that runs DOWN.
    """
    part = inference_frame(df)
    if part.empty:
        return pd.DataFrame()
    part = part[part["pole"].isin(("high", "low"))].copy()
    if part.empty:
        return pd.DataFrame()
    part["key"] = part["kind_pole"]
    frames = [CA.keyed_lift(part, case, min_count, prior_strength)
              for case in sorted(part["case"].unique())]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["channel"] = "inference_pole"
    return out


def pole_key_order(summary: pd.DataFrame) -> List[str]:
    """`kind | pole` rows, the kinds in `INFERENCE_ORDER`, high before low."""
    rank = {k: i for i, k in enumerate(INFERENCE_ORDER)}
    def sort_key(key: str):
        kind, _, pole = key.partition(" | ")
        return (rank.get(kind, len(rank)), 0 if pole == "high" else 1)
    return sorted(summary["key"], key=sort_key)


def inference_words(df: pd.DataFrame,
                    kinds: Optional[Sequence[str]] = None,
                    min_count: int = 20,
                    rare_min_count: int = RARE_MIN_COUNT,
                    rare_kinds: Sequence[str] = RARE_KINDS,
                    prior_strength: float = 25.0) -> pd.DataFrame:
    """What the model WRITES when it makes an inference, and what it costs.

    `kind` names the topic and `pole` names the direction. Neither one says
    what the model actually wrote. This gives the words back, with a price.

    The lift is POOLED over the cases, not measured inside 1 of them. An
    inference phrase is a clause, not a noun phrase: 302,784 spans carry
    154,794 distinct keys, thus almost no phrase clears a per-case floor.
    Each vote instead carries the base rate of ITS OWN case, and the lift is
    the mean of `won - base`, shrunk toward 0.

    This keeps the base-rate correction that makes a lift comparable and it
    lets a phrase spread over 7 cases clear the floor.

    A kind in `rare_kinds` takes `rare_min_count` instead of `min_count`.
    Without it `crime` gives 0 keys and drops out of every word view.
    """
    part = inference_frame(df)
    if part.empty:
        return pd.DataFrame()
    if kinds:
        part = part[part["kind"].isin(list(kinds))]
    part = part[part["image"].isin(["A", "B"]) & part["kind"].notna()].copy()
    if part.empty:
        return pd.DataFrame()

    # The same 2 rules as any directional test: 1 vote for each comparison,
    # and a key on BOTH photographs of a trace separates nothing.
    votes = part.drop_duplicates(["case", "pair_id", "kind", "word", "image"])
    sides = (votes.groupby(["case", "pair_id", "kind", "word"])["image"]
             .transform("nunique"))
    votes = votes[sides == 1]
    label = votes["presented_label"]
    decided = ~label.isin(CA.UNDECIDED)
    won = ((votes["image"].eq("A") & label.isin(CA.WIN_FOR_A))
           | (votes["image"].eq("B") & label.isin(CA.WIN_FOR_B)))
    votes = votes.assign(won=won.astype(float))[decided.to_numpy()]
    if votes.empty:
        return pd.DataFrame()

    # 1 base rate for each CASE, over every inference vote of that case.
    #
    # Warning: do not take the base over (case, kind). 97% of the wealth
    # inferences run 1 way, thus a per-kind base asks "which wealth word pays
    # more than another wealth word" and the whole panel collapses to 0. The
    # case base keeps this figure on the scale of every other lift here.
    base = votes.groupby("case")["won"].transform("mean")
    votes = votes.assign(excess=votes["won"] - base, base=base)

    grouped = votes.groupby(["kind", "word"])  # noqa: E501
    out = grouped.agg(n=("won", "size"), n_cases=("case", "nunique"),
                      wins=("won", "sum"), excess=("excess", "sum"),
                      base=("base", "mean")).reset_index()
    # A rare kind takes its own floor. `min_count` rides along in the table,
    # thus a figure and a reader can both name the floor that a row cleared.
    out["min_count"] = [rare_min_count if k in set(rare_kinds) else min_count
                        for k in out["kind"]]
    out = out[out["n"] >= out["min_count"]]
    if out.empty:
        return pd.DataFrame()
    out["lift"] = out["excess"] / (out["n"] + prior_strength)
    out["win_rate"] = out["wins"] / out["n"]
    out["pole"] = [inference_pole(k) for k in out["word"]]
    out = out.rename(columns={"word": "key"}).drop(columns="excess")
    return (out.sort_values(["kind", "lift"], ascending=[True, False])
            .reset_index(drop=True))


def kind_lift(df: pd.DataFrame, extraction_class: str, key: str = "kind",
              min_count: int = 25, prior_strength: float = 25.0
              ) -> pd.DataFrame:
    """The directional lift of each attribute value, for each case.

    The span names an image, thus the test is the same one a cue takes:
    1 vote for each (trace, key, image), a key on both images is dropped, and
    the rate is shrunk toward the base rate of the case.
    """
    part = df[df["extraction_class"] == extraction_class].copy()
    if part.empty or key not in part.columns:
        return pd.DataFrame()
    part["key"] = part[key].astype("string")
    part = part[part["key"].notna()]
    frames = [CA.keyed_lift(part, case, min_count, prior_strength)
              for case in sorted(part["case"].unique())]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["channel"] = extraction_class
    return out


def kind_summary(tables: pd.DataFrame,
                 order: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """1 row for each key: the spread of its lift, and how it splits."""
    if tables.empty:
        return pd.DataFrame()
    grouped = tables.groupby("key")
    out = pd.DataFrame({
        "n_cases": grouped["case"].nunique(),
        "n": grouped["n"].sum(),
        "win_rate": grouped["wins"].sum() / grouped["n"].sum(),
        "base_rate": grouped["base_rate"].mean(),
        "lift_mean": grouped["lift"].mean(),
        "lift_min": grouped["lift"].min(),
        "lift_max": grouped["lift"].max(),
        "n_pays": grouped["lift"].apply(lambda s: int((s > 0).sum())),
        "n_costs": grouped["lift"].apply(lambda s: int((s < 0).sum())),
    })
    out["lift_range"] = out["lift_max"] - out["lift_min"]
    out["reverses"] = (out["n_pays"] > 0) & (out["n_costs"] > 0)
    out = out.reset_index()
    if order:
        rank = {k: i for i, k in enumerate(order)}
        out["row"] = out["key"].map(rank).fillna(len(rank))
        out = out.sort_values(["row", "lift_mean"], ascending=[True, False])
        out = out.drop(columns="row")
    else:
        out = out.sort_values("lift_mean", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Excerpts: the model's own sentences
# --------------------------------------------------------------------------

#: The columns of the source run that hold a trace.
TRACE_COLUMN = "model_reasoning"

#: Markdown the model writes around a sentence. An excerpt reads better
#: without it, and none of it changes a word.
_MARKUP = re.compile(r"\*\*|\*|^\s*[-*•]\s*|^\s*#+\s*")

_SENTENCE_RIGHT = re.compile(r"[.!?](?:\s|$)|\n")


def trace_texts(df: pd.DataFrame,
                pairs: Optional[pd.DataFrame] = None) -> Dict[tuple, str]:
    """Read the reasoning text of each wanted trace from its source run.

    The extraction corpus does not carry the trace, only the spans and their
    offsets. The run that wrote the trace is named in `source_results_path`,
    1 for each case.

    Args:
        pairs: The rows to read, with a `case` and a `pair_id` column. `None`
            reads every trace of the corpus, which is about 230 MB of text.

    Returns:
        `{(case, pair_id): text}`. A run that is not on disk is skipped, thus
        a missing key is not an error.
    """
    out: Dict[tuple, str] = {}
    sources = df[["case", "source_results_path"]].drop_duplicates()
    wanted = None
    if pairs is not None:
        wanted = {c: set(g["pair_id"]) for c, g in pairs.groupby("case")}
    for case, path in sources.itertuples(index=False):
        if wanted is not None and not wanted.get(case):
            continue
        try:
            frame = pd.read_parquet(path, columns=["pair_id", TRACE_COLUMN])
        except (FileNotFoundError, OSError, ValueError):
            continue
        if wanted is not None:
            frame = frame[frame["pair_id"].isin(wanted[case])]
        for pair_id, text in frame.itertuples(index=False):
            if isinstance(text, str) and text:
                out[(case, pair_id)] = text
    return out


def _anchor(text: str, span: str, start, end) -> tuple:
    """Where the span sits in the trace.

    The stored offsets land exactly on 92% of the spans. The rest drift by a
    quote mark that the extractor normalised, thus a search for the span text
    comes first and the offsets are the fallback.
    """
    found = text.find(span) if span else -1
    if found >= 0:
        return found, found + len(span)
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return 0, min(len(text), 160)


def sentence_around(text: str, start: int, end: int,
                    max_chars: int = 360, min_chars: int = 90) -> str:
    """The sentence that holds a span, cleaned of the markdown around it.

    A 1-sentence excerpt can read as nothing: `It looks very high-end.` names
    neither the photograph nor the thing. When the sentence is shorter than
    `min_chars` this takes the sentence BEFORE it as well, up to `max_chars`.
    """
    if not text:
        return ""
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))

    window_start = max(0, start - max_chars)
    left_window = text[window_start:start]
    # Every sentence start inside the window, nearest to the span last.
    bounds = [window_start + m.end()
              for m in _SENTENCE_RIGHT.finditer(left_window)]
    left = bounds[-1] if bounds else window_start

    right_window = text[end:end + max_chars]
    forward = _SENTENCE_RIGHT.search(right_window)
    right = end + (forward.end() if forward else len(right_window))

    # Walk back 1 sentence at a time until the excerpt can stand alone.
    index = len(bounds) - 1
    while right - left < min_chars and index > 0:
        index -= 1
        left = bounds[index]
    if right - left < min_chars and not bounds:
        left = window_start

    excerpt = text[left:right]
    excerpt = _MARKUP.sub("", excerpt)
    return " ".join(excerpt.split()).strip()


EXCERPT_COLUMNS = (
    "case", "case_label", "kind", "pole", "word", "lift", "n_votes",
    "image", "presented_label", "image_won", "span", "excerpt", "pair_id",
    "char_start", "char_end", "offsets_exact",
)


def excerpts(df: pd.DataFrame, kinds: Optional[Sequence[str]] = None,
             n_per_cell: int = 4, min_count: int = 20, seed: int = 777,
             max_chars: int = 360, min_chars: int = 90) -> pd.DataFrame:
    """The model's own sentences, for each case and each inference kind.

    A lift is a number. This gives the sentence behind it back, so a reader
    can see what the model actually wrote when it inferred wealth about a
    school or a subway entrance.

    The rows of 1 cell are chosen, never sampled at random: the phrases whose
    price is furthest from 0 take the cell, half from the paying end and half
    from the costing end. Inside 1 phrase the span IS drawn at random, with a
    fixed seed.

    Warning: an excerpt needs the source run on disk. `trace_texts` skips a
    run it cannot read, thus a case can be absent. It is not a null result.
    """
    part = inference_frame(df)
    if part.empty:
        return pd.DataFrame()
    if kinds:
        part = part[part["kind"].isin(list(kinds))]
    label = part["presented_label"]
    decided = ~label.isin(CA.UNDECIDED)
    part = part[part["image"].isin(["A", "B"]) & decided].copy()
    if part.empty:
        return pd.DataFrame()
    part["image_won"] = (
        (part["image"].eq("A") & part["presented_label"].isin(CA.WIN_FOR_A))
        | (part["image"].eq("B") & part["presented_label"].isin(CA.WIN_FOR_B))
    )

    priced = inference_words(df, kinds=kinds, min_count=min_count)
    if priced.empty:
        return pd.DataFrame()
    price = priced.set_index(["kind", "key"])[["lift", "n"]]
    part = part.join(price, on=["kind", "word"])
    part = part[part["lift"].notna()]
    if part.empty:
        return pd.DataFrame()

    # 1 row for each (case, kind, phrase), then the 2 ends of the price.
    rows = []
    half = max(1, n_per_cell // 2)
    for (case, kind), group in part.groupby(["case", "kind"]):
        keys = (group[["word", "lift"]].drop_duplicates("word")
                .sort_values("lift"))
        take = pd.concat([keys.head(half), keys.tail(n_per_cell - half)])
        for word in take.drop_duplicates("word")["word"]:
            candidates = group[group["word"] == word]
            rows.append(candidates.sample(n=1, random_state=seed))
    if not rows:
        return pd.DataFrame()
    chosen = pd.concat(rows, ignore_index=True)

    texts = trace_texts(df, chosen[["case", "pair_id"]])
    excerpt, exact = [], []
    for row in chosen.itertuples(index=False):
        text = texts.get((row.case, row.pair_id), "")
        left, right = _anchor(text, row.extraction_text, row.char_start,
                              row.char_end)
        excerpt.append(sentence_around(text, left, right, max_chars,
                                       min_chars))
        exact.append(bool(text) and text[left:right] == row.extraction_text)
    chosen["excerpt"] = excerpt
    chosen["offsets_exact"] = exact
    chosen["span"] = chosen["extraction_text"]
    chosen["n_votes"] = chosen["n"].astype("Int64")
    chosen["case_label"] = [WA.case_label(c) for c in chosen["case"]]
    chosen = chosen[chosen["excerpt"].str.len() > 0]
    order = {k: i for i, k in enumerate(INFERENCE_ORDER)}
    chosen["_row"] = chosen["kind"].map(order).fillna(len(order))
    return (chosen.sort_values(["case", "_row", "lift"],
                               ascending=[True, True, False])
            [list(EXCERPT_COLUMNS)].reset_index(drop=True))


def format_excerpts(frame: pd.DataFrame, max_rows: int = 40) -> str:
    """The excerpts as markdown, with the span in bold inside its sentence."""
    if frame.empty:
        return "_No excerpt. The source runs may not be on disk._"
    lines: List[str] = []
    for kind, group in frame.groupby("kind", sort=False):
        lines.append(f"#### {kind}")
        for row in group.head(max_rows).itertuples(index=False):
            body = row.excerpt
            if row.span and row.span in body:
                body = body.replace(row.span, f"**{row.span}**", 1)
            won = "the photograph WON" if row.image_won else "it LOST"
            lines.append(
                f"- `{row.word}` ({row.pole}, lift {row.lift:+.2f}, "
                f"{int(row.n_votes)} votes) — image {row.image}, {won}\n"
                f"  > {body}"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The figures
# --------------------------------------------------------------------------

def _case_colour(case: str) -> str:
    return WA.CASE_COLOUR.get(case, S.PAL["warmgrey"])


def _wrap(text: str, width: int = 26) -> str:
    """Break a long axis phrase over 2 lines, on a space."""
    if len(text) <= width:
        return text
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n".join(lines[:2])


# --------------------------------------------------------------------------
# The captions
# --------------------------------------------------------------------------
#
# A figure carries no caption prose. Each `caption_<figure>` function builds
# the caption from the same frame that the plot function draws, and the plot
# function attaches it to the figure (`_figmeta.attach`). The notebook shows
# it under the figure, and `_figmeta.save_figure` writes it beside the PNG.

#: The tag of the notebook, the first tag of every figure.
NOTEBOOK_TAG = "trace_channels"


def arm_of_root(root: Optional[str | Path] = None) -> Optional[str]:
    """The name in `X.ARMS` whose root is `root` (default `X.DEFAULT_ROOT`)."""
    target = Path(root) if root else Path(X.DEFAULT_ROOT)
    if not target.is_absolute():
        target = Path(X.REPO_ROOT) / target
    target = target.resolve()
    for name, spec in X.ARMS.items():
        if (Path(X.REPO_ROOT) / spec["root"]).resolve() == target:
            return name
    return None


def _corpus_text(arm: Optional[str], n_traces: Optional[int] = None,
                 n_cases: Optional[int] = None) -> str:
    """1 sentence that names the corpus of a figure.

    `None` takes the arm of `X.DEFAULT_ROOT`, the corpus of the notebook.
    """
    arm = arm or arm_of_root()
    spec = X.ARMS.get(arm or "")
    if spec:
        text = (f"Corpus: the {spec['judge']} traces, read by the "
                f"{spec['extractor']} extractor (arm `{arm}`, schema "
                f"{spec['schema_version']}, the {arm_role(arm)} arm of the "
                "judge)")
    else:
        text = "Corpus: the extraction corpus of the notebook"
    if n_traces:
        text += f", {int(n_traces):,} traces"
    if n_cases:
        text += f" over {int(n_cases)} cases"
    return text + "."


def _meta(name: str, title: str, caption: str, measure: str,
          cases: Sequence[str] = (), arms: Sequence[Optional[str]] = (),
          label: Optional[str] = None) -> Dict:
    """The caption content of 1 figure of this module.

    An arm that is `None` takes the arm of `X.DEFAULT_ROOT`.
    """
    arms = [a for a in (a or arm_of_root() for a in arms) if a]
    schemas = sorted({X.ARMS[a]["schema_version"] for a in arms if a in X.ARMS})
    roots = [X.ARMS[a]["root"] for a in arms if a in X.ARMS]
    return F.figure_meta(
        title, caption, label or F.fig_label(name),
        [NOTEBOOK_TAG, *cases, *arms, *(schemas or ["v3"]), measure],
        module="_trace_channels", version=__version__,
        arm=arms[0] if len(arms) == 1 else (arms or None),
        corpus_root=roots[0] if len(roots) == 1 else (roots or None))


def _cases_of(frame: pd.DataFrame) -> List[str]:
    if frame is None or frame.empty or "case" not in frame.columns:
        return []
    have = set(frame["case"])
    return [c for c in WA.CASE_ORDER if c in have] + sorted(
        have - set(WA.CASE_ORDER))


def caption_axes(table: pd.DataFrame, max_axes: int = 26,
                 arm: Optional[str] = None) -> Dict:
    """The caption of `plot_axes`."""
    keep = table.nlargest(min(max_axes, len(table)), "n")
    n_cases = int(table["n_cases"].max()) if "n_cases" in table else None
    caption = (
        f"Each row is 1 axis phrase that the model names in a `comparison` "
        f"span: the {len(keep)} axes with the most decided comparisons, of "
        f"{len(table)} axes over the count floor ({int(keep['n'].sum()):,} "
        "decided comparisons in the rows). The dot is the share of the "
        "decided comparisons where the final label names the image that the "
        "axis named. The dot area grows with the count, and its colour "
        "follows the same share on the diverging ramp (coral low, teal "
        "high). The dark bar is the 95% Wilson interval, and the grey stem "
        "runs from chance to the dot. The vertical line is 0.5 (chance), and "
        "the shaded band is within 0.1 of chance. A row outside the band "
        "binds the vote: when the model names a winner on that axis, the "
        "final label follows. A row inside the band names an axis that the "
        "model then does not decide on. Warning: the side of the line is "
        "polarity, not quality. `direction` names the image with MORE of the "
        "quality, thus a negative axis word such as `wear` sits on the left "
        "and still binds the vote. The counts pool the cases. "
        + _corpus_text(arm, n_cases=n_cases))
    return _meta("axes-binding", "Which axis binds the vote", caption,
                 "axis_agreement", arms=[arm])


def caption_axis_hedge(table: pd.DataFrame, summary: Optional[Dict] = None,
                       max_axes: int = 22, arm: Optional[str] = None) -> Dict:
    """The caption of `plot_axis_hedge`."""
    keep = table[table["agree_hedged"].notna() & table["agree_plain"].notna()]
    keep = keep.nlargest(min(max_axes, len(keep)), "n")
    pooled = summary or {}
    caption = (
        f"Each row is 1 axis phrase, the {len(keep)} axes with the most "
        "decided comparisons that hold both a plain and a hedged rate. The "
        "teal dot is the agreement of the comparisons where the model did "
        "not hedge, and the coral dot is the agreement of the comparisons "
        "where the model hedged. The arrow runs from plain to hedged: an "
        "arrow to the left says that the model's own hedge marker predicts "
        "that this comparison does not carry the decision. The rows sort by "
        "that drop. The agreement of a negative axis word is folded onto the "
        "polarity of the axis (1 minus the value), thus every row reads on 1 "
        "scale. The vertical line is 0.5 (chance). The number in brackets is "
        "how often the model hedged on that axis. The hedge rate does not "
        "predict a weak axis (Spearman of the hedge rate against the binding "
        f"strength over the axes: {hedge_bind_corr(table):+.2f}); the drop "
        "inside 1 axis is where the marker earns its keep.")
    if pooled:
        caption += (f" Over every axis, the agreement is "
                    f"{pooled['agree_plain']:.3f} plain against "
                    f"{pooled['agree_hedged']:.3f} hedged, over "
                    f"{int(pooled.get('n_decided', 0)):,} decided comparisons.")
    caption += " " + _corpus_text(arm)
    return _meta("axes-hedge", "What the hedge marker costs an axis", caption,
                 "hedge_cost", arms=[arm])


def caption_abstention(dose: pd.DataFrame, stats: pd.DataFrame,
                       arm: Optional[str] = None) -> Dict:
    """The caption of `plot_abstention`."""
    missing = stats[stats["auc"].isna()]
    never = missing[missing["n_abstain"] == 0]["case_label"].tolist()
    few = missing[missing["n_abstain"] > 0]["case_label"].tolist()
    n_traces = int(stats["traces"].sum()) if "traces" in stats else None
    caption = (
        "The share of the traces that abstain (the judge answers `NotSure`), "
        "against the count of the uncertainty spans in the trace. 1 line for "
        "each case; the colour is the case. The x axis holds the fixed "
        f"version 3 bins ({', '.join(DOSE_LABELS)} spans). A bin with fewer "
        "than 30 traces has no point. The AUC in the legend is the "
        "separation of an abstention from a decision by the span count "
        "alone (0.5 is no separation). A case needs 25 abstentions or more "
        "for an AUC.")
    if never:
        caption += " " + ", ".join(never) + " never abstained, thus no line."
    if few:
        caption += (" " + ", ".join(few)
                    + " abstained too seldom for an AUC, thus no line.")
    caption += (" Warning: the left end of a curve is a small group of short "
                "traces. Read the rise above 20 spans. "
                + _corpus_text(arm, n_traces, len(stats)))
    return _meta("abstention-dose", "The abstention dose curve", caption,
                 "abstention_auc", cases=_cases_of(stats), arms=[arm])


def caption_abstention_arms(dose: pd.DataFrame, stats: pd.DataFrame) -> Dict:
    """The caption of `plot_abstention_arms`."""
    arms = [a for a in stats["arm"].drop_duplicates().tolist()]
    by_case = stats.groupby("case_label")["n_abstain"].sum() \
        if "case_label" in stats else pd.Series(dtype=float)
    never = sorted(by_case[by_case == 0].index.tolist())
    n_traces = int(stats["traces"].sum()) if "traces" in stats else None
    caption = (
        "The abstention dose curve of each judge, under its 2 extractors (the "
        "2 x 2 of judge and extractor, schema version 3). 1 panel for each "
        "judge; the panel title names the 2 extractors. 1 colour for each "
        "case. The solid line is the primary extractor and the dashed line "
        "is the control extractor. The y axis is the share of the traces "
        "that abstain (`NotSure`). The x axis is the quantile of the "
        "uncertainty count INSIDE the arm, because the 2 extractors of 1 "
        "judge mark up to 2.5x different counts. A case with fewer than the "
        "floor of abstentions in an arm (25 by default) has no line in that "
        "arm.")
    if never:
        caption += " " + ", ".join(never) + " never abstains in any arm."
    caption += (f" Arms: {', '.join(f'`{a}`' for a in arms)}"
                + (f"; {n_traces:,} traces in all." if n_traces else "."))
    return _meta("abstention-dose-arms", "The abstention in the 2 x 2", caption,
                 "abstention_auc", cases=_cases_of(stats), arms=arms)


def caption_reasons(mix: pd.DataFrame, arm: Optional[str] = None) -> Dict:
    """The caption of `plot_reasons`."""
    parts = ", ".join(f"`{r}` ({REASON_LABEL[r]})" for r in REASON_ORDER)
    caption = (
        "1 bar for each case: the share of the uncertainty spans of the case "
        f"by the reason that the extractor gives. The reasons, left to right: "
        f"{parts}. The number beside a case is the uncertainty spans for each "
        "trace, and the rows sort by it. `other` is the reason that the "
        "extractor could not type, and it takes a large share; read the 3 "
        "named reasons against each other, never against the whole. "
        + _corpus_text(arm, n_cases=len(mix)))
    return _meta("abstention-reasons", "What the model says it cannot do",
                 caption, "uncertainty_reason_share", cases=_cases_of(mix),
                 arms=[arm])


#: The text of each named kind ladder: (label, title, what a row is, measure).
KIND_LADDERS: Dict[str, Tuple[str, str, str, str]] = {
    "inference": (
        "inference-kind-lift", "The price of an inference, by topic",
        "Each row is 1 `kind` of the `inference` spans: what the model reads "
        "INTO a photograph. The risk kinds (wealth, class, crime, "
        "demographic) come first. A wealth inference is the most decisive "
        "move in the trace.",
        "inference_kind_lift"),
    "inference_pole": (
        "inference-pole-lift", "The price of an inference, by direction",
        "Each row is 1 `kind` and 1 pole of the `inference` spans. `kind` "
        "names the topic of an inference, and the pole (`high` or `low`, "
        "from 2 short word lists) names which way it runs. A span that "
        "restates the vote (it names the other photograph or the label) "
        "leaves before the count.",
        "inference_pole_lift"),
    "artifact": (
        "artifact-kind-lift", "The price of a photographic flaw",
        "Each row is 1 `kind` of the `image_artifact` spans: a flaw of the "
        "camera, not of the place. A lift far from 0 says that the judgment "
        "tracks how the photograph was taken.",
        "artifact_kind_lift"),
}


def caption_kind_ladder(tables: pd.DataFrame, summary: pd.DataFrame,
                        name: str = "kind", note: str = "",
                        arm: Optional[str] = None, floor: int = 25) -> Dict:
    """The caption of `plot_kind_ladder`.

    `name` picks the text of `KIND_LADDERS`; another name gives a generic
    caption under the label `fig:<name>-ladder`.
    """
    label, title, what, measure = KIND_LADDERS.get(
        name, (f"{name}-ladder", "The price of each attribute value",
               "Each row is 1 attribute value.", f"{name}_lift"))
    caption = (
        f"{what} 1 dot for each case that clears the count floor ({floor} "
        "decided votes); the colour is the case. The dot is the lift: the "
        "win rate of the image that the span names, shrunk toward the base "
        "rate of the case, less that base rate. The grey line runs from the "
        "lowest to the highest lift of the row; a short line says that every "
        "case gives the value the same price. The vertical line is 0, the "
        "base rate of the case. The number beside a row is the cases that "
        f"clear the floor. The rows hold {int(tables['n'].sum()):,} decided "
        "votes in all. Warning: a lift is an association, not a cause; the "
        "model narrates while it decides.")
    if note and note not in caption:
        caption += " " + note
    caption += " " + _corpus_text(arm, n_cases=len(_cases_of(tables)))
    return _meta(label, title, caption, measure, cases=_cases_of(tables),
                 arms=[arm])


def caption_inference_words(words: pd.DataFrame,
                            kinds: Optional[Sequence[str]] = None,
                            n_each: int = 8, min_keys: int = 3,
                            arm: Optional[str] = None) -> Dict:
    """The caption of `plot_inference_words`."""
    counts = words.groupby("kind").size()
    have = [k for k in (kinds or INFERENCE_ORDER) if counts.get(k, 0) >= min_keys]
    caption = (
        f"What the model writes when it makes an inference, and what it "
        f"costs. 1 panel for each `kind` ({', '.join(have)}); a kind with "
        f"fewer than {min_keys} repeated phrases has no panel. Each panel "
        f"holds the {n_each} phrases that go with the winning photograph "
        f"most, and the {n_each} that go with it least. The dot is the lift "
        "of the phrase over the base rate of the case, pooled over the "
        "cases; each vote carries the base rate of its OWN case. A teal dot "
        "pays (the image wins more often than the base rate) and a coral "
        "dot costs. The x label of a panel gives its comparisons and the "
        "count floor of a phrase. The lift pools the cases because an "
        "inference span is a clause, and almost no phrase clears a per-case "
        "floor. " + _corpus_text(arm))
    return _meta("inference-words", "The words of an inference", caption,
                 "inference_word_lift", cases=_cases_of(words), arms=[arm])


# --------------------------------------------------------------------------
# The plot functions
# --------------------------------------------------------------------------

def plot_axes(table: pd.DataFrame, max_axes: int = 26,
              figsize: Tuple[float, float] = (6.6, 7.4),
              arm: Optional[str] = None):
    """Each named axis, at the share of the votes it agrees with.

    The vertical line is 0.5, which is chance. An axis far from it BINDS the
    vote: name a winner on that axis and the final label follows. An axis near
    it does not, and those are the rows to read first.

    Warning: the side of the line is polarity, not quality. `direction` names
    the image with more of the quality, thus a negative word such as `wear`
    sits on the left and still binds the vote hard.

    The caption is `caption_axes`, attached to the figure.
    """
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("no axes to draw")
    keep = table.nlargest(min(max_axes, len(table)), "n")
    keep = keep.sort_values("agree").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvspan(0.4, 0.6, color=S.PAL["cream"], alpha=0.45, zorder=0)
    ax.axvline(0.5, color=S.EDGE, linewidth=0.8, alpha=0.7, zorder=1)
    sizes = 18 + 46 * (np.sqrt(keep["n"]) / np.sqrt(keep["n"]).max())
    for y, row in keep.iterrows():
        ax.plot([0.5, row["agree"]], [y, y], color=S.PAL["warmgrey"],
                linewidth=1.0, alpha=0.6, zorder=2, solid_capstyle="round")
        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=S.EDGE,
                linewidth=0.8, alpha=0.45, zorder=3)
    ax.scatter(keep["agree"], range(len(keep)), s=sizes,
               c=[S.CMAP_DIV(v) for v in keep["agree"]],
               edgecolors=S.EDGE, linewidths=0.35, zorder=4)

    ax.set_yticks(range(len(keep)), [_wrap(a) for a in keep["axis"]])
    ax.set_ylim(-0.8, len(keep) - 0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("share of the decided comparisons where the final label\n"
                  "names the image the axis named")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    return F.attach(fig, caption_axes(table, max_axes, arm))


def plot_axis_hedge(table: pd.DataFrame, summary: Optional[Dict] = None,
                    max_axes: int = 22,
                    figsize: Tuple[float, float] = (6.6, 6.4),
                    arm: Optional[str] = None):
    """What the model's own hedge marker costs an axis.

    1 row for each axis. The teal dot is the agreement of the PLAIN
    comparisons, and the coral dot is the agreement of the HEDGED ones. The
    arrow runs from plain to hedged, thus a leftward arrow says the hedge
    predicts that this comparison will not carry the decision.

    A negative axis word runs the other way, because `direction` names the
    image with more of the quality. `agree` is folded onto the polarity of
    the axis before it is drawn, so every row reads on 1 scale.

    The caption is `caption_axis_hedge`, attached to the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if table.empty:
        raise ValueError("no axes to draw")
    keep = table[table["agree_hedged"].notna() & table["agree_plain"].notna()]
    if keep.empty:
        raise ValueError("no axis carries both a plain and a hedged rate")
    keep = keep.nlargest(min(max_axes, len(keep)), "n").copy()
    # Fold a negative axis word onto the same scale as a positive one.
    flip = keep["polarity"].eq("negative")
    for column in ("agree", "agree_plain", "agree_hedged"):
        keep[column] = np.where(flip, 1.0 - keep[column], keep[column])
    keep["drop"] = keep["agree_plain"] - keep["agree_hedged"]
    keep = keep.sort_values("drop").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.5, color=S.EDGE, linewidth=0.8, alpha=0.6, zorder=1)
    for y, row in keep.iterrows():
        ax.annotate("", xy=(row["agree_hedged"], y),
                    xytext=(row["agree_plain"], y),
                    arrowprops=dict(arrowstyle="->", color=S.PAL["warmgrey"],
                                    linewidth=1.0, alpha=0.75,
                                    shrinkA=2.5, shrinkB=2.5), zorder=2)
    ax.scatter(keep["agree_plain"], range(len(keep)), s=26,
               color=S.PAL["teal"], edgecolors=S.EDGE, linewidths=0.35,
               zorder=3)
    ax.scatter(keep["agree_hedged"], range(len(keep)), s=26,
               color=S.PAL["coral"], edgecolors=S.EDGE, linewidths=0.35,
               zorder=4)

    # The bracket holds the hedge rate of the axis: a data label.
    labels = [f"{_wrap(a, 24)}  [{h:.0%}]"
              for a, h in zip(keep["axis"], keep["hedge"])]
    ax.set_yticks(range(len(keep)), labels)
    ax.set_ylim(-0.8, len(keep) - 0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("agreement with the final label, folded onto the polarity "
                  "of the axis")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markersize=4.2,
                   markerfacecolor=S.PAL["teal"], markeredgecolor=S.EDGE,
                   markeredgewidth=0.3, label="the model did not hedge"),
            Line2D([], [], marker="o", linestyle="none", markersize=4.2,
                   markerfacecolor=S.PAL["coral"], markeredgecolor=S.EDGE,
                   markeredgewidth=0.3, label="the model hedged"),
        ],
        loc="lower left", frameon=True, framealpha=0.92, fontsize=6.5,
    )
    fig.tight_layout()
    return F.attach(fig, caption_axis_hedge(table, summary, max_axes, arm))


def plot_abstention(dose: pd.DataFrame, stats: pd.DataFrame,
                    figsize: Tuple[float, float] = (6.4, 4.4),
                    arm: Optional[str] = None):
    """The abstention rate climbs with the count of the uncertainty spans.

    1 line for each case. A case that never abstained cannot be drawn, and
    the caption (`caption_abstention`, attached to the figure) names it.
    """
    import matplotlib.pyplot as plt

    if dose.empty:
        raise ValueError("no dose curve to draw")
    usable = stats[stats["auc"].notna()]["case"].tolist()
    fig, ax = plt.subplots(figsize=figsize)
    xs = range(len(DOSE_LABELS))
    for case in usable:
        row = (dose[dose["case"] == case].set_index("bin")
               .reindex(list(DOSE_LABELS)))
        auc = float(stats.loc[stats["case"] == case, "auc"].iloc[0])
        ax.plot(xs, row["abstain_rate"].to_numpy(), marker="o", markersize=4.2,
                linewidth=1.4, color=_case_colour(case),
                markeredgecolor=S.EDGE, markeredgewidth=0.3,
                label=f"{WA.case_label(case)}  (AUC {auc:.2f})")

    ax.set_xticks(list(xs), list(DOSE_LABELS))
    ax.set_xlabel("uncertainty spans in the trace (version 3 bins)")
    ax.set_ylabel("share of the traces that abstain")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=6.5)
    fig.tight_layout()
    return F.attach(fig, caption_abstention(dose, stats, arm))


def _short_model(name: str) -> str:
    """`gemma-4-12b` gives `Gemma-4-12B`; `qwen3.5-9b` gives `Qwen3.5-9B`."""
    head, _, size = str(name).rpartition("-")
    return f"{head[:1].upper()}{head[1:]}-{size.upper()}" if head else str(name)


def plot_abstention_arms(dose: pd.DataFrame, stats: pd.DataFrame,
                         figsize: Tuple[float, float] = (7.2, 3.6)):
    """The dose curve of each judge, under its 2 extractors.

    1 panel for each judge. 1 colour for each case. The solid line is the
    primary extractor and the dashed line is the control. The x axis is the
    quantile of the uncertainty count INSIDE the arm, because the 2
    extractors of 1 judge mark up to 2.5x different counts. A case with too
    few abstentions in an arm (`auc` is `NaN`) has no line in that arm.

    The caption is `caption_abstention_arms`, attached to the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if dose.empty or stats.empty:
        raise ValueError("no dose curve to draw")
    judges = sorted(stats["judge"].unique())
    fig, axes = plt.subplots(1, len(judges), figsize=figsize, sharey=True,
                             squeeze=False)
    bins = [b for b in dose["bin"].drop_duplicates().tolist()]
    bins = sorted(bins, key=lambda b: int(str(b)[1:]) if str(b)[1:].isdigit()
                  else 0)
    xs = np.arange(len(bins))
    seen: List[str] = []
    for ax, judge in zip(axes[0], judges):
        part = stats[stats["judge"] == judge]
        arms = (part.drop_duplicates("arm")
                .sort_values("role", ascending=False))
        notes = []
        for arm in arms.itertuples():
            style = "-" if arm.role == "primary" else "--"
            notes.append(f"{'solid' if arm.role == 'primary' else 'dashed'}: "
                         f"{_short_model(arm.extractor)}")
            rows = part[(part["arm"] == arm.arm) & part["auc"].notna()]
            for case in rows["case"]:
                curve = (dose[(dose["arm"] == arm.arm) & (dose["case"] == case)]
                         .set_index("bin").reindex(bins))
                ax.plot(xs, curve["abstain_rate"].to_numpy(float),
                        linestyle=style, marker="o",
                        markersize=3.6 if arm.role == "primary" else 2.8,
                        linewidth=1.4 if arm.role == "primary" else 1.0,
                        color=_case_colour(case), markeredgecolor=S.EDGE,
                        markeredgewidth=0.3,
                        alpha=1.0 if arm.role == "primary" else 0.8)
                if case not in seen:
                    seen.append(case)
        ax.set_title(f"{_short_model(judge)} judge\n" + ", ".join(notes),
                     fontsize=7.4)
        ax.set_xticks(list(xs), bins)
        ax.set_xlabel("quantile of the uncertainty count in the arm")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)
    axes[0][0].set_ylabel("share of the traces that abstain")

    order = [c for c in stats.sort_values("case")["case"].unique() if c in seen]
    handles = [Line2D([0], [0], color=_case_colour(c), marker="o",
                      markersize=3.6, linewidth=1.4,
                      markeredgecolor=S.EDGE, markeredgewidth=0.3,
                      label=WA.case_label(c)) for c in order]
    handles += [Line2D([0], [0], color=S.EDGE, linestyle="-", linewidth=1.2,
                       label="primary extractor"),
                Line2D([0], [0], color=S.EDGE, linestyle="--", linewidth=1.0,
                       label="control extractor")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4),
               frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return F.attach(fig, caption_abstention_arms(dose, stats))


def plot_reasons(mix: pd.DataFrame,
                 figsize: Tuple[float, float] = (6.4, 3.8),
                 arm: Optional[str] = None):
    """What the model says it cannot do, in each case.

    The caption is `caption_reasons`, attached to the figure.
    """
    import matplotlib.pyplot as plt

    if mix.empty:
        raise ValueError("no reasons to draw")
    ordered = mix.sort_values("spans_per_trace").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    left = np.zeros(len(ordered))
    for reason in REASON_ORDER:
        if reason not in ordered.columns:
            continue
        width = ordered[reason].to_numpy(float)
        ax.barh(range(len(ordered)), width, left=left, height=0.68,
                color=REASON_COLOUR[reason], edgecolor=S.EDGE,
                linewidth=S.EDGE_LW, label=REASON_LABEL[reason])
        left = left + width

    # The number in brackets is the uncertainty spans for each trace: a data
    # label.
    labels = [f"{r.case_label}  ({r.spans_per_trace:.1f}/trace)"
              for r in ordered.itertuples()]
    ax.set_yticks(range(len(ordered)), labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the uncertainty spans of the case")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4,
              frameon=False, fontsize=6.8)
    fig.tight_layout()
    return F.attach(fig, caption_reasons(mix, arm))


def plot_kind_ladder(tables: pd.DataFrame, summary: pd.DataFrame,
                     figsize: Tuple[float, float] = (6.4, 4.2),
                     xlabel: Optional[str] = None, note: str = "",
                     legend_loc: str = "lower right", name: str = "kind",
                     arm: Optional[str] = None):
    """Each attribute value at its price, with 1 dot for each case.

    The vertical line is 0, which is the base rate of the case. A short row
    says every case gave the value the same price.

    `name` picks the caption text (`KIND_LADDERS`). `note` is more caption
    text; the figure does not draw it. The caption is `caption_kind_ladder`,
    attached to the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if tables.empty or summary.empty:
        raise ValueError("no keys to draw")
    keys = list(summary["key"])[::-1]
    cases = [c for c in WA.CASE_ORDER if c in set(tables["case"])]
    wide = tables.pivot_table(index="key", columns="case", values="lift")

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.EDGE, linewidth=0.8, alpha=0.7, zorder=1)
    for y, key in enumerate(keys):
        row = wide.loc[key].dropna()
        if len(row) > 1:
            ax.plot([row.min(), row.max()], [y, y], color=S.PAL["warmgrey"],
                    linewidth=0.9, alpha=0.55, zorder=2,
                    solid_capstyle="round")
        for case in row.index:
            ax.scatter(row[case], y, s=26, color=_case_colour(case),
                       edgecolors=S.EDGE, linewidths=0.3, alpha=0.95, zorder=3)

    # The number in brackets is the cases that clear the floor: a data label.
    counts = summary.set_index("key")["n_cases"]
    ax.set_yticks(range(len(keys)),
                  [f"{k.replace('_', ' ')}  ({int(counts[k])})"
                   for k in keys])
    ax.set_ylim(-0.8, len(keys) - 0.2)
    ax.set_xlabel(xlabel or "lift over the base rate of the case")
    ax.grid(axis="y", alpha=0.15)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=4.0,
                      markeredgecolor=S.EDGE, markeredgewidth=0.3,
                      markerfacecolor=_case_colour(c),
                      label=WA.case_label(c)) for c in cases]
    if legend_loc == "below":
        # A ladder whose rows fill BOTH ends has no free corner inside.
        ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                  fontsize=6.4, bbox_to_anchor=(0.5, -0.14))
    else:
        ax.legend(handles=handles, loc=legend_loc, frameon=True,
                  framealpha=0.9, fontsize=6.2, ncol=2)
    fig.tight_layout()
    return F.attach(fig, caption_kind_ladder(tables, summary, name, note, arm))


def plot_inference_words(words: pd.DataFrame,
                        kinds: Optional[Sequence[str]] = None,
                        n_each: int = 8, min_keys: int = 3, n_cols: int = 2,
                        panel: Tuple[float, float] = (4.2, 3.8),
                        arm: Optional[str] = None):
    """What the model WRITES when it makes an inference, and what it costs.

    1 panel for each kind. Each panel holds the `n_each` phrases that go with
    the winning photograph most, and the `n_each` that go with it least. A
    dot takes the colour of its pole, thus a panel shows the direction and
    the words at once.

    The shared x label (`fig.supxlabel`) is an axis label, not a caption.
    The caption is `caption_inference_words`, attached to the figure.
    """
    import matplotlib.pyplot as plt

    if words.empty:
        raise ValueError("no words to draw")
    # A kind with almost no repeated phrase draws a 1-row panel that reads as
    # a result.
    counts = words.groupby("kind").size()
    have = [k for k in (kinds or INFERENCE_ORDER)
            if counts.get(k, 0) >= min_keys]
    if not have:
        raise ValueError("no kind to draw")
    n_rows = -(-len(have) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(panel[0] * n_cols, panel[1] * n_rows))
    flat = [a for row in axes for a in row]

    for ax, kind in zip(flat, have):
        group = words[words["kind"] == kind].sort_values("lift")
        take = pd.concat([group.head(n_each), group.tail(n_each)])
        take = take.drop_duplicates("key").sort_values("lift")
        y = list(range(len(take)))
        colours = [S.PAL["coral"] if v < 0 else S.PAL["teal"]
                   for v in take["lift"]]
        ax.axvline(0.0, color=S.EDGE, linewidth=0.8, alpha=0.7, zorder=1)
        ax.hlines(y, 0, take["lift"], color=colours, linewidth=1.4, alpha=0.7,
                  zorder=2)
        ax.scatter(take["lift"], y, s=26, c=colours, edgecolors=S.EDGE,
                   linewidths=0.3, zorder=3)
        ax.set_yticks(y, [_wrap(k, 34) for k in take["key"]], fontsize=6.0)
        ax.set_ylim(-0.8, len(take) - 0.2)
        floor = int(group["min_count"].iloc[0]) if "min_count" in group else 0
        label = f"{kind}  ({int(group['n'].sum()):,} comparisons"
        label += f", floor {floor})" if floor else ")"
        ax.set_xlabel(label, fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.12)
    for ax in flat[len(have):]:
        ax.axis("off")

    fig.supxlabel("lift over the base rate of the case, pooled over the "
                  "cases", fontsize=8)
    fig.tight_layout()
    return F.attach(fig, caption_inference_words(words, kinds, n_each,
                                                 min_keys, arm))


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

CAPTION = (
    "The 4 channels of the reasoning trace that a cue count cannot read. "
    "The model names the AXIS of each comparison in its own words and says "
    "which photograph wins on it; it marks a comparison as hedged; it writes "
    "an uncertainty span with a reason; it reads wealth, class, and safety "
    "INTO a photograph; and it names the flaws of the camera. Each channel "
    "names an image, thus each one takes the same directional test as a cue."
)


# --------------------------------------------------------------------------
# The paper tables
# --------------------------------------------------------------------------

#: The judges of the 2 x 2, the gemma judge first.
JUDGE_ORDER = ("gemma-4-12b", "qwen3.5-9b")

#: The columns of the marker table: (AUC column, printed head, group).
MARKER_TABLE_COLUMNS = (
    ("auc", "all", "count"),
    ("auc_core", "core", "count"),
    ("auc_marker_qualifier", "qualif.", "marker"),
    ("auc_marker_contrastive", "contr.", "marker"),
    ("auc_marker_explicit_doubt", "doubt", "marker"),
    ("auc_marker_reconsideration", "recons.", "marker"),
    ("auc_cannot_see", "cannot see", "reason"),
    ("auc_looks_equal", "look equal", "reason"),
)


def _judge_arms(stats: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    """Give `(judge, [arms])` with the primary arm of each judge first."""
    out = []
    for judge in JUDGE_ORDER:
        sub = stats[stats["judge"] == judge].drop_duplicates("arm")
        if sub.empty:
            continue
        sub = sub.assign(_p=sub["role"].ne("primary"))
        out.append((judge, sub.sort_values(["_p", "arm"])["arm"].tolist()))
    return out


def latex_abstention_auc(stats: pd.DataFrame, min_abstain: int = 25,
                         label: str = "tab:abstention-auc-arms") -> str:
    """The AUC of the total uncertainty count, case x arm, as LaTeX.

    `stats` is the long frame of `abstention_from_traces` over the 2 x 2.
    Each judge gets 1 column for its abstention count and 1 column for each
    extractor, the primary extractor first. A case under `min_abstain`
    abstentions in an arm gives a dash, because `abstention_table` gives
    its AUC as `NaN`.
    """
    if stats is None or stats.empty or "arm" not in stats.columns:
        return "% no abstention results across the arms\n"
    judges = _judge_arms(stats)
    groups = [(f"{L.model_label(j)} judge", 1 + len(arms)) for j, arms in judges]
    heads = ["Case"]
    for _, arms in judges:
        heads.append("abst.")
        heads += [L.model_label(X.ARMS[a]["extractor"]) for a in arms]
    first = judges[0][1][0]
    order = (stats[stats["arm"] == first].sort_values(
        "auc", ascending=False, na_position="last")["case"].tolist())
    order += [c for c in stats["case"].unique() if c not in order]
    body = []
    for case in order:
        cells = [L.case_label(case)]
        for _, arms in judges:
            hit = stats[(stats["arm"] == arms[0]) & (stats["case"] == case)]
            cells.append(L.fmt_int(hit["n_abstain"].iloc[0]) if len(hit) else L.DASH)
            for a in arms:
                hit = stats[(stats["arm"] == a) & (stats["case"] == case)]
                cells.append(L.fmt(hit["auc"].iloc[0]) if len(hit) else L.DASH)
        body.append(L.row(cells))
    n_cols = len(heads)
    caption = (
        "\\textbf{Does the count of doubt predict an abstention?} The AUC of "
        "the number of uncertainty spans in a trace, an abstention against a "
        "decision. Each judge's traces are read by 2 extractors; the first "
        "extractor is the primary arm. \\emph{abst.} counts the abstentions "
        f"of the judge in 1,000 traces. A dash marks fewer than {min_abstain} "
        "abstentions."
    )
    return L.table("_trace_channels", __version__, caption, label,
                   "l" + "r" * (n_cols - 1),
                   L.group_header(groups) + [L.row(heads)], body,
                   resize=n_cols > 6)


def latex_abstention_markers(stats: pd.DataFrame,
                             arms: Optional[Sequence[str]] = None,
                             label: str = "tab:abstention-auc-markers") -> str:
    """The AUC of each marker and reason count, on the primary arms, as LaTeX.

    Each row is 1 case of 1 judge. A case without an AUC (fewer abstentions
    than the floor) leaves the table. `arms` defaults to the primary arm of
    each judge in `stats`.
    """
    if stats is None or stats.empty or "arm" not in stats.columns:
        return "% no abstention results across the arms\n"
    if arms is None:
        arms = [a for _, names in _judge_arms(stats) for a in names
                if arm_role(a) == "primary"]
    cols = [c for c in MARKER_TABLE_COLUMNS if c[0] in stats.columns]
    groups: List[Tuple[str, int]] = []
    for _, _, g in cols:
        if groups and groups[-1][0] == g:
            groups[-1] = (g, groups[-1][1] + 1)
        else:
            groups.append((g, 1))
    n_cols = 1 + len(cols)
    body: List[str] = []
    for arm in arms:
        sub = stats[(stats["arm"] == arm) & stats["auc"].notna()]
        if sub.empty:
            continue
        if body:
            body.append("\\midrule")
        body.append(L.row([f"\\multicolumn{{{n_cols}}}{{l}}"
                           f"{{\\emph{{{L.arm_label(arm)}}}}}"]))
        for _, r in sub.sort_values("auc", ascending=False).iterrows():
            body.append(L.row([f"\\quad {L.case_label(r['case'])}"]
                              + [L.fmt(r[c]) for c, _, _ in cols]))
    caption = (
        "\\textbf{Which kind of doubt predicts an abstention?} The AUC of "
        "1 count alone, on the primary arm of each judge (judge / extractor). "
        "\\emph{core} is the total less the reconsideration spans. A value "
        "below .50 goes with a decision: the judge that says the 2 photographs "
        "look equal answers \\emph{Same}."
    )
    return L.table("_trace_channels", __version__, caption, label,
                   "l" + "r" * len(cols),
                   L.group_header(groups) + [L.row(["Case"] + [h for _, h, _ in cols])],
                   body, resize=True)


def export_latex(arm_stats: pd.DataFrame, out_dir: Path,
                 tag: str = "trace_channels", version: str = "",
                 gate_ok: bool = True) -> List[str]:
    """Write the 2 abstention tables as `<tag>_<table>_v<version>.tex`.

    Warning: give `master/outputs/` as `out_dir`, not `master/figures/`.
    """
    if not gate_ok and not tag.endswith("_stale"):
        tag = f"{tag}_stale"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (("abstention_auc", latex_abstention_auc(arm_stats)),
                       ("abstention_markers", latex_abstention_markers(arm_stats))):
        path = out_dir / f"{tag}_{name}_v{version}.tex"
        path.write_text(text)
        written.append(str(path))
    return written


def export(df: pd.DataFrame, out_dir: Path, tag: str = "trace_channels",
           axis_min_count: int = 400, kind_min_count: int = 25,
           excerpt_rows: int = 4, version: str = "",
           gate_ok: bool = True, across_arms: bool = True,
           arms_tables: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
           tex_dir: Optional[Path] = None, arm: Optional[str] = None
           ) -> List[str]:
    """Write the figures and the tables of all 4 channels.

    Each PNG gets a `.json` sidecar with its caption (`_figmeta.save_figure`).

    Args:
        excerpt_rows: Sentences to quote for each case and inference kind.
        across_arms: Read the 4 version 3 arms and write the abstention of
            the 2 x 2. This adds about a minute and several GB.
        arms_tables: The `(stats, dose)` of `abstention_across_arms`, when
            the caller has them. Then the function reads no arm.
        tex_dir: The folder for the LaTeX tables of the 2 x 2 (see
            `export_latex`). `None` writes no table. The tables do not go
            to `out_dir`, because `out_dir` is `master/figures/`.
        arm: The arm of `df`, for the captions. `None` takes the arm of
            `X.DEFAULT_ROOT`.
    """
    import matplotlib.pyplot as plt

    if not gate_ok:
        tag = f"{tag}_stale"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    arm = arm or arm_of_root()

    def _save(fig, name: str) -> None:
        side = save_figure(fig, out_dir / name, dpi=300, bbox_inches="tight",
                           provenance={"notebook_version": version or None})
        plt.close(fig)
        written.extend([name, side.name])

    def _csv(frame: pd.DataFrame, name: str) -> None:
        if frame is None or frame.empty:
            return
        frame.round(4).to_csv(out_dir / name, index=False)
        written.append(name)

    # 1. The axes.
    axes = axis_table(df, min_count=axis_min_count)
    if not axes.empty:
        _save(plot_axes(axes, arm=arm), f"{tag}_axes.png")
        _save(plot_axis_hedge(axes, hedge_summary(df), arm=arm),
              f"{tag}_axes_hedge.png")
        _csv(axes, f"{tag}_axes.csv")
    _csv(axis_by_case(df), f"{tag}_axes_by_case.csv")

    # 2. The abstention.
    traces = trace_frame(df)
    stats = abstention_table(traces)
    dose = abstention_dose(traces)
    if not stats.empty and stats["auc"].notna().any():
        _save(plot_abstention(dose, stats, arm=arm), f"{tag}_abstention.png")
    _csv(stats, f"{tag}_abstention_by_case.csv")
    _csv(dose, f"{tag}_abstention_dose.csv")
    _csv(auc_long(stats), f"{tag}_abstention_markers.csv")
    mix = reason_mix(df)
    if not mix.empty:
        _save(plot_reasons(mix, arm=arm), f"{tag}_abstention_reasons.png")
        _csv(mix, f"{tag}_abstention_reasons.csv")

    # 2b. The abstention in each arm of the 2 x 2.
    if arms_tables is None and across_arms:
        arms_tables = abstention_across_arms(require_canonical=gate_ok)
    if arms_tables is not None:
        arm_stats, arm_dose = arms_tables
        _csv(arm_stats, f"{tag}_abstention_arms.csv")
        _csv(arm_dose, f"{tag}_abstention_arms_dose.csv")
        if not arm_stats.empty and arm_stats["auc"].notna().any():
            _save(plot_abstention_arms(arm_dose, arm_stats),
                  f"{tag}_abstention_arms.png")
        if tex_dir is not None:
            written += export_latex(arm_stats, tex_dir, tag=tag,
                                    version=version)

    # 3 and 4. The inference and the artifact.
    for name, cls, order in (
        ("inference", INFERENCE_CLASS, INFERENCE_ORDER),
        ("artifact", ARTIFACT_CLASS, ARTIFACT_ORDER),
    ):
        tables = kind_lift(df, cls, min_count=kind_min_count)
        if tables.empty:
            continue
        summary = kind_summary(tables, order=order)
        _save(plot_kind_ladder(tables, summary, name=name, arm=arm),
              f"{tag}_{name}.png")
        _csv(tables, f"{tag}_{name}_lift.csv")
        _csv(summary, f"{tag}_{name}_summary.csv")

    # 3b. The direction of an inference, and the words it is made of.
    poles = pole_lift(df, min_count=kind_min_count)
    if not poles.empty:
        pole_keys = kind_summary(poles, order=None)
        pole_keys = pole_keys.set_index("key").loc[
            pole_key_order(pole_keys)].reset_index()
        _save(plot_kind_ladder(
            poles, pole_keys, figsize=(6.4, 5.0), legend_loc="below",
            name="inference_pole", arm=arm),
            f"{tag}_inference_pole.png")
        _csv(poles, f"{tag}_inference_pole_lift.csv")
        _csv(pole_keys, f"{tag}_inference_pole_summary.csv")
    _csv(pole_mix(df), f"{tag}_inference_pole_mix.csv")
    _csv(inference_coverage(df, kind_min_count=kind_min_count),
         f"{tag}_inference_coverage.csv")

    words = inference_words(df, min_count=kind_min_count)
    if not words.empty:
        _save(plot_inference_words(words, arm=arm), f"{tag}_inference_words.png")
        _csv(words, f"{tag}_inference_words.csv")

    # The sentences behind the numbers. This one needs the source runs on
    # disk, thus it can come back empty on a machine that holds only the
    # extraction corpus.
    quoted = excerpts(df, n_per_cell=excerpt_rows, min_count=kind_min_count)
    _csv(quoted, f"{tag}_inference_excerpts.csv")

    caption = f"{CAPTION}\n\nNotebook {version}, `_trace_channels` {__version__}."
    (out_dir / f"{tag}_caption.txt").write_text(caption + "\n")
    written.append(f"{tag}_caption.txt")
    return written
