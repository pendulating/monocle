"""The version 3 extraction corpora, read on the Integrative Complexity codebook.

Version 3 of the extraction schema (`conf/extract/urban_cues_v3.yaml`) puts
the span classes on the IC codebook. This module holds the arithmetic of 3
notebooks. Each notebook asks 1 or 2 questions:

| Notebook | Question |
|----------|----------|
| `master/ic_perspectives.py` | Which perspectives, inferences, and dimensions go with a win of the image they count for? |
| `master/ic_grounding.py` | Which inferences and dimensions ground each perspective? |
| `master/ic_differentiation.py` | How often does the model doubt a verdict and then keep it? What does it set aside? |

The unit
--------
A trace is `(case, pair_id)`. The 7 cases draw from 1 pair table, thus
`pair_id` alone pools 7 traces. `tid` holds the 2 together.

A free name (`perspective.name`, `inference.name`, `dimension.name`,
`dismissal.target`) is the model's own short phrase. `normalise_names` joins
the spelling variants only (`_cues_across.normalise_cue`): case, hyphens,
articles, degree adverbs, and plurals. `maintenance`, `well maintained`, and
`upkeep` stay 3 keys. Every ranked table applies a minimum count.

Warning: a perspective, an inference, and an uncertainty rate are facts about
the extractor. The Qwen3.8-27B extractor marks about 2x the perspectives and
the uncertainty spans of the Gemma-4-31B extractor on the same traces
(`research/docs/langextract-trace-extraction.md`). Read every count of
these classes beside the other extractor of the same judge.

Warning: this is an association, not a cause. The model narrates while it
decides, thus a span can follow the judgment rather than drive it.

See `research/docs/ic-v3-analysis.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _cues_across as CA
import _latex as L
import _extractions as X
import _figmeta as F
import _style as S
import _words_across as WA
# Re-exported, thus a notebook can call `I.caption_md(fig)`.
from _figmeta import caption_md, captioned, save_figure  # noqa: F401

__version__ = "1.2.0"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: The columns a v3 analysis reads. The loader gives about 60; these 22 hold
#: everything the 3 notebooks need, at about a third of the memory.
KEEP_COLUMNS = (
    "case", "pair_id", "judge_model", "extractor_model", "schema_version",
    "presented_label", "extraction_class", "extraction_text", "char_start",
    "char_end", "image", "valence", "kind", "name", "favors", "wins", "source",
    "label", "marker", "reason", "target", "about",
)

WIN_FOR_A = CA.WIN_FOR_A
WIN_FOR_B = CA.WIN_FOR_B
UNDECIDED = CA.UNDECIDED

#: For each class of question 1: the column that names the image the span
#: counts for, and the attribute that joins the name in the key.
LIFT_CLASSES: Dict[str, Tuple[str, Optional[str]]] = {
    "perspective": ("favors", None),
    "inference": ("image", "kind"),
    "dimension": ("image", "valence"),
}

#: The classes that can ground a perspective (question 2).
PARTNER_CLASSES = ("inference", "dimension")

#: The 3 nesting relations of question 2, by character offsets.
RELATIONS = ("inside_perspective", "contains_perspective", "same_span")

#: The values a grounding heat map can show.
GROUNDING_VALUES = {"log2_lift": "log2 lift of the co-occurrence",
                    "nest_rate": "share of the perspective traces where the 2 spans nest"}

ABOUT_ORDER = ("premise", "cue", "perspective")
ABOUT_COLOUR = {"premise": S.PAL["blue"], "cue": S.PAL["amber"],
                "perspective": S.PAL["coral"], "none": S.PAL["warmgrey"]}

#: The 4 groups of question 3, in the order of the stacked bar.
GROUP_ORDER = ("no_verdict", "steady", "pseudo", "revised")
GROUP_LABEL = {
    "no_verdict": "no verdict before the decision",
    "steady": "verdict, no reconsideration after it",
    "pseudo": "verdict, reconsideration, decision unchanged",
    "revised": "verdict, reconsideration, decision changed",
}
GROUP_COLOUR = {"no_verdict": S.PAL["slate"], "steady": S.PAL["cream"],
                "pseudo": S.PAL["coral"], "revised": S.PAL["teal"]}

#: The shortest dismissal target that a text test may search for. A shorter
#: target ("it", "A") matches almost any clause.
MIN_TARGET_CHARS = 3


# --------------------------------------------------------------------------
# Arms and loading
# --------------------------------------------------------------------------

def v3_arms() -> List[str]:
    """The arms of `X.ARMS` with schema version 3, in their order."""
    return [n for n, s in X.ARMS.items() if s.get("schema_version") == "v3"]


def present_arms() -> List[str]:
    """The v3 arms whose corpus is on disk."""
    return list(X.arm_roots(v3_arms(), present_only=True))


def arm_role(arm: str) -> str:
    """`primary` when `X.PRIMARY_ARMS` names the arm, else `control`."""
    return "primary" if arm in set(X.PRIMARY_ARMS.values()) else "control"


def primary_arms() -> List[str]:
    """The primary arm of each judge, the gemma judge first."""
    return [X.PRIMARY_ARMS[j] for j in ("gemma-4-12b", "qwen3.5-9b")
            if j in X.PRIMARY_ARMS]


def sibling_arm(arm: str) -> Optional[str]:
    """The other v3 arm of the same judge: the same traces, the other
    extractor. `None` when there is none."""
    judge = X.ARMS[arm]["judge"]
    for name in v3_arms():
        if name != arm and X.ARMS[name]["judge"] == judge:
            return name
    return None


def arm_label(arm: str) -> str:
    """`gemma-4-12b traces, qwen3.8-27b extractor (primary)`."""
    spec = X.ARMS[arm]
    return (f"{spec['judge']} traces, {spec['extractor']} extractor "
            f"({arm_role(arm)})")


def trace_id(df: pd.DataFrame) -> pd.Series:
    """The key of 1 trace: `case|pair_id`. `pair_id` alone pools 7 traces."""
    return df["case"].astype(str) + "|" + df["pair_id"].astype(str)


def normalise_names(values: pd.Series) -> pd.Series:
    """Normalise a column of free names, 1 call for each distinct value.

    The rule is `_cues_across.normalise_cue`: lower case, a hyphen becomes a
    space, an article and a degree adverb go, and each word becomes
    singular. A missing name stays missing.
    """
    uniq = pd.unique(values.dropna())
    mapping = {v: CA.normalise_cue(v) for v in uniq}
    return values.map(mapping).astype("object")


def slim(df: pd.DataFrame, arm: Optional[str] = None) -> pd.DataFrame:
    """Keep the columns of `KEEP_COLUMNS`, and add `tid`, `key_name`, and
    `key_target`.

    `key_name` is the normalised `name` (perspective, inference, dimension).
    `key_target` is the normalised `target` (dismissal).
    """
    cols = [c for c in KEEP_COLUMNS if c in df.columns]
    out = df[cols].copy()
    for col in KEEP_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out["tid"] = trace_id(out)
    out["key_name"] = normalise_names(out["name"])
    out["key_target"] = normalise_names(out["target"])
    if arm is not None:
        out["arm"] = arm
    for col in ("case", "extraction_class", "presented_label", "arm"):
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def load_arm(arm: str, require_canonical: bool = True) -> pd.DataFrame:
    """Read 1 v3 arm through `_extractions.load`, then `slim` it.

    The loader applies the paper filters: quotable, not duplicate, judge
    answered, 1 decision for each trace (the last verdict becomes the
    decision when none was written).

    Raises:
        KeyError: when the arm is not a v3 arm.
        FileNotFoundError: when its corpus is not on disk.
    """
    if arm not in v3_arms():
        raise KeyError(f"{arm!r} is not a v3 arm; the v3 arms are {v3_arms()}")
    spec = X.arm_roots([arm], present_only=False)[arm]
    df = X.load(root=spec["root"], require_canonical=require_canonical)
    out = slim(df, arm)
    del df
    return out


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Traces and spans of each IC class, by case: spans for each trace."""
    traces = df.groupby("case", observed=True)["tid"].nunique().rename("traces")
    classes = ["dimension", "inference", "perspective", "verdict",
               "uncertainty", "dismissal", "decision"]
    counts = (df[df["extraction_class"].isin(classes)]
              .groupby(["case", "extraction_class"], observed=True).size()
              .unstack(fill_value=0))
    counts = counts.reindex(columns=[c for c in classes if c in counts.columns])
    per = counts.div(traces, axis=0).round(2)
    per.columns = [f"{c}_per_trace" for c in per.columns]
    out = pd.concat([traces, per], axis=1).reset_index()
    out.insert(1, "case_label", out["case"].map(WA.case_label))
    return out


# --------------------------------------------------------------------------
# Question 1: perspectives, inferences, and dimensions against the vote
# --------------------------------------------------------------------------

def vote_frame(df: pd.DataFrame, extraction_class: str) -> pd.DataFrame:
    """The rows the directional test reads for 1 class.

    | Class | `image` (the image the span counts for) | `key` |
    |-------|------------------------------------------|-------|
    | perspective | `favors` | the name |
    | inference | `image` | `name \\| kind` |
    | dimension | `image` | `name \\| valence` |

    A dimension with valence `bad` on image A thus loses when A wins: its
    lift is negative. An inference carries no valence, thus its `kind` names
    the topic and never the direction.
    """
    side, attr = LIFT_CLASSES[extraction_class]
    part = df[df["extraction_class"] == extraction_class]
    key = part["key_name"].astype("object")
    if attr is not None:
        extra = part[attr].astype("object")
        key = key.where(extra.isna(), key + " | " + extra.astype(str))
        key = key.where(extra.notna(), None)
    out = pd.DataFrame({
        "case": part["case"].astype(str).to_numpy(),
        "pair_id": part["pair_id"].to_numpy(),
        "presented_label": part["presented_label"].astype(str).to_numpy(),
        "image": part[side].astype("object").to_numpy(),
        "key": key.to_numpy(),
    })
    if extraction_class == "perspective":
        out["wins"] = part["wins"].astype("object").to_numpy()
    return out[out["key"].notna()].reset_index(drop=True)


def lift_tables(df: pd.DataFrame, extraction_class: str,
                min_count: int = 25, prior_strength: float = 25.0
                ) -> pd.DataFrame:
    """The lift of each key, for each case, in 1 long table.

    The arithmetic is `_cues_across.keyed_lift`: 1 vote for each
    (trace, key, image); a key on both images in 1 trace is dropped; an
    undecided label (Same, NotSure) does not vote; the win rate is shrunk
    toward the base rate by `prior_strength` pseudo votes.

    The base rate is the win rate of ALL the votes of the class in the case,
    before the count floor. Then the floor drops the keys with fewer than
    `min_count` decided votes.

    Columns: `case`, `key`, `n_mentions`, `n` (decided votes), `wins`,
    `win_rate`, `shrunk_rate`, `ci_low`, `ci_high` (Wilson, of the raw rate),
    `base_rate`, `lift` (`shrunk_rate - base_rate`), `ci_lift_low`,
    `ci_lift_high` (the interval less the base), `pays` and `costs` (the
    interval clears the base), and `extraction_class`. A perspective adds
    `wins_yes_share`: the share of its rows where the extractor wrote
    `wins = yes`.
    """
    part = vote_frame(df, extraction_class)
    if part.empty:
        return pd.DataFrame()
    frames = []
    for case in sorted(part["case"].unique()):
        table = CA.keyed_lift(part, case, min_count=1,
                              prior_strength=prior_strength)
        if table.empty:
            continue
        frames.append(table[table["n"] >= min_count])
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["ci_lift_low"] = out["ci_low"] - out["base_rate"]
    out["ci_lift_high"] = out["ci_high"] - out["base_rate"]
    out["pays"] = out["ci_low"] > out["base_rate"]
    out["costs"] = out["ci_high"] < out["base_rate"]
    out["extraction_class"] = extraction_class
    if extraction_class == "perspective":
        share = (part.assign(_yes=part["wins"].eq("yes"))
                 .groupby(["case", "key"])["_yes"].mean()
                 .rename("wins_yes_share").reset_index())
        out = out.merge(share, on=["case", "key"], how="left")
    return out.sort_values(["case", "lift"], ascending=[True, False]
                           ).reset_index(drop=True)


def all_lift_tables(df: pd.DataFrame, min_count: int = 25,
                    prior_strength: float = 25.0) -> Dict[str, pd.DataFrame]:
    """`lift_tables` for the 3 classes of question 1."""
    return {cls: lift_tables(df, cls, min_count, prior_strength)
            for cls in LIFT_CLASSES}


def case_ends(tables: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """The `n` keys of highest lift and the `n` of lowest, for each case.

    Only a positive lift enters the top and only a negative lift the bottom,
    thus a case with few keys can give fewer than `n` of either. `end` is
    `top` or `bottom`, and `rank` counts from the end.
    """
    if tables.empty:
        return pd.DataFrame()
    frames = []
    for case, grp in tables.groupby("case", sort=True):
        top = grp[grp["lift"] > 0].nlargest(n, "lift").assign(end="top")
        bottom = grp[grp["lift"] < 0].nsmallest(n, "lift").assign(end="bottom")
        for part in (top, bottom):
            part = part.copy()
            part["rank"] = range(1, len(part) + 1)
            frames.append(part)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out.insert(1, "case_label", out["case"].map(WA.case_label))
    return out


def key_summary(tables: pd.DataFrame, min_cases: int = 3) -> pd.DataFrame:
    """1 row for each key over the cases: does its sign hold?

    | Column | Meaning |
    |--------|---------|
    | `n_cases` | cases where the key clears the count floor |
    | `n_pays`, `n_costs` | cases with a positive, a negative lift |
    | `n_sig_pays`, `n_sig_costs` | cases whose interval clears the base |
    | `holds` | `n_cases >= min_cases` and every case has the same sign |
    | `sign` | `+`, `-`, or `mixed` |
    """
    if tables.empty:
        return pd.DataFrame()
    grouped = tables.groupby("key")
    out = pd.DataFrame({
        "n_cases": grouped["case"].nunique(),
        "n": grouped["n"].sum(),
        "lift_mean": grouped["lift"].mean(),
        "lift_min": grouped["lift"].min(),
        "lift_max": grouped["lift"].max(),
        "n_pays": grouped["lift"].apply(lambda s: int((s > 0).sum())),
        "n_costs": grouped["lift"].apply(lambda s: int((s < 0).sum())),
        "n_sig_pays": grouped["pays"].sum().astype(int),
        "n_sig_costs": grouped["costs"].sum().astype(int),
    }).reset_index()
    same = (out["n_pays"] == out["n_cases"]) | (out["n_costs"] == out["n_cases"])
    out["holds"] = same & (out["n_cases"] >= min_cases)
    out["sign"] = np.where(out["n_pays"] == out["n_cases"], "+",
                           np.where(out["n_costs"] == out["n_cases"], "-",
                                    "mixed"))
    return out.sort_values("lift_mean", ascending=False).reset_index(drop=True)


def consistent_keys(tables: pd.DataFrame, min_cases: int = 3,
                    n_each: int = 10) -> pd.DataFrame:
    """The keys that hold 1 sign over at least `min_cases` cases: the
    `n_each` of highest and the `n_each` of lowest mean lift."""
    summary = key_summary(tables, min_cases)
    if summary.empty:
        return summary
    held = summary[summary["holds"]]
    top = held[held["sign"] == "+"].nlargest(n_each, "lift_mean")
    bottom = held[held["sign"] == "-"].nsmallest(n_each, "lift_mean")
    return (pd.concat([top, bottom.iloc[::-1]])
            .sort_values("lift_mean", ascending=False).reset_index(drop=True))


def sign_hold_rate(tables: pd.DataFrame, min_cases: int = 3) -> Dict[str, float]:
    """Of the keys in `min_cases` cases or more, the share that hold 1 sign."""
    summary = key_summary(tables, min_cases)
    if summary.empty:
        return {"n_keys": 0, "n_hold": 0, "share_hold": float("nan")}
    many = summary[summary["n_cases"] >= min_cases]
    n_hold = int(many["holds"].sum())
    return {"n_keys": int(len(many)), "n_hold": n_hold,
            "share_hold": n_hold / len(many) if len(many) else float("nan")}


def wins_check(df: pd.DataFrame) -> pd.DataFrame:
    """Does the extractor's `wins` attribute agree with the vote?

    For each case: of the perspective rows that favor A or B under a decided
    label, the share whose favored image won, split by `wins`. A `wins = yes`
    perspective should win almost always, because the attribute says the
    decision follows it. A low share says the extractor reads `wins` loosely.
    """
    part = vote_frame(df, "perspective")
    part = part[part["image"].isin(["A", "B"])
                & ~part["presented_label"].isin(UNDECIDED)]
    if part.empty:
        return pd.DataFrame()
    won = ((part["image"].eq("A") & part["presented_label"].isin(WIN_FOR_A))
           | (part["image"].eq("B") & part["presented_label"].isin(WIN_FOR_B)))
    part = part.assign(won=won)
    out = (part.groupby(["case", "wins"])["won"].agg(["size", "mean"])
           .rename(columns={"size": "n_rows", "mean": "favored_wins"})
           .reset_index())
    out.insert(1, "case_label", out["case"].map(WA.case_label))
    return out


def extractor_agreement(tables_a: pd.DataFrame, tables_b: pd.DataFrame
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The same keys under the 2 extractors of 1 judge.

    Returns:
        `(merged, summary)`. `merged` holds each (case, class, key) that
        clears the floor in BOTH arms, with `lift_a` and `lift_b`.
        `summary` holds, for each case and class: `n_a`, `n_b` (keys in each
        arm), `n_shared`, `spearman` of the 2 lifts, and `same_sign` (the
        share of the shared keys whose sign agrees).
    """
    cols = ["case", "extraction_class", "key", "n", "lift"]
    if tables_a.empty or tables_b.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = tables_a[cols].merge(tables_b[cols],
                                  on=["case", "extraction_class", "key"],
                                  suffixes=("_a", "_b"))
    rows = []
    for (case, cls), grp_a in tables_a.groupby(["case", "extraction_class"]):
        grp_b = tables_b[(tables_b["case"] == case)
                         & (tables_b["extraction_class"] == cls)]
        shared = merged[(merged["case"] == case)
                        & (merged["extraction_class"] == cls)]
        rho = (shared["lift_a"].corr(shared["lift_b"], method="spearman")
               if len(shared) >= 5 else float("nan"))
        same = (float((np.sign(shared["lift_a"])
                       == np.sign(shared["lift_b"])).mean())
                if len(shared) else float("nan"))
        rows.append({"case": case, "case_label": WA.case_label(case),
                     "extraction_class": cls, "n_a": len(grp_a),
                     "n_b": len(grp_b), "n_shared": len(shared),
                     "spearman": rho, "same_sign": same})
    return merged, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Question 2: what grounds a perspective
# --------------------------------------------------------------------------

def partner_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The inference and dimension spans, with the key of question 1
    (`name | kind`, `name | valence`) as `partner`."""
    frames = []
    for cls in PARTNER_CLASSES:
        attr = LIFT_CLASSES[cls][1]
        part = df[(df["extraction_class"] == cls) & df["key_name"].notna()
                  & df[attr].notna()]
        frames.append(pd.DataFrame({
            "tid": part["tid"].to_numpy(),
            "case": part["case"].astype(str).to_numpy(),
            "partner_class": cls,
            "partner": (part["key_name"].astype(str) + " | "
                        + part[attr].astype(str)).to_numpy(),
            "q_start": part["char_start"].to_numpy(),
            "q_end": part["char_end"].to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def perspective_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The perspective spans with a name: `tid`, `case`, `perspective`,
    `p_start`, `p_end`."""
    part = df[(df["extraction_class"] == "perspective")
              & df["key_name"].notna()]
    return pd.DataFrame({
        "tid": part["tid"].to_numpy(),
        "case": part["case"].astype(str).to_numpy(),
        "perspective": part["key_name"].to_numpy(),
        "p_start": part["char_start"].to_numpy(),
        "p_end": part["char_end"].to_numpy(),
    })


def cooccurrence(df: pd.DataFrame, min_perspective: int = 50,
                 min_pair: int = 25) -> pd.DataFrame:
    """How often each partner key sits in the same trace as a perspective.

    The unit is the trace: a perspective named 3 times and a partner named 2
    times in 1 trace count 1 time.

    | Column | Meaning |
    |--------|---------|
    | `n_perspective_traces` | traces that name the perspective |
    | `n_both` | traces that name the perspective AND the partner |
    | `share` | `n_both / n_perspective_traces` |
    | `expected` | `n_both` if the partner were independent of the perspective inside each case |
    | `lift` | `n_both / expected`; 1 is independence |

    `expected` sums over the cases: the perspective traces of the case times
    the share of the case's traces that name the partner. Thus a pair of a
    road quality perspective and a road quality dimension does not read as
    a link only because both live in 1 case.

    Only a perspective in `min_perspective` traces or more, and a pair in
    `min_pair` traces or more, enter the table.
    """
    pers = perspective_frame(df)[["tid", "case", "perspective"]].drop_duplicates()
    part = partner_frame(df)[["tid", "case", "partner_class", "partner"]
                             ].drop_duplicates()
    if pers.empty or part.empty:
        return pd.DataFrame()
    n_p = pers.groupby("perspective").size()
    keep_p = n_p[n_p >= min_perspective].index
    pers = pers[pers["perspective"].isin(keep_p)]
    pairs = pers.merge(part[["tid", "partner_class", "partner"]], on="tid")
    counts = (pairs.groupby(["perspective", "partner_class", "partner"])
              .size().rename("n_both").reset_index())
    counts = counts[counts["n_both"] >= min_pair]
    if counts.empty:
        return pd.DataFrame()

    traces_case = df.groupby("case", observed=True)["tid"].nunique()
    traces_case.index = traces_case.index.astype(str)
    cases = list(traces_case.index)
    p_case = (pers.groupby(["perspective", "case"]).size().unstack(fill_value=0)
              .reindex(columns=cases, fill_value=0))
    q_case = (part.groupby(["partner_class", "partner", "case"]).size()
              .unstack(fill_value=0).reindex(columns=cases, fill_value=0))
    q_share = q_case.div(traces_case.reindex(cases).to_numpy(), axis=1)
    p_mat = p_case.reindex(counts["perspective"]).to_numpy(dtype=float)
    q_idx = pd.MultiIndex.from_arrays([counts["partner_class"],
                                       counts["partner"]])
    q_mat = q_share.reindex(q_idx).to_numpy(dtype=float)
    counts["expected"] = (p_mat * q_mat).sum(axis=1)
    counts["n_perspective_traces"] = counts["perspective"].map(n_p).astype(int)
    counts["share"] = counts["n_both"] / counts["n_perspective_traces"]
    counts["lift"] = counts["n_both"] / counts["expected"]
    counts["log2_lift"] = np.log2(counts["lift"])
    return counts.sort_values(["perspective", "lift"], ascending=[True, False]
                              ).reset_index(drop=True)


def nest_pairs(pers: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """Every (perspective span, partner span) pair of 1 trace that nests.

    | `relation` | Rule, by character offsets |
    |------------|----------------------------|
    | `inside_perspective` | the partner lies inside the perspective clause |
    | `contains_perspective` | the perspective lies inside the partner clause (the mode word inside an inference) |
    | `same_span` | the 2 spans have the same offsets: 1 clause, 2 classes |

    A span with no offset never nests.
    """
    pairs = pers.merge(part.drop(columns=["case"], errors="ignore"), on="tid")
    pairs = pairs.dropna(subset=["p_start", "p_end", "q_start", "q_end"])
    same = (pairs["q_start"] == pairs["p_start"]) & (pairs["q_end"] == pairs["p_end"])
    inside = ((pairs["q_start"] >= pairs["p_start"])
              & (pairs["q_end"] <= pairs["p_end"]) & ~same)
    contains = ((pairs["p_start"] >= pairs["q_start"])
                & (pairs["p_end"] <= pairs["q_end"]) & ~same)
    relation = np.select([same, inside, contains],
                         ["same_span", "inside_perspective",
                          "contains_perspective"], default="")
    pairs = pairs.assign(relation=relation)
    return pairs[pairs["relation"] != ""].reset_index(drop=True)


def nesting(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The nested pairs, counted 2 ways, 1 case at a time to hold the memory.

    Returns:
        `(by_key, overview)`.

        `by_key`: for each (perspective, partner_class, partner, relation):
        `n_nested` (span pairs) and `n_traces`. The relation `any` counts a
        trace 1 time when it nests the pair in more than 1 relation.

        `overview`: for each case, partner class, and relation: the share of
        the perspective spans with at least 1 such partner (`share_spans`),
        out of `n_perspective_spans`.
    """
    pers_all = perspective_frame(df)
    part_all = partner_frame(df)
    keyed, over = [], []
    for case in sorted(pers_all["case"].unique()):
        pers = pers_all[pers_all["case"] == case].reset_index(drop=True)
        pers["p_row"] = np.arange(len(pers))
        part = part_all[part_all["case"] == case]
        pairs = nest_pairs(pers, part)
        if not pairs.empty:
            keyed.append(pairs.groupby(["perspective", "partner_class",
                                        "partner", "relation"])
                         .agg(n_nested=("tid", "size"),
                              n_traces=("tid", "nunique")).reset_index())
            # A trace can nest 1 pair in 2 relations. The `any` row counts
            # each trace 1 time.
            keyed.append(pairs.groupby(["perspective", "partner_class",
                                        "partner"])
                         .agg(n_nested=("tid", "size"),
                              n_traces=("tid", "nunique")).reset_index()
                         .assign(relation="any"))
        for cls in PARTNER_CLASSES:
            for rel in RELATIONS:
                hit = pairs[(pairs["partner_class"] == cls)
                            & (pairs["relation"] == rel)]["p_row"].nunique() \
                    if not pairs.empty else 0
                over.append({"case": case, "case_label": WA.case_label(case),
                             "partner_class": cls, "relation": rel,
                             "n_perspective_spans": len(pers),
                             "share_spans": hit / len(pers) if len(pers) else
                             float("nan")})
    by_key = (pd.concat(keyed, ignore_index=True) if keyed else pd.DataFrame())
    if not by_key.empty:
        by_key = (by_key.groupby(["perspective", "partner_class", "partner",
                                  "relation"])[["n_nested", "n_traces"]]
                  .sum().reset_index())
    return by_key, pd.DataFrame(over)


def grounding_table(cooc: pd.DataFrame, by_key: pd.DataFrame) -> pd.DataFrame:
    """The co-occurrence table with the nested counts of each relation.

    Adds `n_inside` (partner inside the perspective clause), `n_contains`
    (perspective inside the partner clause), `n_same` (1 clause, 2 classes),
    and `n_nested` (any of the 3), each in traces. `nest_share` is
    `n_nested / n_both`: of the traces that name both, the share where the
    2 spans nest. `nest_rate` is `n_nested / n_perspective_traces`: of the
    traces that name the perspective, the share where this partner holds it
    or sits in it.
    """
    if cooc.empty:
        return pd.DataFrame()
    out = cooc.copy()
    names = {"inside_perspective": "n_inside",
             "contains_perspective": "n_contains", "same_span": "n_same",
             "any": "n_nested"}
    if by_key.empty:
        for col in names.values():
            out[col] = 0
    else:
        wide = (by_key.pivot_table(index=["perspective", "partner_class",
                                          "partner"], columns="relation",
                                   values="n_traces", aggfunc="sum",
                                   fill_value=0)
                .rename(columns=names).reset_index())
        out = out.merge(wide, on=["perspective", "partner_class", "partner"],
                        how="left")
        for col in names.values():
            if col not in out.columns:
                out[col] = 0
            out[col] = out[col].fillna(0).astype(int)
    out["nest_share"] = out["n_nested"] / out["n_both"]
    out["nest_rate"] = out["n_nested"] / out["n_perspective_traces"]
    return out


def top_partners(table: pd.DataFrame, n_perspectives: int = 15, n: int = 5,
                 by: str = "lift") -> pd.DataFrame:
    """For the `n_perspectives` most named perspectives: the `n` partners of
    each class with the highest `by` (`lift`, `n_both`, `n_nested`, or
    `nest_rate`). A rank by `by` keeps only the rows where it is above 0."""
    if table.empty:
        return pd.DataFrame()
    heads = (table.drop_duplicates("perspective")
             .nlargest(n_perspectives, "n_perspective_traces")["perspective"])
    part = table[table["perspective"].isin(heads) & (table[by] > 0)]
    out = (part.sort_values(by, ascending=False)
           .groupby(["perspective", "partner_class"], sort=False).head(n))
    order = {p: i for i, p in enumerate(heads)}
    return (out.assign(_o=out["perspective"].map(order))
            .sort_values(["_o", "partner_class", by],
                         ascending=[True, True, False])
            .drop(columns="_o").reset_index(drop=True))


def nested_partners(by_key: pd.DataFrame, relation: str, n: int = 25
                    ) -> pd.DataFrame:
    """The (perspective, partner) pairs that nest most often in 1 relation."""
    if by_key.empty:
        return pd.DataFrame()
    part = by_key[by_key["relation"] == relation]
    return part.nlargest(n, "n_traces").reset_index(drop=True)


# --------------------------------------------------------------------------
# Question 3: pseudo-differentiation
# --------------------------------------------------------------------------

_DIRECTION = {**{k: "A" for k in WIN_FOR_A}, **{k: "B" for k in WIN_FOR_B},
              **{k: "undecided" for k in UNDECIDED}}


def label_direction(label) -> Optional[str]:
    """`A`, `B`, or `undecided` for a label; `None` for a missing label."""
    return _DIRECTION.get(label) if isinstance(label, str) else None


def differentiation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """1 row for each trace, with the flags of question 3.

    The window: a verdict counts when it starts before the decision; a
    reconsideration counts when it starts after the FIRST verdict and before
    the decision. A span after the decision cannot change it.

    | Column | Meaning |
    |--------|---------|
    | `has_decision` | the trace holds a decision with a label |
    | `n_verdicts` | verdicts before the decision |
    | `has_verdict` | `n_verdicts >= 1` |
    | `n_recon_after` | `uncertainty` spans with marker `reconsideration` in the window |
    | `n_unc_after` | `uncertainty` spans of any marker in the window |
    | `decision_unchanged` | the decision label equals the LAST verdict before it |
    | `unchanged_first` | the decision label equals the FIRST verdict |
    | `direction_unchanged` | the last verdict and the decision name the same winner (A, B, or undecided) |
    | `pseudo` | has_verdict, `n_recon_after >= 1`, decision_unchanged |
    | `revised` | has_verdict, `n_recon_after >= 1`, not decision_unchanged |
    | `pseudo_any`, `revised_any` | the same, with any uncertainty span in place of a reconsideration |
    | `pseudo_first`, `revised_first` | the same, against the first verdict |
    | `group` | `no_verdict`, `steady`, `pseudo`, or `revised` |

    A trace without a decision has every flag False and the group `no_decision`.
    """
    traces = (df[["tid", "case"]].drop_duplicates("tid")
              .assign(case=lambda t: t["case"].astype(str)).set_index("tid"))
    dec = df[df["extraction_class"] == "decision"][["tid", "label", "char_start"]]
    dec = dec[dec["label"].notna()].drop_duplicates("tid", keep="last")
    dec = dec.set_index("tid").rename(columns={"label": "decision",
                                               "char_start": "dec_start"})
    out = traces.join(dec, how="left")
    out["has_decision"] = out["decision"].notna()
    dec_start = out["dec_start"].fillna(np.inf)

    ver = df[(df["extraction_class"] == "verdict") & df["label"].notna()
             ][["tid", "label", "char_start"]].copy()
    ver["dec_start"] = ver["tid"].map(dec_start)
    ver = ver[ver["char_start"].fillna(-1) < ver["dec_start"]]
    ver = ver.sort_values(["tid", "char_start"], kind="stable")
    agg = ver.groupby("tid").agg(n_verdicts=("label", "size"),
                                 first_verdict=("label", "first"),
                                 last_verdict=("label", "last"),
                                 first_start=("char_start", "first"))
    out = out.join(agg, how="left")
    out["n_verdicts"] = out["n_verdicts"].fillna(0).astype(int)
    out["has_verdict"] = out["has_decision"] & (out["n_verdicts"] > 0)

    unc = df[df["extraction_class"] == "uncertainty"][["tid", "char_start",
                                                       "marker"]].copy()
    unc["first_start"] = unc["tid"].map(out["first_start"])
    unc["dec_start"] = unc["tid"].map(dec_start)
    unc = unc[(unc["char_start"] > unc["first_start"])
              & (unc["char_start"] < unc["dec_start"])]
    out["n_unc_after"] = unc.groupby("tid").size().reindex(out.index
                                                           ).fillna(0).astype(int)
    out["n_recon_after"] = (unc[unc["marker"] == "reconsideration"]
                            .groupby("tid").size().reindex(out.index)
                            .fillna(0).astype(int))

    hv = out["has_verdict"]
    out["decision_unchanged"] = hv & (out["decision"] == out["last_verdict"])
    out["unchanged_first"] = hv & (out["decision"] == out["first_verdict"])
    out["direction_unchanged"] = hv & (
        out["decision"].map(label_direction)
        == out["last_verdict"].map(label_direction))
    recon = hv & (out["n_recon_after"] > 0)
    anyu = hv & (out["n_unc_after"] > 0)
    out["reconsiders_after"] = recon
    out["doubts_after"] = anyu
    out["pseudo"] = recon & out["decision_unchanged"]
    out["revised"] = recon & ~out["decision_unchanged"]
    out["pseudo_any"] = anyu & out["decision_unchanged"]
    out["revised_any"] = anyu & ~out["decision_unchanged"]
    out["pseudo_first"] = recon & out["unchanged_first"]
    out["revised_first"] = recon & ~out["unchanged_first"]
    out["revised_direction"] = recon & ~out["direction_unchanged"]
    out["group"] = np.select(
        [~out["has_decision"], ~hv, out["pseudo"], out["revised"]],
        ["no_decision", "no_verdict", "pseudo", "revised"], default="steady")
    return out.reset_index()


def differentiation_summary(flags: pd.DataFrame,
                            by: Sequence[str] = ("case",)) -> pd.DataFrame:
    """The shares of question 3, for each group of `by`.

    Every `share_*` divides by the traces with a decision, except the 3
    `*_given_recon` columns: they divide by the traces with a verdict and a
    reconsideration after it, thus they read "when the model reconsiders,
    how often does the decision move?".

    `mean_recon_<group>` is the mean count of reconsiderations in the window
    for the traces of each group (pseudo, revised). `mean_recon_all_<group>`
    is not given: a trace without a verdict has no window.
    """
    by = list(by)
    rows = []
    base = flags[flags["has_decision"]]
    for keys, grp in (base.groupby(by, sort=True) if by else [((), base)]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(grp)
        recon = grp["reconsiders_after"]
        anyu = grp["doubts_after"]
        row = dict(zip(by, keys))
        row.update({
            "n_traces": n,
            "share_verdict": grp["has_verdict"].mean(),
            "share_recon_after": recon.mean(),
            "share_pseudo": grp["pseudo"].mean(),
            "share_revised": grp["revised"].mean(),
            "revised_given_recon": (grp["revised"].sum() / recon.sum()
                                    if recon.sum() else float("nan")),
            "revised_first_given_recon": (grp["revised_first"].sum() / recon.sum()
                                          if recon.sum() else float("nan")),
            "revised_direction_given_recon": (
                grp["revised_direction"].sum() / recon.sum()
                if recon.sum() else float("nan")),
            "share_pseudo_any": grp["pseudo_any"].mean(),
            "revised_any_given_doubt": (grp["revised_any"].sum() / anyu.sum()
                                        if anyu.sum() else float("nan")),
            "share_pseudo_first": grp["pseudo_first"].mean(),
            "mean_verdicts": grp.loc[grp["has_verdict"], "n_verdicts"].mean(),
            "mean_recon_pseudo": grp.loc[grp["pseudo"], "n_recon_after"].mean(),
            "mean_recon_revised": grp.loc[grp["revised"], "n_recon_after"].mean(),
        })
        for g in GROUP_ORDER:
            row[f"share_{g}"] = float((grp["group"] == g).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if "case" in by and not out.empty:
        out.insert(by.index("case") + 1, "case_label",
                   out["case"].map(WA.case_label))
    return out


def pooled_differentiation(flags: pd.DataFrame) -> Dict[str, float]:
    """The headline shares of question 3 over every case of 1 arm."""
    row = differentiation_summary(flags, by=())
    return {} if row.empty else row.iloc[0].to_dict()


# --------------------------------------------------------------------------
# Question 4: what the model discards
# --------------------------------------------------------------------------

NAMED_CLASSES = ("perspective", "dimension", "inference")


def dismissal_frame(df: pd.DataFrame) -> pd.DataFrame:
    """1 row for each dismissal span, with the flags of question 4.

    A dismissal MATCHES a named span of the same trace (a perspective, a
    dimension, or an inference) by 1 of 2 rules:

    1. name: the span's normalised `name` equals the normalised `target`;
    2. text: the lowered span text holds the lowered `target` or the
       normalised target (`school buildings` finds "a school building"),
       of `MIN_TARGET_CHARS` characters or more.

    | Column | Meaning |
    |--------|---------|
    | `named_<class>` | a span of that class matches, by name |
    | `matched_<class>` | a span of that class matches, by name or by text |
    | `matched_any` | any of the 3 classes matches |
    | `matched_elsewhere` | a match that does NOT overlap the dismissal clause |
    | `named_before` | a match that ends before the dismissal starts: the model named it, then set it aside |

    Warning: a text match inside the same clause is often the same sentence
    marked twice ("I don't see a library building" as a dismissal and as a
    dimension). `matched_elsewhere` and `named_before` leave those out.
    """
    dm = df[df["extraction_class"] == "dismissal"]
    out = pd.DataFrame({
        "tid": dm["tid"].to_numpy(),
        "case": dm["case"].astype(str).to_numpy(),
        "d_row": np.arange(len(dm)),
        "d_start": dm["char_start"].to_numpy(),
        "d_end": dm["char_end"].to_numpy(),
        "target": dm["target"].astype("object").to_numpy(),
        "target_key": dm["key_target"].astype("object").to_numpy(),
        "about": dm["about"].astype("object").fillna("none").to_numpy(),
        "text": dm["extraction_text"].astype("object").to_numpy(),
    })
    flag_cols = ([f"named_{c}" for c in NAMED_CLASSES]
                 + [f"matched_{c}" for c in NAMED_CLASSES]
                 + ["matched_any", "matched_elsewhere", "named_before"])
    for col in flag_cols:
        out[col] = False
    if out.empty:
        return out
    named = df[df["extraction_class"].isin(list(NAMED_CLASSES))
               & df["tid"].isin(set(out["tid"]))]
    named = pd.DataFrame({
        "tid": named["tid"].to_numpy(),
        "n_class": named["extraction_class"].astype(str).to_numpy(),
        "n_key": named["key_name"].astype("object").to_numpy(),
        "n_text": named["extraction_text"].astype(str).str.lower().to_numpy(),
        "n_start": named["char_start"].to_numpy(),
        "n_end": named["char_end"].to_numpy(),
    })
    pairs = out[["d_row", "tid", "d_start", "d_end", "target", "target_key"]
                ].merge(named, on="tid")
    if pairs.empty:
        return out
    tgt = pairs["target"].astype(str).str.lower().str.strip()
    key = pairs["target_key"].astype(str)
    by_name = pairs["target_key"].notna() & (pairs["n_key"] == pairs["target_key"])
    long_enough = pairs["target"].notna() & (tgt.str.len() >= MIN_TARGET_CHARS)
    key_ok = pairs["target_key"].notna() & (key.str.len() >= MIN_TARGET_CHARS)
    by_text = np.array(
        [(lo and t in s) or (ko and k in s) for t, k, s, lo, ko in
         zip(tgt, key, pairs["n_text"], long_enough, key_ok)], dtype=bool)
    hit = by_name | by_text
    overlap = ((pairs["n_start"] < pairs["d_end"])
               & (pairs["n_end"] > pairs["d_start"]))
    before = pairs["n_end"] <= pairs["d_start"]
    pairs = pairs.assign(by_name=by_name, hit=hit,
                         elsewhere=hit & ~overlap, before=hit & before)

    def _rows(mask: pd.Series) -> np.ndarray:
        return np.isin(out["d_row"], pairs.loc[mask, "d_row"].unique())

    for cls in NAMED_CLASSES:
        is_cls = pairs["n_class"] == cls
        out[f"named_{cls}"] = _rows(is_cls & pairs["by_name"])
        out[f"matched_{cls}"] = _rows(is_cls & pairs["hit"])
    out["matched_any"] = _rows(pairs["hit"])
    out["matched_elsewhere"] = _rows(pairs["elsewhere"])
    out["named_before"] = _rows(pairs["before"])
    return out


def about_shares(dm: pd.DataFrame, by: Sequence[str] = ("case",)
                 ) -> pd.DataFrame:
    """Dismissals by `about`: the count and the share, for each group of
    `by`. `n_dismissals` is the total of the group."""
    if dm.empty:
        return pd.DataFrame()
    by = list(by)
    counts = (dm.groupby(by + ["about"]).size().unstack(fill_value=0)
              if by else dm["about"].value_counts().to_frame().T)
    cols = [c for c in ABOUT_ORDER + ("none",) if c in counts.columns]
    counts = counts.reindex(columns=cols, fill_value=0)
    total = counts.sum(axis=1)
    shares = counts.div(total, axis=0)
    shares.columns = [f"share_{c}" for c in shares.columns]
    out = pd.concat([total.rename("n_dismissals"), counts, shares], axis=1)
    out = out.reset_index(drop=not by)
    if "case" in by:
        out.insert(1, "case_label", out["case"].map(WA.case_label))
    return out


def top_targets(dm: pd.DataFrame, n: int = 10, min_count: int = 3,
                by: Sequence[str] = ("case",)) -> pd.DataFrame:
    """The most frequent normalised targets in each group of `by`.

    Columns: `target_key`, `n`, `share` (of the group's dismissals),
    `about_mode` (the most frequent `about`), `share_premise`,
    `share_matched` (`matched_any`), and `share_named_before`.
    """
    if dm.empty:
        return pd.DataFrame()
    by = list(by)
    part = dm[dm["target_key"].notna()]
    totals = part.groupby(by).size() if by else pd.Series({(): len(part)})
    grp = part.groupby(by + ["target_key"])
    table = pd.DataFrame({
        "n": grp.size(),
        "about_mode": grp["about"].agg(lambda s: s.value_counts().index[0]),
        "share_premise": grp["about"].agg(lambda s: float((s == "premise").mean())),
        "share_matched": grp["matched_any"].mean(),
        "share_named_before": grp["named_before"].mean(),
    }).reset_index()
    table = table[table["n"] >= min_count]
    if by:
        table["share"] = table["n"] / table.set_index(by).index.map(totals).to_numpy()
    else:
        table["share"] = table["n"] / len(part)
    out = (table.sort_values(by + ["n"], ascending=[True] * len(by) + [False])
           .groupby(by, sort=False).head(n) if by else table.nlargest(n, "n"))
    out = out.reset_index(drop=True)
    if "case" in by:
        out.insert(1, "case_label", out["case"].map(WA.case_label))
    return out


def match_summary(dm: pd.DataFrame, by: Sequence[str] = ("case",)
                  ) -> pd.DataFrame:
    """How often a dismissed target is a named span of the same trace.

    The share of the dismissals with each flag of `dismissal_frame`, for
    each group of `by`.
    """
    if dm.empty:
        return pd.DataFrame()
    by = list(by)
    flags = ([f"named_{c}" for c in NAMED_CLASSES]
             + [f"matched_{c}" for c in NAMED_CLASSES]
             + ["matched_any", "matched_elsewhere", "named_before"])
    if by:
        out = dm.groupby(by)[flags].mean()
        out.insert(0, "n_dismissals", dm.groupby(by).size())
        out = out.reset_index()
    else:
        out = dm[flags].mean().to_frame().T
        out.insert(0, "n_dismissals", len(dm))
    if "case" in by:
        out.insert(1, "case_label", out["case"].map(WA.case_label))
    return out


def dismissal_rate(df: pd.DataFrame, dm: pd.DataFrame) -> pd.DataFrame:
    """Dismissals for each 100 traces, and the share of traces with one, by
    case."""
    traces = df.groupby("case", observed=True)["tid"].nunique()
    traces.index = traces.index.astype(str)
    n = dm.groupby("case").size().reindex(traces.index, fill_value=0)
    with_one = dm.groupby("case")["tid"].nunique().reindex(traces.index,
                                                           fill_value=0)
    out = pd.DataFrame({"traces": traces, "n_dismissals": n,
                        "per_100_traces": 100 * n / traces,
                        "share_traces_with_one": with_one / traces})
    out = out.reset_index().rename(columns={"index": "case"})
    out.insert(1, "case_label", out["case"].map(WA.case_label))
    return out


# --------------------------------------------------------------------------
# Every arm
# --------------------------------------------------------------------------

def arm_bundle(df: pd.DataFrame, arm: str, min_count: int = 25,
               questions: Sequence[str] = ("perspectives", "grounding",
                                           "differentiation")
               ) -> Dict[str, pd.DataFrame]:
    """Every table of the asked questions for 1 arm, with an `arm` column.

    The keys: `lift_<class>` (question 1), `cooc`, `nest_by_key`,
    `nest_overview`, `grounding` (question 2), `flags`, `diff_case`,
    `dismissals`, `about_case`, `match_case`, `rate_case` (questions 3, 4).
    """
    out: Dict[str, pd.DataFrame] = {}
    if "perspectives" in questions:
        for cls, table in all_lift_tables(df, min_count).items():
            out[f"lift_{cls}"] = table
        out["wins_check"] = wins_check(df)
    if "grounding" in questions:
        out["cooc"] = cooccurrence(df, min_pair=min_count)
        out["nest_by_key"], out["nest_overview"] = nesting(df)
        out["grounding"] = grounding_table(out["cooc"], out["nest_by_key"])
    if "differentiation" in questions:
        out["flags"] = differentiation_flags(df)
        out["diff_case"] = differentiation_summary(out["flags"])
        out["dismissals"] = dismissal_frame(df)
        out["about_case"] = about_shares(out["dismissals"])
        out["match_case"] = match_summary(out["dismissals"])
        out["rate_case"] = dismissal_rate(df, out["dismissals"])
    for name, table in out.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            table.insert(0, "arm", arm)
    return out


def across_arms(arms: Optional[Sequence[str]] = None,
                require_canonical: bool = True,
                questions: Sequence[str] = ("differentiation",),
                min_count: int = 25) -> Dict[str, Dict[str, pd.DataFrame]]:
    """`arm_bundle` for each arm, 1 corpus in memory at a time.

    Warning: the 4 v3 arms take about a minute, and a qwen-trace arm holds
    up to 1.1 M rows.
    """
    names = list(arms) if arms else present_arms()
    out = {}
    for arm in names:
        df = load_arm(arm, require_canonical=require_canonical)
        out[arm] = arm_bundle(df, arm, min_count=min_count, questions=questions)
        del df
    return out


def stack(bundles: Dict[str, Dict[str, pd.DataFrame]], key: str) -> pd.DataFrame:
    """Join 1 table of every arm into 1 long table, with `judge`,
    `extractor`, and `role`."""
    frames = [b[key] for b in bundles.values()
              if key in b and not b[key].empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.insert(1, "judge", out["arm"].map(lambda a: X.ARMS[a]["judge"]))
    out.insert(2, "extractor", out["arm"].map(lambda a: X.ARMS[a]["extractor"]))
    out.insert(3, "role", out["arm"].map(arm_role))
    return out


def differentiation_by_arm(bundles: Dict[str, Dict[str, pd.DataFrame]]
                           ) -> pd.DataFrame:
    """The pooled shares of question 3, 1 row for each arm."""
    rows = []
    for arm, b in bundles.items():
        if "flags" not in b:
            continue
        row = pooled_differentiation(b["flags"].drop(columns="arm",
                                                     errors="ignore"))
        row.update({"arm": arm, "judge": X.ARMS[arm]["judge"],
                    "extractor": X.ARMS[arm]["extractor"],
                    "role": arm_role(arm)})
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    first = ["arm", "judge", "extractor", "role"]
    return out[first + [c for c in out.columns if c not in first]]


def dismissals_by_arm(bundles: Dict[str, Dict[str, pd.DataFrame]]
                      ) -> pd.DataFrame:
    """The pooled `about` shares and match shares of question 4, 1 row for
    each arm."""
    rows = []
    for arm, b in bundles.items():
        dm = b.get("dismissals")
        if dm is None or dm.empty:
            continue
        n_traces = int(b["rate_case"]["traces"].sum())
        row = {"arm": arm, "judge": X.ARMS[arm]["judge"],
               "extractor": X.ARMS[arm]["extractor"], "role": arm_role(arm),
               "n_dismissals": len(dm),
               "per_100_traces": 100 * len(dm) / n_traces if n_traces else
               float("nan")}
        for a in ABOUT_ORDER:
            row[f"share_{a}"] = float((dm["about"] == a).mean())
        for col in ("matched_perspective", "matched_dimension",
                    "matched_inference", "matched_any", "matched_elsewhere",
                    "named_before"):
            row[col] = float(dm[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

class EmptyFigure(ValueError):
    """A figure has no row to draw. The export skips it; any other error
    stops the export."""


def _case_colour(case: str) -> str:
    return WA.CASE_COLOUR.get(case, S.PAL["warmgrey"])


def _wrap(text: str, width: int = 28) -> str:
    """Cut a long key to 1 line of `width` characters."""
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


CLASS_NOUN = {"perspective": "perspective (the image it favors)",
              "inference": "inference (name | kind, on its image)",
              "dimension": "dimension (name | valence, on its image)"}

#: The notebook that shows each figure family, the first tag of a figure.
NOTEBOOK_OF = {"perspectives": "ic_perspectives", "grounding": "ic_grounding",
               "differentiation": "ic_differentiation"}


# --------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------
#
# A figure carries no caption prose. Each `caption_<figure>` function builds
# the caption from the frame that the plot function draws, and the plot
# function attaches it (`_figmeta.attach`). The notebook shows it under the
# figure, and `_figmeta.save_figure` writes it beside the PNG.

def _arm_text(arm: Optional[str]) -> str:
    """1 sentence that names the arm of a figure."""
    spec = X.ARMS.get(arm or "")
    if not spec:
        return "Corpus: the version 3 extraction corpus of the notebook."
    return (f"Corpus: the {spec['judge']} traces, read by the "
            f"{spec['extractor']} extractor (arm `{arm}`, the {arm_role(arm)} "
            f"arm of the judge, schema {spec['schema_version']}, 1,000 traces "
            "for each case).")


def _cases(frame: pd.DataFrame) -> List[str]:
    if frame is None or frame.empty or "case" not in frame.columns:
        return []
    return _ordered_cases(frame["case"])


def _meta(family: str, label: str, title: str, caption: str, measure: str,
          cases: Sequence[str] = (), arms: Sequence[Optional[str]] = ()) -> Dict:
    """The caption content of 1 figure of this module.

    A figure of 1 arm takes the arm as the suffix of its label, as the
    LaTeX tables do (`tab:ic-nesting-gemma-by-qwen27-v3`).
    """
    arms = [a for a in arms if a]
    suffix = arms[0] if len(arms) == 1 else None
    roots = [X.ARMS[a]["root"] for a in arms if a in X.ARMS]
    return F.figure_meta(
        title, caption, F.fig_label("ic", label, suffix),
        [NOTEBOOK_OF.get(family, family), *cases, *arms, "v3", measure],
        module=_MOD, version=__version__,
        arm=arms[0] if len(arms) == 1 else (arms or None),
        corpus_root=roots[0] if len(roots) == 1 else (roots or None))


def caption_case_ends(ends: pd.DataFrame, extraction_class: str = "perspective",
                      arm: Optional[str] = None) -> Dict:
    """The caption of `plot_case_ends`."""
    n_end = int(ends.groupby(["case", "end"]).size().max()) if "end" in ends \
        else None
    noun = CLASS_NOUN.get(extraction_class, extraction_class)
    caption = (
        f"The keys of highest and of lowest lift in each case, for the "
        f"`{extraction_class}` class (a key is the {noun}). 1 panel for each "
        "case; the panel title gives the base rate of the case, the win rate "
        "of all the decided votes of the class in the case."
        + (f" Each panel holds up to {n_end} keys at each end." if n_end else "")
        + " The bar is the lift: the win rate of the image that the span "
        "counts for, shrunk toward the base rate, less the base rate. A teal "
        "bar pays (that image wins more often than the base rate) and a "
        "coral bar costs. The whisker is the 95% Wilson interval of the raw "
        "win rate, less the base rate. The count in brackets is the decided "
        "votes of the key; a key needs 25. Warning: a lift is an "
        "association, not a cause. " + _arm_text(arm))
    return _meta("perspectives", f"{extraction_class}-ends",
                 f"The ends of each case: {extraction_class}", caption,
                 f"{extraction_class}_lift", _cases(ends), [arm])


def caption_consistent(tables: pd.DataFrame, consistent: pd.DataFrame,
                       extraction_class: str = "perspective",
                       arm: Optional[str] = None) -> Dict:
    """The caption of `plot_consistent`."""
    min_cases = int(consistent["n_cases"].min()) if "n_cases" in consistent \
        and len(consistent) else 3
    caption = (
        f"The `{extraction_class}` keys that hold 1 sign of lift in every "
        f"case where they clear the count floor (25 decided votes), in "
        f"{min_cases} cases or more: {len(consistent)} keys. Each row is 1 "
        "key, with 1 dot for each case; the colour is the case. The dot is "
        "the lift over the base rate of the case. The grey line runs from "
        "the lowest to the highest lift of the row. The vertical line is 0, "
        "the base rate. The number beside a row is the cases that clear the "
        "floor. " + _arm_text(arm))
    return _meta("perspectives", f"{extraction_class}-consistent",
                 f"The keys of 1 sign: {extraction_class}", caption,
                 f"{extraction_class}_sign_hold", _cases(tables), [arm])


def caption_extractor_agreement(merged: pd.DataFrame,
                                extraction_class: str = "perspective",
                                labels: Tuple[str, str] = ("primary", "control"),
                                arm: Optional[str] = None) -> Dict:
    """The caption of `plot_extractor_agreement`."""
    part = merged[merged["extraction_class"] == extraction_class]
    rho = part["lift_a"].corr(part["lift_b"], method="spearman")
    caption = (
        f"The lift of each `{extraction_class}` key under the 2 extractors "
        f"of 1 judge: x is the lift under {labels[0]}, y the lift under "
        f"{labels[1]}. 1 dot for each (case, key) that clears the count "
        "floor under both extractors; the colour is the case. A dot on the "
        "diagonal is a key that both extractors price the same. The title "
        f"gives the shared keys ({len(part)}) and the Spearman correlation "
        f"({rho:.2f}). A lift that holds along the diagonal comes from the "
        "traces; a lift that moves comes from the extractor. "
        + _arm_text(arm))
    return _meta("perspectives", f"{extraction_class}-extractor-agreement",
                 f"The 2 extractors of 1 judge: {extraction_class}", caption,
                 f"{extraction_class}_extractor_agreement", _cases(part),
                 [arm])


def caption_grounding(table: pd.DataFrame, partner_class: str = "inference",
                      value: str = "log2_lift", n_perspectives: int = 12,
                      n_partners: int = 16, min_both: int = 25,
                      arm: Optional[str] = None) -> Dict:
    """The caption of `plot_grounding`."""
    if value == "log2_lift":
        read = ("The colour is the log2 lift of the co-occurrence in 1 trace "
                "against independence, inside each case (0 is independence; "
                "teal is above, coral below, on a diverging ramp).")
        name, title = "cooc", "co-occurrence"
    else:
        read = ("The colour is the share of the perspective's traces where "
                "the partner span nests with the perspective span, by "
                "character offsets (a sequential ramp from 0).")
        name, title = "nested", "nesting"
    caption = (
        f"What grounds a perspective: the {n_perspectives} most named "
        f"perspectives (rows) against their `{partner_class}` partner spans "
        f"(columns). The columns are the {n_partners} partners with the "
        "highest value on any row. " + read + f" An empty cell has fewer "
        f"than {min_both} traces that hold both spans. Warning: a "
        "co-occurrence is not a cause. " + _arm_text(arm))
    return _meta("grounding", f"grounding-{partner_class}-{name}",
                 f"What grounds a perspective: {partner_class}, {title}",
                 caption, f"grounding_{value}", [], [arm])


def caption_nesting(overview: pd.DataFrame, arm: Optional[str] = None) -> Dict:
    """The caption of `plot_nesting`."""
    caption = (
        "Where a perspective sits in the trace. For each case, the share of "
        "the perspective spans that nest with a partner span, by character "
        "offsets. The left panel reads the `inference` partners, the right "
        "panel the `dimension` partners. Teal (`inside perspective`): the "
        "partner lies inside the perspective span. Amber (`contains "
        "perspective`): the perspective lies inside the partner clause. "
        "Slate (`same span`): the 2 spans share their offsets. A bar counts "
        "a perspective span with at least 1 such partner; 1 span can count "
        "in more than 1 relation, thus the 3 bars of a case do not sum to "
        "1. "
        + _arm_text(arm))
    return _meta("grounding", "nesting", "Where a perspective sits", caption,
                 "nesting_share", _cases(overview), [arm])


def caption_differentiation(summary: pd.DataFrame,
                            arm: Optional[str] = None) -> Dict:
    """The caption of `plot_differentiation`."""
    groups = "; ".join(f"{GROUP_LABEL[g]}" for g in GROUP_ORDER)
    n = int(summary["n_traces"].sum()) if "n_traces" in summary else None
    caption = (
        "How often the model doubts a verdict and keeps it. 1 bar for each "
        "case; the bar splits the traces with a decision into 4 groups, left "
        f"to right: {groups}. The third group (coral) is the "
        "pseudo-differentiation: a verdict, a reconsideration after it, and "
        "the same decision. The percent inside a segment is its share; a "
        "segment under 7% has no number."
        + (f" {n:,} traces with a decision in all." if n else "")
        + " " + _arm_text(arm))
    return _meta("differentiation", "pseudo-by-case",
                 "Pseudo-differentiation by case", caption, "share_pseudo",
                 _cases(summary), [arm])


DIFF_VALUE_TEXT = {
    "revised_given_recon": (
        "the share of the traces that reconsider where the decision then "
        "changes", "The decision change after a reconsideration"),
    "share_pseudo": (
        "the share of the traces with a decision that are "
        "pseudo-differentiation (a verdict, a reconsideration, the same "
        "decision)", "Pseudo-differentiation in the 2 x 2"),
}


def caption_differentiation_arms(diff_long: pd.DataFrame,
                                 value: str = "revised_given_recon") -> Dict:
    """The caption of `plot_differentiation_arms`."""
    what, title = DIFF_VALUE_TEXT.get(value, (value.replace("_", " "), value))
    arms = [a for a in v3_arms() if a in set(diff_long.get("arm", []))]
    caption = (
        f"The value is {what}, for each case under each arm of the 2 x 2 of "
        "judge and extractor. 1 panel for each judge; the 2 extractors of "
        "the judge read the same traces and share 1 row of the case. The "
        "teal circle is the primary extractor and the tan square is the "
        "control extractor. A value that holds along a row comes from the "
        "traces; a value that moves comes from the extractor. Arms: "
        + ", ".join(f"`{a}`" for a in arms) + ".")
    return _meta("differentiation", f"pseudo-arms-{value}", title, caption,
                 value, _cases(diff_long), arms)


def caption_about(about_case: pd.DataFrame, arm: Optional[str] = None) -> Dict:
    """The caption of `plot_about`."""
    n = int(about_case["n_dismissals"].sum()) if "n_dismissals" in about_case \
        else None
    caption = (
        "What the model sets aside. 1 bar for each case: the share of the "
        "dismissal spans by `about`, the thing the model dismisses. Blue is "
        "a premise (often the facility of the prompt), amber a cue, coral a "
        "perspective, and grey a dismissal with no `about`. The count in "
        "brackets is the dismissals of the case"
        + (f" ({n:,} in all)." if n else ".") + " " + _arm_text(arm))
    return _meta("differentiation", "dismissal-about",
                 "What the model sets aside", caption, "dismissal_about_share",
                 _cases(about_case), [arm])


def caption_top_targets(targets: pd.DataFrame, n: int = 6,
                        arm: Optional[str] = None) -> Dict:
    """The caption of `plot_top_targets`."""
    caption = (
        f"The {n} most frequent dismissal targets of each case, after the "
        "target text is normalised. 1 panel for each case. The bar length "
        "is the count of the dismissals of that target. The colour is the "
        "most frequent `about` of the target: blue a premise, amber a cue, "
        "coral a perspective. A target needs 3 dismissals or more. "
        + _arm_text(arm))
    return _meta("differentiation", "dismissal-targets",
                 "The most dismissed targets", caption, "dismissal_count",
                 _cases(targets), [arm])


# --------------------------------------------------------------------------
# The plot functions
# --------------------------------------------------------------------------

def plot_case_ends(ends: pd.DataFrame, extraction_class: str = "perspective",
                   n_cols: int = 2, panel: Tuple[float, float] = (3.4, 2.1),
                   arm: Optional[str] = None):
    """The top and the bottom keys of each case, at their lift.

    1 panel for each case. A bar is the lift over the base rate of the case;
    the whisker is the Wilson interval of the raw rate, less the base.
    Teal pays (the image the span counts for wins more often than the base),
    coral costs. The caption is `caption_case_ends`, attached to the figure.
    """
    import matplotlib.pyplot as plt

    if ends.empty:
        raise EmptyFigure("no keys to draw")
    cases = [c for c in WA.CASE_ORDER if c in set(ends["case"])]
    n_rows = int(np.ceil(len(cases) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False,
                             figsize=(panel[0] * n_cols, panel[1] * n_rows))
    lo = float(ends["ci_lift_low"].min())
    hi = float(ends["ci_lift_high"].max())
    for ax, case in zip(axes.flat, cases):
        part = ends[ends["case"] == case].sort_values("lift")
        y = np.arange(len(part))
        colours = [S.PAL["teal"] if v > 0 else S.PAL["coral"]
                   for v in part["lift"]]
        ax.barh(y, part["lift"], color=colours, height=0.7, zorder=2)
        ax.errorbar(part["win_rate"] - part["base_rate"], y,
                    xerr=[(part["win_rate"] - part["ci_low"]).clip(lower=0),
                          (part["ci_high"] - part["win_rate"]).clip(lower=0)],
                    fmt="none", ecolor=S.EDGE, elinewidth=0.5, alpha=0.6,
                    zorder=3)
        ax.axvline(0, color=S.EDGE, linewidth=0.7)
        ax.set_yticks(y, [f"{_wrap(k)} ({int(n)})"
                          for k, n in zip(part["key"], part["n"])],
                      fontsize=5.6)
        base = float(part["base_rate"].iloc[0])
        ax.set_title(f"{WA.case_label(case)} (base {base:.2f})",
                     fontsize=7.2, color=_case_colour(case))
        ax.grid(axis="y", alpha=0.0)
        ax.set_xlim(min(lo, -0.05) - 0.02, max(hi, 0.05) + 0.02)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    # The shared x label. The reading of the brackets and the whisker is in
    # the caption.
    fig.supxlabel(
        f"lift of the {CLASS_NOUN.get(extraction_class, extraction_class)} "
        "over the base rate of the case", fontsize=6.6)
    fig.tight_layout()
    return F.attach(fig, caption_case_ends(ends, extraction_class, arm))


def plot_consistent(tables: pd.DataFrame, consistent: pd.DataFrame,
                    extraction_class: str = "perspective",
                    arm: Optional[str] = None):
    """The keys that hold 1 sign over the cases: 1 dot for each case.

    The figure is `_trace_channels.plot_kind_ladder`, thus it reads like the
    inference and artifact ladders of `master/trace_channels.py`.
    """
    import _trace_channels as TC

    if consistent.empty:
        raise EmptyFigure("no key holds its sign")
    part = tables[tables["key"].isin(set(consistent["key"]))]
    height = max(2.6, 0.17 * len(consistent) + 1.0)
    fig = TC.plot_kind_ladder(
        part, consistent, figsize=(6.4, height), legend_loc="below",
        xlabel=f"lift of the {CLASS_NOUN.get(extraction_class, '')} over "
               "the base rate of the case")
    # The ladder puts its legend just under the axis, on top of the x label.
    # Move it under the label.
    legend = fig.axes[0].get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((0.5, -0.32 * 2.6 / height - 0.05))
    return F.attach(fig, caption_consistent(tables, consistent,
                                            extraction_class, arm))


def plot_extractor_agreement(merged: pd.DataFrame,
                             extraction_class: str = "perspective",
                             labels: Tuple[str, str] = ("primary", "control"),
                             arm: Optional[str] = None):
    """The lift of each shared key under the 2 extractors of 1 judge.

    A dot on the diagonal is a key that both extractors price the same. The
    colour is the case.
    """
    import matplotlib.pyplot as plt

    part = merged[merged["extraction_class"] == extraction_class]
    if part.empty:
        raise EmptyFigure("no shared key")
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for case, grp in part.groupby("case"):
        ax.scatter(grp["lift_a"], grp["lift_b"], s=10, alpha=0.8,
                   color=_case_colour(case), edgecolors=S.EDGE,
                   linewidths=0.25, label=WA.case_label(case))
    lim = float(np.nanmax(np.abs(part[["lift_a", "lift_b"]].to_numpy()))) + 0.03
    ax.plot([-lim, lim], [-lim, lim], color=S.EDGE, linewidth=0.6, alpha=0.6)
    ax.axhline(0, color=S.EDGE, linewidth=0.4, alpha=0.5)
    ax.axvline(0, color=S.EDGE, linewidth=0.4, alpha=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"lift, {labels[0]}")
    ax.set_ylabel(f"lift, {labels[1]}")
    rho = part["lift_a"].corr(part["lift_b"], method="spearman")
    ax.set_title(f"{extraction_class}: {len(part)} shared keys, "
                 f"Spearman {rho:.2f}", fontsize=7.2)
    ax.legend(fontsize=5.4, frameon=False, loc="upper left")
    fig.tight_layout()
    return F.attach(fig, caption_extractor_agreement(
        merged, extraction_class, labels, arm))


def plot_grounding(table: pd.DataFrame, partner_class: str = "inference",
                   value: str = "log2_lift", n_perspectives: int = 12,
                   n_partners: int = 16, min_both: int = 25,
                   arm: Optional[str] = None):
    """A heat map: the most named perspectives against their partners.

    `value` is `log2_lift` (the co-occurrence against independence inside
    each case; 0 is independence; a diverging ramp) or `nest_rate` (the share
    of the perspective's traces where the partner nests with it; a
    sequential ramp). The columns are the partners with the highest value on
    any row, among the cells with `min_both` traces or more. An empty cell
    is under the floor.
    """
    import matplotlib.pyplot as plt

    part = table[(table["partner_class"] == partner_class)
                 & (table["n_both"] >= min_both)]
    if value == "nest_rate":
        part = part[part["n_nested"] > 0]
    if part.empty:
        raise EmptyFigure("no partner over the floor")
    heads = (table.drop_duplicates("perspective")
             .nlargest(n_perspectives, "n_perspective_traces")["perspective"]
             .tolist())
    part = part[part["perspective"].isin(heads)]
    cols = (part.groupby("partner")[value].max()
            .nlargest(n_partners).index.tolist())
    grid = (part.pivot_table(index="perspective", columns="partner",
                             values=value)
            .reindex(index=heads, columns=cols))
    fig, ax = plt.subplots(figsize=(0.32 * len(cols) + 2.4,
                                    0.24 * len(heads) + 1.9))
    vals = grid.to_numpy(dtype=float)
    if value == "log2_lift":
        lim = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
        im = ax.imshow(vals, cmap=S.CMAP_DIV, vmin=-lim, vmax=lim, aspect="auto")
    else:
        top = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
        im = ax.imshow(vals, cmap=S.CMAP_SEQ, vmin=0, vmax=top, aspect="auto")
    ax.set_xticks(range(len(cols)), [_wrap(c, 30) for c in cols], rotation=60,
                  ha="right", fontsize=5.6)
    ax.set_yticks(range(len(heads)), heads, fontsize=6.2)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=5.6)
    cbar.set_label(GROUNDING_VALUES.get(value, value), fontsize=6.0)
    ax.set_title(f"What grounds a perspective: the {partner_class} spans",
                 fontsize=7.4)
    fig.tight_layout()
    return F.attach(fig, caption_grounding(table, partner_class, value,
                                           n_perspectives, n_partners,
                                           min_both, arm))


def plot_nesting(overview: pd.DataFrame, arm: Optional[str] = None):
    """The share of the perspective spans that nest with an inference or a
    dimension, by case and relation."""
    import matplotlib.pyplot as plt

    if overview.empty:
        raise EmptyFigure("no nesting")
    cases = [c for c in WA.CASE_ORDER if c in set(overview["case"])]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    colours = {"inside_perspective": S.PAL["teal"],
               "contains_perspective": S.PAL["amber"],
               "same_span": S.PAL["slate"]}
    width = 0.26
    for ax, cls in zip(axes, PARTNER_CLASSES):
        part = overview[overview["partner_class"] == cls]
        for i, rel in enumerate(RELATIONS):
            vals = (part[part["relation"] == rel].set_index("case")
                    ["share_spans"].reindex(cases))
            ax.bar(np.arange(len(cases)) + (i - 1) * width, vals, width,
                   color=colours[rel], label=rel.replace("_", " "))
        ax.set_xticks(range(len(cases)), [WA.case_label(c) for c in cases],
                      rotation=35, ha="right", fontsize=6.0)
        ax.set_title(f"{cls} spans", fontsize=7.2)
        ax.grid(axis="x", alpha=0.0)
    axes[0].set_ylabel("share of the perspective spans")
    axes[1].legend(fontsize=5.6, frameon=False)
    fig.tight_layout()
    return F.attach(fig, caption_nesting(overview, arm))


def plot_differentiation(summary: pd.DataFrame, title: str = "",
                         arm: Optional[str] = None):
    """1 bar for each case: the 4 groups of question 3 as shares of the
    traces with a decision."""
    import matplotlib.pyplot as plt

    if summary.empty:
        raise EmptyFigure("no traces")
    part = summary.set_index("case").reindex(
        [c for c in WA.CASE_ORDER if c in set(summary["case"])])
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    left = np.zeros(len(part))
    y = np.arange(len(part))[::-1]
    for g in GROUP_ORDER:
        vals = part[f"share_{g}"].to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=GROUP_COLOUR[g], height=0.7,
                label=GROUP_LABEL[g])
        # The percent inside a segment is a data label.
        for yy, l, v in zip(y, left, vals):
            if v >= 0.07:
                ax.text(l + v / 2, yy, f"{v:.0%}", ha="center", va="center",
                        fontsize=5.6, color=S.text_on(GROUP_COLOUR[g]))
        left = left + vals
    ax.set_yticks(y, [WA.case_label(c) for c in part.index], fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the traces with a decision")
    ax.grid(axis="y", alpha=0.0)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))
    if title:
        ax.set_title(title, fontsize=7.4)
    fig.tight_layout()
    return F.attach(fig, caption_differentiation(summary, arm))


def plot_differentiation_arms(diff_long: pd.DataFrame,
                              value: str = "revised_given_recon"):
    """`value` for each case under each arm: 1 dot for each arm, the 2
    extractors of 1 judge on 1 row."""
    import matplotlib.pyplot as plt

    if diff_long.empty:
        raise EmptyFigure("no arm")
    judges = sorted(diff_long["judge"].unique())
    fig, axes = plt.subplots(1, len(judges), figsize=(3.3 * len(judges), 2.6),
                             sharex=True, squeeze=False)
    markers = {"primary": "o", "control": "s"}
    for ax, judge in zip(axes.flat, judges):
        part = diff_long[diff_long["judge"] == judge]
        cases = [c for c in WA.CASE_ORDER if c in set(part["case"])]
        y = {c: i for i, c in enumerate(cases[::-1])}
        for arm, grp in part.groupby("arm"):
            role = arm_role(arm)
            ax.scatter(grp[value], grp["case"].map(y), s=22,
                       marker=markers[role],
                       color=S.PAL["teal"] if role == "primary" else S.PAL["tan"],
                       edgecolors=S.EDGE, linewidths=0.3,
                       label=f"{X.ARMS[arm]['extractor']} ({role})", zorder=3)
        ax.set_yticks(list(y.values()), [WA.case_label(c) for c in y],
                      fontsize=6.2)
        ax.set_title(f"{judge} traces", fontsize=7.2)
        ax.set_xlabel(value.replace("_", " "), fontsize=6.6)
        ax.legend(fontsize=5.6, frameon=False, loc="lower right")
    fig.tight_layout()
    return F.attach(fig, caption_differentiation_arms(diff_long, value))


def plot_about(about_case: pd.DataFrame, title: str = "",
               arm: Optional[str] = None):
    """1 bar for each case: the dismissals by `about`, with the count."""
    import matplotlib.pyplot as plt

    if about_case.empty:
        raise EmptyFigure("no dismissal")
    part = about_case.set_index("case").reindex(
        [c for c in WA.CASE_ORDER if c in set(about_case["case"])])
    fig, ax = plt.subplots(figsize=(5.0, 2.4))
    y = np.arange(len(part))[::-1]
    left = np.zeros(len(part))
    for a in ABOUT_ORDER + ("none",):
        col = f"share_{a}"
        if col not in part.columns:
            continue
        vals = part[col].fillna(0).to_numpy(dtype=float)
        ax.barh(y, vals, left=left, color=ABOUT_COLOUR[a], height=0.7, label=a)
        left += vals
    ax.set_yticks(y, [f"{WA.case_label(c)} ({int(n)})" for c, n in
                      zip(part.index, part["n_dismissals"])], fontsize=6.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the dismissals (the count in brackets)")
    ax.grid(axis="y", alpha=0.0)
    ax.legend(fontsize=5.8, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.25))
    if title:
        ax.set_title(title, fontsize=7.4)
    fig.tight_layout()
    return F.attach(fig, caption_about(about_case, arm))


def plot_top_targets(targets: pd.DataFrame, n: int = 6, n_cols: int = 2,
                     panel: Tuple[float, float] = (3.3, 1.6),
                     arm: Optional[str] = None):
    """The most frequent targets of each case, coloured by their `about`."""
    import matplotlib.pyplot as plt

    if targets.empty:
        raise EmptyFigure("no target")
    cases = [c for c in WA.CASE_ORDER if c in set(targets["case"])]
    n_rows = int(np.ceil(len(cases) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(panel[0] * n_cols, panel[1] * n_rows))
    for ax, case in zip(axes.flat, cases):
        part = targets[targets["case"] == case].nlargest(n, "n").iloc[::-1]
        y = np.arange(len(part))
        ax.barh(y, part["n"], color=[ABOUT_COLOUR.get(a, S.PAL["warmgrey"])
                                     for a in part["about_mode"]], height=0.7)
        ax.set_yticks(y, [_wrap(t, 30) for t in part["target_key"]],
                      fontsize=5.6)
        ax.set_title(WA.case_label(case), fontsize=7.0,
                     color=_case_colour(case))
        ax.grid(axis="y", alpha=0.0)
        ax.tick_params(axis="x", labelsize=5.6)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=ABOUT_COLOUR[a], label=a)
                        for a in ABOUT_ORDER], loc="lower center", ncol=3,
               fontsize=6.0, frameon=False, bbox_to_anchor=(0.5, -0.02))
    # The x label sits on the lowest panel of each column, thus it does not
    # touch the legend. The reading of the colour is in the caption.
    flat = list(axes.flat)
    for col in range(n_cols):
        shown = [i for i in range(col, len(cases), n_cols)]
        if shown:
            flat[shown[-1]].set_xlabel("dismissals", fontsize=6.6)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return F.attach(fig, caption_top_targets(targets, n, arm))


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

class _Writer:
    """Write a figure or a table into 1 folder, and keep the names."""

    def __init__(self, out_dir: Path, prefix: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.written: List[str] = []

    def fig(self, make, name: str) -> None:
        """Save the figure with its `.json` caption sidecar."""
        import matplotlib.pyplot as plt
        try:
            fig = make()
        except EmptyFigure:
            return
        side = save_figure(fig, self.out_dir / f"{self.prefix}_{name}.png",
                           dpi=300, bbox_inches="tight")
        plt.close(fig)
        self.written += [f"{self.prefix}_{name}.png", side.name]

    def csv(self, frame: Optional[pd.DataFrame], name: str) -> None:
        if frame is None or frame.empty:
            return
        frame.round(4).to_csv(self.out_dir / f"{self.prefix}_{name}.csv",
                              index=False)
        self.written.append(f"{self.prefix}_{name}.csv")


def _prefix(tag: str, arm: str, gate_ok: bool) -> str:
    return f"{tag}_{arm}" + ("" if gate_ok else "_stale")


def export_perspectives(bundle: Dict[str, pd.DataFrame], out_dir: Path,
                        arm: str, gate_ok: bool = True, n_ends: int = 5,
                        min_cases: int = 3,
                        sibling: Optional[Dict[str, pd.DataFrame]] = None,
                        tag: str = "ic_perspectives") -> List[str]:
    """Write the files of question 1 for 1 arm.

    `sibling` is the bundle of the other extractor of the judge; with it the
    function writes the agreement of the 2 extractors too.
    """
    w = _Writer(out_dir, _prefix(tag, arm, gate_ok))
    for cls in LIFT_CLASSES:
        tables = bundle.get(f"lift_{cls}", pd.DataFrame())
        if tables.empty:
            continue
        ends = case_ends(tables, n_ends)
        held = consistent_keys(tables, min_cases)
        w.csv(tables, f"{cls}_lift")
        w.csv(ends, f"{cls}_ends")
        w.csv(key_summary(tables, min_cases), f"{cls}_summary")
        w.fig(lambda: plot_case_ends(ends, cls, arm=arm), f"{cls}_ends")
        w.fig(lambda: plot_consistent(tables, held, cls, arm=arm),
              f"{cls}_consistent")
        if sibling is not None:
            merged, agree = extractor_agreement(
                tables, sibling.get(f"lift_{cls}", pd.DataFrame()))
            w.csv(agree, f"{cls}_extractor_agreement")
            w.fig(lambda: plot_extractor_agreement(
                merged, cls, (X.ARMS[arm]["extractor"],
                              X.ARMS[sibling_arm(arm)]["extractor"]),
                arm=arm),
                f"{cls}_extractor_agreement")
    w.csv(bundle.get("wins_check"), "wins_check")
    return w.written


def export_grounding(bundle: Dict[str, pd.DataFrame], out_dir: Path, arm: str,
                     gate_ok: bool = True, tag: str = "ic_grounding"
                     ) -> List[str]:
    """Write the files of question 2 for 1 arm."""
    w = _Writer(out_dir, _prefix(tag, arm, gate_ok))
    table = bundle.get("grounding", pd.DataFrame())
    w.csv(table, "table")
    w.csv(top_partners(table, by="lift"), "top_by_lift")
    w.csv(top_partners(table, by="n_both"), "top_by_count")
    w.csv(top_partners(table, by="nest_rate"), "top_by_nesting")
    by_key = bundle.get("nest_by_key", pd.DataFrame())
    for rel in RELATIONS:
        w.csv(nested_partners(by_key, rel), f"nested_{rel}")
    w.csv(bundle.get("nest_overview"), "nesting_overview")
    for cls in PARTNER_CLASSES:
        for value, name in (("log2_lift", "cooc"), ("nest_rate", "nested")):
            w.fig(lambda: plot_grounding(table, cls, value, arm=arm),
                  f"heatmap_{cls}_{name}")
    w.fig(lambda: plot_nesting(bundle.get("nest_overview", pd.DataFrame()),
                               arm=arm),
          "nesting")
    return w.written


def export_differentiation(bundle: Dict[str, pd.DataFrame], out_dir: Path,
                           arm: str, gate_ok: bool = True,
                           tag: str = "ic_differentiation") -> List[str]:
    """Write the files of questions 3 and 4 for 1 arm."""
    w = _Writer(out_dir, _prefix(tag, arm, gate_ok))
    w.csv(bundle.get("diff_case"), "pseudo_by_case")
    w.fig(lambda: plot_differentiation(bundle.get("diff_case", pd.DataFrame()),
                                       arm_label(arm), arm=arm),
          "pseudo_by_case")
    dm = bundle.get("dismissals", pd.DataFrame())
    w.csv(bundle.get("about_case"), "dismissal_about")
    w.csv(bundle.get("match_case"), "dismissal_match")
    w.csv(bundle.get("rate_case"), "dismissal_rate")
    targets = top_targets(dm.drop(columns="arm", errors="ignore"), n=10)
    w.csv(targets, "dismissal_targets")
    w.fig(lambda: plot_about(bundle.get("about_case", pd.DataFrame()),
                             arm_label(arm), arm=arm), "dismissal_about")
    w.fig(lambda: plot_top_targets(targets, arm=arm), "dismissal_targets")
    return w.written


def export_across_arms(bundles: Dict[str, Dict[str, pd.DataFrame]],
                       out_dir: Path, gate_ok: bool = True) -> List[str]:
    """Write the tables of every arm side by side: question 3 and 4 by arm,
    and the nesting overview by arm."""
    suffix = "" if gate_ok else "_stale"
    w = _Writer(out_dir, f"ic_differentiation_arms{suffix}")
    diff_long = stack(bundles, "diff_case")
    w.csv(differentiation_by_arm(bundles), "pooled")
    w.csv(diff_long, "by_case")
    w.fig(lambda: plot_differentiation_arms(diff_long), "revised_given_recon")
    w.fig(lambda: plot_differentiation_arms(diff_long, "share_pseudo"),
          "share_pseudo")
    w.csv(dismissals_by_arm(bundles), "dismissal_pooled")
    w.csv(stack(bundles, "about_case"), "dismissal_about")
    written = list(w.written)
    g = _Writer(out_dir, f"ic_grounding_arms{suffix}")
    g.csv(stack(bundles, "nest_overview"), "nesting_overview")
    return written + g.written


# --------------------------------------------------------------------------
# Paper tables (LaTeX)
# --------------------------------------------------------------------------
#
# Each `latex_<table>` function writes 1 float in the house convention of
# `_latex.py`. The functions read the frames that the notebooks compute, and
# `export_*_latex` writes them next to the CSV files of `master/outputs/`.

_MOD = "_ic_v3"


def _tex_key(arm: str) -> str:
    """The `\\label` suffix of an arm: `gemma_by_qwen27_v3` -> `gemma-by-qwen27-v3`."""
    return arm.replace("_", "-")


def _key_tex(name) -> str:
    """Escape a free-text key. A `|` (the join of `name | kind`) prints as a
    dash in the OT1 text font, thus it becomes `\\textbar{}`."""
    return L.escape(name).replace("|", r"\textbar{}")


def _groups(groups, lead: int, n_cols: int) -> List[str]:
    """`L.group_header`, with empty cells after the last group, thus the
    header row has the column count of the tabular."""
    lines = L.group_header(groups, lead=lead)
    pad = n_cols - lead - sum(w for _, w in groups)
    if pad > 0:
        lines[0] = lines[0][:-len(" \\\\")] + " &" * pad + " \\\\"
    return lines


def _ordered_cases(cases) -> List[str]:
    present = set(cases)
    order = [c for c in WA.CASE_ORDER if c in present]
    return order + sorted(present - set(order))


def latex_perspective_ends(ends: pd.DataFrame, arm: str, n: int = 3) -> str:
    """The `n` perspectives of highest and of lowest lift in each case.

    `ends` is the frame of `case_ends` for the perspective class. Each case
    takes `n` rows: a top key and its lift on the left, a bottom key and its
    lift on the right. The base rate of the case stands on its first row.
    """
    head = [L.row(["", "", r"\multicolumn{2}{c}{highest lift}",
                   r"\multicolumn{2}{c}{lowest lift}"]),
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            L.row(["Case", "base", "perspective", "lift", "perspective", "lift"])]
    body: List[str] = []
    if not ends.empty:
        for i, case in enumerate(_ordered_cases(ends["case"])):
            grp = ends[ends["case"] == case]
            top = grp[grp["end"] == "top"].sort_values("rank").head(n)
            bot = grp[grp["end"] == "bottom"].sort_values("rank").head(n)
            if i:
                body.append(r"\midrule")
            for k in range(max(len(top), len(bot), 1)):
                cells = [L.case_label(case) if k == 0 else "",
                         L.fmt(grp["base_rate"].iloc[0]) if k == 0 else ""]
                for part in (top, bot):
                    if k < len(part):
                        r = part.iloc[k]
                        cells += [_key_tex(r["key"]), L.fmt(r["lift"], signed=True)]
                    else:
                        cells += [L.DASH, L.DASH]
                body.append(L.row(cells))
    caption = (
        r"\textbf{Which perspectives go with a win of the image they favor?} "
        f"{L.arm_label(arm)} (judge / extractor). The {n} perspectives of highest "
        "and of lowest lift in each case. The lift is the shrunk win rate less "
        "the base rate of the case (\\emph{base}); a key needs 25 decided votes.")
    return L.table(_MOD, __version__, caption,
                   f"tab:ic-perspective-ends-{_tex_key(arm)}", "lrlrlr",
                   head, body, resize=True)


def latex_sign_table(tables: pd.DataFrame, arm: str, min_cases: int = 3) -> str:
    """The cross-case sign table: the lift of each perspective in each case.

    `tables` is the perspective lift frame (`lift_tables`). A row is a key in
    `min_cases` cases or more, by mean lift. A dash marks a case where the key
    does not clear the count floor. A bullet in the last column marks a
    key that holds 1 sign in every case where it clears the floor.
    """
    summary = key_summary(tables, min_cases) if not tables.empty else pd.DataFrame()
    cases = _ordered_cases(tables["case"]) if not tables.empty else []
    head = [L.row(["Perspective"] + [L.case_label(c) for c in cases]
                  + ["mean", "1 sign"])]
    body: List[str] = []
    if not summary.empty:
        many = summary[summary["n_cases"] >= min_cases].sort_values(
            "lift_mean", ascending=False)
        lift = tables.set_index(["key", "case"])["lift"]
        for _, r in many.iterrows():
            cells = [_key_tex(r["key"])]
            for c in cases:
                v = lift.get((r["key"], c), float("nan"))
                cells.append(L.fmt(v, signed=True))
            cells += [L.fmt(r["lift_mean"], signed=True),
                      r"$\bullet$" if bool(r["holds"]) else ""]
            body.append(L.row(cells))
    rate = sign_hold_rate(tables, min_cases) if not tables.empty else \
        {"n_keys": 0, "n_hold": 0}
    caption = (
        r"\textbf{Does a perspective keep its sign across the cases?} "
        f"{L.arm_label(arm)} (judge / extractor). The lift of each perspective "
        f"in each case, for the {rate['n_keys']} perspectives in {min_cases} "
        f"cases or more; {rate['n_hold']} hold 1 sign in every case "
        r"($\bullet$). A dash marks a case under the floor of 25 decided votes.")
    return L.table(_MOD, __version__, caption,
                   f"tab:ic-perspective-signs-{_tex_key(arm)}",
                   "l" + "r" * (len(cases) + 1) + "c", head, body, resize=True)


def latex_nesting(overview: pd.DataFrame, arm: str) -> str:
    """The share of the perspective spans that nest with a partner, by case.

    `overview` is the second frame of `nesting`. The columns hold the 3
    relations for the inference and for the dimension partners.
    """
    rel_head = {"contains_perspective": "in", "inside_perspective": "holds",
                "same_span": "same"}
    rels = ("contains_perspective", "inside_perspective", "same_span")
    head = [*L.group_header([("inference", 3), ("dimension", 3)], lead=2),
            L.row(["Case", "spans"] + [rel_head[r] for r in rels] * 2)]
    body: List[str] = []
    if not overview.empty:
        for case in _ordered_cases(overview["case"]):
            grp = overview[overview["case"] == case]
            cells = [L.case_label(case), L.fmt_int(grp["n_perspective_spans"].iloc[0])]
            for cls in PARTNER_CLASSES:
                for rel in rels:
                    hit = grp[(grp["partner_class"] == cls) & (grp["relation"] == rel)]
                    cells.append(L.fmt_pct(hit["share_spans"].iloc[0], 1)
                                 if len(hit) else L.DASH)
            body.append(L.row(cells))
    caption = (
        r"\textbf{Where does a perspective sit in the trace?} "
        f"{L.arm_label(arm)} (judge / extractor). The percent of the perspective "
        r"spans that lie inside a partner clause (\emph{in}), hold a partner "
        r"(\emph{holds}), or share its offsets (\emph{same}).")
    return L.table(_MOD, __version__, caption,
                   f"tab:ic-nesting-{_tex_key(arm)}", "lrrrrrrr", head, body,
                   resize=True)


def latex_nesting_arms(overview_long: pd.DataFrame,
                       partner_class: str = "inference",
                       relation: str = "contains_perspective") -> str:
    """The share of the perspective spans inside a partner clause, for the 4
    arms side by side (the 2 x 2 of judge and extractor).

    `overview_long` is `stack(bundles, "nest_overview")`.
    """
    part = overview_long[(overview_long["partner_class"] == partner_class)
                         & (overview_long["relation"] == relation)] \
        if not overview_long.empty else overview_long
    arms = [a for a in v3_arms() if not part.empty and a in set(part["arm"])]
    judges: List[str] = []
    for a in arms:
        if X.ARMS[a]["judge"] not in judges:
            judges.append(X.ARMS[a]["judge"])
    arms = [a for j in judges for a in arms if X.ARMS[a]["judge"] == j]
    groups = [(L.model_label(j), sum(X.ARMS[a]["judge"] == j for a in arms))
              for j in judges]
    head = [*L.group_header(groups, lead=1),
            L.row(["Case"] + [L.model_label(X.ARMS[a]["extractor"])
                              + ("$^{*}$" if arm_role(a) == "primary" else "")
                              for a in arms])]
    body: List[str] = []
    if arms:
        for case in _ordered_cases(part["case"]):
            cells = [L.case_label(case)]
            for a in arms:
                hit = part[(part["case"] == case) & (part["arm"] == a)]
                cells.append(L.fmt_pct(hit["share_spans"].iloc[0], 1)
                             if len(hit) else L.DASH)
            body.append(L.row(cells))
    caption = (
        r"\textbf{Is the nesting of a perspective a fact of the trace or of "
        r"the extractor?} The percent of the perspective spans that lie inside "
        f"an {partner_class} clause, by judge (group) and extractor (column). "
        r"$^{*}$ marks the primary arm of the judge.")
    return L.table(_MOD, __version__, caption, "tab:ic-nesting-arms",
                   "l" + "r" * len(arms), head, body, resize=True)


def latex_strongest_nests(grounding: pd.DataFrame, arm: str, n: int = 10,
                          min_traces: int = 100) -> str:
    """The perspectives whose spans nest most often with 1 partner.

    `grounding` is the frame of `grounding_table`. Each perspective in
    `min_traces` traces or more gives its 1 partner of highest `nest_rate`;
    the table keeps the `n` highest.
    """
    head = [L.row(["Perspective", "partner", "class", "traces", "nest", "lift"])]
    body: List[str] = []
    if not grounding.empty:
        part = grounding[grounding["n_perspective_traces"] >= min_traces]
        part = (part.sort_values("nest_rate", ascending=False)
                .drop_duplicates("perspective").head(n))
        for _, r in part.iterrows():
            body.append(L.row([_key_tex(r["perspective"]), _key_tex(r["partner"]),
                               L.escape(r["partner_class"]),
                               L.fmt_int(r["n_perspective_traces"]),
                               L.fmt(r["nest_rate"]), L.fmt(r["lift"])]))
    caption = (
        r"\textbf{Which clause holds each perspective?} "
        f"{L.arm_label(arm)} (judge / extractor). For the perspectives in "
        f"{min_traces} traces or more, the partner they nest with most often. "
        r"\emph{nest} is the share of the perspective's traces where the 2 "
        r"spans nest; \emph{lift} is the co-occurrence lift (1 is independence).")
    return L.table(_MOD, __version__, caption,
                   f"tab:ic-strongest-nests-{_tex_key(arm)}", "lllrrr",
                   head, body, resize=True)


_DIFF_COLUMNS = (("n_traces", "traces"), ("share_verdict", "verdict"),
                 ("share_pseudo", "pseudo"), ("share_revised", "revised"),
                 ("revised_given_recon", "chg."),
                 ("revised_direction_given_recon", "win."))


def latex_pseudo_by_case(diff_case: pd.DataFrame, arm: str) -> str:
    """The shares of question 3 by case, for 1 arm (`differentiation_summary`)."""
    head = [*L.group_header([("share of the traces", 3),
                             ("given a reconsideration", 2)], lead=2),
            L.row(["Case"] + [h for _, h in _DIFF_COLUMNS])]
    body: List[str] = []
    if not diff_case.empty:
        for case in _ordered_cases(diff_case["case"]):
            r = diff_case[diff_case["case"] == case].iloc[0]
            body.append(L.row([L.case_label(case), L.fmt_int(r["n_traces"])]
                              + [L.fmt(r[c]) for c, _ in _DIFF_COLUMNS[1:]]))
    caption = (
        r"\textbf{How often does the model doubt a verdict and keep it?} "
        f"{L.arm_label(arm)} (judge / extractor). \\emph{{pseudo}}: a verdict, "
        "a reconsideration after it, and the same decision; \\emph{revised}: "
        "the decision changes. Of the traces that reconsider, \\emph{chg.} is "
        r"the share where the decision changes and \emph{win.} where the winner changes.")
    return L.table(_MOD, __version__, caption,
                   f"tab:ic-pseudo-by-case-{_tex_key(arm)}", "lrrrrrr",
                   head, body, resize=True)


def latex_pseudo_arms(diff_arms: pd.DataFrame,
                      dism_arms: Optional[pd.DataFrame] = None) -> str:
    """The pooled shares of question 3 for the 4 arms (the 2 x 2 of judge and
    extractor), with the dismissal rate of question 4 when `dism_arms` is
    given (`differentiation_by_arm`, `dismissals_by_arm`)."""
    with_dm = dism_arms is not None and not dism_arms.empty
    head = [*_groups([("share of the traces", 3),
                      ("given a reconsideration", 2)], 2, 7 + int(with_dm)),
            L.row(["Judge / extractor"] + [h for _, h in _DIFF_COLUMNS]
                  + (["dism."] if with_dm else []))]
    body: List[str] = []
    last_judge = None
    if not diff_arms.empty:
        order = {a: i for i, a in enumerate(v3_arms())}
        frame = diff_arms.assign(_o=diff_arms["arm"].map(order)).sort_values(
            ["judge", "_o"])
        for _, r in frame.iterrows():
            if last_judge is not None and r["judge"] != last_judge:
                body.append(r"\midrule")
            last_judge = r["judge"]
            name = L.arm_label(r["arm"]) + ("$^{*}$" if r["role"] == "primary" else "")
            cells = [name, L.fmt_int(r["n_traces"])] + [
                L.fmt(r[c]) for c, _ in _DIFF_COLUMNS[1:]]
            if with_dm:
                hit = dism_arms[dism_arms["arm"] == r["arm"]]
                cells.append(f"{hit['per_100_traces'].iloc[0]:.0f}" if len(hit)
                             else L.DASH)
            body.append(L.row(cells))
    caption = (
        r"\textbf{Is pseudo-differentiation a fact of the trace or of the "
        r"extractor?} The shares of the traces with a decision, for each judge "
        r"and extractor. Columns as in the by-case table"
        + (r"; \emph{dism.} is the dismissals for each 100 traces" if with_dm else "")
        + r". $^{*}$ marks the primary arm of the judge.")
    return L.table(_MOD, __version__, caption, "tab:ic-pseudo-arms",
                   "lrrrrrr" + ("r" if with_dm else ""), head, body, resize=True)


def latex_dismissals(about_case: pd.DataFrame, rate_case: pd.DataFrame,
                     targets: pd.DataFrame, arm: str) -> str:
    """The dismissals of 1 arm by case: the rate, the `about` shares, and the
    most frequent target (`about_shares`, `dismissal_rate`, `top_targets`)."""
    head = [*_groups([("share by \\emph{about}", 3)], 2, 6),
            L.row(["Case", "per 100", "premise", "cue", "persp.", "top target ($n$)"])]
    body: List[str] = []
    if not about_case.empty:
        for case in _ordered_cases(about_case["case"]):
            a = about_case[about_case["case"] == case].iloc[0]
            rate = rate_case[rate_case["case"] == case] if not rate_case.empty \
                else rate_case
            tops = targets[targets["case"] == case].sort_values("n", ascending=False) \
                if not targets.empty else targets
            top = (f"{_key_tex(tops['target_key'].iloc[0])} ({L.fmt_int(tops['n'].iloc[0])})"
                   if len(tops) else L.DASH)
            body.append(L.row(
                [L.case_label(case),
                 f"{rate['per_100_traces'].iloc[0]:.0f}" if len(rate) else L.DASH]
                + [L.fmt(a.get(f"share_{x}", float("nan"))) for x in ABOUT_ORDER]
                + [top]))
    caption = (
        r"\textbf{What does the model set aside?} "
        f"{L.arm_label(arm)} (judge / extractor). The dismissals for each 100 "
        r"traces, the share of each \emph{about}, and the most frequent "
        "normalised target. A premise target is often the facility of the prompt.")
    return L.table(_MOD, __version__, caption,
                   f"tab:ic-dismissals-{_tex_key(arm)}", "lrrrrl", head, body,
                   resize=True)


def _write_tex(out_dir: Path, prefix: str, name: str, version: str,
               tex: str) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{name}_v{version}.tex"
    (out_dir / fname).write_text(tex)
    return fname


def export_perspectives_latex(bundle: Dict[str, pd.DataFrame], out_dir: Path,
                              arm: str, version: str, gate_ok: bool = True,
                              n_ends: int = 3, min_cases: int = 3,
                              tag: str = "ic_perspectives") -> List[str]:
    """Write the 2 tables of question 1 for 1 arm: the ends and the signs."""
    tables = bundle.get("lift_perspective", pd.DataFrame())
    if tables.empty:
        return []
    prefix = _prefix(tag, arm, gate_ok)
    return [
        _write_tex(out_dir, prefix, "ends", version,
                   latex_perspective_ends(case_ends(tables, n_ends), arm, n_ends)),
        _write_tex(out_dir, prefix, "signs", version,
                   latex_sign_table(tables, arm, min_cases)),
    ]


def export_grounding_latex(bundle: Dict[str, pd.DataFrame], out_dir: Path,
                           arm: str, version: str, gate_ok: bool = True,
                           tag: str = "ic_grounding") -> List[str]:
    """Write the 2 tables of question 2 for 1 arm: the nesting shares and the
    strongest nests."""
    over = bundle.get("nest_overview", pd.DataFrame())
    if over.empty:
        return []
    prefix = _prefix(tag, arm, gate_ok)
    return [
        _write_tex(out_dir, prefix, "nesting", version, latex_nesting(over, arm)),
        _write_tex(out_dir, prefix, "strongest_nests", version,
                   latex_strongest_nests(bundle.get("grounding", pd.DataFrame()), arm)),
    ]


def export_differentiation_latex(bundle: Dict[str, pd.DataFrame], out_dir: Path,
                                 arm: str, version: str, gate_ok: bool = True,
                                 tag: str = "ic_differentiation") -> List[str]:
    """Write the 2 tables of questions 3 and 4 for 1 arm."""
    diff = bundle.get("diff_case", pd.DataFrame())
    if diff.empty:
        return []
    prefix = _prefix(tag, arm, gate_ok)
    dm = bundle.get("dismissals", pd.DataFrame())
    targets = top_targets(dm.drop(columns="arm", errors="ignore"), n=10) \
        if not dm.empty else pd.DataFrame()
    return [
        _write_tex(out_dir, prefix, "pseudo_by_case", version,
                   latex_pseudo_by_case(diff, arm)),
        _write_tex(out_dir, prefix, "dismissals", version,
                   latex_dismissals(bundle.get("about_case", pd.DataFrame()),
                                    bundle.get("rate_case", pd.DataFrame()),
                                    targets, arm)),
    ]


def export_across_arms_latex(bundles: Dict[str, Dict[str, pd.DataFrame]],
                             out_dir: Path, diff_version: str,
                             grounding_version: Optional[str] = None,
                             gate_ok: bool = True) -> List[str]:
    """Write the 2 x 2 tables: pseudo-differentiation by arm (tag
    `ic_differentiation_arms`), and the nesting by arm (tag
    `ic_grounding_arms`) when the bundles hold question 2."""
    suffix = "" if gate_ok else "_stale"
    written: List[str] = []
    diff = differentiation_by_arm(bundles)
    if not diff.empty:
        written.append(_write_tex(
            out_dir, f"ic_differentiation_arms{suffix}", "pseudo", diff_version,
            latex_pseudo_arms(diff, dismissals_by_arm(bundles))))
    over = stack(bundles, "nest_overview")
    if not over.empty:
        written.append(_write_tex(
            out_dir, f"ic_grounding_arms{suffix}", "nesting",
            grounding_version or diff_version, latex_nesting_arms(over)))
    return written
