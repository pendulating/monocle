"""The aesthetic hierarchy over the URBAN CUES, across the cases.

`_words_across` counts words and phrases. It answers "does the model use the
same vocabulary in every case", and the answer is a weak yes: the top of the
word list is mostly what the case is ABOUT. A word count cannot separate the
subject from the judgment.

This module counts CUES instead. The extraction stage
(`dagspaces/pairwisevqa/stages/trace_extract.py`) turns each trace into typed
spans, and a `visual_evidence` span carries 3 things a word never can:

| Field | What it gives |
|-------|---------------|
| `image` | WHICH photograph the cue sits on |
| `valence` | good, bad, or neutral, in the model's own reading |
| `category` | 1 of 11 urban attribute types |

The price of a cue
------------------
Because a cue names its image, the question turns directional. A cue on image A
wins when the label says A. Thus each cue carries a PRICE: how often the
photograph that holds it wins.

That price is the aesthetic hierarchy. `chain-link fence` and `construction
site` cost the model everywhere. `well-maintained` and `very clean` pay
everywhere. The cross-case question is then sharp: does a cue keep its price
when the question changes from a school to a subway entrance?

The lift, not the rate
----------------------
A raw win rate cannot be compared across cases. The base rate differs — a cue
sits on the winning image 59% of the time in the schools case and 70% in the
libraries case, because the model names more cues for the photograph it
prefers. Every number here is a LIFT over the base rate of its own case.

The units
---------
| `unit` | 1 key | Shared by every case? |
|--------|-------|-----------------------|
| `category` | `upkeep` | yes, 11 keys |
| `category_valence` | `upkeep \\| good` | yes, up to 33 keys |
| `cue` | `chain-link fence` | the head only |
| `cue_valence` | `chain-link fence \\| bad` | the head only |

**`category_valence` is the unit that answers the question.** `category` alone
pools `good greenery` with `bad greenery`, and those 2 carry opposite prices,
so the pooled key reads near zero and the hierarchy disappears. Measured on the
2026-08-13 corpus, the cross-case Spearman of the lift is 0.35 for `category`
and 0.92 for `category_valence`.

Warning: this is an association, not a cause
--------------------------------------------
The model narrates while it decides, thus a cue can follow the judgment rather
than drive it. Read a price as "the cues that go with a win".

Warning: the corpus is smaller than the battery
-----------------------------------------------
The extraction corpus covers the cases and the raters that the extraction stage
has run on, and that is not always all 7 cases and both raters. `coverage()`
reports what is present, and every figure names it. Do not read a missing case
as a null result.

See `research/docs/aesthetic-hierarchy.md` and
`research/docs/langextract-trace-extraction.md`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import re

import numpy as np
import pandas as pd

from research.notebooks.cvpr import _extractions as X
from research.notebooks.cvpr import _style as S
from research.notebooks.cvpr import _words_across as WA

__version__ = "1.0.0"

# A cue rate is cues for each 100 traces.
RATE_PER = 100

# The label sets of the directional test. They mirror `_extractions`.
WIN_FOR_A = ("More", "MuchMore")
WIN_FOR_B = ("Less", "MuchLess")
UNDECIDED = ("Same", "NotSure")

CUE_CLASS = "visual_evidence"

UNITS = ("category_valence", "category", "cue", "cue_valence",
         "cue_norm", "cue_norm_valence")

#: The cases that pair photographs of a NAMED facility. Road quality and
#: street photography pair random street images, thus they share almost no
#: cue word with the rest: 17 keys against 52, and 13 of the 17 name a car.
#: A word-level ladder over every case is therefore not possible, and this
#: tuple names the set over which it IS.
FACILITY_CASES = ("libraries", "parks_plazas", "restaurants", "schools",
                  "subway_safety")

#: Words that a normalised cue key drops. An article or a quantifier does not
#: change WHICH thing the model named.
_ARTICLES = frozenset((
    "a", "an", "the", "some", "several", "many", "few", "this", "that",
    "these", "those", "its", "their", "his", "her",
))

#: Degree adverbs. `very clean` and `clean` name the same thing at 2
#: strengths, and the win lift measures the strength itself.
_DEGREE = frozenset((
    "very", "quite", "somewhat", "extremely", "fairly", "relatively",
    "slightly", "highly", "really", "rather", "mostly", "generally",
    "overall", "pretty",
))

#: Words that end in `s` and are not plural.
_KEEP_S = frozenset((
    "glass", "grass", "class", "moss", "access", "process", "bus", "status",
    "premises", "series", "species", "news", "focus", "lens", "canvas",
))

_IRREGULAR = {
    "people": "person", "children": "child", "leaves": "leaf",
    "shelves": "shelf", "benches": "bench", "bushes": "bush",
    "brushes": "brush", "canopies": "canopy",
}

#: `sidewalk is clean` becomes `clean sidewalk`.
_PREDICATE = re.compile(
    r"^(.+?)\s+(?:is|are|was|were|looks?|appears?|seems?)\s+(.+)$")

_NON_WORD = re.compile(r"[^\w\s]")

# The order the paper uses for a valence. It runs bad to good, so a legend and
# a colour ramp read the same way.
VALENCE_ORDER = ("bad", "neutral", "good")

VALENCE_COLOUR = {
    "bad": S.PAL["coral"],
    "neutral": S.PAL["warmgrey"],
    "good": S.PAL["teal"],
}


# --------------------------------------------------------------------------
# Load and describe
# --------------------------------------------------------------------------

def load(require_canonical: bool = True, root: Optional[str | Path] = None
         ) -> pd.DataFrame:
    """Read the extraction corpus. `_extractions.load` holds the gate.

    Version 3 (2026-09-28) names the cue class `dimension` and gives it a
    free `name` in place of the `category` enum. This module reads the cue
    class as `visual_evidence` with a `category` in about 1,400 lines, thus
    the bridge below maps the version 3 rows onto those names and records
    the move in `cue_class_source`. A `category` under version 3 is the
    model's own name for the attribute, lowered, not a value of the enum,
    and a key needs `min_count` rows as before.
    """
    df = X.load(root=root, require_canonical=require_canonical)
    if "extraction_class" in df.columns and (df.extraction_class == "dimension").any():
        df = df.copy()
        df["cue_class_source"] = df.extraction_class.where(df.extraction_class == "dimension", None)
        df.loc[df.extraction_class == "dimension", "extraction_class"] = CUE_CLASS
        if "category" not in df.columns and "name" in df.columns:
            df["category"] = df["name"]
        elif "name" in df.columns:
            df["category"] = df["category"].fillna(df["name"])
    return df


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """What the corpus holds: cases, raters, traces, and cues.

    Read this before a figure. A case that the extraction stage has not run on
    is absent, and an absent case is not a null result.
    """
    traces = (df.groupby(["case", "judge_model"])["pair_id"].nunique()
                .rename("traces").reset_index())
    cues = (df[df.extraction_class == CUE_CLASS]
            .groupby(["case", "judge_model"]).size().rename("cues").reset_index())
    out = traces.merge(cues, on=["case", "judge_model"], how="left")
    out["cues_per_trace"] = (out["cues"] / out["traces"]).round(1)
    out["case_label"] = out["case"].map(WA.CASE_LABEL).fillna(out["case"])
    return out.sort_values("case").reset_index(drop=True)


def missing_cases(df: pd.DataFrame) -> List[str]:
    """The battery cases that the corpus does not hold."""
    have = set(df["case"].unique())
    return [c for c in WA.CASE_ORDER if c not in have]


# --------------------------------------------------------------------------
# The unit
# --------------------------------------------------------------------------

def _singular(word: str) -> str:
    """Make 1 word singular with rules only. No stemmer, no word list."""
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word in _KEEP_S or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes", "zes")):
        return word[:-2]
    if word.endswith(("ss", "us", "is")):
        return word
    return word[:-1] if word.endswith("s") else word


def normalise_cue(text) -> Optional[str]:
    """Reduce a cue span to a key that joins its own spelling variants.

    The 5 rules run in order:

    1. A slash becomes `or` and an ampersand becomes `and`, thus a
       coordination stays a coordination: `debris/dirt` gives
       `debris or dirt`, never the phrase `debris dirt`. Other punctuation
       and a hyphen become a space, thus `well-maintained` joins
       `well maintained`.
    2. `X is Y` becomes `Y X`. `sidewalk is clean` joins `clean sidewalk`.
    3. An article or a quantifier goes. `the tree` joins `tree`.
    4. A degree adverb goes. `very clean` joins `clean`.
    5. Each word becomes singular. `trees` joins `tree`.

    **Warning: this joins spelling variants ONLY.** `clean`, `well maintained`
    and `tidy` stay 3 keys, because they are 3 words the model chose. That is
    the point of the unit: the key must carry the model's own word.

    The rules are deterministic and they read no corpus, thus a key does not
    move when the corpus grows.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    # 5.9% of the spans hold a slash, and it always coordinates: the model
    # wrote `debris/dirt` to name 2 things. Dropping the slash would give
    # `debris dirt`, which reads as 1 thing that `dirt` heads.
    key = str(text).replace("/", " or ").replace("&", " and ")
    key = key.replace("-", " ").replace("_", " ")
    key = " ".join(_NON_WORD.sub(" ", key).lower().split())
    if not key:
        return None
    match = _PREDICATE.match(key)
    if match:
        key = f"{match.group(2)} {match.group(1)}"
    words = [w for w in key.split() if w not in _ARTICLES and w not in _DEGREE]
    if not words:
        words = key.split()
    return " ".join(_singular(w) for w in words) or None


def unit_keys(df: pd.DataFrame, unit: str = "category_valence") -> pd.Series:
    """Build the key column of the chosen unit.

    A row with a missing part gives `None`, and every step drops it. A cue with
    no valence is a defect of the extractor, not a neutral cue.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    if unit == "category":
        keys = df["category"].astype("string")
    elif unit == "category_valence":
        keys = (df["category"].astype("string") + " | "
                + df["valence"].astype("string"))
    else:
        # `cue` keeps the span as the model wrote it. `cue_norm` joins the
        # spelling variants of 1 word. See `normalise_cue`.
        reduce = (normalise_cue if unit.startswith("cue_norm") else X._span_key)
        text = pd.Series([reduce(t) for t in df["extraction_text"]],
                         index=df.index, dtype="string")
        keys = (text if unit in ("cue", "cue_norm")
                else text + " | " + df["valence"].astype("string"))
    return keys.where(~keys.str.contains("<NA>", na=True))


def cue_frame(df: pd.DataFrame, unit: str = "category_valence",
              classes: Sequence[str] = (CUE_CLASS,)) -> pd.DataFrame:
    """The rows a cue figure reads: 1 class, 1 named image, 1 usable key."""
    part = df[df["extraction_class"].isin(list(classes))].copy()
    part["key"] = unit_keys(part, unit)
    part = part[part["key"].notna()]
    return part


# --------------------------------------------------------------------------
# How often the model names a cue
# --------------------------------------------------------------------------

def rates_frame(df: pd.DataFrame, unit: str = "category_valence",
                classes: Sequence[str] = (CUE_CLASS,)) -> pd.DataFrame:
    """Count each key by case and rater, as cues for each 100 traces.

    The columns match `_words_across.rates_frame`, thus every figure of that
    module reads this table without a change. Only the denominator differs: a
    word rate divides by the words of the corpus, and a cue rate divides by its
    TRACES. A cue is a thing the model named, thus "per 100 traces" is the
    number a reader can hold.
    """
    part = cue_frame(df, unit, classes)
    frames = []
    for (case, model), grp in part.groupby(["case", "judge_model"]):
        traces = int(grp["pair_id"].nunique()) or 1
        counts = grp["key"].value_counts()
        frames.append(pd.DataFrame({
            "case": case,
            "model": WA.short_model(model),
            "word": counts.index.astype(str),
            "count": counts.to_numpy(),
            "corpus_words": traces,
            "rate": counts.to_numpy() * RATE_PER / traces,
        }))
    if not frames:
        return pd.DataFrame(columns=["case", "model", "word", "count",
                                     "corpus_words", "rate"])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# The price of a cue
# --------------------------------------------------------------------------

def win_lift(
    df: pd.DataFrame,
    case: str,
    unit: str = "category_valence",
    classes: Sequence[str] = (CUE_CLASS,),
    min_count: int = 25,
    prior_strength: float = 25.0,
) -> pd.DataFrame:
    """How often the photograph that holds each key won, in 1 case.

    This generalises `_extractions.win_rates` from the cue TEXT to any unit,
    and it keeps the same 3 rules that make the number honest:

    1. **1 vote for each comparison.** A trace that names `trash can` 3 times
       for image A still describes 1 comparison.
    2. **A key on BOTH images is dropped.** A fence in A and in B separates
       nothing, and counting it twice adds 1 win and 1 loss.
    3. **The rate is shrunk toward the base rate.** A key seen 4 times would
       otherwise read 100% and take the top of the scale.

    Returns:
        A row for each key, with `n`, `wins`, `win_rate`, `shrunk_rate`, the
        Wilson interval, `base_rate`, and `lift`.

    `lift` is `shrunk_rate - base_rate`. Use it, never the raw rate: the base
    rate runs from 0.59 to 0.70 over the cases, because the model names more
    cues for the photograph it prefers, and a raw rate would carry that
    difference into every comparison between cases.
    """
    return keyed_lift(cue_frame(df[df["case"] == case], unit, classes),
                      case, min_count, prior_strength)


def keyed_lift(part: pd.DataFrame, case: str, min_count: int = 25,
               prior_strength: float = 25.0) -> pd.DataFrame:
    """The directional test itself, over a frame that already holds `key`.

    `win_lift` reads the cue class. `_trace_channels` reads the comparison,
    the inference, and the image-artifact classes. All of them need the SAME
    3 rules, thus all of them call this function.

    Args:
        part: Rows with `case`, `pair_id`, `image`, `presented_label`, `key`.
        case: The case to score. A base rate holds for 1 case only.
    """
    if part.empty or "image" not in part.columns:
        return pd.DataFrame()
    part = part[(part["case"] == case) & part["image"].isin(["A", "B"])]
    if part.empty:
        return pd.DataFrame()

    votes = part.drop_duplicates(subset=["pair_id", "key", "image"])
    sides = votes.groupby(["pair_id", "key"])["image"].transform("nunique")
    votes = votes[sides == 1]
    if votes.empty:
        return pd.DataFrame()

    label = votes["presented_label"]
    decided = ~label.isin(UNDECIDED)
    won = ((votes["image"].eq("A") & label.isin(WIN_FOR_A))
           | (votes["image"].eq("B") & label.isin(WIN_FOR_B)))

    flat = pd.DataFrame({"key": votes["key"].to_numpy(),
                         "decided": decided.to_numpy(),
                         "won": (won & decided).to_numpy()})
    grouped = flat.groupby("key")
    table = pd.DataFrame({
        "n_mentions": grouped.size(),
        "n": grouped["decided"].sum(),
        "wins": grouped["won"].sum(),
    })
    table["undecided_share"] = (1 - table["n"] / table["n_mentions"]).round(3)
    table = table[table["n"] >= min_count]
    if table.empty:
        return pd.DataFrame()

    base = float(table["wins"].sum()) / float(table["n"].sum())
    table["win_rate"] = table["wins"] / table["n"]
    table["shrunk_rate"] = ((table["wins"] + prior_strength * base)
                            / (table["n"] + prior_strength))
    low, high = zip(*(X._wilson(int(w), int(n))
                      for w, n in zip(table["wins"], table["n"])))
    table["ci_low"], table["ci_high"] = low, high
    table["base_rate"] = base
    table["lift"] = table["shrunk_rate"] - base
    table["case"] = case
    return table.reset_index().sort_values("lift", ascending=False)


def lift_tables(df: pd.DataFrame, unit: str = "category_valence",
                classes: Sequence[str] = (CUE_CLASS,), min_count: int = 25,
                prior_strength: float = 25.0) -> pd.DataFrame:
    """`win_lift` for every case in the corpus, in 1 long table."""
    frames = [win_lift(df, c, unit, classes, min_count, prior_strength)
              for c in sorted(df["case"].unique())]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def case_effects(tables: pd.DataFrame) -> pd.DataFrame:
    """The effect of each cue in each case, on a scale that pools.

    A raw win rate cannot be compared between cases: the base rate runs from
    0.47 to 0.60, thus the same cue reads differently for a reason that has
    nothing to do with the cue. The log odds ratio measures a cue against the
    base rate of its OWN case:

        theta = log( odds(win rate) / odds(base rate) )

    `theta` is 0 when the cue goes with the winner exactly as often as a cue
    picked at random in that case. It is positive when the photograph that
    holds the cue wins more often.

    A cell of 0 wins or 0 losses takes the Haldane-Anscombe 0.5 correction,
    thus `theta` stays finite. `pays` and `costs` hold the 95% interval
    ENTIRELY on one side of 0.
    """
    if tables.empty:
        return pd.DataFrame()
    out = tables[["key", "case", "n", "wins", "base_rate", "lift"]].copy()
    out = out[out["n"] > 0]
    total = out["n"].to_numpy(dtype=float)
    w = np.clip(out["wins"].to_numpy(dtype=float), 0.5, total - 0.5)
    base = out["base_rate"].to_numpy(dtype=float)
    out["theta"] = np.log(w / (total - w)) - np.log(base / (1.0 - base))
    out["se"] = np.sqrt(1.0 / w + 1.0 / (total - w))
    out["pays"] = out["theta"] - 1.96 * out["se"] > 0
    out["costs"] = out["theta"] + 1.96 * out["se"] < 0
    return out


def cue_meta(tables: pd.DataFrame, min_cases: int = 3) -> pd.DataFrame:
    """Pool each cue over the cases, and test whether it travels.

    | Column | Meaning |
    |--------|---------|
    | `theta` | the inverse-variance pooled log odds ratio |
    | `theta_lo`, `theta_hi` | the 95% interval of `theta` |
    | `p_effect` | is the pooled effect different from 0? |
    | `q`, `p_hetero` | Cochran's Q: do the cases disagree beyond noise? |
    | `i2` | the share of the variance that is disagreement |
    | `n_pays`, `n_costs` | cases whose OWN interval clears 0 |
    | `lift_mean` | the same effect on the familiar win-share scale |

    **Warning: do not select on `p_hetero` alone.** The median `i2` runs 0.66
    to 0.87, thus almost every cue is heterogeneous at this sample size, and
    the test separates nothing. `outliers` selects on a SIGN REVERSAL, which
    needs each side to clear 0 on its own.
    """
    from scipy import stats

    eff = case_effects(tables)
    if eff.empty:
        return pd.DataFrame()
    eff = eff.copy()
    eff["weight"] = 1.0 / eff["se"] ** 2

    rows: List[Dict[str, Any]] = []
    for key, g in eff.groupby("key"):
        k = len(g)
        if k < min_cases:
            continue
        wt = g["weight"].to_numpy()
        th = g["theta"].to_numpy()
        pooled = float((wt * th).sum() / wt.sum())
        se = float(np.sqrt(1.0 / wt.sum()))
        q = float((wt * (th - pooled) ** 2).sum())
        dof = k - 1
        rows.append({
            "key": key,
            "n_cases": k,
            "n": int(g["n"].sum()),
            "theta": pooled,
            "theta_lo": pooled - 1.96 * se,
            "theta_hi": pooled + 1.96 * se,
            "p_effect": float(2 * stats.norm.sf(abs(pooled / se))),
            "q": q,
            "p_hetero": float(stats.chi2.sf(q, dof)) if dof > 0 else np.nan,
            "i2": max(0.0, (q - dof) / q) if q > 0 else 0.0,
            "theta_min": float(th.min()),
            "theta_max": float(th.max()),
            "theta_range": float(th.max() - th.min()),
            "lift_mean": float(g["lift"].mean()),
            "lift_min": float(g["lift"].min()),
            "lift_max": float(g["lift"].max()),
            "n_pays": int(g["pays"].sum()),
            "n_costs": int(g["costs"].sum()),
            "cases_pay": ";".join(sorted(g.loc[g["pays"], "case"])),
            "cases_cost": ";".join(sorted(g.loc[g["costs"], "case"])),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("theta", ascending=False).reset_index(drop=True)


def cue_valence_mode(df: pd.DataFrame, unit: str = "cue_norm",
                     classes: Sequence[str] = (CUE_CLASS,)) -> pd.DataFrame:
    """The valence the extractor gave each key most often, and how surely.

    The extractor labels a span `good`, `neutral` or `bad`. That label is an
    opinion ABOUT the cue, and the effect is what the cue DID. The 2 can
    disagree, and a disagreement is the most readable kind of outlier.
    """
    part = cue_frame(df, unit, classes)
    part = part[part["valence"].isin(VALENCE_ORDER)]
    if part.empty:
        return pd.DataFrame(columns=["key", "valence", "valence_purity"])
    counts = part.groupby(["key", "valence"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("key")["n"].transform("sum")
    best = counts.sort_values("n").drop_duplicates("key", keep="last")
    return (best[["key", "valence", "share"]]
            .rename(columns={"share": "valence_purity"})
            .reset_index(drop=True))


#: How big a pooled log odds ratio must be to count as a real effect. 1.0 is
#: an odds ratio of 2.7 against the base rate of the case.
STRONG_THETA = 1.0


def outliers(df: pd.DataFrame, unit: str = "cue_norm", min_count: int = 25,
             min_cases: int = 3, alpha: float = 0.01) -> pd.DataFrame:
    r"""Isolate the cues a reader should look at, and say why each one earns it.

    3 kinds of cue earn a place. They are NOT ranked against each other: the
    `flag` column names the kind, and `rank_in_flag` sorts within it.

    | `flag` | The test | What it means |
    |--------|----------|---------------|
    | `reverses` | 1 case pays and another costs, each clearing 0 alone | The same word carries opposite prices. |
    | `neutral_but_decisive` | label is `neutral`, \|theta\| > 1.0, no reversal | The model calls it neutral and then judges by it. |
    | `contradicts_valence` | the label fights the pooled effect, no reversal | The extractor called it good, and it lost. |
    | `strong_and_consistent` | label is `good` or `bad`, \|theta\| > 1.0, no reversal | The cue carries the standard in every case. |

    **`contradicts_valence` is empty on the 2026-08-31 corpus, and that is a
    result.** Over the 53 cues that carry a `good` or a `bad` label and do not
    reverse, the sign of the effect agrees with the label 53 times. The
    opinion the extractor wrote about a cue matches what the cue then did.

    `reverses` is the selective test. It needs BOTH sides to clear 0 on their
    own, thus a difference of size alone does not qualify and a single noisy
    case cannot make a row.

    **Warning: do not read `theta` on a row that reverses.** The pooled value
    weights each case by its precision, thus the largest case drives it, and a
    single number cannot describe a cue that pays in 1 case and costs in
    another. Read `theta_min`, `theta_max`, `cases_pay` and `cases_cost`.

    **Warning: `min_cases` is 3, not 2.** Over 2 cases a reversal rests on 1
    comparison, and Cochran's Q carries 1 degree of freedom.

    `minority_case` names the case that stands alone against the others. It is
    empty when the split is not 1 against the rest.
    """
    tables = lift_tables(df, unit=unit, min_count=min_count)
    if tables.empty:
        return pd.DataFrame()
    meta = cue_meta(tables, min_cases=min_cases)
    if meta.empty:
        return meta
    meta = meta.merge(cue_valence_mode(df, unit), on="key", how="left")

    reverses = (meta["n_pays"] > 0) & (meta["n_costs"] > 0)
    big = ((meta["theta"].abs() > STRONG_THETA)
           & (meta["p_effect"] < alpha) & ~reverses)
    labelled = meta["valence"].isin(["good", "bad"])
    strong = big & labelled
    # The extractor called it neutral and the judgments then turned on it.
    # These are the cues the model uses without calling them evaluative.
    neutral_decisive = big & (meta["valence"] == "neutral")
    # The label says good and the cue lost, or the label says bad and it won.
    # A cue that REVERSES cannot earn this flag: `theta` pools by inverse
    # variance, thus the case with the most data drives it, and a pooled sign
    # means nothing when the cases point both ways. `dark` is the example: it
    # pools to +0.47 because road quality is large, while subway safety sits
    # at -4.68.
    contra = ((((meta["valence"] == "good") & (meta["theta_hi"] < 0))
               | ((meta["valence"] == "bad") & (meta["theta_lo"] > 0)))
              & ~reverses)

    def _minority(row) -> str:
        pay = [c for c in row["cases_pay"].split(";") if c]
        cost = [c for c in row["cases_cost"].split(";") if c]
        if len(pay) == 1 and len(cost) > 1:
            return pay[0]
        if len(cost) == 1 and len(pay) > 1:
            return cost[0]
        return ""

    meta["reverses"] = reverses
    meta["contradicts_valence"] = contra
    meta["neutral_but_decisive"] = neutral_decisive
    meta["strong_and_consistent"] = strong
    meta["minority_case"] = meta.apply(_minority, axis=1)
    meta["flag"] = [
        "+".join(name for name, on in
                 (("reverses", a), ("contradicts_valence", b),
                  ("neutral_but_decisive", c), ("strong_and_consistent", d))
                 if on)
        for a, b, c, d in zip(reverses, contra, neutral_decisive, strong)
    ]
    meta = meta[meta["flag"] != ""].copy()
    if meta.empty:
        return meta
    # A reversal sorts by how far apart the 2 sides sit. The other kinds sort
    # by the size of the effect they claim.
    meta["sort_value"] = np.where(meta["reverses"], meta["theta_range"],
                                  meta["theta"].abs())
    # The file must open on the rows that carry the most: a cue that reverses
    # AND fights its own label first, a cue that only agrees with itself last.
    priority = {"reverses+contradicts_valence": 0, "reverses": 1,
                "contradicts_valence": 2, "neutral_but_decisive": 3,
                "strong_and_consistent": 4}
    meta["flag_rank"] = meta["flag"].map(priority).fillna(5).astype(int)
    meta = meta.sort_values(["flag_rank", "sort_value"],
                            ascending=[True, False])
    meta["rank_in_flag"] = meta.groupby("flag").cumcount() + 1
    return meta.drop(columns=["flag_rank"]).reset_index(drop=True)


OUTLIER_COLUMNS = [
    "flag", "rank_in_flag", "key", "valence", "valence_purity",
    "n_cases", "n", "theta", "theta_lo", "theta_hi", "p_effect",
    "theta_min", "theta_max", "theta_range", "i2", "p_hetero",
    "lift_mean", "lift_min", "lift_max",
    "n_pays", "n_costs", "cases_pay", "cases_cost", "minority_case",
]


def lift_matrix(tables: pd.DataFrame, complete_only: bool = True
                ) -> pd.DataFrame:
    """The key-by-case matrix of the lift.

    Args:
        complete_only: Keep a key only when EVERY case scored it. A key that 1
            case never names cannot say whether the price persists, and a
            missing cell would otherwise become a 0, which reads as "no lift"
            rather than "not measured".
    """
    if tables.empty:
        return pd.DataFrame()
    wide = tables.pivot(index="key", columns="case", values="lift")
    wide.columns.name = None
    if complete_only:
        wide = wide.dropna()
    return wide[WA._ordered_columns(wide.columns)]


def lift_summary(matrix: pd.DataFrame, tables: pd.DataFrame) -> pd.DataFrame:
    """Give each key its place in the hierarchy, and how much it moves.

    | Column | Meaning |
    |--------|---------|
    | `mean_lift` | the mean lift over the cases: the height of the rung |
    | `sd_lift` | how much the lift moves between cases |
    | `spread` | the widest gap, max minus min |
    | `min_lift`, `max_lift` | the ends of that gap |
    | `same_sign` | every case put the key on the same side of its base rate |
    | `n` | the decided comparisons behind it, over the cases |
    """
    if matrix.empty:
        return pd.DataFrame()
    values = matrix.to_numpy(dtype=float)
    totals = tables.groupby("key")["n"].sum()
    out = pd.DataFrame({
        "key": matrix.index,
        "mean_lift": values.mean(axis=1),
        "sd_lift": values.std(axis=1, ddof=0),
        "spread": values.max(axis=1) - values.min(axis=1),
        "min_lift": values.min(axis=1),
        "max_lift": values.max(axis=1),
        "same_sign": (values > 0).all(axis=1) | (values < 0).all(axis=1),
        "n_groups": len(matrix.columns),
    })
    out["n"] = out["key"].map(totals).fillna(0).astype(int)
    out["valence"] = [_valence_of(k) for k in out["key"]]
    return out.sort_values("mean_lift", ascending=False).reset_index(drop=True)


def _valence_of(key: str) -> Optional[str]:
    """Read the valence out of a key, when the unit carries one."""
    tail = str(key).split(" | ")[-1].strip()
    return tail if tail in VALENCE_ORDER else None


def agreement(matrix: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho of the lift between every case pair.

    A high rho says the 2 cases put the cues in the same order, which is what
    "1 aesthetic hierarchy" means.
    """
    return WA.agreement(matrix)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _case_colour(case: str) -> str:
    return WA.CASE_COLOUR.get(case, S.PAL["warmgrey"])


def _pretty(key: str) -> str:
    """`upkeep | good` reads better as `good upkeep` on an axis."""
    parts = [p.strip() for p in str(key).split(" | ")]
    if len(parts) == 2 and parts[1] in VALENCE_ORDER:
        return f"{parts[1]} {parts[0]}"
    return str(key)


def plot_ladder(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    max_keys: int = 30,
    figsize: Tuple[float, float] = (6.4, 5.6),
):
    """The hierarchy itself: each key at its price, with 1 dot for each case.

    The rows run from the cue that costs most to the cue that pays most. A
    short row says every case gave the key the same price. A long row says the
    price moved with the question.

    The vertical line is 0, which is the base rate of each case. A cue to the
    right of it goes with the winning photograph more often than a cue picked
    at random.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if matrix.empty:
        raise ValueError("no keys to draw")
    if len(summary) > max_keys:
        # Take the 2 ENDS, never the head. `lift_summary` sorts by the mean
        # lift, thus a head would drop the cues that cost most, which are the
        # rows a reader looks for first.
        half = max_keys // 2
        keep = pd.concat([summary.head(max_keys - half), summary.tail(half)])
        keep = keep.drop_duplicates("key")
    else:
        keep = summary
    keys = list(keep["key"])[::-1]           # the best price at the top
    cases = list(matrix.columns)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.EDGE, linewidth=0.8, alpha=0.7, zorder=1)
    for y, key in enumerate(keys):
        row = matrix.loc[key]
        ax.plot([row.min(), row.max()], [y, y], color=S.PAL["warmgrey"],
                linewidth=0.9, alpha=0.55, zorder=2, solid_capstyle="round")
        for case in cases:
            ax.scatter(row[case], y, s=22, color=_case_colour(case),
                       edgecolors=S.EDGE, linewidths=0.3, alpha=0.95, zorder=3)

    ax.set_yticks(range(len(keys)), [_pretty(k) for k in keys])
    ax.set_ylim(-0.8, len(keys) - 0.2)
    ax.set_xlabel("lift over the base rate of the case\n"
                  "(share of the wins the photograph holding the cue takes)")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(
        handles=[Line2D([], [], marker="o", linestyle="none", markersize=4.0,
                        markeredgecolor=S.EDGE, markeredgewidth=0.3,
                        markerfacecolor=_case_colour(c),
                        label=WA.case_label(c)) for c in cases],
        loc="lower right", frameon=True, framealpha=0.9, fontsize=6.5,
    )
    fig.tight_layout()
    return fig


def plot_case_ladder(
    tables: pd.DataFrame,
    n_each: int = 9,
    n_cols: int = 4,
    panel: Tuple[float, float] = (3.5, 3.4),
):
    """What each case values, in the words the model wrote.

    1 panel for each case. Each panel holds the `n_each` cues that go with the
    winning photograph most, and the `n_each` that go with it least. A key
    keeps the model's own word, thus a panel reads as a vocabulary and not as
    a taxonomy.

    A cross-case figure cannot show this. A word must recur in every case to
    take a row of `plot_ladder`, and only 17 words do. Each case on its own
    carries 148 to 323, thus the diversity lives INSIDE a case.

    Args:
        tables: `lift_tables(df, unit="cue_norm")`.
        n_each: rows to draw at each end.
    """
    import matplotlib.pyplot as plt

    if tables.empty:
        raise ValueError("no rows to draw")
    cases = [c for c in WA.CASE_ORDER if c in set(tables["case"])]
    n_rows = -(-len(cases) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(panel[0] * n_cols, panel[1] * n_rows))
    flat = [a for row in axes for a in row]

    for ax, case in zip(flat, cases):
        g = tables[tables["case"] == case].sort_values("lift")
        # The 2 ends of the ladder. A middle row sits near the base rate and
        # it carries no reading, thus the panel leaves it out.
        take = pd.concat([g.head(n_each), g.tail(n_each)])
        take = take.drop_duplicates("key").sort_values("lift")
        y = range(len(take))
        colours = [S.PAL["coral"] if v < 0 else S.PAL["teal"]
                   for v in take["lift"]]
        ax.axvline(0.0, color=S.EDGE, linewidth=0.8, alpha=0.7, zorder=1)
        ax.hlines(list(y), 0, take["lift"], color=colours, linewidth=1.4,
                  alpha=0.75, zorder=2)
        ax.scatter(take["lift"], list(y), s=20, c=colours,
                   edgecolors=S.EDGE, linewidths=0.3, zorder=3)
        ax.set_yticks(list(y), list(take["key"]), fontsize=6.5)
        ax.set_ylim(-0.8, len(take) - 0.2)
        ax.set_xlabel(WA.case_label(case), fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.12)
    for ax in flat[len(cases):]:
        ax.axis("off")

    fig.supxlabel("lift over the base rate of the case\n"
                  "(share of the wins the photograph holding the cue takes)",
                  fontsize=8)
    fig.tight_layout()
    return fig


def head_noun(key: str) -> str:
    """The last word of a key. An English noun phrase puts its head last.

    **Warning: this is a rule, not a parser.** `power line` and `elevated
    train line` both give `line`, and they are not the same thing.
    `modifier_variants` therefore takes a head that is 1 word on the ladder,
    where the head IS the whole key and the grouping cannot go wrong.
    """
    return str(key).split()[-1]


#: A word that turns a phrase into an attachment, not a modifier. `brick
#: building with large window` ends in `window`, but `brick building with
#: large` does not describe a window the way `large glass` does.
_PHRASAL = frozenset((
    "with", "of", "on", "in", "at", "and", "or", "along", "near", "next",
    "from", "by", "under", "behind", "around", "lining",
))


def modifier_variants(tables: pd.DataFrame, head: str, min_cases: int = 1,
                      drop_phrasal: bool = True,
                      max_words: int = 4) -> pd.DataFrame:
    """Every key that ends in `head`, with the lift of each case.

    The `modifier` column holds the key with the head word removed, thus a
    row reads as the word the model added: `large glass`, `small`, `older`.
    The bare head takes the label `(alone)`.

    Args:
        drop_phrasal: leave out a key whose modifier holds a preposition or a
            conjunction. Such a key names another thing that HOLDS the head,
            and it does not describe the head.
        max_words: the longest modifier to keep, in words.
    """
    part = tables[tables["key"].map(head_noun) == head].copy()
    if part.empty:
        return part
    part["modifier"] = part["key"].map(
        lambda k: " ".join(str(k).split()[:-1]) or "(alone)")
    counts = part.groupby("key")["case"].nunique()
    keep = set(counts[counts >= min_cases].index)
    if drop_phrasal:
        words = part.drop_duplicates("key").set_index("key")["modifier"]
        bad = {k for k, m in words.items()
               if m != "(alone)" and (set(m.split()) & _PHRASAL
                                      or len(m.split()) > max_words)}
        keep -= bad
    # The bare head always keeps its row: it IS the row the panel opens.
    keep |= {head}
    return part[part["key"].isin(keep)]


def spread_heads(summary: pd.DataFrame, tables: pd.DataFrame,
                 n_heads: int = 6, min_variants: int = 5) -> List[str]:
    """The 1-word ladder keys whose price moves most between the cases.

    A 1-word key is its own head, thus its variants group without a parser.
    A key of 2 words or more is left out: `fire escape` would give `escape`,
    and `power line` would join `elevated train line`.
    """
    if summary.empty:
        return []
    one = summary[summary["key"].str.split().str.len() == 1]
    order = one.sort_values("spread", ascending=False)["key"]
    out: List[str] = []
    for key in order:
        if len(modifier_variants(tables, key)["key"].unique()) >= min_variants:
            out.append(key)
        if len(out) >= n_heads:
            break
    return out


def plot_ladder_modifiers(
    tables: pd.DataFrame,
    summary: pd.DataFrame,
    heads: Optional[Sequence[str]] = None,
    n_heads: int = 9,
    n_label: int = 6,
    min_cases: int = 1,
    figsize: Tuple[float, float] = (7.6, 8.8),
):
    """The widest ladder rows, with the modifier written along each spread.

    A row keeps the shape of `plot_ladder`: 1 grey line for the spread of the
    bare word, and 1 filled dot for each case. It then GAINS the phrases that
    end in that word. Each of those points sits at its own price on the same
    row, takes the colour of its case, and carries the word the model added.

    A reader thus sees 2 things at once: how far the bare word moved between
    the cases, and which descriptor sits at each place along that line. A
    label outside the grey line names a price the ladder alone cannot show.

    Args:
        n_label: labelled variant points for each row. The 2 ends win.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if tables.empty:
        raise ValueError("no rows to draw")
    heads = list(heads) if heads else spread_heads(summary, tables, n_heads)
    if not heads:
        raise ValueError("no 1-word head carries enough variants")
    cases = [c for c in WA.CASE_ORDER if c in set(tables["case"])]

    # The best price at the top, by the mean of the bare word.
    order = sorted(
        heads,
        key=lambda h: tables[tables["key"] == h]["lift"].mean(),
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.EDGE, linewidth=0.8, alpha=0.7, zorder=1)

    # Pass 1 draws every point and remembers where a label must go. The label
    # itself waits for pass 2, which needs the final x limits to measure how
    # wide a word is in DATA units.
    pending: List[Tuple[int, float, str]] = []
    for y, head in enumerate(order):
        part = modifier_variants(tables, head, min_cases)
        bare = part[part["key"] == head]
        var = part[part["key"] != head]

        if not bare.empty:
            ax.plot([bare["lift"].min(), bare["lift"].max()], [y, y],
                    color=S.PAL["warmgrey"], linewidth=1.6, alpha=0.65,
                    zorder=2, solid_capstyle="round")
        # Every variant point, unlabelled: they draw the real reach of the row.
        if not var.empty:
            ax.scatter(var["lift"], [y] * len(var), s=9,
                       c=[_case_colour(c) for c in var["case"]],
                       edgecolors="none", alpha=0.35, zorder=3)
        if not bare.empty:
            ax.scatter(bare["lift"], [y] * len(bare), s=34,
                       c=[_case_colour(c) for c in bare["case"]],
                       edgecolors=S.EDGE, linewidths=0.4, zorder=5)

        if var.empty:
            continue
        # The points of interest are the 2 ends. Choose distinct MODIFIERS,
        # never points: `bare tree` in 3 cases is 1 descriptor, and 3 copies
        # of the word would take the room 3 other descriptors need.
        means = var.groupby("modifier")["lift"].mean().sort_values()
        half = n_label // 2
        chosen = list(dict.fromkeys(
            list(means.index[:half]) + list(means.index[-(n_label - half):])))
        shown = var[var["modifier"].isin(chosen)]
        # Every case that used the descriptor keeps its own point.
        ax.scatter(shown["lift"], [y] * len(shown), s=26,
                   c=[_case_colour(c) for c in shown["case"]],
                   edgecolors=S.EDGE, linewidths=0.4, zorder=6)
        pending += [(y, float(means[m]), m) for m in chosen]

    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.relim()
    ax.autoscale_view(scaley=False)
    lo, hi = ax.get_xlim()
    # A word of `n` characters is about `0.55 * n * fontsize` points wide.
    # Turn that into data units so 2 labels can be tested for a collision.
    axis_points = fig.get_size_inches()[0] * 72.0 * 0.82
    per_point = (hi - lo) / max(axis_points, 1.0)
    fontsize = 5.4

    def _half_width(word: str) -> float:
        return 0.5 * 0.55 * fontsize * len(word) * per_point

    levels = [(8, "bottom"), (-8, "top"), (18, "bottom"), (-18, "top")]
    for y in range(len(order)):
        row = sorted((x, m) for ry, x, m in pending if ry == y)
        taken: Dict[int, Tuple[float, float]] = {}
        for x, modifier in row:
            width = _half_width(modifier)
            level = 0
            for level in range(len(levels)):
                if level not in taken:
                    break
                last_x, last_w = taken[level]
                if x - last_x > width + last_w + 0.006:
                    break
            taken[level] = (x, width)
            offset, valign = levels[level]
            ax.annotate(
                modifier, (x, y),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va=valign, fontsize=fontsize, color=S.EDGE,
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.75),
            )

    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.75, len(order) - 0.25)
    ax.set_xlabel("lift over the base rate of the case\n"
                  "(share of the wins the photograph holding the cue takes)")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(
        handles=[Line2D([], [], marker="o", linestyle="none", markersize=4.0,
                        markeredgecolor=S.EDGE, markeredgewidth=0.3,
                        markerfacecolor=_case_colour(c),
                        label=WA.case_label(c)) for c in cases]
        + [Line2D([], [], marker="o", linestyle="none", markersize=5.2,
                  markerfacecolor=S.PAL["warmgrey"], markeredgecolor=S.EDGE,
                  markeredgewidth=0.4, label="the word alone"),
           Line2D([], [], marker="o", linestyle="none", markersize=3.0,
                  markerfacecolor=S.PAL["warmgrey"], markeredgecolor="none",
                  alpha=0.5, label="with a modifier")],
        loc="lower right", frameon=True, framealpha=0.92, fontsize=6.2,
    )
    fig.tight_layout()
    return fig


def plot_price(
    summary: pd.DataFrame,
    max_labels: int = 30,
    figsize: Tuple[float, float] = (6.8, 4.4),
):
    """The scatter: the price of a cue against how much that price moves.

    | Axis | Meaning |
    |------|---------|
    | x | the mean lift over the cases — the height of the rung |
    | y | the spread of that lift between cases |

    A point low on y keeps its price whatever the question asks. The colour is
    the model's OWN reading of the cue, thus the figure needs no outside
    judgment about which cues should be good.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if summary.empty:
        raise ValueError("no keys to draw")
    x = summary["mean_lift"].to_numpy()
    y = summary["sd_lift"].to_numpy()
    n = summary["n"].to_numpy(dtype=float)
    size = 18 + 40 * (np.sqrt(n) / max(np.sqrt(n).max(), 1e-9))
    valence = list(summary["valence"])
    colours = [VALENCE_COLOUR.get(v, S.PAL["blue"]) for v in valence]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0.0, color=S.EDGE, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(float(np.median(y)), color=S.PAL["warmgrey"], linewidth=0.7,
               linestyle=(0, (4, 3)), alpha=0.8, zorder=1)
    ax.scatter(x, y, s=size, c=colours, edgecolors=S.EDGE,
               linewidths=np.where(summary["same_sign"], 0.9, 0.25),
               alpha=0.95, zorder=3)

    ax.set_xlabel("mean lift over the base rate of each case "
                  "(the price of the cue)")
    ax.set_ylabel(f"spread of the price over the "
                  f"{int(summary['n_groups'].iloc[0])} cases (SD)")
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0.10)

    picks = summary.reindex(
        summary["mean_lift"].abs().sort_values(ascending=False).index
    ).head(max_labels)
    WA._place_labels(
        ax,
        list(zip(picks["mean_lift"], picks["sd_lift"])),
        [_pretty(k) for k in picks["key"]],
        [VALENCE_COLOUR.get(v, S.PAL["blue"]) for v in picks["valence"]],
    )

    seen = [v for v in VALENCE_ORDER if v in set(valence)]
    if seen:
        ax.legend(
            handles=[Line2D([], [], marker="o", linestyle="none",
                            markersize=4.5, markeredgecolor=S.EDGE,
                            markeredgewidth=0.3,
                            markerfacecolor=VALENCE_COLOUR[v],
                            label=f"the model calls it {v}") for v in seen],
            loc="upper left", frameon=True, framealpha=0.9, fontsize=6.5,
        )
    fig.tight_layout()
    return fig


def plot_price_pairs(matrix: pd.DataFrame, n_cols: int = 4,
                     figsize: Tuple[float, float] = (7.0, 4.0)):
    """1 panel for each case: its price against the price of the other cases.

    The axes are linear, because a lift is signed. A tight diagonal says the
    case prices the cues the way the rest of the battery does.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    if matrix.shape[1] < 2:
        raise ValueError("2 cases or more are needed")
    cols = list(matrix.columns)
    n_rows = int(math.ceil(len(cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    lo = float(matrix.to_numpy().min())
    hi = float(matrix.to_numpy().max())
    pad = 0.08 * (hi - lo)

    for ax, col in zip(axes, cols):
        others = matrix[[c for c in cols if c != col]].mean(axis=1)
        rho, _ = spearmanr(matrix[col], others)
        ax.axhline(0, color=S.EDGE, linewidth=0.5, alpha=0.4, zorder=1)
        ax.axvline(0, color=S.EDGE, linewidth=0.5, alpha=0.4, zorder=1)
        ax.plot([lo, hi], [lo, hi], linewidth=0.6, color=S.EDGE, alpha=0.5,
                zorder=2)
        ax.scatter(others, matrix[col], s=12, color=_case_colour(col),
                   edgecolors=S.EDGE, linewidths=0.2, alpha=0.9, zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        _halo = dict(facecolor=S.PAPER, edgecolor="none", alpha=0.8, pad=1.2)
        ax.text(0.04, 0.96, WA.case_label(col), transform=ax.transAxes,
                fontsize=7, va="top", color=S.EDGE, bbox=_halo, zorder=4)
        ax.text(0.96, 0.05, f"rho = {rho:.2f}", transform=ax.transAxes,
                fontsize=6.5, va="bottom", ha="right", color=S.EDGE,
                bbox=_halo, zorder=4)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.supxlabel("price in the other cases (lift)", fontsize=8)
    fig.supylabel("price in this case (lift)", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    tables: pd.DataFrame,
    rate_summary: Optional[pd.DataFrame],
    out_dir: Path,
    tag: str = "cue_hierarchy",
    max_keys: int = 30,
    gate_ok: bool = True,
    provenance: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Write the ladder, the 2 scatters, the agreement block, and the tables.

    Args:
        gate_ok: Did `_extractions.load` accept the corpus? When it did not,
            every name takes a `_stale` suffix, and the export writes a
            provenance file beside the figures.

    Warning: the suffix is not decoration. A stale corpus comes from trace runs
    that the registry does not name, thus its figures answer an older question.
    A file called `cue_hierarchy_ladder.png` must always come from the
    registered runs, and no reader should have to check which.
    """
    import matplotlib.pyplot as plt

    if not gate_ok:
        tag = f"{tag}_stale"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _save(fig, name: str) -> None:
        fig.savefig(out_dir / name, dpi=300)
        plt.close(fig)
        written.append(name)

    _save(plot_ladder(matrix, summary, max_keys=max_keys), f"{tag}_ladder.png")
    _save(plot_price(summary), f"{tag}_price.png")
    _save(plot_price_pairs(matrix), f"{tag}_price_pairs.png")
    if matrix.shape[1] >= 2:
        _save(WA.plot_agreement(matrix), f"{tag}_agreement.png")

    summary.round(4).to_csv(out_dir / f"{tag}_keys.csv", index=False)
    written.append(f"{tag}_keys.csv")
    matrix.round(4).to_csv(out_dir / f"{tag}_lift_matrix.csv")
    written.append(f"{tag}_lift_matrix.csv")
    tables.round(4).to_csv(out_dir / f"{tag}_per_case.csv", index=False)
    written.append(f"{tag}_per_case.csv")
    pairs = agreement(matrix)
    if not pairs.empty:
        pairs.round(4).to_csv(out_dir / f"{tag}_agreement.csv", index=False)
        written.append(f"{tag}_agreement.csv")
    if rate_summary is not None and not rate_summary.empty:
        rate_summary.round(4).to_csv(out_dir / f"{tag}_rates.csv", index=False)
        written.append(f"{tag}_rates.csv")
    if provenance is not None and not provenance.empty:
        provenance.to_csv(out_dir / f"{tag}_provenance.csv", index=False)
        written.append(f"{tag}_provenance.csv")
    return written


def export_words(
    df: pd.DataFrame,
    out_dir: Path,
    tag: str = "cue_words",
    min_count_case: int = 50,
    min_count_shared: int = 25,
    max_keys: int = 44,
    gate_ok: bool = True,
) -> List[str]:
    """Write the 2 word-level figures and their tables.

    The word unit answers a different question from `export`. `export` asks
    whether 1 aesthetic standard crosses the cases, and it must abstract the
    words away to get a key that every case scores. This function keeps the
    word and gives up some of that reach:

    | Figure | Cases | Keys | What it answers |
    |--------|-------|------|-----------------|
    | `_per_case.png` | 7 | 78 to 315 each | What does each case value? |
    | `_facility_ladder.png` | 5 | about 50 | Do the words price the same? |

    Road quality and street photography pair random street images, thus they
    share almost no word with the other cases. The shared ladder therefore
    runs over `FACILITY_CASES`, and the per-case figure holds all 7.
    """
    import matplotlib.pyplot as plt

    if not gate_ok:
        tag = f"{tag}_stale"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _save(fig, name: str) -> None:
        fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    per_case = lift_tables(df, unit="cue_norm", min_count=min_count_case)
    if per_case.empty:
        return written
    _save(plot_case_ladder(per_case), f"{tag}_per_case.png")
    per_case.round(4).to_csv(out_dir / f"{tag}_per_case.csv", index=False)
    written.append(f"{tag}_per_case.csv")

    facility = df[df["case"].isin(FACILITY_CASES)]
    shared = lift_tables(facility, unit="cue_norm", min_count=min_count_shared)
    if shared.empty:
        return written
    matrix = lift_matrix(shared)
    if matrix.empty or matrix.shape[1] < 2:
        return written
    summary = lift_summary(matrix, shared)
    _save(plot_ladder(matrix, summary, max_keys=max_keys, figsize=(6.6, 8.6)),
          f"{tag}_facility_ladder.png")
    summary.round(4).to_csv(out_dir / f"{tag}_facility_keys.csv", index=False)
    matrix.round(4).to_csv(out_dir / f"{tag}_facility_matrix.csv")
    written += [f"{tag}_facility_keys.csv", f"{tag}_facility_matrix.csv"]
    pairs = agreement(matrix)
    if not pairs.empty:
        pairs.round(4).to_csv(out_dir / f"{tag}_facility_agreement.csv",
                              index=False)
        written.append(f"{tag}_facility_agreement.csv")
    # The outlier table reads EVERY case, not the facility set: a cue that
    # reverses does so because 1 case disagrees, and dropping a case would
    # hide exactly the row a reader wants.
    odd = outliers(df, min_count=min_count_shared)
    if not odd.empty:
        (odd[[c for c in OUTLIER_COLUMNS if c in odd.columns]]
         .round(4).to_csv(out_dir / f"{tag}_outliers.csv", index=False))
        written.append(f"{tag}_outliers.csv")
    # The widest rows of the shared ladder, opened up by their modifiers.
    try:
        _save(plot_ladder_modifiers(shared, summary), f"{tag}_modifiers.png")
    except ValueError:
        pass
    return written


def provenance_table(df: pd.DataFrame, gate_ok: bool) -> pd.DataFrame:
    """Name the corpus behind a figure: the case, the rater, and the source.

    A cue figure is 2 steps downstream of a judgment run — the judge wrote the
    trace, and the extractor read it. Thus it must name both, and it must say
    whether the registry accepts the source.
    """
    cols = [c for c in ("case", "judge_model", "extractor_model", "sweep",
                        "schema_name", "schema_version", "source_results_path")
            if c in df.columns]
    out = df[cols].drop_duplicates().sort_values("case").reset_index(drop=True)
    out["registry_gate"] = "passed" if gate_ok else "FAILED, the corpus is stale"
    return out
