"""Structured extractions over the reasoning traces.

`_traces.py` counts words. This module counts CLAIMS. The extraction stage
(`dagspaces/urbanpairvqa/stages/trace_extract.py`) turns each trace into typed
spans with attributes and character offsets, and this module reads them.

What a word count cannot answer, and this can:

| Question | Where the answer is |
|----------|--------------------|
| Which image holds a cue? | `visual_evidence.image` |
| Is the cue good or bad in the model's reading? | `visual_evidence.valence` |
| Does the model go past the pixels? | class `inference` |
| Does it reason about the people present? | class `person_reference` |
| Does a cue go with a label? | `label_association` |
| Can I quote the sentence? | `is_quotable` and the offsets |

Where the data is
-----------------
`scripts/merge_trace_extractions.py` writes 1 parquet for each case, under
`data/trace_extractions/`. Point `load` somewhere else with `root`.

Warning: count `is_quotable` rows only. A `match_lesser` row holds a sentence
the model COMPOSED out of the text plus its own words, and its offsets point at
a fragment of it. `load` drops those rows by default.

Warning: an attribute value is a free string, not an enum. The guided schema
fixes the class names and the attribute names; the model still writes a value
that the prompt never lists. `normalize` maps a value onto the declared
vocabulary and keeps the original in `<attribute>_raw`. Read
`vocabulary_report` before you quote a rate.

See `vlm-narratives-docs/langextract-trace-extraction.md`.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

import _provenance as P
import _traces as T

__version__ = "1.0.0"

REPO_ROOT = P.REPO_ROOT
DEFAULT_ROOT = REPO_ROOT / "data" / "trace_extractions"

# The schema this module reads. A parquet of another version does not pool with
# this one: a class or an attribute may mean something else.
SCHEMA_NAME = "urban_cues"
SCHEMA_VERSION = "v1"

# The declared vocabulary, from `conf/extract/urban_cues_v1.yaml`.
#
# Warning: keep this in step with the config, and raise `SCHEMA_VERSION` when
# either one changes. A value that is absent here becomes `other`, thus a stale
# list quietly moves real signal into the `other` bucket.
VOCABULARY: Dict[str, Dict[str, tuple]] = {
    "visual_evidence": {
        "image": ("A", "B", "both", "unclear"),
        "valence": ("good", "bad", "neutral"),
        "category": (
            "people", "cleanliness", "upkeep", "greenery", "light", "signage",
            "traffic", "construction", "commerce", "architecture", "other",
        ),
    },
    "inference": {
        "image": ("A", "B", "both", "unclear"),
        "kind": (
            "safety", "quality", "upkeep", "wealth", "class", "crime",
            "demographic", "other",
        ),
    },
    "person_reference": {
        "image": ("A", "B", "both", "unclear"),
        "used_in_judgment": ("yes", "no"),
    },
    "image_artifact": {
        "image": ("A", "B", "both", "unclear"),
        "kind": ("blur", "angle", "occlusion", "time_of_day", "weather", "other"),
    },
    "comparison": {
        "direction": ("A", "B", "neither"),
        # `dimension` is a free phrase on purpose. Do not force it.
        "hedged": ("yes", "no"),
    },
    "uncertainty": {
        "reason": ("looks_equal", "cannot_see", "out_of_view", "other"),
    },
    "decision": {
        "label": T.LABELS,
    },
}

CLASSES = tuple(VOCABULARY)

# The classes that carry the risk result: the model goes past the pixels, or it
# reasons about the people in the photograph.
RISK_KINDS = ("wealth", "class", "demographic", "crime")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def find_files(root: Optional[str | Path] = None) -> List[Path]:
    """Find the merged extraction parquets, 1 for each case."""
    base = Path(root) if root else DEFAULT_ROOT
    if not base.is_absolute():
        base = REPO_ROOT / base
    return sorted(base.glob("*_extractions.parquet"))


def registry_mismatch(root: Optional[str | Path] = None) -> List[str]:
    """Name every extraction file that does NOT come from a registered run.

    An extraction corpus is 1 step downstream of a thinking run: it holds the
    spans an extractor read out of that run's traces. Thus a corpus built from
    an older run describes an older PROMPT, and its class rates and its quotes
    answer a question the paper no longer asks. The word clouds beside them
    would come from the new runs, and no reader could see the difference.

    This compares the `source_results_path` of each corpus with the trace runs
    of the canonical registry. An empty list means the corpus is current.
    """
    import os

    import _canonical as C

    reg = {(r.case, r.model_config): os.path.realpath(r.results_path)
           for r in C.runs(kind="trace")}
    problems: List[str] = []
    for f in find_files(root):
        cols = ["case", "judge_model", "sweep", "source_results_path"]
        try:
            rows = pd.read_parquet(f, columns=cols).drop_duplicates()
        except Exception as exc:
            problems.append(f"{f.name}: cannot read its provenance ({exc})")
            continue
        for r in rows.itertuples():
            want = reg.get((r.case, r.judge_model))
            if want is None:
                problems.append(
                    f"{r.case}: no registered trace run for {r.judge_model}")
            elif os.path.realpath(str(r.source_results_path)) != want:
                problems.append(
                    f"{r.case}: extractions come from sweep {r.sweep} "
                    f"({Path(str(r.source_results_path)).name}), not from the "
                    f"registered run ({Path(want).name})")
    return problems


def load(
    cases: Optional[Sequence[str]] = None,
    root: Optional[str | Path] = None,
    quotable_only: bool = True,
    attributes: bool = True,
    normalize_values: bool = True,
    require_canonical: bool = True,
) -> pd.DataFrame:
    """Read every case, and give 1 row for each extraction.

    Args:
        cases: Keep only these cases. `None` keeps all of them, which the
            `distinctive` score needs as its background.
        quotable_only: Drop the rows whose span cannot be quoted. See the
            warning at the top.
        attributes: Open `attributes_json` into a column for each attribute.
        normalize_values: Map each attribute value onto the declared
            vocabulary, and keep the original in `<attribute>_raw`.
        require_canonical: Stop when the corpus does not come from the
            canonical trace runs. Keep this True for every paper figure.

    Raises:
        FileNotFoundError: when no parquet is under `root`.
        ValueError: when a file carries another schema version.
    """
    files = find_files(root)
    if not files:
        raise FileNotFoundError(
            f"no extraction parquet under {root or DEFAULT_ROOT}. "
            "Run scripts/merge_trace_extractions.py first."
        )
    if require_canonical:
        stale = registry_mismatch(root)
        if stale:
            raise RuntimeError(
                "the extraction corpus does not come from the canonical trace "
                "runs:\n  " + "\n  ".join(stale)
                + "\nRun the extraction stage on the registered runs, then "
                  "scripts/merge_trace_extractions.py. "
                  "Pass require_canonical=False only to look at the old corpus."
            )
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    versions = set(df.get("schema_version", pd.Series(dtype=str)).dropna().unique())
    if versions and versions != {SCHEMA_VERSION}:
        raise ValueError(
            f"this module reads {SCHEMA_NAME}/{SCHEMA_VERSION}, and the data holds "
            f"{sorted(versions)}. Two schema versions do not pool."
        )

    if cases:
        df = df[df.case.isin(list(cases))]
    # Keep the silent-trace rows out of every count. They carry no class.
    df = df[df.extraction_class.notna()].copy()
    if quotable_only and "is_quotable" in df.columns:
        df = df[df.is_quotable].copy()
    if attributes:
        df = attach_attributes(df)
        if normalize_values:
            df = normalize(df)
    return df.reset_index(drop=True)


def attach_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Open `attributes_json` into 1 column for each attribute name."""
    if "attributes_json" not in df.columns:
        return df
    parsed = df.attributes_json.fillna("{}").map(json.loads)
    wide = pd.json_normalize(parsed)
    wide.index = df.index
    # A name that already exists stays. The stage owns those columns.
    wide = wide[[c for c in wide.columns if c not in df.columns]]
    return pd.concat([df, wide], axis=1)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map each attribute value onto the declared vocabulary.

    The original value stays in `<attribute>_raw`, thus nothing is lost and
    `vocabulary_report` can name what moved.

    `dimension` is a free phrase, so it is only lowered and stripped.
    """
    out = df.copy()
    for attribute in _attribute_names():
        if attribute not in out.columns:
            continue
        raw = out[attribute].map(_clean_value)
        out[f"{attribute}_raw"] = raw
        out[attribute] = [
            _to_vocabulary(cls, attribute, value)
            for cls, value in zip(out.extraction_class, raw)
        ]
    return out


def _attribute_names() -> List[str]:
    names: List[str] = []
    for attrs in VOCABULARY.values():
        for name in attrs:
            if name not in names:
                names.append(name)
    return names


def _clean_value(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.replace(" ", "_").replace("-", "_")


def _to_vocabulary(extraction_class: str, attribute: str, value):
    """Give the declared value, or `other`. Keep a free field as it is.

    Warning: coerce before you compare. An attribute value is whatever the
    model wrote inside the JSON, thus it can arrive as a number or a bool, not
    only as a string: `{"used_in_judgment": 1}` is valid JSON and pandas then
    holds a float. A bare `value.lower()` raises `'float' object has no
    attribute 'lower'` on such a row.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if not isinstance(value, str):
        value = str(value)
    allowed = VOCABULARY.get(str(extraction_class), {}).get(attribute)
    if allowed is None:
        # Either a free field such as `dimension`, or an attribute this class
        # does not declare. Lower it and leave it.
        return value.lower()
    for option in allowed:
        if value.lower() == str(option).lower():
            return option
    return "other"


def vocabulary_report(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Name the values that `normalize` moved into `other`.

    Read this before you quote a rate. A large `other` means the vocabulary
    under-covers the case, not that the model said nothing.
    """
    rows: List[Dict[str, object]] = []
    for extraction_class, attrs in VOCABULARY.items():
        part = df[df.extraction_class == extraction_class]
        if part.empty:
            continue
        for attribute in attrs:
            raw_col = f"{attribute}_raw"
            if attribute not in part.columns or raw_col not in part.columns:
                continue
            moved = part[part[attribute] == "other"]
            declared = {str(o).lower() for o in attrs[attribute]}
            # `.astype(str)` first: a column of numbers has no `.str` accessor,
            # and a raw value is whatever the model wrote in its JSON.
            raw_text = moved[raw_col].astype(str)
            unlisted = moved[~raw_text.str.lower().isin(declared)]
            if unlisted.empty:
                continue
            counts = unlisted[raw_col].value_counts()
            rows.append({
                "extraction_class": extraction_class,
                "attribute": attribute,
                "moved_to_other": int(len(unlisted)),
                "share_of_class": round(len(unlisted) / len(part), 4),
                "top_values": ", ".join(
                    f"{v} ({n})" for v, n in counts.head(top).items()
                ),
            })
    return pd.DataFrame(rows).sort_values(
        "moved_to_other", ascending=False, ignore_index=True
    ) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------


_TOTALS_CACHE: Dict[str, pd.Series] = {}


def trace_totals(root: Optional[str | Path] = None) -> pd.Series:
    """Count every trace of each case, before any filter.

    Warning: this is the denominator of every rate, and it is NOT the number of
    traces left in a filtered frame. A trace whose spans were all dropped, or
    that held no extraction at all, still asked the question and still got an
    answer. Leave it out of the denominator and every rate reads too high.

    The read is 2 columns, thus it costs little, and the result is cached.
    """
    key = str(root or DEFAULT_ROOT)
    if key not in _TOTALS_CACHE:
        frames = [
            pd.read_parquet(f, columns=["case", "pair_id"]) for f in find_files(root)
        ]
        if not frames:
            return pd.Series(dtype=int)
        whole = pd.concat(frames, ignore_index=True)
        _TOTALS_CACHE[key] = whole.groupby("case").pair_id.nunique()
    return _TOTALS_CACHE[key]


def trace_counts(df: pd.DataFrame, root: Optional[str | Path] = None) -> pd.Series:
    """Give the denominator of a rate: every trace of each case.

    This falls back to the traces present in `df` only when the files cannot be
    read, for example in a test that builds a frame by hand.
    """
    totals = trace_totals(root)
    present = df.groupby("case").pair_id.nunique()
    if totals.empty:
        return present
    return totals.reindex(present.index).fillna(present).astype(int)


def class_rates(df: pd.DataFrame, per: int = 100) -> pd.DataFrame:
    """Give the number of extractions of each class, for each `per` traces.

    A rate, never a raw count: the cases hold the same 11,000 traces today, but
    a later case may not, and a raw count would then compare 2 different things.
    """
    traces = trace_counts(df)
    table = pd.crosstab(df.extraction_class, df.case).astype(float)
    for case in table.columns:
        table[case] = table[case] / max(1, int(traces.get(case, 0))) * per
    return table.round(2)


def attribute_rates(
    df: pd.DataFrame, extraction_class: str, attribute: str, per: int = 100
) -> pd.DataFrame:
    """Give the rate of each value of 1 attribute, for each `per` traces."""
    part = df[df.extraction_class == extraction_class]
    if part.empty or attribute not in part.columns:
        return pd.DataFrame()
    traces = trace_counts(df)
    table = pd.crosstab(part[attribute], part.case).astype(float)
    for case in table.columns:
        table[case] = table[case] / max(1, int(traces.get(case, 0))) * per
    return table.round(2)


def risk_panel(df: pd.DataFrame, per: int = 100) -> pd.DataFrame:
    """The result the framework exists to find, for each `per` traces.

    2 things a word count cannot see, because their words are ordinary:

    - the model infers wealth, class, a demographic, or crime from a street,
    - the model reasons about the people in the photograph.
    """
    traces = trace_counts(df)
    rows: Dict[str, Dict[str, float]] = {}

    inference = df[(df.extraction_class == "inference") & df.get("kind").notna()]
    for kind in RISK_KINDS:
        part = inference[inference["kind"] == kind]
        rows[f"inference: {kind}"] = _rate_by_case(part, traces, per)

    people = df[df.extraction_class == "person_reference"]
    rows["person_reference: any"] = _rate_by_case(people, traces, per)
    if "used_in_judgment" in people.columns:
        rows["person_reference: used in judgment"] = _rate_by_case(
            people[people.used_in_judgment == "yes"], traces, per
        )

    artifact = df[df.extraction_class == "image_artifact"]
    rows["image_artifact: any"] = _rate_by_case(artifact, traces, per)

    table = pd.DataFrame(rows).T
    return table.reindex(sorted(table.columns), axis=1).round(2)


def _rate_by_case(part: pd.DataFrame, traces: pd.Series, per: int) -> Dict[str, float]:
    counts = part.groupby("case").size() if len(part) else pd.Series(dtype=float)
    return {
        case: round(float(counts.get(case, 0)) / max(1, int(n)) * per, 2)
        for case, n in traces.items()
    }


# ---------------------------------------------------------------------------
# The distinctive score, over claims instead of words
# ---------------------------------------------------------------------------

# What 1 unit of the count is.
UNITS = ("class", "class_attr", "text", "class_text")


# The class whose text is scaffold, not content. Every trace ends with "I'll go
# with "Same"", thus the span says nothing about the case. The label of that
# decision is already a column, so nothing is lost by leaving it out of a
# distinctive table.
SCAFFOLD_CLASSES = ("decision",)


def counters(
    df: pd.DataFrame,
    unit: str = "class_text",
    exclude_classes: Sequence[str] = (),
) -> Dict[str, Counter]:
    """Build 1 counter for each case, in the unit you name.

    | `unit` | 1 key looks like |
    |--------|------------------|
    | `class` | `visual_evidence` |
    | `class_attr` | `visual_evidence:category=greenery` |
    | `text` | `graffiti` |
    | `class_text` | `visual_evidence:graffiti` |

    `class_text` is the direct upgrade over the word cloud: the unit is a thing
    the model named, not a word it used.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    if exclude_classes:
        df = df[~df.extraction_class.isin(list(exclude_classes))]
    keys = _unit_keys(df, unit)
    out: Dict[str, Counter] = {}
    for case, key in zip(df.case, keys):
        if key is None:
            continue
        out.setdefault(case, Counter())[key] += 1
    return out


def _unit_keys(df: pd.DataFrame, unit: str) -> List[Optional[str]]:
    if unit == "class":
        return [str(c) for c in df.extraction_class]
    if unit == "text":
        return [_span_key(t) for t in df.extraction_text]
    if unit == "class_text":
        return [
            None if _span_key(t) is None else f"{c}:{_span_key(t)}"
            for c, t in zip(df.extraction_class, df.extraction_text)
        ]
    # class_attr: 1 key for each attribute the class declares.
    keys: List[Optional[str]] = []
    for row in df.itertuples():
        cls = str(row.extraction_class)
        parts = []
        for attribute in VOCABULARY.get(cls, {}):
            value = getattr(row, attribute, None)
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                parts.append(f"{attribute}={value}")
        keys.append(f"{cls}:{'|'.join(parts)}" if parts else cls)
    return keys


def _span_key(text) -> Optional[str]:
    """Reduce a span to a comparable key: lower case, 1 space between words.

    Warning: this is NOT a lemma. `trash can` and `trash cans` stay 2 keys. A
    stemmer would join them, and it would also join words that mean different
    things. Read the top of a table, not its tail.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    key = " ".join(str(text).lower().split())
    return key or None


def distinctive(
    df: pd.DataFrame,
    case: str,
    unit: str = "class_text",
    max_rows: int = 60,
    min_count: int = 5,
    prior_weight: float = 500.0,
    exclude_classes: Sequence[str] = SCAFFOLD_CLASSES,
) -> pd.DataFrame:
    """Score what 1 case names that the others do not.

    This is the same log-odds ratio with an informative Dirichlet prior that
    `_traces.distinctive_scores` applies to words (Monroe, Colaresi, and Quinn,
    2008). Only the unit changes: a claim instead of a word.

    Warning: the score needs 2 cases or more. With 1 case there is no
    background, and this returns the plain counts.

    `exclude_classes` drops the scaffold. Every trace ends with a decision
    sentence, thus `decision:i'll go with "same"` would rank high in every case
    and say nothing.
    """
    by_case = counters(df, unit=unit, exclude_classes=exclude_classes)
    target = by_case.get(case, Counter())
    if not target:
        return pd.DataFrame()
    background: Counter = Counter()
    for other, counter in by_case.items():
        if other != case:
            background.update(counter)
    if not background:
        rows = [{"unit": k, "score": float(n), "count": n} for k, n in
                target.most_common(max_rows)]
        return pd.DataFrame(rows)

    scores = T.distinctive_scores(
        target, background, prior_weight=prior_weight, min_count=min_count
    )
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_rows]
    return pd.DataFrame([
        {
            "unit": key,
            "score": round(score, 2),
            "count": target.get(key, 0),
            "count_elsewhere": background.get(key, 0),
        }
        for key, score in top if score > 0
    ])


# ---------------------------------------------------------------------------
# The label, and the quotes
# ---------------------------------------------------------------------------


def label_association(
    df: pd.DataFrame,
    case: str,
    extraction_class: str = "visual_evidence",
    attribute: Optional[str] = "valence",
    label_column: str = "presented_label",
) -> pd.DataFrame:
    """Ask whether a claim goes with a judgment.

    For each label, this gives the number of matching extractions for each 100
    traces that carry that label. Read it beside the base rate: a class that is
    common everywhere tells you nothing.

    Warning: this is an association, not a cause. The trace is what the model
    WROTE, and the label is what it answered. Neither one made the other.
    """
    part = df[df.case == case]
    if part.empty or label_column not in part.columns:
        return pd.DataFrame()
    traces = part.groupby(label_column).pair_id.nunique()

    target = part[part.extraction_class == extraction_class]
    if attribute and attribute in target.columns:
        counts = pd.crosstab(target[label_column], target[attribute])
    else:
        counts = target.groupby(label_column).size().to_frame("count")

    table = counts.astype(float)
    for label in table.index:
        table.loc[label] = table.loc[label] / max(1, int(traces.get(label, 0))) * 100
    table["traces"] = [int(traces.get(label, 0)) for label in table.index]
    return table.round(2)


def quotes(
    df: pd.DataFrame,
    case: str,
    extraction_class: str,
    n: int = 10,
    attribute: Optional[str] = None,
    value: Optional[str] = None,
    seed: int = 777,
) -> pd.DataFrame:
    """Give quotable spans, with the pair and the offsets that locate them.

    Every row here passed the alignment test, thus the text is the model's own
    text and a reader can find it in the trace.
    """
    part = df[(df.case == case) & (df.extraction_class == extraction_class)]
    if attribute and value is not None and attribute in part.columns:
        part = part[part[attribute] == value]
    if part.empty:
        return pd.DataFrame()
    take = part.sample(n=min(n, len(part)), random_state=seed)
    columns = [
        "pair_id", "presented_label", "extraction_text", "char_start", "char_end",
        "alignment_status",
    ]
    extra = [a for a in _attribute_names() if a in take.columns]
    return take[[c for c in columns + extra if c in take.columns]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Win rates: does a cue go with the image that won?
# ---------------------------------------------------------------------------

# The labels that decide a comparison. `Same` is a real judgment, and `NotSure`
# is an abstention, but neither one names a winner, thus neither can enter a
# win rate. Both shares are reported beside it.
_WIN_FOR_A = ("More", "MuchMore")
_WIN_FOR_B = ("Less", "MuchLess")
_UNDECIDED = ("Same", "NotSure")


def win_rates(
    df: pd.DataFrame,
    case: str,
    classes: Sequence[str] = ("visual_evidence",),
    min_count: int = 25,
    prior_strength: float = 25.0,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Ask, for each cue, how often the image it was attached to won.

    The extraction names the image, thus the question can be directional. A cue
    on image A wins when the label says A; a cue on image B wins when the label
    says B. That is much stronger than a test of presence: both images live in
    ONE trace, so presence alone cannot say which image a cue described.

    Args:
        classes: Which extraction classes to score. The default is the cue
            class. `inference` also works and reads differently.
        min_count: Drop a cue seen fewer times than this.
        prior_strength: The weight of the prior, in observations. See below.

    Returns:
        A row for each cue: `n`, `wins`, `win_rate`, `shrunk_rate`, the Wilson
        interval, and the share of comparisons that no label decided.

    Three rules keep the number honest:

    1. **1 vote for each comparison.** A trace that names "trash can" 3 times
       for image A still describes 1 comparison. The count is over distinct
       (trace, cue, image) triples.
    2. **A cue on BOTH images is dropped.** If the model sees a fence in A and
       in B, the fence separates nothing, and counting it twice adds 1 win and
       1 loss to every such pair.
    3. **The rate is shrunk toward the base rate.** A cue seen 4 times would
       otherwise read 100% and take the top of the colour scale. This is the
       same idea as the Dirichlet prior of the distinctive score.

    Warning: this is an association, not a cause. The model narrates while it
    decides, so a cue may follow the judgment rather than drive it. Read it as
    "the cues that accompany a win", never as "the cues that win".
    """
    part = df[(df.case == case) & df.extraction_class.isin(list(classes))]
    if part.empty or "image" not in part.columns:
        return pd.DataFrame()
    part = part[part["image"].isin(["A", "B"])].copy()
    if part.empty or "presented_label" not in part.columns:
        return pd.DataFrame()

    part["cue"] = [_span_key(t) for t in part.extraction_text]
    part = part[part.cue.notna()]

    # Rule 1: 1 vote for each (trace, cue, image).
    votes = part.drop_duplicates(subset=["pair_id", "cue", "image"])

    # Rule 2: drop a cue that the model attached to BOTH images of one trace.
    sides = votes.groupby(["pair_id", "cue"])["image"].transform("nunique")
    votes = votes[sides == 1]
    if votes.empty:
        return pd.DataFrame()

    label = votes.presented_label
    decided = ~label.isin(_UNDECIDED)
    won = (
        (votes.image.eq("A") & label.isin(_WIN_FOR_A))
        | (votes.image.eq("B") & label.isin(_WIN_FOR_B))
    )

    frame = pd.DataFrame({
        "cue": votes.cue.values,
        "decided": decided.values,
        "won": (won & decided).values,
    })
    grouped = frame.groupby("cue")
    table = pd.DataFrame({
        "n_mentions": grouped.size(),
        "n": grouped.decided.sum(),
        "wins": grouped.won.sum(),
    })
    table["undecided_share"] = (
        1 - table["n"] / table["n_mentions"]
    ).round(3)
    table = table[table["n"] >= min_count]
    if table.empty:
        return pd.DataFrame()

    table["win_rate"] = (table.wins / table.n).round(4)
    base = float(table.wins.sum()) / float(table.n.sum())
    table["shrunk_rate"] = (
        (table.wins + prior_strength * base) / (table.n + prior_strength)
    ).round(4)
    low, high = zip(*(_wilson(w, n) for w, n in zip(table.wins, table.n)))
    table["ci_low"] = [round(v, 4) for v in low]
    table["ci_high"] = [round(v, 4) for v in high]
    table["base_rate"] = round(base, 4)

    table = table.sort_values("shrunk_rate", ascending=False).reset_index()
    return table.head(max_rows) if max_rows else table


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple:
    """The Wilson interval of a proportion.

    Warning: do not use the normal interval here. A cue with 30 of 30 wins
    gives a normal interval of zero width, which claims certainty from 30
    observations.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def unit_win_rates(
    df: pd.DataFrame,
    case: str,
    min_pairs: int = 8,
    prior_strength: float = 8.0,
) -> pd.DataFrame:
    """Score the UNITS, not the cues: how often does each place win?

    A cue win rate asks what the model says. A unit win rate asks which
    schools, entrances, or parks it prefers. The 2 are different questions, and
    only this one can put a photograph on the page.

    The numbers come from the source run, not from the extractions: the results
    parquet holds a label for every pair, and `pairs.parquet` beside it holds
    the unit ids and the image paths. Thus a pair whose spans were all dropped
    still votes.

    Warning: the swap matters. `presented_label` describes the image the model
    saw FIRST, and `is_swapped` says which side that was. Read `unit_uid_a` as
    the model's image A only when the pair was not swapped.

    Returns:
        A row for each unit: `n`, `wins`, `win_rate`, `shrunk_rate`, the unit
        name, and 1 image path. Empty when the manifest is not on disk.
    """
    part = df[df.case == case]
    if part.empty or "source_results_path" not in part.columns:
        return pd.DataFrame()
    results_path = str(part.source_results_path.iloc[0])
    pairs_path = os.path.join(os.path.dirname(results_path), "pairs.parquet")
    if not (os.path.exists(results_path) and os.path.exists(pairs_path)):
        return pd.DataFrame()

    # Warning: a case that pairs IMAGES has no unit at all. `road_quality` and
    # `street_photography` run `pair_sampler.mode=image`, and their manifest
    # carries no `unit_uid_a`, because a block of street is not a facility.
    # There is nothing to rank, thus this returns empty and the figure keeps
    # its words.
    import pyarrow.parquet as pq

    columns = [
        "pair_id", "is_swapped", "unit_uid_a", "unit_uid_b",
        "unit_name_a", "unit_name_b", "presented_left_path", "presented_right_path",
    ]
    available = set(pq.ParquetFile(pairs_path).schema_arrow.names)
    missing = [c for c in columns if c not in available]
    if missing:
        print(
            f"[{case}] the pair manifest has no {missing[0]}; this case pairs "
            "images, not units, thus it has no unit ranking"
        )
        return pd.DataFrame()

    labels = pd.read_parquet(results_path, columns=["pair_id", "presented_label"])
    # The facing filter scores how well a face points at the unit. Take it
    # when it is there, so the picture can be the best of the unit's images
    # rather than an arbitrary one.
    for side in ("a", "b"):
        for extra in (f"attribution_confidence_{side}", f"distance_to_unit_ft_{side}"):
            if extra in available:
                columns.append(extra)
    manifest = pd.read_parquet(pairs_path, columns=columns)
    joined = labels.merge(manifest, on="pair_id", how="inner")
    if joined.empty:
        return pd.DataFrame()

    swapped = joined.is_swapped.fillna(False).astype(bool)
    # The model's image A is the one it saw first, thus the swap decides which
    # unit that was. The image paths already carry the presented order.
    unit_a = joined.unit_uid_b.where(swapped, joined.unit_uid_a)
    unit_b = joined.unit_uid_a.where(swapped, joined.unit_uid_b)
    name_a = joined.unit_name_b.where(swapped, joined.unit_name_a)
    name_b = joined.unit_name_a.where(swapped, joined.unit_name_b)

    decided = ~joined.presented_label.isin(_UNDECIDED)
    a_won = joined.presented_label.isin(_WIN_FOR_A)
    b_won = joined.presented_label.isin(_WIN_FOR_B)

    def _side(column: str, default):
        """Give the value of the side the model saw first, then the second."""
        col_a, col_b = f"{column}_a", f"{column}_b"
        if col_a not in joined.columns:
            return pd.Series(default, index=joined.index), pd.Series(default, index=joined.index)
        first = joined[col_b].where(swapped, joined[col_a])
        second = joined[col_a].where(swapped, joined[col_b])
        return first, second

    conf_a, conf_b = _side("attribution_confidence", float("nan"))
    dist_a, dist_b = _side("distance_to_unit_ft", float("nan"))

    long = pd.concat([
        pd.DataFrame({
            "unit": unit_a, "unit_name": name_a,
            "image_path": joined.presented_left_path,
            "confidence": conf_a, "distance_ft": dist_a,
            "decided": decided, "won": a_won & decided,
        }),
        pd.DataFrame({
            "unit": unit_b, "unit_name": name_b,
            "image_path": joined.presented_right_path,
            "confidence": conf_b, "distance_ft": dist_b,
            "decided": decided, "won": b_won & decided,
        }),
    ], ignore_index=True)

    grouped = long.groupby("unit")
    # Warning: a unit is NOT one photograph. It holds 8 images at the median,
    # and each image is judged about 1 time. Thus the win rate belongs to the
    # unit, and any single picture is only an example of what the model saw.
    # Show the best-facing one, never an arbitrary one.
    best = long.sort_values("confidence", ascending=False, na_position="last")
    best = best.drop_duplicates("unit").set_index("unit")
    table = pd.DataFrame({
        "n": grouped.decided.sum(),
        "wins": grouped.won.sum(),
        "unit_name": grouped.unit_name.first(),
        "n_images": grouped.image_path.nunique(),
        "image_path": best.image_path,
        "image_confidence": best.confidence.round(3),
        "image_distance_ft": best.distance_ft.round(0),
    })
    # The prior comes from EVERY unit, before the `min_pairs` filter. Each
    # decided pair gives 1 winner and 1 loser, thus the base rate over the
    # whole set is 0.5 by construction. Take it after the filter and it falls
    # to about 0.37, because a unit reaches 8 decided comparisons only when the
    # model keeps deciding about it, and it decides more readily against a
    # place. Shrinking toward that biased value pushes every unit down.
    base = float(table.wins.sum()) / max(1.0, float(table["n"].sum()))

    table = table[table["n"] >= min_pairs]
    if table.empty:
        return pd.DataFrame()

    table["win_rate"] = (table.wins / table.n).round(4)
    table["shrunk_rate"] = (
        (table.wins + prior_strength * base) / (table.n + prior_strength)
    ).round(4)
    table["base_rate"] = round(base, 4)
    return table.sort_values("shrunk_rate", ascending=False).reset_index()


def valence_consistency(df: pd.DataFrame, case: str) -> pd.DataFrame:
    """Test the model against itself.

    The model labels each cue `good`, `bad`, or `neutral`, and it also picks a
    winner. Those 2 statements should agree: a `good` cue should sit on the
    image that won. Where they do not, the trace and the judgment disagree.
    """
    rows = []
    part = df[(df.case == case) & (df.extraction_class == "visual_evidence")]
    if part.empty or "valence" not in part.columns:
        return pd.DataFrame()
    for valence, group in part.groupby("valence"):
        table = win_rates(
            pd.concat([group.assign(case=case)]), case, min_count=1, prior_strength=0.0
        )
        if table.empty:
            continue
        rows.append({
            "valence": valence,
            "cues": int(len(table)),
            "decided_comparisons": int(table.n.sum()),
            "win_rate": round(float(table.wins.sum()) / float(table.n.sum()), 4),
        })
    return pd.DataFrame(rows)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """State what the corpus holds. The paper cites this table."""
    rows = []
    for case, part in df.groupby("case"):
        rows.append({
            "case": case,
            "traces": int(part.pair_id.nunique()),
            "extractions": int(len(part)),
            "for_each_trace": round(len(part) / max(1, part.pair_id.nunique()), 1),
            "judge_model": ", ".join(sorted(set(part.judge_model.dropna()))),
            "extractor_model": ", ".join(
                sorted({os.path.basename(str(m)) for m in part.extractor_model.dropna()})
            ),
            "schema": f"{part.schema_name.iloc[0]}/{part.schema_version.iloc[0]}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_class_rates(
    df: pd.DataFrame,
    case: Optional[str] = None,
    per: int = 100,
    title: Optional[str] = None,
):
    """Draw the class mix of each case. Return the figure.

    The house palette comes from `_style`, so this figure and a word cloud read
    as one system.
    """
    import matplotlib.pyplot as plt

    import _style as S

    table = class_rates(df, per=per)
    if table.empty:
        raise ValueError("no extraction to draw")
    order = table.mean(axis=1).sort_values(ascending=True).index
    table = table.loc[order]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(table) * max(1, len(table.columns) / 3) + 1.6))
    colors = S.INK * (len(table.columns) // len(S.INK) + 1)
    table.plot(
        kind="barh", ax=ax, color=colors[: len(table.columns)],
        edgecolor=S.EDGE, linewidth=S.EDGE_LW, width=0.8,
    )
    ax.set_xlabel(f"extractions for each {per} traces")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=10, color=S.EDGE)
    ax.legend(frameon=False, fontsize=8)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout()
    return fig


def plot_risk_panel(df: pd.DataFrame, per: int = 100, title: Optional[str] = None):
    """Draw the risk rates: inference past the pixels, and people."""
    import matplotlib.pyplot as plt

    import _style as S

    table = risk_panel(df, per=per)
    if table.empty:
        raise ValueError("no extraction to draw")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 2.0))
    colors = S.INK * (len(table.columns) // len(S.INK) + 1)
    table.plot(
        kind="barh", ax=ax, color=colors[: len(table.columns)],
        edgecolor=S.EDGE, linewidth=S.EDGE_LW, width=0.8,
    )
    ax.set_xlabel(f"extractions for each {per} traces")
    ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=10, color=S.EDGE)
    ax.legend(frameon=False, fontsize=8)
    fig.patch.set_facecolor(S.PAPER)
    fig.tight_layout()
    return fig


def plot_win_cloud(
    df: pd.DataFrame,
    case: str,
    classes: Sequence[str] = ("visual_evidence",),
    min_count: int = 25,
    max_words: int = 120,
    cmap=None,
    title: Optional[str] = None,
    width: int = 1600,
    height: int = 900,
):
    """Draw the cues, sized by how often the model names them and coloured by
    how often the image they sit on wins. Return `(figure, table)`.

    This merges the 2 questions a word cloud has to keep apart: WHAT the model
    talks about (the size) and WHAT IT REWARDS (the colour).

    Warning: the colour diverges around the BASE RATE, not around 50%. A cue
    lands on the winning image about 60% of the time in every case, because the
    model lists more cues for the image it prefers. Centre the scale at 0.5 and
    almost every word turns green, which says nothing.

    Warning: the default ramp is coral-to-teal, NOT red-to-green. About 8% of
    men cannot separate red from green. Pass `cmap` to override it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm, to_hex

    import _style as S

    table = win_rates(df, case, classes=classes, min_count=min_count)
    if table.empty:
        raise ValueError(f"no cue of {case} reaches min_count={min_count}")
    table = table.nlargest(max_words, "n_mentions")

    base = float(table.base_rate.iloc[0])
    rates = dict(zip(table.cue, table.shrunk_rate))
    lo, hi = float(table.shrunk_rate.min()), float(table.shrunk_rate.max())
    # TwoSlopeNorm needs the centre strictly inside the range.
    lo, hi = min(lo, base - 1e-3), max(hi, base + 1e-3)
    norm = TwoSlopeNorm(vmin=lo, vcenter=base, vmax=hi)
    ramp = cmap if cmap is not None else S.CMAP_DIV

    def _colour(word, **kwargs):
        return to_hex(ramp(norm(rates.get(word, base))))

    cloud = T.make_cloud(
        dict(zip(table.cue, table.n_mentions)), width=width, height=height,
    )
    cloud.recolor(color_func=_colour)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color=S.EDGE)
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=ramp), ax=ax,
        orientation="horizontal", fraction=0.05, pad=0.04,
    )
    bar.set_label(
        f"win rate of the image the cue sits on  (base rate {base:.2f})",
        fontsize=7,
    )
    bar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig, table


def plot_win_block(
    df: pd.DataFrame,
    case: str,
    classes: Sequence[str] = ("visual_evidence",),
    min_count: int = 25,
    max_words: Optional[int] = None,
    cmap=None,
    title: Optional[str] = None,
    fontsize: float = 11.0,
    title_fontsize: float = 11.5,
    label_fontsize: float = 9.5,
    width_in: float = 9.0,
    separator: str = "  ·  ",
    unit_photos: int = 0,
    photo_in: float = 1.15,
):
    """Set the cues as a block of text, best win rate first. Return `(fig, table)`.

    This is the compact form of the win cloud, and it reads better. A cloud
    spends its area on 2 channels at once: the size says how often the model
    names a cue, and the colour says how often that cue's image wins. Area is
    the hardest channel to compare, and the 2 variables then fight.

    The block drops the size channel. Reading ORDER carries the win rate, and
    the colour repeats it, thus a reader scans from the best cue to the worst
    without comparing 2 areas. It also takes about a third of the height.

    The count of each cue is in the returned table, not in the picture.

    Args:
        unit_photos: Put a row of this many photographs above the block, from
            the units the model ranks highest, and the same number below it,
            from the units it ranks lowest. 0 leaves them out.

    Warning: a photograph row shows UNITS, and the words show CUES. They are 2
    different statistics of the same run, and the middle of each scale differs:
    a cue sits on the winning image about 60% of the time, while a unit wins
    exactly half of its decided comparisons by construction.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import TwoSlopeNorm, to_hex

    import _style as S

    table = win_rates(df, case, classes=classes, min_count=min_count)
    if table.empty:
        raise ValueError(f"no cue of {case} reaches min_count={min_count}")
    if max_words:
        table = table.head(max_words)
    table = table.sort_values("shrunk_rate", ascending=False).reset_index(drop=True)

    base = float(table.base_rate.iloc[0])
    lo = min(float(table.shrunk_rate.min()), base - 1e-3)
    hi = max(float(table.shrunk_rate.max()), base + 1e-3)
    norm = TwoSlopeNorm(vmin=lo, vcenter=base, vmax=hi)
    ramp = cmap if cmap is not None else S.CMAP_DIV
    words = [
        (cue + (separator if i < len(table) - 1 else ""), to_hex(ramp(norm(rate))))
        for i, (cue, rate) in enumerate(zip(table.cue, table.shrunk_rate))
    ]

    # 2 passes. The first lays the words out to learn how many lines they need,
    # and the second draws them on a figure of exactly that height. The wrap
    # depends on the WIDTH alone, so the count the first pass returns holds for
    # any height.
    #
    # Warning: size the axes in INCHES, not in a fraction of the figure. A
    # fixed fraction leaves the axes too short as the line count grows, and the
    # last lines then run under the colour bar.
    line_in = fontsize / 72.0 * 1.55
    lines = _layout_words(words, fontsize, width_in, block_in=12.0, ax=None, fig=None)

    # The margins follow the type sizes. A title of 11.5 pt needs more room
    # than one of 9 pt, and the colour bar needs its tick row AND its label.
    #
    # No title by default: these figures go into LaTeX, where `\\caption` says
    # what the picture is. A title inside the image repeats the caption and
    # spends height that the page needs.
    top_in = (title_fontsize / 72.0 * 2.6) if title else 0.10
    bar_in = 0.15
    bottom_in = bar_in + label_fontsize / 72.0 * 4.2
    block_in = lines * line_in + 0.12

    # The photograph rows, when asked for. Each row is 1 band of images plus a
    # caption line.
    units = unit_win_rates(df, case) if unit_photos else pd.DataFrame()
    if unit_photos and units.empty:
        print(f"[{case}] no unit manifest on disk; the figure keeps the words only")
    photos = unit_photos if (unit_photos and not units.empty) else 0
    # A band holds the images and a caption line of 2 lines under each one.
    #
    # No heading over a band. These figures go into LaTeX, where `\\caption`
    # says which row is which. `head_in` is now plain separation from the
    # words, not room for text.
    head_in = 0.12
    cap_in = label_fontsize / 72.0 * 3.4
    row_in = (head_in + photo_in + cap_in) if photos else 0.0

    height_in = block_in + top_in + bottom_in + 2 * row_in

    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([
        0.01, (bottom_in + row_in) / height_in, 0.98, block_in / height_in,
    ])
    ax.axis("off")
    _layout_words(words, fontsize, width_in, block_in, ax=ax, fig=fig)
    if title:
        if photos:
            # A photograph band takes the top of the figure, thus the title
            # cannot hang off the word axes: it would land on the captions of
            # that band.
            fig.text(
                0.5, 1 - (top_in * 0.42) / height_in, title,
                ha="center", va="center", fontsize=title_fontsize, color=S.EDGE,
            )
        else:
            ax.set_title(title, fontsize=title_fontsize, color=S.EDGE, pad=6)
    cbar_ax = fig.add_axes([
        0.24, (bottom_in - bar_in - label_fontsize / 72.0 * 1.5) / height_in,
        0.52, bar_in / height_in,
    ])
    bar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=ramp), cax=cbar_ax, orientation="horizontal",
    )
    bar.set_label(f"win rate  (base rate {base:.2f})", fontsize=label_fontsize)
    bar.ax.tick_params(labelsize=label_fontsize - 1.0)

    if photos:
        # Each band runs, from its own bottom upward: caption, image, heading.
        _photo_row(
            fig, units.head(photos), width_in, height_in, photo_in,
            band_bottom_in=height_in - top_in - row_in,
            cap_in=cap_in, head_in=head_in,
            fontsize=label_fontsize - 1.0, ramp=ramp,
        )
        _photo_row(
            fig, units.tail(photos).iloc[::-1], width_in, height_in, photo_in,
            band_bottom_in=bottom_in,
            cap_in=cap_in, head_in=head_in,
            fontsize=label_fontsize - 1.0, ramp=ramp,
        )
    return fig, table


def _photo_row(
    fig,
    units: pd.DataFrame,
    width_in: float,
    height_in: float,
    photo_in: float,
    band_bottom_in: float,
    cap_in: float,
    head_in: float,
    fontsize: float,
    ramp,
) -> None:
    """Draw 1 horizontal band of unit photographs, with a caption under each.

    The band runs from `band_bottom_in` upward: the caption, then the images.
    It carries NO heading: the LaTeX caption names the rows, and the frame
    colour already separates the winners from the losers.

    Warning: a unit win rate has its own scale, and it is NOT the scale of the
    colour bar. A unit wins half its decided comparisons by construction, so
    the frames use a norm centred on 0.5 of their own.
    """
    from matplotlib.colors import TwoSlopeNorm
    from PIL import Image

    import _style as S

    if units.empty:
        return
    n = len(units)
    gap_in = 0.10
    span_in = n * photo_in + (n - 1) * gap_in
    left_in = (width_in - span_in) / 2.0
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    image_bottom_in = band_bottom_in + cap_in
    for i, row in enumerate(units.itertuples()):
        x_in = left_in + i * (photo_in + gap_in)
        ax = fig.add_axes([
            x_in / width_in, image_bottom_in / height_in,
            photo_in / width_in, photo_in / height_in,
        ])
        ax.set_xticks([])
        ax.set_yticks([])
        try:
            with Image.open(row.image_path) as image:
                # A face is 1024x1024. The page never shows it larger than
                # about 1 inch, thus a full read is wasted work and a large
                # file.
                image = image.convert("RGB")
                image.thumbnail((420, 420))
                ax.imshow(image)
        except Exception as exc:
            ax.text(0.5, 0.5, "no image", ha="center", va="center", fontsize=6)
            print(f"[photo] {row.image_path}: {exc}")
        for side in ax.spines.values():
            side.set_edgecolor(ramp(norm(float(row.shrunk_rate))))
            side.set_linewidth(1.8)
        # Warning: keep the caption narrower than the photograph. A unit name
        # runs to 38 characters ("34 St-Penn Station (Easement - Street)"), and
        # at this size that is 3 times the width of the frame, so 2 captions
        # collide and neither can be read.
        #
        # Warning: cut the name in the MIDDLE. "BOLD CHARTER SCHOOL" and "BOLD
        # CHARTER SCHOOL ANNEX" are 2 different units, and a cut at the end
        # gives both the same caption.
        # 1.15 in holds about 16 characters at this size. Measured, not
        # guessed: at 22 the captions of 2 neighbours ran into each other.
        name = str(row.unit_name)
        if len(name) > 16:
            name = name[:8] + "…" + name[-7:]
        # The camera stands about 117 ft away at the median, thus the distance
        # is part of what the reader needs to judge the picture.
        distance = getattr(row, "image_distance_ft", None)
        far = "" if distance is None or pd.isna(distance) else f", {int(distance)} ft"
        ax.set_xlabel(
            f"{name}\n{row.win_rate:.0%} of {int(row.n)}{far}",
            fontsize=fontsize - 1.5, color=S.EDGE, labelpad=2,
        )


def _layout_words(
    words: Sequence[tuple],
    fontsize: float,
    width_in: float,
    block_in: float,
    ax=None,
    fig=None,
) -> int:
    """Flow coloured words across lines. Return the number of lines used.

    `block_in` is the height of the TEXT BLOCK in inches, which fixes the line
    spacing in axes units. The wrap itself depends only on the width, thus the
    line count this returns is the same for any height.

    Warning: measure each word with the renderer. A guess from the character
    count is wrong for a proportional face, and the block then wraps early on
    one line and overflows on the next.
    """
    import matplotlib.pyplot as plt

    own_figure = ax is None
    if own_figure:
        fig = plt.figure(figsize=(width_in, block_in))
        ax = fig.add_axes([0.01, 0.0, 0.98, 1.0])
        ax.axis("off")
    renderer = fig.canvas.get_renderer()

    line_height = (fontsize / 72.0 * 1.55) / block_in
    x, y, lines = 0.0, 1.0, 1
    for text, colour in words:
        item = ax.text(x, y, text, color=colour, fontsize=fontsize,
                       va="top", ha="left", transform=ax.transAxes)
        width = item.get_window_extent(renderer=renderer).width / ax.bbox.width
        if x > 0 and x + width > 1.0:
            item.remove()
            x, y, lines = 0.0, y - line_height, lines + 1
            item = ax.text(x, y, text, color=colour, fontsize=fontsize,
                           va="top", ha="left", transform=ax.transAxes)
            width = item.get_window_extent(renderer=renderer).width / ax.bbox.width
        x += width
    if own_figure:
        plt.close(fig)
    return lines


def plot_win_bars(
    df: pd.DataFrame,
    case: str,
    classes: Sequence[str] = ("visual_evidence",),
    min_count: int = 25,
    top: int = 12,
    cmap=None,
    title: Optional[str] = None,
):
    """Draw the strongest cues at each end, with their Wilson intervals.

    A cloud cannot be read as a number: no reader can compare 2 word sizes, and
    none of it carries an interval. This is the figure the paper quotes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    import _style as S

    table = win_rates(df, case, classes=classes, min_count=min_count)
    if table.empty:
        raise ValueError(f"no cue of {case} reaches min_count={min_count}")
    base = float(table.base_rate.iloc[0])
    picked = pd.concat([table.head(top), table.tail(top)]).drop_duplicates("cue")
    picked = picked.sort_values("shrunk_rate")

    ramp = cmap if cmap is not None else S.CMAP_DIV
    norm = TwoSlopeNorm(
        vmin=min(picked.shrunk_rate.min(), base - 1e-3), vcenter=base,
        vmax=max(picked.shrunk_rate.max(), base + 1e-3),
    )
    colours = [ramp(norm(v)) for v in picked.shrunk_rate]

    fig, ax = plt.subplots(figsize=(7.2, 0.30 * len(picked) + 1.4))
    y = range(len(picked))
    ax.barh(list(y), picked.win_rate, color=colours,
            edgecolor=S.EDGE, linewidth=S.EDGE_LW, height=0.78)
    ax.errorbar(
        picked.win_rate, list(y),
        xerr=[picked.win_rate - picked.ci_low, picked.ci_high - picked.win_rate],
        fmt="none", ecolor=S.EDGE, elinewidth=0.6, capsize=1.6,
    )
    ax.axvline(base, color=S.EDGE, linestyle="--", linewidth=0.8)
    ax.text(base, len(picked) - 0.3, f" base rate {base:.2f}", fontsize=6.5,
            color=S.EDGE, va="top")
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(picked.cue, picked.n)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("win rate of the image the cue sits on, with the Wilson interval")
    if title:
        ax.set_title(title, fontsize=10, color=S.EDGE)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def export(
    case: str,
    df: pd.DataFrame,
    out_dir: str | Path,
    unit: str = "class_text",
    max_rows: int = 60,
    min_count: int = 25,
    unit_photos: int = 6,
) -> List[str]:
    """Write the tables and the figures of 1 case into that case's folder.

    Args:
        min_count: The fewest decided comparisons a cue needs. Raise it for a
            camera-ready figure: the tail is mostly rare wordings, and the
            block shrinks fast.
        unit_photos: Photographs in each band of the win block. 0 leaves the
            bands out.
    """
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    def _csv(frame: pd.DataFrame, name: str, index: bool = False) -> None:
        if frame is None or frame.empty:
            return
        frame.to_csv(out_dir / name, index=index)
        written.append(name)

    _csv(coverage(df), f"{case}_extraction_coverage.csv")
    _csv(class_rates(df), f"{case}_class_rates.csv", index=True)
    _csv(risk_panel(df), f"{case}_risk_panel.csv", index=True)
    _csv(distinctive(df, case, unit=unit, max_rows=max_rows),
         f"{case}_distinctive_{unit}.csv")
    _csv(vocabulary_report(df), f"{case}_vocabulary_report.csv")
    _csv(label_association(df, case), f"{case}_label_association.csv", index=True)
    _csv(quotes(df, case, "inference", n=20), f"{case}_quotes_inference.csv")
    _csv(win_rates(df, case, min_count=min_count), f"{case}_win_rates.csv")
    _csv(unit_win_rates(df, case), f"{case}_unit_win_rates.csv")
    _csv(valence_consistency(df, case), f"{case}_valence_consistency.csv")

    for name, builder in (
        (f"{case}_class_rates.png", lambda: plot_class_rates(df, case=case)),
        (f"{case}_risk_panel.png", lambda: plot_risk_panel(df)),
        (f"{case}_win_block.png",
         lambda: plot_win_block(df, case, min_count=min_count,
                                unit_photos=unit_photos)[0]),
        (f"{case}_win_bars.png", lambda: plot_win_bars(df, case, min_count=min_count)),
    ):
        try:
            fig = builder()
        except ValueError:
            continue
        fig.savefig(out_dir / name, dpi=200)
        plt.close(fig)
        written.append(name)
    return written
