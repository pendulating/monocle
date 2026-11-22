"""Turn the IC ingredient table into Integrative Complexity codes.

`ic_schema` returns ingredients and no code. This module is the later step it
names: it reads the long ingredient table, builds 1 row for each trace, and
gives that trace a code. Every input is a located span, thus every code can be
audited back to the words that produced it.

Why the split
-------------
The GPU run is expensive and the thresholds are not settled. A threshold change
here costs seconds; a change in the prompt costs 59 GPU-hours. Thus the run
writes ingredients, and the code lives here.

The scale
---------
Integrative Complexity runs 1 to 7. This module derives 1 to 6.

| Code | Name | What the trace shows |
|------|------|----------------------|
| 1 | no differentiation | 1 view. The trace names cues and decides |
| 2 | transitional | a hedge or a set-aside alternative, but no second view |
| 3 | differentiation | 2 or more views, or cues on both sides, held apart |
| 4 | transitional | differentiation, plus a link that is not justified |
| 5 | integration | a justified weighing that relates the differentiated views |
| 6 | integration + context | the weighing holds under a named condition, or 2 mechanisms |
| 7 | higher-order | OUT OF REACH. See the warning below |

**Warning: 7 is not reachable from this schema.** Code 7 needs an organizing
principle above the integrations, and `ic_schema` holds no ingredient for it.
A reader must read this scale as 1 to 6 and never as "no trace reached 7".

What counts, and what does not
------------------------------
- Only a LOCATED span counts. `quote_found` false means the extractor wrote a
  quote that no search finds in the trace, thus it is a defect and not data.
- A justification counts only when its own sub-quote is located.
- An absence inside a CUT answer is not a zero. `truncated` marks such a trace,
  and `code_table` keeps it so a caller can drop it.

See `vlm-narratives-docs/ic-ingredient-extraction.md`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

__version__ = "1.0.0"

# The schema this module reads. A table of another version does not pool with
# this one: an ingredient rate belongs to a version, because the position of a
# class in the prompt changes how much of it the model reports.
READS_SCHEMA = ("ic_ingredients", "v2")

MAX_CODE = 6


@dataclass(frozen=True)
class Thresholds:
    """Every number a code depends on. Change 1 and run again; no GPU needed."""

    # A perspective is "developed" when it carries this many located supporting
    # spans. 2 is the codebook's own bar: 1 quote restates the perspective.
    supporting_quotes: int = 2
    # Differentiation needs this many distinct dimensions, when no perspective
    # is developed.
    distinct_dimensions: int = 2
    # Code 6 needs this many distinct weighing mechanisms, when no condition is
    # named.
    mechanisms_for_context: int = 2


DEFAULT = Thresholds()


def _attrs(value: Any) -> Dict[str, Any]:
    """Read the `attrs_json` column of 1 row."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _located(spans: Any) -> int:
    """Count the sub-quotes that a search really found in the trace."""
    out = 0
    for span in spans or []:
        if isinstance(span, dict) and span.get("method") not in (None, "none"):
            out += 1
    return out


def _norm(name: Any) -> str:
    """Reduce a dimension name so 2 spellings of 1 idea meet.

    `Cleanliness of the platform` and `platform cleanliness` both become
    `cleanliness of the platform` in word order, thus a sorted word set is the
    key. Without this the distinct count follows the wording and not the idea.
    """
    text = re.sub(r"[^a-z ]+", " ", str(name or "").lower())
    words = [w for w in text.split() if len(w) > 2 and w not in _STOP]
    return " ".join(sorted(set(words)))


_STOP = frozenset(
    "the a an and or of in on at to for with its their this that both image "
    "images photo photos picture pictures side".split()
)


def trace_components(rows: pd.DataFrame, t: Thresholds = DEFAULT) -> Dict[str, Any]:
    """Read 1 trace and return every component a code rests on.

    Args:
        rows: The ingredient rows of 1 trace. A row whose `ingredient_type` is
            null means the trace gave nothing.

    Returns:
        A flat dict. Every count is a count of LOCATED ingredients.
    """
    real = rows[rows["ingredient_type"].notna()]
    located = real[real["quote_found"].astype(bool)]

    out: Dict[str, Any] = {
        "n_ingredients": int(len(real)),
        "n_unlocated": int(len(real) - len(located)),
        "truncated": bool(rows["answer_truncated"].max()) if len(rows) else False,
    }

    by = {kind: located[located["ingredient_type"] == kind]
          for kind in ("dimension", "perspective", "verdict", "weighing",
                       "dismissal", "reconsideration", "hedge")}
    for kind, part in by.items():
        out[f"n_{kind}"] = int(len(part))

    # ---- differentiation -------------------------------------------------
    dims = by["dimension"]
    dim_attrs = [_attrs(v) for v in dims["attrs_json"]]
    names = {_norm(n) for n in dims["name"] if _norm(n)}
    out["n_dimensions_distinct"] = int(len(names))
    images = {str(a.get("image", "")) for a in dim_attrs}
    valences = {str(a.get("valence", "")) for a in dim_attrs}
    out["dimensions_both_images"] = bool({"A", "B"} <= images or "both" in images)
    out["dimensions_both_valences"] = bool({"good", "bad"} <= valences)
    out["n_evaluative"] = sum(1 for a in dim_attrs if a.get("type") == "evaluative")

    persp_attrs = [_attrs(v) for v in by["perspective"]["attrs_json"]]
    out["n_perspectives_developed"] = sum(
        1 for a in persp_attrs
        if _located(a.get("supporting_quotes")) >= t.supporting_quotes
    )
    out["perspectives_both_sides"] = bool(
        {"A", "B"} <= {str(a.get("favors", "")) for a in persp_attrs}
    )

    # A trace that lists cues for 1 image only has NOT differentiated. It has
    # described. The codebook calls that pseudo-differentiation.
    out["differentiated"] = bool(
        out["n_perspectives_developed"] >= 2
        or (out["n_perspectives_developed"] >= 1 and out["perspectives_both_sides"])
        or (out["n_dimensions_distinct"] >= t.distinct_dimensions
            and (out["dimensions_both_images"] or out["dimensions_both_valences"]))
    )
    out["pseudo_differentiation"] = bool(
        out["n_dimensions_distinct"] >= t.distinct_dimensions
        and not out["dimensions_both_images"]
        and not out["dimensions_both_valences"]
        and out["n_perspectives_developed"] == 0
    )

    # ---- integration -----------------------------------------------------
    weigh_attrs = [_attrs(v) for v in by["weighing"]["attrs_json"]]
    out["n_weighings_justified"] = sum(
        1 for a in weigh_attrs if _located(a.get("justification_quotes")) >= 1
    )
    out["n_weighings_conditional"] = sum(
        1 for a in weigh_attrs
        if _located(a.get("condition_quotes")) >= 1 or a.get("mechanism") == "conditional"
    )
    out["mechanisms"] = sorted({str(a.get("mechanism", "")) for a in weigh_attrs} - {""})
    out["n_mechanisms"] = len(out["mechanisms"])
    out["integrated"] = bool(out["n_weighings_justified"] >= 1 and out["differentiated"])
    out["context_sensitive"] = bool(
        out["n_weighings_conditional"] >= 1
        or out["n_mechanisms"] >= t.mechanisms_for_context
    )

    # ---- the softer signals ---------------------------------------------
    hedge_attrs = [_attrs(v) for v in by["hedge"]["attrs_json"]]
    out["n_hedges_justified"] = sum(
        1 for a in hedge_attrs if _located(a.get("justification_quotes")) >= 1
    )
    out["hedge_affects_conclusion"] = any(
        bool(a.get("affects_conclusion")) for a in hedge_attrs
    )

    # A revised verdict needs the ORDER of the spans, not the array order: the
    # extractor may report the final label first.
    verdicts = by["verdict"].sort_values("char_start", na_position="last")
    labels = [str(_attrs(v).get("label", "")) for v in verdicts["attrs_json"]]
    labels = [x for x in labels if x]
    out["verdict_labels"] = labels
    out["verdict_revised"] = bool(len(set(labels)) > 1)
    finals = [str(_attrs(v).get("label", "")) for v in verdicts["attrs_json"]
              if _attrs(v).get("is_final")]
    out["verdict_final"] = finals[-1] if finals else (labels[-1] if labels else "")
    return out


def score_from_components(c: Dict[str, Any], t: Thresholds = DEFAULT) -> int:
    """Give 1 trace its code. The rungs are ordered, thus the last one wins."""
    code = 1
    # 2: the trace shows that another reading exists, without developing one.
    if c["n_hedge"] >= 1 or c["n_dismissal"] >= 1:
        code = 2
    if c["differentiated"]:
        code = 3
        # 4: the views meet, but nothing justifies the link.
        if c["n_weighing"] >= 1 or c["verdict_revised"] or c["n_reconsideration"] >= 1:
            code = 4
    if c["integrated"]:
        code = 5
        if c["context_sensitive"]:
            code = 6
    return min(code, MAX_CODE)


def code_table(df: pd.DataFrame, t: Thresholds = DEFAULT) -> pd.DataFrame:
    """Build 1 row for each trace: the components, and the code.

    Args:
        df: The long ingredient table, 1 or many cases.

    Raises:
        ValueError: when the table holds another schema version.
    """
    _check_schema(df)
    keep = [c for c in ("case", "judge_model", "extractor_model", "sweep",
                        "pair_id", "relative_label", "presented_label",
                        "source_results_path", "schema_version") if c in df.columns]
    # Group on the RUN and the trace, never on the trace alone. `doc_id` is the
    # pair id, which each run numbers from 0, thus the same id names a
    # different pair in every case. A group on `doc_id` alone silently pools 7
    # cases into 1 trace and gives that trace the case of whichever row came
    # first.
    group_keys = [c for c in ("case", "judge_model", "doc_id") if c in df.columns]
    out: List[Dict[str, Any]] = []
    for _, rows in df.groupby(group_keys, sort=False):
        comp = trace_components(rows, t)
        head = rows.iloc[0]
        rec: Dict[str, Any] = {"doc_id": head["doc_id"]}
        rec.update({k: head[k] for k in keep})
        rec.update(comp)
        rec["ic_code"] = score_from_components(comp, t)
        out.append(rec)
    table = pd.DataFrame(out)
    # A list column cannot go to parquet without a cast, and a reader wants the
    # text anyway.
    for col in ("mechanisms", "verdict_labels"):
        if col in table.columns:
            table[col] = table[col].apply(lambda v: "|".join(v) if isinstance(v, list) else "")
    return table


def _check_schema(df: pd.DataFrame) -> None:
    name = set(df.get("schema_name", pd.Series(dtype=str)).dropna().unique())
    version = set(df.get("schema_version", pd.Series(dtype=str)).dropna().unique())
    want_name, want_version = READS_SCHEMA
    if name and name != {want_name}:
        raise ValueError(f"this module reads {want_name}, and the table holds {sorted(name)}")
    if version and version != {want_version}:
        raise ValueError(
            f"this module reads {want_name}/{want_version}, and the table holds "
            f"{sorted(version)}. Two versions do not pool: the position of a class "
            f"in the prompt changes how much of it the model reports.")


def summarize(codes: pd.DataFrame, drop_truncated: bool = True) -> pd.DataFrame:
    """Give 1 row for each case: the mean code, the shares, and the components.

    Args:
        drop_truncated: Leave out a trace whose answer the token cap cut. An
            absence in a cut answer is not a zero, thus a cut trace pushes every
            rate down.
    """
    df = codes[~codes["truncated"]] if drop_truncated else codes
    rows = []
    for case, part in df.groupby("case"):
        rec: Dict[str, Any] = {
            "case": case,
            "traces": int(len(part)),
            "dropped_truncated": int((codes["case"] == case).sum() - len(part)),
            "mean_code": round(float(part["ic_code"].mean()), 3),
            "median_code": float(part["ic_code"].median()),
        }
        for code in range(1, MAX_CODE + 1):
            rec[f"code_{code}"] = round(float((part["ic_code"] == code).mean()), 4)
        for flag in ("differentiated", "pseudo_differentiation", "integrated",
                     "context_sensitive", "verdict_revised"):
            rec[flag] = round(float(part[flag].mean()), 4)
        rec["dimensions_per_trace"] = round(float(part["n_dimensions_distinct"].mean()), 2)
        rec["unlocated_per_trace"] = round(float(part["n_unlocated"].mean()), 2)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("mean_code", ascending=False).reset_index(drop=True)


def by_label(codes: pd.DataFrame, drop_truncated: bool = True) -> pd.DataFrame:
    """Ask whether the code moves with the judgment the model gave.

    A `Same` or a `NotSure` should cost more reasoning than a `MuchMore`, if the
    code measures anything. This is the first test of the whole pipeline.
    """
    df = codes[~codes["truncated"]] if drop_truncated else codes
    if "relative_label" not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["case", "relative_label"]).agg(
        traces=("ic_code", "size"),
        mean_code=("ic_code", "mean"),
        integrated=("integrated", "mean"),
    ).reset_index()
    g["mean_code"] = g["mean_code"].round(3)
    g["integrated"] = g["integrated"].round(4)
    return g


def thresholds_used(t: Thresholds = DEFAULT) -> Dict[str, Any]:
    """The numbers behind a table, for the provenance of a figure."""
    out = asdict(t)
    out.update({"module_version": __version__, "reads_schema": "/".join(READS_SCHEMA),
                "max_code": MAX_CODE})
    return out
