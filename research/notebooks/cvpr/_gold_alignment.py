"""Compare the human spans of the gold set with the spans of each extractor.

The gold set is a Label Studio project (`research/scripts/build_gold_set.py`). A human
marks spans in a reasoning trace, and each extractor arm of `_extractions.ARMS`
holds spans for the same trace. This module reads both and measures how well
they line up.

Why not the scorer alone
------------------------
`research/scripts/score_gold_set.py` matches spans 1 to 1 by class and a Jaccard
overlap. The first annotated trace (2026-09-22) shows 2 facts that break that
measure:

1. **The human marks nested spans.** An `inference` often USES a piece of
   `visual_evidence` to make its claim, and the annotator marks both: the
   evidence inside the inference. One region can also carry 2 labels. Thus a
   character can belong to 2 classes at one time.
2. **The human marks whole clauses; the extractor quotes a phrase.** The
   human wrote 124 spans, the primary extractor 35, the control 18. The
   extractor span most often sits INSIDE the human span. A Jaccard of 0.5
   calls that a miss.

Thus the measures here read the characters, not the spans:

| Measure | Reads |
|---------|-------|
| `char_prf` | For each class, the characters the human and the extractor both mark. A nested character counts in each of its classes. |
| `ledger` | For each extractor span: exact, inside, contains, partial, other class, or none, against the human. |
| `human_coverage` | For each human span: how much of it a same-class extractor span covers. |
| `confusion` | The innermost human class under each extractor span, against the class the extractor gave. |
| `nesting` | The human's nested pairs, and which side each extractor took. |
| `decisions` | The human marks every candidate decision; the paper keeps the last one. |

Every function takes plain span tables (`pandas.DataFrame`) with the columns
`start`, `end`, `cls`, `text`, and the loader functions build them from the
export and from the corpora.

See `research/docs/langextract-trace-extraction.md`, "The gold set".
"""
from __future__ import annotations

import glob
import json
import os
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import _extractions as X
import _figmeta as F
import _latex as L
# Re-exported, thus a notebook can call `G.caption_md(fig)`.
from _figmeta import caption_md, captioned, save_figure  # noqa: F401

__version__ = "1.7.0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXPORT_DIR = REPO_ROOT / "research" / "notebooks" / "cvpr" / "cues_annotations"
GOLD_DIR = REPO_ROOT / "data" / "gold_set"

# The classes of both schema generations. Version 2 (7 classes) names
# `visual_evidence`; version 3 (12 classes, 2026-09-28) names `dimension`
# and adds the IC classes. A table reads the classes present in its data.
CLASSES = ("visual_evidence", "dimension", "inference", "perspective", "weighing", "justification",
           "dismissal", "verdict", "decision", "comparison", "uncertainty", "person_reference",
           "image_artifact")

CLASS_COLOURS = {
    "visual_evidence": "#4C9F70",
    "dimension": "#4C9F70",
    "inference": "#D9822B",
    "perspective": "#6C3FC5",
    "weighing": "#B5175A",
    "justification": "#1F7A8C",
    "dismissal": "#A93226",
    "verdict": "#B7950B",
    "decision": "#111111",
    "comparison": "#2E86C1",
    "uncertainty": "#8E44AD",
    "person_reference": "#C0392B",
    "image_artifact": "#7F8C8D",
}

# The Label Studio control name for each attribute of each class. `kind` has
# 2 controls, because `inference` and `image_artifact` do not share values.
CONTROL_OF = {
    ("inference", "kind"): "inference_kind",
    ("image_artifact", "kind"): "artifact_kind",
}

SPAN_COLUMNS = ["start", "end", "cls", "text", "region_id", "attributes"]

LEDGER_ORDER = ("exact", "inside", "contains", "partial", "other_class", "none")


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------

def find_exports(export_dir: Path | str = EXPORT_DIR) -> List[Path]:
    """The Label Studio exports, newest last."""
    paths = [Path(p) for p in glob.glob(os.path.join(str(export_dir), "*.json"))]
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def _trim(text: str, start: int, end: int) -> Tuple[int, int]:
    """Move the bounds inward past the white space. The human selection
    often takes a leading or a trailing space; the extractor never does."""
    s, e = start, end
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def spans_from_results(results: Sequence[Dict[str, Any]], text: str) -> pd.DataFrame:
    """One row for each (region, label). The choices attach by region id.

    A region with 2 labels gives 2 rows, with the same `region_id`. The
    bounds drop the white space at both ends, and `text` is the slice.
    """
    rows: List[Dict[str, Any]] = []
    attrs_by_region: Dict[str, Dict[str, str]] = {}
    for r in results:
        if r.get("type") != "choices":
            continue
        choices = (r.get("value") or {}).get("choices") or []
        if choices:
            attrs_by_region.setdefault(str(r.get("id")), {})[r["from_name"]] = str(choices[0])
    for r in results:
        if r.get("type") != "labels":
            continue
        v = r["value"]
        labels = v.get("labels") or []
        s, e = _trim(text, int(v["start"]), int(v["end"]))
        if e <= s:
            continue
        for label in labels:
            rows.append({
                "start": s, "end": e, "cls": label, "text": text[s:e],
                "region_id": str(r.get("id")), "attributes": dict(attrs_by_region.get(str(r.get("id")), {})),
            })
    return pd.DataFrame(rows, columns=SPAN_COLUMNS)


def image_checks(annotations: Sequence[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """The task-level answers: `shows_a` and `shows_b` (the depiction check
    of the sets of 2026-09-23 to 2026-09-30, since removed) and `notes` (the
    free text box of 2026-09-30 on: depiction notes, or anything else). The
    first annotation that answers wins. An export without them gives None."""
    out: Dict[str, Optional[str]] = {"shows_a": None, "shows_b": None, "notes": None}
    for a in annotations:
        for r in a.get("result", []):
            name = r.get("from_name")
            if name not in out or out[name] is not None:
                continue
            value = r.get("value") or {}
            if r.get("type") == "choices" and value.get("choices"):
                out[name] = str(value["choices"][0])
            elif r.get("type") == "textarea" and value.get("text"):
                text = value["text"]
                out[name] = "\n".join(text) if isinstance(text, list) else str(text)
    return out


def load_export(path: Path | str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read one export. Give (tasks, human_spans).

    `tasks` has 1 row for each task: task_id, case, judge, pair_id,
    presented_label, item, shows_a, shows_b, conf_min, conf_floor,
    conf_quantile, text, chars, n_annotations. The `conf_*` columns are the
    facing-filter confidence of the weaker image, the floor of its case, and
    its quantile in the case; None in an export from before 2026-09-23.
    `human_spans` has 1 row for each (task, annotation, region, label).
    """
    with open(path) as fh:
        data = json.load(fh)
    task_rows, span_frames = [], []
    for t in data:
        d = t["data"]
        text = d["text"]
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        checks = image_checks(anns)
        task_rows.append({
            "task_id": d.get("task_id") or str(t["id"]), "case": d.get("case"), "judge": d.get("judge"),
            "pair_id": d.get("pair_id"), "presented_label": d.get("presented_label"),
            "item": d.get("item"), "shows_a": checks.get("shows_a"), "shows_b": checks.get("shows_b"),
            "notes": checks.get("notes"),
            "conf_min": d.get("conf_min"), "conf_floor": d.get("conf_floor"),
            "conf_quantile": d.get("conf_quantile"),
            "text": text, "chars": len(text), "n_annotations": len(anns),
        })
        for a in anns:
            spans = spans_from_results(a.get("result", []), text)
            spans["task_id"] = d.get("task_id") or str(t["id"])
            spans["annotation_id"] = a.get("id")
            spans["annotator"] = a.get("completed_by")
            spans["lead_time_s"] = a.get("lead_time")
            span_frames.append(spans)
    tasks = pd.DataFrame(task_rows)
    human = pd.concat(span_frames, ignore_index=True) if span_frames else pd.DataFrame(
        columns=SPAN_COLUMNS + ["task_id", "annotation_id", "annotator", "lead_time_s"])
    return tasks, human


# ---------------------------------------------------------------------------
# The extractors
# ---------------------------------------------------------------------------

def arms_for(judges: Iterable[str], schema_version: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """The arms of `_extractions.ARMS` that read these judges and hold files."""
    judges = set(judges)
    out = {}
    for name, spec in X.arm_roots(present_only=True).items():
        if spec["judge"] not in judges:
            continue
        if schema_version and spec["schema_version"] != schema_version:
            continue
        out[name] = spec
    return out


def load_arm_spans(tasks: pd.DataFrame, arms: Optional[Dict[str, Dict[str, str]]] = None,
                   last_decision_only: bool = True, require_canonical: bool = True) -> pd.DataFrame:
    """The extractor spans of every annotated trace, for each arm.

    The rows are the ones `_extractions.load` keeps for the paper: quotable,
    not duplicate, judge answered, and 1 decision when `last_decision_only`.
    """
    if arms is None:
        arms = arms_for(tasks.judge.dropna().unique())
    frames = []
    keys = tasks[["task_id", "case", "judge", "pair_id"]]
    for name, spec in arms.items():
        part_tasks = keys[keys.judge == spec["judge"]]
        if part_tasks.empty:
            continue
        df = X.load(root=spec["root"], cases=sorted(part_tasks.case.unique()),
                    require_canonical=require_canonical, last_decision_only=last_decision_only)
        df = df[df.pair_id.isin(set(part_tasks.pair_id))]
        df = df.merge(part_tasks[["task_id", "case", "pair_id"]], on=["case", "pair_id"], how="inner")
        df = df[df.char_start.notna() & df.char_end.notna()]
        rows = []
        for e in df.itertuples():
            attrs = json.loads(e.attributes_json) if isinstance(e.attributes_json, str) else {}
            attrs = {CONTROL_OF.get((e.extraction_class, k), k): str(v) for k, v in attrs.items()
                     if v is not None and str(v) != ""}
            rows.append({
                "arm": name, "extractor": spec["extractor"], "schema_version": spec["schema_version"],
                "task_id": e.task_id, "start": int(e.char_start), "end": int(e.char_end),
                "cls": e.extraction_class, "text": str(e.extraction_text), "region_id": f"{name}_{e.Index}",
                "attributes": attrs, "alignment_status": getattr(e, "alignment_status", None),
                "is_verbatim": bool(getattr(e, "is_verbatim", True)),
            })
        frames.append(pd.DataFrame(rows))
    cols = ["arm", "extractor", "schema_version", "task_id"] + SPAN_COLUMNS + ["alignment_status", "is_verbatim"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Character-level agreement
# ---------------------------------------------------------------------------

def char_masks(spans: pd.DataFrame, n_chars: int) -> Dict[str, np.ndarray]:
    """One boolean mask for each class. A nested character is True in each
    class that covers it."""
    masks = {c: np.zeros(n_chars, dtype=bool) for c in CLASSES}
    for s in spans.itertuples():
        if s.cls in masks:
            masks[s.cls][max(0, s.start):min(n_chars, s.end)] = True
    return masks


def _prf(tp: float, n_pred: float, n_gold: float) -> Dict[str, float]:
    p = tp / n_pred if n_pred else float("nan")
    r = tp / n_gold if n_gold else float("nan")
    if n_pred and n_gold:
        f = 2 * p * r / (p + r) if (p + r) else 0.0
    elif n_pred or n_gold:
        f = 0.0
    else:
        f = float("nan")
    return {"precision": p, "recall": r, "f1": f}


def char_prf(human: pd.DataFrame, pred: pd.DataFrame, n_chars: int) -> pd.DataFrame:
    """Precision, recall, and F1 over the characters, for each class and in all.

    The `all` row is micro: it sums the (character, class) pairs. Thus a
    nested character contributes to each class it carries.
    """
    hm, pm = char_masks(human, n_chars), char_masks(pred, n_chars)
    rows, tp_all, np_all, ng_all = [], 0, 0, 0
    for c in CLASSES:
        tp = int((hm[c] & pm[c]).sum())
        n_p, n_g = int(pm[c].sum()), int(hm[c].sum())
        tp_all += tp; np_all += n_p; ng_all += n_g
        rows.append({"cls": c, "chars_human": n_g, "chars_pred": n_p, "chars_both": tp, **_prf(tp, n_p, n_g)})
    rows.append({"cls": "all", "chars_human": ng_all, "chars_pred": np_all, "chars_both": tp_all,
                 **_prf(tp_all, np_all, ng_all)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Span-level readings
# ---------------------------------------------------------------------------

def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def innermost_human(human: pd.DataFrame, start: int, end: int, min_share: float = 0.5
                    ) -> Optional[pd.Series]:
    """The shortest human span that covers at least `min_share` of [start, end)."""
    length = max(1, end - start)
    best = None
    for h in human.itertuples():
        if _overlap(start, end, h.start, h.end) / length >= min_share:
            if best is None or (h.end - h.start) < (best.end - best.start):
                best = h
    return None if best is None else human.loc[best.Index]


def ledger(human: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """One row for each extractor span, with its status against the human.

    | Status | Meaning |
    |--------|---------|
    | exact | a human span of the same class has the same bounds |
    | inside | the span sits inside a human span of the same class |
    | contains | the span holds a human span of the same class |
    | partial | the span overlaps a same-class human span, but neither holds the other |
    | other_class | it overlaps human spans of other classes only |
    | none | no human span touches it |

    `human_cls` is the class of the innermost human span that covers at
    least half of the extractor span, or None.
    """
    rows = []
    for p in pred.itertuples():
        same = human[human.cls == p.cls]
        status = "none"
        best_j = 0.0
        for h in same.itertuples():
            ov = _overlap(p.start, p.end, h.start, h.end)
            if ov <= 0:
                continue
            if h.start == p.start and h.end == p.end:
                status = "exact"; best_j = 1.0
                break
            if h.start <= p.start and p.end <= h.end:
                cand = "inside"
            elif p.start <= h.start and h.end <= p.end:
                cand = "contains"
            else:
                cand = "partial"
            j = ov / (max(p.end, h.end) - min(p.start, h.start))
            rank = {"none": 0, "partial": 1, "contains": 2, "inside": 3}
            if rank[cand] > rank.get(status, 0) or (cand == status and j > best_j):
                status, best_j = cand, j
        inner = innermost_human(human, p.start, p.end)
        if status == "none" and inner is not None and inner.cls != p.cls:
            status = "other_class"
        elif status == "none":
            # A touch by another class, under half the span, still counts.
            touched = human[(human.start < p.end) & (human.end > p.start)]
            if not touched.empty:
                status = "other_class"
        rows.append({
            **{k: getattr(p, k) for k in ("arm", "task_id", "start", "end", "cls", "text") if hasattr(p, k)},
            "status": status, "jaccard": round(best_j, 3),
            "human_cls": None if inner is None else inner.cls,
            "human_text": None if inner is None else inner.text,
        })
    return pd.DataFrame(rows)


def human_coverage(human: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """One row for each human span: the share of its characters that a
    same-class extractor span covers, and a status: hit (>= 0.5), touched
    (> 0), or missed."""
    n = int(max(human.end.max() if len(human) else 0, pred.end.max() if len(pred) else 0)) + 1
    pm = char_masks(pred, n)
    rows = []
    for h in human.itertuples():
        covered = float(pm[h.cls][h.start:h.end].mean()) if h.cls in pm and h.end > h.start else 0.0
        status = "hit" if covered >= 0.5 else ("touched" if covered > 0 else "missed")
        rows.append({
            **{k: getattr(h, k) for k in ("task_id", "annotation_id", "region_id", "start", "end", "cls", "text")
               if hasattr(h, k)},
            "covered": round(covered, 3), "status": status,
        })
    return pd.DataFrame(rows)


def confusion(led: pd.DataFrame) -> pd.DataFrame:
    """Innermost human class (rows) against the extractor class (columns),
    in extractor spans. A span under no human span sits in the `none` row."""
    if led.empty:
        return pd.DataFrame()
    t = led.assign(human_cls=led.human_cls.fillna("none"))
    table = pd.crosstab(t.human_cls, t.cls)
    order = [c for c in CLASSES if c in table.index] + (["none"] if "none" in table.index else [])
    return table.reindex(index=order, columns=[c for c in CLASSES if c in table.columns]).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------

def nested_pairs(human: pd.DataFrame) -> pd.DataFrame:
    """The (outer, inner) pairs of human spans in one annotation.

    A pair is nested when the inner span sits inside the outer and the outer
    is longer. A pair is co-labelled when 2 labels share the same bounds;
    it appears one time, with the classes in name order.
    """
    rows = []
    for (task_id, ann_id), part in human.groupby(["task_id", "annotation_id"], dropna=False):
        items = list(part.itertuples())
        for o in items:
            for i in items:
                if o.Index == i.Index or not (o.start <= i.start and i.end <= o.end):
                    continue
                same_bounds = (o.start, o.end) == (i.start, i.end)
                if same_bounds and not (o.cls < i.cls):
                    continue
                rows.append({
                    "task_id": task_id, "annotation_id": ann_id,
                    "outer_cls": o.cls, "inner_cls": i.cls,
                    "outer_start": o.start, "outer_end": o.end, "inner_start": i.start, "inner_end": i.end,
                    "outer_text": o.text, "inner_text": i.text,
                    "co_labelled": same_bounds,
                })
    return pd.DataFrame(rows, columns=["task_id", "annotation_id", "outer_cls", "inner_cls", "outer_start",
                                       "outer_end", "inner_start", "inner_end", "outer_text", "inner_text",
                                       "co_labelled"])


def nesting_by_arm(pairs: pd.DataFrame, pred: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """For each nested pair and each arm: did the extractor mark the outer
    span, the inner span, both, or neither? A side counts as marked when a
    same-class extractor span covers at least `min_share` of it."""
    if pairs.empty or pred.empty:
        return pd.DataFrame()
    rows = []
    for arm, ppart in pred.groupby("arm"):
        for pr in pairs.itertuples():
            tp = ppart[ppart.task_id == pr.task_id]

            def covered(start, end, cls):
                m = np.zeros(max(1, end - start), dtype=bool)
                for p in tp[tp.cls == cls].itertuples():
                    lo, hi = max(start, p.start), min(end, p.end)
                    if hi > lo:
                        m[lo - start:hi - start] = True
                return float(m.mean())

            outer = covered(pr.outer_start, pr.outer_end, pr.outer_cls) >= min_share
            inner = covered(pr.inner_start, pr.inner_end, pr.inner_cls) >= min_share
            took = "both" if outer and inner else "outer" if outer else "inner" if inner else "neither"
            rows.append({"arm": arm, "task_id": pr.task_id, "outer_cls": pr.outer_cls, "inner_cls": pr.inner_cls,
                         "co_labelled": pr.co_labelled, "took": took})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decisions and attributes
# ---------------------------------------------------------------------------

def decisions(human: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """For each task and arm: how many decisions the human marked, the last
    one, and whether the extractor's decision overlaps it."""
    rows = []
    for task_id, hpart in human.groupby("task_id"):
        hd = hpart[hpart.cls == "decision"].sort_values("start")
        last = hd.iloc[-1] if len(hd) else None
        for arm, ppart in pred[pred.task_id == task_id].groupby("arm"):
            pdn = ppart[ppart.cls == "decision"].sort_values("start")
            plast = pdn.iloc[-1] if len(pdn) else None
            hit = bool(last is not None and plast is not None
                       and _overlap(last.start, last.end, plast.start, plast.end) > 0)
            earlier = 0
            if len(hd) > 1:
                for h in hd.iloc[:-1].itertuples():
                    earlier += int(any(_overlap(h.start, h.end, p.start, p.end) > 0 for p in pdn.itertuples()))
            rows.append({
                "task_id": task_id, "arm": arm, "human_decisions": len(hd),
                "human_last": None if last is None else last.text,
                "extractor_decisions": len(pdn),
                "extractor_last": None if plast is None else plast.text,
                "last_matches": hit, "earlier_matched": earlier,
            })
    return pd.DataFrame(rows)


def attribute_agreement(led: pd.DataFrame, human: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """On the exact and inside matches: the share of attribute values the
    extractor and the human share, for each arm and attribute. Empty when the
    human set no attribute."""
    rows: Dict[Tuple[str, str], List[int]] = {}
    hits = led[led.status.isin(["exact", "inside", "contains"])]
    for r in hits.itertuples():
        p = pred[(pred.arm == r.arm) & (pred.task_id == r.task_id) & (pred.start == r.start)
                 & (pred.end == r.end) & (pred.cls == r.cls)]
        if p.empty:
            continue
        pattrs = p.iloc[0].attributes or {}
        h = innermost_human(human[(human.task_id == r.task_id) & (human.cls == r.cls)], r.start, r.end)
        if h is None:
            continue
        for name, hv in (h.attributes or {}).items():
            pv = pattrs.get(name)
            if pv is None:
                continue
            k = rows.setdefault((r.arm, name), [0, 0])
            k[1] += 1
            k[0] += int(str(pv).lower() == str(hv).lower())
    out = [{"arm": a, "attribute": n, "compared": v[1], "agreement": round(v[0] / v[1], 3) if v[1] else float("nan")}
           for (a, n), v in rows.items()]
    return pd.DataFrame(out, columns=["arm", "attribute", "compared", "agreement"])


# ---------------------------------------------------------------------------
# The whole picture
# ---------------------------------------------------------------------------

def compare(tasks: pd.DataFrame, human: pd.DataFrame, pred: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Run every measure over every (task, annotation, arm). Give the tables:
    counts, char_prf, ledger, coverage, nesting, decisions, attributes, summary."""
    chars = dict(zip(tasks.task_id, tasks.chars))
    prf_rows, led_frames, cov_frames = [], [], []
    for (task_id, ann_id), hpart in human.groupby(["task_id", "annotation_id"], dropna=False):
        n = int(chars.get(task_id, max(hpart.end.max(), 0))) + 1
        for arm, ppart in pred[pred.task_id == task_id].groupby("arm"):
            t = char_prf(hpart, ppart, n)
            t.insert(0, "arm", arm); t.insert(0, "annotation_id", ann_id); t.insert(0, "task_id", task_id)
            prf_rows.append(t)
            led = ledger(hpart, ppart)
            led["annotation_id"] = ann_id
            led_frames.append(led)
            cov = human_coverage(hpart, ppart)
            cov["arm"] = arm
            cov_frames.append(cov)
    prf = pd.concat(prf_rows, ignore_index=True) if prf_rows else pd.DataFrame()
    led = pd.concat(led_frames, ignore_index=True) if led_frames else pd.DataFrame()
    cov = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame()

    counts = pd.concat([
        human.groupby(["task_id", "annotation_id", "cls"]).size().rename("n").reset_index().assign(source="human"),
        pred.groupby(["task_id", "arm", "cls"]).size().rename("n").reset_index().rename(columns={"arm": "source"}),
    ], ignore_index=True)

    pairs = nested_pairs(human)
    nest = nesting_by_arm(pairs, pred)
    dec = decisions(human, pred)
    attrs = attribute_agreement(led, human, pred) if not led.empty else pd.DataFrame()

    summary_rows = []
    if not led.empty:
        for arm, lpart in led.groupby("arm"):
            cpart = cov[cov.arm == arm]
            ppart = prf[(prf.arm == arm) & (prf.cls == "all")]
            n_h = len(human)
            summary_rows.append({
                "arm": arm,
                "extractor": pred[pred.arm == arm].extractor.iloc[0],
                "spans_pred": len(lpart), "spans_human": n_h,
                "granularity": round(len(lpart) / n_h, 2) if n_h else float("nan"),
                "pred_same_class": round(lpart.status.isin(["exact", "inside", "contains", "partial"]).mean(), 3),
                "pred_exact": round((lpart.status == "exact").mean(), 3),
                "pred_inside": round((lpart.status == "inside").mean(), 3),
                "pred_other_class": round((lpart.status == "other_class").mean(), 3),
                "pred_none": round((lpart.status == "none").mean(), 3),
                "human_hit": round((cpart.status == "hit").mean(), 3) if len(cpart) else float("nan"),
                "human_touched": round((cpart.status != "missed").mean(), 3) if len(cpart) else float("nan"),
                "char_precision": round(ppart.precision.mean(), 3) if len(ppart) else float("nan"),
                "char_recall": round(ppart.recall.mean(), 3) if len(ppart) else float("nan"),
                "char_f1": round(ppart.f1.mean(), 3) if len(ppart) else float("nan"),
                "decision_last_matches": round(dec[dec.arm == arm].last_matches.mean(), 3)
                if not dec.empty and (dec.arm == arm).any() else float("nan"),
            })
    summary = pd.DataFrame(summary_rows).sort_values("char_f1", ascending=False) if summary_rows else pd.DataFrame()
    return {"counts": counts, "char_prf": prf, "ledger": led, "coverage": cov, "nested_pairs": pairs,
            "nesting": nest, "decisions": dec, "attributes": attrs, "summary": summary}


def class_table(prf: pd.DataFrame, value: str = "f1") -> pd.DataFrame:
    """Class (rows) by arm (columns), with the mean of `value` over the
    annotated tasks."""
    if prf.empty:
        return pd.DataFrame()
    t = prf.pivot_table(index="cls", columns="arm", values=value, aggfunc="mean")
    return t.reindex([c for c in list(CLASSES) + ["all"] if c in t.index]).round(3)


def caption_char_prf(prf: pd.DataFrame, arms: Sequence[str]) -> Dict[str, Any]:
    """The caption of `plot_char_prf`."""
    n_tasks = int(prf["task_id"].nunique()) if "task_id" in prf else 0
    n_ann = int(prf[["task_id", "annotation_id"]].drop_duplicates().shape[0]) \
        if {"task_id", "annotation_id"} <= set(prf.columns) else 0
    schemas = sorted({X.ARMS[a]["schema_version"] for a in arms if a in X.ARMS})
    caption = (
        "The character agreement of each extractor arm with the human gold "
        "spans, by class. Left: the character recall (the share of the "
        "human-marked characters of the class that the arm also marks with "
        "that class). Right: the character precision (the share of the "
        "arm's characters of the class that the human also marks). 1 bar "
        "for each arm, in the legend order; a bar is the mean over the "
        f"annotated traces ({n_tasks} traces, {n_ann} annotations). A "
        "missing class of an arm reads 0. This is the exact reading by "
        "character; the bounds of the human and of the extractor differ, "
        "and the positional alignment table gives the relaxed reading. "
        "Arms: "
        + ", ".join(f"`{a}`" for a in arms) + ".")
    return F.figure_meta(
        "Character agreement with the gold set", caption,
        "fig:gold-char-agreement",
        ["gold_alignment", *arms, *(schemas or ["v3"]), "char_recall",
         "char_precision"],
        module="_gold_alignment", version=__version__)


def plot_char_prf(prf: pd.DataFrame, arms: Sequence[str],
                  figsize: Tuple[float, float] = (7.2, 2.6)):
    """The character recall and precision of each arm, by class.

    The caption is `caption_char_prf`, attached to the figure.
    """
    import matplotlib.pyplot as plt

    if prf.empty:
        raise ValueError("no overlap to plot")
    arms = list(arms)
    classes = [c for c in CLASSES if c in set(prf.cls)]
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, value in zip(axes, ("recall", "precision")):
        t = class_table(prf, value).reindex(classes)
        x = np.arange(len(classes))
        w = 0.8 / max(1, len(arms))
        for i, arm in enumerate(arms):
            if arm not in t.columns:
                continue
            ax.bar(x + i * w - 0.4 + w / 2, t[arm].fillna(0).values, width=w,
                   label=f"{arm}", edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title(f"character {value}")
    axes[0].set_ylabel("share of characters")
    axes[1].legend(frameon=False, fontsize=5, ncol=2)
    fig.tight_layout()
    return F.attach(fig, caption_char_prf(prf, arms))


# ---------------------------------------------------------------------------
# A view of one trace
# ---------------------------------------------------------------------------

def _rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_trace(text: str, human: pd.DataFrame, pred: pd.DataFrame) -> str:
    """HTML of the trace. The background is the innermost human class; a
    second, outer human class shows as a left border of its colour. An
    extractor span is an underline in the colour of its class.

    The marks come from runs of characters with the same styling, thus a
    nested or a co-labelled human span renders without broken tags.
    """
    n = len(text)
    depth = np.zeros(n, dtype=int)
    inner = [None] * n
    outer = [None] * n
    # Longest first, thus the shortest wins the background.
    ordered = human.assign(length=human.end - human.start).sort_values("length", ascending=False)
    for h in ordered.itertuples():
        for i in range(max(0, h.start), min(n, h.end)):
            if inner[i] is not None and inner[i] != h.cls:
                outer[i] = inner[i]
            inner[i] = h.cls
            depth[i] += 1
    pcls = [None] * n
    for p in pred.itertuples():
        for i in range(max(0, p.start), min(n, p.end)):
            pcls[i] = p.cls
    parts = []
    i = 0
    while i < n:
        j = i
        key = (inner[i], outer[i], pcls[i])
        while j < n and (inner[j], outer[j], pcls[j]) == key:
            j += 1
        chunk = escape(text[i:j]).replace("\n", "<br>")
        style = []
        if key[0]:
            style.append(f"background:{_rgba(CLASS_COLOURS[key[0]], 0.28)}")
        if key[1]:
            style.append(f"border-left:3px solid {CLASS_COLOURS[key[1]]};padding-left:1px")
        if key[2]:
            style.append(f"border-bottom:3px solid {CLASS_COLOURS[key[2]]}")
        parts.append(f'<span style="{";".join(style)}">{chunk}</span>' if style else chunk)
        i = j
    legend = " ".join(
        f'<span style="background:{_rgba(c, 0.28)};padding:0 4px;margin-right:6px">{k}</span>'
        for k, c in CLASS_COLOURS.items())
    return (f'<div style="font-family:serif;font-size:14px;line-height:1.7;max-width:900px">'
            f'<p style="font-size:12px">Background: the human, innermost class. Left border: an outer human class. '
            f'Underline: the extractor.</p><p style="font-size:12px">{legend}</p>{"".join(parts)}</div>')


def span_ledger_view(human: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    """The human and the extractor spans of one task in text order, side by
    side: one row for each span, with its source and class."""
    h = human[["start", "end", "cls", "text"]].assign(source="human")
    cols = ["start", "end", "cls", "text", "arm"]
    p = pred[cols].rename(columns={"arm": "source"})
    return pd.concat([h, p], ignore_index=True).sort_values(["start", "end", "source"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Positional alignment: the relaxed reading (2026-09-28)
# ---------------------------------------------------------------------------
#
# The human marks a clause; the extractor quotes a phrase. The exact bounds
# differ, and so does the scaffold the judge prints at the head of a row:
# "Image A", "Image B", a bold header ("**Vibe:**"), a list marker. The
# relaxed reading masks that scaffold on both sides and then asks 1 question
# of each span: does a span of the same class on the other side cover at
# least `min_share` of the SHORTER span's content characters? A phrase inside
# a clause counts, and a clause around a phrase counts.

import re

# What counts as scaffold: the image names, a bold header that ends with a
# colon, the bold marks themselves, a list marker at the head of a line, and
# white space. A bold place name ("**Sutton Park**") keeps its letters.
_SCAFFOLD_PATTERNS = [
    re.compile(r"\*\*[^*\n]{1,60}:\*\*"),          # **Image A Analysis:**, **Vibe:**
    re.compile(r"\*\*"),                              # the bold marks alone
    re.compile(r"\bImage [AB]\b\s*[:,.]?"),          # Image A, Image B:
    re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+"),  # a list marker at the head of a line
    re.compile(r"\s+"),
]


def scaffold_mask(text: str) -> np.ndarray:
    """True at each character that is scaffold, not content."""
    mask = np.zeros(len(text), dtype=bool)
    for pat in _SCAFFOLD_PATTERNS:
        for m in pat.finditer(text):
            mask[m.start():m.end()] = True
    return mask


def _content(mask: np.ndarray, start: int, end: int) -> np.ndarray:
    """The content characters of [start, end), as indices."""
    lo, hi = max(0, start), min(len(mask), end)
    if hi <= lo:
        return np.zeros(0, dtype=int)
    idx = np.arange(lo, hi)
    return idx[~mask[lo:hi]]


def positional_match(human: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray,
                     min_share: float = 0.5, same_class: bool = True
                     ) -> Tuple[List[Tuple[int, int, float]], pd.Series, pd.Series]:
    """Match the spans by position, 1 to 1, best share first.

    A pair qualifies when the content characters they share cover at least
    `min_share` of the SHORTER span's content. With `same_class`, the classes
    must agree. Give the pairs as (human index, pred index, share), and 2
    boolean Series: the human spans that matched, the pred spans that matched.
    """
    h_sets = {i: set(_content(mask, r.start, r.end).tolist()) for i, r in human.iterrows()}
    p_sets = {j: set(_content(mask, r.start, r.end).tolist()) for j, r in pred.iterrows()}
    cands = []
    for i, hr in human.iterrows():
        hs = h_sets[i]
        if not hs:
            continue
        for j, pr in pred.iterrows():
            if same_class and hr.cls != pr.cls:
                continue
            ps = p_sets[j]
            if not ps:
                continue
            inter = len(hs & ps)
            if inter == 0:
                continue
            share = inter / min(len(hs), len(ps))
            if share >= min_share:
                cands.append((share, i, j))
    cands.sort(key=lambda t: -t[0])
    used_h, used_p, pairs = set(), set(), []
    for share, i, j in cands:
        if i in used_h or j in used_p:
            continue
        used_h.add(i); used_p.add(j); pairs.append((i, j, share))
    return (pairs,
            pd.Series([i in used_h for i in human.index], index=human.index, dtype=bool),
            pd.Series([j in used_p for j in pred.index], index=pred.index, dtype=bool))


def positional_prf(human: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray,
                   min_share: float = 0.5) -> pd.DataFrame:
    """Span precision, recall, and F1 under the positional rule, for each
    class, in all (`all`, same class required), and with the class ignored
    (`any_class`). The gap between `all` and `any_class` is the class
    disagreement alone; the gap under `any_class` is the boundary alone."""
    rows = []
    for c in CLASSES:
        h, p = human[human.cls == c], pred[pred.cls == c]
        pairs, _, _ = positional_match(h, p, mask, min_share, same_class=True)
        rows.append({"cls": c, "n_human": len(h), "n_pred": len(p), "matched": len(pairs),
                     **_prf(len(pairs), len(p), len(h))})
    for name, same in (("all", True), ("any_class", False)):
        pairs, _, _ = positional_match(human, pred, mask, min_share, same_class=same)
        rows.append({"cls": name, "n_human": len(human), "n_pred": len(pred), "matched": len(pairs),
                     **_prf(len(pairs), len(pred), len(human))})
    return pd.DataFrame(rows)


def content_char_prf(human: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """`char_prf` over the content characters alone: the scaffold is out of
    both the numerator and the denominator."""
    n = len(mask)
    hm, pm = char_masks(human, n), char_masks(pred, n)
    keep = ~mask
    rows, tp_all, np_all, ng_all = [], 0, 0, 0
    for c in CLASSES:
        tp = int((hm[c] & pm[c] & keep).sum())
        n_p, n_g = int((pm[c] & keep).sum()), int((hm[c] & keep).sum())
        tp_all += tp; np_all += n_p; ng_all += n_g
        rows.append({"cls": c, "chars_human": n_g, "chars_pred": n_p, "chars_both": tp, **_prf(tp, n_p, n_g)})
    rows.append({"cls": "all", "chars_human": ng_all, "chars_pred": np_all, "chars_both": tp_all,
                 **_prf(tp_all, np_all, ng_all)})
    return pd.DataFrame(rows)


def compare_positional(tasks: pd.DataFrame, human: pd.DataFrame, pred: pd.DataFrame,
                       min_share: float = 0.5) -> Dict[str, pd.DataFrame]:
    """The positional reading over every (task, annotation, arm).

    Give `by_task` (1 row for each task, annotation, arm, class), `by_arm`
    (micro over the tasks: the counts sum, then P/R/F1), `by_extractor`
    (the 2 arms of 1 extractor pooled over both judges), and `content_chars`
    (the character reading on the content characters, by arm and class).
    """
    texts = dict(zip(tasks.task_id, tasks.text))
    masks = {tid: scaffold_mask(t) for tid, t in texts.items()}
    frames, cframes = [], []
    for (task_id, ann_id), hpart in human.groupby(["task_id", "annotation_id"], dropna=False):
        if task_id not in masks:
            continue
        for arm, ppart in pred[pred.task_id == task_id].groupby("arm"):
            t = positional_prf(hpart, ppart, masks[task_id], min_share)
            t.insert(0, "arm", arm); t.insert(0, "annotation_id", ann_id); t.insert(0, "task_id", task_id)
            t["extractor"] = ppart.extractor.iloc[0]
            t["judge"] = tasks.set_index("task_id").loc[task_id, "judge"]
            frames.append(t)
            c = content_char_prf(hpart, ppart, masks[task_id])
            c.insert(0, "arm", arm); c.insert(0, "task_id", task_id)
            c["extractor"] = ppart.extractor.iloc[0]
            cframes.append(c)
    by_task = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    content = pd.concat(cframes, ignore_index=True) if cframes else pd.DataFrame()

    def micro(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        g = df.groupby(keys + ["cls"], sort=False)[["n_human", "n_pred", "matched"]].sum().reset_index()
        g["tasks"] = df.groupby(keys + ["cls"], sort=False).task_id.nunique().values
        prf = g.apply(lambda r: pd.Series(_prf(r.matched, r.n_pred, r.n_human)), axis=1)
        return pd.concat([g, prf.round(3)], axis=1)

    def micro_chars(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        g = df.groupby(keys + ["cls"], sort=False)[["chars_human", "chars_pred", "chars_both"]].sum().reset_index()
        prf = g.apply(lambda r: pd.Series(_prf(r.chars_both, r.chars_pred, r.chars_human)), axis=1)
        return pd.concat([g, prf.round(3)], axis=1)

    if by_task.empty:
        return {"by_task": by_task, "by_arm": pd.DataFrame(), "by_extractor": pd.DataFrame(),
                "content_chars": pd.DataFrame(), "content_chars_by_extractor": pd.DataFrame()}
    return {
        "by_task": by_task,
        "by_arm": micro(by_task, ["arm", "extractor", "judge"]),
        "by_extractor": micro(by_task, ["extractor"]),
        "content_chars": micro_chars(content, ["arm", "extractor"]),
        "content_chars_by_extractor": micro_chars(content, ["extractor"]),
    }


def positional_table(by: pd.DataFrame, key: str = "arm", value: str = "f1") -> pd.DataFrame:
    """Class (rows) by `key` (columns), the `value` of the positional reading."""
    if by.empty:
        return pd.DataFrame()
    t = by.pivot_table(index="cls", columns=key, values=value)
    order = [c for c in list(CLASSES) + ["all", "any_class"] if c in t.index]
    return t.reindex(order).round(3)


# ---------------------------------------------------------------------------
# The 2 x 2: agreement between the extractors of 1 judge (2026-09-30)
# ---------------------------------------------------------------------------
#
# Every arm of 1 judge reads the same 1,000 traces of each case (the seeded
# sample). Thus the 2 extractors of a judge can be read against each other
# on those traces with the same positional rule as the human alignment: one
# arm plays the reference, the other the prediction. A pair of arms that
# agree on a class produce a rate that is about the traces; a pair that
# disagree produce a rate that is about the extractor.


def arm_profile(arms: Optional[Sequence[str]] = None, require_canonical: bool = True) -> pd.DataFrame:
    """One row for each arm: what its corpus holds after the paper filters.

    `spans_per_trace`, the share of each class, the decision share, and the
    perspectives for each trace. A rung of the ladder in 1 line.
    """
    rows = []
    for name, spec in X.arm_roots(arms, present_only=True).items():
        df = X.load(root=spec["root"], require_canonical=require_canonical)
        if df.empty:
            continue
        traces = df.groupby(["case", "pair_id"]).ngroups
        counts = df.extraction_class.value_counts()
        row = {"arm": name, "judge": spec["judge"], "extractor": spec["extractor"],
               "schema_version": spec["schema_version"], "traces": traces, "rows": len(df),
               "spans_per_trace": round(len(df) / traces, 1),
               "decision_share": round(df[df.extraction_class == "decision"].groupby(["case", "pair_id"]).ngroups / traces, 3),
               "perspectives_per_trace": round(counts.get("perspective", 0) / traces, 2),
               "verbatim": round(float(df.is_verbatim.mean()), 3) if "is_verbatim" in df else None}
        for c in CLASSES:
            if c in counts.index:
                row[f"share_{c}"] = round(counts[c] / len(df), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def shared_traces(a: pd.DataFrame, b: pd.DataFrame, traces_per_case: int, seed: int = 20260930) -> pd.DataFrame:
    """A seeded draw of (case, pair_id) that both frames hold."""
    ka = a[["case", "pair_id"]].drop_duplicates()
    kb = b[["case", "pair_id"]].drop_duplicates()
    both = ka.merge(kb, on=["case", "pair_id"])
    parts = [g.sample(n=min(traces_per_case, len(g)), random_state=seed)
             for _, g in both.groupby("case", sort=True)]
    return pd.concat(parts, ignore_index=True) if parts else both


def _spans_of(df: pd.DataFrame, case: str, pair_id: str) -> pd.DataFrame:
    part = df[(df.case == case) & (df.pair_id == pair_id)]
    part = part[part.char_start.notna() & part.char_end.notna()]
    out = pd.DataFrame({
        "start": part.char_start.astype(int).values, "end": part.char_end.astype(int).values,
        "cls": part.extraction_class.values, "text": part.extraction_text.astype(str).values,
    })
    out["region_id"] = [f"r{i}" for i in range(len(out))]
    out["attributes"] = [{} for _ in range(len(out))]
    return out


def compare_arm_pairs(judge_arms: Dict[str, Sequence[str]], traces_per_case: int = 20,
                      min_share: float = 0.5, require_canonical: bool = True,
                      texts: Optional[Dict[Tuple[str, str], str]] = None) -> Dict[str, pd.DataFrame]:
    """Read the 2 extractors of each judge against each other.

    Args:
        judge_arms: judge -> (reference arm, other arm). The first arm is
            the reference: its spans are the "human" side of the positional
            rule, and precision reads from the other arm.
        traces_per_case: a seeded draw of shared traces for each case.
        texts: (case, pair_id) -> the trace text, for the scaffold mask.
            Without it the mask is empty and the rule reads every character.

    Give `by_case` (1 row for each judge, case, class), `by_judge` (micro
    over the cases), and `chars` (the content-character reading).
    """
    rows, crows = [], []
    for judge, (ref_arm, oth_arm) in judge_arms.items():
        ref = X.load(root=X.arm_roots([ref_arm])[ref_arm]["root"], require_canonical=require_canonical)
        oth = X.load(root=X.arm_roots([oth_arm])[oth_arm]["root"], require_canonical=require_canonical)
        keys = shared_traces(ref, oth, traces_per_case)
        for k in keys.itertuples():
            h, p = _spans_of(ref, k.case, k.pair_id), _spans_of(oth, k.case, k.pair_id)
            n = int(max(h.end.max() if len(h) else 0, p.end.max() if len(p) else 0)) + 1
            text = (texts or {}).get((k.case, k.pair_id))
            mask = scaffold_mask(text) if text else np.zeros(n, dtype=bool)
            if len(mask) < n:
                mask = np.concatenate([mask, np.zeros(n - len(mask), dtype=bool)])
            t = positional_prf(h, p, mask, min_share)
            t.insert(0, "pair_id", k.pair_id); t.insert(0, "case", k.case); t.insert(0, "judge", judge)
            t["reference"], t["other"] = ref_arm, oth_arm
            rows.append(t)
            c = content_char_prf(h, p, mask)
            c.insert(0, "pair_id", k.pair_id); c.insert(0, "case", k.case); c.insert(0, "judge", judge)
            crows.append(c)
    if not rows:
        return {"by_case": pd.DataFrame(), "by_judge": pd.DataFrame(), "chars": pd.DataFrame()}
    long = pd.concat(rows, ignore_index=True)
    chars = pd.concat(crows, ignore_index=True)

    def micro(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        g = df.groupby(keys + ["cls"], sort=False)[["n_human", "n_pred", "matched"]].sum().reset_index()
        g = g.rename(columns={"n_human": "n_reference", "n_pred": "n_other"})
        prf = g.apply(lambda r: pd.Series(_prf(r.matched, r.n_other, r.n_reference)), axis=1)
        return pd.concat([g, prf.round(3)], axis=1)

    def micro_chars(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        g = df.groupby(keys + ["cls"], sort=False)[["chars_human", "chars_pred", "chars_both"]].sum().reset_index()
        prf = g.apply(lambda r: pd.Series(_prf(r.chars_both, r.chars_pred, r.chars_human)), axis=1)
        return pd.concat([g, prf.round(3)], axis=1)

    # A `pair_id` occurs in more than 1 case, thus the count reads the
    # (case, pair_id) key.
    traces = long.drop_duplicates(["judge", "case", "pair_id"]).groupby("judge").size()
    return {"by_case": micro(long, ["judge", "case"]), "by_judge": micro(long, ["judge"]),
            "chars": micro_chars(chars, ["judge"]), "traces": traces.to_dict()}


def agreement_table(by: pd.DataFrame, value: str = "f1") -> pd.DataFrame:
    """Class (rows) by judge (columns)."""
    if by.empty:
        return pd.DataFrame()
    t = by.pivot_table(index="cls", columns="judge", values=value)
    order = [c for c in list(CLASSES) + ["all", "any_class"] if c in t.index]
    return t.reindex(order).round(3)


# ---------------------------------------------------------------------------
# The paper tables (LaTeX)
# ---------------------------------------------------------------------------
#
# 2 tables carry the result of the gold set. The first reads each extractor
# against the human, the second reads the 2 extractors of a judge against
# each other. Both use the positional rule and the house format of `_latex`.

POSITIONAL_LABEL = "tab:gold-positional-alignment"
AGREEMENT_LABEL = "tab:gold-extractor-agreement"

POSITIONAL_CAPTION = (
    "\\textbf{How well do the extractors line up with a human?} Span F1 of "
    "each extractor against 1 annotator on {tasks} gold traces, by class. "
    "A span matches a span of the same class when their shared content "
    "covers half of the shorter span; \\emph{any class} ignores the class "
    "and reads the boundary alone."
)
AGREEMENT_CAPTION = (
    "\\textbf{Do the 2 extractors of a judge agree?} Span agreement on "
    "{traces} shared traces of each judge, by class, with the same "
    "positional rule. The primary extractor is the reference: P reads the "
    "spans of the control, R the spans of the primary."
)


def _class_tex(cls: str) -> str:
    """`visual_evidence` -> `visual evidence`; `all` -> `all classes`."""
    return {"all": "all classes", "any_class": "any class"}.get(cls, L.escape(str(cls).replace("_", " ")))


def _class_rows(present: Iterable[str]) -> Tuple[List[str], List[str]]:
    """The class rows in `CLASSES` order, and the 2 summary rows.

    `present` holds the classes with enough spans to print.
    """
    present = set(present)
    return ([c for c in CLASSES if c in present],
            [c for c in ("all", "any_class") if c in present])


# The extractors in column order. The 2 large extractors come first: the
# paper reads them, and the small arms are a check.
EXTRACTOR_ORDER = ("qwen3.8-27b", "gemma-4-31b", "qwen3.5-9b", "gemma-4-12b")
LARGE_EXTRACTORS = ("qwen3.8-27b", "gemma-4-31b")


def _ordered_arms(by_arm: pd.DataFrame, arms: Optional[Sequence[str]] = None) -> List[str]:
    """The arms by judge (gemma first), by `EXTRACTOR_ORDER` in each judge.

    `arms=None` keeps the arms of the 2 large extractors (`LARGE_EXTRACTORS`).
    """
    names = list(dict.fromkeys(by_arm["arm"])) if arms is None else list(arms)
    if arms is None:
        names = [a for a in names if X.ARMS.get(a, {}).get("extractor") in LARGE_EXTRACTORS]
    judges = [j for j in ("gemma-4-12b", "qwen3.5-9b")
              if any(X.ARMS.get(a, {}).get("judge") == j for a in names)]
    judges += sorted({X.ARMS.get(a, {}).get("judge", "") for a in names} - set(judges))
    rank = {e: i for i, e in enumerate(EXTRACTOR_ORDER)}
    out = []
    for j in judges:
        mine = [a for a in names if X.ARMS.get(a, {}).get("judge", "") == j]
        out += sorted(mine, key=lambda a: (rank.get(X.ARMS[a]["extractor"], 99), names.index(a)))
    return out


def latex_positional_alignment(by_arm: pd.DataFrame, arms: Optional[Sequence[str]] = None,
                               value: str = "f1", n_tasks: Optional[int] = None,
                               min_spans: int = 10, caption: Optional[str] = None,
                               label: str = POSITIONAL_LABEL) -> str:
    """The extractor-against-human table: class (rows) by arm (columns).

    The arms group by judge under a `\\multicolumn` header, and each column
    names the extractor. 2 count rows close the table: the spans of the
    extractor and the spans of the human, on the traces of that judge.

    Args:
        by_arm: `compare_positional(...)["by_arm"]`.
        arms: The arms to print. `None` prints every arm of `by_arm`.
        value: The measure of a cell: `f1`, `precision`, or `recall`.
        n_tasks: The number of annotated traces, for the caption. `None`
            reads the largest `tasks` of the `all` rows of each judge.
        min_spans: A class row prints when the human marks this many spans
            of the class on the traces of 1 judge or more.
    """
    if by_arm is None or by_arm.empty:
        return "% no positional alignment\n"
    order = [a for a in _ordered_arms(by_arm, arms) if a in set(by_arm["arm"])]
    by_judge: Dict[str, List[str]] = {}
    for a in order:
        by_judge.setdefault(X.ARMS[a]["judge"], []).append(a)
    if n_tasks is None:
        alls = by_arm[(by_arm["cls"] == "all") & by_arm["arm"].isin(order)]
        n_tasks = int(alls.groupby("judge")["tasks"].max().sum()) if "tasks" in alls else 0
    shown = by_arm[by_arm["arm"].isin(order)]
    cell = shown.set_index(["arm", "cls"])
    big = shown.groupby("cls")["n_human"].max()
    classes, summary = _class_rows(big[(big >= min_spans) | big.index.isin(["all", "any_class"])].index)

    def line(cls: str) -> str:
        return L.row([_class_tex(cls)] + [
            L.fmt(cell.loc[(a, cls), value]) if (a, cls) in cell.index else L.DASH
            for a in order])

    def count(col: str) -> List[str]:
        return [L.fmt_int(cell.loc[(a, "all"), col]) if (a, "all") in cell.index else L.DASH
                for a in order]

    head = L.group_header([(f"{L.model_label(j)} traces", len(v)) for j, v in by_judge.items()])
    head.append(L.row(["Class"] + [L.model_label(X.ARMS[a]["extractor"]) for a in order]))
    body = [line(c) for c in classes] + ["\\midrule"] + [line(c) for c in summary]
    body += ["\\midrule", L.row(["extractor spans"] + count("n_pred")),
             L.row(["human spans"] + count("n_human"))]
    text = caption if caption is not None else POSITIONAL_CAPTION.replace("{tasks}", str(n_tasks))
    if value != "f1":
        text = text.replace("Span F1", f"Span {value}")
    # 4 extractor names do not fit 1 column at the body size, thus the box
    # scales the table to the column width.
    return L.table("_gold_alignment", __version__, text, label,
                   "l" + "r" * len(order), head, body, resize=True)


def latex_extractor_agreement(by_judge: pd.DataFrame,
                              traces: Optional[Dict[str, int]] = None,
                              pairs: Optional[Dict[str, Sequence[str]]] = None,
                              min_spans: int = 10, caption: Optional[str] = None,
                              label: str = AGREEMENT_LABEL) -> str:
    """The 2 x 2 agreement table: class (rows) by judge, with F1, P, and R.

    Args:
        by_judge: `compare_arm_pairs(...)["by_judge"]`.
        traces: `compare_arm_pairs(...)["traces"]`, judge -> the shared
            traces read. The caption states the range.
        pairs: judge -> (reference arm, other arm), as given to
            `compare_arm_pairs`. The caption names the reference extractor
            of each judge. `None` reads `X.PRIMARY_ARMS`.
        min_spans: A class row prints when the 2 arms of 1 judge or more
            hold this many spans of the class together.
    """
    if by_judge is None or by_judge.empty:
        return "% no extractor agreement\n"
    judges = [j for j in ("gemma-4-12b", "qwen3.5-9b") if j in set(by_judge["judge"])]
    judges += sorted(set(by_judge["judge"]) - set(judges))
    cell = by_judge.set_index(["judge", "cls"])
    big = (by_judge["n_reference"] + by_judge["n_other"]).groupby(by_judge["cls"]).max()
    classes, summary = _class_rows(big[(big >= min_spans) | big.index.isin(["all", "any_class"])].index)

    def line(cls: str) -> str:
        cells = []
        for j in judges:
            for col in ("f1", "precision", "recall"):
                cells.append(L.fmt(cell.loc[(j, cls), col]) if (j, cls) in cell.index else L.DASH)
        return L.row([_class_tex(cls)] + cells)

    head = L.group_header([(f"{L.model_label(j)} traces", 3) for j in judges])
    head.append(L.row(["Class"] + ["F1", "P", "R"] * len(judges)))
    body = [line(c) for c in classes] + ["\\midrule"] + [line(c) for c in summary]
    vals = sorted(set(int(v) for v in (traces or {}).values()))
    n_txt = ("--".join(L.fmt_int(v) for v in (vals[0], vals[-1])) if len(vals) > 1
             else (L.fmt_int(vals[0]) if vals else "the"))
    text = caption if caption is not None else AGREEMENT_CAPTION.replace("{traces}", n_txt)
    refs = [(j, (pairs or {}).get(j, (X.PRIMARY_ARMS.get(j),))[0]) for j in judges]
    refs = [f"{L.model_label(X.ARMS[a]['extractor'])} on the {L.model_label(j)} traces"
            for j, a in refs if a in X.ARMS]
    if caption is None and refs:
        text += " The reference is " + " and ".join(refs) + "."
    return L.table("_gold_alignment", __version__, text, label,
                   "l" + "rrr" * len(judges), head, body, resize=True)
