#!/usr/bin/env python
"""Score the extractors and the annotators against the gold set.

Input: the Label Studio export of the gold-set project as JSON (Export >
JSON, or `GET /api/projects/<id>/export?exportType=JSON`), and the prediction
files that `scripts/build_gold_set.py` wrote for each extractor arm.

What it measures
----------------
1. Extractor against human, for each class: precision, recall, and F1 of the
   spans. A predicted span matches a human span when they share the class and
   their character ranges overlap by at least `--overlap` (Jaccard, default
   0.5). Each human span matches at most 1 predicted span.
2. Attribute agreement on the matched spans: the share where the extractor's
   value equals the human's, for each attribute.
3. Human against human, for each class: the same span F1 between the 2
   annotators of a task, and the attribute agreement on their matched spans.
   This is the ceiling: an extractor cannot be held to more than 2 people
   agree on.

When a task holds 2 annotations, the extractor is scored against each and
the mean is reported; the union of the 2 is not used, because it would count a
span 1 person rejected.

Use:
    python scripts/score_gold_set.py data/gold_set/export.json
    python scripts/score_gold_set.py export.json --overlap 0.5 --out data/gold_set/scores

Output: `scores_by_class.csv`, `scores_attributes.csv`, `agreement_humans.csv`,
and a short summary on stdout.

See `vlm-narratives-docs/langextract-trace-extraction.md`, "The gold set".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ROOT = "/share/ju/matt/mllmsci"
GOLD = os.path.join(ROOT, "data", "gold_set")

__version__ = "1.0.0"

CLASSES = ("visual_evidence", "inference", "person_reference", "image_artifact",
           "comparison", "uncertainty", "decision")


# ---------------------------------------------------------------------------
# Reading spans out of Label Studio results
# ---------------------------------------------------------------------------

def spans_from_results(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn a Label Studio result list into spans with their attributes.

    A `labels` result gives the span; the `choices` results with the same
    region id (or the same start and end) give its attributes.

    A region with 2 labels gives 2 spans with the same bounds and id: the
    annotator marks a co-labelled span that way (an `inference` that is also
    a `comparison`). Nested spans are separate regions and pass through as
    they stand.
    """
    spans: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for r in results:
        if r.get("type") != "labels":
            continue
        v = r["value"]
        start, end = int(v["start"]), int(v["end"])
        for label in v.get("labels") or []:
            spans[(start, end, label)] = {"start": start, "end": end, "class": label,
                                          "text": v.get("text", ""), "id": r.get("id"), "attrs": {}}
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in spans.values():
        if s.get("id"):
            by_id[s["id"]].append(s)
    for r in results:
        if r.get("type") != "choices":
            continue
        v = r["value"]
        choices = v.get("choices") or []
        if not choices:
            continue
        targets = by_id.get(r.get("id")) or []
        if not targets and "start" in v and "end" in v:
            targets = [s for (a, b, _), s in spans.items() if (a, b) == (int(v["start"]), int(v["end"]))]
        for target in targets:
            target["attrs"][r["from_name"]] = choices[0]
    return list(spans.values())


def jaccard(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union else 0.0


def match_spans(pred: List[Dict[str, Any]], gold: List[Dict[str, Any]],
                overlap: float) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Greedy 1-to-1 matching by class and overlap, best overlap first."""
    pairs = []
    for p in pred:
        for g in gold:
            if p["class"] != g["class"]:
                continue
            j = jaccard(p, g)
            if j >= overlap:
                pairs.append((j, id(p), id(g), p, g))
    pairs.sort(key=lambda t: -t[0])
    used_p, used_g, out = set(), set(), []
    for _, ip, ig, p, g in pairs:
        if ip in used_p or ig in used_g:
            continue
        used_p.add(ip); used_g.add(ig); out.append((p, g))
    return out


def prf(pred: List[Dict[str, Any]], gold: List[Dict[str, Any]], overlap: float,
        cls: Optional[str] = None) -> Dict[str, float]:
    if cls:
        pred = [p for p in pred if p["class"] == cls]
        gold = [g for g in gold if g["class"] == cls]
    matched = match_spans(pred, gold, overlap)
    tp = len(matched)
    precision = tp / len(pred) if pred else float("nan")
    recall = tp / len(gold) if gold else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (pred and gold and tp) else (
        0.0 if (pred or gold) else float("nan"))
    return {"n_pred": len(pred), "n_gold": len(gold), "tp": tp,
            "precision": precision, "recall": recall, "f1": f1}


def attribute_agreement(matched: List[Tuple[Dict[str, Any], Dict[str, Any]]]
                        ) -> Dict[str, Tuple[int, int]]:
    """For each attribute name: (agreements, comparisons) over matched spans."""
    out: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for p, g in matched:
        for name, gv in g["attrs"].items():
            pv = p["attrs"].get(name)
            if pv is None:
                continue
            out[name][1] += 1
            out[name][0] += int(str(pv).lower() == str(gv).lower())
    return {k: (v[0], v[1]) for k, v in out.items()}


# ---------------------------------------------------------------------------
# The export and the predictions
# ---------------------------------------------------------------------------

def load_export(path: str) -> Dict[str, List[List[Dict[str, Any]]]]:
    """task_id -> list of annotations, each a list of spans."""
    with open(path) as fh:
        tasks = json.load(fh)
    out: Dict[str, List[List[Dict[str, Any]]]] = {}
    for t in tasks:
        tid = t["data"].get("task_id") or str(t["id"])
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        out[tid] = [spans_from_results(a.get("result", [])) for a in anns]
    return out


def load_predictions(gold_dir: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """arm -> task_id -> spans."""
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for path in sorted(glob.glob(os.path.join(gold_dir, "predictions_*.json"))):
        arm = os.path.basename(path)[len("predictions_"):-len(".json")]
        with open(path) as fh:
            preds = json.load(fh)
        out[arm] = {p["task_id"]: spans_from_results(p["result"]) for p in preds}
    return out


def score(export_path: str, gold_dir: str, overlap: float, out_dir: str) -> None:
    export = load_export(export_path)
    predictions = load_predictions(gold_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows, attr_rows, human_rows = [], [], []
    for arm, by_task in predictions.items():
        agg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        attr_agg: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        n_tasks = 0
        for tid, anns in export.items():
            if not anns or tid not in by_task:
                continue
            n_tasks += 1
            pred = by_task[tid]
            for gold in anns:
                w = 1.0 / len(anns)
                for cls in CLASSES + ("all",):
                    r = prf(pred, gold, overlap, None if cls == "all" else cls)
                    for k in ("n_pred", "n_gold", "tp"):
                        agg[cls][k] += w * r[k]
                for name, (a, n) in attribute_agreement(match_spans(pred, gold, overlap)).items():
                    attr_agg[name][0] += a * w
                    attr_agg[name][1] += n * w
        for cls, c in agg.items():
            p = c["tp"] / c["n_pred"] if c["n_pred"] else float("nan")
            r = c["tp"] / c["n_gold"] if c["n_gold"] else float("nan")
            f = 2 * p * r / (p + r) if (c["n_pred"] and c["n_gold"] and c["tp"]) else 0.0
            rows.append({"arm": arm, "class": cls, "tasks": n_tasks, "n_pred": round(c["n_pred"], 1),
                         "n_gold": round(c["n_gold"], 1), "tp": round(c["tp"], 1),
                         "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)})
        for name, (a, n) in attr_agg.items():
            attr_rows.append({"arm": arm, "attribute": name, "compared": round(n, 1),
                              "agreement": round(a / n, 3) if n else float("nan")})

    # Human against human, on the tasks with 2 annotations.
    hagg: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hattr: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    n_double = 0
    for tid, anns in export.items():
        if len(anns) < 2:
            continue
        n_double += 1
        a, b = anns[0], anns[1]
        for cls in CLASSES + ("all",):
            r = prf(a, b, overlap, None if cls == "all" else cls)
            for k in ("n_pred", "n_gold", "tp"):
                hagg[cls][k] += r[k]
        for name, (ag, n) in attribute_agreement(match_spans(a, b, overlap)).items():
            hattr[name][0] += ag; hattr[name][1] += n
    for cls, c in hagg.items():
        p = c["tp"] / c["n_pred"] if c["n_pred"] else float("nan")
        r = c["tp"] / c["n_gold"] if c["n_gold"] else float("nan")
        f = 2 * p * r / (p + r) if (c["n_pred"] and c["n_gold"] and c["tp"]) else 0.0
        human_rows.append({"class": cls, "tasks_with_2": n_double, "spans_a": int(c["n_pred"]),
                           "spans_b": int(c["n_gold"]), "matched": int(c["tp"]), "f1": round(f, 3)})
    for name, (ag, n) in hattr.items():
        human_rows.append({"class": f"attr:{name}", "tasks_with_2": n_double, "spans_a": n,
                           "spans_b": n, "matched": ag, "f1": round(ag / n, 3) if n else float("nan")})

    by_class = pd.DataFrame(rows)
    attrs = pd.DataFrame(attr_rows)
    humans = pd.DataFrame(human_rows)
    by_class.to_csv(os.path.join(out_dir, "scores_by_class.csv"), index=False)
    attrs.to_csv(os.path.join(out_dir, "scores_attributes.csv"), index=False)
    humans.to_csv(os.path.join(out_dir, "agreement_humans.csv"), index=False)

    pd.set_option("display.width", 200)
    n_annotated = sum(1 for a in export.values() if a)
    print(f"[score] {len(export)} tasks in the export, {n_annotated} annotated, "
          f"{n_double} with 2 annotations, overlap >= {overlap}")
    if not by_class.empty:
        print("\n== extractor against human, span F1 by class ==")
        print(by_class.pivot(index="class", columns="arm", values="f1").reindex(list(CLASSES) + ["all"]).to_string())
    if not attrs.empty:
        print("\n== attribute agreement on matched spans ==")
        print(attrs.pivot(index="attribute", columns="arm", values="agreement").to_string())
    if not humans.empty:
        print("\n== human against human ==")
        print(humans.to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="the Label Studio JSON export of the gold-set project")
    ap.add_argument("--gold-dir", default=GOLD)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(GOLD, "scores"))
    args = ap.parse_args()
    score(args.export, args.gold_dir, args.overlap, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
