#!/usr/bin/env python
"""Build the Label Studio gold set for the urban-cues extraction.

The extraction is a measurement, and it needs a human anchor. This script
draws the traces, writes them as Label Studio tasks with NO pre-annotation,
and writes the spans of each large extractor as separate prediction files
for the scorer. The annotators label blind; the extractors are scored after.

The sample
----------
30 traces by default: 15 of each judge, spread over the 7 cases (2 for each
case, plus 1 more for subway_safety). Every trace has an answer
(`judge_answered`), and every trace is under a length cap, so that 1 trace is
a task an annotator can finish. The cap keeps the sample readable and it
biases the sample toward shorter traces; the doc says so.

The seed is fixed, thus the same call gives the same 30 traces.

The confidence floor (2026-09-23)
---------------------------------
A judge sometimes argues with its prompt: "I don't see a subway entrance in
Image B". That is a separate reasoning task, and the gold set does not mix it
in. The facing filter (`sv/nyc/curation/filter_facing.py`) gives
each image an `attribution_confidence` in [0, 1]: the share of the unit that
sits inside the face, times a cubic falloff in the distance to the unit. The
score is geometric, not a model's read. It sits in the image manifest of each
run, and the builder joins it to both images of a pair.

The score runs low. The mean over a case is 0.09 (schools) to 0.38 (subway),
and no case but subway has a pair whose weaker image reaches 0.5. Thus the
floor is a quantile inside the case: a pair qualifies when its WEAKER image
(`conf_min`) sits at or above the `--conf-quantile` (default 0.75) of the
pair minima of that case and judge. The 2 cases whose unit is the image
itself (road_quality, street_photography) carry no score and take no floor.

The traces a person already annotated stay in the set (`--pin`, default: the
annotated task_ids of the carried export), whatever their score. Their
`conf_min` is in the task data and in `sample.csv`; the alignment notebook
can drop them.

The images
----------
Each task shows the 2 photographs above the trace, as the judge saw them:
Image A is `presented_left_path` and Image B is `presented_right_path`. Half
the pairs are swapped, thus `image_path_a` is NOT always Image A. The files
are copied under `data/gold_set/images/` and served by Label Studio from its
local-files root. The annotator answers, for each image, whether it shows
the item the prompt names (a subway station entrance, a school, ...). The
judges express doubt about that in their traces (2026-09-22), and the
question records the fact.

Carried annotations
-------------------
`--carry-annotations <export.json>` copies the annotations of an earlier
project onto the tasks with the same `task_id`, so that a rebuild of the set
loses no work. The newest export in `research/notebooks/cvpr/cues_annotations/` is the
default; pass `--carry-annotations none` for a clean set.

Output
------
    data/gold_set/label_config.xml        the Label Studio labelling interface
    data/gold_set/tasks_blind.json        30 tasks: text, images, metadata, carried annotations, no predictions
    data/gold_set/images/<task_id>_{A,B}.jpg  the 2 photographs of each task
    data/gold_set/sample.csv              the trace table: case, judge, pair_id, chars
    data/gold_set/predictions_<arm>.json  the spans of each extractor arm, as
                                          Label Studio predictions, for the scorer

Use
---
    python research/scripts/build_gold_set.py
    python research/scripts/build_gold_set.py --n-per-judge 15 --max-chars 9000 --seed 20260922

See `research/docs/langextract-trace-extraction.md`, "The gold set".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "cvpr"))
import _extractions as X  # noqa: E402

__version__ = "1.6.0"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "gold_set")

# The registered thinking runs. The same table lives in
# `dagspaces/pairwisevqa/submit_trace_shards.py`.
from dagspaces.pairwisevqa.submit_trace_shards import (  # noqa: E402
    CASE_ORDER, GEMMA_RUNS, QWEN_RUNS,
)

JUDGES = {"gemma-4-12b": GEMMA_RUNS, "qwen3.5-9b": QWEN_RUNS}

# The extractor arms whose spans the scorer compares with the humans: the
# primary and the control of each judge.
SCORED_ARMS = {
    "gemma-4-12b": ["gemma_by_qwen27_v2", "gemma_by_gemma31_v2"],
    "qwen3.5-9b": ["qwen_by_gemma31_v2", "qwen_by_qwen27_v2"],
}

# What the prompt says both photographs show, for each case. The annotator
# confirms it for each image.
ITEM_OF_INTEREST = {
    "subway_safety": "a subway station entrance",
    "libraries": "a public library building",
    "schools": "a K-12 school",
    "road_quality": "a roadway",
    "parks_plazas": "a park or a public plaza",
    "restaurants": "a restaurant",
    "street_photography": "a city block",
}

IMAGE_DIR_NAME = "images"

# Version 3 of the schema (2026-09-28): the 12 classes on the IC codebook.
# `--schema v2` keeps the 7 classes of the first set.
CLASS_COLOURS_V2 = {
    "visual_evidence": "#4C9F70",
    "inference": "#D9822B",
    "person_reference": "#C0392B",
    "image_artifact": "#7F8C8D",
    "comparison": "#2E86C1",
    "uncertainty": "#8E44AD",
    "decision": "#111111",
}
CLASS_COLOURS_V3 = {
    "dimension": "#4C9F70",
    "inference": "#D9822B",
    "perspective": "#6C3FC5",
    "weighing": "#B5175A",
    "justification": "#1F7A8C",
    "dismissal": "#A93226",
    "verdict": "#B7950B",
    "decision": "#111111",
    "comparison": "#2E86C1",
    "uncertainty": "#C24CB0",
    "person_reference": "#D35400",
    "image_artifact": "#7F8C8D",
}
CLASS_COLOURS = CLASS_COLOURS_V3

# The attribute controls the interface shows for a selected span. A tuple
# (name, title, values) is a Choices control; values None is a free text box
# (a TextArea, for the free names of the codebook). `kind` is split in 2,
# because `inference` and `image_artifact` share the name but not the values.
LABELS = ["MuchLess", "Less", "Same", "More", "MuchMore", "NotSure"]
ATTRIBUTES_V2 = [
    ("image", "Image", ["A", "B", "both", "unclear"]),
    ("valence", "Valence (visual_evidence)", ["good", "bad", "neutral"]),
    # The v2 values, written out. `X.VOCABULARY` is the union of every
    # version, and single-v1 adds values that v2 never declared.
    ("category", "Category (visual_evidence)",
     ["people", "cleanliness", "upkeep", "greenery", "light", "signage", "traffic",
      "construction", "commerce", "architecture", "other"]),
    ("inference_kind", "Kind (inference)",
     ["safety", "quality", "upkeep", "wealth", "class", "crime", "demographic", "other"]),
    ("artifact_kind", "Kind (image_artifact)", list(X.VOCABULARY["image_artifact"]["kind"])),
    ("used_in_judgment", "Used in judgment (person_reference)", ["yes", "no"]),
    ("direction", "Direction (comparison): the image with MORE of the quality", ["A", "B", "neither"]),
    ("hedged", "Hedged (comparison)", ["yes", "no"]),
    ("reason", "Reason (uncertainty)", list(X.VOCABULARY["uncertainty"]["reason"])),
    ("label", "Label (decision)", LABELS),
]
ATTRIBUTES_V3 = [
    ("image", "Image (dimension, inference, person_reference, image_artifact)", ["A", "B", "both", "unclear"]),
    ("valence", "Valence (dimension)", ["good", "bad", "neutral"]),
    ("name", "Name (dimension, inference, perspective): the attribute observed, evaluated, or the mode", None),
    ("inference_kind", "Kind (inference)", ["safety", "quality", "upkeep", "wealth", "class", "crime", "demographic", "other"]),
    ("favors", "Favors (perspective, weighing): the image the standard counts FOR", ["A", "B", "neither"]),
    ("wins", "Wins (perspective): the decision follows this standard", ["yes", "no"]),
    ("source", "Source (perspective)", ["world_knowledge", "task_convention", "other"]),
    ("mechanism", "Mechanism (weighing)", ["weighing", "balancing", "prioritization", "trade_off", "conditional"]),
    ("of", "Of (justification): what the reason belongs to", ["weighing", "perspective", "uncertainty", "dismissal"]),
    ("role", "Role (justification)", ["reason", "condition"]),
    ("target", "Target (dismissal): what is set aside", None),
    ("about", "About (dismissal)", ["cue", "perspective", "premise"]),
    ("label", "Label (verdict, decision)", LABELS),
    ("direction", "Direction (comparison): the image with MORE of the quality", ["A", "B", "neither"]),
    ("dimension", "Dimension (comparison): the quality compared", None),
    ("hedged", "Hedged (comparison)", ["yes", "no"]),
    ("reason", "Reason (uncertainty)", ["looks_equal", "cannot_see", "out_of_view", "other"]),
    ("marker", "Marker (uncertainty)", ["qualifier", "contrastive", "explicit_doubt", "reconsideration"]),
    ("used_in_judgment", "Used in judgment (person_reference)", ["yes", "no"]),
    ("artifact_kind", "Kind (image_artifact)", ["blur", "angle", "occlusion", "time_of_day", "weather", "other"]),
]
ATTRIBUTES = ATTRIBUTES_V3

# The labels of the first set, mapped onto version 3 for a carried annotation.
RENAME_V2_TO_V3 = {"visual_evidence": "dimension"}


def label_config() -> str:
    """The Label Studio interface: 7 span labels, attributes for each region."""
    labels = "\n".join(
        f'    <Label value="{cls}" background="{colour}"/>'
        for cls, colour in CLASS_COLOURS.items()
    )
    choices = []
    for name, title, values in ATTRIBUTES:
        if values is None:
            choices.append(
                f'    <Header value="{title}" size="5"/>\n'
                f'    <TextArea name="{name}" toName="trace" perRegion="true" rows="1" '
                f'maxSubmissions="1" placeholder="a short phrase" required="false"/>'
            )
            continue
        opts = "\n".join(f'      <Choice value="{v}"/>' for v in values)
        choices.append(
            f'    <Header value="{title}" size="5"/>\n'
            f'    <Choices name="{name}" toName="trace" perRegion="true" '
            f'choice="single" showInline="true" required="false">\n{opts}\n    </Choices>'
        )
    choices_xml = "\n".join(choices)
    return f"""<View>
  <Style>.lsf-richtext__container {{ font-size: 15px; line-height: 1.5; }}</Style>
  <Header value="Case: $case   |   Judge: $judge   |   The judge answered: $presented_label"/>
  <Header value="The 2 photographs, as the judge saw them. The prompt told the judge that both show $item." size="5"/>
  <View style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
    <View>
      <Header value="Image A" size="4"/>
      <Image name="image_a" value="$image_a" zoom="true" maxWidth="100%"/>
    </View>
    <View>
      <Header value="Image B" size="4"/>
      <Image name="image_b" value="$image_b" zoom="true" maxWidth="100%"/>
    </View>
  </View>
  <Header value="Notes: depiction (does an image show $item?), or anything else about this trace." size="5"/>
  <TextArea name="notes" toName="trace" rows="3" maxSubmissions="1" editable="true" placeholder="Optional. Free text."/>
  <Header value="Mark each span the model wrote. Copy the span as it stands; do not extend it to a sentence. One class for each span. Then set its attributes below." size="5"/>
  <Labels name="span" toName="trace" showInline="true">
{labels}
  </Labels>
  <Text name="trace" value="$text" granularity="symbol"/>
  <View visibleWhen="region-selected" style="padding: 8px; background: #f4f4f4; border-radius: 6px;">
    <Header value="Attributes of the selected span. Fill only the ones that belong to its class." size="4"/>
{choices_xml}
  </View>
</View>
"""


def control_pair_ids(judge: str, case: str) -> set:
    """The traces that the control arm of `judge` read.

    The control arms read 2,000 traces for each case, not 11,000. A gold trace
    must sit in that subset, or the control extractor has no spans on it and
    the scorer cannot compare the 2 large extractors on the same trace.
    """
    arm = X.ARMS[SCORED_ARMS[judge][1]]
    path = os.path.join(ROOT, arm["root"], f"{case}_extractions.parquet")
    return set(pd.read_parquet(path, columns=["pair_id"]).pair_id.astype(str))


def manifest_path(results_path: str) -> Optional[str]:
    """The image manifest the run read: `data.parquet_path` of the Hydra
    config in the job folder above the results file."""
    import yaml
    d = os.path.dirname(results_path)
    while d not in ("", "/") and not os.path.isdir(os.path.join(d, ".hydra")):
        d = os.path.dirname(d)
    cfg_path = os.path.join(d, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    return (cfg.get("data") or {}).get("parquet_path")


# The judge doubts that the image shows the thing the prompt names. Measured
# 2026-09-30 over the version 3 samples: 50% to 82% of the unit-case traces
# match, and the facing-filter confidence barely moves that share (35% to
# 80% in the top decile). Thus the clear set excludes the trace itself.
REFERENT_DOUBT = re.compile(
    r"(don'?t|do not|can'?t|cannot|not) (really |clearly |actually )?(see|find|spot|make out|identify)\b"
    r"|is there (a|an|even) |isn'?t (a|an|really) |doesn'?t (show|appear to show|look like)"
    r"|no (visible|obvious|clear) (subway|library|school|park|plaza|restaurant|entrance)"
    r"|hard to (see|tell|find)|where is the|(prompt|question) (says|states|is wrong|might be wrong)",
    re.I,
)


def image_confidence(results: pd.DataFrame, results_path: str) -> pd.DataFrame:
    """Add the facing diagnostics of both images to the pair rows, from the
    run's manifest: `conf_a`, `conf_b`, `conf_min`, `conf_mean`
    (`attribution_confidence`), and, when the manifest holds them,
    `db_max` (the larger of the 2 bearing offsets, degrees off the face
    centre) and `dist_max` (the larger of the 2 distances to the unit, feet).
    A run whose manifest carries no score (the image is the unit) gets NaN."""
    out = results.copy()
    for c in ("conf_a", "conf_b", "conf_min", "conf_mean", "db_max", "dist_max"):
        out[c] = float("nan")
    mp = manifest_path(results_path)
    if not mp or not os.path.exists(mp):
        return out
    try:
        m = pd.read_parquet(mp, columns=["sample_id", "attribution_confidence"])
    except Exception:
        return out
    m = m.drop_duplicates("sample_id").set_index("sample_id").attribution_confidence
    out["conf_a"] = out.sample_id_a.map(m)
    out["conf_b"] = out.sample_id_b.map(m)
    out["conf_min"] = out[["conf_a", "conf_b"]].min(axis=1)
    out["conf_mean"] = out[["conf_a", "conf_b"]].mean(axis=1)
    try:
        g = pd.read_parquet(mp, columns=["sample_id", "delta_bearing_deg", "distance_to_unit_ft"])
    except Exception:
        return out
    g = g.drop_duplicates("sample_id").set_index("sample_id")
    out["db_max"] = pd.concat([out.sample_id_a.map(g.delta_bearing_deg), out.sample_id_b.map(g.delta_bearing_deg)], axis=1).max(axis=1)
    out["dist_max"] = pd.concat([out.sample_id_a.map(g.distance_to_unit_ft), out.sample_id_b.map(g.distance_to_unit_ft)], axis=1).max(axis=1)
    return out


# The rows the version 3 arms read: `trace_extract.load_traces` keeps the
# traces of `min_chars` or more, then `_sample` draws `runtime.sample_n` of
# them with the seed of `pair_sampler.pair_seed`. The same call on the same
# results file gives the same rows, thus the 2 arms of 1 judge read the same
# 1,000 traces of each case, and a gold trace drawn here sits in both.
V3_SAMPLE_N = 1000
V3_SAMPLE_SEED = 777
V3_MIN_CHARS = 200


def v3_sample_pair_ids(results_path: str, n: int = V3_SAMPLE_N, seed: int = V3_SAMPLE_SEED) -> set:
    """The pair_ids of the version 3 sample of 1 judge run, reproduced."""
    df = pd.read_parquet(results_path, columns=["pair_id", "model_reasoning"])
    df["model_reasoning"] = df.model_reasoning.fillna("").astype(str)
    df = df[df.model_reasoning.str.len() >= V3_MIN_CHARS].reset_index(drop=True)
    if n >= len(df):
        return set(df.pair_id.astype(str))
    return set(df.sample(n=n, random_state=seed).pair_id.astype(str))


def verify_v3_sharing(arms: Sequence[str]) -> pd.DataFrame:
    """For each case: do the merged version 3 arms hold the reproduced sample,
    and nothing else? A merged arm with fewer rows (a run in flight) reports
    the share of its rows that sit in the sample."""
    rows = []
    for judge, runs in JUDGES.items():
        for case in CASE_ORDER:
            expected = v3_sample_pair_ids(os.path.join(ROOT, runs[case]))
            for arm in arms:
                spec = X.ARMS[arm]
                if spec["judge"] != judge:
                    continue
                path = os.path.join(ROOT, spec["root"], f"{case}_extractions.parquet")
                if not os.path.exists(path):
                    rows.append({"judge": judge, "case": case, "arm": arm, "rows": 0, "in_sample": None, "equal": None})
                    continue
                got = set(pd.read_parquet(path, columns=["pair_id"]).pair_id.astype(str))
                rows.append({"judge": judge, "case": case, "arm": arm, "rows": len(got),
                             "in_sample": round(len(got & expected) / len(got), 4) if got else None,
                             "equal": got == expected})
    return pd.DataFrame(rows)


def sample_traces(n_per_judge: int, max_chars: int, seed: int,
                  conf_quantile: float = 0.75, pin: Optional[Sequence[str]] = None,
                  in_v3_sample: bool = False, exclude: Optional[Sequence[str]] = None,
                  max_delta_bearing: Optional[float] = None, max_distance_ft: Optional[float] = None,
                  exclude_referent_doubt: bool = False) -> pd.DataFrame:
    """Draw the traces: answered, under the cap, in the control subset, above
    the confidence floor of the case, spread over the cases. A pinned task_id
    stays in and counts toward its case."""
    rows: List[pd.DataFrame] = []
    pin = set(pin or [])
    exclude = set(exclude or [])
    per_case = n_per_judge // len(CASE_ORDER)
    extra = n_per_judge - per_case * len(CASE_ORDER)
    for judge, runs in JUDGES.items():
        for i, case in enumerate(CASE_ORDER):
            take = per_case + (1 if i < extra else 0)
            results_path = os.path.join(ROOT, runs[case])
            df = pd.read_parquet(
                results_path,
                columns=["pair_id", "sample_id_a", "sample_id_b", "model_reasoning", "model_response",
                         "presented_label", "presented_left_path", "presented_right_path"],
            )
            df = image_confidence(df, results_path)
            df["chars"] = df.model_reasoning.fillna("").str.len()
            df["case"] = case
            df["judge"] = judge
            df["task_id"] = [f"{judge.split('-')[0]}_{case}_{pid}" for pid in df.pair_id]
            has_score = df.conf_min.notna().any()
            floor = float(df.conf_min.quantile(conf_quantile)) if has_score else float("nan")
            df["conf_floor"] = floor
            df["conf_quantile"] = (df.conf_min.rank(pct=True) if has_score else float("nan"))
            answered = df.model_response.fillna("").str.strip() != ""
            in_control = df.pair_id.astype(str).isin(control_pair_ids(judge, case))
            if in_v3_sample:
                # The version 3 arms read 1,000 traces of each case, and the
                # control subset of version 2 is another draw. A gold trace
                # for version 3 must sit in the version 3 sample.
                in_control = df.pair_id.astype(str).isin(v3_sample_pair_ids(results_path))
            not_excluded = ~df.task_id.isin(exclude)
            above = (df.conf_min >= floor) if has_score else pd.Series(True, index=df.index)
            # The clear set (2026-09-30): the unit sits near the centre of
            # both faces and close to both cameras, and the judge never
            # doubts that the image shows it.
            if has_score and max_delta_bearing is not None and df.db_max.notna().any():
                above = above & (df.db_max <= max_delta_bearing)
            if has_score and max_distance_ft is not None and df.dist_max.notna().any():
                above = above & (df.dist_max <= max_distance_ft)
            df["referent_doubt"] = df.model_reasoning.fillna("").str.contains(REFERENT_DOUBT)
            if exclude_referent_doubt:
                above = above & ~df.referent_doubt
            pinned = df[df.task_id.isin(pin)]
            pool = df[answered & in_control & (df.chars >= 200) & (df.chars <= max_chars) & above
                      & ~df.task_id.isin(pin) & not_excluded]
            n_draw = max(0, take - len(pinned))
            pick = pd.concat([pinned, pool.sample(n=n_draw, random_state=seed + i)], ignore_index=True)
            rows.append(pick)
    return pd.concat(rows, ignore_index=True)


def copy_images(sample: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Copy the 2 photographs of each task under `<out_dir>/images/`, as the
    judge saw them: A is the left (first) image, B the right (second). Give
    the sample with `image_a_file` and `image_b_file`."""
    import shutil
    img_dir = os.path.join(out_dir, IMAGE_DIR_NAME)
    os.makedirs(img_dir, exist_ok=True)
    a_files, b_files = [], []
    for r in sample.itertuples():
        for side, src, store in (("A", r.presented_left_path, a_files), ("B", r.presented_right_path, b_files)):
            ext = os.path.splitext(str(src))[1] or ".jpg"
            name = f"{r.task_id}_{side}{ext}"
            dst = os.path.join(img_dir, name)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
            store.append(name)
    out = sample.copy()
    out["image_a_file"] = a_files
    out["image_b_file"] = b_files
    return out


def carry_annotations(export_path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    """The annotations of an earlier export, by task_id, as Label Studio
    import expects them: a list of {"result": [...]} for each task."""
    if not export_path or export_path == "none":
        return {}
    with open(export_path) as fh:
        data = json.load(fh)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for t in data:
        tid = t["data"].get("task_id")
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled") and a.get("result")]
        if tid and anns:
            out[tid] = [{"result": _rename_labels(a["result"]), "lead_time": a.get("lead_time")} for a in anns]
    return out


def _rename_labels(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map the labels of an earlier set onto the current classes
    (`RENAME_V2_TO_V3`). A label the current interface does not hold would
    not render."""
    out = []
    for r in results:
        r = json.loads(json.dumps(r))
        if r.get("type") == "labels":
            labels = r.get("value", {}).get("labels") or []
            r["value"]["labels"] = [RENAME_V2_TO_V3.get(x, x) for x in labels]
        out.append(r)
    return out


def local_file_url(name: str, out_dir: str = OUT_DIR) -> str:
    """The Label Studio URL of a file under the local-files root
    (`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT`, which is `data/gold_set`). A
    set built in a folder under that root keeps the folder in the URL."""
    rel = os.path.relpath(os.path.join(os.path.abspath(out_dir), IMAGE_DIR_NAME, name), os.path.abspath(OUT_DIR))
    return f"/data/local-files/?d={rel}"


def _num(v: Any) -> Optional[float]:
    """A float for JSON, or None for NaN."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)


def tasks_from(sample: pd.DataFrame,
               carried: Optional[Dict[str, List[Dict[str, Any]]]] = None,
               out_dir: str = OUT_DIR) -> List[Dict[str, Any]]:
    """The tasks: the text, the 2 images, the metadata, and any carried
    annotations. No prediction."""
    tasks = []
    carried = carried or {}
    for r in sample.itertuples():
        task: Dict[str, Any] = {
            "data": {
                "text": r.model_reasoning,
                "image_a": local_file_url(r.image_a_file, out_dir),
                "image_b": local_file_url(r.image_b_file, out_dir),
                "item": ITEM_OF_INTEREST.get(r.case, "the item the prompt names"),
                "case": r.case,
                "judge": r.judge,
                "pair_id": r.pair_id,
                "presented_label": r.presented_label,
                "task_id": r.task_id,
                "chars": int(r.chars),
                "conf_a": _num(r.conf_a), "conf_b": _num(r.conf_b),
                "conf_min": _num(r.conf_min), "conf_floor": _num(r.conf_floor),
                "conf_quantile": _num(r.conf_quantile),
            }
        }
        if r.task_id in carried:
            task["annotations"] = carried[r.task_id]
        tasks.append(task)
    return tasks


def predictions_from(sample: pd.DataFrame, arm: str) -> List[Dict[str, Any]]:
    """The spans of 1 extractor arm as Label Studio predictions, for the scorer.

    Only the rows that `load` keeps: quotable, not duplicate, judge answered,
    last decision. The result names the arm as `model_version`.
    """
    spec = X.ARMS[arm]
    df = X.load(root=os.path.join(ROOT, spec["root"]), require_canonical=True)
    df = df[df.pair_id.isin(set(sample.pair_id))]
    out = []
    by_key = {(r.case, r.pair_id): r for r in sample.itertuples()}
    for (case, pair_id), part in df.groupby(["case", "pair_id"]):
        row = by_key.get((case, pair_id))
        if row is None:
            continue
        results = []
        for i, e in enumerate(part.sort_values("char_start").itertuples()):
            if pd.isna(e.char_start) or pd.isna(e.char_end):
                continue
            rid = f"{arm}_{i}"
            results.append({
                "id": rid, "from_name": "span", "to_name": "trace", "type": "labels",
                "value": {"start": int(e.char_start), "end": int(e.char_end),
                          "text": str(e.extraction_text), "labels": [e.extraction_class]},
            })
            attrs = json.loads(e.attributes_json) if isinstance(e.attributes_json, str) else {}
            for name, value in attrs.items():
                control = name
                if e.extraction_class == "inference" and name == "kind":
                    control = "inference_kind"
                if e.extraction_class == "image_artifact" and name == "kind":
                    control = "artifact_kind"
                if control not in {a[0] for a in ATTRIBUTES}:
                    continue
                results.append({
                    "id": rid, "from_name": control, "to_name": "trace", "type": "choices",
                    "value": {"start": int(e.char_start), "end": int(e.char_end),
                              "choices": [str(value)]},
                })
        out.append({
            "task_id": row.task_id, "case": case, "pair_id": pair_id,
            "model_version": arm, "result": results,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-judge", type=int, default=15)
    ap.add_argument("--max-chars", type=int, default=9000,
                    help="the length cap of a trace; the median qwen trace is about 8,000")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--conf-quantile", type=float, default=0.75,
                    help="a pair qualifies when its weaker image sits at or above this quantile of the "
                         "pair minima of its case; cases without a score take no floor")
    ap.add_argument("--pin", nargs="*", default=None,
                    help="task_ids that stay in the set whatever their score; default: the annotated "
                         "tasks of the carried export")
    ap.add_argument("--max-delta-bearing", type=float, default=None,
                    help="both images: the unit at most this many degrees off the face centre (the clear set: 30)")
    ap.add_argument("--max-distance-ft", type=float, default=None,
                    help="both images: the camera at most this far from the unit, in feet (the clear set: 120)")
    ap.add_argument("--exclude-referent-doubt", action="store_true",
                    help="drop a trace in which the judge doubts that the image shows the item (REFERENT_DOUBT)")
    ap.add_argument("--in-v3-sample", action="store_true",
                    help="draw from the 1,000-row sample the version 3 arms read, not the version 2 control subset")
    ap.add_argument("--exclude-sample", default=None,
                    help="a sample.csv of an earlier set; its task_ids are not drawn again")
    ap.add_argument("--verify-sharing", action="store_true",
                    help="print, for each case, whether the merged version 3 arms hold the reproduced sample")
    ap.add_argument("--config-only", action="store_true",
                    help="write label_config.xml into --out and stop; the sample and the tasks stay as they are")
    ap.add_argument("--schema", choices=["v2", "v3"], default="v3",
                    help="the class set of the labelling interface; v3 is the IC set of 2026-09-28")
    ap.add_argument("--carry-annotations", default=None,
                    help="an earlier export whose annotations move onto the same task_ids; "
                         "default: the newest export in research/notebooks/cvpr/cues_annotations; `none` for a clean set")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    global CLASS_COLOURS, ATTRIBUTES, RENAME_V2_TO_V3
    if args.schema == "v2":
        CLASS_COLOURS, ATTRIBUTES, RENAME_V2_TO_V3 = CLASS_COLOURS_V2, ATTRIBUTES_V2, {}
    if args.config_only:
        with open(os.path.join(args.out, "label_config.xml"), "w") as fh:
            fh.write(label_config())
        print(f"[gold] wrote {os.path.join(args.out, 'label_config.xml')}")
        return 0
    export_path = args.carry_annotations
    if export_path is None:
        import glob
        exports = sorted(glob.glob(os.path.join(ROOT, "research", "notebooks", "cvpr", "cues_annotations", "*.json")),
                         key=os.path.getmtime)
        export_path = exports[-1] if exports else "none"
    carried = carry_annotations(export_path)
    print(f"[gold] carried annotations: {sum(len(v) for v in carried.values())} on {len(carried)} tasks "
          f"from {os.path.basename(export_path) if export_path != 'none' else 'none'}")
    pin = list(carried) if args.pin is None else args.pin
    exclude = list(pd.read_csv(args.exclude_sample).task_id) if args.exclude_sample else []
    if args.verify_sharing:
        v3_arms = [a for a in X.ARMS if X.ARMS[a]["schema_version"] == "v3"]
        print(verify_v3_sharing(v3_arms).to_string(index=False))
    sample = sample_traces(args.n_per_judge, args.max_chars, args.seed,
                           conf_quantile=args.conf_quantile, pin=pin,
                           in_v3_sample=args.in_v3_sample, exclude=exclude,
                           max_delta_bearing=args.max_delta_bearing, max_distance_ft=args.max_distance_ft,
                           exclude_referent_doubt=args.exclude_referent_doubt)
    sample = copy_images(sample, args.out)
    below = sample[sample.conf_min.notna() & (sample.conf_min < sample.conf_floor)]
    print(f"[gold] confidence floor: quantile {args.conf_quantile} of the pair minima in each case; "
          f"{len(below)} pinned traces sit below it: {sorted(below.task_id)}")
    sample[["task_id", "judge", "case", "pair_id", "presented_label", "chars",
            "conf_a", "conf_b", "conf_min", "conf_floor", "conf_quantile", "db_max", "dist_max",
            "referent_doubt"]].round(4).to_csv(
        os.path.join(args.out, "sample.csv"), index=False)
    with open(os.path.join(args.out, "label_config.xml"), "w") as fh:
        fh.write(label_config())
    with open(os.path.join(args.out, "tasks_blind.json"), "w") as fh:
        json.dump(tasks_from(sample, carried, args.out), fh, indent=1)
    print(f"[gold] {len(sample)} traces, {int(sample.chars.sum()):,} characters, "
          f"median {int(sample.chars.median()):,}")
    print(sample.groupby(["judge", "case"]).size().unstack(0).to_string())
    print(sample.groupby("case")[["conf_min", "conf_floor", "db_max", "dist_max"]].mean().round(2).to_string())
    print(f"[gold] traces with referent doubt in the set: {int(sample.referent_doubt.sum())} of {len(sample)}")
    for judge, arms in SCORED_ARMS.items():
        for arm in arms:
            preds = predictions_from(sample[sample.judge == judge], arm)
            n = sum(len([r for r in p["result"] if r["type"] == "labels"]) for p in preds)
            path = os.path.join(args.out, f"predictions_{arm}.json")
            with open(path, "w") as fh:
                json.dump(preds, fh, indent=1)
            print(f"[gold] {arm}: {len(preds)} tasks, {n} spans -> {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
