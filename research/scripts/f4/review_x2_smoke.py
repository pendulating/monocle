#!/usr/bin/env python
"""F4: review the smoke runs of the x2 round (D130), for a person and for the go decision.

For each arm under `--root` (default `multirun/f4x2_smoke`), the review prints:

| Block | Content |
|-------|---------|
| counts | texts, spans for each text, quotable, verbatim, rescued (`alignment_rescue`), `none` items |
| defects | texts with only `none`; copy suspects (`x2_retry.copy_flags`); loops (a span 10 or more times in 1 text) |
| long traces | for texts over 6,000 characters, the share of quotable spans in the second half |
| classes | the class mix and the attribute mix |
| old run | spans for each text against the run of record on the same texts |

`--street-border` adds the check of the 3 subjects: the `descriptors-v6-street`
targets against the innermost `street-v3` cue class, on the same texts.

The last line of each arm is a verdict: PASS when quotable >= 0.95, copy
suspects <= 3% of the texts (the retry pass reads them), loops == 0, and texts with only `none` <= 0.2 (the cue arms) or a value
that the arm allows. A FAIL names the test.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/review_x2_smoke.py --street-border
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from x2_retry import copy_flags  # noqa: E402

ROOT = "/share/ju/matt/mllmsci"
IN = os.path.join(ROOT, "data/f4_traces/extract_input")
CONF = os.path.join(ROOT, "dagspaces/pairwisevqa/conf/extract")
SCHEMA = {
    "x2_cues": "urban_cues_single_v2", "x2_cues_street": "urban_cues_street_v3",
    "x2_desc_restaurant": "aesthetic_descriptors_restaurant_v6",
    "x2_desc_street": "aesthetic_descriptors_street_v6", "x2_captions": "caption_options_v4",
    # F9 (D142): the cue schemas plus the class `verdict`.
    "x2_cues_subj": "urban_cues_single_subj_v1", "x2_cues_street_subj": "urban_cues_street_subj_v1",
}
SCHEMA.update({"x2r" + a[2:]: s + "r" for a, s in list(SCHEMA.items())})
OLD = {
    "x2_cues": [f"{ROOT}/data/f4_traces/extractions/f4_restaurant_extractions.parquet",
                f"{ROOT}/data/f4_traces/extractions/f4_caption_extractions.parquet"],
    "x2_cues_street": glob.glob(f"{ROOT}/multirun/f4_cues_street_dsv4_smoke_v2/runs/*/outputs/extractions/*.parquet"),
    "x2_desc_restaurant": [f"{ROOT}/data/f4_traces/descriptors/f4_restaurant_desc_final.parquet"],
    "x2_desc_street": [f"{ROOT}/data/f4_traces/descriptors_street/f4_street_desc_final.parquet"],
    "x2_captions": glob.glob(f"{ROOT}/data/f4_traces/captions/*final*.parquet")
    or glob.glob(f"{ROOT}/data/f4_traces/captions/*_extractions.parquet"),
}
# The share of texts with only `none` that an arm may have. A caption answer
# with no caption is a real result (about 5% of all answers under v3).
NONE_MAX = {"x2_captions": 0.35, "x2_desc_restaurant": 0.15, "x2_desc_street": 0.35}

__version__ = "1.2.0"


def load(root: str, arm: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(root, arm, "runs", "*", "outputs", "extractions", "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def example_spans(arm: str) -> set:
    ex = yaml.safe_load(open(os.path.join(CONF, SCHEMA[arm] + ".yaml")))["examples"]
    return {e["extraction_text"] for x in ex for e in x["extractions"] if e["extraction_class"] != "none"}


def texts() -> pd.Series:
    fs = glob.glob(os.path.join(IN, "f4_x2smoke_*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=["doc_id", "text"]) for f in fs]).drop_duplicates("doc_id").set_index("doc_id").text


def review(arm: str, d: pd.DataFrame, txt: pd.Series) -> None:
    print("\n" + "=" * 90 + f"\n{arm}  schema {sorted(d.schema_version.dropna().unique())}  ({len(d.doc_id.unique())} texts)")
    sp = d[d.extraction_class.notna()]
    real = sp[sp.extraction_class != "none"]
    per = d.groupby("doc_id").agg(model=("model", "first"), field=("field", "first"))
    per["real"] = real.groupby("doc_id").size().reindex(per.index).fillna(0)
    per["none"] = sp[sp.extraction_class == "none"].groupby("doc_id").size().reindex(per.index).fillna(0)
    q = real.is_quotable.fillna(False)
    resc = real.get("alignment_rescue", pd.Series(index=real.index, dtype=object))
    only_none = float(((per.real == 0) & (per.none > 0)).mean())
    empty = int(((per.real == 0) & (per.none == 0)).sum())
    print(f"spans {len(real)} (+{len(sp) - len(real)} none); quotable {q.mean():.3f}; verbatim "
          f"{real.is_verbatim.fillna(False).mean():.3f}; rescued {int(resc.notna().sum())} "
          f"({resc.value_counts().to_dict()}); only-none texts {only_none:.2f}; empty texts {empty}")
    tab = per.groupby(["model", "field"]).agg(texts=("real", "size"), spans_per_text=("real", "mean"),
                                              only_none=("none", lambda s: int(((per.loc[s.index, "real"] == 0) & (s > 0)).sum())))
    print(tab.round(1).to_string())
    flags = copy_flags(d, example_spans(arm))
    copies = int(flags.copy_suspect.sum())
    rep = real.groupby(["doc_id", "extraction_text"]).size()
    loops = int((rep >= 10).groupby(level=0).any().sum())
    print(f"copy suspects (x2_retry.copy_flags): {copies} texts; loops (a span >= 10 times): {loops} texts; "
          f"max repeat {int(rep.max()) if len(rep) else 0}")
    long_ids = [i for i in per.index if len(txt.get(i, "")) > 6000]
    if long_ids:
        lq = real[real.doc_id.isin(long_ids) & q]
        half = [r.char_start >= len(txt[r.doc_id]) / 2 for r in lq.itertuples()]
        print(f"long texts {len(long_ids)}: quotable spans {len(lq)}, share in the second half {sum(half) / max(1, len(half)):.2f}")
    print("classes: " + json.dumps(real.extraction_class.value_counts().to_dict()))
    attrs = real.attributes_json.map(lambda s: json.loads(s) if isinstance(s, str) else {})
    for k in sorted({k for a in attrs for k in a} - {"head", "slot", "style"}):
        vc = attrs.map(lambda a: a.get(k)).value_counts().head(12).to_dict()
        print(f"  {k}: {vc}")
    old = [f for f in OLD.get(arm, []) if os.path.exists(f)]
    if old:
        o = pd.concat([pd.read_parquet(f, columns=["doc_id", "extraction_class", "is_quotable"]) for f in old])
        o = o[o.doc_id.isin(per.index) & o.extraction_class.notna() & o.is_quotable.fillna(False)]
        on = o.groupby("doc_id").size().reindex(per.index).fillna(0)
        nn = real[q].groupby("doc_id").size().reindex(per.index).fillna(0)
        cmp = pd.DataFrame({"old": on, "new": nn}).join(per[["model", "field"]]).groupby(["model", "field"]).mean().round(1)
        print("quotable spans per text, old run against x2:\n" + cmp.to_string())
    fails = []
    if (real.extraction_class == "verdict").any() or arm.endswith("_subj"):
        fails += verdict_checks(real, per)
    if q.mean() < 0.95:
        fails.append("quotable")
    if copies > 0.03 * len(per):
        fails.append("copies")
    if loops:
        fails.append("loops")
    if only_none > NONE_MAX.get(arm, 0.2):
        fails.append("only-none")
    print(f"VERDICT {arm}: " + ("PASS" if not fails else "FAIL " + ",".join(fails)))


# F9 (`f9-extraction-design.md`, section 6.1): every gemma and Qwen answer
# gives a verdict, and no verdict span is the echo of the question.
QUESTIONS = ("does this look like a nice restaurant to eat at", "would this street make a good spot for taking a photo")


def verdict_checks(real: pd.DataFrame, per: pd.DataFrame) -> list:
    v = real[real.extraction_class == "verdict"]
    labels = v.attributes_json.map(lambda s: json.loads(s).get("label") if isinstance(s, str) else None)
    print("verdict labels: " + json.dumps(labels.value_counts(dropna=False).to_dict()))
    resp = per[(per.field == "response")]
    has = resp.index.isin(v.doc_id)
    thinking = ~resp.model.str.startswith("llama")
    share = float(has[thinking.values].mean()) if thinking.any() else 1.0
    print(f"response texts with a verdict span: gemma and Qwen {share:.2f}; Llama "
          f"{float(has[(~thinking).values].mean()) if (~thinking).any() else float('nan'):.2f}; "
          f"reasoning texts {per[per.field == 'reasoning'].index.isin(v.doc_id).mean():.2f}")
    echo = v.extraction_text.fillna("").str.lower().str.strip(" ?\"'*").isin(QUESTIONS)
    print(f"verdict spans that echo the question: {int(echo.sum())}")
    fails = []
    if share < 0.9:
        fails.append("verdict-coverage")
    if echo.any():
        fails.append("echo")
    return fails


def innermost(cues: pd.DataFrame, doc: str, a: int, b: int):
    c = cues[(cues.doc_id == doc) & (cues.char_start <= a) & (cues.char_end >= b)]
    if c.empty:
        return "none"
    c = c.assign(w=c.char_end - c.char_start).sort_values("w")
    return c.extraction_class.iloc[0]


def street_border(root: str) -> None:
    cues = load(root, "x2_cues_street")
    desc = load(root, "x2_desc_street")
    if cues.empty or desc.empty:
        print("\nstreet border: a run is missing")
        return
    cues = cues[cues.extraction_class.notna() & (cues.extraction_class != "none") & cues.is_quotable.fillna(False)]
    desc = desc[(desc.extraction_class == "descriptor") & desc.is_quotable.fillna(False)].copy()
    desc["target"] = desc.attributes_json.map(lambda s: json.loads(s).get("target"))
    desc["cue"] = [innermost(cues, r.doc_id, r.char_start, r.char_end) for r in desc.itertuples()]
    print("\n" + "=" * 90 + "\nstreet border: descriptors-v6-street target against the innermost street-v3 class")
    print(pd.crosstab(desc.target, desc.cue, margins=True).to_string())
    want = {"place": ("visual_evidence", "inference"), "current_photo": ("image_artifact",),
            "planned_photo": ("photo_advice",)}
    for t, cls in want.items():
        s = desc[desc.target == t]
        if len(s):
            print(f"  {t}: {s.cue.isin(cls).mean():.2f} in {cls}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(ROOT, "multirun/f4x2_smoke"))
    ap.add_argument("--arms", nargs="*", default=list(SCHEMA))
    ap.add_argument("--street-border", action="store_true")
    args = ap.parse_args()
    txt = texts()
    for arm in args.arms:
        d = load(args.root, arm)
        if d.empty:
            print(f"\n{arm}: no output")
            continue
        review(arm, d, txt)
    if args.street_border:
        street_border(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
