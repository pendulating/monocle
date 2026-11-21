#!/usr/bin/env python
"""F4: print a review of a street cue smoke run (`street-v1`, D125), for a person to read.

The review holds:

- the counts for each (model, field): texts, texts with no span, spans, the
  quotable share, and the spans of each class;
- the copies: a span that equals an example span and is not verbatim;
- the attribute mix of each class;
- the border check: the `descriptors-v5` spans of the same texts, by
  `target`, and the cue class that holds each. A `place` descriptor should
  fall in `visual_evidence` or `inference`; a `photo` descriptor should fall
  outside them (D126);
- `--show N` texts in full, each followed by its cue spans.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/review_street_cue_smoke.py \
        --run multirun/f4_cues_street_dsv4_smoke --show 12 > review.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
SCHEMA = os.path.join(ROOT, "dagspaces/urbanpairvqa/conf/extract/urban_cues_street_v2.yaml")
INPUT = os.path.join(ROOT, "data/f4_traces/extract_input/f4_streetsmoke_desc.parquet")
DESC = os.path.join(ROOT, "data/f4_traces/descriptors_street/f4_street_desc_final.parquet")
TRACK_C = ("visual_evidence", "inference")

__version__ = "1.0.0"


def border(spans: pd.DataFrame) -> pd.DataFrame:
    """Give 1 row for each quotable descriptor of the smoke texts, with the cue class that holds it."""
    desc = pd.read_parquet(DESC)
    desc = desc[desc.doc_id.isin(set(spans.doc_id)) & desc.is_quotable.fillna(False)].copy()
    desc["target"] = desc.attributes_json.map(lambda s: json.loads(s).get("target"))
    cues = spans[spans.is_quotable.fillna(False)][["doc_id", "extraction_class", "char_start", "char_end"]]
    m = desc.reset_index().merge(cues.rename(columns={"char_start": "c0", "char_end": "c1"}), on="doc_id")
    m = m[(m.char_start >= m.c0) & (m.char_end <= m.c1)]
    holder = m.groupby("index").extraction_class_y.first()
    desc["cue_class"] = holder.reindex(desc.index).fillna("none").values
    desc["in_track_c"] = desc.cue_class.isin(TRACK_C)
    return desc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the base dir of the smoke run")
    ap.add_argument("--schema", default=SCHEMA)
    ap.add_argument("--show", type=int, default=12, help="texts to print in full")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run, "runs/*/outputs/extractions/*.parquet")))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    texts = pd.read_parquet(INPUT).set_index("doc_id").text
    d["attr"] = d.attributes_json.apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    spans = d[d.extraction_class.notna()]
    examples = yaml.safe_load(open(args.schema))["examples"]
    example_spans = {e["extraction_text"] for x in examples for e in x["extractions"]}

    per = d.groupby("doc_id").agg(model=("model", "first"), field=("field", "first"),
                                  n=("extraction_class", "count"), q=("is_quotable", "sum"))
    tab = per.groupby(["model", "field"]).agg(texts=("n", "size"), empty=("n", lambda s: int((s == 0).sum())),
                                              spans=("n", "sum"), quotable=("q", "sum"))
    cls = pd.crosstab([spans.model, spans.field], spans.extraction_class)
    print(f"== {per.shape[0]} texts, {len(spans)} spans, {int(spans.is_quotable.sum())} quotable "
          f"({spans.is_quotable.mean():.1%}), schema {d.schema_version.dropna().unique().tolist()}")
    print(tab.join(cls).to_string())
    print("\n== alignment\n" + spans.alignment_status.fillna("none").value_counts().to_string())
    copies = spans[spans.extraction_text.isin(example_spans) & ~spans.is_verbatim]
    print(f"\n== copies of an example span that are not in the text: {len(copies)}")
    if len(copies):
        print(copies[["doc_id", "extraction_text"]].to_string(index=False))
    for c, g in spans.groupby("extraction_class"):
        vals = g["attr"].map(lambda a: ", ".join(f"{k}={v}" for k, v in sorted(a.items())))
        print(f"\n== {c}: attributes\n" + vals.value_counts().head(15).to_string())

    b = border(spans)
    print("\n== border check: the descriptors-v5 spans of these texts, by target and cue class")
    print(pd.crosstab(b.target, b.cue_class, margins=True).to_string())
    print("\nshare in Track C, by target and model:")
    print(b.groupby(["target", "model"]).in_track_c.mean().unstack(0).round(2).to_string())
    bad = b[(b.target == "photo") & b.in_track_c]
    print(f"\n== photo descriptors inside Track C: {len(bad)}")
    if len(bad):
        print(bad[["doc_id", "extraction_text", "cue_class"]].head(30).to_string(index=False))
    lost = b[(b.target == "place") & ~b.in_track_c]
    print(f"\n== place descriptors outside Track C: {len(lost)} (cue class: "
          f"{lost.cue_class.value_counts().to_dict()})")
    if len(lost):
        print(lost[["doc_id", "extraction_text", "cue_class"]].head(30).to_string(index=False))

    show = per.groupby("field", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, args.show // 2)), random_state=args.seed)).index
    for doc in show:
        print("\n" + "=" * 100 + f"\n{doc}\n" + "-" * 100)
        print(texts[doc].strip("\n")[:6000])
        print("-" * 100)
        for _, r in spans[spans.doc_id == doc].iterrows():
            flag = "" if r.is_quotable else f"  [{r.alignment_status}]"
            print(f"  {r.extraction_class:17} {r.extraction_text!r:70} {r["attr"]}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
