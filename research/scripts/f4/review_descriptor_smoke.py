#!/usr/bin/env python
"""F4: print a review of a descriptor smoke run, for a person to read.

The review holds:

- the counts for each (model, case, field): texts, texts with no descriptor,
  descriptors, and quotable descriptors;
- the copies: a span that equals an example span and is not verbatim;
- the attribute mix (vibe, aspect, negated) and the top heads and slots;
- `--show N` texts in full, each followed by its descriptors.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/review_descriptor_smoke.py \
        --run multirun/f4_descriptors_dsv4_smoke --show 12 > review.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCHEMA = os.path.join(ROOT, "research/conf/pairwisevqa/extract/aesthetic_descriptors_v3.yaml")
INPUT = os.path.join(ROOT, "data/f4_traces/extract_input/f4_descsmoke_desc.parquet")

__version__ = "1.0.0"


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
    attrs = d.attributes_json.apply(lambda s: json.loads(s) if isinstance(s, str) else {})
    for k in ("head", "vibe", "slot", "aspect", "negated"):
        d[k] = attrs.apply(lambda a: a.get(k))
    spans = d[d.extraction_class.notna()]
    examples = yaml.safe_load(open(args.schema))["examples"]
    example_spans = {e["extraction_text"] for x in examples for e in x["extractions"]}

    per = d.groupby("doc_id").agg(model=("model", "first"), case=("case", "first"), field=("field", "first"),
                                  n=("extraction_class", "count"), q=("is_quotable", "sum"))
    tab = per.groupby(["model", "case", "field"]).agg(texts=("n", "size"), empty=("n", lambda s: int((s == 0).sum())),
                                                      descriptors=("n", "sum"), quotable=("q", "sum"))
    print(f"== {per.shape[0]} texts, {len(spans)} descriptors, {int(spans.is_quotable.sum())} quotable "
          f"({spans.is_quotable.mean():.1%}), schema {d.schema_version.dropna().unique().tolist()}")
    print(tab.to_string())
    print("\n== alignment\n" + spans.alignment_status.fillna("none").value_counts().to_string())
    copies = spans[spans.extraction_text.isin(example_spans) & ~spans.is_verbatim]
    print(f"\n== copies of an example span that are not in the text: {len(copies)}")
    if len(copies):
        print(copies[["doc_id", "extraction_text"]].to_string(index=False))
    for k in ("vibe", "aspect", "negated"):
        print(f"\n== {k}\n" + spans[k].fillna("(none)").value_counts().to_string())
    print("\n== top 40 heads\n" + spans["head"].str.lower().value_counts().head(40).to_string())
    print("\n== slots (not none)\n" + spans.loc[spans.slot.str.lower() != "none", "slot"].str.lower()
          .value_counts().head(40).to_string())
    print("\n== top vibe heads\n" + spans.loc[spans.vibe == "yes", "head"].str.lower().value_counts().head(30).to_string())

    show = per.groupby(["case", "field"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, args.show // 4)), random_state=args.seed)).index
    for doc in show:
        print("\n" + "=" * 100 + f"\n{doc}\n" + "-" * 100)
        print(texts[doc].strip("\n")[:6000])
        print("-" * 100)
        for _, r in spans[spans.doc_id == doc].iterrows():
            flag = "" if r.is_quotable else f"  [{r.alignment_status}]"
            print(f"  {r.extraction_text!r:50} head={r['head']!r} vibe={r.vibe} slot={r.slot!r} "
                  f"aspect={r.aspect} neg={r.negated}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
