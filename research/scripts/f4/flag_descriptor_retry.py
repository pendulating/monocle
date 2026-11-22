#!/usr/bin/env python
"""F4: flag the texts whose descriptors copy an example, and write the retry input (D121).

The full `descriptors-v3` run (job 94638) gave 20 reasoning traces a copy of
an example: 9 to 16 example spans that are not in the text. A text is
flagged when `--min-copies` or more of its spans equal an example span and
are not verbatim. Every other text has 0 to 2 such spans, and those are case
variants that still align ("Cozy" for "cozy").

Output:

| File | Content |
|------|---------|
| `data/f4_traces/extract_input/f4_restaurant_desc_retry.parquet` | the flagged rows of `f4_restaurant_desc.parquet` |
| `data/f4_traces/descriptors/f4_restaurant_desc_retry_flags.csv` | `doc_id`, `model`, `field`, `copies` |

`--set street` does the same for the street case (`descriptors-v5`, D124):
it reads `data/f4_traces/descriptors_street/` and writes
`f4_street_desc_retry.parquet` and `f4_street_desc_retry_flags.csv`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/flag_descriptor_retry.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/flag_descriptor_retry.py --set street
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
# set -> (extractions, input, schema, retry input, flags)
SETS = {
    "restaurant": ("data/f4_traces/descriptors/f4_restaurant_desc_extractions.parquet",
                   "data/f4_traces/extract_input/f4_restaurant_desc.parquet",
                   "dagspaces/pairwisevqa/conf/extract/aesthetic_descriptors_v3.yaml",
                   "data/f4_traces/extract_input/f4_restaurant_desc_retry.parquet",
                   "data/f4_traces/descriptors/f4_restaurant_desc_retry_flags.csv"),
    "street": ("data/f4_traces/descriptors_street/f4_street_desc_extractions.parquet",
               "data/f4_traces/extract_input/f4_street_desc.parquet",
               "dagspaces/pairwisevqa/conf/extract/aesthetic_descriptors_v5.yaml",
               "data/f4_traces/extract_input/f4_street_desc_retry.parquet",
               "data/f4_traces/descriptors_street/f4_street_desc_retry_flags.csv"),
}

__version__ = "1.1.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-copies", type=int, default=5)
    ap.add_argument("--set", choices=sorted(SETS), default="restaurant")
    args = ap.parse_args()
    EXTRACTIONS, INPUT, SCHEMA, OUT_INPUT, OUT_FLAGS = (os.path.join(ROOT, x) for x in SETS[args.set])

    d = pd.read_parquet(EXTRACTIONS)
    d = d[d.extraction_class.notna()].copy()
    examples = yaml.safe_load(open(SCHEMA))["examples"]
    spans = {x["extraction_text"] for e in examples for x in e["extractions"]}
    d["copied"] = d.extraction_text.isin(spans) & ~d.is_verbatim
    per = d.groupby("doc_id").agg(model=("model", "first"), field=("field", "first"), copies=("copied", "sum"))
    flagged = per[per.copies >= args.min_copies].reset_index()

    texts = pd.read_parquet(INPUT)
    texts[texts.doc_id.isin(set(flagged.doc_id))].to_parquet(OUT_INPUT, index=False)
    flagged.to_csv(OUT_FLAGS, index=False)
    print(flagged.groupby(["model", "field"]).size().to_string())
    print(f"[retry] {len(flagged)} texts flagged; wrote {OUT_INPUT} and {OUT_FLAGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
