#!/usr/bin/env python
"""F4: flag the caption answers that lost a caption, and write the retry input.

The full `caption-v3` run (job 94267) lost the caption of about 64 Llama
answers in 2 ways (D105):

| Reason | Test |
|--------|------|
| `copied_example` | a span equals a span of the examples and does not occur in the answer |
| `missed_offer` | no extraction, and a regex finds a caption that the answer offers: a quote after "caption ...:", a fully quoted answer, or a hashtag |

An answer is flagged only when it also has no usable caption (a quotable
`caption` span). The retry pass reads the flagged answers again with
`caption-v3r` (the same schema, the examples in a new order).

Output:

| File | Content |
|------|---------|
| `data/f4_traces/extract_input/f4_captions_retry.parquet` | the flagged rows of `f4_captions.parquet` |
| `data/f4_traces/captions/f4_captions_retry_flags.csv` | `doc_id`, `model`, `reason` |

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/flag_caption_retry.py
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
EXTRACTIONS = os.path.join(ROOT, "data/f4_traces/captions/f4_captions_extractions.parquet")
INPUT = os.path.join(ROOT, "data/f4_traces/extract_input/f4_captions.parquet")
SCHEMA = os.path.join(ROOT, "dagspaces/urbanpairvqa/conf/extract/caption_options_v3.yaml")
OUT_INPUT = os.path.join(ROOT, "data/f4_traces/extract_input/f4_captions_retry.parquet")
OUT_FLAGS = os.path.join(ROOT, "data/f4_traces/captions/f4_captions_retry_flags.csv")

# A caption that the answer offers: a quote after "caption ...:" or
# "instagram ...:", a hashtag, or an answer that opens with a quote mark.
OFFER = re.compile(r'(?i)(caption|instagram)[^\n]{0,80}?:\s*\n*\s*["“]|#\w|^\s*["“]')

__version__ = "1.0.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extractions", default=EXTRACTIONS)
    args = ap.parse_args()

    d = pd.read_parquet(args.extractions)
    texts = pd.read_parquet(INPUT)
    by_id = texts.set_index("doc_id").text
    examples = yaml.safe_load(open(SCHEMA))["examples"]
    example_spans = {e["extraction_text"] for x in examples for e in x["extractions"]}

    d["copied"] = d.extraction_text.isin(example_spans) & ~d.is_verbatim.fillna(False)
    d["usable"] = (d.extraction_class == "caption") & d.is_quotable.fillna(False)
    per = d.groupby("doc_id").agg(model=("model", "first"), copied=("copied", "any"),
                                  usable=("usable", "sum"),
                                  silent=("extraction_class", lambda s: s.isna().all()))
    per["offers"] = [bool(OFFER.search(by_id[i])) for i in per.index]
    per["reason"] = None
    per.loc[per.silent & per.offers, "reason"] = "missed_offer"
    per.loc[per.copied, "reason"] = "copied_example"
    flagged = per[(per.usable == 0) & per.reason.notna()]

    texts[texts.doc_id.isin(flagged.index)].to_parquet(OUT_INPUT, index=False)
    flagged.reset_index()[["doc_id", "model", "reason"]].to_csv(OUT_FLAGS, index=False)
    print(flagged.groupby(["model", "reason"]).size().to_string())
    print(f"[retry] {len(flagged)} answers flagged; wrote {OUT_INPUT} and {OUT_FLAGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
