#!/usr/bin/env python
"""F4: find every hashtag in the final answers of the caption case, by regex.

A hashtag has a fixed form, thus a regex finds each one exactly and needs no
model (D101). The LangExtract caption job also extracts the hashtag blocks;
compare the 2 to check that job.

The pattern: a `#`, then letters, digits, or `_`, with at least 1 letter.
The `#` must not follow a letter, a digit, `_`, or `&` (thus `C#`, `&#39;`, and
`Option #1` do not count). `\\w` is Unicode, thus `#café` is 1 hashtag. A
space ends a hashtag: `#nyc vibes` gives `#nyc`.

Output `data/f4_traces/captions/f4_caption_hashtags.parquet`, 1 row for each
hashtag in each answer: `doc_id`, `model`, `image_idx`, `hashtag` (as
written), `tag` (lower case, no `#`), `char_start`, `char_end`, `position`
(the order in the answer, from 0). An answer with no hashtag has no row; the
summary counts those answers.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/caption_hashtags.py
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/caption_hashtags.py --traces data/f4_traces/b2/f4_traces_b2.parquet \
        --out data/f4_traces/b2/f4_caption_hashtags_b2.parquet

`--case street_caption` reads the caption prompt on the street views (D133).
Give the street corpus and an `--out` of its own:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/caption_hashtags.py --case street_caption \
        --traces data/f4_traces/f4_street_traces_1k.parquet \
        --out data/f4_traces/street_captions/f4_street_caption_hashtags.parquet
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

ROOT = "/share/ju/matt/mllmsci"
TRACES = os.path.join(ROOT, "data/f4_traces/f4_traces_1k.parquet")
OUT = os.path.join(ROOT, "data/f4_traces/captions/f4_caption_hashtags.parquet")

HASHTAG = re.compile(r"(?<![\w&])#(?=\w*[^\W\d_])\w+")

__version__ = "1.2.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=TRACES, help="the trace corpus (batch 2: data/f4_traces/b2/f4_traces_b2.parquet)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--case", default="caption", choices=["caption", "street_caption"])
    args = ap.parse_args()
    if args.case != "caption" and os.path.abspath(args.out) == OUT:
        raise SystemExit("give --out: the default path holds the hashtags of the caption case")

    t = pd.read_parquet(args.traces, columns=["model", "case", "image_idx", "response"])
    t = t[t.case == args.case]
    rows = []
    for r in t.itertuples(index=False):
        doc = f"{r.model}|{args.case}|{r.image_idx}|response"
        for k, m in enumerate(HASHTAG.finditer(r.response or "")):
            rows.append({"doc_id": doc, "model": r.model, "image_idx": r.image_idx,
                         "hashtag": m.group(0), "tag": m.group(0)[1:].lower(),
                         "char_start": m.start(), "char_end": m.end(), "position": k})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)

    per = out.groupby(["model", "image_idx"]).size()
    summary = pd.DataFrame({
        "answers": t.groupby("model").size(),
        "with_hashtag": per.groupby(level=0).size(),
        "hashtags": out.groupby("model").size(),
        "distinct_tags": out.groupby("model").tag.nunique(),
        "per_answer_p50": per.groupby(level=0).median(),
    }).fillna(0).astype(int)
    print(summary.to_string())
    print(f"[hashtags] {len(out):,} hashtags, {out.tag.nunique():,} distinct tags; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
