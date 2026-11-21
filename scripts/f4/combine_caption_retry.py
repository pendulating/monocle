#!/usr/bin/env python
"""F4: put the retry rows in place of the main rows of the flagged answers.

The main pass (`caption-v3`, job 94267) lost the caption of 64 Llama answers.
The retry pass (`caption-v3r`, D105) read those 64 again. This script writes
the caption table that the NTA analysis reads:

- the rows of every answer that was not flagged come from the main pass;
- the rows of a flagged answer come from the retry pass;
- `pass` names the pass of each row (`main` or `retry`), and `retry_reason`
  the flag (`copied_example` or `missed_offer`);
- `still_failed` is true for a flagged answer that again has no usable caption.

Warning: 2 schema versions meet here on purpose (D105). They differ only in
the order of the examples. The column `schema_version` keeps the version of
each row.

Output `data/f4_traces/captions/f4_captions_final.parquet`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/combine_caption_retry.py
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

ROOT = "/share/ju/matt/mllmsci"
CAPTIONS = os.path.join(ROOT, "data/f4_traces/captions")
MAIN = os.path.join(CAPTIONS, "f4_captions_extractions.parquet")
FLAGS = os.path.join(CAPTIONS, "f4_captions_retry_flags.csv")
RETRY_RUN = os.path.join(ROOT, "multirun/f4_captions_retry_dsv4/runs/f4_captions_retry__0000/outputs/extractions")
OUT = os.path.join(CAPTIONS, "f4_captions_final.parquet")

__version__ = "1.0.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retry-dir", default=RETRY_RUN)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    main_rows = pd.read_parquet(MAIN)
    flags = pd.read_csv(FLAGS)
    files = sorted(glob.glob(os.path.join(args.retry_dir, "*.parquet")))
    if len(files) != 1:
        raise SystemExit(f"want 1 retry parquet in {args.retry_dir}, found {len(files)}")
    retry = pd.read_parquet(files[0])
    missing = set(flags.doc_id) - set(retry.doc_id)
    if missing:
        raise SystemExit(f"{len(missing)} flagged answers have no retry row")

    kept = main_rows[~main_rows.doc_id.isin(set(flags.doc_id))].assign(**{"pass": "main"})
    redo = retry[retry.doc_id.isin(set(flags.doc_id))].assign(**{"pass": "retry"})
    out = pd.concat([kept, redo], ignore_index=True)
    out = out.merge(flags[["doc_id", "reason"]].rename(columns={"reason": "retry_reason"}), on="doc_id", how="left")
    usable = (out.extraction_class == "caption") & out.is_quotable.fillna(False)
    good = out[usable].groupby("doc_id").size()
    out["still_failed"] = out.retry_reason.notna() & ~out.doc_id.isin(good.index)
    if out.doc_id.nunique() != main_rows.doc_id.nunique():
        raise SystemExit("the combined table lost or added an answer")
    out.to_parquet(args.out, index=False)

    f = out[out.retry_reason.notna()].groupby("doc_id").agg(
        model=("model", "first"), reason=("retry_reason", "first"), failed=("still_failed", "first"))
    print(f.groupby(["model", "reason"]).failed.agg(flagged="size", still_failed="sum").to_string())
    print(f"[combine] {out.doc_id.nunique():,} answers, {int(usable.sum()):,} usable captions; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
