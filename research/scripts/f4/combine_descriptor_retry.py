#!/usr/bin/env python
"""F4: put the retry rows in place of the main rows of the flagged texts (D121).

The main pass (`descriptors-v3`, job 94638) gave 20 restaurant traces a copy
of an example. The retry pass (`descriptors-v3r`) read those 20 again. This
script writes the descriptor table that the analysis reads:

- the rows of every text that was not flagged come from the main pass;
- the rows of a flagged text come from the retry pass;
- `pass` names the pass of each row (`main` or `retry`);
- `still_copied` is true for a flagged text whose retry again holds
  `--min-copies` or more example spans that are not in the text.

Warning: 2 schema versions meet here on purpose (D121). They differ only in
the order of the examples. The column `schema_version` keeps the version of
each row.

Output `data/f4_traces/descriptors/f4_restaurant_desc_final.parquet`.

`--set street` does the same for the street case (`descriptors-v5` and
`descriptors-v5r`, D124) and writes
`data/f4_traces/descriptors_street/f4_street_desc_final.parquet`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/combine_descriptor_retry.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/combine_descriptor_retry.py --set street
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
# set -> (main extractions, flags, retry run, retry schema, output)
SETS = {
    "restaurant": ("data/f4_traces/descriptors/f4_restaurant_desc_extractions.parquet",
                   "data/f4_traces/descriptors/f4_restaurant_desc_retry_flags.csv",
                   "multirun/f4_descriptors_retry_dsv4/runs/f4_descriptors_retry__0000/outputs/extractions",
                   "dagspaces/pairwisevqa/conf/extract/aesthetic_descriptors_v3r.yaml",
                   "data/f4_traces/descriptors/f4_restaurant_desc_final.parquet"),
    "street": ("data/f4_traces/descriptors_street/f4_street_desc_extractions.parquet",
               "data/f4_traces/descriptors_street/f4_street_desc_retry_flags.csv",
               "multirun/f4_descriptors_street_retry_dsv4/runs/f4_descriptors_street_retry__0000/outputs/extractions",
               "dagspaces/pairwisevqa/conf/extract/aesthetic_descriptors_v5r.yaml",
               "data/f4_traces/descriptors_street/f4_street_desc_final.parquet"),
}

__version__ = "1.2.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), default="restaurant")
    ap.add_argument("--retry-dir", default=None)
    ap.add_argument("--min-copies", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    MAIN, FLAGS, RETRY_RUN, SCHEMA, OUT = (os.path.join(ROOT, x) for x in SETS[args.set])
    args.retry_dir = args.retry_dir or RETRY_RUN
    args.out = args.out or OUT

    main_rows = pd.read_parquet(MAIN)
    flags = pd.read_csv(FLAGS)
    if flags.empty:
        # No text was flagged (the v5 street run): the main pass is the table.
        out = main_rows.assign(**{"pass": "main", "still_copied": False})
        out.to_parquet(args.out, index=False)
        print(f"[combine] {out.doc_id.nunique():,} texts; 0 flagged; wrote {args.out}")
        return 0
    files = sorted(glob.glob(os.path.join(args.retry_dir, "*.parquet")))
    if len(files) != 1:
        raise SystemExit(f"want 1 retry parquet in {args.retry_dir}, found {len(files)}")
    retry = pd.read_parquet(files[0])
    ids = set(flags.doc_id)
    if ids - set(retry.doc_id):
        raise SystemExit(f"{len(ids - set(retry.doc_id))} flagged texts have no retry row")

    kept = main_rows[~main_rows.doc_id.isin(ids)].assign(**{"pass": "main"})
    redo = retry[retry.doc_id.isin(ids)].assign(**{"pass": "retry"})
    spans = {x["extraction_text"] for e in yaml.safe_load(open(SCHEMA))["examples"] for x in e["extractions"]}
    copied = redo.extraction_text.isin(spans) & ~redo.is_verbatim.fillna(False)
    n_copied = copied.groupby(redo.doc_id).sum()
    out = pd.concat([kept, redo], ignore_index=True)
    out["still_copied"] = out.doc_id.isin(n_copied[n_copied >= args.min_copies].index)
    if out.doc_id.nunique() != main_rows.doc_id.nunique():
        raise SystemExit("the combined table lost or added a text")
    out.to_parquet(args.out, index=False)

    r = redo[redo.extraction_class.notna()]
    per = r.groupby("doc_id").agg(spans=("is_quotable", "size"), quotable=("is_quotable", "sum"))
    per = per.reindex(sorted(ids), fill_value=0).assign(copied=n_copied.reindex(sorted(ids), fill_value=0))
    print(per.to_string())
    print(f"[combine] {out.doc_id.nunique():,} texts; {len(ids)} from the retry; "
          f"{int(out.drop_duplicates('doc_id').still_copied.sum())} still copied; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
