#!/usr/bin/env python3
"""Cut a smaller pair table out of a larger draw, row for row.

Why
---
The 100,000-pair draw at seed 777 holds the same CANONICAL pairs as the
1,000,000-pair draw, but about 8% of them come up in the opposite A/B order,
thus only 91.7% of the `pair_id` rows match. An arm that must join the
1,000,000-pair battery pair for pair cannot use a fresh draw. It must use the
rows of the parent table itself.

This script keeps every row of the PARENT table whose `canonical_pair_id` is
in the SELECT table. Every kept row is a byte-for-byte row of the parent: the
same images, the same presentation order, the same `pair_id`, the same repeat
draws. Thus a run on the cut joins the parent run on `pair_id` at 100%.

What it writes
--------------
    <out-dir>/<case>_pairs.parquet   the cut
    <out-dir>/<case>_pairs.json      the PARENT fingerprint plus a `subset` block

The runner (`_load_pair_table`) accepts the cut for a job whose settings equal
the parent's in everything but `max_pairs`, and whose `max_pairs` equals the
number of canonical pairs the cut kept. `prebuild_pair_tables.py` refuses to
draw over a subset sidecar.

Use
---
    python scripts/subset_pair_tables.py \\
        --parent multirun/pair_tables_1m --select multirun/pair_tables_100k \\
        --out-dir multirun/pair_tables_100k_of_1m
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List, Optional

import pandas as pd


def cut_one(parent_dir: str, select_dir: str, out_dir: str, case: str) -> dict:
    t0 = time.time()
    parent_pq = os.path.join(parent_dir, f"{case}_pairs.parquet")
    select_pq = os.path.join(select_dir, f"{case}_pairs.parquet")
    with open(os.path.join(parent_dir, f"{case}_pairs.json")) as fh:
        parent_doc = json.load(fh)
    parent = pd.read_parquet(parent_pq)
    select = pd.read_parquet(select_pq, columns=["canonical_pair_id"])
    ids = set(select["canonical_pair_id"])
    missing = len(ids - set(parent["canonical_pair_id"]))
    if missing:
        raise SystemExit(f"{case}: {missing} selected canonical pairs are not "
                         f"in the parent. The select table is not nested.")
    cut = parent[parent["canonical_pair_id"].isin(ids)].reset_index(drop=True)

    os.makedirs(out_dir, exist_ok=True)
    out_pq = os.path.join(out_dir, f"{case}_pairs.parquet")
    out_json = os.path.join(out_dir, f"{case}_pairs.json")
    tmp = f"{out_pq}.tmp{os.getpid()}"
    cut.to_parquet(tmp, index=False)
    os.replace(tmp, out_pq)
    with open(out_json, "w") as fh:
        json.dump({
            "case": case,
            "rows": int(len(cut)),
            # The PARENT fingerprint, on purpose: the rows come from that draw.
            "fingerprint": parent_doc["fingerprint"],
            "subset": {
                "of": os.path.abspath(parent_pq),
                "parent_rows": int(len(parent)),
                "selected_by": f"canonical_pair_id in {os.path.abspath(select_pq)}",
                "canonical_pairs": int(cut["canonical_pair_id"].nunique()),
                "rows": int(len(cut)),
            },
        }, fh, indent=2)
    return {"case": case, "rows": int(len(cut)),
            "canonical_pairs": int(cut["canonical_pair_id"].nunique()),
            "seconds": round(time.time() - t0, 1)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True, help="dir of the larger draw")
    ap.add_argument("--select", required=True, help="dir of the smaller draw")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args(argv)

    cases = args.cases or sorted(
        os.path.basename(p)[: -len("_pairs.parquet")]
        for p in glob.glob(os.path.join(args.select, "*_pairs.parquet")))
    for case in cases:
        r = cut_one(args.parent, args.select, args.out_dir, case)
        print(f"[subset] {r['case']:20s} rows={r['rows']:7d} "
              f"canonical={r['canonical_pairs']:7d} {r['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
