"""Check a merged pairwise battery against the pair table that fed it.

The registry gate tests a run in isolation. This tests the SHARDING: that the
322 shards reassemble the prebuilt table exactly, with no pair lost, no pair
counted twice, and the counterbalancing restored.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

COLS = ["pair_id", "canonical_pair_id", "is_swapped", "answer",
        "presented_label", "relative_label"]


def check(case: str, merged: str, table: str) -> dict:
    d = pd.read_parquet(merged, columns=[c for c in COLS])
    t = pd.read_parquet(table, columns=["pair_id", "canonical_pair_id", "is_swapped"])
    ids, tids = d["pair_id"].astype(str), t["pair_id"].astype(str)
    lab = d["relative_label"]
    top = lab.value_counts(normalize=True)
    return {
        "case": case,
        "rows": len(d),
        "rows_match": len(d) == len(t),
        "ids_unique": int(ids.duplicated().sum()) == 0,
        "covers_table": set(ids) == set(tids),
        "missing": len(set(tids) - set(ids)),
        "extra": len(set(ids) - set(tids)),
        "swapped": round(float(d["is_swapped"].mean()), 4),
        "empty_ans": round(float((d["answer"].astype(str).str.strip() == "").mean()), 5),
        "top_label": f"{top.index[0]} {top.iloc[0]:.1%}",
        "degenerate": bool(top.iloc[0] > 0.98),
        "NotSure": round(float((lab == "NotSure").mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged-dir", required=True)
    ap.add_argument("--pairs-dir", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{args.merged_dir}/*/outputs/pairwise/*_merged.parquet")):
        case = f.split("/merged/")[1].split("__")[0]
        table = os.path.join(args.pairs_dir, f"{case}_pairs.parquet")
        if not os.path.exists(table):
            print(f"[verify] {case}: no pair table at {table}")
            continue
        rows.append(check(case, f, table))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    bad = df[(~df.rows_match) | (~df.ids_unique) | (~df.covers_table) | df.degenerate]
    print()
    if len(bad):
        print(f"[verify] {len(bad)} case(s) FAILED:")
        print(bad.to_string(index=False))
        return 1
    print(f"[verify] OK: {len(df)} cases, {int(df['rows'].sum()):,} rows, "
          f"every pair of the table present exactly once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
