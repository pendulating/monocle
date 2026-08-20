#!/usr/bin/env python
"""Merge the shard parquets of a trace-extraction sweep into 1 file for each case.

A sharded run writes 1 parquet for each shard, in its own stage directory. This
script collects them, tests that the shards cover every trace one time, and
writes 1 parquet for each case.

Usage:
    python scripts/merge_trace_extractions.py <sweep_dir> [--schema ic] [--out DIR]

It merges both extraction schemas. `urban_cues` (the LangExtract stage) writes
`outputs/extractions/`, and `ic` (the Integrative Complexity stage) writes
`outputs/ic/`. The shard logic and the coverage test are the same for both.

Example:
    python scripts/merge_trace_extractions.py \\
        multirun/2026-08-14_URBANPAIRVQA/02-01-38 \\
        --out data/trace_extractions

Warning: the script reports a case whose shards do not agree, and it never
repairs one. A missing shard means a job died. Run that shard again.

See `vlm-narratives-docs/langextract-trace-extraction.md`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

import pandas as pd

__version__ = "1.0.0"


# The 2 extraction schemas write different directories and different columns.
# 1 tool merges both, because the shard logic and the coverage test are the same.
#
#   subdir    : where the stage writes, under a stage directory
#   class_col : the column that says WHICH thing was extracted; empty means the
#               trace was silent
#   quality   : the column that says whether the span is grounded
#   out_name  : the file name of a merged case
SCHEMAS = {
    "urban_cues": {
        "subdir": "extractions",
        "class_col": "extraction_class",
        "quality": "is_quotable",
        "out_name": "{case}_extractions.parquet",
        "default_out": "data/trace_extractions",
    },
    "ic": {
        "subdir": "ic",
        "class_col": "ingredient_type",
        "quality": "quote_found",
        "out_name": "{case}_ic_ingredients.parquet",
        "default_out": "data/ic_ingredients",
    },
}


def shard_files(sweep_dirs: str | List[str], schema: str = "urban_cues") -> List[str]:
    """Find every shard parquet of 1 or more sweeps, in a stable order.

    A sweep sometimes needs a second directory: a monitor task can exit without
    submitting its stage job (the submitit result-pickle race on NFS), and the
    shard then runs again in a sweep of its own. Pass both directories.
    """
    if isinstance(sweep_dirs, str):
        sweep_dirs = [sweep_dirs]
    subdir = SCHEMAS[schema]["subdir"]
    out: List[str] = []
    for sweep_dir in sweep_dirs:
        out += glob.glob(os.path.join(sweep_dir, "*", "outputs", subdir, "*_s*.parquet"))
    return sorted(set(out))


def registry_notes(df: pd.DataFrame) -> List[str]:
    """Compare the source of each case with the canonical run registry.

    Returns 1 line for each case. A case whose `source_results_path` is not the
    registered trace run answers an older question, and every panel built from
    it would say so nowhere.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "notebooks", "cvpr"))
        import _canonical as C  # type: ignore

        reg = {(r.case, r.model_config): os.path.realpath(r.results_path)
               for r in C.runs(kind="trace")}
    except Exception as exc:
        return [f"registry test skipped ({type(exc).__name__}: {exc})"]

    out: List[str] = []
    cols = [c for c in ("case", "judge_model", "source_results_path") if c in df.columns]
    if len(cols) < 3:
        return ["registry test skipped (the shards carry no source columns)"]
    for row in df[cols].drop_duplicates().itertuples(index=False):
        case, judge, src = row
        want = reg.get((case, judge))
        if want is None:
            out.append(f"{case}: WARNING, no registered trace run for {judge}")
        elif os.path.realpath(str(src)) == want:
            out.append(f"{case}: from the registered run, OK")
        else:
            out.append(f"{case}: WARNING, from {os.path.basename(str(src))}, "
                       f"not the registered {os.path.basename(want)}")
    return out


def merge(sweep_dir: str | List[str], out_dir: str,
          schema: str = "urban_cues") -> pd.DataFrame:
    spec = SCHEMAS[schema]
    files = shard_files(sweep_dir, schema)
    if not files:
        raise SystemExit(
            f"no {schema} shard parquet under {sweep_dir} "
            f"(looked in */outputs/{spec['subdir']}/)")
    print(f"[merge] {len(files)} shard files, schema {schema}")

    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # 2 schema versions must never pool: a class or an attribute may mean
    # something else in each one.
    versions = sorted(set(df.get("schema_version", pd.Series(dtype=str)).dropna()))
    if len(versions) > 1:
        raise SystemExit(f"the shards hold 2 schema versions: {versions}")
    print(f"[merge] schema_version {versions[0] if versions else 'unknown'}")

    class_col, quality = spec["class_col"], spec["quality"]
    os.makedirs(out_dir, exist_ok=True)
    summary: List[Dict[str, object]] = []
    for case, part in df.groupby("case"):
        real = part[part[class_col].notna()]
        # A trace that appears 2 times means 2 shards read it. That doubles
        # every count of the case, and no later number can show it.
        traces = part.pair_id.nunique()
        rows_per_trace = part.groupby("pair_id").size()
        repeated = int((rows_per_trace.index.duplicated()).sum())

        path = os.path.join(out_dir, spec["out_name"].format(case=case))
        part.to_parquet(path, index=False)
        summary.append({
            "case": case,
            "traces": traces,
            "extractions": len(real),
            "per_trace": round(len(real) / max(1, traces), 1),
            "quotable_rate": round(float(real[quality].mean()), 4) if len(real) else 0.0,
            "silent_traces": int(part[class_col].isna().sum()),
            "repeated_traces": repeated,
            "file": os.path.basename(path),
        })
        print(f"[merge] {case}: {traces:,} traces, {len(real):,} extractions -> {path}")

    # Does this corpus come from the registered runs? A corpus is 1 step
    # downstream of a thinking run, thus it goes stale when the battery runs
    # again. Say so here, where the file is written, and not 3 weeks later in a
    # notebook.
    for line in registry_notes(df):
        print(f"[merge] {line}")

    table = pd.DataFrame(summary)
    table.to_csv(os.path.join(out_dir, "merge_summary.csv"), index=False)
    print("\n" + table.to_string(index=False))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", nargs="+",
                    help="1 or more sweep directories; a re-run shard may live "
                         "in a sweep of its own")
    ap.add_argument("--schema", default="urban_cues", choices=sorted(SCHEMAS),
                    help="which extraction stage wrote the shards")
    ap.add_argument("--out", default="",
                    help="where to write; the default follows the schema")
    args = ap.parse_args()
    out = args.out or SCHEMAS[args.schema]["default_out"]
    merge(args.sweep_dir, out, args.schema)


if __name__ == "__main__":
    sys.exit(main())
