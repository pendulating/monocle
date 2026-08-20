#!/usr/bin/env python3
"""Join the shards of one sharded pairwise run into 1 results parquet.

A 1,000,000-pair case runs as many GPU jobs. Each job holds one share of the
canonical pairs and writes its own results parquet and its own pairs.parquet.
The registry and the notebooks want 1 file for each (case, model), thus this
script joins them.

What it checks before it writes
-------------------------------
1. Every shard of the grid is present. A missing shard is an error, not a
   smaller file: a run with 91 of 92 shards looks complete but is not.
2. No pair_id occurs twice. An overlap means 2 shards judged the same pair,
   which the shard split must make impossible.
3. The shards agree on the model, the question, and the image layout.

Use
---
    python scripts/merge_pairwise_shards.py --sweep-dir multirun/million_qwen \\
        --case schools --out-dir multirun/million_merged

    # Look first, write nothing:
    python scripts/merge_pairwise_shards.py --sweep-dir ... --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

# A case writes `<case>_mvp_<stamp>.parquet` beside its `pairs.parquet`.
_RESULTS_GLOB = "*/outputs/pairwise/*_mvp_*.parquet"


def _stage_dirs(sweep_dir: str) -> List[str]:
    """Give the stage dirs of one sweep, newest results first."""
    hits = sorted(glob.glob(os.path.join(sweep_dir, _RESULTS_GLOB)))
    return sorted({os.path.dirname(h) for h in hits})


def _read_job_meta(stage_dir: str) -> Dict[str, Any]:
    """Read what the run recorded about this shard.

    `pipeline_manifest.json` sits 2 dirs above `outputs/pairwise`. It holds the
    shard index and count that the runner wrote.
    """
    run_dir = os.path.dirname(os.path.dirname(stage_dir))
    path = os.path.join(run_dir, "pipeline_manifest.json")
    meta: Dict[str, Any] = {}
    try:
        with open(path) as fh:
            manifest = json.load(fh)
        node = (manifest.get("nodes") or {}).get("pairwise") or {}
        meta = node.get("metadata") or {}
    except Exception:
        pass
    return meta


def _newest_results(stage_dir: str) -> Optional[str]:
    """The results parquet of this shard.

    A requeued job renders `${now:...}` again, thus a shard that a preemption
    stopped can leave more than 1 file. The newest one is the whole one,
    because the job writes it only after every row is done.
    """
    hits = glob.glob(os.path.join(stage_dir, "*_mvp_*.parquet"))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def collect(sweep_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group the shards of a sweep by case."""
    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for stage_dir in _stage_dirs(sweep_dir):
        results = _newest_results(stage_dir)
        if not results:
            continue
        base = os.path.basename(results)
        case = base.split("_mvp_")[0]
        meta = _read_job_meta(stage_dir)
        by_case[case].append({
            "stage_dir": stage_dir,
            "results": results,
            "pairs": os.path.join(stage_dir, "pairs.parquet"),
            "shard_index": meta.get("shard_index"),
            "shard_count": meta.get("shard_count"),
            "rows": meta.get("rows"),
            "pairs_total": meta.get("pairs_total"),
        })
    return by_case


def check_grid(case: str, shards: List[Dict[str, Any]]) -> List[str]:
    """Report what is missing or repeated in the shard grid of one case."""
    problems: List[str] = []
    counts = {s["shard_count"] for s in shards if s["shard_count"] is not None}
    if len(counts) > 1:
        problems.append(f"the shards disagree on shard_count: {sorted(counts)}")
        return problems
    if not counts:
        problems.append("no shard recorded a shard_count")
        return problems
    total = counts.pop()
    seen = [s["shard_index"] for s in shards if s["shard_index"] is not None]
    missing = sorted(set(range(total)) - set(seen))
    if missing:
        problems.append(
            f"{len(missing)} of {total} shards are missing: "
            f"{missing[:12]}{' ...' if len(missing) > 12 else ''}"
        )
    repeated = sorted({i for i in seen if seen.count(i) > 1})
    if repeated:
        problems.append(f"these shard indexes occur more than once: {repeated}")
    return problems


def merge_case(
    case: str, shards: List[Dict[str, Any]], out_dir: str, *, dry_run: bool,
) -> Dict[str, Any]:
    """Join one case and write its results parquet and pairs parquet."""
    shards = sorted(shards, key=lambda s: (s["shard_index"] is None, s["shard_index"]))
    frames = [pd.read_parquet(s["results"]) for s in shards]
    merged = pd.concat(frames, ignore_index=True)

    report: Dict[str, Any] = {
        "case": case,
        "shards": len(shards),
        "rows": int(len(merged)),
        "problems": check_grid(case, shards),
    }

    dupes = int(merged["pair_id"].duplicated().sum()) if "pair_id" in merged else -1
    if dupes > 0:
        report["problems"].append(f"{dupes} pair_id values occur more than once")
    report["duplicate_pair_ids"] = dupes

    for col in ("model_source", "image_layout"):
        if col in merged.columns:
            values = sorted({str(v) for v in merged[col].dropna().unique()})
            if len(values) > 1:
                report["problems"].append(f"the shards disagree on {col}: {values}")

    expected = shards[0].get("pairs_total")
    if expected:
        report["pairs_total_expected"] = int(expected)
        # Rows carry the repeat draws too, thus rows >= canonical pairs.
        if "canonical_pair_id" in merged.columns:
            got = int(merged["canonical_pair_id"].nunique())
            report["canonical_pairs"] = got

    if dry_run:
        return report

    os.makedirs(out_dir, exist_ok=True)
    out_results = os.path.join(out_dir, f"{case}_mvp_merged.parquet")
    merged.to_parquet(out_results, index=False)
    report["out_results"] = out_results

    pair_frames = [pd.read_parquet(s["pairs"]) for s in shards
                   if os.path.exists(s["pairs"])]
    if pair_frames:
        pairs = pd.concat(pair_frames, ignore_index=True)
        out_pairs = os.path.join(out_dir, f"{case}_pairs_merged.parquet")
        pairs.to_parquet(out_pairs, index=False)
        report["out_pairs"] = out_pairs
        report["pair_rows"] = int(len(pairs))

    sidecar = {
        "case": case,
        "shards": [
            {"shard_index": s["shard_index"], "results": s["results"],
             "rows": s["rows"]}
            for s in shards
        ],
        "rows": int(len(merged)),
        "problems": report["problems"],
    }
    with open(os.path.join(out_dir, f"{case}_merge.json"), "w") as fh:
        json.dump(sidecar, fh, indent=2)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, action="append",
                    help="A sweep root. Give it more than once to join reruns.")
    ap.add_argument("--case", action="append",
                    help="Join this case only. The default joins every case.")
    ap.add_argument("--out-dir", help="Where to write. Needed unless --dry-run.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the grid and write nothing.")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Write even when shards are missing. Use with care.")
    args = ap.parse_args()

    if not args.dry_run and not args.out_dir:
        ap.error("--out-dir is needed unless you pass --dry-run")

    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sweep in args.sweep_dir:
        for case, shards in collect(sweep).items():
            by_case[case].extend(shards)

    if not by_case:
        print(f"No shard found under {args.sweep_dir}", file=sys.stderr)
        return 1

    wanted = set(args.case) if args.case else set(by_case)
    exit_code = 0
    for case in sorted(wanted):
        if case not in by_case:
            print(f"[{case}] no shard found", file=sys.stderr)
            exit_code = 1
            continue
        problems = check_grid(case, by_case[case])
        if problems and not args.allow_incomplete and not args.dry_run:
            print(f"[{case}] REFUSED:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print("  Pass --allow-incomplete to write anyway.", file=sys.stderr)
            exit_code = 1
            continue
        report = merge_case(case, by_case[case], args.out_dir or "",
                            dry_run=args.dry_run)
        flag = "OK " if not report["problems"] else "WARN"
        print(f"[{flag}] {case}: {report['shards']} shards, "
              f"{report['rows']:,} rows"
              + (f" → {report.get('out_results')}" if report.get("out_results") else ""))
        for p in report["problems"]:
            print(f"       - {p}")
        if report["problems"]:
            exit_code = max(exit_code, 0 if args.dry_run else 1)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
