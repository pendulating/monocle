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
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

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


_SHARD_DIR_RE = re.compile(r"__(\d+)$")


def _shard_index_from_dir(stage_dir: str) -> Optional[int]:
    """Read the shard index out of the run directory name.

    `submit_shards` names a run directory `<case>__<index>`, such as
    `parks_plazas__0051`. The index is also in `pipeline_manifest.json`, but a
    job that dies after it writes the parquet and before it writes the manifest
    leaves no manifest at all. Measured 2026-08-20: 7 of 966 shards of the
    1,000,000-pair battery are in that state. Their data is whole, thus the
    name is the right fallback and NOT a reason to call the grid incomplete.
    """
    run_dir = os.path.dirname(os.path.dirname(stage_dir))
    m = _SHARD_DIR_RE.search(os.path.basename(run_dir))
    return int(m.group(1)) if m else None


def _read_pipeline(stage_dir: str) -> str:
    """Read the pipeline name from the `.hydra` record of one shard.

    Do NOT rebuild this name from the results file. A `*_nocity` pipeline
    inherits its base case and keeps the output name of that case, thus
    `libraries_mvp_...parquet` comes from BOTH `pairwise_libraries_mvp` and
    `pairwise_libraries_mvp_nocity`. Only the `.hydra` record tells them apart,
    and the registry reads the prompt through that name.
    """
    run_dir = os.path.dirname(os.path.dirname(stage_dir))
    f = os.path.join(run_dir, ".hydra", "overrides.yaml")
    try:
        for raw in (yaml.safe_load(open(f)) or []):
            if isinstance(raw, str) and raw.startswith("pipeline="):
                return raw.partition("=")[2].strip()
    except Exception:
        pass
    return ""


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
        shard_index = meta.get("shard_index")
        if shard_index is None:
            shard_index = _shard_index_from_dir(stage_dir)
        by_case[case].append({
            "pipeline": _read_pipeline(stage_dir),
            "stage_dir": stage_dir,
            "results": results,
            "pairs": os.path.join(stage_dir, "pairs.parquet"),
            "shard_index": shard_index,
            "shard_count": meta.get("shard_count"),
            "rows": meta.get("rows"),
            "pairs_total": meta.get("pairs_total"),
        })
    # A shard with no manifest also carries no shard_count. Every shard of a
    # case agrees on it, thus take the value the other shards report.
    for shards in by_case.values():
        counts = Counter(s["shard_count"] for s in shards
                         if s["shard_count"] is not None)
        if len(counts) == 1:
            only = next(iter(counts))
            for s in shards:
                if s["shard_count"] is None:
                    s["shard_count"] = only
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


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sweep_model(sweep: str) -> str:
    """Read the model config that a sweep pins, such as `qwen3.5-9b/instruct`."""
    f = os.path.join(REPO_ROOT, "dagspaces", "urbanpairvqa", "conf", "sweep",
                     f"{sweep}.yaml")
    with open(f) as fh:
        cfg = yaml.safe_load(fh) or {}
    model = (((cfg.get("hydra") or {}).get("sweeper") or {})
             .get("params") or {}).get("model")
    if not model:
        raise SystemExit(f"{f} pins no model under hydra.sweeper.params.model")
    return str(model)


def _shard_hydra_config(shard: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read the composed config that one shard ran, or None."""
    run_dir = os.path.dirname(os.path.dirname(shard["stage_dir"]))
    f = os.path.join(run_dir, ".hydra", "config.yaml")
    try:
        cfg = yaml.safe_load(open(f))
    except Exception:
        return None
    return cfg if isinstance(cfg, dict) else None


def write_hydra_record(stage_dir: str, sweep: str, model: str,
                      pipeline: str,
                      shards: Optional[List[Dict[str, Any]]] = None) -> None:
    """Write the `.hydra` pair that the canonical registry reads.

    The registry reads the pipeline, the model, the question, the system
    turn, and the image layout from `.hydra/config.yaml`. The record must
    show the config the shards RAN, not a fresh composition.

    When a shard holds `.hydra/config.yaml`, copy it. `submit_shards` composes
    each shard with the `--set` overrides of the arm, and only that file
    carries them: the shard `overrides.yaml` lists the base overrides alone.
    A fresh composition from the sweep loses the arm. The merged runs of
    2026-09-04 (paraphrase, persona) showed the canonical prompt for that
    reason, and the registry classed them as canonical and dropped them.

    When no shard holds a record, compose the config from the sweep, the
    model, and the pipeline. That is enough for an arm that a pipeline name
    carries, such as `*_nocity`.

    The shard index stays as the first shard wrote it: a merged run holds
    every shard, and no reader uses the field.
    """
    from omegaconf import OmegaConf
    sys.path.insert(0, REPO_ROOT)

    records = [c for c in (_shard_hydra_config(s) for s in (shards or []))
               if c is not None]
    if records:
        prompts = {json.dumps(c.get("prompt"), sort_keys=True, default=str)
                   for c in records}
        if len(prompts) > 1:
            raise SystemExit(
                f"{stage_dir}: the shards disagree on the prompt section of "
                "their .hydra/config.yaml; refuse to write one record for them")
        cfg = OmegaConf.create(records[0])
    else:
        from dagspaces.urbanpairvqa.submit_shards import compose_cfg
        cfg = compose_cfg(sweep, model, pipeline, 0, stage_dir, [])
    hydra_dir = os.path.join(stage_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(hydra_dir, "config.yaml"))
    with open(os.path.join(hydra_dir, "overrides.yaml"), "w") as fh:
        yaml.safe_dump([f"+sweep={sweep}", f"model={model}",
                        f"pipeline={pipeline}"], fh)


def merge_case(
    case: str, shards: List[Dict[str, Any]], out_dir: str, *, dry_run: bool,
    sweep: str = "", model: str = "",
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

    pipelines = sorted({s["pipeline"] for s in shards if s.get("pipeline")})
    if len(pipelines) > 1:
        report["problems"].append(f"the shards disagree on the pipeline: {pipelines}")
    elif not pipelines:
        report["problems"].append("no shard recorded a pipeline in its .hydra")
    report["pipeline"] = pipelines[0] if len(pipelines) == 1 else ""

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

    # Write the shape of a stage directory, NOT 2 flat files. The canonical
    # registry scans for `.hydra` beside `outputs/pairwise/`, and it reads the
    # results as the parquet there that is not `pairs.parquet`. The model goes
    # in the directory name, or a second rater would overwrite the first.
    short = model.split("/")[0] if model else "unknown"
    stage_dir = os.path.join(out_dir, f"{case}__{short}", "outputs", "pairwise")
    os.makedirs(stage_dir, exist_ok=True)
    out_results = os.path.join(stage_dir, f"{case}_mvp_merged.parquet")
    merged.to_parquet(out_results, index=False)
    report["out_results"] = out_results
    report["stage_dir"] = os.path.dirname(os.path.dirname(stage_dir))

    pair_frames = [pd.read_parquet(s["pairs"]) for s in shards
                   if os.path.exists(s["pairs"])]
    if pair_frames:
        pairs = pd.concat(pair_frames, ignore_index=True)
        out_pairs = os.path.join(stage_dir, "pairs.parquet")
        pairs.to_parquet(out_pairs, index=False)
        report["out_pairs"] = out_pairs
        report["pair_rows"] = int(len(pairs))

    if sweep and model and report["pipeline"]:
        write_hydra_record(report["stage_dir"], sweep, model, report["pipeline"],
                           shards)

    sidecar = {
        "case": case,
        "pipeline": report["pipeline"],
        "sweep": sweep,
        "model": model,
        "shards": [
            {"shard_index": s["shard_index"], "results": s["results"],
             "rows": s["rows"]}
            for s in shards
        ],
        "rows": int(len(merged)),
        "problems": report["problems"],
    }
    with open(os.path.join(report["stage_dir"], "merge.json"), "w") as fh:
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
    ap.add_argument("--sweep",
                    help="The sweep that ran these shards, such as "
                         "`million_proxy_qwen9b`. It names the model and it "
                         "writes the `.hydra` that the registry reads.")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="Write even when shards are missing. Use with care.")
    args = ap.parse_args()

    if not args.dry_run and not args.out_dir:
        ap.error("--out-dir is needed unless you pass --dry-run")
    model = sweep_model(args.sweep) if args.sweep else ""
    if not args.dry_run and not args.sweep:
        ap.error("--sweep is needed unless you pass --dry-run: without it the "
                 "merged run holds no `.hydra` and the registry cannot read it")

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
                            dry_run=args.dry_run, sweep=args.sweep or "",
                            model=model)
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
