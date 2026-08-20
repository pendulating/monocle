#!/usr/bin/env python3
"""Draw the pair table of each case once, so the shards do not repeat it.

Why
---
Drawing 1,100,000 pairs costs about 195 seconds. The draw is deterministic:
the seed fixes it, thus every shard of a case builds the SAME table and then
keeps its own share. Over the 966 jobs of the 1,000,000-pair battery that is
about 51 GPU-hours of repeated work, and a preemption makes a job pay it again.

The table does not depend on the model. Thus 1 file for each case serves the 92
qwen shards AND the 46 gemma shards.

A shard reads the parquet in about 3 seconds.

What it writes
--------------
    <out-dir>/<case>_pairs.parquet   the table
    <out-dir>/<case>_pairs.json      the sampler settings that drew it

The runner compares that sidecar against its own config and stops when they
differ, thus a table from other settings cannot quietly change the science.

Use
---
    python scripts/prebuild_pair_tables.py --sweep million_proxy_qwen9b \\
        --out-dir /share/pierson/matt/mllmsci/multirun/pair_tables_1m
    python scripts/prebuild_pair_tables.py --sweep ... --cases pairwise_schools_mvp
    python scripts/prebuild_pair_tables.py --sweep ... --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_CONF_DIR = os.path.join(_ROOT, "dagspaces", "urbanpairvqa", "conf")


def _compose(sweep: str, case: str, extra: List[str]):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    overrides = [f"+sweep={sweep}", f"pipeline={case}"] + list(extra)
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_CONF_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


def build_one(sweep: str, case: str, out_dir: str, extra: List[str],
              force: bool) -> Dict[str, Any]:
    """Draw 1 case and write its parquet and sidecar."""
    from dagspaces.urbanpairvqa.orchestrator import (
        _load_pairwise_manifest, build_pair_table, pair_fingerprint,
    )

    short = case.replace("pairwise_", "").replace("_mvp", "")
    out_parquet = os.path.join(out_dir, f"{short}_pairs.parquet")
    out_json = os.path.join(out_dir, f"{short}_pairs.json")

    cfg = _compose(sweep, case, extra)
    t0 = time.time()
    manifest_df = _load_pairwise_manifest(cfg, None)
    want = pair_fingerprint(cfg, len(manifest_df))

    if os.path.exists(out_parquet) and os.path.exists(out_json) and not force:
        try:
            with open(out_json) as fh:
                have = json.load(fh).get("fingerprint")
        except Exception:
            have = None
        if have == want:
            return {"case": short, "status": "kept", "path": out_parquet,
                    "seconds": round(time.time() - t0, 1)}

    pairs = build_pair_table(cfg, manifest_df)
    os.makedirs(out_dir, exist_ok=True)
    tmp = f"{out_parquet}.tmp{os.getpid()}"
    pairs.to_parquet(tmp, index=False)
    os.replace(tmp, out_parquet)
    with open(out_json, "w") as fh:
        json.dump({"case": short, "pipeline": case, "rows": int(len(pairs)),
                   "fingerprint": want}, fh, indent=2)
    return {"case": short, "status": "built", "path": out_parquet,
            "rows": int(len(pairs)),
            "mb": round(os.path.getsize(out_parquet) / 1e6, 1),
            "seconds": round(time.time() - t0, 1)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--set", dest="extra", action="append", default=[])
    ap.add_argument("--force", action="store_true",
                    help="Draw again even when the sidecar already matches.")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args(argv)

    from dagspaces.urbanpairvqa.submit_shards import read_grid
    _, all_cases, _ = read_grid(args.sweep)
    cases = args.cases or all_cases
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[prebuild] sweep : {args.sweep}")
    print(f"[prebuild] cases : {len(cases)}")
    print(f"[prebuild] out   : {args.out_dir}")

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    workers = max(1, min(args.workers, len(cases)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(build_one, args.sweep, c, args.out_dir,
                            args.extra, args.force): c for c in cases}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"[prebuild] FAILED {futs[fut]}: "
                      f"{type(exc).__name__}: {exc}", file=sys.stderr)
                results.append({"case": futs[fut], "status": "failed"})
                continue
            results.append(r)
            extra = f", {r.get('rows', 0):,} rows, {r.get('mb', 0)} MB" if r["status"] == "built" else ""
            print(f"[prebuild] {r['status']:6s} {r['case']:20s} "
                  f"{r['seconds']:6.1f}s{extra}")

    bad = [r for r in results if r["status"] == "failed"]
    print(f"[prebuild] done in {time.time() - t0:.1f}s, "
          f"{len(results) - len(bad)} of {len(results)} ready")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
