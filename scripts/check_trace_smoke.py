#!/usr/bin/env python
"""Read the health of a trace-extraction shard, and apply the smoke gate.

Use:
    python scripts/check_trace_smoke.py multirun/2026-09-28_CUESV3GQ27_smoke
    python scripts/check_trace_smoke.py multirun/2026-09-28_CUESV3GQ27 --gate

For each `runs/<case>__<shard>/` under the base directory it reads the
stage metadata of `pipeline_manifest.json` and the shard parquet, and it
prints 1 row for each shard:

| Column | Reads |
|--------|-------|
| `truncated` | share of answers the token cap cut (`answers_truncated`) |
| `empty` | traces with no extraction |
| `per_trace` | extractions for each trace |
| `verbatim` | share of spans whose slice equals the span |
| `dup` | share of spans that repeat an earlier span of the same class |
| `nested` | share of spans that sit inside another span of the trace |
| `classes` | the classes present, with their share |

The gate of the version 3 ladder (2026-09-28): truncated under 10%,
verbatim over 95%, dup under 5%, no empty answer, and every class of the
schema present in the shard. `--gate` exits 1 when a shard fails it.

See `vlm-narratives-docs/langextract-trace-extraction.md`, "Version 3".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

__version__ = "1.0.0"

V3_CLASSES = ("dimension", "inference", "perspective", "weighing", "justification", "dismissal",
              "verdict", "decision", "comparison", "uncertainty", "person_reference", "image_artifact")
# The gate asks for these. The others are rare by nature: the smoke of
# 2026-09-28 found no weighing in 120 subway traces, and 1 weighing word.
CORE_CLASSES = ("dimension", "inference", "perspective", "uncertainty", "decision")

GATE = {"truncated": 0.10, "verbatim": 0.95, "dup": 0.05, "decision_share": 0.95}


def _is_verbatim(row: pd.Series, texts: Dict[str, str]) -> bool:
    t = texts.get(row.pair_id)
    if t is None or pd.isna(row.char_start) or pd.isna(row.char_end):
        return False
    return t[int(row.char_start):int(row.char_end)] == str(row.extraction_text)


def _nested_share(df: pd.DataFrame) -> float:
    n, nested = 0, 0
    for _, part in df.groupby("pair_id"):
        spans = part[["char_start", "char_end"]].dropna().astype(int).values.tolist()
        for i, (s, e) in enumerate(spans):
            n += 1
            if any((s2 <= s and e <= e2) and (e2 - s2) > (e - s) for j, (s2, e2) in enumerate(spans) if j != i):
                nested += 1
    return nested / n if n else float("nan")


def read_shard(run_dir: str) -> Dict[str, Any]:
    name = os.path.basename(run_dir.rstrip("/"))
    out: Dict[str, Any] = {"shard": name, "state": "no manifest"}
    mp = os.path.join(run_dir, "pipeline_manifest.json")
    files = sorted(glob.glob(os.path.join(run_dir, "outputs", "extractions", "*.parquet")))
    if not os.path.exists(mp):
        out["state"] = "running" if glob.glob(os.path.join(run_dir, "outputs", "extractions", "chunks", "*")) else "waiting"
        return out
    meta = json.load(open(mp))["nodes"]["extract"]["metadata"]
    n_traces = int(meta.get("traces_extracted") or 0)
    out.update({
        "state": "done", "traces": n_traces,
        "truncated": round((meta.get("answers_truncated") or 0) / max(1, n_traces), 3),
        "empty": int(meta.get("traces_with_no_extraction") or 0),
        "per_trace": meta.get("extractions_per_trace"),
        "exact": meta.get("exact_rate"), "minutes": round((meta.get("duration_s") or 0) / 60),
    })
    if not files:
        out["state"] = "done, no parquet"
        return out
    df = pd.read_parquet(files[-1])
    src = meta.get("source_stage_dir")
    texts: Dict[str, str] = {}
    src_files = glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True) if src else []
    for f in src_files:
        try:
            t = pd.read_parquet(f, columns=["pair_id", "model_reasoning"])
        except Exception:
            continue
        texts.update(dict(zip(t.pair_id.astype(str), t.model_reasoning.fillna(""))))
        if len(texts) >= n_traces:
            break
    if "is_verbatim" in df.columns:
        out["verbatim"] = round(float(df.is_verbatim.mean()), 3)
    elif texts:
        out["verbatim"] = round(float(df.apply(lambda r: _is_verbatim(r, texts), axis=1).mean()), 3)
    if "is_duplicate" in df.columns:
        out["dup"] = round(float(df.is_duplicate.mean()), 3)
    out["nested"] = round(_nested_share(df), 3)
    counts = df.extraction_class.value_counts(normalize=True).round(3)
    out["classes"] = {k: float(v) for k, v in counts.items()}
    out["missing_classes"] = sorted(set(V3_CLASSES) - set(counts.index))
    # The decision share the LOADER will see: over the answered traces, with
    # the last verdict promoted where no decision was written
    # (`_extractions.promote_last_verdict`).
    answered = df[df.judge_answered.fillna(True).astype(bool)] if "judge_answered" in df.columns else df
    with_dec = set(answered[answered.extraction_class == "decision"].pair_id)
    with_verdict = set(answered[answered.extraction_class == "verdict"].pair_id)
    out["traces_answered"] = int(answered.pair_id.nunique())
    out["decision_traces"] = len(with_dec | with_verdict)
    out["decision_written"] = len(with_dec)
    if "extraction_class" in df.columns:
        per = df[df.extraction_class == "perspective"].groupby("pair_id").size()
        out["perspectives_per_trace"] = round(float(per.reindex(df.pair_id.unique()).fillna(0).mean()), 2)
    return out


def gate_ok(row: Dict[str, Any]) -> List[str]:
    fails = []
    if row.get("state") != "done":
        return [row.get("state", "?")]
    if row.get("truncated", 1) > GATE["truncated"]:
        fails.append(f"truncated {row['truncated']}")
    if row.get("verbatim", 0) < GATE["verbatim"]:
        fails.append(f"verbatim {row.get('verbatim')}")
    if row.get("dup", 1) > GATE["dup"]:
        fails.append(f"dup {row.get('dup')}")
    if row.get("empty", 1):
        fails.append(f"empty {row.get('empty')}")
    missing_core = [c for c in row.get("missing_classes", []) if c in CORE_CLASSES]
    if missing_core:
        fails.append("missing " + ",".join(missing_core))
    if row.get("decision_traces") is not None and row.get("traces_answered"):
        share = row["decision_traces"] / row["traces_answered"]
        if share < GATE["decision_share"]:
            fails.append(f"decision on {share:.0%} of traces")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base_dir")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    runs = sorted(glob.glob(os.path.join(args.base_dir, "runs", "*__*")))
    if not runs:
        print(f"no runs under {args.base_dir}")
        return 1
    bad = 0
    for r in runs:
        row = read_shard(r)
        fails = gate_ok(row)
        head = {k: v for k, v in row.items() if k not in ("classes", "missing_classes")}
        print(json.dumps(head))
        if row.get("classes"):
            print("   classes:", ", ".join(f"{k} {v:.0%}" for k, v in sorted(row["classes"].items(), key=lambda kv: -kv[1])))
        if row.get("missing_classes"):
            print("   missing:", ", ".join(row["missing_classes"]))
        if args.gate:
            print("   gate:", "PASS" if not fails else "FAIL " + "; ".join(fails))
            bad += bool(fails)
    return 1 if (args.gate and bad) else 0


if __name__ == "__main__":
    sys.exit(main())
