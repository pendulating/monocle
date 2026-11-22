#!/usr/bin/env python
"""F4: merge the shard parquets of the span extraction into 1 file for each case.

`research/scripts/merge_trace_extractions.py` keys on `pair_id`. The F4 rows key on
`doc_id` (`<model>|<case>|<image_idx>|<field>`), thus this script does the
same work for F4. It tests the coverage against the input
(`data/f4_traces/extract_input/`): every text must appear in the output, and
no text may appear from 2 shards.

Output (default `data/f4_traces/extractions/`):

| File | Content |
|------|---------|
| `f4_<case>_extractions.parquet` | 1 row for each extraction; a text with no extraction has 1 row with a null class |
| `merge_summary.csv` | for each (case, model, field): texts, extractions, rate of quotable spans, silent texts |
| `merge_manifest.json` | the shards, the coverage, the schema version |

`--schema <conf/extract yaml>` (the x2 round, D130) adds 2 text flags to
every row, and the summary counts the `none` items apart:

| Column | Meaning |
|--------|---------|
| `copy_suspect` | the text has 2 or more spans that equal an example span and are not quotable, or no quotable span and 1 such span (`x2_retry.copy_flags`); drop the text |
| `example_source` | the text holds the text of a prompt example; the extractor saw its answer in the prompt |

With `--schema`, the merge also runs the substring rescue again
(`langextract_backend.rescue_offsets`, with the mode "markup") on every
span that is not quotable, and writes `alignment_rescue`. The first steps
of the x2 run used a backend with no "markup" mode; this step makes all
arms equal.

`--retry-runs <dir>` takes the runs of the retry pass (`x2_retry.py`): its
rows replace all rows of each text that it read. The retry schema version
is the first version and "r"; only these 2 versions pool.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/merge_extractions.py multirun/f4_extract_dsv4/runs
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/merge_extractions.py multirun/f4_captions_dsv4/runs \\
        --cases captions --out data/f4_traces/captions
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # the repo root
from x2_retry import copy_flags  # noqa: E402

ROOT = "/share/ju/matt/mllmsci"
INPUT_DIR = os.path.join(ROOT, "data/f4_traces/extract_input")
OUT_DIR = os.path.join(ROOT, "data/f4_traces/extractions")

__version__ = "1.4.0"


def shard_files(runs_dir: str) -> list[str]:
    """The final parquet of each shard. A shard dir holds 1; `chunks/` is the resume unit."""
    out = []
    for shard_dir in sorted(glob.glob(os.path.join(runs_dir, "f4_*__*"))):
        finals = sorted(glob.glob(os.path.join(shard_dir, "outputs/extractions/*.parquet")))
        if len(finals) > 1:
            raise SystemExit(f"{shard_dir} holds {len(finals)} final parquets; remove the stale one")
        out += finals
    return out


def rescue(part: pd.DataFrame, text: pd.Series) -> pd.DataFrame:
    """Find the spans that are not quotable in their text again, and mark them quotable."""
    from dagspaces.common.langextract_backend import rescue_offsets

    if "alignment_rescue" not in part:
        part["alignment_rescue"] = None
    todo = part.index[part["extraction_class"].notna() & (part["extraction_class"] != "none")
                      & ~part["is_quotable"].fillna(False).astype(bool)]
    n = 0
    for i in todo:
        r = part.loc[i]
        hint = None if pd.isna(r["char_start"]) else int(r["char_start"])
        hit = rescue_offsets(text.get(r["doc_id"], ""), r["extraction_text"], None, hint)
        if hit is None:
            continue
        part.loc[i, ["char_start", "char_end"]] = [hit[0], hit[1]]
        part.loc[i, "is_quotable"] = True
        part.loc[i, "is_verbatim"] = hit[2] == "exact"
        part.loc[i, "alignment_rescue"] = hit[2]
        n += 1
    print(f"[merge] rescue: {n} of {len(todo)} spans that are not quotable are found again")
    return part


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_dir")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--cases", nargs="+", default=["restaurant", "caption"],
                    help="the input files f4_<case>.parquet to check; captions for the caption job")
    ap.add_argument("--allow-partial", action="store_true", help="write even when texts are missing")
    ap.add_argument("--schema", help="the schema yaml; adds copy_suspect and example_source")
    ap.add_argument("--retry-runs", help="the runs dir of the retry pass; its rows replace the first rows")
    args = ap.parse_args()

    files = shard_files(args.runs_dir)
    if not files:
        raise SystemExit(f"no shard parquet under {args.runs_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    versions = sorted(df["schema_version"].dropna().unique())
    if len(versions) != 1:
        raise SystemExit(f"the shards hold {versions}; 2 schema versions never pool")
    retried = set()
    if args.retry_runs:
        rfiles = shard_files(args.retry_runs)
        if rfiles:
            r = pd.concat([pd.read_parquet(f) for f in rfiles], ignore_index=True)
            rv = sorted(r["schema_version"].dropna().unique())
            if rv != [versions[0] + "r"]:
                raise SystemExit(f"the retry runs hold {rv}, want {versions[0]}r")
            retried = set(r["doc_id"])
            df = pd.concat([df[~df["doc_id"].isin(retried)], r], ignore_index=True)
            files += rfiles
        print(f"[merge] retry: {len(retried)} texts from {args.retry_runs}")

    os.makedirs(args.out, exist_ok=True)
    examples = yaml.safe_load(open(args.schema))["examples"] if args.schema else []
    ex_spans = {x["extraction_text"] for e in examples for x in e["extractions"] if x["extraction_class"] != "none"}
    coverage, summary = {}, []
    for case in args.cases:
        texts = pd.read_parquet(os.path.join(INPUT_DIR, f"f4_{case}.parquet"), columns=["doc_id", "text"])
        want = texts["doc_id"]
        part = df[df["doc_id"].isin(set(want))]
        # A text read by 2 shards has 2 rows of extraction_index 0 (or 2 null rows).
        firsts = part[part["extraction_index"].fillna(0) == 0].groupby("doc_id").size()
        doubled = int((firsts > 1).sum())
        missing = int((~want.isin(set(part["doc_id"]))).sum())
        coverage[case] = {"texts": int(want.size), "missing": missing, "read_twice": doubled}
        print(f"[merge] {case}: {want.size:,} texts, {missing:,} missing, {doubled:,} read twice")
        if doubled:
            raise SystemExit(f"{case}: {doubled} texts come from 2 shards")
        if missing and not args.allow_partial:
            raise SystemExit(f"{case}: {missing} texts have no row; run the missing shards (or pass --allow-partial)")
        if args.schema:
            part = rescue(part.copy(), texts.set_index("doc_id")["text"])
            flags = copy_flags(part, ex_spans)
            part["copy_suspect"] = part["doc_id"].map(flags.copy_suspect).fillna(False).astype(bool)
            part["retried"] = part["doc_id"].isin(retried)
            # An example text is a cut of its source; its first 200 characters find the source.
            heads = [e["text"].strip()[:200] for e in examples]
            src = {d for d, t in zip(texts["doc_id"], texts["text"]) if any(h and h in t for h in heads)}
            part["example_source"] = part["doc_id"].isin(src)
            n_bad = int(flags.copy_suspect.sum())
            coverage[case + "_flags"] = {"copy_suspect": n_bad, "example_source": len(src),
                                         "retried": len(retried & set(want))}
            print(f"[merge] {case}: {n_bad} copy suspects, {len(src)} example sources, "
                  f"{len(retried & set(want))} retried")
        part.to_parquet(os.path.join(args.out, f"f4_{case}_extractions.parquet"), index=False)
        for (model, field), g in part.groupby(["model", "field"]):
            cls = g["extraction_class"]
            real = g[cls.notna() & (cls != "none")]
            summary.append({
                "case": case, "model": model, "field": field,
                "texts": g["doc_id"].nunique(), "extractions": len(real),
                "per_text": round(len(real) / max(1, g["doc_id"].nunique()), 1),
                "quotable_rate": round(float(real["is_quotable"].mean()), 4) if len(real) else 0.0,
                "verbatim_rate": round(float(real["is_verbatim"].mean()), 4) if len(real) else 0.0,
                "silent_texts": int(g["extraction_class"].isna().sum()),
                "none_items": int((cls == "none").sum()),
                "texts_without_cue": int(g["doc_id"].nunique() - real["doc_id"].nunique()),
                "copy_suspects": int(g.loc[g.get("copy_suspect", pd.Series(False, index=g.index)), "doc_id"].nunique()),
            })
    table = pd.DataFrame(summary)
    table.to_csv(os.path.join(args.out, "merge_summary.csv"), index=False)
    with open(os.path.join(args.out, "merge_manifest.json"), "w") as fh:
        json.dump({
            "script": os.path.relpath(__file__, ROOT), "version": __version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs_dir": os.path.abspath(args.runs_dir), "shards": [os.path.relpath(f, ROOT) for f in files],
            "schema_version": versions[0], "retry_runs": args.retry_runs, "coverage": coverage,
        }, fh, indent=1)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
