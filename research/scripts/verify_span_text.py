#!/usr/bin/env python
"""Add `is_verbatim`, `judge_answered`, and `is_duplicate` to a merged corpus.

`is_quotable` reads the alignment status. The fuzzy aligner accepts a span
when 75% of its tokens match the source, thus a `match_fuzzy` row can differ
from the text at its offsets. Measured on 2026-09-20: 12% of the fuzzy rows
of the gemma extractor and 20% of the qwen extractor did, and some changed the
meaning.

`judge_answered` is False when the judge wrote no answer. A judge that
reaches its token cap inside the thought block writes none, and the pairwise
stage then gives the row the label `Same`. That label is a fallback, not a
judgment. On the qwen3.5-9b thinking runs it is up to 5.7% of the traces.

The stage writes both columns from 2026-09-20 on. This script adds them to a
corpus from before that date. It reads the source trace of each row through
`source_results_path`, compares the text at the offsets with the extraction
text, reads the answer, and writes the parquet again in place. It also adds
`verbatim_rate` and `unanswered_traces` to `merge_summary.csv`.

Usage:
    python research/scripts/verify_span_text.py data/trace_extractions [more roots]
    python research/scripts/verify_span_text.py --all

`--all` names the 4 version 1 roots of the 2 x 2.

Warning: the script writes the parquet in place. It changes no existing
column. Run it one time for each root; a second run gives the same result.

See `research/docs/langextract-trace-extraction.md`.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

import pandas as pd

__version__ = "1.1.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V1_ROOTS = (
    "data/trace_extractions",
    "data/trace_extractions_qwen9b",
    "data/trace_extractions_gemma12b_self",
    "data/trace_extractions_qwen9b_self",
)


def source_flags(df: pd.DataFrame, reasoning_column: str = "model_reasoning",
                 response_column: str = "model_response",
                 id_column: str = "pair_id") -> pd.DataFrame:
    """Give `is_verbatim` and `judge_answered` for each row, from the source.

    `is_verbatim`: the source text at the offsets is the span. A row with no
    offsets, no class, or no source gets False. The comparison strips the
    white space at the 2 ends and nothing else, the same as
    `langextract_backend.is_verbatim`.

    `judge_answered`: the source row holds a non-empty answer. A row with no
    source gets True, so that a missing file does not read as a silent judge.
    """
    out = pd.DataFrame({"is_verbatim": False, "judge_answered": True}, index=df.index)
    if "source_results_path" not in df.columns:
        return out
    for source, part in df.groupby("source_results_path"):
        path = str(source)
        if not os.path.exists(path):
            print(f"[verify] source missing, rows stay False: {path}", flush=True)
            continue
        src = (pd.read_parquet(path, columns=[id_column, reasoning_column, response_column])
               .drop_duplicates(id_column).set_index(id_column))
        texts = src[reasoning_column].fillna("").astype(str)
        answered = src[response_column].fillna("").astype(str).str.strip() != ""
        out.loc[part.index, "judge_answered"] = (
            part[id_column].astype(str).map(answered).fillna(True).astype(bool).to_numpy()
        )
        ok = part.char_start.notna() & part.char_end.notna() & part.extraction_text.notna()
        sub = part[ok]
        source_text = sub[id_column].astype(str).map(texts)
        starts = sub.char_start.astype(int).to_numpy()
        ends = sub.char_end.astype(int).to_numpy()
        spans = sub.extraction_text.astype(str).str.strip().to_numpy()
        flags = [
            (isinstance(t, str) and t[s:e].strip() == span)
            for t, s, e, span in zip(source_text.to_numpy(), starts, ends, spans)
        ]
        out.loc[sub.index, "is_verbatim"] = flags
    return out


def duplicate_flags(df: pd.DataFrame) -> pd.Series:
    """Mark every later copy of the same class and text inside 1 trace.

    The same rule as `langextract_backend.annotated_to_rows`: the first copy
    stays. A model that reaches its token cap often loops, and the loop
    writes 1 span hundreds of times.
    """
    if df.empty or "extraction_class" not in df.columns:
        return pd.Series(False, index=df.index)
    order = df.sort_values("extraction_index") if "extraction_index" in df.columns else df
    key = pd.DataFrame({
        "pair": order["pair_id"].astype(str),
        "cls": order["extraction_class"].astype(str),
        "text": order["extraction_text"].astype(str).str.strip(),
    }, index=order.index)
    dup = key.duplicated(keep="first") & order["extraction_class"].notna()
    return dup.reindex(df.index).fillna(False).astype(bool)


def verify_root(root: str) -> pd.DataFrame:
    """Add the column to every case file under `root`, and give the rates."""
    base = root if os.path.isabs(root) else os.path.join(REPO_ROOT, root)
    files = sorted(glob.glob(os.path.join(base, "*_extractions.parquet")))
    if not files:
        raise SystemExit(f"no *_extractions.parquet under {base}")
    rows: List[Dict[str, object]] = []
    for f in files:
        df = pd.read_parquet(f)
        flags = source_flags(df)
        df["is_verbatim"] = flags.is_verbatim.astype(bool)
        df["judge_answered"] = flags.judge_answered.astype(bool)
        df["is_duplicate"] = duplicate_flags(df)
        real = df[df.extraction_class.notna()]
        quotable = real[real.is_quotable] if "is_quotable" in real.columns else real
        fuzzy = real[real.alignment_status == "match_fuzzy"]
        rows.append({
            "case": str(df.case.iloc[0]) if len(df) else os.path.basename(f),
            "rows": int(len(real)),
            "verbatim_rate": round(float(real.is_verbatim.mean()), 4) if len(real) else 0.0,
            "verbatim_of_quotable": (
                round(float(quotable.is_verbatim.mean()), 4) if len(quotable) else 0.0
            ),
            "fuzzy_not_verbatim": (
                round(float((~fuzzy.is_verbatim).mean()), 4) if len(fuzzy) else 0.0
            ),
            "unanswered_traces": int(df[~df.judge_answered].pair_id.nunique()),
            "duplicate_rate": round(float(real.is_duplicate.mean()), 4) if len(real) else 0.0,
            "file": os.path.basename(f),
        })
        df.to_parquet(f, index=False)
        print(f"[verify] {os.path.basename(f)}: verbatim {rows[-1]['verbatim_rate']:.4f}",
              flush=True)
    table = pd.DataFrame(rows)

    summary_path = os.path.join(base, "merge_summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        by_case = table.set_index("case")
        summary["verbatim_rate"] = summary["case"].map(by_case.verbatim_rate)
        summary["unanswered_traces"] = summary["case"].map(by_case.unanswered_traces)
        summary["duplicate_rate"] = summary["case"].map(by_case.duplicate_rate)
        summary.to_csv(summary_path, index=False)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", help="corpus roots to write in place")
    ap.add_argument("--all", action="store_true", help="the 4 version 1 roots")
    args = ap.parse_args()
    roots = list(args.roots) + (list(V1_ROOTS) if args.all else [])
    if not roots:
        ap.error("name a root, or pass --all")
    for root in roots:
        print(f"[verify] {root}", flush=True)
        table = verify_root(root)
        print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
