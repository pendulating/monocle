#!/usr/bin/env python
"""F4: check the answers of each (model, case) run.

For each `data/f4_traces/runs/<model>/<case>/responses.parquet`, the script
counts the answers, the errors, the finish reasons (`length` = cut at
`max_tokens`), the answers with a thinking text, and the empty final answers.
It gives the quantiles of the completion tokens. It writes `run_check.parquet`
and prints a table. It measures the runs; it does not analyze the traces.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/check_runs.py [--runs data/f4_traces/runs]
"""
from __future__ import annotations

import argparse
import glob
import os

import polars as pl

ROOT = "/share/ju/matt/mllmsci"
SAMPLE = os.path.join(ROOT, "data/f4_traces/f4_restaurants_1k.parquet")

__version__ = "1.0.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(ROOT, "data/f4_traces/runs"))
    args = ap.parse_args()

    n_images = pl.read_parquet(SAMPLE).height
    rows = []
    for p in sorted(glob.glob(os.path.join(args.runs, "*", "*", "responses.parquet"))):
        case_dir = os.path.dirname(p)
        model, case = os.path.basename(os.path.dirname(case_dir)), os.path.basename(case_dir)
        d = pl.read_parquet(p)
        ok = d.filter(pl.col("error").is_null())
        tok = ok["completion_tokens"]
        rows.append({
            "model": model, "case": case, "answers": d.height, "expected": n_images,
            "errors": d.height - ok.height,
            "cut": int((ok["finish_reason"] == "length").sum()),
            "with_reasoning": int((ok["reasoning"].fill_null("").str.len_chars() > 0).sum()),
            "empty_response": int((ok["response"].fill_null("").str.strip_chars().str.len_chars() == 0).sum()),
            "tok_p50": tok.quantile(0.5), "tok_p90": tok.quantile(0.9), "tok_max": tok.max(),
        })
    if not rows:
        raise SystemExit(f"no responses.parquet under {args.runs}")
    t = pl.DataFrame(rows)
    t.write_parquet(os.path.join(args.runs, "run_check.parquet"))
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200):
        print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
