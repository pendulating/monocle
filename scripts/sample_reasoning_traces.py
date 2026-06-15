#!/usr/bin/env python3
"""Sample N reasoning traces from a pairwise VQA stage output parquet to CSV.

Pulls a reproducible random sample of rows and writes the identifying columns
plus the captured ``model_reasoning`` trace, for manual qualitative review.

By default only rows with a non-empty reasoning trace are eligible (first-batch
thinking drops produce blanks); pass ``--include-empty`` to sample from all rows.

Example:

    python scripts/sample_reasoning_traces.py \\
        multirun/2026-05-10_URBANPAIRVQA/11-15-11/0/outputs/pairwise/sterility_large_20260510_111527.parquet \\
        -n 40 --out reports/pairwise/sterility_reasoning_sample_40.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Identifying columns kept alongside the reasoning trace, when present.
ID_COLUMNS = [
    "pair_id",
    "sample_id_a",
    "sample_id_b",
    "image_path_a",
    "image_path_b",
    "presented_order",
    "is_swapped",
    "relative_label",
    "relative_score",
    "model_reasoning",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_parquet", type=Path, help="Pairwise VQA stage output parquet.")
    p.add_argument("-n", "--n", type=int, default=40, help="Number of traces to sample. Default: 40.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV. Default: <parquet_dir>/<stem>.reasoning_sample_<n>.csv.",
    )
    p.add_argument("--seed", type=int, default=1234, help="RNG seed for the sample. Default: 1234.")
    p.add_argument(
        "--include-empty",
        action="store_true",
        help="Allow rows with an empty/missing reasoning trace into the sample.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.output_parquet.exists():
        raise SystemExit(f"Parquet not found: {args.output_parquet}")

    df = pd.read_parquet(args.output_parquet)
    if "model_reasoning" not in df.columns:
        raise SystemExit(f"{args.output_parquet} has no 'model_reasoning' column.")

    eligible = df
    if not args.include_empty:
        reasoning = df["model_reasoning"].fillna("").astype(str).str.strip()
        eligible = df[reasoning.str.len() > 0]
        if eligible.empty:
            raise SystemExit(
                "No non-empty reasoning traces; rerun with --include-empty to sample anyway."
            )

    k = min(args.n, len(eligible))
    if k < args.n:
        print(f"[WARN] only {k} eligible rows (requested {args.n}).")
    sample = eligible.sample(n=k, random_state=args.seed)

    cols = [c for c in ID_COLUMNS if c in sample.columns]
    sample = sample[cols]

    out = args.out or args.output_parquet.with_suffix("").with_name(
        f"{args.output_parquet.stem}.reasoning_sample_{k}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"Wrote {k} reasoning traces -> {out}")
    print(f"Columns: {cols}")


if __name__ == "__main__":
    main()
