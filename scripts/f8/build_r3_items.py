#!/usr/bin/env python
"""F8.3: write the items file of the R3 readout for 1 model and 1 source.

| Source | Answers | `item_id` |
|--------|---------|-----------|
| `f4` | the F4 pool, canonical batch, cases restaurant and street | `f4|{case}|{image_idx}` |
| `f4b2` | the F4 pool, batch 2, the same cases | `f4|{case}|{image_idx}` |
| `cue` | the F8.2 answers in `{runs_root}/{model}/{case}_{arm}/` | `cue|{case}|{arm}|{image_idx}` |
| `dc` | the F8.5 answers (D136), the same layout, 4 arms | `dc|{case}|{arm}|{image_idx}` |

An answer that stopped at the token limit is not an item: F4 `cut`, or
`finish_reason` `length` for the cue answers.

Use:
    python scripts/f8/build_r3_items.py --model gemma-4-12b --source f4 --out data/f8/items/gemma-4-12b-f4.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POOL = {"restaurant": "data/f4_traces/pooled/f4_traces_1k.parquet",
        "street": "data/f4_traces/pooled/f4_street_traces_1k.parquet"}
KEEP = ["item_id", "source", "case", "arm", "image_idx", "sample_id", "prompt", "response"]


def f4_items(model: str, batch: str) -> pd.DataFrame:
    parts = []
    for case, path in POOL.items():
        d = pd.read_parquet(path)
        d = d[(d["model"] == model) & (d["case"] == case) & (d["batch"] == batch) & ~d["cut"].astype(bool)]
        d = d[d["error"].isna()].copy()
        d["arm"] = "none"
        d["item_id"] = [f"f4|{case}|{i}" for i in d["image_idx"]]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


ARMS = {"cue": ("rich", "poor"), "dc": ("w_poor", "n_match", "n_rich", "city")}


def cue_items(model: str, runs_root: str, source: str = "cue") -> pd.DataFrame:
    parts = []
    for case in ("restaurant", "street"):
        for arm in ARMS[source]:
            p = Path(runs_root) / model / f"{case}_{arm}" / "responses.parquet"
            d = pd.read_parquet(p)
            d = d[d["error"].isna() & (d["finish_reason"] != "length")].copy()
            d["case"], d["arm"] = case, arm
            d["item_id"] = [f"{source}|{case}|{arm}|{i}" for i in d["image_idx"]]
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True, choices=["f4", "f4b2", "cue", "dc"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-root", default="data/f8/runs", help="the dir of the cue answers (source cue or dc)")
    args = ap.parse_args()
    if args.source in ARMS:
        d = cue_items(args.model, args.runs_root, args.source)
    else:
        d = f4_items(args.model, "canonical" if args.source == "f4" else "b2")
    d["source"] = args.source
    assert d["item_id"].is_unique
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    d[KEEP].to_parquet(args.out, index=False)
    print(f"[items] {args.model} {args.source}: {len(d)} items -> {args.out}")


if __name__ == "__main__":
    main()
