#!/usr/bin/env python
"""F4: write the stratified smoke input of the descriptor schema.

The smoke run reads `--per` texts from each (model, case, field) cell of the
F4 extraction input. The texts of the schema examples are left out, thus a
span that equals an example span shows a copy.

Output `data/f4_traces/extract_input/f4_descsmoke.parquet`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_descriptor_smoke.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
INPUT = os.path.join(ROOT, "data/f4_traces/extract_input/f4_{}.parquet")
SCHEMA = os.path.join(ROOT, "dagspaces/pairwisevqa/conf/extract/aesthetic_descriptors_v1.yaml")
OUT = INPUT.format("descsmoke")

__version__ = "1.0.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per", type=int, default=6, help="texts in each (model, case, field) cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    texts = pd.concat([pd.read_parquet(INPUT.format(c)) for c in ("restaurant", "caption")], ignore_index=True)
    examples = {e["text"].strip("\n") for e in yaml.safe_load(open(SCHEMA))["examples"]}
    keep = ~texts.text.str.strip("\n").apply(lambda t: any(t.startswith(e) for e in examples))
    print(f"[smoke] left out {int((~keep).sum())} example texts")
    out = (texts[keep].groupby(["model", "case", "field"], group_keys=False)
           .apply(lambda g: g.sample(min(len(g), args.per), random_state=args.seed)))
    out.to_parquet(OUT, index=False)
    print(out.groupby(["model", "case", "field"]).size().unstack("field").to_string())
    print(f"[smoke] {len(out)} texts; wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
