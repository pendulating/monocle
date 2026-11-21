#!/usr/bin/env python
"""F10.8: build the extraction input from the traces of the read prompt.

The cases of `scripts/f4/serve_and_query.sbatch`:

| Case | Prompt | Texts |
|------|--------|-------|
| `cool_word` | `Is this place cool? Answer with one word: yes or no.` | the trace; the answer is 1 word |
| `cool_open` | `Is this place cool?` | the trace and the answer |

Each image has 3 repeats, thus `doc_id` is `<model>|<case>|<image_idx>.<repeat>|<field>`.

Output: `data/f10/hop_reason/traces.parquet` (1 row for each answer) and
`data/f4_traces/extract_input/f4_cool_x2.parquet`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/build_cool_input.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/f4_traces/runs-cool"
OUT = ROOT / "data/f10/hop_reason"
INPUT = ROOT / "data/f4_traces/extract_input/f4_cool_x2.parquet"
__version__ = "1.0.0"


def read_runs() -> pd.DataFrame:
    rows = []
    for f in sorted(RUNS.glob("*/*/responses.jsonl")):
        d = pd.DataFrame(json.loads(line) for line in f.read_text().splitlines())
        d["case"] = f.parent.name
        rows.append(d)
    d = pd.concat(rows, ignore_index=True).rename(columns={"model_key": "model"})
    d["image_idx"], d["repeat"] = d["image_idx"].astype(int), d["repeat"].astype(int)
    images = pd.read_parquet(ROOT / "data/f10/hop_all/images.parquet", columns=["image_idx", "nta2020", "borough"])
    return d.merge(images, on="image_idx", validate="m:1")


def as_input(d: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for field in ("reasoning", "response"):
        p = d[(d[field].fillna("").str.len() > 0) & ~((field == "response") & (d["case"] == "cool_word"))].copy()
        p["text"], p["field"] = p[field].str.replace("‑", "-", regex=False), field
        parts.append(p)
    x = pd.concat(parts, ignore_index=True)
    what = x["field"].map({"reasoning": "the private reasoning trace", "response": "the final answer"})
    x["context"] = [f'This text is {w} of {m} to the prompt: "{p}"' for w, m, p in zip(what, x["model"], x["prompt"])]
    x["doc_id"] = x["model"] + "|" + x["case"] + "|" + x["image_idx"].astype(str) + "." + x["repeat"].astype(str) + "|" + x["field"]
    x["uid"], x["camis"], x["dba"] = x["sample_id"], "", x["nta2020"]
    x["cut"], x["source_run"], x["max_tokens"] = x["finish_reason"] == "length", "runs-cool", 4096
    keep = ["doc_id", "text", "model", "case", "field", "image_idx", "uid", "camis", "dba", "borough", "cut", "source_run", "max_tokens", "completion_tokens", "context"]
    if not x["doc_id"].is_unique:
        raise SystemExit("doc_id is not unique")
    return x[keep].sort_values(["model", "field", "doc_id"]).reset_index(drop=True)


def main() -> None:
    d = read_runs()
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "traces.parquet", index=False)
    x = as_input(d)
    x.to_parquet(INPUT, index=False)
    print(d.groupby(["model", "case"]).agg(n=("image_idx", "size"), images=("image_idx", "nunique"), cut=("finish_reason", lambda v: (v == "length").sum()),
                                           errors=("error", lambda v: v.astype(str).ne("None").sum()), tokens=("completion_tokens", "median")))
    print(x.groupby(["model", "case", "field"]).size(), len(x))


if __name__ == "__main__":
    main()
