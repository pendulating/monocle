#!/usr/bin/env python
"""F10.1: build the items of the text swap of the place name (D148).

Each image gets 5 arms. Only the place name changes. Rules:
vlm-narratives-docs/facct-runbook.md, F10.1.

| Column | Content |
|--------|---------|
| `item_id` | `{case}_{image_idx}_{arm}` |
| `case`, `image_idx`, `sample_id` | the image |
| `borough` | the borough of the image: Manhattan or Bronx |
| `arm`, `question` | the arm and its question |

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/build_name_sample.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__version__ = "1.0.0"

SEED = 20261030
N = 500
BOROUGHS = ("Manhattan", "Bronx")
POOLS = {"restaurant": "data/f4_traces/subj/f4_restaurants_10k.parquet",
         "street": "data/f4_traces/subj/f4_street_10k.parquet"}
ARMS = {"manhattan": "Manhattan", "bronx": "the Bronx", "brooklyn": "Brooklyn", "queens": "Queens", "none": None}
OUT = Path("data/f10")


def question(name: str | None) -> str:
    return f"Is this place in {name} cool?" if name else "Is this place cool?"


def main() -> None:
    rows = []
    for case, path in POOLS.items():
        d = pd.read_parquet(path, columns=["image_idx", "sample_id", "recording_id", "borough"])
        d = d.assign(borough=d["borough"].str.title()).drop_duplicates("recording_id")
        for boro in BOROUGHS:
            part = d[d["borough"] == boro]
            assert len(part) >= N, (case, boro, len(part))
            part = part.sample(N, random_state=SEED).sort_values("image_idx")
            for arm, name in ARMS.items():
                rows.append(part[["image_idx", "sample_id", "borough"]].assign(case=case, arm=arm, question=question(name)))
    items = pd.concat(rows, ignore_index=True)
    items["item_id"] = items["case"] + "_" + items["image_idx"].astype(str) + "_" + items["arm"]
    assert items["item_id"].is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    items.to_parquet(OUT / "name_items.parquet", index=False)
    print(items.groupby(["case", "borough"])["image_idx"].nunique().to_string())
    print(len(items), "items", items.drop_duplicates("arm")[["arm", "question"]].to_string(index=False))


if __name__ == "__main__":
    main()
