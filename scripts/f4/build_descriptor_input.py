#!/usr/bin/env python
"""F4: write the input of the descriptor extraction (`descriptors-v2` and later).

The input is the cue run input (`f4_restaurant.parquet`, `f4_caption.parquet`)
with 1 change: the non-breaking hyphen (U+2011) becomes "-". The qwen3.8-27b
texts hold 15,113 of them, and the extractor writes "-" in its place, thus
the span does not align (smoke job 94507). The change keeps the length of each
text, thus every offset stays valid against the original text and against
the cue spans.

The smoke input keeps the 120 `doc_id` values of the first smoke run.

Output in `data/f4_traces/extract_input/`:

| File | Rows |
|------|------|
| `f4_restaurant_desc.parquet` | the restaurant texts |
| `f4_caption_desc.parquet` | the caption texts |
| `f4_descsmoke_desc.parquet` | the 120 smoke texts |
| `f4_descsmoke_restaurant_desc.parquet` | the 60 restaurant texts of the smoke input (D115) |
| `f4_street_desc.parquet` | the texts of the street (angle) case (D120) |
| `f4_streetsmoke_desc.parquet` | 6 street texts for each (model, field), without the v4 example text |

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f4/build_descriptor_input.py
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = "/share/ju/matt/mllmsci"
DIR = os.path.join(ROOT, "data/f4_traces/extract_input")

__version__ = "1.2.0"


def normalize(text: str) -> str:
    return text.replace("‑", "-")


def main() -> int:
    smoke_ids = set(pd.read_parquet(os.path.join(DIR, "f4_descsmoke.parquet")).doc_id)
    both = []
    for case in ("restaurant", "caption", "street"):
        d = pd.read_parquet(os.path.join(DIR, f"f4_{case}.parquet"))
        new = d.text.map(normalize)
        assert (new.str.len() == d.text.str.len()).all()
        print(f"[desc-input] {case}: {int((new != d.text).sum()):,} of {len(d):,} texts changed")
        d = d.assign(text=new)
        d.to_parquet(os.path.join(DIR, f"f4_{case}_desc.parquet"), index=False)
        both.append(d)
    street = both.pop()
    example = "qwen3.5-9b|street|48|response"   # a v4 example (D122)
    st = street[street.doc_id != example]
    st = st.groupby(["model", "field"], group_keys=False).sample(n=6, random_state=0)
    st.to_parquet(os.path.join(DIR, "f4_streetsmoke_desc.parquet"), index=False)
    print(f"[desc-input] street smoke: {len(st)} texts")
    smoke = pd.concat(both, ignore_index=True)
    smoke = smoke[smoke.doc_id.isin(smoke_ids)]
    assert len(smoke) == len(smoke_ids)
    smoke.to_parquet(os.path.join(DIR, "f4_descsmoke_desc.parquet"), index=False)
    smoke[smoke.case == "restaurant"].to_parquet(os.path.join(DIR, "f4_descsmoke_restaurant_desc.parquet"), index=False)
    print(f"[desc-input] smoke: {len(smoke)} texts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
