#!/usr/bin/env python
"""F10.8: merge the 3 extractions of the read prompt into `data/f10/hop_reason/spans.parquet`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python scripts/f10/merge_cool_spans.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/f10/hop_reason"
__version__ = "1.0.0"


def main() -> None:
    parts = []
    schemas = {"cues": "urban_cues_cool_v1", "desc": "aesthetic_descriptors_street_v6", "criteria": "cool_criteria_v1"}
    for arm, schema in schemas.items():
        target = OUT / "merged" / arm
        subprocess.run([sys.executable, str(ROOT / "scripts/f4/merge_extractions.py"), str(ROOT / "multirun/f10x_cool" / arm / "runs"), "--out", str(target),
                        "--cases", "cool_x2", "--schema", str(ROOT / "dagspaces/urbanpairvqa/conf/extract" / f"{schema}.yaml")], check=True)
        d = pd.read_parquet(target / "f4_cool_x2_extractions.parquet")
        parts.append(d.assign(arm=arm))
    d = pd.concat(parts, ignore_index=True)
    d.to_parquet(OUT / "spans.parquet", index=False)
    print(d.groupby(["arm", "model", "extraction_class"]).size().to_string())
    print("texts", d["doc_id"].nunique(), "quotable", round(float(d["is_quotable"].mean()), 4))


if __name__ == "__main__":
    main()
