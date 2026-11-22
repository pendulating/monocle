#!/usr/bin/env python
"""F11 (R4): build the input of the DeepSeek-V4-Flash read of the pairwise traces.

The reference R4 of F11 (`facct-runbook.md`, D150, D151) reads pairwise traces
with the F9 extractor (DeepSeek-V4-Flash) and the pairwise schema
(`urban_cues/v2`). The divergence of these spans from the Qwen3.8-27B spans
of the same traces is the extractor effect between the 2 paradigms.

The script takes 1,000 first answers (no repeats) of each (model, case) from
the canonical trace runs. Each pair is also in the corpus of the control
extractor (2,000 pairs; Gemma-4-31B for gemma, Qwen3.8-27B for qwen), thus R1
and R4 read the same traces. The seed is fixed.

Output: `data/f4_traces/extract_input/f4_f11pw.parquet`, 4,000 rows, with the
columns of the pairwise results that the extraction stage reads, plus
`f11_model` and `f11_case`.

Use:
    .venv-vllm025cu129/bin/python research/scripts/f4/build_f11_input.py
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = "/share/ju/matt/mllmsci"
CANON = os.path.join(ROOT, "research/notebooks/cvpr/canonical_data/trace/canonical")
OUT = os.path.join(ROOT, "data/f4_traces/extract_input/f4_f11pw.parquet")
MODELS = {"gemma-4-12b": "gemma12b", "qwen3.5-9b": "qwen9b"}
#: The control extractor of each judge (`_paradigm_kl.PAIRWISE_EXTRACTORS`).
CONTROL = {"gemma-4-12b": "gemma31b", "qwen3.5-9b": "qwen27b"}
CASES = ("restaurants", "street_photography")
N = 1000
SEED = 0
COLUMNS = ["pair_id", "model_reasoning", "model_response", "presented_label",
           "presented_score", "relative_label"]

__version__ = "1.1.0"


def main() -> int:
    frames = []
    for model, short in MODELS.items():
        for case in CASES:
            link = os.path.join(CANON, f"{case}__{model}")
            results = pd.read_parquet(os.path.join(link, "results.parquet"))
            pairs = pd.read_parquet(os.path.join(link, "pairs.parquet"),
                                    columns=["pair_id", "repeat_idx"])
            ctl = pd.read_parquet(
                os.path.join(ROOT, f"data/trace_extractions_v2_{short}_by_{CONTROL[model]}",
                             f"{case}_extractions.parquet"), columns=["pair_id"])
            first = set(pairs.loc[pairs.repeat_idx == 0, "pair_id"]) & set(ctl.pair_id)
            df = results[results.pair_id.isin(first)][COLUMNS]
            df = df[df.model_reasoning.fillna("").str.len() >= 200]
            df = df.sample(N, random_state=SEED).sort_values("pair_id")
            df["f11_model"], df["f11_case"] = model, case
            frames.append(df)
            print(f"{model} {case}: {len(first)} eligible, {len(df)} taken")
    out = pd.concat(frames, ignore_index=True)
    # The id must be unique over the 4 runs: the pair ids repeat across runs.
    out["doc_id"] = out.f11_model + "|" + out.f11_case + "|" + out.pair_id
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
