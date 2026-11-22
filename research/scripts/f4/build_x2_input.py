#!/usr/bin/env python
"""F4: build the inputs of the x2 extraction round (D130), and its smoke sets.

The x2 round reruns every F4 extraction with the schemas that apply the
advisor review of 2026-10-26 (`single-v2`, `street-v3`, `descriptors-v6`,
`caption-v4`). Against the earlier inputs:

1. Every text has U+2011 as "-" (`build_descriptor_input.py`), thus the
   offsets of all x2 runs match each other.
2. A `context` column holds 1 line for LangExtract's per-document
   `additional_context`: the field, the model, and the prompt. The line tells
   the extractor which input shape it reads before it reads the examples.

Output, in `data/f4_traces/extract_input/`:

| File | Rows |
|------|------|
| `f4_<case>_x2.parquet` | every text of restaurant, caption, street; `f4_captions_x2` holds the caption answers only |
| `f4_x2smoke_<case>.parquet` | the smoke set of each case (below) |
| `x2_smoke_doc_ids.json` | the doc_ids of every smoke set; no example may come from them |
| `f4_street_captions_<suffix>.parquet` | with `--street-captions` only: the `street_caption` answers (D133); the script writes no other file |

The smoke sets (seed 20261026):

| Set | Texts |
|-----|-------|
| `restaurant` | 4 per (model, field), and the 10 longest qwen3.5-9b restaurant traces (the loops of the single-v1 run) |
| `caption` | 4 per (model, field), and 10 caption answers that got a whole example copy under single-v1 |
| `street` | the 60 texts of `f4_streetsmoke_desc` (the v1 and v2 smokes), and the 10 longest qwen3.5-9b street traces |
| `captions` | 6 caption answers per model, and 10 Llama answers that are 1 quoted text |

Batch 2 (D132): `--in-dir` reads the `build_extract_input.py` output of the
batch, `--suffix` names the files (`f4_<case>_<suffix>.parquet`, always in
`extract_input/`, the folder that `extract_shard.py` reads), and `--no-smoke`
writes no smoke set. The smoke code reads files of the first round only.

F9 (D142, `vlm-narratives-docs/f9-extraction-design.md`): `--subj` reads
`data/f4_traces/subj/extract_input/f4_{restaurant,street}_subj.parquet` and
writes `f4_<case>_x2.parquet`, the smoke sets `f4_x2smoke_<case>.parquet`
(4 texts for each (model, field) and the 10 longest qwen3.5-9b traces,
without the example texts of the schema) and `x2_smoke_doc_ids_subj.json`.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_x2_input.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_x2_input.py \
        --in-dir data/f4_traces/b2/extract_input --suffix b2x2 --no-smoke
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_x2_input.py --subj
"""
from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
IN = os.path.join(ROOT, "data/f4_traces/extract_input")
EXTR = os.path.join(ROOT, "data/f4_traces/extractions")
SEED = 20261026
PROMPTS = {
    "restaurant": "Look and tell me more about this restaurant.",
    "caption": "Look and write an Instagram caption for what's in front of me.",
    "street": "Look and tell me a good angle to take a picture from.",
    "street_caption": "Look and write an Instagram caption for what's in front of me.",
    "restaurant_subj": "Does this look like a nice restaurant to eat at?",
    "street_subj": "Would this street make a good spot for taking a photo?",
}
SUBJ_IN = os.path.join(ROOT, "data/f4_traces/subj/extract_input")
# The schema of each F9 case. Its header lists the example texts.
SUBJ_SCHEMA = {"restaurant_subj": "urban_cues_single_subj_v1", "street_subj": "urban_cues_street_subj_v1"}

__version__ = "1.3.0"


def with_context(d: pd.DataFrame) -> pd.DataFrame:
    """Normalize U+2011 and add the context line. The text length does not change."""
    d = d.copy()
    d["text"] = d.text.str.replace("‑", "-", regex=False)
    what = d.field.map({"reasoning": "the private reasoning trace", "response": "the final answer"})
    d["context"] = [f"This text is {w} of {m} to the prompt: \"{PROMPTS[c]}\""
                    for w, m, c in zip(what, d.model, d.case)]
    return d


def copied_docs() -> set:
    """Give the texts whose single-v1 spans are 5 or more copies of example spans, not in the text."""
    ex = yaml.safe_load(open(os.path.join(ROOT, "research/conf/pairwisevqa/extract/urban_cues_single_v1.yaml")))
    spans = {e["extraction_text"] for x in ex["examples"] for e in x["extractions"]}
    d = pd.concat([pd.read_parquet(os.path.join(EXTR, f"f4_{c}_extractions.parquet"),
                                   columns=["doc_id", "extraction_text", "is_verbatim"]) for c in ("restaurant", "caption")])
    c = d[d.extraction_text.isin(spans) & ~d.is_verbatim.fillna(False)].groupby("doc_id").size()
    return set(c[c >= 5].index)


def example_doc_ids(schema: str) -> set:
    """Give the doc_ids of the example texts that the header of a schema lists."""
    path = os.path.join(ROOT, f"research/conf/pairwisevqa/extract/{schema}.yaml")
    return {m.group(1) for m in re.finditer(r"^#\s+(\S+\|\w+\|\d+\|(?:reasoning|response))\s*$",
                                            open(path).read(), re.M)}


def subj() -> int:
    """Write the F9 inputs and their smoke sets (D142)."""
    ids = {}
    for case, schema in SUBJ_SCHEMA.items():
        d = with_context(pd.read_parquet(os.path.join(SUBJ_IN, f"f4_{case}.parquet")))
        d.to_parquet(os.path.join(IN, f"f4_{case}_x2.parquet"), index=False)
        ex = example_doc_ids(schema)
        if not ex or not ex <= set(d.doc_id):
            raise SystemExit(f"{schema}: the example doc_ids are missing from the header or the input")
        pool = d[~d.doc_id.isin(ex)]
        r = pool.groupby(["model", "field"], group_keys=False).sample(n=4, random_state=SEED)
        t = pool[(pool.model == "qwen3.5-9b") & (pool.field == "reasoning") & ~pool.doc_id.isin(r.doc_id)]
        t = t.assign(n=t.text.str.len()).nlargest(10, "n").drop(columns="n")
        sm = pd.concat([r, t]).drop_duplicates("doc_id")
        sm.to_parquet(os.path.join(IN, f"f4_x2smoke_{case}.parquet"), index=False)
        ids[case] = sorted(sm.doc_id)
        print(f"[x2] {case}: {len(d)} texts; longest {d.text.str.len().max():,} chars; smoke {len(sm)} texts; "
              f"{len(ex)} example texts kept out; U+2011 left {int(d.text.str.contains('‑').sum())}")
    with open(os.path.join(IN, "x2_smoke_doc_ids_subj.json"), "w") as f:
        json.dump(ids, f, indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=IN, help="the folder of f4_<case>.parquet and f4_captions.parquet")
    ap.add_argument("--suffix", default="x2", help="the output name: f4_<case>_<suffix>.parquet in extract_input/")
    ap.add_argument("--no-smoke", action="store_true", help="write no smoke set")
    ap.add_argument("--street-captions", action="store_true",
                    help="write only f4_street_captions_<suffix>.parquet, from <in-dir>/f4_street_captions.parquet (D133)")
    ap.add_argument("--subj", action="store_true", help="write the F9 inputs and smoke sets only (D142)")
    args = ap.parse_args()
    if args.subj:
        return subj()
    if args.street_captions:
        d = with_context(pd.read_parquet(os.path.join(args.in_dir, "f4_street_captions.parquet")))
        if set(d.case) != {"street_caption"} or set(d.field) != {"response"}:
            raise SystemExit("f4_street_captions.parquet must hold street_caption answers only")
        d.to_parquet(os.path.join(IN, f"f4_street_captions_{args.suffix}.parquet"), index=False)
        print(f"[x2] street captions: {len(d)} answers; longest {d.text.str.len().max():,} chars")
        return 0
    if args.in_dir != IN and not args.no_smoke:
        raise SystemExit("the smoke sets read the first-round files; give --no-smoke with --in-dir")
    full = {}
    for case in ("restaurant", "caption", "street"):
        d = with_context(pd.read_parquet(os.path.join(args.in_dir, f"f4_{case}.parquet")))
        full[case] = d
        d.to_parquet(os.path.join(IN, f"f4_{case}_{args.suffix}.parquet"), index=False)
    caps = with_context(pd.read_parquet(os.path.join(args.in_dir, "f4_captions.parquet")))
    caps.to_parquet(os.path.join(IN, f"f4_captions_{args.suffix}.parquet"), index=False)
    if args.no_smoke:
        for case, d in full.items():
            print(f"[x2] {case}: {len(d)} texts; longest {d.text.str.len().max():,} chars")
        print(f"[x2] captions: {len(caps)} answers")
        return 0

    def strat(d, n):
        return d.groupby(["model", "field"], group_keys=False).sample(n=n, random_state=SEED)

    def longest(d, model, k, drop):
        t = d[(d.model == model) & (d.field == "reasoning") & ~d.doc_id.isin(drop)]
        return t.assign(n=t.text.str.len()).nlargest(k, "n").drop(columns="n")

    smoke = {}
    r = strat(full["restaurant"], 4)
    smoke["restaurant"] = pd.concat([r, longest(full["restaurant"], "qwen3.5-9b", 10, set(r.doc_id))])
    c = strat(full["caption"], 4)
    cp = sorted(d for d in copied_docs() if "|caption|" in d and d.endswith("|response") and d not in set(c.doc_id))
    cp = full["caption"][full["caption"].doc_id.isin(pd.Series(cp).sample(n=min(10, len(cp)), random_state=SEED))]
    smoke["caption"] = pd.concat([c, cp])
    s60 = set(pd.read_parquet(os.path.join(IN, "f4_streetsmoke_desc.parquet")).doc_id)
    s = full["street"][full["street"].doc_id.isin(s60)]
    smoke["street"] = pd.concat([s, longest(full["street"], "qwen3.5-9b", 10, s60)])
    k = caps.groupby("model", group_keys=False).sample(n=6, random_state=SEED)
    llama = caps[caps.model.str.startswith("llama") & ~caps.doc_id.isin(k.doc_id)]
    stripped = llama.text.str.strip()
    quoted = llama[stripped.str.startswith('"') & stripped.str.endswith('"') & (stripped.str.count('"') == 2)]
    smoke["captions"] = pd.concat([k, quoted.sample(n=min(10, len(quoted)), random_state=SEED)])

    ids = {}
    for name, d in smoke.items():
        d = d.drop_duplicates("doc_id")
        d.to_parquet(os.path.join(IN, f"f4_x2smoke_{name}.parquet"), index=False)
        ids[name] = sorted(d.doc_id)
        print(f"[x2] smoke {name}: {len(d)} texts")
    with open(os.path.join(IN, "x2_smoke_doc_ids.json"), "w") as f:
        json.dump(ids, f, indent=1)
    for case, d in full.items():
        print(f"[x2] {case}: {len(d)} texts; longest {d.text.str.len().max():,} chars")
    print(f"[x2] captions: {len(caps)} answers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
