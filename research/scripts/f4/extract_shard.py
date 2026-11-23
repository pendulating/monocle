#!/usr/bin/env python
"""F4: run 1 shard of the span extraction inside the current SLURM job.

This is the parent extraction path (d-0012, D96), unchanged: the config of
`pipeline=extract_traces`, the stage `trace_extract`, and `run_shard_job`.
`submit_trace_shards.py` submits through submitit to the old GPU partitions.
The F4 extractor needs 1 B300 on the g3 cluster (`--clusters=g3`), thus
`research/scripts/f4/extract.sbatch` calls this script from a plain sbatch job.

The input is `data/f4_traces/extract_input/f4_<case>.parquet`
(`build_extract_input.py`): 1 row for each reasoning trace or final answer.
`--arm captions` reads `f4_captions.parquet` (the caption answers only) with
the schema `caption_options_v3` (D101 to D104).

Output, as in `submit_trace_shards.py`:
    <base-dir>/runs/f4_<case>__<shard:04d>/outputs/extractions/<schema>_s<k>_<stamp>.parquet
    <base-dir>/runs/f4_<case>__<shard:04d>/pipeline_manifest.json
    <base-dir>/runs/f4_<case>__<shard:04d>/chunks/   (the resume unit)

A requeued job skips each chunk that exists. Merge with:
    python research/scripts/merge_trace_extractions.py <base-dir>/runs --out <root>

Use:
    python research/scripts/f4/extract_shard.py --case restaurant --shard 0 --shards 4 --base-dir DIR
    python research/scripts/f4/extract_shard.py --case restaurant --shard 0 --shards 4 --base-dir DIR \\
        --set runtime.sample_n=50 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

INPUT_DIR = os.path.join(ROOT, "data/f4_traces/extract_input")
KEEP = ["model", "case", "field", "image_idx", "uid", "camis", "dba", "borough",
        "cut", "source_run", "max_tokens", "completion_tokens"]

# The arms, in the shape of `submit_trace_shards.ARMS`.
_COMMON = [
    "trace_extract.reasoning_column=text",
    "trace_extract.id_column=doc_id",
    "trace_extract.keep_columns=[" + ",".join(KEEP) + "]",
    # The input mixes 6 judges. The `model` column of each row names its own.
    "trace_extract.judge_model=f4_mixed",
    # No length filter (d-0014.2, D98). The parent value of 200 drops 122
    # short answers, and 116 of them are real Llama answers.
    "trace_extract.min_chars=1",
]
ARMS = {
    # The urban cue spans of every trace and answer (D93 to D100).
    "cues": dict(
        experiment="F4CUES_DSV4",
        extractor="deepseek-v4-flash/instruct",
        schema="urban_cues_single_v1",
        runs={f"f4_{c}": os.path.join(INPUT_DIR, f"f4_{c}.parquet") for c in ("restaurant", "caption")},
        tp=1, seqs=64, mem=200, extra=list(_COMMON),
    ),
    # The urban cue spans of the street (angle) case, with the class
    # `photo_advice` (D125). Version 2 fixes missed descriptions (D128). The input is the descriptor input, thus the
    # offsets match the descriptor run. `streetsmoke` is the smoke input.
    "cues_street": dict(
        experiment="F4CUES_STREET_DSV4",
        extractor="deepseek-v4-flash/instruct",
        schema="urban_cues_street_v2",
        runs={f"f4_{c}": os.path.join(INPUT_DIR, f"f4_{c}_desc.parquet") for c in ("street", "streetsmoke")},
        tp=1, seqs=64, mem=200, extra=list(_COMMON),
    ),
    # The captions and hashtags of the caption answers (D101 to D104).
    "captions": dict(
        experiment="F4CAPS_DSV4",
        extractor="deepseek-v4-flash/instruct",
        schema="caption_options_v3",
        runs={"f4_captions": os.path.join(INPUT_DIR, "f4_captions.parquet")},
        tp=1, seqs=128, mem=200, extra=list(_COMMON),
    ),
    # The retry pass of the captions arm (D105): 64 flagged Llama answers.
    "captions_retry": dict(
        experiment="F4CAPS_DSV4_RETRY",
        extractor="deepseek-v4-flash/instruct",
        schema="caption_options_v3r",
        runs={"f4_captions_retry": os.path.join(INPUT_DIR, "f4_captions_retry.parquet")},
        tp=1, seqs=128, mem=200, extra=list(_COMMON),
    ),
    # The aesthetic descriptors of every trace and answer (D109 to D112).
    # The input holds "-" for U+2011 (`research/scripts/f4/build_descriptor_input.py`);
    # `descsmoke` is the stratified smoke input of 120 texts.
    "descriptors": dict(
        experiment="F4DESC_DSV4",
        extractor="deepseek-v4-flash/instruct",
        schema="aesthetic_descriptors_v3",
        # The caption case stays out of the full run (D115); its input
        # remains for a later choice.
        runs={f"f4_{c}": os.path.join(INPUT_DIR, f"f4_{c}_desc.parquet")
              for c in ("restaurant", "caption", "descsmoke", "descsmoke_restaurant")},
        tp=1, seqs=64, mem=200, extra=list(_COMMON),
    ),
    # The descriptors of the street (angle) case, with the attribute `target`
    # (D120, D122). Version 5 adds a Llama angle example (D123); the version 4
    # run is in multirun/f4_descriptors_street_dsv4, the version 5 run uses
    # F4X_TAG=v5. `streetsmoke` is the smoke input.
    "descriptors_street": dict(
        experiment="F4DESC_STREET_DSV4",
        extractor="deepseek-v4-flash/instruct",
        schema="aesthetic_descriptors_v5",
        runs={f"f4_{c}": os.path.join(INPUT_DIR, f"f4_{c}_desc.parquet") for c in ("street", "streetsmoke")},
        tp=1, seqs=64, mem=200, extra=list(_COMMON),
    ),
    # The retry pass of the street descriptors (D124): the texts that got a
    # copy of an example under version 5.
    "descriptors_street_retry": dict(
        experiment="F4DESC_STREET_DSV4_RETRY",
        extractor="deepseek-v4-flash/instruct",
        schema="aesthetic_descriptors_v5r",
        runs={"f4_descriptors_street_retry": os.path.join(INPUT_DIR, "f4_street_desc_retry.parquet")},
        tp=1, seqs=64, mem=200, extra=list(_COMMON),
    ),
    # The retry pass of the descriptors arm (D121): 20 restaurant traces that
    # got a copy of an example.
    "descriptors_retry": dict(
        experiment="F4DESC_DSV4_RETRY",
        extractor="deepseek-v4-flash/instruct",
        schema="aesthetic_descriptors_v3r",
        runs={"f4_descriptors_retry": os.path.join(INPUT_DIR, "f4_restaurant_desc_retry.parquet")},
        tp=1, seqs=64, mem=200, extra=list(_COMMON),
    ),
}
# The x2 round (D130): every extraction again, with the schemas that apply
# the advisor review of 2026-10-26. The input is `f4_<case>_x2.parquet`
# (`build_x2_input.py`): U+2011 is "-", and the column `context` holds 1 line
# of `additional_context` for each text. `x2smoke_<case>` are the smoke sets.
_X2 = [c for c in _COMMON if not c.startswith("trace_extract.keep_columns")] + [
    "trace_extract.keep_columns=[" + ",".join(KEEP + ["context"]) + "]",
    # The stage config has no such key; "+" adds it.
    "+trace_extract.context_column=context",
]


def _x2(experiment, schema, cases, seqs=128):
    return dict(experiment=experiment, extractor="deepseek-v4-flash/instruct", schema=schema,
                runs={f"f4_{c}": os.path.join(INPUT_DIR, f"f4_{c}.parquet") for c in cases},
                tp=1, seqs=seqs, mem=200, extra=list(_X2))


# `<case>_b2x2` are the inputs of batch 2 (D132), with the same arms.
ARMS.update({
    "x2_cues": _x2("F4X2_CUES", "urban_cues_single_v2",
                   ("restaurant_x2", "caption_x2", "x2smoke_restaurant", "x2smoke_caption",
                    "restaurant_b2x2", "caption_b2x2")),
    "x2_cues_street": _x2("F4X2_CUES_STREET", "urban_cues_street_v3", ("street_x2", "x2smoke_street", "street_b2x2")),
    "x2_desc_restaurant": _x2("F4X2_DESC_RESTAURANT", "aesthetic_descriptors_restaurant_v6",
                              ("restaurant_x2", "x2smoke_restaurant", "restaurant_b2x2")),
    "x2_desc_street": _x2("F4X2_DESC_STREET", "aesthetic_descriptors_street_v6",
                          ("street_x2", "x2smoke_street", "street_b2x2")),
    "x2_captions": _x2("F4X2_CAPTIONS", "caption_options_v4", ("captions_x2", "x2smoke_captions", "captions_b2x2",
                                                                # The caption prompt on the street views (D133).
                                                                "street_captions_x2", "street_captions_b2x2")),
})
# F9 (D139, D142; `research/docs/f9-extraction-design.md`): the cue
# schemas plus the class `verdict`. Each arm reads the F9 answers and, again,
# the F4 answers of the same images, thus the F9 test has 1 schema on both
# sides. Their spans go under `multirun/f4x2_subj/`, apart from the x2 spans
# of record.
ARMS.update({
    "x2_cues_subj": _x2("F9X2_CUES_SUBJ", "urban_cues_single_subj_v1",
                        ("restaurant_subj_x2", "x2smoke_restaurant_subj", "restaurant_x2", "restaurant_b2x2",
                         "x2smoke_restaurant")),
    "x2_cues_street_subj": _x2("F9X2_CUES_STREET_SUBJ", "urban_cues_street_subj_v1",
                               ("street_subj_x2", "x2smoke_street_subj", "street_x2", "street_b2x2",
                                "x2smoke_street")),
})
# The retry arms of the x2 round: the copy suspects of each arm, with the
# examples in reverse order (`research/scripts/f4/x2_retry.py`). `x2r_<arm>` reads the
# suspects of the full run, `x2rsmoke_<arm>` those of the smoke run, and
# `b2x2r_<arm>` those of batch 2, `scx2r_<arm>` those of the street captions (D133).
for _arm in [a for a in ARMS if a.startswith("x2_")]:
    ARMS["x2r" + _arm[2:]] = _x2(ARMS[_arm]["experiment"] + "_RETRY", ARMS[_arm]["schema"] + "r",
                                 (f"x2r_{_arm}", f"x2rsmoke_{_arm}", f"b2x2r_{_arm}", f"scx2r_{_arm}"))

# F11, the reference R4 (D150, D151): DeepSeek-V4-Flash reads 1,000 pairwise
# traces of each (model, case) with the PAIRWISE schema `urban_cues_v2`, as
# Qwen3.8-27B read them. The input is `build_f11_input.py`. The 4 runs share 1
# input; the columns `f11_model` and `f11_case` name the run of each row.
# F10.8: the traces of the read prompt: the F9 street cues with the read
# question, the street descriptors, and the new criteria schema
# (`research/scripts/f10/build_cool_schemas.py`).
ARMS["cool_cues"] = _x2("F10X_COOL_CUES", "urban_cues_cool_v1", ("cool_x2",))
ARMS["cool_desc"] = _x2("F10X_COOL_DESC", "aesthetic_descriptors_street_v6", ("cool_x2",))
ARMS["cool_criteria"] = _x2("F10X_COOL_CRITERIA", "cool_criteria_v1", ("cool_x2",))

ARMS["f11_pw_cues"] = dict(
    experiment="F11_PW_CUES_DSV4",
    extractor="deepseek-v4-flash/instruct",
    schema="urban_cues_v2",
    runs={"f4_f11pw": os.path.join(INPUT_DIR, "f4_f11pw.parquet"),
          "f4_f11pwsmoke": os.path.join(INPUT_DIR, "f4_f11pwsmoke.parquet")},
    tp=1, seqs=64, mem=200,
    extra=[
        "trace_extract.reasoning_column=model_reasoning",
        "trace_extract.id_column=doc_id",
        "trace_extract.keep_columns=[pair_id,presented_label,presented_score,relative_label,f11_model,f11_case]",
        "trace_extract.judge_model=f11_mixed",
        "trace_extract.case=f11_pairwise",
    ],
)

ARM = ARMS["cues"]

__version__ = "1.12.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="cues", choices=sorted(ARMS))
    cases = sorted({r[len("f4_"):] for a in ARMS.values() for r in a["runs"]})
    ap.add_argument("--case", required=True, choices=cases,
                    help="restaurant or caption for the cues and descriptors arms; else the name of the arm")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--shard-rows", type=int, default=250, help="rows in 1 chunk, the resume unit")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="more Hydra overrides")
    ap.add_argument("--dry-run", action="store_true", help="compose and print the config; run nothing")
    args = ap.parse_args()

    from dagspaces.pairwisevqa.orchestrator import run_shard_job
    from dagspaces.pairwisevqa.submit_shards import build_payload
    from research.traces.submit_trace_shards import compose_cfg, write_hydra_record

    arm = ARMS[args.arm]
    case = f"f4_{args.case}"
    if case not in arm["runs"]:
        raise SystemExit(f"the {args.arm} arm has no case {args.case}; it has {sorted(arm['runs'])}")
    out_root = os.path.join(args.base_dir, "runs", f"{case}__{args.shard:04d}")
    cfg = compose_cfg(arm, case, args.shard, args.shards, args.shard_rows, out_root,
                      list(arm["extra"]) + list(args.set))
    payload = build_payload(cfg)
    eng = payload["cfg"]["model"]["engine_kwargs"]
    ex = payload["cfg"]["extract"]
    print(f"[f4-extract] {case} shard {args.shard}/{args.shards} -> {out_root}")
    print(f"[f4-extract] model={payload['cfg']['model']['model_source']} schema={ex['name']}/{ex['schema_version']}")
    print(f"[f4-extract] max_model_len={eng.get('max_model_len')} seqs={eng.get('max_num_seqs')} "
          f"tp={eng.get('tensor_parallel_size')} max_tokens={ex['max_tokens']} buffer={ex['max_char_buffer']} "
          f"min_chars={payload['cfg']['trace_extract']['min_chars']}")
    if args.dry_run:
        return 0
    write_hydra_record(cfg, out_root, f"f4_{args.arm}_dsv4", arm, case, args.shard)
    result = run_shard_job(payload)
    print("[f4-extract] metadata:", json.dumps(result.get("metadata", {}), default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
