#!/usr/bin/env python
"""F4: the retry pass of the x2 round (D130) for the texts that got a copy of an example.

The extractor sometimes gives the answer of 1 prompt example in place of an
answer for the input text. The caption smoke run gave 4 of 46 answers the 3
spans of example 3, and no span of their own; 3 of the 4 begin almost as
example 2 does. LangExtract keeps the order of the examples, thus the same
prompt with the examples in a different order gives a different neighbor.
This follows the retry passes of D105, D121, and D124.

| Command | Work |
|---------|------|
| `schemas` | write `<schema>r.yaml` for each x2 schema: the same text, the positive examples in reverse order, each `none` example at its place, `schema_version` + "r" |
| `flag --root R --prefix P` | read the runs of each arm under `R`, flag the copy suspects, and write `extract_input/f4_<P>_<arm>.parquet` (the x2 input rows of those texts) and `<R>/<P>_flags.csv` |

A copy suspect is a text with 2 or more spans that equal an example span and
are not quotable, or a text with no quotable span and 1 such span. The merge
(`merge_extractions.py --retry-runs`) puts the retry rows in place of the
first rows, and flags the texts again.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/x2_retry.py schemas
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/x2_retry.py flag --root multirun/f4x2_smoke --prefix x2rsmoke
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/x2_retry.py flag --root multirun/f4x2 --prefix x2r
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/x2_retry.py flag --root multirun/f4x2_b2 --prefix b2x2r
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/x2_retry.py flag --root multirun/f4x2_sc --prefix scx2r
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/x2_retry.py flag --root multirun/f4x2_subj --prefix x2r \
        --arms x2_cues_subj x2_cues_street_subj

`--arms` limits both commands. F9 (D142) gives its 2 arms only: then the
`flag` output files (`f4_x2r_x2_cues_subj.parquet`) and `<root>/x2r_flags.csv`
under `multirun/f4x2_subj` touch no F4 file.

Warning: the prefix names the output files. `x2r` writes the inputs of the
canonical retry; give batch 2 the prefix `b2x2r`, and the street captions of
both batches (D133) the prefix `scx2r`.
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import yaml

ROOT = "/share/ju/matt/mllmsci"
IN = os.path.join(ROOT, "data/f4_traces/extract_input")
CONF = os.path.join(ROOT, "dagspaces/pairwisevqa/conf/extract")
# arm -> (schema, the x2 input files that the arm reads: smoke, full, and batch 2 of D132)
ARMS = {
    "x2_cues": ("urban_cues_single_v2", ("restaurant_x2", "caption_x2", "x2smoke_restaurant", "x2smoke_caption",
                                         "restaurant_b2x2", "caption_b2x2")),
    "x2_cues_street": ("urban_cues_street_v3", ("street_x2", "x2smoke_street", "street_b2x2")),
    "x2_desc_restaurant": ("aesthetic_descriptors_restaurant_v6", ("restaurant_x2", "x2smoke_restaurant", "restaurant_b2x2")),
    "x2_desc_street": ("aesthetic_descriptors_street_v6", ("street_x2", "x2smoke_street", "street_b2x2")),
    "x2_captions": ("caption_options_v4", ("captions_x2", "x2smoke_captions", "captions_b2x2",
                                           "street_captions_x2", "street_captions_b2x2")),
    # F9 (D142): the F9 answers and the F4 answers of the same images, 1 schema.
    "x2_cues_subj": ("urban_cues_single_subj_v1", ("restaurant_subj_x2", "x2smoke_restaurant_subj", "restaurant_x2",
                                                   "restaurant_b2x2", "x2smoke_restaurant")),
    "x2_cues_street_subj": ("urban_cues_street_subj_v1", ("street_subj_x2", "x2smoke_street_subj", "street_x2",
                                                          "street_b2x2", "x2smoke_street")),
}
F9_ARMS = ("x2_cues_subj", "x2_cues_street_subj")

__version__ = "1.3.0"


def example_spans(schema: str) -> set:
    ex = yaml.safe_load(open(os.path.join(CONF, schema + ".yaml")))["examples"]
    return {e["extraction_text"] for x in ex for e in x["extractions"] if e["extraction_class"] != "none"}


def copy_flags(d: pd.DataFrame, spans: set) -> pd.DataFrame:
    """Give, for each text of `d`: `copies` (example spans that are not quotable), `quotable`, `copy_suspect`."""
    cls = d.extraction_class
    real = d[cls.notna() & (cls != "none")]
    q = real.is_quotable.fillna(False).astype(bool)
    copied = real.extraction_text.isin(spans) & ~q
    per = pd.DataFrame({"copies": copied.groupby(real.doc_id).sum(), "quotable": q.groupby(real.doc_id).sum()})
    per = per.reindex(d.doc_id.unique()).fillna(0).astype(int)
    per["copy_suspect"] = (per.copies >= 2) | ((per.copies >= 1) & (per.quotable == 0))
    return per


def write_schemas(arms, out_dir: str = CONF) -> None:
    from dagspaces.common.langextract_backend import spec_from_config, validate_examples
    for schema, _ in (ARMS[a] for a in arms):
        src = os.path.join(CONF, schema + ".yaml")
        text = open(src).read()
        cfg = yaml.safe_load(text)
        ex = cfg["examples"]
        none_at = [i for i, e in enumerate(ex) if all(x["extraction_class"] == "none" for x in e["extractions"])]
        pos = [e for i, e in enumerate(ex) if i not in none_at][::-1]
        out = []
        for i in range(len(ex)):
            out.append(ex[i] if i in none_at else pos.pop(0))
        cfg["examples"] = out
        cfg["schema_version"] = str(cfg["schema_version"]) + "r"
        problems = validate_examples(spec_from_config(cfg))
        if problems:
            raise SystemExit(f"{schema}r: {problems}")
        head = (f"# @package extract\n# {cfg['schema_version']} (D130): the RETRY pass of `{schema}.yaml`, written by\n"
                f"# `research/scripts/f4/x2_retry.py schemas`. The same text; the positive examples are in\n"
                f"# reverse order, and each `none` example stays at its place. It reads only the\n"
                f"# texts that got a copy of an example in the first pass.\n")
        with open(os.path.join(out_dir, schema + "r.yaml"), "w") as fh:
            fh.write(head)
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True, width=100)
        print(f"[retry] wrote {schema}r.yaml ({len(ex)} examples, none at {none_at})")


def flag(root: str, prefix: str, arms) -> None:
    rows = []
    for arm in arms:
        schema, cases = ARMS[arm]
        files = glob.glob(os.path.join(root, arm, "runs", "*", "outputs", "extractions", "*.parquet"))
        if not files:
            print(f"[retry] {arm}: no runs under {root}")
            continue
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        per = copy_flags(d, example_spans(schema))
        bad = set(per.index[per.copy_suspect])
        src = pd.concat([pd.read_parquet(os.path.join(IN, f"f4_{c}.parquet")) for c in cases
                         if os.path.exists(os.path.join(IN, f"f4_{c}.parquet"))])
        src = src[src.doc_id.isin(bad)].drop_duplicates("doc_id")
        out = os.path.join(IN, f"f4_{prefix}_{arm}.parquet")
        if src.empty:
            # No input file: the queue skips the retry step of this arm.
            if os.path.exists(out):
                os.remove(out)
        else:
            src.to_parquet(out, index=False)
        print(f"[retry] {arm}: {len(per)} texts, {len(bad)} copy suspects -> {os.path.relpath(out, ROOT)}")
        rows += [{"arm": arm, "doc_id": i, **per.loc[i].to_dict()} for i in sorted(bad)]
    pd.DataFrame(rows).to_csv(os.path.join(root, f"{prefix}_flags.csv"), index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("schemas")
    sc.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    sc.add_argument("--out-dir", default=CONF, help="a check can write the files elsewhere and diff them")
    f = sub.add_parser("flag")
    f.add_argument("--root", required=True)
    f.add_argument("--prefix", required=True, choices=["x2r", "x2rsmoke", "b2x2r", "scx2r"])
    # The F4 roots hold no F9 arm, and the F9 root no F4 arm; an arm with no runs is skipped.
    f.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    args = ap.parse_args()
    if args.cmd == "schemas":
        write_schemas(args.arms, args.out_dir)
    else:
        flag(os.path.join(ROOT, args.root) if not os.path.isabs(args.root) else args.root, args.prefix, args.arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
