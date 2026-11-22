#!/usr/bin/env python
"""F4: merge the (model, case) runs into 1 trace corpus.

Each row is 1 answer: 6 models x 2 cases x 1,000 images. Most answers come
from `data/f4_traces/runs/`. Two runs have a replacement (`REPLACE`):

| Run | Replacement | Why |
|-----|-------------|-----|
| qwen3.5-9b, restaurant | the 156 answers cut at 4,096 come from `runs-maxtok16384/` (16,384 tokens) | D91, D92: the reruns are new samples; combining is accepted |
| llama-3.2-11b, caption | all 1,000 answers come from `runs-llamactx/` | the first run had a context of 8,192, thus the answers stopped at 1,761 tokens, not 4,096 (D89) |

The column `source_run` names the folder of each answer, and `max_tokens`
gives its limit. `cut` is true when the answer stopped at the limit.

`--set street` merges the street photography runs (D116 to D120) into a
separate corpus, `f4_street_traces_1k.parquet`: 6 models x 2 cases (`street`,
`street_caption`) x 985 images, with no replacement. The street rows have no
restaurant: `uid` is the recording id, and `camis` and `dba` are null. The
NTA of each image is in `f4_street_1k.parquet` (join on `image_idx`).

Batch 2 (D132) gives its own sample, its own run folder, and its own
replacement. With `--runs` other than `runs`, `REPLACE` does not apply; only
the `--replace` items apply (`MODEL:CASE:FOLDER:MAX_TOKENS:MODE`, MODE
`cut_only` or `all`). `--cases` selects the cases (batch 2 has no
`street_caption`).

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/merge_runs.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/merge_runs.py --set street
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/merge_runs.py --sample data/f4_traces/b2/f4_restaurants_b2.parquet \
        --runs runs-b2 --replace qwen3.5-9b:restaurant:runs-b2-maxtok16384:16384:cut_only \
        --out data/f4_traces/b2/f4_traces_b2.parquet
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/merge_runs.py --set street --cases street \
        --sample data/f4_traces/b2/f4_street_b2.parquet --runs runs-b2 --out data/f4_traces/b2/f4_street_traces_b2.parquet
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import polars as pl

ROOT = "/share/ju/matt/mllmsci"
F4 = os.path.join(ROOT, "data/f4_traces")
SAMPLE = os.path.join(F4, "f4_restaurants_1k.parquet")
OUT = os.path.join(F4, "f4_traces_1k.parquet")
STREET_SAMPLE = os.path.join(F4, "f4_street_1k.parquet")
STREET_OUT = os.path.join(F4, "f4_street_traces_1k.parquet")
STREET_CASES = ["street", "street_caption"]
MODELS = ["gemma-4-12b", "gemma-4-31b", "qwen3.5-9b", "qwen3.8-27b", "llama-3.2-11b", "llama-3.2-90b"]
CASES = ["restaurant", "caption"]
# (model, case) -> (folder, max_tokens, "cut_only" or "all")
REPLACE = {
    ("qwen3.5-9b", "restaurant"): ("runs-maxtok16384", 16384, "cut_only"),
    ("llama-3.2-11b", "caption"): ("runs-llamactx", 4096, "all"),
}
COLS = ["image_idx", "sample_id", "seed", "prompt", "sampling", "chat_template_kwargs", "reasoning",
        "response", "finish_reason", "prompt_tokens", "completion_tokens", "error"]

__version__ = "1.2.1"


def _read(folder: str, model: str, case: str, max_tokens: int) -> pl.DataFrame:
    p = os.path.join(F4, folder, model, case, "responses.parquet")
    if not os.path.isfile(p):
        raise SystemExit(f"missing {p}")
    return (pl.read_parquet(p).select(COLS)
            .with_columns(pl.lit(model).alias("model"), pl.lit(case).alias("case"),
                          pl.lit(folder).alias("source_run"), pl.lit(max_tokens).alias("max_tokens")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["restaurant", "street"], default="restaurant")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample", default=None, help="the image table (default: the canonical sample of the set)")
    ap.add_argument("--runs", default="runs", help="the run folder under data/f4_traces/ (batch 2: runs-b2)")
    ap.add_argument("--cases", nargs="+", default=None, help="the cases (default: all cases of the set)")
    ap.add_argument("--replace", action="append", default=None, metavar="MODEL:CASE:FOLDER:MAX_TOKENS:MODE",
                    help="a replacement run; more than 1 is permitted. When given, REPLACE does not apply")
    args = ap.parse_args()
    street = args.set == "street"
    cases, sample_path = (STREET_CASES, STREET_SAMPLE) if street else (CASES, SAMPLE)
    cases = args.cases or cases
    sample_path = args.sample or sample_path
    if args.replace is not None:
        replace = {}
        for item in args.replace:
            model, case, folder, max_tokens, mode = item.split(":")
            if mode not in ("cut_only", "all"):
                raise SystemExit(f"--replace {item}: MODE is cut_only or all")
            replace[(model, case)] = (folder, int(max_tokens), mode)
    else:
        replace = {} if street or args.runs != "runs" else REPLACE
    out = args.out or (STREET_OUT if street else OUT)
    sample = pl.read_parquet(sample_path)
    if street:
        sample = sample.select("image_idx", pl.col("recording_id").alias("uid"),
                               pl.lit(None, pl.Utf8).alias("camis"), pl.lit(None, pl.Utf8).alias("dba"),
                               "borough", "image_path")
    else:
        sample = sample.select("image_idx", "uid", "camis", "dba", "borough", "image_path")

    parts = []
    for model in MODELS:
        for case in cases:
            base = _read(args.runs, model, case, 4096)
            rep = replace.get((model, case))
            if rep is None:
                parts.append(base)
                continue
            folder, max_tokens, mode = rep
            new = _read(folder, model, case, max_tokens)
            # A run with error rows in its jsonl stores the token counts as float
            # (pandas). Cast the rerun to the types of the base run; a value that is
            # not a whole number stops the cast.
            new = new.select([pl.col(c).cast(base.schema[c], strict=True) if c in base.schema else pl.col(c)
                              for c in new.columns])
            if mode == "all":
                parts.append(new)
            else:
                cut = base.filter(pl.col("finish_reason") == "length")["image_idx"]
                if set(cut.to_list()) != set(new["image_idx"].to_list()):
                    raise SystemExit(f"{model}/{case}: the rerun does not hold the cut images")
                parts.append(pl.concat([base.filter(~pl.col("image_idx").is_in(cut.implode())), new]))
    df = (pl.concat(parts)
          .with_columns((pl.col("finish_reason") == "length").alias("cut"))
          .join(sample, on="image_idx", how="left")
          .sort("model", "case", "image_idx"))
    n = df.group_by("model", "case").len()
    if n["len"].min() != sample.height or n.height != len(MODELS) * len(cases):
        raise SystemExit(f"incomplete corpus:\n{n}")
    df.write_parquet(out)

    summary = (df.group_by("model", "case")
               .agg(pl.len().alias("answers"), pl.col("error").is_not_null().sum().alias("errors"),
                    pl.col("cut").sum().alias("cut"),
                    (pl.col("response").fill_null("").str.strip_chars().str.len_chars() == 0).sum().alias("empty_response"),
                    pl.col("completion_tokens").median().alias("tok_p50"),
                    pl.col("source_run").unique().sort().str.join(", ").alias("sources"))
               .sort("model", "case"))
    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "set": args.set, "rows": df.height, "sample": os.path.relpath(os.path.abspath(sample_path), ROOT),
        "runs": args.runs, "cases": cases,
        "replacements": {f"{m}/{c}": list(v) for (m, c), v in replace.items()},
        "summary": summary.to_dicts(),
    }
    with open(out.replace(".parquet", "_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    with pl.Config(tbl_rows=20, tbl_cols=10, tbl_width_chars=200):
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
