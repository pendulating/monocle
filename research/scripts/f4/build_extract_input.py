#!/usr/bin/env python
"""F4: write the input of the span extraction, 1 parquet for each case.

Each row is 1 text for the extractor (d-0012, D95):

| field | Rows | Source column |
|-------|------|---------------|
| reasoning | the 4 thinking models (Gemma, Qwen) | `reasoning` |
| response | all 6 models | `response` |

The script drops a text that is empty. All 44 empty answers are traces that
stopped at the token limit before the answer. Any other filter (for example
the `min_chars` of the stage) is a setting of the extraction run.

A third file, `f4_captions.parquet`, holds the final answers of the caption
case only (6,000 rows). The caption extraction reads it (D101 to D104).

`doc_id` is `<model>|<case>|<image_idx>|<field>`. The stage joins its output
rows back to this table on `doc_id`. The file name gives the case to the stage
(`f4_restaurant.parquet` gives `f4_restaurant`).

With `--src data/f4_traces/f4_street_traces_1k.parquet`, the script writes
`f4_street.parquet` and `f4_street_caption.parquet` (D116 to D120). It
writes `f4_captions.parquet` only when the source holds the caption case, and
`f4_street_captions.parquet` (the final answers of `street_caption`, D133)
only when the source holds that case. The manifest takes the name of the
source.

`--response-from-trace MODEL:IMAGE_IDX:MARKER` (F9, D142) fills 1 empty
`response`: the text after the last MARKER in `reasoning` becomes the
response text (the model wrote its answer inside the trace). MARKER takes
backslash escapes (`\n`). The column `response_from_trace` is true for that
row only. The `reasoning` text does not change.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_extract_input.py
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_extract_input.py \
        --src data/f4_traces/f4_street_traces_1k.parquet
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/build_extract_input.py \
        --src data/f4_traces/subj/f9_restaurant_traces.parquet --out-dir data/f4_traces/subj/extract_input \
        --response-from-trace 'qwen3.8-27b:8826:Let me draft.\n\n"'
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import polars as pl

ROOT = "/share/ju/matt/mllmsci"
F4 = os.path.join(ROOT, "data/f4_traces")
SRC = os.path.join(F4, "f4_traces_1k.parquet")
OUT_DIR = os.path.join(F4, "extract_input")
KEEP = ["model", "case", "field", "image_idx", "uid", "camis", "dba", "borough",
        "cut", "source_run", "max_tokens", "completion_tokens", "response_from_trace"]

__version__ = "1.4.0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--response-from-trace", action="append", default=[], metavar="MODEL:IMAGE_IDX:MARKER")
    args = ap.parse_args()

    d = pl.read_parquet(args.src).with_columns(pl.lit(False).alias("response_from_trace"))
    filled = []
    for item in args.response_from_trace:
        model, idx, marker = item.split(":", 2)
        marker = marker.encode().decode("unicode_escape")
        hit = d.filter((pl.col("model") == model) & (pl.col("image_idx") == int(idx)))
        if hit.height != 1:
            raise SystemExit(f"--response-from-trace {item}: {hit.height} rows, not 1")
        trace, resp = hit["reasoning"][0] or "", hit["response"][0] or ""
        if resp.strip():
            raise SystemExit(f"--response-from-trace {item}: the response is not empty")
        at = trace.rfind(marker)
        if at < 0:
            raise SystemExit(f"--response-from-trace {item}: the marker is not in the trace")
        text = trace[at + len(marker):]
        mask = (pl.col("model") == model) & (pl.col("image_idx") == int(idx))
        d = d.with_columns(pl.when(mask).then(pl.lit(text)).otherwise(pl.col("response")).alias("response"),
                           (pl.col("response_from_trace") | mask).alias("response_from_trace"))
        filled.append({"model": model, "image_idx": int(idx), "marker": marker, "start": at + len(marker),
                       "chars": len(text)})
    parts = []
    for field in ("reasoning", "response"):
        # The flag marks the copied response only, not the trace of the same answer.
        parts.append(d.with_columns(pl.col(field).fill_null("").alias("text"), pl.lit(field).alias("field"),
                                    (pl.col("response_from_trace") & (field == "response")).alias("response_from_trace")))
    rows = pl.concat(parts)
    dropped = rows.filter(pl.col("text").str.strip_chars() == "")
    rows = (rows.filter(pl.col("text").str.strip_chars() != "")
            .with_columns(pl.concat_str([pl.col("model"), pl.col("case"), pl.col("image_idx").cast(pl.Utf8),
                                         pl.col("field")], separator="|").alias("doc_id"))
            .select(["doc_id", "text"] + KEEP)
            .sort("model", "field", "image_idx"))
    if rows["doc_id"].n_unique() != rows.height:
        raise SystemExit("doc_id is not unique")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}
    for case in sorted(rows["case"].unique()):
        part = rows.filter(pl.col("case") == case)
        part.write_parquet(os.path.join(args.out_dir, f"f4_{case}.parquet"))
        summary[case] = (part.group_by("model", "field").len().sort("model", "field")
                         .select(pl.concat_str(["model", "field"], separator="/"), "len").rows())
    # The caption job reads the final answers of the caption case only.
    captions = rows.filter((pl.col("case") == "caption") & (pl.col("field") == "response"))
    if captions.height:
        captions.write_parquet(os.path.join(args.out_dir, "f4_captions.parquet"))
        summary["captions"] = captions.group_by("model").len().sort("model").rows()
    # The same for the caption prompt on the street views (D133).
    street_captions = rows.filter((pl.col("case") == "street_caption") & (pl.col("field") == "response"))
    if street_captions.height:
        street_captions.write_parquet(os.path.join(args.out_dir, "f4_street_captions.parquet"))
        summary["street_captions"] = street_captions.group_by("model").len().sort("model").rows()
    manifest = {
        "script": os.path.relpath(__file__, ROOT), "version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": os.path.relpath(args.src, ROOT), "rows": rows.height,
        "dropped_empty": dropped.group_by("model", "case", "field").len().rows(),
        "response_from_trace": filled,
        "rows_by_case": summary,
    }
    name = "manifest.json" if os.path.abspath(args.src) == SRC else \
        f"manifest_{os.path.basename(args.src).replace('.parquet', '')}.json"
    with open(os.path.join(args.out_dir, name), "w") as fh:
        json.dump(manifest, fh, indent=1)
    with pl.Config(tbl_rows=30):
        print(rows.group_by("case", "model", "field").len().sort("case", "model", "field"))
    print(f"dropped {dropped.height} empty texts; wrote {rows.height} rows to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
