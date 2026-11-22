#!/usr/bin/env python
"""F4: pool the canonical batch and batch 2 (D132) into 1 set of tables.

Each pooled file has the relative path of its canonical file under
`--out` (default `data/f4_traces/pooled/`). Thus a notebook needs only a new
data root (`F4_DATA=pooled`). The file names keep "1k"; the manifest gives
the true row counts.

| Pooled file | Canonical (under `--canonical-root`) | Batch 2 |
|-------------|--------------------------------------|---------|
| `f4_traces_1k.parquet` | same path | `b2/f4_traces_b2.parquet` |
| `f4_street_traces_1k.parquet` | same path | `b2/f4_street_traces_b2.parquet` |
| `f4_restaurants_nta.parquet` | same path | `b2/f4_restaurants_b2_nta.parquet` |
| `f4_street_1k.parquet` | same path | `b2/f4_street_b2.parquet` |
| `extract_input/f4_<case>_x2.parquet` | same path | `<b2-input-dir>/f4_<case>_b2x2.parquet` |
| `x2/<arm>/f4_<case>_x2_extractions.parquet` | same path | `b2/x2/<arm>/f4_<case>_b2x2_extractions.parquet` |
| `captions/f4_caption_hashtags.parquet` | same path | `b2/f4_caption_hashtags_b2.parquet` |
| `street_captions/f4_street_caption_hashtags.parquet` | same path | `b2/f4_street_caption_hashtags_b2.parquet` |

The street captions (D133) add `extract_input/f4_street_captions_x2.parquet`
and `x2/street_captions/f4_street_captions_x2_extractions.parquet`, with the
same rule. Since D133 the `street_caption` case of the street corpus must
have `--expect` images too.

Each file gets the column `batch` (`canonical` or `b2`). The restaurant NTA
table also gets `nta_capacity`, `nta_n` and `weight` (`nta_capacity /
nta_n`) from the batch 2 draw (`sample_images_nta.py`). The pooled
restaurant set is stratified by NTA, thus a rate for the whole city needs
`weight`. A rate inside 1 NTA needs no weight.

The script stops when:

- the columns of the 2 parts of a file are different;
- a key occurs in both parts (`doc_id`, else (`model`, `case`, `image_idx`), else `image_idx`);
- a case of a trace corpus does not have `--expect` images for each model;
- an extraction table has a schema version outside the x2 set.

Use:
    .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/pool_batches.py
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
F4 = os.path.join(ROOT, "data/f4_traces")
SCHEMAS = {"single-v2", "street-v3", "descriptors-v6-restaurant", "descriptors-v6-street", "caption-v4"}
SCHEMAS |= {s + "r" for s in SCHEMAS}
# (arm dir, case) of each extraction table
EXTRACTIONS = [("cues", "restaurant"), ("cues", "caption"), ("cues_street", "street"),
               ("desc_restaurant", "restaurant"), ("desc_street", "street"), ("captions", "captions"),
               ("street_captions", "street_captions")]

__version__ = "1.1.0"


def pairs(c: str, b: str, inp: str) -> list[tuple[str, str, str]]:
    """Give (pooled relative path, canonical path, batch 2 path) for each file."""
    out = [
        ("f4_traces_1k.parquet", os.path.join(c, "f4_traces_1k.parquet"), os.path.join(b, "f4_traces_b2.parquet")),
        ("f4_street_traces_1k.parquet", os.path.join(c, "f4_street_traces_1k.parquet"),
         os.path.join(b, "f4_street_traces_b2.parquet")),
        ("f4_restaurants_nta.parquet", os.path.join(c, "f4_restaurants_nta.parquet"),
         os.path.join(b, "f4_restaurants_b2_nta.parquet")),
        ("f4_street_1k.parquet", os.path.join(c, "f4_street_1k.parquet"), os.path.join(b, "f4_street_b2.parquet")),
        ("captions/f4_caption_hashtags.parquet", os.path.join(c, "captions/f4_caption_hashtags.parquet"),
         os.path.join(b, "f4_caption_hashtags_b2.parquet")),
        ("street_captions/f4_street_caption_hashtags.parquet",
         os.path.join(c, "street_captions/f4_street_caption_hashtags.parquet"),
         os.path.join(b, "f4_street_caption_hashtags_b2.parquet")),
    ]
    for case in ("restaurant", "caption", "street", "captions", "street_captions"):
        out.append((f"extract_input/f4_{case}_x2.parquet", os.path.join(c, f"extract_input/f4_{case}_x2.parquet"),
                    os.path.join(inp, f"f4_{case}_b2x2.parquet")))
    for arm, case in EXTRACTIONS:
        out.append((f"x2/{arm}/f4_{case}_x2_extractions.parquet",
                    os.path.join(c, f"x2/{arm}/f4_{case}_x2_extractions.parquet"),
                    os.path.join(b, f"x2/{arm}/f4_{case}_b2x2_extractions.parquet")))
    return out


def keys(d: pd.DataFrame) -> set:
    if "doc_id" in d:
        return set(d.doc_id)
    cols = [k for k in ("model", "case", "image_idx") if k in d]
    return set(map(tuple, d[cols].drop_duplicates().itertuples(index=False)))


def check_traces(name: str, d: pd.DataFrame, expect: int) -> dict:
    n = d.groupby(["model", "case"]).image_idx.nunique()
    bad = n[n != expect]
    if len(bad):
        raise SystemExit(f"{name}: not {expect} images for each (model, case):\n{bad}")
    return {f"{m}/{c}": int(v) for (m, c), v in n.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical-root", default=F4)
    ap.add_argument("--b2-root", default=os.path.join(F4, "b2"))
    ap.add_argument("--b2-input-dir", default=os.path.join(F4, "extract_input"),
                    help="the folder of f4_<case>_b2x2.parquet")
    ap.add_argument("--out", default=os.path.join(F4, "pooled"))
    ap.add_argument("--expect", type=int, default=10000, help="the images of each case after the pool; 0 = no count check (tests)")
    args = ap.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(args.canonical_root):
        raise SystemExit("--out must not be the canonical root")

    draw = pd.read_parquet(os.path.join(args.b2_root, "f4_restaurants_b2.parquet"),
                           columns=["NTA2020", "nta_capacity", "nta_n"]).drop_duplicates("NTA2020")
    report = {}
    for rel, cpath, bpath in pairs(args.canonical_root, args.b2_root, args.b2_input_dir):
        c, b = pd.read_parquet(cpath), pd.read_parquet(bpath)
        if list(c.columns) != list(b.columns):
            raise SystemExit(f"{rel}: the columns differ\n canonical {list(c.columns)}\n b2 {list(b.columns)}")
        both = keys(c) & keys(b)
        if both:
            raise SystemExit(f"{rel}: {len(both)} keys in both batches, for example {sorted(both)[:3]}")
        d = pd.concat([c.assign(batch="canonical"), b.assign(batch="b2")], ignore_index=True)
        entry = {"canonical": os.path.relpath(cpath, ROOT), "b2": os.path.relpath(bpath, ROOT),
                 "rows": {"canonical": len(c), "b2": len(b)}}
        if rel.startswith("x2/"):
            versions = set(d.schema_version.dropna())
            if versions - SCHEMAS:
                raise SystemExit(f"{rel}: schema versions outside the x2 set: {sorted(versions - SCHEMAS)}")
            entry["schema_versions"] = sorted(versions)
        if args.expect and rel in ("f4_traces_1k.parquet", "f4_street_traces_1k.parquet"):
            entry["images"] = check_traces(rel, d, args.expect)
        if rel == "f4_restaurants_nta.parquet":
            if args.expect and len(d) != args.expect:
                raise SystemExit(f"{rel}: {len(d)} images, not {args.expect}")
            d = d.merge(draw, on="NTA2020", how="left", validate="many_to_one")
            if d.nta_n.isna().any():
                raise SystemExit(f"{rel}: an NTA has no allocation in the batch 2 draw")
            cnt = d.groupby("NTA2020").size()
            if not (cnt == d.groupby("NTA2020").nta_n.first()).all():
                raise SystemExit(f"{rel}: an NTA count differs from its allocation")
            d["weight"] = d.nta_capacity / d.nta_n
            entry["ntas"] = int(cnt.size)
        if rel == "f4_street_1k.parquet":
            if args.expect and len(d) != args.expect:
                raise SystemExit(f"{rel}: {len(d)} images, not {args.expect}")
            entry["ntas"] = int(d.nta2020.nunique())
            entry["per_nta"] = {str(k): int(v) for k, v in d.groupby("nta2020").size().value_counts().items()}
        out = os.path.join(args.out, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        d.to_parquet(out, index=False)
        report[rel] = entry
        print(f"[pool] {rel}: {len(c):,} + {len(b):,} rows")

    manifest = {"script": os.path.relpath(__file__, ROOT), "version": __version__,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "expect": args.expect, "files": report}
    with open(os.path.join(args.out, "pool_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"[pool] wrote {len(report)} files and pool_manifest.json to {os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
