#!/usr/bin/env python
"""RQ1 figure: pick the 8 images for the per-patch R-lens words.

The figure puts 2 things side by side for each image: the words that the
R-lens reads from the image patches of gemma-4-12b, and the inference spans
of the final answer of gemma-4-12b to the same image (F4). It shows the gap
between what the patches carry and what the answer says.

| Cell | Images |
|------|--------|
| restaurant, low-income NTA | 2 |
| restaurant, high-income NTA | 2 |
| street, low-income NTA | 2 |
| street, high-income NTA | 2 |

The income groups are the bottom and the top fifth of the images of the case,
by the median household income of the image's NTA (ACS). An image can be
drawn only when the final answer of gemma-4-12b holds 2 inference spans or
more of the kinds in `KINDS` (the canonical F4 batch, x2 extraction). The
draw uses a fixed seed.

Outputs (data/rq1/):
| File | Content |
|------|---------|
| `patch_words_sample.parquet` | 1 row for each image: case, group, image_idx, image_path, NTA, income, photo_id, lens_case |
| `patch_words_spans.parquet` | the inference spans of the final answer of each image |

Use:
    F4_DATA=canonical .venv-mllmsci-vllm025cu129/bin/python research/scripts/rq1/patch_words_sample.py
    # another draw, for a check of the figure (the seed draws only the images)
    ... research/scripts/rq1/patch_words_sample.py --seed 3 --out data/rq1/seeds/s3
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault("F4_DATA", "canonical")  # the image_idx of the canonical batch
from research.notebooks.facct import _f4 as F4  # noqa: E402
from research.scripts.f8.build_cue_sample import canonical_images  # noqa: E402

SEED = 20261029
MODEL = "gemma-4-12b"  # the model of the R-lens
N_PER_CELL = 2
MIN_SPANS = 2
# The inference kinds that go past the shop front: not the kind of business,
# the place, the time or the brand, which a sign or a logo can give.
KINDS = ("atmosphere", "neighborhood_character", "popularity", "quality", "upkeep", "wealth", "demographic")
LENS_CASE = {"restaurant": "restaurants", "street": "street_photography"}
OUT = REPO / "data" / "rq1"


def income_groups(img: pd.DataFrame) -> pd.DataFrame:
    acs = pd.read_parquet(REPO / "data" / "f4_traces" / "nta_acs.parquet",
                          columns=["ntaname", "median_household_income"])
    img = img.merge(acs.rename(columns={"ntaname": "nta_name", "median_household_income": "income"}),
                    on="nta_name", how="left", validate="m:1")
    lo, hi = img["income"].quantile([0.2, 0.8])
    img["group"] = np.where(img["income"] <= lo, "low", np.where(img["income"] >= hi, "high", None))
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    raw = F4.load_extractions({c: p for c, p in F4.EXTRACTIONS.items() if c in ("restaurant", "street")})
    sp = F4.spans(raw)
    sp = sp[(sp["model"] == MODEL) & (sp["field"] == "response") & (sp["extraction_class"] == "inference")]
    print("inference kinds of the final answer:\n", sp.groupby(["case", "kind"]).size().to_string())
    sp = sp[sp["kind"].isin(KINDS)]
    n_spans = sp.groupby(["case", "image_idx"]).size().rename("n_spans").reset_index()

    rng = np.random.default_rng(a.seed)
    picks = []
    for case, img in canonical_images().items():
        img = income_groups(img).merge(n_spans[n_spans["case"] == case].drop(columns="case"),
                                       on="image_idx", how="left").fillna({"n_spans": 0})
        for group in ("low", "high"):
            pool = img[(img["group"] == group) & (img["n_spans"] >= MIN_SPANS)].sort_values("image_idx")
            print(f"{case} {group}: {len(pool)} images can be drawn of {int((img['group'] == group).sum())}")
            take = pool.iloc[rng.choice(len(pool), N_PER_CELL, replace=False)]
            picks.append(take.assign(case=case))
    sample = pd.concat(picks, ignore_index=True)[
        ["case", "group", "image_idx", "image_path", "nta_name", "boro", "income", "n_spans"]]
    sample["photo_id"] = [f"{c}_{i}" for c, i in zip(sample["case"], sample["image_idx"])]
    # The monocle question of the case. The image comes first in the prompt, so
    # the patch words do not depend on it.
    sample["lens_case"] = sample["case"].map(LENS_CASE)
    keep = sp.merge(sample[["case", "image_idx"]], on=["case", "image_idx"])
    keep = keep[["case", "image_idx", "kind", "valence", "extraction_text", "char_start"]].sort_values(
        ["case", "image_idx", "char_start"])
    a.out.mkdir(parents=True, exist_ok=True)
    sample.assign(seed=a.seed).to_parquet(a.out / "patch_words_sample.parquet", index=False)
    keep.to_parquet(a.out / "patch_words_spans.parquet", index=False)
    pd.set_option("display.width", 200)
    print(sample.to_string(index=False))
    for r in sample.itertuples():
        print(f"\n{r.photo_id} ({r.group}, {r.nta_name}, ${r.income:,.0f})")
        for s in keep[(keep["case"] == r.case) & (keep["image_idx"] == r.image_idx)].itertuples():
            print(f"  [{s.kind}] {s.extraction_text}")


if __name__ == "__main__":
    main()
