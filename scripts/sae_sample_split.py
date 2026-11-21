"""Split the all-NTA sample for the dictionary work.

300 images per NTA train the dictionaries; the rest (200, or fewer in
the 1 NTA under the cap) are held out for selection and validation.
The split is by image inside each NTA under a fixed seed, so every NTA
sits in both halves. Writes `outputs/_monocle/sae/sample/images.parquet`
with a `split` column (`train` | `heldout`).

    .venv-vllm025cu129/bin/python scripts/sae_sample_split.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/share/ju/matt/mllmsci")
SRC = REPO / "outputs/_monocle/probe_words_all/sample/images.parquet"
OUT = REPO / "outputs/_monocle/sae/sample"
TRAIN_PER_NTA = 300
SEED = 777


def main() -> int:
    df = pd.read_parquet(SRC)
    rng = np.random.default_rng(SEED)
    df["split"] = "heldout"
    for nta, idx in df.groupby("nta2020").indices.items():
        pick = rng.permutation(idx)[:TRAIN_PER_NTA]
        df.iloc[pick, df.columns.get_loc("split")] = "train"
    df = df.sort_values("image_path").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "images.parquet", index=False)
    summ = {"images": int(len(df)), "ntas": int(df.nta2020.nunique()),
            "train": int((df.split == "train").sum()), "heldout": int((df.split == "heldout").sum()),
            "train_per_nta": TRAIN_PER_NTA, "seed": SEED, "source": str(SRC)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
