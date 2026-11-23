"""Draw the all-neighborhood sample of the single-word probe.

Round 3 read the 10 highest and 10 lowest NTAs of each proxy. This draw
takes EVERY residential NTA of the round-2 pool (197) and up to
`--per-nta` images in each, spread over the NTA's tracts, so the test
becomes a dose-response over 197 neighborhoods
(`research/scripts/probe_word_dose.py`) and the extremes contrast is a special
case of it. Writes `<out>/images.parquet` with 1 row per image and the
NTA proxies, and `<out>/summary.json`.

    .venv-vllm025cu129/bin/python research/scripts/probe_sample_all.py --per-nta 500 \\
        --out outputs/_monocle/probe_words_all/sample
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.scripts.probe_sample_nta import DATA_NTA, DATA_R1, NTA_COLS, draw_nta, log, tract_to_nta  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-nta", type=int, default=500)
    ap.add_argument("--min-img", type=int, default=60)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    nta = pd.read_parquet(DATA_NTA / "nta_proxies.parquet")
    nta["homeless_per_capita"] = nta["homeless_density"] * nta["area_km2"] / nta["population"]
    cat = pd.read_parquet(DATA_R1 / "catalog_tracts.parquet").merge(
        tract_to_nta(), on="boroct2020", how="left")
    pool = nta[nta["n_img"] >= args.min_img]
    log(f"{len(nta)} NTAs, {len(pool)} with >= {args.min_img} images")
    rng = np.random.default_rng(args.seed)
    parts = []
    for r in pool.sort_values("nta2020").itertuples():
        parts.append(draw_nta(cat, r.nta2020, args.per_nta, rng))
    df = pd.concat(parts, ignore_index=True)
    df = df[["image_path", "sample_id", "geoid", "boroct2020", "nta2020", "face", "recording_id"]]
    df = df.merge(nta[[c for c in NTA_COLS if c in nta]], on="nta2020", how="left")
    df = df.sort_values("image_path").reset_index(drop=True)
    df.to_parquet(out / "images.parquet", index=False)
    per = df.groupby("nta2020").size()
    summ = {"images": int(len(df)), "ntas": int(df["nta2020"].nunique()),
            "tracts": int(df["geoid"].nunique()), "per_nta": args.per_nta,
            "min_per_nta": int(per.min()), "ntas_at_cap": int((per == args.per_nta).sum()),
            "seed": args.seed}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    log(json.dumps(summ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
