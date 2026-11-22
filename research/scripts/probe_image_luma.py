"""Luminance statistics per image, for the luminance control of the `Black` token.

The base model's `Black` association rests on a token that is also a
colour word. If the association follows how dark the picture is, a
partial correlation that controls for the image luminance removes it.
This script writes, per image of the sample: the mean luminance (Rec.
601, 0 to 255), its SD, and the share of pixels darker than 50.

    .venv-vllm025cu129/bin/python research/scripts/probe_image_luma.py --workers 8
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path("/share/ju/matt/mllmsci")
SAMPLE = REPO / "outputs/_monocle/probe_words_all/sample/images.parquet"
OUT = REPO / "outputs/_monocle/probe_black_control/luma.parquet"


def stats(path: str) -> dict:
    try:
        im = Image.open(path).convert("L")
        im.draft("L", (256, 256))
        a = np.asarray(im.resize((128, 128)), dtype=np.float32)
        return {"image_path": path, "luma_mean": float(a.mean()), "luma_sd": float(a.std()),
                "dark_share": float((a < 50).mean())}
    except Exception as e:  # a missing file must not stop the pass
        return {"image_path": path, "luma_mean": np.nan, "luma_sd": np.nan,
                "dark_share": np.nan, "error": repr(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(SAMPLE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = pd.read_parquet(args.images, columns=["image_path"])["image_path"].tolist()
    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap(stats, paths, chunksize=64)):
            rows.append(r)
            if i % 5000 == 0:
                print(f"[luma] {i}/{len(paths)}", flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[luma] wrote {args.out}: {len(df)} rows, {int(df.luma_mean.isna().sum())} failed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
