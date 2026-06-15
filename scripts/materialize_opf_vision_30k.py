"""Materialize a 30k Cyclomedia manifest for the OPF vision-head Stage 1
scale-up training run.

Stratified across all five borough datasets (manhattan, brooklyn, queens,
bronx, si) and the four street-level cube faces (F/B/L/R), 1500 images
per (dataset, face) cell = 30,000 rows. U/D faces are excluded — the OPF
vision head trains on street-level scenes only. Sampling is uniform
random within each cell; deterministic via a fixed seed.

Output schema matches what
``dagspaces.opf_vision_labels.stages._common.load_input_frame`` expects:
``image_path`` (canonical /share/ju path), ``sample_id``,
``recording_id``, ``face``, ``dataset`` plus the same metadata columns
the existing manhattan_2025_1k manifest carries (latitude, longitude,
yawDegrees).

Usage:
    python scripts/materialize_opf_vision_30k.py
"""

from __future__ import annotations

import argparse
import os

import polars as pl

from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog


DEFAULT_OUTPUT = "/share/pierson/matt/mllmsci/data/cyclomedia/opf_vision_30k.parquet"
DEFAULT_DATASETS = (
    "manhattan_2025_1k",
    "brooklyn_2025_1k",
    "queens_2025_1k",
    "bronx_2025_1k",
    "si_2025_1k",
)
DEFAULT_FACES = ("F", "B", "L", "R")

# Columns we keep in the materialized manifest. Same shape as the existing
# manhattan_2025_1k_scratch parquet so the dagspace's _common.load_input_frame
# code path doesn't need to change.
KEEP_COLUMNS = (
    "sample_id",
    "image_path",
    "recording_id",
    "face",
    "dataset",
    "latitude",
    "longitude",
    "yawDegrees",
    "recordedAt",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--per-cell", type=int, default=1500,
                   help="Rows per (dataset, face) stratum.")
    p.add_argument("--seed", type=int, default=20260425)
    p.add_argument("--catalog-root", default="/share/ju/cyclomedia/catalog/v1")
    args = p.parse_args()

    cat = CyclomediaCatalog(root=args.catalog_root)
    print(f"[materialize] catalog at {cat.root}")
    print(f"[materialize] datasets: {cat.datasets()}")

    parts: list[pl.DataFrame] = []
    for dataset in DEFAULT_DATASETS:
        for face in DEFAULT_FACES:
            df = cat.query(
                datasets=[dataset],
                faces={face},
                columns=list(KEEP_COLUMNS),
            )
            n = df.height
            take = min(n, args.per_cell)
            sampled = df.sample(n=take, seed=args.seed, with_replacement=False)
            parts.append(sampled)
            print(f"  {dataset:>20s}  face={face}  pool={n:>9,d}  sampled={take:,d}")

    out = pl.concat(parts, how="vertical_relaxed")
    # Final shuffle so consumers that take ``head(n)`` for downstream
    # subsampling get a globally-random subset instead of a borough-block.
    out = out.sample(n=out.height, seed=args.seed, shuffle=True)
    print(f"[materialize] total rows: {out.height:,d}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.write_parquet(args.output)
    print(f"[materialize] wrote {args.output}")
    print(f"[materialize] file size: {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
