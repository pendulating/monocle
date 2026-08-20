"""Materialize a citywide Cyclomedia street-level manifest for pairwise VQA.

Stratified uniformly across all five borough datasets (manhattan, brooklyn,
queens, bronx, si) and the four street-level cube faces (F/B/L/R):
``--per-cell`` images per (dataset, face) cell, 20 cells total. U/D faces are
excluded (street-level scenes only). Sampling is uniform random within each
cell; deterministic via a fixed seed; a final global shuffle means a downstream
``head(n)`` subsample is still borough-balanced.

This is the general-purpose sibling of ``materialize_opf_vision_30k.py`` (which
is hard-wired to the OPF vision-head 30k training manifest). It backs the
``urbanpairvqa`` street-photography case (image-mode pairing over citywide
blocks); the default 25,000/cell = 500,000 rows = 100,000 per borough gives a
100k-pair run ~0.4x block reuse (most runs sample distinct blocks).

Output schema matches the existing manhattan_2025_1k / opf_vision_30k manifests
(``image_path`` canonical /share/ju path, ``sample_id``, ``recording_id``,
``face``, ``dataset`` + latitude/longitude/yawDegrees/recordedAt), so the
urbanpairvqa data config + ``_load_pairwise_manifest`` path need no changes.

Usage:
    python scripts/materialize_cyclomedia_citywide.py            # 500k default
    python scripts/materialize_cyclomedia_citywide.py --per-cell 5000   # 100k
"""

from __future__ import annotations

import argparse
import os

import polars as pl

from dagspaces.common.cyclomedia_catalog import CyclomediaCatalog


DEFAULT_OUTPUT = "/share/pierson/matt/mllmsci/data/cyclomedia/cyclomedia_all_2025_citywide_500k.parquet"
DEFAULT_DATASETS = (
    "manhattan_2025_1k",
    "brooklyn_2025_1k",
    "queens_2025_1k",
    "bronx_2025_1k",
    "si_2025_1k",
)
DEFAULT_FACES = ("F", "B", "L", "R")

# Same shape as the existing manhattan_2025_1k / opf_vision_30k manifests so the
# urbanpairvqa data config + _load_pairwise_manifest path don't need to change.
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
    p.add_argument("--per-cell", type=int, default=25000,
                   help="Rows per (dataset, face) stratum. 25000 → 500k total "
                        "(100k/borough). 20 cells = 5 boroughs × 4 faces.")
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--catalog-root", default="/share/ju/cyclomedia/catalog/v1")
    args = p.parse_args()

    cat = CyclomediaCatalog(root=args.catalog_root)
    print(f"[materialize] catalog at {cat.root}")
    print(f"[materialize] per-cell={args.per_cell:,} → target {args.per_cell * 20:,} rows "
          f"({args.per_cell * 4:,}/borough)")

    parts: list[pl.DataFrame] = []
    short_cells = []
    for dataset in DEFAULT_DATASETS:
        for face in DEFAULT_FACES:
            df = cat.query(
                datasets=[dataset],
                faces={face},
                columns=list(KEEP_COLUMNS),
            )
            n = df.height
            take = min(n, args.per_cell)
            if take < args.per_cell:
                short_cells.append((dataset, face, n))
            sampled = df.sample(n=take, seed=args.seed, with_replacement=False)
            parts.append(sampled)
            print(f"  {dataset:>20s}  face={face}  pool={n:>9,d}  sampled={take:,d}")

    out = pl.concat(parts, how="vertical_relaxed")
    # Final shuffle so consumers that take ``head(n)`` for downstream
    # subsampling get a globally-random (borough-balanced) subset.
    out = out.sample(n=out.height, seed=args.seed, shuffle=True)
    print(f"[materialize] total rows: {out.height:,d}")
    if short_cells:
        print(f"[materialize] WARNING: {len(short_cells)} cell(s) under per-cell target: {short_cells}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.write_parquet(args.output)
    print(f"[materialize] wrote {args.output}")
    print(f"[materialize] file size: {os.path.getsize(args.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
