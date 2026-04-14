#!/usr/bin/env python3
"""Create a parquet dataset from Cyclomedia panoramic cube-face images for VQA.

Cyclomedia data is organized as:
    {root}/{group}/{recording}/faces/{face}.jpg
    {root}/{group}/{recording}/manifest.json

Each recording has 6 cube faces: F(ront), B(ack), L(eft), R(ight), U(p), D(own).
The manifest.json contains the imageId and lat/lon coordinates.

This script walks the directory tree in parallel, optionally parses manifests,
filters to selected faces, and outputs a parquet file ready for the urbanvqa pipeline.

Usage:
    # Build manifest with horizontal faces only (default, fast -- skips JSON parsing)
    python scripts/create_cyclomedia_dataset.py \
        --image_dir /share/ju/cyclomedia/raw/manhattan_2025_1k \
        --output_path data/cyclomedia_manhattan_scaffolding.parquet

    # Include lat/lon from manifest.json (slower -- reads ~97K JSON files)
    python scripts/create_cyclomedia_dataset.py \
        --image_dir /share/ju/cyclomedia/raw/manhattan_2025_1k \
        --output_path data/cyclomedia_manhattan_scaffolding.parquet \
        --parse_manifests

    # Include all faces
    python scripts/create_cyclomedia_dataset.py \
        --image_dir /share/ju/cyclomedia/raw/manhattan_2025_1k \
        --output_path data/cyclomedia_manhattan_all_faces.parquet \
        --faces F,B,L,R,U,D

    # Only forward-facing images
    python scripts/create_cyclomedia_dataset.py \
        --image_dir /share/ju/cyclomedia/raw/manhattan_2025_1k \
        --output_path data/cyclomedia_manhattan_front_only.parquet \
        --faces F

    # Limit to first N recordings for testing
    python scripts/create_cyclomedia_dataset.py \
        --image_dir /share/ju/cyclomedia/raw/manhattan_2025_1k \
        --output_path data/cyclomedia_manhattan_test.parquet \
        --max_recordings 100

    # Control parallelism
    python scripts/create_cyclomedia_dataset.py \
        --image_dir /share/ju/cyclomedia/raw/manhattan_2025_1k \
        --output_path data/cyclomedia_manhattan_scaffolding.parquet \
        --workers 64

    # Enrich with catalog metadata (timestamps, vehicle heading, etc.)
    # Required for trajectory-based graph construction in urbanroamvqa.
    # Catalog CSVs are at /share/ju/cyclomedia/pull/recordings_*_2025_*.csv
    python scripts/create_cyclomedia_dataset.py \
        --image_dir /share/ju/cyclomedia/raw/manhattan_2025_1k \
        --output_path data/cyclomedia_manhattan_enriched.parquet \
        --parse_manifests \
        --catalog_csv /share/ju/cyclomedia/pull/recordings_manhattan_2025_chunks/manhattan_2025_part1of4.csv \
        --catalog_csv /share/ju/cyclomedia/pull/recordings_manhattan_2025_chunks/manhattan_2025_part2of4.csv \
        --catalog_csv /share/ju/cyclomedia/pull/recordings_manhattan_2025_chunks/manhattan_2025_part3of4.csv \
        --catalog_csv /share/ju/cyclomedia/pull/recordings_manhattan_2025_chunks/manhattan_2025_part4of4.csv
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from tqdm import tqdm


# Cube-face labels in Cyclomedia panoramic captures
ALL_FACES = {"F", "B", "L", "R", "U", "D"}
HORIZONTAL_FACES = {"F", "B", "L", "R"}

# Nominal bearing (degrees 0–360) of each cube face when F is 0° (forward).
# U/D have no single horizontal direction → None.
FACE_BEARING_DEG: dict[str, Optional[float]] = {
    "F": 0.0,
    "R": 90.0,
    "B": 180.0,
    "L": 270.0,
    "U": None,
    "D": None,
}


def _safe_float(value, /) -> Optional[float]:
    """Return float(value) or None if invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None





def _process_recording(
    rec_path: str,
    rec_name: str,
    face_filenames: set[str],
    parse_manifests: bool,
) -> Optional[list[tuple]]:
    """Process a single recording directory. Returns rows or None if skipped.

    Uses EAFP (try/except) instead of stat-before-open to minimize NFS round-trips.
    """
    faces_dir = os.path.join(rec_path, "faces")

    # Scan faces dir -- EAFP: skip the isdir() check, just try scandir
    try:
        present_files = {
            e.name for e in os.scandir(faces_dir)
            if e.is_file(follow_symlinks=False)
        }
    except (OSError, NotADirectoryError):
        return None

    # Intersect with requested faces
    matched = face_filenames & present_files
    if not matched:
        return None

    # Parse manifest if requested -- EAFP: just try to open it
    recording_id = rec_name
    latitude = None
    longitude = None

    if parse_manifests:
        manifest_path = os.path.join(rec_path, "manifest.json")
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            rid = data.get("imageId")
            if rid:
                recording_id = rid
            label = data.get("label", "")
            if label and "," in label:
                parts = label.split(",")
                try:
                    longitude = float(parts[0])
                    latitude = float(parts[1])
                except (ValueError, IndexError):
                    pass
        except (OSError, json.JSONDecodeError):
            pass

    # Build rows for matched faces
    rows = []
    for face_fn in matched:
        face_label = face_fn[0]  # "F.jpg" -> "F"
        rows.append((
            f"{recording_id}_{face_label}",
            os.path.join(faces_dir, face_fn),
            recording_id,
            face_label,
            latitude,
            longitude,
        ))
    return rows


def _enumerate_recording_dirs(image_dir: str, workers: int) -> list[tuple[str, str]]:
    """Phase 1: Enumerate all recording directories in parallel.

    Scans group directories concurrently to build a flat list of
    (recording_path, recording_name) tuples. This is a lightweight
    operation -- just readdir calls, no file I/O.
    """
    # Collect group dirs
    try:
        group_entries = [
            e.path for e in os.scandir(image_dir)
            if e.is_dir(follow_symlinks=False)
        ]
    except OSError as exc:
        raise ValueError(f"Cannot read image_dir: {exc}") from exc

    all_recordings: list[tuple[str, str]] = []

    def _list_recordings_in_group(group_path: str) -> list[tuple[str, str]]:
        try:
            return [
                (e.path, e.name)
                for e in os.scandir(group_path)
                if e.is_dir(follow_symlinks=False)
            ]
        except OSError:
            return []

    # Parallel enumeration of groups (light I/O, just readdir)
    effective = min(workers, len(group_entries))
    with ThreadPoolExecutor(max_workers=effective) as pool:
        with tqdm(
            pool.map(_list_recordings_in_group, group_entries),
            total=len(group_entries),
            desc="  Enumerating groups",
            unit="grp",
        ) as pbar:
            for result in pbar:
                all_recordings.extend(result)
                pbar.set_postfix(recordings=len(all_recordings))

    return all_recordings


def _load_catalog(catalog_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate Cyclomedia recording catalog CSVs.

    These catalogs (from the Cyclomedia WFS API) contain temporal and vehicle
    metadata: recordedAt, recorderDirection, yawDegrees, orientation, height, etc.
    """
    dfs = []
    for path in catalog_paths:
        print(f"  Loading catalog: {path}")
        df = pd.read_csv(path)
        dfs.append(df)
    catalog = pd.concat(dfs, ignore_index=True)
    # Normalize join key
    if "imageId" in catalog.columns:
        catalog = catalog.rename(columns={"imageId": "recording_id"})
    catalog["recording_id"] = catalog["recording_id"].astype(str)
    catalog = catalog.drop_duplicates(subset=["recording_id"])
    print(f"  Catalog: {len(catalog):,} unique recordings")
    return catalog


# Columns to join from the catalog (skip columns already in the base dataset)
CATALOG_JOIN_COLUMNS = [
    "recording_id",
    "recordedAt",
    "recorderDirection",
    "yawDegrees",
    "orientation",
    "orientationPrecision",
    "yawPrecisionDegrees",
    "statePlaneX",
    "statePlaneY",
    "locationSRS",
    "height",
    "heightSystem",
    "groundLevelOffset",
    "latitudePrecision",
    "longitudePrecision",
    "heightPrecision",
    "year",
    "panoramaTileSchema",
    "tileSchema",
    "hasDepthMap",
    "isAuthorized",
    "productType",
]


def create_cyclomedia_dataset(
    image_dir: str,
    output_path: str,
    faces: Optional[set[str]] = None,
    max_recordings: Optional[int] = None,
    workers: int = 32,
    parse_manifests: bool = False,
    catalog_paths: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Create a parquet dataset from Cyclomedia panoramic images.

    Two-phase parallel scan:
      Phase 1: Enumerate all recording directories across all groups (parallel readdir).
      Phase 2: Process each recording in a flat thread pool (parallel faces scan + optional manifest read).

    This maximizes NFS concurrency by processing recordings independently across
    all groups simultaneously, rather than processing groups one-at-a-time per thread.

    The output parquet contains image paths and metadata only -- no prompt column.
    Prompts are supplied at inference time via the pipeline's prompt YAML config.

    Args:
        image_dir: Root directory of the Cyclomedia dataset.
        output_path: Path to output parquet file.
        faces: Set of face labels to include (default: horizontal F,B,L,R).
        max_recordings: Cap on number of recordings to process (for testing).
        workers: Number of threads for parallel scanning (default: 32).
        parse_manifests: If True, parse manifest.json for imageId and lat/lon.
            Adds ~97K file reads -- significantly slower. Default: False.
        catalog_paths: Optional list of Cyclomedia recording catalog CSV paths.
            When provided, enriches the dataset by joining temporal/vehicle metadata
            (recordedAt, recorderDirection, yawDegrees, etc.) from the WFS catalog.
            Required for trajectory-based graph construction in urbanroamvqa.

    Returns:
        DataFrame with columns: sample_id, image_path,
        recording_id, face, latitude, longitude (+ catalog columns if enriched).
    """
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")

    if faces is None:
        faces = HORIZONTAL_FACES
    faces = {f.upper() for f in faces}
    invalid = faces - ALL_FACES
    if invalid:
        raise ValueError(f"Invalid face labels: {invalid}. Must be subset of {ALL_FACES}")

    face_filenames = {f"{f}.jpg" for f in faces}

    print(f"Scanning Cyclomedia dataset at {image_dir}")
    print(f"  Faces: {sorted(faces)} | Workers: {workers} | Manifests: {'yes' if parse_manifests else 'skip'}")
    t0 = time.monotonic()

    # --- Phase 1: Enumerate all recording directories ---
    recordings = _enumerate_recording_dirs(image_dir, workers)
    t1 = time.monotonic()
    print(f"  Found {len(recordings)} recording directories ({t1 - t0:.1f}s)")

    if max_recordings is not None and len(recordings) > max_recordings:
        recordings = recordings[:max_recordings]
        print(f"  Trimmed to {max_recordings} recordings (--max_recordings)")

    # --- Phase 2: Process recordings in flat thread pool ---
    all_rows: list[tuple] = []
    recordings_with_faces = 0
    recordings_skipped = 0

    effective_workers = min(workers, len(recordings))

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {
            pool.submit(
                _process_recording,
                rec_path, rec_name, face_filenames, parse_manifests,
            ): idx
            for idx, (rec_path, rec_name) in enumerate(recordings)
        }

        with tqdm(
            as_completed(futures),
            total=len(futures),
            desc="  Scanning recordings",
            unit="rec",
        ) as pbar:
            for future in pbar:
                result = future.result()
                if result is not None:
                    all_rows.extend(result)
                    recordings_with_faces += 1
                else:
                    recordings_skipped += 1

                pbar.set_postfix(
                    images=len(all_rows),
                    valid=recordings_with_faces,
                    skip=recordings_skipped,
                )

    if not all_rows:
        raise ValueError(
            f"No face images found in {image_dir}. "
            "Expected structure: {{group}}/{{recording}}/faces/{{F,B,L,R,U,D}}.jpg"
        )

    elapsed_total = time.monotonic() - t0

    # --- Build DataFrame from tuples (faster than list-of-dicts) ---
    columns = ["sample_id", "image_path", "recording_id", "face", "latitude", "longitude"]
    df = pd.DataFrame(all_rows, columns=columns)

    # --- Enrich with catalog metadata ---
    if catalog_paths:
        print(f"\nEnriching dataset with catalog metadata...")
        catalog = _load_catalog(catalog_paths)
        # Select only columns that exist in catalog and are in our join list
        available = [c for c in CATALOG_JOIN_COLUMNS if c in catalog.columns]
        catalog_subset = catalog[available]
        n_before = len(df)
        df = df.merge(catalog_subset, on="recording_id", how="left")
        n_matched = df["recordedAt"].notna().sum() if "recordedAt" in df.columns else 0
        n_unique_matched = df.loc[df["recordedAt"].notna(), "recording_id"].nunique() if "recordedAt" in df.columns else 0
        print(f"  Joined {len(available) - 1} catalog columns")
        print(f"  Matched: {n_unique_matched:,} / {df['recording_id'].nunique():,} unique recordings "
              f"({n_matched:,} / {len(df):,} rows)")
        # Fill lat/lon from catalog if missing from manifest
        if "latitude" in df.columns and df["latitude"].isna().any():
            for src, dst in [("lat", "latitude"), ("lon", "longitude")]:
                if src in catalog.columns:
                    cat_col = catalog.set_index("recording_id")[src]
                    mask = df["latitude"].isna() if dst == "latitude" else df["longitude"].isna()
                    df.loc[mask, dst] = df.loc[mask, "recording_id"].map(cat_col)
            n_filled = df["latitude"].notna().sum() - (n_before - df["latitude"].isna().sum())
            if n_filled > 0:
                print(f"  Filled {n_filled:,} missing lat/lon from catalog")

    # Summary
    print(f"\nDataset summary:")
    print(f"  Recordings with faces: {recordings_with_faces}")
    print(f"  Recordings skipped: {recordings_skipped}")
    print(f"  Total images: {len(df)}")
    print(f"  Total time: {elapsed_total:.1f}s ({len(df) / max(elapsed_total, 0.01):.0f} images/s)")
    print(f"  Face distribution:")
    for face_label, count in df["face"].value_counts().sort_index().items():
        print(f"    {face_label}: {count}")
    if parse_manifests and df["latitude"].notna().any():
        print(f"  Latitude range: [{df['latitude'].min():.5f}, {df['latitude'].max():.5f}]")
        print(f"  Longitude range: [{df['longitude'].min():.5f}, {df['longitude'].max():.5f}]")

    # Save to parquet
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved dataset to {output_path}")
    print(f"  {len(df)} rows x {len(df.columns)} columns")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Create a parquet dataset from Cyclomedia panoramic face images"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Root directory of Cyclomedia dataset (e.g. /share/ju/cyclomedia/raw/manhattan_2025_1k)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output parquet file",
    )
    parser.add_argument(
        "--faces",
        type=str,
        default="F,B,L,R",
        help="Comma-separated face labels to include (default: F,B,L,R). Options: F,B,L,R,U,D",
    )
    parser.add_argument(
        "--max_recordings",
        type=int,
        default=None,
        help="Maximum number of recordings to process (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of threads for parallel directory scanning (default: 32)",
    )
    parser.add_argument(
        "--parse_manifests",
        action="store_true",
        default=False,
        help="Parse manifest.json for imageId and lat/lon (slower, adds ~97K file reads)",
    )
    parser.add_argument(
        "--catalog_csv",
        type=str,
        action="append",
        default=None,
        help="Path to Cyclomedia recording catalog CSV (from WFS API). "
             "Can be specified multiple times to concatenate chunks. "
             "Enriches dataset with recordedAt, recorderDirection, etc.",
    )

    args = parser.parse_args()

    faces = {f.strip().upper() for f in args.faces.split(",")}

    df = create_cyclomedia_dataset(
        image_dir=args.image_dir,
        output_path=args.output_path,
        faces=faces,
        max_recordings=args.max_recordings,
        workers=args.workers,
        parse_manifests=args.parse_manifests,
        catalog_paths=args.catalog_csv,
    )

    print("\nDataset preview:")
    print(df.head(10).to_string(index=False))
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
