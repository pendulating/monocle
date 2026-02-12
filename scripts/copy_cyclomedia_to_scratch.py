#!/usr/bin/env python3
"""Copy Cyclomedia images from NFS to /scratch with a flatter directory layout.

Source layout (4 levels):
    /share/ju/cyclomedia/raw/manhattan_2025_1k/{group}/{recording}/faces/{face}.jpg

Target layout (1 level of grouping):
    /scratch/mwf62/cyclomedia/manhattan_2025_1k/{group}/{sample_id}.jpg

Reads the existing parquet manifest to get all paths (no scanning needed),
builds new paths vectorized, copies in parallel, and saves an updated parquet.

Usage:
    python scripts/copy_cyclomedia_to_scratch.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def _copy_one(pair: tuple[str, str]) -> tuple[bool, str]:
    """Copy a single file. Returns (success, error_msg)."""
    src, dst = pair
    try:
        shutil.copy2(src, dst)
        return True, ""
    except Exception as e:
        return False, f"{src}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Copy Cyclomedia images to /scratch (flat layout)")
    parser.add_argument(
        "--parquet",
        default="/share/pierson/matt/mllmsci/data/cyclomedia/manhattan_2025_1k.parquet",
        help="Path to existing parquet manifest",
    )
    parser.add_argument(
        "--dest-root",
        default="/scratch/mwf62/cyclomedia/manhattan_2025_1k",
        help="Destination root on /scratch",
    )
    parser.add_argument(
        "--output-parquet",
        default="/share/pierson/matt/mllmsci/data/cyclomedia/manhattan_2025_1k_scratch.parquet",
        help="Path for updated parquet manifest with /scratch paths",
    )
    parser.add_argument("--workers", type=int, default=64, help="Number of parallel copy threads")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip files already copied")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    args = parser.parse_args()

    # ── 1. Read parquet and build new paths (vectorized) ──
    print(f"Reading parquet: {args.parquet}", flush=True)
    df = pd.read_parquet(args.parquet)
    total = len(df)
    print(f"  {total:,} rows, columns: {list(df.columns)}", flush=True)

    # Extract group prefix from sample_id (first 5 chars = group dir)
    # sample_id format: "{recording_id}_{face}" e.g. "W0ELLJQ5_F"
    # group = first 5 chars of recording_id = first 5 chars of sample_id
    dest_root = args.dest_root
    groups = df["sample_id"].str[:5]
    new_paths = dest_root + "/" + groups + "/" + df["sample_id"] + ".jpg"

    df["image_path_original"] = df["image_path"]
    src_paths = df["image_path"].values
    dst_paths = new_paths.values
    df["image_path"] = new_paths

    print(f"  Sample mapping:", flush=True)
    for i in range(min(3, total)):
        print(f"    {src_paths[i]}", flush=True)
        print(f"    -> {dst_paths[i]}", flush=True)

    # ── 2. Pre-create group directories ──
    unique_groups = groups.unique()
    print(f"\nCreating {len(unique_groups)} group directories on {dest_root}", flush=True)
    for g in unique_groups:
        os.makedirs(os.path.join(dest_root, g), exist_ok=True)

    # ── 3. Build copy list, optionally skipping existing ──
    copy_pairs = list(zip(src_paths, dst_paths))
    if args.skip_existing:
        before = len(copy_pairs)
        copy_pairs = [(s, d) for s, d in copy_pairs if not os.path.exists(d)]
        skipped = before - len(copy_pairs)
        if skipped:
            print(f"  Skipping {skipped:,} already-copied files", flush=True)

    if not copy_pairs:
        print("All files already copied!", flush=True)
    else:
        print(f"\nCopying {len(copy_pairs):,} files with {args.workers} threads...", flush=True)

        t0 = time.time()
        done = 0
        errors = 0
        last_report = t0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_copy_one, pair): pair for pair in copy_pairs}
            for fut in as_completed(futures):
                ok, err_msg = fut.result()
                done += 1
                if not ok:
                    errors += 1
                    if errors <= 10:
                        print(f"  ERROR: {err_msg}", flush=True)

                now = time.time()
                if now - last_report >= 5.0 or done == len(copy_pairs):
                    elapsed = now - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    pct = done / len(copy_pairs) * 100
                    eta = (len(copy_pairs) - done) / rate if rate > 0 else 0
                    print(
                        f"  [{pct:5.1f}%] {done:,}/{len(copy_pairs):,} | "
                        f"{rate:.0f} files/s | "
                        f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s | "
                        f"errors {errors}",
                        flush=True,
                    )
                    last_report = now

        elapsed = time.time() - t0
        print(f"\nCopy done: {done:,} files in {elapsed:.1f}s ({done/elapsed:.0f} files/s), {errors} errors", flush=True)

    # ── 4. Save updated parquet ──
    print(f"\nSaving updated parquet: {args.output_parquet}", flush=True)
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    print(f"  {len(df):,} rows, columns: {list(df.columns)}", flush=True)
    print(f"\nDone! Use the new manifest with:", flush=True)
    print(f"  data.parquet_path={args.output_parquet}", flush=True)


if __name__ == "__main__":
    main()
