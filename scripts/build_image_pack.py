#!/usr/bin/env python3
"""Copy the unique images of the pair tables into a pack of blob files.

Why
---
The pairvqa stage reads each image from the raw tree on `/share/ju`. A cold
open costs about 8 NFS round trips, and a battery of 1,000,000 pairs makes
15,400,000 of them over only 1,100,000 unique files. The server collapses
above about 30 shards. See `vlm-narratives-docs/nfs-load-pairvqa-sweeps.md`.

This script reads each unique image ONE time and writes the bytes into blob
files of `--per-blob` images (about 4 GB at 16,384). A stage that sets
`runtime.image_pack=<out-dir>` then reads each image with 1 `pread`, and the
bytes are the same, thus the results do not change.

What it writes
--------------
    <out-dir>/blob-NNNN.bin        the image bytes
    <out-dir>/parts/index-NNNN.parquet   the index of 1 blob (path, blob, offset, length)
    <out-dir>/index.parquet        all parts, sorted by path
    <out-dir>/missing.txt          the paths that could not be read
    <out-dir>/manifest.json        written LAST; the reader trusts only a pack
                                   that holds it

A run that stops can start again with the same arguments. It keeps every blob
whose index part exists and builds only the rest.

Use
---
    python scripts/build_image_pack.py \\
        --tables multirun/pair_tables_1m/*_pairs.parquet \\
        --out-dir multirun/pair_tables_1m/image_pack --workers 8
    python scripts/build_image_pack.py --paths list.txt --out-dir ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dagspaces.common.image_pack import (  # noqa: E402
    INDEX_NAME, MANIFEST_NAME, PACK_VERSION, blob_name,
)

PATH_COLUMNS = ("image_path", "image_path_a", "image_path_b")


def collect_paths(tables: List[str], path_lists: List[str]) -> List[str]:
    """The sorted unique image paths of the tables and the lists."""
    unique = set()
    for table in tables:
        schema = pq.read_schema(table)
        cols = [c for c in PATH_COLUMNS if c in schema.names]
        if not cols:
            raise SystemExit(f"{table} holds none of the columns {PATH_COLUMNS}")
        frame = pq.read_table(table, columns=cols)
        for c in cols:
            unique.update(p for p in frame.column(c).to_pylist() if p)
        print(f"[pack] {os.path.basename(table)}: {len(unique)} unique paths so far",
              flush=True)
    for lst in path_lists:
        with open(lst) as fh:
            unique.update(ln.strip() for ln in fh if ln.strip())
    # Sorted by path: the siblings of a directory sit together, thus the
    # source server reads 1 directory at a time.
    return sorted(unique)


def read_file(path: str) -> Tuple[str, Optional[bytes], str]:
    try:
        with open(path, "rb") as fh:
            return path, fh.read(), ""
    except Exception as exc:  # noqa: BLE001
        return path, None, f"{type(exc).__name__}: {exc}"


def build_blob(k: int, paths: List[str], out_dir: str, workers: int,
               missing_fh) -> Tuple[int, int]:
    """Write blob k and its index part. Gives (images written, bytes written)."""
    blob_path = os.path.join(out_dir, blob_name(k))
    part_path = os.path.join(out_dir, "parts", f"index-{k:04d}.parquet")
    tmp_blob = f"{blob_path}.tmp"
    rows_path: List[str] = []
    rows_off: List[int] = []
    rows_len: List[int] = []
    offset = 0
    with open(tmp_blob, "wb") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        for path, data, err in pool.map(read_file, paths):
            if data is None:
                missing_fh.write(f"{path}\t{err}\n")
                missing_fh.flush()
                continue
            out.write(data)
            rows_path.append(path)
            rows_off.append(offset)
            rows_len.append(len(data))
            offset += len(data)
    os.replace(tmp_blob, blob_path)
    table = pa.table({
        "path": pa.array(rows_path, pa.string()),
        "blob": pa.array([k] * len(rows_path), pa.int32()),
        "offset": pa.array(rows_off, pa.int64()),
        "length": pa.array(rows_len, pa.int64()),
    })
    tmp_part = f"{part_path}.tmp"
    pq.write_table(table, tmp_part)
    os.replace(tmp_part, part_path)
    return len(rows_path), offset


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", nargs="*", default=[],
                    help="Pair tables or any parquet with an image path column. "
                         "Globs are accepted.")
    ap.add_argument("--paths", nargs="*", default=[],
                    help="Text files with 1 image path on each line.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-blob", type=int, default=16384,
                    help="Images in 1 blob. 16,384 is about 4 GB.")
    ap.add_argument("--workers", type=int, default=8,
                    help="Threads that read the source files. Keep this small: "
                         "the source server is the shared one.")
    args = ap.parse_args(argv)

    tables = sorted({t for pat in args.tables for t in glob.glob(pat)})
    if not tables and not args.paths:
        raise SystemExit("give --tables or --paths")
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(os.path.join(out_dir, MANIFEST_NAME)):
        print(f"[pack] {out_dir} is complete already; remove {MANIFEST_NAME} to build again")
        return 0
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)

    t0 = time.time()
    paths = collect_paths(tables, args.paths)
    n_blobs = (len(paths) + args.per_blob - 1) // args.per_blob
    print(f"[pack] {len(paths)} unique images in {n_blobs} blobs "
          f"(collected in {time.time() - t0:.0f}s)", flush=True)

    total_images = 0
    total_bytes = 0
    with open(os.path.join(out_dir, "missing.txt"), "a") as missing_fh:
        for k in range(n_blobs):
            part_path = os.path.join(out_dir, "parts", f"index-{k:04d}.parquet")
            if os.path.exists(part_path) and os.path.exists(os.path.join(out_dir, blob_name(k))):
                part = pq.read_table(part_path)
                total_images += part.num_rows
                total_bytes += sum(part.column("length").to_pylist())
                print(f"[pack] blob {k}/{n_blobs}: exists, kept", flush=True)
                continue
            t1 = time.time()
            n, b = build_blob(k, paths[k * args.per_blob:(k + 1) * args.per_blob],
                              out_dir, args.workers, missing_fh)
            total_images += n
            total_bytes += b
            dt = time.time() - t1
            print(f"[pack] blob {k + 1}/{n_blobs}: {n} images, {b / 1e9:.2f} GB in "
                  f"{dt:.0f}s ({n / max(dt, 1e-9):.0f} images/s); "
                  f"total {total_bytes / 1e9:.1f} GB, elapsed {(time.time() - t0) / 60:.1f} min",
                  flush=True)

    parts = [pq.read_table(os.path.join(out_dir, "parts", f"index-{k:04d}.parquet"))
             for k in range(n_blobs)]
    index = pa.concat_tables(parts).sort_by("path")
    tmp_index = os.path.join(out_dir, INDEX_NAME + ".tmp")
    pq.write_table(index, tmp_index)
    os.replace(tmp_index, os.path.join(out_dir, INDEX_NAME))

    manifest = {
        "version": PACK_VERSION,
        "n_images": int(index.num_rows),
        "n_blobs": n_blobs,
        "per_blob": args.per_blob,
        "total_bytes": int(total_bytes),
        "sources": tables + list(args.paths),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp_manifest = os.path.join(out_dir, MANIFEST_NAME + ".tmp")
    with open(tmp_manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    os.replace(tmp_manifest, os.path.join(out_dir, MANIFEST_NAME))
    print(f"[pack] complete: {index.num_rows} images, {total_bytes / 1e9:.1f} GB, "
          f"{(time.time() - t0) / 60:.1f} min → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
