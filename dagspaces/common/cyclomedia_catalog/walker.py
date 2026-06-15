"""Walk the Cyclomedia raw tree and emit one row per face JPEG.

The on-disk layout is:

    {raw_root}/{dataset}/{group}/{recording}/faces/{F,B,L,R,U,D}.jpg
    {raw_root}/{dataset}/{group}/{recording}/manifest.json

Walking NFS is metadata-bound. `fd` (Rust) parallelizes readdir much better
than a Python `os.walk`; we shell out to it and stream lines back. A threaded
`os.scandir` fallback keeps the module portable on hosts without `fd`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Optional

import polars as pl

from .schema import ALL_FACES

__all__ = ["walk_dataset", "FD_DEFAULT_PATH", "WalkResult"]

log = logging.getLogger(__name__)

# Absolute path to the cluster-local fd binary. Set by the user at
# /share/ju/matt/.cargo/bin/fd; prepend to PATH so subprocess can find it
# even when the venv doesn't expose cargo.
FD_DEFAULT_PATH = "/share/ju/matt/.cargo/bin/fd"


@dataclass(frozen=True)
class WalkResult:
    """A flat DataFrame plus bookkeeping from one dataset walk."""

    frames: pl.DataFrame
    dataset: str
    used_fd: bool


def _resolve_fd(fd_path: Optional[str]) -> Optional[str]:
    if fd_path and os.path.isfile(fd_path) and os.access(fd_path, os.X_OK):
        return fd_path
    found = shutil.which("fd")
    if found:
        return found
    if os.path.isfile(FD_DEFAULT_PATH) and os.access(FD_DEFAULT_PATH, os.X_OK):
        return FD_DEFAULT_PATH
    return None


def _parse_face_jpeg_path(path: str, dataset_root: str, dataset: str) -> Optional[tuple[str, str, str, str]]:
    """Parse a face JPEG path into (dataset, group, recording, face).

    Expected tail: {group}/{recording}/faces/{X.jpg}. Returns None for paths
    that don't match (e.g. misplaced files).
    """
    rel = os.path.relpath(path, dataset_root)
    parts = rel.split(os.sep)
    # parts = [group, recording, 'faces', 'F.jpg']
    if len(parts) != 4 or parts[2] != "faces":
        return None
    fname = parts[3]
    if not fname.endswith(".jpg") or len(fname) < 5:
        return None
    face = fname[0]
    if face not in ALL_FACES:
        return None
    return dataset, parts[0], parts[1], face


def _walk_with_fd(fd: str, dataset_root: str, dataset: str) -> list[tuple]:
    """Return list of (dataset, group, recording, face, image_path, file_size, file_mtime)."""
    # fd -0 emits NUL-separated paths → robust to weird filenames
    cmd = [
        fd,
        "--type", "f",
        "--extension", "jpg",
        "--absolute-path",
        "-0",
        ".",
        dataset_root,
    ]
    log.info("walker: fd cmd = %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, check=True)
    raw = proc.stdout
    # trailing NUL: split then drop empties
    paths = [p for p in raw.split(b"\x00") if p]

    rows: list[tuple] = []
    # Stat is I/O-bound on NFS → threads help even though GIL is held during
    # the actual os.stat syscall (glibc releases it).
    def _stat_one(path_bytes: bytes):
        path = path_bytes.decode("utf-8", errors="replace")
        parsed = _parse_face_jpeg_path(path, dataset_root, dataset)
        if parsed is None:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        ds, group, recording, face = parsed
        return (ds, group, recording, face, path, int(st.st_size), int(st.st_mtime))

    with ThreadPoolExecutor(max_workers=32) as pool:
        for r in pool.map(_stat_one, paths):
            if r is not None:
                rows.append(r)
    return rows


def _walk_with_scandir(dataset_root: str, dataset: str) -> list[tuple]:
    """Fallback: threaded os.scandir walk. Same return shape as _walk_with_fd."""
    rows: list[tuple] = []

    try:
        groups = [e.path for e in os.scandir(dataset_root) if e.is_dir(follow_symlinks=False)]
    except OSError as exc:
        raise ValueError(f"Cannot read dataset_root {dataset_root}: {exc}") from exc

    def _walk_group(group_path: str) -> list[tuple]:
        group = os.path.basename(group_path)
        local: list[tuple] = []
        try:
            recs = [e.path for e in os.scandir(group_path) if e.is_dir(follow_symlinks=False)]
        except OSError:
            return local
        for rec_path in recs:
            recording = os.path.basename(rec_path)
            faces_dir = os.path.join(rec_path, "faces")
            try:
                for e in os.scandir(faces_dir):
                    if not e.is_file(follow_symlinks=False):
                        continue
                    fname = e.name
                    if not fname.endswith(".jpg") or len(fname) < 5:
                        continue
                    face = fname[0]
                    if face not in ALL_FACES:
                        continue
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    local.append(
                        (dataset, group, recording, face, e.path, int(st.st_size), int(st.st_mtime))
                    )
            except OSError:
                continue
        return local

    with ThreadPoolExecutor(max_workers=min(64, max(1, len(groups)))) as pool:
        for result in pool.map(_walk_group, groups):
            rows.extend(result)
    return rows


def walk_dataset(
    raw_root: str,
    dataset: str,
    fd_path: Optional[str] = None,
) -> WalkResult:
    """Walk one dataset under `raw_root` and return a Polars DataFrame of face rows."""
    dataset_root = os.path.join(raw_root, dataset)
    if not os.path.isdir(dataset_root):
        raise ValueError(f"Dataset directory not found: {dataset_root}")

    fd = _resolve_fd(fd_path)
    if fd is not None:
        log.info("walker: using fd at %s", fd)
        rows = _walk_with_fd(fd, dataset_root, dataset)
        used_fd = True
    else:
        log.warning("walker: fd not found, falling back to threaded os.scandir")
        rows = _walk_with_scandir(dataset_root, dataset)
        used_fd = False

    schema = {
        "dataset": pl.Categorical,
        "group": pl.Utf8,
        "recording_dir": pl.Utf8,
        "face": pl.Categorical,
        "image_path": pl.Utf8,
        "file_size": pl.Int64,
        "file_mtime_unix": pl.Int64,
    }
    if not rows:
        df = pl.DataFrame(schema=schema)
    else:
        df = pl.DataFrame(rows, schema=list(schema.keys()), orient="row").cast(schema)
    return WalkResult(frames=df, dataset=dataset, used_fd=used_fd)


def walk_datasets(
    raw_root: str,
    datasets: Iterable[str],
    fd_path: Optional[str] = None,
) -> pl.DataFrame:
    """Walk multiple datasets and vertically concat the results."""
    frames = []
    for ds in datasets:
        res = walk_dataset(raw_root, ds, fd_path=fd_path)
        log.info("walker: %s → %d face rows (fd=%s)", ds, res.frames.height, res.used_fd)
        frames.append(res.frames)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")
