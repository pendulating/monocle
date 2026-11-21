"""Read images from a pack of blob files instead of the raw NFS tree.

Why
---
A pairvqa row opens 2 JPEG files under `/share/ju/cyclomedia/raw`. The tree is
deep, thus 1 cold open costs about 8 NFS round trips (LOOKUP, ACCESS, OPEN,
READ, CLOSE, GETATTR). A battery of 1,000,000 pairs opens 15,400,000 files
over 1,100,000 unique images, and above about 30 shards the `ju-nfs-02` server
collapses. See `vlm-narratives-docs/nfs-load-pairvqa-sweeps.md`.

A pack holds the same JPEG bytes in a few large blob files. A read of 1 image
is then 1 `pread` of about 240 kB, which is 1 NFS READ, with no directory
work. The bytes are the same, thus the model input and the results do not
change.

Layout of a pack directory
--------------------------
    manifest.json          written last; a pack without it is not complete
    index.parquet          columns: path, blob, offset, length (sorted by path)
    blob-0000.bin ...      the image bytes, one after the other

Use
---
    pack = activate("/share/ju/.../image_pack")   # once for each process
    data = read_image_bytes("/share/ju/.../faces/L.jpg")  # bytes, or None

`read_image_bytes` gives None when no pack is active or the path is not in
the index. The caller then reads the file as before. `prefetch` reads a list
of paths with a thread pool and keeps the bytes until `drop_cache`.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Tuple

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.parquet"
PACK_VERSION = 1


def blob_name(k: int) -> str:
    return f"blob-{k:04d}.bin"


def strip_file_url(path: str) -> str:
    """Give the plain path of a `file://` URL, or the input when it is a path."""
    return path[len("file://"):] if path.startswith("file://") else path


class ImagePack:
    """One open pack. The methods are safe to call from more than 1 thread."""

    def __init__(self, pack_dir: str):
        self.pack_dir = str(pack_dir)
        manifest_path = os.path.join(self.pack_dir, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"image pack {self.pack_dir} holds no {MANIFEST_NAME}; the build "
                f"is not complete, or the path is wrong"
            )
        with open(manifest_path) as fh:
            self.manifest = json.load(fh)
        if int(self.manifest.get("version", 0)) != PACK_VERSION:
            raise ValueError(
                f"image pack {self.pack_dir} has version "
                f"{self.manifest.get('version')}, and this reader needs {PACK_VERSION}"
            )
        self._index: Dict[str, Tuple[int, int, int]] = self._load_index()
        self._fds: Dict[int, int] = {}
        self._fd_lock = threading.Lock()
        self._cache: Dict[str, bytes] = {}
        self.hits = 0
        self.misses = 0

    def _load_index(self) -> Dict[str, Tuple[int, int, int]]:
        import pyarrow.parquet as pq

        table = pq.read_table(os.path.join(self.pack_dir, INDEX_NAME))
        paths = table.column("path").to_pylist()
        blobs = table.column("blob").to_pylist()
        offsets = table.column("offset").to_pylist()
        lengths = table.column("length").to_pylist()
        return {p: (b, o, n) for p, b, o, n in zip(paths, blobs, offsets, lengths)}

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, path: str) -> bool:
        return strip_file_url(str(path)) in self._index

    def _fd(self, blob: int) -> int:
        fd = self._fds.get(blob)
        if fd is not None:
            return fd
        with self._fd_lock:
            fd = self._fds.get(blob)
            if fd is None:
                fd = os.open(os.path.join(self.pack_dir, blob_name(blob)), os.O_RDONLY)
                self._fds[blob] = fd
        return fd

    def _read(self, entry: Tuple[int, int, int]) -> bytes:
        blob, offset, length = entry
        fd = self._fd(blob)
        out = bytearray()
        while len(out) < length:
            chunk = os.pread(fd, length - len(out), offset + len(out))
            if not chunk:
                raise IOError(
                    f"image pack {self.pack_dir}: {blob_name(blob)} ends at "
                    f"{offset + len(out)}, and the index needs {offset + length}"
                )
            out.extend(chunk)
        return bytes(out)

    def get(self, path: str) -> Optional[bytes]:
        """Give the bytes of 1 image, or None when the path is not in the pack."""
        key = strip_file_url(str(path))
        data = self._cache.get(key)
        if data is not None:
            self.hits += 1
            return data
        entry = self._index.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        return self._read(entry)

    def prefetch(self, paths: Iterable[str], workers: int = 8) -> int:
        """Read these paths with a thread pool and keep the bytes in the cache.

        The reads go in offset order inside each blob, which keeps the server
        reads close to sequential. Gives the number of images read.
        """
        wanted: List[Tuple[Tuple[int, int, int], str]] = []
        seen = set()
        for raw in paths:
            key = strip_file_url(str(raw))
            if key in seen or key in self._cache:
                continue
            seen.add(key)
            entry = self._index.get(key)
            if entry is not None:
                wanted.append((entry, key))
        if not wanted:
            return 0
        wanted.sort()
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            blobs = list(pool.map(lambda w: self._read(w[0]), wanted))
        for (_, key), data in zip(wanted, blobs):
            self._cache[key] = data
        return len(wanted)

    def drop_cache(self) -> None:
        self._cache = {}

    def cache_bytes(self) -> int:
        return sum(len(v) for v in self._cache.values())

    def close(self) -> None:
        with self._fd_lock:
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds = {}
        self.drop_cache()


_ACTIVE: Optional[ImagePack] = None


def activate(pack_dir: Optional[str]) -> Optional[ImagePack]:
    """Open a pack for this process. None or "" switches the pack off."""
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.close()
    _ACTIVE = ImagePack(pack_dir) if pack_dir else None
    return _ACTIVE


def active() -> Optional[ImagePack]:
    return _ACTIVE


def read_image_bytes(path: str) -> Optional[bytes]:
    """Bytes of the image from the active pack, or None when there is no hit."""
    pack = _ACTIVE
    if pack is None:
        return None
    return pack.get(path)


def stats_line() -> str:
    pack = _ACTIVE
    if pack is None:
        return "image_pack=off"
    return f"image_pack hits={pack.hits} misses={pack.misses} dir={pack.pack_dir}"
