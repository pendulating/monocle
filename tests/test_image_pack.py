"""The image pack: build from files, read the same bytes back."""
import io
import os
import subprocess
import sys

import pytest
from PIL import Image

from dagspaces.common import image_pack

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_images(root, n):
    paths = []
    for i in range(n):
        d = os.path.join(root, "raw", f"d{i % 3}", f"e{i}", "faces")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "L.jpg")
        Image.new("RGB", (8 + i, 8), (i, 2 * i, 3 * i)).save(p, "JPEG", quality=90)
        paths.append(p)
    return paths


def build(tmp_path, paths, per_blob):
    lst = tmp_path / "paths.txt"
    lst.write_text("\n".join(paths) + "\n")
    out = tmp_path / "pack"
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "build_image_pack.py"),
         "--paths", str(lst), "--out-dir", str(out), "--per-blob", str(per_blob),
         "--workers", "2"],
        check=True, cwd=ROOT,
    )
    return out


def test_pack_gives_the_same_bytes(tmp_path):
    paths = make_images(tmp_path, 7)
    out = build(tmp_path, paths, per_blob=3)
    pack = image_pack.ImagePack(str(out))
    assert len(pack) == 7
    assert pack.manifest["n_blobs"] == 3
    for p in paths:
        assert pack.get(p) == open(p, "rb").read()
        assert pack.get("file://" + p) == open(p, "rb").read()
        assert Image.open(io.BytesIO(pack.get(p))).size == Image.open(p).size
    assert pack.get(str(tmp_path / "not_there.jpg")) is None
    assert pack.misses == 1


def test_prefetch_and_cache(tmp_path):
    paths = make_images(tmp_path, 5)
    out = build(tmp_path, paths, per_blob=2)
    pack = image_pack.ImagePack(str(out))
    n = pack.prefetch(["file://" + p for p in paths] + paths[:2], workers=3)
    assert n == 5
    assert pack.cache_bytes() == sum(os.path.getsize(p) for p in paths)
    assert pack.prefetch(paths) == 0
    for p in paths:
        assert pack.get(p) == open(p, "rb").read()
    pack.drop_cache()
    assert pack.cache_bytes() == 0
    assert pack.get(paths[0]) == open(paths[0], "rb").read()


def test_missing_source_goes_to_missing_txt(tmp_path):
    paths = make_images(tmp_path, 3)
    gone = str(tmp_path / "raw" / "gone.jpg")
    out = build(tmp_path, paths + [gone], per_blob=10)
    pack = image_pack.ImagePack(str(out))
    assert len(pack) == 3
    assert gone in (out / "missing.txt").read_text()
    assert pack.get(gone) is None


def test_incomplete_pack_is_refused(tmp_path):
    (tmp_path / "half").mkdir()
    with pytest.raises(FileNotFoundError):
        image_pack.ImagePack(str(tmp_path / "half"))


def test_module_registry(tmp_path):
    paths = make_images(tmp_path, 2)
    out = build(tmp_path, paths, per_blob=10)
    assert image_pack.read_image_bytes(paths[0]) is None
    image_pack.activate(str(out))
    assert image_pack.read_image_bytes(paths[0]) == open(paths[0], "rb").read()
    assert "hits=1" in image_pack.stats_line()
    image_pack.activate(None)
    assert image_pack.read_image_bytes(paths[0]) is None
