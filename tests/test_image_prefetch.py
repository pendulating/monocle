"""The one-batch-ahead image prefetcher of the single-process path."""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

from dagspaces.common import image_pack
from dagspaces.common import vllm_inference as vi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_images(root, n):
    paths = []
    for i in range(n):
        d = os.path.join(root, "raw", f"e{i}", "faces")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "L.jpg")
        arr = (np.random.RandomState(i).rand(64, 64, 3) * 255).astype("uint8")
        Image.fromarray(arr).save(p, "JPEG", quality=90)
        paths.append(p)
    return paths


def build(tmp_path, paths):
    lst = tmp_path / "paths.txt"
    lst.write_text("\n".join(paths) + "\n")
    out = tmp_path / "pack"
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "build_image_pack.py"),
         "--paths", str(lst), "--out-dir", str(out), "--per-blob", "4", "--workers", "2"],
        check=True, cwd=ROOT,
    )
    return str(out)


def messages(pa, pb):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "file://" + pa}},
        {"type": "text", "text": "Image A."},
        {"type": "image", "image": pb},
        {"type": "text", "text": "Which?"},
    ]}]


def check_prefetcher(pack, paths, workers):
    pf = vi._BatchPrefetcher(pack, workers, "test")
    key0, key1 = (0, 2), (2, 4)
    pf.submit(key0, ["file://" + paths[0], paths[1], paths[0]])
    assert pf.take(key1) is None          # not the batch in flight
    pf.submit(key0, ["file://" + paths[0], paths[1]])
    got = pf.take(key0)
    assert set(got) == {paths[0], paths[1]}
    for p in (paths[0], paths[1]):
        assert np.array_equal(np.asarray(got[p]), np.asarray(Image.open(p).convert("RGB")))
    if pack is not None:
        assert pack.cache_bytes() == 0    # the prefetcher drops the byte cache
    # the hydration helpers use the decoded images and read the rest
    hyd = vi._hydrate_messages_for_chat(messages(paths[0], paths[2]), got)
    blocks = [b for b in hyd[0]["content"] if b["type"] == "image_pil"]
    assert blocks[0]["image_pil"] is got[paths[0]]
    assert np.array_equal(np.asarray(blocks[1]["image_pil"]),
                          np.asarray(Image.open(paths[2]).convert("RGB")))
    imgs = vi._extract_images_from_messages(messages(paths[1], paths[3]), got)
    assert imgs[0] is got[paths[1]] and len(imgs) == 2
    assert pf.take(key0) is None          # taken once
    assert pf.batches == 1
    pf.close()


def test_prefetcher_with_pack(tmp_path):
    paths = make_images(tmp_path, 4)
    pack = image_pack.ImagePack(build(tmp_path, paths))
    check_prefetcher(pack, paths, workers=4)


def test_prefetcher_without_pack(tmp_path):
    paths = make_images(tmp_path, 4)
    check_prefetcher(None, paths, workers=1)


def test_failed_prefetch_falls_back(tmp_path):
    pf = vi._BatchPrefetcher(None, 1, "test")
    pf.submit((0, 1), [str(tmp_path / "gone.jpg")])
    assert pf.take((0, 1)) == {}          # a missing file is dropped, not raised
    pf.close()
