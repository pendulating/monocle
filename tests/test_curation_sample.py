"""Unit tests for dagspaces.common.curation.sample.sample_images."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest

from dagspaces.common.curation.sample import sample_images


def _make_image_tree(tmp_path: Path, n_images: int = 20) -> tuple[Path, pl.DataFrame]:
    """Build a synthetic tree of tiny JPEG files + a manifest parquet.

    Returns (parquet_path, df).
    """
    src_root = tmp_path / "src"
    src_root.mkdir()
    rows = []
    datasets = ["brooklyn_2025_1k", "queens_2025_1k", "bronx_2025_1k"]
    faces = ["F", "B", "L", "R"]
    from PIL import Image
    for i in range(n_images):
        ds = datasets[i % len(datasets)]
        face = faces[i % len(faces)]
        rec = f"REC{i:04d}"
        sample_id = f"{rec}_{face}"
        d = src_root / ds / rec / "faces"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{face}.jpg"
        Image.new("RGB", (8, 8), color=(i, i, i)).save(p, "JPEG")
        rows.append({
            "sample_id": sample_id,
            "recording_id": rec,
            "face": face,
            "dataset": ds,
            "image_path": str(p),
            "latitude": 40.7 + i * 1e-4,
            "longitude": -74.0 + i * 1e-4,
        })
    df = pl.DataFrame(rows)
    parquet = tmp_path / "curated.parquet"
    df.write_parquet(parquet)
    return parquet, df


class TestSampleImages:

    def test_copy_mode_happy_path(self, tmp_path):
        parquet, _ = _make_image_tree(tmp_path, n_images=20)
        out = tmp_path / "insp"
        r = sample_images(str(parquet), str(out), k=5, mode="copy", seed=0, workers=2)
        assert r.n_sampled == 5
        assert r.n_exported == 5
        assert r.n_missing == 0
        assert r.n_failed == 0
        files = sorted(os.listdir(out / "images"))
        assert len(files) == 5
        # All filenames follow <dataset>__<sample_id>.jpg pattern
        for f in files:
            assert "__" in f
            assert f.endswith(".jpg")
        # Manifest has a row per sampled image, with export_status=ok
        mf = pl.read_parquet(out / "manifest.parquet")
        assert mf.height == 5
        assert set(mf["export_status"].to_list()) == {"ok"}
        # JSON summary
        summary = json.loads((out / "manifest.json").read_text())
        assert summary["mode"] == "copy"
        assert summary["k_requested"] == 5 and summary["n_exported_ok"] == 5

    def test_symlink_mode(self, tmp_path):
        parquet, _ = _make_image_tree(tmp_path, n_images=12)
        out = tmp_path / "insp"
        r = sample_images(str(parquet), str(out), k=4, mode="symlink", seed=0)
        assert r.mode == "symlink" and r.n_exported == 4
        for entry in (out / "images").iterdir():
            assert entry.is_symlink(), f"{entry} is not a symlink"
            # Absolute target (so the inspection dir can be moved)
            assert os.path.isabs(os.readlink(entry))

    def test_stratified_sample_even_split(self, tmp_path):
        parquet, _ = _make_image_tree(tmp_path, n_images=30)
        out = tmp_path / "insp"
        r = sample_images(str(parquet), str(out), k=9, stratify_by="dataset", seed=0)
        mf = pl.read_parquet(out / "manifest.parquet")
        # 3 datasets, k=9 → 3 per dataset
        counts = mf.group_by("dataset").len().sort("dataset")
        assert counts["len"].to_list() == [3, 3, 3]

    def test_k_greater_than_population_caps(self, tmp_path, caplog):
        parquet, _ = _make_image_tree(tmp_path, n_images=5)
        out = tmp_path / "insp"
        r = sample_images(str(parquet), str(out), k=100, seed=0)
        assert r.n_sampled == 5 and r.n_exported == 5

    def test_missing_source_file_counted(self, tmp_path):
        parquet, df = _make_image_tree(tmp_path, n_images=10)
        # Corrupt one image_path
        df2 = df.with_columns(
            pl.when(pl.col("sample_id") == df["sample_id"][0])
              .then(pl.lit("/definitely/not/a/real/file.jpg"))
              .otherwise(pl.col("image_path"))
              .alias("image_path")
        )
        df2.write_parquet(parquet)
        out = tmp_path / "insp"
        r = sample_images(str(parquet), str(out), k=10, seed=0)
        assert r.n_sampled == 10
        assert r.n_exported + r.n_missing == 10
        assert r.n_missing >= 1
        mf = pl.read_parquet(out / "manifest.parquet")
        assert "missing" in mf["export_status"].to_list()

    def test_refuse_non_empty_dir_without_force(self, tmp_path):
        parquet, _ = _make_image_tree(tmp_path, n_images=4)
        out = tmp_path / "insp"
        out.mkdir()
        (out / "something.txt").write_text("x")
        with pytest.raises(FileExistsError):
            sample_images(str(parquet), str(out), k=2, seed=0)
        # With force, it proceeds.
        r = sample_images(str(parquet), str(out), k=2, seed=0, force=True)
        assert r.n_exported == 2

    def test_reproducible_with_seed(self, tmp_path):
        parquet, _ = _make_image_tree(tmp_path, n_images=50)
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        r1 = sample_images(str(parquet), str(out_a), k=7, seed=42)
        r2 = sample_images(str(parquet), str(out_b), k=7, seed=42)
        a = sorted(os.listdir(out_a / "images"))
        b = sorted(os.listdir(out_b / "images"))
        assert a == b
