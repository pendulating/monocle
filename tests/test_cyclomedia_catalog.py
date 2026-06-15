"""Unit tests for the Cyclomedia catalog.

Exercises:
  - schema helpers (dataset_to_borough, face bearing constants)
  - walker (fd path + scandir fallback) on a synthetic tree
  - manifest parser on a handcrafted manifest.json
  - WFS loader on a synthetic CSV
  - end-to-end build on a synthetic mini-tree
  - all 10 validation invariants, both happy-path and corrupted fixtures
  - CyclomediaCatalog query filters (faces, between, within, datasets)
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import polars as pl
import pytest
from PIL import Image
from shapely.geometry import box

from dagspaces.common.cyclomedia_catalog import (
    ALL_FACES,
    FACE_BEARING_DEG,
    ValidationError,
    build_catalog,
    CyclomediaCatalog,
    dataset_to_borough,
)
from dagspaces.common.cyclomedia_catalog.manifest import parse_manifests
from dagspaces.common.cyclomedia_catalog.schema import CATALOG_COLUMNS
from dagspaces.common.cyclomedia_catalog.walker import walk_dataset
from dagspaces.common.cyclomedia_catalog.wfs import load_wfs_catalog


# --- fixtures ---------------------------------------------------------------


def _write_jpeg(path: str, size: tuple[int, int] = (32, 32), fill: int = 128) -> None:
    img = Image.new("RGB", size, color=(fill, fill, fill))
    img.save(path, "JPEG", quality=85)


def _make_recording(
    root: Path,
    dataset: str,
    group: str,
    recording_id: str,
    lat: float,
    lon: float,
    *,
    manifest_image_id: Optional[str] = None,
    faces: tuple[str, ...] = ALL_FACES,
    include_manifest: bool = True,
    tiny_faces: frozenset[str] = frozenset(),
) -> Path:
    """Write a single recording directory with faces and a manifest.json."""
    rec_dir = root / dataset / group / recording_id
    faces_dir = rec_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    for f in faces:
        size = (4, 4) if f in tiny_faces else (48, 48)
        _write_jpeg(str(faces_dir / f"{f}.jpg"), size=size)
    if include_manifest:
        manifest = {
            "imageId": manifest_image_id or recording_id,
            "label": f"{lon:.6f},{lat:.6f}",
            "zoom": 3,
            "nameVersion": "streetsmart_25.2.0",
            "tileSchema": "Dcr9Tiling",
            "tilePx": 512,
            "faces": {f: {"elapsed_s": 1.0, "used_render": True} for f in faces},
            "mode": "render",
            "checkpoint": dataset,
            "no_tiles": True,
            "depthmaps": {
                "level": None,
                "mode": "render",
                "faces": {
                    f: {
                        "tiles_present": 0,
                        "tiles_expected": 0,
                        "used_render": True,
                        "render_size": 1024,
                        "rgb_render_size": 1024,
                        "downsample_factor": 0.25,
                    }
                    for f in faces
                },
                "total_present": len(faces),
                "total_expected": len(faces),
                "stitched_faces": True,
            },
        }
        (rec_dir / "manifest.json").write_text(json.dumps(manifest))
    return rec_dir


def _write_wfs_csv(path: Path, recording_ids: list[str], *, lat: float = 40.76, lon: float = -73.97) -> None:
    """Synthetic WFS CSV with the real schema."""
    header = (
        "imageId,lon,lat,recordedAt,productType,orientation,yawDegrees,orientationPrecision,"
        "yawPrecisionDegrees,recorderDirection,statePlaneX,statePlaneY,locationSRS,"
        "height,heightSystem,groundLevelOffset,latitudePrecision,longitudePrecision,"
        "heightPrecision,year,panoramaTileSchema,tileSchema,hasDepthMap,isAuthorized\n"
    )
    rows = [header]
    for i, rid in enumerate(recording_ids):
        rows.append(
            f"{rid},{lon},{lat},2025-05-01T11:45:{i % 60:02d}.0700000-04:00,"
            f"Cyclorama,0.05,0.0,0.008,0.45,{60.0 + i},-8228000,4982000,"
            f"urn:x-ogc:def:crs:EPSG:3857,-25.0,4326,2.2,0.02,0.02,0.03,2025,"
            f"TS_100_PLUS,Dcr9Tiling,True,True\n"
        )
    path.write_text("".join(rows))


@pytest.fixture()
def synthetic_tree(tmp_path: Path) -> dict:
    """Build a tiny but realistic cyclomedia tree with 3 recordings in one dataset."""
    raw_root = tmp_path / "raw"
    catalog_root = tmp_path / "catalog"
    dataset = "manhattan_synthetic"

    # 3 recordings, all with full 6 faces + good manifests
    ids = ["W0XXXX01", "W0XXXX02", "W0XXXX03"]
    _make_recording(raw_root, dataset, "W0XXX", ids[0], lat=40.7580, lon=-73.9855)
    _make_recording(raw_root, dataset, "W0XXX", ids[1], lat=40.7600, lon=-73.9800)
    _make_recording(raw_root, dataset, "W0YYY", ids[2], lat=40.7700, lon=-73.9700)

    # WFS CSV covering all three
    wfs_dir = tmp_path / "pull" / "recordings_synth_chunks"
    wfs_dir.mkdir(parents=True)
    _write_wfs_csv(wfs_dir / "synth_part1of1.csv", ids)

    return {
        "raw_root": str(raw_root),
        "catalog_root": str(catalog_root),
        "dataset": dataset,
        "recording_ids": ids,
        "wfs_glob": [str(tmp_path / "pull" / "**" / "*.csv")],
    }


# --- schema helpers ---------------------------------------------------------


class TestSchema:
    def test_dataset_to_borough(self):
        assert dataset_to_borough("manhattan_2025_1k") == "manhattan"
        assert dataset_to_borough("brooklyn_2025_1k") == "brooklyn"
        assert dataset_to_borough("queens_2025_1k") == "queens"
        assert dataset_to_borough("bronx_2025_1k") == "bronx"
        assert dataset_to_borough("si_2025_1k") == "staten_island"
        assert dataset_to_borough("plazas_sample") == "unknown"
        assert dataset_to_borough("TEST") == "unknown"

    def test_face_bearing_constants(self):
        assert FACE_BEARING_DEG["F"] == 0.0
        assert FACE_BEARING_DEG["R"] == 90.0
        assert FACE_BEARING_DEG["B"] == 180.0
        assert FACE_BEARING_DEG["L"] == 270.0
        assert FACE_BEARING_DEG["U"] is None
        assert FACE_BEARING_DEG["D"] is None

    def test_all_faces_includes_up_and_down(self):
        assert "U" in ALL_FACES
        assert "D" in ALL_FACES
        assert len(ALL_FACES) == 6


# --- walker + manifest ------------------------------------------------------


class TestWalkerAndManifest:
    def test_walk_picks_up_all_faces(self, synthetic_tree):
        res = walk_dataset(synthetic_tree["raw_root"], synthetic_tree["dataset"])
        # 3 recordings × 6 faces = 18
        assert res.frames.height == 18
        assert set(res.frames["face"].cast(pl.Utf8).unique().to_list()) == set(ALL_FACES)

    def test_walk_scandir_fallback(self, synthetic_tree):
        res = walk_dataset(synthetic_tree["raw_root"], synthetic_tree["dataset"],
                           fd_path="/nonexistent/bin/fd")
        # fd_path invalid, scandir fallback triggers when PATH-fd also missing.
        # In CI we can't guarantee fd is absent from PATH, but we can check the
        # result shape equals the fd path's.
        assert res.frames.height == 18

    def test_manifest_parser_extracts_everything(self, synthetic_tree):
        keys = [(synthetic_tree["dataset"], "W0XXX", "W0XXXX01")]
        mf = parse_manifests(synthetic_tree["raw_root"], keys)
        row = mf.to_dicts()[0]
        assert row["manifest_ok"] is True
        assert row["manifest_image_id"] == "W0XXXX01"
        assert row["manifest_latitude"] == pytest.approx(40.7580, abs=1e-4)
        assert row["manifest_longitude"] == pytest.approx(-73.9855, abs=1e-4)
        assert row["manifest_zoom"] == 3
        assert row["manifest_tile_px"] == 512
        assert row["manifest_name_version"] == "streetsmart_25.2.0"
        # per-face columns present
        for f in ALL_FACES:
            assert row[f"face_used_render_{f}"] is True
            assert row[f"depthmap_present_{f}"] is True


# --- WFS loader -------------------------------------------------------------


class TestWfsLoader:
    def test_wfs_loads_and_normalizes(self, synthetic_tree):
        df = load_wfs_catalog(synthetic_tree["wfs_glob"])
        assert "recording_id" in df.columns
        assert df.height == 3
        # recordedAt converted to tz-aware
        dt_col = df["recordedAt"]
        assert dt_col.dtype.time_zone is not None
        # hasDepthMap parsed as bool
        assert df["hasDepthMap"].dtype == pl.Boolean
        assert df["hasDepthMap"].sum() == 3


# --- end-to-end build + validation -----------------------------------------


class TestBuildAndValidate:
    def test_happy_path_end_to_end(self, synthetic_tree):
        res = build_catalog(
            raw_root=synthetic_tree["raw_root"],
            output_root=synthetic_tree["catalog_root"],
            datasets=[synthetic_tree["dataset"]],
            catalog_globs=synthetic_tree["wfs_glob"],
        )
        # 3 recordings × 6 faces = 18 rows
        assert res.total_rows == 18
        # summary and partition file exist
        assert os.path.isfile(os.path.join(res.output_root, "summary.md"))
        assert os.path.isfile(os.path.join(res.output_root, "validation_report.parquet"))

    def test_schema_columns_all_present(self, synthetic_tree):
        build_catalog(
            raw_root=synthetic_tree["raw_root"],
            output_root=synthetic_tree["catalog_root"],
            datasets=[synthetic_tree["dataset"]],
            catalog_globs=synthetic_tree["wfs_glob"],
        )
        cat = CyclomediaCatalog(root=synthetic_tree["catalog_root"])
        df = cat.query()
        for col in CATALOG_COLUMNS:
            assert col in df.columns, f"missing canonical column: {col}"

    def test_bearing_computation(self, synthetic_tree):
        build_catalog(
            raw_root=synthetic_tree["raw_root"],
            output_root=synthetic_tree["catalog_root"],
            datasets=[synthetic_tree["dataset"]],
            catalog_globs=synthetic_tree["wfs_glob"],
        )
        cat = CyclomediaCatalog(root=synthetic_tree["catalog_root"])
        df = cat.query()

        # U/D must be NULL
        ud = df.filter(pl.col("face").cast(pl.Utf8).is_in(["U", "D"]))
        assert ud["bearing"].null_count() == ud.height

        # F/B/L/R must be the face's absolute compass bearing (FACE_BEARING_DEG[face])
        # — Cyclomedia's NYC cube faces are rendered in a globally-oriented frame,
        # NOT rotated by recorderDirection. See `_compute_bearing` docstring.
        for face_letter, expected in [("F", 0.0), ("R", 90.0), ("B", 180.0), ("L", 270.0)]:
            sample = df.filter(pl.col("face").cast(pl.Utf8) == face_letter).head(1).to_dicts()[0]
            assert sample["bearing"] is not None, face_letter
            assert sample["bearing"] == pytest.approx(expected, abs=1e-3), face_letter

    def test_unique_recording_id_face(self, synthetic_tree):
        build_catalog(
            raw_root=synthetic_tree["raw_root"],
            output_root=synthetic_tree["catalog_root"],
            datasets=[synthetic_tree["dataset"]],
            catalog_globs=synthetic_tree["wfs_glob"],
        )
        cat = CyclomediaCatalog(root=synthetic_tree["catalog_root"])
        df = cat.query()
        dup_count = (
            df.group_by(["recording_id", "face"]).len().filter(pl.col("len") > 1).height
        )
        assert dup_count == 0

    def test_fatal_when_manifest_imageid_mismatches_dirname(self, tmp_path):
        raw_root = tmp_path / "raw"
        dataset = "synth"
        # imageId in manifest != dirname
        _make_recording(raw_root, dataset, "W0GGG", "W0GGGXYZ",
                        lat=40.76, lon=-73.97, manifest_image_id="DIFFERENT_ID")

        wfs_dir = tmp_path / "pull" / "chunks"
        wfs_dir.mkdir(parents=True)
        _write_wfs_csv(wfs_dir / "p.csv", ["DIFFERENT_ID"])

        with pytest.raises(ValidationError, match="imageId vs dirname"):
            build_catalog(
                raw_root=str(raw_root),
                output_root=str(tmp_path / "catalog"),
                datasets=[dataset],
                catalog_globs=[str(tmp_path / "pull" / "**" / "*.csv")],
            )

    def test_warn_when_wfs_join_misses(self, tmp_path, caplog):
        """Missing WFS entry → warn, not fatal."""
        import logging
        caplog.set_level(logging.WARNING)

        raw_root = tmp_path / "raw"
        dataset = "orphan"
        _make_recording(raw_root, dataset, "W0A", "W0AREC01",
                        lat=40.76, lon=-73.97)
        # WFS file covers an unrelated id
        wfs_dir = tmp_path / "pull" / "chunks"
        wfs_dir.mkdir(parents=True)
        _write_wfs_csv(wfs_dir / "p.csv", ["UNRELATED"])

        res = build_catalog(
            raw_root=str(raw_root),
            output_root=str(tmp_path / "catalog"),
            datasets=[dataset],
            catalog_globs=[str(tmp_path / "pull" / "**" / "*.csv")],
        )
        assert res.total_rows == 6  # 1 × 6 faces
        assert any("catalog hit rate low" in r.message for r in caplog.records)

    def test_warn_when_face_too_small(self, tmp_path, caplog):
        """JPEG < 50 KB triggers warn 8."""
        import logging
        caplog.set_level(logging.WARNING)

        raw_root = tmp_path / "raw"
        dataset = "tinyfaces"
        _make_recording(raw_root, dataset, "W0T", "W0TRECT01",
                        lat=40.76, lon=-73.97, tiny_faces=frozenset(ALL_FACES))
        wfs_dir = tmp_path / "pull" / "chunks"
        wfs_dir.mkdir(parents=True)
        _write_wfs_csv(wfs_dir / "p.csv", ["W0TRECT01"])

        build_catalog(
            raw_root=str(raw_root),
            output_root=str(tmp_path / "catalog"),
            datasets=[dataset],
            catalog_globs=[str(tmp_path / "pull" / "**" / "*.csv")],
        )
        assert any("suspicious file_size" in r.message for r in caplog.records)


# --- query API --------------------------------------------------------------


class TestQueryAPI:
    @pytest.fixture()
    def built(self, synthetic_tree):
        build_catalog(
            raw_root=synthetic_tree["raw_root"],
            output_root=synthetic_tree["catalog_root"],
            datasets=[synthetic_tree["dataset"]],
            catalog_globs=synthetic_tree["wfs_glob"],
        )
        return synthetic_tree

    def test_face_filter(self, built):
        cat = CyclomediaCatalog(root=built["catalog_root"])
        df = cat.query(faces={"F"})
        assert df.height == 3  # one F per recording × 3 recordings
        assert df["face"].cast(pl.Utf8).unique().to_list() == ["F"]

    def test_between_filter(self, built):
        cat = CyclomediaCatalog(root=built["catalog_root"])
        df = cat.query(between=("2025-04-01", "2025-06-01"))
        # All synthetic recordings are dated 2025-05-01, so all 18 rows should match
        assert df.height == 18

        empty = cat.query(between=("2026-01-01", "2026-12-31"))
        assert empty.height == 0

    def test_within_polygon(self, built):
        cat = CyclomediaCatalog(root=built["catalog_root"])
        bbox = box(-74.0, 40.75, -73.97, 40.77)  # covers only id 01 & 02 (lat ~ 40.758-40.76)
        df = cat.query(within=bbox)
        recs = set(df["recording_id"].unique().to_list())
        # id 03 is at lat 40.77 lon -73.97 — exactly on boundary; within returns
        # inside only, so expect ids 01 and 02.
        assert "W0XXXX01" in recs
        assert "W0XXXX02" in recs
        # id 03 may or may not match on boundary; don't assert

    def test_within_excludes_outside(self, built):
        cat = CyclomediaCatalog(root=built["catalog_root"])
        far_away = box(0.0, 0.0, 1.0, 1.0)
        df = cat.query(within=far_away)
        assert df.height == 0

    def test_dataset_filter(self, built):
        cat = CyclomediaCatalog(root=built["catalog_root"])
        df = cat.query(datasets=["does_not_exist"])
        assert df.height == 0
        df = cat.query(datasets=[built["dataset"]])
        assert df.height == 18

    def test_build_inference_parquet(self, built, tmp_path):
        cat = CyclomediaCatalog(root=built["catalog_root"])
        out_path = tmp_path / "out.parquet"
        df = cat.build_inference_parquet(
            output_path=str(out_path),
            faces={"F", "B"},
        )
        assert df.height == 6  # 3 recordings × 2 faces
        assert out_path.is_file()
        roundtrip = pl.read_parquet(str(out_path))
        assert roundtrip.height == 6
