"""Tests of the NYC layer of the Cyclomedia catalog (sv.cyclomedia_catalog).

The catalog code and its tests are in image-forward
(``dagspaces.sources.cyclomedia``, ``tests/test_sources_cyclomedia.py``).
Here: the NYC settings (borough, box, time zone, folders), the old import
paths and the CLI with the NYC defaults. All data is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from PIL import Image

import sv.cyclomedia_catalog as svc
from sv.cyclomedia_catalog import (
    ALL_FACES,
    DEFAULT_CATALOG_ROOT,
    CyclomediaCatalog,
    build_catalog,
    dataset_to_borough,
    rejoin_wfs,
)
from sv.cyclomedia_catalog import cli as nyc_cli
from sv.cyclomedia_catalog import nyc
from sv.cyclomedia_catalog.wfs import DEFAULT_CATALOG_GLOB, load_wfs_catalog

# --- a synthetic pull --------------------------------------------------------


def _make_recording(root: Path, dataset: str, group: str, rid: str, lat: float, lon: float) -> None:
    faces = root / dataset / group / rid / "faces"
    faces.mkdir(parents=True, exist_ok=True)
    for f in ALL_FACES:
        Image.new("RGB", (48, 48), color=(128, 128, 128)).save(faces / f"{f}.jpg", "JPEG")
    manifest = {"imageId": rid, "label": f"{lon:.6f},{lat:.6f}", "zoom": 3,
                "faces": {f: {"elapsed_s": 1.0, "used_render": True} for f in ALL_FACES}}
    (root / dataset / group / rid / "manifest.json").write_text(json.dumps(manifest))


def _write_wfs(path: Path, ids: list[str]) -> None:
    rows = ["imageId,lon,lat,recordedAt,recorderDirection,year,hasDepthMap,isAuthorized\n"]
    # 11:45 at -04:00 is 15:45 UTC.
    rows += [f"{rid},-73.98,40.76,2025-05-01T11:45:00.0000000-04:00,60.0,2025,True,True\n" for rid in ids]
    path.write_text("".join(rows))


@pytest.fixture()
def pull(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    _make_recording(raw, "manhattan_2025_1k", "W0AAA", "W0AAA001", 40.7580, -73.9855)
    _make_recording(raw, "si_2025_1k", "W0BBB", "W0BBB001", 40.5800, -74.1500)
    wfs = tmp_path / "pull" / "c"
    wfs.mkdir(parents=True)
    _write_wfs(wfs / "p.csv", ["W0AAA001", "W0BBB001"])
    return {"raw": str(raw), "catalog": str(tmp_path / "catalog"),
            "globs": [str(tmp_path / "pull" / "**" / "*.csv")]}


# --- the NYC settings ---------------------------------------------------------


def test_dataset_to_borough():
    assert dataset_to_borough("manhattan_2025_1k") == "manhattan"
    assert dataset_to_borough("brooklyn_2025_1k") == "brooklyn"
    assert dataset_to_borough("queens_2025_1k") == "queens"
    assert dataset_to_borough("bronx_2025_1k") == "bronx"
    assert dataset_to_borough("si_2025_1k") == "staten_island"
    assert dataset_to_borough("plazas_sample") == "unknown"
    assert dataset_to_borough("TEST") == "unknown"


def test_the_nyc_defaults():
    assert DEFAULT_CATALOG_ROOT == "/share/ju/cyclomedia/catalog/v1"
    assert DEFAULT_CATALOG_GLOB == nyc.WFS_GLOBS
    assert nyc.TZ == "America/New_York"
    assert svc.NYC_BBOX["lat_min"] < 40.7 < svc.NYC_BBOX["lat_max"]


def test_the_shims_use_the_framework_code():
    from dagspaces.sources.cyclomedia import catalog as fw_catalog
    from dagspaces.sources.cyclomedia import validation as fw_validation
    assert issubclass(CyclomediaCatalog, fw_catalog.CyclomediaCatalog)
    assert svc.ValidationError is fw_validation.ValidationError
    from sv.cyclomedia_catalog.schema import CATALOG_COLUMNS, FACE_BEARING_DEG  # noqa: F401
    from sv.cyclomedia_catalog.walker import FD_DEFAULT_PATH, walk_dataset  # noqa: F401


def test_the_ocr_configs_name_a_handler_that_loads():
    from dagspaces.ocr.data_handlers.base import OCRDataHandler

    root = Path(__file__).resolve().parents[1] / "conf" / "ocr" / "data"
    paths = {line.split(":", 1)[1].strip()
             for f in root.glob("cyclomedia*.yaml")
             for line in f.read_text().splitlines() if line.startswith("handler:")}
    assert paths == {"dagspaces.sources.cyclomedia.ocr:CyclomediaHandler"}
    assert type(OCRDataHandler.get_handler(paths.pop())).__name__ == "CyclomediaHandler"


# --- builds with the NYC settings ---------------------------------------------


def test_wfs_loader_uses_new_york_time(pull):
    df = load_wfs_catalog(pull["globs"])
    assert df["recordedAt"].dtype.time_zone == "America/New_York"
    assert df["recordedAt"].dt.hour().to_list() == [11, 11]


def test_build_gives_boroughs_and_new_york_time(pull):
    res = build_catalog(pull["raw"], pull["catalog"], catalog_globs=pull["globs"])
    assert res.total_rows == 12
    df = CyclomediaCatalog(pull["catalog"]).query()
    by_rec = dict(zip(df["recording_id"].to_list(), df["borough"].cast(pl.Utf8).to_list()))
    assert by_rec == {"W0AAA001": "manhattan", "W0BBB001": "staten_island"}
    # The framework default is UTC; the NYC wrapper must give New York time.
    assert df["recordedAt"].dtype.time_zone == "America/New_York"
    assert df["recordedAt"].dt.hour().unique().to_list() == [11]
    # A date with no time zone is a New York date.
    assert CyclomediaCatalog(pull["catalog"]).query(between=("2025-05-01T11:00", "2025-05-01T12:00")).height == 12


def test_the_framework_default_is_utc(pull):
    from dagspaces.sources.cyclomedia import build_catalog as fw_build

    fw_build(pull["raw"], pull["catalog"], catalog_globs=pull["globs"])
    df = CyclomediaCatalog(pull["catalog"]).query()
    assert df["recordedAt"].dtype.time_zone == "UTC"
    assert set(df["borough"].cast(pl.Utf8).to_list()) == {"unknown"}


def test_rejoin_keeps_the_nyc_settings(pull):
    build_catalog(pull["raw"], pull["catalog"], catalog_globs=pull["globs"])
    res = rejoin_wfs(pull["catalog"], ["manhattan_2025_1k"], catalog_globs=pull["globs"], raw_root=pull["raw"])
    assert res.total_rows == 6
    df = CyclomediaCatalog(pull["catalog"]).query(datasets=["manhattan_2025_1k"])
    assert df["recordedAt"].dtype.time_zone == "America/New_York"
    assert set(df["borough"].cast(pl.Utf8).to_list()) == {"manhattan"}


def test_the_nyc_box_check_warns_outside_nyc(tmp_path, caplog):
    import logging

    caplog.set_level(logging.WARNING)
    raw = tmp_path / "raw"
    _make_recording(raw, "paris_2025", "W0P", "W0P001", 48.85, 2.35)
    (tmp_path / "pull").mkdir()
    _write_wfs(tmp_path / "pull" / "p.csv", ["W0P001"])
    build_catalog(str(raw), str(tmp_path / "cat"), catalog_globs=[str(tmp_path / "pull" / "*.csv")])
    assert any("outside the bbox" in r.message for r in caplog.records)


def test_the_cli_uses_the_nyc_settings(pull, tmp_path):
    rc = nyc_cli.main(["build", "--raw-root", pull["raw"], "--output", pull["catalog"],
                       "--catalog-csv-glob", *pull["globs"]])
    assert rc == 0
    df = CyclomediaCatalog(pull["catalog"]).query()
    assert df["recordedAt"].dtype.time_zone == "America/New_York"
    assert "staten_island" in df["borough"].cast(pl.Utf8).to_list()
    out = tmp_path / "s"
    rc = nyc_cli.main(["sample", "-74.0,40.75,-73.97,40.77", "-n", "1", "--catalog", pull["catalog"],
                       "--out", str(out)])
    assert rc == 0
    import pandas as pd

    table = pd.read_parquet(out / "images.parquet")
    assert table["sample_id"].tolist() == ["W0AAA001_B", "W0AAA001_F", "W0AAA001_L", "W0AAA001_R"]


def test_the_cli_defaults_are_the_cluster_folders(monkeypatch):
    import dagspaces.sources.cyclomedia.cli as fw_cli

    seen: dict = {}
    monkeypatch.setattr(fw_cli, "main", lambda argv, **kw: seen.update(kw) or 0)
    assert nyc_cli.main(["validate"]) == 0
    assert seen["arg_defaults"]["output"] == "/share/ju/cyclomedia/catalog/v1"
    assert seen["arg_defaults"]["raw_root"] == "/share/ju/cyclomedia/raw"
    assert seen["tz"] == "America/New_York" and seen["region_of"] is dataset_to_borough
