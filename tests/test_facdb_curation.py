"""Unit tests for dagspaces.common.curation.facdb.*"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import box

from dagspaces.common.curation.facdb import (
    HIERARCHY_LEVELS,
    UnknownCategoryError,
    load_categorization,
    validate_filter_values,
)
from dagspaces.common.curation.facdb import facdb_facilities as fb_mod
from dagspaces.common.curation import geom as geom_mod
from dagspaces.common.curation.facdb.fetch import _in_clause
from dagspaces.common.curation.facdb.normalize import (
    COMMON_COLUMNS,
    normalize_facdb,
)
from dagspaces.common.curation.facdb.validation import (
    FacdbValidationError,
    run_validation,
)


# ---------------------------------------------------------------- categorization

class TestCategorization:

    def test_hierarchy_levels(self):
        assert HIERARCHY_LEVELS == ("facdomain", "facgroup", "facsubgrp", "factype")

    def test_load_ok(self):
        c = load_categorization()
        assert c["version"].startswith("25")
        assert "HEALTH AND HUMAN SERVICES" in c["domains"]
        assert "SCHOOLS (K-12)" in c["groups"]
        assert len(c["domains"]) == 7

    def test_validate_canonicalizes_case_and_whitespace(self):
        out = validate_filter_values("facdomain", ["  health and human services  "])
        assert out == ["HEALTH AND HUMAN SERVICES"]

    def test_validate_accepts_multiple(self):
        out = validate_filter_values(
            "facdomain",
            ["parks, gardens, and historical sites", "libraries and cultural programs"],
        )
        assert set(out) == {
            "PARKS, GARDENS, AND HISTORICAL SITES",
            "LIBRARIES AND CULTURAL PROGRAMS",
        }

    def test_validate_unknown_raises_with_suggestion(self):
        with pytest.raises(UnknownCategoryError, match="healht"):
            validate_filter_values("facdomain", ["healht"])

    def test_validate_bad_level_raises(self):
        with pytest.raises(ValueError, match="unknown hierarchy level"):
            validate_filter_values("bogus", ["x"])


# ---------------------------------------------------------------- fetch

class TestFetchClause:

    def test_escapes_quotes(self):
        # dictionary values don't contain quotes today, but be defensive
        c = _in_clause("facdomain", ["O'BRIEN"])
        assert "O''BRIEN" in c and "upper(facdomain)" in c

    def test_multi_value_in(self):
        c = _in_clause("facgroup", ["LIBRARIES", "SCHOOLS (K-12)"])
        assert c == "upper(facgroup) IN ('LIBRARIES','SCHOOLS (K-12)')"


# ---------------------------------------------------------------- normalize

def _raw(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "uid": "", "facname": "", "address": "", "city": "NEW YORK",
        "zipcode": "", "boro": "MANHATTAN", "borocode": "1",
        "bin": "1000001", "bbl": "1000010001",
        "latitude": "40.75", "longitude": "-73.98",
        "xcoord": "0", "ycoord": "0",
        "facdomain": "HEALTH AND HUMAN SERVICES",
        "facgroup": "HEALTH CARE", "facsubgrp": "HOSPITAL", "factype": "GENERAL HOSPITAL",
        "capacity": "0", "captype": "", "opname": "", "opabbrev": "", "optype": "",
        "overagency": "", "overabbrev": "", "overlevel": "", "servarea": "LOCAL",
        "cd": "105", "council": "3", "nta2020": "MN05", "ct2020": "005",
        "schooldist": "", "policeprct": "", "datasource": "dcp_colp",
    }
    return pl.DataFrame([{**defaults, **r} for r in rows])


class TestNormalize:

    def test_happy_path(self):
        raw = _raw([{"uid": "U1", "facname": "BELLEVUE HOSPITAL"}])
        out = normalize_facdb(raw)
        assert out.height == 1
        assert out["uid"][0] == "U1"
        assert out["permit_id"][0] == "U1"   # shared geom API key
        assert out["facname"][0] == "BELLEVUE HOSPITAL"
        assert out["borough"][0] == "MANHATTAN"
        assert out["raw_latitude"][0] == pytest.approx(40.75)
        # hierarchy columns already uppercased
        assert out["facdomain"][0] == "HEALTH AND HUMAN SERVICES"

    def test_normalize_empty(self):
        out = normalize_facdb(pl.DataFrame())
        assert out.height == 0
        assert set(out.columns) == set(COMMON_COLUMNS)


# ---------------------------------------------------------------- buffer + validation + orchestrator

@pytest.fixture
def synthetic_buildings() -> gpd.GeoDataFrame:
    """Three 10m squares at known BINs."""
    def square(cx, cy, h=0.0001):
        return box(cx - h, cy - h, cx + h, cy + h)
    return gpd.GeoDataFrame(
        {"bin": ["1000001", "1000002", "1000003"]},
        geometry=[
            square(-73.98, 40.75),
            square(-73.981, 40.751),
            square(-73.982, 40.752),
        ],
        crs="EPSG:4326",
    )


class TestOrchestratorEndToEnd:

    def test_build_happy_path(self, monkeypatch, synthetic_buildings, tmp_path):
        raw_df = _raw([
            {"uid": "U1", "facname": "SITE 1", "bin": "1000001",
             "facdomain": "HEALTH AND HUMAN SERVICES", "facgroup": "HEALTH CARE"},
            {"uid": "U2", "facname": "SITE 2", "bin": "1000002",
             "facdomain": "HEALTH AND HUMAN SERVICES", "facgroup": "HEALTH CARE"},
            {"uid": "U3", "facname": "SITE 3", "bin": "9999999",  # BIN miss → nearest fallback
             "latitude": "40.75", "longitude": "-73.98",
             "facdomain": "HEALTH AND HUMAN SERVICES", "facgroup": "HEALTH CARE"},
        ])

        def fake_fetch_facdb(*, cache_path=None, refresh=False, **_kw):
            from dagspaces.common.curation.socrata import FetchResult
            return FetchResult(df=raw_df, pages=1, total_rows=raw_df.height,
                               page_rows=[raw_df.height])

        monkeypatch.setattr(fb_mod, "fetch_facdb", fake_fetch_facdb)
        monkeypatch.setattr(geom_mod, "load_buildings", lambda _p: synthetic_buildings)

        out = tmp_path / "facdb_health"
        r = fb_mod.build(
            out=str(out),
            facdomain=["HEALTH AND HUMAN SERVICES"],
            buildings_path="/nonexistent.parquet",
        )
        assert r.total_publishable == 3
        assert Path(r.facilities_parquet).is_file()
        assert Path(r.coverage_geojson).is_file()
        assert Path(r.manifest_path).is_file()

        m = json.loads(Path(r.manifest_path).read_text())
        assert m["status"] == "OK"
        assert m["filters"]["facdomain"] == ["HEALTH AND HUMAN SERVICES"]

        pq = pl.read_parquet(r.facilities_parquet)
        assert {"uid", "facname", "facdomain", "facgroup", "geom_source", "geom_wkb"} <= set(pq.columns)
        # 2 BIN-exact + 1 nearest fallback
        gs = pq["geom_source"].value_counts().to_dicts()
        gs_map = {r["geom_source"]: r["count"] for r in gs}
        assert gs_map.get("bin_polygon", 0) == 2
        assert gs_map.get("nearest_polygon", 0) == 1

    def test_unknown_filter_raises_before_fetch(self, monkeypatch, tmp_path):
        called = []
        monkeypatch.setattr(fb_mod, "fetch_facdb", lambda **_: called.append(True))
        out = tmp_path / "facdb"
        with pytest.raises(UnknownCategoryError):
            fb_mod.build(out=str(out), facdomain=["invalid domain name"])
        assert not called

    def test_fatal_duplicate_uid_blocks(self, monkeypatch, synthetic_buildings, tmp_path):
        raw_df = _raw([
            {"uid": "DUP", "facname": "A", "bin": "1000001"},
            {"uid": "DUP", "facname": "B", "bin": "1000002"},
        ])
        monkeypatch.setattr(
            fb_mod, "fetch_facdb",
            lambda **_: _make_fetch_result(raw_df),
        )
        monkeypatch.setattr(geom_mod, "load_buildings", lambda _p: synthetic_buildings)
        out = tmp_path / "facdb"
        with pytest.raises(FacdbValidationError, match="duplicate"):
            fb_mod.build(out=str(out),
                         facdomain=["HEALTH AND HUMAN SERVICES"],
                         buildings_path="/nonexistent.parquet")
        # summary written even on fatal
        assert (out / "summary.md").is_file()
        assert (out / "validation_report.parquet").is_file()
        # manifest records FATAL status
        m = json.loads((out / "manifest.json").read_text())
        assert m["status"] == "FATAL"


def _make_fetch_result(df: pl.DataFrame):
    from dagspaces.common.curation.socrata import FetchResult
    return FetchResult(df=df, pages=1, total_rows=df.height, page_rows=[df.height])
