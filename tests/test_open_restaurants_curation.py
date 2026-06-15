"""Unit tests for dagspaces.common.curation.open_restaurants.*"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import box

from dagspaces.common.curation import geom as geom_mod
from dagspaces.common.curation.open_restaurants import (
    LICENSE_TYPES,
    UnknownLicenseTypeError,
    validate_license_types,
)
from dagspaces.common.curation.open_restaurants import open_restaurants as orr_mod
from dagspaces.common.curation.open_restaurants.fetch import (
    fetch_open_restaurants,
    _quote_in,
)
from dagspaces.common.curation.open_restaurants.normalize import (
    COMMON_COLUMNS,
    normalize_open_restaurants,
)
from dagspaces.common.curation.open_restaurants.validation import (
    OpenRestaurantsValidationError,
)


# ---------------------------------------------------------------- license_types

class TestLicenseTypes:

    def test_vocab(self):
        assert LICENSE_TYPES == ("Sidewalk", "Roadway")

    def test_validate_canonicalizes_case(self):
        assert validate_license_types(["sidewalk", "ROADWAY"]) == ["Sidewalk", "Roadway"]

    def test_validate_unknown_raises_with_suggestion(self):
        with pytest.raises(UnknownLicenseTypeError, match="sidewlk"):
            validate_license_types(["sidewlk"])


# ---------------------------------------------------------------- fetch

class TestFetchClause:

    def test_quote_in_escapes(self):
        assert _quote_in(["O'BRIEN"]) == "'O''BRIEN'"

    def test_where_always_requires_coords(self, monkeypatch):
        captured = {}

        def fake_fetch_socrata(url, *, where, select, **_kw):
            captured["where"] = where
            captured["select"] = select
            from dagspaces.common.curation.socrata import FetchResult
            return FetchResult(df=pl.DataFrame(), pages=0, total_rows=0)

        import dagspaces.common.curation.open_restaurants.fetch as fetch_mod
        monkeypatch.setattr(fetch_mod, "fetch_socrata", fake_fetch_socrata)
        fetch_open_restaurants(license_types=["Sidewalk"], boroughs=["MANHATTAN"])
        assert "latitude IS NOT NULL" in captured["where"]
        assert "upper(license_type) IN ('SIDEWALK')" in captured["where"]
        assert "upper(borough) IN ('MANHATTAN')" in captured["where"]


# ---------------------------------------------------------------- normalize

def _raw(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "business_legal_name": "ACME LLC", "assumed_name_s": "ACME CAFE",
        "street": "1 MAIN ST", "city": "NEW YORK", "borough": "Manhattan",
        "postcode": "10001", "license_type": "Sidewalk", "license_status": "Issued",
        "license_issue_date": "2025-01-01T00:00:00.000",
        "license_expiration_date": "2029-01-01T00:00:00.000",
        "latitude": "40.75", "longitude": "-73.98",
        "council_district": "3", "community_board": "105",
        "bin": "1000001", "bbl": "1000010001", "ct2020": "005", "nta2020": "MN05",
    }
    return pl.DataFrame([{**defaults, **r} for r in rows])


class TestNormalize:

    def test_happy_path(self):
        out = normalize_open_restaurants(_raw([{}]))
        assert out.height == 1
        assert out["facname"][0] == "ACME CAFE"            # prefers DBA name
        assert out["permit_id"][0] == out["uid"][0]        # shared geom API key
        assert out["borough"][0] == "MANHATTAN"            # canonical uppercase
        assert out["raw_latitude"][0] == pytest.approx(40.75)
        assert out["datasource"][0] == "dcwp:fpeh-f7ci"

    def test_facname_falls_back_to_legal_name(self):
        out = normalize_open_restaurants(_raw([{"assumed_name_s": ""}]))
        assert out["facname"][0] == "ACME LLC"

    def test_borough_from_bbl_when_text_missing(self):
        # Null borough text, BBL starts with 3 → Brooklyn.
        out = normalize_open_restaurants(_raw([{"borough": None, "bbl": "3000010001"}]))
        assert out["borough"][0] == "BROOKLYN"

    def test_uid_unique_for_colliding_base_key(self):
        # Two licenses, identical bbl + license_type + coords → suffixed UIDs.
        out = normalize_open_restaurants(_raw([
            {"business_legal_name": "A"}, {"business_legal_name": "B"},
        ]))
        assert out["uid"].n_unique() == 2

    def test_normalize_empty(self):
        out = normalize_open_restaurants(pl.DataFrame())
        assert out.height == 0
        assert set(out.columns) == set(COMMON_COLUMNS)


# ---------------------------------------------------------------- orchestrator

@pytest.fixture
def synthetic_buildings() -> gpd.GeoDataFrame:
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


def _make_fetch_result(df: pl.DataFrame):
    from dagspaces.common.curation.socrata import FetchResult
    return FetchResult(df=df, pages=1, total_rows=df.height, page_rows=[df.height])


class TestOrchestratorEndToEnd:

    def test_build_happy_path(self, monkeypatch, synthetic_buildings, tmp_path):
        raw_df = _raw([
            {"business_legal_name": "S1", "assumed_name_s": "SITE 1", "bin": "1000001",
             "bbl": "1000010001"},
            {"business_legal_name": "S2", "assumed_name_s": "SITE 2", "bin": "1000002",
             "bbl": "1000010002", "license_type": "Roadway"},
            {"business_legal_name": "S3", "assumed_name_s": "SITE 3", "bin": "9999999",  # BIN miss
             "bbl": "1000010003", "latitude": "40.75", "longitude": "-73.98"},
        ])
        monkeypatch.setattr(orr_mod, "fetch_open_restaurants",
                            lambda **_: _make_fetch_result(raw_df))
        monkeypatch.setattr(geom_mod, "load_buildings", lambda _p: synthetic_buildings)

        out = tmp_path / "open_restaurants"
        r = orr_mod.build(out=str(out), buildings_path="/nonexistent.parquet")

        assert r.total_publishable == 3
        assert Path(r.restaurants_parquet).is_file()
        assert Path(r.coverage_geojson).is_file()
        m = json.loads(Path(r.manifest_path).read_text())
        assert m["status"] == "OK"

        pq = pl.read_parquet(r.restaurants_parquet)
        assert {"uid", "facname", "license_type", "geom_source", "geom_wkb"} <= set(pq.columns)
        assert pq["uid"].n_unique() == pq.height
        gs_map = {row["geom_source"]: row["count"]
                  for row in pq["geom_source"].value_counts().to_dicts()}
        assert gs_map.get("bin_polygon", 0) == 2
        assert gs_map.get("nearest_polygon", 0) == 1

    def test_bad_geocode_rows_dropped(self, monkeypatch, synthetic_buildings, tmp_path):
        raw_df = _raw([
            {"business_legal_name": "GOOD", "bin": "1000001", "bbl": "1000010001"},
            # No BIN, lat/lon out in Massachusetts Bay → dropped before bbox check.
            {"business_legal_name": "BAD", "assumed_name_s": "BAD", "bin": None,
             "bbl": None, "latitude": "42.12", "longitude": "-70.84"},
        ])
        monkeypatch.setattr(orr_mod, "fetch_open_restaurants",
                            lambda **_: _make_fetch_result(raw_df))
        monkeypatch.setattr(geom_mod, "load_buildings", lambda _p: synthetic_buildings)

        out = tmp_path / "open_restaurants"
        r = orr_mod.build(out=str(out), buildings_path="/nonexistent.parquet")
        assert r.total_publishable == 1          # bad geocode dropped, build still OK
        assert r.funnel["after_in_nyc_or_bin"] == 1

    def test_unknown_license_type_raises_before_fetch(self, monkeypatch, tmp_path):
        called = []
        monkeypatch.setattr(orr_mod, "fetch_open_restaurants",
                            lambda **_: called.append(True))
        with pytest.raises(UnknownLicenseTypeError):
            orr_mod.build(out=str(tmp_path / "x"), license_types=["bogus"])
        assert not called

    def test_fatal_duplicate_uid_blocks(self, monkeypatch, synthetic_buildings, tmp_path):
        # Force a duplicate uid past normalize by stubbing it.
        raw_df = _raw([{"bin": "1000001"}])
        monkeypatch.setattr(orr_mod, "fetch_open_restaurants",
                            lambda **_: _make_fetch_result(raw_df))
        monkeypatch.setattr(geom_mod, "load_buildings", lambda _p: synthetic_buildings)

        def dup_normalize(_df):
            base = normalize_open_restaurants(raw_df)
            return pl.concat([base, base])  # two identical rows → duplicate uid

        monkeypatch.setattr(orr_mod, "normalize_open_restaurants", dup_normalize)
        out = tmp_path / "open_restaurants"
        with pytest.raises(OpenRestaurantsValidationError, match="duplicate"):
            orr_mod.build(out=str(out), buildings_path="/nonexistent.parquet")
        assert (out / "summary.md").is_file()
        m = json.loads((out / "manifest.json").read_text())
        assert m["status"] == "FATAL"
