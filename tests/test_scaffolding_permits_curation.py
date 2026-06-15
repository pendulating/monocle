"""Unit tests for the scaffold/shed permits curation pipeline.

Exercises:
  - normalize_dob_now + normalize_bis scaffold_type derivation
  - point fallback for unmatched BIN
  - orchestrator end-to-end on synthetic fixtures
  - validation fatal (duplicate permit_id) blocks output
  - validation warn surfaces in summary.md without blocking
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from dagspaces.common.curation.permits import buffer as buffer_mod
from dagspaces.common.curation.permits import fetch as fetch_mod
from dagspaces.common.curation.permits import scaffolding_permits as sp_mod
from dagspaces.common.curation.permits.normalize import (
    normalize_bis,
    normalize_dob_now,
    normalize_to_common,
)
from dagspaces.common.curation.permits.validation import (
    PermitValidationError,
    run_validation,
)


# ----------------------------------------------------------------------- fixtures

def _dob_now_raw(rows: list[dict]) -> pl.DataFrame:
    """Build a DOB NOW-shaped raw DataFrame from dicts with only the fields used."""
    defaults = {
        "job_filing_number": "", "filing_status": "", "filing_date": None,
        "first_permit_date": "2024-01-01T00:00:00.000",
        "current_status_date": None, "signoff_date": None,
        "latitude": "40.75", "longitude": "-73.98",
        "scaffold": "0", "shed": "0",
        "borough": "MANHATTAN",
        "house_no": "1", "street_name": "MAIN ST",
        "block": "1", "lot": "1",
        "bin": "1000001",
        "initial_cost": "0", "job_type": "", "job_description": "",
    }
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


def _bis_raw(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "borough": "MANHATTAN", "bin__": "1000001",
        "house__": "1", "street_name": "MAIN ST",
        "job__": "100000000", "job_type": "A2",
        "block": "1", "lot": "1",
        "work_type": "EQ", "permit_status": "ISSUED", "filing_status": "INITIAL",
        "permit_type": "EQ", "permit_subtype": "SH", "permit_sequence__": "01",
        "filing_date": "01/01/2019", "issuance_date": "01/05/2019",
        "expiration_date": "01/05/2020", "job_start_date": "01/05/2019",
        "permit_si_no": "9000000",
        "gis_latitude": "40.75", "gis_longitude": "-73.98",
        "owner_s_business_name": "",
    }
    filled = [{**defaults, **r} for r in rows]
    return pl.DataFrame(filled)


@pytest.fixture
def synthetic_buildings() -> gpd.GeoDataFrame:
    """Three building polygons at fixed BINs, WGS84. Small ~10-m squares."""
    def square(cx: float, cy: float, half_deg: float = 0.0001) -> Polygon:
        return box(cx - half_deg, cy - half_deg, cx + half_deg, cy + half_deg)

    gdf = gpd.GeoDataFrame(
        {"bin": ["1000001", "1000002", "1000003"]},
        geometry=[
            square(-73.9800, 40.7500),
            square(-73.9810, 40.7510),
            square(-73.9820, 40.7520),
        ],
        crs="EPSG:4326",
    )
    return gdf


# -------------------------------------------------------------------- normalize

class TestNormalize:

    def test_dob_now_scaffold_type_both(self):
        raw = _dob_now_raw([{"job_filing_number": "Q1-I1", "scaffold": "1", "shed": "1"}])
        out = normalize_dob_now(raw)
        assert out["scaffold_type"].to_list() == ["both"]
        assert out["source"].to_list() == ["dob_now"]
        assert out["permit_id"].to_list() == ["Q1-I1"]

    def test_dob_now_scaffold_type_scaffold_only(self):
        raw = _dob_now_raw([{"job_filing_number": "Q2-I1", "scaffold": "1", "shed": "0"}])
        assert normalize_dob_now(raw)["scaffold_type"].to_list() == ["scaffold"]

    def test_dob_now_scaffold_type_shed_only(self):
        raw = _dob_now_raw([{"job_filing_number": "Q3-I1", "scaffold": "0", "shed": "1"}])
        assert normalize_dob_now(raw)["scaffold_type"].to_list() == ["shed"]

    def test_dob_now_parses_iso_datetime(self):
        raw = _dob_now_raw([{"job_filing_number": "Q4-I1", "scaffold": "1",
                             "first_permit_date": "2024-06-15T12:30:45.000"}])
        out = normalize_dob_now(raw)
        issue = out["issue_date"][0]
        assert issue is not None
        assert issue.year == 2024 and issue.month == 6 and issue.day == 15

    def test_bis_permit_subtype_mapping(self):
        raw = _bis_raw([
            {"permit_si_no": "A", "permit_subtype": "SH"},
            {"permit_si_no": "B", "permit_subtype": "SD"},
            {"permit_si_no": "C", "permit_subtype": "SF"},
        ])
        out = normalize_bis(raw)
        assert out["scaffold_type"].to_list() == ["shed", "scaffold", "scaffold"]
        assert out["permit_id"].to_list() == ["A", "B", "C"]

    def test_bis_parses_mmddyyyy_date(self):
        raw = _bis_raw([{"permit_si_no": "A", "issuance_date": "06/15/2021"}])
        out = normalize_bis(raw)
        issue = out["issue_date"][0]
        assert issue.year == 2021 and issue.month == 6 and issue.day == 15

    def test_normalize_to_common_concats_sources(self):
        dn = _dob_now_raw([{"job_filing_number": "Q1-I1", "scaffold": "1"}])
        bs = _bis_raw([{"permit_si_no": "P1"}])
        out = normalize_to_common(dn, bs)
        assert out.height == 2
        assert set(out["source"].to_list()) == {"dob_now", "bis"}


# -------------------------------------------------------------------- buffer

class TestAttachGeometry:

    def test_bin_polygon_match(self, synthetic_buildings):
        dn = _dob_now_raw([{"job_filing_number": "Q1-I1", "scaffold": "1", "bin": "1000001"}])
        normalized = normalize_dob_now(dn)
        gdf = buffer_mod.attach_geometry(
            normalized, buffer_ft=80.0, buildings_gdf=synthetic_buildings
        )
        assert len(gdf) == 1
        assert gdf["geom_source"].iloc[0] == "bin_polygon"
        # Buffered polygon must contain the building's bbox
        assert gdf.geometry.iloc[0].contains(synthetic_buildings.geometry.iloc[0].centroid)

    def test_nearest_polygon_fallback_for_unmatched_bin(self, synthetic_buildings):
        """BIN miss but lat/lon near a building → pick the nearest building polygon."""
        dn = _dob_now_raw([{
            "job_filing_number": "Q2-I1", "scaffold": "1",
            "bin": "9999999",   # not in synthetic_buildings
            "latitude": "40.7500", "longitude": "-73.9800",  # right at building #1 centroid
        }])
        normalized = normalize_dob_now(dn)
        gdf = buffer_mod.attach_geometry(
            normalized, buffer_ft=80.0, buildings_gdf=synthetic_buildings,
            nearest_max_ft=200.0,
        )
        assert len(gdf) == 1
        assert gdf["geom_source"].iloc[0] == "nearest_polygon"
        # Distance should be very small (permit is essentially on the building)
        assert gdf["match_dist_ft"].iloc[0] < 50.0
        # Matched polygon should contain building #1's centroid
        assert gdf.geometry.iloc[0].contains(synthetic_buildings.geometry.iloc[0].centroid)

    def test_point_fallback_when_no_building_within_threshold(self, synthetic_buildings):
        """BIN miss AND lat/lon > threshold from any building → fall back to point."""
        dn = _dob_now_raw([{
            "job_filing_number": "Q3-I1", "scaffold": "1",
            "bin": "9999999",
            "latitude": "40.8500", "longitude": "-73.8800",  # far from synthetic_buildings
        }])
        normalized = normalize_dob_now(dn)
        gdf = buffer_mod.attach_geometry(
            normalized, buffer_ft=80.0, buildings_gdf=synthetic_buildings,
            nearest_max_ft=200.0,
        )
        assert gdf["geom_source"].iloc[0] == "point"
        geom = gdf.geometry.iloc[0]
        assert not geom.is_empty and geom.is_valid
        assert 0.0 < geom.area < 1e-4   # tiny buffered point in deg²

    def test_none_when_no_bin_and_no_latlon(self, synthetic_buildings):
        dn = _dob_now_raw([{
            "job_filing_number": "Q3-I1", "scaffold": "1",
            "bin": "9999999",
            "latitude": None, "longitude": None,
        }])
        normalized = normalize_dob_now(dn)
        gdf = buffer_mod.attach_geometry(
            normalized, buffer_ft=80.0, buildings_gdf=synthetic_buildings
        )
        assert gdf["geom_source"].iloc[0] == "none"


# -------------------------------------------------------------------- validation

def _publishable_fixture(synthetic_buildings, n_dob=2, n_bis=2) -> gpd.GeoDataFrame:
    """Build a small publishable (BIN-matched) GeoDataFrame."""
    dob = _dob_now_raw([
        {"job_filing_number": f"Q{i}-I1", "scaffold": "1", "bin": bn}
        for i, bn in enumerate(["1000001", "1000002"][:n_dob], start=1)
    ])
    bis = _bis_raw([
        {"permit_si_no": f"B{i}", "bin__": bn}
        for i, bn in enumerate(["1000002", "1000003"][:n_bis], start=1)
    ])
    normalized = normalize_to_common(dob, bis)
    gdf = buffer_mod.attach_geometry(
        normalized, buffer_ft=80.0, buildings_gdf=synthetic_buildings
    )
    return gdf[gdf["geom_source"].isin(["bin_polygon", "point"])].reset_index(drop=True)


class TestValidation:

    def _funnel(self, publishable, dob_n, bis_n):
        return {
            "dob_now_raw": dob_n,
            "bis_raw": bis_n,
            "after_normalize": dob_n + bis_n,
            "after_date_clip": dob_n + bis_n,
            "after_valid_scaffold_type": dob_n + bis_n,
            "after_attach_geometry": len(publishable),
            "publishable": len(publishable),
            "dob_now_dropped_null_first_permit_date": 0,
        }

    def _pagination(self):
        return {
            "dob_now": {"pages": 1, "page_rows": [2], "total_rows": 2, "truncated_likely": False},
            "bis": {"pages": 1, "page_rows": [2], "total_rows": 2, "truncated_likely": False},
        }

    def test_happy_path(self, synthetic_buildings, tmp_path):
        pub = _publishable_fixture(synthetic_buildings)
        result = run_validation(
            pub, str(tmp_path),
            funnel=self._funnel(pub, 2, 2),
            pagination=self._pagination(),
            cutoff="2025-12-31", buffer_ft=80.0,
        )
        assert result.fatal_violations == []
        assert os.path.isfile(result.summary_path)
        assert os.path.isfile(result.report_path)
        assert result.coverage is not None and not result.coverage.is_empty

    def test_fatal_duplicate_permit_id_blocks(self, synthetic_buildings, tmp_path):
        pub = _publishable_fixture(synthetic_buildings)
        # Duplicate one row to violate fatal #2
        pub = gpd.GeoDataFrame(
            pub.iloc[[0, 0, 1]].reset_index(drop=True),
            geometry="geometry", crs=pub.crs,
        )
        with pytest.raises(PermitValidationError, match="duplicate"):
            run_validation(
                pub, str(tmp_path),
                funnel=self._funnel(pub, 2, 2),
                pagination=self._pagination(),
                cutoff="2025-12-31", buffer_ft=80.0,
            )
        # summary.md + report still written for diagnosis
        assert (tmp_path / "summary.md").is_file()
        assert (tmp_path / "validation_report.parquet").is_file()

    def test_fatal_empty_source_blocks(self, synthetic_buildings, tmp_path):
        pub = _publishable_fixture(synthetic_buildings)
        funnel = self._funnel(pub, 2, 2)
        funnel["bis_raw"] = 0   # simulate BIS API returning nothing
        with pytest.raises(PermitValidationError, match="BIS returned 0 rows"):
            run_validation(
                pub, str(tmp_path),
                funnel=funnel,
                pagination=self._pagination(),
                cutoff="2025-12-31", buffer_ft=80.0,
            )

    def test_warn_low_bin_match_does_not_block(self, synthetic_buildings, tmp_path):
        """A permit with BIN not in buildings AND lat/lon far from any building
        drops its match rate below threshold but the build should still succeed."""
        # Non-BIN permits placed far from synthetic_buildings (~5 km away) so
        # the nearest-building fallback at 200 ft also misses, forcing `point`.
        far_lat, far_lon = "40.80", "-73.90"
        dob = _dob_now_raw([
            {"job_filing_number": "Q1-I1", "scaffold": "1", "bin": "1000001"},
            {"job_filing_number": "Q2-I1", "scaffold": "1", "bin": "999", "latitude": far_lat, "longitude": far_lon},
            {"job_filing_number": "Q3-I1", "scaffold": "1", "bin": "998", "latitude": far_lat, "longitude": far_lon},
        ])
        bis = _bis_raw([
            {"permit_si_no": "B1", "bin__": "997", "gis_latitude": far_lat, "gis_longitude": far_lon},
        ])
        normalized = normalize_to_common(dob, bis)
        gdf = buffer_mod.attach_geometry(
            normalized, buffer_ft=80.0, buildings_gdf=synthetic_buildings,
            nearest_max_ft=200.0,
        )
        pub = gdf[gdf["geom_source"].isin(["bin_polygon", "nearest_polygon", "point"])].reset_index(drop=True)
        assert len(pub) == 4
        result = run_validation(
            pub, str(tmp_path),
            funnel={
                "dob_now_raw": 3, "bis_raw": 1,
                "after_normalize": 4, "after_date_clip": 4,
                "after_valid_scaffold_type": 4, "after_attach_geometry": 4,
                "publishable": 4, "dob_now_dropped_null_first_permit_date": 0,
            },
            pagination=self._pagination(),
            cutoff="2025-12-31", buffer_ft=80.0,
            bin_match_warn_threshold=0.85,
        )
        assert result.fatal_violations == []
        # Warn must surface in summary.md
        summary = (tmp_path / "summary.md").read_text()
        assert "below" in summary and "threshold" in summary
        # Overall match rate is 1/4 = 25%
        assert result.metrics["bin_match_rate_overall_pct"] == pytest.approx(25.0)


# -------------------------------------------------------------------- orchestrator

class TestSince:

    def test_since_drops_early_permits(self):
        """normalize_to_common passes through all dates; the scaffolding_permits
        orchestrator's client-side clip applies the [since, cutoff] window.
        Verifying the clip logic via normalize + explicit filter expression."""
        dob = _dob_now_raw([
            {"job_filing_number": "OLD", "scaffold": "1",
             "first_permit_date": "2019-06-15T00:00:00.000"},
            {"job_filing_number": "KEEP", "scaffold": "1",
             "first_permit_date": "2021-03-10T00:00:00.000"},
        ])
        bis = _bis_raw([])
        normalized = normalize_to_common(dob, bis)
        assert normalized.height == 2

        # Apply the same since/cutoff clip the orchestrator applies
        import datetime as _dt
        since_dt = _dt.datetime(2020, 1, 1)
        cutoff_dt = _dt.datetime(2025, 12, 31, 23, 59, 59)
        clipped = normalized.filter(
            pl.col("issue_date").is_not_null()
            & (pl.col("issue_date") >= since_dt)
            & (pl.col("issue_date") <= cutoff_dt)
        )
        assert clipped.height == 1
        assert clipped["permit_id"].to_list() == ["KEEP"]


class TestOrchestratorEndToEnd:

    def test_build_happy_path(self, monkeypatch, synthetic_buildings, tmp_path):
        """End-to-end build with Socrata monkey-patched to return fixtures."""
        dob_df = _dob_now_raw([
            {"job_filing_number": "Q1-I1", "scaffold": "1", "bin": "1000001"},
            {"job_filing_number": "Q2-I1", "shed": "1", "bin": "1000002"},
        ])
        bis_df = _bis_raw([
            {"permit_si_no": "B1", "bin__": "1000003", "permit_subtype": "SH"},
        ])

        def fake_fetch_dob_now(*a, **kw):
            from dagspaces.common.curation.socrata import FetchResult
            return FetchResult(df=dob_df, pages=1, total_rows=dob_df.height,
                               page_rows=[dob_df.height])

        def fake_fetch_bis(*a, **kw):
            from dagspaces.common.curation.socrata import FetchResult
            return FetchResult(df=bis_df, pages=1, total_rows=bis_df.height,
                               page_rows=[bis_df.height])

        monkeypatch.setattr(sp_mod, "fetch_dob_now", fake_fetch_dob_now)
        monkeypatch.setattr(sp_mod, "fetch_bis", fake_fetch_bis)
        # Patch the buildings loader to use our synthetic frame.
        # load_buildings moved to the shared curation.geom module; patch there.
        from dagspaces.common.curation import geom as _geom_mod
        monkeypatch.setattr(_geom_mod, "load_buildings", lambda _p: synthetic_buildings)

        out = tmp_path / "permits_out"
        r = sp_mod.build(
            out=str(out), cutoff="2025-12-31", buffer_ft=80.0,
            buildings_path="/nonexistent.parquet",
        )
        assert r.total_publishable == 3
        assert os.path.isfile(r.permits_parquet)
        assert os.path.isfile(r.permits_geojson)
        assert os.path.isfile(r.coverage_geojson)
        assert os.path.isfile(r.manifest_path)
        # coverage.geojson round-trips
        with open(r.coverage_geojson) as f:
            fc = json.load(f)
        assert fc["type"] == "FeatureCollection"
        # permits.parquet has the expected columns
        pq = pl.read_parquet(r.permits_parquet)
        assert {"source", "permit_id", "scaffold_type", "geom_source", "geom_wkb"} <= set(pq.columns)
