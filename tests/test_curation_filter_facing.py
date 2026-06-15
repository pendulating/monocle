"""Unit tests for dagspaces.common.curation.filter_facing.filter_facing.

Synthetic geometry: one building polygon directly north of a recording
location. Build a small parquet with 6 rows at the same lat/lon, one per
face (F/B/L/R/U/D) with bearings set so face bearings point in cardinal
directions. Only the face aimed at the building should survive the filter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import box
from shapely.wkb import dumps as wkb_dumps

from dagspaces.common.curation.filter_facing import filter_facing


def _synth_parquet_and_coverage(tmp_path: Path) -> tuple[Path, Path]:
    """A recording at (lon, lat) with one building ~20 m north of it.

    Using NYC coords so the default NY State Plane projection is sensible.
    Building is ~20 m to the north, ~5 m wide/tall. Rays cast north (bearing=0)
    should hit it; rays cast south, east, west should not (within 30 m default).
    """
    rec_lat, rec_lon = 40.7500, -73.9800

    # A 5m × 5m building box ~18 m north of the recording point.
    # 1 deg lat ≈ 111 km, so 18 m ≈ 0.0001618 deg, 5 m ≈ 0.0000450 deg.
    building = box(
        rec_lon - 0.00003,
        rec_lat + 0.00014,
        rec_lon + 0.00003,
        rec_lat + 0.00020,
    )
    cov_path = tmp_path / "coverage.geojson"
    gpd.GeoDataFrame(
        {"name": ["test_building"]},
        geometry=[building],
        crs="EPSG:4326",
    ).to_file(cov_path, driver="GeoJSON")

    # recorderDirection = 0 → F face points north; bearings = (0 + face_offset) % 360
    rows = [
        # face, bearing (absolute, degrees, 0=N)
        {"face": "F", "bearing": 0.0},     # looks north → hits building
        {"face": "R", "bearing": 90.0},    # east
        {"face": "B", "bearing": 180.0},   # south (away from building)
        {"face": "L", "bearing": 270.0},   # west
        {"face": "U", "bearing": None},    # no bearing
        {"face": "D", "bearing": None},    # no bearing
    ]
    for i, r in enumerate(rows):
        r["sample_id"] = f"REC1_{r['face']}"
        r["latitude"] = rec_lat
        r["longitude"] = rec_lon
        r["image_path"] = f"/dev/null/{i}.jpg"
        r["dataset"] = "synthetic_1k"

    df = pl.DataFrame(rows, schema_overrides={"bearing": pl.Float64})
    parquet = tmp_path / "curated.parquet"
    df.write_parquet(parquet)
    return parquet, cov_path


class TestFilterFacing:

    def test_only_facing_face_kept(self, tmp_path):
        parquet, cov = _synth_parquet_and_coverage(tmp_path)
        out = tmp_path / "filtered.parquet"
        r = filter_facing(
            str(parquet), str(cov), str(out),
            ray_length_m=30.0,
        )
        assert r.in_rows == 6
        assert r.horizontal_rows == 4        # dropped U, D
        assert r.with_bearing_rows == 4
        assert r.kept_rows == 1              # only F-face ray hits the building
        kept = pl.read_parquet(out)
        assert kept.height == 1
        assert kept["face"].to_list() == ["F"]
        assert r.per_face_kept == {"F": 1}

    def test_ray_length_too_short_drops_all(self, tmp_path):
        parquet, cov = _synth_parquet_and_coverage(tmp_path)
        out = tmp_path / "filtered.parquet"
        # 5 m ray can't reach an 18-m-away building, even from F face.
        r = filter_facing(
            str(parquet), str(cov), str(out),
            ray_length_m=5.0,
        )
        assert r.kept_rows == 0

    def test_ray_length_long_still_drops_off_axis(self, tmp_path):
        """Even with a long ray, a ray pointed away from the building should not hit it."""
        parquet, cov = _synth_parquet_and_coverage(tmp_path)
        out = tmp_path / "filtered.parquet"
        r = filter_facing(
            str(parquet), str(cov), str(out),
            ray_length_m=500.0,
        )
        # F still hits. B/L/R still miss (pointed wrong way). U/D still excluded.
        assert r.kept_rows == 1
        kept = pl.read_parquet(out)
        assert kept["face"].to_list() == ["F"]

    def test_manifest_contents(self, tmp_path):
        parquet, cov = _synth_parquet_and_coverage(tmp_path)
        out = tmp_path / "filtered.parquet"
        filter_facing(str(parquet), str(cov), str(out))
        manifest_path = tmp_path / "filtered_filter_facing_manifest.json"
        assert manifest_path.is_file()
        m = json.loads(manifest_path.read_text())
        assert m["in_rows"] == 6 and m["kept_rows"] == 1
        assert m["ray_length_m"] == 30.0
        assert m["per_face_kept"] == {"F": 1}
        assert m["per_dataset_kept"] == {"synthetic_1k": 1}

    def test_overwrite_guard(self, tmp_path):
        parquet, cov = _synth_parquet_and_coverage(tmp_path)
        out = tmp_path / "filtered.parquet"
        filter_facing(str(parquet), str(cov), str(out))
        with pytest.raises(FileExistsError):
            filter_facing(str(parquet), str(cov), str(out))
        filter_facing(str(parquet), str(cov), str(out), overwrite=True)

    def test_missing_columns_raises(self, tmp_path):
        parquet, cov = _synth_parquet_and_coverage(tmp_path)
        # Drop the bearing column
        df = pl.read_parquet(parquet).drop("bearing")
        bad = tmp_path / "bad.parquet"
        df.write_parquet(bad)
        with pytest.raises(ValueError, match="bearing"):
            filter_facing(str(bad), str(cov), str(tmp_path / "out.parquet"))


# ---------------------------------------------------------------------------
# Fix F — occlusion check (per-unit mode + nyc_buildings.parquet)
# ---------------------------------------------------------------------------


# Lat/lon scaling at NYC latitudes: 1 m ≈ 9.0e-6 deg lat, 1.19e-5 deg lon at lat 40.75.
_M_LAT = 9.0e-6
_M_LON = 1.19e-5
_REC_LAT = 40.7500
_REC_LON = -73.9800


def _box_around(lat_m: float, lon_m: float, half_m: float):
    """Axis-aligned box centered at (lat_m, lon_m) meters offset from the
    recording, extending ±``half_m`` meters in both axes. Returned in WGS84."""
    return box(
        _REC_LON + (lon_m - half_m) * _M_LON,
        _REC_LAT + (lat_m - half_m) * _M_LAT,
        _REC_LON + (lon_m + half_m) * _M_LON,
        _REC_LAT + (lat_m + half_m) * _M_LAT,
    )


def _synth_occlusion_fixtures(
    tmp_path: Path,
    *,
    include_occluder: bool = True,
    include_side: bool = False,
    unit_bin: str = "LIB",
) -> tuple[Path, Path, Path]:
    """Build synthetic (curated_parquet, units_parquet, buildings_parquet).

    Layout (recording at origin, north = +lat):
      - Library BIN="LIB" at ~30 m north, 20 m × 20 m footprint.
      - Unit polygon: 60 m × 60 m centered at 30 m north (the 80-ft buffer
        analog), so recording-pointing-north sample rays at 10/20/30 m land
        inside the unit polygon → Fix A passes.
      - Recording at (0, 0) with a single F-face row (bearing=0, north).
      - If ``include_occluder``: BIN="OCC" 5 m × 5 m at 15 m north — sits
        on the recording→library segment and should trigger a strict pierce.
      - If ``include_side``: BIN="SIDE" 10 m × 10 m at +20 m east, off the
        north-pointing segment — sjoin won't match, so no drop.
    """
    # --- Buildings parquet (what Fix F reads) ---
    lib_bldg = _box_around(lat_m=30, lon_m=0, half_m=10)    # 20 m square
    bldgs = [{"bin": unit_bin, "geometry": lib_bldg}]
    if include_occluder:
        occ_bldg = _box_around(lat_m=15, lon_m=0, half_m=2.5)  # 5 m square
        bldgs.append({"bin": "OCC", "geometry": occ_bldg})
    if include_side:
        side_bldg = _box_around(lat_m=0, lon_m=20, half_m=5)  # 10 m square
        bldgs.append({"bin": "SIDE", "geometry": side_bldg})
    buildings_gdf = gpd.GeoDataFrame(bldgs, geometry="geometry", crs="EPSG:4326")
    buildings_path = tmp_path / "nyc_buildings.parquet"
    buildings_gdf.to_parquet(buildings_path)

    # --- Units parquet (unit_uid="U1", 60 m × 60 m buffered polygon, bin=LIB) ---
    unit_poly = _box_around(lat_m=30, lon_m=0, half_m=30)   # 60 m square
    units_df = pl.DataFrame(
        {
            "uid": ["U1"],
            "facname": ["Test Library"],
            "bin": [unit_bin if unit_bin else ""],
            "geom_wkb": [wkb_dumps(unit_poly, hex=False)],
        }
    )
    units_path = tmp_path / "facilities.parquet"
    units_df.write_parquet(units_path)

    # --- Curated parquet: single F-face row, bearing=0 (north) ---
    curated = pl.DataFrame(
        [
            {
                "sample_id": "REC1_F",
                "face": "F",
                "bearing": 0.0,
                "latitude": _REC_LAT,
                "longitude": _REC_LON,
                "unit_uid": "U1",
                "image_path": "/dev/null/0.jpg",
                "dataset": "synthetic_1k",
            }
        ],
        schema_overrides={"bearing": pl.Float64},
    )
    curated_path = tmp_path / "curated.parquet"
    curated.write_parquet(curated_path)

    return curated_path, units_path, buildings_path


class TestFilterFacingOcclusion:

    def test_drops_when_non_unit_building_pierces_segment(self, tmp_path):
        parquet, units, buildings = _synth_occlusion_fixtures(
            tmp_path, include_occluder=True
        )
        out = tmp_path / "filtered.parquet"
        r = filter_facing(
            str(parquet), coverage_geojson=None, output_parquet=str(out),
            units_parquet=str(units),
            buildings_path=str(buildings),
            occlusion=True,
            ray_length_m=30.0,
        )
        assert r.mode == "per_unit"
        # Fix A passes (library's 60 m unit polygon catches the ray samples),
        # but Fix F drops because the OCC polygon sits between camera and library.
        assert r.dropped_by_occlusion == 1
        assert r.occlusion_processable == 1
        assert r.kept_rows == 0

    def test_keeps_when_only_off_axis_building_exists(self, tmp_path):
        parquet, units, buildings = _synth_occlusion_fixtures(
            tmp_path, include_occluder=False, include_side=True
        )
        out = tmp_path / "filtered.parquet"
        r = filter_facing(
            str(parquet), coverage_geojson=None, output_parquet=str(out),
            units_parquet=str(units),
            buildings_path=str(buildings),
            occlusion=True,
            ray_length_m=30.0,
        )
        # Library + off-axis SIDE building; segment doesn't cross SIDE.
        assert r.dropped_by_occlusion == 0
        assert r.kept_rows == 1

    def test_no_occlusion_flag_disables_check(self, tmp_path):
        parquet, units, buildings = _synth_occlusion_fixtures(
            tmp_path, include_occluder=True
        )
        out = tmp_path / "filtered.parquet"
        r = filter_facing(
            str(parquet), coverage_geojson=None, output_parquet=str(out),
            units_parquet=str(units),
            buildings_path=str(buildings),
            occlusion=False,
            ray_length_m=30.0,
        )
        # Same geometry as the drops-when-pierced test, but Fix F disabled.
        assert r.dropped_by_occlusion == 0
        assert r.occlusion_processable == 0
        assert r.kept_rows == 1

    def test_pass_through_when_unit_bin_absent(self, tmp_path):
        # unit_bin="" → units parquet carries empty BIN → occlusion can't
        # look up a blocker anchor → pass-through even with OCC polygon present.
        parquet, units, buildings = _synth_occlusion_fixtures(
            tmp_path, include_occluder=True, unit_bin="",
        )
        out = tmp_path / "filtered.parquet"
        r = filter_facing(
            str(parquet), coverage_geojson=None, output_parquet=str(out),
            units_parquet=str(units),
            buildings_path=str(buildings),
            occlusion=True,
            ray_length_m=30.0,
        )
        assert r.occlusion_processable == 0
        assert r.dropped_by_occlusion == 0
        assert r.kept_rows == 1

    def test_manifest_records_occlusion_fields(self, tmp_path):
        parquet, units, buildings = _synth_occlusion_fixtures(
            tmp_path, include_occluder=True
        )
        out = tmp_path / "filtered.parquet"
        filter_facing(
            str(parquet), coverage_geojson=None, output_parquet=str(out),
            units_parquet=str(units),
            buildings_path=str(buildings),
            occlusion=True,
            ray_length_m=30.0,
        )
        m = json.loads((tmp_path / "filtered_filter_facing_manifest.json").read_text())
        assert m["mode"] == "per_unit"
        assert m["dropped_by_occlusion"] == 1
        assert m["occlusion_processable"] == 1
        assert m["buildings_source"] == str(buildings)
