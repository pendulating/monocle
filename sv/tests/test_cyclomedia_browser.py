"""Tests of the NYC browser layer (sv.cyclomedia) over the framework code.

The cube and depth tests are in image-forward
(``tests/test_sources_cyclomedia.py``). Here: the NYC folders of the shim,
the shim names, and the queries on the real NYC recording index (marked
``data``; they skip when the index is not built).
"""

from __future__ import annotations

import os

import pytest

from sv.cyclomedia import catalog as cat
from sv.cyclomedia import cubemap as cube
from sv.cyclomedia import depth as dep


def test_the_shims_are_the_framework_code():
    from dagspaces.sources.cyclomedia import cubemap, depth, recordings

    assert cube.cube_to_equirect is cubemap.cube_to_equirect
    assert cube._face_uv_from_ray is cubemap._face_uv_from_ray
    assert dep.decode_depth is depth.decode_depth and dep.NO_RETURN == 0
    assert cat.nearest_recordings is recordings.nearest_recordings
    assert set(cube.__all__) >= {"FACES", "cube_cross", "load_faces"}


def test_recording_dir_builds_path_without_touching_the_filesystem():
    """Paths are constructed from catalog fields -- never by globbing NFS."""
    got = cat.recording_dir("manhattan_2025_1k", "W0CDN", "W0CDN0T5")
    assert got == "/share/ju/cyclomedia/raw/manhattan_2025_1k/W0CDN/W0CDN0T5"
    assert cat.FACE_GLOB == "/share/ju/cyclomedia/catalog/v1/by_dataset/**/*.parquet"


_HAS_INDEX = os.path.exists(cat.DEFAULT_INDEX_PATH)
_needs_index = pytest.mark.skipif(
    not _HAS_INDEX, reason="recording index not built (run build_recording_index)"
)


@pytest.fixture(scope="module")
def con():
    c = cat.connect()
    cat.load_recording_index(c)
    return c


@pytest.mark.data
@_needs_index
def test_nearest_recordings_are_sorted_and_within_radius(con):
    df = cat.nearest_recordings(con, lat=40.7580, lon=-73.9855, k=10, radius_m=150)

    assert len(df) == 10
    assert (df["dist_m"] <= 150).all()
    assert df["dist_m"].is_monotonic_increasing
    assert (df["borough"] == "manhattan").all()


@pytest.mark.data
@_needs_index
def test_nearest_recordings_honours_the_filter(con):
    df = cat.nearest_recordings(
        con, lat=40.7580, lon=-73.9855, k=5, radius_m=500, where="borough = 'brooklyn'"
    )
    # Times Square is nowhere near Brooklyn, so the filter must empty the result
    # rather than quietly returning Manhattan rows.
    assert df.empty


@pytest.mark.data
@_needs_index
def test_recordings_in_bbox_samples_instead_of_truncating(con):
    """When the render cap bites, the sample must still span the whole box."""
    lat0, lat1, lon0, lon1 = 40.750, 40.765, -73.995, -73.975
    df = cat.recordings_in_bbox(con, lat0, lat1, lon0, lon1, limit=500)

    assert len(df) == 500
    assert df.attrs["sampled"] is True
    assert df.attrs["total_in_bbox"] > 500

    # A truncating implementation would collapse into one corner; a sample spans
    # the box. Require coverage of most of each axis.
    assert (df["latitude"].max() - df["latitude"].min()) > 0.6 * (lat1 - lat0)
    assert (df["longitude"].max() - df["longitude"].min()) > 0.6 * (lon1 - lon0)


@pytest.mark.data
@_needs_index
def test_recordings_in_bbox_returns_everything_under_the_cap(con):
    df = cat.recordings_in_bbox(con, 40.7580, 40.7585, -73.9855, -73.9850, limit=5000)

    assert df.attrs["sampled"] is False
    assert len(df) == df.attrs["total_in_bbox"]


@pytest.mark.data
@_needs_index
def test_overview_grid_conserves_the_recording_count(con):
    """Binning must not lose or duplicate recordings."""
    where = "borough = 'staten_island'"
    grid = cat.overview_grid(con, cell_m=500, where=where)
    total = int(
        con.execute(f"SELECT count(*) FROM recordings WHERE {where}").fetchone()[0]
    )

    assert int(grid["n"].sum()) == total
    assert len(grid) < total  # it actually aggregated


@pytest.mark.data
@_needs_index
def test_recording_index_has_one_row_per_recording(con):
    """recording_id is unique -- cross-dataset duplicates are collapsed."""
    n_rows, n_ids = con.execute(
        "SELECT count(*), count(DISTINCT recording_id) FROM recordings"
    ).fetchone()
    assert n_rows == n_ids
