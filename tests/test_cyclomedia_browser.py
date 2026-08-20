"""Tests for the Cyclomedia browser layers (catalog / cubemap / depth).

The cube-geometry tests are the load-bearing ones: they pin down the claim that
the cube is *north-referenced* (F faces true north), which the whole panorama
and any downstream georeferencing depends on.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from dagspaces.common.cyclomedia import catalog as cat
from dagspaces.common.cyclomedia import cubemap as cube
from dagspaces.common.cyclomedia import depth as dep

# ---------------------------------------------------------------- cube geometry


def test_face_ray_matches_documented_directions():
    """Face centres (u=v=0) point along their documented compass axes."""
    zero = np.array([0.0])
    centres = {f: cube.face_ray(f, zero, zero)[0] for f in cube.FACES}

    # World frame is (East, North, Up).
    np.testing.assert_allclose(centres["F"], [0, 1, 0], atol=1e-12)  # north
    np.testing.assert_allclose(centres["R"], [1, 0, 0], atol=1e-12)  # east
    np.testing.assert_allclose(centres["B"], [0, -1, 0], atol=1e-12)  # south
    np.testing.assert_allclose(centres["L"], [-1, 0, 0], atol=1e-12)  # west
    np.testing.assert_allclose(centres["U"], [0, 0, 1], atol=1e-12)  # zenith
    np.testing.assert_allclose(centres["D"], [0, 0, -1], atol=1e-12)  # nadir


def test_face_ray_and_uv_inversion_round_trip():
    """_face_uv_from_ray inverts face_ray for every face across the whole face."""
    grid = np.linspace(-0.95, 0.95, 7)
    uu, vv = np.meshgrid(grid, grid)

    for face in cube.FACES:
        rays = cube.face_ray(face, uu.ravel(), vv.ravel())
        x, y, z = rays[:, 0], rays[:, 1], rays[:, 2]
        masks, uv = cube._face_uv_from_ray(x, y, z)

        # Every ray from a face must land back on that same face.
        assert masks[face].all(), f"{face}: rays did not map back to their own face"

        u_back, v_back = uv[face]
        np.testing.assert_allclose(u_back, uu.ravel(), atol=1e-9, err_msg=f"{face} u")
        np.testing.assert_allclose(v_back, vv.ravel(), atol=1e-9, err_msg=f"{face} v")


def test_equirect_places_faces_by_compass_bearing():
    """A solid-colour cube lands in the panorama where the compass says it should.

    This is the real assertion behind "F faces north": if the face orientation or
    the bearing convention were wrong, the colours would land in the wrong
    columns/rows.
    """
    colours = {
        "F": (255, 0, 0),
        "R": (0, 255, 0),
        "B": (0, 0, 255),
        "L": (255, 255, 0),
        "U": (255, 0, 255),
        "D": (0, 255, 255),
    }
    faces = {
        f: np.tile(np.array(c, dtype=np.uint8), (64, 64, 1)) for f, c in colours.items()
    }

    w, h = 720, 360
    pano = cube.cube_to_equirect(faces, width=w, height=h)

    horizon = h // 2

    def at(bearing_deg: float, row: int) -> tuple:
        col = int(bearing_deg / 360.0 * w) % w
        return tuple(pano[row, col])

    # Column 0 is north; bearing increases eastward.
    assert at(0, horizon) == colours["F"], "bearing 0 should be F (north)"
    assert at(90, horizon) == colours["R"], "bearing 90 should be R (east)"
    assert at(180, horizon) == colours["B"], "bearing 180 should be B (south)"
    assert at(270, horizon) == colours["L"], "bearing 270 should be L (west)"

    # Poles.
    assert tuple(pano[0, w // 2]) == colours["U"], "top row should be U (zenith)"
    assert tuple(pano[h - 1, w // 2]) == colours["D"], "bottom row should be D (nadir)"


def test_equirect_preserves_dtype_and_channels():
    """Single-channel uint16 (depth codes) stitch without being coerced to RGB."""
    faces = {f: np.full((32, 32), i + 1, dtype=np.uint16) for i, f in enumerate(cube.FACES)}
    pano = cube.cube_to_equirect(faces, width=128, height=64)

    assert pano.shape == (64, 128)
    assert pano.dtype == np.uint16
    assert set(np.unique(pano)) == {1, 2, 3, 4, 5, 6}


def test_equirect_missing_face_uses_fill():
    """A missing face leaves its region at `fill` rather than blowing up."""
    faces = {
        f: np.full((32, 32, 3), 200, dtype=np.uint8)
        for f in cube.FACES
        if f != "U"
    }
    pano = cube.cube_to_equirect(faces, width=128, height=64, fill=0)

    assert (pano[0, :] == 0).all(), "zenith row should be fill where U is missing"
    assert (pano[32, :] == 200).all(), "horizon should still be painted"


def test_cube_cross_layout_places_faces():
    """The unfolded cross puts each face in its documented slot."""
    colours = {f: (i * 40, 0, 0) for i, f in enumerate(cube.FACES)}
    faces = {
        f: np.tile(np.array(c, dtype=np.uint8), (16, 16, 1)) for f, c in colours.items()
    }
    cross = cube.cube_cross(faces, pad=0)

    assert cross.shape == (3 * 16, 4 * 16, 3)

    def cell(r: int, c: int) -> tuple:
        return tuple(cross[r * 16 + 8, c * 16 + 8])

    assert cell(0, 1) == colours["U"]
    for i, f in enumerate(("L", "F", "R", "B")):
        assert cell(1, i) == colours[f]
    assert cell(2, 1) == colours["D"]


# -------------------------------------------------------------------- depth


def test_decode_depth_packs_r_and_g_big_endian():
    """code = R*256 + G; the blue channel is ignored."""
    img = np.array(
        [[[0, 0, 0], [1, 0, 255], [66, 200, 7], [255, 255, 0]]], dtype=np.uint8
    )
    code = dep.decode_depth(img)

    assert code.dtype == np.uint16
    np.testing.assert_array_equal(code, [[0, 256, 66 * 256 + 200, 65535]])


def test_decode_depth_zero_is_no_return_not_near():
    """Zero is the no-return sentinel, and must not read as "very close"."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = (70, 10, 0)
    code = dep.decode_depth(img)

    valid = code > dep.NO_RETURN
    assert valid.sum() == 1
    assert code[0, 0] == 70 * 256 + 10


def test_to_metres_applies_the_calibrated_affine_decode():
    """range_m = (code - 16384) / 250, i.e. a 4 mm quantum."""
    code = np.array([[16384, 16384 + 250, 16384 + 2500, 65535]], dtype=np.uint16)
    m = dep.to_metres(code)

    np.testing.assert_allclose(m[0, :3], [0.0, 1.0, 10.0], atol=1e-4)
    assert m[0, 3] == pytest.approx(dep.MAX_RANGE_M, abs=1e-3)
    assert dep.MAX_RANGE_M == pytest.approx(196.604, abs=1e-3)


def test_to_metres_maps_no_return_to_nan_not_a_negative_distance():
    """Sentinel 0 would otherwise decode to -65.5 m and poison any aggregate."""
    code = np.array([[0, 20_000]], dtype=np.uint16)
    m = dep.to_metres(code)

    assert np.isnan(m[0, 0])
    assert m[0, 1] == pytest.approx((20_000 - 16384) / 250.0)
    assert np.nanmax(m) > 0


def test_depth_stats_reports_coverage_over_valid_pixels_only():
    code = np.zeros((10, 10), dtype=np.uint16)
    code[:, :5] = 20_000  # half the image is a valid return

    stats = dep.depth_stats(code)
    assert stats["n_valid"] == 50
    assert stats["pct_valid"] == pytest.approx(50.0)
    assert stats["pct_no_return"] == pytest.approx(50.0)
    assert stats["p50"] == pytest.approx(20_000)


def test_colorize_depth_paints_no_return_pixels_distinctly():
    code = np.zeros((4, 4), dtype=np.uint16)
    code[0, :] = 17_000
    code[1, :] = 30_000

    rgb = dep.colorize_depth(code, invalid_rgb=(0, 0, 0))
    assert rgb.shape == (4, 4, 3)
    assert (rgb[2:] == 0).all(), "no-return rows should be painted invalid_rgb"
    assert not (rgb[0] == 0).all(), "valid rows should be coloured"
    # Near and far must not collapse to the same colour.
    assert not np.array_equal(rgb[0], rgb[1])


def test_colorize_depth_shared_scale_is_comparable_across_faces():
    """Explicit vmin/vmax make two faces directly comparable."""
    a = np.full((4, 4), 18_000, dtype=np.uint16)
    b = np.full((4, 4), 18_000, dtype=np.uint16)
    b[0, 0] = 40_000  # b has a far pixel; a does not

    ca = dep.colorize_depth(a, vmin=16_000, vmax=40_000)
    cb = dep.colorize_depth(b, vmin=16_000, vmax=40_000)

    # Same code -> same colour under a shared scale, regardless of face content.
    np.testing.assert_array_equal(ca[1, 1], cb[1, 1])


# ------------------------------------------------------------------- catalog


def test_recording_dir_builds_path_without_touching_the_filesystem():
    """Paths are constructed from catalog fields -- never by globbing NFS."""
    got = cat.recording_dir("manhattan_2025_1k", "W0CDN", "W0CDN0T5")
    assert got == "/share/ju/cyclomedia/raw/manhattan_2025_1k/W0CDN/W0CDN0T5"


_HAS_INDEX = os.path.exists(cat.DEFAULT_INDEX_PATH)
_needs_index = pytest.mark.skipif(
    not _HAS_INDEX, reason="recording index not built (run build_recording_index)"
)


@pytest.fixture(scope="module")
def con():
    c = cat.connect()
    cat.load_recording_index(c)
    return c


@_needs_index
def test_nearest_recordings_are_sorted_and_within_radius(con):
    df = cat.nearest_recordings(con, lat=40.7580, lon=-73.9855, k=10, radius_m=150)

    assert len(df) == 10
    assert (df["dist_m"] <= 150).all()
    assert df["dist_m"].is_monotonic_increasing
    assert (df["borough"] == "manhattan").all()


@_needs_index
def test_nearest_recordings_honours_the_filter(con):
    df = cat.nearest_recordings(
        con, lat=40.7580, lon=-73.9855, k=5, radius_m=500, where="borough = 'brooklyn'"
    )
    # Times Square is nowhere near Brooklyn, so the filter must empty the result
    # rather than quietly returning Manhattan rows.
    assert df.empty


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


@_needs_index
def test_recordings_in_bbox_returns_everything_under_the_cap(con):
    df = cat.recordings_in_bbox(con, 40.7580, 40.7585, -73.9855, -73.9850, limit=5000)

    assert df.attrs["sampled"] is False
    assert len(df) == df.attrs["total_in_bbox"]


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


@_needs_index
def test_recording_index_has_one_row_per_recording(con):
    """recording_id is unique -- cross-dataset duplicates are collapsed."""
    n_rows, n_ids = con.execute(
        "SELECT count(*), count(DISTINCT recording_id) FROM recordings"
    ).fetchone()
    assert n_rows == n_ids
