"""Tests for the stored-truth path and the block pooling.

`--truth-from` lets a gate reuse an input ablation that a previous run
measured. That saves 256 forward passes for each image, so an error here would
silently score every instrument against the wrong ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle.input_gate import load_truth  # noqa: E402
from research.monocle.patch_map_causal import patch_groups, pool_by_groups  # noqa: E402

N = 256


class TestPoolByGroups:
    def test_group_one_is_the_identity(self):
        v = np.random.default_rng(0).normal(size=N)
        assert np.allclose(pool_by_groups(v, 1), v)

    def test_the_total_is_preserved(self):
        v = np.random.default_rng(1).normal(size=N)
        for g in (2, 4, 8, 16):
            assert pool_by_groups(v, g).sum() == pytest.approx(v.sum())

    def test_the_size_falls_by_the_square(self):
        v = np.random.default_rng(2).normal(size=N)
        for g in (1, 2, 4, 8, 16):
            assert pool_by_groups(v, g).size == N // (g * g)

    def test_a_block_holds_the_right_cells(self):
        v = np.arange(N, dtype=float)
        grid = v.reshape(16, 16)
        got = pool_by_groups(v, 2).reshape(8, 8)
        assert got[0, 0] == pytest.approx(grid[0:2, 0:2].sum())
        assert got[3, 5] == pytest.approx(grid[6:8, 10:12].sum())
        assert got[7, 7] == pytest.approx(grid[14:16, 14:16].sum())

    def test_it_is_row_major_like_patch_groups(self):
        v = np.random.default_rng(3).normal(size=N)
        want = np.array([v[idx].sum() for idx in patch_groups(N, 4)])
        assert np.allclose(pool_by_groups(v, 4), want)

    def test_a_grid_that_does_not_divide_is_an_error(self):
        with pytest.raises(ValueError):
            pool_by_groups(np.zeros(N), 3)

    def test_a_map_that_is_not_square_is_an_error(self):
        with pytest.raises(ValueError):
            pool_by_groups(np.zeros(10), 2)


def write(tmp_path: Path, rows: list[dict], name="m.parquet") -> str:
    p = tmp_path / name
    pd.DataFrame(rows).to_parquet(p, index=False)
    return str(p)


class TestLoadTruth:
    def test_the_causal_maps_layout(self, tmp_path):
        rng = np.random.default_rng(0)
        want = {f"i{i}": rng.normal(size=N) for i in range(3)}
        path = write(tmp_path, [{"image_id": k, "layer": None, "values": v}
                                for k, v in want.items()])
        got = load_truth(path, "subway_safety", 1)
        assert set(got) == set(want)
        for k in want:
            assert np.allclose(got[k], want[k])

    def test_the_input_gate_layout(self, tmp_path):
        rng = np.random.default_rng(1)
        want = {f"i{i}": rng.normal(size=N) for i in range(3)}
        path = write(tmp_path, [{"target": k, "layer": 12, "causal": v}
                                for k, v in want.items()])
        got = load_truth(path, "subway_safety", 1)
        assert set(got) == set(want)

    def test_repeated_layers_give_one_map(self, tmp_path):
        rng = np.random.default_rng(2)
        want = {f"i{i}": rng.normal(size=N) for i in range(4)}
        rows = [{"target": k, "layer": ly, "causal": v}
                for ly in (0, 6, 12) for k, v in want.items()]
        got = load_truth(write(tmp_path, rows), "subway_safety", 1)
        assert len(got) == 4
        for k in want:
            assert np.allclose(got[k], want[k])

    def test_the_group_pools_the_truth(self, tmp_path):
        v = np.arange(N, dtype=float)
        path = write(tmp_path, [{"image_id": "i0", "values": v}])
        got = load_truth(path, "subway_safety", 2)["i0"]
        assert got.size == 64
        assert got.sum() == pytest.approx(v.sum())

    def test_a_scalar_row_is_skipped(self, tmp_path):
        path = write(tmp_path, [{"image_id": "i0", "values": [1.0]},
                                {"image_id": "i1", "values": np.zeros(N)}])
        got = load_truth(path, "subway_safety", 1)
        assert set(got) == {"i1"}

    def test_missing_columns_raise(self, tmp_path):
        path = write(tmp_path, [{"foo": 1, "bar": 2}])
        with pytest.raises(ValueError):
            load_truth(path, "subway_safety", 1)

    def test_the_sign_is_not_changed(self, tmp_path):
        """Both layouts store the CONTRIBUTION, so no negation may happen."""
        v = np.linspace(-1, 1, N)
        path = write(tmp_path, [{"image_id": "i0", "values": v}])
        assert np.allclose(load_truth(path, "subway_safety", 1)["i0"], v)


class TestRealRun:
    """The stored truth of the 2026-08-23 overnight run must still load."""

    PATH = REPO / "outputs/_monocle/causal_maps/subway_safety_absent/maps.parquet"

    @pytest.mark.skipif(not PATH.exists(), reason="the overnight run is absent")
    def test_twenty_images_load(self):
        got = load_truth(str(self.PATH), "subway_safety", 1)
        assert len(got) == 20
        assert all(v.size == N for v in got.values())

    @pytest.mark.skipif(not PATH.exists(), reason="the overnight run is absent")
    def test_the_pooled_total_matches(self):
        fine = load_truth(str(self.PATH), "subway_safety", 1)
        coarse = load_truth(str(self.PATH), "subway_safety", 2)
        for k in fine:
            assert coarse[k].sum() == pytest.approx(fine[k].sum(), abs=1e-6)
