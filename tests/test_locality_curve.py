"""Tests for the locality curve join.

The curve decides which layer a causal claim can use, so a silent join error
would send the whole ladder to the wrong layer.
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
sys.path.insert(0, str(REPO / "research" / "scripts"))

from research.scripts.locality_curve import curve, full_r, input_truth  # noqa: E402

N_PATCH = 256


def make_truth(n_img: int, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {f"img{i}": rng.normal(size=N_PATCH) for i in range(n_img)}


def donor_frame(truth: dict, layers: dict[int, float], seed: int = 1
                ) -> pd.DataFrame:
    """A donor map that mixes the truth with noise by a per-layer weight."""
    rng = np.random.default_rng(seed)
    rows = []
    for layer, w in layers.items():
        for tgt, gold in truth.items():
            noise = rng.normal(size=N_PATCH)
            rows.append({"layer": layer, "target": tgt, "donor": "d0",
                         "causal": w * gold + (1.0 - w) * noise})
    return pd.DataFrame(rows)


class TestFullR:
    def test_identical_is_one(self):
        a = np.random.default_rng(0).normal(size=N_PATCH)
        assert full_r(a, a) == pytest.approx(1.0)

    def test_negated_is_minus_one(self):
        a = np.random.default_rng(0).normal(size=N_PATCH)
        assert full_r(a, -a) == pytest.approx(-1.0)

    def test_constant_is_nan(self):
        a = np.ones(N_PATCH)
        assert np.isnan(full_r(a, np.random.default_rng(0).normal(size=N_PATCH)))

    def test_nan_entries_are_dropped(self):
        a = np.random.default_rng(0).normal(size=N_PATCH)
        b = a.copy()
        b[:10] = np.nan
        assert full_r(a, b) == pytest.approx(1.0)


class TestInputTruth:
    def test_one_map_for_each_image(self):
        gold = make_truth(3)
        df = pd.DataFrame([
            {"layer": ly, "target": t, "causal": m}
            for ly in (0, 4, 8) for t, m in gold.items()])
        got = input_truth(df)
        assert set(got) == set(gold)
        for t in gold:
            assert np.allclose(got[t], gold[t])

    def test_a_scalar_column_is_skipped(self):
        gold = make_truth(1)
        df = pd.DataFrame([
            {"layer": 0, "target": "img0", "causal": None},
            {"layer": 4, "target": "img0", "causal": gold["img0"]}])
        got = input_truth(df)
        assert np.allclose(got["img0"], gold["img0"])

    def test_empty_when_no_array(self):
        df = pd.DataFrame([{"layer": 0, "target": "img0", "causal": None}])
        assert input_truth(df) == {}


class TestCurve:
    def test_a_local_layer_scores_high_and_a_mixed_layer_does_not(self):
        gold = make_truth(8)
        df = donor_frame(gold, {0: 1.0, 24: 0.0})
        tab = curve(df, gold, k=50).set_index("layer")
        assert tab.loc[0, "r_full"] == pytest.approx(1.0, abs=1e-6)
        assert abs(tab.loc[24, "r_full"]) < 0.2

    def test_the_curve_falls_with_depth(self):
        gold = make_truth(10)
        df = donor_frame(gold, {0: 1.0, 8: 0.7, 16: 0.4, 24: 0.05})
        tab = curve(df, gold, k=50).sort_values("layer")
        vals = tab["r_full"].to_numpy()
        assert np.all(np.diff(vals) < 0)

    def test_the_null_is_near_zero_for_unrelated_images(self):
        gold = make_truth(12)
        df = donor_frame(gold, {0: 1.0})
        tab = curve(df, gold, k=50)
        assert abs(float(tab["r_null"].iloc[0])) < 0.3
        assert float(tab["excess"].iloc[0]) > 0.6

    def test_shared_structure_lifts_the_null(self):
        """A ramp in every map must lift the null, not the excess."""
        rng = np.random.default_rng(3)
        ramp = np.linspace(-2, 2, N_PATCH)
        gold = {f"img{i}": ramp + 0.3 * rng.normal(size=N_PATCH)
                for i in range(10)}
        df = donor_frame(gold, {0: 1.0})
        tab = curve(df, gold, k=50)
        assert float(tab["r_null"].iloc[0]) > 0.5
        assert float(tab["excess"].iloc[0]) < 0.5

    def test_a_size_mismatch_is_skipped(self):
        gold = make_truth(4)
        df = donor_frame(gold, {0: 1.0})
        df.at[0, "causal"] = np.zeros(64)
        tab = curve(df, gold, k=50)
        assert int(tab["n_pairs"].iloc[0]) == 3

    def test_an_unknown_target_is_skipped(self):
        gold = make_truth(4)
        df = donor_frame(gold, {0: 1.0})
        df.loc[0, "target"] = "not_an_image"
        tab = curve(df, gold, k=50)
        assert int(tab["n_pairs"].iloc[0]) == 3

    def test_it_is_deterministic(self):
        gold = make_truth(8)
        df = donor_frame(gold, {0: 0.5})
        a = curve(df, gold, k=50)
        b = curve(df, gold, k=50)
        pd.testing.assert_frame_equal(a, b)

    def test_many_donors_for_one_target_all_count(self):
        gold = make_truth(3)
        rows = []
        for d in ("d0", "d1", "d2"):
            for t, m in gold.items():
                rows.append({"layer": 12, "target": t, "donor": d, "causal": m})
        tab = curve(pd.DataFrame(rows), gold, k=50)
        assert int(tab["n_pairs"].iloc[0]) == 9
        assert tab["r_full"].iloc[0] == pytest.approx(1.0, abs=1e-6)
