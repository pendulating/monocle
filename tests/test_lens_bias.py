"""Tests for the lens scoring metric (research/monocle/lens_bias.py).

The metric decides the ladder, so it must be exact. `--verify` reproduces the
stage-4 table from the real parquets; these tests cover the mechanics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.monocle import lens_bias as lb  # noqa: E402

RNG = np.random.default_rng(777)


def frame(n_maps=3, size=100, **cols):
    """A maps.parquet-shaped frame with one layer."""
    rows = []
    for i in range(n_maps):
        row = {"layer": 18, "target": f"img{i}",
               "causal": RNG.normal(size=size)}
        for k, v in cols.items():
            row[k] = v(row["causal"]) if callable(v) else v
        rows.append(row)
    return pd.DataFrame(rows)


class TestAsMap:
    def test_none_is_none(self):
        assert lb.as_map(None) is None

    def test_scalar_is_none(self):
        """A null lens column stores a scalar, not an array."""
        assert lb.as_map(np.float64("nan")) is None
        assert lb.as_map(3.0) is None

    def test_array_survives(self):
        out = lb.as_map(np.arange(5.0))
        assert out is not None and out.shape == (5,)


class TestTopKScore:
    def test_identical_maps_score_one(self):
        c = RNG.normal(size=100)
        sign, r = lb.top_k_score(c, c.copy(), k=50)
        assert r == pytest.approx(1.0)
        assert sign == pytest.approx(1.0)

    def test_negated_map_scores_minus_one(self):
        c = RNG.normal(size=100)
        sign, r = lb.top_k_score(c, -c, k=50)
        assert r == pytest.approx(-1.0)
        assert sign == pytest.approx(0.0)

    def test_absent_candidate_is_nan(self):
        assert np.isnan(lb.top_k_score(RNG.normal(size=100), None)[1])

    def test_too_few_patches_is_nan(self):
        c = RNG.normal(size=10)
        assert np.isnan(lb.top_k_score(c, c.copy(), k=50)[1])

    def test_constant_candidate_is_nan(self):
        c = RNG.normal(size=100)
        assert np.isnan(lb.top_k_score(c, np.ones(100), k=50)[1])

    def test_mismatched_sizes_is_nan(self):
        assert np.isnan(
            lb.top_k_score(RNG.normal(size=100), RNG.normal(size=60))[1])

    def test_only_the_top_k_count(self):
        """Noise outside the top k must not move the score."""
        c = np.concatenate([np.full(50, 10.0) + RNG.normal(size=50), 
                            RNG.normal(size=50) * 0.01])
        m = c.copy()
        m[50:] = RNG.normal(size=50) * 500  # garbage, all outside the top 50
        assert lb.top_k_score(c, m, k=50)[1] == pytest.approx(1.0)


class TestNanMean:
    def test_ignores_nan(self):
        assert lb.nanmean([1.0, float("nan"), 3.0]) == pytest.approx(2.0)

    def test_all_nan_is_nan_and_silent(self):
        with np.errstate(all="raise"):
            assert np.isnan(lb.nanmean([float("nan")] * 3))

    def test_empty_is_nan(self):
        assert np.isnan(lb.nanmean([]))


class TestScoreMaps:
    def test_perfect_lens_scores_one(self):
        df = frame(lens_delta=lambda c: c.copy(), taylor=lambda c: c.copy())
        out = lb.score_maps(df, k=50)
        assert out.lens_delta_r.iloc[0] == pytest.approx(1.0)
        assert out.bias_r.iloc[0] == pytest.approx(1.0)

    def test_absent_lens_column_is_nan_not_a_crash(self):
        out = lb.score_maps(frame(taylor=lambda c: c.copy()), k=50)
        assert np.isnan(out.lens_delta_r.iloc[0])
        assert np.isnan(out.bias_r.iloc[0])

    def test_null_lens_rows_score_nan(self):
        """A layer the lens is not fitted at must not score a number."""
        df = frame(taylor=lambda c: c.copy(), lens_delta=None)
        out = lb.score_maps(df, k=50)
        assert np.isnan(out.lens_delta_r.iloc[0])
        assert np.isnan(out.bias_r.iloc[0])

    def test_bias_compares_lens_to_taylor_not_to_causal(self):
        """bias must read the LOCAL map, so an anti-taylor lens goes negative
        even when it correlates with causal."""
        df = frame(n_maps=2, taylor=lambda c: -c, lens_delta=lambda c: c.copy())
        out = lb.score_maps(df, k=50)
        assert out.lens_delta_r.iloc[0] == pytest.approx(1.0)
        assert out.bias_r.iloc[0] == pytest.approx(-1.0)

    def test_n_maps_is_reported(self):
        assert lb.score_maps(frame(n_maps=4), k=50).n_maps.iloc[0] == 4


class TestGuards:
    def test_k_default_matches_the_baseline(self):
        assert lb.DEFAULT_K == 50, "changing k voids the stage-4 baseline"

    def test_stage4_table_covers_both_runs(self):
        assert set(lb.STAGE4) == {"_J2", "_R2"}
        for want in lb.STAGE4.values():
            assert set(want) == {12, 18, 24}

    def test_needs_runs_or_verify(self):
        with pytest.raises(SystemExit):
            lb.main([])


class TestPairedCompare:
    """The paired test is what the preregistration commits to."""

    @staticmethod
    def two_runs(delta=0.0, n=6):
        """Two runs over the SAME pairs; run B's lens map is shifted toward
        the causal map by `delta`."""
        rows_a, rows_b = [], []
        for i in range(n):
            c = RNG.normal(size=100)
            noise = RNG.normal(size=100)
            a = noise
            b = delta * c + (1 - delta) * noise
            base = {"layer": 18, "target": f"img{i}", "causal": c,
                    "taylor": c.copy()}
            rows_a.append({**base, "lens_delta": a})
            rows_b.append({**base, "lens_delta": b})
        return pd.DataFrame(rows_a), pd.DataFrame(rows_b)

    def test_identical_runs_show_no_difference(self):
        a, _ = self.two_runs()
        out = lb.paired_compare(a, a, "lens_delta")
        assert out.delta.iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert out.n_pairs.iloc[0] == 6

    def test_a_real_improvement_is_detected(self):
        a, b = self.two_runs(delta=0.6)
        out = lb.paired_compare(a, b, "lens_delta").iloc[0]
        assert out.delta > 0
        assert out.lo > 0, "a large real gain must exclude zero"
        assert out.n_up == 6

    def test_pairs_are_matched_by_target_not_by_order(self):
        a, b = self.two_runs(delta=0.6)
        shuffled = b.iloc[::-1].reset_index(drop=True)
        straight = lb.paired_compare(a, b, "lens_delta").delta.iloc[0]
        reversed_ = lb.paired_compare(a, shuffled, "lens_delta").delta.iloc[0]
        assert straight == pytest.approx(reversed_)

    def test_only_shared_pairs_are_compared(self):
        a, b = self.two_runs(delta=0.5, n=6)
        out = lb.paired_compare(a, b.iloc[:4], "lens_delta")
        assert out.n_pairs.iloc[0] == 4

    def test_unfitted_layer_is_not_comparable(self):
        a, b = self.two_runs()
        a["lens_delta"] = None
        b["lens_delta"] = None
        out = lb.paired_compare(a, b, "lens_delta").iloc[0]
        assert np.isnan(out.delta)
        assert out.n_pairs == 0

    def test_is_deterministic(self):
        a, b = self.two_runs(delta=0.4)
        one = lb.paired_compare(a, b, "lens_delta")
        two = lb.paired_compare(a, b, "lens_delta")
        assert one.lo.iloc[0] == two.lo.iloc[0]

    def test_per_pair_r_returns_one_row_per_pair(self):
        a, _ = self.two_runs(n=5)
        out = lb.per_pair_r(a, "taylor")
        assert len(out) == 5
        assert out.r.iloc[0] == pytest.approx(1.0)
