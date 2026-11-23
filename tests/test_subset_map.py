"""The subset map must recover an additive truth and must say when it cannot."""
from __future__ import annotations

import numpy as np
import pytest

from research.monocle.subset_map import draw_sets, fit_values, holdout_r2  # noqa: E402


class TestDrawSets:
    def test_it_gives_the_asked_count(self):
        rng = np.random.default_rng(0)
        s = draw_sets(64, 50, (1, 2, 4), rng)
        assert len(s) == 50

    def test_the_sizes_walk_the_grid(self):
        rng = np.random.default_rng(0)
        s = draw_sets(64, 12, (1, 2, 4), rng)
        assert [len(x) for x in s] == [1, 2, 4] * 4

    def test_a_set_holds_no_repeat(self):
        rng = np.random.default_rng(0)
        for s in draw_sets(16, 40, (4, 8), rng):
            assert len(set(s)) == len(s)

    def test_every_cell_is_in_range(self):
        rng = np.random.default_rng(1)
        for s in draw_sets(16, 40, (4, 8), rng):
            assert all(0 <= c < 16 for c in s)


class TestFitValues:
    def test_it_recovers_an_additive_truth(self):
        rng = np.random.default_rng(7)
        n = 16
        true = rng.normal(size=n)
        sets = draw_sets(n, 600, (1, 2, 4, 8), rng)
        dc = np.array([true[s].sum() for s in sets])
        w = fit_values(sets, dc, n, alpha=1e-6)
        assert np.corrcoef(w, true)[0, 1] > 0.99

    def test_an_intercept_does_not_break_it(self):
        rng = np.random.default_rng(8)
        n = 16
        true = rng.normal(size=n)
        sets = draw_sets(n, 600, (1, 2, 4, 8), rng)
        dc = np.array([true[s].sum() + 3.5 for s in sets])
        w = fit_values(sets, dc, n, alpha=1e-6)
        assert np.corrcoef(w, true)[0, 1] > 0.99

    def test_the_shape_is_one_value_for_each_cell(self):
        rng = np.random.default_rng(9)
        sets = draw_sets(8, 60, (1, 2), rng)
        w = fit_values(sets, np.zeros(60), 8, alpha=1.0)
        assert w.shape == (8,)


class TestHoldoutR2:
    def test_an_additive_truth_scores_near_one(self):
        rng = np.random.default_rng(11)
        n = 16
        true = rng.normal(size=n)
        sets = draw_sets(n, 800, (1, 2, 4, 8), rng)
        dc = np.array([true[s].sum() for s in sets])
        assert holdout_r2(sets, dc, n, 1e-6, rng) > 0.95

    def test_a_saturated_truth_scores_low(self):
        # A judgment that saturates: the set effect is a squashed sum, so no
        # additive map can follow it.
        rng = np.random.default_rng(12)
        n = 16
        true = rng.normal(size=n)
        sets = draw_sets(n, 800, (1, 2, 4, 8), rng)
        dc = np.array([np.tanh(3.0 * true[s].sum()) for s in sets])
        r2 = holdout_r2(sets, dc, n, 1e-6, rng)
        assert r2 < 0.95

    def test_pure_noise_scores_at_or_below_zero(self):
        rng = np.random.default_rng(13)
        sets = draw_sets(16, 400, (1, 2, 4, 8), rng)
        dc = rng.normal(size=400)
        assert holdout_r2(sets, dc, 16, 1.0, rng) < 0.3


class TestSign:
    def test_the_contribution_negates_the_weight(self):
        # A cell that holds up the positive label makes `c` FALL when it goes,
        # so its fitted weight is negative and its contribution is positive.
        rng = np.random.default_rng(14)
        n = 8
        sets = draw_sets(n, 400, (1, 2), rng)
        true = np.zeros(n)
        true[3] = -2.0                      # deleting cell 3 lowers c
        dc = np.array([true[s].sum() for s in sets])
        w = fit_values(sets, dc, n, alpha=1e-6)
        values = -w
        assert values[3] == pytest.approx(2.0, abs=0.05)
        assert int(np.argmax(values)) == 3
