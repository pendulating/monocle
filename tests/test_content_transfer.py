"""The content transfer test. See monocle/content_transfer.py."""
import numpy as np
import pandas as pd
import pytest

from monocle import content_transfer as ct


class TestTransferSets:
    def v(self):
        return np.linspace(-1.0, 1.0, 64)

    def test_the_safe_arm_takes_the_highest_cells(self):
        s = ct.transfer_sets(self.v(), 3, np.random.default_rng(0), 2)
        assert set(s["transfer_safe"][0]) == {61, 62, 63}

    def test_the_danger_arm_takes_the_lowest_cells(self):
        s = ct.transfer_sets(self.v(), 3, np.random.default_rng(0), 2)
        assert set(s["transfer_danger"][0]) == {0, 1, 2}

    def test_the_two_targeted_arms_are_matched_in_size(self):
        """A gap between them cannot come from how much changed."""
        s = ct.transfer_sets(self.v(), 8, np.random.default_rng(0), 3)
        assert len(s["transfer_safe"][0]) == len(s["transfer_danger"][0]) == 8

    def test_the_random_arm_has_several_draws(self):
        s = ct.transfer_sets(self.v(), 4, np.random.default_rng(0), 5)
        assert len(s["transfer_random"]) == 5


class TestSummarize:
    def frame(self, safe, danger, rnd, ok=True):
        base = {"target_id": "b", "source_id": "a", "k": 8,
                "c_target": 10.0, "c_source": 3.0, "target_worth_ok": ok,
                "dispersion": 4.0}
        rows = [{**base, "arm": "transfer_safe", "dc": safe},
                {**base, "arm": "transfer_danger", "dc": danger}]
        rows += [{**base, "arm": "transfer_random", "dc": r} for r in rnd]
        return pd.DataFrame(rows)

    def test_the_sign_runs_the_other_way_from_a_deletion(self):
        """Putting safe content IN must raise `c`, not lower it."""
        s = ct.summarize(self.frame(+2.0, -1.0, [0.0, 0.1]))
        assert bool(s.safe_sign_ok.iloc[0]) and bool(s.danger_sign_ok.iloc[0])

    def test_separation_is_safe_minus_danger(self):
        s = ct.summarize(self.frame(+2.0, -1.0, [0.0]))
        assert s.separation.iloc[0] == pytest.approx(3.0)

    def test_a_backwards_transfer_fails_both_signs(self):
        s = ct.summarize(self.frame(-2.0, +1.0, [0.0]))
        assert not bool(s.safe_sign_ok.iloc[0])
        assert not bool(s.danger_sign_ok.iloc[0])

    def test_z_uses_the_random_spread(self):
        s = ct.summarize(self.frame(+1.0, -1.0, [0.0, 1.0, -1.0]))
        assert s.separation_z.iloc[0] == pytest.approx(2.0)

    def test_a_flat_random_arm_gives_no_z(self):
        s = ct.summarize(self.frame(1.0, -1.0, [0.0, 0.0, 0.0]))
        assert np.isnan(s.separation_z.iloc[0])

    def test_it_keeps_both_image_names(self):
        """The map comes from the source and the test runs on the target."""
        s = ct.summarize(self.frame(1.0, -1.0, [0.0, 0.5]))
        assert s.source_id.iloc[0] == "a" and s.target_id.iloc[0] == "b"


class TestContract:
    def test_the_default_cells_are_big_enough_to_hold_a_thing(self):
        assert ct.DEFAULT_GROUP >= 2

    def test_the_smallest_k_moves_a_real_area(self):
        assert min(ct.DEFAULT_KS) >= 4
