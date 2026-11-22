"""The residual-stream swap. See research/monocle/patch_swap.py."""
import numpy as np
import pandas as pd
import pytest
import torch

from research.monocle import patch_swap as ps


class TestPatcher:
    def out(self, tuple_form):
        h = torch.zeros(1, 6, 4)
        return (h, "extra") if tuple_form else h

    @pytest.mark.parametrize("tuple_form", [False, True])
    def test_overwrites_only_the_named_positions(self, tuple_form):
        p = ps.Patcher([2], torch.ones(1, 4))
        got = p(None, None, self.out(tuple_form))
        h = got[0] if tuple_form else got
        assert torch.equal(h[0, 2], torch.ones(4))
        assert torch.equal(h[0, 1], torch.zeros(4))
        assert torch.equal(h[0, 3], torch.zeros(4))

    def test_keeps_the_rest_of_the_tuple(self):
        got = ps.Patcher([0], torch.ones(1, 4))(None, None, self.out(True))
        assert isinstance(got, tuple) and got[1] == "extra"

    def test_does_not_mutate_the_input_tensor(self):
        h = torch.zeros(1, 6, 4)
        ps.Patcher([1], torch.ones(1, 4))(None, None, h)
        assert torch.equal(h, torch.zeros(1, 6, 4))

    def test_writes_every_named_position(self):
        got = ps.Patcher([0, 5], torch.ones(2, 4))(None, None, self.out(False))
        assert torch.equal(got[0, 0], torch.ones(4))
        assert torch.equal(got[0, 5], torch.ones(4))


class TestConditions:
    def block(self):
        return list(range(8, 264))

    def arms(self):
        return ps.conditions(self.block(), np.random.default_rng(0), 300)

    def test_last_is_the_final_patch(self):
        assert self.arms()["last"] == [263]

    def test_first_is_the_opening_patch(self):
        assert self.arms()["first"] == [8]

    def test_mid_is_neither_endpoint(self):
        m = self.arms()["mid"]
        assert len(m) == 1 and m[0] not in (8, 263)

    def test_random_avoids_both_reported_endpoints(self):
        """A random draw that lands on `last` would confound the control."""
        for s in range(30):
            a = ps.conditions(self.block(), np.random.default_rng(s), 300)
            assert a["random"][0] not in (8, 263)
            assert 8 not in a["random_8"] and 263 not in a["random_8"]

    def test_random_8_is_eight_distinct_positions(self):
        r = self.arms()["random_8"]
        assert len(r) == 8 and len(set(r)) == 8

    def test_all_covers_the_whole_block(self):
        assert self.arms()["all"] == self.block()

    def test_has_a_positive_control_on_the_answer_position(self):
        """A null swap cannot tell a working hook from a dead one."""
        assert self.arms()["answer"] == [300]

    def test_the_control_is_outside_the_image_block(self):
        assert self.arms()["answer"][0] not in self.block()

    def test_single_patch_arms_move_one_position(self):
        a = self.arms()
        for arm in ("last", "mid", "first", "random"):
            assert len(a[arm]) == 1


class TestSummarize:
    def frame(self):
        rows = []
        for arm, rec in (("last", 0.8), ("mid", 0.1)):
            for i in range(4):
                rows.append({"layer": 46, "arm": arm, "recovery": rec,
                             "delta": rec * 2.0})
        return pd.DataFrame(rows)

    def test_reports_mean_recovery_per_arm(self):
        s = ps.summarize(self.frame()).set_index("arm")
        assert s.loc["last", "mean_recovery"] == pytest.approx(0.8)
        assert s.loc["mid", "mean_recovery"] == pytest.approx(0.1)

    def test_abs_delta_ignores_sign(self):
        f = self.frame()
        f.loc[f["arm"] == "mid", "delta"] = -0.2
        s = ps.summarize(f).set_index("arm")
        assert s.loc["mid", "mean_abs_delta"] == pytest.approx(0.2)

    def test_counts_the_pairs(self):
        assert set(ps.summarize(self.frame())["n"]) == {4}


class TestContract:
    def test_min_gap_is_positive(self):
        """recovery divides by the donor-target gap."""
        assert ps.MIN_GAP > 0

    def test_default_layers_cover_peak_and_collapse(self):
        assert 43 in ps.DEFAULT_LAYERS and 46 in ps.DEFAULT_LAYERS

    def test_default_layers_include_an_early_depth(self):
        """A late-only sweep cannot separate 'absorbed already' from 'no-op'."""
        assert min(ps.DEFAULT_LAYERS) <= 20

    def test_control_floor_is_demanding(self):
        assert ps.CONTROL_FLOOR >= 0.5

    def test_deepest_default_layer_leaves_little_computation(self):
        """The control is judged there, so it must be near the top."""
        assert max(ps.DEFAULT_LAYERS) >= 46
