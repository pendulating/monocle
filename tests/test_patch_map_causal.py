"""The causal patch map. See monocle/patch_map_causal.py."""
import numpy as np
import pandas as pd
import pytest
import torch

from monocle import patch_map_causal as pm


class TestBatchPatcher:
    def out(self, b=3, seq=6, d=4, tuple_form=False):
        h = torch.zeros(b, seq, d)
        return (h, "extra") if tuple_form else h

    def test_each_row_gets_its_own_position(self):
        vals = torch.stack([torch.full((4,), float(i + 1)) for i in range(3)])
        got = pm.BatchPatcher([1, 2, 3], vals)(None, None, self.out())
        assert torch.equal(got[0, 1], torch.full((4,), 1.0))
        assert torch.equal(got[1, 2], torch.full((4,), 2.0))
        assert torch.equal(got[2, 3], torch.full((4,), 3.0))

    def test_leaves_other_positions_alone(self):
        vals = torch.ones(3, 4)
        got = pm.BatchPatcher([1, 2, 3], vals)(None, None, self.out())
        assert torch.equal(got[0, 2], torch.zeros(4))
        assert torch.equal(got[1, 1], torch.zeros(4))

    def test_does_not_mutate_the_input(self):
        h = torch.zeros(3, 6, 4)
        pm.BatchPatcher([1, 2, 3], torch.ones(3, 4))(None, None, h)
        assert torch.equal(h, torch.zeros(3, 6, 4))

    def test_keeps_the_rest_of_the_tuple(self):
        got = pm.BatchPatcher([0, 1, 2], torch.ones(3, 4))(
            None, None, self.out(tuple_form=True))
        assert isinstance(got, tuple) and got[1] == "extra"

    def test_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="positions but"):
            pm.BatchPatcher([1, 2], torch.ones(3, 4))


class TestGroupPatcher:
    def out(self, b=2, seq=8, d=4):
        return torch.zeros(b, seq, d)

    def test_a_row_gets_its_whole_group(self):
        vals = [torch.full((2, 4), 1.0), torch.full((2, 4), 2.0)]
        got = pm.BatchPatcher([[1, 2], [4, 5]], vals)(None, None, self.out())
        assert torch.equal(got[0, 1], torch.full((4,), 1.0))
        assert torch.equal(got[0, 2], torch.full((4,), 1.0))
        assert torch.equal(got[1, 4], torch.full((4,), 2.0))

    def test_each_position_keeps_its_own_donor_vector(self):
        """A group is not 1 vector written 4 times."""
        vals = [torch.tensor([[1.0] * 4, [9.0] * 4])]
        got = pm.BatchPatcher([[0, 3]], vals)(None, None, self.out(b=1))
        assert got[0, 0, 0] == 1.0 and got[0, 3, 0] == 9.0

    def test_leaves_the_rest_alone(self):
        got = pm.BatchPatcher([[1, 2]], [torch.ones(2, 4)])(
            None, None, self.out(b=1))
        assert torch.equal(got[0, 0], torch.zeros(4))

    def test_rejects_a_group_short_of_values(self):
        with pytest.raises(ValueError, match="own donor vector"):
            pm.BatchPatcher([[1, 2, 3]], [torch.ones(2, 4)])

    def test_rejects_a_count_mismatch(self):
        with pytest.raises(ValueError, match="positions but"):
            pm.BatchPatcher([[1], [2]], [torch.ones(1, 4)])


class TestPatchGroups:
    def test_a_16x16_grid_gives_64_groups_of_4(self):
        g = pm.patch_groups(256, 2)
        assert len(g) == 64 and all(len(x) == 4 for x in g)

    def test_every_patch_is_used_once(self):
        flat = [i for g in pm.patch_groups(256, 2) for i in g]
        assert sorted(flat) == list(range(256))

    def test_a_group_is_a_square_block_not_a_row(self):
        """[0,1,16,17] is a 2x2 block. [0,1,2,3] would be a strip."""
        assert pm.patch_groups(256, 2)[0] == [0, 1, 16, 17]

    def test_the_order_stays_row_major(self):
        """So grid_of reads the result with no reshuffle."""
        g = pm.patch_groups(256, 2)
        assert g[1] == [2, 3, 18, 19] and g[8] == [32, 33, 48, 49]

    def test_group_1_is_the_whole_grid(self):
        assert pm.patch_groups(256, 16) == [list(range(256))]

    def test_rejects_a_non_square_count(self):
        with pytest.raises(ValueError, match="not square"):
            pm.patch_groups(255, 2)

    def test_rejects_a_size_that_does_not_divide(self):
        with pytest.raises(ValueError, match="does not divide"):
            pm.patch_groups(256, 3)


class TestAgreement:
    def test_identical_maps_agree_perfectly(self):
        x = np.random.default_rng(0).normal(size=64)
        a = pm.agreement(x, x, k=10)
        assert a["pearson"] == pytest.approx(1.0)
        assert a["spearman"] == pytest.approx(1.0)
        assert a["sign_agree"] == pytest.approx(1.0)
        assert a["top_overlap"] == pytest.approx(1.0)

    def test_negated_map_anticorrelates(self):
        x = np.random.default_rng(1).normal(size=64)
        a = pm.agreement(x, -x, k=10)
        assert a["pearson"] == pytest.approx(-1.0)
        assert a["sign_agree"] == pytest.approx(0.0)
        assert a["top_overlap"] == pytest.approx(1.0)   # same magnitudes

    def test_independent_maps_do_not_agree(self):
        rng = np.random.default_rng(2)
        a = pm.agreement(rng.normal(size=256), rng.normal(size=256))
        assert abs(a["pearson"]) < 0.3

    def test_a_flat_map_returns_nan_not_a_number(self):
        a = pm.agreement(np.zeros(64), np.arange(64.0))
        assert np.isnan(a["pearson"])

    def test_ignores_non_finite_entries(self):
        x = np.arange(64.0)
        y = x.copy(); y[0] = np.nan
        assert pm.agreement(x, y)["pearson"] == pytest.approx(1.0)


class TestTaylorMap:
    """<g, h_donor - h> is the fair test of the gradient itself."""

    def test_is_the_dot_of_the_gradient_with_the_move(self):
        g = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        h = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        donor = torch.tensor([[3.0, 9.0], [9.0, 4.0]])
        got = pm.taylor_map(g, h, donor)
        assert got == pytest.approx([3.0, 8.0])

    def test_subtracts_the_target_not_the_mean(self):
        """It must use h_donor - h, NOT the centered activation."""
        g = torch.ones(2, 2)
        h = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        donor = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
        assert pm.taylor_map(g, h, donor) == pytest.approx([2.0, 2.0])

    def test_is_zero_when_the_donor_matches(self):
        g = torch.randn(4, 3)
        h = torch.randn(4, 3)
        assert pm.taylor_map(g, h, h) == pytest.approx([0.0] * 4)

    def test_differs_from_the_production_map(self):
        """The 2 maps dot the gradient with DIFFERENT vectors."""
        g = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        h = torch.tensor([[2.0, 0.0], [0.0, 6.0]])
        donor = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
        assert not np.allclose(pm.taylor_map(g, h, donor), pm.cheap_map(g, h))


class TestCheapMap:
    def test_centers_before_the_dot_product(self):
        g = torch.ones(2, 2)
        h = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        # mean is [2,2]; centered rows are [-1,-1] and [+1,+1]
        assert pm.cheap_map(g, h) == pytest.approx([-2.0, 2.0])


class TestSummarize:
    def frame(self):
        return pd.DataFrame([
            {"layer": 20, "pearson": 0.8, "spearman": 0.7, "sign_agree": 0.9,
             "top_overlap": 0.5, "all_recovery": 0.9, "causal_max_abs": 1.0,
             "causal_top5pct": 0.4, "taylor_pearson": 0.9,
             "taylor_spearman": 0.8, "taylor_sign_agree": 0.9,
             "taylor_top_overlap": 0.6, "cheap_vs_taylor_pearson": 0.1},
            {"layer": 20, "pearson": 0.6, "spearman": 0.5, "sign_agree": 0.7,
             "top_overlap": 0.3, "all_recovery": 0.7, "causal_max_abs": 2.0,
             "causal_top5pct": 0.6, "taylor_pearson": 0.7,
             "taylor_spearman": 0.6, "taylor_sign_agree": 0.7,
             "taylor_top_overlap": 0.4, "cheap_vs_taylor_pearson": 0.3},
        ])

    def test_averages_within_a_layer(self):
        s = pm.summarize(self.frame()).set_index("layer")
        assert s.loc[20, "pearson"] == pytest.approx(0.7)
        assert s.loc[20, "t_pearson"] == pytest.approx(0.8)
        assert s.loc[20, "n"] == 2


class TestContract:
    def test_default_layers_sit_in_the_causal_window(self):
        """Patches hold the judgment to about L24 and are spent by L30."""
        assert max(pm.DEFAULT_LAYERS) <= 28
        assert 20 in pm.DEFAULT_LAYERS

    def test_window_floor_is_set(self):
        assert 0 < pm.WINDOW_FLOOR < 1


# ---------------------------------------------------------------------------
# The lens arms. See monocle/patch_map_causal.py lens_maps.
# ---------------------------------------------------------------------------
class FakeLens:
    """Transport is a scale, so a readout is easy to predict by hand."""

    def __init__(self, scale=2.0, layers=(12, 24)):
        self.jacobians = {l: None for l in layers}
        self.source_layers = list(layers)
        self.scale = scale

    def transport(self, h, layer):
        return h * self.scale


class FakeUnembed:
    """`unembed` is the identity on the first 2 columns."""

    @staticmethod
    def unembed(x):
        return x


class TestLensMaps:
    def test_the_readout_is_pos_minus_neg(self):
        h = torch.tensor([[3.0, 1.0], [5.0, 4.0]])
        got = pm.lens_readout(FakeLens(2.0), FakeUnembed, h, 12, 0, 1)
        assert got.tolist() == [4.0, 2.0]

    def test_own_scores_each_patch_alone(self):
        h = torch.tensor([[3.0, 1.0]])
        donor = torch.tensor([[9.0, 1.0]])
        own, _ = pm.lens_maps(FakeLens(1.0), FakeUnembed, h, donor, 12, 0, 1)
        assert own.tolist() == [2.0]

    def test_delta_is_the_swap(self):
        h = torch.tensor([[3.0, 1.0]])
        donor = torch.tensor([[9.0, 1.0]])
        _, delta = pm.lens_maps(
            FakeLens(1.0), FakeUnembed, h, donor, 12, 0, 1)
        assert delta.tolist() == [6.0]

    def test_delta_is_zero_for_an_identical_donor(self):
        h = torch.randn(7, 2)
        _, delta = pm.lens_maps(FakeLens(3.0), FakeUnembed, h, h, 12, 0, 1)
        assert np.allclose(delta, 0.0)

    def test_one_score_for_each_patch(self):
        h, donor = torch.randn(11, 2), torch.randn(11, 2)
        own, delta = pm.lens_maps(
            FakeLens(), FakeUnembed, h, donor, 24, 0, 1)
        assert own.shape == (11,) and delta.shape == (11,)


class TestBaselineOffset:
    """Why `batched_baseline` exists. See its docstring."""

    def test_pearson_does_not_see_an_offset(self):
        rng = np.random.default_rng(0)
        c, m = rng.normal(size=64), rng.normal(size=64)
        a = pm.agreement(c, m)
        b = pm.agreement(c + 0.7, m)
        assert a["pearson"] == pytest.approx(b["pearson"])

    def test_sign_agreement_does_see_an_offset(self):
        rng = np.random.default_rng(0)
        c = rng.normal(scale=0.3, size=256)
        m = c + rng.normal(scale=0.05, size=256)
        clean = pm.agreement(c, m)["sign_agree"]
        offset = pm.agreement(c + 0.5, m)["sign_agree"]
        assert clean > 0.9
        assert offset < clean - 0.2
