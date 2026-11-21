"""The directed deletion test. See monocle/deletion_curve.py."""
import numpy as np
import pandas as pd
import pytest

from monocle import deletion_curve as dc


class TestRankCells:
    def test_the_positive_order_starts_at_the_highest(self):
        v = np.array([0.0, 5.0, -3.0, 1.0])
        safe, _ = dc.rank_cells(v)
        assert safe[0] == 1 and safe[1] == 3

    def test_the_negative_order_starts_at_the_lowest(self):
        v = np.array([0.0, 5.0, -3.0, 1.0])
        _, danger = dc.rank_cells(v)
        assert danger[0] == 2

    def test_the_two_orders_are_opposite(self):
        v = np.random.default_rng(0).normal(size=32)
        safe, danger = dc.rank_cells(v)
        assert list(safe) == list(danger[::-1])

    def test_it_does_not_disturb_the_map(self):
        v = np.array([0.0, 5.0, -3.0, 1.0])
        dc.rank_cells(v)
        assert v.tolist() == [0.0, 5.0, -3.0, 1.0]


class TestArmSets:
    def v(self):
        return np.linspace(-1.0, 1.0, 64)

    def test_every_arm_holds_k_cells(self):
        arms = dc.arm_sets(self.v(), 8, np.random.default_rng(0))
        for draws in arms.values():
            for d in draws:
                assert len(d) == 8

    def test_the_random_arm_has_several_draws(self):
        arms = dc.arm_sets(self.v(), 4, np.random.default_rng(0), n_random=5)
        assert len(arms["random"]) == 5
        assert len(arms["top_safe"]) == 1

    def test_the_random_draws_differ(self):
        """1 draw gives no spread, and the spread is the yardstick."""
        arms = dc.arm_sets(self.v(), 8, np.random.default_rng(0), n_random=6)
        sets = {tuple(sorted(d)) for d in arms["random"]}
        assert len(sets) > 1

    def test_a_random_draw_never_repeats_a_cell(self):
        arms = dc.arm_sets(self.v(), 16, np.random.default_rng(1))
        for d in arms["random"]:
            assert len(set(d)) == 16

    def test_the_two_targeted_arms_are_opposite_ends(self):
        v = self.v()
        arms = dc.arm_sets(v, 3, np.random.default_rng(0))
        assert set(arms["top_safe"][0]) == {61, 62, 63}
        assert set(arms["top_danger"][0]) == {0, 1, 2}

    def test_rejects_k_over_the_cell_count(self):
        with pytest.raises(ValueError, match="k must be in 1..64"):
            dc.arm_sets(self.v(), 65, np.random.default_rng(0))

    def test_rejects_k_of_zero(self):
        with pytest.raises(ValueError, match="k must be in"):
            dc.arm_sets(self.v(), 0, np.random.default_rng(0))


class TestCellsToPatches:
    def test_a_2x2_cell_gives_4_patches(self):
        blocks = [[0, 1, 16, 17], [2, 3, 18, 19]]
        assert dc.cells_to_patches([0], blocks) == [0, 1, 16, 17]

    def test_two_cells_give_eight_patches(self):
        blocks = [[0, 1, 16, 17], [2, 3, 18, 19]]
        assert len(dc.cells_to_patches([0, 1], blocks)) == 8

    def test_the_result_is_sorted_and_unique(self):
        blocks = [[5, 1], [1, 9]]
        assert dc.cells_to_patches([0, 1], blocks) == [1, 5, 9]


class TestLoadRanking:
    def write(self, tmp_path, cells, n=3, name="maps.parquet"):
        rng = np.random.default_rng(0)
        pd.DataFrame([{"image_id": f"im{i}",
                       "values": list(rng.normal(size=cells))}
                      for i in range(n)]).to_parquet(tmp_path / name)
        return tmp_path / name

    def test_reads_a_map_for_each_image(self, tmp_path):
        out = dc.load_ranking(self.write(tmp_path, 64), 2)
        assert set(out) == {"im0", "im1", "im2"}
        assert out["im0"].size == 64

    def test_refuses_a_grid_that_does_not_match_the_group(self, tmp_path):
        """A 256-cell map cannot rank 64 cells."""
        with pytest.raises(SystemExit, match="different grid"):
            dc.load_ranking(self.write(tmp_path, 256), 2)

    def test_a_fine_map_matches_group_one(self, tmp_path):
        assert dc.load_ranking(self.write(tmp_path, 256), 1)["im0"].size == 256

    def test_refuses_a_frame_with_no_values(self, tmp_path):
        f = tmp_path / "bad.parquet"
        pd.DataFrame([{"image_id": "im0"}]).to_parquet(f)
        with pytest.raises(SystemExit, match="no values column"):
            dc.load_ranking(f, 2)

    def test_refuses_a_frame_with_no_image_id(self, tmp_path):
        f = tmp_path / "bad.parquet"
        pd.DataFrame([{"values": [0.0] * 64}]).to_parquet(f)
        with pytest.raises(SystemExit, match="no image_id column"):
            dc.load_ranking(f, 2)


class TestSummarize:
    def frame(self, safe, danger, rnd, worth=10.0, ok=True):
        rows = [{"image_id": "im1", "k": 4, "arm": "top_safe", "dc": safe,
                 "image_worth": worth, "worth_ok": ok},
                {"image_id": "im1", "k": 4, "arm": "top_danger", "dc": danger,
                 "image_worth": worth, "worth_ok": ok}]
        rows += [{"image_id": "im1", "k": 4, "arm": "random", "dc": r,
                  "image_worth": worth, "worth_ok": ok} for r in rnd]
        return pd.DataFrame(rows)

    def test_separation_is_danger_minus_safe(self):
        s = dc.summarize(self.frame(-2.0, 3.0, [0.0, 0.1, -0.1]))
        assert s.separation.iloc[0] == pytest.approx(5.0)

    def test_the_signs_the_colour_promises(self):
        s = dc.summarize(self.frame(-2.0, 3.0, [0.0]))
        assert bool(s.safe_sign_ok.iloc[0]) and bool(s.danger_sign_ok.iloc[0])

    def test_a_backwards_map_fails_both_signs(self):
        s = dc.summarize(self.frame(+2.0, -3.0, [0.0]))
        assert not bool(s.safe_sign_ok.iloc[0])
        assert not bool(s.danger_sign_ok.iloc[0])

    def test_z_measures_against_the_random_spread(self):
        s = dc.summarize(self.frame(-1.0, 1.0, [0.0, 1.0, -1.0]))
        assert s.separation_z.iloc[0] == pytest.approx(2.0)

    def test_a_flat_random_arm_gives_no_z(self):
        """No spread means no yardstick, so the test must not invent one."""
        s = dc.summarize(self.frame(-1.0, 1.0, [0.0, 0.0, 0.0]))
        assert np.isnan(s.separation_z.iloc[0])

    def test_one_random_draw_gives_no_z(self):
        s = dc.summarize(self.frame(-1.0, 1.0, [0.0]))
        assert np.isnan(s.separation_z.iloc[0])

    def test_keep_only_flips_the_separation(self):
        """Keeping the safe cells must read as a success, not a failure."""
        f = self.frame(+2.0, -1.0, [0.0, 0.1])   # safe high, danger low
        s = dc.summarize(f, keep_only=True)
        assert s.separation.iloc[0] == pytest.approx(3.0)
        assert bool(s.safe_sign_ok.iloc[0])
        assert bool(s.danger_sign_ok.iloc[0])

    def test_the_same_numbers_read_as_failure_under_deletion(self):
        f = self.frame(+2.0, -1.0, [0.0, 0.1])
        s = dc.summarize(f, keep_only=False)
        assert s.separation.iloc[0] == pytest.approx(-3.0)
        assert not bool(s.safe_sign_ok.iloc[0])

    def test_keep_only_signs_are_judged_against_random(self):
        """With most of the picture gone, every arm falls. Order is the test."""
        f = self.frame(-5.0, -9.0, [-7.0, -7.2])
        s = dc.summarize(f, keep_only=True)
        assert bool(s.safe_sign_ok.iloc[0]) and bool(s.danger_sign_ok.iloc[0])

    def test_the_flag_is_recorded(self):
        s = dc.summarize(self.frame(1.0, -1.0, [0.0]), keep_only=True)
        assert bool(s.keep_only.iloc[0])

    def test_a_weak_image_keeps_its_flag(self):
        s = dc.summarize(self.frame(-1.0, 1.0, [0.0, 0.2], ok=False))
        assert not bool(s.worth_ok.iloc[0])
