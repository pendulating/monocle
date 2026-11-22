"""Which word of the question carries it. See research/monocle/question_tokens.py."""
import numpy as np
import pytest
import torch

from research.monocle import question_tokens as qt
from research.monocle.attn_route import BOI_TOKEN_ID, EOI_TOKEN_ID

IMG = 258880


def ids(n_before=2, n_patches=4, n_after=3):
    return torch.tensor([7] * n_before + [BOI_TOKEN_ID] + [IMG] * n_patches
                        + [EOI_TOKEN_ID] + [8] * n_after + [9])


class TestQuestionPositions:
    def test_it_finds_the_words_after_the_image(self):
        x = ids()
        q, _ = qt.question_positions(x, [3, 4, 5, 6], int(x.shape[0]) - 1)
        assert q == [8, 9, 10]

    def test_the_answer_is_not_a_question_token(self):
        x = ids()
        last = int(x.shape[0]) - 1
        q, _ = qt.question_positions(x, [3, 4, 5, 6], last)
        assert last not in q

    def test_the_null_control_precedes_the_image(self):
        x = ids()
        _, n = qt.question_positions(x, [3, 4, 5, 6], int(x.shape[0]) - 1)
        assert n == [0, 1]

    def test_the_delimiters_are_in_neither(self):
        x = ids()
        q, n = qt.question_positions(x, [3, 4, 5, 6], int(x.shape[0]) - 1)
        assert 2 not in q + n and 7 not in q + n


class TestRecovery:
    def test_no_move_is_zero(self):
        assert qt.recovery(5.0, 5.0, 9.0) == pytest.approx(0.0)

    def test_all_the_way_is_one(self):
        assert qt.recovery(9.0, 5.0, 9.0) == pytest.approx(1.0)

    def test_half_way_is_a_half(self):
        assert qt.recovery(7.0, 5.0, 9.0) == pytest.approx(0.5)

    def test_it_works_when_the_donor_is_lower(self):
        assert qt.recovery(3.0, 5.0, 1.0) == pytest.approx(0.5)

    def test_a_pair_with_no_gap_gives_no_number(self):
        """Dividing by a near-zero gap would invent a huge recovery."""
        assert np.isnan(qt.recovery(5.0, 5.0, 5.0))

    def test_overshoot_is_reported_not_clipped(self):
        assert qt.recovery(13.0, 5.0, 9.0) == pytest.approx(2.0)


class TestGuards:
    def test_a_close_pair_is_skipped_not_divided_by(self):
        """The first version read a null control of 103.6 for want of this."""
        assert qt.MIN_GAP >= 1.0

    def test_the_null_floor_is_tight(self):
        """Every row now has a self row in its own batch, so it can be."""
        assert qt.NULL_FLOOR <= 1e-3

    def test_each_set_is_paired_with_a_self_swap_of_itself(self):
        """A self row at OTHER positions does not cancel: the recorded values
        come from a batch-1 forward and are written into a batched one, so
        writing any recorded value is already a small change."""
        import inspect
        src = inspect.getsource(qt.swap_positions)
        assert "* 2" in src
        assert "own_h[g, :] for g in block" in src
        assert "while len(block) < batch" in src

    def test_recovery_still_refuses_a_zero_gap(self):
        assert np.isnan(qt.recovery(5.0, 5.0, 5.0))


class TestContract:
    def test_the_band_covers_the_handover(self):
        """The relay put the question tokens at blocks 25 to 32."""
        assert min(qt.DEFAULT_LAYERS) <= 25
        assert max(qt.DEFAULT_LAYERS) >= 32

    def test_the_band_floor_is_a_real_gate(self):
        assert 0.0 < qt.BAND_FLOOR < 1.0
