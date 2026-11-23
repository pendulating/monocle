"""The mask search must blend, keep the graph, and expand cells to slots."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from research.monocle.mask_search import MaskBlender  # noqa: E402


def _h(b=1, n=8, d=4):
    return torch.arange(b * n * d, dtype=torch.float32).reshape(b, n, d)


class TestMaskBlender:
    def test_a_mask_of_ones_changes_nothing(self):
        h = _h()
        slots = [2, 3, 4, 5]
        out = MaskBlender(slots, torch.ones(4))(None, None, h)
        assert torch.allclose(out, h)

    def test_a_mask_of_zeros_gives_the_mean_patch(self):
        h = _h()
        slots = [2, 3, 4, 5]
        out = MaskBlender(slots, torch.zeros(4))(None, None, h)
        base = h[:, slots, :].mean(dim=1, keepdim=True)
        assert torch.allclose(out[:, slots, :], base.expand(-1, 4, -1))

    def test_it_leaves_the_other_positions_alone(self):
        h = _h()
        slots = [2, 3]
        out = MaskBlender(slots, torch.zeros(2))(None, None, h)
        keep = [0, 1, 4, 5, 6, 7]
        assert torch.allclose(out[:, keep, :], h[:, keep, :])

    def test_it_does_not_edit_the_input_in_place(self):
        h = _h()
        before = h.clone()
        MaskBlender([2, 3], torch.zeros(2))(None, None, h)
        assert torch.allclose(h, before)

    def test_a_tuple_output_stays_a_tuple(self):
        h = _h()
        extra = torch.zeros(3)
        out = MaskBlender([1], torch.ones(1))(None, None, (h, extra))
        assert isinstance(out, tuple) and len(out) == 2
        assert torch.allclose(out[1], extra)

    def test_the_gradient_reaches_the_mask(self):
        h = _h()
        w = torch.full((4,), 0.5, requires_grad=True)
        out = MaskBlender([2, 3, 4, 5], w)(None, None, h)
        out.sum().backward()
        assert w.grad is not None
        assert torch.isfinite(w.grad).all()
        # Position 2 is far from the mean, so its gradient is not zero.
        assert w.grad.abs().max() > 0

    def test_half_a_mask_is_the_midpoint(self):
        h = _h()
        slots = [2, 3, 4, 5]
        out = MaskBlender(slots, torch.full((4,), 0.5))(None, None, h)
        base = h[:, slots, :].mean(dim=1, keepdim=True)
        want = 0.5 * h[:, slots, :] + 0.5 * base
        assert torch.allclose(out[:, slots, :], want)


class TestCellToSlot:
    """1 mask value must cover every slot of its block."""

    def test_a_group_shares_one_value(self):
        from research.monocle.patch_map_causal import patch_groups
        groups = patch_groups(16, 2)
        n_cell, n_slot = len(groups), 16
        m = torch.zeros(n_cell, n_slot)
        for ci, g in enumerate(groups):
            for p in g:
                m[ci, p] = 1.0
        # Every slot belongs to exactly 1 cell.
        assert torch.allclose(m.sum(dim=0), torch.ones(n_slot))
        cell = torch.arange(n_cell, dtype=torch.float32)
        w = cell @ m
        for ci, g in enumerate(groups):
            for p in g:
                assert float(w[p]) == pytest.approx(float(ci))


class TestScoreSign:
    """The score must carry the same sign as the measured map."""

    def test_a_cell_cut_to_make_c_fall_scores_positive(self):
        # `down` removes the cell (mask 0); `up` keeps it (mask 1).
        m_dn = np.array([0.0, 1.0])
        m_up = np.array([1.0, 1.0])
        score = m_up - m_dn
        assert score[0] > 0 and score[1] == 0


class TestAttnMasker:
    """The soft attention mask must hide a key and keep the graph."""

    def _mask(self, dims):
        import torch
        return torch.zeros(dims)

    def test_a_weight_of_one_adds_nothing(self):
        import torch
        from research.monocle.mask_search import AttnMasker
        m = self._mask((1, 4, 6, 6))
        _, kw = AttnMasker([2, 3], torch.ones(2))(None, (), {"attention_mask": m})
        assert torch.allclose(kw["attention_mask"], m)

    def test_a_small_weight_drives_the_key_down(self):
        import torch
        from research.monocle.mask_search import AttnMasker
        m = self._mask((1, 4, 6, 6))
        _, kw = AttnMasker([2], torch.tensor([1e-6]))(
            None, (), {"attention_mask": m})
        out = kw["attention_mask"]
        assert float(out[..., :, 2].max()) < -10.0
        keep = [0, 1, 3, 4, 5]
        assert torch.allclose(out[..., :, keep], m[..., :, keep])

    def test_a_three_dim_mask_works(self):
        import torch
        from research.monocle.mask_search import AttnMasker
        m = self._mask((1, 6, 6))
        _, kw = AttnMasker([1], torch.tensor([1e-6]))(
            None, (), {"attention_mask": m})
        assert float(kw["attention_mask"][..., :, 1].max()) < -10.0

    def test_the_gradient_reaches_the_weight(self):
        import torch
        from research.monocle.mask_search import AttnMasker
        w = torch.full((2,), 0.5, requires_grad=True)
        m = self._mask((1, 2, 4, 4))
        _, kw = AttnMasker([1, 2], w)(None, (), {"attention_mask": m})
        kw["attention_mask"].sum().backward()
        assert w.grad is not None and torch.isfinite(w.grad).all()
        assert w.grad.abs().min() > 0

    def test_a_boolean_mask_is_refused(self):
        import torch
        from research.monocle.mask_search import AttnMasker
        m = torch.zeros((1, 2, 4, 4), dtype=torch.bool)
        with pytest.raises(RuntimeError, match="boolean"):
            AttnMasker([1], torch.ones(1))(None, (), {"attention_mask": m})

    def test_a_missing_mask_is_refused(self):
        import torch
        from research.monocle.mask_search import AttnMasker
        with pytest.raises(RuntimeError, match="eager"):
            AttnMasker([1], torch.ones(1))(None, (), {})
