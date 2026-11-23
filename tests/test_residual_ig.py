"""Integrated gradients in the residual must be complete and must write slots."""
from __future__ import annotations

import pytest
import torch

from research.monocle.residual_ig import SlotWriter  # noqa: E402


class TestSlotWriter:
    def test_it_writes_only_the_slots(self):
        h = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        z = torch.zeros(2, 4)
        out = SlotWriter([1, 3], z)(None, None, h)
        assert torch.allclose(out[:, [1, 3], :], z.unsqueeze(0))
        keep = [0, 2, 4, 5]
        assert torch.allclose(out[:, keep, :], h[:, keep, :])

    def test_it_does_not_edit_in_place(self):
        h = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        before = h.clone()
        SlotWriter([0], torch.ones(1, 4))(None, None, h)
        assert torch.allclose(h, before)

    def test_a_tuple_output_stays_a_tuple(self):
        h = torch.zeros(1, 4, 2)
        extra = torch.ones(3)
        out = SlotWriter([0], torch.ones(1, 2))(None, None, (h, extra))
        assert isinstance(out, tuple) and torch.allclose(out[1], extra)

    def test_the_gradient_reaches_the_written_tensor(self):
        h = torch.zeros(1, 4, 2)
        z = torch.ones(2, 2, requires_grad=True)
        out = SlotWriter([1, 2], z)(None, None, h)
        out.sum().backward()
        assert z.grad is not None and torch.allclose(z.grad, torch.ones(2, 2))


class TestCompleteness:
    """IG must sum to the difference of the 2 ends, for a linear function."""

    def test_a_linear_score_is_complete(self):
        # c(z) = <w, sum of z>. The gradient is constant, so IG is exact.
        torch.manual_seed(0)
        n, d, steps = 8, 5, 16
        w = torch.randn(d)
        h = torch.randn(n, d)
        hbar = h.mean(0, keepdim=True)
        delta = h - hbar
        total = torch.zeros_like(delta)
        for k in range(steps):
            a = (k + 0.5) / steps
            z = (hbar + a * delta).detach().requires_grad_(True)
            c = (z @ w).sum()
            (g,) = torch.autograd.grad(c, z)
            total += g
        ig = (delta * (total / steps)).sum(1)
        c_real = float((h @ w).sum())
        c_base = float((hbar.expand_as(h) @ w).sum())
        assert float(ig.sum()) == pytest.approx(c_real - c_base, abs=1e-4)

    def test_a_saturated_score_is_still_close(self):
        # A score that reads 1 patch, flat at the real point and steep at the
        # baseline. IG stays complete to the step error.
        torch.manual_seed(1)
        n, d, steps = 8, 5, 256
        w = torch.randn(d)
        h = torch.randn(n, d) * 2.0
        hbar = h.mean(0, keepdim=True)
        delta = h - hbar
        base0 = hbar[0].detach().clone()

        def score(z):
            return torch.tanh(3.0 * ((z[0] - base0) @ w))

        total = torch.zeros_like(delta)
        for k in range(steps):
            a = (k + 0.5) / steps
            z = (hbar + a * delta).detach().requires_grad_(True)
            (g,) = torch.autograd.grad(score(z), z)
            total += g
        ig = (delta * (total / steps)).sum(1)
        gap = float(score(h)) - float(score(hbar.expand_as(h)))
        assert abs(gap) > 0.5
        assert float(ig.sum()) == pytest.approx(gap, abs=0.02)

    def test_the_plain_gradient_is_NOT_complete_when_saturated(self):
        # This is the reason the module exists.
        #
        # Warning: a score that reads only the SUM over the patches cannot
        # show this. The baseline is the MEAN patch, so a blend toward it
        # leaves the sum unchanged and both ends agree. The score below reads
        # 1 patch against the baseline, so it is 0 at the baseline and flat at
        # the real point.
        torch.manual_seed(2)
        n, d = 8, 5
        w = torch.randn(d)
        h = torch.randn(n, d) * 3.0
        hbar = h.mean(0, keepdim=True)
        delta = h - hbar
        base0 = hbar[0].detach().clone()

        def score(z):
            return torch.tanh(3.0 * ((z[0] - base0) @ w))

        z1 = h.detach().clone().requires_grad_(True)
        (g1,) = torch.autograd.grad(score(z1), z1)
        gxa = float((delta * g1).sum())
        gap = float(score(h)) - float(score(hbar.expand_as(h)))
        assert abs(gap) > 0.9          # the true change is nearly the full range
        assert abs(gxa) < 0.05         # the gradient at the flat top says nothing
        assert abs(gxa - gap) > 0.85
