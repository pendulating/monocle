"""CPU-only unit tests for the multimodal Jacobian-lens fitter core.

No GPU, no gemma load, no jlens/gemma imports at module level. The pure
estimator (`monocle.jlens_fit_mm.jacobian_for_inputs`) is exercised against a
tiny per-position linear toy model whose true patch->target Jacobian is known
in closed form, so the estimator's correctness (and its source/target mask
semantics) can be pinned exactly on CPU.

ActivationRecorder is pulled in *inside* jacobian_for_inputs via a two-venv
fallback (installed jlens first, else the vendored copy on sys.path), so these
tests run in the plain .venv where jlens is not pip-installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import jlens_fit_mm as jf  # noqa: E402


# ---------------------------------------------------------------------------
# Toy model: two per-position linear blocks, no cross-position mixing.
# ---------------------------------------------------------------------------
class ToyModel(nn.Module):
    """h = embed(input_ids); h = block0(h); h = block1(h); return h.

    Per-position Linear(d, d, bias=False) blocks mean there is NO cross-position
    mixing. For source layer 0 (block0 output) and target layer 1 (block1
    output): dh_1[p'] / dh_0[p] = W1 if p' == p else 0. Params are frozen so
    the ActivationRecorder can mark the layer-0 output as the grad leaf (matches
    HFLensModel, which freezes every parameter).
    """

    def __init__(self, d: int, vocab: int = 32, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.block0 = nn.Linear(d, d, bias=False)
        self.block1 = nn.Linear(d, d, bias=False)
        self.layers = nn.ModuleList([self.block0, self.block1])
        self.seen_batch_dims: list[int] = []
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_ids=None, pixel_values=None, **kw):
        self.seen_batch_dims.append(int(input_ids.shape[0]))
        h = self.embed(input_ids)
        h = self.block0(h)
        h = self.block1(h)
        return h


class DuckLens:
    """Minimal lens_model: .layers / .n_layers / .d_model."""

    def __init__(self, model: ToyModel) -> None:
        self.layers = model.layers
        self.n_layers = len(model.layers)
        self.d_model = model.d


def _make(d: int = 6, seq: int = 5, vocab: int = 32):
    model = ToyModel(d=d, vocab=vocab)
    lens = DuckLens(model)
    input_ids = torch.randint(0, vocab, (1, seq))
    inputs = {"input_ids": input_ids}
    return model, lens, inputs


# ---------------------------------------------------------------------------
# 1. Exactness: J_0 == W1 when every source position is inside the target mask.
# ---------------------------------------------------------------------------
class TestJacobianExact:
    @pytest.mark.parametrize("dim_batch", [2, 6])
    def test_restricted_source_all_targets(self, dim_batch: int):
        d, seq = 6, 5
        model, lens, inputs = _make(d=d, seq=seq)
        W1 = model.block1.weight.detach().clone()  # [d, d]

        source_positions = torch.tensor([2, 3])
        target_mask = torch.ones(seq, dtype=torch.bool)
        target_mask[seq - 1] = False  # all-True-except-last; still covers 2,3

        J = jacobian_for_inputs_helper(
            model, lens, inputs, source_positions, target_mask,
            dim_batch=dim_batch)

        assert set(J.keys()) == {0}
        assert J[0].shape == (d, d)
        assert torch.allclose(J[0], W1, atol=1e-5), (J[0] - W1).abs().max()


# ---------------------------------------------------------------------------
# 2. Mask semantics: targets disjoint from sources -> J_0 == 0.
# ---------------------------------------------------------------------------
class TestMaskSemantics:
    def test_targets_exclude_sources_zero(self):
        d, seq = 6, 5
        model, lens, inputs = _make(d=d, seq=seq)

        source_positions = torch.tensor([1, 2])
        target_mask = torch.zeros(seq, dtype=torch.bool)
        target_mask[4] = True  # target only at position 4, disjoint from 1,2

        J = jacobian_for_inputs_helper(
            model, lens, inputs, source_positions, target_mask, dim_batch=3)

        assert torch.allclose(J[0], torch.zeros(d, d), atol=1e-6), \
            J[0].abs().max()


# ---------------------------------------------------------------------------
# 3. Batch replication of an extra float tensor (fake pixel_values).
# ---------------------------------------------------------------------------
class TestBatchReplication:
    def test_extra_float_tensor_replicated(self):
        d, seq = 6, 5
        model, lens, inputs = _make(d=d, seq=seq)
        inputs = {**inputs, "pixel_values": torch.randn(1, 3, 8)}  # 3-D float

        source_positions = torch.tensor([2, 3])
        target_mask = torch.ones(seq, dtype=torch.bool)
        target_mask[seq - 1] = False

        model.seen_batch_dims.clear()
        dim_batch = 4
        J = jacobian_for_inputs_helper(
            model, lens, inputs, source_positions, target_mask,
            dim_batch=dim_batch)

        # forward saw a batch of exactly dim_batch rows.
        assert model.seen_batch_dims
        assert all(b == dim_batch for b in model.seen_batch_dims)
        # and the result is still exact (the extra tensor did not perturb it).
        assert torch.allclose(J[0], model.block1.weight, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Validation guards.
# ---------------------------------------------------------------------------
class TestValidation:
    def test_empty_source_positions_raises(self):
        model, lens, inputs = _make()
        with pytest.raises(ValueError):
            jf.jacobian_for_inputs(
                model, lens, inputs,
                torch.tensor([], dtype=torch.long),
                torch.ones(5, dtype=torch.bool),
                source_layers=[0], target_layer=1, dim_batch=2)

    def test_source_layer_ge_target_raises(self):
        model, lens, inputs = _make()
        with pytest.raises(ValueError):
            jf.jacobian_for_inputs(
                model, lens, inputs,
                torch.tensor([2, 3]),
                torch.ones(5, dtype=torch.bool),
                source_layers=[1], target_layer=1, dim_batch=2)

    def test_empty_target_mask_raises(self):
        model, lens, inputs = _make()
        with pytest.raises(ValueError):
            jf.jacobian_for_inputs(
                model, lens, inputs,
                torch.tensor([2, 3]),
                torch.zeros(5, dtype=torch.bool),
                source_layers=[0], target_layer=1, dim_batch=2)


def jacobian_for_inputs_helper(
    model, lens, inputs, source_positions, target_mask, *, dim_batch,
):
    """Fit source layer 0, target layer 1 (the toy's only fittable pair)."""
    return jf.jacobian_for_inputs(
        model, lens, inputs, source_positions, target_mask,
        source_layers=[0], target_layer=1, dim_batch=dim_batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
