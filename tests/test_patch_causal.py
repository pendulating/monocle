"""CPU tests for the causal-calibration geometry and scoring.

Every torch/transformers call that needs a GPU stays inside `main`, so this
file imports on a login node.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import patch_causal as pc  # noqa: E402


def raster_pid(n_rows: int, n_cols: int, n_slots: int) -> torch.Tensor:
    """Processor-shaped (x, y) ids: live patches first, then -1 padding."""
    pid = torch.full((n_slots, 2), -1, dtype=torch.long)
    for r in range(n_rows):
        for c in range(n_cols):
            pid[r * n_cols + c] = torch.tensor([c, r])
    return pid


class TestGrid:
    def test_reads_a_square_grid_and_its_padding(self):
        n_r, n_c, valid = pc.grid_from_position_ids(raster_pid(16, 16, 280))
        assert (n_r, n_c) == (16, 16)
        assert valid.size == 256
        assert valid[0] == 0 and valid[-1] == 255

    def test_reads_a_non_square_grid(self):
        n_r, n_c, valid = pc.grid_from_position_ids(raster_pid(14, 19, 280))
        assert (n_r, n_c) == (14, 19)
        assert valid.size == 266

    def test_rejects_a_non_raster_order(self):
        """A transposed map is silent corruption, so it must raise."""
        pid = raster_pid(4, 4, 16)
        pid[[1, 2]] = pid[[2, 1]]
        with pytest.raises(RuntimeError, match="raster order"):
            pc.grid_from_position_ids(pid)

    def test_rejects_a_bad_shape(self):
        with pytest.raises(ValueError, match=r"\(n_slots, 2\)"):
            pc.grid_from_position_ids(torch.zeros(10))

    def test_rejects_an_all_padding_image(self):
        with pytest.raises(ValueError, match="no valid"):
            pc.grid_from_position_ids(torch.full((8, 2), -1))


class TestBlocks:
    def test_exact_division(self):
        b = pc.pooled_blocks(6, 6, 3)
        assert len(b) == 4
        assert all(idx.size == 9 for _, _, idx in b)

    def test_ragged_last_block(self):
        """16 / 3 gives five blocks of 3 and one of 1 on each axis."""
        b = pc.pooled_blocks(16, 16, 3)
        assert len(b) == 36
        sizes = sorted({idx.size for _, _, idx in b})
        assert sizes == [1, 3, 9]

    def test_blocks_partition_the_grid_exactly_once(self):
        seen = np.concatenate([idx for _, _, idx in pc.pooled_blocks(16, 16, 3)])
        assert sorted(seen) == list(range(256))

    def test_pool_one_is_one_block_per_patch(self):
        assert len(pc.pooled_blocks(4, 4, 1)) == 16

    def test_rejects_a_bad_pool(self):
        with pytest.raises(ValueError):
            pc.pooled_blocks(4, 4, 0)


class TestSum:
    def test_sums_not_averages(self):
        """Ablating a block removes every patch in it, so the prediction adds."""
        blocks = pc.pooled_blocks(2, 2, 2)
        out = pc.sum_over_blocks(np.array([1.0, 2.0, 3.0, 4.0]), blocks)
        assert out.tolist() == [10.0]

    def test_totals_are_preserved_across_blocks(self):
        v = np.arange(256, dtype=float)
        out = pc.sum_over_blocks(v, pc.pooled_blocks(16, 16, 3))
        assert out.sum() == pytest.approx(v.sum())


class TestScoring:
    def test_perfect_predictor(self):
        v = np.array([1.0, -2.0, 3.0, -4.0])
        assert pc.safe_corr(v, v) == pytest.approx(1.0)
        assert pc.sign_agreement(v, v) == 1.0

    def test_inverted_predictor(self):
        v = np.array([1.0, -2.0, 3.0, -4.0])
        assert pc.safe_corr(v, -v) == pytest.approx(-1.0)
        assert pc.sign_agreement(v, -v) == 0.0

    def test_constant_side_is_nan_not_zero(self):
        assert np.isnan(pc.safe_corr(np.ones(4), np.arange(4.0)))

    def test_sign_agreement_ignores_exact_zeros(self):
        a = np.array([1.0, 0.0, -1.0])
        b = np.array([2.0, 5.0, -3.0])
        assert pc.sign_agreement(a, b) == 1.0

    def test_sign_agreement_is_nan_when_nothing_is_live(self):
        assert np.isnan(pc.sign_agreement(np.zeros(3), np.zeros(3)))


class TestSummary:
    def frame(self, flip: bool = False):
        import pandas as pd
        n = 24
        rng = np.random.default_rng(0)
        truth = rng.normal(size=n)
        return pd.DataFrame({
            "pair_id": ["p0"] * n, "slot": ["A"] * 12 + ["B"] * 12,
            "true_delta": truth, "c": np.ones(n),
            "pred_pixel": truth * (-1 if flip else 1),
            "pred_layer_46": rng.normal(size=n),
        })

    def test_ranks_the_best_predictor_first(self):
        s = pc.summarize("t", "judgment", self.frame())
        assert s["predictors"][0]["predictor"] == "pred_pixel"
        assert s["predictors"][0]["r_all"] == pytest.approx(1.0)

    def test_an_inverted_predictor_ranks_last(self):
        s = pc.summarize("t", "judgment", self.frame(flip=True))
        assert s["predictors"][-1]["predictor"] == "pred_pixel"
        assert s["predictors"][-1]["r_all"] == pytest.approx(-1.0)

    def test_reports_per_slot_correlations(self):
        s = pc.summarize("t", "judgment", self.frame())
        row = s["predictors"][0]
        assert row["r_A"] == pytest.approx(1.0)
        assert row["r_B"] == pytest.approx(1.0)


class TestCli:
    def test_defaults(self):
        a = pc.build_parser().parse_args([])
        assert a.cases == ["subway_safety"]
        assert a.pool == 3 and a.n_pairs == 50 and a.contrast == "judgment"

    def test_rejects_an_unregistered_case(self):
        with pytest.raises(SystemExit):
            pc.build_parser().parse_args(["--cases", "nope"])


class TestReplicate:
    def inputs(self):
        return {
            "input_ids": torch.arange(6).reshape(1, 6),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "pixel_values": torch.arange(2 * 4 * 3).reshape(2, 4, 3).float(),
            "image_position_ids": raster_pid(2, 2, 4).unsqueeze(0).repeat(2, 1, 1),
        }

    def test_text_and_image_rows_repeat_separately(self):
        out = pc.replicate(self.inputs(), 3)
        assert out["input_ids"].shape == (3, 6)
        assert out["pixel_values"].shape == (6, 4, 3)

    def test_images_keep_the_a_b_a_b_order(self):
        """The model consumes images in sequence order, so A,B must repeat as a
        unit -- A,A,B,B would attach the wrong picture to every second row."""
        out = pc.replicate(self.inputs(), 3)["pixel_values"]
        src = self.inputs()["pixel_values"]
        for r in range(3):
            assert torch.equal(out[2 * r], src[0])
            assert torch.equal(out[2 * r + 1], src[1])

    def test_n_one_is_unchanged(self):
        src = self.inputs()
        out = pc.replicate(src, 1)
        for k in src:
            assert torch.equal(out[k], src[k])

    def test_passes_non_tensors_through(self):
        out = pc.replicate({"x": "keep", "input_ids": torch.ones(1, 2)}, 2)
        assert out["x"] == "keep"

    def test_rejects_a_bad_n(self):
        with pytest.raises(ValueError):
            pc.replicate(self.inputs(), 0)


class TestFp32Contrast:
    class FakeLens:
        def __init__(self, w, cap=None):
            import torch.nn as nn
            self._lm_head = nn.Linear(w.shape[1], w.shape[0], bias=False)
            with torch.no_grad():
                self._lm_head.weight.copy_(w)
            self._final_norm = nn.Identity()
            self._logit_softcap = cap

    def test_matches_the_plain_difference_without_a_cap(self):
        w = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        h = torch.tensor([[3.0, 5.0]])
        c = pc.fp32_contrast(self.FakeLens(w), h, 0, 1)
        assert c.item() == pytest.approx(3.0 - 10.0)

    def test_returns_float32_from_a_bfloat16_residual(self):
        w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        h = torch.tensor([[3.0, 5.0]], dtype=torch.bfloat16)
        c = pc.fp32_contrast(self.FakeLens(w), h, 0, 1)
        assert c.dtype == torch.float32

    def test_applies_the_softcap(self):
        w = torch.eye(2) * 100.0
        h = torch.tensor([[1.0, 0.0]])
        c = pc.fp32_contrast(self.FakeLens(w, cap=30.0), h, 0, 1)
        # 100 and 0 -> 30*tanh(10/3) and 0
        assert c.item() == pytest.approx(30.0 * float(torch.tanh(
            torch.tensor(100.0 / 30.0))))

    def test_handles_a_batch(self):
        w = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        h = torch.tensor([[3.0, 5.0], [1.0, 0.0]])
        c = pc.fp32_contrast(self.FakeLens(w), h, 0, 1)
        assert c.shape == (2,)
        assert c.tolist() == pytest.approx([-2.0, 1.0])
