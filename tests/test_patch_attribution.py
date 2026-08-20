"""CPU tests for the per-patch decision attribution.

Covers the pure pieces: contrast selection, grad x activation, map
normalization, the long-row layout, and the summary. A real gradient needs the
model, so `attribute_pair` is exercised by the GPU smoke, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path("/share/pierson/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import canonical  # noqa: E402
from monocle import patch_attribution as pa  # noqa: E402

LABELS = {"MuchLess": 100, "Less": 7, "Same": 8, "More": 9, "MuchMore": 100,
          "NotSure": 42}
LABELS_NO_ABSTAIN = {k: v for k, v in LABELS.items() if k != "NotSure"}


# ---------------------------------------------------------------------------
# contrast
# ---------------------------------------------------------------------------
class TestContrast:
    def test_judgment_is_more_minus_less(self):
        assert pa.contrast_token_ids("judgment", LABELS) == (9, 7, "More", "Less")

    def test_abstain_is_notsure_minus_more(self):
        assert pa.contrast_token_ids("abstain", LABELS) == (
            42, 9, "NotSure", "More")

    def test_abstain_refuses_a_run_without_abstention(self):
        """Better to fail than to silently measure a different contrast."""
        with pytest.raises(KeyError, match="did not enable abstention"):
            pa.contrast_token_ids("abstain", LABELS_NO_ABSTAIN)

    def test_unknown_contrast_raises(self):
        with pytest.raises(ValueError, match="unknown contrast"):
            pa.contrast_token_ids("vibes", LABELS)

    def test_abstain_uses_the_canonical_label_name(self):
        assert canonical.NOT_SURE in LABELS


# ---------------------------------------------------------------------------
# grad x activation
# ---------------------------------------------------------------------------
class TestGradXActivation:
    def test_is_a_row_wise_dot_product(self):
        g = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        a = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        assert pa.grad_x_activation(g, a).tolist() == [17.0, 53.0]

    def test_sign_is_preserved(self):
        """A patch can push the contrast either way; the sign says which."""
        g = torch.tensor([[-1.0, 0.0]])
        a = torch.tensor([[2.0, 9.0]])
        assert pa.grad_x_activation(g, a).item() == -2.0

    def test_zero_gradient_gives_zero(self):
        out = pa.grad_x_activation(torch.zeros(3, 4), torch.randn(3, 4))
        assert torch.equal(out, torch.zeros(3))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            pa.grad_x_activation(torch.zeros(2, 4), torch.zeros(3, 4))

    def test_returns_one_value_per_position(self):
        out = pa.grad_x_activation(torch.randn(256, 8), torch.randn(256, 8))
        assert out.shape == (256,)


# ---------------------------------------------------------------------------
# map normalization
# ---------------------------------------------------------------------------
class TestNormalize:
    def test_max_scaling_is_still_available(self):
        out = pa.normalize_map(np.array([-4.0, 2.0, 1.0]), percentile=100)
        assert out.tolist() == [-1.0, 0.5, 0.25]

    def test_a_single_outlier_no_longer_flattens_the_map(self):
        """The L46 case: one huge patch, the rest small but not nothing.

        Max-scaling drives the small cells to ~0.01 and the map reads as
        empty. The percentile clip keeps them visible.
        """
        # A realistic pooled grid: 36 cells, one of them dominant.
        v = np.array([100.0] + [1.0, 2.0, 3.0, 4.0] * 8 + [1.0, 2.0, 3.0])
        flat = pa.normalize_map(v, percentile=100)
        clipped = pa.normalize_map(v, percentile=95)
        assert flat[1:].max() < 0.05, "max-scaling flattens the map"
        assert clipped[1:].max() > 0.5, "the clip keeps the rest legible"
        assert clipped[0] == 1.0, "the outlier saturates rather than vanishing"

    def test_the_default_percentile_suits_a_pooled_grid(self):
        """A pool of 3 leaves 36 cells; the 99th would clip nothing there."""
        v = np.array([100.0] + [1.0] * 35)
        assert pa.normalize_map(v, percentile=99)[1:].max() < 0.05
        assert pa.normalize_map(v)[1:].max() == 1.0

    def test_clipping_keeps_values_inside_the_unit_range(self):
        out = pa.normalize_map(np.random.randn(500) * 9, percentile=90)
        assert out.min() >= -1.0 and out.max() <= 1.0

    def test_keeps_negative_values_negative(self):
        out = pa.normalize_map(np.array([-8.0, 4.0]))
        assert out[0] < 0 < out[1]

    def test_an_all_zero_map_does_not_divide_by_zero(self):
        out = pa.normalize_map(np.zeros(5))
        assert out.tolist() == [0.0] * 5

    def test_output_reaches_the_unit_range(self):
        out = pa.normalize_map(np.random.randn(100) * 17, percentile=100)
        assert np.abs(out).max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# long rows
# ---------------------------------------------------------------------------
class TestPatchRows:
    def setup_method(self):
        d = 4
        # Patch activations must VARY, or centering makes them all zero.
        act = torch.arange(10 * d, dtype=torch.float).reshape(1, 10, d)
        self.grads = {6: torch.ones(1, 10, d), 42: torch.ones(1, 10, d) * 2}
        self.acts = {6: act, 42: act}
        self.blocks = [torch.tensor([2, 3, 4, 5]), torch.tensor([6, 7, 8, 9])]
        self.grids = {"A": (2, 2), "B": (2, 2)}

    def rows(self):
        r, _ = pa.patch_rows(self.grads, self.acts, self.blocks, self.grids,
                             {"case": "c", "cond": "prod", "pair_id": "p0"})
        return r

    def diag(self):
        _, d = pa.patch_rows(self.grads, self.acts, self.blocks, self.grids,
                             {"case": "c", "cond": "prod", "pair_id": "p0"})
        return d

    def test_one_row_per_layer_slot_and_patch(self):
        assert len(self.rows()) == 2 * 2 * 4

    def test_both_image_blocks_appear(self):
        assert {r["slot"] for r in self.rows()} == {"A", "B"}

    def test_grid_coordinates_are_row_major(self):
        a6 = [r for r in self.rows() if r["slot"] == "A" and r["layer"] == 6]
        assert [(r["patch_row"], r["patch_col"]) for r in a6] == [
            (0, 0), (0, 1), (1, 0), (1, 1)]

    def test_carries_both_centered_and_raw(self):
        """The raw value stays so the artefact remains measurable."""
        r = self.rows()[0]
        assert "attrib" in r and "attrib_raw" in r
        assert r["attrib"] != r["attrib_raw"]

    def test_centering_removes_the_block_mean(self):
        """With a constant gradient, centered attributions sum to ~0."""
        a6 = [r["attrib"] for r in self.rows()
              if r["slot"] == "A" and r["layer"] == 6]
        assert sum(a6) == pytest.approx(0.0, abs=1e-4)

    def test_the_two_blocks_are_centered_independently(self):
        """Image B must not be centered by image A's mean."""
        rows = self.rows()
        for slot in ("A", "B"):
            vals = [r["attrib"] for r in rows
                    if r["slot"] == slot and r["layer"] == 6]
            assert sum(vals) == pytest.approx(0.0, abs=1e-4)

    def test_base_fields_are_carried(self):
        assert all(r["case"] == "c" and r["pair_id"] == "p0"
                   for r in self.rows())

    def test_diagnostic_reports_one_row_per_layer_and_slot(self):
        d = self.diag()
        assert len(d) == 4
        assert all("shared_ratio" in r for r in d)


class TestCentering:
    def test_center_subtracts_the_patch_mean(self):
        a = torch.tensor([[1.0, 1.0], [3.0, 5.0]])
        out = pa.center_over_patches(a)
        assert out.mean(dim=0).abs().max() < 1e-6

    def test_center_rejects_a_batched_tensor(self):
        with pytest.raises(ValueError, match="n_patches"):
            pa.center_over_patches(torch.zeros(1, 4, 8))

    def test_shared_ratio_is_large_when_a_massive_component_dominates(self):
        """The measured gemma case: a huge shared vector, tiny spread."""
        base = torch.zeros(16, 8)
        base[:, 3] = 250.0
        jitter = torch.randn(16, 8) * 0.1
        assert pa.shared_component_ratio(base + jitter) > 100

    def test_shared_ratio_is_small_without_one(self):
        assert pa.shared_component_ratio(torch.randn(64, 16)) < 2.0

    def test_shared_ratio_rejects_a_batched_tensor(self):
        with pytest.raises(ValueError, match="n_patches"):
            pa.shared_component_ratio(torch.zeros(1, 4, 8))


class TestPooling:
    def test_pool_one_is_a_no_op(self):
        v = np.arange(16, dtype=float)
        out, r, c = pa.pool_map(v, 4, 4, 1)
        assert (out is v) and (r, c) == (4, 4)

    def test_pool_two_means_each_block(self):
        v = np.array([1., 2., 3., 4.,
                      5., 6., 7., 8.,
                      9., 10., 11., 12.,
                      13., 14., 15., 16.])
        out, r, c = pa.pool_map(v, 4, 4, 2)
        assert (r, c) == (2, 2)
        assert out.tolist() == [3.5, 5.5, 11.5, 13.5]

    def test_sixteen_by_three_gives_a_six_by_six_grid(self):
        """What the 6x6 request resolves to: 5 full bands plus a thin edge."""
        out, r, c = pa.pool_map(np.zeros(256), 16, 16, 3)
        assert (r, c) == (6, 6)
        assert out.size == 36

    def test_mean_keeps_the_thin_edge_band_comparable(self):
        """A sum would make the 1-wide edge look artificially quiet."""
        v = np.ones(256)
        out, _, _ = pa.pool_map(v, 16, 16, 3)
        assert out.min() == pytest.approx(1.0)
        assert out.max() == pytest.approx(1.0)

    def test_pool_tokens_merges_by_summed_probability(self):
        """A token several patches agree on beats one a single patch shouts."""
        toks = {0: ["road"], 1: ["road"], 16: ["road"], 17: ["SHOUT"]}
        prbs = {0: [0.3], 1: [0.3], 16: [0.3], 17: [0.8]}
        out = pa.pool_tokens(toks, prbs, 16, 16, 3, k=2)
        assert out[0][0] == "road"

    def test_pool_tokens_is_a_truncation_at_pool_one(self):
        out = pa.pool_tokens({0: ["a", "b", "c", "d"]}, {0: [1, 1, 1, 1]},
                             4, 4, 1, k=2)
        assert out[0] == ["a", "b"]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def attrib_frame():
    """One pair, one layer, one slot: 19 near-zero patches and 1 large one."""
    vals = [0.01] * 19 + [10.0]
    return pd.DataFrame([
        {"cond": "prod", "layer": 42, "slot": "A", "pair_id": "p0",
         "patch_idx": i, "attrib": v} for i, v in enumerate(vals)])


class TestSummary:
    def test_concentration_finds_a_single_dominant_patch(self, tmp_path):
        s = pa.summarize_case("c", attrib_frame(), "judgment", "More", "Less",
                              tmp_path)
        g = s["by_group"][0]
        # top 5% of 20 patches = 1 patch, holding 10.0 of 10.19 total.
        assert g["top5pct_share"] == pytest.approx(10.0 / 10.19, rel=1e-3)

    def test_reports_both_signed_and_absolute_means(self, tmp_path):
        s = pa.summarize_case("c", attrib_frame(), "judgment", "More", "Less",
                              tmp_path)
        g = s["by_group"][0]
        assert g["mean_attrib"] == pytest.approx(g["mean_abs_attrib"])
        assert g["max_abs_attrib"] == 10.0

    def test_records_the_contrast_it_used(self, tmp_path):
        s = pa.summarize_case("c", attrib_frame(), "abstain", "NotSure",
                              "More", tmp_path)
        assert s["contrast"] == "abstain"
        assert (s["positive"], s["negative"]) == ("NotSure", "More")

    def test_an_empty_frame_summarizes_to_nothing(self, tmp_path):
        s = pa.summarize_case("c", pd.DataFrame(), "judgment", "More", "Less",
                              tmp_path)
        assert s["by_group"] == [] and s["n_pairs"] == 0


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
class TestRender:
    def test_overlay_upscales_the_source(self):
        """Upscaled so the readout words stay legible in a cell."""
        from PIL import Image

        im = Image.new("RGB", (64, 48), (10, 10, 10))
        out = pa.render_attrib(im, np.random.randn(16), 4, 4, upscale=2)
        assert out.size == (128, 96)

    def test_overlay_accepts_pooled_tokens(self):
        from PIL import Image

        im = Image.new("RGB", (320, 320), (10, 10, 10))
        out = pa.render_attrib(
            im, np.random.randn(256), 16, 16, pool=3,
            tokens={i: ["road", "sign"] for i in range(36)})
        assert out.size == (640, 640)

    def test_the_ramp_diverges_around_zero(self):
        neg, zero, pos = pa._div_color(-1.0), pa._div_color(0.0), pa._div_color(1.0)
        assert zero == (255, 255, 255)
        assert neg[2] > neg[0]          # teal side: more blue than red
        assert pos[0] > pos[2]          # coral side: more red than blue


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
class TestCli:
    def test_defaults(self):
        args = pa.build_parser().parse_args([])
        assert args.cases == list(canonical.CASES)
        assert args.contrast == "judgment"
        assert args.n_pairs == 200
        assert args.kind == "proxy"

    def test_contrast_override(self):
        args = pa.build_parser().parse_args(
            ["--contrast", "abstain", "--cases", "schools", "--n-maps", "0"])
        assert args.contrast == "abstain"
        assert args.n_maps == 0

    def test_rejects_an_unknown_contrast(self):
        with pytest.raises(SystemExit):
            pa.build_parser().parse_args(["--contrast", "vibes"])


# ---------------------------------------------------------------------------
# positional diagnostics
# ---------------------------------------------------------------------------
class TestPositional:
    def test_row_profile_is_uniform_on_a_flat_map(self):
        pr = pa.row_profile(np.ones(256), 16, 16)
        assert pr.shape == (16,)
        assert pr.min() == pytest.approx(1 / 16)

    def test_row_profile_finds_a_hot_last_row(self):
        """The measured L46 case: the causal aggregation row dominates."""
        v = np.ones(256) * 0.01
        v[240:] = 1.0
        assert pa.row_profile(v, 16, 16)[-1] > 0.8

    def test_row_profile_uses_magnitude_not_sign(self):
        """A row of strong negative attributions is not a quiet row."""
        v = np.zeros(256)
        v[240:] = -1.0
        assert pa.row_profile(v, 16, 16)[-1] == pytest.approx(1.0)

    def test_difference_cancels_a_shared_positional_pattern(self):
        """Both arms carry the same causal geometry; only the rest survives."""
        geom = np.zeros(256)
        geom[240:] = 5.0
        question = np.zeros(256)
        question[100] = 2.0
        diff = pa.prod_minus_neutral(geom + question, geom)
        assert diff[240:].max() == pytest.approx(0.0)
        assert diff[100] == pytest.approx(2.0)

    def test_difference_rejects_mismatched_grids(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            pa.prod_minus_neutral(np.zeros(256), np.zeros(64))

    def test_correlation_flags_a_question_independent_map(self):
        """The measured libraries L46 case: r = 0.98 between the arms."""
        a = np.random.randn(256)
        assert pa.map_correlation(a, a) == pytest.approx(1.0)
        assert pa.map_correlation(a, a + np.random.randn(256) * 0.05) > 0.9

    def test_correlation_is_nan_on_a_constant_map(self):
        assert np.isnan(pa.map_correlation(np.ones(256), np.random.randn(256)))

    def test_correlation_rejects_mismatched_grids(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            pa.map_correlation(np.zeros(256), np.zeros(64))
