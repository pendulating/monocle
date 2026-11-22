"""What is in the squares? See research/monocle/cell_statistics.py."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from research.monocle import cell_statistics as cs


class TestCellBoxes:
    def test_a_64_cell_grid_is_8_by_8(self):
        b = cs.cell_boxes(64)
        assert len(b) == 64
        assert b[0] == (0, 0, 96, 96)

    def test_row_major_order(self):
        """Cell 1 is to the RIGHT of cell 0, not below it."""
        b = cs.cell_boxes(64)
        assert b[1][0] > b[0][0] and b[1][1] == b[0][1]
        assert b[8][1] > b[0][1] and b[8][0] == b[0][0]

    def test_the_boxes_reach_the_far_edge(self):
        assert cs.cell_boxes(64)[-1][2] == cs.MODEL_SIDE

    def test_rejects_a_non_square_count(self):
        with pytest.raises(ValueError, match="not a square grid"):
            cs.cell_boxes(63)


class TestCellStats:
    def half_dark(self):
        """Top half black, bottom half white."""
        a = np.zeros((cs.MODEL_SIDE, cs.MODEL_SIDE, 3), dtype=np.uint8)
        a[cs.MODEL_SIDE // 2:] = 255
        return Image.fromarray(a)

    def test_it_finds_the_dark_half(self):
        d = cs.cell_stats(self.half_dark(), 64)
        top = d[d.row < 4].luminance.mean()
        bottom = d[d.row >= 4].luminance.mean()
        assert top < 0.1 and bottom > 0.9

    def test_a_flat_cell_has_no_contrast(self):
        d = cs.cell_stats(self.half_dark(), 64)
        assert d[d.row == 0].contrast.max() < 1e-4

    def test_a_grey_image_has_no_saturation(self):
        a = np.full((64, 64, 3), 128, dtype=np.uint8)
        d = cs.cell_stats(Image.fromarray(a), 64)
        assert d.saturation.max() < 1e-4

    def test_a_colour_image_has_saturation(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8); a[..., 0] = 255
        d = cs.cell_stats(Image.fromarray(a), 64)
        assert d.saturation.min() > 0.9

    def test_the_row_and_col_match_the_cell_index(self):
        d = cs.cell_stats(self.half_dark(), 64)
        assert d.loc[9, "row"] == 1 and d.loc[9, "col"] == 1

    def test_it_gives_one_row_for_each_cell(self):
        assert len(cs.cell_stats(self.half_dark(), 256)) == 256


class TestFits:
    def frame(self, n=64, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        lum = rng.random(n)
        return pd.DataFrame({
            "image_id": "im1", "cell": range(n), "row": 0, "col": 0,
            "luminance": lum, "contrast": rng.random(n),
            "edge": rng.random(n), "saturation": rng.random(n),
            "value": lum * 2.0 + rng.normal(0, noise, n)})

    def test_it_finds_a_planted_relation(self):
        out = cs.fit_within_image(self.frame(), ["luminance", "contrast"])
        assert out[out.prop == "luminance"].r.iloc[0] > 0.95

    def test_an_unrelated_property_reads_near_zero(self):
        out = cs.fit_within_image(self.frame(n=256), ["contrast"])
        assert abs(out[out.prop == "contrast"].r.iloc[0]) < 0.2

    def test_the_joint_fit_recovers_the_planted_weight(self):
        j = cs.multi_fit(self.frame(), ["luminance", "contrast", "edge",
                                        "saturation", "row", "col"])
        assert j.beta_luminance.iloc[0] > 0.9
        assert j.r2.iloc[0] > 0.95

    def test_a_flat_property_does_not_divide_by_zero(self):
        """`row` and `col` are constant in this fixture."""
        out = cs.fit_within_image(self.frame(), ["row"])
        assert out.r.iloc[0] == 0.0 or np.isnan(out.r.iloc[0])

    def test_the_fit_runs_inside_each_image(self):
        a = self.frame(seed=0); b = self.frame(seed=1)
        b["image_id"] = "im2"
        out = cs.fit_within_image(pd.concat([a, b]), ["luminance"])
        assert set(out.image_id) == {"im1", "im2"}
