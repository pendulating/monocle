"""The 3-column caption figure. See research/monocle/caption_panel.py."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from research.monocle import caption_panel as cp


def photo(side=64, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(40, 220, (side, side, 3), dtype=np.uint8))


def cell_box(img, cell, n_cells=64):
    """The pixels of one cell, in a rendered panel."""
    n_rows, n_cols = cp.cm.grid_of(n_cells)
    W, H = img.size
    cw, ch = W / n_cols, H / n_rows
    r, c = cell // n_cols, cell % n_cols
    # Step inside the border, which is drawn in the pole colour.
    pad = 4
    return np.array(img.crop((int(c * cw) + pad, int(r * ch) + pad,
                              int((c + 1) * cw) - pad,
                              int((r + 1) * ch) - pad)), dtype=float)


class TestColumns:
    def test_the_order_reads_safe_then_whole_then_danger(self):
        assert cp.COLUMNS == ("keep_safe", "whole_image", "keep_danger")

    def test_the_poles_are_the_map_ramp_ends(self):
        """A column must not carry a colour the colour bar does not name."""
        assert cp.TEAL == cp.pa._div_color(-1.0)
        assert cp.CORAL == cp.pa._div_color(1.0)

    def test_teal_is_not_coral(self):
        assert cp.TEAL != cp.CORAL


class TestHighlight:
    def test_a_kept_cell_keeps_its_colour(self):
        """The ghost must not reach the cells the model was given."""
        img = cp.highlight(photo(), [0], 64, cp.TEAL)
        kept, gone = cell_box(img, 0), cell_box(img, 63)
        assert kept.std() > gone.std()

    def test_a_removed_cell_is_still_visible(self):
        """`ghost` fades, it does not erase. A white hole hides the context."""
        img = cp.highlight(photo(), [0], 64, cp.TEAL)
        gone = cell_box(img, 63)
        assert gone.std() > 1.0, "the removed cells went flat"
        assert gone.mean() < 254.0, "the removed cells went white"

    def test_a_removed_cell_is_paler_than_a_kept_one(self):
        img = cp.highlight(photo(), [0], 64, cp.TEAL)
        assert cell_box(img, 63).mean() > cell_box(img, 0).mean()

    def test_the_cell_index_is_row_major(self):
        """It must match `overlay`, which reshapes the values row-major."""
        img = cp.highlight(photo(), [1], 64, cp.TEAL)   # row 0, column 1
        assert cell_box(img, 1).std() > cell_box(img, 8).std()

    def test_keeping_every_cell_ghosts_nothing(self):
        img = cp.highlight(photo(), range(64), 64, cp.TEAL)
        assert cell_box(img, 0).std() == pytest.approx(
            cell_box(img, 63).std(), rel=0.6)

    def test_it_refuses_a_grid_that_is_not_square(self):
        with pytest.raises(ValueError, match="not square"):
            cp.highlight(photo(), [0], 63, cp.TEAL)

    def test_the_whole_photograph_is_returned(self):
        """Not a crop. The reader has to see the scene the map was built on."""
        img = cp.highlight(photo(side=64), [0], 64, cp.TEAL, upscale=2)
        assert img.size == (128, 128)


class TestWrap:
    def draw(self):
        return ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def test_no_line_is_wider_than_the_column(self):
        d, f = self.draw(), cp.font(13)
        text = "the quick brown fox jumps over the lazy dog " * 4
        for line in cp.wrap(text, f, 200, d):
            assert d.textlength(line, font=f) <= 200

    def test_every_word_survives(self):
        d, f = self.draw(), cp.font(13)
        text = "a dark interior space with a grated metal doorway"
        assert " ".join(cp.wrap(text, f, 120, d)).split() == text.split()

    def test_an_empty_caption_gives_no_lines(self):
        assert cp.wrap("", cp.font(13), 200, self.draw()) == []

    def test_one_long_word_is_kept_not_dropped(self):
        got = cp.wrap("supercalifragilistic", cp.font(13), 10, self.draw())
        assert got == ["supercalifragilistic"]


class TestTriptych:
    """The caption readout. It sizes its block to the words."""

    def build(self, captions=None, **kw):
        values = np.linspace(-1.0, 1.0, 64)
        caps = captions or {a: "a short sentence." for a in cp.COLUMNS}
        kw.setdefault("readout", "caption")
        return cp.triptych(
            "IMG_A", photo(), values, list(range(16)), list(range(48, 64)),
            caps, {a: 1.0 for a in cp.COLUMNS}, vmax=1.0,
            case="subway_safety", c_image=2.5, **kw)

    def test_it_is_three_columns_wide(self):
        w = 100
        assert self.build(width=w, gap=10).size[0] == 3 * w + 4 * 10

    def test_a_long_caption_makes_a_taller_panel(self):
        short = self.build()
        long = self.build({a: "word " * 60 for a in cp.COLUMNS})
        assert long.size[1] > short.size[1]

    def test_the_height_follows_the_longest_of_the_three(self):
        one = self.build({"keep_safe": "word " * 60, "whole_image": "a.",
                          "keep_danger": "b."})
        all3 = self.build({a: "word " * 60 for a in cp.COLUMNS})
        assert one.size[1] == all3.size[1]

    def test_a_missing_caption_does_not_raise(self):
        """An arm can be absent from a run; the column stays blank."""
        self.build({"keep_safe": "only this one."})

    def test_it_rejects_an_unknown_case(self):
        with pytest.raises(ValueError, match="unknown case"):
            cp.triptych("IMG_A", photo(), np.zeros(64), [0], [1], {}, {},
                        vmax=1.0, case="nope", c_image=0.0,
                        readout="caption")


class TestLoadRun:
    def test_it_says_which_file_is_missing(self, tmp_path):
        with pytest.raises(SystemExit, match="captions.parquet is missing"):
            cp.load_run(tmp_path)

    def test_an_old_run_is_named_as_such(self, tmp_path):
        """A run made before 2026-08-23 has no map, so it cannot be drawn."""
        import pandas as pd
        pd.DataFrame([{"a": 1}]).to_parquet(tmp_path / "captions.parquet")
        with pytest.raises(SystemExit, match="measured again"):
            cp.load_run(tmp_path)


class TestReadouts:
    """`rating` and `thinking` ask the SAME question of the model."""

    def build(self, readout, **kw):
        values = np.linspace(-1.0, 1.0, 64)
        kw.setdefault("c_rate", {a: 1.0 for a in cp.COLUMNS})
        return cp.triptych(
            "IMG_A", photo(), values, list(range(16)), list(range(48, 64)),
            {a: "word " * 20 for a in cp.COLUMNS},
            {a: 1.0 for a in cp.COLUMNS}, vmax=1.0, case="subway_safety",
            c_image=2.5, readout=readout, **kw)

    def test_all_three_are_named(self):
        assert cp.READOUTS == ("rating", "thinking", "caption")

    def test_the_default_generates_nothing(self):
        """`rating` is the default because no text means no hallucination."""
        import inspect
        sig = inspect.signature(cp.triptych)
        assert sig.parameters["readout"].default == "rating"

    def test_an_unknown_readout_is_refused(self):
        with pytest.raises(ValueError, match="readout must be"):
            self.build("guess")

    def test_the_rating_readout_needs_its_numbers(self):
        with pytest.raises(ValueError, match="needs a `c`"):
            self.build("rating", c_rate=None)

    def test_the_thinking_readout_needs_its_numbers(self):
        with pytest.raises(ValueError, match="needs a `c`"):
            self.build("thinking", c_rate=None)

    def test_a_rating_panel_ignores_the_caption_length(self):
        """It shows a verdict, so long words must not stretch it."""
        short = cp.triptych(
            "IMG_A", photo(), np.zeros(64), [0], [1], {a: "hi" for a in cp.COLUMNS},
            {}, vmax=1.0, case="subway_safety", c_image=0.0,
            readout="rating", c_rate={a: 1.0 for a in cp.COLUMNS})
        assert self.build("rating").size[1] == short.size[1]

    def test_a_thinking_panel_is_taller_than_a_rating_one(self):
        """It carries the verdict AND the trace."""
        assert self.build("thinking").size[1] > self.build("rating").size[1]


class TestVerdictBlock:
    def block(self, c):
        img = Image.new("RGB", (300, 90), (255, 255, 255))
        d = ImageDraw.Draw(img)
        cp.verdict_block(d, 4, 4, 280, c, cp.sg.rating("subway_safety"),
                         {"big": cp.font(17, True), "num": cp.font(12, True)})
        return img

    def near(self, img, colour, tol=40):
        a = np.array(img, dtype=int)
        return bool((np.abs(a - np.array(colour)).sum(-1) < tol).any())

    def test_a_positive_c_paints_the_safe_pole(self):
        assert self.near(self.block(+4.0), cp.TEAL)

    def test_a_negative_c_paints_the_danger_pole(self):
        assert self.near(self.block(-4.0), cp.CORAL)

    def test_both_poles_show_when_the_model_is_unsure(self):
        img = self.block(0.0)
        assert self.near(img, cp.TEAL) and self.near(img, cp.CORAL)

    def test_it_returns_the_next_free_y(self):
        img = Image.new("RGB", (300, 90))
        d = ImageDraw.Draw(img)
        y = cp.verdict_block(d, 0, 10, 280, 1.0,
                             cp.sg.rating("subway_safety"),
                             {"big": cp.font(17, True),
                              "num": cp.font(12, True)})
        assert y > 10


class TestGroupScaling:
    """`k` must follow the grid, or cell size and kept area move together."""

    def test_a_quarter_is_a_quarter_at_every_group(self):
        from research.monocle import caption_loop as cl
        for g in (1, 2, 4):
            assert cl.k_for(g, 0.25) / cl.cells_for(g) == pytest.approx(0.25)

    def test_the_cell_count_falls_with_the_square_of_the_group(self):
        from research.monocle import caption_loop as cl
        assert [cl.cells_for(g) for g in (1, 2, 4, 8, 16)] == \
            [256, 64, 16, 4, 1]

    def test_a_fixed_k_means_different_areas(self):
        """This is the trap the fraction exists to avoid."""
        from research.monocle import caption_loop as cl
        assert cl.k_for(2, 0.25, fixed=16) == 16      # a quarter of 64
        assert cl.k_for(4, 0.25, fixed=16) == 16      # ALL of 16

    def test_it_refuses_a_k_larger_than_the_grid(self):
        from research.monocle import caption_loop as cl
        with pytest.raises(ValueError, match="more than the 16 cells"):
            cl.k_for(4, 0.25, fixed=17)

    def test_it_refuses_a_group_that_does_not_divide(self):
        from research.monocle import caption_loop as cl
        with pytest.raises(ValueError, match="does not divide"):
            cl.cells_for(3)

    def test_a_coarse_grid_still_keeps_one_cell(self):
        from research.monocle import caption_loop as cl
        assert cl.k_for(8, 0.25) == 1

    def test_it_refuses_a_fraction_outside_the_range(self):
        from research.monocle import caption_loop as cl
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="keep-fraction"):
                cl.k_for(2, bad)

    def test_a_fixed_k_cannot_serve_several_groups(self):
        from research.monocle import caption_loop as cl
        with pytest.raises(SystemExit, match="cannot serve several groups"):
            cl.main(["--group", "1", "2", "--k", "8"])
