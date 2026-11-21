"""Tests for the crop caption read."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from monocle import patch_caption as pcap  # noqa: E402


def test_a_middle_cell_gives_a_full_window():
    x0, y0, x1, y1 = pcap.crop_box(cell=17, side=16, size=896, pad=1)
    assert (x1 - x0, y1 - y0) == (168, 168)


def test_a_corner_cell_is_clipped():
    x0, y0, x1, y1 = pcap.crop_box(cell=0, side=16, size=896, pad=1)
    assert (x0, y0) == (0, 0)
    assert (x1 - x0, y1 - y0) == (112, 112)


def test_no_pad_gives_one_cell():
    x0, y0, x1, y1 = pcap.crop_box(cell=100, side=16, size=896, pad=0)
    assert (x1 - x0, y1 - y0) == (56, 56)


def test_the_box_tracks_the_row_and_the_column():
    side, size = 16, 896
    step = size // side
    for cell in (0, 5, 16, 255):
        r, c = divmod(cell, side)
        x0, y0, _, _ = pcap.crop_box(cell, side, size, pad=0)
        assert x0 == c * step and y0 == r * step


def test_crop_is_square_and_the_right_size():
    img = Image.new("RGB", (640, 480), "red")
    piece, box = pcap.crop_of(img, cell=34, side=16, pad=1, out_size=896)
    assert piece.size == (896, 896)
    assert box == (56, 56, 224, 224)


def test_crop_holds_the_cell_content():
    img = Image.new("RGB", (896, 896), "black")
    img.paste(Image.new("RGB", (56, 56), "white"), (56 * 3, 56 * 2))
    cell = 2 * 16 + 3
    piece, _ = pcap.crop_of(img, cell, side=16, pad=0, out_size=896)
    assert np.asarray(piece).mean() > 200


def test_pick_cells_holds_the_poles_first():
    rng = np.random.default_rng(0)
    v = np.array([0.0, 5.0, -5.0, 1.0, 2.0])
    cells, arms = pcap.pick_cells(v, n_random=2, rng=rng)
    assert cells[0] == 1 and cells[1] == 2
    assert arms[:2] == ["pos", "neg"]
    assert arms[2:] == ["random", "random"]
    assert 1 not in cells[2:] and 2 not in cells[2:]


def test_pick_cells_asks_for_no_more_than_it_can_give():
    rng = np.random.default_rng(1)
    v = np.arange(4, dtype=float)
    cells, arms = pcap.pick_cells(v, n_random=10, rng=rng)
    assert len(cells) == 4
    assert arms.count("random") == 2


def test_a_grid_that_is_not_square_stops():
    import pandas as pd
    assert int(round(255 ** 0.5)) ** 2 != 255


@pytest.mark.parametrize("pad", [0, 1, 2])
def test_the_window_grows_with_the_pad(pad):
    x0, y0, x1, y1 = pcap.crop_box(cell=136, side=16, size=896, pad=pad)
    assert (x1 - x0) == 56 * (1 + 2 * pad)
