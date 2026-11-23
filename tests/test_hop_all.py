"""Tests for the split of a prompt in `hop_all.py`."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/scripts/f10"))
from research.scripts.f10.hop_all import align_text, parts, tail_names  # noqa: E402


def test_parts_split_the_prompt():
    assert parts([1, 2, 9, 9, 9, 3, 4], 9) == ([0, 1], [2, 3, 4], [5, 6])


def test_parts_stop_on_patches_that_are_not_1_block():
    with pytest.raises(RuntimeError):
        parts([9, 1, 9, 2], 9)


def test_tail_names_keep_the_old_names():
    assert tail_names([5, 6, 7], {"after_image": 5, "final": 7}) == ["after_image", "t01", "final"]


def test_align_text_gives_the_tail_index_of_each_text_token():
    assert align_text([8, 4, 5, 6], [1, 4, 5, 6], 4) == [1, 2, 3]


def test_align_text_stops_when_the_texts_differ():
    with pytest.raises(RuntimeError):
        align_text([8, 4, 5, 6], [4, 5, 7], 4)
