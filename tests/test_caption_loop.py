"""The caption loop. See research/monocle/caption_loop.py."""
import numpy as np
import pytest

from research.monocle import caption_loop as cl


class TestKeepOnly:
    def blocks(self, n=4):
        return [[i * 4 + j for j in range(4)] for i in range(n)]

    def test_it_removes_the_complement(self):
        """Keeping cell 0 means removing the patches of cells 1, 2 and 3."""
        out = cl.keep_only_patches([0], self.blocks(), 4)
        assert out == list(range(4, 16))

    def test_keeping_everything_removes_nothing(self):
        assert cl.keep_only_patches(range(4), self.blocks(), 4) == []

    def test_keeping_nothing_removes_everything(self):
        assert len(cl.keep_only_patches([], self.blocks(), 4)) == 16

    def test_two_kept_cells_leave_two_removed(self):
        out = cl.keep_only_patches([1, 2], self.blocks(), 4)
        assert sorted(out) == list(range(0, 4)) + list(range(12, 16))


class TestDescribedText:
    def test_the_caption_is_quoted_into_the_question(self):
        t = cl.described_text("subway_safety", "A dark stairwell.")
        assert '"A dark stairwell."' in t

    def test_the_rating_question_survives(self):
        t = cl.described_text("subway_safety", "x")
        assert "Safe" in t and "Dangerous" in t

    def test_whitespace_around_the_caption_is_trimmed(self):
        t = cl.described_text("subway_safety", "  hello  \n")
        assert '"hello"' in t

    def test_the_description_comes_first(self):
        t = cl.described_text("subway_safety", "zzz")
        assert t.index("zzz") < t.index("Dangerous")


class TestContract:
    def test_the_caption_question_is_open(self):
        """A closed question would put the label back in the prompt."""
        q = cl.CAPTION_QUESTION.lower()
        assert "safe" not in q and "dangerous" not in q

    def test_the_default_keeps_a_quarter_of_the_picture(self):
        assert cl.DEFAULT_K == 16 and cl.DEFAULT_GROUP == 2

    def test_a_caption_has_room_to_say_something(self):
        assert cl.MAX_NEW >= 24
