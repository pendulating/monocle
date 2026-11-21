"""The label token contract. See monocle/labels.py."""
import pytest

from monocle import labels as lb


class StubTokenizer:
    """Splits on a fixed table, so the tests need no model on disk."""

    TABLE = {
        "Safe": [100], "Neutral": [101], "Dangerous": [102],
        "Unsafe": [7, 100],                 # Un + safe
        "Very unsafe": [9, 200],            # Very + _unsafe
        "Very safe": [9, 201],              # Very + _safe, collides on `Very`
        "More": [300], "Less": [301],
        "MuchMore": [9, 300], "MuchLess": [9, 301],
        "1": [11], "7": [17],
        # Two distinct strings that normalize onto ONE token. Rare, but it is
        # the only way a single-token label set can still collide.
        "Safe ": [100],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self.TABLE[text])


@pytest.fixture
def tk():
    return StubTokenizer()


class TestContractHolds:
    def test_accepts_the_three_point_word_scale(self, tk):
        got = lb.check_label_tokens(tk, ["Safe", "Neutral", "Dangerous"])
        assert got == {"Safe": 100, "Neutral": 101, "Dangerous": 102}

    def test_accepts_the_pairwise_contrast_poles(self, tk):
        assert lb.check_label_tokens(tk, ["More", "Less"]) == {
            "More": 300, "Less": 301}

    def test_encodes_without_a_leading_space(self, tk):
        """The forced prefix ends in a quote, so the label starts a string."""
        assert lb.label_token_ids(tk, ["Safe"])["Safe"] == [100]


class TestContractFails:
    def test_rejects_a_multi_token_label(self, tk):
        with pytest.raises(ValueError, match="not 1 token"):
            lb.check_label_tokens(tk, ["Safe", "Unsafe"])

    def test_rejects_the_seven_point_word_scale(self, tk):
        """`Very safe` and `Very unsafe` are the reason we use single words."""
        with pytest.raises(ValueError, match="not 1 token"):
            lb.check_label_tokens(tk, ["Very safe", "Very unsafe"])

    def test_rejects_the_pairwise_ordinal_labels(self, tk):
        """`MuchMore` and `MuchLess` are multi-token AND share `Much`.

        The length rule fires first, which is why the pairwise contrast uses
        `More` and `Less` instead of the outer 2 points of the scale.
        """
        with pytest.raises(ValueError, match="not 1 token"):
            lb.check_label_tokens(tk, ["MuchMore", "MuchLess"])

    def test_rejects_two_labels_that_encode_to_one_token(self, tk):
        with pytest.raises(ValueError, match="share token id"):
            lb.check_label_tokens(tk, ["Safe", "Safe "])

    def test_rejects_duplicates(self, tk):
        with pytest.raises(ValueError, match="duplicate"):
            lb.check_label_tokens(tk, ["Safe", "Safe"])

    def test_names_the_offending_label(self, tk):
        with pytest.raises(ValueError, match="Unsafe"):
            lb.check_label_tokens(tk, ["Safe", "Unsafe"])
