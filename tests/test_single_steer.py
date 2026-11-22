"""CPU tests for research/monocle/single_steer.py (the single-image steering rung).

These cover the pure helpers: the prompt text, the single-token word gate,
the polar scoring, the image dedup, and the summary shape. The GPU path runs
under research/monocle/monocle.sub.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO = Path("/share/ju/matt/mllmsci")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.monocle import single_steer  # noqa: E402

MODEL_DIR = Path(single_steer.MODEL_DIR_DEFAULT)


class FakeTokenizer:
    """encode() from a fixed word->ids table; unknown words split per char."""

    def __init__(self, table: dict[str, list[int]]):
        self.table = table

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text in self.table:
            return list(self.table[text])
        return [1000 + ord(c) for c in text]


# ---------------------------------------------------------------------------
# Word tables and prompt text
# ---------------------------------------------------------------------------
def test_every_case_has_words_subject_question():
    from research.monocle import canonical
    for case in canonical.CASES:
        assert case in single_steer.CASE_WORDS
        assert case in single_steer.CASE_SUBJECTS
        assert case in single_steer.CASE_QUESTIONS


def test_user_text_names_both_poles():
    for case, (pos, neg) in single_steer.CASE_WORDS.items():
        text = single_steer.user_text(case)
        assert f"- {pos}\n" in text
        assert f"- {neg}\n" in text
        assert single_steer.CASE_QUESTIONS[case] in text
        assert '"answer"' in text


def test_user_text_unknown_case_raises():
    with pytest.raises(ValueError):
        single_steer.user_text("not_a_case")


def test_question_words_match_label_words():
    """The question must use the same polar words the label list offers."""
    for case, (pos, neg) in single_steer.CASE_WORDS.items():
        q = single_steer.CASE_QUESTIONS[case]
        assert pos in q, f"{case}: question does not name {pos!r}"
        assert neg in q, f"{case}: question does not name {neg!r}"


# ---------------------------------------------------------------------------
# The single-token gate
# ---------------------------------------------------------------------------
def test_polar_token_ids_accepts_single_tokens():
    tok = FakeTokenizer({"safe": [11], "dangerous": [22]})
    assert single_steer.polar_token_ids(tok, "subway_safety") == (11, 22)


def test_polar_token_ids_rejects_multi_token():
    tok = FakeTokenizer({"safe": [11], "dangerous": [22, 23]})
    with pytest.raises(ValueError, match="single-token"):
        single_steer.polar_token_ids(tok, "subway_safety")


def test_polar_token_ids_rejects_collision():
    tok = FakeTokenizer({"safe": [11], "dangerous": [11]})
    with pytest.raises(ValueError, match="share token id"):
        single_steer.polar_token_ids(tok, "subway_safety")


@pytest.mark.skipif(not MODEL_DIR.is_dir(), reason="model dir not mounted")
def test_all_case_words_single_token_on_real_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    for case in single_steer.CASE_WORDS:
        pos_id, neg_id = single_steer.polar_token_ids(tok, case)
        assert pos_id != neg_id


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_score_polar_logits_math():
    logits = torch.zeros(50)
    logits[7] = 3.0
    logits[9] = 1.0
    m = single_steer.score_polar_logits(logits, 7, 9)
    assert m["logit_diff"] == pytest.approx(2.0)
    assert m["argmax_pos"] is True
    assert m["top_token_id"] == 7
    assert 0.0 < m["p_neg"] < m["p_pos"] < 1.0


def test_score_polar_logits_negative_pole():
    logits = torch.zeros(50)
    logits[9] = 5.0
    m = single_steer.score_polar_logits(logits, 7, 9)
    assert m["logit_diff"] == pytest.approx(-5.0)
    assert m["argmax_pos"] is False
    assert m["top_token_id"] == 9


# ---------------------------------------------------------------------------
# Image dedup
# ---------------------------------------------------------------------------
def test_unique_image_paths_order_and_dedup():
    rows = [
        {"presented_left_path": "a.jpg", "presented_right_path": "b.jpg"},
        {"presented_left_path": "b.jpg", "presented_right_path": "c.jpg"},
        {"presented_left_path": "", "presented_right_path": "a.jpg"},
    ]
    assert single_steer.unique_image_paths(rows) == ["a.jpg", "b.jpg", "c.jpg"]


def test_unique_image_paths_empty():
    assert single_steer.unique_image_paths([]) == []


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def test_summarize_groups_cells():
    df = pd.DataFrame([
        {"direction": "signal", "layer": 42, "alpha": 0.0,
         "logit_diff": 1.0, "argmax_pos": True, "p_pos": 0.6, "p_neg": 0.2,
         "top_token_id": 7},
        {"direction": "signal", "layer": 42, "alpha": 0.0,
         "logit_diff": -1.0, "argmax_pos": False, "p_pos": 0.2, "p_neg": 0.6,
         "top_token_id": 9},
        {"direction": "control", "layer": 42, "alpha": 2.0,
         "logit_diff": 0.5, "argmax_pos": True, "p_pos": 0.5, "p_neg": 0.3,
         "top_token_id": 7},
    ])
    s = single_steer.summarize(df)
    cell = s["signal/L42/a+0"]
    assert cell["n"] == 2
    assert cell["mean_logit_diff"] == pytest.approx(0.0)
    assert cell["frac_argmax_pos"] == pytest.approx(0.5)
    assert cell["n_top_tokens"] == 2
    assert "frac_on_contract" not in cell
    assert "control/L42/a+2" in s


def test_summarize_contract_rate():
    df = pd.DataFrame([
        {"direction": "signal", "layer": 42, "alpha": 2.0,
         "logit_diff": 1.0, "argmax_pos": True, "p_pos": 0.6, "p_neg": 0.2,
         "top_token_id": 7},
        {"direction": "signal", "layer": 42, "alpha": 2.0,
         "logit_diff": 1.0, "argmax_pos": True, "p_pos": 0.6, "p_neg": 0.2,
         "top_token_id": 71},   # the space-prefixed pos variant
        {"direction": "signal", "layer": 42, "alpha": 2.0,
         "logit_diff": 1.0, "argmax_pos": True, "p_pos": 0.6, "p_neg": 0.2,
         "top_token_id": 9},
        {"direction": "signal", "layer": 42, "alpha": 2.0,
         "logit_diff": 1.0, "argmax_pos": True, "p_pos": 0.6, "p_neg": 0.2,
         "top_token_id": 999},  # off contract
    ])
    s = single_steer.summarize(df, pos_ids={7, 71}, neg_ids={9})
    cell = s["signal/L42/a+2"]
    assert cell["frac_top_pos"] == pytest.approx(0.5)
    assert cell["frac_top_neg"] == pytest.approx(0.25)
    assert cell["frac_on_contract"] == pytest.approx(0.75)


def test_polar_variant_ids_includes_space_form():
    tok = FakeTokenizer({
        "safe": [11], " safe": [111],
        "dangerous": [22], " dangerous": [2, 2]})  # space form multi-token
    pos, neg = single_steer.polar_variant_ids(tok, "subway_safety")
    assert pos == {11, 111}
    assert neg == {22}


def test_polar_variant_ids_rejects_overlap():
    tok = FakeTokenizer({
        "safe": [11], " safe": [111],
        "dangerous": [22], " dangerous": [111]})
    with pytest.raises(ValueError, match="overlap"):
        single_steer.polar_variant_ids(tok, "subway_safety")
