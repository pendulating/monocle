"""The label and question overrides must reach the prompt and the readout."""
from __future__ import annotations

import pytest

from research.monocle import single as sg  # noqa: E402


def test_the_case_pair_is_in_the_prompt():
    t = sg.prompt_text("subway_safety")
    assert "- Safe" in t and "- Dangerous" in t


def test_a_label_override_rewrites_the_prompt():
    t = sg.prompt_text("subway_safety",
                       {"positive": "Good", "negative": "Poor"})
    assert "- Good" in t and "- Poor" in t
    assert "Safe" not in t and "Dangerous" not in t


def test_a_label_override_keeps_the_question():
    t = sg.prompt_text("subway_safety",
                       {"positive": "Good", "negative": "Poor"})
    assert "How safe does this station entrance look?" in t


def test_a_question_override_rewrites_the_question():
    q = "How well lit is this station entrance?"
    t = sg.prompt_text("subway_safety", {"question": q})
    assert q in t
    assert "How safe does this station entrance look?" not in t


def test_both_overrides_compose():
    q = "How well lit is this station entrance?"
    t = sg.prompt_text("subway_safety",
                       {"question": q, "positive": "Good", "negative": "Poor"})
    assert q in t and "- Good" in t and "- Poor" in t


def test_the_case_config_is_not_changed_by_an_override():
    sg.prompt_text("subway_safety", {"positive": "Good", "negative": "Poor"})
    assert "- Safe" in sg.prompt_text("subway_safety")


@pytest.mark.parametrize("case", ["restaurants", "libraries"])
def test_every_case_keeps_its_own_pair(case):
    t = sg.prompt_text(case)
    assert "- Good" in t and "- Poor" in t
