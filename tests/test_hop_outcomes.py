import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/f10"))

from hop_outcomes import outcome  # noqa: E402


def test_each_form_of_a_verdict_is_a_verdict():
    assert [outcome(t) for t in ("Yes", "yes", " yes", " Yes")] == ["yes"] * 4
    assert [outcome(t) for t in ("No", "no", " no", " No")] == ["no"] * 4


def test_any_other_first_token_is_unable():
    assert [outcome(t) for t in ("Subject", "Unable", "I", "Whether", "")] == ["unable"] * 5
