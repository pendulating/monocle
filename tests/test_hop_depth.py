import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/scripts/f10"))
from research.scripts.f10.build_depth_sample import stratified  # noqa: E402
from research.scripts.f10.hop_depth import locate  # noqa: E402


def test_stratified_takes_3_or_all():
    pool = pd.DataFrame({"image_idx": range(9), "nta2020": ["a"] * 5 + ["b"] * 3 + ["c"]})
    assert stratified(pool).groupby("nta2020").size().to_dict() == {"a": 3, "b": 3, "c": 1}


def test_stratified_is_fixed_by_the_seed():
    pool = pd.DataFrame({"image_idx": range(40), "nta2020": ["a", "b"] * 20})
    assert stratified(pool).equals(stratified(pool.sample(frac=1, random_state=1)))


def test_locate_with_image():
    # bos, image, image, delimiter, Is, this, place, cool, ?, end, final
    seq = [1, 9, 9, 8, 20, 21, 22, 23, 24, 2, 3]
    at = locate(seq, 9, 20, 22, 23, 24)
    assert at == {"after_image": 3, "first_text": 4, "place": 6, "after_place": 7, "after_cool": 8, "final": 10}


def test_locate_with_no_image_has_no_delimiter():
    assert "after_image" not in locate([1, 20, 21, 22, 23, 24, 3], 9, 20, 22, 23, 24)


def test_locate_stops_on_other_tokens():
    with pytest.raises(RuntimeError):
        locate([1, 20, 22, 99, 24, 3], None, 20, 22, 23, 24)


def test_ask_changes_the_noun_and_the_adjective():
    from research.scripts.f10.hop_depth import QUESTION, ask
    assert ask() == QUESTION
    assert ask("ugly", "scene") == "Is this scene ugly? Answer with one word: yes or no."
