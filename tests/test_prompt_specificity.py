"""Is the map about the question? See monocle/prompt_specificity.py."""
import numpy as np
import pytest

from monocle import prompt_specificity as ps


class TestPromptSpec:
    def test_a_real_case_comes_from_the_shared_table(self):
        assert ps.prompt_spec("subway_safety")["positive"] == "Safe"

    def test_the_control_is_not_a_case(self):
        """`daylight` must not leak into the 7-case battery."""
        from monocle import single as sg
        assert "daylight" not in sg.CASE_RATING

    def test_the_control_asks_nothing_about_quality(self):
        s = ps.prompt_spec("daylight")
        assert s["positive"] == "Day" and s["negative"] == "Night"

    def test_an_unknown_prompt_raises(self):
        with pytest.raises(ValueError, match="unknown case"):
            ps.prompt_spec("not_a_prompt")

    def test_the_reference_is_first(self):
        assert ps.DEFAULT_PROMPTS[0] == "subway_safety"

    def test_the_control_is_in_the_default_set(self):
        assert "daylight" in ps.DEFAULT_PROMPTS

    def test_the_image_pool_is_fixed(self):
        """The comparison moves the question, never the photograph."""
        assert ps.IMAGE_CASE == "subway_safety"


class TestCompareMaps:
    def test_identical_maps_correlate_perfectly(self):
        v = np.random.default_rng(0).normal(size=64)
        out = ps.compare_maps({"a": v, "b": v.copy()})
        assert out[0]["pearson"] == pytest.approx(1.0)
        assert out[0]["top_overlap"] == pytest.approx(1.0)

    def test_a_flipped_map_anticorrelates(self):
        v = np.random.default_rng(0).normal(size=64)
        out = ps.compare_maps({"a": v, "b": -v})
        assert out[0]["pearson"] == pytest.approx(-1.0)
        assert out[0]["top_overlap"] == pytest.approx(0.0)

    def test_unrelated_maps_sit_near_zero(self):
        rng = np.random.default_rng(1)
        out = ps.compare_maps({"a": rng.normal(size=256),
                               "b": rng.normal(size=256)})
        assert abs(out[0]["pearson"]) < 0.25

    def test_every_pair_appears_once(self):
        v = np.zeros(64) + np.arange(64)
        out = ps.compare_maps({"a": v, "b": v, "c": v})
        assert len(out) == 3

    def test_the_pair_names_are_sorted(self):
        v = np.arange(64.0)
        out = ps.compare_maps({"zeta": v, "alpha": v})
        assert out[0]["prompt_a"] == "alpha"

    def test_the_top_and_bottom_are_read_separately(self):
        """A map may agree on what is safe and not on what is dangerous."""
        v = np.arange(64.0)
        w = v.copy()
        w[0:8], w[8:16] = v[8:16], v[0:8]   # move the lowest 8 cells only
        out = ps.compare_maps({"a": v, "b": w})
        assert out[0]["top_overlap"] == pytest.approx(1.0)
        assert out[0]["bottom_overlap"] == pytest.approx(0.0)
