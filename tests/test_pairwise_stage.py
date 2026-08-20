"""Tests for the urbanpairvqa stage logic — label canonicalization, ordinal
scoring, and the optional 'Not sure' (abstention) toggle.

The abstention option is OFF by default and is enabled per run/sweep via
``prompt.structured_output.allow_not_sure=true``. When on it (a) appends the
configured label to the guided-decoding answer enum, (b) appends a guidance
line to the rendered prompt, and (c) scores the abstention as NaN — excluded
from the ordinal scale rather than collapsed into "Same".
"""
from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf

from dagspaces.urbanpairvqa.stages.pairwise_vqa import (
    _augment_schema_with_not_sure,
    _canonicalize_label,
    _derive_labels,
    _make_pairwise_preprocess,
    _not_sure_enabled,
    _not_sure_label,
    _render_pair_prompt,
)


_BASE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string",
                   "enum": ["MuchLess", "Less", "Same", "More", "MuchMore"]},
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def _cfg(allow_not_sure: bool, *, label: str = "NotSure"):
    return OmegaConf.create({
        "prompt": {
            "system": "S",
            "user_template": "Compare A and B.",
            "structured_output": {
                "enabled": True,
                "allow_not_sure": allow_not_sure,
                "not_sure_label": label,
                "json_schema": _BASE_SCHEMA,
            },
        },
        "sampling_params_vqa": {"temperature": 0.0, "stop": []},
    })


class TestCanonicalize:
    @pytest.mark.parametrize("raw,expected", [
        ("MuchMore", "MuchMore"),
        ("much more", "MuchMore"),
        ("less", "Less"),
        ("Same", "Same"),
        ("equal", "Same"),
        ("", "Same"),
        ("totally unparseable", "Same"),
    ])
    def test_ordinal(self, raw, expected):
        assert _canonicalize_label(raw) == expected

    @pytest.mark.parametrize("raw", [
        "NotSure", "Not sure", "not_sure", "unsure", "I am not sure",
        '{"answer": "NotSure"}', "cannot tell",
    ])
    def test_abstention_recognized(self, raw):
        assert _canonicalize_label(raw) == "NotSure"

    def test_custom_label_recognized(self):
        # A configured label string maps to the canonical NotSure token.
        assert _canonicalize_label("Cannot determine",
                                   not_sure_label="Cannot determine") == "NotSure"


class TestScoring:
    def test_no_abstention_stays_int(self):
        df = pd.DataFrame({"answer": ["MuchMore", "Less", "Same"],
                           "is_swapped": [False, True, False]})
        out = _derive_labels(df)
        assert str(out["relative_score"].dtype) == "int64"
        # Swap inverts: Less (-1) presented on a swapped pair -> More (+1).
        assert out.loc[1, "relative_label"] == "More"
        assert out.loc[1, "relative_score"] == 1

    def test_abstention_scores_nan(self):
        df = pd.DataFrame({"answer": ["MuchMore", "Not sure", "Same"],
                           "is_swapped": [False, True, False]})
        out = _derive_labels(df)
        assert str(out["relative_score"].dtype) == "float64"
        assert out.loc[1, "relative_label"] == "NotSure"
        assert pd.isna(out.loc[1, "relative_score"])
        assert out.loc[0, "relative_score"] == 2.0
        # Downstream coerce + dropna naturally drops the abstention.
        coerced = pd.to_numeric(out["relative_score"], errors="coerce")
        assert int(coerced.notna().sum()) == 2

    def test_abstention_is_swap_symmetric(self):
        # Swapping A/B must not resolve the uncertainty.
        df = pd.DataFrame({"answer": ["NotSure"], "is_swapped": [True]})
        out = _derive_labels(df)
        assert out.loc[0, "relative_label"] == "NotSure"


class TestSchemaAugmentation:
    def test_appends_label(self):
        schema = {"properties": {"answer": {"enum": ["MuchLess", "Same"]}}}
        out = _augment_schema_with_not_sure(schema, "NotSure")
        assert out["properties"]["answer"]["enum"][-1] == "NotSure"

    def test_idempotent(self):
        schema = {"properties": {"answer": {"enum": ["Same", "NotSure"]}}}
        out = _augment_schema_with_not_sure(schema, "NotSure")
        assert out["properties"]["answer"]["enum"].count("NotSure") == 1

    def test_missing_enum_is_noop(self):
        schema = {"properties": {}}
        # Must not raise; leaves schema unchanged.
        assert _augment_schema_with_not_sure(schema, "NotSure") == {"properties": {}}


class TestToggleWiring:
    def test_helpers(self):
        assert _not_sure_enabled(_cfg(True)) is True
        assert _not_sure_enabled(_cfg(False)) is False
        assert _not_sure_label(_cfg(True, label="Unsure")) == "Unsure"

    def test_enum_toggle_in_preprocess(self):
        row = {"presented_left_path": "/a.jpg", "presented_right_path": "/b.jpg",
               "prompt": "P", "pair_id": "x"}
        enum_on = (_make_pairwise_preprocess(_cfg(True))(row)
                   ["sampling_params"]["guided_decoding"]["json"]
                   ["properties"]["answer"]["enum"])
        enum_off = (_make_pairwise_preprocess(_cfg(False))(row)
                    ["sampling_params"]["guided_decoding"]["json"]
                    ["properties"]["answer"]["enum"])
        assert "NotSure" in enum_on
        assert "NotSure" not in enum_off

    def test_prompt_guidance_toggle(self):
        assert "NotSure" in _render_pair_prompt({"pair_id": "x"}, _cfg(True))
        assert "NotSure" not in _render_pair_prompt({"pair_id": "x"}, _cfg(False))


class TestImageLayout:
    ROW = {"presented_left_path": "/a.jpg", "presented_right_path": "/b.jpg",
           "prompt": "P", "pair_id": "x"}

    @staticmethod
    def _content(layout):
        cfg = _cfg(False)
        if layout is not None:
            cfg.prompt.image_layout = layout
        return _make_pairwise_preprocess(cfg)(dict(TestImageLayout.ROW))["messages"][1]["content"]

    @staticmethod
    def _kinds(content):
        return [block["type"] for block in content]

    def test_default_is_images_then_text(self):
        for layout in (None, "images_then_text"):
            content = self._content(layout)
            assert self._kinds(content) == ["image_url", "image_url", "text"]
            assert content[0]["image_url"]["url"] == "file:///a.jpg"
            assert content[1]["image_url"]["url"] == "file:///b.jpg"
            assert content[2]["text"] == "P"

    def test_interleaved_labels(self):
        content = self._content("interleaved_labels")
        assert self._kinds(content) == ["text", "image_url", "text", "image_url", "text"]
        assert content[0]["text"] == "Image A:"
        assert content[1]["image_url"]["url"] == "file:///a.jpg"
        assert content[2]["text"] == "Image B:"
        assert content[3]["image_url"]["url"] == "file:///b.jpg"
        assert content[4]["text"] == "P"

    def test_text_first(self):
        content = self._content("text_first")
        assert self._kinds(content) == ["text", "image_url", "image_url"]
        assert content[0]["text"] == "P"
        assert content[1]["image_url"]["url"] == "file:///a.jpg"
        assert content[2]["image_url"]["url"] == "file:///b.jpg"

    def test_unknown_layout_raises(self):
        cfg = _cfg(False)
        cfg.prompt.image_layout = "side_by_side_collage"
        with pytest.raises(ValueError, match="image_layout"):
            _make_pairwise_preprocess(cfg)
