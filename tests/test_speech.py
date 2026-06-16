"""Unit tests for the urbanspeech anti-hallucination logic.

Covers the two GPU-free pieces that fix granite-speech hallucinating filler
over non-speech audio:
  - VAD speech-window packing (dagspaces.urbanspeech.stages.vad.pack_segments)
  - the hallucination post-filter (asr._make_hallucination_flagger)
"""

import pytest
from omegaconf import OmegaConf

from dagspaces.urbanspeech.stages.vad import pack_segments, total_speech_seconds
from dagspaces.urbanspeech.stages.asr import (
    _make_hallucination_flagger,
    _normalize_for_match,
    _segment_bounds,
)


class TestPackSegments:
    def test_empty(self):
        assert pack_segments([], 30.0) == []

    def test_merge_within_gap(self):
        # Two segments 1s apart merge when join_gap_s >= 1.
        assert pack_segments([(0.0, 5.0), (6.0, 9.0)], 30.0, join_gap_s=2.0) == [(0.0, 9.0)]

    def test_keep_separate_beyond_gap(self):
        # A large silence gap keeps windows separate (non-speech not bridged).
        out = pack_segments([(0.0, 5.0), (40.0, 50.0)], 30.0, join_gap_s=2.0)
        assert out == [(0.0, 5.0), (40.0, 50.0)]

    def test_split_long_segment(self):
        # A segment longer than chunk_seconds is sliced into chunk windows.
        out = pack_segments([(0.0, 75.0)], 30.0)
        assert out == [(0.0, 30.0), (30.0, 60.0), (60.0, 75.0)]

    def test_window_never_exceeds_chunk(self):
        out = pack_segments([(0.0, 10.0), (10.5, 25.0), (25.2, 40.0)], 30.0, join_gap_s=1.0)
        assert all(end - start <= 30.0 + 1e-6 for start, end in out)

    def test_total_speech_seconds(self):
        assert total_speech_seconds([(0.0, 5.0), (10.0, 12.5)]) == pytest.approx(7.5)


class TestSegmentBounds:
    def test_converts_seconds_to_samples_and_clamps(self):
        # 16 kHz; a window past the end clamps to n_samples.
        bounds = _segment_bounds([(0.0, 1.0), (1.5, 100.0)], 16000, 16000 * 2, 30.0, 1.0)
        assert bounds[0] == (0, 16000)
        # second window clamped to total length (2 s = 32000 samples)
        assert bounds[-1][1] == 32000

    def test_drops_zero_length(self):
        assert _segment_bounds([(1.0, 1.0)], 16000, 16000 * 5, 30.0, 1.0) == []


def _flagger(**kw):
    cfg = OmegaConf.create({"asr": {"hallucination_filter": {"enabled": True, **kw}}})
    return _make_hallucination_flagger(cfg)


class TestHallucinationFilter:
    def test_disabled_by_default_config(self):
        cfg = OmegaConf.create({"asr": {}})
        enabled, flag = _make_hallucination_flagger(cfg)
        assert enabled is False
        assert flag("Thank you for watching") is False

    @pytest.mark.parametrize("text", [
        "Thank you.",
        "Thank you very much.",
        "Thank you very much for watching.",
        "Thank you for watching!",
        "Thank you. Thank you. Thank you very much for watching.",
        "Thank you very much for your attention today, ladies and gentlemen.",
    ])
    def test_flags_filler(self, text):
        _, flag = _flagger()
        assert flag(text) is True

    @pytest.mark.parametrize("text", [
        "No, no automatic. You see it come up. And what are they for?",
        "Yeah, so we're reporters curious but good luck with it. Thanks a lot for watching.",
        "Because you are researchers, yeah? You do things ethically.",
        "Happy birthday!",
    ])
    def test_spares_real_speech(self, text):
        _, flag = _flagger()
        assert flag(text) is False

    def test_empty_not_flagged(self):
        _, flag = _flagger()
        assert flag("") is False

    def test_custom_phrase_list(self):
        _, flag = _flagger(phrases=["subscribe to the channel"])
        assert flag("Subscribe to the channel!") is True
        assert flag("Thank you very much") is False  # not in custom list


class TestNormalize:
    def test_strips_punctuation_and_lowercases(self):
        assert _normalize_for_match("Thank You, Sir!") == "thank you sir"
