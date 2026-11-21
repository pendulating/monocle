"""Tests for the gold-set scorer: the span matcher and the F1."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "score_gold_set", Path(__file__).resolve().parent.parent / "scripts" / "score_gold_set.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def span(start, end, cls, **attrs):
    return {"start": start, "end": end, "class": cls, "text": "", "id": None, "attrs": attrs}


class TestMatcher:
    def test_same_class_and_half_overlap_match(self):
        pred = [span(0, 10, "visual_evidence")]
        gold = [span(2, 12, "visual_evidence")]
        assert len(S.match_spans(pred, gold, 0.5)) == 1

    def test_another_class_does_not_match(self):
        assert S.match_spans([span(0, 10, "inference")], [span(0, 10, "visual_evidence")], 0.5) == []

    def test_one_gold_span_matches_once(self):
        pred = [span(0, 10, "decision"), span(1, 9, "decision")]
        gold = [span(0, 10, "decision")]
        r = S.prf(pred, gold, 0.5)
        assert r["tp"] == 1 and r["precision"] == 0.5 and r["recall"] == 1.0

    def test_attribute_agreement_counts_shared_names_only(self):
        pred = [span(0, 10, "visual_evidence", image="A", valence="good")]
        gold = [span(0, 10, "visual_evidence", image="A", valence="bad", category="upkeep")]
        agree = S.attribute_agreement(S.match_spans(pred, gold, 0.5))
        assert agree == {"image": (1, 1), "valence": (0, 1)}


class TestResults:
    def test_choices_attach_to_their_region(self):
        results = [
            {"id": "r1", "type": "labels", "from_name": "span", "to_name": "trace",
             "value": {"start": 5, "end": 14, "text": "trash can", "labels": ["visual_evidence"]}},
            {"id": "r1", "type": "choices", "from_name": "image", "to_name": "trace",
             "value": {"start": 5, "end": 14, "choices": ["B"]}},
        ]
        spans = S.spans_from_results(results)
        assert spans == [{"start": 5, "end": 14, "class": "visual_evidence", "text": "trash can",
                          "id": "r1", "attrs": {"image": "B"}}]
