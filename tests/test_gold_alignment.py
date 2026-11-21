"""Tests for the gold-set alignment measures: nested spans, the character
agreement, the ledger, the coverage, and the decisions."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "cvpr"))
import _gold_alignment as G  # noqa: E402

TEXT = "Image A shows a bank. It looks like a wealthy area. So A is More."
#       0         1         2         3         4         5         6
#       0123456789012345678901234567890123456789012345678901234567890123456


def spans(rows, **extra):
    df = pd.DataFrame(rows, columns=["start", "end", "cls"])
    df["text"] = [TEXT[s:e] for s, e in zip(df.start, df.end)]
    df["region_id"] = [f"r{i}" for i in range(len(df))]
    df["attributes"] = [{} for _ in range(len(df))]
    for k, v in extra.items():
        df[k] = v
    return df


class TestExport:
    def test_two_labels_on_one_region_give_two_rows(self):
        results = [
            {"id": "r1", "type": "labels", "from_name": "span", "to_name": "trace",
             "value": {"start": 22, "end": 51, "text": TEXT[22:51], "labels": ["inference", "comparison"]}},
        ]
        df = G.spans_from_results(results, TEXT)
        assert list(df.cls) == ["inference", "comparison"]
        assert set(df.region_id) == {"r1"}

    def test_white_space_at_the_ends_is_trimmed(self):
        results = [
            {"id": "r1", "type": "labels", "from_name": "span", "to_name": "trace",
             "value": {"start": 21, "end": 52, "text": TEXT[21:52], "labels": ["inference"]}},
        ]
        df = G.spans_from_results(results, TEXT)
        assert (df.start.iloc[0], df.end.iloc[0]) == (22, 51)
        assert df.text.iloc[0] == "It looks like a wealthy area."

    def test_choices_attach_by_region_id(self):
        results = [
            {"id": "r1", "type": "labels", "from_name": "span", "to_name": "trace",
             "value": {"start": 0, "end": 20, "text": TEXT[0:20], "labels": ["visual_evidence"]}},
            {"id": "r1", "type": "choices", "from_name": "image", "to_name": "trace",
             "value": {"start": 0, "end": 20, "choices": ["A"]}},
        ]
        df = G.spans_from_results(results, TEXT)
        assert df.attributes.iloc[0] == {"image": "A"}


class TestCharacters:
    def test_a_nested_character_counts_in_both_classes(self):
        human = spans([(22, 51, "inference"), (38, 50, "visual_evidence")])
        masks = G.char_masks(human, len(TEXT))
        assert masks["inference"][40] and masks["visual_evidence"][40]

    def test_a_phrase_inside_the_clause_has_full_precision_and_partial_recall(self):
        human = spans([(22, 51, "inference")])
        pred = spans([(38, 50, "inference")])
        t = G.char_prf(human, pred, len(TEXT)).set_index("cls")
        assert t.loc["inference", "precision"] == 1.0
        assert t.loc["inference", "recall"] == pytest.approx(12 / 29)

    def test_all_row_is_micro_over_classes(self):
        human = spans([(0, 20, "visual_evidence"), (22, 51, "inference")])
        pred = spans([(0, 20, "visual_evidence")])
        t = G.char_prf(human, pred, len(TEXT)).set_index("cls")
        assert t.loc["all", "recall"] == pytest.approx(20 / 49)


class TestLedger:
    def test_statuses(self):
        human = spans([(0, 20, "visual_evidence"), (22, 51, "inference"), (53, 66, "decision")])
        pred = spans([(0, 20, "visual_evidence"),     # exact
                      (38, 50, "inference"),          # inside
                      (22, 66, "inference"),          # contains
                      (10, 30, "visual_evidence"),    # partial
                      (53, 66, "comparison"),         # other_class
                      ])
        led = G.ledger(human, pred)
        assert list(led.status) == ["exact", "inside", "contains", "partial", "other_class"]
        assert led.human_cls.iloc[-1] == "decision"

    def test_innermost_class_wins_the_confusion(self):
        human = spans([(22, 51, "inference"), (38, 50, "visual_evidence")])
        pred = spans([(38, 50, "visual_evidence")])
        led = G.ledger(human, pred)
        assert led.status.iloc[0] == "exact"
        conf = G.confusion(led.assign(arm="a"))
        assert conf.loc["visual_evidence", "visual_evidence"] == 1

    def test_a_span_with_no_human_is_none(self):
        human = spans([(0, 20, "visual_evidence")])
        pred = spans([(53, 66, "decision")])
        assert G.ledger(human, pred).status.iloc[0] == "none"


class TestCoverage:
    def test_hit_touched_missed(self):
        human = spans([(0, 20, "visual_evidence"), (22, 51, "inference"), (53, 66, "decision")])
        pred = spans([(0, 15, "visual_evidence"), (48, 51, "inference")])
        cov = G.human_coverage(human, pred)
        assert list(cov.status) == ["hit", "touched", "missed"]


class TestNesting:
    def test_pairs_and_the_side_the_extractor_took(self):
        human = spans([(22, 51, "inference"), (38, 50, "visual_evidence"), (22, 51, "comparison")],
                      task_id="t", annotation_id=1)
        pairs = G.nested_pairs(human)
        kinds = set(zip(pairs.outer_cls, pairs.inner_cls, pairs.co_labelled))
        assert ("inference", "visual_evidence", False) in kinds
        assert ("comparison", "visual_evidence", False) in kinds
        assert ("comparison", "inference", True) in kinds
        assert len(pairs) == 3
        pred = spans([(38, 50, "visual_evidence")], task_id="t", arm="a")
        took = G.nesting_by_arm(pairs, pred)
        assert set(took.took) == {"inner", "neither"}


class TestDecisions:
    def test_last_decision_and_earlier_ones(self):
        human = spans([(22, 51, "decision"), (53, 66, "decision")], task_id="t", annotation_id=1)
        pred = pd.concat([
            spans([(22, 51, "decision")], task_id="t", arm="early"),
            spans([(53, 66, "decision")], task_id="t", arm="last"),
        ], ignore_index=True)
        d = G.decisions(human, pred).set_index("arm")
        assert d.loc["early", "last_matches"] == False  # noqa: E712
        assert d.loc["early", "earlier_matched"] == 1
        assert d.loc["last", "last_matches"] == True  # noqa: E712
        assert d.loc["last", "human_decisions"] == 2


class TestView:
    def test_render_handles_nested_and_co_labelled_spans(self):
        human = spans([(22, 51, "inference"), (38, 50, "visual_evidence"), (22, 51, "comparison")])
        pred = spans([(38, 50, "visual_evidence")])
        html = G.render_trace(TEXT, human, pred)
        assert "wealthy area" in html
        assert html.count("<span") >= 3


class TestPositional:
    TEXT2 = "**Image A Analysis:**\n- Image A shows a clean street.\n- It looks safe."

    def test_scaffold_mask_covers_headers_names_and_markers(self):
        m = G.scaffold_mask(self.TEXT2)
        t = self.TEXT2
        assert m[t.index("**Image A Analysis:**")]
        assert m[t.index("Image A shows")]
        assert not m[t.index("shows")]
        assert m[t.index("- It")]
        assert not m[t.index("safe")]

    def test_a_phrase_inside_a_clause_matches_by_position(self):
        t = self.TEXT2
        m = G.scaffold_mask(t)
        s, e = t.index("- Image A shows a clean street."), t.index("street.") + len("street.")
        human = spans([(s, e, "visual_evidence")])
        human["text"] = t[s:e]
        ps, pe = t.index("clean street"), t.index("clean street") + len("clean street")
        pred = spans([(ps, pe, "visual_evidence")])
        pairs, hm, pm = G.positional_match(human, pred, m, 0.5)
        assert len(pairs) == 1 and pairs[0][2] == 1.0

    def test_the_class_gate_and_the_any_class_row(self):
        t = self.TEXT2
        m = G.scaffold_mask(t)
        s, e = t.index("It looks safe."), len(t)
        human = spans([(s, e, "inference")])
        pred = spans([(s, e, "visual_evidence")])
        prf = G.positional_prf(human, pred, m, 0.5).set_index("cls")
        assert prf.loc["all", "matched"] == 0
        assert prf.loc["any_class", "matched"] == 1

    def test_the_scaffold_alone_never_matches(self):
        t = self.TEXT2
        m = G.scaffold_mask(t)
        s, e = 0, len("**Image A Analysis:**")
        human = spans([(s, e, "visual_evidence")])
        pred = spans([(s, e, "visual_evidence")])
        pairs, _, _ = G.positional_match(human, pred, m, 0.5)
        assert pairs == []


class TestTaskLevelFields:
    def test_notes_and_checks_are_read_from_the_first_annotation(self):
        anns = [{"result": [
            {"type": "textarea", "from_name": "notes", "to_name": "trace", "value": {"text": ["Image B shows no library."]}},
            {"type": "choices", "from_name": "shows_a", "to_name": "image_a", "value": {"choices": ["yes"]}},
        ]}]
        out = G.image_checks(anns)
        assert out == {"shows_a": "yes", "shows_b": None, "notes": "Image B shows no library."}
