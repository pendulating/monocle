"""Tests for the IC code derivation.

The codes decide what the paper says about reasoning complexity, thus each rung
of the scale gets a test that builds the smallest table that reaches it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from dagspaces.common import ic_codes as IC


def row(kind, *, name="", quote="q", found=True, attrs=None, index=0,
        doc="d1", truncated=False, case="subway_safety", label="More",
        char_start=0):
    """Build 1 ingredient row, in the shape `ic_schema.ingredient_rows` writes."""
    return {
        "case": case,
        "judge_model": "gemma-4-12b/instruct_thinking",
        "extractor_model": "qwen",
        "sweep": "looks_thinking_gemma12b",
        "pair_id": doc,
        "doc_id": doc,
        "relative_label": label,
        "presented_label": label,
        "schema_name": "ic_ingredients",
        "schema_version": "v2",
        "ingredient_type": kind,
        "ingredient_index": index,
        "name": name,
        "quote": quote,
        "char_start": char_start,
        "char_end": char_start + len(quote),
        "quote_method": "exact" if found else "none",
        "quote_found": found,
        "attrs_json": json.dumps(attrs or {}),
        "n_sub_quotes": 0,
        "n_sub_quotes_found": 0,
        "answer_truncated": truncated,
        "answer_repaired": False,
        "parse_error": "",
    }


def span(text="s", found=True):
    return {"text": text, "char_start": 0 if found else None,
            "char_end": 1 if found else None, "method": "exact" if found else "none"}


def dim(name, image="A", valence="good", **kw):
    return row("dimension", name=name,
               attrs={"type": "evaluative", "image": image, "valence": valence}, **kw)


def frame(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------- the rungs

def test_code_1_single_view():
    df = frame([dim("clean platform"),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.ic_code.iloc[0] == 1
    assert not codes.differentiated.iloc[0]


def test_code_2_hedge_without_a_second_view():
    df = frame([dim("clean platform"),
                row("hedge", attrs={"marker_type": "qualifier",
                                    "justification_quotes": [], "affects_conclusion": False}),
                row("verdict", attrs={"label": "More", "is_final": True})])
    assert IC.code_table(df).ic_code.iloc[0] == 2


def test_code_3_two_dimensions_on_both_images():
    df = frame([dim("clean platform", image="A"),
                dim("broken tiles", image="B", valence="bad"),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.ic_code.iloc[0] == 3
    assert codes.differentiated.iloc[0]


def test_code_4_link_without_a_justification():
    df = frame([dim("clean platform", image="A"),
                dim("broken tiles", image="B", valence="bad"),
                row("weighing", attrs={"mechanism": "trade_off",
                                       "justification_quotes": [], "condition_quotes": []}),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.ic_code.iloc[0] == 4
    assert not codes.integrated.iloc[0]


def test_code_5_justified_weighing():
    df = frame([dim("clean platform", image="A"),
                dim("broken tiles", image="B", valence="bad"),
                row("weighing", attrs={"mechanism": "trade_off",
                                       "justification_quotes": [span()],
                                       "condition_quotes": []}),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.ic_code.iloc[0] == 5
    assert codes.integrated.iloc[0]


def test_code_6_named_condition():
    df = frame([dim("clean platform", image="A"),
                dim("broken tiles", image="B", valence="bad"),
                row("weighing", attrs={"mechanism": "conditional",
                                       "justification_quotes": [span()],
                                       "condition_quotes": [span("at night")]}),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.ic_code.iloc[0] == 6
    assert codes.context_sensitive.iloc[0]


def test_the_scale_stops_at_6():
    assert IC.MAX_CODE == 6


# ------------------------------------------------------- what does not count

def test_an_unlocated_quote_does_not_count():
    """A quote no search finds is a defect. It must not lift a code."""
    df = frame([dim("clean platform", image="A"),
                dim("broken tiles", image="B", valence="bad", found=False),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.n_unlocated.iloc[0] == 1
    assert not codes.differentiated.iloc[0]


def test_an_unlocated_justification_does_not_integrate():
    df = frame([dim("clean platform", image="A"),
                dim("broken tiles", image="B", valence="bad"),
                row("weighing", attrs={"mechanism": "trade_off",
                                       "justification_quotes": [span(found=False)],
                                       "condition_quotes": []}),
                row("verdict", attrs={"label": "More", "is_final": True})])
    assert not IC.code_table(df).integrated.iloc[0]


def test_cues_on_one_image_are_pseudo_differentiation():
    df = frame([dim("clean platform", image="A"),
                dim("bright lights", image="A"),
                dim("wide stairs", image="A"),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.pseudo_differentiation.iloc[0]
    assert not codes.differentiated.iloc[0]


def test_a_developed_perspective_differentiates_on_its_own():
    df = frame([dim("clean platform", image="A"),
                row("perspective", name="a rider at night",
                    attrs={"favors": "B", "is_winner": False,
                           "supporting_quotes": [span(), span("s2")]}),
                row("perspective", name="a rider by day",
                    attrs={"favors": "A", "is_winner": True,
                           "supporting_quotes": [span(), span("s2")]}),
                row("verdict", attrs={"label": "More", "is_final": True})])
    codes = IC.code_table(df)
    assert codes.n_perspectives_developed.iloc[0] == 2
    assert codes.differentiated.iloc[0]


def test_one_supporting_quote_is_not_developed():
    df = frame([row("perspective", name="a rider at night",
                    attrs={"favors": "B", "is_winner": False,
                           "supporting_quotes": [span()]})])
    assert IC.code_table(df).n_perspectives_developed.iloc[0] == 0


# ------------------------------------------------------------- the details

def test_verdict_revision_follows_the_span_order():
    """The extractor may report the final label first. The offsets decide."""
    df = frame([dim("clean platform"),
                row("verdict", attrs={"label": "More", "is_final": True}, char_start=900),
                row("verdict", attrs={"label": "Same", "is_final": False}, char_start=100)])
    codes = IC.code_table(df)
    assert codes.verdict_revised.iloc[0]
    assert codes.verdict_final.iloc[0] == "More"
    assert codes.verdict_labels.iloc[0] == "Same|More"


def test_two_spellings_of_one_dimension_count_once():
    df = frame([dim("Cleanliness of the platform"),
                dim("platform cleanliness", image="B")])
    assert IC.code_table(df).n_dimensions_distinct.iloc[0] == 1


def test_a_silent_trace_gives_a_row_and_code_1():
    empty = row("dimension")
    empty["ingredient_type"] = None
    empty["quote_found"] = False
    df = frame([empty])
    codes = IC.code_table(df)
    assert codes.n_ingredients.iloc[0] == 0
    assert codes.ic_code.iloc[0] == 1


def test_another_schema_version_raises():
    df = frame([dim("clean platform")])
    df["schema_version"] = "v1"
    with pytest.raises(ValueError, match="do not pool"):
        IC.code_table(df)


def test_summarize_drops_a_cut_answer():
    good = [dim("clean platform", doc="d1"), dim("broken tiles", image="B", doc="d1")]
    cut = [dim("clean platform", doc="d2", truncated=True)]
    codes = IC.code_table(frame(good + cut))
    table = IC.summarize(codes)
    assert table.traces.iloc[0] == 1
    assert table.dropped_truncated.iloc[0] == 1
    assert IC.summarize(codes, drop_truncated=False).traces.iloc[0] == 2


def test_by_label_splits_on_the_judgment():
    a = [dim("clean platform", doc="d1", label="Same")]
    b = [dim("clean platform", doc="d2", label="More")]
    table = IC.by_label(IC.code_table(frame(a + b)))
    assert set(table.relative_label) == {"Same", "More"}


def test_thresholds_are_reported():
    used = IC.thresholds_used()
    assert used["supporting_quotes"] == 2
    assert used["reads_schema"] == "ic_ingredients/v2"


def test_a_threshold_change_moves_a_code():
    """The whole design is that a threshold changes without a GPU run."""
    df = frame([dim("clean platform", image="A"),
                row("perspective", name="a rider at night",
                    attrs={"favors": "B", "is_winner": False,
                           "supporting_quotes": [span()]})])
    strict = IC.code_table(df)
    loose = IC.code_table(df, IC.Thresholds(supporting_quotes=1))
    assert strict.n_perspectives_developed.iloc[0] == 0
    assert loose.n_perspectives_developed.iloc[0] == 1


def test_two_cases_that_share_a_doc_id_stay_apart():
    """`doc_id` is the pair id, and every run numbers its pairs from 0.

    A group on `doc_id` alone pools the cases and reports 1 trace where there
    are 7. This test is the guard: the same id in 2 cases must give 2 rows.
    """
    a = [dim("clean platform", doc="0", case="subway_safety"),
         dim("broken tiles", image="B", valence="bad", doc="0", case="subway_safety")]
    b = [dim("mown lawn", doc="0", case="parks_plazas")]
    codes = IC.code_table(frame(a + b))
    assert len(codes) == 2
    assert set(codes.case) == {"subway_safety", "parks_plazas"}
    assert codes[codes.case == "subway_safety"].n_dimensions_distinct.iloc[0] == 2
    assert codes[codes.case == "parks_plazas"].n_dimensions_distinct.iloc[0] == 1
