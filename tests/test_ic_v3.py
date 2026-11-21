"""Unit tests for `notebooks/cvpr/_ic_v3.py`.

These tests build small frames by hand. They read no corpus and need no GPU.
They cover the lift of question 1, the nesting lookup of question 2, the
pseudo-differentiation flags of question 3, and the dismissal join of
question 4.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "cvpr"))

import _ic_v3 as I  # noqa: E402


def _row(case, pair, presented, cls, text="x", start=None, end=None, **attrs):
    row = {"case": case, "pair_id": pair, "presented_label": presented,
           "extraction_class": cls, "extraction_text": text,
           "char_start": start, "char_end": end}
    row.update(attrs)
    return row


def _frame(rows):
    return I.slim(pd.DataFrame(rows), "test_arm")


# --------------------------------------------------------------------------
# Question 1
# --------------------------------------------------------------------------

def _lift_corpus():
    """40 traces of 1 case. `maintenance` favors the winner in 30 of 30
    decided traces; `standard` favors the loser in 10 of 10; 5 undecided
    traces name `maintenance` for A."""
    rows = []
    for i in range(30):
        rows.append(_row("schools", f"m{i}", "More", "perspective",
                         name="Maintenance", favors="A", wins="yes"))
    for i in range(10):
        rows.append(_row("schools", f"s{i}", "Less", "perspective",
                         name="standard", favors="A", wins="no"))
    for i in range(5):
        rows.append(_row("schools", f"u{i}", "Same", "perspective",
                         name="maintenance", favors="A", wins="no"))
    return _frame(rows)


def test_lift_base_rate_counts_every_vote():
    tables = I.lift_tables(_lift_corpus(), "perspective", min_count=5,
                           prior_strength=0.0)
    by_key = tables.set_index("key")
    # 30 wins of 40 decided votes, over every key, before the floor.
    assert by_key.loc["maintenance", "base_rate"] == pytest.approx(0.75)
    assert by_key.loc["maintenance", "win_rate"] == 1.0
    assert by_key.loc["standard", "win_rate"] == 0.0
    assert by_key.loc["maintenance", "lift"] == pytest.approx(0.25)
    assert by_key.loc["standard", "lift"] == pytest.approx(-0.75)
    # The undecided traces do not vote but they count as mentions.
    assert by_key.loc["maintenance", "n"] == 30
    assert by_key.loc["maintenance", "n_mentions"] == 35
    assert by_key.loc["maintenance", "pays"]
    assert by_key.loc["standard", "costs"]


def test_lift_floor_drops_rare_key_but_keeps_its_base():
    tables = I.lift_tables(_lift_corpus(), "perspective", min_count=20,
                           prior_strength=0.0)
    assert list(tables["key"]) == ["maintenance"]
    assert tables["base_rate"].iloc[0] == pytest.approx(0.75)


def test_lift_shrinks_toward_base():
    tables = I.lift_tables(_lift_corpus(), "perspective", min_count=5,
                           prior_strength=10.0)
    row = tables.set_index("key").loc["standard"]
    # (0 + 10 * 0.75) / (10 + 10) = 0.375
    assert row["shrunk_rate"] == pytest.approx(0.375)
    assert row["lift"] == pytest.approx(0.375 - 0.75)


def test_lift_drops_key_on_both_images_and_neither():
    rows = [
        _row("schools", "p1", "More", "perspective", name="age", favors="A"),
        _row("schools", "p1", "More", "perspective", name="age", favors="B"),
        _row("schools", "p2", "More", "perspective", name="age", favors="neither"),
        _row("schools", "p3", "More", "perspective", name="age", favors="A"),
    ]
    tables = I.lift_tables(_frame(rows), "perspective", min_count=1)
    assert int(tables.set_index("key").loc["age", "n"]) == 1


def test_lift_key_of_dimension_and_inference():
    rows = [
        _row("parks_plazas", "p1", "Less", "dimension", name="Trees",
             image="B", valence="good"),
        _row("parks_plazas", "p2", "Less", "inference", name="upkeep",
             image="A", kind="upkeep"),
    ]
    df = _frame(rows)
    dim = I.lift_tables(df, "dimension", min_count=1)
    inf = I.lift_tables(df, "inference", min_count=1)
    assert dim["key"].tolist() == ["tree | good"]
    assert dim["wins"].tolist() == [1]
    assert inf["key"].tolist() == ["upkeep | upkeep"]
    assert inf["wins"].tolist() == [0]


def test_wins_yes_share_and_key_summary():
    tables = I.lift_tables(_lift_corpus(), "perspective", min_count=5)
    share = tables.set_index("key")["wins_yes_share"]
    assert share["maintenance"] == pytest.approx(30 / 35)
    assert share["standard"] == 0.0
    other = tables.assign(case="libraries")
    summary = I.key_summary(pd.concat([tables, other]), min_cases=2)
    assert summary.set_index("key").loc["standard", "holds"]
    assert summary.set_index("key").loc["standard", "sign"] == "-"


def test_case_ends_split_by_sign():
    tables = I.lift_tables(_lift_corpus(), "perspective", min_count=5)
    ends = I.case_ends(tables, n=5)
    assert set(ends.loc[ends["end"] == "top", "key"]) == {"maintenance"}
    assert set(ends.loc[ends["end"] == "bottom", "key"]) == {"standard"}


def test_extractor_agreement_counts_shared_keys():
    a = I.lift_tables(_lift_corpus(), "perspective", min_count=5)
    b = a.copy()
    b.loc[b["key"] == "standard", "lift"] = 0.1
    merged, summary = I.extractor_agreement(a, b)
    assert len(merged) == 2
    assert summary["n_shared"].iloc[0] == 2
    assert summary["same_sign"].iloc[0] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Question 2
# --------------------------------------------------------------------------

def _nest_corpus():
    """1 trace: a perspective `modern` at 10-16 inside an inference clause at
    0-40; a dimension at 12-15 inside the perspective; a dimension at 50-60
    apart; a dimension with the same offsets as the perspective."""
    return _frame([
        _row("schools", "p1", "More", "perspective", "modern", 10, 16,
             name="modern", favors="A"),
        _row("schools", "p1", "More", "inference", "a modern glass facade",
             0, 40, name="modernity", image="A", kind="quality"),
        _row("schools", "p1", "More", "dimension", "der", 12, 15,
             name="glass", image="A", valence="neutral"),
        _row("schools", "p1", "More", "dimension", "far", 50, 60,
             name="trees", image="B", valence="good"),
        _row("schools", "p1", "More", "dimension", "modern", 10, 16,
             name="facade", image="A", valence="good"),
        _row("schools", "p2", "Less", "dimension", "y", 0, 5,
             name="trees", image="B", valence="good"),
    ])


def test_nest_pairs_relations():
    df = _nest_corpus()
    pairs = I.nest_pairs(I.perspective_frame(df), I.partner_frame(df))
    rel = dict(zip(pairs["partner"], pairs["relation"]))
    assert rel["modernity | quality"] == "contains_perspective"
    assert rel["glass | neutral"] == "inside_perspective"
    assert rel["facade | good"] == "same_span"
    assert "tree | good" not in rel


def test_nesting_overview_and_grounding_table():
    df = _nest_corpus()
    by_key, overview = I.nesting(df)
    over = overview.set_index(["partner_class", "relation"])["share_spans"]
    assert over[("inference", "contains_perspective")] == 1.0
    assert over[("dimension", "inside_perspective")] == 1.0
    assert over[("inference", "inside_perspective")] == 0.0
    cooc = I.cooccurrence(df, min_perspective=1, min_pair=1)
    table = I.grounding_table(cooc, by_key).set_index("partner")
    assert table.loc["modernity | quality", "n_contains"] == 1
    assert table.loc["tree | good", "n_nested"] == 0
    assert table.loc["modernity | quality", "nest_rate"] == 1.0


def test_cooccurrence_expected_within_case():
    df = _nest_corpus()
    cooc = I.cooccurrence(df, min_perspective=1, min_pair=1).set_index("partner")
    # `trees` sits in 2 of the 2 traces of the case, thus the 1 perspective
    # trace expects 1 co-occurrence and sees 1: lift 1.
    assert cooc.loc["tree | good", "expected"] == pytest.approx(1.0)
    assert cooc.loc["tree | good", "lift"] == pytest.approx(1.0)
    # `modernity` sits in 1 of 2 traces: expected 0.5, lift 2.
    assert cooc.loc["modernity | quality", "lift"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Question 3
# --------------------------------------------------------------------------

def _diff_corpus():
    rows = []
    # pseudo: verdict More, reconsideration, decision More.
    rows += [_row("libraries", "a", "More", "verdict", start=10, label="More"),
             _row("libraries", "a", "More", "uncertainty", start=20,
                  marker="reconsideration"),
             _row("libraries", "a", "More", "decision", start=30, label="More")]
    # revised: verdict More, 2 reconsiderations, decision Less.
    rows += [_row("libraries", "b", "Less", "verdict", start=10, label="More"),
             _row("libraries", "b", "Less", "uncertainty", start=15,
                  marker="reconsideration"),
             _row("libraries", "b", "Less", "uncertainty", start=18,
                  marker="reconsideration"),
             _row("libraries", "b", "Less", "decision", start=30, label="Less")]
    # steady: verdict, a qualifier only, decision unchanged.
    rows += [_row("libraries", "c", "Same", "verdict", start=10, label="Same"),
             _row("libraries", "c", "Same", "uncertainty", start=20,
                  marker="qualifier"),
             _row("libraries", "c", "Same", "decision", start=30, label="Same")]
    # no verdict: a reconsideration BEFORE any answer; and 1 after the
    # decision in trace a's twin, which must not count.
    rows += [_row("libraries", "d", "More", "uncertainty", start=5,
                  marker="reconsideration"),
             _row("libraries", "d", "More", "decision", start=30, label="More")]
    # first verdict More, last verdict Less, decision Less: unchanged
    # against the last, changed against the first.
    rows += [_row("libraries", "e", "Less", "verdict", start=10, label="More"),
             _row("libraries", "e", "Less", "uncertainty", start=15,
                  marker="reconsideration"),
             _row("libraries", "e", "Less", "verdict", start=20, label="Less"),
             _row("libraries", "e", "Less", "decision", start=30, label="Less"),
             _row("libraries", "e", "Less", "uncertainty", start=40,
                  marker="reconsideration")]
    # a strength change only: More then MuchMore.
    rows += [_row("libraries", "f", "MuchMore", "verdict", start=10, label="More"),
             _row("libraries", "f", "MuchMore", "uncertainty", start=20,
                  marker="reconsideration"),
             _row("libraries", "f", "MuchMore", "decision", start=30,
                  label="MuchMore")]
    return _frame(rows)


def test_differentiation_flags():
    flags = I.differentiation_flags(_diff_corpus())
    f = flags.set_index(flags["tid"].str.split("|").str[1])
    assert f.loc["a", "group"] == "pseudo"
    assert f.loc["b", "group"] == "revised"
    assert f.loc["b", "n_recon_after"] == 2
    assert f.loc["c", "group"] == "steady"
    assert f.loc["c", "pseudo_any"]
    assert f.loc["d", "group"] == "no_verdict"
    assert f.loc["e", "group"] == "pseudo"
    assert f.loc["e", "n_recon_after"] == 1      # the span after the decision is out
    assert f.loc["e", "n_verdicts"] == 2
    assert f.loc["e", "revised_first"]
    assert f.loc["f", "revised"]
    assert not f.loc["f", "revised_direction"]   # A wins in both
    assert f.loc["b", "revised_direction"]


def test_differentiation_summary_shares():
    flags = I.differentiation_flags(_diff_corpus())
    s = I.differentiation_summary(flags).iloc[0]
    assert s["n_traces"] == 6
    assert s["share_verdict"] == pytest.approx(5 / 6)
    assert s["share_pseudo"] == pytest.approx(2 / 6)
    assert s["share_revised"] == pytest.approx(2 / 6)
    assert s["revised_given_recon"] == pytest.approx(2 / 4)
    assert s["mean_recon_revised"] == pytest.approx(1.5)
    assert s["mean_recon_pseudo"] == pytest.approx(1.0)
    shares = sum(s[f"share_{g}"] for g in I.GROUP_ORDER)
    assert shares == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Question 4
# --------------------------------------------------------------------------

def _dismissal_corpus():
    return _frame([
        # Named before, by name: a perspective `prestige`, then dismissed.
        _row("schools", "a", "More", "perspective", "prestige matters", 0, 16,
             name="Prestige"),
        _row("schools", "a", "More", "dismissal", "but prestige is not the task",
             20, 48, target="prestige", about="perspective"),
        # A premise dismissal whose only match is the SAME clause, marked as
        # a dimension too.
        _row("schools", "b", "More", "dimension",
             "I don't see a school building", 0, 29, name="signage",
             image="A", valence="neutral"),
        _row("schools", "b", "More", "dismissal",
             "I don't see a school building", 0, 29,
             target="school buildings", about="premise"),
        # No match at all; a 1-letter target never matches by text.
        _row("schools", "c", "More", "inference", "a b c", 0, 5, name="a",
             image="A", kind="quality"),
        _row("schools", "c", "More", "dismissal", "ignore it", 10, 19,
             target="b", about="cue"),
    ])


def test_dismissal_join():
    dm = I.dismissal_frame(_dismissal_corpus()).set_index("tid")
    a, b, c = dm.loc["schools|a"], dm.loc["schools|b"], dm.loc["schools|c"]
    assert a["named_perspective"] and a["matched_any"]
    assert a["matched_elsewhere"] and a["named_before"]
    # `school buildings` normalises to `school building`; the text rule
    # finds `school building` inside the overlapping dimension clause.
    assert b["target_key"] == "school building"
    assert b["matched_dimension"] and not b["named_dimension"]
    assert not b["matched_elsewhere"] and not b["named_before"]
    assert not c["matched_any"]


def test_dismissal_summaries():
    dm = I.dismissal_frame(_dismissal_corpus())
    about = I.about_shares(dm).iloc[0]
    assert about["n_dismissals"] == 3
    assert about["share_premise"] == pytest.approx(1 / 3)
    targets = I.top_targets(dm, n=5, min_count=1)
    assert set(targets["target_key"]) == {"prestige", "school building", "b"}
    match = I.match_summary(dm).iloc[0]
    assert match["matched_any"] == pytest.approx(2 / 3)
    assert match["named_before"] == pytest.approx(1 / 3)
