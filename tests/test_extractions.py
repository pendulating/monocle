"""Unit tests for the extraction analysis module, `notebooks/cvpr/_extractions.py`.

These tests build a small frame by hand. They read no parquet and need no GPU.
They cover the parts where a quiet error changes a number in the paper: the
denominator of a rate, the vocabulary mapping, and the unit of the distinctive
score.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks" / "cvpr"))

import _extractions as X  # noqa: E402


def _row(case, pair, cls, text, attrs, label="Same", quotable=True):
    return {
        "case": case,
        "pair_id": pair,
        "presented_label": label,
        "extraction_class": cls,
        "extraction_text": text,
        "attributes_json": json.dumps(attrs) if attrs else None,
        "is_quotable": quotable,
        "alignment_status": "match_exact" if quotable else "match_lesser",
        "judge_model": "gemma-4-12b/instruct_thinking",
        "extractor_model": "/zoo/gemma-4-12B-it",
        "schema_name": X.SCHEMA_NAME,
        "schema_version": X.SCHEMA_VERSION,
    }


@pytest.fixture
def frame() -> pd.DataFrame:
    rows = [
        _row("subway_safety", "p1", "visual_evidence", "trash can",
             {"image": "A", "valence": "bad", "category": "cleanliness"}),
        _row("subway_safety", "p1", "person_reference", "people waiting",
             {"image": "A", "used_in_judgment": "yes"}),
        _row("subway_safety", "p2", "inference", "looks like a rough area",
             {"image": "B", "kind": "crime"}),
        # A value the prompt never lists. `normalize` must move it to `other`
        # and keep the original.
        _row("subway_safety", "p2", "inference", "a bright modern entrance",
             {"image": "A", "kind": "architecture"}),
        _row("schools", "p3", "visual_evidence", "yellow school bus",
             {"image": "A", "valence": "good", "category": "other"}),
        _row("schools", "p3", "inference", "a wealthy district",
             {"image": "A", "kind": "wealth"}),
    ]
    return X.attach_attributes(pd.DataFrame(rows))


class TestNormalize:
    def test_a_declared_value_survives(self, frame):
        out = X.normalize(frame)
        crime = out[out.extraction_text == "looks like a rough area"]
        assert crime["kind"].iloc[0] == "crime"

    def test_an_unlisted_value_becomes_other(self, frame):
        out = X.normalize(frame)
        row = out[out.extraction_text == "a bright modern entrance"]
        assert row["kind"].iloc[0] == "other"

    def test_the_original_value_is_kept(self, frame):
        """Nothing is lost. The report needs the original to name it."""
        out = X.normalize(frame)
        row = out[out.extraction_text == "a bright modern entrance"]
        assert row["kind_raw"].iloc[0] == "architecture"

    def test_the_report_names_what_moved(self, frame):
        report = X.vocabulary_report(X.normalize(frame))
        assert not report.empty
        line = report[report.attribute == "kind"].iloc[0]
        assert "architecture" in line.top_values

    def test_a_declared_other_is_not_reported(self, frame):
        """`other` is a real option of `category`. It did not move."""
        report = X.vocabulary_report(X.normalize(frame))
        assert report[report.attribute == "category"].empty


class TestRates:
    def test_the_denominator_is_every_trace(self, frame, monkeypatch):
        """A trace with no quotable span still asked the question.

        Leave it out of the denominator and every rate reads too high.
        """
        monkeypatch.setattr(
            X, "trace_totals",
            lambda root=None: pd.Series({"subway_safety": 4, "schools": 2}),
        )
        rates = X.class_rates(frame, per=100)
        # 1 visual_evidence over 4 subway traces, not over the 2 that appear.
        assert rates.loc["visual_evidence", "subway_safety"] == 25.0
        assert rates.loc["visual_evidence", "schools"] == 50.0

    def test_it_falls_back_to_the_frame(self, frame, monkeypatch):
        monkeypatch.setattr(X, "trace_totals", lambda root=None: pd.Series(dtype=int))
        rates = X.class_rates(frame, per=100)
        assert rates.loc["visual_evidence", "subway_safety"] == 50.0

    def test_the_risk_panel_holds_the_risk_rows(self, frame, monkeypatch):
        monkeypatch.setattr(
            X, "trace_totals",
            lambda root=None: pd.Series({"subway_safety": 2, "schools": 2}),
        )
        panel = X.risk_panel(X.normalize(frame))
        assert panel.loc["inference: crime", "subway_safety"] == 50.0
        assert panel.loc["inference: wealth", "schools"] == 50.0
        assert panel.loc["person_reference: used in judgment", "subway_safety"] == 50.0


class TestCounters:
    def test_class_text_is_the_unit(self, frame):
        counters = X.counters(frame, unit="class_text")
        assert counters["schools"]["visual_evidence:yellow school bus"] == 1

    def test_a_span_key_is_case_folded(self, frame):
        rows = pd.concat([frame, frame.assign(extraction_text="Trash Can")])
        counters = X.counters(rows, unit="class_text")
        assert counters["subway_safety"]["visual_evidence:trash can"] >= 2

    def test_the_scaffold_class_is_dropped(self, frame):
        rows = pd.concat([
            frame,
            pd.DataFrame([_row("schools", "p3", "decision", 'I\'ll go with "Same"',
                               {"label": "Same"})]),
        ], ignore_index=True)
        counters = X.counters(X.attach_attributes(rows), unit="class_text",
                              exclude_classes=X.SCAFFOLD_CLASSES)
        assert not any(k.startswith("decision:") for k in counters["schools"])

    def test_an_unknown_unit_raises(self, frame):
        with pytest.raises(ValueError):
            X.counters(frame, unit="nonsense")


class TestWinRates:
    """The 3 rules that keep a win rate honest, and the direction itself."""

    def _cue(self, pair, image, label, text="graffiti"):
        return _row("subway_safety", pair, "visual_evidence", text,
                    {"image": image, "valence": "bad", "category": "other"},
                    label=label)

    def test_the_direction_follows_the_image(self):
        """A cue on A wins when A wins; the same cue on B wins when B wins."""
        rows = [self._cue(f"a{i}", "A", "More") for i in range(10)]
        rows += [self._cue(f"b{i}", "B", "Less") for i in range(10)]
        df = X.normalize(X.attach_attributes(pd.DataFrame(rows)))
        out = X.win_rates(df, "subway_safety", min_count=1, prior_strength=0.0)
        assert out.iloc[0]["n"] == 20
        assert out.iloc[0]["win_rate"] == 1.0

    def test_a_cue_on_the_losing_image_scores_zero(self):
        rows = [self._cue(f"a{i}", "A", "Less") for i in range(10)]
        df = X.normalize(X.attach_attributes(pd.DataFrame(rows)))
        out = X.win_rates(df, "subway_safety", min_count=1, prior_strength=0.0)
        assert out.iloc[0]["win_rate"] == 0.0

    def test_one_vote_for_each_comparison(self):
        """A trace that names a cue 3 times still describes 1 comparison."""
        rows = [self._cue("p1", "A", "More") for _ in range(3)]
        rows += [self._cue("p2", "A", "More")]
        df = X.normalize(X.attach_attributes(pd.DataFrame(rows)))
        out = X.win_rates(df, "subway_safety", min_count=1, prior_strength=0.0)
        assert out.iloc[0]["n"] == 2

    def test_a_cue_on_both_images_is_dropped(self):
        """A fence in A and in B separates nothing."""
        rows = [self._cue("p1", "A", "More"), self._cue("p1", "B", "More")]
        rows += [self._cue("p2", "A", "More")]
        df = X.normalize(X.attach_attributes(pd.DataFrame(rows)))
        out = X.win_rates(df, "subway_safety", min_count=1, prior_strength=0.0)
        assert out.iloc[0]["n"] == 1

    def test_an_undecided_label_leaves_the_rate(self):
        """`Same` and `NotSure` name no winner, thus neither can vote."""
        rows = [self._cue("p1", "A", "More"), self._cue("p2", "A", "Same"),
                self._cue("p3", "A", "NotSure")]
        df = X.normalize(X.attach_attributes(pd.DataFrame(rows)))
        out = X.win_rates(df, "subway_safety", min_count=1, prior_strength=0.0)
        assert out.iloc[0]["n"] == 1
        assert out.iloc[0]["n_mentions"] == 3
        assert out.iloc[0]["undecided_share"] == pytest.approx(2 / 3, abs=0.001)

    def test_shrinkage_holds_a_rare_cue_back(self):
        """A cue seen 3 times must not reach the top of the colour scale.

        The test is how FAR each estimate moves toward the base rate, not which
        cue ends up higher. A well-observed cue that already sits at the base
        rate cannot move, so a comparison between the two proves nothing.
        """
        rows = [self._cue(f"r{i}", "A", "More", text="rare") for i in range(3)]
        rows += [self._cue(f"c{i}", "A", "More", text="common") for i in range(300)]
        rows += [self._cue(f"d{i}", "A", "Less", text="common") for i in range(100)]
        df = X.normalize(X.attach_attributes(pd.DataFrame(rows)))
        out = X.win_rates(df, "subway_safety", min_count=1, prior_strength=25.0)
        rare = out[out.cue == "rare"].iloc[0]
        common = out[out.cue == "common"].iloc[0]
        base = float(rare.base_rate)

        assert rare.win_rate == 1.0
        # 3 observations carry almost no weight against a prior of 25, thus the
        # estimate lands nearer the base rate than its own raw value.
        assert abs(rare.shrunk_rate - base) < abs(rare.win_rate - base) / 2
        # 400 observations dominate the same prior, thus this one barely moves.
        assert abs(common.shrunk_rate - common.win_rate) < 0.02

    def test_the_interval_is_wilson_not_normal(self):
        """A perfect record must not give an interval of zero width."""
        low, high = X._wilson(30, 30)
        assert high == 1.0
        assert low < 1.0


class TestUnitWinRates:
    """The unit statistic, whose whole risk is the A/B swap.

    `presented_label` describes the image the model saw FIRST. `is_swapped`
    says which side that was. Read `unit_uid_a` as the model's image A on a
    swapped pair and both photograph rows come out inverted.
    """

    def _run(self, tmp_path, rows):
        """Write a results parquet and a pairs manifest, then score them."""
        out = tmp_path / "outputs" / "pairwise"
        out.mkdir(parents=True)
        results = out / "subway_safety_mvp_20260813_013722.parquet"
        pd.DataFrame([
            {"pair_id": r["pair_id"], "presented_label": r["presented_label"]}
            for r in rows
        ]).to_parquet(results, index=False)
        pd.DataFrame([
            {k: v for k, v in r.items() if k != "presented_label"} for r in rows
        ]).to_parquet(out / "pairs.parquet", index=False)

        frame = pd.DataFrame([{
            "case": "subway_safety", "pair_id": rows[0]["pair_id"],
            "source_results_path": str(results),
            "extraction_class": "visual_evidence", "extraction_text": "x",
            "attributes_json": None, "is_quotable": True,
        }])
        return X.unit_win_rates(frame, "subway_safety", min_pairs=1,
                                prior_strength=0.0)

    def _pair(self, i, swapped, label):
        return {
            "pair_id": f"p{i}", "presented_label": label, "is_swapped": swapped,
            "unit_uid_a": "unit_A", "unit_uid_b": "unit_B",
            "unit_name_a": "A place", "unit_name_b": "B place",
            "presented_left_path": "/left.jpg", "presented_right_path": "/right.jpg",
        }

    def test_an_unswapped_pair_credits_side_a(self, tmp_path):
        """`More` means the first image won, and that is unit_uid_a here."""
        table = self._run(tmp_path, [self._pair(0, False, "More")])
        won = table.set_index("unit").win_rate
        assert won["unit_A"] == 1.0
        assert won["unit_B"] == 0.0

    def test_a_swapped_pair_credits_side_b(self, tmp_path):
        """The same label, but the first image was unit_uid_b."""
        table = self._run(tmp_path, [self._pair(0, True, "More")])
        won = table.set_index("unit").win_rate
        assert won["unit_B"] == 1.0
        assert won["unit_A"] == 0.0

    def test_the_photograph_follows_the_presented_order(self, tmp_path):
        """The winner's frame must show the image the model saw as A."""
        table = self._run(tmp_path, [self._pair(0, True, "More")])
        winner = table.set_index("unit").loc["unit_B"]
        assert winner.image_path == "/left.jpg"

    def test_an_undecided_label_votes_for_nobody(self, tmp_path):
        table = self._run(tmp_path, [self._pair(0, False, "Same"),
                                     self._pair(1, False, "NotSure")])
        assert table.empty or int(table["n"].sum()) == 0

    def test_the_base_rate_is_one_half(self, tmp_path):
        """Each decided pair gives 1 winner and 1 loser, thus the base is 0.5.

        Take it after the `min_pairs` filter and it is not.
        """
        rows = [self._pair(i, i % 2 == 0, "More" if i % 3 else "Less")
                for i in range(20)]
        table = self._run(tmp_path, rows)
        assert float(table.base_rate.iloc[0]) == 0.5

    def test_a_missing_manifest_returns_empty(self, tmp_path):
        """An older sweep may have no manifest. That must not raise."""
        frame = pd.DataFrame([{
            "case": "subway_safety", "pair_id": "p0",
            "source_results_path": str(tmp_path / "gone" / "x.parquet"),
            "extraction_class": "visual_evidence", "extraction_text": "x",
            "attributes_json": None, "is_quotable": True,
        }])
        assert X.unit_win_rates(frame, "subway_safety").empty


class TestDistinctive:
    def test_it_names_what_one_case_says_alone(self):
        """A word the target uses and the background does not must score high."""
        rows = []
        for i in range(40):
            rows.append(_row("schools", f"s{i}", "visual_evidence", "school bus",
                             {"image": "A", "valence": "good", "category": "other"}))
            rows.append(_row("subway_safety", f"t{i}", "visual_evidence", "turnstile",
                             {"image": "A", "valence": "neutral", "category": "other"}))
        df = X.attach_attributes(pd.DataFrame(rows))
        top = X.distinctive(df, "schools", unit="class_text", min_count=1)
        assert top.iloc[0]["unit"] == "visual_evidence:school bus"
        assert top.iloc[0]["score"] > 0

    def test_one_case_alone_gives_counts(self, frame):
        only = frame[frame.case == "schools"]
        out = X.distinctive(only, "schools", min_count=1)
        assert not out.empty
        assert "count" in out.columns
