"""Tests for the proxemics distance report."""

from __future__ import annotations

import pandas as pd

from trashbot_annotator.proxemics import (
    REPORT_COLUMNS,
    build_report,
    format_report,
)
from trashbot_annotator.taxonomy import PROXEMIC_EVENTS


def annotations_frame() -> pd.DataFrame:
    """Build a small consolidated annotations frame with event rows."""
    deposit = PROXEMIC_EVENTS["trash_deposit"]
    gaze = PROXEMIC_EVENTS["gaze_at_robot"]
    rows = [
        {"place": "albee-square", "event": deposit, "distance_m": 0.5, "zone": "personal"},
        {"place": "albee-square", "event": deposit, "distance_m": 1.0, "zone": "personal"},
        {"place": "tappen-park", "event": deposit, "distance_m": None, "zone": "unknown"},
        {"place": "tappen-park", "event": gaze, "distance_m": 4.0, "zone": "public"},
        # a talk row without an event must stay out of the report
        {"place": "tappen-park", "event": None, "distance_m": None, "zone": None},
    ]
    df = pd.DataFrame(rows)
    df["category"] = None
    return df


def test_report_counts_and_quantiles() -> None:
    """The report counts events and summarizes the distances."""
    report = build_report(annotations_frame())

    assert list(report.columns) == list(REPORT_COLUMNS)
    deposit = report[report["event_key"] == "trash_deposit"].iloc[0]
    assert deposit["n_events"] == 3
    assert deposit["n_with_distance"] == 2
    assert deposit["distance_median"] == 0.75
    assert deposit["zone_personal"] == 2
    assert deposit["zone_unknown"] == 1

    toss = report[report["event_key"] == "trash_toss"].iloc[0]
    assert toss["n_events"] == 0
    assert toss["distance_median"] is None or pd.isna(toss["distance_median"])


def test_report_grouping_by_place() -> None:
    """Each place adds one group next to the (all) group."""
    report = build_report(annotations_frame(), group_by=["place"])
    groups = set(report["group"])
    assert "(all)" in groups
    assert "place=albee-square" in groups
    assert "place=tappen-park" in groups


def test_format_report_text() -> None:
    """The text form names the questions and the zone counts."""
    report = build_report(annotations_frame())
    text = format_report(report)
    assert "How close are people at a trash deposit?" in text
    assert "events=3" in text
    assert "No events found." in text
