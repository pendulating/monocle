"""Tests for the repeat-interaction pairing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trashbot_annotator.repeats import (
    PAIRS_COLUMNS,
    cluster_repeats,
    find_repeats,
    parse_date_from_sample_id,
    select_event_windows,
)


def frames_manifest() -> pd.DataFrame:
    """Build a frames manifest for three events across two clips."""
    return pd.DataFrame(
        [
            {
                "event_id": "clip_a__w0",
                "sample_id": "clip_a",
                "place": "albee-square",
                "date": "2023-07-11",
            },
            {
                "event_id": "clip_a__w5",
                "sample_id": "clip_a",
                "place": "albee-square",
                "date": "2023-07-11",
            },
            {
                "event_id": "clip_b__w2",
                "sample_id": "clip_b",
                "place": "albee-square",
                "date": "2023-07-12",
            },
        ]
    )


def embeddings_frame() -> pd.DataFrame:
    """Build embeddings: two near-identical vectors, one orthogonal."""
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    w = np.array([0.99, 0.14, 0.0], dtype=np.float32)
    u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return pd.DataFrame(
        [
            {"event_id": "clip_a__w0", "embedding": v.tolist()},
            {"event_id": "clip_a__w5", "embedding": v.tolist()},
            {"event_id": "clip_b__w2", "embedding": w.tolist()},
            {"event_id": "clip_c__missing", "embedding": None},
        ]
    ).assign(embedding_dim=3)


def test_find_repeats_cross_sample() -> None:
    """Only cross-clip pairs over the threshold survive."""
    pairs = find_repeats(embeddings_frame(), frames_manifest(), min_sim=0.9)

    assert list(pairs.columns) == list(PAIRS_COLUMNS)
    # clip_a__w0/clip_a__w5 sit in one clip, so they drop out. The
    # orthogonal vector never crosses the threshold.
    got = {(r["event_id_a"], r["event_id_b"]) for r in pairs.to_dict("records")}
    assert got == {
        ("clip_a__w0", "clip_b__w2"),
        ("clip_a__w5", "clip_b__w2"),
    }
    row = pairs.iloc[0]
    assert row["same_place"]
    assert not row["same_date"]


def test_find_repeats_within_sample() -> None:
    """With cross_sample_only off, same-clip pairs appear."""
    pairs = find_repeats(
        embeddings_frame(), frames_manifest(), min_sim=0.9, cross_sample_only=False
    )
    got = {(r["event_id_a"], r["event_id_b"]) for r in pairs.to_dict("records")}
    assert ("clip_a__w0", "clip_a__w5") in got


def test_cluster_repeats() -> None:
    """Pairs chain into one cluster of three events."""
    pairs = find_repeats(
        embeddings_frame(), frames_manifest(), min_sim=0.9, cross_sample_only=False
    )
    clusters = cluster_repeats(pairs)
    assert len(clusters) == 1
    assert clusters.iloc[0]["n_events"] == 3


def test_parse_date_from_sample_id() -> None:
    """Both date spellings inside a sample id parse to ISO form."""
    assert (
        parse_date_from_sample_id("Bu_TrashbotWoz_2023-7-11-albee-square_x")
        == "2023-07-11"
    )
    assert (
        parse_date_from_sample_id("trashbot_summer_2026_joy_t2t_20260626_cam_f")
        == "2026-06-26"
    )
    assert parse_date_from_sample_id("clip_without_date") is None


def test_select_event_windows() -> None:
    """Only windows with a person event qualify, once per window."""
    df = pd.DataFrame(
        [
            {"sample_id": "a", "window_idx": 0, "start_s": 0.0, "end_s": 8.0,
             "category": "Instruction — commanding the robot", "event": None},
            {"sample_id": "a", "window_idx": 0, "start_s": 0.0, "end_s": 8.0,
             "category": None, "event": "Gaze — a person looks at the robot or tracks it"},
            {"sample_id": "a", "window_idx": 1, "start_s": 8.0, "end_s": 16.0,
             "category": None, "event": None},
        ]
    )
    windows = select_event_windows(df)
    assert len(windows) == 1
    assert windows.iloc[0]["t_frame_s"] == 4.0
