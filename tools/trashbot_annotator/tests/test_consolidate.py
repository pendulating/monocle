"""Tests for JSONL consolidation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trashbot_annotator.consolidate import OUTPUT_COLUMNS, consolidate_dir
from trashbot_annotator.taxonomy import (
    ANNOTATOR_NAME,
    NORM_DIMENSIONS,
    PROXEMIC_EVENTS,
    TALK_CATEGORIES,
)


def window_line(**overrides: object) -> str:
    """Build one JSONL line for a window record."""
    record: dict[str, object] = {
        "schema_version": 1,
        "video_path": "/share/ju/robot_norms/data/Han_2025_Trashbots/2025-07-01-tappen-park/onboard_camera/a.mp4",
        "sample_id": "Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a.mp4",
        "window_idx": 0,
        "t_start_s": 0.0,
        "t_end_s": 8.0,
        "frame_indices": [0, 1, 2],
        "timestamps_s": [0.0, 1.0, 2.0],
        "model": "google/gemma-4-12B-it",
        "response_raw": None,
        "annotation": None,
        "error": None,
    }
    record.update(overrides)
    return json.dumps(record)


def write_fixture(out_dir: Path) -> None:
    """Write fixture JSONL files into the out-dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    good = {
        "interactions": [
            {
                "category": "instruction",
                "address": "to_robot",
                "who": "caregiver",
                "evidence": "Parent says: push it.",
            },
            {
                "category": "banter",
                "address": "shouting",
                "who": "?",
                "evidence": "Unknown label here.",
            },
        ],
        "norm_dimensions": [
            {"dimension": NORM_DIMENSIONS[0], "relevance": 2, "evidence": "Child smiles at it."},
            {"dimension": "Privacy vibes", "relevance": 99, "evidence": "Bad label and score."},
        ],
        "proxemic_events": [
            {
                "event": "trash_deposit",
                "who": "child",
                "distance_m": 0.8,
                "evidence": "The child drops a cup into the robot.",
            },
            {
                "event": "hug",
                "who": "?",
                "distance_m": -2,
                "zone": "near",
                "evidence": "Bad event and zone.",
            },
        ],
        "summary": "A family guides the robot.",
    }
    errored = window_line(
        window_idx=1,
        t_start_s=8.0,
        t_end_s=16.0,
        error="request_error: connection refused",
    )
    lines = [
        window_line(annotation=good),
        errored,
        window_line(
            annotation={
                "interactions": [],
                "norm_dimensions": [],
                "summary": "Nothing happens.",
            },
            window_idx=2,
            t_start_s=16.0,
            t_end_s=24.0,
        ),
    ]
    (out_dir / "Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_consolidate_columns_and_expansion(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Rows expand per interaction and per dimension with exact columns."""
    out_dir = tmp_path / "jsonl"
    write_fixture(out_dir)
    out = tmp_path / "consolidated.parquet"

    df, warnings = consolidate_dir(out_dir, out)

    assert list(df.columns) == list(OUTPUT_COLUMNS)
    # Window 0: 2 interactions + 2 dimensions + 2 events = 6 rows.
    # Window 1 has an error, so it yields no rows.
    # Window 2 has empty lists, so it yields no rows.
    assert len(df) == 6
    assert (df["window_idx"] == 0).all()
    assert len(pd.read_parquet(out)) == 6


def test_invalid_label_handling(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Invalid labels land in the raw columns and count as warnings."""
    out_dir = tmp_path / "jsonl"
    write_fixture(out_dir)

    df, warnings = consolidate_dir(out_dir, tmp_path / "c.parquet")

    cats = df["category"].dropna().tolist()
    assert cats == [TALK_CATEGORIES["instruction"]]
    assert df["category_raw"].dropna().tolist() == ["banter"]

    dims = df["dimension"].dropna().tolist()
    assert dims == [NORM_DIMENSIONS[0]]
    assert df["dimension_raw"].dropna().tolist() == ["Privacy vibes"]

    # Relevance 99 is out of range, so it becomes null.
    rel = df["relevance"].dropna().tolist()
    assert rel == [2]
    assert warnings["invalid_category"] == 1
    assert warnings["invalid_dimension"] == 1
    assert warnings["invalid_relevance"] == 1
    assert warnings["windows_without_annotation"] == 1
    assert warnings["invalid_address"] == 1
    assert warnings["invalid_event"] == 1
    assert warnings["invalid_zone"] == 1
    assert warnings["invalid_distance"] == 1


def test_proxemic_event_handling(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Events land in event rows with a distance and a zone."""
    out_dir = tmp_path / "jsonl"
    write_fixture(out_dir)

    df, _warnings = consolidate_dir(out_dir, tmp_path / "c.parquet")

    events = df[df["event"].notna()]
    assert events["event"].tolist() == [PROXEMIC_EVENTS["trash_deposit"]]
    row = events.iloc[0]
    # 0.8 m sits inside the personal zone; the zone derives from the
    # distance because the model gave none.
    assert row["distance_m"] == 0.8
    assert row["zone"] == "personal"
    assert row["who"] == "child"

    bad = df[df["event_raw"].notna()]
    assert bad["event_raw"].tolist() == ["hug"]
    # The distance -2 is invalid, so the derived zone is unknown.
    assert bad.iloc[0]["zone"] == "unknown"
    assert bad.iloc[0]["zone_raw"] == "near"
    assert pd.isna(bad.iloc[0]["distance_m"])


def test_interaction_address(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A valid address carries through; an invalid one lands in raw."""
    out_dir = tmp_path / "jsonl"
    write_fixture(out_dir)

    df, _warnings = consolidate_dir(out_dir, tmp_path / "c.parquet")

    talk = df[df["category"].notna() | df["category_raw"].notna()]
    assert talk["address"].dropna().tolist() == ["to_robot"]
    assert talk["address_raw"].dropna().tolist() == ["shouting"]


def test_metadata_from_manifest_join(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A manifest join fills dataset, place, and capture type."""
    out_dir = tmp_path / "jsonl"
    write_fixture(out_dir)

    manifest = pd.DataFrame(
        [
            {
                "sample_id": "Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a.mp4",
                "video_path": "/share/ju/robot_norms/data/Han_2025_Trashbots/2025-07-01-tappen-park/onboard_camera/a.mp4",
                "dataset": "Han_2025_Trashbots",
                "place": "tappen-park",
                "capture_type": "onboard_camera",
            }
        ]
    )
    df, _warnings = consolidate_dir(out_dir, tmp_path / "c.parquet", manifest=manifest)

    assert (df["dataset"] == "Han_2025_Trashbots").all()
    assert (df["place"] == "tappen-park").all()
    assert (df["capture_type"] == "onboard_camera").all()


def test_metadata_derived_from_video_path(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Without a manifest, metadata comes from the video path."""
    out_dir = tmp_path / "jsonl"
    write_fixture(out_dir)

    df, _warnings = consolidate_dir(out_dir, tmp_path / "c.parquet")

    assert (df["dataset"] == "Han_2025_Trashbots").all()
    assert (df["place"] == "tappen-park").all()
    assert (df["capture_type"] == "onboard_camera").all()


def test_fixed_row_fields(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Annotator name, model, times, and summary carry through."""
    out_dir = tmp_path / "jsonl"
    write_fixture(out_dir)

    df, _warnings = consolidate_dir(out_dir, tmp_path / "c.parquet")
    row = df.iloc[0]

    assert row["annotator"] == ANNOTATOR_NAME
    assert row["model"] == "google/gemma-4-12B-it"
    assert row["start_s"] == 0.0
    assert row["end_s"] == 8.0
    assert row["summary"] == "A family guides the robot."
    assert row["sample_id"] == "Han_2025_Trashbots_2025-07-01-tappen-park_onboard_camera_a.mp4"
    # created_at parses as a timestamp.
    assert pd.Timestamp(row["created_at"]) is not None
