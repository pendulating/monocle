"""Tests for the speech classification channel."""

from __future__ import annotations

import json

import pandas as pd

from tests.stub_server import start_stub_server, stop_stub_server
from trashbot_annotator.speech import (
    SPEECH_COLUMNS,
    build_context,
    classify_corpus,
    consolidate_speech,
    extract_json_object,
    load_utterance_corpus,
    render_speech_prompt,
)

# The stub answers every request with this speech payload.
SPEECH_PAYLOAD: dict[str, object] = {
    "address": "about_robot",
    "kind": "joke",
    "operator": False,
    "evidence": "They laugh about the robot.",
}


def corpus_frame() -> pd.DataFrame:
    """Build a small utterance corpus frame."""
    return pd.DataFrame(
        [
            {
                "doc_id": "0",
                "dataset": "Bu_TrashbotWoz",
                "place": "albee-square",
                "capture_type": "other",
                "sample_id": "clip_a",
                "chunk_idx": 0,
                "start_s": 0.0,
                "end_s": 3.0,
                "text": "Look at that little robot.",
            },
            {
                "doc_id": "1",
                "dataset": "Bu_TrashbotWoz",
                "place": "albee-square",
                "capture_type": "other",
                "sample_id": "clip_a",
                "chunk_idx": 1,
                "start_s": 3.0,
                "end_s": 6.0,
                "text": "Come here, robot!",
            },
        ]
    )


def test_load_corpus_accepts_transcript_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The loader renames transcript to text and drops empty rows."""
    df = corpus_frame().rename(columns={"text": "transcript"})
    df.loc[1, "transcript"] = "  "
    path = tmp_path / "corpus.parquet"
    df.to_parquet(path, index=False)

    loaded = load_utterance_corpus(path)
    assert "text" in loaded.columns
    assert len(loaded) == 1


def test_context_and_prompt_rendering() -> None:
    """The context marks the target line; the template fills both slots."""
    texts = ["one", "two", "three"]
    context = build_context(texts, 1, radius=1)
    assert ">> two" in context
    assert "   one" in context or "  one" in context

    prompt = render_speech_prompt("A {CONTEXT} B {TEXT} C", "t", "ctx")
    assert prompt == "A ctx B t C"


def test_extract_json_object() -> None:
    """The parser finds the JSON object inside noise."""
    assert extract_json_object('noise {"a": 1} tail') == {"a": 1}
    assert extract_json_object("no json here") is None


def test_classify_and_consolidate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The classifier streams JSONL; the parquet carries the eavesdrop flag."""
    server, thread = start_stub_server(response_payload=SPEECH_PAYLOAD)
    try:
        jsonl = tmp_path / "speech.jsonl"
        warnings = classify_corpus(
            corpus_frame(), server_url=server.base_url, out_jsonl=jsonl
        )
        assert warnings["calls"] == 2
        assert warnings["request_errors"] == 0

        df = consolidate_speech(jsonl, tmp_path / "speech.parquet")
        assert list(df.columns) == list(SPEECH_COLUMNS)
        assert len(df) == 2
        assert (df["address"] == "about_robot").all()
        assert df["eavesdrop"].all()

        # A second run skips the classified rows.
        warnings2 = classify_corpus(
            corpus_frame(), server_url=server.base_url, out_jsonl=jsonl
        )
        assert warnings2["calls"] == 0
        assert warnings2["skipped_resume"] == 2
    finally:
        stop_stub_server(server, thread)


def test_operator_speech_is_not_eavesdrop(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Operator talk about the robot does not count as eavesdropping."""
    payload = dict(SPEECH_PAYLOAD, operator=True)
    server, thread = start_stub_server(response_payload=payload)
    try:
        jsonl = tmp_path / "speech.jsonl"
        classify_corpus(corpus_frame(), server_url=server.base_url, out_jsonl=jsonl)
        df = consolidate_speech(jsonl, tmp_path / "speech.parquet")
        assert not df["eavesdrop"].any()
    finally:
        stop_stub_server(server, thread)


def test_jsonl_records_carry_ids(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Each JSONL line keeps the corpus keys for joins."""
    server, thread = start_stub_server(response_payload=SPEECH_PAYLOAD)
    try:
        jsonl = tmp_path / "speech.jsonl"
        classify_corpus(corpus_frame(), server_url=server.base_url, out_jsonl=jsonl)
        lines = [json.loads(l) for l in jsonl.read_text().splitlines()]
        assert {l["doc_id"] for l in lines} == {"0", "1"}
        assert all(l["sample_id"] == "clip_a" for l in lines)
    finally:
        stop_stub_server(server, thread)
