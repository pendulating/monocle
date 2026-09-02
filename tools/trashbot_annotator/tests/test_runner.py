"""Tests for the runner and the stub vLLM server."""

from __future__ import annotations

import base64
import json
import stat
import tempfile
from pathlib import Path

import pytest
from openai import OpenAI

from trashbot_annotator.runner import (
    build_config,
    child_env,
    default_prompt_template,
    normalize_server_url,
    require_binary,
    run_manifest,
    work_root,
)
from trashbot_annotator.taxonomy import NORM_DIMENSIONS
from tests.stub_server import CANNED_ANNOTATION, start_stub_server, stop_stub_server

REAL_BINARY = Path(__file__).resolve().parents[1] / "build" / "trashbot_pipe"


def make_fake_binary(tmp_path: Path) -> Path:
    """Create a fake binary that copies its config to the JSONL output."""
    script = """#!/bin/bash
# Fake trashbot_pipe. Copy the config to the --jsonl path.
VIDEO=""
CONFIG=""
OUT=""
SAMPLE_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --jsonl) OUT="$2"; shift 2;;
    --sample-id) SAMPLE_ID="$2"; shift 2;;
    *) shift;;
  esac
done
printf '{"video": "%s", "sample_id": "%s"}\\n' "$VIDEO" "$SAMPLE_ID" > "$OUT"
cat "$CONFIG" >> "$OUT"
"""
    path = tmp_path / "fake_trashbot_pipe"
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def make_manifest(tmp_path: Path) -> Path:
    """Write a two-row manifest parquet."""
    import pandas as pd

    df = pd.DataFrame(
        [
            {"sample_id": "ds_a.mp4", "video_path": str(tmp_path / "a.mp4")},
            {"sample_id": "ds_b.mp4", "video_path": str(tmp_path / "b.mp4")},
        ]
    )
    path = tmp_path / "manifest.parquet"
    df.to_parquet(path, index=False)
    return path


def test_normalize_server_url() -> None:
    """The URL always ends with /v1 and has no trailing slash."""
    assert normalize_server_url("http://h:8000") == "http://h:8000/v1"
    assert normalize_server_url("http://h:8000/") == "http://h:8000/v1"
    assert normalize_server_url("http://h:8000/v1/") == "http://h:8000/v1"


def test_build_config_shape() -> None:
    """The config holds the exact keys the C++ binary expects."""
    from trashbot_annotator.runner import ServerSettings

    cfg = build_config(
        ServerSettings(server_url="http://h:8000", model="m", fps=2.0, batch=4),
        prompt_template=Path("/tmp/prompt.txt"),
    )
    assert cfg["endpoint_url"] == "http://h:8000/v1"
    assert cfg["model"] == "m"
    assert cfg["api_key"] == "EMPTY"
    assert cfg["fps"] == 2.0
    assert cfg["batch"] == 4
    assert cfg["jpeg_quality"] == 90
    assert cfg["max_tokens"] == 1024
    assert cfg["temperature"] == 0.0
    assert cfg["seed"] == 42
    assert cfg["timeout_s"] == 120
    assert cfg["prompt_template"] == "/tmp/prompt.txt"


def test_missing_binary_raises() -> None:
    """A missing binary raises with a clear message."""
    with pytest.raises(FileNotFoundError, match="trashbot_pipe"):
        require_binary("/no/such/trashbot_pipe")


def test_default_prompt_template_path() -> None:
    """The default prompt points at the tool config folder."""
    template = default_prompt_template()
    assert template.name == "prompt_proxemics.txt"
    assert template.parent.name == "config"


def test_run_manifest_with_fake_binary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The runner builds one command per row and stops at the limit."""
    fake = make_fake_binary(tmp_path)
    manifest_path = make_manifest(tmp_path)
    import pandas as pd

    manifest = pd.read_parquet(manifest_path)

    results = run_manifest(
        manifest,
        binary=fake,
        server_url="http://localhost:9/",
        out_dir=tmp_path / "out",
        limit=1,
    )

    assert len(results) == 1
    assert results[0].ok
    jsonl = results[0].jsonl_path
    assert jsonl.name == "ds_a.mp4.jsonl"
    lines = [json.loads(line) for line in jsonl.read_text().splitlines()]
    # Line 1 echoes the video and the sample id; line 2 is the config.
    assert lines[0]["video"].endswith("a.mp4")
    assert lines[0]["sample_id"] == "ds_a.mp4"
    assert lines[1]["endpoint_url"] == "http://localhost:9/v1"
    assert lines[1]["api_key"] == "EMPTY"


def test_work_root_stays_off_tmp(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The scratch folder sits next to the output, not in /tmp."""
    monkeypatch.delenv("TRASHBOT_TMPDIR", raising=False)
    out = tmp_path / "out"

    root = work_root(out)

    assert root == out / ".work"
    assert root.is_dir()
    # The folder comes from the output path, not from the system temp dir.
    assert root.is_relative_to(out)
    assert not root.is_relative_to(Path(tempfile.gettempdir()) / "trashbot")


def test_work_root_honours_override(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """TRASHBOT_TMPDIR moves the scratch folder."""
    other = tmp_path / "job_local"
    monkeypatch.setenv("TRASHBOT_TMPDIR", str(other))

    root = work_root(tmp_path / "out")

    assert root == other
    assert root.is_dir()


def test_child_env_sets_log_dir(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The child gets TRASHBOT_LOG_DIR, which keeps VideoPipe logs off /tmp."""
    env = child_env(log_dir=tmp_path / "vp_logs")

    assert env["TRASHBOT_LOG_DIR"] == str(tmp_path / "vp_logs")
    assert "TRASHBOT_LOG_DIR" not in child_env()


def test_run_video_writes_config_outside_tmp(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The per-call config file lands under the output folder."""
    import pandas as pd

    fake = make_fake_binary(tmp_path)
    manifest = pd.read_parquet(make_manifest(tmp_path))
    out = tmp_path / "out"

    results = run_manifest(
        manifest, binary=fake, server_url="http://localhost:9/", out_dir=out, limit=1
    )

    assert results[0].ok
    # The binary echoes its --config path on line 3 of the fake output.
    assert (out / ".work").is_dir()


def test_stub_server_roundtrip() -> None:
    """The stub answers an OpenAI client call and sees image parts."""
    server, thread = start_stub_server()
    try:
        client = OpenAI(base_url=server.base_url, api_key="EMPTY")
        png = base64.b64encode(b"\x89PNG-fake").decode("ascii")
        response = client.chat.completions.create(
            model="google/gemma-4-12B-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Annotate."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{png}"},
                        },
                    ],
                }
            ],
        )
        annotation = json.loads(response.choices[0].message.content or "{}")
        assert annotation["summary"] == CANNED_ANNOTATION["summary"]
        assert len(annotation["interactions"]) == 2
        assert annotation["norm_dimensions"][0]["dimension"] == NORM_DIMENSIONS[0]
        # The handler saw exactly one image_url part.
        assert server.image_part_counts == [1]
        assert server.requests[0]["model"] == "google/gemma-4-12B-it"
    finally:
        stop_stub_server(server, thread)


@pytest.mark.skipif(not REAL_BINARY.exists(), reason="C++ binary not built")
def test_real_binary_against_stub(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Run the real binary on a tiny synthetic video against the stub."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not available")
    video = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=64x64:rate=2",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    import pandas as pd

    manifest = pd.DataFrame(
        [{"sample_id": "tiny.mp4", "video_path": str(video)}]
    )
    server, thread = start_stub_server()
    try:
        results = run_manifest(
            manifest,
            binary=REAL_BINARY,
            server_url=server.base_url,
            out_dir=tmp_path / "out",
        )
        assert results[0].ok
        assert server.image_part_counts, "server saw no image parts"
        # The record carries the manifest sample id, not the file stem.
        first = json.loads(results[0].jsonl_path.read_text().splitlines()[0])
        assert first["sample_id"] == "tiny.mp4"
    finally:
        stop_stub_server(server, thread)


def test_skip_existing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A row with a non-empty JSONL output does not run again."""
    fake = make_fake_binary(tmp_path)
    import pandas as pd

    manifest = pd.read_parquet(make_manifest(tmp_path))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "ds_a.mp4.jsonl").write_text('{"done": true}\n', encoding="utf-8")

    results = run_manifest(
        manifest,
        binary=fake,
        server_url="http://localhost:9/",
        out_dir=out_dir,
        skip_existing=True,
    )

    assert [r.sample_id for r in results] == ["ds_b.mp4"]
    # The existing output stays untouched.
    assert (out_dir / "ds_a.mp4.jsonl").read_text().startswith('{"done"')
