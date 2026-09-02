"""Run the C++ trashbot_pipe binary over manifest rows.

The binary processes one video per call. It reads a JSON config, calls
the vLLM server once per analysis window, and writes one JSON line per
window to the output file.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_MODEL: str = "google/gemma-4-12B-it"
DEFAULT_FPS: float = 1.0
DEFAULT_BATCH: int = 8
DEFAULT_JPEG_QUALITY: int = 90
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_SEED: int = 42
DEFAULT_TIMEOUT_S: int = 120
API_KEY: str = "EMPTY"


@dataclass(frozen=True)
class ServerSettings:
    """Connection and sampling settings for one run."""

    server_url: str
    model: str = DEFAULT_MODEL
    fps: float = DEFAULT_FPS
    batch: int = DEFAULT_BATCH


@dataclass(frozen=True)
class VideoRunResult:
    """Outcome of one binary invocation."""

    sample_id: str
    video_path: str
    jsonl_path: Path
    returncode: int
    stderr_tail: str

    @property
    def ok(self) -> bool:
        """Return True when the binary exited with status zero."""
        return self.returncode == 0


def normalize_server_url(server_url: str) -> str:
    """Normalize the server URL to a base URL that ends with "/v1".

    Example: "http://host:8000" becomes "http://host:8000/v1".
    """
    url = server_url.strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def default_prompt_template() -> Path:
    """Return the default prompt template path.

    The prompt lives in the config folder of the annotator tool. The
    proxemics prompt is the default contract. The old normativity
    prompt stays in config/ for reruns of the old contract.
    """
    tool_root = Path(__file__).resolve().parents[1]
    return tool_root / "config" / "prompt_proxemics.txt"


def build_config(settings: ServerSettings, prompt_template: Path) -> dict[str, object]:
    """Build the config dict for the C++ binary."""
    return {
        "endpoint_url": normalize_server_url(settings.server_url),
        "model": settings.model,
        "api_key": API_KEY,
        "fps": settings.fps,
        "batch": settings.batch,
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "seed": DEFAULT_SEED,
        "timeout_s": DEFAULT_TIMEOUT_S,
        "prompt_template": str(prompt_template.resolve()),
    }


def require_binary(binary: str | Path) -> Path:
    """Check that the C++ binary exists. Return its path.

    Raise FileNotFoundError with a build hint when it does not exist.
    """
    path = Path(binary)
    if not path.is_file():
        raise FileNotFoundError(
            f"trashbot_pipe binary not found at '{path}'. "
            "Build the C++ VideoPipe app first "
            "(expected at tools/trashbot_annotator/build/trashbot_pipe)."
        )
    return path


def jsonl_path_for(out_dir: Path, sample_id: str) -> Path:
    """Return the output JSONL path for one sample id."""
    return out_dir / f"{sample_id}.jsonl"


def versions_env_path() -> Path:
    """Return the path of the vendored toolchain versions file."""
    tool_root = Path(__file__).resolve().parents[1]
    return tool_root / "third_party" / "versions.env"


def read_versions_env() -> dict[str, str]:
    """Read KEY="value" pairs from versions.env. Return {} when absent.

    The setup script writes the file. It records the vendored OpenCV and
    FFmpeg roots. We need those roots for the child library search path.
    """
    path = versions_env_path()
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" not in line or line.startswith("#"):
            continue
        key, _, raw = line.partition("=")
        value = raw.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def work_root(out_dir: Path) -> Path:
    """Choose the folder for the per-call scratch files.

    Warning: do not write to /tmp. This cluster gives /tmp 15 GB for all
    users, and one batch of runs filled it. The default sits next to the
    output. TRASHBOT_TMPDIR overrides it, for example with a job-local
    disk.
    """
    override = os.environ.get("TRASHBOT_TMPDIR", "").strip()
    root = Path(override) if override else out_dir / ".work"
    root.mkdir(parents=True, exist_ok=True)
    return root


def child_env(log_dir: Path | None = None) -> dict[str, str]:
    """Build the child process environment for one binary call.

    The binary links against the vendored OpenCV and the FFmpeg tree.
    Both must appear in LD_LIBRARY_PATH when the caller did not source
    env.sh. Missing entries degrade to the inherited environment.

    log_dir sets TRASHBOT_LOG_DIR, which moves the VideoPipe log file off
    /tmp.
    """
    env = os.environ.copy()
    if log_dir is not None:
        env["TRASHBOT_LOG_DIR"] = str(log_dir)
    versions = read_versions_env()
    libs = [
        str(Path(versions[key]) / "lib")
        for key in ("OPENCV_ROOT", "FFMPEG_ROOT")
        if versions.get(key)
    ]
    if not libs:
        return env
    existing = env.get("LD_LIBRARY_PATH", "")
    merged = ":".join([*libs, existing]) if existing else ":".join(libs)
    env["LD_LIBRARY_PATH"] = merged
    return env


def run_video(
    binary: Path,
    video_path: str,
    sample_id: str,
    settings: ServerSettings,
    out_dir: Path,
    prompt_template: Path,
    overlays_dir: Path | None = None,
    max_windows: int | None = None,
) -> VideoRunResult:
    """Invoke the binary once for one video.

    The runner writes a temp config file per call and removes it after
    the call. With overlays_dir set, it also asks the binary for an
    annotated overlay video named after the sample id. With
    max_windows set, the binary stops after that many windows; use it
    for smoke runs.
    """
    jsonl_path = jsonl_path_for(out_dir, sample_id)
    config = build_config(settings, prompt_template)
    work = work_root(out_dir)
    tmp = tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        prefix="trashbot_cfg_",
        dir=str(work),
        delete=False,
        encoding="utf-8",
    )
    try:
        json.dump(config, tmp)
        tmp.close()
        cmd = [
            str(binary),
            "--video",
            video_path,
            "--config",
            tmp.name,
            "--jsonl",
            str(jsonl_path),
            "--sample-id",
            sample_id,
        ]
        if overlays_dir is not None:
            overlays_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--overlay", str(overlays_dir / sample_id)]
        if max_windows is not None:
            cmd += ["--max-windows", str(max_windows)]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=child_env(log_dir=work / "vp_logs"),
        )
    finally:
        Path(tmp.name).unlink(missing_ok=True)

    if proc.returncode != 0 and jsonl_path.is_file():
        # Keep the partial output for debugging, but out of the way of
        # a later --skip-existing pass.
        jsonl_path.rename(jsonl_path.with_suffix(".jsonl.failed"))

    stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
    return VideoRunResult(
        sample_id=sample_id,
        video_path=video_path,
        jsonl_path=jsonl_path,
        returncode=proc.returncode,
        stderr_tail=" | ".join(stderr_tail),
    )


def run_manifest(
    manifest: pd.DataFrame,
    binary: str | Path,
    server_url: str,
    model: str = DEFAULT_MODEL,
    out_dir: str | Path = ".",
    fps: float | None = None,
    batch: int | None = None,
    limit: int | None = None,
    prompt_template: str | Path | None = None,
    overlays_dir: str | Path | None = None,
    max_windows: int | None = None,
    skip_existing: bool = False,
) -> list[VideoRunResult]:
    """Run the binary over each manifest row in sequence.

    Stop early when limit is set. Return one result per processed row.
    With overlays_dir set, each video also gets an annotated overlay mp4.
    With max_windows set, each video stops after that many windows.
    With skip_existing, a row whose JSONL output exists and is not
    empty does not run again; use it to resume a batch.
    """
    binary_path = require_binary(binary)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    overlays_path = Path(overlays_dir) if overlays_dir is not None else None
    template = (
        Path(prompt_template) if prompt_template is not None else default_prompt_template()
    )
    settings = ServerSettings(
        server_url=server_url,
        model=model,
        fps=DEFAULT_FPS if fps is None else fps,
        batch=DEFAULT_BATCH if batch is None else batch,
    )

    rows = manifest.to_dict("records")
    if skip_existing:
        kept = []
        for row in rows:
            done = jsonl_path_for(out_path, str(row["sample_id"]))
            if done.is_file() and done.stat().st_size > 0:
                continue
            kept.append(row)
        skipped = len(rows) - len(kept)
        if skipped:
            print(f"Skipping {skipped} video(s) with existing JSONL output")
        rows = kept
    if limit is not None:
        rows = rows[: max(limit, 0)]

    results: list[VideoRunResult] = []
    total = len(rows)
    print(f"Running trashbot_pipe on {total} video(s); server={settings.server_url}")
    for i, row in enumerate(rows, start=1):
        result = run_video(
            binary=binary_path,
            video_path=str(row["video_path"]),
            sample_id=str(row["sample_id"]),
            settings=settings,
            out_dir=out_path,
            prompt_template=template,
            overlays_dir=overlays_path,
            max_windows=max_windows,
        )
        status = "ok" if result.ok else f"FAILED rc={result.returncode}"
        print(f"  [{i}/{total}] {result.sample_id}: {status}")
        results.append(result)
    failed = sum(1 for r in results if not r.ok)
    print(f"Done: {len(results) - failed} ok, {failed} failed")
    return results
