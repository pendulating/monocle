"""Audio extraction stage: isolate audio tracks from video clips via ffmpeg.

Decodes each video's audio stream to 16 kHz mono 16-bit PCM WAV — the input
format granite-speech expects — so the GPU ASR stage never touches video
containers. Extraction is CPU-only and parallelised with a thread pool
(ffmpeg subprocesses release the GIL), so it runs on a cheap CPU launcher.

Output is a manifest parquet with one row per video:
  sample_id, video_path, audio_path, has_audio, audio_duration_s,
  sample_rate, extract_error (+ any metadata columns from the input parquet)
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from omegaconf import DictConfig

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts", ".m4v"}


# ---------------------------------------------------------------------------
# Input listing
# ---------------------------------------------------------------------------

def _sanitize_sample_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def _list_videos(cfg: DictConfig) -> pd.DataFrame:
    """Build the video listing from data.parquet_path or data.video_dir."""
    parquet_path = str(getattr(cfg.data, "parquet_path", "") or "")
    if parquet_path:
        df = pd.read_parquet(parquet_path)
        columns = getattr(cfg.data, "columns", None)
        video_col = str(getattr(columns, "video_path", "video_path")) if columns else "video_path"
        id_col = str(getattr(columns, "sample_id", "sample_id")) if columns else "sample_id"
        if video_col not in df.columns:
            raise ValueError(
                f"Column '{video_col}' not found in {parquet_path} "
                f"(available: {list(df.columns)})"
            )
        df = df.rename(columns={video_col: "video_path"})
        if id_col in df.columns and id_col != "sample_id":
            df = df.rename(columns={id_col: "sample_id"})
        if "sample_id" not in df.columns:
            df["sample_id"] = [
                _sanitize_sample_id(os.path.splitext(os.path.basename(p))[0])
                for p in df["video_path"]
            ]
        return df

    video_dir = str(getattr(cfg.data, "video_dir", "") or "")
    if not video_dir:
        raise ValueError("data.parquet_path or data.video_dir must be set")
    pattern = str(getattr(cfg.data, "video_glob", "") or "**/*.mp4")
    paths = sorted(
        p for p in glob.glob(os.path.join(video_dir, pattern), recursive=True)
        if os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No videos matched {pattern!r} under {video_dir}")
    sample_ids: List[str] = []
    seen: Dict[str, int] = {}
    for p in paths:
        sid = _sanitize_sample_id(os.path.splitext(os.path.relpath(p, video_dir))[0])
        if sid in seen:
            seen[sid] += 1
            sid = f"{sid}_{seen[sid]}"
        else:
            seen[sid] = 0
        sample_ids.append(sid)
    return pd.DataFrame({"sample_id": sample_ids, "video_path": paths})


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg
# ---------------------------------------------------------------------------

def _probe_video(video_path: str) -> Dict[str, Any]:
    """Return {'has_audio': bool, 'duration_s': float|None} for a video."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type",
            "-of", "json", video_path,
        ],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[-500:]}")
    info = json.loads(proc.stdout or "{}")
    has_audio = any(
        s.get("codec_type") == "audio" for s in info.get("streams", [])
    )
    duration: Optional[float] = None
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass
    return {"has_audio": has_audio, "duration_s": duration}


def _extract_one(
    sample_id: str,
    video_path: str,
    audio_dir: str,
    sample_rate: int,
    skip_existing: bool,
) -> Dict[str, Any]:
    """Extract one video's audio track to 16-bit PCM WAV."""
    result: Dict[str, Any] = {
        "sample_id": sample_id,
        "audio_path": None,
        "has_audio": False,
        "audio_duration_s": None,
        "sample_rate": sample_rate,
        "extract_error": None,
    }
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        probe = _probe_video(video_path)
        result["audio_duration_s"] = probe["duration_s"]
        if not probe["has_audio"]:
            result["extract_error"] = "no_audio_stream"
            return result

        audio_path = os.path.join(audio_dir, f"{sample_id}.wav")
        if not (skip_existing and os.path.exists(audio_path) and os.path.getsize(audio_path) > 44):
            tmp_path = audio_path + ".tmp.wav"
            proc = subprocess.run(
                [
                    "ffmpeg", "-y", "-nostdin", "-v", "error",
                    "-i", video_path,
                    "-vn", "-ac", "1", "-ar", str(sample_rate),
                    "-c:a", "pcm_s16le", "-f", "wav",
                    tmp_path,
                ],
                capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")
            os.replace(tmp_path, audio_path)

        result["audio_path"] = audio_path
        result["has_audio"] = True
    except Exception as exc:  # noqa: BLE001 — per-video failures must not kill the batch
        result["extract_error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_extract_audio_stage(cfg: DictConfig) -> str:
    """Extract audio from all input videos; write the manifest parquet.

    Returns:
        Path to the manifest parquet.
    """
    df = _list_videos(cfg)

    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if sample_n:
        df = df.head(int(sample_n))
        print(f"[extract_audio] Limited to {len(df)} rows for debug", flush=True)

    output_path = str(getattr(cfg.runtime, "output_path", None) or "")
    if not output_path:
        output_path = os.path.abspath("audio_manifest.parquet")
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    audio_dir = str(getattr(cfg.audio_extraction, "audio_dir", None) or "")
    if not audio_dir:
        audio_dir = os.path.join(os.path.dirname(output_path), "audio")
    audio_dir = os.path.abspath(os.path.expanduser(audio_dir))
    os.makedirs(audio_dir, exist_ok=True)

    sample_rate = int(getattr(cfg.audio_extraction, "sample_rate", 16000))
    num_workers = int(getattr(cfg.audio_extraction, "num_workers", 8))
    skip_existing = bool(getattr(cfg.audio_extraction, "skip_existing", True))

    rows = df.to_dict("records")
    print(f"[extract_audio] {len(rows)} videos -> {audio_dir} "
          f"(sr={sample_rate}, workers={num_workers})", flush=True)

    results: Dict[str, Dict[str, Any]] = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(
                _extract_one, str(row["sample_id"]), str(row["video_path"]),
                audio_dir, sample_rate, skip_existing,
            ): str(row["sample_id"])
            for row in rows
        }
        for fut in as_completed(futures):
            res = fut.result()
            results[res["sample_id"]] = res
            n_done += 1
            if n_done % 100 == 0 or n_done == len(rows):
                n_failed = sum(1 for r in results.values() if not r["has_audio"])
                print(f"[extract_audio] {n_done}/{len(rows)} done "
                      f"({n_failed} without audio)", flush=True)

    for col in ("audio_path", "has_audio", "audio_duration_s", "sample_rate", "extract_error"):
        df[col] = [results[str(sid)][col] for sid in df["sample_id"]]

    df.to_parquet(output_path, index=False)
    n_ok = int(df["has_audio"].sum())
    print(f"[extract_audio] Wrote {output_path}: {n_ok}/{len(df)} with audio", flush=True)
    return output_path
