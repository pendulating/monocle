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

# .insv is Insta360's raw container — an MP4-family stream ffmpeg can decode
# audio from directly (the proprietary metadata it ignores).
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".ts", ".m4v", ".insv"}


# ---------------------------------------------------------------------------
# Input listing
# ---------------------------------------------------------------------------

def _sanitize_sample_id(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def _exclude_tokens(cfg: DictConfig) -> List[str]:
    """Blacklist substrings from data.video_exclude (str or list)."""
    raw = getattr(cfg.data, "video_exclude", None)
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [str(t).lower() for t in raw if str(t).strip()]


def _is_excluded(rel_path: str, tokens: List[str]) -> bool:
    """True if any blacklist token appears (case-insensitive) in rel_path."""
    low = rel_path.lower()
    return any(tok in low for tok in tokens)


def _list_videos(cfg: DictConfig) -> pd.DataFrame:
    """Build the video listing from data.parquet_path or data.video_dir."""
    exclude = _exclude_tokens(cfg)
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
        if exclude:
            n_before = len(df)
            df = df[~df["video_path"].astype(str).str.lower().apply(
                lambda p: _is_excluded(p, exclude))].reset_index(drop=True)
            print(f"[extract_audio] Excluded {n_before - len(df)} videos "
                  f"matching {exclude}", flush=True)
        return df

    video_dir = str(getattr(cfg.data, "video_dir", "") or "")
    if not video_dir:
        raise ValueError("data.parquet_path or data.video_dir must be set")
    pattern = str(getattr(cfg.data, "video_glob", "") or "**/*")
    all_paths = [
        p for p in glob.glob(os.path.join(video_dir, pattern), recursive=True)
        if os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS
    ]
    if exclude:
        kept = [p for p in all_paths
                if not _is_excluded(os.path.relpath(p, video_dir), exclude)]
        print(f"[extract_audio] Excluded {len(all_paths) - len(kept)} videos "
              f"matching {exclude}", flush=True)
        all_paths = kept
    paths = sorted(all_paths)
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

    # Voice Activity Detection: detect speech intervals per video so the GPU
    # ASR stage only transcribes audio that actually contains speech. This is
    # the fix for granite-speech hallucinating filler ("Thank you for watching")
    # over ambient non-speech audio, which a pure RMS gate cannot catch.
    _run_vad_pass(cfg, df)

    df.to_parquet(output_path, index=False)
    n_ok = int(df["has_audio"].sum())
    print(f"[extract_audio] Wrote {output_path}: {n_ok}/{len(df)} with audio", flush=True)
    return output_path


def _run_vad_pass(cfg: DictConfig, df: pd.DataFrame) -> None:
    """Add speech_segments / n_speech_segments / speech_duration_s to df.

    Mutates ``df`` in place. Runs sequentially (Silero VAD is a GIL-bound torch
    model — keep it out of the ffmpeg thread pool) over every extracted WAV.
    Per-file VAD failures degrade gracefully to an empty segment list so a bad
    file never kills the run; with VAD disabled the columns are written as
    null, and the ASR stage falls back to whole-file chunking.
    """
    vad_cfg = getattr(cfg.audio_extraction, "vad", None)
    enabled = bool(getattr(vad_cfg, "enabled", False)) if vad_cfg is not None else False
    if not enabled:
        df["speech_segments"] = None
        df["n_speech_segments"] = None
        df["speech_duration_s"] = None
        print("[extract_audio] VAD disabled; ASR will chunk whole files", flush=True)
        return

    from .audio_io import read_wav
    from .vad import speech_segments, total_speech_seconds

    opts = dict(
        threshold=float(getattr(vad_cfg, "threshold", 0.5)),
        min_speech_duration_ms=int(getattr(vad_cfg, "min_speech_duration_ms", 250)),
        min_silence_duration_ms=int(getattr(vad_cfg, "min_silence_duration_ms", 300)),
        speech_pad_ms=int(getattr(vad_cfg, "speech_pad_ms", 200)),
        max_speech_duration_s=float(getattr(vad_cfg, "max_speech_duration_s", 30.0)),
    )
    print(f"[extract_audio] Running VAD (Silero) over extracted audio: {opts}", flush=True)

    segs_col: List[Any] = []
    n_col: List[Any] = []
    dur_col: List[Any] = []
    n_done = 0
    n_speech = 0
    for _, row in df.iterrows():
        audio_path = row.get("audio_path")
        if not row.get("has_audio") or not audio_path:
            segs_col.append(None)
            n_col.append(None)
            dur_col.append(None)
            continue
        try:
            audio, sr = read_wav(str(audio_path))
            segs = speech_segments(audio, sr, **opts)
        except Exception as exc:  # noqa: BLE001 — bad files must not kill the pass
            # Store None (not []) so ASR falls back to whole-file chunking for
            # this clip rather than dropping it as "no speech".
            print(f"[extract_audio] VAD failed for {audio_path}: {exc}", flush=True)
            segs_col.append(None)
            n_col.append(None)
            dur_col.append(None)
            continue
        # Store as list-of-lists so parquet/pyarrow round-trips cleanly.
        segs_col.append([[s, e] for s, e in segs])
        n_col.append(len(segs))
        dur_col.append(total_speech_seconds(segs))
        n_done += 1
        if segs:
            n_speech += 1

    df["speech_segments"] = segs_col
    df["n_speech_segments"] = n_col
    df["speech_duration_s"] = dur_col
    print(f"[extract_audio] VAD done: {n_speech}/{n_done} clips contain speech", flush=True)
