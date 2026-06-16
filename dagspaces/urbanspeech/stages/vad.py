"""Voice Activity Detection for the urbanspeech pipeline (Silero VAD).

granite-speech — like every attention-based ASR model — hallucinates
YouTube-outro filler ("Thank you for watching", "Thank you very much")
over *non-speech* audio. Urban video clips are dominated by ambient
traffic/wind, which is non-speech but not silent, so the cheap RMS energy
gate in the ASR stage cannot catch it (loud != speech).

Silero VAD is a tiny, CPU-friendly torch model that distinguishes speech
from noise. We run it once per video in the (CPU) extract_audio stage and
persist the resulting speech intervals into the manifest, so the GPU ASR
stage only ever transcribes audio that actually contains speech — which is
both the correct fix for the hallucinations and a large compute saving
(non-speech chunks never reach the model).

This mirrors the VAD-gating approach faster-whisper / WhisperX use to solve
the same failure mode.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Silero VAD operates on 16 kHz (or 8 kHz) mono audio. extract_audio already
# resamples to 16 kHz, so this is the expected path.
_SUPPORTED_SR = (8000, 16000)

# Module-level singleton — the model is ~2 MB and load is cheap, but reloading
# it per video would dominate the VAD pass.
_MODEL = None


def load_vad_model():
    """Lazily load and cache the Silero VAD model (torch, CPU)."""
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def speech_segments(
    audio: np.ndarray,
    sr: int,
    *,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 200,
    max_speech_duration_s: float = 30.0,
) -> List[Tuple[float, float]]:
    """Return merged speech intervals (start_s, end_s) for a mono waveform.

    ``audio`` is float32 in [-1, 1]; ``sr`` must be 8 kHz or 16 kHz (Silero's
    supported rates). Returns an empty list when no speech is detected.
    """
    if sr not in _SUPPORTED_SR:
        raise ValueError(
            f"Silero VAD supports {_SUPPORTED_SR} Hz, got {sr} Hz. "
            "extract_audio should resample to 16 kHz before VAD."
        )
    if audio.size == 0:
        return []

    import torch
    from silero_vad import get_speech_timestamps

    model = load_vad_model()
    # Silero expects a float32 torch tensor; ensure contiguous (np.frombuffer
    # views are read-only, which torch.from_numpy dislikes).
    tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    ts = get_speech_timestamps(
        tensor,
        model,
        threshold=threshold,
        sampling_rate=sr,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    return [(float(seg["start"]), float(seg["end"])) for seg in ts]


def total_speech_seconds(segments: List[Tuple[float, float]]) -> float:
    """Sum the duration of a list of (start_s, end_s) segments."""
    return float(sum(max(0.0, end - start) for start, end in segments))


def pack_segments(
    segments: List[Tuple[float, float]],
    chunk_seconds: float,
    *,
    join_gap_s: float = 0.0,
) -> List[Tuple[float, float]]:
    """Pack speech intervals into transcription windows of <= chunk_seconds.

    Adjacent speech segments separated by <= ``join_gap_s`` are merged before
    packing so a single utterance split by a brief pause is not fragmented.
    A segment longer than ``chunk_seconds`` is split into chunk_seconds slices
    (granite-speech's max_model_len caps how much audio one request can hold).
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s[0])
    chunk = max(float(chunk_seconds), 0.1)

    windows: List[Tuple[float, float]] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None

    def _emit(start: float, end: float) -> None:
        # Split windows that exceed chunk_seconds (long uninterrupted speech).
        s = start
        while end - s > chunk:
            windows.append((s, s + chunk))
            s += chunk
        if end - s > 0:
            windows.append((s, end))

    for seg_start, seg_end in ordered:
        if cur_start is None:
            cur_start, cur_end = seg_start, seg_end
            continue
        # Extend the current window if the segment is contiguous (within
        # join_gap_s) AND the window stays within chunk_seconds.
        if seg_start - cur_end <= join_gap_s and seg_end - cur_start <= chunk:
            cur_end = seg_end
        else:
            _emit(cur_start, cur_end)
            cur_start, cur_end = seg_start, seg_end
    if cur_start is not None:
        _emit(cur_start, cur_end)
    return windows
