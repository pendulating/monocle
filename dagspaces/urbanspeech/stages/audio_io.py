"""Shared WAV reader for the urbanspeech stages.

extract_audio guarantees 16-bit PCM mono WAVs, so both the VAD pass
(extract_audio) and the ASR pass can read them with the stdlib ``wave``
module — no librosa/soundfile dependency.
"""

from __future__ import annotations

import wave
from typing import Tuple

import numpy as np


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV into a float32 mono array in [-1, 1]."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample_width={sample_width}: {path}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return audio, sr
