"""Speech recognition stage: granite-speech transcription via vLLM.

Consumes the audio manifest written by the extract_audio stage (16 kHz mono
PCM WAV per video) and produces one transcript per video.

granite-speech specifics (per the model card):
  - The speech encoder is wired in through a LoRA adapter that ships inside
    the model repo, so the engine needs ``enable_lora=True`` /
    ``max_lora_rank=64`` and every audio request must carry
    ``LoRARequest("speech", 1, <model dir>)``.
  - The user turn embeds the audio with a literal ``<|audio|>`` placeholder;
    the waveform itself goes in ``multi_modal_data={"audio": (array, sr)}``.

Long clips are split into ``asr.chunk_seconds`` windows (the model is built
for short utterances; default max_model_len is 2048), transcribed chunk by
chunk, and re-joined per video. WAVs are read with the stdlib ``wave``
module — extract_audio guarantees 16-bit PCM, so no librosa/soundfile needed.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from dagspaces.common.vllm_inference import (
    _build_engine_kwargs,
    _shutdown_llm,
    get_pcie_nccl_env_vars,
    get_vllm_runtime_env_vars,
)

from .audio_io import read_wav as _read_wav
from .vad import pack_segments


# ---------------------------------------------------------------------------
# Audio loading / chunking
# ---------------------------------------------------------------------------

def _segment_bounds(
    segments: List[Tuple[float, float]], sr: int, n_samples: int, chunk_s: float, join_gap_s: float
) -> List[Tuple[int, int]]:
    """Convert VAD speech segments (seconds) into sample-index windows.

    Packs segments into <= chunk_s windows (so non-speech audio between
    utterances is never transcribed) and clamps to the waveform length.
    """
    windows = pack_segments(segments, chunk_s, join_gap_s=join_gap_s)
    bounds: List[Tuple[int, int]] = []
    for start_s, end_s in windows:
        start = max(0, int(round(start_s * sr)))
        end = min(n_samples, int(round(end_s * sr)))
        if end > start:
            bounds.append((start, end))
    return bounds


def _chunk_bounds(
    n_samples: int, sr: int, chunk_s: float, overlap_s: float, min_chunk_s: float
) -> List[Tuple[int, int]]:
    """Split [0, n_samples) into windows of chunk_s with optional overlap."""
    chunk = max(int(chunk_s * sr), 1)
    step = max(chunk - int(overlap_s * sr), 1)
    min_samples = int(min_chunk_s * sr)
    bounds = []
    start = 0
    while start < n_samples:
        end = min(start + chunk, n_samples)
        if end - start >= min_samples or not bounds:
            bounds.append((start, end))
        start += step
    return bounds


def _is_silent(chunk: np.ndarray, rms_threshold: float) -> bool:
    """True if a chunk's RMS energy is below rms_threshold.

    granite-speech (like every attention-based ASR) hallucinates filler such
    as "Thank you. Thank you." on silent / near-silent audio, so chunks below
    the threshold are dropped before inference rather than transcribed. The
    waveform is float32 in [-1, 1]; a quiet room sits around 0.001–0.005 RMS
    and conversational speech is typically >0.02, so ~0.005–0.01 is a safe
    cutoff. ``rms_threshold <= 0`` disables the gate (transcribe everything).
    """
    if rms_threshold <= 0.0 or chunk.size == 0:
        return False
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return rms < rms_threshold


# ---------------------------------------------------------------------------
# Hallucination post-filter
# ---------------------------------------------------------------------------

# Default filler phrases granite-speech emits over residual non-speech audio
# that survives VAD (e.g. music, crowd noise). Matched as substrings against a
# normalized (lowercased, punctuation-stripped) chunk; only *whole* chunks that
# are essentially just the filler are flagged.
_DEFAULT_HALLUCINATION_PHRASES = [
    "thank you for watching",
    "thanks for watching",
    "thank you very much for watching",
    "thank you for your attention",
    "thank you very much for your attention",
    "thank you very much",
    "thank you",
    "please subscribe",
    "see you next time",
    "i'll see you next time",
]


def _normalize_for_match(text: str) -> str:
    """Lowercase and strip punctuation/whitespace for phrase matching."""
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


def _make_hallucination_flagger(cfg: DictConfig):
    """Return (enabled, flag_fn) where flag_fn(text) -> True if filler.

    A chunk is flagged when, after normalization, it is *dominated* by a
    blocklist phrase (the phrase covers >= ``min_coverage`` of its characters).
    This drops "Thank you very much for watching." while sparing a real
    sentence that merely contains "thank you" in passing.
    """
    fcfg = getattr(cfg.asr, "hallucination_filter", None)
    enabled = bool(getattr(fcfg, "enabled", False)) if fcfg is not None else False
    if not enabled:
        return False, (lambda _t: False)

    phrases_raw = getattr(fcfg, "phrases", None)
    phrases = [str(p) for p in phrases_raw] if phrases_raw else list(_DEFAULT_HALLUCINATION_PHRASES)
    norm_phrases = sorted(
        (p for p in (_normalize_for_match(p) for p in phrases) if p),
        key=len,
        reverse=True,
    )
    min_coverage = float(getattr(fcfg, "min_coverage", 0.6))

    def _flag(text: str) -> bool:
        norm = _normalize_for_match(text)
        if not norm:
            return False
        # Strip every blocklist phrase (repeatedly) and see how much non-filler
        # text remains. A chunk that is essentially all filler — including
        # repeated/concatenated phrases like "thank you thank you for watching"
        # — collapses to near-empty and is flagged; a real sentence that merely
        # ends with "thanks for watching" keeps a large residual and is spared.
        residual = norm
        for ph in norm_phrases:
            if ph:
                residual = residual.replace(ph, " ")
        residual = "".join(residual.split())
        return len(residual) <= (1.0 - min_coverage) * len(norm)

    return True, _flag


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_asr_stage(cfg: DictConfig) -> str:
    """Transcribe every audio file in the manifest; write per-video parquet.

    Returns:
        Path to the output parquet.
    """
    manifest_path = str(getattr(cfg.asr, "audio_manifest_path", None) or "")
    if not manifest_path:
        raise ValueError("asr.audio_manifest_path must be set (extract_audio output)")
    df = pd.read_parquet(manifest_path)
    print(f"[asr] Read manifest: {manifest_path} ({len(df)} rows)", flush=True)

    sample_n = getattr(getattr(cfg, "runtime", {}), "sample_n", None)
    if sample_n:
        df = df.head(int(sample_n))
        print(f"[asr] Limited to {len(df)} rows for debug", flush=True)

    output_path = str(getattr(cfg.runtime, "output_path", None) or "")
    if not output_path:
        output_path = os.path.abspath("transcripts.parquet")
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chunk_s = float(getattr(cfg.asr, "chunk_seconds", 30.0))
    overlap_s = float(getattr(cfg.asr, "chunk_overlap_seconds", 0.0))
    min_chunk_s = float(getattr(cfg.asr, "min_chunk_seconds", 0.2))
    batch_size = int(getattr(cfg.asr, "batch_size", 64)) or 64
    output_column = str(getattr(cfg.asr, "output_column", "transcript"))
    question = str(getattr(cfg.asr, "question", "can you transcribe the speech into a written format?"))

    # VAD-driven chunking: when the manifest carries per-video speech_segments
    # (written by extract_audio's VAD pass), transcribe only those windows so
    # non-speech audio never reaches the model. Falls back to whole-file
    # windowing when segments are absent (old manifests) or use_vad disabled.
    use_vad = bool(getattr(cfg.asr, "use_vad_segments", True))
    vad_join_gap_s = float(getattr(cfg.asr, "vad_join_gap_seconds", 1.0))
    has_vad_col = "speech_segments" in df.columns

    # Anti-repetition / hallucination controls. granite-speech is a small
    # autoregressive model and, under plain greedy decoding, both loops on hard
    # audio ("Literally, this is the set." x20) and emits filler on silence
    # ("Thank you. Thank you."). repetition/frequency/presence penalties break
    # the loops; the RMS gate drops silent chunks before they reach the model.
    repetition_penalty = float(getattr(cfg.asr, "repetition_penalty", 1.0) or 1.0)
    frequency_penalty = float(getattr(cfg.asr, "frequency_penalty", 0.0) or 0.0)
    presence_penalty = float(getattr(cfg.asr, "presence_penalty", 0.0) or 0.0)
    silence_rms = float(getattr(cfg.asr, "silence_rms", 0.0) or 0.0)

    # Set runtime env vars before importing vLLM.
    for k, v in {**get_pcie_nccl_env_vars(), **get_vllm_runtime_env_vars()}.items():
        os.environ.setdefault(k, v)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_source = str(cfg.model.model_source)
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"<|audio|>{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"[asr] Prompt: {prompt!r}", flush=True)

    engine_kwargs = _build_engine_kwargs(cfg)
    # This stage drives a single engine directly; drop the auto-detected DP
    # hint that run_vllm_inference() would otherwise consume.
    engine_kwargs.pop("data_parallel_size", None)
    print(f"[asr] Initializing vLLM: "
          f"{ {k: v for k, v in engine_kwargs.items() if k != 'model'} }", flush=True)
    print(f"[asr] Model: {model_source}", flush=True)
    llm = LLM(**engine_kwargs)

    lora_request = None
    if bool(getattr(cfg.model, "speech_lora", True)):
        # The speech adapter lives inside the model repo itself.
        lora_request = LoRARequest("speech", 1, model_source)
        print(f"[asr] Speech LoRA: {model_source}", flush=True)

    sampling_params = SamplingParams(
        temperature=float(getattr(cfg.asr, "temperature", 0.0)),
        max_tokens=int(getattr(cfg.asr, "max_tokens", 256)),
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    print(f"[asr] Sampling: temp={sampling_params.temperature} "
          f"max_tokens={sampling_params.max_tokens} "
          f"repetition_penalty={repetition_penalty} "
          f"frequency_penalty={frequency_penalty} "
          f"presence_penalty={presence_penalty} silence_rms={silence_rms}", flush=True)

    streaming_io = bool(getattr(getattr(cfg, "runtime", {}), "streaming_io", False))
    parts_dir = output_path + ".parts"
    if streaming_io:
        os.makedirs(parts_dir, exist_ok=True)

    rows = df.to_dict("records")
    chunk_transcripts: Dict[int, Dict[int, str]] = {}
    asr_errors: Dict[int, str] = {}

    # Stream rows -> chunk batches so only one batch of waveforms is resident.
    pending: List[Dict[str, Any]] = []  # {row_idx, chunk_idx, start_s, end_s, audio}
    n_chunks_done = 0
    n_parts = 0
    n_skipped_silent = 0

    def _flush_batch(batch: List[Dict[str, Any]]) -> None:
        nonlocal n_chunks_done, n_parts
        inputs = [
            {
                "prompt": prompt,
                "multi_modal_data": {"audio": (item["audio"], item["sr"])},
            }
            for item in batch
        ]
        outputs = llm.generate(inputs, sampling_params, lora_request=lora_request)
        part_rows = []
        for item, out in zip(batch, outputs):
            text = out.outputs[0].text.strip() if out.outputs else ""
            chunk_transcripts.setdefault(item["row_idx"], {})[item["chunk_idx"]] = text
            part_rows.append({
                "sample_id": rows[item["row_idx"]].get("sample_id"),
                "chunk_idx": item["chunk_idx"],
                "start_s": item["start_s"],
                "end_s": item["end_s"],
                "text": text,
            })
        n_chunks_done += len(batch)
        print(f"[asr] {n_chunks_done} chunks transcribed", flush=True)
        if streaming_io and part_rows:
            part_path = os.path.join(parts_dir, f"part_{n_parts:06d}.parquet")
            pd.DataFrame(part_rows).to_parquet(part_path, index=False)
            n_parts += 1

    for row_idx, row in enumerate(rows):
        audio_path = row.get("audio_path")
        if not row.get("has_audio") or not audio_path:
            continue
        try:
            audio, sr = _read_wav(str(audio_path))
        except Exception as exc:  # noqa: BLE001 — bad files must not kill the run
            asr_errors[row_idx] = f"wav_read_failed: {exc}"
            continue

        # Prefer VAD speech windows; fall back to whole-file chunking.
        segments = row.get("speech_segments") if has_vad_col else None
        if use_vad and segments is not None:
            seg_pairs = [(float(s[0]), float(s[1])) for s in segments]
            if not seg_pairs:
                # VAD ran and found no speech -> nothing to transcribe.
                asr_errors[row_idx] = "no_speech"
                continue
            bounds = _segment_bounds(seg_pairs, sr, len(audio), chunk_s, vad_join_gap_s)
        else:
            bounds = _chunk_bounds(len(audio), sr, chunk_s, overlap_s, min_chunk_s)

        for chunk_idx, (start, end) in enumerate(bounds):
            clip = audio[start:end]
            # Drop silent/near-silent chunks rather than letting the model
            # hallucinate filler over them (no-op when silence_rms <= 0).
            if _is_silent(clip, silence_rms):
                n_skipped_silent += 1
                continue
            pending.append({
                "row_idx": row_idx,
                "chunk_idx": chunk_idx,
                "start_s": start / sr,
                "end_s": end / sr,
                "audio": clip,
                "sr": sr,
            })
            if len(pending) >= batch_size:
                _flush_batch(pending)
                pending = []
    if pending:
        _flush_batch(pending)
        pending = []

    _shutdown_llm(llm, stage_name="asr")

    # Post-filter: flag residual hallucinated filler and collapse cross-chunk
    # repeats when joining. Flagged chunks are excluded from the joined
    # transcript but retained in chunk_transcripts (+ chunk_flags) for audit.
    filter_enabled, is_hallucination = _make_hallucination_flagger(cfg)
    collapse_dupes = bool(
        getattr(getattr(cfg.asr, "hallucination_filter", None), "collapse_consecutive_duplicates", True)
    )

    # Assemble per-video transcripts.
    transcripts: List[Any] = []
    chunk_lists: List[Any] = []
    flag_lists: List[Any] = []
    n_chunks_col: List[int] = []
    n_filtered_col: List[int] = []
    error_col: List[Any] = []
    n_filtered_total = 0
    for row_idx, row in enumerate(rows):
        chunks = chunk_transcripts.get(row_idx)
        if chunks:
            ordered = [chunks[i] for i in sorted(chunks)]
            flags = [bool(t) and is_hallucination(t) for t in ordered]
            n_filtered_total += sum(flags)
            kept: List[str] = []
            prev_norm = None
            for text, flagged in zip(ordered, flags):
                if not text or flagged:
                    continue
                norm = _normalize_for_match(text)
                if collapse_dupes and norm and norm == prev_norm:
                    continue
                kept.append(text)
                prev_norm = norm
            transcripts.append(" ".join(kept).strip() or None)
            chunk_lists.append(ordered)
            flag_lists.append(flags)
            n_chunks_col.append(len(ordered))
            n_filtered_col.append(int(sum(flags)))
            error_col.append(None if kept else "all_chunks_filtered")
        else:
            transcripts.append(None)
            chunk_lists.append(None)
            flag_lists.append(None)
            n_chunks_col.append(0)
            n_filtered_col.append(0)
            error_col.append(asr_errors.get(row_idx) or row.get("extract_error") or "no_audio")

    df[output_column] = transcripts
    df["chunk_transcripts"] = chunk_lists
    df["chunk_flags"] = flag_lists
    df["n_chunks"] = n_chunks_col
    df["n_chunks_filtered"] = n_filtered_col
    df["asr_error"] = error_col
    df["asr_model_source"] = model_source

    df.to_parquet(output_path, index=False)
    n_ok = int(df[output_column].notna().sum())
    print(f"[asr] Wrote {output_path}: {n_ok}/{len(df)} transcribed "
          f"({n_chunks_done} chunks, {n_skipped_silent} silent chunks skipped, "
          f"{n_filtered_total} chunks flagged as filler"
          f"{' [filter off]' if not filter_enabled else ''})",
          flush=True)
    return output_path
