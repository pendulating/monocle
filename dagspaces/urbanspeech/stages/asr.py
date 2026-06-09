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
import wave
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


# ---------------------------------------------------------------------------
# Audio loading / chunking
# ---------------------------------------------------------------------------

def _read_wav(path: str) -> Tuple[np.ndarray, int]:
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
    )

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
        for chunk_idx, (start, end) in enumerate(
            _chunk_bounds(len(audio), sr, chunk_s, overlap_s, min_chunk_s)
        ):
            pending.append({
                "row_idx": row_idx,
                "chunk_idx": chunk_idx,
                "start_s": start / sr,
                "end_s": end / sr,
                "audio": audio[start:end],
                "sr": sr,
            })
            if len(pending) >= batch_size:
                _flush_batch(pending)
                pending = []
    if pending:
        _flush_batch(pending)
        pending = []

    _shutdown_llm(llm, stage_name="asr")

    # Assemble per-video transcripts.
    transcripts: List[Any] = []
    chunk_lists: List[Any] = []
    n_chunks_col: List[int] = []
    error_col: List[Any] = []
    for row_idx, row in enumerate(rows):
        chunks = chunk_transcripts.get(row_idx)
        if chunks:
            ordered = [chunks[i] for i in sorted(chunks)]
            transcripts.append(" ".join(t for t in ordered if t).strip())
            chunk_lists.append(ordered)
            n_chunks_col.append(len(ordered))
            error_col.append(None)
        else:
            transcripts.append(None)
            chunk_lists.append(None)
            n_chunks_col.append(0)
            error_col.append(asr_errors.get(row_idx) or row.get("extract_error") or "no_audio")

    df[output_column] = transcripts
    df["chunk_transcripts"] = chunk_lists
    df["n_chunks"] = n_chunks_col
    df["asr_error"] = error_col
    df["asr_model_source"] = model_source

    df.to_parquet(output_path, index=False)
    n_ok = int(df[output_column].notna().sum())
    print(f"[asr] Wrote {output_path}: {n_ok}/{len(df)} transcribed "
          f"({n_chunks_done} chunks)", flush=True)
    return output_path
