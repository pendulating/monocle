---
title: urbanspeech Dagspace
category: concept
created: 2026-06-16
updated: 2026-06-16
tags: [dagspace, asr, speech, vad, granite-speech, vllm, hallucination]
---

# urbanspeech Dagspace

Speech recognition over video clips: isolate each video's audio with ffmpeg,
then transcribe it with **granite-speech** via vLLM on the JU partition
(RTX A5000). One of the six [[concept-dagspaces|dagspaces]]; follows the same
two-tier SLURM orchestration pattern as `urbanvqa` / `urbanembed`.

## Pipeline

```
videos → [extract_audio] → audio_manifest.parquet → [asr] → transcripts.parquet
         (CPU launcher)                              (GPU launcher, vLLM)
```

| Stage | File | Launcher | Output |
|-------|------|----------|--------|
| `extract_audio` | `stages/extract_audio.py` | `slurm_cpu_ju` | `audio_manifest.parquet` (one row/video) |
| `asr` | `stages/asr.py` | `slurm_gpu_ju_1x` | `transcripts.parquet` |

Pipelines: `conf/pipeline/asr_videos.yaml` (single model),
`conf/pipeline/asr_videos_granite_ablation.yaml` (2b vs 8b side by side).
Models: `granite_speech_3_3_2b` (default), `_3_3_8b` (TP=2), `_4_1_2b`
(newer backbone, no speech LoRA).

### extract_audio
- ffmpeg decodes each video's audio to **16 kHz mono 16-bit PCM WAV** (the
  format granite-speech expects). CPU-only, thread-pooled. `skip_existing`
  reuses WAVs from a prior partial run.
- Runs a **VAD pass** (see below) and writes speech timestamps into the manifest.
- Manifest columns: `sample_id, video_path, audio_path, has_audio,
  audio_duration_s, sample_rate, extract_error, speech_segments,
  n_speech_segments, speech_duration_s`.

### asr
- Reads the manifest, loads each WAV with the stdlib `wave` module
  (`stages/audio_io.py`), and transcribes **speech windows only**.
- granite-speech specifics: the speech encoder is a LoRA adapter shipped inside
  the model repo, so the engine needs `enable_lora=True` / `max_lora_rank=64`
  and every request carries `LoRARequest("speech", 1, <model dir>)`. The user
  turn embeds a literal `<|audio|>` placeholder; the waveform goes in
  `multi_modal_data={"audio": (array, sr)}`.
- Output columns: `transcript` (joined), `chunk_transcripts` (per-window),
  `chunk_flags`, `n_chunks`, `n_chunks_filtered`, `asr_error`,
  `asr_model_source`.

## Hallucination control (the hard problem)

granite-speech — like every attention-based ASR model — **hallucinates
YouTube-outro filler** ("Thank you for watching", "Thank you very much for your
attention") over **non-speech audio**. Urban clips are dominated by ambient
traffic/wind, which is non-speech but *not silent*, so naive mitigations fail.

**What does NOT work (and why):**

| Mitigation | Why it fails |
|------------|--------------|
| RMS silence gate (`asr.silence_rms`) | Gates loud vs quiet, not speech vs noise. Ambient traffic sails over the threshold. On a 47-clip sample it dropped 1/1642 chunks. |
| `repetition_penalty` / `frequency_penalty` | Operate *within* one `generate()` call. Each chunk emits a short, internally-non-repetitive phrase; the repetition is *across* chunks, invisible to per-request penalties. High `frequency_penalty` also corrupts legitimately repeated words. |

**What works — VAD-gated transcription** (mirrors faster-whisper / WhisperX):

1. **Silero VAD** (`stages/vad.py`) runs in the CPU `extract_audio` stage and
   detects speech intervals per video → persisted as `speech_segments`.
2. The `asr` stage packs those segments into ≤`chunk_seconds` windows
   (`pack_segments`) and transcribes **only speech**. Non-speech audio never
   reaches the model, so the filler is never generated. Clips with zero detected
   speech yield `transcript=None` / `asr_error="no_speech"`.
3. **Post-filter backstop** (`asr._make_hallucination_flagger`): a removal-based
   phrase matcher flags residual filler (music, crowd noise) and collapses
   cross-chunk duplicates. Flagged chunks are excluded from the joined
   `transcript` but retained in `chunk_transcripts` + `chunk_flags` for audit.

**Measured impact (47-clip sample):** total chunks 1642 → 910 (~45% fewer);
4/47 noise-only clips skipped entirely; a 29-min recycling clip went 59 → 3
windows (0% speech). Interview clips (95–97% speech) keep essentially all audio.

### Key config (`conf/config.yaml`)

```yaml
audio_extraction.vad:
  enabled: true
  threshold: 0.5               # speech-probability cutoff
  min_speech_duration_ms: 250
  min_silence_duration_ms: 300
  speech_pad_ms: 200
  max_speech_duration_s: 30.0
asr:
  use_vad_segments: true       # consume manifest speech_segments
  vad_join_gap_seconds: 1.0    # merge segments separated by <= this gap
  repetition_penalty: 1.2
  frequency_penalty: 0.0       # 0 once VAD does the heavy lifting
  silence_rms: 0.005           # cheap secondary gate
  hallucination_filter:
    enabled: true
    phrases: null              # null = built-in filler list
    min_coverage: 0.6
    collapse_consecutive_duplicates: true
```

Disabling `audio_extraction.vad.enabled` (or `asr.use_vad_segments`) makes the
asr stage fall back to whole-file chunking, so **old manifests still run**.

## Files

| File | Role |
|------|------|
| `stages/extract_audio.py` | ffmpeg extraction + VAD pass |
| `stages/vad.py` | Silero VAD wrapper, `speech_segments()`, `pack_segments()` |
| `stages/audio_io.py` | shared stdlib-`wave` WAV reader |
| `stages/asr.py` | segment-driven chunking, vLLM inference, hallucination post-filter |
| `orchestrator.py` | `ExtractAudioRunner`, `AsrRunner`, stage registry |
| `conf/` | config, data, model, pipeline groups |
| `tests/test_speech.py` | VAD packing + filter unit tests (GPU-free) |

## See also
- [[concept-dagspaces]] — the shared dagspace architecture
