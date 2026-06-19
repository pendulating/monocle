---
title: "UrbanSpeech — Speech Recognition over Video"
category: dagspace
created: 2026-06-09
updated: 2026-06-16
tags:
  - dagspace
  - asr
  - speech
  - audio
  - granite-speech
  - vllm
  - ju-partition
---

# UrbanSpeech — Speech Recognition over Video

UrbanSpeech is the dagspace for **automatic speech recognition (ASR) over video clips**. A CPU preprocessing stage isolates each video's audio track with ffmpeg (so the GPU stage never decodes video containers), then a vLLM stage transcribes the audio with IBM **granite-speech** — **3.3** (2b default, 8b ablation) or **4.1-2b** (`model=granite_speech_4_1_2b`). Runs on the **JU partition** (RTX A5000, 24 GB, sm_86).

## Purpose

- Batch transcription of speech in urban video datasets (dashcam, street-level clips)
- Audio-channel isolation as preprocessing for faster GPU throughput
- 2b vs 8b model ablation on the same audio manifest

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanspeech/cli.py` | Hydra CLI entry point |
| `dagspaces/urbanspeech/orchestrator.py` | DAG engine; defines `ExtractAudioRunner` and `AsrRunner` |
| `dagspaces/urbanspeech/stages/extract_audio.py` | ffmpeg audio isolation → manifest parquet |
| `dagspaces/urbanspeech/stages/asr.py` | granite-speech transcription via vLLM |
| `dagspaces/urbanspeech/conf/model/granite_speech_3_3_{2b,8b}.yaml` | granite-speech-3.3 model configs (LoRA speech adapter) |
| `dagspaces/urbanspeech/conf/model/granite_speech_4_1_2b.yaml` | granite-speech-4.1-2b config (no LoRA; `speech_lora: false`) |
| `dagspaces/urbanspeech/conf/pipeline/asr_videos.yaml` | extract_audio → asr (2b) |
| `dagspaces/urbanspeech/conf/pipeline/asr_videos_granite_ablation.yaml` | extract once, transcribe with 2b AND 8b |
| `dagspaces/urbanspeech/conf/data/generic_videos.yaml` | Video inputs (parquet or directory glob) |

## Pipeline Stages

### 1. `extract_audio` (CPU, launcher `slurm_cpu_ju`)

- Lists videos from `data.parquet_path` (carries metadata columns through) or `data.video_dir` + `data.video_glob` (default `**/*`, recursive; the extension allow-list `{.mp4,.mov,.mkv,.avi,.webm,.ts,.m4v,.insv}` decides what counts as video, so non-video siblings are dropped). `.insv` is Insta360's raw container — ffmpeg reads its h264+aac streams directly
- `data.video_exclude` blacklists clips whose path contains any listed substring (case-insensitive, matched on the path relative to `video_dir`) — e.g. `'data.video_exclude=[Interview]'` skips a subfolder; applies to both the parquet and directory-scan inputs
- ffprobe detects whether an audio stream exists; ffmpeg decodes to **16 kHz mono 16-bit PCM WAV** (granite-speech's expected input)
- Parallel thread pool (`audio_extraction.num_workers`, default 8); `skip_existing: true` makes re-runs resumable
- Output: `audio_manifest.parquet` — one row per video with `sample_id, video_path, audio_path, has_audio, audio_duration_s, extract_error`

### 2. `asr` (GPU, launcher `slurm_gpu_ju_1x` / `_2x`)

- Reads the manifest, splits each WAV into `asr.chunk_seconds` windows (default 30 s; model default `max_model_len` is 2048)
- WAVs read with stdlib `wave` (extraction guarantees 16-bit PCM — no librosa/soundfile dependency)
- One vLLM request per chunk: prompt is the granite chat template around `<|audio|>{asr.question}`, waveform passed via `multi_modal_data={"audio": (array, sr)}`
- Greedy decoding (`asr.temperature: 0.0`, `asr.max_tokens: 256` per chunk) with anti-repetition penalties (see below)
- **Silence gate**: chunks whose RMS energy is below `asr.silence_rms` (default 0.005) are dropped before inference, not transcribed — stops the model hallucinating filler over quiet stretches (logged as "N silent chunks skipped"). `silence_rms: 0` disables it
- Chunk transcripts re-joined per video; output parquet adds `transcript`, `chunk_transcripts`, `n_chunks`, `asr_error`, `asr_model_source`
- `runtime.streaming_io: true` flushes chunk-level parquet parts to `<output>.parts/` for crash recovery

## granite-speech Specifics

| Aspect | Detail |
|--------|--------|
| Architecture | `GraniteSpeechForConditionalGeneration` (`model_type: granite_speech`, supported by vLLM ≤0.19) |
| Speech adapter (3.3) | A **LoRA adapter inside the model repo** wires in the audio encoder — engine needs `enable_lora: true`, `max_lora_rank: 64`, and each audio request carries `LoRARequest("speech", 1, <model dir>)` (`model.speech_lora: true`) |
| Speech adapter (4.1-2b) | **No LoRA.** `config.json` has `"has_lora_adapter": false`; encoder/projector/LLM weights are all in the indexed safetensors, so `speech_lora: false` and `engine_kwargs` omits `enable_lora`/`max_lora_rank`. The repo's `out_llm.safetensors` is an auxiliary CTC head for the HF processor — **not** in the weight index, so vLLM ignores it |
| Audio input | 16 kHz mono float32 array, `limit_mm_per_prompt: {audio: 1}` |
| Checkpoints | 3.3: `/share/pierson/matt/zoo/models/granite-speech-3.3-{2b,8b}`; 4.1: `…/granite-speech-4.1-2b` |
| Sizing | 3.3-2b / 4.1-2b fit one A5000 (TP=1; 4.1-2b bf16 ≈4.6 GB); 3.3-8b bf16 ≈17 GB → **TP=2** on 24 GB A5000s |
| Prompt note (4.1-2b) | Default prompt yields raw lowercase text; pass `asr.question="transcribe the speech with proper punctuation and capitalization."` for punctuated/cased output. `max_model_len` can be 4096 (granite-4.0-1b-base backbone) |

> **4.1 -plus / -nar are not wired up.** The `granite-speech-4.1-2b-plus` (speaker-attributed ASR + word-level timestamps, no punctuation/casing) and `-nar` (non-autoregressive CTC editing) variants use distinct architectures — `GraniteSpeechPlusForConditionalGeneration` (`granite_speech_plus`) and `GraniteSpeechNarForASR` (`granite_speech_nar`) — that vLLM 0.19 does **not** support (only `GraniteSpeechForConditionalGeneration` is registered). Running them would require a transformers-based inference path (or a newer vLLM); deferred as of 2026-06-16.

## JU Partition Launchers

Shared launchers in `dagspaces/common/conf/hydra/launcher/` (see [[slurm-deployment]]):

| Launcher | Resources | Use |
|----------|-----------|-----|
| `slurm_cpu_ju` | 4 CPUs, 32 GB | extract_audio |
| `slurm_gpu_ju_1x` | 1× A5000, 4 CPUs | granite-speech-2b (TP=1) |
| `slurm_gpu_ju_2x` | 2× A5000, 8 CPUs | granite-speech-8b (TP=2) |
| `slurm_gpu_ju_4x` | 4× A5000, 16 CPUs | larger TP jobs |

JU notes: partition `ju`, node `ju-compute-01` (4× RTX A5000, 32 CPUs, ~251 GB). A5000 is sm_86 PCIe-only like the A6000s, so the same NCCL settings apply (`TORCH_CUDA_ARCH_LIST=8.6`, `NCCL_P2P_DISABLE=1`, …). CPU requests are deliberately small (4/GPU) because other groups' CPU jobs share the node. `runtime.job_memory_gb` defaults to 64 here (not 256) for the same reason.

## Usage

```bash
# Two-stage pipeline, 2b model
python -m dagspaces.urbanspeech.cli -m pipeline=asr_videos \
  data.video_dir=/path/to/clips

# 2b vs 8b ablation (extract once, transcribe twice)
python -m dagspaces.urbanspeech.cli -m pipeline=asr_videos_granite_ablation \
  data.video_dir=/path/to/clips

# granite-speech-4.1-2b instead of 3.3-2b (CLI model= override wins over the
# pipeline default); add the punctuation prompt for cased/punctuated output
python -m dagspaces.urbanspeech.cli -m pipeline=asr_videos \
  model=granite_speech_4_1_2b data.video_dir=/path/to/clips \
  asr.question="transcribe the speech with proper punctuation and capitalization."

# Parquet input with metadata columns carried through
python -m dagspaces.urbanspeech.cli -m pipeline=asr_videos \
  data.parquet_path=/path/to/videos.parquet runtime.sample_n=100
```

The 8b node in the ablation pipeline swaps the model via node `overrides`
(`model.model_source`, `model.engine_kwargs.tensor_parallel_size: 2`) — no
separate config group selection needed at runtime.

## Repetition / Hallucination Controls

granite-speech is a small autoregressive model and, under plain greedy decoding, fails two ways on real-world street/interview audio:

| Symptom | Example | Lever |
|---------|---------|-------|
| Decode loop on hard audio | `"Literally, this is the set." × 20`, `"I know it's good." × 40` | `asr.repetition_penalty` (default **1.2**; >1 penalizes already-emitted tokens). For severe runs add `asr.frequency_penalty≈0.5` (scales with repeat count) |
| Filler on silence / non-speech | `"Thank you. Thank you. Thank you."` while nobody speaks | `asr.silence_rms` (default **0.005**) drops low-energy chunks before inference |

- All four knobs (`repetition_penalty`, `frequency_penalty`, `presence_penalty`, `silence_rms`) live in the global `asr:` block in `config.yaml` and flow into every pipeline node, so a CLI override like `asr.frequency_penalty=0.5 asr.silence_rms=0.01` works without editing pipeline YAML.
- vLLM 0.19 does **not** support `no_repeat_ngram_size` (`TypeError`), so the penalties above are the only in-engine anti-loop levers — confirmed against `vllm.SamplingParams`.
- The RMS gate is energy-based (numpy, no extra deps): a quiet room sits ≈0.001–0.005 RMS and conversational speech is >0.02. It catches true silence but **not** loud non-speech (traffic, music) — proper VAD (e.g. Silero via `torchaudio`) segmentation is the next step if noise-driven hallucination persists.

## Limitations / Notes

- Videos without an audio stream are kept in the output with `transcript = null` and `asr_error = "no_audio_stream"` — no need to pre-filter
- Chunking is non-overlapping by default; words split at a 30 s boundary can be garbled at chunk seams (`asr.chunk_overlap_seconds` exists but joined text will then duplicate words)
- The ASR stage drives a single vLLM engine per job (no data-parallel worker fan-out); scale by sharding the manifest across jobs if needed
- granite-speech is English-centric (3.3 adds limited multilingual; 4.1 widens it — a smoke test transcribed mixed English/French cleanly); expect degraded quality on other languages

## See Also

- [[architecture]] — DAG execution model shared by all dagspaces
- [[slurm-deployment]] — launcher configs, server.env, submitit
- [[vllm-inference]] — shared engine-kwargs builder used by the asr stage
- [[guide-custom-stages]] — the pattern these stages follow
