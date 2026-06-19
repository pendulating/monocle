---
title: CLI & Quick Reference
category: reference
created: 2026-04-06
updated: 2026-06-16
tags: [reference, cli, commands, cheatsheet]
---

# CLI & Quick Reference

Quick lookup for commands, file locations, environment variables, and common patterns.

## Dagspace CLI Entry Points

| Dagspace | Command | Purpose |
|----------|---------|---------|
| [[urban-vqa]] | `python -m dagspaces.urbanvqa.cli` | Visual Question Answering |
| [[urban-ocr]] | `python -m dagspaces.urbanocr.cli` | OCR / text spotting |
| [[urban-pair-vqa]] | `python -m dagspaces.urbanpairvqa.cli` | Pairwise comparison VQA |
| [[urban-roam-vqa]] | `python -m dagspaces.urbanroamvqa.cli` | Multi-step street traversal |
| [[urban-embed]] | `python -m dagspaces.urbanembed.cli` | Embedding inference |
| [[urban-speech]] | `python -m dagspaces.urbanspeech.cli` | Speech recognition over video |

All accept Hydra overrides appended to the command.

## Common Commands

```bash
# Install
uv pip install -e .

# Local debug run (100 samples)
python -m dagspaces.urbanvqa.cli pipeline=vqa_cyclomedia_scaffolding runtime.debug=true runtime.sample_n=100

# SLURM GPU run (4 GPUs)
python -m dagspaces.urbanvqa.cli hydra/launcher=slurm_gpu_4x pipeline=vqa_cyclomedia_scaffolding

# Multirun sweep
python -m dagspaces.urbanvqa.cli -m data=bayflood_1k,bayflood_nearby model=vllm_multimodal_qwen3_vl_2b,vllm_multimodal_qwen3_vl_30b

# Tests
pytest tests/test_vqa.py -v

# Type check
pyright
```

## Common Hydra Overrides

| Override | Effect |
|----------|--------|
| `runtime.debug=true` | Enable debug logging |
| `runtime.sample_n=100` | Limit to N samples |
| `model.batch_size=32` | Change inference batch size |
| `data.parquet_path=/path/to/data.parquet` | Override input data |
| `hydra/launcher=slurm_gpu_4x` | Submit to SLURM with 4 GPUs |
| `hydra/launcher=ray_local` | Run locally with Ray |
| `sampling_params.temperature=0.0` | Greedy decoding |
| `sampling_params.max_tokens=512` | Limit output length |
| `wandb.mode=offline` | Disable W&B upload |
| `+hydra.job.name=my_experiment` | Custom job name |

## Key File Locations

| Path | Purpose |
|------|---------|
| `dagspaces/` | All pipeline implementations |
| `dagspaces/common/` | Shared infrastructure |
| `dagspaces/common/conf/model/` | Model configs (symlinked as `models/`) |
| `dagspaces/common/conf/hydra/launcher/` | Launcher configs (symlinked as `launchers/`) |
| `server.env` | Site-specific SLURM/env settings |
| `pipeline_manifest.json` | Execution results tracker |
| `PIPELINE_ISSUES.md` | Known performance issues |
| `tests/test_vqa.py` | VQA unit tests |
| `tests/test_roaming_vqa.py` | Roaming VQA tests |
| `scripts/` | Dataset creation & model download utilities |
| `notebooks/` | Analysis notebooks |
| `docs/` | User guide, config guide, custom stages guide |
| `documentation/` | Implementation reports & verification |

## Environment Variables

| Variable | Source | Purpose |
|----------|--------|---------|
| `SLURM_PARTITION` | `server.env` | SLURM partition name |
| `VENV` | `server.env` | Path to Python venv |
| `PROJECT_ROOT` | `server.env` | Project root path |
| `NCCL_P2P_DISABLE=1` | `server.env` | Disable P2P for PCIe machines |
| `NCCL_IB_DISABLE=1` | `server.env` | Disable InfiniBand |
| `NCCL_SHM_DISABLE=1` | `server.env` | Disable shared memory |
| `VLLM_LOGGING_LEVEL` | runtime | Control vLLM log verbosity |
| `WANDB_MODE` | runtime | online/offline/disabled |
| `WANDB_DISABLE_SERVICE=true` | hardcoded | Disable W&B service daemon |

## Available Models

| Config Name | Model | Size | Notes |
|-------------|-------|------|-------|
| `vllm_multimodal_qwen3_vl_2b` | Qwen3-VL-2B | ~4GB | Fits on single A6000 |
| `vllm_multimodal_qwen3_vl_2b_awq` | Qwen3-VL-2B AWQ | Smaller | Quantized |
| `vllm_multimodal_qwen3_vl_30b` | Qwen3-VL-30B | Large | Needs TP≥2 |
| `vllm_multimodal_qwen2.5_vl_3b` | Qwen2.5-VL-3B | ~6GB | |
| `vllm_multimodal_qwen2.5_vl_7b` | Qwen2.5-VL-7B | ~14GB | |
| `vllm_multimodal_cambrian1_13b` | Cambrian-13B | ~26GB | |
| `vllm_multimodal_internvl` | InternVL | Varies | |
| `vllm_multimodal_phi3` | Phi-3 Vision | ~8GB | |
| `vllm_multimodal_smolvlm` | SmolVLM | Small | |
| `qwen3_vl_embedding_2b` | Qwen3-VL-Embed-2B | ~4GB | Embedding only |
| `qwen3_vl_embedding_8b` | Qwen3-VL-Embed-8B | ~16GB | Embedding only |
| `granite_speech_3_3_2b` | granite-speech-3.3-2b | ~5GB | ASR (urbanspeech-local) |
| `granite_speech_3_3_8b` | granite-speech-3.3-8b | ~17GB | ASR, TP=2 on A5000 (urbanspeech-local) |
| `granite_speech_4_1_2b` | granite-speech-4.1-2b | ~4.6GB | ASR, no LoRA (urbanspeech-local) |

### Group-style configs (pairwise/VQA model sweeps)

Newer configs are **groups** under `dagspaces/common/conf/model/<group>/` selected as `model=<group>/<variant>` (e.g. `model=gemma-4-12b/instruct`). Used by the schools/restaurants pairwise sweeps.

| Config | Model | Size (bf16) | TP | Notes |
|--------|-------|------|----|-------|
| `gemma-4-e2b/instruct` | Gemma-4-E2B-IT | ~5GB | 1 | weakest rater (≈0% Much* labels) |
| `gemma-4-e4b/instruct` | Gemma-4-E4B-IT | ~8GB | 1 | |
| `gemma-4-12b/{instruct,base}` | Gemma-4-12B(-IT) | ~24GB | 1 | dense; fits one 48GB A6000; added for the capability-ladder schools sweep |
| `gemma-4-31b/{instruct,base}` | Gemma-4-31B(-IT) | ~62GB | 2 | |
| `qwen3.5-{2b,4b,9b}/instruct` | Qwen3.5-{2B,4B,9B} | ~4/8/18GB | 1 | 9b/4b set `limit_mm_per_prompt.image=2` for pairwise |
| `phi-4/multimodal-instruct` | Phi-4-multimodal | ~11GB | 1 | degenerate rater (~83% "Same") |

## SLURM Launchers

| Config | GPUs | Typical Use |
|--------|------|-------------|
| `ray_local` | Local | Development/debugging |
| `slurm_cpu` | 0 | CPU-only processing |
| `slurm_cpu_beefy` | 0 | Heavy CPU workloads |
| `slurm_gpu_1x` | 1 | Small models (2B-3B) |
| `slurm_gpu_2x` | 2 | Medium models (7B) |
| `slurm_gpu_3x` | 3 | Large models |
| `slurm_gpu_4x` | 4 | Standard production (30B) |
| `slurm_gpu_6x` | 6 | Very large models |
| `slurm_gpu_klara_1x/2x/4x` | 1/2/4 | klara node (A6000) via pierson partition |
| `slurm_cpu_ju` | 0 | JU partition CPU stages (audio extraction) |
| `slurm_gpu_ju_1x/2x/4x` | 1/2/4 | JU partition (RTX A5000, 24GB) |
| `slurm_monitor` | — | Monitoring jobs |

## Stage Types

| Stage Name | Dagspace | Runner Class | Purpose |
|------------|----------|-------------|---------|
| `vqa` | urbanvqa | `VQARunner` | Multimodal VQA |
| `ocr` | urbanocr | `OCRRunner` | Text spotting |
| `pairwise_vqa` | urbanpairvqa | `PairwiseVQARunner` | Image pair comparison |
| `roaming_vqa` | urbanroamvqa | `RoamingVQARunner` | Agent navigation |
| `embed` | urbanembed | `EmbedRunner` | Image embedding |
| `extract_audio` | urbanspeech | `ExtractAudioRunner` | ffmpeg audio isolation from video |
| `asr` | urbanspeech | `AsrRunner` | granite-speech transcription |

See [[guide-custom-stages]] for how to add new stage types.

## See Also

- [[guide-bootstrapping]] — Full setup walkthrough
- [[config-system]] — Detailed config documentation
- [[troubleshooting]] — When things go wrong
