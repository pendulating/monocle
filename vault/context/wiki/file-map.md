---
title: Project File Map
category: reference
created: 2026-04-06
updated: 2026-06-09
tags: [reference, files, structure, map]
---

# Project File Map

Complete file tree with purpose annotations. Use this to orient yourself in the codebase.

## Root Level

```
mllmsci/
├── pyproject.toml              # Package manifest (Python 3.12, uv)
├── pyrightconfig.json          # Type checker config
├── constraints.txt             # Python < 3.13
├── .python-version             # 3.12
├── server.env                  # Site-specific SLURM/env vars (DO NOT COMMIT)
├── server.env.example          # Template for server.env
├── pipeline_manifest.json      # Execution results tracker
├── PIPELINE_ISSUES.md          # Known performance bugs
├── CLAUDE.md                   # Claude Code guidance
├── README.md                   # Main project docs
├── models/ → dagspaces/common/conf/model/
├── launchers/ → dagspaces/common/conf/hydra/launcher/
├── datasets/ → dagspaces/common/conf/data/
├── dagspaces/                  # All pipeline implementations
├── scripts/                    # Utilities
├── notebooks/                  # Analysis
├── tests/                      # Unit tests
├── docs/                       # User-facing guides
├── documentation/              # Implementation reports
├── data/                       # Input datasets
├── papers/                     # UAIR research paper (submodule)
├── sub/                        # Cambrian submodule
└── vault/                      # This wiki
```

## dagspaces/ Structure

Each dagspace follows the same pattern:

```
dagspaces/<name>/
├── __init__.py
├── cli.py                      # Entry point (cleans SLURM env vars)
├── orchestrator.py             # Stage registry + execution
├── stages/
│   └── <stage>.py              # Inference implementation
└── conf/
    ├── config.yaml             # Root Hydra config
    ├── data/                   # Data source configs (local overrides; shared in common)
    ├── prompt/                 # Prompt template configs
    ├── model/                  # Model configs (may inherit common)
    └── pipeline/               # DAG definition configs
```

### dagspaces/common/ (Shared Infrastructure)

```
dagspaces/common/
├── config_schema.py            # PipelineGraphSpec, PipelineNodeSpec, ArtifactSpec
├── orchestrator.py             # StageExecutionContext, StageResult, SLURM dispatch
├── vllm_inference.py           # run_vllm_inference() — core GPU inference
├── wandb_logger.py             # WandbConfig, WandbLogger
├── stage_utils.py              # ensure_dotenv(), extract_last_json(), etc.
├── multiprocessing_utils.py    # Ray setup, SLURM CPU detection
├── logging_filters.py          # vLLM log throttling
├── resource_tracker_patch.py   # Multiprocessing workaround
├── runners/
│   └── base.py                 # StageRunner ABC
└── conf/
    ├── data/                   # Shared data configs (16 datasets)
    ├── model/                  # Shared model configs (14+ models)
    └── hydra/launcher/         # Shared SLURM launcher configs (9 launchers)
```

### dagspaces/urbanvqa/ (Extras)

```
dagspaces/urbanvqa/
├── schema_builders.py          # JSON schema helpers for guided decoding
├── verification_core.py        # Answer verification (embed/NLI/combo)
├── prompts/
│   ├── vqa.py, decision_tree.py, techniques.py, unified.py
├── prompt_opt/                 # GEPA prompt optimization
│   ├── runner.py, dataset.py, lm_resolver.py
│   ├── gepa_adapter.py, multimodal_reflection.py, visualize.py
├── stages/
│   ├── vqa.py                  # Core VQA inference
│   ├── persistent_vllm.py      # Persistent model loading
│   └── persistent_cambrian.py  # Cambrian-specific persistent loading
└── gepa_cli.py                 # GEPA CLI entry point
```

### dagspaces/urbanocr/ (Extras)

```
dagspaces/urbanocr/
├── tiling.py                   # Image tiling for large images
├── data_handlers/
│   ├── base.py                 # OCRDataHandler ABC + factory
│   ├── cyclomedia.py           # Cyclomedia data loader
│   └── generic.py              # Generic image loader
└── prompts/
    └── ocr_preprocessing.py    # OCR prompts + output schema
```

### dagspaces/urbanpairvqa/ (Extras)

```
dagspaces/urbanpairvqa/
└── samplers/
    └── cyclomedia_pairs.py     # Pair generation + counterbalancing
```

### dagspaces/urbanroamvqa/ (Extras)

```
dagspaces/urbanroamvqa/
├── graph/
│   ├── street_graph.py         # StreetGraph, Neighbor, face system
│   └── builder.py              # Graph construction (OSMNX, KNN, H3)
└── samplers/
    └── seed_sampler.py         # Walk starting point selection
```

### dagspaces/urbanspeech/

```
dagspaces/urbanspeech/
├── cli.py                      # Hydra entry point
├── orchestrator.py             # ExtractAudioRunner + AsrRunner, DAG engine
├── stages/
│   ├── extract_audio.py        # ffmpeg video → 16 kHz mono WAV + manifest
│   └── asr.py                  # granite-speech transcription via vLLM
└── conf/
    ├── config.yaml             # audio_extraction + asr config blocks
    ├── data/generic_videos.yaml        # parquet or video_dir+glob inputs
    ├── model/granite_speech_3_3_2b.yaml # also _8b (TP=2 on A5000), 4_1_2b (no LoRA)
    └── pipeline/asr_videos.yaml         # also asr_videos_granite_ablation
```

## scripts/

```
scripts/
├── create_cyclomedia_dataset.py    # Build parquet from Cyclomedia images
├── create_nexar_dataset.py         # Build parquet from Nexar dashcam
├── create_bayflood_splits.py       # Dataset splits for BayFlood
├── copy_cyclomedia_to_scratch.py   # Copy images for faster access
├── download_mllm_model.py          # Download models from HuggingFace
├── download_qwen3_july25_*.py      # Qwen3 model downloaders
├── download_cambrian_13b.py        # Cambrian model downloader
├── install_flash_attention.sh      # FlashAttention build script
├── run_qwen30b_vllm.sub           # SLURM job script
├── run_qwen3vl30b_vllm.sub        # SLURM job script
├── condense.py                     # Article metadata condensation
├── optimize_prompts_nli.py         # NLI prompt optimization
└── df_2_ls.py                      # DataFrame utility
```

## tests/

```
tests/
├── test_vqa.py                 # VQA: templates, image prep, JSON extraction
└── test_roaming_vqa.py         # Roaming: graph, bearings, checkpoints
```

## See Also

- [[project-overview]] — What everything is for
- [[architecture]] — How it all connects
- [[shared-infrastructure]] — The common layer in detail
