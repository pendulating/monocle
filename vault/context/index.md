# Wiki Index

Content catalog for the MLLMSCI project wiki. Organized by category. Updated on every ingest.

## Overview

- [[project-overview]] — What MLLMSCI/UAIR is, tech stack, five dagspaces, design principles, data domains, models
- [[architecture]] — System design: DAG execution, stage runners, data flow, config composition, SLURM integration

## Dagspaces

- [[urban-vqa]] — Core VQA pipeline: prompts, guided decoding, verification, prompt optimization (GEPA)
- [[urban-ocr]] — Text spotting: pluggable data handlers, automatic tiling, flat detection output
- [[urban-pair-vqa]] — Pairwise comparison: ordinal labels, counterbalancing, pair generation, diagnostics, always-on "Not sure" abstention, the consolidated 7-case ranking battery (subway/libraries/schools/road-quality/parks under the public-investment umbrella, plus restaurants + street photography) and its minimal prompt contract, `deprecated/` config sub-groups, `prompt.image_layout` presentation variants, Reviewer-2 robustness battery
- [[urban-roam-vqa]] — Agent navigation: board-first street graph, legal-move walk simulation, absolute-frame faces, checkpointing
- [[urban-embed]] — Image embedding: Ray actor pool, Qwen3-VL-Embedding, pooling strategies
- [[urban-speech]] — Speech recognition over video: ffmpeg audio isolation + granite-speech 3.3 (2b/8b) or 4.1-2b via vLLM on JU partition A5000s
- [[urban-vit]] — ViT training + inference: DINOv3/TIPSv2 with LoRA, multi-head masked-loss classifiers, WebDataset shards, A6000/klara
- [[artifact-gen]] — Geospatial raster generation: text query → IDW-interpolated GeoTIFF from embeddings

## Infrastructure

- [[shared-infrastructure]] — dagspaces/common/ module inventory: config_schema, orchestrator, stage_utils, logging
- [[cyclomedia-catalog]] — Polars + polars-st spatial/temporal catalog over /share/ju/cyclomedia/raw (31.5M rows, 5 boroughs, QC-cleared; rejoin-wfs fast path)
- [[scaffolding-permits-curation]] — DOB NOW + BIS scaffold/shed permits → 80-ft buffered coverage + Cyclomedia rows near permits. Two sub-datasets: `through_2025` (323k permits, 6.3M images) and `2020_through_2025` (57k permits, new)
- [[facdb-curation]] — NYC DCP Facilities Database (`ji82-xba5`) — filter at any of 4 hierarchy levels (facdomain > facgroup > facsubgrp > factype), 80-ft buffered coverage, shared `geom.py` with permits
- [[dohmh-restaurants-curation]] — DOHMH Restaurant Inspection Results (`43nn-pn8j`) as a proxy for ALL NYC restaurants. Two-step: build → inspection-level rows; opt-in `aggregate-restaurants` → one row per CAMIS
- [[subway-entrances-curation]] — MTA Subway Entrances/Exits (`i9wp-a4ja`, data.ny.gov) — points-only geometry: skips BIN match, buffers entrance lat/lon directly. Filter by entrance_type / division / borough / route
- [[open-restaurants-curation]] — DCWP Open Restaurants / Dining Out NYC outdoor-dining licenses (`fpeh-f7ci`) — 1.3k restaurants licensed for Sidewalk/Roadway dining, BIN building polygon + 80-ft buffer, shared `geom.py`. Filter by license_type / borough; synthesizes a unique `uid` (no native key)
- [[vllm-inference]] — run_vllm_inference() deep dive: multimodal, LoRA, reasoning parsers, guided decoding
- [[config-system]] — Hydra composition: config groups, searchpath, overrides, env interpolation, CLI usage
- [[slurm-deployment]] — SLURM integration: launcher configs, server.env, submitit, job lifecycle

## Guides

- [[guide-bootstrapping]] — Newcomer setup: install, env config, first local run, first SLURM run, testing
- [[guide-custom-stages]] — Extending the framework: StageRunner subclass, registration, pipeline YAML
- [[guide-browser-search]] — Pure client-side image search: PCA index, learned projection, ONNX text encoder
- [[guide-validation-pipeline]] — Rigorous evaluation for city stakeholders: annotation, stratified testing, calibration, equity
- [[guide-compliance-map]] — Scaffolding permit compliance map: DoB cross-reference, spatial join, folium interactive map
- [[guide-neighborhood-aggregation]] — NTA-level aggregation of pairwise rankings: `notebooks/css/neighborhoods.py`, join pairs.parquet, point-in-polygon (EPSG:2263), dual unit-first + zone-first TrueSkill
- [[guide-pairwise-difference-testing]] — On-the-fly group t-tests over urbanpairvqa runs (`scripts/pairwise_vqa_difference_report.py`): head-to-head + rating-level tests, all-pairs matrix, multi-model replication via `--aggregation-dir`, JSONL experiment registry, W&B mirror to `URBANPAIRVQA-ANALYSIS`
- [[guide-pairwise-regression-testing]] — Covariate regressions over urbanpairvqa ratings (`scripts/pairwise_vqa_regression_report.py`): unit-level OLS/WLS + pair-level Δx validation, controls/partial R², screen mode, school covariates builder (`scripts/build_school_covariates.py`, DOE + PLUTO joins); shares registry/W&B via `scripts/pairwise_analysis_common.py`
- [[guide-cyclomedia-browser]] — Interactive marimo browser over the Cyclomedia imagery (`notebooks/cyclomedia/browser.py` + `dagspaces/common/cyclomedia/`): DuckDB-backed citywide density map → click → nearest recordings → cube faces, equirectangular panorama, depth maps. Recording-level index (5.24M rows) cached from the catalog

## Concepts

- [[concept-tiling]] — How UrbanOCR splits large images and remaps coordinates
- [[concept-counterbalancing]] — How UrbanPairVQA handles presentation order bias
- [[concept-guided-decoding]] — vLLM structured output via JSON schema / enum constraints
- [[concept-verification]] — Post-inference answer filtering via embeddings, NLI, and LLM judges
- [[concept-street-graph]] — Board-first street graph (OSM board + imagery attachment, QA-gated connectivity), face frames, legal moves
- [[concept-embedding-thresholds]] — Background Gaussian fit + σ thresholds for pseudo-labeling from embedding similarity
- [[concept-chunked-dp-worker]] — Why `llm.chat(big_list)` stalls + OOMs on multimodal batches, and the chunked DP-full worker pattern that fixes it
- [[concept-trueskill]] — Bayesian rating aggregation of pairwise VQA outputs: ordinal-score → `rate_1vs1` recipe, `mu - 3*sigma` for ranking, gotchas
- [[concept-facing-filter]] — Per-unit facing pipeline (A–E): attribution dedup, ray-vs-own-polygon, 22.5° bearing cone, 200-ft distance cap, quadratic `attribution_confidence` score feeding the weighted pair sampler
- [[concept-self-explanation-ladder]] — GEPA self-distillation on the pairwise judges: escalating channels (labels → captions → contrastive captions → the image pairs themselves) recover only visual proxies, never the evaluative concept; the `[val-eval]` scoring trap
- [[concept-cyclomedia-depth-maps]] — Depth PNG encoding (`code = R*256 + G`, `0` = no return) + the metric calibration, **resolved 2026-07-29**: `range_m = (code - 16384) / 250`, Euclidean range along the pixel ray, 4 mm quantum, 0–196.6 m. Scale from collinear camera baselines (249.86 ± 0.17 codes/m, flat 19→52 m, which is what ruled out log/inverse); zero-point from cross-projection consistency. Trap: `groundLevelOffset` is *not* the rendered camera height and anchoring on it is 24 σ off
- [[concept-activation-probe]] — HF hidden-state extraction from `gemma4_unified` (48×3840, 0.33 s/pair); the Gemma massive-activation trap (raw cosine ≈1.0 is an artifact — always center); preliminary result: the judgment *is* represented, just not sayable
- [[concept-monocle]] — Per-patch logit-lens word clouds over cyclomedia images (`monocle/`): encoder-free gemma-4 48-px patches → next-token top-k ÷ image-global distribution → PNG/SVG overlay; verified 16×16 row-major geometry; the substring-token-matching trap
- [[concept-jlens-monocle]] — Depth-resolved patch readout via the Jacobian lens (`sub/jacobian-lens`, fitted for gemma-4-12b, 8 layers): per-layer overlays + depth GIF + HTML scrubber; text-reading ignites at L36 while color localizes at L12; the hidden_states double-norm and force_bos double-BOS traps

## Reference

- [[cli-reference]] — Commands, overrides, env vars, models table, launchers table, stage types
- [[file-map]] — Annotated project file tree with every directory and key file

## Troubleshooting

- [[troubleshooting]] — Batch size collapse, object store growth, job hanging, common errors
