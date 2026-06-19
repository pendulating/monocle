# Wiki Activity Log

## [2026-06-16] add | urban-speech — granite-speech-4.1-2b model support
- New config `dagspaces/urbanspeech/conf/model/granite_speech_4_1_2b.yaml`. Same vLLM model class as 3.3 (`GraniteSpeechForConditionalGeneration`, supported by vLLM 0.19) so the asr stage needs **no code change** — but 4.1-2b ships **no LoRA** (`config.json` `"has_lora_adapter": false`, no `adapter_*.safetensors`), so `speech_lora: false` and `engine_kwargs` omits `enable_lora`/`max_lora_rank`. The repo's `out_llm.safetensors` is an auxiliary CTC head not in the weight index — vLLM ignores it. `max_model_len: 4096` (granite-4.0-1b-base backbone).
- Downloaded weights to `/share/pierson/matt/zoo/models/granite-speech-4.1-2b` (~4.6 GB bf16). Verified on an A6000: vLLM loads with no LoRA and transcribes the model's `multilingual_sample.wav` through the exact stage path. Default prompt → raw lowercase; `asr.question="transcribe the speech with proper punctuation and capitalization."` → punctuated/cased.
- Run via `model=granite_speech_4_1_2b` (CLI override wins over the `asr_videos` pipeline default — verified by Hydra compose). Updated `asr_videos.yaml` doc comment.
- **Deferred:** the `-plus` (speaker-attributed ASR + word timestamps) and `-nar` variants use distinct archs (`granite_speech_plus` / `granite_speech_nar`) that vLLM 0.19 does **not** register; would need a transformers inference path. Documented in [[urban-speech]].
- Updated [[urban-speech]], [[file-map]], [[cli-reference]], [[index]].

## [2026-06-15] update | urban-speech — video_exclude blacklist, .insv support, glob default
- Added `data.video_exclude` (str or list) to `urbanspeech` extract_audio stage: skips any clip whose path contains a listed substring (case-insensitive, matched on path relative to `video_dir`); applies to both parquet and directory-scan inputs. Default `[]`. Motivated by `/share/ju/robot_norms/data` layout (nested `Oct 1/{Landfill,Recycle,Interview}/` folders with spaces).
- Added `.insv` (Insta360 raw — h264+aac in an mp4-family container) to the video extension allow-list and changed the default `video_glob` from `**/*.mp4` to `**/*` so the allow-list governs discovery (previously a narrow glob silently dropped non-mp4 formats already in the allow-list). Verified ffprobe/ffmpeg extract audio from a real `.insv`. Updated [[urban-speech]] stage notes.
- New intern onboarding guide `docs/urbanspeech_intern_guide.md` (beginner-level, pastable commands) for running ASR over the robot_norms videos.

## [2026-04-29] add | pairwise restaurants — full 6-model sweep mirror
- New artifact: `dagspaces/urbanpairvqa/conf/sweep/restaurants_all_models.yaml` — exact mirror of `libraries_all_models.yaml` (gemma-4-e2b, gemma-4-e4b, phi-4/multimodal-instruct, qwen3.5-{2,4,9}b), targeting `pairwise_restaurants_mvp`. Same `array_parallelism: 2` cap. Coexists with the strict-≤4B variant (`restaurants_all_models_4b.yaml`).
- Verified Hydra compose for each of the 6 model overrides resolves to the expected `/share/pierson/matt/zoo/models/...` path.
- Submit: `python -m dagspaces.urbanpairvqa.cli --multirun +sweep=restaurants_all_models pipeline=pairwise_restaurants_mvp`.

## [2026-04-29] add | pairwise restaurants MVP + ≤4B model sweep
- New artifacts:
  - `dagspaces/common/conf/data/cyclomedia_near_restaurants_facing.yaml` — points at the 490,851-row / 18,488-camis restaurants facing parquet at `curation/dohmh_restaurants_inspected_all/`. Schema and `metadata_columns` mirror the libraries facing config.
  - `dagspaces/urbanpairvqa/conf/prompt/pairwise_restaurant_eat_at_ordinal.yaml` — restaurant-exterior prompt: "Observe the exterior of these two restaurants. Which would you rather eat at?" 5-point ordinal scale + JSON schema, same shape as `pairwise_library_maintained_ordinal`.
  - `dagspaces/urbanpairvqa/conf/pipeline/pairwise_restaurants_mvp.yaml` — mirrors `pairwise_libraries_mvp.yaml`; flips `allow_replacement` to `false` (18,488 unique camis vs 247 libraries → ~170M canonical pairs without replacement). Default `model: qwen3.5-4b/instruct` (largest in the strict ≤4B sweep set), `pair_sampler.weight_column: attribution_confidence`, `counterbalance_mode: balanced`, `max_pairs: 100000`.
  - `dagspaces/urbanpairvqa/conf/sweep/restaurants_all_models_4b.yaml` — Hydra multirun across the natively-multimodal instruction-tuned strict ≤4B lineup: gemma-4-e2b, qwen3.5-2b, qwen3.5-4b. Drops gemma-4-e4b (~4B effective / ~8B total — soft >4B), phi-4-multimodal (~5.6B), qwen3.5-9b vs `libraries_all_models.yaml`.
- Verified Hydra compose for pipeline + each sweep model (4 model overrides resolve to expected `model_source` paths). No code changes — purely configs.
- Submit:
  - `python -m dagspaces.urbanpairvqa.cli -m pipeline=pairwise_restaurants_mvp` (single Gemma-4-E4B run)
  - `python -m dagspaces.urbanpairvqa.cli --multirun +sweep=restaurants_all_models_4b pipeline=pairwise_restaurants_mvp` (4-model sweep, `array_parallelism: 2`)
- Updated wiki: `urban-pair-vqa.md` (new prompt + pipeline + data + sweep tables; backlink to `dohmh-restaurants-curation`).

## [2026-04-23] add | facing filter — Fix F line-of-sight occlusion check
- Motivation: Castle Hill Library (pair_id `unit_00002196` in the last sweep) depicted the adjacent Top Banana Food Market, not the library — camera on the side street, library hidden behind the market. A+C+D+E all pass because the geometry (ray hits unit polygon, face points at centroid, distance acceptable) is correct; the filter had no way to reason about line-of-sight through NYC building footprints.
- Added **Fix F (strict-pierce occlusion)** in `dagspaces/common/curation/filter_facing.py`: after Fix A, for each row with a library BIN, build `LineString(recording → unit_bin.representative_point)` in EPSG:2263 and STRtree-sjoin against `nyc_buildings.parquet`. A candidate blocker passes the "strict pierce" test iff both segment endpoints lie outside its polygon (sjoin already guarantees intersection → entry + exit). Self-matches (candidate BIN == unit BIN) are skipped. Rows with no BIN anchor fall through as pass-through.
- `representative_point` is used (not centroid) so concave / L-shaped / courtyard footprints don't produce a target outside the polygon. Strict pierce (two boundary crossings) preferred over lenient intersect to avoid dropping rows where a corner of a neighbor building clips the segment — typically the image's foreground frame, not an occluder.
- Plumbed through `_load_units_from_parquet` (now also carries `bin`), `FilterFacingResult` (`dropped_by_occlusion`, `occlusion_processable`, `buildings_source`), the facing filter's manifest, `materialize.materialize(facing_occlusion=True, facing_buildings_path=...)`, and the CLI: `filter-facing --no-occlusion --buildings-path`, `materialize-cyclomedia --no-facing-occlusion --facing-buildings-path`.
- Tests: added 5 synthetic cases to `tests/test_curation_filter_facing.py` (occluder between camera/library → drop; off-axis side building → keep; `--no-occlusion` pass-through; no-BIN unit pass-through; manifest field contents). All 11 facing tests pass; adjacent curation suites (37 tests) unchanged.
- Re-materialized `facdb_libraries` (SLURM 829629): 55,164 → **11,721** kept rows (21.2%, was 12,947 pre-F). Fix F dropped 3,265 of the 41,653 post-A rows with a known library BIN (~7.8%). Runtime +35 s per run dominated by loading `nyc_buildings.parquet` (~1.08M polygons). Per-face distribution remains balanced (F=3040, R=3031, B=2851, L=2799). Attribution-confidence mean 0.096 / median 0.081 — higher than pre-F (0.05) because F preferentially drops the lower-confidence side-street occluded rows.
- Castle Hill spot-check: went 67 rows → 63 rows; 4 dropped are all L-face / bearing=270° side-street shots (`W0CPPIZ3_L`, `W0CPPIZ2_L`, `W0EE4HL2_L`, `W0EE4HL1_L`), exactly the Top Banana Food Market occlusion pattern.
- Updated wiki: `concept-facing-filter.md` (Fix F block, updated formula table, empirical table gains 2026-04-23 row, new tuning knobs, new known edge cases); `dagspaces/common/conf/data/cyclomedia_near_libraries_facing.yaml` comment block reflects the new row count + F semantics.

## [2026-04-22] rebalance-2 | facing filter — visibility-fraction angular, cubic distance
- User direction: within the 45° cone, proximity should dominate; the angular penalty should only kick in when the unit is actually being clipped out of the face. The prior `linear(Δθ/45)` penalty demoted close-off-axis rows even when the unit was fully in frame.
- Replaced the angular factor with a physical **visibility fraction**: treat the unit as a disk of characteristic half-width `H_REF = 25 ft`. Its image-plane angular half-width is `atan(H_REF/D)`. Angular term = fraction of the unit's angular span that still falls inside the face's [−45°, +45°] FOV after clipping. Fully-visible rows get `visibility = 1`; partially-cropped rows get the remaining visible fraction.
- Distance term stays cubic. So `conf = visibility_fraction × (1 − D/200)³`.
- Added `DEFAULT_UNIT_HALF_WIDTH_FT = 25.0` constant to `filter_facing.py`; ~15-line change in Fix E block. All 6 facing tests still pass.
- HEISKELL ranking now: top 3 are fully-visible B-face rows at 65.5 / 68.5 / 69.7 ft (conf 0.304 / 0.284 / 0.276); the physically-closest row W0D5JL31_B (64.7 ft, 29° off-axis, ~12% clipped) sits at rank 4 (conf 0.270) — just behind the uncropped rows at similar distances, exactly as requested.
- Confidence distribution on the 12,947 kept rows: mean 0.093 / median 0.077 / max ≈0.56. Higher mean than the linear-angular formula because most rows are fully visible and only pay the distance penalty.
- Regenerated `machine-beholder/audits/libraries_facing_sample.pdf` (250 MB). Updated `concept-facing-filter.md` Fix E block + reference-value table.

## [2026-04-22] rebalance | facing filter — 45° cone + linear angle × cubic distance
- Audit showed close-off-axis rows scoring far too low: e.g. a library row at Δθ=19° / 69 ft scored 0.01 under the 22.5° + all-quadratic formula, even though the library occupies ~40% of the frame at that distance. User direction: "closer distances should be prioritized over better angles... MUCH more important for the subject to be as close as possible to the lens."
- Widened `DEFAULT_BEARING_TOL_DEG` 22.5° → 45° (the face's actual half-FOV). The tight cone was over-protective given that Fix E already penalizes edge-of-frame via centeredness → 0. Anything inside the 90° face now qualifies; the continuous score does the fine-grained prioritization.
- Rebalanced Fix E confidence formula to linear angle × **cubic** distance: `conf = clip(1 − Δθ/45°, 0, 1) × clip(1 − D/200ft, 0, 1)³`. Cubic distance makes close-dominates-far concrete: 50 ft scores 3.4× 100 ft; 80 ft scores 1.7× 100 ft. Angular falloff stays linear — at the cone edge the linear ramp already hits 0, no need to also square.
- Implementation: 3-line change in `dagspaces/common/curation/filter_facing.py` (`DEFAULT_BEARING_TOL_DEG = 45.0`, drop square on angular, change square → cube on distance) plus docstring updates. All 6 facing tests still pass. Re-ran filter-facing in-place (0.4s; no re-materialize needed since the hard gates didn't change) — 12,947 kept rows across **all 236 libraries** (up from 6,376 / 234), conf mean 0.050 / max 0.506.
- HEISKELL spot-check: top 4 rows are now B-face at 68–75 ft with Δθ ≤ 15°, conf ≈ 0.19–0.24. W0E5PQKD_B (the face the user originally flagged) sits at rank 3 with conf 0.22.
- Regenerated `machine-beholder/audits/libraries_facing_sample.pdf` (249 MB). Updated `concept-facing-filter.md` table of iterations (morning → afternoon → evening → night), new reference-value table, new tuning-knob hint.

## [2026-04-22] fix | absolute-frame cube — dropped recorderDirection from bearing computation
- Empirically verified (via visual audit of the HEISKELL library pages + 13,792 W 20th St recordings) that Cyclomedia's NYC cube faces are rendered in a **globally-oriented frame**: F=N, R=E, B=S, L=W. Does NOT rotate with vehicle heading. Orientation column (camera yaw, radians) is within ±0.2° of 0 across 100% of NYC rows, confirming the panorama's F-face is anchored to north.
- Smoking gun in the audit: two HEISKELL recordings with opposing `recorderDirection` (119° ESE vs −59° WNW — 180° apart) both showed the same north-side scaffolded building in their F.jpg files and the same south-side library in their B.jpg files. That's mathematically impossible under a vehicle-relative convention and exactly what an absolute-frame cube predicts.
- Fixed `dagspaces/common/cyclomedia_catalog/indexer.py::_compute_bearing` to drop the `recorder_direction` term; now returns `FACE_BEARING_DEG[face]` directly. Updated `schema.py` comment on `FACE_BEARING_DEG` to reflect absolute-frame semantics. Updated `tests/test_cyclomedia_catalog.py::test_bearing_computation` with the new expected values; all 20 catalog tests still pass.
- Rebuilt the catalog via `rejoin-wfs` fast path (363s for all 5 boroughs, 31.5M rows rewritten). Re-materialized `facdb_libraries` (SLURM 785377, 60s): 55,164 → **6,376** facing rows (up from 5,497 under the buggy bearings) covering **234/236** libraries. Per-face distribution is now balanced (F=1637, R=1672, B=1577, L=1490) vs the clearly skewed mornings that were masking the bug.
- Regenerated `machine-beholder/audits/libraries_facing_sample.pdf` (253 MB, 235 pages). Spot-check: HEISKELL's top 4 rows are now all B-face at Δθ ≤ 11° — including `W0E5PQKD_B`, the face the user originally flagged as the face that visually shows the library.
- Updated wiki pages: `cyclomedia-catalog.md` (formula + caveat box), `concept-facing-filter.md` (empirical-behavior table now tracks three runs; added bug entry to known-edge-cases), `concept-street-graph.md` (face-system block rewritten + scheduled-fix note on roam-VQA resolver), `artifact-gen.md` (ray accumulation pseudocode + scheduled-fix note on `raster.py::_compute_face_bearings`). Downstream code (`filter_facing`, `street_graph::_face_for_bearing`, `raster::_compute_face_bearings`, `roaming_vqa::abs_bearing`) all still compose `yaw + FACE_BEARING_DEG` the same broken way — they now operate on correct bearings only because the catalog `bearing` column is correct, but direct recomputations still need patching.
- Sibling wiki in `/share/pierson/matt/mllmsci/vault/urban-visual-analytics/wiki/cyclomedia-data.md` and `/share/ju/matt/shedfolio/.vault/wiki/cyclomedia-data.md` still states "F is aligned with vehicle heading" — left as-is for now; not in this vault's scope.

## [2026-04-22] tighten | facing filter — 22.5° cone + quadratic confidence
- Cut `DEFAULT_BEARING_TOL_DEG` 45 → 22.5 in `dagspaces/common/curation/filter_facing.py` so the unit must sit in the center 45° of the face's 90° FOV. Audits of the previous 45° output showed low-confidence rows (Δθ near the 45° edge, ≥100 ft) where the library was a sliver at the frame corner, often occluded — the filter said "ray hits polygon" but the image did not depict the library.
- Replaced both linear falloff terms in `attribution_confidence` with quadratic: `angular² × distance²`. Concentrates the weighted sampler on centrally-placed + physically-close rows; distance² is also a proxy for the `angular size ∝ 1/D` prominence effect.
- Routed `materialize.py` + `cli.py` through `filter_facing.DEFAULT_BEARING_TOL_DEG` / `DEFAULT_MAX_DISTANCE_FT` so there's a single source of truth for the defaults.
- Re-materialize on `facdb_libraries` (SLURM job 782475, 89s): 55,164 → **5,497** kept (10.0%, was 23.5% at 45° linear), covering 235 / 236 libraries. Drops: A 11,104, C 38,245, D 318. Confidence mean 0.075 / max 0.573 (squeeze from squaring is expected; relative ordering is what matters downstream).
- Regenerated `machine-beholder/audits/libraries_facing_sample.pdf` (252 MB, 236 pages) via `scripts/facing_audit_report.py` — top-of-page thumbnails are now noticeably more "unit-is-the-focus".
- Confirmed along the way that face-offset bearings are already applied at catalog index time (`cyclomedia_catalog/indexer.py::_compute_bearing` adds `FACE_BEARING_DEG[face]` to `recorderDirection`); the facing filter's `bearing` column is truthfully the face bearing, not the vehicle bearing.

## [2026-04-22] update | weighted pair sampler implemented
- Implemented the `weight_column: Optional[str]` parameter on `dagspaces/urbanpairvqa/samplers/cyclomedia_pairs.py::build_unit_random_pairs` as planned in `concept-facing-filter.md`. Within-unit draws now use `rng.choice(indices, p=weights)`; unit-level pair selection stays uniform. Threaded through `dagspaces/urbanpairvqa/orchestrator.py` and surfaced as `pair_sampler.weight_column` (default `null` in `conf/config.yaml`, `attribution_confidence` in `conf/pipeline/pairwise_libraries_mvp.yaml`).
- Graceful fallbacks: missing column → uniform + one-line warning; per-unit all-zero weights → uniform for that unit only; NaN/negative values clipped to 0 at load. Added `TestWeightedWithinUnit` class to `tests/test_pairwise_unit_sampler.py` covering the weighted-bias, missing-column, per-unit-zero, and clipping paths (all 15 tests pass).
- Wiki flipped "Planned" → implemented in both `urban-pair-vqa.md` and `concept-facing-filter.md`; `sampling` tag added to the concept page.

## [2026-04-22] create | concept-facing-filter — per-unit attribution + confidence score
- Added `vault/context/wiki/concept-facing-filter.md` documenting the five-fix overhaul of the Cyclomedia-vs-coverage facing filter (A: ray must hit the row's own `unit_uid` polygon, B: closest-attribution dedup at materialize time, C: 45° bearing cone, D: 200-ft distance cap, E: continuous `attribution_confidence ∈ [0, 1]`). Implementation landed in `dagspaces/common/curation/filter_facing.py` (A+C+D+E), `dagspaces/common/curation/permits/materialize.py` (B + threading of new knobs), and `dagspaces/common/curation/cli.py` (new `--facing-max-distance-ft` / `--bearing-tol-deg` / `--max-distance-ft` / `--units` flags).
- Smoke-tested on existing `curation/facdb_libraries/cyclomedia_near_libraries.parquet`: 59,080 → 13,632 kept rows (23%), with 11,723 dropped by A (cross-attribution), 32,589 by C (off-axis), 1,136 by D (too-far). Confidence distribution: mean 0.19, max 0.77. Diagnostic columns `bearing_to_unit_deg`, `delta_bearing_deg`, `distance_to_unit_ft`, `attribution_confidence` now populated in per-unit mode.
- Planned next step captured on the same concept page + in `urban-pair-vqa.md`: thread `weight_column: Optional[str]` into `build_unit_random_pairs` for confidence-weighted image draws within each unit. Within-unit weighting only; unit-level selection stays uniform to preserve per-library coverage for TrueSkill.
- Updated `facdb-curation.md` and `scaffolding-permits-curation.md` to replace their old "Orientation filter" sections with pointers to the concept page. Added a See-also link from `cyclomedia-catalog.md`. Added `updated: 2026-04-22` to `urban-pair-vqa.md`. Updated `index.md` concept list.

## [2026-04-22] create | concept-trueskill — aggregating pair outputs into ranked lists
- Added `vault/context/wiki/concept-trueskill.md` documenting how `urbanpairvqa` `relative_score` rows get turned into per-entity ratings via the `trueskill` package. Covers the 5-point ordinal → 3-outcome collapse, the winner-first `rate_1vs1` idiom, `mu - 3*sigma` as the conservative display rating, and gotchas (row-order sensitivity, draw-probability tuning, unit-granularity choice).
- Canonical recipe lifted from `notebooks/css/wealth.ipynb` (the only existing in-tree implementation — library-, PUMA-, and tract-level ratings).
- Added `scripts/pairwise_vqa_report.py` — markdown report generator over a stage-output parquet + companion `pairs.parquet`: ordinal label distribution, position-bias diagnostic, reasoning-trace length histogram, word cloud over captured thinking, full TrueSkill ranking (sorted errorbar scatter + mu/sigma histograms + top/bottom-N tables). First exercised on `multirun/2026-04-21_URBANPAIRVQA/21-31-18` (5000 library pairs, 247 libraries).
- Updated `urban-pair-vqa.md` with a "Downstream Aggregation" section pointing at the new concept page and the utility script. Updated `index.md`.

## [2026-04-21] create | urbanvit — new dagspace for ViT training + inference
- Added `dagspaces/urbanvit/` — sixth dagspace, first one dedicated to pure-ViT (non-VLM) workloads. Orchestrator + six stages (`shard`, `train`, `eval`, `extract`, `classify`, `collate`) following the urbanembed pattern, with `dagspaces/common/orchestrator.py` reused for SLURM submit/wait/recover.
- Two pipelines: `train_scaffolding` (shard → train → eval) and `infer_scaffolding` (shard → extract → classify → collate). Target compute: RTX A6000 on the `klara` partition via three new launchers (`slurm_gpu_klara_{1x,2x,4x}.yaml`).
- **Split decision (captured in `docs/plans/urbanvit-dagspace.md`):** group-aware split on `recording_id` — all 4 faces of a Cyclomedia pano land in the same split. Coarser `group`-level splitting rejected for v1 (imbalanced); H3-based spatial partitioning noted as the future upgrade path.
- **Multi-head design:** heads declared in `conf/heads/*.yaml` as an ordered list with optional `conditional_on` for hierarchical tasks (e.g. `fancy_vs_standard` only trains on rows where `scaffolding_any == 1`). Missing labels are masked out of per-head CE loss, so populating parquet columns later doesn't require retraining from scratch.
- **Storage / compute:** shards on `/scratch/$USER/urbanvit/shards/`, checkpoints + features parquets on shared. LoRA via `peft` (default r=16, α=32 on `attn.qkv`/`attn.proj`); bf16 autocast + `torch.compile(mode="max-autotune")`. WebDataset tar shards (pip-installable), FFCV stub present but raises `NotImplementedError` pending install-complexity evaluation.
- **Labels**: none yet. Plan is to bootstrap via embedding retrieval (top/bottom-scoring images from `urbanembed` → weak labels → parquet columns `label_scaffolding`, `label_fancy`). Train stage aborts with a pointer to the plan if no label columns are populated.
- Models: DINOv3 ViT-B/16 raw `.pth` (loaded via `timm` + strict=False, register tokens + RoPE keys mismatch expected) and TIPSv2 ViT-B/14 HF w/ custom code (loaded via `AutoModel.from_pretrained(trust_remote_code=True)`).
- Added deps to `pyproject.toml`: `peft>=0.14.0`, `timm>=1.0.15`, `webdataset>=0.2.100`. `uv sync` required before first run.
- New wiki page `urban-vit.md`; index updated.

## [2026-04-22] feature | unit-aware materialize + unit-mode pairwise VQA
- **Materialize sjoin target changed**: `materialize-cyclomedia` now sjoins catalog points against the per-unit buffered polygons from `facilities.parquet` / `permits.parquet` (auto-detected in the curation dir) rather than against the dissolved `coverage.geojson` MultiPolygon. Every output row now carries `unit_uid`, `unit_name`, `unit_dist_ft` (0.0 — image is inside the buffer). `sjoin_dataset_chunk` takes a `units_gdf` instead of `coverage_gdf`; internal join is still STRtree-backed via GEOS. CLI gains `--units-path` override; auto-detect works for both FacDB (`uid` + `facname`) and scaffolding (`permit_id` + `address`).
- **Keep-all-matches semantic**: an image that sits inside N overlapping unit buffers produces N rows (one per `(image, unit)` pair). Natural for unit-level sampling. For image-mode sampling the downstream should `.unique(['sample_id'])`.
- **Re-materialized `facdb_libraries`**: 59,080 unfiltered rows (up from 55,164 pre-change; duplication factor 1.077) → 47,557 facing, 247 unique libraries attributed. A notable cluster: Brooklyn Central Library has 4 sibling FacDB uids sharing one BIN, so its 537 images show up under each of the 4 unit_uids.
- **urbanpairvqa `build_unit_random_pairs`** (new): groups by `unit_column` (default `unit_uid`), samples 2 distinct units, draws 1 image per unit. Persists `unit_uid_a`/`unit_name_a`/`unit_uid_b`/`unit_name_b` in pair rows plus the standard metadata-column suffixes. Repeat-observations re-sample within-unit each time. 50k library pairs from 247 units built in 11.4s.
- **`pair_sampler.mode`** config flag: `image` (existing behavior) vs `unit` (new). Default `image` — no backcompat break.
- **Pairs persisted to disk**: orchestrator now writes `<run_dir>/pairs.parquet` + `pairs.meta.json` before calling the inference stage. Rerun-with-different-model + post-hoc audit for free.
- **New library-maintained prompt**: `pairwise_library_maintained_ordinal.yaml` — ordinal MuchLess/Less/Same/More/MuchMore on "which library looks better maintained". Guided decoding via JSON schema.
- **New pipeline** `pairwise_libraries_mvp.yaml`: mode=unit, max_pairs=50k, balanced counterbalance, Qwen3-VL-8B-Thinking on `slurm_gpu_2x`.
- **Tests**: 10 new unit-sampler tests + 1 persist-pairs test. 74/74 curation + pairwise tests pass.

## [2026-04-22] default | materialize-cyclomedia now chains filter-facing by default
- `materialize-cyclomedia` now runs `filter-facing` against the unfiltered parquet immediately after writing it, producing a sibling `<name>_facing.parquet`. Downstream consumers (`sample-images`, training, labeling) should prefer the `_facing` parquet — it drops faces looking away from the coverage mask (~20% of rows for libraries, ~10% for scaffold permits). Unfiltered copy kept on disk as audit.
- CLI: new `--no-facing` flag opts out; new `--facing-ray-length-m N` tunes the ray (default 30 m, matches `artifact_gen/raster.py`).
- Python: `materialize(facing=True, facing_ray_length_m=30.0)` defaults. `MaterializeResult` now carries `facing_output_path` + `facing_rows`; manifest.json records `facing_enabled`, `facing_ray_length_m`, `facing_output_path`, `facing_rows`.
- **First FacDB materialization** (`curation/facdb_libraries`): 55,164 unfiltered → 44,141 facing (80.0% kept in 0.4s). 253 library-polygon coverage, 5 boroughs, 13,803 unique recordings captured. Wiki pages (scaffolding + facdb) updated to note the new default and the two-parquet layout.

## [2026-04-22] build | facdb-curation — second curation family over NYC DCP Facilities Database
- Added `dagspaces/common/curation/facdb/` — fetch/normalize/validate/orchestrate pattern mirroring `permits/`, but for Socrata `ji82-xba5` (FacDB, 34.7k rows). Filter at any of 4 hierarchy levels (`facdomain` → `facgroup` → `facsubgrp` → `factype`); values validated against a **frozen dictionary JSON** baked from `curation/facilities_data_dictionary.xlsx` (Categorization sheet, 616 factype rows across 7 domains / 25 groups / 72 subgroups). Typos fail fast with difflib suggestions. The xlsx stays as a reference; runtime source of truth is `dagspaces/common/curation/facdb/categorization.json`.
- CLI: `python -m dagspaces.common.curation facdb-facilities --out <dir> [--facdomain ...] [--facgroup ...] [--facsubgrp ...] [--factype ...]`. Multi-value flags at any level; intersect across levels.
- **Shared `geom.py`** — extracted `attach_geometry` (3-stage BIN-exact → nearest-within-200ft → point fallback) out of `permits/buffer.py` into `curation/geom.py` with column-name parameters. `permits/buffer.py` is now a thin wrapper preserving the old signature. Both sub-dataset families share the polygon-buffering machinery.
- Validation: 7 fatal checks (ji82-xba5 non-empty, unique uid, supported geom_source, non-null facdomain, geom validity, NYC bbox, coverage union). Polygon-match warn threshold lower than permits' 85% → 75% because FacDB has many park/roadway rows with no BIN.
- **Bad-geocode pre-filter**: FacDB has a small number of rows with `bin='0' AND latitude=longitude=0.0` (e.g. QUEENSBRIDGE and RAVENSWOOD libraries). Orchestrator drops these before `attach_geometry` so they don't poison the NYC-bbox fatal.
- **Dogfooded on libraries** (`--facgroup LIBRARIES`): 255 raw → 253 publishable, **100.00% polygon match** (96.44% BIN-exact + 3.56% nearest-within-200ft), **1.53 km² coverage**, all 7 fatals pass, **34s** end-to-end.
- Tests: 13 passing (categorization load + validate with difflib suggestions + typo raise; SoQL IN-clause quote escaping + multi-value; normalize happy path + empty input; orchestrator end-to-end with monkey-patched fetch + buildings; unknown-filter raises before fetch; duplicate uid blocks with summary still written). Existing 30 curation tests (permits/sample/filter-facing) still pass after the `geom.py` refactor (one monkey-patch target moved — updated in the test).
- Wiki page `facdb-curation.md` added with CLI examples, hierarchy reference, output schema table, and the integration blurb for filter-facing/materialize-cyclomedia/sample-images downstream. Index entry added under Infrastructure.

## [2026-04-22] tool | filter-facing — drop faces not oriented toward coverage
- Added `dagspaces/common/curation/filter_facing.py` + `filter-facing` CLI subcommand. Geometric post-filter complementing the point-in-polygon materialize step: from each row's recording position, cast a forward ray in the face's absolute bearing direction; keep only rows whose ray lands in coverage. Matches the `rays` pattern in `dagspaces/artifact_gen/stages/raster.py` (same 30-m default `ray_length_m`, same bearing convention). U/D faces and null-bearing rows drop unconditionally.
- **Toggle, not default** — produces a sibling `<name>_facing.parquet` next to the unfiltered curated parquet plus a `<name>_filter_facing_manifest.json`. `sample-images` stays coverage-unaware; user points it at whichever parquet they want.
- **Implementation note**: the literal line-vs-polygon `gpd.sjoin(predicate='intersects')` ran for 24+ minutes on 3.97M rows with zero progress before I killed it — a 30-m LineString's bounding box overlaps far more STRtree candidates than a point's. Switched to an approximation: sample N points evenly along the forward ray (N = max(3, ceil(ray_length_m / 10)) auto-scaling) and do a point-in-polygon sjoin. Semantically equivalent for 80-ft-buffered coverage (buffers are much wider than a 10-m sample spacing). End-to-end on the real 2020-2025 parquet: **25 s** (~60× speedup) vs >24 min for line-based.
- Real dogfood on `scaffolding_permits_2020_through_2025`: 3,970,154 → 3,581,915 rows (90.2% kept), 388,219 dropped. Per-recording breakdown: unfiltered 99.4% of recordings had all 4 horizontal faces kept; after filter, 70.5% keep all 4, 19.6% keep 3, 9.8% keep 1-2. The filter is most active where permit coverage is sparse (fewer surrounding buildings); in dense Manhattan most recordings have permitted buildings in multiple directions and all 4 faces survive.
- Tests: 6 passing (synthetic building north of recording; only F-face ray hits; B/L/R/U/D drop; ray-length-too-short drops all; long-ray still correctly drops off-axis thanks to auto-scaling; manifest contents; overwrite guard; missing-column error).
- Wiki page gets a new "Orientation filter (optional)" section above the sampling section documenting the two-step workflow and toggle semantics.

## [2026-04-22] tool | sample-images — export K images from any curated parquet
- Added `dagspaces/common/curation/sample.py` + CLI subcommand `python -m dagspaces.common.curation sample-images`. Works against any parquet with an `image_path` column; Cyclomedia is just one consumer.
- Modes: **copy** (default, safe to tar+relocate, preserves mtime via `shutil.copy2`) and **symlink** (fast, local-only, uses absolute source paths). `--workers N` (default 8) parallelizes the I/O via ThreadPoolExecutor.
- `-k K` samples K rows (capped at `df.height` with a warn). `--seed` makes runs reproducible. `--stratify-by COL` splits K evenly across distinct values of COL (common: `dataset`, `face`). Shortfall in one stratum redistributes deterministically to strata with room.
- Output layout: `<out>/images/<dataset>__<sample_id>.jpg` + `<out>/manifest.parquet` (full provenance + `export_filename` + `export_status`) + `<out>/manifest.json` (counts + timings). The `<dataset>__` prefix disambiguates the ~24k cross-dataset sample_id dupes inherited from catalog warn check #11.
- Safety: refuses non-empty `--out` unless `--force`. Missing source files are counted as `missing` in the manifest (non-fatal, logged). Returns exit code 0 on clean run, 2 on user error (ValueError/FileExistsError), 3 if any files failed to export.
- Dogfooded against `scaffolding_permits_2020_through_2025`: 200 symlinks stratified 40/borough in **4.5s**, 0 missing. Wiki page gets a new "Sampling images for inspection" section with both usage patterns documented.
- Tests: `tests/test_curation_sample.py` covers copy mode, symlink mode, stratified-even-split, k>population cap, missing-source counting, force/no-force, seed reproducibility — 7/7 passing.

## [2026-04-22] build | scaffolding_permits_2020_through_2025 — second sub-dataset with date-range filter
- Added `--since YYYY-MM-DD` CLI flag (lower bound on `issue_date`) to complement the existing `--cutoff`. Threaded through `fetch_dob_now`, `fetch_bis` (server-side year-level prune for BIS, since `issuance_date` is plain text), the orchestrator's client-side date clip (source of truth), and the validation summary header.
- Built `curation/scaffolding_permits_2020_through_2025/` with `--since 2020-01-01 --cutoff 2025-12-31`: **57,492 publishable permits** (DOB NOW: 52,985; BIS: 4,516 — BIS tiny in this window because it was deprecated in favor of DOB NOW around 2020). Polygon match **99.97%** (bin_exact 93.04% + nearest_polygon 6.93%); coverage **116.23 km²** (14.94% of NYC). All 8 fatal validation checks pass. Build time 130s (longer than through_2025 because BIS server-side year filter rejects 99% of its data, which apparently takes Socrata a minute to compute).
- SLURM job (738408, after a first submission hit a docs-bug with `CURATION_ROOT=x sbatch` env-var propagation and ran against the wrong dir) materialized the Cyclomedia sub-dataset in **28.4 s**: 3,970,154 rows (120 MB) across 5 boroughs. Per-dataset: bronx 683k, brooklyn 1.21M, manhattan 1.34M, queens 691k, si 48k. Overall ~63% of the 6.3M through_2025 curated rows — consistent with the 6-year window being much denser in permits per unit area than the 37-year historical set.
- **Fixed SLURM script docstring** to clarify that `CURATION_ROOT=x sbatch ...` does NOT propagate — pass the curation root as a positional arg, or use `--export=CURATION_ROOT=x`. The script's positional-arg handling was always correct; only the docs were misleading.
- Added `test_since_drops_early_permits` to pytest suite (17/17 passing). Updated wiki with a new "Built sub-datasets" table and renamed naming convention to `scaffolding_permits_<since>_through_<cutoff>/` for windowed pulls.

## [2026-04-21] rename | scaffolding_permits_2025 → scaffolding_permits_through_2025
- The old dir name `scaffolding_permits_2025` was misleading — it suggested a "permits from 2025" filter when in fact the `2025` referred to the `issue_date <= 2025-12-31` **cutoff**, and the actual contents span 1989-06-09 → 2025-12-31 (37 years, ~10k permits/year 2018-2025 and ~1-2k/year in the 1990s). Renamed to `curation/scaffolding_permits_through_2025/` to make the cutoff semantics explicit. Future refreshes use the same `through_<year>[Q<quarter>]` convention.
- Updated all references: wiki page (status line + new naming-convention callout), log (this entry), plan doc, notebook (hardcoded `Path(...)`), materialize module docstrings, CLI help text, and SLURM script default `CURATION_ROOT`. Historical `.slurm_jobs/*.err/.out` logs intentionally left as-is (they're immutable records).
- Regenerated `cyclomedia_materialize_manifest.json` + `materialize_progress.json` via a fresh materialize run — the only files that carried absolute paths with the old name. Permits build `manifest.json`, `permits.parquet`, `permits.geojson`, `coverage.geojson` had no embedded paths and didn't need regeneration.

## [2026-04-21] build | scaffolding-permits-curation cyclomedia sub-dataset materialized (6.3M rows, 38s)
- Added `dagspaces/common/curation/permits/materialize.py` + `materialize-cyclomedia` CLI subcommand + `scripts/materialize_scaffolding_cyclomedia.sub`. Per-borough chunking (5 chunks) writes `chunks/<dataset>.parquet` incrementally and atomically updates `materialize_progress.json` after each so operators can tail mid-run.
- **Abandoned `CyclomediaCatalog.query(within=...)` for bulk curation.** First submission hung for 45+ min with zero progress on one borough. Root cause: `polars-st.st.within(literal_multipoly)` has no spatial index — each row decodes `geom_wkb` and does a GEOS containment test against the full 5,388-part coverage MultiPolygon. Also researched: GeoPolars (still prototype, sjoin [issue #27](https://github.com/geopolars/geopolars/issues/27) open since June 2022), DuckDB SPATIAL_JOIN (fast, but not currently a dep). Settled on `gpd.sjoin(predicate='within')` — STRtree-backed via GEOS, already a project dep.
- **`sjoin_dataset_chunk`** builds the points GeoDataFrame from the catalog's `latitude`/`longitude` columns (no WKB decode), runs STRtree-indexed sjoin, and gathers matched row indices back into the Polars frame. Empirical timing per borough: scan 0.4-1.5s + points 0.8-2.3s + **sjoin 1.0-5.0s** + gather 0.02-0.1s + write 0.1-0.3s. Total job: **38 s for 6,296,458 rows**. ~70× faster than even the polars-st cold-run estimate (if it had ever completed).
- Output at `curation/scaffolding_permits_through_2025/cyclomedia_near_permits.parquet` (191 MB). Per-dataset hit rates vs catalog F/B/L/R counts: bronx 35%, brooklyn 42%, manhattan 75%, queens 17%, si 4% (Manhattan dense with scaffolding; SI has few permits).
- Added wiki design decision #11 capturing the polars-st limitation and the sjoin fast path. Revisit when GeoPolars sjoin ships.

## [2026-04-21] improve | scaffolding-permits-curation nearest-building fallback (95.28% → 99.98% match)
- Investigated the 15,282 point-fallback rows from the initial build: **100% had BINs that are legitimately absent from `nyc_buildings.parquet`** (5,997 unique missing BINs, all with valid 6-7 digit borough-prefixed format). The buildings file's `last_edited_date` goes up to 2026-04-10, so staleness isn't the primary driver — BIS permits reference BINs for demolished buildings (retired in DoITT) and DOB NOW references BINs for new construction (not yet published).
- Tested two recovery strategies: BBL (block+lot) fallback recovered 35.49% of failures, nearest-building-polygon-within-200ft recovered **99.66%** of failures at p50 distance 36 ft / p99 130 ft. Nearest-building dominates; BBL is redundant because BBL-matched buildings are always the closest buildings anyway.
- **Added nearest-building fallback as stage 2 of `attach_geometry`** in `dagspaces/common/curation/permits/buffer.py` (sjoin_nearest in EPSG:2263 with a 200-ft cap, configurable via `--nearest-max-ft`). New `geom_source` enum value `nearest_polygon` distinct from `bin_polygon`, plus a new `match_dist_ft` column for drill-down. Validator accepts the expanded set, splits per-source × per-borough table into `bin_exact`/`nearest`/`point` columns, and emits nearest-distance percentiles in `summary.md`.
- **Rebuilt `curation/scaffolding_permits_through_2025/`** with the fallback: overall polygon match rate **99.98%** (bin_exact 95.28% + nearest_polygon 4.71%), all 10 source×borough combinations ≥ 99.82%, only **52 rows** remain on point fallback (all permits with no building within 200 ft — parks, bridges, plazas). Nearest-distance p50=36 ft, p99=130 ft, max=194 ft. Coverage grew slightly from 186.81 km² → 188.82 km² (polygon fallbacks are larger than bare point buffers).
- Updated wiki with new design decision #10, index entry, and plan cross-reference.

## [2026-04-21] build | scaffolding-permits-curation implemented + built for 2025-12-31 cutoff
- Implemented `dagspaces/common/curation/` (reusable `socrata.py` fetcher) and `dagspaces/common/curation/permits/` (6 modules: `fetch.py`, `normalize.py`, `buffer.py`, `validation.py`, `scaffolding_permits.py`, plus the CLI in `curation/cli.py` + `__main__.py`). Test suite at `tests/test_scaffolding_permits_curation.py` (15 tests covering normalize, buffer, validation fatal/warn paths, end-to-end orchestrator with monkey-patched Socrata).
- **First real build at `curation/scaffolding_permits_through_2025/`:** 323,474 publishable permits (73,371 DOB NOW + 255,237 BIS → 5,041 post-cutoff trim + 93 no-geometry drops). Overall BIN → polygon match rate 95.28% (range 87.59-97.20% across source × borough — BIS consistently ~95-97%, DOB NOW slightly lower at 87-97% because DOB NOW often references BINs for new-construction buildings not yet in `nyc_buildings.parquet`). Coverage 186.81 km² (24.01% of NYC land). All 8 fatal validation checks pass.
- **Two design decisions discovered during implementation** (added to the wiki page as #8 and #9): (a) BIS's `issuance_date` is a **plain-text `MM/DD/YYYY` column**, not a Socrata floating timestamp — the natural `issuance_date <= '2025-12-31T23:59:59'` server filter does lexicographic string comparison and silently passes ~5% of post-cutoff rows. Fix: year-level `substring(issuance_date, 7, 4) <= 'YYYY'` server prune, then exact client-side clip after parse. The dropped-permit funnel in `summary.md` reports how many rows the client-side clip removed (5,041 in the first real build). (b) `coverage.geojson` is written as a FeatureCollection of per-Polygon features (5,541 polygons for this build), not a single dissolved MultiPolygon — the giant MultiPolygon exceeds pyogrio's default 16 MB per-feature size limit. Downstream `CyclomediaCatalog.query(within=...)` re-dissolves via `unary_union` on intake, so semantics are preserved.
- **Socrata fetch hardening:** don't retry 4xx client errors (they won't become valid on retry); do retry 429/5xx transients with exponential backoff.
- Validation module emits `summary.md` (8 fatal + 12 warn checks, BIN match by source × borough, dropped-permit funnel, top-20 BINs by permit count, pagination table, scaffold_type / permit_status distributions) + `validation_report.parquet` (per-permit boolean drill-down). Fatal failures still write both artifacts so diagnosis doesn't require a second Socrata pull.
- Updated wiki page status planned → built; updated index entry with build stats.

## [2026-04-21] plan | scaffolding-permits-curation page created (first sub-dataset bootstrap)
- Created `vault/context/wiki/scaffolding-permits-curation.md` documenting the planned DOB NOW + BIS scaffold/shed permit curation pipeline. Full implementation plan at `docs/plans/scaffolding-permits-curation.md`.
- Sources: DOB NOW (`w9ak-ipjd`, scaffold/shed flags) unioned with BIS (`ipu4-2q9a`, `permit_subtype ∈ {SH,SD,SF}`), unauthenticated Socrata, 50k-row paginated fetch, parquet cache. Filter: `issue_date ≤ 2025-12-31`, all 5 boroughs, all permit statuses (expired/signed-off included).
- Spatial mask: BIN-joined to `data/geo/nyc_buildings.parquet`, 80-ft buffer in EPSG:2263. Rows with unmatched BIN fall back to an 80-ft circle around `(lat, lon)` from the API, tagged `geom_source='point'`.
- Output layout: `curation/scaffolding_permits_through_2025/` with `permits.parquet`, `permits.geojson`, `coverage.geojson` (single dissolved MultiPolygon, the consumable for `CyclomediaCatalog.query(within=...)`), and `by_source/` raw parquets.
- Key design decisions documented on the wiki page: (1) DOB NOW filings with null `first_permit_date` dropped — "issued" means a permit was actually issued, not just filed; (2) expired/signed-off permits included because the purpose is image retrieval near any historical scaffold, not current compliance; (3) BIS and DOB NOW copies kept separate (no cross-source dedupe); (4) 80-ft buffer chosen for image-recall over the compliance notebook's 50-ft default.
- **Added post-pull validation module** `dagspaces/common/curation/permits/validation.py` modeled on `cyclomedia_catalog/validation.py`: 8 fatal + 12 warn checks emit `validation_report.parquet` + `summary.md`. Headline metric is BIN → building-polygon match rate broken out per source × per borough. Fatals refuse to publish outputs; warns only log. Dropped-permit funnel surfaces how much recall is leaked at each preprocessing step (fetch → normalize → join → buffer); Socrata pagination-truncation detection guards against silent API cutoff.
- Module landing site: `dagspaces/common/curation/permits/` (with top-level `dagspaces/common/curation/socrata.py` reusable across future sub-datasets). CLI: `python -m dagspaces.common.curation scaffolding-permits --cutoff 2025-12-31 --buffer-ft 80 --out curation/scaffolding_permits_through_2025/`.
- Added index entry under Infrastructure; awaiting plan approval before implementation.

## [2026-04-21] update | cyclomedia-catalog: QC clear + rejoin_wfs manifest fix
- **Queens rebuild completed (job 691726):** after the long-tail queens fd walk finally finished, full catalog is **31,534,741 rows** (queens jumped from 0.20M → 11.63M once the remaining pull landed). All 4 fatal validation checks pass; WFS hit rate stays 100% on every dataset. Sampled 20 `image_path` rows — all exist and `file_size` matches `os.path.getsize` exactly; 1000/1000 `geom_wkb` decode to `(longitude, latitude)` within 1e-9.
- **Warnings explained and cleared as benign:** (1) 3.49% of rows have `file_size ≤ 50KB`, but the breakdown by face shows the signal is entirely in **U** (sky) and **D** (ground) cube faces — the four horizontal faces (F/B/L/R) have only single-digit small files per dataset. Those two faces legitimately compress to <50KB. (2) 128,059 cross-dataset `(recording_id, face)` pairs sit on adjacent-borough boundaries (BK↔Queens 99,563; Bronx↔Manhattan 19,056; BK↔Manhattan 5,252; Manhattan↔Queens 4,188). Within-dataset uniqueness is perfect, so downstream queries just need to scope by `dataset` for strict dedup.
- **Bug fix in `indexer.py::rejoin_wfs`:** the rejoin path only wrote `rejoined_datasets` to `manifest.json` and never unioned into the `datasets` field, so `manhattan_2025_1k` (which only ever landed via rejoin) was missing from `datasets` even though its partitions and `row_counts` entry were present. Added `merged_datasets = sorted(set(existing_manifest.get("datasets", [])) | set(row_counts.keys()))` mirroring the `build_catalog` path, and patched the live `manifest.json` on disk.
- Updated `wiki/cyclomedia-catalog.md` (status line: 20.1M → 31.5M, removed the queens 0.20M caveat, added warning explanations) and bumped `updated:` to 2026-04-21.

## [2026-04-20] update | cyclomedia-catalog: WFS glob fix, relaxed uniqueness, rejoin-wfs fast path
- **Root cause of low brooklyn (1.38%) + SI (0.00%) hit rates** in the first full build: `DEFAULT_CATALOG_GLOB` matched `recordings_*_chunks/*.csv` (manhattan, manhattan_latter, queens) and `recordings_*_part*.csv` (bronx), but brooklyn + SI WFS data only live under `/share/ju/cyclomedia/pull/out_catalog/recordings_{brooklyn,si}_2025.csv`. Extended the glob to include `out_catalog/` — WFS now loads 6.3M unique recording_ids vs 3.3M before.
- **Relaxed fatal check #4** from `unique(recording_id, face)` → `unique(dataset, recording_id, face)`. The pull pipeline uses a lat/lon bbox per borough, so ~4,159 recordings along NYC borough edges legitimately exist in two borough raw dirs (bronx↔manhattan: 3,176 recordings; brooklyn↔manhattan: 875; brooklyn↔queens: 108; SI has no overlaps). The catalog now faithfully mirrors disk; callers that want a single copy should `.unique(subset=["recording_id","face"])` at query time. Added warn #11 to surface the cross-dataset overlap count in `summary.md` (current: 24,956 pairs).
- **Added `rejoin_wfs()` fast path** in `dagspaces/common/cyclomedia_catalog/indexer.py` that re-runs only the WFS join against existing partitions, skipping the expensive fd walks and manifest parses. Reads each `by_dataset/dataset=X/` partition, strips WFS-sourced columns (`recordedAt`, `recorderDirection`, `bearing`, `catalog_hit`, `year`, lat/lon fallback, ...), renames current `latitude`/`longitude` back to `manifest_latitude`/`manifest_longitude` so the same coalesce runs, calls the extracted `_join_wfs_and_derive()`, `rmtree`s the old per-dataset dir, and rewrites. Refactored `_build_dataset_rows` into `_explode_walk_with_manifests` + `_join_wfs_and_derive` shared by both paths. New CLI subcommand `rejoin-wfs --datasets ...`.
- **Full-borough build completed (2026-04-19 → 2026-04-20):** 9.7h SLURM job on `ju` partition wrote 16.86M rows across brooklyn/queens/bronx/si; prior manhattan run had added another 3.24M. After rejoin, **20,106,960 rows** total with **100% WFS hit rate on every dataset**. `rejoin_wfs` completed all 5 datasets in **238.6s** on the login node — a 150× speedup over rebuild.
- **Queens coverage caveat documented:** only 33,059 queens recordings walked on disk vs 1.94M in the WFS CSVs (~1.7% pulled). Surfaced in the wiki page header.
- Updated `wiki/cyclomedia-catalog.md` (status, module table, schema note, indexer pipeline with two-phase split, sanity check #4 + #11, CLI section, new decisions 6/7/8) and bumped `updated:` to 2026-04-20.

## [2026-04-17] build | cyclomedia-catalog implemented and smoke-tested
- Shipped `dagspaces/common/cyclomedia_catalog/` (8 files: `schema.py`, `walker.py`, `manifest.py`, `wfs.py`, `indexer.py`, `validation.py`, `catalog.py`, `cli.py`). Added `polars` 1.39.3 + `polars-st` 0.7.0 to `pyproject.toml`; added `pytest` 9.0.3 as dev dep.
- **Smoke test (plazas_sample):** 4,737 face rows built in ~6s (4,737 jpegs walked by fd in ~1.8s, 791 manifests parsed in ~2s, WFS join against 3.27M-row catalog, partitioned parquet written). All 4 fatal checks pass; warns reflect plazas_sample being a custom (non-borough) pull: WFS hit rate 10.77%, 33/4737 rows outside NYC bbox (real plazas outside the strict rectangle), 217 rows with tiny file_size. Spot-check: `W0CJTS42/R` → `bearing = (119.49 + 90) mod 360 = 209.49` matches stored value exactly.
- **Test suite:** `tests/test_cyclomedia_catalog.py` with 20 tests covering schema helpers, walker (fd + scandir paths), manifest parser, WFS loader, end-to-end build on a synthetic 3-recording tree, each fatal validator (including a deliberately mismatched `imageId` fixture), each warn validator, and every query filter (`faces`, `between`, `within`, `datasets`, `build_inference_parquet`). All 20 pass in ~2s.
- **Known gotchas (fixed):** walker emits `dataset` as Categorical but manifests from dicts are Utf8 → cast both to Utf8 before join; Polars 1.39 is strict about string→datetime literals and about tz-aware vs tz-naive comparison → `_coerce_datetime` localizes bare dates to `America/New_York` to match the catalog's `recordedAt` tz; `is_in(Series)` is deprecated → catalog_hit now derived from a sentinel `_wfs_hit` column merged during the left join.
- **Not yet done:** full borough-wide build (manhattan/brooklyn/queens/bronx/si), incremental refresh command, shim rewrite of `scripts/create_cyclomedia_dataset.py`, pipeline-config migration.

## [2026-04-17] update | cyclomedia-catalog switched to Polars + polars-st
- Flipped decision #2: query engine is now **Polars + `polars-st`** instead of DuckDB. Rationale: all-Python surface with no SQL layer, single lazy chain (`scan_parquet → filter → collect`) composes cleanly with the rest of the codebase; `polars-st` wraps GEOS for point-in-polygon (`st.from_wkb` / `.st.within`) which covers every spatial query this catalog needs. Storage layout (hive-partitioned parquet with `geom_wkb`) is unchanged, so this is reversible.
- Added a query-implementation sketch to the plan showing `pl.scan_parquet` + hive partitioning + the `polars-st` spatial filter.
- **Toolchain:** user installed cargo/rust at `/share/ju/matt/.cargo` + `/share/ju/matt/.rustup` and `fd` v10.4.2 at `/share/ju/matt/.cargo/bin/fd`. Indexer will invoke by absolute path or PATH-prepend; `os.scandir` fallback retained for portability.
- Wiki page, index tagline, and plan all updated.

## [2026-04-17] update | cyclomedia-catalog design decisions locked
- Reviewed with user; open questions resolved. Plan + wiki updated in place.
- **Decisions:** catalog at `/share/ju/cyclomedia/catalog/v1/` | DuckDB + `spatial` (Polars-native spatial too immature for `ST_Within`) | manual refresh only, no cron | keep all six faces (U/D `bearing=NULL`) | `borough` derived from dataset name with a **bounding-rectangle caveat** documented (edge recordings may lie in a neighboring borough; callers needing polygon-accurate borough reverse-geocode at query time).
- **Schema expanded** to capture every WFS catalog column *and* every non-trivial manifest.json field (per-face render provenance, per-face depthmap presence, `manifest.checkpoint` for pull-batch cross-check). Verified against a sample `manhattan_2025_1k` manifest.
- Ready to implement.

## [2026-04-17] create | cyclomedia-catalog plan + wiki page
- Proposed a centralized DuckDB-backed catalog at `/share/ju/cyclomedia/catalog/v1/` to replace the per-run NFS walk in `scripts/create_cyclomedia_dataset.py`. Schema: one row per `(recording_id, face)` with lat/lon, `geom_wkb`, `recordedAt`, `recorderDirection`, derived `bearing`, and validation flags. Query API takes `within=gdf` (spatial), `between=` (temporal), `faces`, `datasets`; returns a DataFrame matching the old script's output shape so all five dagspaces consume it unchanged.
- **Indexer**: `fd`-based walk (fallback to threaded `os.scandir` if `fd` missing on cluster) → manifest parse → join WFS catalog CSVs → partitioned parquet `by_dataset/year=...`. 10-invariant sanity-check suite (recording_id↔face mapping, manifest imageId vs dirname, catalog hit rate, NYC bbox, JPEG truncation sniff, symlink-escape, etc.) with a pytest fixture on `plazas_sample`.
- **Plan:** `docs/plans/cyclomedia-catalog.md` (milestones, open questions, migration path).
- **Wiki:** created `wiki/cyclomedia-catalog.md`; added to `index.md` under Infrastructure.

## [2026-04-13] fix | UrbanEmbed DP worker chunked streaming + partial recovery
- **Incident (2026-04-12):** 1,038,932-row embed job on `slurm_gpu_4x` wasted 25h of A6000 time. Rank 0 straggled past the Python watchdog's 24h timeout; ranks 1/2/3 finished at 25.3/25.5/25.9h. `_run_data_parallel_embed` unconditionally `os.unlink()`-ed every worker pickle in its cleanup loop *before* the `raise`, so the three good ranks' embeddings existed on /scratch for a few ms and were then deleted. Stage output dir was empty. Two compounding bugs: (1) the unconditional cleanup, (2) `streaming_io=True` in the DP path only ran *after* `_run_data_parallel_embed` returned, so workers accumulating all embeddings in RAM before a single end-of-run pickle write meant a kill mid-run left nothing recoverable. See [[urban-embed#Fault Tolerance and Partial Recovery]].
- **Fixes:**
  - `slurm_gpu_4x.yaml`: `timeout_min` 2880 → 4320 (48h → 72h hard SLURM limit).
  - `_DP_EMBED_WORKER_SCRIPT`: workers now write `chunk{idx:05d}.pkl` into `{result_path}.chunks/` every `CHUNK_BATCHES=50` batches via atomic temp+fsync+rename (`_atomic_write_pickle`). A kill at 95% progress preserves 95% of embeddings.
  - `_run_data_parallel_embed`: default `timeout` 86400 → 255600s (~71h, ~1h under SLURM for cleanup headroom). Return type changed from `List[Any]` to `Tuple[List[Any], List[str]]`. Per-rank error tracking; chunks from each rank are loaded + concatenated; `None` placeholders fill gaps so `len(all_embeddings) == len(prompt_texts)` regardless of failures. Cleanup only unlinks chunk files/markers for ranks that fully succeeded — failing ranks' chunk dirs are left on disk and their paths are printed for manual recovery.
  - `dagspaces/urbanembed/stages/embed.py::run_embed_stage`: unpacks the new tuple, runs the merge+streaming loop unconditionally (postprocess already handles `None` embeddings), then raises `RuntimeError` *after* the final parquet flush so partial results are always persisted before the exception propagates. The raised message names the output directory that holds the recovered data.
- **Wiki:** updated [[urban-embed]] (new "Fault Tolerance and Partial Recovery" section, two-layer Checkpointing row, Multi-GPU Scaling bullet list with worker-side streaming + watchdog sizing), updated [[concept-chunked-dp-worker]] "See also" entry to point at the new embed DP streaming behavior.
- **Smoke tested** the chunk-load + `None`-placeholder + orphan-`.tmp`-sweep logic via an inline script; no vLLM needed.

## [2026-04-12] update | Building-footprint matching for compliance map
- Rewrote `scaffolding_compliance_map.ipynb` to use NYC building footprint polygons (`data/geo/nyc_buildings.parquet`, 1M+ buildings) instead of arbitrary 100m radius KD-tree matching
- Buildings joined to DoB filings via BIN, footprints buffered by configurable distance (default 50 ft) to cover sidewalk + near-street zone
- Buffer sweep analysis: 20ft=6.5%, 50ft=85%, 80ft=94.5% match (camera is in the roadway, not on the sidewalk)
- Added building-centric view: cross-tab of permit lifecycle x detection presence
- Updated `guide-compliance-map.md` wiki page

## [2026-04-12] add | Scaffolding permit compliance map notebooks + guide
- Created two compliance map notebooks in `notebooks/scaffolding/`:
  - `scaffolding_compliance_map.ipynb` — rerank-based: 2,019 camera-position detections from cross-encoder reranking
  - `scaffolding_compliance_raster.ipynb` — raster-based: ~1,059 hotspots from ray-accumulated GeoTIFF heatmaps with scaffold type (green/white) classification via signed_diff raster
- Both cross-reference against NYC DoB permit data (Dataset 1: filings via Socrata, Dataset 2: permit issuance via BIN join) to classify as permitted/expired/unpermitted
- KD-tree spatial matching on EPSG:2263 (NY State Plane), 100m radius, folium interactive map output
- Raster approach advantages: positions represent estimated scaffold location (ray convergence), aggregates multiple observations, includes scaffold type
- Created `wiki/guide-compliance-map.md` documenting both approaches, added to `index.md` under Guides

## [2026-04-10] create | concept-chunked-dp-worker distillation page
- Created `wiki/concept-chunked-dp-worker.md` — the *why* behind the chunked DP-full worker fix: vLLM 0.19 serial multimodal rendering, leaky LRU cache (vllm #15294, #35191), chunk-size tuning table (64 is the sweet spot), pre-resize rationale, mandatory multimodal engine kwargs (`mm_processor_cache_gb=2`, `mm_encoder_tp_mode="data"`), `max_num_seqs` sizing from the KV cache concurrency report, streaming parquet shard path resolution via `HydraConfig.sweep.dir`, why `use_tqdm=False` under SLURM, and how the pattern replaces the old Ray-based flow.
- Cross-linked from `[[vllm-inference#Chunked DP Worker]]` (the implementation) and `[[troubleshooting#Issue 4 Multimodal Rendering Bottleneck Engine Core OOM urbanpairvqa]]` (the concrete incident).
- Added to `index.md` under Concepts.
- Scope split note: the parallel UVA vault at `/share/ju/matt/vaults/uva/` is consumer-facing (Shedfolio etc.) and intentionally does NOT carry this deep-internal page; its `mllmsci-framework.md` / `urban-pairvqa-pipeline.md` were trimmed to high-level consumer notes pointing back at this vault.

## [2026-04-10] fix | Chunked DP worker for multimodal vllm_inference
- `dagspaces/common/vllm_inference.py`: rewrote `_DP_FULL_WORKER_SCRIPT` to process its row shard in chunks of 256 instead of one giant `llm.chat(big_list)` call. Each chunk now (1) preprocesses rows, (2) decodes + resizes both PIL images per row in parallel CPU threads (16-thread pool, PIL releases the GIL), (3) rewrites image blocks to vLLM's `image_pil` format, (4) calls `llm.chat(use_tqdm=False)`, (5) postprocesses, (6) drops chunk-local state. Streaming progress (rate, ETA, img-decode vs gen breakdown) logged every `log_every` rows (default 1000).
- `_run_data_parallel_full` and `run_vllm_inference` now thread `chunk_size`, `log_every`, `image_max_pixels`, `image_load_workers` from `cfg.model.*` into the worker task dict.
- `_build_engine_kwargs` auto-sets `mm_processor_cache_gb=2` and `mm_encoder_tp_mode="data"` for multimodal models (caps the LRU cache that leaked in vLLM #15294, #35191; enables vision-encoder DP across GPUs in vLLM 0.19+).
- `dagspaces/urbanpairvqa/conf/model/vllm_multimodal_qwen3_vl_8b_thinking.yaml`: bumped `max_num_seqs` 8→16 and `max_num_batched_tokens` 8192→16384 to use the KV cache the engine reports (~22.7x concurrency at 8192 tokens). Added `mm_processor_cache_gb`, `mm_encoder_tp_mode`, and the new `chunk_size`/`log_every`/`image_load_workers` knobs.
- Updated [[vllm-inference]] with a new "Chunked DP Worker" section and revised "Known Issues".
- Updated [[troubleshooting]] with new "Issue 4: Multimodal Rendering Bottleneck + Engine-Core OOM (urbanpairvqa)" entry documenting the symptom, root cause, and the chunked-worker mitigation.
- Root cause for the original 12h render + OOM-at-3000-rows: vLLM 0.19's `llm.chat(list)` does not pipeline multimodal preprocessing with generation; it serially renders every conversation through the HF processor (~5 it/s for 2×1024² images on Qwen3-VL) before any GPU work, while the engine-core RSS climbs because of the leaky multimodal LRU cache.

## [2026-04-08] update | Add Shedfolio downstream project reference
- Updated `project-overview.md` with Downstream Projects section referencing Shedfolio (`/share/ju/matt/shedfolio/`)
- Shedfolio uses MLLMSCI dagspaces for citywide scaffold detection, type classification, and DOB permit validation

## [2026-04-08] add | Ray accumulation interpolation for artifact_gen
- Added directional flow vector interpolation as alternative to IDW in raster stage
- Each face image casts a ray (K meters) in its absolute bearing direction; cells along ray accumulate score with linear distance decay
- Bearing computed from `(yawDegrees + FACE_BEARING[face]) % 360` using Cyclomedia face geometry (F=0°, R=90°, B=180°, L=270°)
- Bresenham line rasterization, optional normalization by ray count per cell
- Updated `artifact-gen.md` wiki page with full design documentation

## [2026-04-08] add | Artifact Gen dagspace for geospatial raster generation
- Created `dagspaces/artifact_gen/` — new dagspace for generating GeoTIFF rasters from text query + geolocated image embeddings
- Implements `RasterRunner` stage: loads urbanembed embeddings, encodes text query via BGE+projection (CPU) or Qwen (GPU), computes cosine similarity, interpolates onto regular grid via IDW, outputs float32 GeoTIFF with CRS metadata
- Uses `rasterio` for GeoTIFF output, `pyproj` for CRS transformations, `scipy.spatial.cKDTree` for spatial neighbor queries
- Pipeline config `raster_from_embeddings.yaml` consumes three upstream urbanembed artifact directories (embeddings, PCA, projection)
- Added `rasterio>=1.3.0` to `pyproject.toml`
- Created `vault/context/wiki/artifact-gen.md` wiki page, updated index

## [2026-04-06] bootstrap | Initial wiki creation
- Created vault structure: `vault/context/{wiki,sources,raw}`
- Created `WIKI_SCHEMA.md` with conventions and workflows
- Created `index.md` content catalog
- Generated initial wiki pages from full codebase exploration:
  - Project overview, architecture, all 5 dagspaces
  - Shared infrastructure (vLLM, Ray, W&B, orchestrator)
  - Configuration system (Hydra, models, launchers, pipelines)
  - Guides (bootstrapping, custom stages)
  - Concepts (tiling, counterbalancing, guided decoding, verification)
  - Reference (CLI commands, file locations)
  - Troubleshooting (known performance issues)

## [2026-04-07] update | Trajectory graph: Street Smart scoring + global connectivity
- Extracted actual navigation algorithm from StreetSmartApi.js v26.1 (minified bundle)
- Replaced heuristic heading-diff filter with the real Street Smart scoring function: `S(dist, angle) = angle > π/4 ? -1 : (1 - angle/π) / (1 + 0.1·√dist)`
- Added component bridging (Phase 4) — graph now 99.9% connected (1 component of 259K+ nodes)
- Documented the complete algorithm: dual system (arrow-key spatial scoring + LRS route cruise mode)
- Updated STREETSMART_API_REFERENCE.md with deobfuscated JS source code

## [2026-04-07] add | Validation pipeline guide for city stakeholders
- Created `vault/context/wiki/guide-validation-pipeline.md` — 7-phase evaluation pipeline
- Covers: human baseline (inter-annotator agreement), pseudo-label quality audit (TREC pooling), stratified test set design (sample size calculations), model evaluation (disaggregated metrics with CIs), calibration (conformal prediction), external validation (DOB permit cross-reference), stakeholder deliverables (model cards, failure analysis)
- 25+ papers cited with direct links: NIST AI frameworks, conformal prediction (Angelopoulos & Bates), NYC scaffolding detection precedent (arXiv:2402.06801), active learning, geographic equity
- Key precedent: NYC dashcam scaffolding paper achieved 78% recall/79% precision and found 529 unpermitted structures via DOB cross-validation

## [2026-04-07] add | Embedding similarity threshold analysis
- Created `vault/context/wiki/concept-embedding-thresholds.md` — methods for finding decision boundary in embedding space
- Empirical analysis: background Gaussian fit on 1M cosine similarities, excess signal detection at 2.5–3.0σ (60–16x above expected)
- Literature survey: 15+ papers on background distribution modeling, knee detection, pseudo-labeling, VLM distillation, score distribution theory
- Recommended pipeline: threshold at 2.5–3.0σ for positives, below 0.5σ for negatives, exclude ambiguous zone

## [2026-04-07] add | Browser search pipeline stages for urbanembed
- Created `dagspaces/urbanembed/stages/build_browser_index.py` — PCA + uint8 quantization stage (CPU-only, `slurm_cpu_beefy`)
- Created `dagspaces/urbanembed/stages/train_query_projection.py` — bge→Qwen linear projection training stage (GPU, `slurm_gpu_2x`)
- Added `BuildBrowserIndexRunner` and `TrainQueryProjectionRunner` to orchestrator, registered in `_STAGE_REGISTRY`
- Added `browser_index` and `query_projection` config groups to `dagspaces/urbanembed/conf/config.yaml`
- Created `dagspaces/urbanembed/conf/pipeline/browser_search_cyclomedia.yaml` — 3-node DAG: embed → build_browser_index → train_query_projection
- Updated `urban-embed.md` with stage docs and config tables
- Updated `guide-browser-search.md` to reference pipeline stages as recommended path

## [2026-04-07] implement | Browser-based image search MVP
- Created `scripts/build_browser_index.py` — PCA + uint8 quantization + artifact export from embed parquet output
- Created `scripts/train_query_projection.py` — trains linear projection from bge-small (384d) to PCA-reduced Qwen space (256d), requires GPU
- Created `viz/embedding_search/` — Vite + React web app: text search via Transformers.js ONNX encoder, optimized uint8 dot product search, deck.gl map view with score-colored scatter plot
- Updated `vault/context/wiki/guide-browser-search.md` with implementation details, file inventory, and setup instructions

## [2026-04-07] add | Browser-based image search design
- Created `vault/context/wiki/guide-browser-search.md` — architecture plan for pure client-side search over Qwen3-VL-Embedding vectors
- Key design: PCA dimensionality reduction + uint8 quantization + learned linear projection from browser-sized text encoder (bge-small) to Qwen embedding space
- Target: <300ms query latency, ~380MB static assets, no server

## [2026-04-07] add | Rerank stage for urbanembed
- Added `dagspaces/urbanembed/stages/rerank.py` — two-phase retrieval: embedding cosine recall + Qwen3-VL-Reranker-8B cross-encoder reranking via vLLM `llm.score()`
- Added `dagspaces/urbanembed/conf/model/qwen3_vl_reranker_8b.yaml` — model config with `hf_overrides` for `Qwen3VLForSequenceClassification`
- Added `dagspaces/urbanembed/conf/pipeline/rerank_cyclomedia.yaml` — standalone rerank pipeline
- Updated `dagspaces/urbanembed/orchestrator.py` — added `RerankRunner` to stage registry
- Updated `dagspaces/urbanembed/conf/config.yaml` — added `reranking:` config group
- Updated `vault/context/wiki/urban-embed.md` with rerank stage documentation

## [2026-04-07] add | Trajectory graph builder for urbanroamvqa
- Added `build_trajectory_graph()` to `dagspaces/urbanroamvqa/graph/builder.py` — reconstructs capture vehicle passes from timestamps + heading, chains sequential recordings, connects across passes at intersections
- Created `dagspaces/urbanroamvqa/conf/graph/trajectory.yaml` config
- Created `dagspaces/urbanroamvqa/graph/STREETSMART_API_REFERENCE.md` — reverse-engineering reference for Cyclomedia Street Smart viewer navigation
- Extended `scripts/create_cyclomedia_dataset.py` with `--catalog_csv` flag for WFS catalog enrichment
- Updated `vault/context/wiki/urban-roam-vqa.md` with trajectory graph documentation

## [2026-04-06] update | Consolidate shared data configs
- Moved 16 data configs from dagspace-local to dagspaces/common/conf/data/
- Created root-level datasets/ symlink (following models/, launchers/ pattern)
- Dagspace-specific configs remain local (OCR handlers, pairwise, roaming)
- Updated config-system.md, file-map.md, CLAUDE.md

## [2026-04-07] update | Fix vLLM data-parallel inference for multimodal + vLLM 0.19
- Rewrote `_run_data_parallel()` to follow vLLM 0.19 DP pattern: workers set `VLLM_DP_*` env vars, create `LLM()` without `data_parallel_size` in kwargs
- Fixed multimodal DP: workers now receive image file paths (strings) and load images lazily via PIL — old code silently dropped PIL images
- Added auto-detection: `dp_size = total_gpus // tp_size` when not explicitly configured
- Implemented `_run_data_parallel_embed()` for embedding DP — same subprocess pattern with `LLM(runner="pooling")` and `llm.embed()`
- Updated `run_embed_stage()` and `run_vllm_embed()` with DP branching
- Updated vllm-inference.md: rewrote DP section for subprocess + env var pattern
- Rewrote urban-embed.md: replaced stale Ray actor pool description with vLLM pooling architecture

## [2026-04-28] create | DOHMH restaurants curation mode + aggregate-restaurants utility
- New `dagspaces/common/curation/dohmh/` package: fetch / normalize / validation / build orchestrator / cuisines vocab / aggregate
- `dohmh-restaurants` CLI: pulls DOHMH Restaurant Inspection Results (Socrata `43nn-pn8j`), filters by cuisine + borough, geocodes via shared `geom.py`, writes `restaurants.parquet` (inspection-level — one row per `(camis, inspection_date)`, violation rows collapsed Critical-first)
- `aggregate-restaurants` CLI: opt-in step that collapses to one row per CAMIS, emits `n_inspections` / `n_grade_a/b/c` / `first_inspection_date` aggregates → `restaurants_aggregated.parquet`
- `permits/materialize.py` autodetect now prefers `restaurants_aggregated.parquet`, refuses bare `restaurants.parquet` with a helpful error pointing at `aggregate-restaurants`; `_load_units` rejects duplicate unit IDs to prevent silent sjoin inflation
- New wiki page `dohmh-restaurants-curation.md`; updated `index.md`

## [2026-04-28] create | Subway entrances curation mode (points-only geometry)
- New `dagspaces/common/curation/subway/` package: fetch / normalize / validation / build orchestrator / entrance_types vocab
- `subway-entrances` CLI: pulls MTA Permanent Station Entrances/Exits (data.ny.gov `i9wp-a4ja`, ~2.1k rows / 485 stations / 13 types), filters by entrance_type + division + borough + route (whole-token match against `daytime_routes`)
- Geometry path is **point-only**: buffers entrance lat/lon directly in EPSG:2263, skips the shared BIN→nearest→point fallback (wrong for sidewalk stairs)
- Synthesizes stable uid from `station_id + entrance_type + rounded(lat, lon, 7dp)` so rebuilds aren't churn-prone
- `permits/materialize.py` autodetect now also recognizes `entrances.parquet`
- New wiki page `subway-entrances-curation.md`; updated `index.md`

## [2026-05-01] add | NYC K-12 schools pairwise VQA pipeline + sampler canonical-pair fix
- Materialized `cyclomedia_near_schools_facing.parquet` from `facdb_schools_k_12/facilities.parquet` (3,103 publishable schools → 709,745 raw cyclomedia rows → 130,727 facing rows / 2,287 unique schools after per-unit facing filter; 18.4% kept)
- New configs: `prompt/pairwise_school_send_child_ordinal.yaml` (open-ended "based on appearance" framing per user — no exterior-cue constraint), `data/cyclomedia_near_schools_facing.yaml`, `pipeline/pairwise_schools_mvp.yaml` (Qwen3.5-4B default, `max_pairs=100,000`, `allow_replacement=false`), `sweep/schools_all_models.yaml`
- Fixed `_sample_distinct_canonical_pairs` in `samplers/cyclomedia_pairs.py`: under `allow_replacement=False`, `max_pairs` was previously silently capped at `n_units // 2` (perfect matching only). Now: `max_pairs=None` keeps perfect matching, explicit `max_pairs` samples distinct canonical pairs up to `C(n_units, 2)` via rejection sampling (sparse) or enumerate-and-shuffle (dense ≤10M). Tests at `tests/test_pairwise_unit_sampler.py::TestUnitSampler::test_no_replacement_*`
- Trimmed `max_model_len` on six common multimodal model configs to fit the 2×1024×1024 pairwise prompt: Qwen3.5-{2,4,9}B 16384→6144 (+ pinned `max_pixels: 1048576`), Gemma-4-{E2B,E4B} 16384→4096, Phi-4-MM kept at 8192 (HD-tile expansion makes it tight: 5,409 prompt tokens for 2 images)
- Updated `urban-pair-vqa.md` (new prompt/pipeline/data/sweep entries + new "max_pairs semantics" subsection)

## [2026-05-03] add | Pairwise visual-monotony/sterility pipeline
- New `prompt/pairwise_sterility_ordinal.yaml`: 5-point ordinal comparison of visual monotony/sterility, restricted to observable visual cues (repetitive forms, blank facades, empty sidewalks, regimented spatial design); explicitly forbids speculation about residents/income/safety
- New `pipeline/pairwise_cyclomedia_sterility_large.yaml`: random image-pair sampling (mode=image, no unit materialization) from `cyclomedia_pairwise_manhattan_2025_1` against Qwen3.5-9B with thinking enabled on `slurm_gpu_4x`; 480k pairs, balanced counterbalance, 4,800 repeats; `sampling_params_vqa.max_tokens=6144` to leave headroom for the reasoning trace
- New `common/conf/model/qwen3.5-9b/instruct_thinking.yaml`: thinking-enabled Qwen3.5-9B variant (`thinking_mode: on` + legacy `chat_template_kwargs.enable_thinking: true`, `engine_kwargs.structured_outputs_config.reasoning_parser: qwen3` so vLLM holds JSON grammar until after `</think>`, `max_model_len: 12288`, `max_num_seqs: 16`, `limit_mm_per_prompt.image: 2`)
- Updated `urban-pair-vqa.md` prompt + pipeline tables

## [2026-06-01] add | Open Restaurants (Dining Out NYC) curation family
- New curation module `dagspaces/common/curation/open_restaurants/` (license_types/fetch/normalize/validation/open_restaurants) pulling DCWP `fpeh-f7ci` outdoor-dining licenses; BIN-polygon geometry (shared `geom.py`) + 80-ft buffer, filter by license_type/borough, synthesized unique `uid` (dataset has no native key)
- Wired `open-restaurants` CLI subcommand; added `open_restaurants.parquet` to `materialize-cyclomedia` auto-detect; added `scripts/materialize_open_restaurants_cyclomedia.sub` + data config `cyclomedia_near_open_restaurants_facing.yaml`; tests in `tests/test_open_restaurants_curation.py` (14 pass)
- Built `curation/open_restaurants_all/` (1,309 licenses, 100% polygon match, 4.03 km² coverage) and materialized Cyclomedia: 191,964 image rows → 36,284 facing across 1,039 restaurants
- New wiki page [[open-restaurants-curation]]; added to `index.md` Infrastructure list

## [2026-06-01] add | Image-distribution preview report script
- New reusable `scripts/image_distribution_report.py` — bird's-eye PDF for any curation `*_facing.parquet` (cover stats, distribution charts, borough-coloured point map, stratified image montage); complements per-unit `scripts/facing_audit_report.py`. Reuses rcParams/accents from `pairwise_vqa_report.py` + thumbnail loader from `facing_audit_report.py`; optional `--units-parquet` join adds borough/category breakdowns + coverage ratio
- Generated `curation/open_restaurants_all/cyclomedia_near_open_restaurants_facing_distribution.pdf` (36,284 images / 1,039 restaurants)
- Documented under [[open-restaurants-curation]] "Preview / QA reports"

## [2026-05-18] add | Zone-geometry aggregation in pairwise_vqa_report.py
- `scripts/pairwise_vqa_report.py` gained `--zone-geojson`/`--zone-id-column`/`--zone-name-column` + `--coords-parquet` lookup flags: spatial-joins each image point to a containing polygon and rates the zone via the existing TrueSkill path, for image-mode runs with no `unit_uid_*` (e.g. sterility)
- Documented in `concept-trueskill.md` (new "Zone-geometry aggregation" section + Utility-script note on the multi-model `pairwise_vqa_aggregation_report.py`)
- Regenerated the sterility run as a tract-level report (NYC 2020 Census Tracts, 327 tracts rated): `reports/pairwise/sterility_large_by_tract.report.{md,pdf}`

## [2026-06-05] add | sample_reasoning_pdfs.py reasoning-trace PDF utility
- New `scripts/sample_reasoning_pdfs.py`: renders per-pair PDFs (the two images in presented order + verdict + full paginated `model_reasoning` trace) from a pairwise VQA output parquet; `--combined`, `--decisive-only`, `--include-empty`, `--seed`, `--title` flags. matplotlib PdfPages + PIL, no extra deps.
- Documented in `urban-pair-vqa.md` (new "Inspecting Reasoning Traces" section alongside the existing CSV `sample_reasoning_traces.py`)
- First use: qwen3.5-9b/instruct_thinking restaurants + schools 10k-pair sweeps -> reports/pairwise/{restaurants,schools}_reasoning_pdfs/

## [2026-06-08] update | concept-counterbalancing — population vs per-pair levels + correct `balanced` semantics
- Corrected a wrong claim: `balanced` mode does NOT present each pair in both orderings. It is a marginal 50/50 split via `obs_idx % 2`; a canonical pair is shown in only one order. Verified against `cyclomedia_pairs.py`.
- Added "Population vs per-pair counterbalancing" section: two levels (aggregate de-biasing vs full per-pair both-orders averaging), when each suffices, and the note that a full/both-orders sampler mode is NOT yet implemented.
- Added worked example from the qwen3.5-9b thinking restaurants/schools ~1k runs: presented_score mean ≈ −0.49 (left position bias) collapses to ≈0 after de-swapping (95% CI includes 0), SE≈0.045 at n≈1,100. Clarified `repeat_fraction` is a reliability probe, not a de-biasing mechanism.

## [2026-06-09] create | guide-neighborhood-aggregation — NTA-level pairwise ranking notebook
- New guide for `notebooks/css/neighborhoods.py`: joins pairwise output to sibling pairs.parquet (geo lives only there), locates each unit (mean recording lat/lon), point-in-polygon to NYC NTA 2020 (`data/geo/nynta2020_26b`, EPSG:2263), then two side-by-side NTA rankings — (A) direct NTA-vs-NTA zone TrueSkill, (B) mean of unit μ. Marimo, reactive dataset/min-comparisons controls; persists per-NTA + per-unit tables to notebooks/css/results/.
- Results snapshot (qwen3.5-9b/instruct non-thinking, 100k pairs): 100% unit→NTA assignment; restaurants 219 NTAs, schools 207; methods (A) vs (B) agree at Spearman ρ=0.905; top restaurant NTAs SoHo/Park Slope/West Village/Williamsburg (face-valid).
- Cross-linked from concept-trueskill (geographic-area unit row → unit-mode NTA notebook).

## [2026-06-09] update | troubleshooting — Issue 5: recompile interrupted urbanpairvqa run from streaming chunks
- Documented the failure mode: run finishes but consolidated `<dataset>_mvp_<ts>.parquet` missing; only `streaming/urbanpairvqa_pairwise/rank*_part*.parquet` chunks exist. Chunks are PRE-postprocess (15 raw cols), missing the 5 derived label/score cols from `_derive_labels()`.
- Recovery: new `scripts/recompile_streaming_pairwise.py` (concat chunks in row order + re-apply `_derive_labels`). Used it to recover the restaurants 100k run (110k rows, symmetric relative_label).

## [2026-06-09] create | urban-speech — new urbanspeech dagspace (ASR over video) + JU/A5000 launchers
- New dagspace `dagspaces/urbanspeech/`: `extract_audio` stage (parallel ffmpeg, video → 16 kHz mono PCM WAV + manifest parquet) then `asr` stage (granite-speech-3.3 via vLLM: speech LoRA from the model repo, `<|audio|>` prompt, 30 s chunking, greedy decoding, per-video transcript reassembly).
- Models downloaded to zoo: `granite-speech-3.3-2b` (TP=1, one A5000) and `-8b` (TP=2). Pipelines: `asr_videos` (2b) and `asr_videos_granite_ablation` (extract once, transcribe with both).
- New shared launchers for the **ju** partition (ju-compute-01, 4x RTX A5000 24 GB, sm_86 PCIe): `slurm_cpu_ju`, `slurm_gpu_ju_{1,2,4}x`. CPU requests kept small (4/GPU) because the node is shared.
- Created [[urban-speech]]; updated [[cli-reference]] (CLI, models, launchers, stage types), [[slurm-deployment]] (launcher table incl. klara, A5000 NCCL note), index.

## [2026-06-09] update | urban-roam-vqa, concept-street-graph — board-first street network redesign
- New canonical builder `graph/board_builder.py` (`graph_type: board`, `conf/graph/board_25m.yaml`, now the default + both pipelines): OSM network → consolidate intersections → largest component → discretize at uniform 25m pitch → attach imagery (greedy unique nearest + heading alignment) → contract imageless nodes (bounded reconnect) → QA gate asserts exactly 1 connected component. Legacy builders demoted to comparison-only.
- Walk mechanics fixed: `arrival_face()` now returns the backtrack face (was inverted — agents could never continue straight); menu = `legal_faces()` (resolvable, unvisited, imaged), so illegal moves are filtered, not fatal; dead-end turnaround; missing single faces dropped not fatal; seeds show all legal faces (`initial_face: ""`).
- Face frame: `StreetGraph.face_frame` defaults to `absolute` (F=N/R=E/B=S/L=W, matching the verified Cyclomedia NYC catalog finding) — closes the "known bug" note in concept-street-graph; `relative` available for vehicle-oriented data.
- Other fixes: KNN builder yaw normalization + hard error when osmnx missing + osmnx 2.x bbox order; trajectory builder yaw/sort misalignment, 4-direction Street Smart scoring (turns at intersections), honest post-bridge component report; config-fingerprinted graph caches (`graph/cache.py`); `compute_graph_diagnostics()` + `_validate_config()` in orchestrator.
- Tests: `tests/test_roaming_vqa.py` updated to new semantics + new coverage (legal faces, face frames, diagnostics, board builder unit + end-to-end with synthetic OSM, cache roundtrip) — 107 passing.

## [2026-06-09] update | urban-roam-vqa — Manhattan board built + marimo validation notebook
- Built the canonical board via new `scripts/build_roam_board_graph.py` (composes the exact pipeline graph config so the fingerprinted cache is pipeline-loadable): 36,210 nodes / 39,720 edges / 1 component / 99.2% imagery coverage / 25m median pitch. Artifacts pinned in `conf/graph/board_25m.yaml`: `data/graphs/roamvqa_board_25m_manhattan_2025_1.pkl` + `data/osm/manhattan_2025_drive.graphml`. (Fix along the way: `ox.truncate.largest_component` is directed-only; use `nx.connected_components` after `to_undirected`.)
- New marimo notebook `notebooks/roaming/network_validation.py` (supersedes the Jupyter `network_validation.ipynb`): QA summary with pass/fail checks, degree + pitch distributions, coverage map vs raw recordings, face/street alignment, legal-move audit, interactive folium close-up, 500 VLM-free random walks over the real mechanics, verdict cell. Headless run confirms: 0 zero-move nodes, all walks reach max steps, face-alignment max 45.0°; 11% face-shadowed directed edges at complex intersections noted as a known limitation.

## [2026-06-09] update | urban-roam-vqa — post-build review: geometry-orientation bug fixed, board rebuilt
- Post-build review caught a real defect: `ox.convert.to_undirected` leaves edge geometry direction arbitrary vs the (u,v) order `edges()` yields; 53% of Manhattan edges were flipped, so `_discretize_edges` anchored chains at the wrong end — 2,799 junction hops up to 2.4km and 7,109 same-bearing shadowed corridor nodes. Fix: re-anchor geometry at u before interpolating (+ regression test, 108 passing).
- Rebuilt board: pitch p90 66m → 27m (p10/50/90 = 23/25/27m), shadowed directed edges 11.1% → 2.1% (deg-2 shadow nodes 7,109 → 26), random-walk mean step 24.7m, still 1 component / 99.2% coverage / 36,317 nodes.
- Prompt templates: replaced hardcoded "three street views" with count-agnostic wording in tourist, tourist_independent, accessibility_surveyor, greenery_seeker (menus are now 1-4 panels).

## [2026-06-09] update | urban-roam-vqa — per-row guided decoding: illegal faces unrepresentable
- urbanvqa `_make_preprocess` now honors a per-row `guided_decoding` payload column (dict like `{"json": schema}`), overriding the cfg-level schema for that row (both unified and simple branches). Payload form bypasses `_build_guided_decoding_config`'s enum→choice collapse.
- roaming `RoamingStepper` resolves `prompt.structured_output.json_schema` once and attaches a per-row payload with `chosen_face.enum` narrowed to the step's legal faces. Fixes a latent no-op: `prompt.structured_output` previously never reached vLLM from roaming (vqa.py reads only `sampling_params_vqa.structured_output`).
- DP-worker path unchanged (first row's params win — comment added in vllm_inference.py); roaming pins concurrency=1. 7 new tests (narrowing, schema preservation, no-mutation, override/fallback paths); 115 passing. Verified payload materializes as `StructuredOutputsParams(json=...)` on vLLM 0.19.

## [2026-06-11] create | urban-pair-vqa — difference-testing tooling + experiment registry
- New `scripts/pairwise_vqa_difference_report.py`: on-the-fly group-difference tests over a pairwise run ("would the VLM rather eat at Chinese than Italian restaurants?"). Groups resolved from surfaced `<col>_a/_b` pair metadata or an external unit-metadata join on `unit_uid` (cuisine joins from `restaurants_aggregated.parquet` on camis — it is not on the facing manifest). Two tests per comparison: head-to-head (oriented score, repeat-collapsed, t/Wilcoxon/sign) and rating-level (TrueSkill μ, Welch/MWU); `--all-pairs` matrix mode with BH correction + heatmaps; markdown/PDF + tidy `*.tests.parquet`.
- Experiment registry: append-only JSONL at `machine-beholder/difference_tests/registry.jsonl`, deterministic experiment ids, dedupe-and-skip (`--force` to rerun), `--list` browser. Each experiment mirrored to the new W&B project `URBANPAIRVQA-ANALYSIS` (entity urbanekg, `job_type=difference_test`) via the sanctioned `WandbLogger`; W&B failures non-fatal.
- Validated on the may1 restaurants sweep: Chinese vs Italian n=320 direct pairs, no significant preference (h2h p=0.35, rating p=0.83); 12-cuisine matrix 66 pairs, 0 significant after BH, omnibus p=0.39. New wiki page [[guide-pairwise-difference-testing]]; `tests/test_pairwise_difference.py` (21 passing).

## [2026-06-11] update | urban-pair-vqa — multi-model difference testing + schools public-vs-private run
- `scripts/pairwise_vqa_difference_report.py` gains `--aggregation-dir`: discovers per-model runs via the aggregation-report layouts, runs both tests per model (BH within model), and adds a cross-model replication summary — forest plots (±1.96 SE, per-model stars) in pair mode, k-of-N significance heatmaps in matrix mode. Registry records carry `models`/`n_models`; W&B artifact names now keyed on experiment id (128-char cap fix). New SE fields (`mean_oriented_se`, `delta_mu_se`) on both tests. 22 tests passing (new multi-model end-to-end with planted opposite preferences).
- Schools run (may1_sweep, 6 models, FacDB `facsubgrp` joined on `uid`): PUBLIC vs NON-PUBLIC K-12 — null across the board. ~33.6k direct pairs per model, win rates 49.9–50.4%, all h2h p ≥ 0.20, all rating p ≥ 0.50, mean Δμ −0.011; 0/6 models significant on either test. Updated [[guide-pairwise-difference-testing]] with the multi-model mode + caveat (shared pair set → replication counts overstate independence).

## [2026-06-11] update | urban-pair-vqa — schools public-vs-private within each level: still null, leans public if anything
- Three multi-model difference tests on may1_sweep via `factype` (joined on `uid`): elementary PUBLIC vs NON-PUBLIC (570/557 units, ~12.1k direct pairs/model), high school (153/134, 805 pairs), middle (143/34, 182 pairs). Registry ids 12e953bd6d31 / 1754ebfc5cdc / fba7257e4fe7.
- No replicated effect anywhere; the weak directional lean is toward *public*, not private (rating-level: 4/6, 5/6, 6/6 models toward public; 0/6 significant in all three). Only nominal hit: phi-4 high-school head-to-head toward public (+0.134, p=0.013, win 54.8%) — uncorrected across models, no other model close. The pooled facsubgrp null is NOT masking offsetting within-level effects.

## [2026-06-11] update | urban-pair-vqa — may1_sweep CONTAMINATED (images not ingested); corrected schools findings on June 8 run
- User-flagged image-ingestion failure confirmed for the 2026-05-01 sweep: same model (Qwen3.5-9B), same schools dataset — may1 sweep shows Same=45.7% / MuchMore+MuchLess=0.8% vs the known-good 2026-06-08 run (multirun/2026-06-08_URBANPAIRVQA/12-21-15) at Same=17.9% / 8.0%. Blind models defaulting to "Same" explains the across-the-board nulls. **All registry experiments sourced from may1_sweep aggregations (restaurants + schools) are invalid as substantive findings**: f09a2cf1e6ea, 67751c05eb0a, 04c041790c1b, 12e953bd6d31, 1754ebfc5cdc, fba7257e4fe7, 5c53415011d1, 81beddf12765, 9b3428f13233.
- Corrected findings on the good run (Qwen3.5-9B, 110k pairs): **non-public schools significantly preferred** — overall h2h p=7e-21 toward non-public, Δμ=-0.47, d=-0.17 (registry b73639940d52); replicates within elementary (p=4e-12, d=-0.21) and high school (rating p=0.031, d=-0.26); middle same direction, underpowered. **Borough effects large + significant** (registry bf5545c9b6b2): Manhattan ≥ Staten Island > Queens > Brooklyn ≈ Bronx; Manhattan-vs-Bronx Δμ=+0.86, p_adj=1e-4; 5/10 rating-level + 7/10 h2h significant after BH.
- Diagnostic heuristic for future runs: Same-rate ≫ 30% with near-zero MuchLess/MuchMore is a red flag for broken image ingestion; compare against a trusted run of the same model/dataset.
- External data landed in curation/external/ (PLUTO 26v1 858,602 lots; DOE: DBN↔BIN/BBL crosswalk, demographic snapshot 2017-22, ELA/Math 2013-23) for upcoming covariate analyses; READMEs in each dir.

## [2026-06-11] update | urban-pair-vqa — registry cleaned; contaminated experiments redone on the good June 8 run
- Registry: 10 may1-sourced records quarantined to `machine-beholder/difference_tests/registry_removed_contaminated.jsonl` (reason annotated; full backup `registry.jsonl.bak`). Restaurants experiments (Chinese-vs-Italian, 12-cuisine matrix) have no valid source run yet — redo after the restaurants sweep reruns with fixed image ingestion.
- Redone on the good run (Qwen3.5-9B, registry now 7 valid records, all W&B-mirrored): within-level public/non-public (46710257f92f, 1bd48ea8b4ad, 4a9987a4ca2d — non-public preferred, elementary p=4e-12); 3-way sector matrix (6e9e84ad6392): **charter > public** (Δμ=+0.60, p_adj=0.002, d=0.21) and non-public > public (Δμ=+0.47, p_adj=0.001); charter ≈ non-public (ns). Community-district matrix, 52 districts (ca4db08e5935): 138/1326 significant, |d| up to 1.23 — dominant: MN08 UES (+37/-0), BK10 Bay Ridge (+33/-0), MN07 UWS, QN11, QN06; dominated: BK03 Bed-Stuy (0/-24), QN12 Jamaica, BK05 East New York, BX03 Morrisania. Clear socioeconomic gradient in VLM school preference.

## [2026-06-11] update | urban-pair-vqa — restaurants experiments redone on good June 8 run: strong cuisine hierarchy
- Good restaurants run confirmed healthy (multirun/2026-06-08_URBANPAIRVQA/12-21-06, Qwen3.5-9B, 110k pairs: Same=19.9%, Much*=18.9%). Redid both quarantined experiments: Chinese vs Italian (04dca656e2d2) — **Italian strongly preferred** (h2h mean oriented -0.35 over 644 direct pairs, p=2e-15; Δμ=-3.14, rating p=3e-60, d=-0.80). 12-cuisine matrix (4367be8d2f90): 56/66 significant. Standing by mean Δμ vs field: Italian (11-0) > Japanese (10-1) > Coffee/Tea > Bakery > Pizza > American > Mexican ≫ Chicken, Latin American, Chinese, Caribbean, Donuts (0-11). Mirrors the schools CD gradient — VLM preference tracks NYC socioeconomic geography.

## [2026-06-11] create | urban-pair-vqa — regression tooling + school covariates; poverty gradient confirmed
- Refactored shared analysis infra (registry, W&B mirror, run loading, metadata joins, formatting) into `scripts/pairwise_analysis_common.py`; difference tool re-imports, 22 tests unchanged.
- New `scripts/build_school_covariates.py` → `curation/external/school_covariates.parquet` (3,103 uids): FacDB + DOE BIN→DBN crosswalk + demographic snapshot (latest year, enrollment-weighted per building) + PLUTO 26v1 via BBL. Poverty/ENI coverage 1,738 units (83% of DOE sector); README with per-column coverage.
- New `scripts/pairwise_vqa_regression_report.py`: unit-level OLS/WLS (HC3, controls + partial R², standardized β, Spearman) + pair-level Δx slope validation; single/screen/multi-model modes; scatter+fit/residuals/Cook's-d/forest plots; shared registry (mode regression/screen) + W&B job_type=regression. `tests/test_pairwise_regression.py` — 17 tests, 39 total passing.
- Findings (good June 8 Qwen3.5-9B runs): school preference **declines with poverty** (β*=-0.156, pair-level p=3e-34, n=23,682 direct pairs; registry 83ca60a4e9c8), survives borough control (partial R²=0.022, 0d0a10d79263); 10-covariate screen (2f89e166a110): ENI/poverty dominate, building_age weakly −, log_assesstot weakly +. Restaurants: preference does NOT track DOHMH inspection score (R²=0.0000, 52f22556dd20). New wiki page [[guide-pairwise-regression-testing]].

## [2026-06-16] add | models — gemma-4-12b config group + schools capability-ladder sweep
- New model group `dagspaces/common/conf/model/gemma-4-12b/{instruct,base}.yaml` for the freshly-downloaded `google/gemma-4-12B` / `gemma-4-12B-it` (zoo: `/share/pierson/matt/zoo/models/gemma-4-12B{,-it}`, ~24GB bf16, `Gemma4UnifiedForConditionalGeneration`). Mirrors gemma-4-e4b settings (same gemma-4 family) so cross-size comparisons aren't confounded; TP=1 fits one 48GB A6000.
- New sweep `dagspaces/urbanpairvqa/conf/sweep/schools_gemma12b_klara2x.yaml` — single-model (gemma-4-12b/instruct) schools pairwise run, byte-identical to schools_all_models_klara2x.yaml otherwise, to extend the capability ladder and test whether the public/non-public gap + poverty gradient grow with rater capability in the gemma family (they do in qwen 2b→4b→9b).
- Updated [[cli-reference]] models table with a group-style-configs subsection (gemma-4 e2b/e4b/12b/31b, qwen3.5, phi-4).

## [2026-06-18] fix | infra — vLLM-nightly/gemma4_unified stack: LD_PRELOAD propagation + troubleshooting entries
- gemma-4-12b-it schools pairwise run (110k pairs) completed on klara_1x and was folded into the 7-model `machine-beholder/full_agg_20260615` agg; 10-analysis suite re-run + mirrored to W&B (URBANPAIRVQA-ANALYSIS). Result: gemma-12b **breaks** the monotone capability trend — qwen 2b→4b→9b public−nonpublic Δμ = -0.30→-0.37→-0.49 (toward non-public), but gemma e2b→e4b→12b = +0.02→-0.19→**+1.60** (12b flips to strongly favor PUBLIC, largest discrepancy of all 7). Sign is family-specific; capability scales magnitude within a family only.
- Propagated the flashinfer GLIBCXX `LD_PRELOAD=/usr/lib/.../libstdc++.so.6` fix from slurm_gpu_klara_1x to **all 12 GPU launchers** (slurm_gpu_{1x,2x,3x,4x,6x}, slurm_gpu_ju_{1x,2x,4x}, slurm_gpu_klara_{1x,2x,4x}, urbanocr/slurm_gpu_4x). CPU/monitor launchers untouched.
- [[troubleshooting]]: new "vLLM Nightly / gemma4_unified Stack" section — libcudart.so.13 (cu129 install recipe), GLIBCXX_3.4.32/LD_PRELOAD, and the `gemma4_unified` multimodal prompt-format AssertionError (fixed single-process; DP-worker path still pending). [[slurm-deployment]]: documented the LD_PRELOAD line in the launcher setup example.
