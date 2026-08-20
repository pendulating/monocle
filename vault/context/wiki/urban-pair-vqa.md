---
title: "UrbanPairVQA — Pairwise Comparison"
category: dagspace
created: 2026-04-06
updated: 2026-08-11
tags:
  - dagspace
  - pairwise
  - ordinal
  - comparison
  - counterbalancing
  - sampling
  - abstention
  - robustness
---

# UrbanPairVQA — Pairwise Comparison

UrbanPairVQA is the dagspace for **ordinal comparison of image pairs**. Given two urban images, it produces a relative judgment on a 5-point scale (MuchLess / Less / Same / More / MuchMore) for a given attribute (e.g., livability, wealth, parking availability). It handles pair generation, image stitching, counterbalancing for reliability, and label canonicalization.

## Purpose

- Pairwise relative comparison of image pairs via VLM inference
- Ordinal 5-point scale scoring with automatic label normalization
- Counterbalancing (presentation order swapping) for inter-rater reliability
- Reuses the UrbanVQA inference engine for the actual VLM call

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanpairvqa/cli.py` | Hydra CLI entry point |
| `dagspaces/urbanpairvqa/orchestrator.py` | DAG execution engine; defines `PairwiseVQARunner(StageRunner)` and pair manifest loading |
| `dagspaces/urbanpairvqa/stages/pairwise_vqa.py` | Core pairwise VQA stage: pair preparation, stitching, label derivation |
| `dagspaces/urbanpairvqa/samplers/cyclomedia_pairs.py` | Pair generation: `build_global_random_pairs()` with counterbalancing |
| `dagspaces/urbanpairvqa/prompt_opt/gepa_pairwise.py` | GEPA prompt "retranslation" harness: optimizes a case prompt against a production run's own labels; two-image + gemma4_unified support, in-process reflection, ordinal metric (`scripts/gepa_pairwise_subway.sub` to launch) |

## Workflow

```
Image Manifest (parquet with image_path, sample_id, metadata)
  -> build_global_random_pairs()
     -> Random pairing with counterbalance mode
     -> repeat_count for reliability assessment
     -> Metadata propagation (columns_a, columns_b)
  -> _prepare_pairwise_batch()
     -> Load left/right images
     -> _stitch_pair() -- horizontal concatenation at max_height
     -> _render_pair_prompt() -- inject pair_id and comparison instruction
  -> run_vqa_stage() (reuses UrbanVQA inference engine)
  -> _derive_labels()
     -> _canonicalize_label() -- normalize raw answer to ordinal label
     -> Invert label for swapped pairs
     -> Map to numeric score (-2 to +2)
  -> Output Parquet with relative scores
```

## Ordinal Labels and Scoring

The 5-point ordinal scale maps raw VLM answers to standardized labels and numeric scores:

| Label | Score | Meaning |
|-------|-------|---------|
| `MuchLess` | -2 | Left image has much less of the attribute than right |
| `Less` | -1 | Left image has less of the attribute |
| `Same` | 0 | Both images are equivalent |
| `More` | +1 | Left image has more of the attribute |
| `MuchMore` | +2 | Left image has much more of the attribute |

### Label Canonicalization

`_canonicalize_label(value)` normalizes diverse raw answers to the 5-point scale:

- Case-insensitive matching with whitespace/separator stripping
- Alias support: `"yes"` -> `More`, `"no"` -> `Less`, `"equal"` -> `Same`, `"true"` -> `More`, `"false"` -> `Less`
- Abstention aliases (only emitted when the toggle below is on): `"not sure"`, `"unsure"`, `"cannot tell"`, JSON-wrapped `{"answer": "NotSure"}`, the configured `not_sure_label`, etc. -> `NotSure`
- Substring matching as fallback (e.g., `"slightly more"` -> `More`)
- Default to `"Same"` when no match found

### Label Inversion for Counterbalancing

When pairs are presented in swapped order (`is_swapped=True`), the presented label is inverted before computing the relative score:

| Presented | Inverted |
|-----------|----------|
| `MuchLess` | `MuchMore` |
| `Less` | `More` |
| `Same` | `Same` |
| `More` | `Less` |
| `MuchMore` | `MuchLess` |

This ensures that `relative_label` and `relative_score` always reflect the canonical A-vs-B comparison regardless of presentation order.

### "Not sure" abstention (on by default since 2026-08-11)

A sixth option lets a model decline to judge when the two images don't give it
enough signal. **As of the 2026-08-11 consolidation it is ON for all seven
cases** — the battery always offers the out, so a model that cannot judge a pair
abstains rather than being forced onto the ordinal scale. It was opt-in (default
`false`) before that date, so pre-consolidation runs have no abstentions.

Turn it off only for an explicit ablation:

```bash
... pipeline=pairwise_subway_safety_mvp prompt.structured_output.allow_not_sure=false
```

When `prompt.structured_output.allow_not_sure=true`, `pairwise_vqa.py`:

1. **Appends the label to the guided-decoding enum** — `_augment_schema_with_not_sure()` adds `not_sure_label` (default `"NotSure"`) to `json_schema.properties.answer.enum`, so guided decoding will accept it. No-op (with a warning) if the schema has no answer enum.
2. **Adds a guidance line to the prompt** — `_render_pair_prompt()` appends `_not_sure_guidance()` (override via `prompt.structured_output.not_sure_text`) instructing the model to abstain only for true uncertainty, *not* for "looks about equal" (that's still `Same`). The default wording was genericized on 2026-08-11: it used to say "not when the two look about equally **appealing**", which reads as a hint on restaurants/street photography and as a non-sequitur on road quality/libraries. It is shared by all seven cases, so it must stay domain-neutral.
3. **Scores the abstention as `NaN`** — `NotSure` is deliberately absent from `_ORDINAL_SCORE`, so `_score_labels()` maps it to `NaN`. An abstention is **not** a `0`/`Same` judgment and must not be folded into the ordinal scale.

| Toggle state | `relative_score` dtype | Abstentions |
|---|---|---|
| off (ablation only) | `int64` (unchanged from legacy runs) | n/a |
| on, none emitted | `int64` | n/a |
| on, some emitted | `float64` | `NaN` rows for `NotSure` |

Because abstentions are `NaN`, downstream analysis that does `pd.to_numeric(...).dropna(subset=["relative_score"])` (TrueSkill in `scripts/pairwise_vqa_report.py`, `scripts/pairwise_analysis_common.py`) excludes them automatically, while `relative_label` still counts `NotSure` as its own category in the label distribution. The abstention rate is the signal of interest, not a rating.

On a `runtime.skip_inference=true` dry run with the toggle on, the deterministic debug labeler includes `NotSure` so the `NaN` plumbing is exercised without a GPU. The `not_sure_probe` sweep A/Bs `allow_not_sure={false,true}`. Tests: `tests/test_pairwise_stage.py`.

## Counterbalancing

The `build_global_random_pairs()` function in `dagspaces/urbanpairvqa/samplers/cyclomedia_pairs.py` supports three counterbalancing modes:

| Mode | Config Value | Behavior |
|------|-------------|----------|
| **None** | `"none"` | All pairs presented in canonical (A, B) order |
| **Random** | `"random"` | Each pair randomly assigned to canonical or swapped order |
| **Balanced** | `"balanced"` | For each canonical pair, both orderings are generated |

### `is_swapped` Tracking

Every pair row includes an `is_swapped` boolean column indicating whether the presentation order was flipped relative to the canonical pair. This enables:

- Post-hoc agreement analysis between forward and reversed presentations
- Inter-rater reliability metrics
- Detection of position bias in VLM responses

## Pair Generation

`build_global_random_pairs()` generates image pairs from a manifest:

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `manifest_df` | DataFrame | Input manifest with `image_path` and `sample_id` columns |
| `max_pairs` | int (optional) | Maximum number of pairs to generate |
| `seed` | int | Random seed (default: 777) |
| `allow_replacement` | bool | Whether to sample with replacement |
| `counterbalance_mode` | str | `"none"`, `"random"`, or `"balanced"` |
| `repeat_count` | int | Number of repeated evaluations per pair for reliability |
| `repeat_fraction` | float | Fraction of pairs to repeat (alternative to repeat_count) |
| `metadata_columns` | list | Manifest columns to propagate as `{col}_a` / `{col}_b` |

**`max_pairs` semantics under `allow_replacement=False`** (changed 2026-05-01):

| `max_pairs` | Behavior |
|-------------|----------|
| `None` | Perfect matching: `n_units // 2` pairs, each unit appears in at most one pair. |
| explicit int | Sample distinct **canonical** pairs (no `{a, b}` repeats; units may recur across pairs) up to `min(max_pairs, C(n_units, 2))`. |

Before this change, the `allow_replacement=False` branch silently capped at `n_units // 2` regardless of `max_pairs` (perfect matching only). The fix adds rejection-sampling and enumerate-and-shuffle paths in `_sample_distinct_canonical_pairs` (`samplers/cyclomedia_pairs.py`). Tests in `tests/test_pairwise_unit_sampler.py::TestUnitSampler::test_no_replacement_*`.

**Output columns:**

| Column | Description |
|--------|-------------|
| `pair_id` | Unique identifier for this pair instance |
| `canonical_pair_id` | Canonical pair identifier (same for forward/reversed) |
| `repeat_idx` | Repeat index for reliability pairs |
| `sample_id_a`, `sample_id_b` | Original sample IDs |
| `image_path_a`, `image_path_b` | Canonical image paths |
| `presented_left_path`, `presented_right_path` | Actual presentation order paths |
| `presented_order` | `"A_then_B"` or `"B_then_A"` |
| `is_swapped` | Boolean: True if presentation is reversed |
| `{metadata_col}_a`, `{metadata_col}_b` | Propagated metadata per image |

## Image Stitching

`_stitch_pair(path_a, path_b, max_height)` creates a side-by-side composite:

1. Load both images as RGB PIL Images
2. Scale each to `max_height` preserving aspect ratio
3. Create a white canvas of combined width
4. Paste left image at (0,0), right image at (left_width, 0)
5. Return as numpy array

The prompt instructs the VLM to interpret the left half as Image A and the right half as Image B.

### Current presentation path + `prompt.image_layout` (added 2026-07-10)

The production stage (`stages/pairwise_vqa.py::_make_pairwise_preprocess`) does **not** stitch — it passes two separate `image_url` content blocks (`file://` URLs, lazily loaded by vLLM) followed by the text prompt. The optional `prompt.image_layout` key selects the content-block order:

| Value | Content blocks | Use |
|-------|----------------|-----|
| `images_then_text` (default) | image A, image B, prompt | Production behavior — absent key means this |
| `interleaved_labels` | `"Image A:"`, image A, `"Image B:"`, image B, prompt | Reviewer-2 arm C1 — textual anchors adjacent to each image |
| `text_first` | prompt, image A, image B | Reviewer-2 arm C2 — question before images |

Unknown values raise `ValueError` at stage entry (validated even on `skip_inference` dry runs). CLI: `+prompt.image_layout=interleaved_labels`. Tests: `tests/test_pairwise_stage.py::TestImageLayout`. ⚠️ gemma-4-12b (`gemma4_unified`) prompt replacement is fragile to content-block changes — smoke 20 pairs before a full layout run.

## Diagnostics

The orchestrator computes pairwise diagnostics for quality monitoring:

- Label distribution across the 5-point scale
- Entropy of label distribution (higher = more diverse judgments)
- Agreement metrics between repeated pairs
- Position bias detection (forward vs. reversed presentation)

## Configuration

### Prompt Configs (`dagspaces/urbanpairvqa/conf/prompt/`)

**Consolidated 2026-08-11 to exactly seven ranking cases.** Everything else moved to the `deprecated/` sub-group (still selectable as `prompt=deprecated/<name>`). All seven are 5-point ordinal with the `NotSure` abstention **on by default**.

Five of the seven sit under one umbrella — *signs of public investment + government management* — where the umbrella is the construct and each case is a concrete publicly-managed asset that renders it visible. Restaurants (private commerce) and street photography (aesthetics) are standalone contrasts.

| Config | Case | Umbrella | Unit / mode |
|--------|------|----------|-------------|
| `pairwise_subway_safety_ordinal.yaml` | Subway safety — "which station entrance is safer?" | public investment | station entrance / `unit` |
| `pairwise_library_maintained_ordinal.yaml` | Libraries — "which library building is better maintained?" | public investment | library / `unit` |
| `pairwise_school_send_child_ordinal.yaml` | Schools — "which school would you rather send your child to?" | public investment | K-12 school / `unit` |
| `pairwise_road_quality_ordinal.yaml` | Road quality — "which roadway is in better condition?" | public investment | block / `image` |
| `pairwise_parks_plazas_ordinal.yaml` | Parks / plazas — "which one is better maintained?" | public investment | park or plaza / `unit` ⚠️ manifest not built |
| `pairwise_restaurant_eat_at_ordinal.yaml` | Restaurants — "which restaurant would you rather eat at?" | standalone | restaurant / `unit` |
| `pairwise_street_photography_ordinal.yaml` | Street photography — "which block is a more appealing photoshoot location?" | standalone | block / `image` |

#### Minimal prompt contract

Every case now shares one skeleton, and the only text that varies between cases is the unit noun, the question, and the interpretation line:

```
system: null            # no system turn is sent AT ALL — see below

Image A is the first image and Image B is the second image.
Both show <unit noun>.
<Question>?

Return exactly one label from:  MuchLess / Less / Same / More / MuchMore

Interpret this as "<A> is <label> <adjective> than <B>". Use "Same" when they look about the same.
```

Deliberately **absent**, to avoid constraining the output token distribution:

- no enumerated cues (`"e.g. signage, cleanliness, lighting, ..."`)
- **no persona at all** — the system turn is omitted entirely (the pre-consolidation schools prompt opened *"You are a parent in NYC"*; subway used an urban-planner/sociologist persona)
- no `"base your judgement only on observable cues"` / `"do not speculate"` guardrails
- no instruction about what to ignore (weather, time of day, season)
- no examples

#### The three system-prompt conditions (don't confuse them)

`_resolve_system_prompt()` in `stages/pairwise_vqa.py` returns `Optional[str]`; `None` means the `{"role": "system", ...}` message is never added to the list.

| `prompt.system` | What actually renders |
|---|---|
| `"You are ..."` | normal system turn |
| `""` (empty string) | a **vestigial empty system turn** — not the same as none, and a token pattern the models have barely seen. Normalized to `None` in code so nobody trips it |
| `null` / key absent | **no system turn at all** ← the battery default |

Two facts verified against the local checkpoints on 2026-08-11, both of which make this a genuine no-persona condition rather than a rename:

1. **Neither template injects a default persona** when the system turn is absent — checked `Qwen3.5-4B`, `Gemma-4-E4B-it`, `gemma-4-12B-it`. Rendered prompt lengths: Qwen 30 → 22 → 17 tokens for persona → empty → omitted (Gemma 29 → 21 → 16).
2. **Deleting the key used to substitute a *different* persona.** Before 2026-08-11 the stage fell back to a hardcoded `"You are a helpful assistant."`, so "removing the persona" by deleting `system:` silently swapped one for another. That fallback is gone.

⚠️ Re-adding any of these is an **ablation, not a fix** — put it in its own file so the battery stays comparable. The retired reviewer-2 arms are exactly this kind of ablation and now live at `prompt=deprecated/pairwise_subway_safety_ordinal_{paraphrase,nopersona,enumrev,flipped}`.

⚠️ **Prior sweeps are not comparable to runs from here on.** The five pre-existing cases were all rewritten (cue lists and personas stripped), abstention flipped from off to on, and libraries' `additionalProperties` flipped `true → false`. Re-run any baseline before pooling it with post-consolidation results — this includes the verified 7-model schools/restaurants sweeps ([[project_schools_restaurants_verification]]) and the reviewer-2 battery.

#### Deprecated prompts (`conf/prompt/deprecated/`)

`pairwise_wealth_ordinal`, `pairwise_wealth_yes_no`, `pairwise_livability_ordinal`, `pairwise_parking_ordinal`, `pairwise_sterility_ordinal`, `pairwise_relative_ordinal`, `pairwise_driving_safety_ordinal`, `pairwise_street_photography_sparse`, `pairwise_.yaml`, and the four reviewer-2 subway arms. Driving safety was retired in favor of road quality: it asked about *risk to a driver*, whereas road quality asks about the *condition of the road*, which is what reads as a public-investment signal.

### Pipeline Configs (`dagspaces/urbanpairvqa/conf/pipeline/`)

One pipeline per consolidated case; the retired ones moved to the `deprecated/` sub-group alongside their prompts (`pipeline=deprecated/<name>`).

| Config | Description |
|--------|-------------|
| `pairwise_subway_safety_mvp.yaml` | Subway-safety MVP — unit mode over `cyclomedia_near_subway_facing` (1,990 entrances; `max_pairs=100,000`, `allow_replacement=false`; Qwen3.5-4B default; sweep via `+sweep=subway_all_models_klara2x`) |
| `pairwise_libraries_mvp.yaml` | Library-level MVP (Gemma-4-E4B default; sweep via `+sweep=libraries_all_models`) |
| `pairwise_schools_mvp.yaml` | K-12 schools MVP (Qwen3.5-4B default; `max_pairs=100,000`, `allow_replacement=false`; sweep via `+sweep=schools_all_models`) |
| `pairwise_road_quality_mvp.yaml` | Road-quality MVP (added 2026-08-11) — **image mode** over `cyclomedia_all_2025_citywide`, `pair_seed=777` / `max_pairs=100,000` deliberately matching street photography so the two image-mode cases draw comparable block samples. Reuse the street-photography launcher sweeps. Replaces `deprecated/pairwise_driving_safety_mvp.yaml` |
| `pairwise_parks_plazas_mvp.yaml` | Parks/plazas MVP (added 2026-08-11) — mirrors libraries in unit mode. ⚠️ **Not runnable yet**: `cyclomedia_near_parks_facing` has to be curated first (see Data Config below). `max_pairs`/`allow_replacement` are provisional pending the real unit count |
| `pairwise_restaurants_mvp.yaml` | Restaurant-level MVP (Qwen3.5-4B default; ≤4B sweep via `+sweep=restaurants_all_models_4b`) |
| `pairwise_street_photography_mvp.yaml` | Street-photography MVP — **image mode** over `cyclomedia_all_2025_citywide` (500k citywide blocks; random street shots, no unit manifest → TrueSkill N/A; `max_pairs=100,000`; sweep via `+sweep=street_photography_all_models_klara2x`) |

All case pipelines run `max_pairs=100,000` per run. ⚠️ The completed 7-model **restaurants** sweeps were at 50k — re-run them at 100k before pooling with the new cadence; the older runs aren't directly comparable.

Deprecated pipelines: `pairwise_cyclomedia_{ordinal,livability_large,wealth_large,wealth_midsize,wealth_tester_256,sterility_large,parking_tester_*}` and `pairwise_driving_safety_mvp`. Their `override /prompt:` lines were repointed at `deprecated/…` so they still compose.

⚠️ The root `config.yaml` compose-time defaults moved with them: `prompt` and `optional pipeline` now default to the street-photography case (the image-mode one, matching the default `pair_sampler.mode: image`) instead of `pairwise_relative_ordinal` / `pairwise_cyclomedia_ordinal`. Every real run passes `pipeline=`, which overrides both.

### Data Config

| Config | Description |
|--------|-------------|
| `conf/data/cyclomedia_pairwise_manhattan_2025_1.yaml` | Manhattan 2025 Cyclomedia pairwise manifest |
| `common/conf/data/cyclomedia_near_libraries_facing.yaml` | NYC public-library facing manifest (~11.7k rows / 236 units) |
| `common/conf/data/cyclomedia_near_restaurants_facing.yaml` | DOHMH restaurant facing manifest (~490.8k rows / 18,488 camis) |
| `common/conf/data/cyclomedia_near_schools_facing.yaml` | FacDB K-12 schools facing manifest (130.7k rows / 2,287 units, of 3,103 publishable) |
| `common/conf/data/cyclomedia_near_subway_facing.yaml` | MTA subway-entrance facing manifest (45.9k rows / 1,990 entrances); upstream `curation/subway_entrances_all/` (see `dagspaces/common/curation/subway/`) |
| `common/conf/data/cyclomedia_all_2025_citywide.yaml` | Citywide street-level manifest, **500k rows / 100k per borough**, uniformly stratified (25k per dataset×face cell); image-mode source for the street-photography **and road-quality** cases. Built by `scripts/materialize_cyclomedia_citywide.py` (the citywide successor to the 30k `cyclomedia_all_2025_30k` OPF manifest) |
| `common/conf/data/cyclomedia_near_parks_facing.yaml` | FacDB parks/plazas facing manifest — ⚠️ **config written 2026-08-11, parquet NOT built.** Build with `python -m dagspaces.common.curation facdb-facilities --facgroup "PARKS AND PLAZAS" --out curation/facdb_parks_plazas/` then `materialize-cyclomedia` (auto-chains the facing filter). **Curation caveat:** most park/plaza FacDB rows carry no BIN, so the facing filter's building-polygon occlusion check (Fix F) has nothing to aim at, and centroid bearing is a weak proxy for a large or irregular park — audit `attribution_confidence` for discriminative power before trusting unit attribution, or `pair_sampler.weight_column` is inert. See [[concept-facing-filter]] |

### Sweep Configs (`dagspaces/urbanpairvqa/conf/sweep/`)

| Config | Lineup | Notes |
|--------|--------|-------|
| `libraries_all_models.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | All natively-multimodal instruct models |
| `restaurants_all_models.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | Full mirror of `libraries_all_models.yaml` — no size cap |
| `restaurants_all_models_4b.yaml` | gemma-4-e2b, qwen3.5-2b, qwen3.5-4b | Strict ≤4B subset; drops gemma-4-e4b (~4B effective / ~8B total), phi-4-mm (~5.6B), qwen3.5-9b |
| `schools_all_models.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | Same lineup as restaurants/libraries; pairs with `pairwise_schools_mvp.yaml` |
| `subway_all_models_klara2x.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | **Case 3** — same 6-model lineup as the restaurants/schools `klara2x` sweeps (klara_2x stage, `max_tokens=128`); pairs with `pairwise_subway_safety_mvp.yaml` |
| `street_photography_all_models_klara2x.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | **Case 4** — same lineup; pairs with `pairwise_street_photography_mvp.yaml` (image mode) |
| `not_sure_probe.yaml` | qwen3.5-9b, gemma-4-e4b × `allow_not_sure={false,true}` | A/B the abstention toggle (4 runs, 1,000 pairs each); works against any case pipeline |
| `reviewer2_probe_klara1x.yaml` | qwen3.5-2b (negative control), qwen3.5-9b | Reviewer-2 robustness probes: 1,000-pair seed-777 prefix, perturbation arm passed on the CLI (header documents every invocation) |
| `reviewer2_gemma12b_klara1x.yaml` | gemma-4-12b | Same probes for the unified-arch rater (klara_1x only, `.venv-nightly`, serial; no res-half arm — gemma-4 has no `max_pixels` knob) |
| `reviewer2_smallladder_klara1x.yaml` | qwen3.5-2b, qwen3.5-9b, gemma-4-e2b | Fresh same-seed **small-model ladder** for the Case-3 battery (added 2026-07-12). All three run the standard encoder path / stable `.venv` — gemma-4-e2b is `Gemma4ForConditionalGeneration` (`gemma4`, real `gemma4_vision` encoder), **not** the encoder-free `gemma4_unified` of the 12b, so it needs no nightly venv. res-half stays qwen-only |

Invoke a sweep with Hydra multirun:
```bash
python -m dagspaces.urbanpairvqa.cli --multirun \
    +sweep=restaurants_all_models_4b \
    pipeline=pairwise_restaurants_mvp
```

## Confidence-weighted pair sampling

`build_unit_random_pairs` supports **within-unit** weighted image selection via the `weight_column` parameter. Downstream of the per-unit [[concept-facing-filter]], each curated row carries `attribution_confidence ∈ [0, 1]` (higher = more confidently facing its attributed unit); passing that as the weight column biases draws toward the confidently-facing shots.

| Aspect | Behavior |
|---|---|
| Parameter | `weight_column: Optional[str] = None` |
| Where weights apply | Image selection **within** a unit only |
| Where weights don't apply | Unit-level pair selection stays uniform — preserves per-library coverage so TrueSkill sees roughly equal data per unit |
| Missing column | Falls back to uniform; one-line warning printed |
| All-zero / NaN / negative weights in a unit | Clipped to 0; if the whole unit ends up at 0, uniform fallback for that unit only |
| Hydra exposure | `pair_sampler.weight_column: attribution_confidence` (opt-in; default `null`) |

The libraries MVP pipeline (`conf/pipeline/pairwise_libraries_mvp.yaml`) sets `weight_column: attribution_confidence` by default. A manifest produced before the 2026-04-22 facing filter change doesn't have that column — the graceful fallback keeps old runs working. Tests live in `tests/test_pairwise_unit_sampler.py::TestWeightedWithinUnit` (weighted-bias, missing-column, per-unit-all-zero, negative/NaN-clipped).

Expected impact: with mean confidence ≈ 0.19 and max 0.77 on the first library run, weighted sampling biases meaningfully toward the top-quartile rows (`attribution_confidence ≥ 0.3`, ~25% of the pool) without dropping anything.

### Variants available on the same surface

These aren't implemented yet but the config surface is designed to admit them without breaking changes:

- **Top-K per unit** (`weight_top_k: int`) — keep top K rows by weight per unit, then draw uniformly. Stricter than continuous weighting.
- **Confidence floor** (`min_confidence: float`) — hard pre-filter before building `unit_to_indices`. Cleaner than relying on `weight=0` to suppress low-quality rows.

## Reviewer-2 Robustness Baselines (Case 3)

Perturbation battery (added 2026-07-10) showing the subway-safety finding — large raters' safety judgments track tract income + per-capita crime (see log entry) — is not an artifact of prompt wording or image presentation. Analysis: `scripts/pairwise_reviewer2_baselines.py`; probes are 1,000-pair **seed-777 prefixes** of the production 100k draw (`pair_sampler.max_pairs=1000`; never `runtime.sample_n`, which resamples the manifest and breaks the prefix). Models: qwen3.5-9b + gemma-4-12b (signal carriers, two architecture families) + qwen3.5-2b (negative control — perturbations must not awaken signal in a noise-level rater).

| Arm | Rules out | Mechanism |
|-----|-----------|-----------|
| A0 `retest` | (anchor: temp-0.6 test-retest ceiling) | baseline prompt re-run on the 1k prefix |
| A1 `paraphrase` | exact-wording artifact | `prompt=pairwise_subway_safety_ordinal_paraphrase` |
| A2 `nopersona` | persona primes SES stereotypes | `prompt=..._nopersona` |
| A3 `enumrev` | enum position bias | `prompt=..._enumrev` |
| A4 `flipped` | acquiescence bias | `prompt=..._flipped`; ⚠️ scores negated in analysis (`--flip-arms flipped`) |
| A5 `res-half` | reads signage, not scene | `model.engine_kwargs.mm_processor_kwargs.max_pixels=262144` (qwen only) |
| A6 `freeform` | guided decoding distorts | `prompt.structured_output.enabled=false` (parse-fallback rate reported) |
| A7 `temp0` | sampling noise | `sampling_params_vqa.temperature=0.0` |
| C1 `interleaved` | A/B mis-binding | `+prompt.image_layout=interleaved_labels` |
| C2 `textfirst` | attention-order artifact | `+prompt.image_layout=text_first` |

Evaluation: paired join to the production parquet on `pair_id` → exact agreement, linear-weighted kappa, Spearman, direction-flip rate, Same/NotSure deltas, swap-conditioned agreement. **Judge every arm against the A0 retest row, not 1.0** — decoding is temperature 0.6. Tier B (deferred): 25k-prefix outcome replication re-running the tract regression on the strongest arms.

**Tier A/C results (run 2026-07-10; 29/29 probes landed, all exact ordered prefixes, no blind-run signatures).** Primary contrast = probe vs the A0 retest run (`--baseline`, image-identical); each model's own `temp0` row is its decoding-determinism bound. Linear-weighted kappa:

| arm | gemma-4-12b (bound 0.935) | qwen3.5-9b (bound 0.668) | qwen3.5-2b (control) |
|---|--:|--:|--:|
| paraphrase | 0.809 | 0.549 | 0.448 |
| nopersona | 0.812 | 0.580 | 0.625 |
| enumrev | 0.871 | 0.562 | 0.244 |
| flipped (negated) | 0.542 | **−0.040** | 0.237 |
| freeform | 0.858 | 0.349 | 0.207 |
| res-half | n/a | 0.515 | 0.486 |
| interleaved | 0.820 | 0.542 | 0.637 |
| textfirst | 0.757 | **0.107** | 0.294 |

Verdict: **prompt wording is not doing the work** — paraphrase/nopersona/enumrev sit at or near each signal carrier's decoding-noise bound, and gemma-4-12b is robust on every arm including both layouts and unconstrained decoding. Two real sensitivities, both in qwen3.5-9b and both presentation/framing (not wording): (1) `textfirst` collapses its per-pair consistency (κ 0.107, flip rate 0.385) — question-before-images breaks its A/B binding, while gemma-12b is unaffected (0.757); (2) `flipped` does not invert (κ −0.04; Same-rate jumps to 38.9%) — "which looks less safe" is not answered as the inverse of "which looks safer" (gemma-12b partially inverts, κ 0.542 with Same-hedging at 29.6%). The negative control degrades (never gains signal) under every perturbation, as required. Runs/reports: `outputs/reviewer2_subway/agreement_vsA0_20260710_200722/` (vs A0) and the vs-production companion (image-draw-attenuated) in `outputs/reviewer2_subway/`; W&B `URBANPAIRVQA` runs `8x73wvcr` (vs A0) / `98u9ccol` (vs production). Stage dirs: attribute via `.hydra/overrides.yaml` under `multirun/2026-07-10_URBANPAIRVQA/`.

**Tier B outcome replication (run 2026-07-11; 25k-prefix runs, `--tier outcome`):** the tract-level regression replicates under both surviving perturbations, compared like-for-like against production judgments on the SAME 25k pairs. r(income) probe vs production-prefix: qwen3.5-9b paraphrase **0.488 vs 0.517**, nopersona **0.491 vs 0.517**; gemma-4-12b paraphrase **0.444 vs 0.447**, nopersona **0.423 vs 0.447**. Per-capita crime β stays positive and significant in all four signal-carrier arms (0.12–0.20, p<0.001; production-prefix 0.119–0.136). Perturbed-vs-production tract scores correlate 0.91–0.94. The qwen3.5-2b negative control stays at noise (r −0.02/0.06 vs 0.08). Report: `outputs/reviewer2_subway/outcome_20260711_064701/REPORT.md`; W&B run `63iok48y`.

**P0 freebies (no GPU, from the production 100k parquets; `--tier freebies`, run 2026-07-10):** the income signal is order-robust — per-half r(income) unswapped/swapped: qwen3.5-9b **0.512/0.523** (between-half tract-mu r=0.938), gemma-4-12b **0.453/0.450** (0.941), gemma-4-e4b 0.359/0.370 (0.892); small models stay at noise in both halves (qwen3.5-2b 0.051/0.083, half-correlation ≈0). Repeat self-consistency (temp-0.6 ceiling, ~10k repeat pairs): weighted kappa qwen3.5-9b 0.324, gemma-4-12b 0.386, e4b 0.189; small models ≈0 (flip rate ≈ 0.5 = coin-flip).

**Small-model ladder (fresh same-seed re-run, 2026-07-12; `reviewer2_smallladder_klara1x.yaml`, 29/29 probes, all exact ordered prefixes).** The Case-3 battery re-run across three small raters — gemma-4-e2b, qwen3.5-2b, qwen3.5-9b — in one same-day sweep for clean provenance. qwen3.5-2b/9b **reproduce the 2026-07-10 numbers within temp-0.6 noise** (e.g. qwen-9b flipped κ −0.051 vs −0.040, textfirst 0.093 vs 0.107; qwen-2b paraphrase 0.447 vs 0.448, interleaved 0.610 vs 0.637), so the fresh table is directly comparable. Linear-weighted kappa vs each model's own A0 anchor (`agreement_vsA0_smallladder_20260712_131444`, W&B `txmv877e`):

| arm | gemma-4-e2b | qwen3.5-2b | qwen3.5-9b |
|---|--:|--:|--:|
| temp0 (bound) | 0.709 | 0.722 | 0.652 |
| enumrev | 0.584 | 0.240 | 0.550 |
| nopersona | 0.485 | 0.579 | 0.544 |
| paraphrase | **0.058** | 0.447 | 0.518 |
| res-half | n/a | 0.498 | 0.501 |
| interleaved | **0.198** | 0.610 | 0.548 |
| freeform | **0.161** | 0.241 | 0.334 |
| flipped (negated) | **0.108** | **0.180** | **−0.051** |
| textfirst | **−0.003** | 0.246 | **0.093** |

Two takeaways. **(1) The `flipped` and `textfirst` weaknesses are tier-wide, not a qwen-9b quirk.** All three small raters fail to answer the negated "which looks *less* safe" as the inverse (κ ≤ 0.18 for every one; qwen-9b actively anti-correlates at −0.051 with Same jumping to 41%) — acquiescence bias is universal in the small tier. Question-before-images (`textfirst`) breaks A/B binding for e2b (−0.003) and qwen-9b (0.093). gemma-4-12b, by contrast, survives both (0.542 / 0.757) — robustness to *framing/presentation* is the capability-ladder axis, and only the large unified rater clears it. **(2) Robustness to wording/resolution climbs monotonically with capability.** qwen-9b holds ~0.50–0.55 on every non-pathological arm; qwen-2b holds on paraphrase/nopersona/res-half/interleaved but the enum-order scaffold matters (enumrev collapses it to 0.240); gemma-4-e2b survives only enum-order (0.584) and persona-removal (0.485) — a full paraphrase (0.058) or any layout change collapses it, and without guided decoding it dissolves into abstention (`freeform` NotSure 0.55, κ 0.16). e2b's temp0 bound (0.709) is as high as the qwen models', so its fragility is genuine perturbation sensitivity, not just noise. Runs attribute via `.hydra/overrides.yaml` under `multirun/2026-07-12_URBANPAIRVQA/` (arm-group timestamps 12-20-19/26/34/42/50, plus 13-07-15 = qwen-2b interleaved retry after a transient SSH-reset job death).

## Self-Explanation Experiments (Case 3 + restaurants/schools)

A second line of Reviewer-2 work asks not whether a rater is *consistent* but whether the **concept behind its judgments is legible in its own outputs**. Two linked experiments, both on gemma-4-12b:

| Experiment | Question | Answer | Page |
|---|---|---|---|
| **GEPA self-distillation ladder** | Can the model re-derive its own case prompt from its own behavior? | **No.** Across four channels (labels → factual captions → contrastive captions → the pixels themselves) it recovers only visual proxies — subway safety → *"brightly lit"*, restaurant appeal → *"commercial activity"* — and always plateaus 0.05–0.09 ordinal below the true-axis ceiling. The evaluative concept is **never named**. | [[concept-self-explanation-ladder]] |
| **Activation probe** | Is that a *representation* failure or an *articulation* failure? | **Articulation.** A linear probe on the model's hidden states recovers the safety judgment from a *brightness*-prompted (0.886) or even *blind* (0.877) forward pass — well above what the model actually says when asked (0.820). The information was there the whole time. | [[concept-activation-probe]] |

Together: **a "safe" rating is really a brightness-and-openness rating, and the model cannot articulate the substitution even while looking at the photographs it judged** — but it does represent it. Harness: `dagspaces/urbanpairvqa/prompt_opt/gepa_pairwise.py`; probe scripts in `outputs/_actprobe/`.

## Downstream Aggregation

Pair outputs are a match table, not a ranking. To turn the per-pair `relative_score` column into a sorted list of entities ("libraries from most to least maintained"), feed it through **[[concept-trueskill]]**. The `scripts/pairwise_vqa_report.py` utility wraps the end-to-end analysis (label stats, reasoning word cloud, TrueSkill ranking) into a markdown report.

## Inspecting Reasoning Traces

When a run uses a thinking model (e.g. `qwen3.5-9b/instruct_thinking`, with `sampling_params_vqa.max_tokens` raised to ~8192 so the trace isn't truncated), the parsed `<think>` block is persisted to the **`model_reasoning`** column (see [[vllm-inference]] for the reasoning parser). Two scripts pull reproducible random samples for qualitative review:

| Script | Output | Use |
|--------|--------|-----|
| `scripts/sample_reasoning_traces.py` | CSV | ID columns + raw `model_reasoning` text for spreadsheet/grep review |
| `scripts/sample_reasoning_pdfs.py` | PDF (one per pair; `--combined` for a merged file) | Visual review: the two images **in presented order** alongside the verdict and the full paginated trace |

Both default to sampling only rows with a non-empty trace (a thinking model occasionally spends its whole token budget mid-trace and emits no final JSON, leaving `model_reasoning` populated but `model_response`/`answer` blank). Key `sample_reasoning_pdfs.py` flags: `--decisive-only` (skip `Same` verdicts), `--include-empty`, `--combined`, `--title`/`--question` (header context), `--seed`. It renders left = "Image A" (`presented_left_path`), right = "Image B" (`presented_right_path`) so the images line up with how the trace refers to them, and uses matplotlib `PdfPages` + PIL (no extra deps).

```bash
python scripts/sample_reasoning_pdfs.py \
    multirun/.../outputs/pairwise/restaurants_mvp_*.parquet \
    -n 12 --decisive-only --combined \
    --title "Restaurants — Qwen3.5-9B thinking" \
    --out reports/pairwise/restaurants_reasoning_pdfs
```

## Related Pages

- [[architecture]] -- overall pipeline architecture
- [[concept-counterbalancing]] -- counterbalancing methodology and reliability
- [[concept-trueskill]] -- aggregating pair outputs into a ranked list
- [[concept-facing-filter]] -- upstream per-unit facing filter; source of `attribution_confidence`
- [[concept-self-explanation-ladder]] -- GEPA self-distillation: can the model re-derive its own question?
- [[concept-activation-probe]] -- is the judgment represented even though it cannot be said?
- [[dohmh-restaurants-curation]] -- upstream curation for the restaurants MVP / sweep
- [[facdb-curation]] -- upstream curation for the libraries MVP / sweep
- [[urban-vqa]] -- core VQA dagspace (pairwise reuses `run_vqa_stage`)
- [[urban-roam-vqa]] -- street traversal dagspace (shares image stitching pattern)
- [[urban-ocr]] -- OCR dagspace
- [[urban-embed]] -- embedding dagspace
- [[shared-infrastructure]] -- common modules
