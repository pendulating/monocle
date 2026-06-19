---
title: "UrbanPairVQA — Pairwise Comparison"
category: dagspace
created: 2026-04-06
updated: 2026-06-05
tags:
  - dagspace
  - pairwise
  - ordinal
  - comparison
  - counterbalancing
  - sampling
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

## Diagnostics

The orchestrator computes pairwise diagnostics for quality monitoring:

- Label distribution across the 5-point scale
- Entropy of label distribution (higher = more diverse judgments)
- Agreement metrics between repeated pairs
- Position bias detection (forward vs. reversed presentation)

## Configuration

### Prompt Configs (`dagspaces/urbanpairvqa/conf/prompt/`)

| Config | Domain |
|--------|--------|
| `pairwise_livability_ordinal.yaml` | Livability comparison (5-point) |
| `pairwise_wealth_ordinal.yaml` | Wealth comparison (5-point) |
| `pairwise_wealth_yes_no.yaml` | Wealth comparison (binary yes/no) |
| `pairwise_parking_ordinal.yaml` | Parking availability comparison |
| `pairwise_relative_ordinal.yaml` | Generic relative comparison |
| `pairwise_library_maintained_ordinal.yaml` | Library facade-maintenance comparison |
| `pairwise_restaurant_eat_at_ordinal.yaml` | Restaurant exterior "which would you rather eat at" |
| `pairwise_school_send_child_ordinal.yaml` | K-12 school exterior "which would you rather send your child to" — open-ended "based on appearance" framing (no exterior-cue constraint, unlike libraries/restaurants) |
| `pairwise_sterility_ordinal.yaml` | Visual monotony/sterility comparison (5-point); judges only on observable cues (repetitive forms, blank facades, empty sidewalks, etc.) |
| `pairwise_.yaml` | Base pairwise prompt template |

### Pipeline Configs (`dagspaces/urbanpairvqa/conf/pipeline/`)

| Config | Description |
|--------|-------------|
| `pairwise_cyclomedia_ordinal.yaml` | Standard ordinal comparison |
| `pairwise_cyclomedia_livability_large.yaml` | Large-scale livability comparison |
| `pairwise_cyclomedia_wealth_large.yaml` | Large-scale wealth comparison |
| `pairwise_cyclomedia_wealth_midsize.yaml` | Mid-size wealth comparison |
| `pairwise_cyclomedia_wealth_tester_256.yaml` | Small wealth test run |
| `pairwise_cyclomedia_sterility_large.yaml` | Large-scale visual-monotony/sterility comparison (random image pairs from `manhattan_2025_1k`, Qwen3.5-9B with thinking enabled via `qwen3.5-9b/instruct_thinking`; `sampling_params_vqa.max_tokens=6144` to fit the reasoning trace) |
| `pairwise_cyclomedia_parking_tester_*.yaml` | Parking detection test runs |
| `pairwise_libraries_mvp.yaml` | Library-level MVP (Gemma-4-E4B default; sweep via `+sweep=libraries_all_models`) |
| `pairwise_restaurants_mvp.yaml` | Restaurant-level MVP (Qwen3.5-4B default; ≤4B sweep via `+sweep=restaurants_all_models_4b`) |
| `pairwise_schools_mvp.yaml` | K-12 schools MVP (Qwen3.5-4B default; `max_pairs=100,000`, `allow_replacement=false`; sweep via `+sweep=schools_all_models`) |

### Data Config

| Config | Description |
|--------|-------------|
| `conf/data/cyclomedia_pairwise_manhattan_2025_1.yaml` | Manhattan 2025 Cyclomedia pairwise manifest |
| `common/conf/data/cyclomedia_near_libraries_facing.yaml` | NYC public-library facing manifest (~11.7k rows / 236 units) |
| `common/conf/data/cyclomedia_near_restaurants_facing.yaml` | DOHMH restaurant facing manifest (~490.8k rows / 18,488 camis) |
| `common/conf/data/cyclomedia_near_schools_facing.yaml` | FacDB K-12 schools facing manifest (130.7k rows / 2,287 units, of 3,103 publishable) |

### Sweep Configs (`dagspaces/urbanpairvqa/conf/sweep/`)

| Config | Lineup | Notes |
|--------|--------|-------|
| `libraries_all_models.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | All natively-multimodal instruct models |
| `restaurants_all_models.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | Full mirror of `libraries_all_models.yaml` — no size cap |
| `restaurants_all_models_4b.yaml` | gemma-4-e2b, qwen3.5-2b, qwen3.5-4b | Strict ≤4B subset; drops gemma-4-e4b (~4B effective / ~8B total), phi-4-mm (~5.6B), qwen3.5-9b |
| `schools_all_models.yaml` | gemma-4-e2b, gemma-4-e4b, phi-4-mm, qwen3.5-{2,4,9}b | Same lineup as restaurants/libraries; pairs with `pairwise_schools_mvp.yaml` |

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
- [[dohmh-restaurants-curation]] -- upstream curation for the restaurants MVP / sweep
- [[facdb-curation]] -- upstream curation for the libraries MVP / sweep
- [[urban-vqa]] -- core VQA dagspace (pairwise reuses `run_vqa_stage`)
- [[urban-roam-vqa]] -- street traversal dagspace (shares image stitching pattern)
- [[urban-ocr]] -- OCR dagspace
- [[urban-embed]] -- embedding dagspace
- [[shared-infrastructure]] -- common modules
