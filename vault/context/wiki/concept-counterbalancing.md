---
title: "Counterbalancing in Pairwise Comparison"
category: concept
created: 2026-04-06
updated: 2026-06-08
tags:
  - concept
  - pairwise
  - counterbalancing
  - methodology
---

# Counterbalancing in Pairwise Comparison

How UrbanPairVQA handles presentation order bias in side-by-side image comparisons.

## Problem

Vision-language models can exhibit positional bias -- they may systematically prefer the left-side or first-presented image in a pairwise comparison, regardless of actual content. This introduces systematic error into relative judgments (e.g., "which neighborhood looks wealthier?").

## Solution

The framework implements counterbalancing with order tracking. Each image pair can be presented in its canonical order or swapped, and the swap state is recorded so that final labels can be corrected to reflect the true canonical ordering.

## Counterbalancing Modes

Set via the `counterbalance_mode` parameter in the pair sampler:

| Mode | Behavior |
|------|----------|
| `none` | Canonical order only. Image A is always presented on the left. |
| `random` | Each presentation is independently swapped with 50% probability (`rng.integers(0, 2)`). |
| `balanced` | **Marginal** 50/50 split: swap is assigned by observation-index parity (`swap = obs_idx % 2`), so ~half of *all* presentations across the run are swapped. This balances the **population**, not each individual pair — see below. |

The mode is normalized by `_normalize_counterbalance_mode()` in `dagspaces/urbanpairvqa/samplers/cyclomedia_pairs.py`.

> **Correction (2026-06-08):** `balanced` does **not** show "each unique pair in both orderings" (as an earlier version of this page claimed). It only guarantees the *marginal* order distribution is ~50/50. A given `canonical_pair_id` is still presented in exactly one order; the `obs_idx % 2` assignment makes no attempt to pair a canonical pair with its own mirror, and even `repeat_fraction` repeats are not guaranteed to land in the opposite order.

## Population vs per-pair counterbalancing

There are **two distinct levels** of counterbalancing, and they buy different things. `balanced` mode provides only the first.

| Level | What it does | Achieved by | What it de-biases |
|-------|--------------|-------------|-------------------|
| **Population (aggregate)** | ~50% of all presentations swapped across the run | `counterbalance_mode: balanced` (covers 100% of rows) | The **aggregate** distribution / mean / a global ranking — position bias cancels exactly across the dataset |
| **Per-pair (full)** | *Every* canonical pair evaluated in **both** orders, then the two `relative_score`s averaged | Not implemented yet — would need a "both-orders" sampler mode (emit each base pair twice with `swap ∈ {False, True}`) | Each **individual pair's** score — position bias cancels *within* the pair, not just in aggregate. 2× the inference cost. |

**When each is enough:**
- **Population conclusions** (overall distribution, mean comparison, TrueSkill ranking over many units): population counterbalancing is sufficient. The position bias is removed *structurally* by the 50/50 swap regardless of sample size.
- **Valid per-pair verdicts** (cleanest per-comparison deltas, publishable individual numbers): use full counterbalancing.

### Worked example (qwen3.5-9B thinking, ~1,000 pairs each, 2026-06-07)

Restaurants and schools "ordinal" runs, `counterbalance_mode: balanced`, `repeat_fraction: 0.10`:

| Quantity | Restaurants | Schools | Reading |
|----------|-------------|---------|---------|
| `presented_score` mean | **−0.495** | **−0.480** | Raw position bias: ~half a label, model favors the **left** image |
| `relative_score` mean (de-swapped) | **+0.025** | **−0.055** | After counterbalancing → essentially zero |
| 95% CI on de-biased mean | [−0.06, +0.11] | [−0.14, +0.03] | **Both include 0** — no residual directional bias |
| SE (n≈1,100, sd≈1.46) | 0.045 | 0.044 | Population mean pinned to within ±0.09 |
| Repeat sign-agreement | 71% | 66% | ~1 in 3 order-flips reverse the preference (the −0.49 bias acting per-pair) |

**Takeaways:**
1. The −0.49 → ~0 collapse *is* the counterbalancing working. It is driven by the **100%** swap coverage, not the 10% repeats.
2. `repeat_fraction` is a **reliability probe** (test-retest / order-sensitivity), **not** a de-biasing mechanism — repeats barely move an SE that is already 0.045.
3. n≈1,000 gives a tight population estimate (SE ≈ 0.045). Finer slices (per-neighborhood, per-unit, tails) have smaller effective n, so the lever there is **more distinct pairs / comparisons per unit**, not more repeats or counterbalancing.

## Data Flow

### Pair Construction

The `build_global_random_pairs()` function in `dagspaces/urbanpairvqa/samplers/cyclomedia_pairs.py` constructs pair rows with these columns:

| Column | Description |
|--------|-------------|
| `pair_id` | Unique identifier for this specific presentation |
| `canonical_pair_id` | Stable identifier for the underlying pair (independent of swap) |
| `sample_id_a` | Original sample ID for image A |
| `sample_id_b` | Original sample ID for image B |
| `image_path_a` | File path to image A (canonical left) |
| `image_path_b` | File path to image B (canonical right) |
| `presented_left_path` | File path of the image actually shown on the left |
| `presented_right_path` | File path of the image actually shown on the right |
| `presented_order` | `"A_then_B"` (canonical) or `"B_then_A"` (swapped) |
| `is_swapped` | `True` if the images were flipped from canonical order |
| `repeat_idx` | Repetition index for inter-rater reliability |

The `_build_pair_row()` helper constructs each row. When `swap_presented=True`, the left/right paths are reversed and `is_swapped` is set to `True`.

### Label Derivation

After inference, `_derive_labels()` in `dagspaces/urbanpairvqa/stages/pairwise_vqa.py` computes final labels that account for swap state:

1. **`presented_answer`** -- raw model output (refers to the presented ordering)
2. **`presented_label`** -- canonicalized version of the presented answer (e.g., "left", "right", "equal")
3. **`presented_score`** -- ordinal score for the presented label
4. **`relative_label`** -- the corrected label relative to canonical A/B ordering. For swapped pairs, the label is inverted (e.g., "left" becomes "right")
5. **`relative_score`** -- ordinal score for the relative label

The inversion ensures that `relative_label` always refers to the canonical pair ordering (A vs B), regardless of how images were actually presented to the model.

## Inter-Rater Reliability

The pair sampler supports repeated measurements for reliability estimation:

| Parameter | Description |
|-----------|-------------|
| `repeat_count` | Number of exact repeat presentations for a subset of pairs |
| `repeat_fraction` | Fraction of pairs to repeat (e.g., 0.1 = 10%) |

Repeated pairs get distinct `pair_id` values but share the same `canonical_pair_id`, enabling agreement analysis across presentations.

## Diagnostics

Post-inference analysis can compute:

- **Entropy** of label distributions across counterbalanced presentations -- high entropy suggests the model is sensitive to order
- **Agreement metrics** between original and swapped presentations of the same pair -- low agreement indicates positional bias
- **Consistency across repeats** -- measures how often the model gives the same answer for the same pair presented multiple times

## Key Files

| File | Role |
|------|------|
| `dagspaces/urbanpairvqa/samplers/cyclomedia_pairs.py` | Pair construction with counterbalancing and order tracking |
| `dagspaces/urbanpairvqa/stages/pairwise_vqa.py` | `_derive_labels()` for correcting labels post-inference |
| `dagspaces/urbanpairvqa/conf/pipeline/*.yaml` | Pipeline configs specifying counterbalance_mode |

## See Also

- [[urban-pair-vqa]] -- the UrbanPairVQA dagspace overview
