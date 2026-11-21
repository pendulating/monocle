# F11: judgment consistency, pairwise against single-image

This is the record of F11 (`facct-runbook.md`, F11, D150, D151). The test
asks if a model gives a visual cue the same valence when it compares 2
images (the pairwise battery) and when it judges 1 image (F9).

**Status (2026-10-30):** done with R4 shard 0 (job 106921, 250 traces of
each cell). **H11 fails in 11 of 12 cells** (section 4). Shards 1 to 3 of
R4 are optional; they make the R4 interval narrower.

## 1. The data

| Arm | Texts | Extractor and schema | Spans, median words |
|-----|-------|----------------------|---------------------|
| pairwise | the thinking traces of the canonical trace runs; 10,000 pairs; 1 text = 1 side of 1 pair | cross-family: Qwen3.8-27B reads gemma-4-12b, Gemma-4-31B reads qwen3.5-9b; `urban_cues/v2` | 4 to 5 |
| F9 | the `reasoning` field (D151, Q124); 10,000 images | DeepSeek-V4-Flash; `single-subj-v1`, `street-subj-v1` | 8 to 10 |
| R1 | the same pairwise traces, 2,000 pairs | the other extractor of the same schema | 3 to 6 |
| R2 | the F4 `reasoning` field on the same images, about 1,000 images | DeepSeek-V4-Flash, the F9 schema | 8 to 9 |
| R3 | the 1,000 pairwise repeats against their first answers | the primary extractor | 4 to 6 |
| R4 | 250 pairwise traces of each cell (shard 0 of 4), inside the R1 sample | DeepSeek-V4-Flash, `urban_cues/v2` | 4 to 5 |

Step 0 checks:

| Check | Result |
|-------|--------|
| The text that each extraction read | both read the thinking trace: `model_reasoning` (pairwise) and `reasoning` (F9) |
| Sampling | both runs: temperature 1.0, top_p 0.95, thinking on. Gemma: top_k 64 in both. Qwen: the F9 run adds top_k 20 and presence_penalty 1.5; the pairwise run has neither |
| The canonical gate | the 4 pairwise corpora match the canonical registry (`_extractions.registry_mismatch` is empty) |
| The crosswalk (Q125) | the pairwise `people` holds crowds and street life (`lot of activity`), as the F9 `activity` does. The pairwise `other` holds colours and murals, as the F9 `color` does. The crosswalk stays |
| The valence values | `good`, `bad`, `neutral` in both schemas |

**Warning:** the 2 paradigms show different images. The restaurant arms
share 4,954 restaurants but not the view; the street arms share 34
recordings. Thus U1 to U4 compare distributions, not the same images.

## 2. The method

The code is `notebooks/facct/_paradigm_kl.py` (1.3.0). The notebook is
`notebooks/facct/f11_paradigms.py`. The output is in `data/f11/`.

| Item | Value |
|------|-------|
| Units | U1 P(category); U2 P(valence \| category); U3 P(valence \| cue); U4 P(word \| valence) |
| Cue of U3 | 1 content word, or 2 adjacent content words, of 1 span. 25 texts or more in each paradigm |
| Count | 1 text counts 1 key 1 time |
| Divergence | Jensen-Shannon (JS, bits), and KL in both directions. Add 0.5 to each cell. U2 and U3: the mean over the keys, weighted by the mean share of the key |
| Null | a permutation of the arm labels, 200 times: over the pooled clusters (unpaired), or a swap inside each cluster (paired: R1, R2, R3, R4) |
| Point value | `js_excess` = JS less the mean of the null |
| Interval | 1,000 bootstrap draws of the clusters (the pair or the image); the 2.5% and 97.5% distances from the median, put around the point |
| Order | Spearman rho of the net valence (good - bad) / texts of the U3 cues. The R0 column gives the lower rho of the 2 arms |
| Level | `raked`: the valence margins of the 2 arms are set equal first |

Two changes from the plan:

1. **The null.** A JS of 2 samples is larger than 0 when the 2
   distributions are equal, and the bias grows when the sample is small. The
   anchors R1 to R3 have about 1,000 to 1,800 clusters, the paradigm arms
   have about 10,000. Without the null, a small anchor looks larger than it
   is. R0 (the split-half) stays in the table, but the null replaces it as
   the noise floor.
2. **The negation.** The F9 clauses often deny a standard: `the brick isn't
   pristine`, `it doesn't look dingy`. The first cue unit dropped the `not`,
   and `pristine` then had a bad valence in F9 and a good one in the pairwise
   traces. The unit now keeps `not pristine` as its own cue. It uses the
   rules of `_cues_across.normalise_cue` without the predicate swap
   (`X is Y` to `Y X`), because the swap moves words across the negator. The
   share of spans with a negator is 2.8% to 5.6% in F9 and 0.5% to 1.6% in
   the pairwise traces.

## 3. The results

### 3.1 The divergence (U1 to U3)

`js_excess` with the 95% interval. Unraked.

| Model | Case | Unit | Paradigm | R1 extractor | R2 prompt (F4 to F9) | R3 retest |
|-------|------|------|----------|--------------|----------------------|-----------|
| gemma-4-12b | restaurant | U1 | 0.037 [0.035, 0.038] | 0.011 [0.010, 0.013] | 0.028 [0.023, 0.033] | 0.000 |
| gemma-4-12b | restaurant | U2 | 0.012 [0.011, 0.013] | 0.003 [0.002, 0.004] | 0.074 [0.066, 0.084] | 0.000 |
| gemma-4-12b | restaurant | U3 | 0.022 [0.021, 0.023] | 0.007 [0.005, 0.008] | 0.055 [0.049, 0.061] | 0.000 |
| gemma-4-12b | street | U1 | 0.043 [0.042, 0.045] | 0.006 [0.005, 0.007] | 0.018 [0.013, 0.024] | 0.000 |
| gemma-4-12b | street | U2 | 0.062 [0.060, 0.064] | 0.002 [0.001, 0.003] | 0.148 [0.134, 0.162] | 0.000 |
| gemma-4-12b | street | U3 | 0.052 [0.051, 0.053] | 0.003 [0.003, 0.004] | 0.129 [0.118, 0.141] | 0.001 |
| qwen3.5-9b | restaurant | U1 | 0.025 [0.024, 0.026] | 0.009 [0.008, 0.011] | 0.050 [0.044, 0.055] | 0.000 |
| qwen3.5-9b | restaurant | U2 | 0.017 [0.016, 0.018] | 0.001 [0.001, 0.002] | 0.106 [0.097, 0.116] | 0.000 |
| qwen3.5-9b | restaurant | U3 | 0.029 [0.028, 0.030] | 0.004 [0.004, 0.005] | 0.081 [0.075, 0.088] | 0.000 |
| qwen3.5-9b | street | U1 | 0.057 [0.055, 0.059] | 0.010 [0.008, 0.011] | 0.027 [0.022, 0.033] | 0.000 |
| qwen3.5-9b | street | U2 | 0.046 [0.044, 0.048] | 0.006 [0.005, 0.007] | 0.071 [0.063, 0.081] | 0.000 |
| qwen3.5-9b | street | U3 | 0.036 [0.035, 0.037] | 0.006 [0.005, 0.007] | 0.073 [0.066, 0.080] | 0.000 |

| Finding | Evidence |
|---------|----------|
| The paradigm divergence is 3 to 30 times the extractor divergence R1 | all 12 cells; no interval overlaps |
| The retest divergence is 0 | R3 is inside the null in all 12 cells: 1 model, 1 pair, 2 samples give the same distribution |
| An unraked paradigm divergence of U2 and U3 is smaller than the prompt change R2 | all 8 cells; F4 is a description, thus nearly all its spans are `neutral` |
| A raked paradigm divergence of U3 is equal to or larger than the raked R2 | paradigm 0.020, 0.043, 0.021, 0.038; R2 0.015, 0.020, 0.018, 0.015 |
| The street case diverges most | F9 gives the street a bad valence 2 times as often as the pairwise traces (gemma 24% against 10%) |

### 3.2 The order of the cues

Spearman rho of the net valence of the U3 cues, with the 95% interval.

| Model | Case | Cues | Paradigm | R1 extractor | R2 prompt | R3 retest | R0 split-half |
|-------|------|------|----------|--------------|-----------|-----------|---------------|
| gemma-4-12b | restaurant | 1,085 | 0.82 [0.80, 0.83] | 0.90 | 0.52 | 0.78 | 0.90 |
| gemma-4-12b | street | 1,288 | 0.80 [0.79, 0.81] | 0.91 | 0.55 | 0.85 | 0.96 |
| qwen3.5-9b | restaurant | 1,644 | 0.87 [0.86, 0.88] | 0.90 | 0.53 | 0.80 | 0.92 |
| qwen3.5-9b | street | 1,620 | 0.83 [0.82, 0.84] | 0.92 | 0.54 | 0.83 | 0.96 |

**Caution:** a rho depends on the sample size. R3 has about 900 pairs,
thus its rho is low for a sample reason. R1 (about 1,800 pairs) is the
fair comparison for the paradigm rho.

The order of the cues holds across the paradigms: the paradigm rho is
near the extractor rho and far above the prompt change. The words that
move most are comparatives and photographic words. A pairwise trace uses
`darker` and `cleaner` to rank 2 images. A gemma F9 street trace uses
`dramatic`, `scenic`, `landmark` and `stark` for a bad valence, and the
pairwise trace uses them for a good one (39 and 41 sign changes in the 2
street cells, 2 and 9 in the restaurant cells).

### 3.3 The same restaurant (U5, no KL)

The trace run is too sparse: only 31 restaurants have 5 comparisons or
more. Thus the pairwise score is the 1,000,000-pair battery (no thinking).

| Model | x | y | Restaurants | rho [95%] |
|-------|---|---|-------------|-----------|
| gemma-4-12b | battery score | F9 verdict (yes 1, no -1, mixed or graded 0) | 6,825 | 0.035 [0.013, 0.060] |
| gemma-4-12b | battery score | F9 net valence | 6,818 | 0.225 [0.204, 0.248] |
| gemma-4-12b | pairwise net valence | F9 net valence | 4,462 | 0.072 [0.040, 0.103] |
| qwen3.5-9b | battery score | F9 verdict | 7,415 | 0.080 [0.057, 0.103] |
| qwen3.5-9b | battery score | F9 net valence | 7,415 | 0.177 [0.158, 0.198] |
| qwen3.5-9b | pairwise net valence | F9 net valence | 4,706 | 0.040 [0.010, 0.068] |

The F9 verdict is the LAST `verdict` span of the `response` field.
**Caution:** the 2 paradigms show different views of the same restaurant.
A low rho can come from the view, not from the standard.

## 4. R4 and the H11 verdict

R4 reads 237 to 250 pairs of each cell. DeepSeek-V4-Flash writes about 28
spans for each trace, but the span length (4 to 5 words) and the valence
shares are near the primary extractor. `js_excess` (bits), with the 95%
interval:

| Model | Case | Unit | Paradigm | R4 | R1 | Overlap with R4 |
|-------|------|------|----------|----|----|-----------------|
| gemma | restaurant | U1 | 0.037 [0.035, 0.038] | 0.023 [0.016, 0.031] | 0.011 | no |
| gemma | restaurant | U2 | 0.012 [0.011, 0.013] | 0.006 [0.002, 0.011] | 0.003 | **yes** |
| gemma | restaurant | U3 | 0.022 [0.021, 0.023] | 0.005 [0.001, 0.010] | 0.007 | no |
| gemma | street | U1 | 0.043 [0.042, 0.045] | 0.001 [0.000, 0.003] | 0.006 | no |
| gemma | street | U2 | 0.062 [0.060, 0.065] | 0.003 [0.001, 0.006] | 0.002 | no |
| gemma | street | U3 | 0.052 [0.051, 0.053] | 0.004 [0.001, 0.008] | 0.003 | no |
| qwen | restaurant | U1 | 0.025 [0.024, 0.026] | 0.013 [0.009, 0.018] | 0.009 | no |
| qwen | restaurant | U2 | 0.017 [0.016, 0.018] | 0.004 [0.001, 0.008] | 0.001 | no |
| qwen | restaurant | U3 | 0.029 [0.028, 0.030] | 0.002 [-0.001, 0.006] | 0.004 | no |
| qwen | street | U1 | 0.057 [0.055, 0.059] | 0.009 [0.006, 0.013] | 0.010 | no |
| qwen | street | U2 | 0.046 [0.044, 0.048] | 0.008 [0.004, 0.012] | 0.006 | no |
| qwen | street | U3 | 0.036 [0.035, 0.037] | 0.004 [0.002, 0.008] | 0.006 | no |

After the raking, the paradigm U3 is 5 to 25 times R4, and no interval
overlaps (paradigm 0.020 to 0.043, R4 0.002 to 0.004). The U3 rho of R4 is
0.75 to 0.83, lower than R1 (0.90 to 0.92) because R4 has fewer texts.

## 5. The reading

| Claim | Status |
|-------|--------|
| The 2 paradigms rank the cues in nearly the same order | supported: rho 0.80 to 0.87; R1 0.90 to 0.92, R4 0.75 to 0.83 |
| The 2 paradigms give the same distribution of valence (H11) | **rejected** in 11 of 12 cells. The only overlap is gemma restaurant U2, at the upper edge of a wide R4 interval |
| The F9 extractor causes the gap | no: R4 is about the size of R1. A change to DeepSeek-V4-Flash moves the pairwise spans much less than the change of paradigm |
| The gap is smaller than a change of prompt | yes for U2 and U3: R2 (F4 against F9) is 2 to 6 times the paradigm. U1 is mixed |
| The street case diverges most | yes: F9 gemma gives `bad` to 24% of the street spans, the pairwise traces 10% |
| The 2 paradigms rank the same restaurants the same | weak: rho 0.04 to 0.23, on different views |

Thus the models keep the order of the cues across the paradigms, but not the
level of the valence. The difference is a property of the paradigm, not of
the extractor. It is smaller than the change from description to judgment.

## 6. The files

| File | Content |
|------|---------|
| `data/f11/f11_divergence.parquet` | 1 row for each (model, case, unit, comparison, raked) |
| `data/f11/f11_cues.parquet` | the U3 cues of each cell: counts, net valence, JS, `sign_flip` |
| `data/f11/f11_arms.parquet` | the texts, spans, words and shares of each arm |
| `data/f11/f11_places_{gemma12b,qwen9b}.parquet` | 1 row for each F9 restaurant |
| `data/f4_traces/extract_input/f4_f11pw.parquet` | the R4 input: 4,000 traces (`scripts/f4/build_f11_input.py`) |
| `multirun/f4_f11_pw_cues_dsv4/` | the R4 spans (`extract.sbatch`, arm `f11_pw_cues`) |
| `notebooks/facct/figures/f11/` | `f11_divergence`, `f11_cues`, `f11_places` |

To run it again:

```bash
.venv-vllm025cu129/bin/python notebooks/facct/_paradigm_kl.py      # about 10 min
.venv-vllm025cu129/bin/python notebooks/facct/f11_paradigms.py --write
```

To run R4 (1 shard is 250 traces of each cell, about 20 min on 1 B300):

```bash
sbatch --array=0-3%1 --export=ALL,F4X_ARM=f11_pw_cues,F4X_CASE=f11pw,F4X_SHARDS=4 scripts/f4/extract.sbatch
```
