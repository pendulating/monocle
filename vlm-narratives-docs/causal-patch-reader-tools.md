# The fitted ablation predictor: the tools

**Date:** 2026-08-25. **Model:** gemma-4-12B-it. **Cases:** subway_safety,
libraries, restaurants.

This page lists the modules and the scripts that build and test the 1-token
fitted ablation predictor. The science is in
[causal-lens-retrospective.md](causal-lens-retrospective.md).

## What the predictor is

The predictor takes 1 backward pass over a single-image prompt. It scores each of
the 256 image tokens at 9 layers. The score is the dot product of the token's
gradient and its hidden state (gradient-times-input attribution). The predictor
then takes the mean rank over the 9 layers. The strongest rank names the token
that supports the judgment. The weakest names the token that opposes it.

The fitted form weights the 3840 dimensions before the sum. See the table at
the end of this page.

The test compares those 2 tokens against the measured token-ablation effect of
each token, and it repeats the same readout under another case's question.

## The GPU modules

| Module | What it measures |
|---|---|
| `monocle/causal_maps.py` | the token-ablation map: 256 single-token ablations of each photograph |
| `monocle/input_gate.py` | every attribution method at 9 layers, against a stored map |
| `monocle/residual_ig.py` | integrated gradients over the residual stream |
| `monocle/transfer_k1.py` | transfers 1 image token from photograph A into photograph B |
| `monocle/truth_retest.py` | measures the map a second time and compares |
| `monocle/patch_caption.py` | crops the image patch of the top ablation-effect token and captions it |
| `monocle/causal_words.py` | the logit-lens words at the top and bottom tokens of the map |
| `monocle/self_report.py` | asks the model for the reason of its answer, then crops the predictor's image token |

### New flags

`causal_maps.py` and `input_gate.py` both take an answer-word override and a
question override. A change to either makes the logit difference `c` a different
quantity, so a new token-ablation map is necessary.

| Flag | Module | Effect |
|---|---|---|
| `--labels POSITIVE NEGATIVE` | both | replaces the 2 answer words in the prompt and in the readout |
| `--question TEXT` | `causal_maps.py` | replaces the case question |
| `--paraphrase TEXT` | `input_gate.py`, `residual_ig.py` | replaces the case question |
| `--prompt-case CASE` | `input_gate.py` | asks another case's question about the same photographs |
| `--truth-from PATH` | `input_gate.py` | scores against a stored map instead of a fresh one |
| `--reference substitute` | `causal_maps.py` | substitutes other pixels into the image patch instead of a deletion |
| `--save-acts LAYERS` | `input_gate.py` | keeps `h` and `g` at each image token, for the fitted predictor |

**Warning:** a stage job may read a stale copy of a module over NFS. The
attribute cache holds the old file for about a minute. Wait a minute after
you edit a module, or the job runs the old code and writes an output that
looks complete.

**Warning:** match `--seed` to the run that measured the stored map, or every
photograph drops out and the run stops.

**Warning:** match the measure batch size. The map is bit-exact at 1 batch size
and it changes at another. `map_stats.parquet` records `measure_batch`.
`causal_maps60` used 12.

## The CPU scripts

| Script | What it reports |
|---|---|
| `scripts/k1_separation.py` | the 1-token separation at 1 layer, with the controls |
| `scripts/k1_layer_profile.py` | the same by layer, and `--pool` for the mean rank |
| `scripts/k1_position_control.py` | the separation with the position prior removed |
| `scripts/k1_rank_agreement.py` | the agreement over all 256 image tokens, not only the top and bottom tokens |
| `scripts/k1_wording_ensemble.py` | the predictor averaged over several wordings |
| `scripts/k1_confidence.py` | 3 scores that estimate the reliability of a prediction |
| `scripts/transfer_k1_report.py` | the k=1 transfer experiment |
| `scripts/truth_retest_report.py` | the test-retest agreement of the token-ablation map |
| `scripts/crop_caption_report.py` | the caption words of a top ablation-effect image patch against a random patch |
| `scripts/plot_holdout_replication.py` | the first set and the held-out set, side by side |
| `scripts/k1_image_stats.py` | 7 image statistics as predictors, and the token-ablation map with those statistics regressed out |
| `scripts/k1_sign_and_flip.py` | the top half and the bottom half of the map, the answer flip, and 2 predictors combined as 1 |
| `scripts/k1_locality_predict.py` | the top-1 share curve, and a gain predicted from a map |
| `scripts/hole_vs_content_map.py` | the deletion map against the pixel-substitution map |
| `scripts/k1_hole_content_residual.py` | the pixel-substitution map with the deletion map regressed out |
| `scripts/k1_fitted_probe.py` | a ridge regression on `h` or on `g * h`, grouped by photograph |
| `scripts/self_report_report.py` | the model's stated reason against the predictor's crop |

## A run, end to end

The 4 steps below give the headline number for 1 case. Each step is 1 SLURM
job on klara.

1. Measure the map.

```bash
sbatch monocle/monocle.sub monocle.causal_maps --cases restaurants \
  --n-images 60 --group 1 --batch 12 --reference absent --seed 20260821 \
  --out-dir outputs/_monocle/causal_maps60
```

2. Score every attribution method against that map.

```bash
sbatch monocle/monocle.sub monocle.input_gate --case restaurants \
  --seed 20260821 --n-images 60 --layers 12 16 18 20 22 24 26 28 30 \
  --batch 8 --group 1 \
  --truth-from outputs/_monocle/causal_maps60/restaurants_absent/maps.parquet \
  --lens outputs/_monocle/jlens/mm/gemma4_12b_lens.pt \
  --mean-grad outputs/_monocle/mean_grad/restaurants/mean_grad.pt \
  --out-dir outputs/_monocle/gateB60_g1L
```

3. Repeat step 2 with `--prompt-case libraries` and another output directory.
That is the wrong-judgment control.

4. Score it.

```bash
python scripts/k1_layer_profile.py --pool \
  --root outputs/_monocle/gateB60_g1L \
  --xq-root outputs/_monocle/gateB60_g1L_xq --cases restaurants
```

## The tests

| File | Covers |
|---|---|
| `tests/test_k1_separation.py` | the statistic, the null distribution, the Benjamini-Hochberg step |
| `tests/test_transfer_k1.py` | the map loader, the top and bottom tokens, the summary |
| `tests/test_truth_retest.py` | the comparison, the sign convention |
| `tests/test_patch_caption.py` | the crop box at every edge |
| `tests/test_label_override.py` | the answer-word and question overrides |
| `tests/test_causal_words.py` | the logit-lens word readout |

All 64 tests run in under 10 seconds with no GPU.

## The fitted ablation predictor tools, 2026-08-25

| Tool | What it does |
|---|---|
| `scripts/k1_fitted_probe.py` | fits a ridge regression on `g*h` or on `h`, grouped by photograph, with a held-out group |
| `scripts/k1_null_gain.py` | the permutation null distribution for the gain, which is the statistic under test |
| `scripts/k1_probe_transfer.py` | the case-by-case transfer matrix of the fitted weights |
| `scripts/k1_probe_shared.py` | the pooled predictor, leave 1 case out, and the learning curve |
| `scripts/k1_reader_anatomy.py` | the top-m sweep, the trim of the massive-activation dimensions, the participation ratio |
| `scripts/k1_reader_dims.py` | the dimensions that each case selects, and their words through the embedding |
| `scripts/k1_export_reader_cells.py` | writes the fitted map in the shape that `monocle.patch_caption` reads |
| `scripts/k1_reader_flip.py` | the flip rate of the predictor's image token, with the oracle beside it |
| `scripts/k1_target_words.py` | the word list of each case, taken from the question |

**Warning:** `k1_fitted_probe.py` reports `null_mean` and `null_gain`. Quote
`null_gain`. `null_mean` is a null distribution for the `real` column alone,
and it is positive wherever a case has a strong mean layout.

**Warning:** `Gram.fit(drop=...)` takes 1 key or a list of keys. A list
leaves out a whole case. Never store a per-photograph `d` by `d` matrix: at
3840 dimensions each is 118 MB.

The activations come from `monocle.input_gate --save-acts`. The image tokens
come before the question, so `h` is the same under any question and only `g`
carries the question.

## The held-out-photograph experiment, 2026-08-26

The top-and-bottom-token experiment of the predictor ran only on the photographs
that the fit used. This group runs it on photographs that no earlier experiment
used, so we can measure the predictor's advantage out of sample.

| Tool | What it does |
|---|---|
| `monocle.input_gate --save-acts 20 --truth-from <maps>` | writes `h` and `g` for a new photograph set; `--truth-from` skips the 256 ablations |
| `scripts/k1_export_reader_cells.py --pred-root <acts>` | fits on 1 activation store and predicts on another |
| `scripts/k1_export_reader_cells.py --check-against <dir>` | rebuilds an earlier map and reports the rank agreement |
| `scripts/reader_fresh_report.py` | the 2 gaps, the paired map test, and the position of the predictor's image token |

The order of the steps:

1. Measure the map of the held-out photographs with `monocle.causal_maps`.
2. Write the activations with `input_gate`, at the same seed and image count.
3. Export the predictor's map with `--pred-root`, in `shared` mode.
4. Cut and caption the crops with `monocle.patch_caption --maps ...`.
5. Score both stores with `scripts/reader_fresh_report.py`.

**Step 2 is fast.** `--truth-from` reads the map that step 1 already
measured, so the job runs 1 forward pass and 1 backward pass for each
photograph. It costs about 100 seconds for each case, against hours for the
ablations.

**Warning: no output records alpha.** A predictor map holds `image_id` and
`causal` and nothing else, so a later run cannot read its settings back.
Always pass `--check-against` on a rebuild of an earlier set. A median Spearman
rho of 1.0000 shows that the settings match. The experiment of 2026-08-26 used
alpha 1e5, layer 20, feature `gh`.

**Warning: put the case in the permutation stratum.** 60 photographs of 360
carry 2 questions, so a stratum keyed on the photograph and the row band
alone can hold crops of 2 different maps. Key it on the case, the
photograph and the band.

The result is Finding 6 of `shared-negative-evidence.md`.
