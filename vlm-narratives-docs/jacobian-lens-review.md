# Jacobian-lens review: specification, code, and parity

Date: 2026-08-31. Scope: how monocle trains, implements, and uses the
Jacobian lens for causal interpretability of gemma-4-12B-it on the 7
canonical cases.

Files reviewed: `monocle/jlens_fit.py`, `monocle/jlens_fit_mm.py`,
`monocle/jlens_fit_ladder.py`, `monocle/jlens_merge_mm.py`,
`monocle/jlens_read.py`, `monocle/jlens_validate.py`,
`monocle/jlens_compare.py`, `monocle/jlens_steer.py`,
`monocle/judgment_trace.py`, `monocle/safety_workspace.py`,
`monocle/canonical.py`, the vendored `sub/jacobian-lens` package, and the 5
lens documents in this folder.

## Summary

The estimator is a faithful port of the reference method. The project's own
negative results agree with the strongest published critique. The main flaws
are above the estimator, not in it:

1. The main L42 claim rests on a readout with no causal test.
2. Two existing rules contradict the claim.
3. One output of `safety_workspace` conflicts with the project's own
   conclusion on per-image-token lens maps.

The code is careful. The defects are provenance, precision, and consistency
defects, not estimator defects.

## Parity: what already matches the published work

| Item | Where | Finding |
|---|---|---|
| Estimator: cotangent sum over targets, mean over sources and prompts | `sub/jacobian-lens/jlens/fitting.py` | exact parity |
| Unembed = final norm + lm_head + softcap; J=I gives the logit lens | `jlens/hf.py` | parity |
| Attention-sink skip (16 positions); final-position exclusion | `fitting.py` | parity |
| Readout at the decision position, before generated text | `canonical.FORCE_PREFIX` | parity with the JADR decision point |
| First-token classes + the Much* collision check | `canonical.label_classes` | stronger than the reference |
| R-lens replication, with a causal criterion that the reference post lacks | `r-lens.md` | beyond parity |
| The calibration critique, measured independently | `causal-lens-retrospective.md` | beyond parity |
| Estimand statement + corpus preregistration | `lens-corpus-faithfulness.md` | beyond parity |

Two departures are deliberate and correct. The corpus is 128 prompts, not
about 1,000; the community showed that 10 prompts already work. The
multimodal fit decouples the source mask (image tokens) from the target
mask. That defines a new estimand, and the corpus-size series measured
which axis of it matters.

One external point supports the local negative result. A community audit
found that banded Gemma models do not resolve concepts through the fitted
lens, while a Qwen model without the band does. The local finding is the
same fact from the causal side: the fitted translator shows no causal
effect at the image tokens of gemma-4-12B-it. Cite it as corroboration.

## Specification flaws

### F1 — The L42 claim has no causal test of the lens direction

The claim ("the judgment first becomes decodable at the last token
position at L42 under the production prompt, never under the neutral
prompt") is a readout-magnitude claim. The calibration critique warns
against exactly this shape: components at the same readout level can do
very different things. The one causal intervention on the lens direction,
activation addition, returned a null result with a structural cause. The
activation-patching and ablation results on information flow support the
position account, not the depth account. See P1.

### F2 — The agreement threshold rejects the layer that the claim decodes

`judgment_trace` prints the rule: a layer with r below 0.8 across lenses
reflects its fit corpus. `lens-corpus-faithfulness.md` measured L42 at a
median r of 0.67. The claim and the threshold cannot both be correct as
written. The likely resolution: the threshold applies to the all-position
content readout, and the L42 claim decodes a 2-token label contrast at one
position. No document says this. See P3.

### F3 — Part (b) conflicts with the project's own conclusion

`safety_workspace` part (b) renders per-image-token heat maps of
safety-vocabulary mass under the multimodal (mm) lens. The retrospective's
conclusion is explicit: the fitted lens decodes nothing at the image
tokens, and a figure that colors image tokens from a lens readout makes a
claim that the project's results refute. See P7.

### F4 — The layer grid confounds depth with layer type

7 of the 8 fitted layers sit immediately after a global-attention layer,
and the grid has a spacing of 6 layers. "L42" thus means "somewhere in
37-46, at a position after a global-attention layer". The 47-layer mm lens
exists, but `safety_workspace` still reads the 8-layer grid, and its
equal-layer assert blocks the 47-layer lens. See P2 and C4.

### F5 — The neutral condition has the wrong measure for "never"

`position_metrics` reads only p(production label). Under the neutral
prompt the model can hold a different judgment, and that judgment also
appears in the readout. The metric cannot separate "no judgment" from
"another judgment". See P4.

### F6 — The part (a) summary carries no uncertainty

`summarize_part_a` reports group means only. "Never under the neutral
prompt" is a null claim and needs a confidence interval and a power
statement. See P5.

## Code defects

| # | File | Issue | Status |
|---|---|---|---|
| C1 | `monocle/canonical.py` + `canonical-lens-prompt-path.md` | The docstrings say the criterion demands `interleaved_labels`; `REQUIRED_LAYOUT` says `images_then_labels` and marks the other REJECTED. The doc page carries the stale name throughout. | fixed 2026-09-01; 5 more stale doc spots fixed too |
| C2 | `jlens/lens.py` + `jlens_merge_mm.py` | Completed shards save J in fp16; a merge over final files loses precision that a merge over fp32 checkpoints keeps. | fixed 2026-09-01 |
| C3 | `jlens/lens.py` `transport()` | Moves the 59 MB J from host to GPU on every call; part (a) moves about 4 GB per pair. Cache per device. | fixed 2026-09-01 |
| C4 | `safety_workspace.py` | The equal-layer assert forbids a mixed lens set, so the 47-layer lens cannot join. Intersect instead. | fixed 2026-09-01 |
| C5 | `jlens_fit_mm.py`, `jlens_fit.py` | `cmd_smoke` calibrates on the default layers even when the fit runs `--all-layers` or `--dense`, so the smoke test under-reports the peak. | fixed 2026-09-01 |
| C6 | `jlens_fit_ladder.py` | The checkpoint load uses `weights_only=False`; every other loader uses `True`. | fixed 2026-09-01 |
| C7 | the lens `.pt` format | A lens file records no estimand: no rules, masks, corpus, prompt kind, dim_batch, or model revision. | fixed 2026-09-01 |
| C8 | `jlens_fit_mm.py` `_is_replicable` | Bool tensors pass through without replication; a bool attention mask would go out at batch-1 beside batch-8. | fixed 2026-09-01 |
| C9 | `safety_workspace.py` part (b) | The code silently skips a map layer that has no record. | fixed 2026-09-01 |
| C10 | `safety_workspace.vocab_mass_ids` | A vocabulary word with no exact single-token form silently contributes zero ids. | fixed 2026-09-01 |
| C11 | `jlens_fit.py` smoke text | Says "per 250-prompt shard"; the default is 128 prompts in 4 shards. | fixed 2026-09-01 |

## Improvements that add parity to the science

| # | Item | Status |
|---|---|---|
| P1 | Ablate the label direction at L42 (project out span{J42^T w_label} at the last token position under the production prompt, with a matched random-subspace control). Ablation, not activation addition: the reference paper's strongest safety result is an ablation. | done 2026-09-01 — see `label-ablation.md`: ablation does NOT change the judgment; the claim weakens to "decodable" |
| P2 | Repeat the readout of part (a) through the 47-layer lens. Gives 1-layer resolution and removes the global-attention confound. Needs C4. | done 2026-09-01 — see `workspace-depth-47.md`: the judgment becomes decodable over a ramp L40-45, first at the L41 global-attention layer, with a peak at L45 (a layer inside a run of sliding-attention layers), so the layer-type confound is removed and "L42" was an artifact of the grid spacing |
| P3 | State the scope of the 0.8 threshold, and compute the narrower last-token-position agreement from `trace.parquet`. | done 2026-09-01 — the narrow readout exceeds the threshold; see Resolutions |
| P4 | Add a self-agreement column to part (a): depth argmax against the same condition's own final answer. Computable from the saved parquet. | done 2026-09-01 |
| P5 | Paired production-neutral statistics per layer: bootstrap interval, BH over layers. | done 2026-09-01 — `scripts/answer_depth_stats.py`; see Resolutions |
| P6 | A lens registry: path, sha256, rules, masks, corpus + hash, n, layers, dim_batch, fit jobs, validation report. | done 2026-09-01 — `scripts/register_lenses.py` + `lens-registry.md` |
| P7 | Resolve F3: label the part (b) figures as readouts of representation, not of causal effect, or retire them. | done 2026-09-01 |
| P8 | Close or reopen the activation-addition question: one weak-coefficient condition at the centered scale, and a pseudo-inverse pre-image condition. | done 2026-09-01 — question CLOSED; see `steering-weak-alpha.md`: both pre-images are null at a weak coefficient, and the cause is structural (the translator reconstructs about 5 percent of the label direction) |

## Resolutions, 2026-09-01

The fixes above ran, and three flaws now have measured answers.

### F2 resolves in favor of the claim

The 0.8 threshold applies to the all-position content readout. The L42
claim decodes a 2-token label contrast at one position.
`scripts/lens_agreement_answer.py` measures the agreement for that narrow
readout, across images, at the last token position (mm against wikitext,
subway_safety):

| Layer | 36 | 42 | 46 | 47 |
|---|---|---|---|---|
| narrow r (last token position, across images) | 0.97 | 0.91 | 0.96 | 1.00 |
| wide threshold (all positions, median over images) | 0.32 | 0.67 | 0.73 | 1.00 |

The layers that the depth claim rests on exceed the threshold when the
threshold measures the quantity that the claim is about. The early layers
are WORSE under the narrow readout (L12 and L18 are negative), so the
threshold's warning holds there. Caveat: n = 8 images; increase n before a
paper quotes the narrow r.

### F5 and F6 resolve against the wording "never under neutral"

`scripts/answer_depth_stats.py` (300 pairs, paired, 20000-resample
bootstrap, BH over layers) gives these results. The production-minus-neutral
difference at the label position is +0.027 [+0.021, +0.032] in
p(production label) and +0.47 [+0.41, +0.53] in 4-way agreement at mm L42,
and the neutral condition at mm L42 is NOT at its minimum: p(production
label) = 1.1e-3 [7.8e-4, 1.3e-3], agreement 0.22. But that 0.22 is a constant
prediction: the neutral argmax is `Less` for 224 of 300 pairs and `Same`
for the rest, while the production argmax tracks the pair. The supported
statement is: **under the neutral prompt the readout gives a constant
label, not the pair's judgment.** Two more measured facts follow. L30 shows
a dip (the production prompt LOWERS the label probability there, in all 3
lenses, and the effect survives BH), and the early-layer "agreement" is an
artifact of a constant class (sd exactly 0). Thus read the signed-rank and
median columns, not the means, below L36.

### F1 resolves against a causal reading of the lens depth

The ablation ran (`label-ablation.md`, `monocle/label_ablate.py`, 150
pairs, 5 layers, matched controls). Ablation of the rank-5 label subspace
at L42 changes the judgment by +0.043 logits [+0.018, +0.068] over the
matched control, on a judgment scale of about 5 logits, and flips 2.0
percent of answers against 0.7. The effects at L24 and L36 are 8 to 16
times larger, which contradicts an onset at L42 in causal terms. So the
correct statement is: **the judgment is DECODABLE through the lens over
layers 40-45; the lens direction at that depth is not a causal
bottleneck.** This is the calibration critique's prediction, confirmed
locally: a readout depth is not an intervention depth.

### P8 closes the activation-addition question, with the cause measured

The weak-coefficient re-run (`steering-weak-alpha.md`: 100 pairs, layers
24/36/42, 11 values of the coefficient alpha in centered units, transpose
AND pseudo-inverse pre-images, matched controls) finds no regime where the
signal beats its control. The cause is structural: even the pseudo-inverse
pre-image reconstructs the label direction with a relative error of 0.998
or more; the fitted translator's row space holds about 5 percent of it. A
direction that the translator cannot represent cannot act as a steering
vector through it. This closes the 2026-08-23 caveat ("a weaker grid may
find a regime") with a measurement.

### F4 resolves: the onset is a ramp, and the grid named its sample point

The 47-layer readout (`workspace-depth-47.md`, 300 pairs) shows that the
judgment becomes decodable at L40-41 (the first decodable layer is the L41
GLOBAL-attention layer) and increases to a peak at L45, a layer inside a
run of sliding-attention layers, with a dip at L46. So the decodability
does not track the layer type inside the ramp, and the correct depth
statement is "decodable over layers 40-45", not "decodable at L42". The
safety-vocabulary mass is 0.000 at all 47 layers.

### The registry found a precision gap

Every merged lens on disk is fp16, `mm_all47` included; every merge
predates the C2 fix. The known cost is the ~4e-4 relative residual that
the R0 check measured. New merges save fp32; the lenses on disk stay as
they are, and `lens-registry.md` records their hashes.

## The change to the single-image task, 2026-09-01

After the review, the user changed the activation-addition question. The
pairwise prompt template stays for the validation against the ground-truth
variables only; it controls positional bias in the aggregation. The
interpretability question moves to single-image evaluations, because the
lens decodes one next-token distribution and a multi-token label only
shows its first token.

Each case gets one pair of single-token answer words, one positive-valence
term and one negative-valence term. We verified each word on the
gemma-4-12B-it tokenizer:

| Case | Positive | Negative |
|---|---|---|
| subway_safety | safe | dangerous |
| libraries | maintained | worn |
| schools | good | bad |
| road_quality | smooth | damaged |
| parks_plazas | maintained | worn |
| restaurants | delicious | gross |
| street_photography | beautiful | ugly |

The words "appealing", "dislike", "neglected", "inviting", "rundown", and
"shabby" split into 2 or more tokens, so they are not usable. The script
is `monocle/single_steer.py`. Its `--coverage` mode measures the
translator reconstruction error of each case's contrast direction before
any sweep; the P8 null result was structural, and the new directions must meet
this criterion first. Results: see `single-image-steering.md`.

## What the review did not flag

The estimator mathematics, the layer-hook convention, the BOS-token
handling, the resume and merge weighting, the hold-out design of the
corpus-size series, and the R0 harness check are correct as written. The
check that catches a silent failure of the forced prefix, and the
`--check-head` no-effect test, are good practice.

## Related

- `lens-corpus-faithfulness.md` — the estimand and the corpus-size series
- `causal-lens-retrospective.md` — the causal criteria that the lens fails
- `r-lens.md` — the rules study
- `canonical-lens-prompt-path.md` — the mandatory prompt path
- `jlens-47-layer-fit.md` — the dense fit that P2 uses
