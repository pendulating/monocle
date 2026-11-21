# The causal lens: a retrospective and the plan

**Date:** 2026-08-24, corrected 2026-08-25. **Model:** gemma-4-12B-it.
**Cases:** subway_safety, libraries, restaurants, and 4 more added on
2026-08-25: road_quality, schools, parks_plazas, street_photography.

## The goal

We want a method that identifies which image tokens cause an aesthetic
judgment. The judgment is safety, upkeep, or preference. The method must be
causal, not only illustrative.

## The answer, first

**A causal image-token attribution method exists for every judgment we tried.
The method that works is the fitted ablation predictor: a ridge regression
from gradient-times-input attribution to the token-ablation effect. It beats
its control on all 7 of the 7 cases, on photographs outside its fit, and
safety is one of them.**

The difference between gradient-times-input attribution and the fitted
predictor is the main result.

| Method | What it is | Cost | Cases where it beats the control |
|---|---|---|---|
| `gxa`, gradient-times-input attribution | the plain sum of `g * h` over 3840 dimensions | 1 backward pass | 4 of 7 |
| `g*h`, the fitted ablation predictor, fitted per case | a ridge regression on the same 3840 dimensions | 1 backward pass, and a fit on 60 photographs | **7 of 7**, held out |
| `g*h`, 1 pooled weight | 1 ridge regression over all 7 cases at once | 1 backward pass, and no new fit | **7 of 7**, +1.40 pooled |
| the same weight, judgment never seen | fitted on the OTHER 6 cases | 1 backward pass | **6 of 7** |

There is 1 predictor, and not 1 for each judgment. A weight fitted on 6
judgments beats the control on the 7th in 6 of the 7 tries. Only safety
needs its own fit.

Gradient-times-input attribution sums with a weight of 1 on every dimension.
Where the judgment sits in 1 or 2 image tokens of 256, that sum works. Where
it spreads over the whole image, the sum cancels, and a fitted weight does not.

**The line fitted on the single-token share thus predicts where
gradient-times-input attribution works. It does not predict where the
judgment is recoverable.** That correction is the main result of 2026-08-25.

The rest of this page reports gradient-times-input attribution first, because
we built it first and every control below tests it. The fitted predictor
starts at "A fitted ablation predictor beats its control on subway safety"
and runs to the end.

## Gradient-times-input attribution, and why we call it a family of methods

Gradient-times-input attribution beats its control on 4 of the 7 judgments:
the restaurant preference, the library upkeep, the park upkeep and the road
condition. It does not beat its control on subway safety, school quality, or
street photography.

The word family matters. Two conditions, `gxa` and its centered variant
`cheap`, measure the same quantity, and their maps agree at only 0.12 to
0.18. Each is weak. On a held-out set some condition repeats on 3 of the 4
cases, and only on restaurants does the SAME condition repeat.

The method is gradient-times-input attribution on the image tokens, at the
single-token ablation unit, with the ranking pooled over the layers. It costs
1 backward pass. The fitted Jacobian lens beats its control on none of the 7.

**And we can say which cases pass before we run the method.** The share of the
causal effect that sits in 1 image token of 256 is measured from the
token-ablation map alone. A line fitted on the first 3 cases orders the 4 new
ones at a rank correlation of +0.80, and all 7 at +0.75.

| Judgment | `gxa` gain | 95% interval | Passes | Repeats on new photographs | Share in 1 image token |
|---|---|---|---|---|---|
| restaurants, appeal | **+1.37** | +0.65 to +2.14 | yes | **yes, same condition** | 0.0207 |
| libraries, upkeep | **+1.00** | +0.43 to +1.60 | yes | yes, other condition | 0.0159 |
| parks and plazas, upkeep | **+0.82** | +0.23 to +1.45 | yes | yes, other condition | 0.0176 |
| road quality, condition | **+0.66** | +0.09 to +1.23 | yes | **no** | 0.0198 |
| schools, quality | +0.39 | -0.15 to +0.94 | no | -- | 0.0168 |
| street photography, appeal | +0.11 | -0.48 to +0.68 | no | -- | 0.0155 |
| subway, safety | +0.05 | -0.51 to +0.65 | no | -- | 0.0148 |

Six of the 7 cases share the answer words `Good` and `Poor`, so the words do
not make the split.

**Read the 5th column before you quote the 2nd.** Each case that repeats was
run again on 60 photographs that share no image with the first 60. Three of
the 4 repeat, and only restaurants repeats through the SAME condition. See
"The held-out replication of the 2 new cases" for what that costs the claim.

### The main test

The unit is 1 image token of 256. The token-ablation map is the MEASURED
change in the judgment when that token is ablated, for every token. No
pooling and no additive assumption enters.

For each photograph we take 2 tokens. One is the token the attribution method
ranks as the strongest positive. The other is the token it ranks as the
strongest negative. We then read the measured effect of each:

    z = (truth[argmax] - truth[argmin]) / sd(truth over the 256 tokens)

The control is the same photograph, the same token-ablation map and the same
attribution method under ANOTHER case's question. A method that reads the
image content keeps its separation there. A method that reads the judgment
loses it.

| Case | Own question | Other question | Gain | 95% interval | q |
|---|---|---|---|---|---|
| **restaurants** | +1.06 | **-0.31** | **+1.37** | +0.65 to +2.14 | **0.0135** |
| **libraries** | +1.30 | +0.30 | **+1.00** | +0.43 to +1.60 | **0.0201** |
| subway_safety | +0.42 | +0.37 | +0.05 | -0.51 to +0.65 | 0.592 |

`q` is a Benjamini-Hochberg step over all 15 condition-by-case tests. The
ranking is the mean rank over 9 layers, so no layer is chosen.

The ceiling comes from a SECOND measurement of the token-ablation map, not
from the map when it scores itself: +7.49, +4.55 and +4.00. So the method
reaches about 14 per cent of what a perfect attribution method could do on
restaurants, and about 28 per cent on libraries.

### The control that makes it causal

A method that reads the image content would keep its separation under any
question. A method that reads the words would fail under any change of the
words. `gxa` does neither.

| The prompt | restaurants | libraries |
|---|---|---|
| its own question | +1.06 | +1.30 |
| the same question, reworded | +0.95 | +1.13 |
| another case's question | -0.31 | +0.30 |

**It keeps about 9 tenths of its separation under a reworded question, and
loses it under another judgment.** So it tracks what the model is asked to
decide, not how the question is written.

### It replicates on photographs the study never saw

The whole experiment ran again on 60 NEW photographs of each case, sampled
with seed 20260825, with 0 overlap. Nothing was refitted.

| Case | First 60 | 60 new | q on the new set |
|---|---|---|---|
| restaurants | +1.368 | **+1.041** | **0.013** |
| libraries | +0.999 | +0.784 | 0.098 |
| subway_safety | +0.052 | +0.446 | 0.082 |

Restaurants replicates inside its own interval. Libraries replicates in size
and misses the correction, while its centered variant `cheap` passes at
0.013. Subway safety rises and still does not exceed the threshold.

### What the method is: a ranking of the top image tokens on the judged object

Three measurements say what kind of attribution method this is.

| Question | Answer |
|---|---|
| Does it identify the strongest image token? | yes, 0.150 against 0.004 by chance |
| Does it rank the whole map? | no, rank correlation 0.06 over 256 image tokens |
| What is in the image token it identifies? | the restaurant or the library building |
| How many judgments does it beat the control on? | 4 of the 7 we tried |
| Can we say which 4 before we run it? | yes, from the token-ablation map, rank correlation +0.75 |
| Does the fitted predictor beat it? | yes, by about a third, and it holds out of sample |
| Can the model name the image token itself? | no, only the kind of thing in it |

The content measurement is direct: cut the image patch out of the photograph
and ask the model to name what it holds. An image patch at an extreme of the
map names the judged object 68 to 72 per cent of the time. A random image
patch of the SAME photograph gives 41 to 48. The conditions sit in the same
rows of the image. **The image token the METHOD selects does the same**: 70
per cent against 39 on restaurants and 63 against 48 on libraries.

**The 2 extremes hold the same object.** The image patch that pushes the
judgment up and the image patch that pushes it down are described the same
way, p=0.996 on restaurants. So the causality is about the object, and the
SIGN is carried by something no word names.

### The effect of an image token depends on the photograph around it

The token-ablation map ablates a token. A second map instead gives the image
token ANOTHER photograph's pixels and ablates nothing. The 2 maps disagree:
they correlate at r=0.22 to 0.29 and name the same top image token 7 to 30
times in 100. The method beats its control against BOTH, +0.97 and +0.80, and
keeps two-thirds of that after the ablation-effect map is removed from the
content map.

But move the pixels of 1 image token INTO another photograph and the
judgment barely follows. Restaurants give +0.26 logits, against 5 to 6 for
the whole image. The other 2 cases give nothing at all.

**So an image token carries no effect of its own.** Change what sits in an
image token and the judgment moves, by MORE than an ablation moves it. Carry
that same image patch to a new photograph and almost nothing comes with it.

### What does NOT beat the control

- **The fitted Jacobian lens.** Its gain is +0.09, +0.45 and -0.32.
- **The mean-gradient methods.** The mean-gradient method (`meang_own`) and
  the per-position mean-gradient method (`meangp_own`) score the same under
  the wrong question, to 3 decimals. They read the image layout.
- **Every attribution method, on subway safety.** The only condition that
  beats 0 there is a position prior built from the OTHER photographs (+0.84,
  p=0.0002).

### Why the earlier conclusion was negative

The set test ablates k image tokens at once. It fails for every condition,
and 2 measurements say the test, not the attribution method, is the limit.

1. **The judgment saturates.** One ablation unit of 64 already moves it 25 to
   46 per cent of the way that half the image moves it.
2. **Half the joint effect is not additive.** We fitted a ridge regression of
   the set effect on the image-token indicators, on 512 measured ablations of
   each photograph. It holds out an R2 of 0.39 to 0.53. And the SUM of the
   measured 1-token effects reaches only 33 to 51 per cent of the ceiling at
   the 2x2 ablation unit.

Every attribution method here is additive by construction. The measured
ceiling itself fails the baseline threshold at k=16 and k=32 on all 3 cases.
Where the ceiling has no signal, no condition can show one.

**Warning: the k-by-k tables below are superseded.** They report about 108
tests at a nominal 95%, and their scattered passes are the effect of that
multiplicity. Read "The corrected result", "The 1-token test" and "Every
layer pooled" before you quote any number from them.

## The reading order

This page is a laboratory record and it runs in the order the work happened.
The sections below hold the current result. Everything else is history, and
the set-test sections carry their own warning.

| Read | For |
|---|---|
| The answer, first | the result and its controls |
| Every layer pooled: the test that decides | the main statistic |
| The held-out replication | 60 photographs no part of the study saw |
| The layout control | the position prior removed |
| The paraphrase control | the method follows the judgment, not the words |
| Averaging over wordings | 4 wordings, with a matched control |
| The method ranks the top image tokens, it does not reproduce the full map | the unit of the prediction |
| What is in the causal image token | the content, in words |
| Moving 1 image token into another photograph | an image token is not portable |
| The method is not edge energy, brightness or colour | the pixel control |
| Ablating 1 image token almost never changes the answer | the limit |
| How well does the token-ablation map itself repeat? | the ceiling |
| Two tests of why subway safety fails | the case that does not pass |
| The main result, re-derived with no shared code | the correctness check |
| A pre-registered test on 4 cases we never touched | the generalization |
| The method also predicts a content map, not only an ablation-effect map | the second token-ablation map |
| Why the gradient is the part that reads the question | the mechanism |
| Subway safety: 2 more answers ruled out | the case that does not pass |
| The sign split | which end carries the gain |
| A fitted ablation predictor | what gradient-times-input attribution does not recover |
| The held-out replication of the 2 new cases | what it costs the claim |
| All 4 cases that pass | the method's image token holds the judged thing |
| Does the model name the thing the method points at? | the self-report |

The tools are listed in
[causal-patch-reader-tools.md](causal-patch-reader-tools.md).

## How to read one photograph

The method costs 1 backward pass. The recipe:

1. Build the single-image prompt for the case with
   `monocle.single.build_single_inputs`.
2. Find the image-token positions with `monocle.patch_ablate.slots_of`.
3. Take the gradient of the answer logit difference with respect to the
   image-token hidden states at layers 12 to 30, with
   `monocle.patch_attribution.attribute_pair`.
4. For each layer, score each image token as the dot product of its gradient
   and its hidden state. That is `gxa`.
5. Rank the 256 image tokens at each layer, and take the mean rank.
6. The strongest ranks name the tokens that hold the judgment up. The weakest
   name the tokens that push it down.

`monocle/input_gate.py` runs steps 1 to 4 and writes the maps.
`scripts/k1_layer_profile.py --pool` runs steps 5 and 6 and scores them.
`scripts/plot_reader_sheet.py` draws the result on the photograph.

**Warning: use the method only where it is tested.** It passes on restaurants
and on libraries. It fails on subway safety. Before you trust it on a new
judgment, measure the 1-token ablation map on 60 photographs and run the same
test.

## What the record already settles

| Question | Answer | Where |
|---|---|---|
| Does the token-ablation map carry order? | Yes. z=5.7 at k=1, z=3.45 at k=32 | `deletion_curve/subway_safety_g2_self_ablate` |
| Does it survive a substitution with no ablation? | Yes. z=1.8 to 2.6 | `..._self_substitute` |
| Do the image tokens sum to the whole? | No, at any grid size | `causal_maps` sweep |
| Do 2 direct per-token measurements agree? | Only at r=0.26 | 2026-08-23 log |
| Is the map explained by simple image statistics? | No. every \|r\| < 0.07, joint R2=0.185 | `cell_statistics/absent_g2` |
| Does activation patching at layer 20 match the input ablation? | Weakly. r=+0.08, 17 of 20 images positive | this run |
| Does the activation-patching ORDER separate? | Yes at set size. z=2.19 at k=32 | `deletion_curve/..._g2_donor` |
| Are the image tokens sufficient, not only necessary? | Weak. z=1.07 at k=16 | `..._self_ablate_keep` |

## Where the image acts

The attention knockout names the layers. We cut the attention edge from the
image tokens into the text-token stream, and we read the change in the logit
difference (`c`).

| Layers cut | mean delta c | mean absolute delta |
|---|---|---|
| 0-21 | +0.57 | 1.49 |
| **22-29** | **-1.58** | **2.93** |
| 30-47 | +0.40 | 0.46 |
| 0-47 | -7.68 | 7.68 |

The last token position almost never reads an image token directly: every
condition that cuts that edge moves `c` by 0.51 or less. Thus the image
information leaves the image-token positions below layer 30, and most of it
leaves in layers 22 to 29.

The image-token gradients are also small. At layer 36 the gradient norm of the
last token position is 37 times the norm of an image token, and the absolute
attribution is 60 times larger. See `patch_norms/subway_safety`.

## What this predicts

Activation patching at layer L can only act on what the image token did not
yet transfer. Thus the agreement with the input ablation must FALL with depth,
and it must fall hard above layer 22.

- **P1.** The activation-patching agreement falls with depth. It is highest at
  layer 0.
- **P2.** The agreement above layer 30 is inside the shuffled-image baseline.
- **P3.** A method that reads the residual stream scores best below layer 22.
- **P4.** The model-free image-statistic baseline does not separate the
  token-ablation test.
- **P5.** No attribution method reaches the measured ceiling of z=3.45 at
  k=32.

P5 is a prediction of failure. We record it so that a positive result is a
real positive result.

## The criterion we now use

Criterion A scored a lens against layer-20 activation patching. That
patching agrees with the input ablation at r=+0.08, so a lens can pass
criterion A and still not predict the token-ablation map. **Criterion B
scores against the input ablation itself.**

Criterion B costs little. `--truth-from` loads the 256-token ablation map that
the 2026-08-23 run measured, so an image costs 1 backward pass, not 256
forward passes.

## The decisive test

Rank the image tokens by an attribution method. Ablate the top k together.
Read the separation. The conditions:

| Condition | What it tests |
|---|---|
| the token-ablation map | the ceiling |
| `lens_own`, the Jacobian-lens map | the map that we would plot |
| `cheap`, `gxa` | the local gradient |
| `meang_own` | the mean gradient, a constant direction |
| image statistics (edge, luminance, contrast) | the model-free baseline |
| random | the ablation size alone |

An attribution method earns the name "causal lens" when it beats the
model-free baseline and comes near the ceiling.

## What we do not claim

The parts never sum to the whole. A single image-token value is not a result.
Any figure must show a SET of image tokens and an ORDER, never a per-token
magnitude.

---

# Results, 2026-08-25

## Criterion B: no attribution method predicts a single image token

Every attribution method was scored against the token-ablation map on 20
images, at 19 layers. The best entry of the whole table:

| Statistic | Best condition | Layer | Value | 95 percent interval |
|---|---|---|---|---|
| spearman | `gxa` | 0 | +0.054 | [+0.024, +0.082] |
| top_sign | `lens_own` | 6 | +0.565 | [+0.403, +0.721] |

No method predicts a single image token. The constant conditions
(`meang_own`, `meangp_own`) do no worse than the local gradient, because
neither works.

## Why an image token was always the wrong unit

The group sweep measures the ablation of a spatial BLOCK. We compare it
against the sum of the single-token effects inside that block.

| Block | Pearson | Spearman | Slope |
|---|---|---|---|
| 2x2 | +0.328 | +0.322 | 0.21 |
| 4x4 | +0.175 | +0.149 | 0.05 |
| 8x8 | -0.087 | -0.120 | -0.02 |

The parts over-predict the whole by about 5 times. Even the token-ablation
map does not transfer from single tokens to sets. Thus a per-token
correlation cannot judge an attribution method, and the set-ablation test
must.

## The token-ablation experiment, subway_safety, 20 images, 64 ablation units

Median separation z, and the share of the measured ceiling.

| Condition | z at k=8 | z at k=16 | share at k=16 |
|---|---|---|---|
| token-ablation map (ceiling) | 3.59 | 3.36 | 1.00 |
| `meangp_own` L18 | — | 2.26 | 0.67 |
| activation patching from a source image, L20 | — | 1.98 | 0.59 |
| `meang_own` L18 | — | 1.91 | 0.57 |
| `gxa` L18 | — | 1.66 | 0.50 |
| **model-free baseline: edge density** | — | **1.38** | **0.41** |
| `lens_own` L18 | — | 1.23 | 0.37 |
| `cheap` L18 | — | 0.77 | 0.23 |

## The set test against the strongest baseline (superseded)

**Warning: this section reads a k grid, and the multiplicity correction
later removes its passes. Read "The corrected result" and "The 1-token
test" for the numbers that are supported.**

A baseline that is weak at one set size is not a control. Edge density
separates at k=16 and hardly at all at k=8. Luminance does the opposite. Thus
the threshold is the STRONGEST model-free baseline at each k, and the test is
paired on the image.

| k | The baseline | z |
|---|---|---|
| 8 | luminance | +0.81 |
| 16 | edge density | +1.38 |

Gain over that threshold:

| Condition | gain at k=8 | gain at k=16 | exceeds |
|---|---|---|---|
| token-ablation map | **+2.17 [+0.76, +3.58]** | **+1.10 [+0.58, +2.52]** | **yes** |
| `meangp_own` L18 | +1.22 [-0.14, +3.40] | +1.64 [-0.02, +2.78] | no, at the boundary |
| activation patching from a source image, L20 | +0.02 [-0.50, +2.54] | +0.97 [-0.35, +2.09] | no |
| `meang_own` L18 | +0.52 [-0.51, +1.76] | +0.51 [-0.50, +2.27] | no |
| `gxa` L18 | -0.36 [-1.88, +1.03] | -0.07 [-0.75, +1.69] | no |
| `lens_own` L18 | -0.04 [-1.38, +1.51] | -0.42 [-2.24, +0.45] | no |
| `cheap` L18 | -0.11 [-1.17, +1.96] | -0.54 [-2.17, +0.63] | no |

**Only the direct measurement exceeds a model-free baseline.** No low-cost
attribution method does: not the fitted Jacobian lens, not the Rlens, not the
local gradient, not the mean gradient, and not layer-20 activation patching.

The token-ablation map exceeds the threshold, so the test is specific. It
does not merely reward the removal of high-edge image tokens.

`meangp_own` is the one condition that sits on the boundary, with a lower
limit of -0.015 at k=16. That is a question of statistical power, not a
positive result. We raise the image count from 20 to 60 to settle it.

**Warning:** an earlier draft of this page named `meang_own` as a winner. That
draft compared against edge density alone, which is a weak baseline at k=8.
The table above uses the strongest baseline at each k and supersedes it.

## Why the lens fails, measured

Two numbers explain it, and they bound the lens from both sides.

| Where the image acts | Where a constant direction exists |
|---|---|
| The image tokens feed the stream below layer 30, mostly 22-29 | The mean gradient holds a cosine of 0.0004 at layer 0, 0.08 at 16, 0.23 at 18, 0.47 at 30 |

A constant matrix needs a shared direction. A shared direction only appears
above layer 24. The image tokens stop their action near layer 29. The window
where both hold is narrow, and the lens does not exceed the baseline inside
it.

## The image tokens are not the same image tokens

The attribution methods select the top 16 and the bottom 16 image tokens at
CHANCE overlap with the token-ablation map. Chance is 4.0 of 16.

| Condition | top overlap | bottom overlap |
|---|---|---|
| `meangp_own` | 4.75 | 4.55 |
| `gxa` | 4.40 | 4.75 |
| `cheap` | 4.40 | 4.10 |
| `meang_own` | 3.90 | 4.10 |
| `lens_own` | 3.80 | 4.30 |

Thus an attribution method that separates the test does NOT do it because it
names the image tokens the measurement names. Any figure that colours image
tokens from a lens makes a claim that this table refutes.

## How much variance can a constant direction capture?

We kept 32 image-token gradients for each of 48 held-out images, and we read
the eigenvalues of their covariance. The mean-direction share (`mean_share`)
is the share of the variance that lies along the MEAN direction, and a
constant lens can use nothing else.

| Layer | participation ratio | directions for 80 percent | mean_share |
|---|---|---|---|
| 0 | 41.6 | 76 | 0.022 |
| 6 | 7.7 | 8 | 0.096 |
| 12 | 49.5 | 90 | 0.018 |
| 18 | 11.0 | 80 | 0.058 |
| 20 | 55.9 | 93 | 0.076 |
| 24 | 24.4 | 44 | **0.126** |
| 28 | 6.2 | 9 | 0.054 |
| 30 | 5.3 | 7 | 0.094 |

**The mean direction never holds more than 13 percent of the variance.** That
is a measured ceiling on the premise of a constant lens. No corpus can lift
it, because it is a property of the model and the task, not of the fit.

The effective rank is not flat with depth. At layers 6, 28 and 30 about 8
directions hold 80 percent. At layers 12 and 20 it takes 90. So a small basis
is plausible at some layers and not plausible at others.

## The full baseline set

6 baselines were run on the same 20 images and the same token-ablation test.

| Baseline | z at k=8 | z at k=16 |
|---|---|---|
| luminance | **+0.81** | -0.84 |
| column | +0.59 | -0.57 |
| edge density | +0.26 | **+1.38** |
| contrast | -0.05 | +0.96 |
| the lens map on the WRONG photograph | -0.00 | -0.32 |
| row | -0.23 | -0.51 |

Two results matter.

1. **The shuffled baseline sits at zero.** Give a photograph another
   photograph's lens map and the test collapses. Thus the lens map IS
   specific to its own image. It is not a generic layout prior.
2. **A model-free statistic still beats it.** `lens_own` reaches +1.23 at
   k=16, and edge density reaches +1.38 on the same images.

So the lens map is specific to the individual photograph, and it predicts no
better than an edge detector predicts. Both statements are necessary, and
only the pair is correct.

## The layer does not improve it

The same experiment at layer 30, where the mean gradient holds its highest
cosine:

| Condition | gain over the threshold, k=8 | gain over the threshold, k=16 |
|---|---|---|
| `meang_own` L30 | +0.18 [-1.55, +1.16] | -0.60 [-2.25, +0.34] |
| `lens_own` L30 | -0.83 [-1.73, +0.33] | -0.83 [-3.44, +0.19] |
| `gxa` L30 | **-1.33 [-2.39, -0.73]** | **-1.25 [-3.18, -0.10]** |

Layer 30 is WORSE than layer 18 for every attribution method, and `gxa` there
is reliably below the baseline. That agrees with the attention knockout: above
layer 30 the image tokens no longer act, so a method that reads there reads a
position whose information already left.

## A layer range does not improve it either

The attention knockout shows that the image tokens transfer their information
in layers 22 to 29. We summed the attribution over that window, and over every
layer below 30, with each layer scaled to unit spread first.

| Ranking | Spearman against the token-ablation map | top-16 overlap |
|---|---|---|
| `gxa` summed over 0 to 28 | +0.090 | 4.35 |
| `gxa` summed over 22 to 28 | +0.063 | 4.20 |
| `gxa` at L18 alone | +0.046 | 4.40 |
| `meangp_own` summed over 22 to 28 | +0.030 | 3.80 |

Chance overlap is 4.0. The sum over the transfer window does not beat the
single layer, and none of them leaves chance.

## The ablation unit matters as much as the attribution method

Two ceilings were run. One ranks by a map MEASURED on the 64-unit grid. The
other ranks by the 256-token map, summed into the same 64 units.

| k | measured on 64 units | 256 image tokens, then summed | ratio |
|---|---|---|---|
| 4 | +3.74 | +1.63 | 2.30 |
| 8 | +3.59 | +1.52 | 2.36 |
| 16 | +3.36 | +2.04 | 1.64 |
| 32 | +3.45 | +1.93 | 1.78 |

The paired gain of the direct measurement is +1.88 [+1.25, +3.20] at k=4 and
+1.59 [+0.41, +2.20] at k=8.

**A fine map loses most of its power when it is summed.** That is the same
failure of decomposition as the block table above. It limits every
attribution method here, because every attribution method produces a
256-token map. Against the SUMMED ceiling, `meangp_own` is level: 0.97 at
k=4, 1.61 at k=8, 1.11 at k=16.

## What exceeds the strongest model-free baseline

| Condition | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|
| measured on the 64-unit grid | **yes** | **yes** | **yes** | no |
| `meangp_own` L18 | **yes** | no | at the boundary | no |
| 256 image tokens summed to 64 | no | no | no | no |
| activation patching from a source image, `meang_own`, `lens_own`, `gxa`, `cheap` | no | no | no | no |

`meangp_own` at k=4 gains +1.48 [+0.11, +2.91], and it wins on 15 of 20
photographs. It is one constant direction for each image-token POSITION,
fitted on 48 held-out images of the same case.

**Warning:** we test 4 set sizes and 7 conditions. One pass with a lower
limit of +0.11 is weak evidence. The run at 60 images decides it.

The threshold itself moves with k, so it is stated here.

| k | The baseline that wins | z |
|---|---|---|
| 4 | contrast | +0.39 |
| 8 | luminance | +0.81 |
| 16 | edge density | +1.38 |
| 32 | contrast | +2.41 |

At k=32 nothing exceeds the threshold, the token-ablation map included. Half
the image is enough for a model-free statistic to do as well as anything.

## Why a linear attribution method cannot reach the token-ablation map

This part is a proof, not a measurement.

Every attribution method here reads one image token and returns a number:
`<w_p, h_p>`. A lens, an Rlens, a mean gradient and a gradient-times-input
map differ only in `w_p`. The value of a BLOCK is then the sum over its image
tokens:

    score(B) = sum over p in B of <w_p, h_p>

So a linear attribution method is EXACTLY additive by construction. It cannot
represent a block effect that differs from the sum of its parts.

The measurement can. It ablates the block and reads the model, so it sees the
interaction. The block table shows how large that gap is: the parts predict
the whole at Spearman +0.32, and they over-predict its size by about 5 times.

So no linear attribution method can match a measurement that reads the
interaction. That is a limit of the FORM, and a better corpus cannot lift it.

This argument has a limit. It does NOT say that the summed token-ablation map
is the best additive predictor. Another set of weights could sum to a better
block score. It says only that an additive score cannot represent a
non-additive effect.

Where `meangp_own` stands against the summed token-ablation map: 0.97, 1.61
and 1.11 of it at k=4, 8 and 16. So it is already level with that particular
additive predictor. The fitted ablation predictor answers whether a BETTER
additive predictor exists. A predictor fitted against BLOCK effects would
answer it exactly. We fit against single-token effects here, so our predictor
bounds the per-token problem, not the block problem.

## The 3 judgments behave the same

Criterion B was run on all 3 cases the goal names. Best Spearman anywhere in
each table, over 19 layers and 5 conditions:

| Case | Question | Best condition | Layer | Spearman |
|---|---|---|---|---|
| subway_safety | is this place safe? | `gxa` | 0 | +0.054 |
| libraries | how well maintained? | `gxa` | 18 | +0.074 |
| restaurants | how appealing? | `meang_own` | 20 | +0.083 |

No case yields a single-token prediction. The failure is a property of the
method, not of one question.

## The weaker claim fails too

An attribution method might know WHICH image tokens matter and get the
direction wrong. That would still be useful, and it is a different claim. We
tested it. We scored the MAGNITUDE of each map against the magnitude of the
token-ablation map.

| Layer | `gxa` signed / abs | `meangp_own` signed / abs | `lens_own` signed / abs |
|---|---|---|---|
| 6 | +0.003 / +0.020 | +0.007 / +0.004 | -0.027 / +0.024 |
| 18 | +0.046 / +0.024 | +0.022 / +0.013 | +0.024 / +0.025 |
| 30 | +0.020 / +0.050 | +0.033 / +0.033 | +0.004 / +0.017 |

The magnitude map does no better than the signed map. The lens does not know
which image tokens matter, and it does not know the direction either.

# The correction, 2026-08-25

The results above use a grid of 6 set sizes, 6 conditions and 3 cases. That is
about 108 tests at a nominal 95%, and about 5 false passes are expected when
nothing works. **The passes recorded above are the effect of that
multiplicity.** This section gives the corrected analysis. Read it in place
of the k-by-k tables.

## The sign of a baseline is arbitrary

A model-free image statistic has no direction. Rank the image tokens by MINUS
edge density and the token-ablation test swaps its 2 directed conditions, so
the separation flips sign exactly. Thus the strength of a baseline is `|z|`,
never the signed z.

We corrected the baseline-threshold script (`scripts/null_bar.py`) to read
`|z|`. The threshold did not move at any k, so no earlier number changed.

## The statistical-power test

The token-ablation map sets the upper bound. Where the upper bound itself does
not exceed the strongest baseline, the test has no power, and a score there
carries no information.

Two mechanisms close the test at large k. A random k-subset of the 64 blocks
of image tokens shares about k/2 blocks with the top set, so the baseline and
the method converge. And a leave-one-out map measures a marginal effect, while
a set ablation measures a joint effect. The marginal effect cannot predict the
joint effect once the blocks are redundant.

| Case | The k where the upper bound passes |
|---|---|
| subway_safety | 1, 2, 4, 8 |
| libraries | 1, 4 |
| restaurants | 1, 2, 4, 8 |

The upper bound fails at k=16 and k=32 in every case.

## One statistic, one test for each method

For each image we take the mean separation over the k values that pass the
power test. The mean over a geometric grid gives the same weight to each
doubling of the set size, so the large sets do not dominate. Then we run 1
signed-rank test for each method, and a Benjamini-Hochberg step over the
methods. The threshold is chosen inside each bootstrap sample, so the interval
carries the noise of the choice.

`scripts/deletion_auc.py` holds this analysis.

## The corrected result: nothing passes

20 images, layer 18, group 2, the full 6-feature baseline set.

| Case | Best method | Gain | 95% CI | p | p after BH |
|---|---|---|---|---|---|
| subway_safety | meangp_own | +1.57 | [-0.13, +2.26] | 0.017 | 0.086 |
| libraries | gxa | +1.18 | [-0.37, +2.65] | 0.053 | 0.133 |
| restaurants | meangp_own | +1.40 | [-1.14, +2.08] | 0.064 | 0.224 |

No method passes the criterion on any case. `lens_own`, the fitted Jacobian
lens, is last or near last on all 3 cases, and its gain is negative on 2 of
them.

The gradient methods, `gxa` and `meang_own`, hold a positive gain on all 3
cases and win on 13 to 15 images of 20. That is a weak and consistent
direction, not a result. **The sample of 20 images is the limiting
constraint**, so we now repeat every method at 60.

## A control that failed to explain the one pass

A token-ablation map is not flat in space: the middle blocks move the judgment
more than the corners. So an attribution method can score above chance while
it carries no information about the photograph.

We built the position-only map: the mean measured effect at each block, taken
over the other 19 images, and applied unchanged. `scripts/position_prior.py`
holds it.

| Test | The position-only map | The best fitted additive regression |
|---|---|---|
| Pearson r against the token-ablation map | +0.174 | +0.175 |

The paired gain of the regression over the prior is +0.001, and the
signed-rank p is 0.96. So on a correlation the spatial prior explains
everything that an additive method can do.

But the prior fails the set-ablation test at every k (-0.16 at k=4), while
`meangp_own` passes it at k=4. The 2 statistics do not measure the same
quantity: a correlation uses the whole map, and a set ablation uses the 4
extreme blocks. **The prior does not explain the one pass.**

## The upper bound for every additive method

A lens, an Rlens, a mean gradient and a gradient-times-input map are all
linear in the image-token hidden state. A ridge regression fitted on the
measured ablation effects sets the upper bound for all of them.
`scripts/linear_probe.py` holds it. The folds split by image, so a training
token and a test token never share a photograph.

| Ablation unit | Held-out r | Shuffled control |
|---|---|---|
| 256 image tokens, per token | +0.03 to +0.05 | +0.01 to +0.03 |
| 64 blocks, per block | +0.13 to +0.18 | +0.11 to +0.14 |

The block regression is the exact upper bound. A linear score of a block is
the sum of its token scores, and that sum equals the dot product with the
whole hidden state of the block.

Subtract the spatial prior from both sides, and a small image-specific part
survives: r = +0.097 at layer 24, positive on 15 images of 20. A matching test
confirms that it is specific to its own photograph.

| Layer | Own photograph | Other photographs | Rank of own (chance 10.5) | CI |
|---|---|---|---|---|
| 12 | +0.081 | +0.009 | 8.60 | [6.25, 11.20] |
| 18 | +0.045 | +0.007 | 9.35 | [7.10, 11.70] |
| 24 | +0.097 | +0.007 | 7.80 | [5.90, 9.85] |
| 30 | +0.080 | +0.003 | 7.85 | [5.60, 10.20] |

So a real image-specific additive signal exists at layer 24, and it is r about
0.10. It is far too weak to rank the extreme blocks, and it adds nothing to an
image-independent spatial prior.

## What we did next, and what each step gave

| Step | Why | What it gave |
|---|---|---|
| every method at 60 images | 20 images cannot separate a weak effect from noise | the 1-token result, and 2 halves that agree |
| a cross-question control | ask case A's question about case B's images | the control that makes the result causal |
| a set-aware search | the additive family is exhausted, so we leave it | a real set, but no per-block ranking |
| a subset regression | measure the additive share directly | about half, so the set test had half a target |
| 256 image tokens, no sum-pooling | sum-pooling assumes the effects add, and they do not | the ablation unit at which the methods show a signal |
| 9 layers | 1 layer could be chance | a band at layers 16 to 22 |

## The token-ablation map ranks content, not gaps

An ablation removes a block and leaves a gap. A map could rank the gaps rather
than the content, and the set-ablation test would not see the difference. The
substitution edit answers this: the blocks keep their place but hold the
pixels of another photograph.

Median separation z, subway_safety, 20 images:

| Edit | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|
| ablate | +6.00 | +3.34 | +3.74 | +3.59 | +3.36 | +3.45 |
| substitute | +2.10 | +1.74 | +2.18 | +2.61 | +2.81 | +1.78 |
| keep only (sufficiency) | — | — | +0.69 | +0.41 | +1.24 | — |

The ranking survives the substitution, so the token-ablation map ranks
content. The `keep only` row inverts the question: it removes everything
except the top blocks. Those blocks are necessary but they are not sufficient,
which is what a distributed and non-additive judgment predicts.

## The result is stable under the reduction

`separation_z` divides by the spread of the random samples. That spread is 4
times below its median on some images, so a single k can reach `|z| = 23`. We
ran the power-tested test with a mean over k and again with a median over k.

No method passes under either reduction. `meangp_own` on subway_safety
improves under the median (p = 0.004, BH 0.021), but its bootstrap interval
still includes zero (-0.146 to +2.10). The rank test and the interval
disagree, and our rule requires both, so it does not pass.

## A set-aware search, because the additive family is exhausted

`monocle/mask_search.py` learns a mask over the 64 blocks. The mask blends each
image-token hidden state toward the mean image-token hidden state of the same
image, so the sequence keeps its length and the whole path stays
differentiable.

At `m = 1` the gradient for a block is the centered gradient-times-input
attribution. So **1 step of this search is an attribution method that fails,
and many steps give a set**. If the search works and the 1 step does not, the
additive form is the whole failure.

Two runs find the 2 directions: one drives `c` down, one drives it up. The
score of a block is `m_up - m_down`.

20 images for each case, 80 steps, layer 0:

| Case | mean \|c0\| | drop | rise | blocks removed, down | blocks removed, up |
|---|---|---|---|---|---|
| subway_safety | 13.85 | -12.38 | +6.25 | 35.4 | 27.4 |
| libraries | 11.54 | -15.15 | +6.13 | 33.4 | 24.6 |
| restaurants | 5.32 | -7.88 | +8.85 | 32.6 | 26.3 |

The search moves the judgment by 12 to 15 logits, which is near the whole
range. **Sets that flip the judgment exist, and an optimizer finds them.** But
the L1 penalty has little effect: about half the blocks are removed. The
set-ablation test now scores the ranking.

## A map fitted on sets, not on singletons

`monocle/subset_map.py` ablates M random sets of blocks for each image and
regresses the change in `c` on the block indicators. The fit is the best
additive summary of the joint effect, in the least squares sense over the
sampled sets, and it is a Banzhaf-style value.

The token-ablation map ablates 1 block at a time, which is a marginal effect.
The set-ablation test removes a set, which is a joint effect. So a map fitted
on singletons cannot rank sets even when it is exact, and this module corrects
the target rather than the attribution method.

The module also reports a held-out `additive_r2`: the share of the set effect
that any additive map can explain.

## Criterion B at 60 images: the per-block signal is real and it is small

Criterion B scores an attribution method against the measured input ablation,
block by block. At 20 images every method was at zero. At 60 images the
intervals exclude zero, and the numbers are still small. Layer 18, 64 blocks,
60 images.

| Case | Best method | Spearman | 95% CI | Top-block Pearson | 95% CI |
|---|---|---|---|---|---|
| subway_safety | meangp_own | +0.066 | [+0.024, +0.105] | +0.083 | [+0.039, +0.125] |
| libraries | gxa | +0.075 | [+0.038, +0.113] | +0.106 | [+0.056, +0.157] |
| restaurants | meangp_own | +0.082 | [+0.034, +0.129] | +0.138 | [+0.080, +0.195] |

The sign agreement is +0.51 to +0.53 against a chance of 0.50.

`lens_own`, the fitted Jacobian lens, scores -0.023, -0.009 and +0.047. **It
recovers no signal on any case.** Every other method beats it.

These numbers agree with the ridge regression: the image-specific additive
signal is about r = 0.10, and no more. A gradient method carries it; a fitted
lens does not.

## What "small" means here

r = 0.10 explains 1 per cent of the variance of the token-ablation map. A
method that ranks the 4 strongest blocks of 64 needs the order of the
extremes, and a correlation of 0.10 does not give it. Thus the per-block
criterion and the set-ablation test agree: both are negative.

## The corpus changes what the lens says, not whether it recovers the causal map

The study started from a question about the corpus: the lens was fitted on
wikitext, which has no images, while the task concerns photographs.
`compare_wikitext_vs_mm.json` answers the first half.

The wikitext lens and the multimodal lens give almost disjoint word sets for
the same image token. The Jaccard overlap of the top 10 words is 0.00 at
layer 6, 0.036 at layer 18 and 0.070 at layer 24. The corpus decides what the
lens says.

But both lenses fail the causal criterion. `lens_own` in the table above is
the task-fitted lens, and it scores -0.023, -0.009 and +0.047 on the 3 cases.
So a better corpus moves the words and does not move the causal readout. The
2 questions are separate, and only the first one depends on the corpus.

## Why no additive method can work: the judgment saturates at once

This is the clearest measurement in the study. We ablate the strongest blocks,
random blocks and the weakest blocks, and we record the judgment.

Median change in `c`, 20 images, 64 blocks.

| Case | 1 block | 4 blocks | 16 blocks | 32 blocks |
|---|---|---|---|---|
| subway_safety | -1.69 | -2.63 | -2.07 | -5.19 |
| libraries | -0.79 | -0.92 | -2.43 | -3.22 |
| restaurants | -1.27 | -2.20 | -3.43 | -2.74 |

**One block of 64 gives 25 to 46 per cent of what half the photograph gives.**
The curve flattens at once. If the blocks added up, the value at 32 blocks
would be 32 times the value at 1 block, and it is about 2 to 4 times.

On restaurants the curve even turns back: 8 blocks move the judgment further
than 32 do. The removal of more of the photograph moves the judgment less.

`outputs/figures/saturation_curve.png` draws it.

### What this settles

| Claim | Status |
|---|---|
| A block carries a real and large marginal effect | true, and measurable |
| The effects add up | false, at every set size |
| A small set is sufficient for the judgment | false, `keep_only` is weak |
| An additive method can rank sets | false, and this is why |

The judgment is redundant. It depends on many parts of the photograph, and any
one of them nearly suffices. That is not a fault of the lens. It is a property
of the model, and it is the reason the lens cannot exist in the intended form.

## The one thing that works: it names the strongest block

The saturation curve shows that the magnitudes of the map cannot be recovered,
and the set-ablation test shows that its sets cannot be ranked. But one
question stays open, and it is the question that a user of a lens asks:

> Show me the image region that matters most.

`scripts/top_cell_hit.py` scores it. For each photograph, it tests if the
method ranks the measured strongest block first out of 64 blocks. Chance is
1 in 64, or 0.016.

60 images for each case, layer 18, 64 blocks.

| Method | subway_safety | libraries | restaurants |
|---|---|---|---|
| gradient times input | 0.050 | 0.050 | **0.183** |
| centered gradient times input | **0.083** | 0.033 | 0.100 |
| mean gradient, per position | 0.033 | 0.017 | **0.133** |
| mean gradient | 0.000 | 0.017 | 0.017 |
| Jacobian lens | 0.000 | 0.017 | 0.017 |
| — position only (control) | 0.017 | 0.000 | 0.050 |
| — best of 6 saliency features | 0.033 | 0.050 | 0.050 |

**On restaurants, gradient-times-input attribution names the exact strongest
block of 64 in 18.3 per cent of photographs.** That is 12 times chance, and it
is 3.7 times the image-independent spatial prior.

| Comparison | p |
|---|---|
| gradient times input on restaurants against chance | 2.3e-9 |
| gradient times input on restaurants against the position prior | < 0.001 |
| mean gradient per position on restaurants against the position prior | 0.010 |
| centered gradient times input on subway_safety against the position prior | 0.003 |

Every model-free feature stays at chance, and that is with the sign given,
which doubles its chance rate. The best of 6 features reaches 0.050.

The Jacobian lens is at 0.000, 0.017 and 0.017. **It never names the block.**

`outputs/figures/top_cell_n60.png` draws it.

### What this means

The method recovers the top of the ranking and nothing else. The saturation
curve predicts exactly that. One block carries a large marginal effect, so the
strongest block is a real object that an attribution method can find. The
rest of the map is saturated, and no additive score can follow it.

So a causal lens is possible in a narrow form: **it names the image region
that matters most; it does not reproduce the full map.**

Warning: the cross-question control must still pass. If `gxa` names the same
block when we ask the wrong question, it depends on the picture and not on
the judgment.

### When it works, and why

The method names the block exactly when one block stands out. We measure the
lead of the strongest block over the second, in units of the spread of the
map.

| Split | Lead of the true top block |
|---|---|
| gradient times input names it (11 photographs) | 1.59 SD |
| gradient times input misses (49 photographs) | 0.51 SD |

The hits are 3 times more peaked than the misses. The size of the top effect
also separates them: 4.85 against 3.69.

Across the 3 cases the same order holds. Restaurants has the most peaked maps
and the best rate; subway_safety has the flattest.

| Case | Mean lead | Share above 1 SD | Mean top effect | gradient times input top1 |
|---|---|---|---|---|
| subway_safety | 0.40 | 0.10 | 2.70 | 0.050 |
| libraries | 0.59 | 0.18 | 2.28 | 0.050 |
| restaurants | 0.71 | 0.20 | 3.90 | 0.183 |

Warning: 3 cases is not a test. The within-case split above is the evidence;
the table is only consistent with it. Four more cases came later, and the
across-case claim now rests on 7. See "A pre-registered test on 4 cases we
never touched".

We also tested if the method knows when it is correct. Its own lead raises the
hit rate from 0.183 to 0.267 in the top quartile (p = 0.043, 15 photographs).
That is 4 hits against 3, and 1 of 9 tests. **We do not claim a confidence
signal.**

### Warning: the 60-image token-ablation map was sum-pooled, not measured

The 60-image runs took their token-ablation map from `causal_maps60`, which
measured 256 single-token ablations. `input_gate.load_truth` then pools 4
image tokens into a block by a sum.

But the sum is exactly the operation that this page shows to be wrong.
Measured on subway_safety, 20 images:

| Pooled 256 against measured 2x2 | Value |
|---|---|
| Spearman | +0.322 |
| Pearson | +0.328 |
| slope | +0.213 |
| same strongest block | 0.100 |

The 2 targets name the same strongest block in 1 photograph of 10. So the
60-image tables above score the methods against a different object from the
20-image tables, which use a measured 2x2 map.

We now measure the 2x2 map directly at 60 images
(`outputs/_monocle/causal_maps60_g2`) and repeat the criterion and the
top-unit test against it. Until then, read the 60-image top-block numbers as a
result about the sum-pooled map.

## The cross-question control: gxa passes, meangp_own fails

The control asks the wrong question about the correct photographs. The images
and the token-ablation map stay with the case; the question and the 2 answer
tokens come from another case. `input_gate --prompt-case` runs it.

A method that depends on the judgment must lose its ranking here. A method
that depends on the picture keeps it.

Top-block rate, 60 images, 64 blocks, chance 0.016.

| Method | Correct question | Wrong question | Conclusion |
|---|---|---|---|
| **gradient times input, restaurants** | **0.183** | **0.000** | depends on the judgment |
| centered gradient times input, restaurants | 0.100 | 0.000 | depends on the judgment |
| centered gradient times input, subway_safety | 0.083 | 0.017 | depends on the judgment |
| **mean gradient per position, restaurants** | **0.133** | **0.133** | depends on the picture |
| mean gradient, restaurants | 0.017 | 0.017 | no signal |
| Jacobian lens, all 3 | 0.000 to 0.017 | 0.000 to 0.050 | no signal |

**`gxa` falls to zero of 60 under the wrong question.** That is the required
behavior of a causal readout, and it shows it.

`meangp_own` does not move. It keeps 8 hits of 60 under a question it never
received. That is the expected behavior of a per-position constant vector:
the vector does not depend on the question, so its map does not either. **We
withdraw `meangp_own` as a causal attribution method.** Its earlier k=4 pass
on subway_safety must be interpreted the same way.

### The current conclusion

| Method | Recovers image-token causal effects? |
|---|---|
| gradient times input, layer 18 | yes, for the strongest block, on restaurants |
| centered gradient times input | yes, weakly, on restaurants and subway_safety |
| mean gradient, per position | no, it depends on the picture |
| mean gradient | no |
| the fitted Jacobian lens | no, on every statistic and every case |

## The assumption-free version: 256 image tokens, no pooling at all

The pooled-map problem disappears at group 1. There the token-ablation map is
the single-token ablation, the method output is the raw per-token score, and
nothing is summed. So this is the test that carries no additivity assumption
anywhere.

Layer 18, 256 image tokens, 20 images, chance 1 in 256 = 0.0039.

| Method | subway_safety | libraries | restaurants |
|---|---|---|---|
| gradient times input | 0.000 | 0.000 | **0.150** |
| centered gradient times input | 0.000 | 0.000 | **0.100** |
| mean gradient | — | 0.000 | 0.050 |
| mean gradient, per position | — | 0.000 | 0.000 |
| Jacobian lens | 0.000 | 0.000 | 0.000 |

**On restaurants, gradient-times-input attribution names the exact strongest
image token of 256 in 15 per cent of photographs. That is 38 times chance (p <
0.001).** The mean rank it gives the true strongest image token is 92.8,
against a chance of 128.5.

`meangp_own` scores 0.000 here, which agrees with the cross-question control:
it was never a causal attribution method.

20 images cannot resolve subway_safety or libraries at this ablation unit,
because chance predicts 0.08 hits. The 60-image runs are in progress.

### Why this is the version to quote

| Version | Token-ablation map | Assumption |
|---|---|---|
| 64 blocks, 60 images | a sum of 4 image-token ablations | additivity, which is false |
| 64 blocks, 20 images | a measured 2x2 ablation | none, but small |
| **256 image tokens** | **a measured 1-token ablation** | **none** |

The 256-token version asks the plainest form of the question. It asks which
single image token moves the judgment most when it is ablated, and whether a
method can name that token.

## How much an additive method can reach: the oracle sum

Every method here is additive. It gives each image token a number, and a
block of 4 image tokens gets the sum. So the best additive method that could
exist is the sum of the measured 1-token effects. It knows the true answer for
every image token, and it only has to add.

We measured both maps on the same 60 photographs: the 1-token ablation of all
256 tokens, and the joint ablation of all 64 2x2 blocks.

| Case | Spearman | Names the top block | Top 5 | 1-block separation | Upper bound |
|---|---|---|---|---|---|
| subway_safety | +0.344 | 0.13 | 0.27 | +1.65 (+1.23 to +2.12) | +5.06 |
| libraries | +0.417 | 0.17 | 0.33 | +2.55 (+2.00 to +3.08) | +5.18 |
| restaurants | +0.377 | 0.32 | 0.48 | +2.97 (+2.41 to +3.51) | +5.81 |

**The oracle sum reaches 33 to 51 per cent of the upper bound, and it names
the strongest 2x2 block in 13 to 32 per cent of photographs.** It holds the
true per-token answer, so nothing it misses is a fault of a method. The gap is
the part of the joint effect that does not add.

This sets the threshold for every earlier result:

- At 64 blocks, half the target is out of reach of any additive method before
  the method starts.
- At 256 image tokens no sum-pooling occurs, so the threshold does not apply.
  That is why the methods show a signal there and not at 64.
- The set test at k=16 and k=32 asks for a much larger joint effect, and the
  measured upper bound itself fails the baseline threshold there.

`monocle/subset_map.py` measures the same quantity a second way. It ablates
512 random block sets for each photograph and fits a ridge regression of the
effect on the block indicators. The held-out R2 of that fit is the additive
share:

| Case | Held-out additive R2 |
|---|---|
| libraries | 0.534 |
| restaurants | 0.467 |
| subway_safety | 0.389 |

So about half of the joint ablation effect adds, and about half does not. An
additive method cannot reach the half that does not.

## The 1-token test, at 60 images and 256 image tokens

The 20-image run could not resolve 2 of the 3 cases. The 60-image run now
does, and it uses the same measured 1-token map.

### Which image token is strongest, and can a method name it

Layer 18, 256 image tokens, 60 images. Chance is 1 in 256 = 0.0039.

| Method | subway_safety | libraries | restaurants |
|---|---|---|---|
| gradient times input | 0.000 | 0.017 | **0.150** |
| centered gradient times input | **0.033** | **0.033** | **0.083** |
| mean gradient | 0.017 | 0.017 | 0.017 |
| mean gradient, per position | 0.000 | 0.000 | 0.000 |
| Jacobian lens | 0.000 | 0.000 | 0.017 |

The 60-image run repeats the 20-image number for restaurants exactly: 0.150,
which is 38 times chance. Its top-5 rate is 0.250 and its mean rank for the
true strongest image token is 80.4, against a chance of 128.5.

### The same methods under the wrong question

| Method | Case | Correct | Wrong | Mean rank, correct | Mean rank, wrong |
|---|---|---|---|---|---|
| gradient times input | restaurants | 0.150 | 0.033 | 80.4 | 161.2 |
| centered gradient times input | restaurants | 0.083 | 0.000 | 120.2 | 147.0 |
| centered gradient times input | libraries | 0.033 | 0.000 | 111.3 | 104.5 |
| mean gradient | restaurants | 0.017 | 0.017 | 94.0 | 94.1 |

`gxa` on restaurants does not only lose its hits. Its mean rank falls below
chance under the wrong question. The mean-gradient row does not move at all,
to 1 decimal, which is the required behavior of a question-independent
method.

### The directed form of the same test

A hit rate discards the magnitude. The directed statistic keeps it. For each
photograph, take the image token that the method calls the strongest
positive, and the image token it calls the strongest negative, and take the
measured effect of each:

    z = (truth[argmax] - truth[argmin]) / sd(truth over the 256 cells)

`z` is positive when the method orders the 2 extremes correctly. The upper
bound puts the token-ablation map in place of the method.
`scripts/k1_separation.py` runs it, and `tests/test_k1_separation.py` covers
it.

**Warning: this upper bound scores the map against itself.** It is the full
range of the map, and no method can reach it. The correct upper bound comes
from a second measurement, and it is +4.00, +4.55 and +7.49. See "How well
does the token-ablation map itself repeat?" below.

Layer 18, 256 image tokens, 60 images, mean over photographs.

| Method | subway_safety | libraries | restaurants |
|---|---|---|---|
| upper bound | +6.36 | +6.87 | +7.78 |
| gradient times input | -0.31 | **+0.99** | **+1.90** |
| centered gradient times input | +0.26 | **+1.03** | **+0.98** |
| mean gradient | +0.33 | +0.30 | **+0.57** |
| mean gradient, per position | -0.15 | +0.07 | +0.43 |
| Jacobian lens | +0.32 | +0.06 | +0.09 |
| position prior, leave 1 out | **+0.84** | +0.09 | +0.16 |
| shuffled map | -0.10 | +0.03 | +0.02 |

The shuffled row is at 0, as required. The position prior is model-free and
image-independent, and it wins on subway_safety alone. So the subway case has
a prompt-template bias, and no method recovers the picture content there.

### The paired control, which is the test that counts

The same 60 photographs, the same measured map, the same method, and only the
question changes. The gain is the per-photograph difference.

| Case | Method | Correct | Wrong | Gain | 95% interval | q |
|---|---|---|---|---|---|---|
| restaurants | gradient times input | +1.90 | -0.21 | **+2.11** | +1.14 to +3.13 | **0.0004** |
| libraries | centered gradient times input | +1.03 | +0.15 | +0.88 | +0.26 to +1.53 | 0.069 |
| libraries | gradient times input | +0.99 | +0.42 | +0.57 | +0.04 to +1.16 | 0.270 |
| restaurants | centered gradient times input | +0.98 | +0.26 | +0.72 | -0.07 to +1.55 | 0.273 |
| subway_safety | Jacobian lens | +0.32 | -0.14 | +0.46 | +0.00 to +0.95 | 0.270 |
| restaurants | mean gradient | +0.57 | +0.59 | -0.02 | -0.11 to +0.05 | 0.808 |
| subway_safety | gradient times input | -0.31 | -0.02 | -0.29 | -0.86 to +0.30 | 0.861 |

`q` is a Benjamini-Hochberg step over all 15 method-by-case tests. A pass
needs q below 0.05 and a bootstrap interval that excludes 0. Only restaurants
with gradient-times-input attribution passes.

The median reduction gives the same 2 leading results: restaurants `gxa`
+1.15 with an interval of +0.52 to +1.93, and libraries `cheap` +0.66 with
+0.08 to +0.95. So the result does not come from a few large photographs.

### What this changes

The set test showed that no method depends on the judgment. The 1-token test
shows that one does, on 1 case with a strict correction and on 2 cases with a
per-case correction. The 2 results do not conflict. The set test asks an
additive method to predict a joint effect that is about 79 percent
non-additive, and it has no power above k=8. The 1-token test asks the same
method the question it can answer.

## The headline result survives a split and a leave-1-out

A single mean over 60 photographs can hide a few large ones. Two checks show
that it does not.

**A split by a hash of the photograph name.** Each half is an independent
30-photograph test.

| Case | Method | Half | n | Gain | p | Wins |
|---|---|---|---|---|---|---|
| restaurants | `gxa` | 0 | 30 | +1.704 | 0.0098 | 0.70 |
| restaurants | `gxa` | 1 | 30 | +2.508 | 0.0002 | 0.77 |
| libraries | `cheap` | 0 | 24 | +0.425 | 0.165 | 0.62 |
| libraries | `cheap` | 1 | 36 | +1.187 | 0.014 | 0.64 |

Restaurants replicates in both halves on its own. Libraries reaches the
threshold in 1 half of 2, which matches its place just outside the strict
correction.

**Leave 1 photograph out.** The restaurants `gxa` gain is +2.106 over all 60.
Over the 60 leave-1-out means it runs from +1.816 to +2.215. No single
photograph carries it.

## The depth band where an image token is recoverable

The 1-token test at 1 layer could be chance. So we ran it at 9 layers, on the
same 60 photographs and the same measured map, with the wrong-question
control at every layer. Only the method changes with depth.

Restaurants, 256 image tokens, 60 photographs. The 2 columns are the
separation under the case's own question, and the gain over the wrong
question.

| Layer | `gxa` own question | `gxa` gain | `cheap` own question | `cheap` gain |
|---|---|---|---|---|
| 12 | +1.10 | +0.68 | +0.88 | +0.81 |
| 16 | +1.00 | +1.21 | +1.43 | +1.03 |
| **18** | **+1.90** | **+2.11** | +0.98 | +0.72 |
| 20 | +1.60 | +1.42 | +0.85 | +0.45 |
| 22 | +1.16 | +1.37 | +0.53 | +0.38 |
| 24 | +1.20 | +0.56 | +0.43 | +0.02 |
| 26 | +1.18 | +0.74 | +0.14 | +0.19 |
| 28 | +0.30 | +0.25 | +0.33 | +0.45 |
| 30 | +0.21 | +0.06 | +0.05 | +0.40 |

A Benjamini-Hochberg step over all 40 layer-by-method tests keeps 5:

| Layer | Method | Gain | 95% interval | q |
|---|---|---|---|---|
| 18 | `gxa` | +2.106 | +1.138 to +3.133 | 0.0011 |
| 20 | `gxa` | +1.416 | +0.678 to +2.245 | 0.0158 |
| 22 | `gxa` | +1.367 | +0.467 to +2.312 | 0.0158 |
| 16 | `gxa` | +1.212 | +0.448 to +2.035 | 0.0287 |
| 16 | `cheap` | +1.034 | +0.327 to +1.747 | 0.0458 |

**The effect is a band, not a point.** Layers 16 to 22 carry it, and it
decays to nothing by layer 28. A single chance layer would not show that.

This also places the image-token attribution against the late-layer result at
the last token position. The judgment itself becomes decodable at the last
token position at layer 42. The image-token content becomes causally
recoverable at layers 16 to 22. So the 2 stages are far apart, and a method
that recovers an image token must do so early.

## Every layer pooled: the test that decides

A single layer that passes could be the layer that we selected by chance. So
we take the mean rank of each token over all 9 layers, and we run 1 test for
each case and each method. We use ranks, not values, because the scale of a
method changes with depth. `scripts/k1_layer_profile.py --pool` runs it.

The correction is a Benjamini-Hochberg step over all 15 tests, and a pass
needs q below 0.05 and a bootstrap interval that excludes 0.

| Case | Method | Own question | Other question | Gain | 95% interval | q |
|---|---|---|---|---|---|---|
| restaurants | **`gxa`** | +1.062 | -0.306 | **+1.368** | +0.647 to +2.144 | **0.0135** |
| libraries | **`gxa`** | +1.295 | +0.297 | **+0.999** | +0.434 to +1.603 | **0.0201** |
| libraries | `lens_own` | +0.526 | +0.072 | +0.454 | -0.056 to +0.945 | 0.175 |
| libraries | `meangp_own` | +0.592 | +0.335 | +0.256 | -0.054 to +0.670 | 0.196 |
| restaurants | `cheap` | +0.448 | +0.202 | +0.246 | -0.262 to +0.750 | 0.204 |
| restaurants | `meangp_own` | +1.127 | +0.975 | +0.152 | -0.029 to +0.359 | 0.204 |
| libraries | `cheap` | +0.260 | +0.042 | +0.218 | -0.374 to +0.808 | 0.452 |
| subway_safety | `cheap` | +0.461 | +0.382 | +0.080 | -0.430 to +0.594 | 0.452 |
| subway_safety | `gxa` | +0.420 | +0.368 | +0.052 | -0.512 to +0.649 | 0.592 |
| restaurants | `meang_own` | +0.180 | +0.182 | -0.002 | -0.135 to +0.132 | 0.611 |
| subway_safety | `lens_own` | -0.154 | +0.165 | -0.319 | -0.789 to +0.161 | 0.899 |

**Gradient-times-input attribution passes on restaurants and on libraries.**
Nothing passes on subway safety, and nothing else passes anywhere.

The pooled-layer test is also the fairest test we ran. It removes the layer
choice, it holds the multiplicity at 15, and it raises libraries from a near
miss at 1 layer to a pass.

Note that `meangp_own` reaches +1.127 on restaurants under its own question.
Its wrong-question value is +0.975, so almost all of that is prompt-template
bias. The gain is what counts, and its gain is +0.152.
## What 1 backward pass gives: the cost ladder

At 64 ablation units we can measure the token-ablation map and score each attribution method against it.
The map here is the joint ablation of each 2x2 block, so nothing is pooled
on the map side. 20 photographs for each case.

| Attribution method | Forward or backward passes | subway_safety | libraries | restaurants |
|---|---|---|---|---|
| the token-ablation map | 64 forward | +5.13 | +5.12 | +5.70 |
| a fit on 512 random sets | 512 forward | +3.15 | +3.66 | +4.42 |
| the sum of the 256 1-token effects | 256 forward | +1.69 | +2.99 | +3.37 |
| **`gxa`, layers pooled** | **1 backward** | +0.13 | **+1.08** | **+1.28** |

The numbers are the 1-unit separation, in units of the standard deviation over the ablation units.

| Attribution method | Share of the ceiling |
|---|---|
| a fit on 512 random sets | 61 to 78 per cent |
| the sum of the 256 1-token effects | 33 to 59 per cent |
| `gxa`, layers pooled | 2 to 22 per cent |

**One backward pass gives about a fifth of what 512 forward passes give**, on
the 2 cases where the method recovers anything. On subway safety it gives nothing.

The ladder also shows where the loss occurs. A fit on random sets loses about a
quarter of the ceiling to the non-additive part of the effect. The sum of the 1-token effects loses
another quarter to the same cause. A gradient loses the rest, because it is a
first-order estimate of a saturated function.

## The paraphrase control: the predictor tracks the judgment, not the wording

A cross-question drop can have 2 causes. The attribution method tracks the judgment,
or any change of the prompt reduces the separation. The paraphrase separates them.

We reworded the question and kept the judgment and both answer words:

| Case | The question | The reworded question |
|---|---|---|
| restaurants | How appealing is this restaurant to eat at? | Would you want to have a meal at this restaurant? |
| libraries | How well maintained does this library building look? | What is the upkeep of this library building like? |

The photographs, the token-ablation map and the attribution method stay the same.

`gxa`, layers pooled, 60 photographs for each case:

| Case | Its own question | Reworded | Another judgment | Gain over the rewording | Gain over another judgment |
|---|---|---|---|---|---|
| restaurants | +1.062 | **+0.951** | -0.306 | +0.110 (p=0.53) | **+1.368** (q=0.014) |
| libraries | +1.295 | **+1.134** | +0.297 | +0.161 (p=0.38) | **+0.999** (q=0.020) |
| subway_safety | +0.420 | +0.701 | +0.368 | -0.281 (p=0.86) | +0.052 (q=0.59) |

**The predictor keeps 88 to 90 per cent of its separation under a reworded
question, and loses 77 to 100 per cent under another judgment.** So the
cross-question drop comes from the judgment, not from the prompt.

Each other condition moves by less than 0.3 under either change, because each
other condition had no separation to lose.

This control supports the causal claim in the intended sense. The
attribution method tracks the decision that the prompt requests, not the wording of the question
and not the content of the photograph.

## What the cross-question control does and does not change

The control swaps the question. It keeps the photographs and the token-ablation
map fixed. But 2 cases also carry different answer words, so the swap can
move the readout direction as well as the question.

| Case | Its control | Positive word | Negative word | The readout direction |
|---|---|---|---|---|
| **restaurants** | libraries | Good | Poor | **the same** |
| libraries | subway_safety | Good to Safe | Poor to Dangerous | it moves |
| subway_safety | restaurants | Safe to Good | Dangerous to Poor | it moves |

**The restaurants control is the strict one.** Restaurants and libraries share
the 2 answer words, so the logit difference that the method uses is the same vector.
Only the prompt text changes. The +1.37 gain there cannot come from a new
readout direction.

For libraries the answer words change as well, so its +1.00 gain mixes the 2
effects. The paraphrase run separates them: it rewords the question and keeps
the judgment and both answer words.

## Why subway safety fails: 2 explanations we can exclude

**It is not a weak judgment.** Ablation of the whole image moves the subway
logit difference by 5.48 logits, the largest of the 3 cases. Only 8 of 60 subway
photographs fall below the whole-image effect threshold, and that is the fewest of the 3.

**It is not a global image statistic.** We correlated the judgment with 4
whole-image quantities over the same 60 photographs.

| Case | Luminance | Contrast | Edges | Saturation |
|---|---|---|---|---|
| subway_safety | -0.108 | -0.186 | -0.056 | -0.042 |
| libraries | -0.202 | -0.018 | +0.132 | -0.068 |
| restaurants | -0.025 | +0.003 | -0.001 | +0.074 |

Not 1 of the 12 reaches significance. So the safety judgment does not track
whole-image luminance or clutter.

The remaining explanation is the spread of the effect. Subway safety puts the smallest share of its effect in 1
token (0.32 against 0.61 for restaurants), and it carries the strongest position
prior. The cause is real, and it is distributed over many tokens.

## Integrated gradients: it does not beat a plain gradient

A gradient is a first-order estimate, and the judgment saturates, so a path
integral should do better. `monocle/residual_ig.py` runs it in the residual
stream: it moves each image-token hidden state from the mean over image tokens to its own value in `steps`
stages, and it sums the gradient along the path.

**The step count matters.** At 16 steps the completeness gap is 6.21
against a range of 6.19, which means the sum misses the whole effect. At 64
steps a preliminary run falls to 0.74, or 12 per cent. So all the runs use 64
steps.

**Completeness still fails on a fifth of the photographs.** The preliminary figure
does not hold over 60. The table below gives the completeness gap of each photograph
as a share of its own range.

| Run | Median share missed | Share under 25 per cent | Share over 100 per cent |
|---|---|---|---|
| restaurants | 0.40 | 0.45 | 0.27 |
| libraries | 0.12 | 0.62 | 0.18 |
| subway_safety | 0.27 | 0.48 | 0.22 |

So the path integral holds for about half the photographs and fails on
about a fifth. Every number in this section carries that limit. The plain gradient
`gxa` needs no completeness, so the main result does not.

Layer 18, 256 image tokens, 60 photographs, with the cross-question control. The
correction is a Benjamini-Hochberg step over all 6 tests.

| Case | Condition | Own question | Other question | Gain | 95% interval | q |
|---|---|---|---|---|---|---|
| restaurants | integrated gradients | +1.655 | +0.244 | +1.411 | +0.570 to +2.321 | 0.025 |
| libraries | centered gradient x activation | +1.018 | +0.116 | +0.902 | +0.273 to +1.551 | 0.027 |
| **subway_safety** | integrated gradients | +0.895 | +0.171 | +0.724 | +0.073 to +1.382 | 0.027 |
| restaurants | centered gradient x activation | +0.941 | +0.272 | +0.669 | -0.082 to +1.495 | 0.221 |
| libraries | integrated gradients | +0.378 | +0.320 | +0.058 | -0.416 to +0.552 | 0.610 |
| subway_safety | centered gradient x activation | +0.277 | +0.422 | -0.145 | -0.762 to +0.465 | 0.610 |

**The path integral does not beat the plain gradient.** On restaurants it
reaches +1.66 against `gxa` at +1.90 on the same layer. On libraries it
reaches +0.38 against +0.99. So the extra 64 forward passes give nothing there.

**The one subway safety number does not survive its control.** It was the
only condition that moved on that case, so we ran the paraphrase beside it. The
paraphrase keeps the judgment and both answer words, and changes only the
wording. A method that tracks the judgment keeps its separation there.

| Case | Condition | Own wording | Reworded | Drop |
|---|---|---|---|---|
| subway_safety | integrated gradients | +0.895 | +0.278 | 69 per cent |
| restaurants | integrated gradients | +1.655 | +1.261 | 24 per cent |
| libraries | centered gradient x activation | +1.018 | +0.471 | 54 per cent |

The subway number falls by more than two thirds when only the words change.
It tracks the wording, not the judgment. Two facts agree with this. Its value
of +0.895 is close to the subway position prior of +0.838, which is the
strongest prior of the 3 cases. And `gxa`, which passes both controls on the
other 2 cases, recovers nothing here.

So no attribution method recovers subway safety. We reject the candidate.

**The paraphrase separates the 2 attribution methods.** Pooled over every layer,
`gxa` keeps its separation when the words change: restaurants +1.062 to
+0.951, libraries +1.295 to +1.134, and no test passes the correction. The
centered gradient-times-activation method does not: on libraries it falls from +1.018 to +0.471 at
layer 18. A method that tracks the judgment must survive new words, and
only `gxa` does.

## The Rlens question: the logit lens gives the location, not the content

A saliency map gives a location. A logit lens should give a word. So we decoded the lens
words at the 2 extreme tokens of the token-ablation map, and at 12 random image tokens of the same
photograph. The photograph is thus held constant, and only the causal role
changes. `monocle/causal_words.py` runs it, and
`scripts/causal_word_report.py` scores it.

**A causal image token does carry different words than a random image token.** At layer
18, over 60 photographs, the divergence between the extreme-token words and the random
words exceeds a within-photograph permutation null distribution:

| Role | Divergence | Null median | p |
|---|---|---|---|
| raises the judgment | 0.0605 | 0.0318 | 0.0005 |
| lowers the judgment | 0.0509 | 0.0317 | 0.0005 |

**But the 2 extreme tokens do not differ from each other.**

| Case | Divergence, extreme against extreme | Null median | p |
|---|---|---|---|
| restaurants | 0.0483 | 0.0431 | 0.128 |
| libraries | 0.0425 | 0.0437 | 0.622 |

**And the words themselves are not interpretable.** The words that mark a causal
image token at layer 18 are `gautam`, `ghaz`, `lawan`, `mathbf`, `shouting`. At
layer 24 the logit lens gives `<pad>` for most image tokens. Not 1 of them names a thing
in the photograph.

### What this result establishes

| Question | Answer |
|---|---|
| Where is the judgment located? | The gradient identifies the token. |
| In which direction does that token act? | The sign of the gradient gives it. |
| What does the model see there? | The logit lens does not give it. |

So the attribution succeeds at the level of location and sign, and it fails at the
level of content. The logit-lens words separate a strong image token from a weak one,
which is a salience signal, and they carry no information about the judgment.

This agrees with the corpus result. A lens fitted on wikitext and a lens
fitted on multimodal text give word sets that hardly overlap, at a Jaccard of
0.00 to 0.07. The corpus changes the words. It does not change the causal effect.

### The full sweep, 2026-08-25: 7 cases, 8 layers, both lenses

The 2026-07 test above used the Jacobian lens at 3 layers on 3 cases. The Jacobian lens
gives no readable words below layer 30, so that negative result was confounded: we could not tell
if the 2 extreme tokens hold the same content, or if the method could not
decode it. The R-lens exists to close that gap, so we ran the paired design.

Only the lens changes. Same 60 photographs, same seed, same token-ablation map,
the 8 fitted layers, and all 7 cases. `monocle.causal_words` writes the words
and `scripts/rlens_word_report.py` scores them. The Jacobian-lens condition at layers 12, 18 and
24 repeats the 2026-07 run, which checks the scoring script.

**The extreme tokens still do not differ from each other, under either lens, at any layer.**

| Layer | Jacobian lens, cases at q<0.05 | R-lens, cases at q<0.05 |
|---|---|---|
| 6, 12, 18, 24 | 0, 1, 0, 0 of 7 | 0, 0, 0, 0 of 7 |
| 30, 36, 42, 46 | 0, 1, 0, 0 of 7 | 0, 0, 0, 1 of 7 |
| **all 56 rows** | **2** | **1** |

A BH threshold at 0.05 over 56 rows admits about 2.8 rows by chance. Both lenses are
at chance level.

The test also has power at the relevant layer. At layer 42 each case holds 356 to
439 words, and the divergence equals the null median: libraries 0.277 against
0.278, restaurants 0.270 against 0.267. The method can detect a difference, and there is
no difference to detect.

**The R-lens does not give readable words at the layers where the image acts on the answer.**

| Layer | Jacobian lens, nonsense-token rate | R-lens, nonsense-token rate | does the image still act on the answer? |
|---|---|---|---|
| 12 | 0.298 | **0.961** | yes |
| 18 | 0.328 | **0.954** | yes |
| 24 | 0.783 | **0.867** | yes |
| 36 | 0.264 | 0.147 | no |
| 42 | 0.136 | 0.084 | no |

At the layers where the image still acts, the R-lens gives the single token `0` about 95 times in
100. That is pitfall 3 of [r-lens.md](r-lens.md): the norm rule drops the term
that discounts a massive-activation channel, and gemma-4 is a model where those
channels dominate. The Jacobian lens gives more words there, but it gives the same
multilingual nonsense tokens for every case. Its cross-case word overlap is 0.61 to 0.69,
and a readout that never changes with the judgment carries no information about the judgment.

Where the R-lens does give words, at layers 24 to 30, they do not change with
the case: `street`, `pastel`, `scooter`, `neighborhood`, `ornate`. That is 1
list for a subway entrance, a library facade and a park.

**The R-lens is the better lens, and this evidence is new.** The test compares the words at the extreme tokens
against the caption of that same crop and against the caption of another photograph.

| Layer | Jacobian lens passes | R-lens passes |
|---|---|---|
| 36 | 0 of 7 | **3 of 7** |
| 42 | 1 of 7 | **4 of 7**, q=0.009 |
| 46 | 2 of 7 | **3 of 7** |

This extends the "the image-token readout improves" row of [r-lens.md](r-lens.md) from
a synthetic quadrant to 60 street photographs on 7 tasks. But layers 36 to 46
are the layers at which the image no longer acts on the answer.

### What the full sweep establishes

**The sign is not in the content.** Both extreme tokens hold the same objects.
Only the sign of the gradient separates a token that raises the judgment from a token
that lowers it.

That result explains the question-blind hidden state. `h` is the same under any
question, because a forward readout decodes the same content at both extreme tokens. `g`
carries the whole judgment. The 2 factors of the predictor input are thus not 2
views of 1 signal. They are a location and a sign, and the logit lens decodes neither.

## The predictor needs no per-case adjustment

The mean over layers is flat: each layer has the same weight. A fitted weight
could do better, and it does not.

We built a weight over the 9 layers from the per-layer separation of one case,
and we applied it to another case. The gain is the cross-question control gain.

| Weights from | Applied to | Own question | Other question | Gain | 95% interval |
|---|---|---|---|---|---|
| flat | restaurants | +1.062 | -0.306 | +1.368 | +0.622 to +2.083 |
| libraries | restaurants | +1.334 | +0.081 | +1.253 | +0.548 to +2.054 |
| flat | libraries | +1.295 | +0.297 | +0.999 | +0.441 to +1.594 |
| restaurants | libraries | +1.398 | +0.172 | +1.226 | +0.703 to +1.783 |
| flat | subway_safety | +0.420 | +0.368 | +0.052 | -0.494 to +0.640 |
| restaurants | subway_safety | +0.631 | +0.527 | +0.105 | -0.449 to +0.693 |

The flat mean and the fitted weight fall within each other's intervals. So the
predictor needs no per-case adjustment, and the layer profile is a property of the
model, not of the case.

No weight recovers subway safety, which repeats the same result.

## When the predictor works: the judgment must be local

The predictor recovers restaurants, partly recovers libraries, and does not recover subway safety. The
token-ablation maps show why.

First, the effect that 1 token of 256 can have. We compare the strongest single-token
ablation with the ablation of the whole image.

| Case | Whole image | Strongest 1 token | Share | Top 5% of tokens |
|---|---|---|---|---|
| restaurants | 4.93 | 2.29 | **0.61** | 0.165 |
| libraries | 4.21 | 1.17 | 0.37 | 0.149 |
| subway_safety | 5.48 | 1.58 | 0.32 | 0.144 |

The numbers are medians over 60 photographs, in logit-difference units.

**One image token of 256 carries 32 to 61 per cent of the whole image.** That
is the saturation, measured at the finest ablation unit available. It is also why a set
test has little room: the first token takes most of the effect, so 16 tokens
cannot take 16 times as much.

Next, the spread of the effect. For each photograph take the share of the total
absolute effect that is in the single strongest image token, over the 256 measured
1-token ablations.

| Case | Top-1 share | Top-8 share | Effective number of tokens | `gxa` separation |
|---|---|---|---|---|
| restaurants | 0.0207 | 0.116 | 152 | **+1.90** |
| libraries | 0.0159 | 0.100 | 162 | +0.99 |
| subway_safety | 0.0148 | 0.096 | 166 | -0.31 |

The order is the same in both columns. The case whose judgment is concentrated in the
fewest image tokens is the case that an attribution method recovers.

The relation also holds within a case. Over all 180 photographs the rank
correlation between the top-1 share and the `gxa` separation is +0.210
(p=0.005). After we remove the case means it is +0.180 (p=0.016).

**So the predictor works where the judgment is local, and it fails where the
judgment is distributed.** A safety judgment depends on the whole image: the light, the
emptiness, the state of the platform. A preference for a restaurant depends on a
storefront. That is the difference that the attribution method detects.

The subway case also carries the strongest layout bias. A position prior built
from the other 59 photographs reaches +0.84 there (p=0.0002) and about +0.1 on
the other 2 cases. So subway safety has a strong mean layout and no local
cause, which is the worst case for a per-token attribution method.

## A model-based, judgment-free baseline, and it is the strongest one

Edge density and luminance test "better than an image statistic". They do not
test "tracks the judgment computation". The baseline that does is model-based and
judgment-free: the norm of the image-token hidden state. It uses the model, and it uses
no information about the question.

Median |z| of each baseline, subway_safety, 20 images:

| k | residual-stream norm, layer L24 | best image statistic |
|---|---|---|
| 4 | **0.488** | 0.404 (saturation) |
| 8 | **1.456** | 0.814 (luminance) |
| 16 | **2.691** | 1.412 (saturation) |

**The residual-stream norm exceeds every image statistic at every set size.** It nearly
doubles the threshold at k=16.

Against this threshold the last remaining condition fails too. `meangp_own` at k=4 scores
+1.183 with an interval of -0.084 to +3.039, so it no longer passes. The
cross-question control already excluded it; this is a second, independent
reason.

The ceiling still exceeds the higher threshold at k=4, 8 and 16, so the statistical-power criterion holds and
the test keeps its power.

## The mask search over sets does not rank image tokens either

The mask search moves the judgment by 12 to 15 logits, so the sets it finds are
real. But its per-token score fails the token-ablation test at the set sizes that pass the statistical-power test.

subway_safety, 20 images, statistical-power test at set sizes 1, 2, 4, 8, threshold = the residual-stream norm at layer L24.

| Condition | Gain | 95% CI | Wins | Passes |
|---|---|---|---|---|
| the token-ablation map (ceiling) | +3.654 | [+2.48, +4.68] | 20 of 20 | yes |
| mean gradient, per position | +0.904 | [-0.39, +1.68] | 12 | no |
| centered gradient x activation | +0.060 | [-1.46, +1.13] | 10 | no |
| **the mask search** | **-0.208** | [-0.57, +0.78] | 8 | **no** |
| the Jacobian lens | -0.384 | [-1.07, +0.62] | 8 | no |
| gradient x activation | -1.218 | [-1.84, -0.33] | 5 | no |

The mask ranking also has the wrong form. Its separation increases with the set
size, from 0.24 at k=1 to 2.57 at k=32, while the token-ablation map decreases, from 6.00
to 3.45. The search finds a large set, and a large set is where this test has
no power. The 2 facts agree: the mask score is uncorrelated with the token-ablation map
(Spearman +0.06, top-16 overlap 4.3 of 16 against a chance of 4.0).

`gxa` on subway_safety is significantly below the threshold. Its ranking is opposite to
the causal set here. That agrees with the top-token test, where subway_safety is
the case that it does not recover.

## The layout control: the predictor does not identify an average photograph

Some image tokens of the 16x16 grid matter in every photograph. The bottom rows hold
the pavement, the top rows hold the sky. An attribution method that learns only that
pattern would identify the extreme tokens of an average image. Then the separation would
look like a judgment readout, and it would be a readout of the layout.

The control removes the layout. The position prior of a photograph is the
mean token-ablation map of the other 59 photographs, token by token. It is a
leave-1-out mean, so no photograph contributes to its own prior. Subtract it
from the token-ablation map, and only the part specific to this photograph remains.
The strict variant also subtracts the attribution method's own prior.

`scripts/k1_position_control.py` runs all 3 variants beside the
cross-question control. There are 45 tests (3 cases x 5 conditions x 3 variants)
and 1 BH correction over all of them.

| Case | Condition | Variant | Own question | Other judgment | Gain | q |
|---|---|---|---|---|---|---|
| restaurants | gxa | raw | +1.062 | -0.306 | **+1.368** | 0.020 |
| restaurants | gxa | ablation-map prior removed | +0.991 | -0.299 | **+1.290** | 0.022 |
| restaurants | gxa | both priors removed | +1.070 | -0.265 | **+1.336** | 0.020 |
| libraries | gxa | raw | +1.295 | +0.297 | **+0.999** | 0.024 |
| libraries | gxa | ablation-map prior removed | +1.250 | +0.250 | **+1.000** | 0.022 |
| libraries | gxa | both priors removed | +1.279 | +0.261 | **+1.018** | 0.032 |
| subway_safety | gxa | raw | +0.420 | +0.368 | +0.052 | 0.592 |

The predictor loses nothing. On restaurants the gain moves from +1.368 to
+1.336, and on libraries from +0.999 to +1.018. Both stay inside the other
one's interval. No other condition passes any variant.

### The control removes what it is built to remove

`meangp_own` is a per-position constant, so it is a position prior under
another name. Its gain must collapse when the prior is removed, and it does:
+1.127 raw, +0.287 with both priors removed, on restaurants. That is the
internal check. The control has an effect, and gradient-times-input attribution survives it.

## The predictor ranks the top image tokens; it does not reproduce the full map

The 1-token separation uses 2 image tokens of 256. A full map that a person can use
must agree over the whole grid. `scripts/k1_rank_agreement.py` measures that
directly: the rank correlation between the pooled attribution and the token-ablation
map, within each photograph, with the cross-question control beside it.

There are 30 tests (3 cases x 5 conditions x 2 statistics) and 1 BH correction.

| Case | Condition | Statistic | Own question | Other judgment | Gain | q |
|---|---|---|---|---|---|---|
| restaurants | gxa | spearman | +0.056 | +0.015 | +0.041 | 0.016 |
| restaurants | gxa | top-16 recall | 0.182 | 0.153 | +0.029 | 0.030 |
| libraries | gxa | spearman | +0.059 | +0.030 | +0.029 | 0.024 |
| libraries | gxa | top-16 recall | 0.180 | 0.148 | +0.031 | 0.004 |
| subway_safety | gxa | spearman | +0.040 | +0.034 | +0.006 | 0.566 |

The gain is real and it passes the correction. The level is small. A rank
correlation of 0.056 over 256 image tokens does not reproduce the full map. The top-16 recall of
0.18 is above the cross-question value of 0.15, and both are near the 0.125
that a random selection gives.

Together, the 2 results show the form of the attribution method.

| Question | Answer |
|---|---|
| Does it identify the strongest image token? | yes, 0.150 against 0.004 by chance |
| Does it rank the whole map? | almost not at all, rho about 0.06 |

So the predictor identifies the top image tokens. It finds an extreme token, and it does not order the middle of the map.
Any figure that colors all 256 image tokens from this method shows mostly noise.
A figure that marks only the token it identifies is accurate.

### The correct reference for 0.056

A rank correlation of 0.056 appears small next to 1.0. But 1.0 is the
wrong denominator, because no ranking can exceed the agreement of the token-ablation map with
itself. We measured 2 ablation maps of the same photographs, 1 that removes an
image token and 1 that swaps its pixels. Their agreement is the level at which the
question "which image token matters" no longer has 1 answer.

| Case | `gxa` rho | Ablation against pixel swap | The share |
|---|---|---|---|
| restaurants | +0.056 | +0.191 | 0.29 |
| libraries | +0.059 | +0.174 | 0.34 |

**So the predictor reaches about a third of the agreement between 2 reasonable
definitions of the causal effect.** It still ranks only the top tokens and does not reproduce the full map, but a
third is a very different number from 6 per cent.

Warning: this is a construct ceiling, not a noise ceiling. The 2 maps measure
different quantities, and each one is bit-exact at a fixed batch size. So the
number gives the agreement between the definitions, not the noise of 1 map.

## How the gain is distributed over the photographs

A mean over 60 photographs can come from 2 of them. The main result uses a mean,
so we report the distribution.

| Case | Mean gain | Median | 10 per cent trimmed | Positive | Sign test | Top 5 share |
|---|---|---|---|---|---|---|
| restaurants | +1.368 | +1.044 | +1.167 | 0.62 | 0.046 | 0.47 |
| libraries | +0.999 | +0.737 | +0.906 | 0.65 | 0.014 | 0.46 |
| subway_safety | +0.052 | -0.279 | -0.010 | 0.47 | 0.741 | 1.22 |

This table shows 3 results.

The effect does not come from 2 photographs. A 10 per cent trimmed mean removes the 6
largest and the 6 smallest values, and it holds at +1.17 and +0.91. A sign
test, which discards every magnitude, passes on both cases.

The effect is skewed. Five photographs of 60 carry about 47 per
cent of the total gain. So the predictor works well on a minority of the photographs and works
weakly on the majority.

The predictor is not universal. It gives the correct sign on 62 to 65 photographs
of every 100. That is well above the 50 that the other question gives. It is
not a method that gives a reliable result on 1 photograph.

## A mean over wordings raises the readout, not the gain

The paraphrase control gives more than a control. `gxa` keeps 88 to 90 per
cent of its separation when the words change, so the missing tenth is wording
noise. A mean over several wordings of the same judgment must remove some
of it. The mean is over ranks, so no wording sets the scale.

**The control must grow with it.** A first analysis averaged 2 wordings on the
own-question side and kept the other-judgment side at 1 wording. That difference
alone can raise a gain, because 2 maps have less noise than 1. The correct form
gives both sides the same number of wordings and shows the whole curve.
`scripts/k1_wording_ensemble.py` does that.

| Case | Wordings | Own judgment | Other judgment | Gain | 95% interval |
|---|---|---|---|---|---|
| restaurants | 1 | +1.062 | -0.306 | +1.368 | +0.652 to +2.165 |
| restaurants | 2 | +1.227 | +0.158 | +1.068 | +0.389 to +1.780 |
| restaurants | 3 | +1.288 | +0.111 | +1.177 | +0.542 to +1.861 |
| restaurants | 4 | +1.412 | +0.209 | +1.203 | +0.595 to +1.874 |
| libraries | 1 | +1.295 | +0.297 | +0.999 | +0.448 to +1.590 |
| libraries | 2 | +1.393 | +0.140 | +1.254 | +0.691 to +1.820 |
| libraries | 3 | +1.426 | +0.196 | +1.230 | +0.674 to +1.799 |
| libraries | 4 | +1.254 | +0.492 | +0.762 | +0.145 to +1.398 |
| subway_safety | 1 | +0.420 | +0.368 | +0.052 | -0.511 to +0.606 |
| subway_safety | 2 | +0.860 | +0.384 | +0.476 | -0.118 to +1.032 |
| subway_safety | 3 | +0.645 | +0.425 | +0.220 | -0.360 to +0.800 |
| subway_safety | 4 | +0.671 | +0.459 | +0.212 | -0.380 to +0.827 |

The own side increases from +1.06 to +1.41 on restaurants and from +1.30 to
+1.43 on libraries. So the mean works: it removes wording noise, and
a predictor built this way identifies the extreme tokens better.

The other-judgment side increases too, from -0.31 to +0.21. The mean removes noise from the
other-judgment attribution as well, and a less noisy attribution finds more
of the part of the map that any question finds.

**The gain does not grow.** It stays near +1.2 on restaurants at every
count. On libraries it peaks at 2 wordings and falls again at 4. So more
wordings give a better absolute readout and no more judgment specificity. A
predictor that must be reliable on 1 photograph should use them. A study that
must show that the readout tracks the judgment gains nothing.

**The main result holds under 4 different wordings.** Every restaurants row and
every libraries row passes the correction. So the result is not a property of
the 1 question that we wrote.

**The mean over wordings does not recover subway safety.** Its own side doubles, from
+0.420 to +0.860, and the other-judgment side rises with it. No number of wordings
exceeds the threshold. An early 2-wording value of +0.476 was close; it does not
hold when the control gets the same 2 wordings and the count grows.

## The main result, computed again with no shared code

The whole result runs through 3 helpers: `pooled_ranks`, `sep_z` and `table`.
A fault in any of them would carry into every number. So we computed the main result
a second time from the stored parquet, in 20 lines that share no
code with the experiment. We built the ranks by hand. We identified the extreme tokens by
hand. The standard deviation comes from the token-ablation map of that photograph.

| Case | Own question | Other judgment | Gain | p |
|---|---|---|---|---|
| restaurants | +1.062 | -0.306 | +1.368 | 0.0009 |
| libraries | +1.295 | +0.297 | +0.999 | 0.0027 |
| subway_safety | +0.420 | +0.368 | +0.052 | 0.474 |

Every figure agrees to 3 decimals with the experiment.

We checked the token-ablation map the same way. The `causal` column of the criterion file and
the map that `monocle.causal_maps` writes agree exactly, 60 of 60 in all 3
cases. The map is identical across all 9 layer rows of a photograph.
So the criterion test scores the measured map, and not a copy of it.

## Transfer of 1 image patch into another photograph: it carries almost nothing

The token-ablation test measures the effect of a removed token. The transfer test measures the effect of content. Take the
pixels of 1 image patch of photograph A, paste them into photograph B at the same
coordinates, and measure `c` of B. Nothing is removed, so the edit changes content
and only content.

`monocle/transfer_k1.py` runs it at k=1. Each target takes 3 sources. From
each source it takes the map's highest image token, its lowest image token, and 8 arbitrary
image tokens. The statistic is `separation = dc_pos - dc_neg`, paired inside a
(target, source) pair, so the paste area and the paste distance match.

The criterion is exact: a paste of the target's own image patch moves `c` by 0.00000.

| Run | Pairs | dc(high in) | dc(random in) | dc(low in) | Separation | 95% interval | q |
|---|---|---|---|---|---|---|---|
| restaurants, token-ablation map | 180 | +0.202 | -0.035 | -0.059 | **+0.262** | +0.037 to +0.492 | 0.028 |
| libraries, token-ablation map | 180 | +0.001 | +0.004 | -0.054 | +0.055 | -0.052 to +0.168 | 0.714 |
| subway_safety, token-ablation map | 180 | -0.099 | -0.026 | -0.012 | -0.087 | -0.217 to +0.035 | 0.878 |
| restaurants, gxa | 180 | +0.062 | -0.037 | +0.089 | -0.026 | -0.206 to +0.162 | 0.878 |
| libraries, gxa | 180 | +0.010 | +0.005 | -0.023 | +0.034 | -0.087 to +0.149 | 0.714 |

One row passes, and it is small. On restaurants the strongest image token of the token-ablation
map raises the other photograph by +0.20 logits. The whole photograph is
worth 5 to 6 logits, so 1 image patch carries about 4 per cent of a judgment into
a new scene. The token that the attribution method selects carries nothing.

### What the size gap means

Compare the 2 edits on the same image token of the same photograph.

| Edit | What changes | Median strongest token |
|---|---|---|
| remove the image token of this photograph | the token is removed | 2.29 and 1.17 logits |
| swap in other pixels, in place | the token remains | **2.99 and 2.10 logits** |
| transfer the same image patch to a new photograph | the token remains | 0.20 logits, 1 case |

The numbers are restaurants first, then libraries.

**A swap in place has a larger effect than an ablation.** So the model does not only respond to the presence of a
token. Remove the pixels of an image token and the judgment moves; give the
token other pixels and it moves more.

The third row fails. The same image patch, in a new photograph, carries
almost none of its effect. The value of an image token depends on the photograph around it,
and the value does not transfer with the pixels.

## The held-out replication

Every number above comes from the same 60 photographs of each case. Thus we
ran the token-ablation experiment again on 60 new photographs, drawn with seed
20260825. The overlap with the first set is 0 of 60, checked by image id.

We refitted nothing. The mean gradient and the Jacobian lens keep their fits
on other data. Thus those conditions are now also fully out of sample. The
correction is 1 Benjamini-Hochberg (BH) step over all 14 tests.

| Case | Method | Own question | Other judgment | Gain | 95% interval | q |
|---|---|---|---|---|---|---|
| restaurants | gxa | +0.972 | -0.069 | **+1.041** | +0.369 to +1.725 | **0.0096** |
| restaurants | cheap | +0.842 | -0.045 | **+0.887** | +0.300 to +1.479 | **0.0096** |
| libraries | cheap | +0.813 | +0.100 | **+0.713** | +0.204 to +1.240 | **0.0096** |
| libraries | gxa | +0.919 | +0.135 | +0.784 | +0.107 to +1.568 | 0.088 |
| restaurants | meang_own | +0.310 | +0.260 | +0.050 | -0.084 to +0.176 | 0.422 |
| libraries | meang_own | +0.467 | +0.582 | -0.115 | -0.490 to +0.149 | 0.749 |
| restaurants | lens_own | -0.158 | -0.074 | -0.083 | -0.252 to +0.074 | 0.749 |
| libraries | lens_own | -0.251 | +0.011 | -0.261 | -0.785 to +0.218 | 0.749 |
| subway_safety | gxa | +0.810 | +0.364 | +0.446 | -0.077 to +0.952 | 0.082 |
| subway_safety | cheap | +0.270 | +0.303 | -0.032 | -0.599 to +0.522 | 0.749 |

**Restaurants replicates.** The first set gave +1.368 with an interval of
+0.647 to +2.144. The new set gives +1.041, which is inside it.

**Libraries replicates in size but not in q.** The gain is +0.784 against
+0.999 on the first set, and its interval excludes 0. The corrected p is
0.088, so it misses the threshold. The related method `cheap`, which is the
same gradient against a centered activation, passes at 0.0096. We class the
case as a replication of direction that the correction does not confirm.

**Subway safety shows a positive gain and does not pass.** Its gain rises from
+0.052 on the first set to +0.446 held out. The interval still reaches -0.08.
Over both sets subway is the weakest case each time, and it never exceeds the
threshold.

The fitted Jacobian lens still recovers nothing, and the 2 question-blind
conditions still recover nothing, on photographs that no part of this study
saw.

### The paraphrase control out of sample

| Case | Own wording | Reworded | Gain | q |
|---|---|---|---|---|
| restaurants | +0.972 | +0.683 | +0.289 | 0.640 |
| libraries | +0.919 | +0.814 | +0.104 | 0.833 |

The predictor keeps 70 per cent and 89 per cent of its separation when only
the words change, and no test passes. Thus the control holds out of sample
too.

## Two tests of why subway safety fails

The predictor passes on a preference (restaurants) and on upkeep (libraries)
and fails on safety. Two explanations are still open, and each has a test.

| Explanation | The test |
|---|---|
| the answer words `Safe` and `Dangerous` are the problem | ask the same question with `Good` and `Poor` |
| the safety judgment is not local | ask a local safety question about the same photographs |

The 2 cases that pass share the words `Good` and `Poor`, and the case that
fails does not. That is a perfect confound with n=2 against 1, so we must
break it. The first test keeps the subway photographs and the safety question
and changes only the 2 answer words of the logit difference.

The second test keeps the photographs and asks how well lit the entrance is.
Lighting is a safety cue at specific positions in an image. If the predictor
finds it, then the failure comes from the spatial extent of the question "how
safe does it look", and not from safety.

Each test needs its own token-ablation map, because `c` is a different
quantity when the words or the question change. Both maps run at 256 image
tokens over the same 60 photographs.

## Whether the predictor can signal its own reliability

The predictor gets the sign right on about 63 photographs of every 100. A
user with 1 photograph needs to know which kind they hold. A useful score must
use only what the attribution method gives, because the user does not have the
token-ablation map. `scripts/k1_confidence.py` tests 3.

| Score | What it counts |
|---|---|
| `layer_agree` | how many of the 9 layers name the pooled top image token |
| `margin` | the pooled-rank gap between the top image token and the next |
| `concentration` | the share of the absolute attribution in its top image token |

### Layer agreement is the only score that repeats

We split the photographs at the median of the score and compute the gain in
each half. Restaurants and libraries are pooled.

| Set | Score | Above the median | Below | Difference | 95% interval | p |
|---|---|---|---|---|---|---|
| first 60 | layer agreement | +1.772 | +0.748 | +1.023 | +0.081 to +1.980 | 0.017 |
| held out | layer agreement | +1.489 | +0.645 | +0.844 | -0.295 to +1.980 | 0.064 |
| first 60 | concentration | +1.064 | +1.303 | -0.240 | -1.188 to +0.774 | 0.696 |
| held out | concentration | +1.307 | +0.518 | +0.789 | -0.196 to +1.760 | 0.018 |

A photograph whose 9 layers agree is predicted about twice as well. The size
holds across the 2 sets, +1.02 and +0.84. The held-out interval reaches
-0.30, so the score is suggestive and not confirmed.

Concentration shows the opposite pattern: it fails on the first set and
passes on the held-out one. That pattern is noise, not a reliability score.

The rank correlations are weak everywhere, +0.118 on the first set and +0.146
held out, both above 0.10 in p. Thus the relation is strong enough to sort the
photographs into halves and too weak to score 1 photograph.

## Two attribution methods, not one, with little overlap

`gxa` is the gradient times the activation, which is gradient-times-input
attribution. `cheap` is the same gradient times the centered activation. They
appear to be 1 method with a small change. They are not.

| Set | Case | Correlation of the 2 separations | Rank correlation of the 2 maps |
|---|---|---|---|
| first 60 | restaurants | +0.465 | +0.122 |
| first 60 | libraries | -0.152 | +0.144 |
| held out | restaurants | +0.308 | +0.176 |
| held out | libraries | +0.047 | +0.177 |

The 2 maps agree at a rank correlation of 0.12 to 0.18 over 256 image tokens,
and their per-photograph separations agree at 0.05 to 0.47. Thus they are
close to independent attribution methods of the same photograph.

Two things follow.

The evidence is stronger than 1 statistic. Two nearly independent methods
both beat their controls, on 2 cases and on 2 sets of photographs. A single
chance statistic cannot do that.

Which one passes in a given sample is close to chance. On the first 60 `gxa`
passes and `cheap` does not; held out, `cheap` passes on both cases and `gxa`
misses the correction on libraries. This pattern is expected from 2 weak and
nearly independent methods.

## The content of the causal image token: crop it and caption it

The Jacobian-lens readout of a causal image token gives token fragments like
`gautam` and `ghaz`. No attribution method can say what the token depicts.
Thus we asked the model instead. We crop the image patch out of the
photograph, keep 1 ring of neighbour patches for legibility, and ask the model
to name the things it sees. `monocle/patch_caption.py` runs this.

Each photograph gives 6 crops: 1 at the map's highest image token (the top
token), 1 at its lowest (the bottom token), and 4 arbitrary image tokens. The
random crops come from the same photograph, so the scene, the camera and the
light never change.

### The open test finds nothing

The test is the Jensen-Shannon divergence between the word counts of 2
conditions, against a permutation of the condition labels inside each
photograph.

| Comparison | JS | Null model | p |
|---|---|---|---|
| top token against random | 0.0658 | 0.0585 | 0.187 |
| bottom token against random | 0.0446 | 0.0571 | 0.936 |
| top token against bottom token | 0.0614 | 0.0498 | 0.153 |

Nothing passes. The word lists still differ in a consistent way: the top
token names `building`, `library`, `facade`, `window`, and a random crop names
`branches`, `clouds`, `sky`, `road`, `ground`.

### The directed test finds it

The library question asks about the upkeep of a building. Thus the test that
the question implies is whether the causal image token holds the building.
The target list is fixed before the count: building, facade, wall, brick,
window, door, library, entrance, roof, column, stone, glass, sign.

| Comparison | Names a building word | Random crop | Paired difference | p |
|---|---|---|---|---|
| top token | 0.717 | 0.479 | **+0.237** | **0.0004** |
| bottom token | 0.567 | 0.479 | +0.087 | 0.090 |

**The image token that supports the judgment holds the building.** A random
image token of the same photograph holds the sky, the trees or the road.

### It is not the position in the image

A top or bottom token could simply sit where buildings sit.

| Condition | Mean row | Mean column | Names a building word |
|---|---|---|---|
| top token | 7.37 | 7.88 | 0.717 |
| bottom token | 7.20 | 6.75 | 0.567 |
| random | 7.49 | 7.80 | 0.479 |

The 3 conditions sit in the same place. The row does predict the words, at a
rank correlation of -0.281, because the bottom of an image holds the pavement.
But the conditions share a mean row, so the row cannot make the difference.

A stricter match keeps only the random crops within 3 rows of the selected
token. It gives +0.126 with p=0.060 over the 55 photographs that have such a
crop. The direction is the same, and the effect is small.

**This is the content that the Jacobian lens could not give.** The lens gives
the position and the sign. The crop says that the image token holds the
building the question is about.

### Restaurants: the open test passes too

The restaurant question asks about a place to eat. The target list is fixed
before the count: restaurant, storefront, sign, awning, menu, window, door,
entrance, table, chairs, seating, cafe, bar, counter, food.

| Comparison | JS | Null model | p |
|---|---|---|---|
| top token against random | 0.0938 | 0.0632 | **0.0015** |
| bottom token against random | 0.0782 | 0.0616 | **0.0410** |
| top token against bottom token | 0.0336 | 0.0597 | 0.996 |

| Comparison | Names a restaurant word | Random crop | Paired difference | p |
|---|---|---|---|---|
| top token | 0.683 | 0.408 | **+0.275** | **0.0002** |
| bottom token | 0.700 | 0.408 | **+0.292** | **0.0001** |

The top-token crop names `storefront`, `awning`, `sign`, `restaurant`. A
random crop of the same photograph names `car`, `road`, `sky`, `power`.

The row-matched control passes here as well: +0.230 with p=0.0010 for the top
token and +0.237 with p=0.0003 for the bottom token.

### The top and bottom tokens hold the same object

`pos_vs_neg` gives p=0.996 on restaurants and p=0.153 on libraries. Thus the
captions do not describe the 2 tokens differently from each other, and the
bottom token names the restaurant as often as the top token does, 0.700
against 0.683.

This is the same pattern that the lens words gave, now in words a person can
read.

| Question | Answer |
|---|---|
| Is a causal image token different from a random one? | yes, it holds the judged object |
| Is the top token different from the bottom token? | no |

**The causal effect sits in the object; the sign does not.** The image tokens
that decide the judgment sit on the restaurant or on the library building.
The direction of their effect comes from something the crop caption does not
name.

### The predictor's own image token also holds the object

Everything above selects the image token from the token-ablation map. That
describes the causal tokens, not the tokens that the predictor selects. Thus
we ran the crops again, with the image token selected by the method `gxa` at
layer 20, and with the same target lists and the same permutation test.

| Case | Condition | Names the judged object | Random crop | Paired difference | p |
|---|---|---|---|---|---|
| restaurants | top token | 0.700 | 0.392 | **+0.308** | **0.0000** |
| restaurants | bottom token | 0.500 | 0.392 | +0.108 | 0.046 |
| libraries | top token | 0.633 | 0.483 | **+0.150** | **0.0114** |
| libraries | bottom token | 0.617 | 0.483 | +0.133 | 0.0235 |

The open test agrees on the top token of both cases.

| Comparison | JS | Null model | p |
|---|---|---|---|
| restaurants, top token against random | 0.1090 | 0.0650 | **0.0005** |
| restaurants, bottom token against random | 0.0775 | 0.0669 | 0.123 |
| libraries, top token against random | 0.0802 | 0.0624 | **0.0310** |
| libraries, bottom token against random | 0.0531 | 0.0619 | 0.833 |

The predictor's top-token words on restaurants are `sign`, `storefront`,
`restaurant`, `awning`, `door`, `windows`. Its random words are `light`,
`power`, `line`, `surface`, `ground`, `clouds`.

**Thus the predictor selects the restaurant, not only a causal image token.**
The open test is in fact stronger here than on the token-ablation map, 0.0005
against 0.0015 on restaurants and 0.031 against 0.187 on libraries.

One thing changes between the top and bottom tokens. On the token-ablation
map the 2 tokens hold the same object, p=0.996. With the predictor the top
token is the stronger end and the bottom token is weak: p=0.0005 against
0.123 on restaurants, and 0.031 against 0.833 on libraries. `pos_vs_neg`
moves from p=0.996 to p=0.083.

| Which map selects the token | Top token finds the object | Bottom token finds it |
|---|---|---|
| the token-ablation map | yes | yes, as often |
| the predictor `gxa` | yes, more strongly | weakly |

The predictor is thus better when it says what supports the judgment than
what opposes it. A figure should mark the top image token.

## Repeatability of the token-ablation map

Every number in this experiment is scored against the token-ablation map. The
map is the ground truth of the whole study, and no one measured it twice
before. `monocle/truth_retest.py` measures it again on 12 photographs of each
case and compares the 2 measurements.

### At the same batch size the map is exact

| Retest | Correlation | Top image token repeats | Largest difference |
|---|---|---|---|
| batch 12, the size the map used | **1.000000** | 12 of 12 | **0.00e+00** |
| batch 8 | 0.63 to 0.81 | 3 to 8 of 12 | 0.86 to 1.15 |
| batch 7 | 0.58 to 0.74 | 2 to 8 of 12 | 0.86 to 1.15 |

The matched run used a shuffled token order, in a new process, and it
reproduced the stored map bit for bit on all 3 cases. Thus which tokens share
a chunk does not matter, and the run does not matter. Only the batch size
moves the map.

The map subtracts a baseline measured at its own batch size, so a constant
offset already cancels. What remains is a change of shape: at batch 8 a
token's measured effect moves by up to 1.15 logits against a map whose range
is 4.05.

### What this changes, and what it does not

**It does not change the main result.** The own-question condition and the
wrong-judgment condition score against the same stored map. A shared
distortion cancels in the difference, and the gain stands.

**It lowers the ceiling.** The experiment reported a ceiling of +7.78, +6.87
and +6.36 when we scored the token-ablation map in place of the attribution
method. That map scores itself, so the number is the full range and no method
can reach it. A second measurement at another batch size gives the attainable
ceiling.

| Case | Self-scored ceiling | Ceiling from a second measurement | Share |
|---|---|---|---|
| restaurants | 8.09 | +7.49 | 0.93 |
| libraries | 6.29 | +4.55 | 0.72 |
| subway_safety | 6.73 | +4.00 | 0.60 |

Thus `gxa` at +1.06 reaches 14 per cent of what a perfect attribution method
could do on restaurants, and not the 13 per cent the old ceiling implied. The
change is small on restaurants and large on the other 2.

**It is a rule for every future run.** Record `measure_batch`, which is in
`map_stats.parquet`, and match it in any re-measurement. `causal_maps60` used
12.

## The ablation unit: do not pool the tokens

A 2 by 2 block is easier to draw than 1 token of 256. The question is what
pooling costs. Both ablation units are assumption-free at k=1: the reference
is a measured ablation of exactly the unit that is scored.

Layer 18, 60 photographs, `gxa`, with the wrong-question control.

| Case | 64 blocks, gain | 256 tokens, gain |
|---|---|---|
| restaurants | +1.135 [+0.30, +2.01], q=0.048 | **+2.106** [+1.14, +3.13], q=0.0001 |
| libraries | +0.692 [+0.05, +1.33], q=0.066 | +0.574 [+0.04, +1.16], q=0.135 |
| subway_safety | -0.258 | -0.288 |

On restaurants the finest ablation unit gives about twice the gain, and its q
falls from 0.048 to 0.0001. On libraries the 2 units are the same inside their
intervals. Subway safety fails at both.

**Thus the signal is in single tokens, and a 2 by 2 pool loses it.** That
agrees with the form of the attribution method: it ranks 1 image token, and
it does not reproduce a field over the grid. A figure should mark the token,
not shade a block.

## A pre-registered test on 4 cases we never touched

Everything above rests on 3 cases. Two pass and 1 does not, and we explain the
split with locality. Three points do not establish a rule. `sg.CASE_RATING`
holds 4 more cases. No run in this project touched them before.

| Case | Question | Answer words |
|---|---|---|
| `road_quality` | What condition is this roadway in? | Good / Poor |
| `schools` | How good a school does this look to be? | Good / Poor |
| `parks_plazas` | How well maintained does this park or plaza look? | Good / Poor |
| `street_photography` | How appealing is this block for a street photography photoshoot? | Good / Poor |

All 4 use the same answer words as the 2 cases that pass. Thus the answer
words cannot explain a split here. Only the judgment differs.

**We write the prediction now, before we measure.** The locality claim says
that the predictor works when the cause sits in few image tokens. From the
question text alone:

| Case | Where the cause sits | Prediction |
|---|---|---|
| `road_quality` | a crack or a repaired area in 1 place on the surface | passes |
| `schools` | a school facade, like a library facade | passes |
| `parks_plazas` | the ground, the plants and the benches together | weak |
| `street_photography` | the whole block, like subway safety | fails |

Predicted order: `road_quality` about `schools` > `parks_plazas` >
`street_photography`, and `street_photography` does not exceed its control.

There is a second, sharper prediction. The token-ablation map alone gives the
top-1 share, and the top-1 share needs no attribution method. Thus we fit 1
line, from the top-1 share to the separation, on the 3 old cases. That line
then gives a number for each new case from its map alone, before any
criterion runs. We record both predictions here and score them below.

The run: 60 photographs per case at seed 20260821, the same 9 layers, group
1, and the matched wrong-question control. We fit no mean gradient for a new
case, so the `meang_own` and `meangp_own` conditions stay empty. The main
method `gxa` needs no fit, and the position control comes from the
leave-one-out prior in `scripts/k1_position_control.py`, which the stack of
token-ablation maps itself gives.

### The line, fitted and locked before the new maps

`scripts/k1_locality_predict.py --fit-only` fits 1 line. Its 2 columns are
the median top-1 share of a case and the pooled `gxa` gain of the same case.

| Case | top-1 share | pooled gain | the line |
|---|---|---|---|
| restaurants | 0.02072 | +1.368 | +1.442 |
| libraries | 0.01586 | +0.999 | +0.580 |
| subway_safety | 0.01483 | +0.052 | +0.397 |

    gain = -2.231 + 177.24 * top1_share      (r = +0.822 on 3 points)

Three points and 2 parameters give a weak line, and we state this. But the
line is now recorded, and the maps of the 4 new cases do not exist yet. Thus
the number it gives for each new case is a prediction, not a fit.

## The predictor is not edge energy, brightness or colour

An image-token attribution method could reduce to an image statistic. An
image token with a strong edge, a bright token, or a saturated token may
simply be the token the model reacts to. Every such statistic is independent
of the question, so it cannot gain under the wrong-question control. But it
can still explain the raw separation, and an attribution method that finds
only edges does not attribute the judgment.

`scripts/k1_image_stats.py` makes the picture square at 896 pixels, the same
step the model takes, and splits it into the same 16 by 16 grid. Each image
token gets 7 numbers: brightness, contrast within the token, edge energy,
colour saturation, the row, the column, and the distance from the middle.
Each one then scores as its own attribution method against the token-ablation
map. We keep the sign with the larger separation, which makes the test
generous to the statistic.

| Case | Best image statistic | Its separation | `gxa` | q of the statistic |
|---|---|---|---|---|
| restaurants | edge energy | +0.403 | **+1.062** | 0.060 |
| libraries | the row | +0.170 | **+1.295** | 0.655 |
| subway_safety | colour saturation | +0.192 | +0.420 | 0.562 |

**No image statistic passes its own test.** The best of 21 tries is edge
energy on restaurants, and it misses at q=0.060. On the same 60 photographs
`gxa` passes at q=0.008.

The second half removes them. For each photograph we fit 1 regression of the
token-ablation map on the statistics, and we keep the residual. Then `gxa`
scores against that residual, with the same wrong-question control.

| Case | Token-ablation map | Own | Wrong | Gain | 95% CI | q |
|---|---|---|---|---|---|---|
| restaurants | as measured | +1.062 | -0.306 | **+1.368** | [+0.63,+2.14] | 0.008 |
| restaurants | no pixel statistics | +1.008 | -0.371 | **+1.379** | [+0.62,+2.14] | 0.009 |
| restaurants | no position statistics | +1.071 | -0.332 | **+1.402** | [+0.70,+2.14] | 0.008 |
| restaurants | neither | +1.011 | -0.362 | **+1.374** | [+0.65,+2.14] | 0.008 |
| libraries | as measured | +1.295 | +0.297 | **+0.999** | [+0.42,+1.61] | 0.013 |
| libraries | no pixel statistics | +1.344 | +0.282 | **+1.062** | [+0.47,+1.68] | 0.014 |
| libraries | no position statistics | +1.297 | +0.308 | **+0.990** | [+0.38,+1.60] | 0.017 |
| libraries | neither | +1.324 | +0.306 | **+1.019** | [+0.43,+1.66] | 0.015 |
| subway_safety | neither | +0.463 | +0.204 | +0.259 | [-0.31,+0.82] | 0.587 |

**The gain does not move.** Remove all 7 statistics from the token-ablation
map, and the predictor still gains +1.37 and +1.02. Thus the tokens it names
are not the tokens an image statistic names.

This run also reproduces the main result a third time. The `raw` row gives
+1.368 and +0.999, which are the numbers of the pooled test, from code that
shares only `sep_z` with it.

## The sign split: both ends carry the gain

The separation adds a high half and a low half. `scripts/k1_sign_and_flip.py`
scores each half on its own, against the average image token of the same
photograph.

| Case | Method | Half | Own | Wrong | Gain | 95% CI |
|---|---|---|---|---|---|---|
| restaurants | `gxa` | high | +0.535 | +0.046 | +0.488 | [+0.06,+0.96] |
| restaurants | `gxa` | low | +0.527 | -0.353 | +0.880 | [+0.35,+1.44] |
| libraries | `gxa` | high | +0.600 | +0.111 | +0.489 | [+0.09,+0.89] |
| libraries | `gxa` | low | +0.695 | +0.185 | +0.510 | [+0.06,+0.96] |
| subway_safety | `gxa` | high | +0.147 | +0.164 | -0.017 | [-0.44,+0.41] |
| subway_safety | `gxa` | low | +0.272 | +0.203 | +0.069 | [-0.37,+0.50] |

The split doubles the tests and halves each effect, so under 1 BH over all 18
rows nothing reaches q=0.05. The intervals still show the pattern: on both
cases that pass, both halves exclude 0.

**Thus the predictor finds a strongly causal image token at each end.** The
low half is the larger of the 2 on restaurants, +0.880 against +0.488.

Compare this with the crop captions. Those say the opposite about content:
the top-token crop names the judged object and the bottom-token crop does
not. The 2 facts agree. The predictor selects a causal image token at both
ends. Only the high end holds an object that a caption can name.

## Ablation of 1 image token almost never changes the answer

A separation is a shift of a logit. The judgment is an answer. Thus we ask
the harder question: ablate the image token the predictor names, and does the
model change its answer?

The token-ablation map holds a contribution, so an ablation moves `c` by the
negative of the token value. The token that supports the answer is the top
token when `c` is positive and the bottom token when `c` is negative. We take
that token, and compare against all 256 image tokens of the same photograph.

| Case | Method | The predictor's token | An arbitrary token | Difference | q |
|---|---|---|---|---|---|
| restaurants | `gxa` | 0.067 | 0.044 | +0.023 | 0.611 |
| restaurants | `cheap` | 0.083 | 0.043 | +0.040 | 1.000 |
| libraries | `gxa` | 0.017 | 0.009 | +0.007 | 0.737 |
| libraries | `cheap` | 0.017 | 0.009 | +0.007 | 0.737 |
| subway_safety | any | 0.000 | 0.000 | 0.000 | 0.404 |

**One image token of 256 flips the answer on 0 to 8 per cent of
photographs.** The predictor does not beat chance there. On subway safety no
single image token flips the answer.

This is the limit of the whole line of work. The measured `c` runs from -13.8
to +12.8, and the strongest single image token moves it by about 2. The
margin is far larger than 1 token, so the judgment does not depend on a
token. The predictor ranks a contribution, not a decision.

## Two methods combined do not beat 1

`gxa` and `cheap` agree at a map correlation of only 0.12 to 0.18, so their
mean rank should be better than both. It is not.

| Case | Half | `gxa` gain | `cheap` gain | Mean rank gain |
|---|---|---|---|---|
| restaurants | high | +0.488 | +0.324 | +0.578 |
| restaurants | low | +0.880 | -0.077 | -0.056 |
| libraries | high | +0.489 | -0.017 | +0.464 |
| libraries | low | +0.510 | +0.235 | +0.485 |

The mean rank wins on 1 half of 4. It loses by a large margin on the
restaurants low half, where `gxa` alone gives +0.880 and the pair gives
-0.056. `cheap` is not weak noise that cancels. It is a different method, and
the mean of the 2 discards what `gxa` contains.

## The predictor also predicts a content map, not only an ablation-effect map

An ablation removes a token, and a predictor that matches the effect could
respond only to the removal. Thus we measured a second token-ablation map with
the same code and 1 flag changed. `--reference substitute` gives each image
token another photograph's pixels and removes nothing. The same criterion and
the same wrong-question control then ran against that map.

| Case | Method | Own | Wrong | Gain | 95% CI | q |
|---|---|---|---|---|---|---|
| restaurants | `gxa` | +0.531 | -0.434 | **+0.965** | [+0.30,+1.67] | **0.046** |
| restaurants | `cheap` | +0.396 | +0.117 | +0.280 | [-0.36,+0.97] | 0.455 |
| libraries | `gxa` | +1.013 | +0.210 | **+0.803** | [+0.25,+1.35] | **0.046** |
| libraries | `cheap` | +0.215 | +0.037 | +0.178 | [-0.37,+0.73] | 0.455 |
| restaurants | `meangp_own` | +0.759 | +0.763 | -0.005 | [-0.27,+0.26] | 0.672 |
| libraries | `meangp_own` | +0.356 | +0.310 | +0.047 | [-0.21,+0.36] | 0.487 |

**`gxa` passes against the content map on both cases.** The position prior
still fails, so the control still discriminates.

### The 2 maps are not the same map

`scripts/hole_vs_content_map.py` compares them token by token, inside each
photograph.

| Case | n | Pearson r | Spearman | Same top token | Same bottom token |
|---|---|---|---|---|---|
| restaurants | 60 | +0.292 | +0.191 | 0.30 | 0.17 |
| libraries | 60 | +0.219 | +0.174 | 0.07 | 0.12 |

Chance for a matched top token is 1 in 256, so 0.30 and 0.07 are far above
chance and far below agreement. The 2 references make different maps.

### And the predictor predicts the part they do not share

`scripts/k1_hole_content_residual.py` fits 1 regression of the content map on
the ablation-effect map for each photograph, keeps the residual, and scores
the predictor again.

| Case | Token-ablation map | Own | Wrong | Gain | 95% CI | q |
|---|---|---|---|---|---|---|
| restaurants | as measured | +0.531 | -0.434 | +0.965 | [+0.29,+1.65] | 0.037 |
| restaurants | the ablation-effect map removed | +0.300 | -0.362 | **+0.662** | [+0.09,+1.24] | 0.064 |
| libraries | as measured | +1.013 | +0.210 | +0.803 | [+0.26,+1.37] | 0.037 |
| libraries | the ablation-effect map removed | +0.687 | +0.125 | **+0.562** | [+0.05,+1.10] | 0.064 |

About two-thirds of the gain is in the part of the content map that the
ablation-effect map does not carry. Both intervals exclude 0, and both miss
q=0.05 by a small margin.

**Thus the predictor is not fitted to 1 definition of causality.** It finds a
causal image token when a token is removed, and it finds one when a token
changes. The 2 tests still disagree about which token.

## Subway safety: 2 more explanations ruled out, and 1 new fact

Subway safety is the case where the predictor fails. Two candidates stayed
open: the answer words, and the spatial extent of the question. We measured a
new token-ablation map for each, with the same 60 photographs and the same
code.

**Candidate 1: the answer words.** Subway safety is the only case that uses
`Safe` and `Dangerous`. Both cases that pass use `Good` and `Poor`. Thus we
ran subway safety again with `--labels Good Poor`.

**Candidate 2: the spatial extent of the question.** "How safe does this
station entrance look?" depends on the whole image. Thus we asked a local
question of the same photographs: "How well lit is this station entrance?"
The answer words stayed `Good` and `Poor`.

| Test | Own | Wrong | Gain | 95% CI | q |
|---|---|---|---|---|---|
| as it was | +0.420 | +0.368 | +0.052 | [-0.51,+0.61] | 0.947 |
| `Good` and `Poor` | +0.575 | +0.511 | +0.064 | [-0.53,+0.67] | 0.821 |
| "how well lit" | +0.798 | +0.665 | +0.134 | [-0.44,+0.70] | 0.513 |

**Neither candidate holds.** The answer words are not the cause, and a local
question does not make the case pass.

The new fact explains the failure. The top-1 share says how much of the total
effect is in 1 image token, and it is measured from the map, not from an
attribution method.

| The 60 subway photographs, asked | top-1 share | strongest image token |
|---|---|---|
| how safe does it look | 0.01483 | 1.584 |
| how good is it, `Good`/`Poor` | 0.01474 | 1.287 |
| how well lit is it | 0.01555 | 1.327 |
| restaurants, for a scale | 0.02072 | 2.286 |

**The question that sounds local does not make a local map.** Ask about the
light of a station entrance, and the effect still spreads over the image. The
share is 0.0156, against 0.0207 for restaurants.

Thus locality is a property of the scene, not of the words. A station
entrance is a wide, dim, repeated scene, and every judgment of it depends on
the whole image. A block of shopfronts holds its judgment in a storefront. No
wording changes that.

The locality line predicts the "how well lit" gain at +0.525 from its map
alone, and the measured gain is +0.134. Both are inside the noise of a line
fitted on 3 points.

## Why the gradient is the factor that depends on the question

The prompt puts the picture first. The rendered text starts
`<bos><|turn>user\n<|image|>` and only then asks the question. `gemma-4` is a
causal decoder, so a token at an image position attends to nothing that comes
after it.

**Thus the residual stream at an image token cannot hold the question.** We
measured it. The test saves the residual stream under the task's own question
and under another case's question, for the same 60 photographs.

| Layer | cos(own, wrong) | Relative gap | cos(own, another photograph) |
|---|---|---|---|
| 18 | 0.999532 | 0.031 | +0.221 |
| 20 | 0.999452 | 0.034 | +0.205 |
| 42 | 0.997006 | 0.078 | +0.003 |

The 2 questions give the same vector to 4 decimal places. The small gap is
bf16 arithmetic: the 2 prompts have different lengths, so the kernels tile
differently. For comparison, 2 different photographs sit at 0.22 and 0.00.

This determines which factor of `gxa` carries the question.

| Factor of `<g_p, h_p>` | Source | Depends on the question |
|---|---|---|
| `h_p`, the hidden state | the forward pass, before the question | **no** |
| `g_p`, the gradient | the backward pass, from the answer | **yes** |

**So every attribution method built from `h` alone is question-blind by
construction.** That includes the fitted Jacobian lens, which decodes `h`
through a linear map. This explains a pattern that we saw and could not
explain. `lens_own` never shows a gain.
`meang_own` and `meangp_own` score the same to 3 decimals under the wrong
question. Only the methods that use `g` change.

This also changes what a fitted probe on `h` can mean. Such a probe can only
answer how much of the map a question-blind method finds. Its wrong-question
control is 0 by construction, not by evidence, and we must not report it as a
passed control.

Warning: do not report a wrong-question control on an attribution method that
uses `h` only. The control is empty. Only a method that uses `g` has a real
control.

### The prediction from the token-ablation maps, recorded before the tests ran

Three of the 4 new token-ablation maps were complete first. We put their
top-1 shares through the linear fit, which was fixed in advance, and
recorded the predictions before any attribution method ran on them.

| Case | top-1 share | Quartiles | The linear fit predicts |
|---|---|---|---|
| `road_quality` | 0.01984 | 0.0161 to 0.0266 | **+1.285** |
| `parks_plazas` | 0.01758 | 0.0157 to 0.0207 | **+0.884** |
| `schools` | 0.01676 | 0.0141 to 0.0203 | **+0.739** |

The map already disagrees with the guess that we made from the question text.
We guessed `road_quality` about equal to `schools`, then `parks_plazas`. The
maps put `parks_plazas` above `schools`. So the 2 predictions are not the
same prediction, and the results can separate them.

### The result on the 4 new cases

The same criterion test, the same 9 layers, the same pooled rank, and the
matched wrong-question control. One BH family over all 12 tests, which is 4
cases by 3 methods.

| Case | Method | Own question | Wrong question | Gain | 95% CI | q |
|---|---|---|---|---|---|---|
| `parks_plazas` | `gxa` | +0.799 | -0.024 | **+0.822** | [+0.23,+1.45] | 0.054 |
| `road_quality` | `cheap` | +1.362 | +0.417 | **+0.945** | [+0.31,+1.62] | 0.054 |
| `road_quality` | `gxa` | +1.071 | +0.412 | **+0.659** | [+0.09,+1.23] | 0.054 |
| `schools` | `gxa` | +0.303 | -0.088 | +0.391 | [-0.15,+0.94] | 0.256 |
| `schools` | `cheap` | +0.393 | +0.161 | +0.232 | [-0.32,+0.83] | 0.422 |
| `parks_plazas` | `cheap` | +0.297 | +0.343 | -0.047 | [-0.67,+0.63] | 0.562 |
| `street_photography` | `gxa` | +0.258 | +0.147 | +0.111 | [-0.48,+0.68] | 0.490 |
| `street_photography` | `cheap` | +0.205 | +0.111 | +0.094 | [-0.40,+0.62] | 0.490 |
| all 4 | `lens_own` | -- | -- | +0.10 or less | -- | 0.256 or more |

**Two of 4 cases that no earlier run touched give a significant gain.**
`road_quality` and `parks_plazas` both have a confidence interval that
excludes 0 at q=0.054 in a 12-test family. `schools` and
`street_photography` do not.

The fitted Jacobian lens fails on all 4, as it fails on the first 3.

### Scoring the 2 predictions

**The prediction from the question text was partly right.** We guessed
`road_quality` about equal to `schools`, then `parks_plazas`, then
`street_photography` last. The measured order is `parks_plazas`,
`road_quality`, `schools`, `street_photography`. The guess got the last place
right and put `schools` far too high.

**The linear fit did better.** It used only the token-ablation maps, never an
attribution method.

| Case | top-1 share | The linear fit predicted | Measured `gxa` gain |
|---|---|---|---|
| `road_quality` | 0.01984 | +1.285 | +0.659 |
| `parks_plazas` | 0.01758 | +0.884 | +0.822 |
| `schools` | 0.01676 | +0.739 | +0.391 |
| `street_photography` | 0.01548 | +0.512 | +0.111 |

| Statistic | Over the 4 new cases | Over all 7 |
|---|---|---|
| Spearman | **+0.800** | +0.750 |
| Pearson | +0.736 | +0.719 |
| Mean absolute error | 0.359 | -- |

**The linear fit puts the 4 new cases in nearly the right order.** It swaps
only the top 2, whose predictions sit 0.40 apart and whose results sit 0.16
apart. It over-predicts every case by about 0.36, which is the expected
behavior of a line fitted on 3 points with restaurants at the top.

So the locality claim no longer rests on 3 cases only. It predicts, out of
sample, from a quantity that no attribution method touches.

| The claim | Its evidence now |
|---|---|
| the attribution works on more than 2 cases | 4 cases pass of 7 tried |
| locality says which cases | rank correlation +0.75 over 7 cases |
| the answer words do not decide | 6 of 7 cases share `Good`/`Poor` |

### The layout control on the new cases

The first 3 cases passed a control that removes a position prior. The prior
is built for each photograph from the other 59, so it holds no information
about that photograph. `raw` is the plain gain. `truth_resid` removes the
prior from the token-ablation map. `both_resid` removes it from the map and
from the attribution method. One BH family over all 36 tests.

| Case | Method | raw | prior removed from the map | prior removed from both |
|---|---|---|---|---|
| `road_quality` | `gxa` | +0.659 | +0.623 | +0.530 |
| `road_quality` | `cheap` | +0.945 | +0.850 | +0.796 |
| `parks_plazas` | `gxa` | +0.822 | +0.722 | **+0.960** |
| `parks_plazas` | `cheap` | -0.047 | -0.061 | +0.107 |
| `schools` | `gxa` | +0.391 | +0.445 | +0.546 |
| `street_photography` | `gxa` | +0.111 | +0.054 | +0.665 |

**The layout is not the explanation on either new case that passes.** The
`road_quality` gain drops by a fifth and its interval still holds. The
`parks_plazas` gain rises.

One row needs care. `street_photography` moves from +0.111 to +0.665 when
the prior is removed from both sides. It is 1 row of 36 and its plain gain is
not significant, so we record it as a variant that moves, and not as a result. A new set
of photographs must show it again.

## The model names the kind of object, not the image token

The crop captions show that a causal image token holds the judged object.
They do not show that the model can name that image token as its reason. So
we asked the model.

`monocle/self_report.py` runs 3 steps for each photograph. It asks the case
question and records the answer. It then asks a follow-up question with the
same picture: name the one thing you can see that most caused that answer.
It then captions the image token that gradient-times-input attribution
selects and 4 random image tokens of the same photograph.

The test is an overlap. A hit means that a content word of the self-report
also appears in a crop caption. The random crops come from the same
photograph, so the scene never changes. One BH family over all 8 tests.

| Case | Answer | Statistic | The selected crop | A random crop | Paired | q |
|---|---|---|---|---|---|---|
| restaurants | low | any word | 0.250 | 0.125 | +0.125 | 0.094 |
| restaurants | low | share of words | 0.186 | 0.072 | +0.115 | 0.094 |
| restaurants | high | any word | 0.200 | 0.125 | +0.075 | 0.219 |
| restaurants | high | share of words | 0.122 | 0.072 | +0.051 | 0.275 |
| libraries | low | any word | 0.117 | 0.079 | +0.037 | 0.338 |
| libraries | high | any word | 0.067 | 0.079 | -0.013 | 0.338 |

**No row passes the correction.** The 2 restaurants rows at the low answer
come closest, at q=0.094 and about double the rate of a random crop.

**The overlap is low everywhere.** The model's stated reason falls in the
selected crop on 7 to 25 photographs of 100.

The reasons the model gives are plausible. On restaurants it names
`restaurant`, `sign`, `signage`, `storefront`, `outdoor`, `seating`,
`graffiti`, `facade`. On libraries it names `clean`, `facade`, `exterior`,
`sidewalk`, `brickwork`, `maintained`, `construction`. Those are the same
kinds of word that the selected crops hold.

So the 2 agree about the kind of object and not about the location. The
attribution selects an image token that holds a storefront, and the model
says that a storefront caused its answer. For a given photograph the selected
image token is usually not the storefront that the model means.

| Question | Answer |
|---|---|
| Does the selected image token hold the judged object? | yes, 0.70 against 0.39 |
| Does the model name that kind of object as its reason? | yes |
| Is it the same image token, photograph by photograph? | not more than by chance |

This result has the same form as the earlier activation-addition result. The
model can say what kind of object decides a judgment. It cannot say where.

Warning: the reason the model gives is often a proper name, such as
"The Mercury Lounge" or "Nice Ices sign". No crop caption holds those words,
so this test is hard by construction and it under-counts real matches.

## All 4 cases that pass: the selected image token holds the judged object

The crop test now runs on every case that passes, with the image token
selected by the attribution method `gxa` and never by the token-ablation
map. Each target list is fixed before the count and comes from the question,
not from the words we saw. The random crops come from the same photograph.
One BH family over all 8 tests.

| Case | Answer | Names the judged object | A random crop | Paired | q |
|---|---|---|---|---|---|
| restaurants | high | 0.700 | 0.392 | **+0.308** | **0.0003** |
| road_quality | low | 0.683 | 0.396 | **+0.287** | **0.0003** |
| road_quality | high | 0.633 | 0.396 | **+0.237** | **0.0003** |
| parks_plazas | high | 0.600 | 0.371 | **+0.229** | **0.0010** |
| libraries | high | 0.633 | 0.483 | **+0.150** | **0.0182** |
| libraries | low | 0.617 | 0.483 | **+0.133** | **0.0313** |
| restaurants | low | 0.500 | 0.392 | +0.108 | 0.0521 |
| parks_plazas | low | 0.333 | 0.371 | -0.037 | 0.6517 |

**Six of 8 pass, and every case that passes its control passes at its high
answer.** The open Jensen-Shannon test agrees: p=0.0005 on restaurants and
road_quality, p=0.0030 on parks and plazas, p=0.0310 on libraries.

The words show what each image token holds.

| Case | The selected image token's words at the high answer | A random crop of the same photograph |
|---|---|---|
| restaurants | `sign`, `storefront`, `restaurant`, `awning` | `light`, `power`, `line`, `ground` |
| road_quality | `surface`, `asphalt`, `paved`, `crack`, `pavement` | `sky`, `building`, `trees`, `clouds` |
| parks_plazas | `bushes`, `tree`, `green`, `leaves`, `railing`, `lamp` | `surface`, `door`, `buildings`, `roof` |
| libraries | `sidewalk`, `crack`, `wall`, `arched`, `window` | `car`, `ground`, `leaves`, `tire` |

**The attribution selects the object that the question is about**, in every
case where the attribution passes its control.

### The low answer is a different kind of object

`parks_plazas` is the clearest. Its high answer names `bushes`, `tree`,
`green`, `leaves`. Its low answer names `ground`, `road`, `curb`, `asphalt`,
`debris`, and it fails the park-word test at -0.037.

So the object that raises the rating of a plaza is its plants and its
furniture, and the object that lowers it is the roadway and the debris.
Those are 2 different objects, and 1 target list cannot hold both.

The 2 answers behave differently by case.

| Case | High answer | Low answer |
|---|---|---|
| road_quality | the road surface | the road surface, as strongly |
| libraries | the building | the building |
| restaurants | the storefront | the storefront, weakly |
| parks_plazas | the plants | the road and the debris |

Where the judged object fills the image, as a roadway does, both answers sit
on it. Where the judgment is about an object in a scene, the low answer
drifts away from that object.

## The held-out replication of the 2 new cases, and what it costs the claim

`road_quality` and `parks_plazas` both passed on their first 60 photographs.
We measured a token-ablation map for 60 more photographs of each, sampled
with seed 20260825, and ran the same criterion test. The overlap with the
first set is 0 of 60 on both cases, checked by image id.

| Case | Method | Own question | Wrong question | Gain | 95% CI | q |
|---|---|---|---|---|---|---|
| `parks_plazas` | `cheap` | +1.193 | -0.155 | **+1.348** | [+0.71,+2.06] | **0.0007** |
| `road_quality` | `cheap` | +0.733 | +0.196 | +0.537 | [-0.12,+1.21] | 0.197 |
| `parks_plazas` | `gxa` | +0.758 | +0.738 | +0.020 | [-0.50,+0.56] | 0.700 |
| `road_quality` | `gxa` | +0.427 | +0.430 | -0.003 | [-0.58,+0.58] | 0.700 |

**`gxa` does not replicate on either new case.** `cheap` passes on
`parks_plazas`, and passes strongly. `road_quality` fails on both methods out
of sample.

### The whole out-of-sample record, in 1 table

| Case | First set `gxa` | First set `cheap` | Held out `gxa` | Held out `cheap` |
|---|---|---|---|---|
| restaurants | **+1.37** | +0.25 | **+1.04** | **+0.89** |
| libraries | **+1.00** | +0.22 | +0.78 (q=0.098) | **+0.71** |
| parks_plazas | **+0.82** | -0.05 | +0.02 | **+1.35** |
| road_quality | **+0.66** | **+0.95** | -0.00 | +0.54 |
| schools | +0.39 | +0.23 | -- | -- |
| street_photography | +0.11 | +0.09 | -- | -- |
| subway_safety | +0.05 | +0.08 | +0.45 (q=0.082) | -0.03 |

Bold is a pass at q<0.05 in its own family.

**Three of the 4 cases replicate on photographs that no fit saw**, and each
one replicates through at least 1 method. `road_quality` does not.

**But the method that passes does not repeat.** `gxa` passes on restaurants
on both sets. On libraries it passes on the first set and `cheap` passes on
the second. On parks and plazas `gxa` passes on the first set and only
`cheap` passes on the second. On road quality both pass on the first set and
neither passes on the second.

This is the consequence of the earlier result that the 2 methods barely
overlap: their maps correlate at 0.12 to 0.18, and each is weak. Two weak,
nearly independent methods that measure the same quantity will each pass in
some samples and not in others.

### What the claim should say now

| Statement | Support |
|---|---|
| an image-token attribution beats its control on 4 of 7 judgments | first set, supported |
| it replicates on new photographs | 3 of the 4, through some method |
| `gxa` in particular replicates | restaurants only, of the 4 |
| road quality passes | first set only, and it does not repeat |
| locality predicts which cases pass | first-set gains, rank +0.75 over 7 |

**So the supported claim is a family of weak attribution methods, not 1
method.** The family beats its control out of sample on 3 judgments of 7. If
we name the method before the sample, the claim becomes much weaker.

Note: do not quote the gain of a single method as the result of a case
unless the same method passes on a held-out set. Only restaurants meets that
criterion today.

## The predictor: how much does gradient-times-input attribution leave unreached?

`gxa` is analytic. It sums `g_p * h_p` over the 3840 dimensions with a weight
of 1 on each. A ridge regression learns the weights instead, and its result
shows what the whole linear family can reach.

`scripts/k1_fitted_probe.py` fits it. The rules:

- The split groups by photograph. All 256 image tokens of a photograph go to
  the train side or to the test side together. A split over image tokens
  leaks.
- The per-dimension scale comes from the train fold only, folded through the
  normal equations as a diagonal.
- The final row fits on 60 photographs and tests on 60 that share no image.
- `alpha` is chosen on the train set by an inner split over photographs, and
  never on the held-out set.
- A photograph-permutation null model gives z=+0.148, so the pipeline itself
  adds no signal.

Three attribution methods run through the same pipeline, on 3 layers.

| Case | Layer | Method | Own question | Wrong question | Gain | Identifies the top image token | q |
|---|---|---|---|---|---|---|---|
| restaurants | 18 | `gxa`, not fitted | +1.897 | -0.209 | **+2.106** | 0.150 | 0.0006 |
| restaurants | 18 | `g*h`, fitted | **+2.210** | +1.066 | +1.144 | **0.167** | 0.0148 |
| restaurants | 18 | `h` only, question-blind | +0.419 | +0.302 | +0.117 | 0.050 | 0.280 |
| restaurants | 20 | `gxa`, not fitted | +1.605 | +0.188 | +1.416 | 0.100 | 0.0076 |
| restaurants | 20 | `g*h`, fitted | **+2.140** | +0.909 | +1.231 | **0.117** | 0.0043 |
| restaurants | 20 | `h` only, question-blind | +1.459 | +1.350 | +0.108 | 0.083 | 0.280 |
| libraries | 18 | `gxa`, not fitted | +0.994 | +0.420 | +0.574 | 0.017 | 0.202 |
| libraries | 18 | `g*h`, fitted | **+2.084** | +0.682 | **+1.402** | 0.050 | 0.0128 |
| libraries | 20 | `gxa`, not fitted | +1.096 | +0.100 | +0.996 | 0.050 | 0.0230 |
| libraries | 20 | `g*h`, fitted | **+2.346** | +1.162 | **+1.184** | **0.083** | 0.0128 |
| both | 42 | every method | -- | -- | 0.23 or less | 0.00 to 0.02 | 0.47 or more |

Every `g*h` and `h` row is held out. The `gxa` row needs no fit, so it uses
the same 60 photographs on which the predictor was tested.

### What the fit gains

**The predictor raises the separation by a third and the gain holds out of
sample.** Restaurants goes from +1.61 to +2.14 at layer 20, libraries from
+1.10 to +2.35. Both hold on 60 photographs that the fit never saw.

It also identifies the exact strongest image token more often: 0.167 against
0.150 on restaurants at layer 18, and 0.083 against 0.050 on libraries at
layer 20. Chance is 0.004.

**Libraries is the case where the fit matters most.** At layer 18
gradient-times-input attribution fails its control at q=0.20 and the
predictor passes at q=0.013, with the gain up from +0.57 to +1.40.

**Layer 42 gives no gain for any method.** This confirms the layer range from
the layer profile: the token-ablation map is recoverable at layers 16 to 22
and not at late layers.

### The question-blind probe, and what it shows

`h` at an image token does not change with the question. So an `h` probe is
question-blind, and its gain must be 0. It is: +0.117, +0.108, +0.053,
+0.108.

But its own-question side is informative. At layer 20 on restaurants it
reaches **+1.459**, against +1.605 for `gxa`. A method that never receives
the question gets almost all of the raw separation.

The 3 numbers side by side, at layer 20 on restaurants:

| Method | Separation | Depends on the question |
|---|---|---|
| `h` probe, question-blind | +1.459 | no |
| `gxa`, own question | +1.605 | yes |
| `gxa`, wrong question | +0.188 | yes, the wrong one |

**The wrong question is worse than no question.** A question-blind method
beats a misdirected one by 7 times. So the control does not only measure
whether the question helps. It measures that a gradient taken from the wrong
answer points at the wrong image tokens.

That also shows what the gain consists of. Most of the raw separation is
question-blind: an image token that matters for the restaurant judgment
mostly matters under any question. The gain lies in the part that the
question changes, and that part is small and real.

### The 2 by 2 design: the map information under the own and the wrong question

A probe fitted on the gradient of the wrong question, tested against the own
token-ablation map, separates the 2 interpretations.

| Case | Layer | Fitted on the own question | Fitted on the wrong question | Null model |
|---|---|---|---|---|
| restaurants | 18 | **+2.557** | +1.312 | +0.148 |
| restaurants | 20 | **+2.914** | +1.104 | +0.148 |
| libraries | 18 | **+1.576** | +0.330 | +0.148 |
| libraries | 20 | **+2.011** | +0.975 | +0.148 |

A wrong-question fit still beats the null model, so part of the map is
reachable with any question. The own-question fit is 2 to 5 times better.
**So the gradient of the right question carries information about the map
that no other question carries.**

# The retrospective: every experiment, and its support

This page follows the order in which the work happened, so a person who reads
it cannot see the overall structure. This section is the summary of claims. Every
experiment we ran occupies 1 row. A row says what it asked, what it found,
and whether the result is supported.

## 1. The attribution methods we tried

| Attribution method | What it is | Recovers the judgment? |
|---|---|---|
| `gxa` | `<g_p, h_p>`, 1 backward pass | **yes, on 4 of 7 cases** |
| `cheap` | `<g_p, centred h_p>` | **yes, and on different cases** |
| fitted probe on `g * h` (the predictor) | ridge, grouped by photograph | **yes, and best of all** |
| fitted probe on `h` | ridge, no gradient | no, question-blind by construction |
| `lens_own` | the fitted Jacobian lens | **no, on any of 7 cases** |
| `meang_own` | `<mean g over a case, h_p>` | no, it responds to the prompt template |
| `meangp_own` | a per-position mean gradient | no, it is the prompt template itself |
| integrated gradients on the residual stream | 64 steps from a baseline | no, it follows the wording |
| a search fitted on sets | fitted on sets, not single tokens | no |
| 7 image statistics | edges, light, colour, layout | no |

## 2. The ablation units we tried

| Ablation unit | Result |
|---|---|
| a set of k image tokens, k=4 to 32 | nothing passes, the judgment saturates at 1 image token |
| a 2 by 2 block, 64 tokens | passes on restaurants, half the gain |
| **1 image token of 256** | **the ablation unit that works** |

The saturation is the reason. One image token of 256 carries 32 to 61 per
cent of the whole image, so a set of 16 cannot carry 16 times as much.

## 3. The controls the attribution passed

| Control | What it removes | Outcome |
|---|---|---|
| the wrong question | the judgment, not the picture | the primary test |
| the paraphrase | the words, not the judgment | keeps 88 to 90 per cent |
| 4 wordings with a matched control | any 1 wording | the gain stays flat |
| the position prior | an average layout | the gain does not move |
| 7 image statistics | edges, light, colour, layout | the gain does not move |
| the alternative ablation map | 1 definition of the causal effect | passes on the other one too |
| split half and leave 1 out | 1 favorable half | holds |
| the pooled layers | 1 favorable layer | holds, 9 layers by rank |
| a held-out set of 60 | a fitted or chosen sample | 3 of 4 cases, some method |
| a photograph permutation | the pipeline itself | null model z = +0.15 |
| an independent re-derivation | a coding error | reproduces to 3 decimals |

## 4. The claims we made and then corrected

| Claim we made | What we found later |
|---|---|
| "the map ranks ablation effects, not content" | activation patching in place beats ablation |
| "the attribution reaches 14 per cent of the ceiling" | the ceiling was self-scored; the correct one is lower |
| "integrated gradients misses 12 per cent" | that was a preliminary run; it misses a fifth |
| "averaging wordings raises the gain" | the wrong side rises too; the gain is flat |
| "the attribution recovers 4 of 7 judgments" | true on the first set; 3 of 4 repeat, 1 method only |
| "0.056 is 6 per cent of a heat map" | 1.0 was the wrong denominator; it is about a third |
| "a subway safety integrated-gradients map on the residual stream passes" | it followed the wording, not the judgment |

## 5. The candidates we closed

| Candidate | How we closed it |
|---|---|
| subway fails because of its answer words | ran it with `Good`/`Poor`; gain +0.06 |
| subway fails because the question is wide | asked "how well lit"; gain +0.13 |
| subway fails because the judgment is weak | the whole-image ablation is the largest |
| subway fails because of an image statistic | no statistic recovers any case |
| the attribution is edge energy | 21 tries, none passes its own test |
| the attribution is a layout prior | the prior removed, the gain holds |
| 2 methods combine into a better 1 | the mean rank loses to `gxa` alone |
| the attribution can say when to trust itself | no self-contained score replicates |
| the Jacobian lens recovers the map at the image tokens | it fails on all 7 cases and every layer |

## 6. What the attribution is, in 1 table

| Question | Answer | Evidence |
|---|---|---|
| does it identify the strongest image token? | yes | 0.150 against 0.004 |
| does it rank the whole map? | barely | rho 0.056, about a third of the ceiling |
| what is in the image token it identifies? | the judged object | 9 of 14 tests, all 7 cases |
| does 1 image token decide the answer? | **almost never** | the oracle is 3 to 28 per cent |
| does the attribution find that image token? | about half the time | and 4 to 15 times the rate of a random token |
| which judgments does `gxa` recover? | 4 of 7 | and a map alone predicts which |
| which judgments does the predictor recover? | **7 of 7** | every row is held out, q below 0.02 |
| where in depth? | layers 16 to 22 | layer 42 gives no gain for any method |
| which factor depends on the question? | the gradient | `h` is the same vector under any question |
| can a fit beat it? | **yes** | it recovers the 3 cases that `gxa` cannot |
| can the model name the image token? | no | the kind of object, not the location |

## 7. Open questions

| Open question | Why it is open |
|---|---|
| does the attribution hold on a 2nd model? | the ablation needs a new operator, not a flag |
| why is the weight cosine near 0? | the weights transfer, and they do not align |
| why does subway safety need its own fit? | its weight is the concentrated one, ratio 123 |
| does any of this hold on a second model? | every number here is gemma-4-12B |
| why does `road_quality` not repeat? | it passed both methods and then neither |
| why do `gxa` and `cheap` disagree? | their maps agree at 0.12 to 0.18 and we cannot say why |
| is the low answer a different object? | parks says yes, and 1 case is not a test |

# The predictor recovers subway safety

Safety is the first judgment that the goal names, and no analytic attribution
method recovered it. `gxa` gives +0.05 pooled, -0.29 at layer 18 and +0.35
at layer 20. We ruled out the answer words, the width of the question, a weak
judgment and every image statistic. The case appeared closed.

It is not closed. The predictor recovers it.

| Layer | Method | Own question | Wrong question | Gain | q |
|---|---|---|---|---|---|
| 18 | `gxa`, not fitted | -0.306 | -0.018 | -0.288 | 0.469 |
| 18 | `g*h`, fitted, **held out** | **+1.159** | +0.065 | **+1.095** | **0.0020** |
| 20 | `gxa`, not fitted | +0.520 | +0.171 | +0.349 | 0.469 |
| 20 | `g*h`, fitted, **held out** | **+1.797** | -0.015 | **+1.812** | **0.0000** |
| 18 | `h` only, question-blind, held out | +0.539 | +0.469 | +0.070 | 0.181 |
| 20 | `h` only, question-blind, held out | +0.423 | +0.523 | -0.099 | 0.221 |

The held-out rows fit on 60 photographs at seed 20260821 and score on 60 at
seed 20260825, and the 2 sets share no image.

**The gain at layer 20 is +1.81 on photographs that the fit never saw.** That
is larger than any analytic number we got on any case.

The wrong-question side is -0.015. So a predictor fitted on the safety
gradient recovers nothing when it is given a restaurant gradient of the same
photographs.

## What was wrong with the old conclusion

We wrote that subway safety fails because the judgment spreads over the
image. The top-1 share said so, and the relation between locality and gain
predicted a small gain. That account described gradient-times-input
attribution, and we read it as a fact about the model.

The predictor shows that the difference is a property of the attribution
method.

| Attribution method | restaurants | libraries | subway safety |
|---|---|---|---|
| `gxa`, layer 20 | +1.42 | +1.00 | +0.35 |
| fitted `g*h`, layer 20, held out | +1.23 | +1.18 | **+1.81** |

A sum with a weight of 1 on every dimension works where the judgment is
concentrated. Where the judgment spreads, the sum cancels and a fitted weight
does not.

**Locality thus predicts where gradient-times-input attribution works, and
not where the judgment is decodable.** The relation between locality and gain
is a fact about `gxa`. It is not a fact about the model.

## The controls this result needed, and what they gave

Both controls are complete. The result holds.

### 1. The null model we first quoted was the wrong one

The permutation null model in the probe measures the own column. It does not
measure the gain.

A probe that we fit on a shuffled token-ablation map learns the mean map of
the case. That mean map separates any token-ablation map that resembles it.
Subway carries the strongest mean map of the 7 cases, so its own-side null
model is high.

The statistic under test is the gain. The own features and the wrong-question
features both carry the mean map. A correct null model thus gives a gain of 0.

`scripts/k1_null_gain.py` measures it. It uses 20 shuffles over 20
photographs, and it reads each case's ridge strength from the run on disk.

| Case | Layer | Null model on `real` | Null model on the gain |
|---|---|---|---|
| restaurants | 18 | +0.367 | +0.141 |
| restaurants | 20 | +0.165 | +0.122 |
| libraries | 18 | +0.113 | +0.040 |
| libraries | 20 | -0.116 | -0.067 |
| schools | 18 | +0.185 | -0.053 |
| schools | 20 | +0.022 | +0.071 |
| street photography | 18 | +0.070 | +0.109 |
| street photography | 20 | +0.227 | +0.291 |
| **subway safety** | 18 | +0.443 | **+0.212** |
| **subway safety** | 20 | +0.274 | **-0.024** |

The spread over the 20 shuffles is about 0.4, so each mean has a standard
error of about 0.09. Every null-model gain is at 0.

**Subway safety at layer 20 has a null-model gain of -0.02 against a measured
+1.81.** The mean map does not produce that number.

We patched `k1_fitted_probe.py`, and it now reports `null_gain` beside
`null_mean`. The warning in the code says which one to quote.

### 2. The rerun with the position prior removed

The second control removes a leave-1-out position prior from the
token-ablation map before the score. For a held-out photograph the prior comes
only from the train set.

| Layer | Attribution method | Own | Wrong | Gain | q |
|---|---|---|---|---|---|
| 18 | `gxa`, nothing fitted | -0.335 | +0.011 | -0.346 | 0.343 |
| 18 | `g*h`, fitted, **held out** | **+1.004** | +0.061 | **+0.943** | **0.011** |
| 20 | `gxa`, nothing fitted | +0.527 | +0.168 | +0.359 | 0.471 |
| 20 | `g*h`, fitted, held out | +1.013 | +0.274 | +0.739 | 0.056 |

The gain at layer 20 drops from +1.81 to +0.74, and layer 18 holds at +0.94.
The mean map thus carries part of the raw number. The predictor survives
without it.

### 3. What is still weak about subway at layer 20

The 2 by 2 design gives one number that we must report. A probe that we fit
on the wrong question's features reads the subway token-ablation map at
+0.778 at layer 20. At layer 18 the same test gives +0.096.

A large question-blind part (a part that does not depend on the question)
thus sits in the layer-20 fit. The layer-18 result is the clean one, and it
is the one to quote.

| Subway safety, layer 18 | Value |
|---|---|
| fit on own, leave 1 out | +0.804 |
| fit on the wrong question | +0.096 |
| **held out** | **+1.095** |
| held out, mean map removed | **+0.943** |
| null model on the gain | +0.212 |
| `h` only, question-blind | +0.070 |

# The predictor reads every judgment we tried

Safety was the last case that no attribution method could read. It is not the
only case that the fit recovers. Schools and street photography also failed
`gxa` in the pre-registered test, and a fitted probe reads both.

## Schools

| Layer | Attribution method | Own | Wrong | Gain | q |
|---|---|---|---|---|---|
| 18 | `gxa`, nothing fitted | +0.248 | -0.038 | +0.286 | 0.508 |
| 18 | `g*h`, fitted, leave 1 out | +1.753 | +0.379 | **+1.373** | **0.0013** |
| 20 | `gxa`, nothing fitted | +0.868 | +0.063 | +0.805 | 0.032 |
| 20 | `g*h`, fitted, leave 1 out | +2.084 | +0.243 | **+1.841** | **0.0000** |
| 18 | `h` only, question-blind | +0.487 | +0.446 | +0.041 | 0.799 |
| 20 | `h` only, question-blind | +0.649 | +0.571 | +0.078 | 0.799 |

A probe that we fit on the wrong question gives +0.407 at layer 18. The own
fit gives +1.753. At layer 20 the 2 numbers are +0.566 and +2.084.

## Street photography

| Layer | Attribution method | Own | Wrong | Gain | q |
|---|---|---|---|---|---|
| 18 | `gxa`, nothing fitted | +0.422 | +0.086 | +0.337 | 0.216 |
| 18 | `g*h`, fitted, leave 1 out | +1.537 | +0.185 | **+1.351** | **0.0004** |
| 20 | `gxa`, nothing fitted | +0.667 | +0.110 | +0.556 | 0.031 |
| 20 | `g*h`, fitted, leave 1 out | +1.335 | -0.006 | **+1.340** | **0.0000** |

Street photography also gives the best top-token agreement of any case. The
predictor's top image token is the token-ablation map's top image token on
15% of the photographs, at both layers. Chance is 0.4%.

## The 5 cases side by side

Every gain below is against the case's own wrong-question control.

| Case | `gxa`, best layer | fitted `g*h`, best layer | null model on the gain |
|---|---|---|---|
| restaurants | **+1.42** | +1.23 held out | +0.12 |
| libraries | **+1.00** | +1.40 held out | +0.04 |
| schools | +0.81 | **+1.84** leave 1 out | +0.07 |
| street photography | +0.56 | **+1.35** leave 1 out | +0.11 |
| subway safety | +0.35 | **+1.81** held out | -0.02 |

The predictor beats gradient-times-input attribution on the 3 cases where the
latter fails, and it matches it where the latter works.

**The judgment sits in the image-token features of all 5 cases we probed.
Only the attribution method had to change.**

# There is 1 predictor, and not 1 for each judgment

The fitted probe learns 3840 weights on the `g * h` of 1 case. We ask if that
weight reads another judgment. If it does, the predictor is general, and the
judgment enters only through the gradient.

We fit on all 60 photographs of the source case and score all 60 of the
target case. The 2 cases share no photograph, so no photograph is on both
sides. The control is the target's own wrong-question features.

`scripts/k1_probe_transfer.py` builds the 5 by 5 matrix at layer 20. Each
case keeps the ridge strength that its own probe chose.

| Fit on \ Score on | restaurants | libraries | subway | schools | street |
|---|---|---|---|---|---|
| restaurants | **+2.84** | +1.34 | +0.59 | +1.27 | +0.67 |
| libraries | +2.09 | **+1.03** | +0.77 | +1.82 | +0.80 |
| subway safety | +2.27 | +1.12 | **+1.12** | +1.26 | +1.09 |
| schools | +2.17 | +0.83 | +0.90 | **+1.84** | +0.46 |
| street photography | +1.52 | +2.29 | +0.16 | +1.30 | **+1.34** |

The bold entries are the leave-1-out self fits.

**Every one of the 20 cross-case entries is positive. 17 of the 20 beat their
control at q below 0.05.**

| Target | Its own fit | The mean of the 4 other fits |
|---|---|---|
| restaurants | +2.84 | +2.01 |
| libraries | +1.03 | **+1.39** |
| schools | +1.84 | +1.41 |
| street photography | +1.34 | +0.75 |
| subway safety | +1.12 | +0.61 |

A weight learned on library upkeep reads school quality at +1.82, and the
school's own weight reads it at +1.84. A weight learned on street photography
reads library upkeep at +2.29, above the library's own +1.03.

**The predictor is thus shared.** A large part of what the fit learns is not
about the judgment. It is about which of the 3840 residual-stream dimensions
carry image-token content, and which carry noise.

The self fit still wins on 4 of the 5 targets, so a case-specific part exists.
It is the smaller part.

## What this means for the attribution method

Gradient-times-input attribution sums with a weight of 1 on every dimension.
That weight is wrong, and it is wrong in the same way for every judgment.

This is why the fit recovers subway safety, schools and street photography.
The fit does not learn what safety is. It learns which dimensions to ignore.

**A predictor that we fit once, on any judgment, is a general predictor of the
token-ablation map for gemma-4-12B.** That is a much stronger tool than 7
per-case probes.

# The held-out table: 10 of 10

Every fitted number above comes from 1 of 2 splits.

| Split | The fit | The score |
|---|---|---|
| leave 1 out | 59 photographs of the case | the 60th |
| held out | all 60 photographs, seed 20260821 | 60 more, seed 20260825 |

The 2 seeds share no image. We checked the overlap for each case, and it is
0 in every case.

| Case | Layer | `gxa` | Fitted, leave 1 out | Fitted, **held out** | q | Top token |
|---|---|---|---|---|---|---|
| restaurants | 18 | +2.11 | +2.49 | **+1.14** | 0.015 | 0.167 |
| restaurants | 20 | +1.42 | +2.84 | **+1.23** | 0.004 | 0.117 |
| libraries | 18 | +0.57 | +1.20 | **+1.40** | 0.013 | 0.050 |
| libraries | 20 | +1.00 | +1.03 | **+1.18** | 0.013 | 0.083 |
| subway safety | 18 | -0.29 | +0.80 | **+1.10** | 0.002 | 0.033 |
| subway safety | 20 | +0.35 | +1.12 | **+1.81** | 0.000 | 0.050 |
| schools | 18 | +0.29 | +1.35 | **+1.36** | 0.000 | 0.033 |
| schools | 20 | +0.81 | +2.23 | **+1.72** | 0.000 | 0.083 |
| street photography | 18 | +0.34 | +1.35 | **+1.05** | 0.003 | 0.017 |
| street photography | 20 | +0.56 | +1.34 | **+0.93** | 0.002 | 0.050 |

**All 10 rows repeat on 60 photographs that the fit never saw. All 10 beat
their wrong-question control at q below 0.02.**

Compare that with gradient-times-input attribution. On the same held-out
design `gxa` repeated on 3 of 4 cases, and only through the other condition
on 2 of the 3. The fit repeats on 5 of 5, at both layers, through the same
condition.

## The 1 limitation in this table

The ridge strength comes from an inner cross-validation that groups by
photograph, and the folds are random. Two runs of the same case thus select a
slightly different alpha. We measured the size of that effect: the schools
layer-18 leave-1-out gain moved from +1.373 to +1.354 between 2 runs.

That is 0.02 z on a +1.4 result, so it changes no conclusion. The held-out
column is the one to quote, because its fit and its score share no
photograph.

# The full set of fits: 7 cases, 14 rows, all held out

Every row below fits on 60 photographs at seed 20260821 and scores 60 more at
seed 20260825. The 2 sets share no image.

`gxa` is gradient-times-input attribution on the same features, with nothing
fitted. The `wrong` column is the same fitted weight, applied to the features
of another case's question on the same photographs.

| Case | Layer | `gxa` | Own | Wrong | **Gain** | q |
|---|---|---|---|---|---|---|
| restaurants | 18 | +2.11 | +2.21 | +1.07 | **+1.14** | 0.015 |
| restaurants | 20 | +1.42 | +2.14 | +0.91 | **+1.23** | 0.004 |
| libraries | 18 | +0.57 | +2.08 | +0.68 | **+1.40** | 0.013 |
| libraries | 20 | +1.00 | +2.35 | +1.16 | **+1.18** | 0.013 |
| parks and plazas | 18 | +0.57 | +2.56 | +1.48 | **+1.08** | 0.018 |
| parks and plazas | 20 | +0.78 | +2.96 | +1.82 | **+1.13** | 0.012 |
| road quality | 18 | +1.06 | +2.33 | +0.47 | **+1.86** | 0.000 |
| road quality | 20 | +1.46 | +2.44 | +0.74 | **+1.70** | 0.001 |
| schools | 18 | +0.29 | +1.75 | +0.40 | **+1.36** | 0.000 |
| schools | 20 | +0.81 | +2.05 | +0.32 | **+1.72** | 0.000 |
| street photography | 18 | +0.34 | +1.17 | +0.12 | **+1.05** | 0.003 |
| street photography | 20 | +0.56 | +1.11 | +0.18 | **+0.93** | 0.002 |
| subway safety | 18 | -0.29 | +1.16 | +0.07 | **+1.10** | 0.002 |
| subway safety | 20 | +0.35 | +1.80 | -0.02 | **+1.81** | 0.000 |

**All 14 rows beat their control at q below 0.02.** Gradient-times-input
attribution beat its control on 4 of the 7 cases. The predictor beats it on
all 7.

## How much of each readout is question-blind

The `wrong` column is not 0 everywhere. It is the part of the readout that
the fitted weight gets from features taken under another question.

| Case | Wrong at layer 20 | What it says |
|---|---|---|
| subway safety | -0.02 | the readout depends only on the question |
| schools | +0.32 | almost all question-specific |
| street photography | +0.18 | almost all question-specific |
| road quality | +0.74 | mostly question-specific |
| restaurants | +0.91 | about half |
| libraries | +1.16 | about half |
| parks and plazas | +1.82 | mostly question-blind, and the gain is still +1.13 |

Parks and plazas is the extreme case. Its map is largely decodable with no
knowledge of the question, and a question still adds +1.13.

This column is the limit of the claim. The predictor does not read only the
judgment. It reads where the image carries weight, and then the gradient
selects which of those places matter to the question.

# What the fitted weight is, and what it is not

The predictor is `sum_d w_d * g_d * h_d`. That is gradient-times-input
attribution with the residual-stream basis rescaled by `w`. The fit adds no
new mechanism. It only says which of the 3840 dimensions to use.

`scripts/k1_reader_anatomy.py` and `scripts/k1_reader_dims.py` ask what `w`
does. Warning: both fit and score on the same 60 photographs, so their
numbers describe the shape of the weight and not how well it generalizes.

## It is not a removal of the high-magnitude dimensions

The model carries massive activations. An unweighted sum over 3840 dimensions
could be dominated by a few high-magnitude and content-free dimensions. If
so, the correction would need no fit: drop the highest-magnitude dimensions
and sum the rest.

We tested it. The result is the opposite.

| Highest-magnitude dimensions dropped | restaurants | libraries | schools | subway |
|---|---|---|---|---|
| 0, the unweighted sum | **+1.42** | **+1.00** | **+0.81** | +0.35 |
| 1 | +0.41 | +0.21 | +0.20 | +0.74 |
| 8 | +0.50 | +0.23 | -0.36 | -0.31 |
| 64 | -0.14 | +0.37 | -0.12 | +0.03 |
| 256 | -0.03 | +0.24 | +0.04 | -0.15 |

**A removal of those dimensions makes gradient-times-input attribution worse,
not better.** The high-magnitude dimensions carry signal. This approach does
not work.

## It is not a small set of dimensions either

The weight is spread. The participation ratio counts how many dimensions
carry the weight's squared norm.

| Case | Participation ratio, of 3840 |
|---|---|
| subway safety | 123 |
| libraries | 707 |
| schools | 850 |
| restaurants | 928 |
| street photography | 947 |

Subway safety is the concentrated one. The others use a quarter of the
basis.

## The dimensions are mostly not shared, and the predictor still transfers

This is the least expected result on this page. The cosine similarity between
2 fitted weights is near 0, and the top dimensions barely overlap.

| Pair | Shared, of the top 64 | Chance |
|---|---|---|
| **libraries and schools** | **33** | 1.07 |
| libraries and parks | 11 | 1.07 |
| parks and schools | 8 | 1.07 |
| parks and roads | 6 | 1.07 |
| libraries and restaurants | 5 | 1.07 |
| schools and subway | 5 | 1.07 |
| parks and street photography | 0 | 1.07 |

Libraries and schools share half of their top dimensions. Both ask about the
upkeep of a public building, so a shared predictor for that pair is expected.
No other pair comes close.

But the transfer matrix above shows that a weight fitted on street
photography reads library upkeep at +2.29. **The transfer does not come from
the top dimensions.** It comes from the bulk of the small weights.

## The dimensions do not have names

We read each shared dimension through the token embedding, which the model
ties to its unembedding matrix. A column of that matrix ranks the vocabulary
by how much 1 dimension pushes each word.

The result is noise for 13 of the 14 shared dimensions. The words are
unrelated pieces of many scripts.

Dimension 3171 is the 1 exception. Its 2 extremes are `<image|>` and
`<|image>`, the image markers themselves. So 1 of the shared dimensions is
the axis that marks an image token.

**A single residual-stream coordinate is not a name.** The basis is not
aligned to meaning, and a layer-20 residual stream is not yet in the output
basis. This approach does not work either, and the shared-subspace
description replaces it.

# One predictor for all 7 judgments

The transfer matrix is pairwise. This is the pooled form. We put the 420
photographs of the 7 cases into 1 ridge regression and hold out either 1
photograph or 1 whole case. `scripts/k1_probe_shared.py` runs both.

| Split | What the fit never saw |
|---|---|
| pooled, leave 1 out | the target photograph |
| leave 1 case out | the target case, its judgment and all 60 of its photographs |

## The pooled predictor

| Case | Own | Wrong | Gain | q |
|---|---|---|---|---|
| restaurants | +2.48 | +0.54 | **+1.93** | 0.000 |
| libraries | +2.28 | +0.56 | **+1.72** | 0.000 |
| schools | +2.21 | +0.32 | **+1.89** | 0.000 |
| road quality | +3.14 | +1.76 | **+1.39** | 0.001 |
| street photography | +1.32 | +0.23 | **+1.09** | 0.000 |
| parks and plazas | +2.26 | +1.20 | **+1.06** | 0.002 |
| subway safety | +1.45 | +0.70 | **+0.75** | 0.026 |
| **all 420 photographs** | **+2.16** | **+0.76** | **+1.40** | **0.000** |

**One weight vector reads all 7 judgments.** It is not 7 predictors.

## The stronger test: hold out the whole judgment

Here the fit sees 6 judgments and reads the 7th. It never saw the target
case.

| Case | Own | Wrong | Gain | q |
|---|---|---|---|---|
| schools | +2.19 | +0.26 | **+1.93** | 0.000 |
| libraries | +2.22 | +0.61 | **+1.61** | 0.000 |
| restaurants | +2.26 | +0.71 | **+1.55** | 0.000 |
| road quality | +2.75 | +1.62 | **+1.13** | 0.001 |
| parks and plazas | +2.24 | +1.04 | **+1.20** | 0.001 |
| street photography | +1.06 | +0.16 | **+0.90** | 0.002 |
| subway safety | +1.12 | +0.80 | +0.32 | 0.584 |

**6 of the 7 judgments are decodable with a weight that never saw them.**
Schools reads at +1.93, which matches its own pooled number.

Subway safety is the 1 case that needs its own judgment in the fit. It reads
at +1.81 with its own weight and at +0.32 with the weight of the other 6.
That agrees with the weight structure: subway's weight is the concentrated
one, at a participation ratio of 123 against 700 to 950 for the rest.

## The number of photographs the fit needs

We drew random subsets of a given size, fit once, and scored every photograph
outside the subset. Each size uses 5 random samples.

| Case | 5 photographs | 15 | 30 |
|---|---|---|---|
| restaurants | +2.11 | +2.39 | +2.30 |
| schools | +1.04 | +1.59 | +1.83 |
| libraries | +0.89 | +1.02 | +1.31 |
| street photography | +0.69 | +1.03 | +1.31 |
| road quality | +0.78 | +1.22 | +1.04 |
| parks and plazas | +0.92 | +0.80 | +0.87 |
| subway safety | +0.60 | +0.70 | +0.96 |

**5 photographs already give a positive readout on all 7 cases**, and 30
reaches most of the full value. Restaurants needs 5.

Note: the run also holds a 59-photograph row. Do not use it. At 59 of 60 in
the fit each sample scores 1 photograph, so that row rests on 5 numbers.

## What this gives

A person who wants a predictor of the token-ablation map for a new judgment
on this model does not need a new fit. They take the pooled weight, ask their
question, run 1 backward pass, and multiply. The judgment enters through the
gradient.

The 1 exception we found is safety, where the predictor needs its own fit. 30
photographs are enough for that.

# The predictor and the image token that changes the answer

A separation is a shift of a logit. A judgment is an answer. We ran this test
on gradient-times-input attribution and it failed. The named image token
changed the answer on 0 to 8 per cent of photographs, and it did not beat a
random image token.

The predictor names other image tokens, so the test runs again. This time we
add the oracle: the share of photographs where some single image token
changes the answer. Without it a low rate looks like a weak predictor when it
may be a hard ceiling. `scripts/k1_reader_flip.py` measures all 3.

| Case | Predictor | Random token | **Oracle** | Share of the ceiling |
|---|---|---|---|---|
| restaurants | 0.167 | 0.044 | **0.283** | 0.59 |
| parks and plazas | 0.083 | 0.011 | **0.117** | 0.71 |
| schools | 0.067 | 0.009 | **0.083** | 0.80 |
| road quality | 0.067 | 0.011 | **0.133** | 0.50 |
| libraries | 0.017 | 0.010 | **0.033** | 0.50 |
| street photography | 0.017 | 0.007 | **0.083** | 0.20 |
| subway safety | 0.000 | 0.000 | **0.033** | 0.00 |

Every row uses the shared weight, which never saw the target case.

**The ceiling is the main result.** On 5 of the 7 cases fewer than 14
photographs in 100 hold any single image token whose ablation changes the
answer. The model does not decide these judgments on 1 image token. It adds
many small contributions.

**Against that ceiling the predictor is not weak.** It reaches 50 to 80 per
cent of the achievable answer changes on 4 cases. On every case with a
nonzero rate, its rate is 4 to 15 times the random rate.

## What this test cannot say

No row passes the multiple-comparison threshold. Restaurants is the closest
at q=0.057, and the rest are above 0.2.

The reason is statistical power, not the predictor. With 60 photographs and a
ceiling of 3 to 28 per cent, the whole test has between 2 and 17 answer
changes to work with. A larger photograph set would settle it.

**We thus correct the earlier sentence.** We wrote "one image token almost
never changes the answer, and the predictor does not beat chance". The first
half holds and the second half was not measured against the right threshold.
The right sentence is:

> One image token can change the answer on 3 to 28 photographs in 100. When
> it can, the predictor finds it about half the time. 60 photographs are too
> few to put a confidence interval on that.

# Every judgment: the image patch the predictor names holds the judged object

The content test cuts the image patch the predictor names out of the
photograph and asks the model to name what it holds. The earlier run used
gradient-times-input attribution and it covered the 4 cases that method
could read.

This run uses the shared fitted weight, which never saw the target case. It
covers all 7. `scripts/k1_target_words.py` holds each word list, and every
list comes from the question and not from the words we saw.

| Case | End | Names the judged object | A random crop | Paired | q |
|---|---|---|---|---|---|
| restaurants | high | 0.783 | 0.263 | **+0.521** | **0.0000** |
| libraries | high | 0.933 | 0.554 | **+0.379** | **0.0000** |
| road quality | low | 0.767 | 0.408 | **+0.358** | **0.0000** |
| restaurants | low | 0.617 | 0.263 | **+0.354** | **0.0000** |
| street photography | high | 0.667 | 0.321 | **+0.346** | **0.0000** |
| schools | high | 0.967 | 0.629 | **+0.338** | **0.0000** |
| schools | low | 0.850 | 0.629 | **+0.221** | **0.0001** |
| parks and plazas | high | 0.467 | 0.271 | **+0.196** | **0.0013** |
| **subway safety** | high | 0.517 | 0.350 | **+0.167** | **0.0119** |
| libraries | low | 0.667 | 0.554 | +0.113 | 0.130 |
| parks and plazas | low | 0.350 | 0.271 | +0.079 | 0.119 |
| street photography | low | 0.367 | 0.321 | +0.046 | 0.223 |
| road quality | high | 0.433 | 0.408 | +0.025 | 0.344 |
| subway safety | low | 0.333 | 0.350 | -0.017 | 0.487 |

One Benjamini-Hochberg family over the 14 tests. **All 7 cases pass at 1 end
or both, and 9 of the 14 tests pass.**

## The predictor beats gradient-times-input attribution at this too

Both attribution methods ran on the same photographs with the same word list.

| Case, high end | Fitted, shared weight | Gradient-times-input `gxa` | Random |
|---|---|---|---|
| restaurants | **0.783** | 0.600 | 0.263 |
| libraries | **0.933** | 0.750 | 0.554 |
| parks and plazas | **0.467** | 0.467 | 0.271 |

## The words

| Case | The predictor's high end | A random crop |
|---|---|---|
| **subway safety** | `subway`, `station`, `entrance`, `street` | `surface`, `small`, `trees`, `road` |
| restaurants | `storefront`, `sign`, `awning`, `restaurant` | `surface`, `car`, `road`, `ground` |
| libraries | `brick`, `gate`, `wall`, `arched`, `entrance` | `road`, `sky`, `car`, `surface` |
| schools | `brick`, `building`, `windows`, `banner` | `road`, `sky`, `surface`, `car` |
| street photography | `street`, `cars`, `crossing`, `pedestrian` | `surface`, `car`, `branches`, `sky` |
| parks and plazas | `sidewalk`, `bushes`, `fence`, `plants`, `green` | `surface`, `dark`, `road`, `asphalt` |

**The subway row is the one to read.** No attribution method without a fit
ever decoded safety. The predictor points at the subway entrance itself, and
it does so with a weight fitted on 6 other judgments.

## Why road quality inverts

Road quality is the 1 case whose high end fails and whose low end passes at
+0.358. Its high-end words are `objects`, `any`, `cannot`, `solid`.

Those are the words of a refusal. The model says it cannot see any distinct
object, because the crop holds a plain piece of road surface with nothing on
it. A word list of road nouns cannot score that.

The low end names `asphalt`, `crack`, `debris`, `road`, `pavement`. So the
predictor still sits on the roadway at both ends. Only 1 end has anything to
name.

# The 2 parts: a question-blind part and a question-specific part

**Warning:** These 2 parts are not the J-lens and not the R-lens. We built
both of those methods, we fitted both, and we scored both against this same
token-ablation map. Neither one predicts it. Read the next part first.

## What these parts are not

| Attribution method | What it is | Its score against the token-ablation map |
|---|---|---|
| the fitted J-lens | 1 averaged Jacobian, `unembed(J_l h)`, fitted on text | Pearson correlation -0.011 to +0.026 |
| the R-lens | the same fit with 4 LRP rules in the backward pass | a better readout, the same null result against the token-ablation map |

The J-lens is included in every `input_gate` run as the `lens_own` condition.
We scored it on all 7 cases, on the first set and on the held-out set. Not 1
of those 14 numbers leaves the noise band.

The cause of the failure is a measurement, not a guess. A lens carries 1
averaged direction. We measured the variance in that mean direction, and it
never holds more than 13 percent. That number is a property of the model and
the task, so no corpus can lift it. A second measurement agrees from the
other end: our J-lens is unreadable below layer 30, and the image stops
acting on the answer above layer 30. The readable window and the causal
window never meet.

Both failures have 1 cause. A lens replaces the gradient of this input with
an average over a corpus. We tested that same substitution directly, with the
`meang_own` and `meangp_own` conditions, and they failed in the same way. The
predictor below uses the gradient of this photograph under this question. The
average discards exactly that.

`r-lens.md` keeps the R-lens for the job it does well. It improves the text
readout and the image-token readout, and its fit is cheaper and more stable.
It does not predict what an image token does, and its directions do not
steer. Use the R-lens for a readout. Use `causal_maps.py` for what a region
does.

## The 2 parts

The predictor splits cleanly into 2 parts, and each part has its own number.

The model places the image tokens before the question. A causal decoder thus
gives the same hidden state `h` at every image-token position under any
question. So:

| Part | What it needs | What it knows |
|---|---|---|
| **the question-blind part**, a fixed direction on `h` | 1 forward pass | which image token carries causal weight |
| **the question-specific part**, the gradient `g` | 1 more backward pass | which of those the question uses |

## The question-blind part: 1 direction, no gradient, no question

We fit a ridge regression on `h` alone. The predictor is `<w, h_p - mean h>`,
a single direction in the residual stream. Every row below is held out on 60
new photographs. We also remove the mean map of the case from the
token-ablation map first. The direction thus cannot win when it names the
centre of the image.

| Case | Layer 20 | 95% confidence interval | Under another question |
|---|---|---|---|
| parks and plazas | **+1.79** | +1.17 to +2.46 | +1.76 |
| restaurants | **+1.43** | +0.77 to +2.11 | +1.31 |
| schools | **+0.82** | +0.27 to +1.37 | +0.84 |
| libraries | **+0.75** | +0.24 to +1.26 | +0.64 |
| street photography | **+0.63** | +0.21 to +1.04 | +0.68 |
| subway safety | **+0.42** | +0.02 to +0.80 | +0.53 |
| road quality | +0.42 | -0.02 to +0.85 | +0.38 |

**A single fixed direction predicts the token-ablation map on 6 of the 7
cases, and it never sees the question.** Road quality is the 1 marginal row,
and its interval includes zero by 0.02. The last column is the same direction
under another case's question, and it matches the first to 2 decimal places.
That is not a failure of the control. It shows that `h` is question-blind.

At layer 18 the same predictor gives +1.08 down to +0.18. Subway, schools and
road quality cross zero there. Layer 20 is the better depth for this part.

## The question-specific part: what the gradient adds

The gain column of the 14-row table above is exactly what the gradient adds.

| Case | The question-blind part alone, layer 20 | The gain the gradient adds |
|---|---|---|
| subway safety | +0.42 | **+1.81** |
| schools | +0.82 | **+1.72** |
| road quality | +0.42 | **+1.70** |
| restaurants | +1.43 | +1.23 |
| libraries | +0.75 | +1.18 |
| parks and plazas | +1.79 | +1.13 |
| street photography | +0.63 | +0.93 |

**The 2 parts are inverse.** Where the question-blind part reads most, the
gradient adds least. Parks and plazas has the largest question-blind readout
and the smallest question-specific gain. Subway safety has the smallest
question-blind readout and the largest gain.

Two independent measurements thus agree. The `wrong` column of the held-out
table calls parks mostly question-blind and subway fully question-specific.
The question-blind part, fitted on `h` alone, says the same.

## What a user should build

| If you want | Build | Cost |
|---|---|---|
| which image tokens carry causal weight | the question-blind direction | 1 forward pass |
| which image tokens this question uses | the question-blind direction times the gradient | 1 forward and 1 backward pass |

Neither needs a fit for each judgment. The pooled weight serves all 7.

# The answer to the goal, in 1 page

The goal asked for a causal Jlens or Rlens that reads image-token causality
toward an aesthetic judgment: safety, upkeep, and preference. This is what we
have on 2026-08-25.

The answer is not a lens. The fitted J-lens and the R-lens both failed this
test, and the section above gives their numbers and the reason. What works is
a fitted linear predictor on gradient-times-input attribution, which keeps
the gradient of each input instead of an average of it.

## What works

| Claim | Number | Where it is tested |
|---|---|---|
| the predictor reads every judgment | 14 of 14 rows, q below 0.02 | 60 held-out photographs, 7 cases, 2 layers |
| 1 weight serves all 7 judgments | +1.40 pooled | 420 photographs, 1 leave-1-out ridge regression |
| a weight that never saw a judgment reads it | 6 of 7 | leave 1 case out |
| 5 photographs are enough to fit | positive on 7 of 7 | random subsets, scored outside |
| the image patch holds the judged object | 9 of 14 tests, all 7 cases | crop and caption, against a random crop |
| a gradient-free direction predicts the map | 6 of 7 at layer 20 | mean map removed, held out |

## What does not work

| Claim | What we found |
|---|---|
| the unweighted sum reads every judgment | it reads 4 of 7, and locality predicts which |
| a removal of the high-magnitude dimensions corrects the sum | it makes the sum worse at every number of removed dimensions |
| the predictor's dimensions have names | the token embedding gives noise for 13 of 14 |
| the weights align across cases | the cosine similarity is near 0, and the transfer holds |
| 1 image token decides the answer | at best 28 photographs in 100 hold such a token |
| the predictor reproduces the full map | the whole-map rank correlation is 0.056 |
| the fitted J-lens or the R-lens predicts the map | neither leaves the noise band on any of the 7 cases |

## Open questions

| Open question | Why it is open, and what it needs |
|---|---|
| a 2nd model | the ablation acts on the image-token rows of the unified model. An encoder model needs a pixel-space operator, which is a different measurement and not a flag |
| the shared subspace | the weights transfer and do not align. We did not find the subspace that carries the transfer |
| why safety needs its own fit | its weight has a participation ratio of 123 against 700 to 950 |
| the answer-change test | 60 photographs give 2 to 17 answer changes. A larger set would put an interval on it |
| the layer profile on the other 5 cases | we mapped layers 12 to 30 on restaurants and subway only |

## The 1 sentence

**On gemma-4-12B 1 fitted linear predictor on gradient-times-input
attribution names the image token that carries an aesthetic judgment. It
works on all 7 judgments we tried, with 1 backward pass and no fit for each
judgment. The image patch it names holds the object the question is about.
It almost never holds the whole answer, because almost no single image token
does.**

# The layer profile of the predictor: a wider band than the sum

Until now we measured the predictor at layers 18, 20 and 42 only. This
section maps layers 12 to 30 on the 2 cases at the ends of the question-blind
split. Both columns are leave-1-out, and both use the same activations and
the same control.

## Restaurants

| Layer | `gxa` gain | Fitted gain | Own | Wrong | q | Top token |
|---|---|---|---|---|---|---|
| 12 | +0.68 | +1.23 | +1.19 | -0.03 | 0.024 | 0.117 |
| 16 | +1.21 | **+2.51** | +2.45 | -0.06 | 0.000 | **0.217** |
| 18 | +2.11 | **+2.49** | +2.56 | +0.07 | 0.000 | **0.217** |
| 20 | +1.42 | **+2.84** | +2.91 | +0.07 | 0.000 | 0.183 |
| 22 | +1.37 | **+2.31** | +2.54 | +0.23 | 0.000 | 0.133 |
| 24 | +0.56 | **+2.54** | +2.44 | -0.10 | 0.000 | 0.167 |
| 26 | +0.74 | +0.81 | +1.51 | +0.69 | 0.035 | 0.133 |
| 28 | +0.25 | +1.42 | +1.97 | +0.55 | 0.001 | 0.150 |
| 30 | +0.06 | +0.42 | +0.96 | +0.54 | 0.450 | 0.083 |

## Subway safety

| Layer | `gxa` gain | Fitted gain | Own | Wrong | q | Top token |
|---|---|---|---|---|---|---|
| 12 | -0.00 | +0.33 | +0.61 | +0.28 | 0.483 | 0.017 |
| 16 | +0.03 | +0.82 | +0.77 | -0.05 | 0.071 | 0.000 |
| 18 | -0.29 | +0.66 | +0.88 | +0.22 | 0.107 | 0.017 |
| 20 | +0.35 | **+1.12** | +1.51 | +0.39 | **0.002** | 0.033 |
| 22 | -0.25 | **+0.76** | +0.97 | +0.21 | **0.022** | 0.050 |
| 24 | -0.46 | **+0.78** | +1.30 | +0.53 | **0.009** | 0.000 |
| 26 | -0.70 | +0.65 | +1.18 | +0.53 | 0.097 | 0.000 |
| 28 | -0.03 | +0.50 | +1.13 | +0.63 | 0.107 | 0.000 |
| 30 | +0.29 | +0.65 | +0.68 | +0.03 | 0.084 | 0.000 |

## What the profile says

**The fit widens the band.** On restaurants gradient-times-input attribution
peaks at layer 18 and falls to +0.56 by layer 24. The predictor holds above
+2.3 from layer 16 to layer 24. Layer 24 is the widest gap in the whole
study: +0.56 against +2.54.

**Gradient-times-input attribution is negative on subway at 6 of the 9
layers.** It does not fail at 1 depth. It fails at every depth. The predictor
is positive at all 9 and passes at layers 20, 22 and 24.

**Both attribution methods lose their signal after layer 26.** That agrees
with the earlier band and with the null result at layer 42.

**The top-token rate is higher than we reported.** Restaurants reaches 0.217
at layers 16 and 18, against the 0.150 we quoted from layer 20 alone. Chance
is 0.004.

Note: these rows are leave-1-out, not held out. The held-out table above is
the one to quote for the size of the effect. This table is for the shape of
the layer profile.

# The image patch the predictor names, out of sample, 2026-08-26

The word experiment of 2026-08-25 asked what sits in the image patch each map
names. It read the predictor's image patch on the same photographs the fit
read, so the answer could belong to that sample. This run repeats it on the
seed-20260827 photographs, which no earlier experiment read: 7 cases, 60
photographs for each case, 2520 crops.

| The map that names the image patch | disorder words at the negative end | at random | gap | p |
|---|---|---|---|---|
| the predictor's output | 0.129 | 0.030 | **+0.098** | 0.00005 |
| the token-ablation map | 0.069 | 0.028 | +0.041 | 0.028 |

A permutation of the map label inside each (case, photograph, condition)
gives +0.0571 against a null-model value of +0.0002 with sd 0.0166, thus
p = 0.0007. The in-sample run gave +0.0875, so the advantage repeats out of
sample.

**The predictor is not a denoised copy of the token-ablation map.** It agrees
with the token-ablation map at rho 0.09 and names the same worst image token
11 percent of the time. The token-ablation map agrees with itself at rho 0.53
to 0.75 and 8 to 58 percent across batch sizes. A cleaner copy of a noisy map
sits near the retest line. This one sits far below it, so the ridge
regression reads something the ablation extremum does not carry.

**What it does not say.** The words are a property of the crop. No second
causal measure scored the predictor's image patch, so we cannot call the
predictor more correct about the model.

Full record and every table: `shared-negative-evidence.md`, Finding 6.
Tools: `causal-patch-reader-tools.md`.
