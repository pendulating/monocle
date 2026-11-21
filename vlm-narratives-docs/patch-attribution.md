# Patch attribution: which image tokens move the decision

This page depends on the model architecture: 48 layers, 8 of them global,
the bidirectional attention mask over the image tokens, and the bfloat16 cast
in the vision embedder. See
[gemma4-12b-architecture.md](gemma4-12b-architecture.md). On this page, Ln
denotes layer n.

`monocle/patch_attribution.py` scores each image token of a pairwise run.
`monocle/patch_causal.py` tests whether those scores are correct.

Read this page before you read an image-token attribution map. The layer
result it depends on is in [where-the-judgment-lives.md](where-the-judgment-lives.md).

## What the score is

    c          = logit(More) - logit(Less)          at the answer position
    a_l[patch] = < dc/dh_l[patch] , h_l[patch] >    grad x centered activation

The first line is the logit difference at the last token position. The second
line is gradient-times-input attribution: the gradient of the logit difference
times the centered hidden state of an image token. One backward pass gives
`dc/dh_l` at every position. This is one row of the Jacobian, the gradient of
the logit difference, so it costs 1 pass, not the 3,840 that the full matrix
needs.

## The layer determines the correlation

Aggregate a map as (sum over image A) minus (sum over image B) and correlate
it with `c` across 200 pairs of each case. Mean over the 7 cases, one layer at
a time:

| L34 | L36 | L38 | L39 | **L40** | L41 | L42 | **L43** | L44 | L45 | L46 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.398 | 0.559 | 0.403 | 0.398 | **0.766** | 0.720 | 0.789 | **0.805** | 0.679 | 0.735 | 0.234 |

**Use L43.** L42 is below the peak. The usable window is L40 to L45. Two
results follow.

1. A step at L39 to L40: 0.398 to 0.766 in one layer. L38 and L39 are worse
   than L36. Every layer is effectively global here (`sliding_window` 1024 >
   our 703 tokens), so the alternation of local and global attention does not
   explain it. This is an open question.
2. L46 collapses. The cause is 1 image token (see below).

Maps at L18-L24 look meaningful to a person: the library or the subway
entrance is visible. They hold no judgment signal. A readable map is not a
map of contributions.

**Warning:** these numbers show that the map correlates with the decision.
They do not show that the image tokens caused it. At L43 the residual stream
at the image tokens may already carry the answer.

## The last image token of each image concentrates the gradient

The last image token of each image is 1 token of 256, and it holds about 20%
of the map at L42 and 43% at L46. When we drop it, the L46 correlation more
than doubles:

| | L34 | L36 | L42 | L43 | L46 |
|---|---|---|---|---|---|
| all image tokens | 0.398 | 0.559 | 0.789 | 0.805 | **0.234** |
| drop image token 255 | 0.573 | 0.613 | 0.731 | 0.760 | **0.555** |

`patch_norms.py` shows the cause. It decomposes `a = ||g|| ||h|| cos` at that
image token against all the others (25 pairs; subway_safety and libraries
agree to 2%):

| layer | `||h||` ratio | `||g||` ratio |
|---|---|---|
| L30 | 1.04 | 19.2 |
| L36 | 1.44 | 36.8 |
| L42 | 1.25 | 66.4 |
| L46 | 1.00 | **116.9** |

The activation norm is ordinary: the ratio is exactly 1.00 at L46. Thus this
token is not a register token, and the effect is not a norm artifact. The
gradient ratio is 117x. The decision moves when that 1 position moves.

At L46 the only path from an image token to `c` is the attention of layer 47
from the last token position, because `c` reads `h_47[answer]` alone.

**Warning:** we first interpreted this as "the model routes each image to its
last image token". Two later tests refute this. See "The last image token
attracts attention but does not hold the judgment" below. The attention
concentration is real. The routing is not.

Measured layout: the image tokens sit at positions 8-263 and 269-524, and
`boi` (255999) and `eoi` (258882) enclose each set. `mm_token_type_ids` holds
only [0, 1], so `boi` and `eoi` are not image tokens. The bidirectional
attention region is exactly the 256 image tokens, and the last one is adjacent
to the causal boundary.

**Rule:** mask image token 255 at L46. Keep it at L40-L45, where it helps.

## The logit-lens readouts and the attribution map disagree, and they should

The image tokens with the largest attribution have the logit-lens readouts
`<pad>`, `/`, `3`, `pered`, `icing`; ordinary image tokens read `sidewalk`,
`brick`, `orange`. That is not a rendering fault. When we keep only the image
tokens whose readout is a dictionary word, the correlation collapses:

| Case | L30 all -> word | L42 all -> word | L46 all -> word |
|---|---|---|---|
| subway_safety | 0.769 -> 0.134 | 0.833 -> 0.709 | 0.402 -> 0.129 |
| libraries | 0.694 -> 0.072 | 0.828 -> 0.746 | -0.022 -> 0.066 |
| parks_plazas | 0.515 -> 0.165 | 0.904 -> 0.733 | 0.308 -> 0.030 |

The image tokens that name the scene are not the image tokens that move the
answer. Do not mask the image tokens with an unreadable readout to clean a
map. That removes the signal.

The logit-lens readout also depends on the layer. At L30 the readout is
modality control tokens (`<audio|>` 123,834 times over ordinary image tokens).
At L42 the readout is scene content. At L46 the readout collapses to `ed`
(69,308) and `<|channel>` (61,925) everywhere.

The lens fitted at consecutive layers adds 2 regimes. **At L34 the readout
collapses to a control token: `<pad>` is the top-1 readout of 82.5% of all
image tokens.** L45 turns to punctuation (`-` at 30%) before L46 turns to
control tokens. Thus the late collapse is gradual, and it starts at L45.

### The sparse set of fitted layers caused this, not the model

The old set of fitted layers was 6, 12, 18, 24, 30, 36, 42, 46. Each of those
layers except 46 sits directly after a global-attention layer. Thus "readable
at L42, unreadable at L46" could describe the model, or it could describe the
layers we fitted.

We fitted the lens at 13 layers (30, 34, 36-46; 128 prompts) and scored the
top-1 readout of every image token against an English dictionary. 40 pairs,
2 cases, which agree to 2%:

| L30 | L34 | L36 | L37 | L38 | L39 | L40 | L41 | L42 | L43 | L44 | L45 | L46 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 48% | **13%** | 63% | 72% | 66% | 76% | 73% | 77% | **78%** | 74% | 72% | 55% | 47% |

**The attention type of the layer does not predict the readout.** Inside the
flat band L36-L44, where only the attention type changes:

| Attention type | Layers | Mean |
|---|---|---|
| after a global layer | 36, 42 | 70.6% |
| sliding-window layer | 37, 38, 39, 40, 43, 44 | 72.5% |
| global layer | 41 | 77.3% |

The gap between the post-global mean and the sliding-window mean is 1.8
points. The spread within the sliding-window layers is 10 points. Thus depth
explains the readout and the attention type does not.

**The readable window is L36 to L44.** It contains the attribution window
L40-L45 at L40-L44. Thus we can read the layers that decide. The earlier
claim that we could not was an artifact of the set of fitted layers.

Use a dictionary test, not an alphabetic test. A loose "3+ letters" rule
scores L37 at 97%, because `imov`, `pered`, `tered`, and `celain` are
alphabetic fragments, not words. That rule inverts the ranking of L37 and L42.

**Warning:** a layer with a readable logit-lens output is not a layer with a
valid attribution. L37-L39 have readable outputs (66-76%) and hold almost no
judgment signal (r about 0.40). Use L42 or L43, where both hold.

## The last image token attracts attention but does not hold the judgment

`patch_swap.py` does activation patching: it replaces the residual stream of
one image token of a target image with the same token of a source image, at
layer L. It then measures the fraction of the effect recovered

    recovery = (c_swapped - c_target) / (c_donor - c_target)

Activation patching in the residual stream does not cross the bfloat16 pixel
cast, so the failure that stopped all 3 attempts in `patch_causal` does not
apply to it.

The positive control overwrites the last token position. This must move `c`
to the source-image value when little computation remains above L. Measured
on subway_safety, 12 source-target pairs, single-image path:

| L5 | L20 | L30 | L38 | L42 | L43 | L46 |
|---|---|---|---|---|---|---|
| 0.008 | 0.021 | 0.291 | 0.733 | 0.802 | 0.801 | **0.991** |

A correct intervention must produce this curve: the layers above recompute an
early replacement, and a late replacement persists. **An identity replacement
is not a valid control for this test.** A write of the value already present
and no write at all give the same answer. The first version of this module
passed that check while every condition gave zero.

### Result 1: the last image token is ordinary

| Layer | last | mid | random | first |
|---|---|---|---|---|
| L5 | +0.158 | -0.010 | +0.079 | -0.038 |
| L20 | -0.013 | +0.010 | -0.010 | -0.004 |
| L42 | -0.007 | +0.007 | +0.010 | +0.008 |
| L46 | -0.004 | +0.001 | +0.000 | +0.003 |

At no layer does the last image token exceed an ordinary one. At L5 the
`random` condition moves `c` more than `last` (mean |dc| 0.929 against
0.436). Thus the 100x attention concentration marks where the last token
position attends. It does not mark an image token that holds more.

### Result 2: the image information transfers below L30

Replace all 256 image tokens at once:

| L5 | L20 | L30 | L38 | L42 | L43 | L46 |
|---|---|---|---|---|---|---|
| +1.093 | +0.969 | +0.045 | +0.012 | +0.010 | +0.006 | +0.007 |

A whole-image replacement transfers the entire judgment at L5 and L20, and
nothing from L30 up. Thus the image information reaches the last token
position below L30, and by L30 the last token position already holds what it
needs.

**Thus the attribution window L40-L45 is causally inert.** The replacement of
every image token there moves `c` by about 0.1 against a range of 17. The
maps at L42 and L43 correlate with `c` at r about 0.8 because the residual
streams at the image tokens still encode the judgment, from a shared upstream
cause. They do not drive it.

This measurement confirms the warning at the top of this page. Read an
image-token map as a correlate of the decision, never as its cause.

Scope: 12 pairs, 1 case, the single-image path. See
[single-image-lens-path.md](single-image-lens-path.md).

## The gradient map does not predict the measured map

If the image tokens matter below L30, the direct correction is to compute the
gradient-times-input map at L20. That correction does not work, and the
reason matters.

`patch_map_causal.py` measures the map directly: it replaces each of the 256
image tokens alone and records how far `c` moves. Then it tests whether the
1-backward-pass gradient map predicts the measured map. 8 source-target
pairs, subway_safety.

Most image tokens move `c` by almost nothing, so a score over all 256 is
mostly noise. We score the image tokens that move it.

**Warning:** the numbers below replace the 2026-08-22 table. That run took
the baseline `c` at batch 1 and ran the replacements at batch 8, and in
bfloat16 a forward pass depends on its batch size. Every delta of an image
thus carried a constant offset, from -0.50 to +1.23, against a typical
absolute delta of 0.31. A Pearson correlation does not see a constant, but a
sign score sees only the constant. `batched_baseline` corrects it. Jobs
478063 and 478064, 2026-08-23. The first-order Taylor map is scored, which is
the fair test of the gradient (see below).

| Layer | sign accuracy, top 50 | r, top 50 | whole-image replacement, fraction recovered |
|---|---|---|---|
| L12 | 0.61 | +0.20 | 0.96 |
| L18 | 0.59 | +0.35 | 0.98 |
| **L20** | 0.60 | **+0.49** | 1.04 |
| L24 | 0.64 | +0.30 | 0.68 |

**L20 still holds the most signal**, and r rose from +0.43 to +0.49 after the
offset correction. The sign accuracy is now flat across the band, near 0.60,
and 0.60 is not sufficient to colour 256 image tokens. Above L30 no method
can work, because the image tokens have no causal effect there.

This table is the baseline for the R-lens work, and the R-lens does not beat
it: a fitted linear translator reaches r +0.10 where the local gradient
reaches +0.49. See [r-lens.md](r-lens.md).

### The 2 maps answer opposite questions, and that is correct

The gradient-times-input map is `<g_p, h_p - hbar>`, and a replacement moves
the image token along `h_donor - h_p`. Measured, the 2 anti-correlate at -0.4
to -0.6. That is not a fault. If a source-image token is a random sample,
then `h_donor` is near `hbar`, so

    <g, h_donor - h> = -<g, h - hbar>

Thus the gradient-times-input map states a contribution ("this image token
pushes the answer this way"), and a replacement states the effect of removal.
An image token that supports `Safe` must lower `c` when it is removed. With
the sign flipped, the gradient-times-input map scores about as well as the
first-order prediction.

**Rule:** state which of the 2 a figure shows. They differ by a sign.

## The sum is not a total

`Sum over p of a_l[p]` anti-correlates with `c` at -0.93 to -0.96. That is not
a defect of the maps. The activation is centered, so

    sum_p < g_p , h_p - hbar > = sum_p < g_p - gbar , h_p - hbar >

which is a covariance across image tokens with the global term removed by
construction. It has no completeness property. Do not report it as a total.

## The calibration failed 3 times

`patch_causal.py` tried to measure the effect that the scores predict. All 3
attempts failed, and the failures are informative.

### 1. Mean-patch ablation is out of distribution

Replace an input image patch with the mean image patch of its image and run
the model again. Measured on subway_safety, 25 pairs, each image patch of the
16x16 grid:

| | value |
|---|---|
| mean abs delta c, 1 image patch | 0.22 to 0.46 |
| ratio, most to least | 2.09 |
| mean abs c | 4.97 |
| sign, fraction positive | 0.435 |

Sky, pavement, and the subway entrance all move the decision by about the
same amount. The 256 single-patch deltas sum to -9.92 while `c` is +4.57.

The model responds to the flat grey square, not to the missing content. Thus
the ablation measures perturbation sensitivity, and it cannot evaluate the
scores. The correlations it produced (r about 0 at pool size 3 and pool size
1) show nothing about gradient-times-input attribution.

The implementation is not at fault. An empty ablation reproduces `c` exactly,
and `patch_causal` asserts this on the first pair of every run.

### 2. Integrated gradients violates completeness

IG must sum to `c(x) - c(baseline)`. Measured: sum -205.6 against +3.27, a
completeness residual of 6391%. `c` stays inside -3 to +5 along the whole
path, so a correct gradient cannot integrate to -205.

### 3. The pixel gradient is not the model's gradient

Step along the pixel gradient by the distance that the gradient predicts
moves `c` by `pred`. Then measure:

| pred | pair 1 | pair 2 | pair 3 |
|---|---|---|---|
| +0.05 | +1.00 | +0.00 | +0.00 |
| +0.20 | +0.95 | -0.29 | +0.09 |
| +1.00 | -0.14 | -0.25 | +0.18 |
| +4.00 | +0.12 | +0.24 | +0.39 |

The zeros show the cause. `Gemma4UnifiedVisionEmbedder.forward` runs

    self.patch_ln1(pixel_values.to(self.patch_dense.weight.dtype))

and that dtype is bfloat16. A step below bfloat16 resolution rounds back to
the original image, so the model output does not move. Autograd passes the
gradient through `.to()` unchanged, so the analytic gradient reports a
sensitivity that the deployed model does not have.

**The pixel gradient describes an ideal float32 model. The model on the GPU
is a step function at fine scale.** This one fact explains all 3 failures.

Thus: do not use pixel-level gradient attribution on this IT model in
bfloat16. More IG steps do not help.

## What to do

| Goal | Method |
|---|---|
| **an image-token map to publish** | **`causal_maps.py` at L20**: measured, not predicted |
| a low-cost screen of many images | gradient-times-input attribution at **L20**, top 20 image tokens only |
| which position holds the judgment at layer L | `patch_swap.py` |
| whether an attention path carries information | `attn_knockout.py` |
| a map at L40-L46 | none. The image tokens have no causal effect at those layers |

**Warning:** `grad x activation at L42` (gradient-times-input attribution at
that layer) was the recommendation until 2026-08-21. We withdrew it. That map
correlates with `c` at r about 0.8, and the image tokens it highlights change
`c` by about 0.1 out of 17.

### The recommended figure path

`monocle/causal_maps.py` measures the effect of each image token instead of
predicting it.

1. Use **L20**. The module rejects a layer at or above 30.
2. Replace each of the 256 image tokens alone and record the change in `c`.
3. Average over several source images. 1 source image gives the effect of a
   replacement by that specific token; the average gives how much this image
   token matters. The module saves the spread per image token, so an unstable
   token is visible.
4. Flip the sign, so the stored value is the contribution toward the
   positive label. A replacement delta gives the effect when an image token
   is removed, which is the opposite of what the token contributes.
5. Draw every panel on one scale, in logit units. `pa.render_attrib`
   normalizes inside a panel, so a weak map and a strong map look equal
   across a sheet.

### The colour convention

The data and the colour point in opposite directions, and this is by design.

| Item | Direction |
|---|---|
| `values` in `maps.parquet` | the contribution toward the positive label |
| the colour in a panel | coral is the negative label, teal is the positive label |

For `subway_safety` the labels are `Safe` and `Dangerous`, so **coral marks
the danger**. A viewer of a safety map expects red there, and a figure must
not contradict that expectation. `CORAL_SIGN` in `causal_maps.py` holds the
flip, and nothing else in the module changes sign.

**Warning:** name the 2 labels on the colour bar. An earlier bar had the
label `change in the judgment`, which describes the replacement delta. The
image tokens carry the opposite sign of the delta, so a viewer who trusted
the bar label read every panel with the wrong sign.

### Warning: the source-image (donor) reference is retired (2026-08-23)

The layer-20 activation-patching map and the input token-ablation map of the
same photograph correlate at **+0.09**, and their strongest Dangerous image
tokens overlap at chance. Only the token-ablation ranking survives a
token-ablation test. Gemma-4 gives the image tokens a bidirectional attention
mask from layer 0, so a layer-20 residual stream carries the whole scene, and
activation patching there does not isolate 1 image token.

Use `--reference absent`. See [[do-the-colours-mean-anything]] for the tests
that decide it, and for what a map can and cannot show.

### 2 references: source image (donor) and absent

A map entry gives the contribution of an image token relative to a reference.
`--reference` selects the reference.

| Reference | The edit | Meaning |
|---|---|---|
| `donor` (source image) | at L20, replace the token with the same token of another photograph | contribution, against 3 other photographs |
| `absent` | at the input, remove the image token | contribution, against the absence of the token |

`absent` is the better reference, for 3 reasons.

1. It answers the question that a viewer asks. The source-image map gives the
   effect of a replacement by a specific token, which no caption claims.
2. The input is the only place where an image patch is still 1 patch. Gemma-4
   gives the image tokens a bidirectional attention mask, so every image
   token attends to every other image token from layer 0. A layer-20 hidden
   state carries a share of the whole scene.
3. It needs no source images, so it costs 1/3 as much.

**How a token ablation works.** `image_position_ids == (-1, -1)` is the
model's own marker for a slot with no image token, and the processor already
pads 24 slots this way. `get_image_features` strips those slots before it
scatters the image embeddings into the text. So 1 ablation is 2 edits that
must agree:

1. mark the slot `(-1, -1)`, which removes 1 image embedding;
2. remove its `<|image|>` token, which removes 1 slot in the text.

`check_alignment` raises an error when they disagree. The position embedding
is factorized for each image token, so the remaining tokens keep their
coordinates: the model sees a missing token at (3, 7), not a shifted grid.

**Do not** use `<pad>`, `<unk>`, or an `<unusedN>` token. No training example
puts a text token between image tokens.

**Do not** use a flat-colour image patch. `patch_ln1` is a LayerNorm, so
every flat colour gives the same image token. It is content, not absence.

**The 2 controls** run before each image, and both must pass:

| Condition | Meaning | Pass |
|---|---|---|
| `none` | rebuild the inputs, remove nothing | `c` returns to 1e-3 |
| `all` | remove every image token | `c` moves at least 2.0 |

The `none` condition is not a no-op. It runs the same rebuild, so it catches
a bad index or a lost tensor. The `all` condition gives the natural
normalizer, and `parts_over_whole` in `map_stats.parquet` reports the sum over
image tokens against it.

```bash
sbatch --time=01:00:00 monocle/monocle.sub monocle.causal_maps \
    --cases subway_safety --n-images 20 --reference absent --group 2
```

Verify the path on 1 image first:

```bash
sbatch --time=00:30:00 monocle/monocle.sub monocle.patch_ablate \
    --case subway_safety
```

### Groups of image tokens

1 image token covers 48 px of the 768 px model image, which is a tyre or a
piece of concrete. `--group 2` replaces 2 x 2 blocks of image tokens instead,
so 1 ablation unit covers 96 px, and the grid becomes 8 x 8 with 64 units.

Do not use `max_soft_tokens=70` to get the same grid. The supported levels
are 70, 140, 280, 560, and 1120, and the level sets the **input size**, not
the image-token size:

| Level | Model input | Grid | 1 image token, in the 1024 px original |
|---|---|---|---|
| 70 | 384x384 | 8x8 | 128 px |
| 140 | 528x528 | 11x11 | 93 px |
| 280 (production) | 768x768 | 16x16 | 64 px |
| 560 | 1104x1104 | 23x23 | 45 px |

Level 70 gives the 8 x 8 grid, but it discards 3/4 of the pixels first, so
the model judges a blurred scene and the run leaves the production path. The
grouped replacement keeps the production input and costs 1/4 of the GPU time.

```bash
sbatch --time=01:00:00 monocle/monocle.sub monocle.causal_maps \
    --cases subway_safety --n-images 20 --n-donors 3 --layer 20 --group 2
```

Each run writes to its own folder, so no run overwrites the run it is
compared against: `<case>`, `<case>_g2`, `<case>_absent`, `<case>_g2_absent`.

### Additivity of the map

A grouped replacement is also the additivity test. Run the fine map and the
grouped map with the same seed, then:

```bash
python -m monocle.causal_maps --additivity --cases subway_safety --group 2
```

It compares each grouped unit against the sum of the 4 fine image tokens it
contains.

| Ratio group / sum of parts | Interpretation |
|---|---|
| about 1 | the image tokens act independently, and the map decomposes `c` |
| far below 1 | the image tokens are redundant, and the fine map double counts |
| above 1 | the image tokens interact, and the fine map misses the interaction |

**Warning:** the fine map does not decompose the judgment. The 256 image
tokens of 1 image sum to between -162 and +132, against a `c` of -4 to +19,
and the sum correlates with `c` at r = 0.10. Compare image tokens against
each other inside 1 panel. Do not read the colour of a whole panel as the
decision.

### Names and re-rendering

`photograph_id()` in `monocle/single.py` names a photograph. A Cyclomedia
face sits at `.../<unit>/faces/<face>.jpg`, so the file stem is only the face
letter, and 4 photographs of 4 units all write `B.png`. The unit goes in
front: `W0E966FV_B`.

Use `--rerender` to draw the figures again from the parquet, with no GPU:

```bash
python -m monocle.causal_maps --rerender --cases subway_safety
```

It rebuilds each name from the path, so it repairs a run made before the
naming fix without a new measurement. It also removes the panels of the
previous run, so an old name does not survive a rename.

```bash
sbatch --time=03:00:00 monocle/monocle.sub monocle.causal_maps \
    --cases subway_safety --n-images 20 --n-donors 3 --layer 20
```

Output: `outputs/_monocle/causal_maps/<case>/` with `panels/`,
`contact_sheet.png`, `maps.parquet` and `map_stats.parquet`.

Cost: about 256 forward passes for each image and source image, batched. 20
images with 3 source images run in well under an hour on 1 A6000. That cost
is acceptable for figures, and the result is a measured effect.

## Commands

```bash
pytest tests/test_patch_attribution.py tests/test_patch_causal.py \
    tests/test_patch_map_causal.py tests/test_causal_maps.py \
    tests/test_patch_ablate.py -v

# The figure path, absence reference. Prove it on 1 image first.
sbatch --time=00:30:00 monocle/monocle.sub monocle.patch_ablate \
    --case subway_safety
sbatch --time=01:00:00 monocle/monocle.sub monocle.causal_maps \
    --cases subway_safety --n-images 20 --reference absent --group 2

# The donor reference, for the comparison.
sbatch --time=03:00:00 monocle/monocle.sub monocle.causal_maps \
    --cases subway_safety --n-images 20 --n-donors 3 --layer 20

# Draw again from the parquet, or ask whether the map adds up. No GPU.
python -m monocle.causal_maps --rerender --cases subway_safety
python -m monocle.causal_maps --additivity --cases subway_safety --group 2

# Where the judgment sits at each depth, and what each route carries.
sbatch monocle/monocle.sub monocle.patch_swap --cases subway_safety
sbatch monocle/monocle.sub monocle.attn_knockout --cases subway_safety

# The 3 diagnostics, in the order to run them
sbatch monocle/monocle.sub monocle.patch_causal --grad-check --n-pairs 5
sbatch monocle/monocle.sub monocle.patch_causal --probe-path --n-pairs 4
sbatch monocle/monocle.sub monocle.patch_causal --n-pairs 50 --pool 3
```

## Scope

Every number here comes from runs registered under
`image_layout: interleaved_labels`. Production moved to `images_then_labels`
on 2026-08-20. Both layouts bind the second image, so these results hold for
the prompt template they measured. They are not comparable to a re-registered
set of runs. See [canonical-lens-prompt-path.md](canonical-lens-prompt-path.md).
