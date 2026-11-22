# Where the judgment is decodable, by layer

The judgment does not stay at the image tokens. It moves through 3 position
groups, and a method that reads the wrong position group at the wrong layer
measures nothing.

This page is the result of `research/monocle/patch_swap.py` and
`research/monocle/attn_knockout.py`. The method is in
[patch-attribution.md](patch-attribution.md) and the prompt path is in
[single-image-lens-path.md](single-image-lens-path.md).

## Information flow

We patch 1 position group of a target image with the activations of the same
group from a source image at layer L. We then measure the fraction of the
logit difference (`c`) recovered toward the source image. 12 pairs,
subway_safety, single-image path.

| Layer | image tokens | question tokens | last token position |
|---|---|---|---|
| L5 | **1.093** | -0.000 | 0.008 |
| L20 | **0.969** | 0.276 | 0.021 |
| L24 | 0.738 | **0.650** | -0.014 |
| L28 | 0.182 | **0.747** | 0.152 |
| L30 | 0.045 | **0.670** | 0.291 |
| L34 | 0.016 | 0.262 | **0.683** |
| L42 | 0.010 | 0.137 | **0.802** |
| L46 | 0.007 | 0.002 | **0.991** |

| Layers | Position group that holds the judgment |
|---|---|
| 0-21 | the image tokens |
| 22-29 | the transfer to the question tokens |
| 30-41 | the transfer to the last token position |
| 42-47 | the last token position alone |

The transfer is gradual, not abrupt. Whole-image activation patching recovers
0.994 at L22, 0.738 at L24, 0.413 at L26, 0.182 at L28 and 0.045 at L30.

### Information flow at every layer, on the corrected prompt (2026-08-23)

The table above ran before `build_single_inputs` appended the forced answer
prefix, and at 7 layers. The re-run reads all 48 layers on the corrected
prompt. The result is cleaner: the whole-image fraction recovered no longer
overshoots (it reached 1.394 at L22 before, and now it stays between 0.96 and
1.06), and the 2 crossover points are sharp.

| Layers | Position group | Crossover |
|---|---|---|
| 0-22 | the image tokens | image 0.650 = question 0.635 at **L23** |
| 23-30 | the question tokens, peak **0.817 at L27** | question 0.416 vs answer 0.472 at **L31** |
| 31-47 | the last token position, 0.985 by L46 | |

`text_before` and `boi` give **0.000 at all 48 layers**, which is 96 exact
zero values.

The question tokens keep a share of the effect: their condition recovers 0.19
to 0.30 from L33 to L44, long after the last token position holds the largest
share. The transfer is a split between position groups, not a complete move.

### Which word (2026-08-23)

`research/monocle/question_tokens.py` patches 1 question token at a time. Unlike the
image tokens, the question tokens are additive: the 66 single-token patching
effects sum to 1.02 of the whole-question patching effect at L33.

| Word | first layer with an effect | peak |
|---|---|---|
| ` this`, the demonstrative that points at the image | **L22** | 0.066 |
| ` Dangerous`, a label in the prompt | L25 | **0.189** |
| ` Safe`, a label in the prompt | L25 | 0.052 |
| the newline after `model` | L27 | 0.133 |
| `":`, the last token of the prompt | L29 | 0.170 |

The model writes the image evidence onto the question tokens that name the
answer options. See [[do-the-colours-mean-anything]] for the batch-size
requirement of this measurement.

## 2 controls validate the measurement

`text_before` and `boi` sit before the image and give **exactly 0.000** at
every layer. They cannot differ between 2 images, and the measurement confirms
this. That is the negative control.

The positive control patches the last token position. It must move `c` to the
source-image value when little computation remains above L, and it does: 0.008
at L5, 0.291 at L30, 0.802 at L42, **0.991 at L46**.

**Warning:** activation patching with an identical source is not a valid
criterion. A write of the value already present and no write at all give the
same answer. The first version of `patch_swap` passed that criterion while
every condition gave zero.

## The attention knockout agrees

`attn_knockout.py` cuts attention instead of patching activations, so it is an
independent test. Mean |change in c|, 12 images:

| Cut | 0-21 | 22-29 | 30-47 | 0-47 |
|---|---|---|---|---|
| the whole residual stream cut off from the image tokens | 1.490 | **2.933** | 0.458 | 7.681 |
| the last token cut off from the question tokens | 1.498 | 7.925 | 8.799 | **10.274** |

The effect of the removal of image attention peaks at layers 22-29, which is
the transfer that activation patching found. Above L30 the effect is 0.458,
about 6% of the total.

## The last image token receives concentrated attention

The last token position attends to the final image token of the image span
80-105x more than to any other image token. A cut of that 1 attention edge at
layer 47 moves `c` by **0.131** out of a range of 17. Activation patching of
the residual stream at that image token moves `c` no more than patching of an
ordinary image token, at every layer.

Thus the attention concentration is real, and the model loses almost nothing
without it. The correct description is that the last token attends to the
image span through that token. The incorrect description is that the model
routes the image content into that token.

## Consequence for the earlier attribution window

We chose the attribution window L40-L45 because the attribution-map aggregate
tracks `c` at r about 0.8 there. That window has no causal effect. The metric
rewards a layer whose residual stream is linearly aligned with the decision,
which occurs after the model makes the decision.

| Layer | map tracks `c` | image tokens cause `c` |
|---|---|---|
| L18-L24 | about 0 | about 1.0 |
| L43 | 0.805 | 0.006 |

Do not select a layer by that correlation. See the figure path in
[patch-attribution.md](patch-attribution.md).
