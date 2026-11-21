# The R-lens

R-lens is a change to the Jacobian lens, from a 2026 LessWrong post. The
Jacobian lens (J-lens) is a tuned-lens variant: a fitted linear translator per
layer into the final-layer basis, fitted from raw gradients. The R-lens is the
same translator, fitted from relevance coefficients instead of raw gradients.
It makes the early-layer readout legible. We built it, fitted it twice, and
tested it against the measured causal effect of each image token on
2026-08-23.

## Summary of results

| Question | Result |
|---|---|
| Does the text readout improve? | **yes**, by a wide margin |
| Does the image-token readout improve? | **yes** |
| Is the fit cheaper? | **yes** — 40% faster, 10 GiB less memory |
| Is the fit more stable? | **yes** — 15 times more converged |
| Does it predict the causal effect of an image token? | **no** |
| Does it name the content of an image token? | **yes**, at layers 36-46 only |
| Does it give a legible readout at the layers where the image acts? | **no** — 95% uninterpretable tokens at layers 12-24 |
| Do its directions work as steering vectors? | **no**, the same null result as the J-lens |

**Use the R-lens where you want a readout:** a profile across layers, a word
cloud, the layer at which a concept first becomes decodable. **Use
`causal_maps.py` when you need to know the causal effect of an image region.**
Do not use the first in place of the second.

The rest of this page is the proposal as written, then each stage as it ran.

Source: [R-lens: making J-lens more faithful on early
layers](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)

The architecture is in
[gemma4-12b-architecture.md](gemma4-12b-architecture.md). The fit we run today
is `monocle/jlens_fit.py` and `monocle/jlens_fit_mm.py`. The result this page
tries to recover is in [patch-attribution.md](patch-attribution.md).

## What R-lens changes

The fit does not change. The estimator, the corpus, the shards, the merge and
the readout path all stay. R-lens changes what flows backward.

Today the fit averages a raw gradient:

    lens_l(h) = unembed(J_l @ h),  J_l = E[dh_target / dh_l]

R-lens averages a relevance coefficient in the same place. The coefficients
come from Layer-wise Relevance Propagation. Three rules make them:

| Rule | Site | What it does |
|---|---|---|
| norm | RMSNorm | holds the denominator constant |
| identity | GELU | uses the linear part, not the derivative |
| half | a product of 2 branches | splits 50/50, with no double count |

Each rule is a stop-gradient. The forward value does not move, and the pass
count does not move. Thus an R fit costs what a J fit costs.

The post reports 3 results. Early layers show far fewer uninterpretable
tokens. A concept becomes decodable many layers earlier. An ablation of an R
direction hurts accuracy more than an ablation of a J direction or a
logit-lens direction. The gain grows with model size, and the smallest models
show none.

## Why we want it

### The text readout is late

`jlens_validate` stage B reads the prompt "Fact: the capital of the state
containing Dallas is". Top-1 by layer, for our 2 fitted lenses and for the
logit lens:

| Layer | text J-lens | multimodal J-lens | logit lens |
|---|---|---|---|
| 6 | ` mock` | `mathbf` | `[multimodal]` |
| 12 | `tı` | `צור` | `<pad>` |
| 18 | `<pad>` | ` datas` | `s` |
| 24 | `<pad>` | `<image\|>` | `(` |
| 30 | ` Seville` | `bahnhof` | ` situated` |
| 36 | ` Wisconsin` | ` यूनाइटेड` | **` Texas`** |
| 42 | **` Texas`** | **` Texas`** | ` Texas` |

The model answers " Texas". The logit lens finds it at L36. Both fitted
J-lenses find it at L42. Below L30 the 2 J-lenses give the
uninterpretable tokens that the post describes. L30 of the text lens
gives Spain, which is a wrong answer from another country and not a partial
answer.

### The image readout is later still

Stage B also reads the synthetic quadrant image: a dog in the top left, red on
the right. Exact-dog probability mass in the top-left quadrant, where 0.25 is
chance:

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|
| text lens | 0.19 | 0.00 | 0.10 | 0.04 | 0.03 | **0.95** | 0.91 | 0.98 |
| mm lens | 0.24 | 0.25 | 0.20 | 0.25 | 0.01 | 0.25 | **0.83** | 0.99 |

The multimodal lens sits at chance until L42, and we fitted it on image
tokens. That is the worse of the 2.

### The layer mismatch is the real cost

[patch-attribution.md](patch-attribution.md) measures where the image acts.
Activation patching of all 256 image tokens moves the logit difference (`c`)
by +0.969 at L20 and by +0.006 at L43. The image has its full
effect below L30.

So the lens becomes legible at L36-L42, and the image stops acting at
L30. The 2 layer ranges do not overlap. That gap retired the
activation-patching maps, and it is the one thing R-lens claims to close.

### A measured baseline already exists at L20

`patch_map_causal.py` patches the activation of each of the 256 image tokens
on its own and records the change in the logit difference (`dc`). It then
asks whether a 1-pass map predicts that measured activation-patching map. Top-50 image tokens, 8
pairs, subway_safety:

| Layer | sign accuracy | r |
|---|---|---|
| L12 | 0.56 | +0.13 |
| **L20** | **0.65** | **+0.43** |
| L24 | 0.57 | +0.15 |
| L28 | 0.46 | +0.12 |

L20 is the best layer, and it is not good enough to colour 256 image
tokens. This table is the criterion. An R map must exceed 0.65 and +0.43 at
L20.

**Correction, 2026-08-23.** Stage 4 found a batch-size fault in the baseline
that made this table. The sign column is void and the r column was
understated. The corrected criterion is **r +0.49 at L20**, and the
corrected table is in
[patch-attribution.md](patch-attribution.md). The conclusion does not change:
the corrected criterion is harder, and no R condition came near either
version.

## The code change

One new module, `monocle/rlens_rules.py`. It is a context manager. It patches
4 forward passes so that the value stays bit-identical and only the backward
differs.

```python
# norm rule -- Gemma4UnifiedRMSNorm._norm
return hidden_states * torch.pow(mean_squared, -0.5).detach()

# identity rule -- gelu_pytorch_tanh is x * Phi(x), and Phi is bounded
return x * phi_tanh(x).detach()            # forward == gelu(x), exactly

# half rule -- Gemma4UnifiedTextMLP.forward, which is a GeGLU
prod = 0.5 * (g * u.detach() + g.detach() * u)
return self.down_proj(prod)                # forward == g * u, exactly

# softmax rule -- eager attention only
attn = softmax(scores).detach()            # relevance goes through V alone
```

`jlens_fit.py` and `jlens_fit_mm.py` then take 1 flag, `--rules
{jacobian,lrp}`. Nothing else changes. The corpus, `dim_batch`, the shards,
the checkpoints, the merge, `jlens_read`, `jlens_validate` and `jlens_compare`
all work as they are.

## Gemma-4 pitfalls

The post fits Qwen and DeepSeek. Gemma-4 differs in 5 ways that touch these
rules.

1. **Eager attention is mandatory, and it is cheaper.** `jlens_fit` loads with
   `extract.load_model`, which takes the SDPA path, and SDPA hides the
   softmax. An R fit must load with `attn_route.load_eager`. We expected a
   cost from that. There is none — see "The R fit is cheaper" below.
2. **A layer holds 7 RMSNorm sites, not 1.** They are `input`,
   `post_attention`, `pre_feedforward`, `post_feedforward`, `q_norm`, `k_norm`
   and `v_norm`. The norm rule thus has much more effect here than on a
   Llama-style layer. Once the softmax is detached, `q_norm` and `k_norm`
   carry no relevance, so 5 sites matter.
3. **Massive activations are the largest risk.** The norm rule removes the
   only term that discounts a channel with a very large norm. Gemma-4 is the
   model where a raw cosine similarity of 1.0 is an artifact of exactly those
   channels, so the R rows may collapse onto a few dimensions. The check is
   cheap. Report the fraction of each row's variance that sits on the top-5
   channels by mean |h|, for J against R.
4. **`attention_k_eq_v` is true on the 8 global layers**, so V and K are one
   tensor. That breaks the half rule, which assumes 2 separate branches. The
   softmax rule removes the problem, because K then gets relevance only
   through the value path.
5. **`num_kv_shared_layers` is 0.** We do not need the adaptation that the
   post makes for DeepSeek.
6. **The relevance scale is smaller than the gradient scale.** The local
   `|R| / |J|` at L20 on 1 image is 0.004. The fitted translator shrinks
   far less, and by an amount that changes with the layer: L6 x12.8,
   L12 x21.4, L18 x7.3, L24 x3.7, L30 x3.9, L36
   x2.3, L42 x1.8, L46 x1.4. A top-k readout
   cannot see a
   scale, but `scoring.score_patches` takes a softmax over the lens logits,
   and a softmax can. The quadrant test would thus show "no signal" where
   the ranking is in fact perfect. `rlens_rules.match_scale` gives each R_l
   the Frobenius norm of the J_l at the same layer, which moves no ranking.

## How we test it

The post validates with pass@10 on text puzzles and with 30 multihop
questions. We hold a measured causal map that the post does not have.

| Test | Harness | Baseline on disk |
|---|---|---|
| text readout layer | `jlens_validate` stage B | " Texas" at L42 |
| image readout layer | `jlens_validate` quadrant | dog top-left at L36 / L42 |
| J against R agreement | `jlens_compare` | Jaccard and JS by layer |
| **map against the measured activation-patching map** | `patch_map_causal` | 0.65 sign, r +0.43, at L20 |
| direction ablation | `jlens_steer` | a null result with output collapse |

**Correction, 2026-08-23.** Both baselines in this table were
unusable. The map baseline carried a batch-size offset (see above). The
activation-addition baseline came from a 2026-07-24 run that read the
pre-consolidation pairs, which
[canonical-lens-prompt-path.md](canonical-lens-prompt-path.md) says must be
re-run. Stage 4 re-measured both conditions of each, so nothing compares an R
number against a stale J number.

Row 4 is the decisive one. Per layer, correlate the R readout of each image
token with the measured change in the logit difference (`dc`) of that token.
The J answer is known. The correlation peaks at L43, where activation
patching of the whole image moves the logit difference (`c`) by 0.006. If R
moves that peak down to L18-L24, we get a legible attribution method at
the layer where the model decides. It also gives a 1-forward-pass surrogate
for a map that costs 256 forward passes.

Row 5 is free. J directions, used as steering vectors, give a null result with
output collapse. The post predicts that R directions have a larger effect, so
the re-run tests an existing negative result directly.

## The plan

| Stage | Work | GPU cost | Criterion |
|---|---|---|---|
| 1 | `rlens_rules.py`, and its unit tests | none | **PASS** |
| 2 | text R fit, same wikitext corpus, 4 shards | 1 J-fit budget | **MIXED** — the text criterion passes, the image criterion does not |
| 3 | multimodal R fit, same `fit_images.json`, 4 shards | 1 J-fit budget | **PASS** |
| 4 | map against the measured activation-patching map, and the activation-addition re-run | 2 map jobs, 4 activation-addition jobs | **FAIL** — the decisive stage |

Stage 1 is independent and needs no GPU. Stage 2 is the decisive criterion. If
the Dallas prompt does not improve, the rules do not transfer to gemma-4, and
we stop there.

### Stage 1 result, 2026-08-23

`monocle/rlens_rules.py` holds the rules, and `--rules lrp` turns them on in
both fitters. An R fit writes to its own directory, so it can never overwrite
a J shard. `tests/test_rlens_rules.py` holds 52 tests that need no GPU.

Job 476103, `monocle.rlens_rules --check`, gemma-4-12B on klara, 25 s:

| Measure | Value | Result |
|---|---|---|
| max abs logit change | 0.000e+00 | PASS |
| softmax rule calls in 1 forward | 48 of 48 layers | PASS |
| `c` before and after | -3.8750 and -3.8750 | unchanged |
| gradient cosine similarity at L20, 256 image tokens | +0.2735 | the rules change the gradient |
| per-token cosine similarity | median +0.3666, min -0.1737, max +0.6724 | |
| `\|R\| / \|J\|` at L20 | 0.004 | see pitfall 6 |

The forward is exact to the last bit, and the backward is a different vector
that still points in the same general direction. That is what an R-lens must
look like before it is fitted.

3 changes to the plan came out of stage 1.

1. The attention half rule is dropped. Once the softmax is a constant, the
   query-key matmul carries no relevance, and the other matmul has 1
   differentiable input. A half rule there is a factor of 0.5 on every head,
   which a lens readout cannot see. This also removes pitfall 4.
2. The installer checks itself. It runs the stock `_norm` against the rule
   and refuses on any difference, so a later transformers release fails
   with an error instead of a silent fit of a wrong lens.
3. The softmax rule counts its own calls. A config check can miss a fused
   kernel; a count of 0 after a forward pass cannot.

## What R-lens is not

R-lens does not replace the J-lens. J has a first-order causal interpretation,
and R does not. The post says so itself: R tracks the flow of computation more
than the presence of a concept.

For a claim about what the model represents in a street scene, that
difference is the decisive one. So both lenses stay on disk, each figure names
the lens that made it, and no R result enters the summary of claims until the
measured causal map supports it.

### The R fit is cheaper than the J fit, 2026-08-23

Job 476180 against the July J smoke test, job 148106. Same script, same node,
same prompt, 128 tokens with 111 valid positions:

| dim_batch | J s/prompt | R s/prompt | J peak | R peak |
|---|---|---|---|---|
| 8 | 180 | 108 | 31.6 GiB | 26.3 GiB |
| 16 | 175 | 105 | 40.6 GiB | 30.3 GiB |
| 32 | — | 104 | — | 38.3 GiB |

The R fit runs 40% faster and uses 10 GiB less memory, with eager attention,
which we budgeted as an overhead. The softmax rule is the reason. With the
attention probabilities held constant, the backward pass never crosses the
query-key matmul, so the gradients through `q_proj` and `k_proj` and the
whole attention-weight graph do not exist. That saving is larger than the
cost of eager attention in the forward.

The time curve is flat from 8 to 16 to 32. The shards use `dim_batch=16`,
which is also what the July J fit used, so the 2 fits match in every setting
we control.

The multimodal fit repeats the saving. Job 476825 against the July mm smoke
test, 1 image, seq_len 275, 256 source image tokens:

| dim_batch | J s/image | R s/image | J peak | R peak |
|---|---|---|---|---|
| 4 | 450 | 232 | 32.8 GiB | 27.7 GiB |
| 8 | 417 | 230 | 43.4 GiB | 33.1 GiB |
| 12 | — | 216 | — | 38.5 GiB |
| 16 | — | 227 | — | 43.9 GiB |

The mm shards use `dim_batch=8`, which the July mm J fit also used. 12 is 6%
faster for 5.4 GiB more, and 16 is slower than 12.

## Stage 2 result, 2026-08-23

The text R-lens is fitted: 4 shards, 128 wikitext prompts, the same corpus
and the same 8 source layers as the July J lens. Jobs 476263-476266, about 1
hour each. `--rescale-against` matched each layer to the J norm before we
computed any score. Job 476793 ran `jlens_validate`.

### The text readout: a clear pass

Top-5 on "Fact: the capital of the state containing Dallas is":

| Layer | J-lens | R-lens | logit lens |
|---|---|---|---|
| 6 | ` mock` | `,` ` and` ` ` `.` ` a` | `[multimodal]` |
| 12 | `tı` | `\n` ` thing` ` alot` ` she` | `<pad>` |
| 18 | `<pad>` | `\n` `...` ` of` ` or` | `s` |
| 24 | `<pad>` | `\n` `...` ` (` ` of` | `(` |
| 30 | ` Seville` | ` city` ` county` ` statewide` | ` situated` |
| 36 | ` Wisconsin` | **` Texas`** ` Illinois` ` Arizona` | ` Texas` |
| 42 | ` Texas` | ` Texas` | ` Texas` |

3 things change. No uninterpretable token appears at any layer. ` Texas`
arrives at L36 and not L42, so the R-lens matches the logit lens
instead of a lag of 6 layers behind it. L30 decodes geography words
where the J-lens decoded Spain.

**Warning:** function words at L6-L24 do not prove that the lens resolves
early content. A collapse toward the unigram prior looks the same. L30
and L36 argue against that interpretation, because county and state names are
not high-frequency filler. The causal test in stage 4 settles it. A token
list never will.

### The image readout: not a pass

Exact-dog probability mass in the top-left quadrant, and red probability mass
in the right half:

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|
| J dog TL | 0.19 | 0.00 | 0.10 | 0.04 | 0.03 | **0.95** | 0.91 | 0.98 |
| R dog TL | 0.25 | 0.04 | 0.03 | 0.01 | 0.08 | **0.44** | 0.94 | 0.87 |
| J red right | 0.39 | 0.98 | 0.55 | 0.53 | 0.32 | 0.94 | 1.00 | 1.00 |
| R red right | 0.49 | 0.50 | 0.67 | **0.89** | 0.95 | 1.00 | 1.00 | 0.96 |

The dog criterion does not move earlier. Both lenses first hold it at L36,
and the R mass there is 0.44 against 0.95, which is weaker.

The red criterion gives a better result, but not the one the criterion asked
for. The J profile is not monotone: 0.98 at L12, then 0.55, 0.53 and
0.32. A first crossing at L12 that collapses 6 layers later is noise,
not an early readout. The R profile rises once and stays up, and holds from
L24.

So stage 2 is mixed. The text criterion passes by a wide margin. The image
criterion does not, and this lens is fitted on text. Stage 3 fits the same
rules on image tokens, which is the case that the image criterion is really
about.

## Stage 3 result, 2026-08-23

The multimodal R-lens is fitted: 4 shards, 128 images, the same
`fit_images.json` and the same 8 source layers as the July mm J lens. Jobs
476915-476918, about 2 hours each. `--rescale-against` matched each layer to
the mm J norm. Job 477800 ran `jlens_validate`.

### The quadrant readout: a pass

The dog criterion is not a mass threshold. It asks whether the top-left
quadrant holds the most exact-`dog` mass of the 4. Chance is 0.25.

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|
| mm J, dog TL | 0.24 | 0.25 | 0.20 | 0.25 | 0.01 | 0.25 | **0.83** | 0.99 |
| mm R, dog TL | 0.25 | 0.13 | 0.35 | 0.30 | **0.51** | 0.22 | **0.88** | 0.98 |
| mm J, red right | 0.42 | 0.45 | 0.91 | 0.50 | 0.90 | 0.65 | 0.98 | 0.97 |
| mm R, red right | 0.67 | **0.98** | 0.97 | 0.94 | 0.88 | 0.97 | 1.00 | 1.00 |

**The red result is the strong one.** The R-lens holds the right half from
L12 and never drops below 0.88 again. The J-lens oscillates the whole
way: 0.42, 0.45, 0.91, 0.50, 0.90, 0.65. It only settles at L42. A lens
that finds a thing and then loses it 6 layers later did not find it.

The dog criterion first holds at L30 for R against L42 for J, which
is 12 layers earlier and meets the criterion.

**Warning:** the R dog profile is not monotone. TL is the argmax at L30,
is not at L36, and is again from L42. A single crossing at L30
with a gap above it is weaker evidence than the red profile, and we must not
quote L30 as the layer at which the concept first becomes decodable
without that qualification.

### The text prompt, decoded by an image-fitted lens

` Texas` is top-1 at L36, where the mm J lens needed L42. L24
and L30 decode ` town`, ` neighborhoods`, ` county` and ` southernmost`. L6
and L12 still give uninterpretable tokens, which is expected: this lens is
fitted on image tokens only.

### The scale profile says where the gradient fails

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|
| mm, scale to match J | x155.1 | x60.0 | x9.7 | x5.0 | x5.0 | x4.0 | x2.6 | x1.7 |
| text, for comparison | x12.8 | x21.4 | x7.3 | x3.7 | x3.9 | x2.3 | x1.8 | x1.4 |

At L6 the image-token-fitted R translator is 155 times smaller than the
J one, against 12.8 for text. A component that the rules discard thus
dominates the raw gradient on an image token at L6. That is the failure
the whole exercise addresses, and it appears here as a magnitude.

### The fit is also more stable

`max_d_mean` is the relative shift that the next image makes to the running
mean. Same 32 images, same order, shard 0:

| Image | J fit | R fit |
|---|---|---|
| 5 | 4.54e-01 | 5.5e-02 |
| 16 | 1.60e-01 | 2.1e-02 |
| 32 | 1.20e-01 | **8.1e-03** |

The R fit ends 15 times more converged, and it still falls where the J fit
flattened. The remaining variance of the J fit was thus not a shortage
of images. It was variance that the estimator itself carried.

## Stage 4 result, 2026-08-23 — the R-lens fails the causal criterion

Stages 2 and 3 say the R-lens gives a better readout. Stage 4 asks the
question the whole proposal was built around: does it predict the causal
effect of an image token.

### A bug we had to fix first

`patch_map_causal` ran the 256 patched forward passes at batch 8 and took the
baseline `c0` at batch 1. In bfloat16 a forward pass depends on the size of
its batch, so every delta of an image carried a constant offset. Measured:
the per-map median of the 256 deltas sat at +0.06 to +0.18, and single maps
ran from -0.50 to +1.23, against a typical absolute delta of 0.31.

A Pearson score cannot see a constant. A sign score sees nothing else. Thus
the sign column of the 2026-08-22 run is void, and its Pearson column is
sound. `batched_baseline` now takes the baseline at the batch size of the
patched passes. The median correction was 0.349 logits, and the per-map
medians fell to -0.06 to +0.06.

### The result

Top-50 image tokens by absolute causal effect, mean over the 6 source-target
image pairs that pass `MIN_GAP`, `sign / r`. Jobs 478063 and 478064.
`python -m monocle.lens_bias --verify` reproduces every cell of this table.

| Layer | taylor `<g, dh>` | R `lens_delta` | J `lens_delta` |
|---|---|---|---|
| 12 | 0.607 / +0.198 | 0.530 / +0.098 | 0.520 / -0.073 |
| 18 | 0.587 / +0.348 | 0.547 / +0.076 | 0.477 / -0.073 |
| 20 | **0.597 / +0.489** | not fitted | not fitted |
| 24 | 0.637 / +0.299 | 0.503 / -0.013 | 0.537 / +0.040 |

**No lens condition comes near.** The 1-backward-pass local Jacobian stays
the best predictor of an activation-patching effect, and the fix raised it (L20 r
went from +0.347 to +0.489) while the lens conditions stayed at 0. The gap
widened.

R beats J: R is positive at L12 and L18 where J is negative. So the
rules do move the translator from the wrong sign to a weak correct sign. It
is not enough to matter.

`lens_own`, the map we would in fact render, scored -0.02 to +0.00 at every
layer for both lenses. The words that an image token's hidden state decodes
to have no relation to the effect of a change to that token.

The criterion is thus not met. The correct target for any future attempt is
the corrected taylor row: **+0.489 at L20, +0.348 at L18, +0.299 at
L24.**

### Activation addition: a null result with output collapse, on both lenses

The old activation-addition result was not usable. Both 2026-07-24 runs read
the 2026-06-29 pairwise parquet, which predates the prompt consolidation. So
we re-ran all 4 conditions on the canonical path: jobs 478048, 478049,
478051, 478052.

Activation addition collapses the model. Of the 18 (layer, alpha) settings
per condition, the count whose top label holds more than 70% of the 100
pairs:

| Condition | signal | control |
|---|---|---|
| J, last token position | 16/18 | 17/18 |
| R, last token position | 16/18 | 12/18 |
| J, image tokens | 6/18 | 9/18 |
| R, image tokens | 11/18 | 12/18 |

A collapsed condition scores about 0.39 exact agreement, against 0.18 without
activation addition, because one label matches 39% of the production labels.
That is not a steering effect. It is a constant output whose accuracy equals
a class frequency.

Restrict to the 9 settings where neither condition is degenerate, and compare
signal against control at a matched layer and coefficient alpha: signal wins
6 of 9, p = 0.508. R at the image tokens goes 0 of 3, with control ahead by
0.260.

**Thus the null result with output collapse holds, now measured on the
canonical prompt and with both lenses.** The post's result 4 does not
replicate here.

One caveat, and it does not change the result. The coefficient alpha grid runs from 2 to 8
typical residual-stream norms, and at the last token position that is clearly
too strong: 16 of 18 settings collapse. A weaker grid may find a regime that
does not break the model. We did not run one.

**Update, 2026-09-01: we ran the weaker grid, and the caveat is closed.**
[steering-weak-alpha.md](steering-weak-alpha.md) sweeps coefficient alpha in
centered-norm units (about 150x weaker at L24) with both the transpose
and the pseudo-inverse steering direction. The collapse persists for signal
and control alike, the 4 clean settings give a null result, and no setting
follows the sign of alpha. The fitted translator also cannot reconstruct the
label direction at all (relative error 0.998 or worse at every layer), which
gives the null result a structural cause. The activation-addition line of
work is closed, not only untested.

## Conclusion

| Stage | Question | Result |
|---|---|---|
| 1 | Do the rules install exactly? | PASS |
| 2 | Does the text readout improve? | PASS, by a wide margin |
| 2 | Does the image readout improve? | no, but the lens was text-fitted |
| 3 | Does the image-token-fitted readout improve? | PASS |
| 4 | Does the readout predict the causal effect of an image token? | **FAIL** |
| 4 | Do R directions work better than J directions as steering vectors? | **no, both give a null result** |

R-lens is a better readout of gemma-4-12B and a cheaper, more stable fit. It
is not a causal attribution method, and it is not a surrogate for a
token-ablation map.

This is the post's own limitation, measured on our data: R-lens tracks the
flow of the computation, not the presence of a concept. Our measured
token-ablation map separates those 2 things, and the lens lands on the wrong
side.

**The rule is unchanged.** Use the R-lens where you want a readout: a profile
across layers, a word cloud, the layer at which a concept first becomes
decodable. Use `causal_maps.py` when you need to know the causal effect of an
image region. Do not use the first in place of the second. The
activation-patching maps failed for exactly this reason, and R-lens does not
restore them.

## The causal-word sweep, 2026-08-25

Stage 4 asked whether an R map predicts the causal effect of an image token.
It does not. This sweep asks the other question: does the R-lens name the
content of an image token that has a measured ablation effect? The
token-ablation map selects the image tokens, so the lens only describes them.
That is the readout task this page keeps the R-lens for.

The design is paired. Only the lens changes: 7 cases, 60 photographs, the 8
fitted layers, the same seed and the same token-ablation map.
`monocle.causal_words` writes the words and `scripts/rlens_word_report.py`
scores 3 layers. The J condition at layers 12, 18 and 24 repeats the 2026-07
run, which checks the scorer.

### The R-lens names the image region, and the J-lens does not

Each word of each sign goes to the caption of that same image region. The
baseline is the caption of another photograph, so a word must name this
scene.

| Layer | J-lens passes | R-lens passes |
|---|---|---|
| 36 | 0 of 7 | **3 of 7** |
| 42 | 1 of 7 | **4 of 7**, q=0.009 |
| 46 | 2 of 7 | **3 of 7** |

At layer 42 the uninterpretable-token rate falls from 0.136 to 0.084, and the
target-word share rises from 0.040 to 0.062. This is the first test of the
image-token readout on real photographs. The quadrant test used 1 synthetic
image.

### The legible layers still do not reach the layers where the image acts

| Layer | J uninterpretable-token rate | R uninterpretable-token rate | does the image act? |
|---|---|---|---|
| 12 | 0.298 | **0.961** | yes |
| 18 | 0.328 | **0.954** | yes |
| 24 | 0.783 | **0.867** | yes |
| 36 | 0.264 | 0.147 | no |
| 42 | 0.136 | 0.084 | no |

Below layer 30 the R-lens gives the single token `0` about 95 times in 100.
**That is pitfall 3, and it is now measured on real images.** The norm rule
drops the term that discounts a channel with a very large norm, and the R
rows collapse.

Between layers 24 and 30 the R-lens does give words, and they do not change
with the case: `street`, `pastel`, `scooter`, `neighborhood`, `ornate`. One
list serves a subway entrance, a library facade and a park. That is the
collapse toward the unigram prior that the stage 2 warning names.

### The conclusion does not change

**Use the R-lens for a readout above layer 30. Do not use it below layer 30.**
The R-lens is the better lens on every readout measure we have. It is not a
legible readout at the layers where the image acts.

The sweep also settles a question this page could not answer. The 2 signs of
the token-ablation map carry the same words, under either lens, at every
layer: 2 of 56 rows for J and 1 of 56 for R, where chance gives about 2.8.
The sign of an image token's effect is not in the content that it decodes to.
Only the gradient carries it. See
[causal-lens-retrospective.md](causal-lens-retrospective.md).
