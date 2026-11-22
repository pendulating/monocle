# Lens corpus faithfulness

Status: preregistration. We wrote this before we fitted a lens.
Date: 2026-08-24.

## The answer, first

We never wrote down what the Jacobian lens estimates. We took the corpus from
a paper. The corpus is not a nuisance parameter here — it is the first-order
one, and our own diagnostic says so.

| Question | Answer |
|---|---|
| Does the corpus change the readout? | Yes, at every layer we read. |
| Which axis matters most? | Not the text topic. The task shape. |
| Is the image set already faithful? | Yes. It is also held out. |
| Will a faithful corpus fix the causal gate? | Partly. We predict a cap. |

## The estimand

The lens computes one constant matrix:

    J_bar = E[J(x)],  x ~ D

`D` is a choice. We never made it. We inherited WikiText from the reference
work and then added corpora by instinct. Rigour starts one step before the
corpus: name the population, and show that the population matches the claim.

Three claims need three different `D`:

| Claim | Correct D | What we fitted |
|---|---|---|
| gemma-4 maps residual content to logits like this | broad text | `jlens/` |
| Patch content in a street image surfaces as these words | street images, neutral context | `jlens/mm/` |
| The safety judgment enters the workspace at L42 | the production pairwise prompt | nothing |

Our papers make the third claim. Our lens estimates the first or the second.

## What we measured before we started

### The text topic washes out. The modality does not.

Top-10 token overlap on one real Brooklyn face, lens against lens:

| Layer | WikiText vs urban | WikiText vs mm |
|---|---|---|
| 18 | 0.223 | 0.036 |
| 24 | 0.324 | 0.070 |
| 30 | 0.299 | 0.002 |
| 36 | 0.443 | 0.027 |
| 42 | 0.707 | 0.194 |
| 46 | 0.611 | 0.120 |

A change of topic keeps the deep readout. A change of modality does not.

### Our own floor already fails

`research/monocle/judgment_trace.py` prints this rule: a layer below 0.8 reads its
corpus, not the model. Here is that check, mm against WikiText, 8 images,
72 rows:

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 | 47 |
|---|---|---|---|---|---|---|---|---|---|
| median r | 0.08 | 0.32 | 0.14 | 0.53 | 0.41 | 0.32 | 0.67 | 0.73 | 1.00 |

Every layer we read fails the floor. L47 passes, but there the lens is close
to the identity, so it proves nothing.

### The two endpoints of the average

Stage 4 of the R-lens study measured both ends of the corpus width. We
reproduced every cell below from `maps.parquet` before we wrote this.

**The metric.** Take the top 50 patches by absolute causal effect. Score
Pearson `r` between the causal map and the candidate map on those 50. Take
the mean over the 6 donor-target pairs that pass `MIN_GAP`. Every rung uses
this metric, unchanged.

| Layer | Taylor `<g, dh>`, local | J `lens_delta`, fitted | R `lens_delta`, fitted |
|---|---|---|---|
| 12 | +0.198 | -0.073 | +0.098 |
| 18 | **+0.348** | -0.073 | +0.076 |
| 24 | +0.299 | +0.040 | -0.013 |
| 20 | **+0.489** | not fitted | not fitted |

The local instrument wins at every layer. The fitted transport sits at zero:
the J arm runs -0.073 to +0.040 across the three fitted layers.

Warning: L20 is NOT a fitted layer. The lens is fitted at
`[6, 12, 18, 24, 30, 36, 42, 46]`. So +0.489 is the local instrument's best
score, not a layer-matched target. The layer-matched targets are **+0.348 at
L18** and **+0.299 at L24**.

## How much the gate can see

We bootstrapped the stage-4 numbers over the 6 donor-target pairs, which are
the gate's only replicates. 20000 resamples, seed 777.

| Layer | arm | mean | 95% interval | width |
|---|---|---|---|---|
| 12 | taylor | +0.198 | [+0.124, +0.273] | 0.149 |
| 12 | `lens_delta` | -0.073 | [-0.178, +0.068] | 0.246 |
| 18 | taylor | +0.348 | [+0.276, +0.415] | 0.139 |
| 18 | `lens_delta` | **-0.073** | **[-0.172, +0.041]** | **0.213** |
| 20 | taylor | +0.489 | [+0.415, +0.565] | 0.149 |
| 24 | taylor | +0.299 | [+0.088, +0.540] | 0.452 |
| 24 | `lens_delta` | +0.040 | [-0.057, +0.137] | 0.194 |

Three rules come out of this table, and we adopt them now:

1. **L18 is the primary layer.** It has the tightest lens interval of the
   fitted layers, and it carries the layer-matched local target of +0.348.
2. **L24 cannot carry a conclusion.** Its taylor interval is 0.452 wide. We
   report it, and we do not argue from it.
3. **Rung comparisons use a PAIRED test.** Every rung reads the same causal
   maps over the same pairs, so the per-pair difference is far tighter than
   the gap between two intervals. We compare rungs by the 6 (or more)
   per-pair differences, never by whether 2 intervals overlap.

We also raise the replicate count. The gate runs at `--n-images 12`, which
builds 6 pairs. We run every rung at 12 AND at 24 images. The 12-image run
keeps the link to the record. The 24-image run roughly doubles the
replicates, and the analysis-reserve of 32 still holds the fit corpus clear
of it.

## The five axes

We compare the mm fit against the distribution we analyze.

We have TWO analysis distributions, and they differ. The ladder targets the
one that holds the number we must move.

| Analysis | Path | Images | Prompt |
|---|---|---|---|
| Causal gate, `patch_map_causal` | `research/monocle/single.py` | **1** | `rating_2point`, per case |
| Workspace trace, `judgment_trace` | `research/monocle/canonical.py` | 2 | production pairwise |

The causal gate holds the +0.10 against +0.49 result. Thus the ladder targets
the single-image path. A pairwise lens is separate work.

| Axis | mm fit | Causal-gate analysis | Match |
|---|---|---|---|
| Image count | 1 | 1 | **yes** |
| Source positions | image patches | image patches | **yes** |
| Image content | 256 cyclomedia faces | 8519 cyclomedia recordings | **yes** |
| Prompt | "Describe this scene." | `rating_2point`, per case | no |
| Target positions | first patch onward | the last position | no |
| Decode | free | a 1-token label pair | no |

Three axes already match. The image axis is also disjoint: 1 of the 256 fit
recordings appears among the 8519 analysis recordings. So image content cannot
explain the failure. The task shape can.

The target span is our first suspect. We inject cotangents over every position
from the first patch onward. The gate reads ONE position, the one that emits
the label. That average smears the transport over positions we never read.

## The circularity trap

A narrow `D` gives a low-bias lens. It also lets the lens learn the task. In
the limit `D` holds the exact prompt we analyze, and "the judgment appears at
L42" becomes near-tautological.

We do not solve this by a wide corpus. We solve it by a hold-out. Every rung
below fits on cases and images that the analysis never touches.

This leaves a real tension, and we think it is the result rather than an
obstacle. `D` must stay wide enough to give a shared basis, and tight enough
to keep the bias low. Step 5 measures that curve.

## The predictions

We record these now so we cannot explain the outcome after we see it.

| # | Prediction | Falsified when |
|---|---|---|
| P1 | The target span carries the largest single effect. | Another rung moves the gate more. |
| P2 | The prompt text carries a moderate effect. | Its rung moves the gate by less than 0.03. |
| P3 | Image content carries almost no effect. | Its rung moves the gate by more than 0.05. |
| P4 | R3 `lens_delta` reaches +0.15 or above at L18. | It stays below +0.15. |
| P5 | R3 does not reach the layer-matched local score at L18 (+0.348). | A fitted lens reaches +0.32 or above at L18. |
| P6 | Steering stays null. | A steered arm beats its control at p < 0.05. |

P5 is the important one. A tight corpus gives a better constant. It does not
give a variable. If P5 fails, the constant-matrix account of the stage-4 result
is wrong, and we must say so.

The J baseline at L18 is -0.073. So P4 asks for a rise of about 0.22, and P5
says the rise stops short of 0.42.

P6 is separate. Steering failed for its own reason: only 9 of 72 cells stayed
non-degenerate under the alpha grid. A better corpus does not repair that.

## The design

### Step 1 — the estimand

This document. No GPU work.

### Step 2 — the ablation ladder

One change per rung, from the current lens to the faithful one. Every rung
uses the `jacobian` rules, so the corpus stays the only variable.

| Rung | Change from the rung above | Isolates |
|---|---|---|
| R0 | none — the current `jlens/mm/` lens | baseline |
| R1 | target mask = the last position only | the target span |
| R2 | prompt = `rating_2point` for the case, forced prefix | the prompt text |
| R3 | images from the analysis pool, held out | the image content |

R3 is the faithful lens. We re-fit R3 under the `lrp` rules afterwards, so the
rule and the corpus never move together.

R2 makes the lens case-specific. That is a real cost: one lens per case, and
no shared basis across cases. Step 5 measures what that cost buys.

### Step 3 — the gates

We re-run the gates we already have, unchanged, so every number compares
directly to the record.

| Gate | Tool | Number on record |
|---|---|---|
| Readout legibility | `research/monocle/jlens_validate.py` | early layers give punctuation |
| Corpus dependence | `research/monocle/judgment_trace.py` | fails 0.8 at every read layer |
| Causal gate | `research/monocle/patch_map_causal.py` | `lens_delta` -0.073 at L18 |
| Steering | `research/monocle/jlens_steer.py` | 9 of 72 cells usable |

### Step 4 — the bias measure

The distance between the fitted transport and the local truth. We score it in
the currency the gate reads, not in matrix norm, because a matrix norm needs a
full local Jacobian and the gate does not read one:

    bias(l) = r( taylor_map , lens_delta_map ),  top 50 by |causal|

The taylor map IS the local instrument. So this number asks how much of the
local truth the fitted transport keeps. It costs no GPU time: both maps are
already columns of `maps.parquet`. `research/monocle/lens_bias.py` computes it, and it
reproduces the stage-4 table as its own check.

### Step 5 — the width sweep

We fit at several corpus widths and plot two curves against width:

- bias to the local Jacobian, which falls as the corpus narrows
- agreement across held-out inputs, which falls as the corpus narrows

The crossing point is the design answer. It also tells a reader why a lens
must average at all.

## Run log

### R0 reproduces the lens it copies, 2026-08-24

The harness check. R0 uses the same 128 images, the same describe-contexts and
the same target span as `outputs/_monocle/jlens/mm/gemma4_12b_lens.pt`, so it
must match it. `research/scripts/check_ladder_r0.py`:

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|
| rel diff | 0.0004 | 0.0004 | 0.0004 | 0.0002 | 0.0003 | 0.0003 | 0.0003 | 0.0003 |
| cos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Both lenses hold `n_prompts=128`. The tolerance is 0.02, so the check passes
by a factor of 50. The residual is bf16 nondeterminism: R0 splits the corpus
8 ways and the mm fit split it 4 ways.

Warning: the first run of this check printed cosines above 1.0, which cannot
happen. A float32 reduction over 14.7M elements loses that much precision, and
Gemma-4's massive activations widen the range and make it worse. The script
now compares in float64. The gate number is a norm ratio and never moved.

### The R2/R3 memory fault, 2026-08-24

All 16 R2 and R3 jobs stopped with CUDA OOM on image 0. The cause is the
prompt length: `rating_2point` builds 329 tokens where the describe-context
builds 275, and gemma-4 computes logits at EVERY position. At `dim_batch=8`
that tensor is `[8, 329, 262144]`, and the retained graph holds it, and its
softcap temporaries, across all 480 backward passes.

**We did not lower `dim_batch`.** A bf16 forward depends on its batch size —
that is the fault which voided the stage-4 sign column. A smaller `dim_batch`
for R2 and R3 alone would tie the prompt axis to a numerics change, and the
rung would no longer isolate one axis.

We set `logits_to_keep=1` instead. The LM head sits after the last decoder
layer, so it is off the recording path and off the gradient path from
`target_layer` to the source layers. That argument predicts the fitted
Jacobian does not move, and `--check-head` measures it rather than trusting
it. All 8 R1 shards started before the flag existed, so R0 and R1 stay
internally consistent whatever the check returns.

### The LM-head flag is inert, and how we nearly got that wrong

`--check-head` fits one image twice and compares. `--check-head --repeat` runs
ONE setting twice, which sizes the noise floor. Both are needed: a difference
means nothing until you know what an identical repeat costs.

| Config | flag effect | repeat control | reading |
|---|---|---|---|
| R1, `dim_batch=8` | **0.000e+00** | **0.000e+00** | deterministic and exactly inert |
| R2, `dim_batch=4` | 2.17e-03 | 7.45e-04 | nondeterministic |

The `dim_batch=4` row first read as "the flag is not inert". It is not. Put the
2 runs side by side, per layer, as a RELATIVE difference:

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|
| flag | 2.93e-02 | 2.64e-02 | 1.75e-02 | 1.21e-02 | 8.09e-03 | 5.51e-03 | 2.28e-03 | 0.0 |
| noise | 2.55e-02 | 2.26e-02 | 1.72e-02 | 1.16e-02 | 8.04e-03 | 5.64e-03 | 2.30e-03 | 0.0 |
| ratio | 1.15 | 1.17 | 1.02 | 1.04 | 1.01 | 0.98 | 0.99 | -- |

Same curve, same decay, and both exactly zero at L46. That is 2 draws from one
nondeterministic process. The error falls as the backward path shortens, which
is what a reduction-order difference does.

**A lesson about the summary statistic.** `worst abs diff` compares absolute
maxima at different layers, and it turned a ratio of 1.1 into an apparent 3.
The relative per-layer profile is the honest comparison. Report the profile,
not one number.

Every rung fits at `dim_batch=8`, where both numbers are exactly zero.
`dim_batch=4` is a diagnostic config only, forced by the fact that the no-flag
arm does not fit in memory at 329 tokens. Thus R0 and R1 stand as fitted, and
no rung needs a refit.

Warning: we cannot test inertness at `dim_batch=8` x 329 tokens, because the
no-flag arm OOMs there. That is the fault which started this. We test what is
testable instead: whether the true fit config repeats.

### The answer-token rung, 2026-08-26

R2 is the rung that fits the Jacobian AT the token which emits the judgment
label. The `rating_2point` prompt carries the forced prefix `{"answer": "`, so
the last position emits the label and not `{`. All 8 shards ran on 2026-08-25
and then sat unmerged. We merged them and ran gate B.

| Item | Value |
|---|---|
| file | `outputs/_monocle/jlens/ladder/R2/gemma4_12b_lens.pt` |
| images | 128, in 8 shards |
| layers | 6, 12, 18, 24, 30, 36, 42, 46 |
| case | subway_safety |
| non-finite or all-zero layers | none |

**The diagonal test does not apply to a `last` target.** A span target maps a
position to itself, so R0 keeps a diagonal at L46 (3.2e-3). An answer-token
target transports a PATCH residual to the LABEL position, which is another
position, so no diagonal survives. R1 and R2 both sit at about 2e-4 at every
depth. Read that as correct, not as a fault.

### The gate, the input-ablation truth

We ran gate B twice. The first pass read 20 photographs and left 1 survivor
after a BH step, so we raised the count. The 60-photograph ablation map was
already on disk at the same seed, so the second pass cost 3 short jobs and no
new measurement. Every number below comes from the 60-photograph pass.

`lens_own` top-50 Pearson, mean over the 60 photographs. The local `gxa` map
is the control: it reads the same photographs under every rung, and its spread
across the 3 rungs is exactly 0.

| Layer | R0 span | R1 last, describe | R2 answer token | gxa |
|---|---|---|---|---|
| 6 | -0.036 | -0.019 | +0.016 | -0.003 |
| 12 | +0.007 | -0.026 | +0.046 | +0.024 |
| 18 | +0.032 | **+0.087** | **+0.066** | +0.056 |
| 24 | +0.048 | **+0.085** | **+0.116** | +0.085 |
| 30 | +0.036 | **+0.077** | -0.001 | +0.060 |
| 36 | +0.016 | **+0.077** | **+0.063** | +0.020 |
| 42 | +0.014 | +0.043 | +0.013 | +0.029 |
| 46 | +0.020 | +0.037 | +0.030 | +0.016 |

A one-sample test over the 60 photographs asks which arm is above 0. A BH step
over all 24 tests leaves 7 survivors, and the bold cells above are those 7:

| Rung | Survivors, with the q value |
|---|---|
| R0 span | **none** |
| R1 last | L18 (0.007), L24 (0.000), L30 (0.004), L36 (0.001) |
| R2 answer token | L18 (0.007), L24 (0.000), L36 (0.022) |

**The span lens reads nothing at any depth.** Its best cell, L24, gives an
uncorrected p of 0.023 and does not survive the step.

### P1 holds. P2 fails.

**The target span is the whole effect. The token identity is not.** Move the
cotangent from a 500-position span to ANY one late position, and the fitted
transport leaves 0 and stays there. Make that position the label token as
well, and nothing more happens.

R2 against R1, paired by photograph:

| Layer | 6 | 12 | 18 | 24 | 30 | 36 | 42 | 46 |
|---|---|---|---|---|---|---|---|---|
| gain | +0.035 | +0.072 | -0.021 | +0.031 | **-0.078** | -0.015 | -0.030 | -0.007 |
| p | 0.437 | 0.056 | 0.394 | 0.113 | **0.021** | 0.647 | 0.222 | 0.591 |

No layer favours the answer token. The 1 layer that separates the 2 rungs,
L30, favours the DESCRIBE prompt. R1 reads the position that emits the first
word of a description, and it reads the picture as well as the position that
emits the verdict, or better.

| Prediction | Verdict |
|---|---|
| P1 the target span carries the largest single effect | **holds** |
| P2 the prompt text carries a moderate effect | **fails**, the effect is 0 |

### The rung still does not pass the local instrument

R2 against `gxa`, paired by photograph. No layer favours the lens, and L30
favours the local map:

| Layer | 18 | 24 | 30 | 36 |
|---|---|---|---|---|
| lens | +0.066 | +0.116 | -0.001 | +0.063 |
| gxa | +0.056 | +0.085 | +0.060 | +0.020 |
| p | 0.769 | 0.206 | **0.028** | 0.132 |

The fitted transport now MATCHES the 1-backward-pass map at L18, L24 and L36.
It does not pass it, and the standing rule of `r-lens.md` does not move.

### What this closes

We asked whether a Jacobian fitted at the judgment token reads the picture
better than one fitted over a span. It does, and the gain comes from the
narrow target and not from the judgment. R3 moves the image axis alone, and P3
already predicts almost no effect there, so we did not fit it.

The tools and the runs:

| Item | Path |
|---|---|
| the lens | `outputs/_monocle/jlens/ladder/R2/gemma4_12b_lens.pt` |
| the 20-photograph gate | `outputs/_monocle/gateB_ladder{R0,R1,R2}/` |
| the 60-photograph gate | `outputs/_monocle/gateB60_ladder{R0,R1,R2}/` |
| the report | `research/scripts/ladder_gate_b_report.py` |

```bash
python -m research.monocle.jlens_merge_mm --in-dir outputs/_monocle/jlens/ladder/R2 \
    --out outputs/_monocle/jlens/ladder/R2/gemma4_12b_lens.pt --final-only

sbatch --time=02:00:00 research/monocle/monocle.sub monocle.input_gate \
    --case subway_safety --seed 20260821 --n-images 60 \
    --layers 6 12 18 24 30 36 42 46 --batch 8 --group 1 \
    --truth-from outputs/_monocle/causal_maps60/subway_safety_absent/maps.parquet \
    --lens outputs/_monocle/jlens/ladder/R2/gemma4_12b_lens.pt \
    --out-dir outputs/_monocle/gateB60_ladderR2

python research/scripts/ladder_gate_b_report.py --case subway_safety \
    --prefix gateB60_ladder
```

**Warning: `research/scripts/jlens_verify_lens.py` reports a diagonal mass near 0 for
R1 and R2, and that is not a fault.** The diagonal test assumes a span target,
where a position maps to itself. Read the layer count and the finite check
instead.

## What this cannot fix

A constant matrix cannot adapt to the input. The ladder buys a better
constant, not a variable one. Region importance still comes from
`research/monocle/causal_maps.py`. We keep the standing rule from the R-lens study:
read tokens with the lens, and read region importance with the causal maps.
Never swap one for the other.

## Related

- `vlm-narratives-docs/r-lens.md` — the rules study, and the stage-4 gate
- `vlm-narratives-docs/canonical-lens-prompt-path.md` — the mandatory prompt path
- `vlm-narratives-docs/patch-attribution.md` — the L20 scoreboard
