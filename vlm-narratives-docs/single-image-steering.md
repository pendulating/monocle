# Activation addition through the Jacobian lens in the single-image prompt template

Date: 2026-09-01. Script: `research/monocle/single_steer.py`. This study restates
the causal activation-addition question, at the user's direction: single-image
evaluations, with one single-token antonym pair per case. The pairwise prompt
template stays for the census-variable validation results only.

## Reason for the single-image prompt template

The logit lens decodes one next-token distribution. A multi-token label shows
only its first token, so the pairwise labels (`MuchLess` ... `MuchMore`) reduce
to their first tokens. A single-token antonym pair removes that limit and puts
the judgment on a semantic dimension (`safe` vs `dangerous`), not on a
comparative scale token.

## The word table

Each word is exactly one gemma-4-12B-it token (verified; the criterion is
`polar_token_ids`):

| Case | Positive | Negative |
|---|---|---|
| subway_safety | safe | dangerous |
| libraries | maintained | worn |
| schools | good | bad |
| road_quality | smooth | damaged |
| parks_plazas | maintained | worn |
| restaurants | delicious | gross |
| street_photography | beautiful | ugly |

"appealing", "dislike", "neglected", "inviting", "rundown", and "shabby"
split into 2 or more tokens, so they are not usable.

## The row-space coverage criterion, first

Before any sweep, `--coverage` answers the structural question: how much of the
case's logit-difference gradient ``d = grad(logit_pos - logit_neg)`` lies in the row
space of the fitted linear translator ``J_l`` of the lens? The pseudo-inverse pre-image reconstructs the
row-space part of ``d`` and nothing else, so
``rowspace_frac = sqrt(1 - err_pinv^2)`` is the upper bound on the share of the direction
that any steering vector built through the lens can contain.

Result (job 88231, mm_all47 lens, 16 calibration images per case):

| Case | L24 | L36 | L41 | L42 | L45 | L46 |
|---|---|---|---|---|---|---|
| subway_safety | .052 | .029 | .034 | .034 | .036 | .103 |
| libraries | .066 | .034 | .036 | .035 | .037 | .124 |
| schools | .053 | .027 | .032 | .032 | .032 | .104 |
| road_quality | .066 | .039 | .040 | .041 | .043 | .123 |
| parks_plazas | .066 | .034 | .036 | .035 | .037 | .124 |
| restaurants | .058 | .036 | .040 | .040 | .042 | .116 |
| street_photography | .062 | .036 | .039 | .039 | .041 | .110 |

**The translator contains 3 to 7 percent of every case's logit-difference
gradient at every layer through L45, and about 10 to 12 percent at L46.** The
transpose form is worse (err about 1.0000 everywhere). The pairwise null result
(`steering-weak-alpha.md`) measured the same 5 percent for the label
direction. So the obstacle is a property of the averaged Jacobian, not of
the comparative labels: the mean Jacobian is near-singular in every answer
direction we tested, single-token or not.

Full numbers: `outputs/_monocle/single_steer/coverage.json`.

## The behavioral sweep

Design, per case: 16 calibration images give the direction and the centered
scale of the activation addition; 60 disjoint evaluation images take the grid
``layers {24,36,41,42,45} x alphas {0, ±2, ±4, ±8, ±16} x
{signal, matched random control}``, with the pseudo-inverse pre-image and the addition at the last token
position. The metric is the model's own judgment of the image:
``logit_pos - logit_neg`` and the restricted argmax. The same run records a
logit-lens readout of the single-image judgment at each of the 47 layers, at alpha = 0.

### subway_safety (job 88383, 60 evaluation images)

The baseline shows a ceiling effect: the model calls all 60 entrances `safe`
(mean logit difference +15.8). Three regimes appear in the grid:

1. **|alpha| >= 4 is invalid.** The output leaves the set of valid answer
   tokens for both signal and control: the top token is a malformed token
   (`<image|>`, `!!!!`, stray multilingual pieces) and the probability mass
   on the answer words drops to ~0. The logit difference still moves, but
   inside a degenerate distribution. This is the failure mode of the
   raw-norm scale, at a smaller coefficient.
2. **alpha = -2 is a valid activation-addition effect.** The model still
   answers the question. 6 of 60 images flip to `dangerous` at L42 and 8 of
   60 at L45 (top-token counts 54/6 and 52/8). The matched control at the
   same coefficient flips zero images (60/60 `safe`). The mean logit
   difference drops from +15.8 to +2.4 (L42).
3. **L24 has no effect** at any coefficient, in either condition.

So the single-image prompt template gives what the pairwise prompt template
never gave: a coefficient window where a lens-derived steering vector moves
the model's own judgment and a matched control does not. The effect is real
but small, and the window is narrow. The fine coefficient grid below maps it.

**The per-layer readout confirms the premise of the single-image prompt
template.** The Jacobian-lens readout of the answer word under the
single-image prompt reaches p(`safe`) = 0.354 at L45 (0.329 at L46, 0.089 at
L42, ~0 below L40). The pairwise prompt template peaked at 0.069. The readout
of the single-image judgment through the same lens is about 5 times larger.

### restaurants (job 88713, 60 evaluation images)

The baseline shows no ceiling effect (49 `delicious` / 11 `gross`, mean
logit difference +4.9), so this case tests both directions. The
activation-addition effect is strong, bidirectional, and specific:

| Sweep condition (L42) | What the model emits (60 images) |
|---|---|
| signal, alpha -2 or -4 | ` gross` x 60 (the word, space-prefixed) |
| signal, alpha +4 (L45) | `delicious` x 60 |
| control, alpha -2 | ` neither` / malformed tokens; never the opposite answer word |
| control, alpha +-4 | malformed tokens (`<audio\|>`, `1`) |

A note on the valid-answer metric: at L42 the steered model emits the answer
word with a leading space (a different token id). That is the correct
answer with a formatting artifact, so `frac_top_pos` / `frac_top_neg`
count both surface forms (see `polar_variant_ids`). Under that count the
signal stays inside the set of valid answer tokens up to |alpha| = 4 while it
flips the judgment, and the random control leaves that set instead of a flip.

Per-layer readout: p(`delicious`) reaches 0.07-0.09 over L42-46, and the
per-layer restricted argmax at L45-46 agrees with the model's own answer for
93 percent of images. This is informative here because the baseline
splits 49/11.

### Fine coefficient grid, subway_safety (job 89040, alphas in 0.5 steps)

The valid-answer window is alpha in [-2, +3] (signal): every forward pass
still answers the question. Inside the window the response to the
coefficient is smooth and monotonic. L45, mean logit difference (signal | control):

| alpha | -2 | -1.5 | -1 | -0.5 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|
| signal | +1.5 | +6.1 | +10.1 | +12.6 | +15.8 | +25.0 | +32.2 | +31.6 |
| control | +13.2 | +14.6 | +15.8 | +15.6 | +15.8 | +14.7 | +7.1 | +2.7 |

At alpha = -2 the signal moves the judgment by -14.3 logits against the
control's -2.6, and 13 percent of images say `dangerous` (L45; 10
percent at L42) with the control at zero. The output leaves the valid answer
set at alpha <= -2.5 (signal) and at alpha >= +2.5 (control, L45). The
ceiling effect of the baseline bounds the flip fraction: every image starts
at `safe` with a +15.8-logit margin, so only the least-safe images can
cross within the window.

### Fine coefficient grid, restaurants (job 89500, layers 41/42/44/45)

This case shows complete bidirectional semantic control at coefficients
inside the natural variation of the judgment signal across image pairs:

- **alpha = -1 at L42: all 60 images answer `gross`** (space-prefixed,
  inside the valid answer set), from a 49/11 `delicious` baseline. The
  control at the same coefficient flips nothing.
- **alpha = +3 to +4 at L45: 59-60 of 60 answer `delicious`**, again inside
  the valid answer set; the control stays at its baseline logit difference
  (~+5) or leaves the set.
- Paired signal-vs-control effects (10000-resample bootstrap over the 60
  images): L42 alpha -2: **-35.9 logits [-36.2, -35.7]**; L45 alpha +2:
  **+27.8 [+27.5, +28.1]**. The judgment scale itself spans about 5-16
  logits, so the effect saturates the scale.
- The valid-answer window depends on the layer and the sign: L42 stays
  inside the valid answer set over [-4, +1], L45 over [-1, +4] with a brief
  break at -2/-1.5. Outside its window, the logit difference of a sweep
  condition is not an activation-addition result.

### The remaining 5 cases (2026-09-01)

Jobs 94313, 94320, 94321, 94524 and 94525 ran the 5 other cases on the
same fine grid. Each run used layers 41/42/44/45, alphas from -4 to +4,
60 evaluation images, 16 calibration images, the pseudo-inverse pre-image,
and the addition at the last token position. Each run wrote 6240 sweep rows
and 2820 per-layer readout rows.

The window below is the set of coefficients around alpha = 0 where the
signal holds the fraction of valid answers (`frac_on_contract`) at 0.90 or
more, with no break. Only a sweep condition inside the window counts as evidence
of activation addition. The table names the widest window of the 4 layers
and the two coefficients at its edges.

| Case | Baseline pos/neg | Window (layer) | Low coefficient | Signal neg | Control neg | High coefficient | Signal pos | Control pos |
|---|---|---|---|---|---|---|---|---|
| libraries | 54/6 | L42 [-1.5, +2] | -1.5 | 0.900 | 0.000* | +2 | 0.933 | 0.000* |
| schools | 56/4 | L41 [-1, +1] | -1 | 1.000 | 0.117 | +1 | 1.000 | 0.917 |
| road_quality | 36/24 | L41 [-1, +1] | -1 | 1.000 | 0.033 | +1 | 1.000 | 0.550 |
| parks_plazas | 52/8 | L42 [-1, +1] | -1 | 1.000 | 0.050 | +1 | 0.983 | 0.000* |
| street_photography | 45/15 | L42 [-0.5, +1.5] | -0.5 | 0.333 | 0.250 | +1.5 | 0.967 | 0.000* |

A star marks a control condition that leaves the valid answer set. The control
does not flip the judgment there. It emits a token that is not an answer
word.

| Case | Effect at the low coefficient | Effect at the high coefficient | Readout peak | agree_own L44 / L45 / L46 |
|---|---|---|---|---|
| libraries | -22.2 [-22.5, -22.0] | +28.5 [+28.3, +28.7] | L45, 0.234 | 1.000 / 0.967 / 0.933 |
| schools | -11.0 [-11.1, -10.8] | +15.2 [+14.7, +15.7] | L45, 0.110 | 0.783 / 1.000 / 0.917 |
| road_quality | -16.3 [-16.6, -16.0] | +13.4 [+13.2, +13.5] | L46, 0.081 | 0.900 / 0.983 / 0.883 |
| parks_plazas | -20.1 [-20.3, -20.0] | +16.4 [+16.0, +16.8] | L45, 0.200 | 1.000 / 1.000 / 0.967 |
| street_photography | -7.6 [-8.7, -6.6] | +20.0 [+19.8, +20.3] | L44, 0.250 | 0.850 / 0.950 / 1.000 |

The effect is the paired signal-minus-control change of `logit_diff`
over the 60 images. The interval is a 10000-resample bootstrap at 95
percent. The agree_own column is the agreement of the per-layer restricted
argmax with the model's own answer.

**road_quality is the cleanest test of the 7.** Its baseline splits
36/24, so neither answer word shows a ceiling effect. At L41 the coefficient
alpha = -1 turns all 60 roadways to `damaged`, and alpha = +1 turns all 60
to `smooth`. The control at both coefficients stays near its baseline
split. At alpha = +0.5 the control moves 7 images to `damaged`, which is
the wrong direction for a positive coefficient, so that drift is noise.

**street_photography steers fully in both directions; the earlier weak
negative-direction result was an artifact of the coefficient grid.** The
follow-up run (job 96366, `outputs/_monocle/single_steer_fine2/`,
coefficients 0 to -1.5 in 0.1-0.25 steps) finds the negative-direction
window between the steps of the old grid. At L45 and alpha = -0.9, 54 of 60
images answer `ugly` (baseline 15 of 60) with 93 percent of forward passes
inside the valid answer set; at L42 and alpha = -0.8, 44 of 60 answer `ugly`
with 100 percent inside the valid answer set. The control holds its 45/15
baseline split through the whole window. Paired effects: -15.7
[-16.4, -15.1] at L45 alpha -0.9, and -14.9 [-16.2, -13.6] at L42 alpha
-0.8. The response to the coefficient is smooth from 25 percent `ugly` at
alpha -0.25 to 90 percent at -0.9, and the output leaves the valid answer
set only at alpha <= -1. So all 7 cases steer in both available directions.

**The schools control moves a small amount.** At L42 and alpha = -1 the
control moves the negative fraction from 4 of 60 to 13 of 60. The signal
at the same condition moves it to 60 of 60. So the control adds a small
non-specific shift, and the signal still flips every image.

**libraries and parks_plazas repeat the near-ceiling pattern.** Both
start above 52 of 60 on the positive answer word. Both flip at least 54 of
60 images at alpha = -1 or -1.5, and their controls hold the baseline split
at every coefficient inside the valid-answer window.

## The answer

**Yes: causal activation addition through the Jacobian lens works in the
single-image prompt template.** The pseudo-inverse pre-image of the
logit-difference gradient moves the model's own judgment monotonically,
bidirectionally (restaurants), and specifically (a matched random control
never flips one judgment at any coefficient; it only degrades the output
format). The pairwise prompt template never showed this. All 7 cases steer:
each case has a coefficient window where the signal flips judgments and the
matched control stays at or near its baseline split.

The structural paradox is the scientific point: the translator contains
only 3-7 percent of the logit-difference gradient (`err_pinv` >= 0.999), and
the activation addition works anyway. So the row space of the mean Jacobian
understates the causal reach of a layer: the true forward pass amplifies the
small transmitted component about 30-fold. The pairwise null result
(`steering-weak-alpha.md`) was a property of the comparative-label prompt
template, not of the lens or of the model. This result refutes its
structural argument ("5 percent cannot carry the signal") as a sufficient
argument for a null result. The row-space coverage is an upper bound on
reconstruction, not on causal effect.

## What the steering vector is, and is not (2026-09-01, later the same day)

`steer-coord.md` measured the steering vector against the model's own
activations. The pseudo-inverse steering direction is orthogonal to the
principal judgment direction, the direction of largest between-image
variance of the logit difference (cosine about 0.01), and it carries almost
no natural variance. So the claim "moves the model's own judgment" is
supported, and the claim "the direction the model uses to represent its
judgment" is not. The principal judgment direction also steers the
judgment, over a wider coefficient window and with a lower gain. Read that page before you cite
this one.

## Caveats

- All 7 cases ran on the fine grid. That grid holds layers 41, 42, 44
  and 45 only, so the 5 later cases have no L24 or L36 readout.
- The subway_safety baseline shows a ceiling effect (60/60 `safe`), so its
  flip fractions are bounded; restaurants is the informative case.
- The evaluation images come from the image set of the registered pairwise
  runs; the single-image prompt itself is new (no registered single-image
  run exists as a reference for it).
- The activation addition is at the last token position only. The
  image-token-position variant (`--positions patches`) did not run.

## How to repeat

```bash
sbatch --time=00:40:00 research/monocle/monocle.sub monocle.single_steer --smoke
sbatch --time=03:00:00 research/monocle/monocle.sub monocle.single_steer --coverage \
    --layers 24,36,40,41,42,43,44,45,46
sbatch --time=06:00:00 research/monocle/monocle.sub monocle.single_steer \
    --case subway_safety --n-images 60 --pullback pinv
```

## Related

- `jacobian-lens-review.md` — the review that led here (see "The
  single-image reframe")
- `steering-weak-alpha.md` — the pairwise null result of the activation
  addition and its structural cause
- `workspace-depth-47.md` — the pairwise per-layer profile
