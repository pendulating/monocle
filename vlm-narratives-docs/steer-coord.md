# The steering coordinate: what the pseudo-inverse steering direction is, and is not

Date: 2026-09-01. Method: `monocle/steer_coord.py`. Report:
`scripts/steer_coord_report.py`. Output: `outputs/_monocle/steer_coord/`.

The single-image steering study (`single-image-steering.md`) showed that
``v_l = pinv(J_l) d`` moves the model's own judgment when we add it at
the last token position. This page asks what that vector is, in the
geometry of the model's own activations, on the same 60 evaluation images
and the same prompt.

## The 2 measurements

1. **The natural coordinate.** With no activation addition, one forward
   pass per evaluation image records ``h_l[answer]`` at layers 41, 42, 44
   and 45. The coordinate ``s_l = <h_l[answer], v_l>`` is the position of
   the image along the steering direction. We compare it with the model's
   own ``logit_diff``.
2. **The map.** For each image, the 256 single-token ablations of
   `causal_maps` run again under the steering prompt. Each forward pass
   records ``c`` AND ``h_l[answer]``. The change in ``c`` is the
   token-ablation map of the judgment. The change in
   ``h_l[answer]``, projected on ``v_l``, is the token-ablation map of
   the steering coordinate. The stored hidden states serve any later
   direction with no new GPU run.

The evaluation set is the sweep's evaluation set, checked at run time. The
batch is 12, as in `causal_maps60`. Each case ran in about 31 minutes on
one A6000.

## Result 1: the steering direction is orthogonal to the principal judgment direction

The centered last-token hidden state, across the 60 images, is close to
one-dimensional. Its first principal component holds 62 to 83 percent of
the variance and correlates 0.95 to 0.99 with the judgment. The
pseudo-inverse steering direction has a cosine of 0.001 to 0.019 with
that component, and the same with the difference of the 2 answer-class
means.

| Case | PC1 share | r(PC1, judgment) | cos(v, PC1) | sd along v | sd along PC1 |
|---|---|---|---|---|---|
| restaurants (L45) | 0.68 | 0.99 | 0.017 | 0.20 | 11.6 |
| road_quality (L41) | 0.63 | 0.97 | 0.019 | 0.22 | 11.5 |
| subway_safety (L45) | 0.83 | 0.99 | 0.009 | 0.08 | 9.2 |
| libraries (L41) | 0.75 | 0.99 | 0.015 | 0.20 | 13.1 |
| schools (L41) | 0.68 | 0.95 | 0.016 | 0.17 | 10.7 |
| parks_plazas (L41) | 0.72 | 0.98 | 0.018 | 0.26 | 14.2 |
| street_photography (L45) | 0.68 | 0.96 | 0.016 | 0.28 | 16.7 |

The images vary along ``v_l`` by 0.04 to 0.29 units, against 9 to 17
units along the principal judgment direction. The activation-addition
scale is 11 to 14 units, so alpha = 1 puts the last-token hidden state 50
to 300 times farther along ``v_l`` than any natural image sits.

``v_l`` is also not the unembedding contrast: its cosine with ``d`` is
0.05 to 0.09 at every layer.

## Result 2: the natural coordinate correlates, and the correlation comes from a small projection of the principal direction

``s_l`` correlates with the judgment at r = 0.5 to 0.98 at each case's
best layer. That number is not evidence. A random unit direction reaches
|r| = 0.85 to 0.98 at the 95th percentile, because a 1 to 2 percent
projection of the dominant direction into any direction carries the
direction's correlation. The correlation is scale-free, and that small
projection is enough.

The gain is the decisive test. The regression slope of the judgment on
``s_l`` is 20 to 98 logits per unit, but the steering gain from the sweep
at the same layer is 0.6 to 1.8 logits per unit. The natural spread along
``v_l``, multiplied by the causal gain, explains 0.6 to 6.6 percent of the
observed spread of the judgment. So the images do not reach their
judgments by movement along ``v_l``.

## Result 3: the coordinate map is a scaled-down version of the token-ablation map

At each case's best layer the per-image correlation between the
coordinate map and the token-ablation map is high, the sign is correct in
98 to 100 percent of the images, and the top 5 percent of image tokens
overlap 6 to 13 times chance.

| Case | Best layer | median r (signal) | median r (control) | random p95 | top-token overlap |
|---|---|---|---|---|---|
| restaurants | 45 | 0.89 | 0.03 | 0.74 | 0.62 |
| road_quality | 41 | 0.91 | -0.34 | 0.79 | 0.65 |
| subway_safety | 45 | 0.55 | 0.07 | 0.63 | 0.31 |
| libraries | 41 | 0.77 | 0.71 | 0.68 | 0.46 |
| schools | 41 | 0.77 | 0.30 | 0.64 | 0.42 |
| parks_plazas | 41 | 0.81 | 0.58 | 0.65 | 0.50 |
| street_photography | 45 | 0.85 | 0.25 | 0.84 | 0.46 |

Chance overlap is 0.05. The random-direction baseline is high here too,
for the same reason as in Result 2: a token ablation moves the last-token
hidden state mostly along the principal judgment direction, so any
direction with a small projection of that direction gives a correlated
map. The signal exceeds the baseline in 5 of 7 cases (not subway_safety,
not street_photography), and it has the correct sign where a random
direction has either.

The pooled slope of the token-ablation map on the coordinate map is 10 to
28 logits per unit, again 10 to 25 times the steering gain. The image
tokens move ``c`` far more than their movement along
``v_l`` could cause through the steering gain.

The layer matters. The coordinate map tracks the token-ablation map at
L41 and L42 for libraries, schools and parks_plazas, and falls to r = 0.0
to 0.3 by L45. Restaurants and subway_safety track at L45 and not at L41
to L44. road_quality tracks at every layer, and street_photography at L41
and L45.

## What this means

The pseudo-inverse steering direction is **a steering direction, not the
principal judgment direction**. The model's own judgment lies along one
direction of the last-token hidden state. ``v_l`` is orthogonal to that
direction, carries almost no natural variance, and still moves the
judgment at about 1 logit per unit when we add it. Its per-token map is a
signed, scaled-down version of the token-ablation map, because a small
projection of the principal judgment direction is present in it.

So a map of "which image tokens align with the steering direction" is,
to first order, the token-ablation map of the judgment scaled down by
that projection. It does not add a token-level fact that the
token-ablation map lacks.

The open question changes: is the gain along ``v_l`` a property of the
model, or of the activation addition? A direction along which the images
vary, the principal judgment direction, has a natural slope of 0.5 to 1
logit per unit of its own spread. An activation addition along that
direction at the same centered scale, with the same valid-answer
criterion and the same control, gives the comparison.

## Part 2: the principal judgment direction as a steering condition (2026-09-01, evening)

Methods: `scripts/steer_axis_cpu.py` (step 1, CPU) and the new
``--arms`` option of `monocle/single_steer.py` (step 2, GPU). Report:
`scripts/steer_axis_report.py`. Output: `outputs/_monocle/single_steer_axis/`.

### Step 1: the principal direction on the recorded hidden states

The principal direction is the unit regression direction of the judgment
on the last-token hidden state (`single_steer.judgment_axis`). We fitted
it on one half of the 60 evaluation images and evaluated it on the other,
both ways, at layers 41 to 45:

| Quantity | Range over 7 cases |
|---|---|
| cosine of the 2 fold directions | 0.93 to 0.995 |
| cosine of the principal direction with the pseudo-inverse direction | 0.001 to 0.018 |
| held-out r of the judgment on the principal-direction coordinate | 0.95 to 0.99 |
| held-out natural slope, logits per unit | 0.35 to 0.69 |
| pooled slope of the token-ablation map on the principal-direction map | 0.21 to 0.50 |
| median per-image r of the token-ablation map and the principal-direction map | 0.79 to 0.99 |

The natural slope and the token slope agree to within about 30 percent,
so the image tokens that move the judgment move it along the principal
direction, at the natural rate. That is the prediction for the activation
addition.

### Step 2: four conditions in one sweep

Per case: 16 calibration images fit the principal direction and the
pseudo-inverse direction, 60 evaluation images take coefficients from -2
to +2 in steps of 0.25 at layers 41, 42, 44 and 45. The conditions are
``axis``, ``signal`` (the pinv vector), ``control`` (the
pseudo-inverse random control of the sweep) and ``control_raw`` (a random
unit direction with no pull-back, at the same centered scale). About 38
minutes per case.

| Case | principal-direction gain | predicted | pseudo-inverse gain | principal-direction window | pseudo-inverse window | principal direction negative at -2 | principal direction positive at +2 |
|---|---|---|---|---|---|---|---|
| subway_safety | 0.38 | 0.44 | 0.98 | [-2, +2] | [-2, +2] | 0.00 | 1.00 |
| libraries | 0.48 | 0.54 | 0.97 | [-2, +2] | [-1.5, +2] | 0.23 | 1.00 |
| schools | 0.37 | 0.36 | 1.31 | [-2, +2] | [-1, +1.25] | 0.62 | 1.00 |
| road_quality | 0.29 | 0.41 | 0.64 | [-2, +2] | [-1.25, +1] | 1.00 | 0.98 |
| parks_plazas | 0.43 | 0.57 | 1.15 | [-2, +2] | [-1, +1.25] | 0.32 | 0.97 |
| restaurants | 0.31 | 0.35 | 1.43 | [-2, +2] | [-2, +0.75] | 0.87 | 1.00 |
| street_photography | 0.38 | 0.63 | 1.23 | [-2, +2] | [-0.75, +1.5] | 0.75 | 0.90 |

The gain is logits per unit hidden-state norm from the paired
coefficients of 0.5. The table shows the principal-direction gain at L41;
the other 3 layers agree to within 0.1. The window is the coefficient
range where 90 percent or more of the forward passes answer with one of
the answer words. The pseudo-inverse gain and window come from that
condition's widest layer.

Three things hold in every case.

1. **The principal direction steers within the data distribution.** It
   keeps a valid answer at every coefficient to plus or minus 2, which is
   twice the natural spread of the images along it. Its response to the
   coefficient is linear (the gains from coefficients of 0.25 and 0.5
   agree to 0.01). Its gain matches the step 1 prediction within 30
   percent, and the token slope within 10 percent.
2. **The pseudo-inverse direction has 2 to 4 times the gain and a
   narrower window.** It gives an invalid answer at coefficients of 0.75
   to 2, where the principal direction still answers. So the last-token
   hidden state has at least 2 effective steering directions: the model's
   own principal direction, at low gain over a wide window, and the
   pseudo-inverse direction from the Jacobian lens, at high gain over a
   narrow one. The two are orthogonal.
3. **A random direction at the same scale is not a steering vector.** The
   raw random control gives an invalid answer at the first coefficient of
   0.25 in 5 of 7 cases. The pseudo-inverse random control keeps a valid
   answer to about 1 and moves the judgment by -5 to +5 logits with no
   consistent sign.

The flip fractions at -2 follow the baseline. road_quality (36 of 60
positive) flips all 60 both ways. subway_safety (60 of 60, margin +15.8)
moves by -2.9 logits at -2 and flips none. The coefficient that flips a
ceiling case sits outside this grid.

### What the 2 steps settle

The principal judgment direction carries the judgment: the images vary
along it, the token ablations move it, and an activation addition along
it moves the judgment at the same rate. The pseudo-inverse direction is a
steering direction only: orthogonal to the principal direction, near zero
natural variance, high gain, narrow window. Both statements are now
causal.

The apparent contradiction of `single-image-steering.md` (3 to 7 percent
coverage, and steering works) has its answer here. The pseudo-inverse
direction does not need to contain the principal direction. It is a
separate direction that the late layers amplify at 2 to 4 times the
principal direction's rate. Why the mean Jacobian's minimum-norm pre-image
lands on such a direction is the open question.

## Part 3: generation, the gradient, one direction or seven, and the image-token maps (2026-09-01, night)

Methods: `monocle/steer_generate.py`, `monocle/grad_align.py`,
`scripts/steer_axes_cos.py`, `scripts/steer_patch_figures.py`, and the
`grad` condition of `monocle/single_steer.py`. Reports:
`scripts/steer_generate_report.py`, `scripts/grad_align_report.py`,
`scripts/steer_axis_report.py --root outputs/_monocle/single_steer_grad`.

### Generation: the flip persists, and the format persists under the principal direction

The model writes its answer under the sweep's own vectors, greedy, no KV
cache, the vector added at the last prompt position and every generated
position. Layer 41, 60 evaluation images per case. The ``prefix`` mode
starts from the forced ``{"answer": "``, as the sweeps did. The ``free``
mode starts from the bare generation prompt.

| Case | baseline pos | principal direction -2: neg | principal direction +2: pos | pseudo-inverse -0.75: neg | pseudo-inverse +0.75: pos |
|---|---|---|---|---|---|
| subway_safety | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 |
| libraries | 0.90 | 0.23 | 1.00 | 0.20 | 0.98 |
| schools | 0.93 | 0.62 | 1.00 | 0.83 | 1.00 |
| road_quality | 0.60 | 1.00 | 0.98 | 0.90 | 0.93 |
| parks_plazas | 0.87 | 0.32 | 0.97 | 0.32 | 0.98 |
| restaurants | 0.82 | 0.87 | 1.00 | 1.00 | 1.00 |
| street_photography | 0.75 | 0.75 | 0.90 | 0.35 | 0.80 |

Prefix mode, share of 60 images. The principal-direction columns agree
with the sweep's flip counts to within 1 image in every case. Under the
principal direction the closing JSON is intact in 100 percent of
generations. Under the pseudo-inverse direction the word flips and the
continuation after it breaks in restaurants, road_quality and
parks_plazas (``gross" 2``, ``maintained" 8``). The raw random control at
0.25 writes degenerate text in every case.

In free mode the model can abstain, and it does at baseline for
restaurants (70 percent, "I cannot determine, the image does not show any
food") and schools (98 percent). The principal direction does not remove
that abstention. Where the model answers at baseline, the principal
direction flips the written word and keeps the JSON: street_photography
0.88 negative at -2 and 1.00 positive at +2 with JSON intact in all 60,
road_quality 1.00 and 0.82. The pseudo-inverse direction in free mode
breaks the output in 5 of 7 cases at -0.75 (55 to 98 percent with no
answer word) and repeats the word at +0.75 (road_quality, 97 percent
degenerate).

**So the principal direction changes which answer word the model names,
not whether it answers, and it leaves the format intact.** The
pseudo-inverse direction changes the next token and corrupts the
continuation.

### The gradient: a third orthogonal direction, and not a steering vector

One backward pass per image gives ``g_l``, the gradient of the logit
difference at the last token position. Over the 60 evaluation images and
4 layers, in every case:

| Quantity | Range |
|---|---|
| cos(g, mean g), consistency across images | 0.83 to 0.95 (street_photography 0.49 to 0.58) |
| cos(g, principal direction) | 0.003 to 0.006 |
| cos(g, pseudo-inverse direction) | 0.008 to 0.019 |
| norm of g, logits per unit at first order | 45 to 174 |

The gradient as a steering condition (mean unit gradient from the
calibration images, the same grid as part 2) never gives a valid answer
at any nonzero coefficient, in any case, at any layer. The first grid
step is 3.4 units; the linear regime of the gradient lies below it. A
fine grid (coefficients 0.01 to 0.15, restaurants and road_quality)
locates it:

| Coefficient | Norm, units | restaurants L41: measured | first order | answer |
|---|---|---|---|---|
| -0.01 | 0.14 | -1.2 | -5.4 | valid |
| 0 | 0 | +4.9 | +4.9 | valid |
| +0.01 | 0.14 | +15.4 | +15.1 | valid |
| +0.02 | 0.27 | +15.1 | +25.4 | valid (0.95) |
| +0.05 | 0.68 | +4.0 | +56 | invalid |
| +0.15 | 2.04 | +0.1 | +159 | invalid |

The gradient matches its first-order prediction at 0.14 units of norm
(road_quality: +5.5 measured, +6.3 predicted), saturates by 0.27, and
gives an invalid answer at 0.68. Its whole working range is 1 to 2
percent of the principal-direction window. The gradient is the steepest
direction and the direction with the shortest range.

Thus there are three directions, each orthogonal to the other two, and
each moves the logit difference:

| Direction | Natural variance | Gain | Valid-answer window |
|---|---|---|---|
| principal judgment direction | all of it | 0.3 to 0.5 | the whole grid, to 2x the natural spread |
| pseudo-inverse direction | about 1 percent | 0.6 to 1.4 | 0.75 to 2 |
| gradient | about 0 | highest at first order | none on the grid |

### One direction or seven

The 7 principal directions are case-specific. The median off-diagonal
cosine is 0.12 to 0.16 at every layer. The one high pair is libraries and
parks_plazas at 0.85, which share the word pair maintained/worn;
subway_safety and parks_plazas sit at 0.42 to 0.48. The pseudo-inverse
directions of libraries and parks_plazas are the same vector (cosine
0.998), because ``d`` is the same; every other pseudo-inverse pair sits
at 0.1 to 0.35.

Four image statistics (brightness, contrast, edge energy, saturation,
whole-image means) explain 4 to 26 percent of the judgment, and 54
percent for street_photography, where the model judges dark and edge-rich
images as beautiful. The principal-direction coordinate keeps a partial
correlation of 0.92 to 0.99 with the judgment after we control for the 4
statistics. The principal direction is not an image statistic.

### The image-token maps, redrawn

`scripts/steer_patch_figures.py` draws the `causal_maps` figure for 4
maps of each image at layer 41, all in logits: the measured
token-ablation map, the principal-direction map (hidden-state change
along the principal direction times the principal-direction gain), the
pseudo-inverse map (the same with the pseudo-inverse direction), and the
gradient map (the same with the image's own gradient times its norm, the
first-order prediction).

| Case | principal-direction r | pseudo-inverse r | grad r | principal-direction scale | pseudo-inverse scale | grad scale |
|---|---|---|---|---|---|---|
| subway_safety | 0.86 | 0.27 | 0.95 | 0.73 | 0.04 | 0.88 |
| libraries | 0.97 | 0.77 | 0.99 | 0.55 | 0.03 | 0.91 |
| schools | 0.90 | 0.77 | 0.98 | 0.81 | 0.07 | 0.92 |
| road_quality | 0.99 | 0.91 | 1.00 | 0.76 | 0.04 | 0.89 |
| parks_plazas | 0.96 | 0.81 | 0.99 | 0.73 | 0.04 | 0.88 |
| restaurants | 0.94 | 0.29 | 0.99 | 0.83 | 0.04 | 0.89 |
| street_photography | 0.91 | 0.81 | 0.99 | 0.29 | 0.02 | 0.90 |

r is the median per-image correlation with the token-ablation map; scale
is the median magnitude over that of the token-ablation map. The gradient
map reproduces the measured map. The principal-direction map is the same
picture at lower amplitude. The pseudo-inverse map is near blank. Files:
`outputs/_monocle/steer_coord/figures/patch/`, one 60-image contact sheet
per map and case, and a 6-image comparison sheet per case.

## Part 4: the extended coefficient range, and the image-token principal direction (2026-09-01, late)

### Step 4: the ceiling cases at coefficients out to 8

The 3 cases where coefficient -2 flipped fewer than half of the images
ran again at coefficients -8 to +8, layers 41 to 45, the
principal-direction condition and the pseudo-inverse random control.
Layer 41, share of images that answer the positive word:

| Coefficient | -8 | -6 | -5 | -4 | -3 | -2 | 0 | +2 | +3 | +4 to +8 |
|---|---|---|---|---|---|---|---|---|---|---|
| subway_safety | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| libraries | 0.00 | 0.00 | 0.00 | 0.00 | 0.60 | 0.78 | 0.90 | 1.00 | 1.00 | 1.00 |
| parks_plazas | 0.00 | 0.00 | 0.00 | 0.00 | 0.52 | 0.72 | 0.87 | 0.97 | 1.00 | 1.00 |

The principal direction keeps a valid answer at every one of the 13
coefficients in all 3 cases and at all 4 layers. The pseudo-inverse
random control gives an invalid answer at coefficient 3 or 4 on both
sides. Libraries and parks are a coefficient limit: every image flips by
coefficient -4, and the response is linear on both sides (2.6 to 2.9
logits per coefficient unit).

Subway safety does not flip at any coefficient. Its mean logit difference
goes from +15.8 at baseline to +6.8 at -8, and every image still answers
"safe". Two facts explain it:

| Quantity | subway_safety | libraries | parks_plazas |
|---|---|---|---|
| baseline mean logit difference, logits | +15.8 | +11.0 | +9.5 |
| centered scale at layer 41, units of norm per coefficient | 4.0 | 8.6 | 11.4 |
| slope on the negative side, logits per coefficient | 1.1 | 2.6 | 2.0 |
| slope on the positive side, logits per coefficient | 1.5 | 2.8 | 2.9 |

The subway images vary little at the last token position (every one of
them is "safe"), so the centered scale is a third of the other cases',
and a coefficient of 8 is only 32 units of norm. Libraries flips at 34
units. In units of norm the subway response is about 0.28 logits per
unit, in line with the other cases; its negative side flattens (1.1
against 1.5), so a linear extrapolation of the negative branch reaches
zero near 60 units. The subway ceiling is a large baseline margin, not a
limit: the model sits 16 logits into "safe" and the principal direction
takes them back at the same rate as elsewhere, but the grid did not reach
far enough.

Files: `outputs/_monocle/single_steer_axis_ext/<case>/` and the figures
`outputs/_monocle/single_steer_axis_ext/figures/dose_response_*.png`.

## Part 5: the image tokens (2026-09-02)

Question: can a steering vector be mapped to single image tokens? Four
methods, all on the same 60 evaluation images per case:

| Method | Module | What it moves |
|---|---|---|
| uniform image-token activation addition | `single_steer --positions patches --scale centered_mean` | every image token at once, layers 8 to 30 |
| per-token activation addition | `patch_steer` | one image token at a time, layers 16 and 22 |
| ablation at each image-token budget | `patch_budget` | the image the model sees (70, 140, 280 tokens) |
| grouped ablation | `patch_group` | 2 x 2, 4 x 4 and 8 x 8 blocks at the production budget |

and one readout, the per-token gradient (`patch_grads`), scored by
`scripts/patch_grads_report.py`.

### The coefficient of a uniform activation addition

A vector added to all 256 image tokens moves the MEAN image-token vector
by the whole coefficient, and the last token uses that mean. The first
sweep measured the coefficient against one token's spread (21 to 55 units
at layers 12 to 30) and overshot: at coefficient 0.5 even the raw random
control flipped every image at layer 12. The ``centered_mean`` scale
measures the coefficient against the between-image spread of the mean
image-token vector, 3 to 5 units at layers 12 to 22, and the sweep ran
again from -2 to +2 in quarter steps.

### The image-token principal direction carries the judgment at the image-token positions

The principal judgment direction fitted on the image-token hidden states
(every image-token row carries its image's judgment) gives a graded,
monotone, linear response at every layer from 12 to 22, with a valid
answer at every coefficient. Layer 22, mean logit difference at
coefficients -2, 0 and +2, and the positive share at -2 and +2:

| Case | -2 | 0 | +2 | share positive at -2 / +2 |
|---|---|---|---|---|
| subway_safety | +11.3 | +15.8 | +18.3 | 0.95 / 1.00 |
| libraries | +3.3 | +11.0 | +16.7 | 0.72 / 1.00 |
| schools | +1.4 | +6.8 | +10.0 | 0.65 / 1.00 |
| road_quality | -4.0 | +0.1 | +4.1 | 0.15 / 0.82 |
| parks_plazas | -5.2 | +9.5 | +19.4 | 0.18 / 1.00 |
| restaurants | -0.4 | +4.9 | +9.1 | 0.53 / 0.95 |
| street_photography | -4.0 | +5.8 | +13.6 | 0.25 / 0.85 |

The three controls at the same coefficients are erratic: the raw random
direction, the pseudo-inverse direction and the pseudo-inverse random
control each jump by 5 to 15 logits between adjacent coefficients with no
order (restaurants, layer 12, raw random: +4.9 at 0, -6.9 at -0.25, +5.3
at +2). A coherent shift of one natural mean-spread in any direction but
the principal direction takes the mean image-token vector off the data
manifold, and the judgment becomes random, with a valid answer intact. By
layer 22 the random directions go flat (the model no longer uses the
image tokens) while the principal direction still carries the judgment.
At layer 30 every condition has no effect.

**The pseudo-inverse direction does not carry the judgment at the
image-token positions.** Its response is indistinguishable from its
random control at every layer.

### One image token at a time

The per-token activation addition of the image-token principal direction
(one image token, 2 per-token units, layers 16 and 22) gives a mean
response of 0.15 to 0.98 logits per token at layer 16 and 0.02 to 0.07 at
layer 22, near the bf16 precision floor. The maps at +2 and -2 do not
mirror each other (r -0.12 to +0.22 across cases and layers; a linear
response gives -1), the raw random control moves a token about as much as
the principal direction does, and the per-token principal-direction map
correlates with the token-ablation map at -0.02 to +0.10. A single image
token is too small a unit for a direct activation-addition map at this
coefficient. The first-order map from the image-token gradient replaces
it (below).

### The image-token budget

The 70 and 140 token budgets give the model a coarser image (96 and 70
pixels per image token against 48). The judgment persists: the logit
difference at 70 correlates with the 280 logit difference at 0.62 to 0.90
and the answer agrees on 77 to 100 percent of images (the misses are
interpretable: a restaurant with garbage bags on the sidewalk gives
"gross" at 280 and "delicious" at 70). The token-ablation maps do not
persist:

| Case | 70 vs 280 pooled 2x2, median r | 140 vs 280, median r | ablation offset 70 / 280 |
|---|---|---|---|
| subway_safety | 0.06 | 0.09 | -0.10 / +0.07 |
| libraries | 0.08 | 0.09 | -0.13 / -0.01 |
| schools | 0.01 | 0.00 | 0.00 / -0.02 |
| road_quality | 0.05 | 0.09 | -0.23 / -0.06 |
| parks_plazas | 0.09 | 0.08 | -0.05 / -0.06 |
| restaurants | 0.09 | 0.07 | -0.02 / +0.02 |
| street_photography | 0.13 | 0.06 | -0.30 / +0.10 |

The top-token overlap is at chance. Each budget has its own ablation
offset, the mean effect of one missing token. The 280 maps reproduce the
earlier `steer_coord` token-ablation maps bit for bit.

### The alternative: grouped ablation at 280

Ablation of 2 x 2 blocks from the production input keeps the production
judgment. The block map against the 4 single image tokens pooled:

| Case | 2x2 vs pooled tokens, r | sum ratio block / 4 tokens | 4x4 vs pooled, r | 2x2 vs the 70 map, r |
|---|---|---|---|---|
| subway_safety | 0.42 | 0.33 | 0.21 | 0.11 |
| libraries | 0.47 | 0.25 | 0.33 | 0.10 |
| schools | 0.37 | 0.30 | 0.29 | 0.05 |
| road_quality | 0.59 | 0.25 | 0.37 | 0.08 |
| parks_plazas | 0.15 | -0.08 | 0.35 | 0.05 |
| restaurants | -0.05 | 1.40 | 0.32 | -0.01 |
| street_photography | 0.40 | 0.36 | 0.35 | 0.19 |

Single image tokens add up to a block only in part (r about 0.4) and the
block ablation removes about a third of what its 4 tokens remove one by
one: the image tokens of an object are redundant, as the concern about
the token size predicted. The 2 x 2 map at 280 and the 70-token map of
the same grid agree at r 0.05 to 0.19, so the coarse budget is a
different image, not a coarser map of this one.

### The per-token gradient: alignment token by token

One backward pass per image gives the gradient of the logit difference
with respect to every image-token hidden state at layers 12, 16, 20 and
22. For a vector ``v``, ``<g[cell], v>`` is the first-order response of
the answer to ``v`` injected at that cell, and ``<h[cell] - mean, v>`` is
the image token's coordinate along ``v``. Layer 22, medians over the 60
images:

| Quantity | principal direction | pseudo-inverse direction | raw random |
|---|---|---|---|
| mean absolute cos(g[cell], v) over image tokens | 0.002 to 0.007 | 0.011 to 0.014 | 0.014 to 0.016 |
| cos(mean of g over image tokens, v) | +0.009 to +0.020 | -0.001 to +0.022 | -0.012 to +0.013 |
| r of the sensitivity map with the token-ablation map | -0.03 to +0.11 | -0.09 to +0.06 | +0.05 to +0.12 |
| r of the coordinate map with the token-ablation map | -0.03 to +0.10 | -0.03 to +0.06 | |
| share of the sensitivity mass in the top 5 percent of image tokens | 0.31 to 0.61 | 0.33 to 0.47 | 0.32 to 0.52 |
| r of the sensitivity map with the measured per-token activation addition | 0.39 to 0.63 | | |
| measured per-token response over first order | 0.20 to 0.35 | | |

A random unit direction in 3840 dimensions has a cosine of about 0.016
with any vector. The principal direction sits BELOW that floor token by
token (0.002 to 0.007) and carries the judgment only because its 256
small projections share one sign: the uniform activation addition sums
them, and the sum matches the first-order prediction (256 image tokens x
0.7 logits per unit x 0.004 x 8.6 units at coefficient 2 = about 6
logits, against 4 to 10 measured). The pseudo-inverse direction has no
such coherence: its mean-gradient cosine is no larger than a random
direction's. Neither vector's sensitivity map, nor its coordinate map,
resembles the token-ablation map. The gradient-times-input map does not
either (r 0.02 to 0.10): at the token level the ablation effect is not
first order, because the bidirectional image attention mixes the image
tokens before the last token uses them.

The sensitivity maps are sparse and image-specific: the image-token
position explains 4 to 13 percent of their variance, two images agree at
r 0.02 to 0.14, and the only recurring image tokens are the four corners
of the grid, which carry the largest image-token gradients in every case.
So the image tokens through which a vector can act are a property of the
image, but not of its judgment.

The per-token activation addition of the principal direction is real but
damped: the first-order map predicts it at r 0.4 to 0.6, and the measured
response is a fifth to a third of the prediction.

Figures: `outputs/_monocle/patch_grads/figures/cells_<case>_L{16,22}.png`
(photo, token-ablation map, gradient-times-input, principal-direction
sensitivity, pseudo-inverse sensitivity).

### The answer to the image-token question

Can a steering direction from the Jacobian lens be mapped to single image
tokens? No, at every level this work could measure:

| Level | Test | Result |
|---|---|---|
| last token position, readout | project the per-token change of the last-token hidden state on the pseudo-inverse direction (part 3) | near blank, 2 to 7 percent of the token-ablation map's scale |
| image-token positions, uniform activation addition | the pseudo-inverse direction at all 256 image tokens, natural coefficient | erratic, the same as its random control |
| image-token positions, token by token | first-order sensitivity and coordinate per image token | cosine at the random floor, r about 0 with the token-ablation map |

The principal judgment direction maps to the image tokens, but only as a
whole: it carries the judgment at the image-token positions (graded,
linear, with a valid answer, layers 12 to 22), and this effect is a
coherent sum over all 256 image tokens with no single image token above
the random floor. Which image tokens carry the judgment is a different
question, answered by ablation, and no vector's per-token alignment
measures it. The image tokens most aligned with a steering vector are not
the image tokens that decide the judgment.

## Part 6: the image mask, and a model with a vision encoder (2026-09-02, night)

The per-token failure of part 5 has a candidate cause. gemma-4-12B-it has
no vision encoder: an image token is a 48-px square, and the decoder lets
the 256 image tokens attend to each other from layer 0
(`use_bidirectional_attention: "vision"`). After layer 0, no image token
is its own square any more. So part 6 changes the mask over the image
tokens in the 12B, and measures the same set of experiments on
gemma-4-E4B, which has a 16-layer vision encoder in front of its decoder.

### The mask modes

`single_steer.install_image_mask(model, image_token_id, mode)` sets the
mode. A forward pre-hook records the image positions and pops the
multimodal type ids, so the trained block overlay never applies.

| mode | an image token attends to | how |
|---|---|---|
| bidirectional | every image token | the trained mask |
| causal | the earlier image tokens only | no type ids, block ids all -1 |
| self | itself and the text before it | an and-mask on `create_masks_for_generate` of the model's own module |

Warning: the 12B class lives in `modeling_gemma4_unified`, the E-series
in `modeling_gemma4`. Each module imports its own mask builder. The first
self-mask attempt patched `modeling_gemma4` and the 12B ran causal. The
install now patches `sys.modules[type(model).__module__]`, and
`monocle.mask_check` proves the mode before a set of experiments runs: it
replaces 1 image token's pixels and measures which other image tokens
change
at layer 12.

| mode | image tokens before the replaced token | image tokens after the replaced token |
|---|---|---|
| bidirectional | change | change |
| causal | bit-exact | change |
| self | bit-exact | bit-exact |

### The causal mask: the model still answers

Under the causal mask the 12B keeps a valid answer on every forward pass
of every case. The judgment shrinks and drifts, and the image ranking
persists:

| case | mean c, trained | mean c, causal | agree | r logit difference | token-ablation map r (median) |
|---|---|---|---|---|---|
| subway_safety | 15.8 | 10.3 | 1.00 | 0.60 | 0.01 |
| libraries | 11.1 | 5.3 | 0.90 | 0.45 | 0.03 |
| schools | 6.8 | 4.4 | 0.88 | 0.36 | 0.02 |
| road_quality | 0.1 | -2.5 | 0.57 | 0.55 | 0.04 |
| parks_plazas | 9.4 | 5.4 | 0.92 | 0.73 | 0.03 |
| restaurants | 4.9 | 0.5 | 0.63 | 0.55 | 0.02 |
| street_photography | 5.7 | -3.8 | 0.47 | 0.69 | 0.01 |

The token-ablation maps do not persist: token for token, the causal map
has no relation to the trained map (r about 0.02 in every case). The
per-token sd of the causal map is about 60 percent of the trained one.
Street photography flips its sign at the population level.

The steering results under the causal mask, with the principal direction
refit on the hidden states under the causal mask:

| measurement | trained | causal |
|---|---|---|
| last token position, layers 41 to 45, gain per unit | 0.29 to 0.40 (restaurants, road) | 0.27 to 0.39 |
| the same, street_photography | 0.36 to 0.46 | 0.21 to 0.23; +2 reaches 28 percent positive |
| image-token positions, layer 12 to 22, slope per unit | 1.5 to 4.4 | 1.2 to 2.8 |
| per-token cosine of the gradient with the principal direction | 0.000 to 0.001 | 0.000 to 0.001 |
| sensitivity map vs the causal token-ablation map | 0.03 to 0.14 (part 5) | -0.08 to 0.14 |

So the causal mask is not the cause of the per-token failure. The
principal judgment direction still carries the judgment at the last token
and at the image tokens, and the per-token gradient still has no relation
to the principal direction or to the token-ablation map. A causal mask is
also not a locality test: a later image token still attends to every
earlier one. The self mask is the locality test, and its results follow
below.

### The self mask: no image token attends to another

Under the self mask an image token attends to itself and the text before
it, in every layer. The model still emits an answer word on every forward
pass. The judgment changes more than under the causal mask:

| case | mean c, trained | mean c, self | positive share, self | agree | r logit difference | token-ablation map r (median) | per-token sd, self / trained |
|---|---|---|---|---|---|---|---|
| subway_safety | 15.8 | 9.0 | 1.00 | 1.00 | 0.23 | -0.01 | 0.36 |
| libraries | 11.1 | -3.0 | 0.03 | 0.10 | -0.17 | 0.00 | 0.31 |
| schools | 6.8 | 3.9 | 1.00 | 0.92 | 0.21 | -0.02 | 0.43 |
| road_quality | 0.1 | -2.6 | 0.08 | 0.42 | 0.05 | 0.01 | 0.38 |
| parks_plazas | 9.4 | -1.0 | 0.22 | 0.32 | 0.28 | 0.00 | 0.28 |
| restaurants | 4.9 | 4.1 | 1.00 | 0.82 | 0.30 | 0.00 | 0.37 |
| street_photography | 5.7 | 1.6 | 0.82 | 0.87 | 0.71 | -0.02 | 0.21 |

Libraries and parks flip their population sign. The image ranking is
mostly gone (r 0.05 to 0.30, street photography 0.71). The token-ablation
maps have no relation to the trained maps, and the per-token effects
shrink to about a third of the trained spread. So the trained judgment is
a product of the image tokens' attention to each other: with that
attention gone, the model uses a different, weaker image.

The steering results, with the principal direction refit on the
hidden states under the self mask:

| measurement | trained | self |
|---|---|---|
| last token position, gain per unit (layers 41 to 45) | 0.29 to 0.46 | 0.28 to 0.44 |
| image-token positions, slope per unit (layers 12 to 22), monotone | 1.5 to 4.4 | 0.9 to 2.4 |
| raw random control at the image tokens | not monotone | not monotone |
| per-token cosine of the gradient with the principal direction (median) | 0.002 to 0.007 | 0.000 to 0.005 |
| sensitivity map vs the self token-ablation map, r (median) | 0.03 to 0.14 | -0.03 to 0.22 |
| coordinate map vs the self token-ablation map, r (median) | about 0 | 0.05 to 0.18 |
| gradient x input vs the self token-ablation map, r (median) | 0.02 to 0.10 | 0.07 to 0.21 |
| sensitivity top-5 mass | 0.29 to 0.46 | 0.14 to 0.17 |

The refit principal direction carries the judgment at the last token and
at the image tokens under the self mask as before. The per-token gradient
still has no alignment with it, and the first-order maps still do not
recover the token-ablation map (street photography is the best case at r
about 0.2). The sensitivity maps become flat (top-5 mass halves), so the
corners lose their weight.

This closes the bidirectional-attention explanation. The image tokens'
attention to each other is what the trained judgment is made of, and it
is what makes the token-ablation maps what they are. But it is not what
hides the per-token alignment: with no image attention at all, no single
image token's gradient aligns with the principal judgment direction, and
no first-order map recovers the token-ablation map.

### gemma-4-E4B: a vision encoder in front of the decoder

gemma-4-E4B (`Gemma4ForConditionalGeneration`, 42 layers, hidden 2560)
puts a 16-layer vision encoder in front of the decoder and gives the
decoder 256 image tokens with a causal mask. It has no fitted Jacobian
lens, so the set of experiments runs with `--no-lens`: the principal
judgment direction and the raw random control only. `extract.load_model`
loads it through a new branch that reads the architecture from
`config.json` and the template from the directory's `chat_template.jinja`.

| measurement | E4B | 12B (parts 2 to 5) |
|---|---|---|
| last token position, gain per unit (layers 36 to 40) | 0.31 to 0.63, control -0.04 to 0.04 | 0.29 to 0.46 |
| image-token positions, slope per unit (layers 10 to 14) | 0.7 to 3.1, monotone in 6 of 7 cases | 1.5 to 4.4 |
| image-token positions, layers 23 and 26 | exactly 0, both conditions | layer 30 has no effect |
| per-token cosine of the gradient with the principal direction (median) | -0.006 to 0.010 | 0.002 to 0.007 |
| random floor for that cosine | 0.020 | 0.016 |
| sensitivity top-5 mass | 0.31 to 0.49 | sparse |

So the pattern repeats with an encoder in front. The principal judgment
direction carries the judgment at the last token and at the image-token
positions, and no single image token's gradient aligns with it above the
random floor. Restaurants is the weak case at the image tokens (slope 1.4
at layer 10, and the random control moves as much).

Warning: E4B shares the KV cache of its last 18 layers
(`num_kv_shared_layers: 18`). A hidden-state edit at an image position in
layer 23 or later changes no key or value that the last token uses. That
is why layers 23 and 26 give exactly zero. Measure the E4B image tokens
at layers 10 to 19 only.

E4B has no token-ablation map: `patch_ablate` edits the Unified model's
slot table, and the E-series has none. So the E4B per-token test is the
gradient cosine only.

### How the 256 image tokens compose the sum

`scripts/cell_composition_report.py` measures the per-token distribution
of 2 quantities per image, medians over the 60 images and 7 cases
(`outputs/_monocle/cell_composition/`, figure `figures/rank_share.png`):

| quantity | top-5 share | Gini | participation ratio | cancellation | same-sign share | kurtosis |
|---|---|---|---|---|---|---|
| ablation contribution | 0.07 to 0.09 | 0.42 to 0.45 | 0.54 to 0.63 | 1.5 to 2.5 | 0.66 to 0.74 | 0.8 to 4.6 |
| <g, principal direction> per image token | 0.13 to 0.41 | 0.60 to 0.80 | 0.08 to 0.35 | 1.2 to 3.2 | 0.57 to 0.85 | 7 to 64 |
| <g, random> per image token | 0.15 to 0.30 | 0.59 to 0.71 | 0.13 to 0.34 | 2.0 to 9.3 | about 0.5 | 12 to 51 |
| Gaussian, 256 image tokens | 0.06 | 0.41 | 0.64 | 19 | 0.52 | 0 |

The token-ablation map is not heavy-tailed: its magnitude profile is that
of Gaussian noise, and what separates it from noise is the sign (two
thirds of the image tokens push with the net). The gradient projection is
heavy-tailed, but a random direction gives the same tail, so the tail is
the distribution of gradient norms across image tokens and not the
judgment. The two methods disagree in their largest entries, which is why
the largest gradient entries do not predict the largest ablation entries.

### The answer-word projection against the ground-truth variables

The last-token projection on the answer-word directions, at scale and
against the CVPR unit-level ground-truth variables, is in
`pole-projection-proxy.md`.

## Files (part 6)

| File | Content |
|---|---|
| `outputs/_monocle/causal12b/` | the 12B under the causal image mask: `patch_budget`, `patch_grads` (7 cases), `patch_axis_mean`, `single_steer_axis` (3 cases), `causal_mask_report.parquet` |
| `outputs/_monocle/self12b/` | the same under the self mask; `mask_check.json` is the proof of the mode; `self_mask_report.parquet` |
| `outputs/_monocle/e4b/` | gemma-4-E4B: `patch_grads`, `patch_axis_mean` (7 cases), `single_steer_axis` (3 cases) |
| `monocle/mask_check.py` | the substitution check of a mask mode, 1 image |
| `scripts/causal_mask_report.py` | a masked run against the trained run (`--root`, `--label`) |

## Files (part 5)

| File | Content |
|---|---|
| `outputs/_monocle/patch_axis/<case>/` | the uniform image-token sweep at the per-token scale (overshoots; keep for the vectors) |
| `outputs/_monocle/patch_axis_mean/<case>/` | the same at the mean scale; `figures/dose_response_*.png` |
| `outputs/_monocle/patch_steer/<case>/` | per-token activation addition of the image-token principal direction, layers 16 and 22 |
| `outputs/_monocle/patch_budget/<case>/` | token-ablation maps at 70, 140 and 280 tokens; `figures/budget_<case>.png` |
| `outputs/_monocle/patch_group/<case>/` | 2x2, 4x4, 8x8 block ablation at 280; `figures/group_<case>.png` |
| `outputs/_monocle/patch_grads/<case>/` | per-token gradients and hidden states, layers 12 to 22; `figures/cells_*.png` |
| `scripts/patch_*_report.py`, `scripts/patch_grads_figures.py` | the reports and figures |

## Pitfalls

- **Do not report r alone.** At layers 41 to 45 the last-token hidden
  state is one-dimensional across images, so a random direction
  correlates with the judgment at |r| up to 0.98. Report the gain match,
  the cosine with the principal judgment direction, and the sign.
- **The pseudo-inverse control is not a baseline for correlation.** It
  gives r = 0.6 to 0.99 in several cases and layers, for the same projection
  reason.
  It is a baseline for the gain (0.0 to 0.5) and for the map sign.
- **Match the batch.** The baseline and every image token sit in a batch
  of 12. The run asserts that the baseline from the control condition and
  from the map agree.
- **The stored hidden states are fp16.** A coordinate along a unit
  direction is accurate to about 0.01 units. The natural spread along
  ``v_l`` is 0.04 to 0.29, so the measurement is coarse but not blind.

## Files (part 3)

| File | Content |
|---|---|
| `outputs/_monocle/steer_generate/<case>/generations.parquet` | every generation, both modes |
| `outputs/_monocle/grad_align/<case>/` | `grad_dirs.npz`, `eval_grads.npy`, `cosines.parquet` |
| `outputs/_monocle/single_steer_grad/<case>/` | the grad and principal-direction conditions on the part 2 grid |
| `outputs/_monocle/single_steer_grad_fine/<case>/` | the grad condition at coefficients 0.01 to 0.15 |
| `outputs/_monocle/single_steer_axis/axis_image_stats.parquet` | the image statistics test |
| `outputs/_monocle/single_steer_axis/figures/dose_response_*.png` | the 4-condition response figures |
| `outputs/_monocle/steer_coord/figures/patch/` | the redrawn image-token maps |

## Files (part 2)

| File | Content |
|---|---|
| `outputs/_monocle/steer_coord/axis_cpu_report.parquet` | step 1 |
| `outputs/_monocle/single_steer_axis/<case>/steer_results.parquet` | the 4-condition sweep, 16320 rows |
| `outputs/_monocle/single_steer_axis/<case>/vectors.npz` | the 4 directions per layer |
| `outputs/_monocle/single_steer_axis/axis_report.parquet` | per case, layer, condition |
| `outputs/_monocle/single_steer_axis/axis_L41_summary.parquet` | the table above, with the controls |

## Files

| File | Content |
|---|---|
| `<case>/vectors.npz` | ``d``, the control, ``v_signal_L<l>``, ``v_control_L<l>``, scales, pseudo-inverse errors |
| `<case>/natural.parquet`, `natural_acts.npy` | the clean measurement: judgment and last-token hidden state per image |
| `<case>/maps.parquet`, `map_stats.parquet` | the token-ablation map under the steering prompt template, with the controls |
| `<case>/answer_acts.npy` | ``[60, 257, 4, 3840]`` fp16; row 0 is the baseline |
| `natural_report.parquet`, `geometry_report.parquet`, `map_report.parquet` | the tables above |
| `figures/natural_<case>.png` | the natural coordinate against the judgment, per layer |
| `figures/maps_<case>_L<l>.png` | photograph, token-ablation map, coordinate map for 6 images |

## How to repeat

```bash
sbatch --time=00:30:00 monocle/monocle.sub monocle.steer_coord --smoke
sbatch --time=01:30:00 monocle/monocle.sub monocle.steer_coord \
    --case restaurants --n-images 60 --batch 12
python scripts/steer_coord_report.py --cases all --figures
python scripts/steer_axis_cpu.py --cases all
sbatch --time=01:30:00 monocle/monocle.sub monocle.single_steer \
    --case restaurants --n-images 60 --pullback pinv --layers 41,42,44,45 \
    --alphas=-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2 \
    --arms signal,control,axis,control_raw \
    --out-dir outputs/_monocle/single_steer_axis
python scripts/steer_axis_report.py --cases all
```

## Related

- `single-image-steering.md` — the sweep whose vectors this page measures
- `patch-attribution.md` — the token-ablation map method and its layer rules
- `where-the-judgment-lives.md` — the information flow that puts the
  judgment at the last token position by layer 31

### Part 6

```bash
# prove the mask mode, then the battery behind it (afterok)
sbatch monocle/monocle.sub monocle.mask_check --case restaurants --out outputs/_monocle/self12b/mask_check.json
sbatch monocle/monocle.sub monocle.patch_budget --case <case> --budgets 280 --image-mask self --out-dir outputs/_monocle/self12b/patch_budget
sbatch monocle/monocle.sub monocle.patch_grads --case <case> --image-mask self --out-dir outputs/_monocle/self12b/patch_grads
sbatch monocle/monocle.sub monocle.single_steer --case <case> --positions patches --scale centered_mean \
  --arms axis,control_raw --layers 8,12,16,20,22,26,30 --alphas=-2,-1,-0.5,-0.25,0,0.25,0.5,1,2 \
  --image-mask self --out-dir outputs/_monocle/self12b/patch_axis_mean
# --image-mask causal repeats the causal battery under outputs/_monocle/causal12b
# E4B: add --model-dir /share/ju/matt/zoo/models/Gemma-4-E4B-it --no-lens
.venv-3.12/bin/python scripts/causal_mask_report.py --root outputs/_monocle/self12b --label self --cases all
.venv-3.12/bin/python scripts/patch_grads_report.py --cases all --grads-root outputs/_monocle/self12b/patch_grads \
  --vecs-root outputs/_monocle/self12b/patch_axis_mean --removal-root outputs/_monocle/self12b/patch_budget --removal-kind delta
```

### Part 5

```bash
# uniform patch sweep at the mean scale (GPU, one job per case)
sbatch --time=01:30:00 monocle/monocle.sub monocle.single_steer --case restaurants \
    --n-images 60 --pullback pinv --positions patches --scale centered_mean \
    --layers 8,12,16,20,22,26,30 --alphas=-2,-1,-0.5,-0.25,0,0.25,0.5,1,2 \
    --arms axis,control_raw,signal,control --out-dir outputs/_monocle/patch_axis_mean
# per-cell injection, budget maps, grouped removal, patch gradients
sbatch --time=04:30:00 monocle/monocle.sub monocle.patch_steer --case restaurants --layers 16,22
sbatch --time=02:30:00 monocle/monocle.sub monocle.patch_budget --case restaurants --budgets 70,140,280
sbatch --time=01:30:00 monocle/monocle.sub monocle.patch_group --case restaurants --groups 2,4,8
sbatch --time=01:00:00 monocle/monocle.sub monocle.patch_grads --case restaurants
# reports and figures (CPU)
.venv-3.12/bin/python scripts/steer_axis_figure.py --root outputs/_monocle/patch_axis_mean
.venv-3.12/bin/python scripts/patch_steer_report.py --cases all
.venv-3.12/bin/python scripts/patch_budget_report.py --cases all
.venv-3.12/bin/python scripts/patch_group_report.py --cases all
.venv-3.12/bin/python scripts/patch_grads_report.py --cases all
.venv-3.12/bin/python scripts/patch_grads_figures.py --cases all --layer 22
```

### Part 3

```bash
# generation, the gradient, and the gradient arm (GPU, one job per case)
sbatch --time=01:30:00 monocle/monocle.sub monocle.steer_generate --case restaurants
sbatch --time=00:40:00 monocle/monocle.sub monocle.grad_align --case restaurants
sbatch --time=02:00:00 monocle/monocle.sub monocle.single_steer --case restaurants \
    --arms axis,grad --grad-vectors outputs/_monocle/grad_align/restaurants/grad_dirs.npz \
    --out-dir outputs/_monocle/single_steer_grad
# the fine grid
sbatch --time=00:30:00 monocle/monocle.sub monocle.single_steer --case restaurants \
    --arms grad --grad-vectors outputs/_monocle/grad_align/restaurants/grad_dirs.npz \
    --alphas=-0.15,-0.1,-0.05,-0.02,-0.01,0,0.01,0.02,0.05,0.1,0.15 \
    --out-dir outputs/_monocle/single_steer_grad_fine
# reports and figures (CPU)
.venv-3.12/bin/python scripts/steer_generate_report.py --cases all
.venv-3.12/bin/python scripts/grad_align_report.py
.venv-3.12/bin/python scripts/steer_axis_report.py --root outputs/_monocle/single_steer_grad --cases all
.venv-3.12/bin/python scripts/steer_axes_cos.py
.venv-3.12/bin/python scripts/steer_axis_figure.py
.venv-3.12/bin/python scripts/steer_patch_figures.py --cases all
```
