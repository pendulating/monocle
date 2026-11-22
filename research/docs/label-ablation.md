# Label-subspace ablation: the causal test of the L42 claim

Date: 2026-09-01. Review item P1 (see
[jacobian-lens-review.md](jacobian-lens-review.md), flaw F1). Tool:
`research/monocle/label_ablate.py`. Jobs: 80050 (smoke), 80080 (full run).

## The question

The workspace claim says the judgment enters the lens channel at L42 under
the production prompt. That claim rests on a readout. This experiment tests
it by removal: delete the label-direction subspace from the residual at one
block, at the answer position, and read the change in the model's own
judgment. If the judgment uses the channel from L42 up, removal at or above
L42 must move it.

## The verdict, first

**Removal of the label subspace at L42 does not move the judgment in any
meaningful size. The claim must soften to: the judgment is decodable
through the lens at L42.**

The paired excess over the matched control at L42 is +0.043 logits
[+0.018, +0.068] on a judgment whose typical scale is about 5 logits, with
a flip rate of 2.0% against 0.7%. The excess is statistically real
(q=0.0016) and about 1% of the judgment. The depth profile also does not
match an entry at L42: the larger excesses sit BELOW it, at L36 (+0.68) and
L24 (+0.37).

## The design

- Case subway_safety, kind proxy, production prompt, teacher-forced
  `{"answer": "` prefix. 16 calibration pairs, 150 eval pairs, disjoint,
  seed 777.
- Directions: for each first-token class (Much*, Less, Same, More,
  NotSure), the gradient of that class logit with respect to the final
  residual, by autograd through the unembed path. Mean over the
  calibration pairs.
- Pull-back per layer: `v_i = J_l^T d_i` with the mm lens
  (`outputs/_monocle/jlens/mm/gemma4_12b_lens.pt`), orientation invariant
  checked (worst residual 2e-7). QR gives an orthonormal rank-5 basis.
- Ablation: one forward hook per forward removes the basis span from the
  block output at the answer position only.
- Arms per (pair, layer): `none` (baseline), `label`, `ctrl_matched` (the
  same pipeline on 5 random vocabulary rows — it matches the construction
  and the massive-channel concentration), `ctrl_random` (a random rank-5
  subspace of the residual space).
- Readout: the model's own final logits at the answer position. The
  primary statistic is the paired difference of `|dc|`
  (`c = logit(More) - logit(Less)`) between the label arm and the matched
  control, Wilcoxon + a 20000-resample bootstrap, BH over the 5 layers.

## The numbers (150 pairs)

Per arm:

| Layer | Arm | Flip rate | mean d\|c\| | mean dp(prod) | Removed norm frac |
|---|---|---|---|---|---|
| 24 | label | 0.033 | 0.488 | +0.015 | 0.002 |
| 24 | ctrl_matched | 0.007 | 0.123 | +0.008 | 0.001 |
| 30 | label | 0.073 | 2.141 | -0.343 | 0.001 |
| 30 | ctrl_matched | 0.060 | 4.888 | -0.775 | 0.001 |
| 36 | label | 0.047 | 0.976 | +0.007 | 0.000 |
| 36 | ctrl_matched | 0.013 | 0.293 | +0.008 | 0.000 |
| 42 | label | 0.020 | 0.182 | +0.007 | 0.000 |
| 42 | ctrl_matched | 0.007 | 0.139 | +0.002 | 0.000 |
| 46 | label | 0.180 | 0.813 | -0.035 | 0.006 |
| 46 | ctrl_matched | 0.040 | 0.916 | -0.015 | 0.004 |

Label against the matched control, paired `|dc|`, BH over layers (all 5
survive):

| Layer | Excess | 95% interval | q |
|---|---|---|---|
| 24 | +0.365 | [+0.314, +0.417] | 0.0000 |
| 30 | **-2.746** | [-3.349, -2.142] | 0.0000 |
| 36 | +0.684 | [+0.529, +0.850] | 0.0000 |
| **42** | **+0.043** | [+0.018, +0.068] | 0.0016 |
| 46 | -0.103 | [-0.245, +0.053] | 0.0056 |

## How to read this

1. **L42 carries almost nothing that removal can reach.** The label arm
   removes less than 0.1% of the residual norm there, and the judgment
   moves by 1% of its scale more than the matched control. The readout at
   L42 is decodable; it is not load-bearing at that site.
2. **The depth profile contradicts an entry at L42.** The excess is 8
   times larger at L36 and 16 times larger at L24 — both below the claimed
   entry depth. Whatever the label pull-backs touch, they touch it before
   L42, not from L42 up.
3. **L30 is fragile to ANY unembed pull-back.** The matched control moves
   the judgment by 4.9 logits there — more than the label arm (2.1). A
   rank-5 removal along unembed pull-backs at L30 damages the computation
   without regard for the label content. Read no label conclusion from
   L30.
4. **The random-subspace control shows why the matched control matters.**
   It removes 20 to 50 times more norm (it hits the massive-activation
   channels) and flips 77 to 87% of answers at L30-L42. Against that
   control every arm would look causal.
5. **L46 flips without a larger `|dc|`.** The label arm flips 18% of
   answers at L46 (matched: 4%) while it moves `c` LESS than the control.
   At L46 the transport is near the identity, so the removal erases
   label-class evidence directly under the head. The flips there are
   class reshuffles among the non-More/Less pairs, not a judgment
   reversal. Report it; do not build on it.

## Limits

- The removal acts at one position and one block. The relay shows the
  question tokens still hold 0.19 to 0.30 of the judgment through L33-L44,
  so downstream attention can restore what a single-site removal deletes.
  A null here bounds the single-site story only.
- The subspace has rank 5. A redundantly coded judgment (the saturation
  result) can survive a rank-5 deletion.
- The registered subway run predates the 2026-08-20 layout decision. Every
  arm reads the same recorded prompt, so the comparison is internally
  consistent, but a battery re-run will change the forward this page
  reads.

## Files

| Item | Path |
|---|---|
| tool | `research/monocle/label_ablate.py` |
| CPU tests | `tests/test_label_ablate.py` (13 pass) |
| long frame | `outputs/_monocle/label_ablate/ablate.parquet` |
| summary | `outputs/_monocle/label_ablate/summary.json` |
| directions cache | `outputs/_monocle/label_ablate/directions.pt` |
