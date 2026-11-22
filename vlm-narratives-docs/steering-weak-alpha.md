# Activation addition at small coefficient alpha: the line of work closes

Date: 2026-09-01. Review item P8 (see
[jacobian-lens-review.md](jacobian-lens-review.md)).

The 2026-08-23 activation-addition runs used the raw residual-stream norm as
the coefficient scale, and 16 of 18 settings collapsed. The record stated
that a weaker grid was never run. This study runs that grid, with 2 pre-image
forms of the steering vector. **Conclusion: activation addition gives a null
result in the small-coefficient regime too, for both pre-image forms. The
line of work is closed.**

## Design

| Item | Value |
|---|---|
| case / kind | subway_safety / proxy, canonical path |
| lens | `outputs/_monocle/jlens/mm/gemma4_12b_lens.pt` |
| layers | 24, 36, 42 |
| coefficients alpha | -16 to +16 (11 values), in centered-norm units |
| scale | `--scale centered`: median \|\|h - mean(h)\|\| over calibration pairs |
| pre-image forms | `--pullback transpose` (J^T d) and `--pullback pinv` (pinv(J) d) |
| conditions | signal + seed-777 random control, same pipeline |
| pairs | 16 calibration + 100 eval, disjoint |
| jobs | smoke test 47937; conditions 79919 (transpose), 80020 (pinv) |
| output | `outputs/_monocle/steer_weak/{transpose,pinv}/` |

The centered scale is far below the old grid. The raw norm at L24 is 256 and
the centered norm is 1.69, a ratio of about 150. Here, alpha +16 adds a
vector of norm about 27 at L24 where the old grid started near 512.

| Layer | raw norm | centered norm |
|---|---|---|
| 24 | 256.0 | 1.69 |
| 36 | 126.7 | 12.71 |
| 42 | 100.1 | 13.41 |

## The fitted translator cannot represent the label direction

Both pre-image forms fail to reconstruct the label direction `d` through the
fitted linear translator of the Jacobian lens. Relative error
\|\|J v - d\|\| / \|\|d\|\| at the best scalar:

| Layer | transpose | pinv |
|---|---|---|
| 24 | 0.99999 | 0.9981 |
| 36 | 0.99999 | 0.9994 |
| 42 | 0.99996 | 0.9992 |

The pseudo-inverse (pinv) is the best possible linear pre-image, and it
recovers about 4 to 6 percent of the direction. The label direction lies
almost fully outside the row space of the averaged Jacobian. This is a
structural reason that activation addition along a pre-image direction
cannot work here. It agrees with the near-zero diagonal entries of the
middle-layer translators in
[jlens-47-layer-fit.md](jlens-47-layer-fit.md).

## Collapse still dominates, and it is not specific to the direction

A setting is collapsed when one steered answer class holds more than 70
percent of the 100 pairs. Of 30 steered settings per condition:

| Condition | collapsed settings |
|---|---|
| transpose signal | 26 / 30 |
| transpose control | 25 / 30 |
| pinv signal | 22 / 30 |
| pinv control | 25 / 30 |

The control condition collapses as much as the signal condition. Thus the
sensitivity is a property of the last token position, not of the direction.
Collapse reaches 75 percent of settings already at \|alpha\| = 1 (an added
vector of norm 1.7 to 13). By layer: L36 and L42 collapse at 80 to 100
percent under every tested alpha; only L24 retains a small-coefficient
regime. The collapse target has no pattern: 38 settings go to Much, 30 to
More, 25 to Less, 5 to Same, with no relation to the sign of alpha.

## The 4 non-degenerate settings show no effect

Only 4 of 60 (layer, alpha) settings are non-degenerate in both conditions.
Signal minus control, paired over 100 pairs:

| Setting | difference (exact) | difference (ordinal) | Wilcoxon p |
|---|---|---|---|
| pinv L24 a+1 | +0.01 | +0.005 | — |
| pinv L24 a+2 | -0.01 | -0.003 | 1.00 |
| pinv L24 a+4 | -0.05 | -0.025 | 0.018 (the control condition is higher) |
| transpose L24 a+4 | +0.16 | +0.083 | 0.0003 |

The one setting that favors the signal condition does not show a directed
shift. Its label distribution is {More 0.55, Less 0.44, Much 0.01} against
a baseline of {Much 0.67, More 0.25, Less 0.08}. A shift toward More that
also raises Less from 0.08 to 0.44 does not follow the sign. It splits the
dominant Much answer into a near-even More/Less division. The gain in exact
agreement is the class-frequency artifact that the 2026-08-23 record
already names. The 3 pinv settings are the only truly small perturbations
(distributions near baseline), and they show no effect.

## No monotonic response to the coefficient in any setting

A steering vector must raise p(More) - p(Less) with alpha in the signal
condition and not in the control condition. Per-pair slopes over
\|alpha\| <= 2:

| Setting | signal slope | control slope |
|---|---|---|
| transpose L24 | -0.017 | -0.033 |
| pinv L24 | -0.005 | +0.015 |

The signal slope is negative (the wrong sign) at the one layer with a
non-degenerate regime, and the control condition moves as much. The L36 and
L42 slopes describe the transition into collapse, not a directed shift.

## Conclusion

| Question | Answer |
|---|---|
| Does a small-coefficient regime exist without collapse? | Only at L24, alpha <= 4, and only barely |
| Does the signal condition beat the control condition there? | No. 2 of 4 settings, opposite signs, no response in the direction of the sign |
| Does the pseudo-inverse pre-image help? | No. It is the better pre-image and still recovers about 5 percent of d |
| Does this line of work stay open? | **No. It is closed.** |

The 2026-08-23 caveat is now measured and removed: there is no weaker grid
to try. The mean translator does not represent the label direction
(reconstruction error 0.998+), so no pre-image of it can steer the model. The
last token position collapses under any perturbation of even a tenth of its
residual-stream norm.

## Notes

- The registered subway_safety run predates the 2026-08-20 layout decision
  (`canonical.prompt_cfg` warns). Both conditions use the same recorded
  prompt, so the signal-minus-control comparison is internally consistent.
- `research/monocle/jlens_steer.py` now takes `--scale {raw,centered}` (default
  centered; raw reproduces the 2026-08-23 grid) and `--pullback
  {transpose,pinv}`, and records both scales and both reconstruction errors
  in the summary JSON.
