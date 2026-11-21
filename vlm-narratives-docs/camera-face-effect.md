# The camera face effect

**Question:** a photograph in the battery is 1 face of a Cyclomedia cube. Does
the model's judgment change with the face?

**Answer:** yes, and the same face wins in 11 of the 13 cells that give an
estimate. The cause is the sun.

## The face is a compass direction

The cube is fixed to the compass, not to the car. `pairs.parquet` carries the
absolute bearing in `bearing_a` and `bearing_b`, and those columns hold exactly
0, 90, 180 and 270.

| Face | The camera looks toward | The sun sits |
|------|-------------------------|--------------|
| `F` | north (0 deg) | behind the camera, all day |
| `R` | east (90 deg) | in front in the morning, behind in the afternoon |
| `B` | south (180 deg) | in front, near noon |
| `L` | west (270 deg) | behind in the morning, in front in the afternoon |

`dagspaces/urbanroamvqa/graph/street_graph.py` holds the same table as
`FACE_BEARING_DEG`. The facing filter picks the face that points at the unit,
thus 1 recording gives 1 face to a facility pool.

The face is a **camera decision**. A library does not become a better library
when the car photographs it from the south. Every face coefficient must be 0.

## The design

The battery draws the image of each side at random from the pool of the place,
and the pool spans several faces. The face is thus assigned at random inside a
place, and a fixed effect on the place removes every property of the place.

| `fe` | What it holds fixed | Which cases carry it |
|------|---------------------|----------------------|
| `dyad` | the SAME 2 places, compared again | libraries |
| `place` | the place of each side | the other 4 facility cases |
| `place` | the recording, thus the position AND the minute | the 2 image cases |

### Only the libraries case repeats a place pair

| Case | Places | Place pairs | Median repeats |
|------|--------|-------------|----------------|
| libraries | 236 | 27,730 | 39 |
| subway_safety | 1,990 | 1,000,000 | 1 |
| parks_plazas | 2,064 | 1,000,000 | 1 |
| schools | 2,287 | 1,000,000 | 1 |
| restaurants | 18,488 | 997,074 | 1 |

236 libraries give 27,730 possible pairs, thus 1,000,000 draws fall on each
pair about 39 times, with a new pair of faces each time. 99.7% of the libraries
rows sit in a place pair that came up again with a NEW face pair. The other
cases hold too many places, thus a draw is nearly always a new pair.

### The 2 designs agree

On the libraries case the `dyad` design and the `place` design give the same 4
coefficients to 4 decimal places. They differ only in the standard error.

## The model

`y` is the ordinal label of side A against side B, from -2 to +2. An abstention
gives NaN and drops out, the same as in `_provenance.score_units`.

    y = theta_a - theta_b + beta[face_a] - beta[face_b] + e

The 4 `beta` values are identified up to a constant, thus `_faces` reports them
with a mean of 0.

**Warning: fix the side order before a `dyad` group is demeaned.** A run shows
the same place pair many times, and side A holds the 2 places about equally
often. `y` changes sign with that order, thus the group mean is not the
strength of the dyad. `_faces.orient` does the step. Without it the libraries
swing reads 0.053 rather than 0.083.

## The standard error

A comparison touches 2 places, thus 2 comparisons that share a place carry
correlated errors.

| `cluster` | Formula | The question it answers |
|-----------|---------|-------------------------|
| `dyad` | group on the place pair | Does the face move the judgment for THESE places? |
| `dyadic` | place sum minus dyad sum | Would it move it for a NEW place? |
| `robust` | HC0 | only when the draw is random and no FE runs |

`dyadic` is the dyadic-robust variance of Aronow, Samii and Assenova (2015). A
comparison enters the sum of BOTH of its places, thus a dyad is counted twice
and the dyad sum removes the double count.

**The 2 answers differ by a factor of 7.** On libraries / qwen the swing gives
t = 17 under `dyad` and t = 2.3 under `dyadic`, because 236 libraries carry the
whole interval. Report the conservative one, and read the grid for the rest.

## The result

`F` wins in 11 of the 13 cells that give an estimate, and `B` loses in 10. A
named face wins 11 of 13 by chance with p = 4.4e-5.

| Case | Rater | Design | F | R | B | L | Swing |
|------|-------|--------|---|---|---|---|-------|
| subway_safety | gemma-4-12b | place | +0.008 | +0.003 | -0.007 | -0.004 | 0.016 |
| subway_safety | qwen3.5-9b | place | +0.007 | -0.004 | -0.010 | +0.006 | 0.017* |
| libraries | gemma-4-12b | same 2 places | +0.021 | -0.019 | -0.014 | +0.011 | 0.040*** |
| libraries | qwen3.5-9b | same 2 places | +0.034 | -0.001 | -0.049 | +0.016 | 0.083*** |
| schools | gemma-4-12b | place | +0.008 | +0.004 | -0.011 | -0.001 | 0.019* |
| schools | qwen3.5-9b | — | — | — | — | — | — |
| road_quality | gemma-4-12b | place | +0.016 | -0.006 | +0.004 | -0.015 | 0.031*** |
| road_quality | qwen3.5-9b | place | +0.037 | -0.010 | -0.016 | -0.011 | 0.052*** |
| parks_plazas | gemma-4-12b | place | +0.015 | +0.011 | -0.020 | -0.006 | 0.035* |
| parks_plazas | qwen3.5-9b | place | +0.028 | -0.001 | -0.055 | +0.028 | 0.082*** |
| restaurants | gemma-4-12b | place | +0.017 | +0.002 | -0.016 | -0.003 | 0.033*** |
| restaurants | qwen3.5-9b | place | +0.006 | -0.003 | -0.019 | +0.016 | 0.035*** |
| street_photography | gemma-4-12b | place | +0.014 | -0.006 | -0.011 | +0.003 | 0.025*** |
| street_photography | qwen3.5-9b | place | +0.036 | -0.019 | -0.008 | -0.009 | 0.055*** |

`*` p < 0.05, `**` p < 0.01, `***` p < 0.001, under the conservative cluster of
each design.

**The schools / qwen3.5-9b cell gives no estimate.** That rater abstained on
99.7% of the schools comparisons, thus 3,384 usable rows remain out of
1,100,000.

## The mechanism: the sun test

A face effect alone does not name a cause. The cube is fixed to the compass,
thus a face is also a **light condition**, and the light turns during the day
while the street does not. That gives a prediction a reader can falsify.

| Window, local time | The sun sits | Backlit face | Front-lit face |
|--------------------|--------------|--------------|----------------|
| before 11 | east | `R` | `L` |
| 11 to 14 | south | `B` | `F` |
| after 14 | west | `L` | `R` |

- `R - L` must be negative in the morning, near 0 at midday, and positive in
  the afternoon.
- `F - B` must stay positive all day. A north-looking camera never faces the
  sun.
- If the model reads the CITY and not the light, then nothing moves with the
  hour. A street does not change direction at lunchtime.

### The prediction holds, in all 4 image cells

| Case | Rater | Window | n | F | R | B | L | R - L | F - B |
|------|-------|--------|---|---|---|---|---|-------|-------|
| road_quality | gemma-4-12b | morning | 56,802 | +0.020 | -0.036 | +0.021 | -0.004 | **-0.032** | -0.001 |
| road_quality | gemma-4-12b | midday | 248,881 | +0.013 | -0.003 | -0.012 | +0.002 | **-0.005** | +0.025 |
| road_quality | gemma-4-12b | afternoon | 95,412 | +0.027 | +0.015 | +0.005 | -0.046 | **+0.061** | +0.022 |
| road_quality | qwen3.5-9b | morning | 56,930 | +0.023 | -0.065 | +0.001 | +0.042 | **-0.107** | +0.022 |
| road_quality | qwen3.5-9b | midday | 249,396 | +0.036 | -0.009 | -0.035 | +0.008 | **-0.017** | +0.071 |
| road_quality | qwen3.5-9b | afternoon | 95,589 | +0.045 | +0.022 | -0.007 | -0.061 | **+0.083** | +0.052 |
| street_photography | gemma-4-12b | morning | 57,024 | +0.009 | -0.033 | -0.003 | +0.027 | **-0.060** | +0.012 |
| street_photography | gemma-4-12b | midday | 249,776 | +0.022 | -0.008 | -0.017 | +0.002 | **-0.010** | +0.039 |
| street_photography | gemma-4-12b | afternoon | 95,725 | +0.010 | +0.010 | -0.002 | -0.019 | **+0.029** | +0.012 |
| street_photography | qwen3.5-9b | morning | 56,937 | +0.019 | -0.060 | +0.003 | +0.037 | **-0.097** | +0.016 |
| street_photography | qwen3.5-9b | midday | 249,397 | +0.044 | -0.017 | -0.028 | +0.001 | **-0.017** | +0.072 |
| street_photography | qwen3.5-9b | afternoon | 95,618 | +0.036 | +0.007 | -0.003 | -0.040 | **+0.047** | +0.040 |

`R - L` crosses 0 between the morning and the afternoon in all 4 cells, and
`F - B` stays positive in 11 of the 12 windows, and the 12th reads -0.001. Only the 2 image cases carry
`recordedAt`, thus only they can run the test.

## What it does to a published place score

A single judgment is not the paper's output. A place score is, and it is a mean
over many comparisons. The battery draws the face at random, thus most of the
face effect cancels inside that mean.

It does not cancel all the way. A place does not see the 4 faces evenly: a
library on the north side of an east-west street is only visible from a camera
that looks north, thus its pool is nearly all `F`. The **face mix** of a place
is the channel by which the camera reaches its score.

| Quantity | libraries / qwen3.5-9b |
|----------|------------------------|
| Swing, in ordinal units | 0.083 |
| Swing, against the spread of the place scores | 0.24 |
| Percentiles a median place moves under the swing | 9.7 |
| Share of the between-place score variance | 0.4% |

Read the last 2 rows together. A single comparison moves by a quarter of the
spread of the places, but a place score averages about 7,500 comparisons whose
faces are drawn at random. The random draw already does most of the work.

**The number would be far larger for a study that gave each place 1
photograph.** That is the reading the paper needs: the battery is safe because
it samples the face, not because the model ignores it.

## The face also moves the abstention

The same design, run on the `NotSure` rate, gives a second reading. A negative
value says the model abstains LESS when a side carries that face. The design is
symmetric here, because the model abstains on the comparison and not on 1 side
of it.

| Case | Rater | F | R | B | L | t |
|------|-------|---|---|---|---|---|
| libraries | gemma-4-12b | -0.0055 | +0.0036 | -0.0037 | +0.0057 | 8.9 |
| libraries | qwen3.5-9b | -0.0101 | +0.0043 | +0.0055 | +0.0003 | 10.5 |
| schools | gemma-4-12b | -0.0056 | +0.0017 | -0.0066 | +0.0105 | 3.4 |
| parks_plazas | gemma-4-12b | -0.0145 | +0.0120 | -0.0065 | +0.0091 | 4.3 |
| parks_plazas | qwen3.5-9b | -0.0111 | +0.0083 | -0.0074 | +0.0102 | 3.1 |
| restaurants | qwen3.5-9b | -0.0051 | +0.0032 | +0.0071 | -0.0052 | 4.9 |

The table holds the 6 cells whose swing reaches t = 3. `F` carries a negative
value in all 6. A north-looking photograph thus makes the model both more
positive AND more decisive. The 2 image cases give a value near 0, because they
almost never abstain.

## The rank shift, case by case

The correction subtracts the face component from each place score. The table
reports how far a place then moves in the ranking, in percentile points.

| Case | Rater | Places | Median | 90th | Max | Over 5 pts | Spearman |
|------|-------|--------|--------|------|-----|-----------|----------|
| subway_safety | gemma-4-12b | 1,990 | 0.25 | 0.95 | 2.26 | 0.0% | 0.9998 |
| subway_safety | qwen3.5-9b | 1,990 | 0.40 | 1.16 | 2.41 | 0.0% | 0.9997 |
| libraries | gemma-4-12b | 236 | 0.42 | 1.69 | 3.81 | 0.0% | 0.9994 |
| libraries | qwen3.5-9b | 236 | 1.27 | 3.81 | 7.20 | 3.4% | 0.9971 |
| schools | gemma-4-12b | 2,287 | 0.26 | 0.92 | 1.97 | 0.0% | 0.9998 |
| road_quality | gemma-4-12b | 220,841 | 0.72 | 2.51 | 3.40 | 0.0% | 0.9989 |
| road_quality | qwen3.5-9b | 221,439 | 0.66 | 2.00 | 4.14 | 0.0% | 0.9991 |
| parks_plazas | gemma-4-12b | 2,064 | 0.34 | 0.91 | 1.55 | 0.0% | 0.9998 |
| parks_plazas | qwen3.5-9b | 2,064 | 0.92 | 2.47 | 5.33 | 0.1% | 0.9986 |
| restaurants | gemma-4-12b | 18,488 | 0.36 | 1.05 | 1.45 | 0.0% | 0.9998 |
| restaurants | qwen3.5-9b | 18,477 | 0.80 | 2.60 | 3.91 | 0.0% | 0.9988 |
| street_photography | gemma-4-12b | 221,956 | 0.49 | 1.62 | 2.93 | 0.0% | 0.9994 |
| street_photography | qwen3.5-9b | 221,479 | 0.59 | 1.78 | 3.55 | 0.0% | 0.9993 |

**The correction moves no ranking.** Every Spearman reads 0.997 or above, and
only the libraries / qwen3.5-9b cell moves more than 0.1% of its places by more
than 5 percentile points. A published map does not change.

The image rows read a recording, not a place, thus read them as a count of
photographs and not as a place ranking.

## Limits

- **The face is not only the light.** A camera that looks north also sees the
  south wall of the building, which is a different wall. The estimate holds the
  PLACE fixed, not the surface. The sun test separates the two: a wall does not
  turn during the day, and the `R - L` contrast does.
- **It is a lower bound on the camera.** The hour, the season and the weather
  move a photograph as well, and no term here holds them.
- **An image case gives no place score.** Its "place" is 1 recording, which
  sits in about 4 comparisons, thus a score over those 4 is nearly all sampling
  noise. `_faces` returns NaN for the place-level fields of such a case and
  writes a note.

## The files

| File | Purpose |
|------|---------|
| `notebooks/cvpr/_faces.py` | the frame, the estimator, the figures and the table |
| `notebooks/cvpr/master/face_effect.py` | the marimo notebook |
| `scripts/export_face_effect.py` | the reproducible export, with the registry gate |
| `notebooks/cvpr/master/outputs/face_effect_*` | the grid, the table and the LaTeX |
| `notebooks/cvpr/master/figures/face_*` | the coefficient figure and the sun figures |

```bash
.venv-mllmsci-vllm025cu129/bin/python scripts/export_face_effect.py
.venv-mllmsci-vllm025cu129/bin/marimo edit notebooks/cvpr/master/face_effect.py
```

**Warning: the export reads 14 runs of 1,100,000 rows. It needs about 10
minutes.**

## The next step, not yet built

The correction gives each place a score net of its face mix. The natural next
question is whether that correction moves the validation-by-proxy numbers. That
step needs `_proxy_arms`, and nothing here runs it.
