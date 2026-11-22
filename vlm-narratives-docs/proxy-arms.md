# Validation by proxy, run again on every ablation arm

`research/notebooks/cvpr/_proxy_arms.py`, section 4 of
`research/notebooks/cvpr/master/ablations.py`.

## The question

The master table (`results_table.py`) reads the canonical arm alone. This module runs
the same road for EACH prompt ablation, thus a reader can ask what the removed
thing carried.

| Arm | What it removes | The question |
|-----|-----------------|--------------|
| no abstention | the `NotSure` label | Does the model hide its harder calls? |
| no city name | "New York City" in the prompt | Does the place prior carry the agreement? |
| text first | the images-first reading order | Does the reading order carry it? |

| What happens | What it says |
|--------------|--------------|
| The arm agrees LESS | The removed thing carried real signal. |
| The arm agrees MORE | The removed thing carried a bias. |
| No change | The removed thing carried nothing either way. |

## The road

The arm changes only the first step. Everything after it is the code the case
notebooks run:

1. Read the run parquet of the arm and join it to `pairs.parquet`.
2. `_provenance.score_units` (or `score_images`) gives each place a score.
3. `_geography.aggregate` puts the places into the polygons of one layer.
4. Join to the proxy and read n, agreement, Pearson r and Kendall tau-b.

Step 4 reads the proxy parquet that the case notebook exported. A proxy is an
outside measurement, thus it does not move with the arm, and every arm joins
the same file.

`CASE_SPEC` repeats the constants of the 7 case notebooks: the curation root,
the unit table, the `MIN_UNITS_PER_POLYGON` floor, and the park-or-plaza split
that gives one run its 2 table rows.

**Warning: the libraries and the plazas floors moved to 3 on 2026-08-31, and
the 2 case notebooks still read 1.** See
[proxy-audit-fixes.md](proxy-audit-fixes.md) for the 2 lines to change and for
what the floor costs.

**Every arm goes through this module, never through the case export.** A case
notebook can set its own floor, and comparing a recomputed arm with an exported
control would read that difference as a prompt effect.

**The recomputation was exact until 2026-08-31.** On 2026-08-28 the recomputed
canonical rows reproduced all 34 exported rows of the master table — n,
agreement, r and tau-b — to the last digit. The 2 new `min_units` floors break
that match for libraries and plazas until their notebooks run again. Every
other case still matches.

## The trap: an arm that drew fewer pairs

`nocity` and `textfirst` drew 110,000 pairs for each case, and the canonical
battery drew 1,100,000. A place thus takes about 780 comparisons in those arms
against 8,246 in the canonical run. A place score built on 10 times fewer
comparisons carries more sampling noise, and noise attenuates a correlation on
its own.

**Each arm therefore gets its OWN control.** `place_frame(..., match_arm=arm)`
keeps only the comparisons the arm also judged, thus both sides read the same
UNIT pairs in the same A/B order. `match_rate()` reports what stays: 1.000 for
`noabstain`, 0.917 for `nocity`, 0.918 for `textfirst`.

**The join key is `pair_id`, never `canonical_pair_id`.** `pair_id` carries the
presentation order, and about 8% of the shared pairs came up in the opposite
A/B order.

## The trap: the match does not hold the photographs

A matched `pair_id` names the same 2 UNITS in the same A/B order. It does NOT
name the same 2 photographs, outside 2 places.

| Arm or case | Do the photographs match? |
|-------------|---------------------------|
| `noabstain` | Yes, 100%. It reads the same 1,100,000-pair draw. |
| Road quality | Yes, 100%. There the place IS the photograph. |
| `nocity`, `textfirst`, unit cases | 0.18% to 1.65% |

A facility case draws a fresh image for each side when it builds the pairs.
Measured 2026-08-31: libraries 0.18%, schools 0.21%, parks and plazas 0.23%,
subway safety 0.56%, restaurants 1.65%.

This adds NOISE to `d_tau` in `nocity` and `textfirst`, and it adds no
direction: the image draw does not know which arm it serves. **Read a small
`d_tau` in those 2 arms as "no effect that survives the image noise", never as
"no effect".** A large `d_tau` stays readable, because noise cannot make a
difference.

## The interval on `d_tau`

`compare(boot=True)` gives `d_tau_lo` and `d_tau_hi`, a 95% percentile
interval from a bootstrap over AREAS. One resample draws an area index and
reads BOTH arms at it, the same pattern as `_quantile_shift.shift_row`. The 2
arms score the same areas, thus an unpaired interval throws the pairing away
and reads much too wide. `to_display` prints it as `d tau 95%`.

`verdict(tol=0.02)` keeps its band, but the band is a PRACTICAL size and not a
standard error. The marginal error of a Kendall tau-b over 50 areas is near
0.09; a paired difference cancels most of it. Read the interval for
significance, and read the band for size.

## The robustness check: does the abstention enter the area score?

The area score is a plain mean over its places, thus a place that abstained
often pulls the area as hard as a place that answered every time.
`weighted_tau_table()` recomputes the tau with `mean_score_weighted`, which
weights each place by its comparison count, and it reports
`abstention_proxy_r` beside it.

The abstention rate DOES track the proxies (up to r = 0.58 for libraries
income), and the weighted tau still moves the plain tau by under 0.04 on 25 of
the 30 rows that pass the area floor. Thus the agreement does not rest on the
abstention channel. The parks case under qwen3.5-9b is the exception: crime
density moves -0.17 and acquisition year -0.10. Section 4d of the master
notebook shows the frame.

## The trap: a spread is not a ranking

A fall in the correlation does NOT prove the removed thing carried signal. An
arm can squeeze the place scores toward the middle (see
[quantile-shift.md](quantile-shift.md)), and a narrower spread lowers a
correlation with no change of ranking at all.

`spread_ratio` is the standard deviation of the area scores, arm over
canonical. Under 1 says the arm compressed them. **Read `d_tau` against
`spread_ratio`, never alone.**

## The held split (1.1.0)

The tables above ask whether the FULL forced battery agrees with the proxy.
The held split asks the sharper question: do the forced judgments THEMSELVES
carry the signal? `held_table` splits the arm's pairs by what the canonical
arm did on the same pair:

| Side | The pairs it holds |
|------|--------------------|
| `held` | The canonical arm wrote `NotSure`; the arm had to choose. |
| `answered` | The canonical arm gave a label. |

Both halves go down the same road to the proxy, with the same rater and the
same prompt. A subset with fewer pairs gives each place a noisier score, and
noise attenuates a correlation on its own, thus both halves take the size of
the smaller one, with a seed-fixed draw (`HELD_SEED`).

| Reading | Meaning |
|---------|---------|
| `tau held` ≈ `tau answered` | The abstention hid judgments with the same signal. |
| `tau held` ≈ 0 | The forced choices are noise. The abstention was honest. |
| In between | The held calls carry a weaker form of the signal. |

**Warning: the 2 sides split the comparisons of a place, not the places.** A
place with a high abstention rate puts most of its comparisons on the held
side, thus the 2 sides can keep different places and different areas. The
`n_areas` column of each side says what it kept.

## The trap that was: the gemma halves carried a layout change

Before 2026-09-02 the canonical gemma-4-12b runs used `interleaved_labels`
and the `nocity` and `textfirst` arms did not, thus a gemma row of those arms
held the named change AND an image-layout change together. The `nocity` and
`textfirst` gemma halves re-ran on the control's layout on 2026-09-02, and
the `noabstain` gemma half on 2026-09-04. Every gemma row now holds the
named change alone. The `layout_confounded` column stays, `to_display`
prints it as `layout conf.`, and `_ablations.LAYOUT_CONFOUNDED` is empty.

The qwen3.5-9b half of every arm ran 1 layout throughout: `images_then_text`
in the control and `text_first` in the text-first arm, which differ in the
prompt position alone.

## The floors

The table keeps a row that the master table drops, and marks it. Such a row is
worth seeing here, because the rater that abstains most is the one a reader
asks about, but it must not drive the count. `summary(drop_thin=True)` leaves
it out.

| Flag | Rule | The cells it catches |
|------|------|----------------------|
| `thin_coverage` | under `R.MIN_COVERAGE` | Schools, qwen3.5-9b (it abstains on 99.7%) |
| `thin_areas` | under `MIN_AREAS` (20) | POPS completion year; every plazas row at `min_units=3` |

`_results_table.py` owns `MIN_AREAS`, and this module reads it from there.
Before 2026-08-31 each module held its own rule, and the master table held
none, thus the shipped LaTeX starred a POPS row that 12 areas carry. The
master renderers now apply the same floor.

## What it found (2026-08-28)

**Warning: the numbers below predate the 2026-08-31 audit fixes.** The
community district layer then held its 12 joint interest areas, and the
libraries and plazas floors were 1. Read them as the state of the run on that
date, and rebuild section 4d for a current number. See
[proxy-audit-fixes.md](proxy-audit-fixes.md).

At the community district layer, over the 32 cells of each arm that pass both
floors:

| Arm | no change | agrees more | agrees less | median d_tau | median spread |
|-----|-----------|-------------|-------------|--------------|---------------|
| no abstention | 13 | 10 | 9 | +0.001 | 0.73 |
| no city name | 6 | 7 | 19 | -0.032 | 0.88 |
| text first | 3 | 16 | 13 | +0.019 | 0.47 |

**The 3 arms do NOT drift together.** Each one moves the proxy agreement in its
own way, thus the agreement does not rest on one shared cause.

## What it found again (2026-08-31)

The same build after the audit fixes: the JIA areas are gone, and the libraries
and plazas floors are 3. 28 cells of each arm now pass both floors, against 32
before. The plazas case drops out of the count, because 16 areas is under
`MIN_AREAS`.

| Arm | no change | agrees more | agrees less | median d_tau | median spread |
|-----|-----------|-------------|-------------|--------------|---------------|
| no abstention | 9 | 11 | 8 | +0.002 | 0.75 |
| no city name | 5 | 6 | 17 | -0.033 | 0.88 |
| text first | 4 | 12 | 12 | +0.016 | 0.46 |

**The reading does not change.** Every median moves by under 0.001 against the
2026-08-28 build, and `nocity` stays the one arm that costs agreement (17 of 28
cells fall). The fixes remove areas and rows; they do not turn a result over.

### No abstention: it compresses, it does not reorder

The median change of Kendall tau-b is +0.001, the count splits almost evenly,
and the 2 raters lean in opposite directions. But the spread ratio is 0.73. The
held-back judgments shrink the scale and leave the order alone.

The largest single moves are road quality under gemma-4-12b: crime density
falls 0.19 and pothole repairs falls 0.16, both near a spread ratio of 0.55.
The same case under qwen3.5-9b moves the other way (+0.15 and +0.14) at a
spread ratio near 0.94. A rater-specific compression, not a shared effect.

### No city name: the one arm that costs agreement

19 of 32 cells fall, and the median is -0.032. Both raters lean the same way:
gemma-4-12b -0.054 (10 of 17 fall), qwen3.5-9b -0.029 (9 of 15 fall).

**The qwen half settles it.** That half is clean of the layout confound, and
its spread ratio is 0.99 — the scale does not move at all. Thus the fall is a
loss of RANKING, not a compression. **Naming the city carries real information
about the place.** The gemma half falls further (-0.054) but it also compresses
(spread 0.76), so read it as the 2 changes together.

### Text first: the widest scatter and the deepest compression

The median is +0.019, but the spread ratio is 0.47 — this arm halves the scale
of the area scores. The 2 raters split: gemma-4-12b +0.021 (10 of 17 rise),
qwen3.5-9b -0.126 (9 of 15 fall) at a spread ratio of 0.43.

The arm holds the largest single move of the whole table. Subway safety under
qwen3.5-9b against median household income turns over: tau +0.366 to -0.415,
r +0.616 to -0.635, p = 5e-6, with the area score sd falling from 0.175 to
0.065. A real reversal, not a floor artifact.

## Run it

```bash
uv run marimo edit research/notebooks/cvpr/master/ablations.py
# or, from a command
python research/scripts/export_cvpr_ablations.py
```

Section 4 builds the long frame, the drift frame, the direction count, the
arm-by-arm wide table, and 3 figures. The layer control at the top of the
notebook drives it. The export button writes them under
`research/notebooks/cvpr/master/outputs/`.

The stem is `proxy_arms_<layer>_v<version>`:

| File | What it holds |
|------|---------------|
| `<stem>_long.csv` | 1 row for each (arm, cell), every metric |
| `<stem>.csv` and `.tex` | the arm beside its control |
| `<stem>_wide.csv` | 1 row for each cell, 1 column for each arm |
| `<stem>_drift.png` | 1 row for each cell, 1 colour for each arm |
| `<stem>_slopes.png` | 1 panel for each arm, canonical dot to arm dot |
| `<stem>_spread.png` | the change of tau against the compression |

Section 4d also writes `proxy_weighted_tau_<layer>_v<version>.csv`, the
robustness frame of the weighted area mean.

A full build reads the run parquets of 4 arms and does the spatial joins again.
It takes about 5 minutes, and `place_frame` and `aggregate_arm` cache within
the session. The paired interval adds about 0.2 s for each row.
