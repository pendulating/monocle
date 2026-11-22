# The quantile shift of a prompt arm

`research/notebooks/cvpr/_quantile_shift.py`, section 3 of
`research/notebooks/cvpr/master/ablations.py`.

## Why a mean cannot read this arm

The ablation table gives one number for the change of the ordinal score:
`score_delta`, the paired mean. That number is blind to the no-abstention arm.

A pairwise score is antisymmetric. Each comparison adds `+s` to one place and
`-s` to the other, thus the mean over places stays near 0 whatever the prompt
does. A prompt change that squeezes the scores toward the middle, or that
stretches the tails, moves NO mean.

## The statistic

Read the change rank by rank. Sort the place scores of each arm. The 2 arms
score the same places, thus the 2 lists have the same length, and

    delta(tau) = the arm list minus the canonical list, at the same rank

It is signed, it is per rank, and it is in the units of the score.

| Column | Definition | Reading |
|--------|------------|---------|
| `spread_ratio` | `sd_arm` / `sd_canon` | 1.00 = no change; 0.73 = a 27% compression |
| `rank_tau` | Kendall tau-b between the 2 arms' place scores | 1.00 = the order of the places held |
| `d05` | `delta` at the 5th percentile | How far the bottom of the scale moved |
| `d25` `d50` `d75` | `delta` at the quartiles | The body of the distribution |
| `d95` | `delta` at the 95th percentile | How far the top of the scale moved |

A positive `d05` beside a negative `d95` says the arm pulled the 2 ends toward
the middle. `spread_ratio` says the same thing in one number, and it is the
statistic that the `spread` column of `_proxy_arms.py` reports. That module
reads it over the areas of a proxy join and this one reads it over the places,
thus the 2 tables can be read against each other.

The sorted-rank deltas are blind to the identity of a place: a shuffle of the
scores inside one arm moves no `delta`. `rank_tau` (added in 2.1.0) reads that
other axis on the same paired frame. It uses the tau-b variant that
`_results_table.py` reads against the proxies. An arm can compress the scale
and keep the order, or hold the scale and shuffle the places; `spread_ratio`
and `rank_tau` separate those effects.

`TAUS_REPORT` holds the 5 quantiles of the table and `TAUS_HEADLINE` holds the
2 quantiles of the figure. The plotted curve uses a 999-point grid. Only the
figure reads that grid.

## Version 2.0.0 drops the Wasserstein decomposition

Version 1 reported the 1-Wasserstein distance `w1`, a `mean_shift` term, and
`reshape = w1 - |mean_shift|`. The antisymmetry holds `mean_shift` near 0 in
every cell, thus `reshape` only repeated `w1` and the share was near 100%
always. Those 3 columns carried no information, and 2 of them needed an
integral to explain.

Version 2 removes them, along with `grid_w1` and the Wasserstein call, and
reports the sorted-rank deltas and the spread ratio directly. The taus 0.01 and
0.99 also leave: with a floor of 30 places that quantile is the sample end, and
a bootstrap there is not reliable.

## The variable is a PLACE

A comparison carries 1 of 5 ordinal labels, and a quantile of 5 values is a
step function. The continuous variable sits one level up: the mean score of a
place over every comparison it took part in. This is the quantity that
`_provenance.score_units` builds and that the validation-by-proxy table reads
after the geography join.

A place is a facility where `pairs.parquet` names one (`unit_uid`), and an
image where it does not (`sample_id`).

**Warning: 2 of the 7 cases have no continuous score.** Road quality and street
photography pair random street-level images, thus a place is 1 image and it
takes about 4 comparisons. A mean of 4 ordinal values takes 17 distinct values,
and its quantiles step. `MIN_COMPARISONS` (30) drops such a place, and those 2
cases then hold no quantile. `skipped()` prints the reason.

## The pairing

The table reads the places that BOTH arms score. The canonical arm abstains,
thus it scores fewer comparisons for each place, and it can lose a place to the
`MIN_COMPARISONS` floor that the arm keeps. A distribution over a different
place set would read that difference as a prompt effect. `n_arm_only` and
`n_canon_only` report what the restriction costs.

## The interval

The interval resamples PLACES and reads both arms at the same place. The 2 arms
scored the same places, thus an unpaired interval would throw that away and
read much too wide. The same loop gives the interval of each `delta`, of
`spread_ratio`, and of `rank_tau`.

## What it found (2026-08-28)

9 cells of the 14 carry enough places. Version 1 wrote these deltas, and
version 2 computes them the same way:

| Cell | d(0.05) | d(0.95) |
|---|---|---|
| Schools, gemma-4-12b | +0.654 | -0.284 |
| Parks / Plazas, qwen3.5-9b | +0.527 | -0.432 |
| Parks / Plazas, gemma-4-12b | +0.321 | -0.191 |
| Subway safety, gemma-4-12b | +0.158 | -0.144 |
| Libraries, gemma-4-12b | +0.228 | -0.163 |
| Restaurants, qwen3.5-9b | +0.072 | +0.032 |

**The arm compresses.** In 7 of the 9 cells the low quantiles go UP and the
high quantiles go DOWN. The mean stays near 0. Forcing an answer does not move
the middle; it pulls both ends toward it.

Restaurants under qwen3.5-9b is the one clean exception. There every quantile
rises together, thus the move is a shift and not a compression.

**Warning: `spread_ratio` and `rank_tau` are not in this page.** Version 1
reported a 90% span and no rank statistic, thus the table must run again
before a spread or a rank number is quoted.

The compression matters for the proxy table, because a correlation falls with
the spread even when the ranking does not change. See
[proxy-arms.md](proxy-arms.md), which measures that directly.

## Run it

```bash
uv run marimo edit research/notebooks/cvpr/master/ablations.py
# or, from a command
python research/scripts/export_cvpr_ablations.py
```

Section 3 builds the table, the skipped list, and the 2 figures. The export
button writes them under `research/notebooks/cvpr/master/outputs/`.
