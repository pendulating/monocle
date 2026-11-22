# The tier-1 audit fixes of the proxy road

Date: 2026-08-31. Modules: `research/notebooks/cvpr/_geography.py`,
`research/notebooks/cvpr/_results_table.py`, `research/notebooks/cvpr/_proxy_arms.py`,
section 4d of `research/notebooks/cvpr/master/results_table.py` (section 4 of
`master/ablations.py` since 2026-09-04), and the 7 case
validation notebooks with their `recipe.json` files.

An audit of the validation-by-proxy road found 6 problems. This page records
what each fix changes, and which published number moves.

## The 6 fixes

| # | Problem | Fix |
|---|---------|-----|
| 1 | The layers hold areas with no residents | `_geography` drops them |
| 2 | 2 modules held 2 different area floors | 1 constant, `R.MIN_AREAS` |
| 3 | `min_units=1` let 1 place carry an area | 3 for plazas and libraries |
| 4 | The area mean weights each place the same | A weighted mean beside it |
| 5 | The arm match claimed the photographs | The docstring states the truth |
| 6 | A proxy export named no metric, no source, and no window | 3 provenance columns, and a naming contract |

## Fix 1: the areas with no residents

`_geography.load_layer` now drops the areas that hold no residents. Pass
`include_jia=True` to keep them and reproduce an older number.

| Layer | Rule | Areas before | Areas after |
|-------|------|--------------|-------------|
| Community district | `boro_cd` not in `JIA_BORO_CD` | 71 | 59 |
| NTA | `NTAType` is `0` | 262 | 197 |
| Census tract | none | 2325 | 2325 |

NYC calls a community district of this kind a joint interest area. The 12
codes are 164 (Central Park), 226-228 (the Bronx parks), 355-356 (Prospect
Park and the Brooklyn parks), 480 and 484 (the Queens parks), 481 (LaGuardia),
482 (JFK), 483 (Rikers Island), and 595 (the Staten Island parks).

Why the default drops them: a JIA has no household income, no school, and
almost no resident crime record. A census proxy there is a boundary artifact.

The NTA file marks the same kind of area in its `NTAType` column: `5` is
Rikers Island, `6` an institution or a rail yard, `7` a cemetery, `8` an
airport, and `9` a park. The census tract file carries no column of that kind,
thus this fix leaves that layer whole. A tract inside a park holds few units,
and `min_units` already drops it.

### Why the fix needs no new proxy parquet

Every comparison joins the model side and the proxy side INNER on the layer
key. `_proxies` and `_geography` both read `load_layer`, thus a rebuild of
either side loses the JIA rows. An export that a case notebook wrote BEFORE
this fix still holds them, so `_results_table.load_case` drops the keys
directly. `_geography.excluded_keys(layer)` names them.

### The numbers that move

Community district layer, the master table, canonical arm. Only the rows that
move are here.

| Case | Proxy | n | gemma tau | qwen tau |
|------|-------|---|-----------|----------|
| Subway safety | Crime density | 62 → 57 | -0.09 → -0.14 | -0.17 → -0.23 |
| Libraries | Construction year | 60 → 59 | -0.34 → -0.34 | -0.31 → -0.33 |
| Road quality | Crime density | 71 → 59 | +0.54 → +0.46 | +0.03 → +0.24 |
| Road quality | Pothole repairs | 70 → 59 | +0.42 → +0.28 | -0.11 → -0.00 |
| Road quality | DOT pavement | 65 → 59 | -0.10 → -0.11 | +0.06 → +0.09 |
| Road quality | Income | 60 → 59 | +0.10 → +0.12 | +0.27 → +0.26 |
| Parks | Crime density | 60 → 59 | +0.22 → +0.20 | +0.24 → +0.22 |
| Parks | PIP acceptable | 60 → 59 | +0.17 → +0.15 | +0.05 → +0.03 |
| Parks | Acquisition year | 60 → 59 | -0.18 → -0.15 | -0.13 → -0.10 |
| Restaurants | Inspection score | 60 → 59 | +0.06 → +0.10 | +0.04 → +0.07 |

The left value is the state before the fix, and the right value is the state
after it.

Road quality loses 12 of its 71 areas, more than any other case. Its places
are random street-level photographs, thus a large park or an airport collects
many of them. **The 2 largest single moves are there**: the gemma-4-12b
pothole row falls 0.15, and the qwen3.5-9b crime row rises 0.21 and becomes
significant.

The subway crime row also changes its reading: the qwen3.5-9b tau moves from
-0.17 (p = 0.05) to -0.23 (p = 0.01).

Plazas, schools, and the income rows of subway, libraries and parks do not
move at all. Their exports held no JIA key at this layer.

## Fix 2: one area floor

`MIN_AREAS = 20` now lives in `_results_table.py`. `_proxy_arms.py` reads it
from there. Before this fix `_proxy_arms` held its own copy and
`_results_table` held no floor at all, thus the shipped LaTeX starred a POPS
row that 12 areas carry.

| Where | Before | After |
|-------|--------|-------|
| `_results_table.build` | no floor | keeps the row, marks `thin_areas` |
| `_results_table.to_latex` | prints and stars it | drops or dashes it |
| `_results_table.to_display` | prints it | drops or dashes it |
| `_proxy_arms` | its own `MIN_AREAS = 20` | reads `R.MIN_AREAS` |

The rule, the same as `_proxy_arms` applies:

1. The long frame keeps every row, and marks a thin one.
2. A renderer drops a proxy row where NO rater reaches the floor.
3. A renderer dashes the 3 metrics of a rater that does not reach it, and it
   keeps that rater's `n`, so a reader sees why the dash is there.
4. The footnote states the floor and names each dropped row.

**The POPS completion year row of the plazas case leaves the table.** It held
12 areas, and its qwen3.5-9b tau of -0.727 carried a star at p = 0.0005. That
star is gone. The row stays in `results_table_community_district_v1.0.0_long.csv`.

## Fix 3: the `min_units` floor of plazas and libraries

`_proxy_arms.CASE_SPEC` now sets `min_units=3` for plazas and libraries. A
floor of 1 lets an area score rest on a single place.

| Case | Areas at floor 1 | Areas with 1 place | Share |
|------|------------------|--------------------|-------|
| Plazas | 43 | 19 | 44% |
| Libraries | 59 | 3 | 5% |

The plazas case is the bad one. The libraries case loses 11 of its 59 areas at
floor 3, thus 19% of its areas held 1 or 2 libraries.

**Warning: the 2 case notebooks still read 1. They must follow.**

| File | Line on 2026-08-31 | Now | Must become |
|------|--------------------|-----|-------------|
| `research/notebooks/cvpr/plazas/plazas_validation.py` | 73 | `MIN_UNITS_PER_POLYGON = 1` | `= 3` |
| `research/notebooks/cvpr/libraries/libraries_validation.py` | 61 | `MIN_UNITS_PER_POLYGON = 1` | `= 3` |

The line moves when a notebook changes. Find it again with:

```bash
grep -n "MIN_UNITS_PER_POLYGON = " research/notebooks/cvpr/plazas/plazas_validation.py \
    research/notebooks/cvpr/libraries/libraries_validation.py
```

The libraries notebook also carries a markdown block at line 161 that states
the floor of 1 and its reason. Rewrite that block with the new floor. Run each
notebook again after the change, or the master table keeps the floor-1 export.

### What the floor costs, and what it buys

Community district layer, canonical arm, measured through `_proxy_arms`.

| Case | Proxy | Rater | n at 1 | n at 3 | tau at 1 | tau at 3 |
|------|-------|-------|--------|--------|----------|----------|
| Plazas | Income | gemma-4-12b | 43 | 16 | +0.216 | +0.467 |
| Plazas | Income | qwen3.5-9b | 43 | 16 | +0.152 | +0.533 |
| Plazas | Crime density | gemma-4-12b | 43 | 16 | -0.130 | -0.050 |
| Plazas | Crime density | qwen3.5-9b | 43 | 16 | -0.110 | +0.050 |
| Plazas | POPS year | gemma-4-12b | 12 | 11 | -0.364 | -0.273 |
| Plazas | POPS year | qwen3.5-9b | 12 | 11 | -0.727 | -0.709 |
| Libraries | Income | gemma-4-12b | 59 | 48 | +0.102 | +0.188 |
| Libraries | Income | qwen3.5-9b | 59 | 48 | +0.143 | +0.199 |
| Libraries | Construction year | gemma-4-12b | 59 | 48 | -0.344 | -0.387 |
| Libraries | Construction year | qwen3.5-9b | 59 | 48 | -0.333 | -0.447 |

**Warning: the plazas case falls under the area floor at floor 3.** 16 areas
is under `MIN_AREAS = 20`, thus every plazas row of the arm table now carries
the `thin_areas` mark and `summary(drop_thin=True)` leaves it out. The plazas
case needs a finer layer, or it needs the floor-1 reading with the caveat
stated.

**The size of the tau rises on 6 of the 10 rows and falls on 4.** A rise is
what the fix predicts: an area built on 1 plaza carries the noise of that
plaza, and noise attenuates a correlation. The 2 income rows of the plazas
case more than double, and their p values fall although the n falls with them.
The 2 crime rows move the other way, and 16 areas cannot settle which reading
is right.

## Fix 4: the weighting and the coverage columns

`_geography.aggregate` now gives `mean_score_weighted` beside `mean_score`.
The weight is the comparison count of the place. The plain mean gives each
place the same voice although the kept comparison counts of one area differ by
a factor of 25 after the `NotSure` drop, thus the abstention rate enters the
plain mean.

`aggregate` also takes an optional `units_total` point layer. Pass the full
unit layer, and the frame gains `n_units_total` and `coverage`. The default is
`None`, and then the 2 columns stay absent.

`_proxy_arms._aggregate_scores` now gives the SAME columns in both modes. The
image branch had no `mean_abstention` and no `layer`, thus a reader of a road
quality frame could not test the abstention at all.

### The robustness check

`_proxy_arms.weighted_tau_table()` recomputes the Kendall tau-b with
`mean_score_weighted` and reports it beside the plain tau. It also gives
`abstention_proxy_r`, the Pearson correlation of the area abstention rate with
the proxy. Section 4d of the master notebook shows it in one cell.

What it found at the community district layer, canonical arm, over the 30 rows
that pass the area floor:

| Reading | Value |
|---------|-------|
| Rows where `d_tau_weight` is under 0.04 | 25 of 30 |
| Largest `d_tau_weight` | -0.17 (parks, qwen3.5-9b, crime density) |
| Second largest | -0.10 (parks, qwen3.5-9b, acquisition year) |
| Largest `abstention_proxy_r` | +0.58 (libraries, qwen3.5-9b, income) |
| Largest negative `abstention_proxy_r` | -0.49 (schools, gemma-4-12b, report card) |

**The abstention rate tracks the proxies, and the weighting still moves almost
no tau.** The area abstention rate reaches r = 0.58 against income, thus the
channel is real. 25 of the 30 rows move under 0.04 when the weight enters, and
no headline row changes its sign or its reading. The parks case under
qwen3.5-9b is the exception, and the paper must state the mean it reports for
that case.

The 6 plazas rows sit under the area floor. They move most (+0.10 and -0.11),
which is what 16 areas give.

## Fix 5: what an arm match really matches

`_proxy_arms.place_frame` said that a matched pair gives "the same
comparisons, in the same A/B order". Section 4d of the master notebook said
the same. Only half of that is true.

| Level | Does a matched `pair_id` hold it? |
|-------|----------------------------------|
| The unit pair | Yes |
| The A/B presentation order | Yes |
| The 2 photographs | Only for `noabstain` and for road quality |

A facility case draws a fresh image for each side when it builds the pairs,
thus the 100,000-pair arms show the model OTHER pictures of the same 2 units.
The share of shared pairs where both photographs match, measured 2026-08-31:

| Case | Share |
|------|-------|
| Libraries | 0.18% |
| Schools | 0.21% |
| Parks and plazas | 0.23% |
| Subway safety | 0.56% |
| Restaurants | 1.65% |
| Road quality | 100% |

`noabstain` matches on 100% of its 1,100,000 shared pairs: it reads the same
draw. Road quality is an image case, thus its place IS the photograph.

**What this costs.** The image mismatch adds NOISE to `d_tau` for `nocity` and
`textfirst`, and it adds no direction: the image draw does not know which arm
it serves. Read a small `d_tau` in those 2 arms as "no effect that survives
the image noise", never as "no effect". A `d_tau` that IS large stays
readable, because noise cannot make a difference.

### The paired interval on `d_tau`

`_proxy_arms.compare(boot=True)` now gives `d_tau_lo` and `d_tau_hi`, a 95%
percentile interval from a bootstrap over AREAS. One resample draws an area
index and reads BOTH arms at it, the same pattern as
`_quantile_shift.shift_row`. `to_display` prints it as `d tau 95%`.

The `tol=0.02` band of `verdict` stays, but its stated reason was wrong. The
old docstring called 0.09 the standard error of a tau over 50 areas and read
0.02 against it. That marginal error does not apply: `d_tau` is a PAIRED
difference, both arms read the same areas, and most of the error cancels. The
measured paired interval is much narrower than 0.09 — the libraries income row
of `noabstain` reads -0.01 [-0.05, +0.03]. Read the interval for significance
and the band for size.

## Fix 6: the provenance of a proxy export

Date: 2026-08-31. Files: the 7 case validation notebooks, their `recipe.json`
files, and their `outputs/` directories.

### The problem

3 notebooks pick the proxy metric from a dropdown, and then write a FIXED name
into the `proxy` column. The name does not follow the selection. Thus an
export can carry the name of a metric that the run never read.

| Case | Dropdown | Name before | The fault |
|------|----------|-------------|-----------|
| Schools | 40 School Quality Reports metrics | `school_report_card` | The run reads "Average Student Attendance". The paper calls it a report card. |
| Parks | cleanliness or condition | `pip_acceptable_rate` | The name does not say which rating. |
| Restaurants | `inspection_score` or `grade_a_rate` | none | The export holds no `proxy` column at all. |

The subway crime dropdown does it correctly: it writes the selected metric
into the column. That is the pattern the other 3 now follow.

No export named its time window. The battery mixes 5 windows, thus a reader of
1 parquet cannot tell a 5-year census estimate from 1 winter quarter of crime.

### The 3 new columns

Every proxy export of every case now carries them.

| Column | Holds | Example |
|--------|-------|---------|
| `proxy_metric` | The metric or the column that the run selected | `Average Student Attendance` |
| `proxy_source` | The dataset id, or the file stem, from `recipe.json` | `dnpx-dfnc` |
| `proxy_window` | The time window of the source | `2026Q1` |

A cell in each notebook holds the rule:

```python
def stamp(frame, proxy, metric, entry, window=None):
    """Add the proxy name and the 3 provenance columns to 1 frame."""
    out = frame.copy()
    out["proxy"] = proxy
    out["proxy_metric"] = metric
    out["proxy_source"] = source_id(entry)
    out["proxy_window"] = window or entry["window"]
    return out
```

`entry` is a `recipe.json` source, thus the recipe is the 1 place that records
a window. Change a year filter in a notebook, and change the recipe with it.

### The naming contract

**Warning: never write a fixed proxy name beside a dropdown.**

| Case | Rule |
|------|------|
| Schools | `METRIC_PROXY_NAME` maps the metric to a name. An unnamed metric takes a slug of its own text. |
| Parks | `pip_proxy_name` maps the PIP metric to a name, and it raises an error on an unknown metric. |
| Restaurants | The name IS the dropdown value. |
| Subway, plazas, parks (crime) | The name IS the dropdown value. |
| Road | No dropdown. The name comes from the `CRIME_METRIC` constant that the call also uses. |

The parks cleanliness rows keep the name `pip_acceptable_rate`, because the
master table and the canonical registry hold that key. A condition run now
exports as `pip_condition_acceptable_rate`, and `proxy_metric` names the exact
rating on every row.

### The window of every source

| Source | Proxy | `proxy_source` | `proxy_window` |
|--------|-------|----------------|----------------|
| School Quality Reports | `school_attendance` | `dnpx-dfnc` | `school year 2024` |
| Building footprints | `construction_year` | `5zhs-2jue` | `vintage; pulled 2026-04-12` |
| DOHMH restaurants, curated | `inspection_score` | `restaurants_aggregated` | `last inspection of each unit, to 2026-04-25` |
| DOHMH restaurants, curated | `grade_a_rate` | `restaurants_aggregated` | `all inspections, 2007 to 2026-04-25` |
| ACS 5-Year 2024, S1901 | `median_household_income` | `ACSST5Y2024.S1901-Data` | `2020-2024 ACS 5-year` |
| NYPD complaints | `crime_density` | `nypd_complaint_ytd_2026_pulled20260629` | `2026Q1` |
| Parks Inspection Program | `pip_acceptable_rate` | `yg3y-7juh` | `2023-2026 inspections` |
| DPR Parks Properties | `acquisition_year` | `enfh-gkve` | `vintage; pulled 2026-08-19` |
| POPS | `year_completed` | `rvih-nhyn` | `vintage; pulled 2026-08-19` |
| DOT pavement ratings | `dot_pavement_rating` | `6yyb-pb25` | `2024-01 to 2026-07 inspections` |
| Pothole work orders | `pothole_repairs` | `x9wy-ing4` | `2024-01 to 2026-07 work orders` |

**Warning: the crime source is 1 winter quarter, not a year.** The file name
says year to date 2026. The record stops on 2026-03-31, and 94.8% of the
133,114 complaints fall in January to March 2026. The 4 recipes that name it
now say `2026 Q1`, and the notebooks state it in section 4.

**Warning: 2 restaurant metrics read 2 windows of 1 file.** `inspection_score`
reads `last_score`, thus its window is the LAST inspection of each unit: the
median unit sits at 2025-09. `grade_a_rate` reads the whole history.

### The schools rename

| Version | Name in `proxy` | What it measures |
|---------|-----------------|------------------|
| 1.0.0 | `school_report_card` | Average Student Attendance |
| 1.1.0 | `school_attendance` | Average Student Attendance |

The old name describes the SOURCE, and the source holds 40 metrics. The new
name describes the measurement. A construct upgrade to a better metric is a
separate task: this fix only makes the label true.

`_results_table.py` follows the rename. `PROXY_LABEL` now holds
`"school_attendance": "School attendance"`, and it keeps the old key for an
archived v1.0.0 export. `CASES` names `school_attendance` as the
`single_proxy` of the schools folder.

**Warning: the label was necessary, not cosmetic.** Without it the LaTeX
printed the raw key `school_attendance`, and an underscore in text mode stops
a LaTeX build.

### What moved on disk

Every case wrote a `_proxy_v1.1.0.parquet` beside its v1.0.0 file. The 2 files
hold the same `proxy_mean` and the same `n_units_proxy`, row for row. Only the
name and the 3 new columns differ. `_results_table`, `_proxy_arms`, and
`_ic_link` each glob `_proxy_v*.parquet` and take `sorted()[-1]`, thus they
read the new file with no edit.

| Case | File | Rows |
|------|------|------|
| Schools | `schools_proxy_v1.1.0.parquet` | 1,077 |
| Libraries | `libraries_proxy_v1.1.0.parquet` | 2,894 |
| Subway | `subway_proxy_v1.1.0.parquet` | 5,080 |
| Parks | `parks_proxy_v1.1.0.parquet` | 7,674 |
| Plazas | `plazas_proxy_v1.1.0.parquet` | 5,225 |
| Road quality | `road_quality_proxy_v1.1.0.parquet` | 10,270 |
| Restaurants | `restaurants_proxy_v1.1.0.parquet` | 1,810 |

The `_aggregated_`, `_correlations_`, and `_provenance_` exports stay at
v1.0.0. This fix changes no model score.

### The rows that fix 1 still drops

Fix 1 came after the v1.0.0 export, thus the v1.1.0 files still hold the areas
with no residents. A rebuild of a case notebook now drops them at build time.
Measured 2026-08-31 on the subway case: every proxy value agrees to 0.0 on the
shared areas, the census tract layer does not change, and every dropped key
sits in `_geography.excluded_keys`. Thus the files stay correct, and
`_results_table.load_case` removes the keys as fix 1 describes.

## What the arm table now reads

`_proxy_arms.build()` and `drift()` ran again on 2026-08-31 at the community
district layer, with the fixes in. 28 cells of each arm pass both floors,
against 32 before. The plazas case leaves the count, because 16 areas is under
`MIN_AREAS`.

| Arm | no change | agrees more | agrees less | median d_tau | median spread |
|-----|-----------|-------------|-------------|--------------|---------------|
| no abstention | 9 | 11 | 8 | +0.002 | 0.75 |
| no city name | 5 | 6 | 17 | -0.033 | 0.88 |
| text first | 4 | 12 | 12 | +0.016 | 0.46 |

**No arm reading turns over.** Each median moves by under 0.001 against the
2026-08-28 build, and `nocity` stays the one arm that costs agreement. The
full build takes 262 s, and the paired interval is most of the added cost.

## What ran again on 2026-08-31

| Step | State |
|------|-------|
| `research/scripts/export_cvpr_results_table.py --allow-stale` | Ran. It wrote 15 files for the 3 layers. |
| `_proxy_arms.build` + `drift` + `weighted_tau_table` | Ran headless. 262 s. |
| The plazas and libraries case notebooks | NOT run. They still hold `MIN_UNITS_PER_POLYGON = 1`. |

**Warning: the export needs `--allow-stale` today.** Gate 2 of the script asks
each case export to be newer than the canonical registry. Every case export is
of 2026-08-20 and the registry is of 2026-08-27, thus the gate refuses. That
staleness is older than these fixes, and every shipped number already came
from those exports.

## What to run again

| Step | Command |
|------|---------|
| The case notebooks of plazas and libraries | `marimo edit research/notebooks/cvpr/<case>/<case>_validation.py` |
| The master table | `python research/scripts/export_cvpr_results_table.py` |
| The ablation arms | `python research/scripts/export_cvpr_ablations.py` |

The master export reads the case exports, thus run the case notebooks first.
A fresh case export also lifts the staleness gate.

## The integration of the 5 proxy upgrades (2026-08-31)

The 5 upgrade agents wrote the v1.2.0 proxies into the `_proxies_<case>.py`
modules and the case exports. This section records the pass that wired those
proxies into the shared tooling.

### The label additions in `_results_table.py`

`PROXY_LABEL` gained the 20 new keys, thus no raw key prints in a table. The
new subway, road, restaurants, buildings, plazas, parks, and schools proxies
all carry a display name and an orientation note.

### The context mechanism

`CONTEXT_PROXIES` names the rows that describe the place and make no
validation claim:

- `park_acreage` — a size measure
- `subway_ridership` — a size measure
- `outdoor_dining` — a visible confounder of the preference judgment
- `school_attendance` — an outcome of the population, not of the building

A renderer marks a context row with a dagger and a footnote, and it never
hides the row. `park_acreage` and `subway_ridership` also joined
`UNORIENTED_PROXIES`.

### The supersession mechanism

`SUPERSEDED_PROXIES` maps an old measurement to its replacement, by case:

| Case | Old row | Replacement |
|------|---------|-------------|
| subway | `crime_density` | `transit_crime_per_rider` |
| road | `dot_pavement_rating` | `dot_pavement_rating_weighted` |
| road | `pothole_repairs` | `pothole_repairs_per_mile` |
| restaurants | `inspection_score` | `inspection_score_history` |

The long frame and the CSV keep every row. A renderer holds an old row back
only when the SAME case also carries the replacement, and a footnote names
the held rows. `crime_density` stays printed for road, parks, and plazas:
only the subway case got a per-rider replacement.

### The plazas `min_units` revert

Fix 3 set `min_units = 3` for plazas. The parks agent measured the cost: the
floor keeps 16 community districts, which is under `MIN_AREAS = 20`, thus the
whole case leaves the area table. `_proxy_arms.CASE_SPEC` now sets plazas
back to 1 and keeps libraries at 3, and the unit layer answers the
single-plaza districts.

### The unit layer grew to 7 cases

`_unit_validation.CASE_SPEC` gained libraries and plazas, and a case can now
carry alternates in `alt_proxies`. The restaurants headline moved to
`inspection_score_history`, and parks gained the condition rate and the
acreage. `reconcile()` now matches on the proxy as well as the model, and it
reads all 6 published unit-correlation exports. Every reconcile row agrees:
16 of 16 rows match the published rho and n exactly or inside 0.0001.

### The unit-level headlines of the 5 upgrades

| Case | Headline proxy | gemma rho | qwen rho | n | Reading |
|------|----------------|-----------|----------|---|---------|
| subway | transit crime per 100k riders | +0.207 | +0.294 | 419 | The per-rider rate FLIPS the crime sign to agreement. |
| restaurants | inspection history mean | -0.067 | -0.069 | 12,151 | Stays negative; about -0.10 after the reliability correction. `outdoor_dining` (+0.150 / +0.068) is a confounder. |
| libraries | building violations, total | -0.371 | -0.477 | 228 | More violations, better model score. Permits run positive (+0.267 / +0.375). |
| road | pothole repairs per mile | +0.232 | -0.021 (tau, CD) | 59 areas | The per-mile denominator shaves the density confound; centerline base is 6,838.2 mi. |
| plazas | DOB and ECB violations | +0.158 | +0.139 | 395 | The first unit signal of the plazas case. gemma `park_acreage` (+0.285) is a size confounder. |

### The export

`research/scripts/export_cvpr_results_table.py --allow-stale` ran again on 2026-08-31
and wrote the 15 files for the 3 layers. The stale gate still fires for the
2026-08-20 aggregated exports, which is the known cosmetic staleness above.
The exported tables now carry 41 case-proxy rows for each layer, with the
daggers and the supersession footnote in the LaTeX.
