# Validation by proxy at the unit layer

`research/notebooks/cvpr/_unit_validation.py`, section 1b of
`research/notebooks/cvpr/master/results_table.py`.

## Why the unit layer is the headline

The master table validates the model at the community district layer. That
layer is a choice, and the choice decides the result. A community district
holds hundreds of facilities, thus the mean of a facility measure is nearly the
same in every district.

The module measures that flatness directly. `proxy_cv_units` is the
coefficient of variation of the proxy over the units of a join, and
`area_proxy_cv` is the same measure over the areas of the same join.

| Case | Proxy | CV over units | CV over areas |
|------|-------|---------------|---------------|
| Restaurants | DOHMH inspection score | 0.71 | 0.07 |
| Road quality | DOT pavement rating | 0.25 | 0.05 |
| Parks | PIP acceptable rate | 0.14 | 0.06 |
| Schools | School report card | 0.04 | 0.02 |

The socio-economic proxies keep a CV of 0.4 to 0.95 over the same areas. Thus
an area table puts a nearly flat measure beside a measure that varies 10 times
more, and the reader cannot separate the 2.

The unit layer removes the choice. A unit is the object the model looked at and
the object the record grades: 1 park, 1 school, 1 restaurant, 1 image of a
street. The case README states the rule:

> Report the unit value. Use the area value only to show the gap.

## The road from a run to a number

1. Read the canonical run through the registry (`_canonical`).
2. `_provenance.score_units` or `score_images` gives each place a score.
3. Join the unit-level proxy value with the rule of the case notebook.
4. Report n, Spearman rho, Kendall tau-b, and a bootstrap interval.

`with_area()` then reads the AREA value of the same cell through
`_results_table.load_case` and `_results_table.metrics`, which is the
arithmetic of the master table. Thus 1 row holds both layers and the gap.

## The join rule of each case

| Case | Unit | Key | How the proxy reaches the unit |
|------|------|-----|--------------------------------|
| restaurants | 1 restaurant | `unit_uid` | `uid` is `unit_uid`; an exact key join |
| schools | 1 school | `unit_uid` | borough and school name; a name join |
| parks | 1 park | `unit_uid` | the unit point in the DPR polygon |
| road | 1 image | `sample_id` | the nearest rated segment, 100 ft |

### restaurants

`_proxies.restaurant_proxy` reads
`curation/dohmh_restaurants_inspected_all/restaurants_aggregated.parquet` and
writes `proxy_key = uid`. The recipe states that `uid` matches `unit_uid` in
the pairs manifest exactly, thus the map from a unit to a proxy key is the
identity. The metric is `inspection_score`, and `min_inspections = 2` repeats
the notebook constant. The sign flip happens inside `_proxies`, thus a positive
correlation means agreement.

### schools

`_proxies.school_quality_wide` takes the metric `Average Student Attendance`
of the school year `2024`. `_proxies.join_to_units` then joins that table to
the FacDB units by borough and normalized school name.

**Warning: the name join is not complete.** The proxy covers public schools,
and FacDB also holds private, charter, and postgraduate schools, which carry no
DBN. 1,136 of 2,287 scored schools get a value, which is 49.7%. The missing
schools are not a random half.

**Warning: a borough and a name can hold 2 DBNs.** The module averages the
proxy values of such a unit before the merge. Without that step the unit enters
the correlation twice.

### parks

`_proxies.attach_park_property_id` gives each unit its DPR `gispropnum`, and
`_proxies.parks_pip_proxy` gives each property its share of acceptable
inspections since 2023, with `min_inspections = 2`. The metric is
`cleanliness_acceptable_rate` and the distance cap is 500 ft.

**Warning: never use the FacDB polygon of a park.** It is a nearby building
with a median area of 0.0038 km^2, and Central Park covers 3.41 km^2. The
function uses the unit POINT and the true DPR polygon.

### road

`_proxies.pavement_segments` reads the DOT Street Pavement Ratings since 2024
and drops `systemrating = 0.0`, which means NOT RATED.
`_proxies.attach_nearest_segment` gives each image the rating of the nearest
segment within 100 ft, and it breaks a tie by distance. A camera sits in the
roadway, thus a small cap is right; a large cap attaches the rating of another
street.

The proxy of this case is a LINE layer, thus `unit_proxy()` returns None for
road and `unit_key()` carries the value.

## The floors

| Floor | Value | Why |
|-------|-------|-----|
| `MIN_COMPARISONS_FACILITY` | 5 | The `_provenance.score_units` default. A facility sits in hundreds of comparisons, thus the floor drops only a unit that 1 or 2 comparisons describe. |
| `MIN_COMPARISONS_IMAGE` | 1 | The `_provenance.score_images` default. |
| `MIN_UNITS` | 30 | A cell under 30 units gets a `thin_units` mark. |
| `R.MIN_COVERAGE` | 0.5 | A rater under it gets a `thin_coverage` mark. |

**Warning: do not raise the image floor to 5.** An image case spreads its pairs
over a 500,000-image manifest, thus the median image takes about 2 comparisons
and a floor of 5 empties the case. The precision of an image row comes from the
number of images, not from the number of comparisons of 1 image.

**A thin cell stays in the frame.** The master table DROPS a rater under the
coverage floor. This table keeps it and marks it, because the unit layer is
where a reader asks what a heavy abstainer left behind. qwen3.5-9b abstains on
99.7% of the schools pairs and keeps 205 of the 1,136 schools that gemma-4-12b
keeps. Its coverage is 0.01, thus its row carries `thin_coverage = True` and
`area_dropped = True`.

## The interval

The interval is a bootstrap over UNITS, with 400 resamples and the seed
`20260831`. The unit is the sampling unit of the table, thus the resample draws
units and never comparisons. A resample over comparisons asks a different
question and gives a much narrower interval.

## The results

The canonical arm, 2026-08-31. The area value comes from the community
district layer.

| Case | Rater | Units | rho | tau-b | Areas | area tau-b | gap |
|------|-------|-------|-----|-------|-------|------------|-----|
| Restaurants | gemma-4-12b | 16,082 | -0.053 | -0.037 | 59 | +0.10 | +0.13 |
| Restaurants | qwen3.5-9b | 16,073 | -0.053 | -0.037 | 59 | +0.07 | +0.11 |
| Schools | gemma-4-12b | 1,136 | +0.169 | +0.116 | 59 | +0.14 | +0.02 |
| Schools | qwen3.5-9b | 205 | +0.261 | +0.181 | 53 | +0.04 | -0.14 |
| Parks | gemma-4-12b | 1,054 | -0.023 | -0.017 | 59 | +0.15 | +0.17 |
| Parks | qwen3.5-9b | 1,054 | +0.062 | +0.047 | 59 | +0.03 | -0.02 |
| Road quality | gemma-4-12b | 382,176 | +0.084 | +0.059 | 59 | -0.11 | -0.17 |
| Road quality | qwen3.5-9b | 382,235 | +0.045 | +0.031 | 59 | +0.09 | +0.06 |

`gap` is the area tau-b minus the unit tau-b. A large positive gap says that
the aggregation made the agreement.

3 readings stand out. Restaurants and parks show a positive area value over a
zero or negative unit value, which is the pattern the audit predicts. Schools
is the 1 case where the unit value HOLDS the area value, and it is the case
whose proxy grades the unit most directly. Road quality reverses: the unit
value is positive and the gemma area value is negative, thus the aggregation
destroys a real image-level signal there.

**Warning: the qwen3.5-9b schools row rests on 205 schools of a 99.7%
abstainer.** The schools it answered are a selection and not a sample. Do not
read that row as a rater comparison.

## Reconciliation

Parks and road already publish a unit correlation. `reconcile()` reads those
files and tests this module against them. Run on 2026-08-31:

| Case | Rater | Published n | n here | Published rho | rho here | Agrees |
|------|-------|-------------|--------|---------------|----------|--------|
| parks | gemma-4-12b | 1,054 | 1,054 | -0.0233 | -0.02327 | yes |
| parks | qwen3.5-9b | 1,054 | 1,054 | +0.0622 | +0.06221 | yes |
| road | gemma-4-12b | 382,176 | 382,176 | +0.0837 | +0.08371 | yes |
| road | qwen3.5-9b | 382,235 | 382,235 | +0.0454 | +0.04544 | yes |

The unit counts match exactly and every rho matches to 4 decimal places. The
area rows match `_results_table.build()` to the last digit as well, on all 8
cells.

Restaurants and schools publish no unit correlation. This module builds the
first one for them.

**Warning: the prose of the 2 case notebooks disagrees with their own
parquets.** The parks notebook text quotes n = 1,048 and rho = -0.005, and the
road notebook text quotes n = 127,883 and rho = +0.014. Those numbers come from
the earlier 110,000-pair battery. The parquets carry the 1,100,000-pair
battery, and this module reproduces the parquets.

## The subway seam

Subway safety pairs station ENTRANCES, and the outside measurement grades a
STATION. A station holds several entrances, thus a unit row needs 2 things that
do not exist yet:

1. A roll-up from an entrance to its station.
2. A station-level record to grade the station.

`CASE_SPEC` holds no `subway` entry until both land. `SUBWAY_SEAM` in the
module states the missing step. Add the entry with `mode="facility"` and
`join="station"` when the roll-up arrives.

## The API

| Call | What it gives |
|------|---------------|
| `place_scores(folder, model, arm)` | The mean score of each unit |
| `unit_proxy(folder)` | `proxy_key` and `proxy_value` of each unit |
| `unit_key(folder, scores)` | The map from a place to a proxy key |
| `joined(folder, model, arm)` | 1 row for each matched unit |
| `unit_row(folder, model, arm)` | Every number of 1 cell |
| `build(arm=...)` | The long frame, 1 row for each cell |
| `area_rows(layer)` | The area value of each cell |
| `with_area(long, layer)` | The 2 layers side by side |
| `reconcile(long)` | The test against the published parquets |
| `to_display`, `to_latex`, `export` | The renderers |

`arm` defaults to `canonical`. Pass `noabstain`, `nocity`, or `textfirst` to
read an ablation at the unit layer.

**Warning: a build reads the 1,100,000-row run parquets, and the road case runs
a spatial join over about 490,000 image points.** A full build of 4 cases and 2
raters takes about 2.5 minutes.

## The outputs

`export()` writes to `research/notebooks/cvpr/master/outputs/`:

| File | Holds |
|------|-------|
| `unit_validation_v1.0.0.csv` and `.parquet` | The unit rows |
| `unit_validation_community_district_v1.0.0_paired.csv` and `.parquet` | The unit rows with the area value |
| `unit_validation_community_district_v1.0.0.tex` | The paper table |
| `unit_validation_v1.0.0_reconcile.csv` | The reconciliation |

The unit rows carry no layer, thus their stem drops it. The paired rows carry
one, thus their stem holds it.
