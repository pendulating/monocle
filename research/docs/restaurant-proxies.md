# The restaurants proxies

`research/notebooks/cvpr/_proxies_restaurants.py`,
`research/notebooks/cvpr/restaurants/restaurants_validation.py`,
`sv/nyc/curation/dohmh/history.py`.

Export version 1.2.0, 2026-08-31.

## What changed

Up to export version 1.1.0 the restaurants case had one proxy: `last_score`,
the violation points of the single most recent inspection. Version 1.2.0 adds
2 metrics and 1 channel:

| Proxy | Role | What it measures |
|-------|------|------------------|
| `inspection_score_history` | headline | The mean score over the cycle initial inspections |
| `inspection_score` | context | The score of the last inspection (the old headline) |
| `grade_a_rate` | context | The share of inspections that earned grade A |
| `outdoor_dining` | confounder | The restaurant holds a sidewalk or roadway dining license |

Every value leaves `_proxies_restaurants` oriented "higher is better". A DOHMH
score counts violation points, thus a LOW raw score is a CLEAN restaurant, and
the 2 score metrics carry a sign of -1. A positive correlation always means
that the model and the record agree.

## Two construct caveats. Read them first.

### 1. The prompt asks for a preference, not for food safety

The prompt is "Which restaurant would you rather eat at?". That is a
preference. The DOHMH record measures the food safety of the kitchen. The 2
constructs are not the same thing, and nothing makes them the same.

A person can prefer a restaurant with a poor inspection record. Price,
cuisine, a crowd on the sidewalk, and the look of the facade all enter the
preference, and none of them enter the score. Thus a weak correlation is a
statement about 2 different constructs, and not only a statement about the
model.

Warning: do not report a low value here as "the model fails to see food
safety". Report it as "the preference and the inspection record measure
different things, and the overlap is small and negative".

### 2. The grade card hangs in the window

The city requires each restaurant to display its letter grade in a window that
faces the street. The model looks at a street photograph of the same
storefront. The grade card is therefore inside the image, and the proxy can
leak into the input.

This cuts both ways:

- A positive correlation could come from the card and not from the facade.
- A negative correlation, which is what we measure, means that the model does
  not read the card, or reads it and does not use it.

The measured unit values are negative, thus the leak does not explain them.
State the leak anyway. It bounds what a future positive result could mean.

## The history metrics

### Why the inspection type matters

The city gives a re-inspection ONLY after an initial inspection fails. A
re-inspection score is thus conditional on an earlier failure. A mean over
mixed inspection types compares restaurants under different conditions, and
the comparison is not fair.

`inspection_score_history` reads one inspection type:
**`Cycle Inspection / Initial Inspection`**. That is the routine visit that
every operating restaurant receives on a cycle, and no earlier result
conditions it.

| Inspection type | Rows | In the mean |
|-----------------|-----:|-------------|
| Cycle Inspection / Initial Inspection | 45,679 | yes |
| Cycle Inspection / Re-inspection | 17,822 | no |
| Pre-permit (Operational) / Initial Inspection | 10,558 | no |
| Pre-permit (Operational) / Re-inspection | 3,232 | no |
| every other type | 5,826 | no |

A pre-permit inspection happens before the city issues the permit. It grades a
restaurant that does not operate yet, thus it also leaves the mean.

### The window

The pull of 2026-04-27 spans 2007-08-10 to 2026-04-25. That span is
misleading: the source publishes a rolling window, and 98.6% of the rows fall
on 2022-01-01 or later.

The metric applies a floor of **2019-01-01**. The floor drops 335 of the
45,679 cycle initial inspections (0.73%) and 36 of the 21,573 restaurants. A
pre-2019 inspection describes a restaurant before the pandemic, and often a
different operator.

The choice does not drive any number. The mean under the floor and the mean
over the whole span agree at Spearman rho = 0.999, and 127 of 21,573
restaurants take a different value. We keep the floor because it makes the
window honest, not because it moves a result.

The window of the export is therefore
**cycle initial inspections, 2019-01-05 to 2026-04-25**.

### The floor on the count

A mean over 1 inspection is the old one-shot measurement under a new name.
The metric needs **2 or more** cycle initial inspections. 16,271 of the 21,537
restaurants pass that floor, and the mean count is 2.46.

### How noisy was the old proxy

The sidecar carries the standard deviation inside each restaurant, so we can
split the variance of a score into a true part and a noise part.

| Quantity | Value |
|----------|------:|
| Variance inside one restaurant | 140.2 |
| Variance of the observed means | 102.0 |
| Implied variance between restaurants | 45.1 |
| Reliability of ONE inspection | **0.243** |
| Reliability of the mean of 2.46 inspections | **0.442** |

A single inspection is 76% noise. That is the audit's concern, and the number
confirms it. The mean of the history nearly doubles the reliability.

The 2 metrics also disagree at the unit level: the Spearman rho between
`last_score` and the history mean is only 0.502 over 16,119 restaurants.

## The outdoor-dining channel

### The source and the join

`sv/nyc/curation/open_restaurants` already curates the DCWP
"Dining Out NYC" licence dataset (`fpeh-f7ci`). The curated file holds 1,309
issued licenses: 656 Sidewalk and 653 Roadway, issued between 2024-10-09 and
2026-05-29 and pulled on 2026-06-01.

The licence table carries no CAMIS. It carries a BIN, and the unit table
carries the BIN of each restaurant, thus the join runs inside a building:

| Rule | Matched licenses |
|------|-----------------:|
| The same BIN and the same normalized trade name | 1,266 |
| The same BIN, where that BIN holds exactly 1 restaurant | 12 |
| No match | 36 |

A BIN alone is not enough. One building holds up to 18 restaurants in this
data, thus the name rule runs first and the BIN rule catches the rest.

1,273 licenses reach a restaurant, and **1,076 of the 26,973 restaurants
(4.0%)** carry the flag. 638 hold a sidewalk licence and 638 hold a roadway
licence.

### The unit value and the area share

`outdoor_dining` is 1 for a licensed restaurant and 0 for every other
restaurant in the unit table. The metric applies no inspection floor, because
the 0 rows carry the information.

The mean of that flag over an area IS the share of the restaurants of the area
that hold a licence. A frame of the matched restaurants alone would give a
share of 1.0 everywhere.

| Layer | Areas | Mean share | Max share | Areas at 0 |
|-------|------:|-----------:|----------:|-----------:|
| community_district | 59 | 0.025 | 0.139 | 14 |
| nta | 197 | 0.021 | 0.160 | 107 |
| census_tract | 1,615 | 0.023 | 0.444 | 1,261 |

### Warning: this is a confounder, not a validation

Outdoor seating stands on the street. The model sees it in the photograph, and
it plausibly drives the preference directly. A correlation with this channel
therefore does NOT validate the model against an outside record.

Read the row as a measurement of how much of the judgment one visible amenity
explains. The paper must not put it in the same column as an inspection
metric without that label.

## The numbers

### The unit layer, all 4 proxies and both raters

The unit is the restaurant. This case needs no join work: `uid` in the unit
table IS `unit_uid` in the pairs manifest. The place scores come from the
canonical registry, through `_unit_validation.place_scores`. The interval is a
95% bootstrap over 400 resamples of the UNITS.

| Model | Proxy | n | rho | 95% CI | tau | p(tau) |
|-------|-------|--:|----:|--------|----:|-------:|
| gemma-4-12b | inspection_score_history | 12,151 | **-0.067** | -0.086 to -0.049 | -0.045 | 2e-13 |
| gemma-4-12b | inspection_score | 16,082 | -0.053 | -0.069 to -0.038 | -0.037 | 2e-11 |
| gemma-4-12b | grade_a_rate | 16,326 | -0.050 | -0.067 to -0.036 | -0.036 | 1e-10 |
| gemma-4-12b | outdoor_dining | 18,488 | **+0.150** | +0.136 to +0.163 | +0.123 | 2e-92 |
| qwen3.5-9b | inspection_score_history | 12,142 | **-0.069** | -0.087 to -0.053 | -0.046 | 5e-14 |
| qwen3.5-9b | inspection_score | 16,073 | -0.053 | -0.069 to -0.038 | -0.037 | 2e-11 |
| qwen3.5-9b | grade_a_rate | 16,317 | -0.051 | -0.068 to -0.035 | -0.036 | 1e-10 |
| qwen3.5-9b | outdoor_dining | 18,477 | **+0.068** | +0.055 to +0.081 | +0.056 | 2e-20 |

The 3 metrics read different unit sets, thus the table below repeats them on
the 12,151 restaurants that the history mean keeps. The rulers then face the
same restaurants.

| Model | Proxy | n | rho |
|-------|-------|--:|----:|
| gemma-4-12b | inspection_score_history | 12,151 | -0.067 |
| gemma-4-12b | inspection_score | 12,033 | -0.062 |
| gemma-4-12b | grade_a_rate | 12,151 | -0.064 |
| gemma-4-12b | outdoor_dining | 12,151 | +0.163 |
| qwen3.5-9b | inspection_score_history | 12,142 | -0.069 |
| qwen3.5-9b | inspection_score | 12,024 | -0.060 |
| qwen3.5-9b | grade_a_rate | 12,142 | -0.066 |
| qwen3.5-9b | outdoor_dining | 12,142 | +0.076 |

### The answer to the audit question

The audit asked whether a less noisy proxy raises the unit-level agreement
from -0.05. It does not. The agreement stays negative and it moves a little
further from 0.

The reliability numbers explain why, and they make the result stronger. Divide
each rho by the square root of the reliability of its proxy, and the 2 metrics
give the same underlying value:

| Model | Proxy | rho | Reliability | rho corrected |
|-------|-------|----:|------------:|--------------:|
| gemma-4-12b | inspection_score | -0.053 | 0.243 | **-0.108** |
| gemma-4-12b | inspection_score_history | -0.067 | 0.442 | **-0.100** |
| qwen3.5-9b | inspection_score | -0.053 | 0.243 | **-0.108** |
| qwen3.5-9b | inspection_score_history | -0.069 | 0.442 | **-0.103** |

The 4 corrected values sit between -0.100 and -0.108. Thus the negative sign
is not an artifact of a noisy ruler. On a noise-free inspection ruler, the
preference of the model and the inspection record of the restaurant carry a
small, real, NEGATIVE relation of about rho = -0.10, and 2 different raters
give the same answer.

### The area layer, old against new

| Layer | Proxy | gemma-4-12b | qwen3.5-9b |
|-------|-------|------------:|-----------:|
| community_district (n=59) | inspection_score | +0.157 | +0.112 |
| community_district (n=59) | inspection_score_history | +0.066 | +0.119 |
| community_district (n=59) | grade_a_rate | +0.030 | +0.086 |
| community_district (n=59) | outdoor_dining | **+0.716** | **+0.371** |
| nta (n=197) | inspection_score | +0.059 | +0.062 |
| nta (n=197) | inspection_score_history | +0.093 | +0.214 |
| nta (n=197) | grade_a_rate | +0.024 | +0.121 |
| nta (n=197) | outdoor_dining | **+0.655** | **+0.370** |
| census_tract (n=1,316-1,615) | inspection_score | -0.060 | -0.013 |
| census_tract | inspection_score_history | -0.040 | -0.014 |
| census_tract | grade_a_rate | -0.034 | -0.006 |
| census_tract | outdoor_dining | **+0.421** | **+0.171** |

The values are Spearman rho. The archived version 1.0.0 file holds the same
`inspection_score` numbers on 201 NTAs and 60 districts, because that export
came before the JIA filter. Version 1.2.0 keeps 197 NTAs and 59 districts.

Two things stand out.

**The sign flips between the layers.** Every inspection metric is negative at
the unit layer and at the census tract layer, and positive at the district and
the NTA layers. That is the aggregation artifact that `unit-validation.md`
describes. Report the unit value.

**Only outdoor dining varies across areas.** The coefficient of variation of
each proxy over the 59 districts:

| Proxy | CV over districts |
|-------|------------------:|
| inspection_score | 0.074 |
| inspection_score_history | 0.079 |
| grade_a_rate | 0.070 |
| outdoor_dining | **1.325** |

The 3 inspection metrics are nearly flat over the districts, thus an area
correlation on them reads the few districts that move. Outdoor dining varies
18 times more. That, and not a better measurement, is why its area value
reaches +0.72.

## The files

| File | Holds |
|------|-------|
| `sv/nyc/curation/dohmh/history.py` | The sidecar builder |
| `curation/dohmh_restaurants_inspected_all/restaurants_inspection_history.parquet` | 21,537 rows, keyed by camis |
| `research/notebooks/cvpr/_proxies_restaurants.py` | The 4 metrics, the licence join, the unit table |
| `research/notebooks/cvpr/restaurants/recipe.json` | The 4 sources and their windows |
| `research/notebooks/cvpr/restaurants/outputs/restaurants_proxy_v1.2.0.parquet` | 7,082 area rows, 4 proxies |
| `.../restaurants_correlations_v1.2.0.parquet` | The area values above |
| `.../restaurants_unit_correlations_v1.2.0.parquet` | The unit values above |
| `.../restaurants_unit_correlations_common_v1.2.0.parquet` | The same, on one unit set |

### Why the sidecar and not more columns on the aggregate

`restaurants_aggregated.parquet` is the input of 3 artifacts that already
exist: `cyclomedia_near_restaurants.parquet`, the facing filter beside it, and
the pairs manifest of the restaurants case. A new column changes the schema of
that file and makes every consumer read it again.

We rebuilt the aggregate into a temporary path and compared it with the file on
disk. The 47 columns and the 26,973 rows are frame-equal, thus the aggregation
code carries no drift. We still chose the sidecar: it adds the same numbers
with no risk to bytes that 3 downstream artifacts describe.

Join the sidecar on `camis`. Fold it into `aggregate_restaurants` only when a
rebuild of the curation sub-dataset is scheduled.

### Build the sidecar again

```bash
.venv-nightly/bin/python -c "
from sv.nyc.curation.dohmh import build_inspection_history
print(build_inspection_history(
    'curation/dohmh_restaurants_inspected_all/restaurants.parquet',
    overwrite=True))"
```

Warning: this needs polars. `.venv-3.12` has no polars. Use `.venv-nightly` or
`.venv-vllm025cu129`.

## What the master table still needs

The export now carries 4 proxies, and `_results_table` groups by the `proxy`
column. Thus the restaurants case gives 4 rows for each rater instead of 1.
The following changes belong in files that this work did not touch:

1. `_results_table.PROXY_LABEL` holds no key for `inspection_score_history`,
   `outdoor_dining`, or `grade_a_rate`. The renderer prints the raw key today.
2. `_results_table.CASES` still carries `"single_proxy": "inspection_score"`
   for restaurants. That value is a fallback for an export with no `proxy`
   column, and the restaurants export has carried one since version 1.1.0.
   The value is dead, and it would mislabel all 4 proxies if a reader used it.
3. `outdoor_dining` needs a label of its own in the paper table, or the table
   needs a rule that keeps a confounder out of the validation column.
4. `_unit_validation.CASE_SPEC["restaurants"]["proxy"]` still names
   `inspection_score`. Point it at `inspection_score_history` when the paper
   moves to the new headline.

The new export changes no coverage value. `_results_table.coverage_by_cell`
gives the same number for all 14 cells under version 1.1.0 and version 1.2.0,
thus the thin-rater rule of the master table does not move.
