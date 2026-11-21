# Building-condition proxies for libraries and schools

`notebooks/cvpr/_proxies_buildings.py`,
`notebooks/cvpr/libraries/libraries_validation.py`,
`notebooks/cvpr/schools/schools_validation.py`.

Export version **1.2.0**, written 2026-08-31.

## The construct problem

The libraries prompt asks which library BUILDING looks better kept. The schools
prompt asks the same of a school. Until version 1.2.0 neither case validated
against a measurement of a building.

| Case | Proxy before 1.2.0 | What it graded |
|------|--------------------|----------------|
| Libraries | Median household income | The households near the library |
| Schools | Average student attendance | The students of the school |

Income describes an area. Attendance describes a student body. A correlation on
either row can only say that the look of a building tracks the wealth or the
outcomes around it. It cannot say that the model reads the building.

FacDB closes the gap. It gives a BIN to 96.8% of the libraries and to 99.0% of
the schools, thus the city building record reaches each unit by key and needs
no spatial join. The record grades the object the model looked at.

## The 3 building proxies

| Proxy | Source | Metric | Sign |
|-------|--------|--------|------|
| `building_violations_open` | 3h2n-5cm9 + 6bgk-3dad | Open DOB and ECB violations of the BIN | -1 |
| `building_violations_total` | 3h2n-5cm9 + 6bgk-3dad | All DOB and ECB violations of the BIN | -1 |
| `alteration_permits` | ipu4-2q9a | Distinct A1/A2/A3 jobs with a permit issued to the BIN | +1 |

The window is **2016-01-01 to 2026-08-31 (the pull date)** for all 3. One window
lets a reader compare them directly. A wider window would let a fault of 1990
count against a building that the city repaired in 2005.

A DOB violation is open when `violation_category` ends in `ACTIVE`. An ECB
violation is open when `ecb_violation_status` is `ACTIVE`.

The permit count is over the distinct DOB job number, not over the permit rows.
One job takes an initial permit, a renewal, and 1 permit for each trade, thus a
count of rows would call a long job a busy building. Job types A1, A2, and A3
only: a new building and a demolition are not investments in the building the
model looked at.

### Normalization

**A count is a raw count for 1 building, and the module divides it by nothing.**
The unit IS the building, thus the count already describes 1 object. A count for
each square foot would ask whether a building carries more faults than its size
predicts, which is a different question, and no city record says the model
should read size.

**Warning: read a correlation inside 1 case, never across the 2 cases.** A
school campus collects more violations than a branch library because it is
larger, not because it is worse.

**Warning: the violation row and the permit row are not independent.** Over the
raw counts they agree at Spearman 0.49 for the libraries and 0.47 for the
schools, because both rise when work happens on a building.

### Orientation

Every value leaves the module oriented "higher is better", the contract that
`_proxies` keeps. A violation count carries a sign of -1. Thus a positive
correlation always means that the model and the city record agree. Do not undo
the sign in a notebook.

## The schools construct decision

The task asked for a construct-valid schools metric. The 3 candidate sources
gave 1 usable building measurement.

### School Quality Reports (dnpx-dfnc): no building metric exists

The notebook dropdown showed 40 metrics because `list_metrics` takes a `top=40`
argument. The source holds **404 distinct metrics for school year 2024** and
**677 across 2015 to 2024**. A search of every name for the words facility,
building, clean, safe, environment, maintain, condition, physical, trust,
leader, family, collaborative, rigorous, supportive, survey, order, respect,
bully, violence, and climate returns **0 building metrics**. The words that do
match name a Regents subject, such as `Average Regents Score - Living
Environment/Biology`, or an IEP placement, such as `Movement of Students with
IEPs to Less Restrictive Environments`.

The metric families of the source are:

- Attendance, in total and by student group.
- State test proficiency in ELA and mathematics, by grade and student group.
- Core course pass rates.
- Regents scores and pass rates.
- ELL progress and IEP placement movement.

Thus **no facility-condition metric is obtainable from this source**, and
`school_attendance` stays as a context row.

### NYC School Survey: 1 item, and it stops in 2017-18

| Dataset | Year | Holds |
|---------|------|-------|
| `k2zg-756q`, `uzmq-2khh`, `4fcz-jias` | 2019 | The 6 framework scores only. No building item. The 3 files also return identical rows, which is a fault of the source. |
| `sjxh-har6` | 2017-18 | The item-level teacher file. It names item **6f. This school is kept clean.** |

`sjxh-har6` is the newest item-level file the city publishes. The proxy
`school_kept_clean` is the share of teachers who agree or strongly agree with
item 6f, from the sheet `Teacher Pos & Neg %`. The file covers 1,806 schools.

**Warning: the asset is an `.xlsx` blob and not a table.** SODA cannot query
it. `_proxies_buildings._download_survey_workbook` saves it to
`data/external/nyc_school_survey_2018/`, and the module reads the workbook XML
directly, because the project environments carry no `openpyxl`.

**Warning: the window is 2017-18, about 7 years before the model run.** State
the year in the paper.

### Economic Need Index: the better control

`c7ru-d68s` carries the Economic Need Index of each school and each year, and
2021-22 is the newest year it holds. The proxy `school_economic_need` negates
it, thus higher means a wealthier school community and the row reads the same
way as household income.

It beats tract income as the control of this case: a school draws its students
from a zone and not from its own block, and a charter or a specialized school
draws from the whole city. The index measures the students who attend.

**Warning: this is a control, not a building measure.**

### FISP facades: built, not exported

`xubg-57si` holds the Local Law 11 facade filings, and
`_proxies_buildings.facade_status_by_bin` reads them. The law applies to a
building of more than 6 storeys, and a library or a school is mostly low.

| Case | BINs | BINs with a filing | Share |
|------|------|--------------------|-------|
| Libraries | 236 | 28 | 11.9% |
| Schools | 2,375 | 189 | 8.0% |

A correlation over 8% of a case would describe the tall schools alone, thus
**no export row comes from this source**. The builder stays for a reader who
wants that arm.

## Join coverage

Measured 2026-08-31.

| Case | Units | With a BIN | With 1+ violation | With 1+ permit |
|------|-------|------------|-------------------|----------------|
| Libraries | 253 | 245 (96.8%) | 188 (74.3%) | 159 (62.9%) |
| Schools | 3,103 | 3,073 (99.0%) | 2,909 (93.8%) | 2,454 (79.1%) |

A BIN that the record never names takes 0, not a gap. An absent row in the city
record means "no fault was recorded", which is a measurement.

The DBN proxies cover less, and the reason is structural: a private, a
postgraduate, or a non-DOE school carries no DBN at all.

| Proxy | Rows | Matched to a DOE point | Share |
|-------|------|------------------------|-------|
| `school_kept_clean` | 1,806 | 1,731 | 95.8% |
| `school_economic_need` | 1,789 | 1,777 | 99.3% |

At the UNIT layer the DBN proxies reach the FacDB units through
`_proxies_buildings.school_unit_key`. The rule: each unit takes the value of
the nearest DBN point inside 200 ft, in the working CRS. The rule matches
1,834 of the 3,103 units to a DBN point (median distance 55 ft). State the
rule when you quote a number, because a different rule moves `n` by about 5%.

| Proxy | Proxy rows | Units with a value | Share of units |
|-------|------------|--------------------|----------------|
| `school_kept_clean` | 1,806 | 1,667 | 53.7% |
| `school_economic_need` | 1,789 | 1,699 | 54.8% |

## Unit-level numbers

The unit layer is the headline layer. See `unit-validation.md`. Each row reads
the model score of 1 building against the city record of the SAME building.
`n` counts the units that hold both. The brackets hold a 95% bootstrap interval
over 400 resamples of the units, seed 20260831. The notebooks compute these
rows with `_proxies_buildings.unit_correlation` and export them as
`{case}_unit_correlations_v1.2.0.parquet`.

### Libraries

| Model | Proxy | n | rho [95% CI] | tau-b | p |
|-------|-------|---|--------------|-------|---|
| gemma-4-12b | `building_violations_open` | 228 | -0.003 [-0.132, +0.128] | -0.004 | 0.97 |
| gemma-4-12b | `building_violations_total` | 228 | **-0.371** [-0.477, -0.254] | -0.261 | 8e-09 |
| gemma-4-12b | `alteration_permits` | 228 | **+0.267** [+0.142, +0.388] | +0.193 | 5e-05 |
| qwen3.5-9b | `building_violations_open` | 228 | -0.097 [-0.208, +0.036] | -0.073 | 0.14 |
| qwen3.5-9b | `building_violations_total` | 228 | **-0.477** [-0.573, -0.355] | -0.350 | 2e-14 |
| qwen3.5-9b | `alteration_permits` | 228 | **+0.375** [+0.251, +0.507] | +0.276 | 5e-09 |

### Schools

| Model | Proxy | n | rho [95% CI] | tau-b | p |
|-------|-------|---|--------------|-------|---|
| gemma-4-12b | `building_violations_open` | 2,267 | +0.010 [-0.032, +0.056] | +0.007 | 0.62 |
| gemma-4-12b | `building_violations_total` | 2,267 | -0.101 [-0.143, -0.056] | -0.069 | 1e-06 |
| gemma-4-12b | `alteration_permits` | 2,267 | +0.160 [+0.120, +0.202] | +0.115 | 2e-14 |
| gemma-4-12b | `school_kept_clean` | 1,120 | +0.019 [-0.042, +0.077] | +0.013 | 0.52 |
| gemma-4-12b | `school_economic_need` | 1,150 | +0.126 [+0.068, +0.186] | +0.085 | 2e-05 |
| qwen3.5-9b | `building_violations_open` | 369 | +0.294 [+0.193, +0.382] | +0.205 | 9e-09 |
| qwen3.5-9b | `building_violations_total` | 369 | +0.027 [-0.080, +0.126] | +0.018 | 0.61 |
| qwen3.5-9b | `alteration_permits` | 369 | +0.098 [-0.015, +0.205] | +0.069 | 0.06 |
| qwen3.5-9b | `school_kept_clean` | 182 | +0.136 [-0.006, +0.274] | +0.095 | 0.07 |
| qwen3.5-9b | `school_economic_need` | 191 | +0.194 [+0.064, +0.313] | +0.129 | 0.007 |

qwen3.5-9b scores 374 of the 3,103 schools, because it abstains on almost every
schools pair. `_results_table.drop_thin_models` already removes that rater from
the schools case of the master table.

### What the numbers say

The libraries case gives the clearest reading. `building_violations_total`
runs NEGATIVE at -0.37 and -0.48, and the proxy is already negated, thus **the
model calls a library with MORE recorded faults better looking**.
`alteration_permits` runs POSITIVE at +0.27 and +0.38. The 2 rows agree with
each other, because a building the city works on collects both permits and
violations. Read them as 1 signal: the model prefers a library where
construction happens.

The schools case gives the flattest reading. `school_kept_clean` is the one
true facility measurement of that case, and it sits at +0.02 with a p of 0.52
for gemma-4-12b. **The model and the teachers do not agree about which school
building is kept clean.**

## Area-level numbers

The community district layer, which is the master table default.

### Libraries

| Proxy | gemma-4-12b n | rho | qwen3.5-9b n | rho |
|-------|---------------|-----|--------------|-----|
| `building_violations_open` | 47 | -0.175 | 47 | -0.294 |
| `building_violations_total` | 47 | -0.570 | 47 | -0.623 |
| `alteration_permits` | 47 | +0.435 | 47 | +0.485 |
| `median_household_income` | 48 | +0.254 | 48 | +0.297 |
| `construction_year` | 47 | -0.570 | 47 | -0.602 |

### Schools

| Proxy | gemma-4-12b n | rho | qwen3.5-9b n | rho |
|-------|---------------|-----|--------------|-----|
| `building_violations_open` | 59 | +0.176 | 53 | +0.291 |
| `building_violations_total` | 59 | -0.112 | 53 | -0.034 |
| `alteration_permits` | 59 | +0.379 | 53 | +0.196 |
| `school_kept_clean` | 59 | +0.142 | 53 | +0.096 |
| `school_economic_need` | 59 | +0.278 | 53 | +0.264 |
| `school_attendance` | 59 | +0.197 | 53 | +0.047 |
| `construction_year` | 59 | -0.000 | 53 | +0.091 |

The libraries area value of `building_violations_total` is larger than its unit
value, and the schools area value of `school_kept_clean` is larger than its unit
value. That gap is the aggregation effect the unit layer exists to show.

## The libraries polygon floor moved from 1 to 3

`libraries_validation.py` read `MIN_UNITS_PER_POLYGON = 1` up to version 1.1.0,
thus most polygon values were 1 library and not an area mean.
`_proxy_arms.CASE_SPEC` already read 3 for the same case, so the 2 numbers
disagreed. Version 1.2.0 sets the notebook to 3.

Libraries for each polygon, measured 2026-08-31:

| Layer | Polygons with 1+ | Polygons with 3+ |
|-------|------------------|------------------|
| NTA | 166 | 16 |
| Community district | 59 | 49 |
| Census tract | 231 | 3 |

**The area numbers moved.** The master table at the community district layer,
before and after:

| Proxy | Model | n before | tau before | n after | tau after |
|-------|-------|----------|------------|---------|-----------|
| `construction_year` | gemma-4-12b | 59 | -0.344 | 47 | -0.402 |
| `construction_year` | qwen3.5-9b | 59 | -0.333 | 47 | -0.434 |
| `median_household_income` | gemma-4-12b | 59 | +0.102 | 48 | +0.188 |
| `median_household_income` | qwen3.5-9b | 59 | +0.143 | 48 | +0.199 |

Read the community district layer for this case. The NTA layer keeps 16 of 166
polygons and the census tract layer keeps 3 of 231. The unit layer carries the
case.

## The exports

| File | Rows | Change from 1.1.0 |
|------|------|-------------------|
| `libraries/outputs/libraries_proxy_v1.2.0.parquet` | 2,706 | 3 new proxies; the old rows move with the floor |
| `libraries/outputs/libraries_aggregated_v1.2.0.parquet` | — | The floor is 3 |
| `libraries/outputs/libraries_correlations_v1.2.0.parquet` | — | 5 proxies |
| `schools/outputs/schools_proxy_v1.2.0.parquet` | 3,922 | 5 new proxies; the old rows are unchanged |
| `schools/outputs/schools_aggregated_v1.2.0.parquet` | — | Unchanged rule |
| `schools/outputs/schools_correlations_v1.2.0.parquet` | — | 7 proxies |
| `libraries/outputs/libraries_unit_correlations_v1.2.0.parquet` | 6 | New: unit-level rho, tau-b, and CI |
| `schools/outputs/schools_unit_correlations_v1.2.0.parquet` | 10 | New: unit-level rho, tau-b, and CI |

Every proxy row carries `proxy`, `proxy_metric`, `proxy_source`, and
`proxy_window`, and no cell of those 4 columns is null.

The old proxy values are equal on every area that both versions hold.
The schools case lost 1 attendance NTA and 2 construction-year NTAs, and the
libraries case lost 3 income NTAs and 1 income community district. Every lost
area is a joint interest area that `_geography.excluded_keys` names, thus the
loss comes from the geography filter and not from this change.

## The pull

Every query filters on the BIN list of 1 case on the server. The violation
sources hold more than 2,000,000 rows each and the permit source holds more
than 4,000,000, thus a whole-table fetch would reach the row cap.

| Source | Query shape | Rows moved, libraries | Rows moved, schools |
|--------|-------------|-----------------------|---------------------|
| 3h2n-5cm9 | Grouped on the server by BIN and category | ~5 per BIN | 23,372 grouped down |
| 6bgk-3dad | Grouped on the server by BIN and status | ~2 per BIN | 38,822 grouped down |
| ipu4-2q9a | 3 columns, filtered by BIN and job type | ~3,000 | 125,720 |

The permit source cannot filter its window on the server: `issuance_date` holds
2 shapes, `MM/DD/YYYY` for most rows and an ISO timestamp for a few. The module
filters the year in pandas.

**The cache.** `_proxies.fetch` names its cache file from the first 120
characters of the query. A BIN `IN` list differs from the next batch only after
thousands of characters, thus a truncated name would let batch 2 read batch 1's
rows. `_proxies_buildings.fetch` therefore keeps its own cache and names each
file from a SHA-256 of the whole query.

## What a reader must still add

These changes live in files that this work does not own. See the hand-off
notes beside the code.

1. `_results_table.PROXY_LABEL` needs 5 keys, or the master table prints a raw
   proxy name. The reader already picks the 5 proxies up; only the label is
   raw. The keys and the labels:

   | Key | Label |
   |-----|-------|
   | `building_violations_open` | Open building violations (negated) |
   | `building_violations_total` | Building violations (negated) |
   | `alteration_permits` | Alteration permits |
   | `school_kept_clean` | School kept clean (teacher survey) |
   | `school_economic_need` | Economic need index (negated) |

2. `_unit_validation.CASE_SPEC` and `_unit_validation.FOLDERS` need a
   `libraries` entry, or the module's unit view skips the case. The schools
   entry is correct: it names `school_attendance`. The unit numbers of the 5
   new proxies do not wait for that module — the case notebooks export them as
   `{case}_unit_correlations_v1.2.0.parquet`.
