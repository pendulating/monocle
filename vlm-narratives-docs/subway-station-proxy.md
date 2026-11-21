# The subway station proxy

`notebooks/cvpr/_proxies_subway.py`, notebook version 1.2.0 of
`notebooks/cvpr/subway/subway_safety_validation.py`, and the subway rows of
`notebooks/cvpr/_unit_validation.py`.

## The construct

The model answers "which station entrance is safer". Until export v1.1.0 the
case had 2 proxies, and both describe the AREA around an entrance: tract
income, and complaints for each km². Neither one measures the station. The
new headline proxy grades the station itself:

    transit crime for each 100,000 rider entries

The numerator counts the complaints with premise `TRANSIT - NYC SUBWAY`. The
denominator counts the rider entries of the same station complex over the
same months. The rate removes the exposure, thus it separates a busy station
from a dangerous one.

## The sources and the windows

| Source | Resource | Window | Rows kept |
|--------|----------|--------|-----------|
| NYPD complaints, historic | `qgea-i56i` | 2024-01 to 2025-12 | 59,555 |
| NYPD complaints, year to date | `5uac-w243` | 2026-01 to 2026-06 | the rest of 76,729 |
| MTA hourly ridership, 2020-2024 | `wujg-7c2s` | 2024-01 to 2024-12 | server-side sums |
| MTA hourly ridership, from 2025 | `5wq4-mkjj` | 2025-01 to 2026-06 | server-side sums |
| MTA subway stations | `39hk-dx4f` | current | 485 stations |

The premise filter and the ridership sums run ON THE SERVER. The hourly table
holds about 120,000,000 rows for 2020-2024 alone, and the network carries
only the grouped result. The pulls sit under `data/external/` with a
`pull.json` that records the query and the date.

**Warning: do not build this rate from the 2026Q1 complaint file on disk.**
That window gives the median complex 8 events, and a rate over 8 events is
noise. The 30-month window gives it 66.

## The roll-up and the guards

| Step | Rule |
|------|------|
| Entrance → station | `complex_id` where the entrance carries one, `station_id` where it does not. 445 keys. |
| Complaint → station | The nearest MTA station point in EPSG:2263, cap 1,000 ft. The cap keeps 99.0%. |
| Entrance scores → station score | The comparison-weighted mean. It equals the plain mean over the comparisons of the station. |
| Rate floor | A complex under 50,000 rider entries takes no rate. |

Two guards drop stations whose value would lie:

- **The 21 SIR complexes drop.** Neither source covers the Staten Island
  Railway, thus a SIR station would take a rate of 0 on an absence of data
  and lead the safety ranking.
- **5 named-but-unreached stations drop.** The record names them in
  `station_name` but geocodes their events to another station. Their 0 is a
  geocode error, not a quiet station. Roosevelt Island is the clear case.

The coverage: 445 station keys → 422 headline rows, and 2,051 of the 2,120
entrances reach a station that holds a rate.

## What it found (2026-08-31, canonical arm)

Unit level, 419 stations with a model score and a rate:

| Rater | n | Spearman rho [95% CI] | tau-b | Pearson r |
|-------|---|------------------------|-------|-----------|
| gemma-4-12b | 419 | +0.21 [+0.12, +0.29] | +0.14 | +0.12 |
| qwen3.5-9b | 419 | +0.29 [+0.20, +0.38] | +0.20 | +0.22 |

This is the strongest unit-level agreement of the battery. The look of an
entrance carries real information about the crime rate of its own station,
and both raters carry it.

Area level, community district:

| Proxy | gemma tau | qwen tau | n |
|-------|-----------|----------|---|
| transit_crime_per_rider | +0.23 | +0.28 | 54 |
| transit_felony_per_rider | +0.30 | +0.32 | 54 |
| median_household_income | +0.42 | +0.37 | 57 |
| crime_density (area) | −0.14 | −0.23 | 57 |
| subway_ridership (context) | +0.45 | +0.57 | 54 |

Read the contrast: the AREA crime row is negative and the STATION crime row
is positive. The old proxy said the model disagrees with the record. The
station proxy says the model agrees with the record about the station, and
the area row measured the busy blocks around it, without an exposure
denominator.

**Warning: do not read the ridership row as agreement.** It carries no
orientation. A busy station is not a better station. The row shows that both
raters call a busy station safer, which is context for the rate rows.

## The exports

| File | Holds |
|------|-------|
| `subway_proxy_v1.2.0.parquet` | The area export: 2 old proxies (rows unchanged from v1.1.0) + 3 station proxies, on all 3 layers, with the 4 provenance columns |
| `subway_station_proxy_v1.2.0.parquet` | The station table: 1 row for each station and metric |
| `subway_unit_correlations_v1.2.0.parquet` | The unit correlation of the headline rate, both raters |

`_unit_validation.reconcile()` tests the module against the unit export and
agrees to 4 decimals. The old-proxy rows of v1.2.0 are identical to v1.1.0
on every layer, less the 12 JIA keys that the layer filter now drops at read
time.

## Credentials

The 2 Socrata hosts take 2 keys, and `.env` holds both (`NYC_API_ID` /
`NYC_API_SKEY` for the city, `NY_STATE_API_ID` / `NY_STATE_API_SKEY` for the
state). Never write a key into the repository, a notebook cell, or an error
message.
