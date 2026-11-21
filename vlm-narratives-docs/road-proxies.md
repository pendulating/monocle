# The road-quality proxies: the denominator and the demand side

Date: 2026-08-31. This is upgrade U5 of the validation-by-proxy road.

New module: `notebooks/cvpr/_proxies_road.py`. New export:
`notebooks/cvpr/road/outputs/road_quality_proxy_v1.2.0.parquet`. New cache:
`data/external/nyc_street_centerline/`.

`notebooks/cvpr/_proxies.py` does not change. The old builders stay where they
are, thus every published number is still reproducible.

## The 3 problems

| # | Problem | Fix |
|---|---------|-----|
| 1 | `pothole_repairs` divides repairs by km^2 | Divide by the centerline mile |
| 2 | `dot_pavement_rating` takes a plain segment mean | Weight each rating by segment length |
| 3 | The case reads the repair and never the report | Add the 311 complaints |

A 4th proxy joins the case: the years since the last DOT resurfacing.

## The denominator argument

A pothole happens on a roadway, not on land.

A community district holds land, and the roadway inside it is a small part of
that land. A dense street grid packs more roadway into the same square
kilometre than a district of large blocks does. Thus a repair count for each
km^2 rises with street density, and the reader cannot tell that rise apart from
bad pavement.

The measurement confirms it. At the community district layer the old rate
correlates with street density at Spearman **-0.602**, and the corrected rate
at **-0.340**. Street density alone reaches Kendall tau **-0.230** against the
gemma-4-12b score, which is 83% of the size of the whole old pothole row.

The rate needs the roadway itself in the denominator. This upgrade measures the
centerline miles of every polygon and divides by them.

## The length argument

The DOT pavement rating scores a street segment from 0 to 10. A plain mean over
segments gives a 60 ft block and a 1 mi avenue the same voice. A driver meets
the avenue for a mile and the block for 3 seconds.

The corrected mean weights each rating by the length of its own segment. The
segment filters do not change: `_proxies.pavement_segments` still drops
`systemrating = 0.0`, which means NOT RATED, and it still reads the same
window.

The weight moves the number very little. The old and the new area means agree
at Spearman 0.973, and the mean community district moves 0.11 rating points.
The largest move is 0.375 points. The correction is right, but no reading of
the case turns on it.

## Supply and demand

The case read 1 side of the pothole story. The 2 sides are:

| Side | Source | Counts | Warning |
|------|--------|--------|---------|
| Supply | Closed work orders, `x9wy-ing4` | The city repair | A fast borough looks worse |
| Demand | 311 complaints, `erm2-nwe9` | The resident report | A quiet borough looks better |

Neither source counts potholes. Read the 2 together, and read both beside the
DOT pavement rating, which is the only source that scores the pavement itself.

The 2 rates agree at Spearman **0.819** over the 59 community districts, thus
they are not the same measurement. The demand side tracks the model a little
better than the supply side on both raters.

The city record holds 147,824 complaints and 49,589 closed work orders in
almost the same window. Do not read that ratio as a repair backlog. One work
order can answer many complaints, and the 2 sources open a row under different
rules.

## The street centerline

Source: NYC Street Centerline (CSCL), `inkn-q76z`.

| Item | Value |
|------|-------|
| Filter | `rw_type in (1,2) AND status='2'` |
| Segments | 103,429 of 122,263 |
| Centerline miles | 6,838.2 |
| Street (`rw_type` 1) | 6,438.0 miles |
| Highway (`rw_type` 2) | 400.2 miles |
| Working CRS | EPSG:2263 |

NYC DOT publishes about 6,300 centerline miles of street, thus the total is
right. The excluded types are a bridge, a tunnel, a boardwalk, a path, a step
street, a driveway, a ramp, an alley, a non-physical segment, a U-turn, and a
ferry route. A pothole work order and a 311 pothole complaint both name a
street or a highway, thus the numerator and the denominator match.

Each line is CUT at the polygon border. A segment that crosses 2 areas gives
each one its true part, thus a sum over 1 layer returns the roadway of the
city.

| Layer | Areas | Centerline miles |
|-------|-------|------------------|
| Community district | 59 | 6,612.7 |
| NTA | 197 | 6,429.7 |
| Census tract | 2,321 | 6,836.2 |

The first 2 layers sit under the city total because `_geography` drops the
areas that hold no residents. The census tract layer keeps them, thus it
returns almost the whole city.

### The cache that other cases can read

| File | Holds |
|------|-------|
| `data/external/nyc_street_centerline/centerline_miles_by_layer.parquet` | `layer`, `area_key`, `centerline_miles`, `n_segments` |
| `data/external/nyc_street_centerline/cscl_roadway_lines_2263.parquet` | The 103,429 lines, in EPSG:2263 |
| `data/external/nyc_street_centerline/centerline_pull.json` | The filter, the counts, and the pull date |

The frame is long, because the 3 layers name their key differently. Call
`_proxies_road.miles_for_layer(layer)` to get 1 layer with its own key column.
Any case that needs a roadway denominator reads this file and pulls nothing.

## The sources and their windows

| Source | Id | Rows the pull moves | Window |
|--------|-----|--------------------|--------|
| Street Pavement Ratings | `6yyb-pb25` | 125,924 rated segments | 2024-01 to 2026-07 |
| Pothole work orders, closed | `x9wy-ing4` | 49,589 orders | 2024-01-01 to 2026-07-30 |
| 311 pothole complaints | `erm2-nwe9` | 147,824 found, 68,108 located | 2024-01-01 to 2026-08-29 |
| DOT in-house resurfacing | `ffaf-8mrv` | 252,054 jobs on 92,087 blocks | 2002-03-25 to 2026-08-14 |
| Street centerline (CSCL) | `inkn-q76z` | 103,429 segments | pulled 2026-08-31 |

Every pull filters on the server. 311 holds more than 40,000,000 rows, thus a
pull without `$where` and `$select` is forbidden.

### Warning: 54% of the 311 rows carry no coordinate

The filter finds 147,824 pothole complaints, and 68,108 of them carry a
latitude. The located share differs by borough: Queens 42%, Bronx 43%, Staten
Island 49%, Brooklyn 51%, Manhattan 54%.

The `community_board` column is present on every row, so the whole set gives a
true count at the community district layer. The located rate and the whole-set
rate rank the 59 districts at **Spearman 0.92**. Thus the located subset is
safe for an ORDER and it is not safe for a LEVEL. The export builds on the
located subset, because the NTA layer and the tract layer need a point.

### Warning: 2026-03 holds 5 times the normal month

The month of 2026-03 gives 24,379 complaints against about 4,500 in the same
month of 2025. The rise appears in all 5 boroughs, thus it is a pothole season
and not a data artifact.

### Warning: the resurfacing age is right-censored

`ffaf-8mrv` holds the IN-HOUSE work of DOT only, from 2002. A block that a
capital contract rebuilt is absent, and a block that DOT never paved since 2002
is absent. The proxy reads "how long since DOT last paved this block", never
"how old the pavement is". 92,087 blocks carry a date, against 103,429
centerline segments, thus the record reaches most of the city.

## Old against new

The community district layer, 59 areas, the 2 canonical raters. Kendall tau-b,
the measure the master table prints. Measured 2026-08-31.

| Proxy | Transform | gemma-4-12b | qwen3.5-9b |
|-------|-----------|-------------|------------|
| Pothole repairs | Old, for each km^2 | +0.276 | -0.002 |
| Pothole repairs | **New, for each mile** | **+0.232** | **-0.020** |
| DOT pavement rating | Old, plain mean | -0.107 | +0.089 |
| DOT pavement rating | **New, length-weighted** | **-0.091** | **+0.057** |
| 311 pothole complaints | **New, for each mile** | **+0.248** | **+0.066** |
| Years since resurfacing | **New, length-weighted** | **-0.032** | **+0.094** |
| Street density | The denominator alone | -0.230 | -0.010 |

The same table with Spearman rho:

| Proxy | Transform | gemma-4-12b | qwen3.5-9b |
|-------|-----------|-------------|------------|
| Pothole repairs | Old, for each km^2 | +0.412 | +0.019 |
| Pothole repairs | New, for each mile | +0.340 | -0.032 |
| DOT pavement rating | Old, plain mean | -0.153 | +0.142 |
| DOT pavement rating | New, length-weighted | -0.129 | +0.103 |
| 311 pothole complaints | New, for each mile | +0.368 | +0.112 |
| Years since resurfacing | New, length-weighted | -0.047 | +0.132 |
| Street density | The denominator alone | -0.334 | -0.005 |

The fall holds at the other 2 layers, so it is not an artifact of 1 geography.
Kendall tau against the gemma-4-12b score:

| Proxy | Census tract | Community district | NTA |
|-------|--------------|--------------------|-----|
| Pothole repairs, old | +0.220 | +0.276 | +0.310 |
| Pothole repairs, new | +0.192 | +0.232 | +0.274 |
| Pavement rating, old | +0.061 | -0.107 | +0.001 |
| Pavement rating, new | +0.055 | -0.091 | -0.007 |
| Complaints for each mile | +0.182 | +0.248 | +0.269 |
| Street density | -0.105 | -0.230 | -0.178 |

### What the table says

1. **The audit was right about the denominator.** The land denominator carries
   street density into the proxy. The correction cuts the gemma-4-12b tau at
   every layer, by 0.028 at the tract, 0.044 at the district, and 0.036 at the
   NTA. It removes a measure the case never meant to take.

2. **The old and the new rate still agree.** Spearman 0.941 over the 59
   districts. The denominator changes the size of the result and not its sign,
   thus no published reading of this case turns over.

3. **The length weight changes nothing.** Spearman 0.973 between the 2 pavement
   means. Take the correction because it is right, not because it moves a
   number.

4. **The demand side is the better proxy of the 2 pothole rates.** Complaints
   for each mile reach +0.248 and +0.066, above the repair rate on both
   raters. The demand side is also the only pothole row where qwen3.5-9b
   leaves zero.

5. **No pothole rate tracks the pavement rating.** The corrected repair rate
   and the length-weighted rating agree at Spearman -0.05, and the complaint
   rate at +0.12. A pothole count and a pavement score measure different things
   at the area level, so the paper must not call them 1 proxy.

6. **The resurfacing age is a null.** -0.032 and +0.094. The model does not
   read the DOT paving cycle, and the censoring of the source is a fair reason
   to expect that.

## The image level does not move

The image comparison of section 4 gives each image the rating of its nearest
rated segment, inside 100 ft. The length weight enters an AREA mean only, thus
the image join sees nothing of this upgrade.

I checked it on 2026-08-31. Section 4 ran again with `_proxies_road` loaded and
returned the shipped table exactly:

| Model | n | Pearson r | Spearman rho |
|-------|---|-----------|--------------|
| gemma-4-12b | 382,176 | +0.0828 | +0.0837 |
| qwen3.5-9b | 382,235 | +0.0519 | +0.0454 |

Both rows match `road_quality_unit_correlations_v1.0.0.parquet` to 4 decimal
places. The upgrade writes no new image table.

## The export

`road_quality_proxy_v1.2.0.parquet`, 20,445 rows, the same 10 columns that
version 1.1.0 has.

| Block | Rows | Comes from |
|-------|------|------------|
| Version 1.1.0, whole | 10,270 | Copied, row for row |
| The 4 new proxies | 10,175 | `_proxies_road.area_export` |

The old rows do not change. A reader of version 1.2.0 can reconcile every
published number, and `pothole_repairs` for each km^2 stays as the context of
`pothole_repairs_per_mile`.

`merge_export` raises an error when a proxy name appears on both sides. A
duplicate name would give a reader 2 values for 1 polygon, thus each corrected
proxy takes a name of its own.

| `proxy` | `proxy_metric` | `proxy_source` | `proxy_window` |
|---------|----------------|----------------|----------------|
| `dot_pavement_rating_weighted` | systemrating, weighted by segment length | `6yyb-pb25` | 2024-01 to 2026-07 inspections |
| `pothole_repairs_per_mile` | closed work orders for each centerline mile | `x9wy-ing4 + inkn-q76z` | 2024-01 to 2026-07 work orders |
| `pothole_complaints_per_mile` | 311 complaints for each centerline mile | `erm2-nwe9 + inkn-q76z` | 2024-01 to 2026-08 complaints |
| `resurfacing_age` | years since the last in-house paving, length-weighted | `ffaf-8mrv` | 2002-03 to 2026-08 paving, age as of 2026-08-14 |

A rate reads 2 sources, thus `proxy_source` names both: the count, and the
centerline that makes the denominator.

The notebook writes a second file,
`road_quality_denominator_comparison_v1.2.0.parquet`. It holds the old-against-
new table of all 3 layers, so a reader can rebuild the tables above.

### The model-side exports do not move

This upgrade touched the proxy side only. The export cell now writes the
model-side tables at `MODEL_EXPORT_VERSION` and it SKIPS a file that already
exists. Thus a run of the notebook cannot quietly replace a shipped model
score.

## What the master table still needs

`_results_table.PROXY_LABEL` holds no name for the 4 new proxies, thus the
table prints the raw key. `_results_table.py` belongs to the master table and
this upgrade does not own it. Add:

```python
    "dot_pavement_rating_weighted": "DOT pavement rating (length-weighted)",
    "pothole_repairs_per_mile": "Pothole repairs per mile (negated)",
    "pothole_complaints_per_mile": "Pothole complaints per mile (negated)",
    "resurfacing_age": "Years since resurfacing (negated)",
```

The road case now gives the table 8 proxy rows for each rater. The 3 old rows
that a corrected row replaces are context, not results. A reader of the paper
needs a rule that prints the corrected row and holds the old one back. That
rule belongs in `_results_table.py`, thus this upgrade names it and does not
write it.

## The files

| File | Role |
|------|------|
| `notebooks/cvpr/_proxies_road.py` | The corrected builders and the export helpers |
| `notebooks/cvpr/road/road_quality_validation.py` | Section 6, and the export cell |
| `notebooks/cvpr/road/recipe.json` | 7 sources: 2 more proxies, and the denominator |
| `notebooks/cvpr/road/outputs/road_quality_proxy_v1.2.0.parquet` | The export |
| `notebooks/cvpr/road/outputs/road_quality_denominator_comparison_v1.2.0.parquet` | Old against new |
| `data/external/nyc_street_centerline/` | The centerline pull and the miles cache |
