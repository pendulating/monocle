# The vintage proxies

The validation-by-proxy table asks whether a model score agrees with an outside
measurement. The vintage proxies ask a different question: **does the model
prefer a newer unit?** They enter the same table, at the same 3 geography
layers, but they carry no orientation.

| Case | Field | Source | Key | Units with a year |
|------|-------|--------|-----|-------------------|
| Libraries | `construction_year` | Building footprints, on disk | `bin` | 244 / 253 (96.4%) |
| Schools | `construction_year` | Building footprints, on disk | `bin` | 3,026 / 3,103 (97.5%) |
| Parks | `acquisitiondate` | DPR Parks Properties | `gispropnum` | 1,946 / 2,078 (93.6%) |
| Plazas | `year_completed` | POPS | `bbl` | 347 / 504 (68.8%) |

## Warning: a year is not a quality measure

Every other proxy arrives oriented "higher is better", thus a positive value
means agreement. A year does not measure quality, so **a positive value here
says only that the model calls a newer unit better**. State it as a finding.
Never read it as agreement. `_results_table.UNORIENTED_PROXIES` names the 3
keys.

## Why each case needs its own field

A building year fits a library and a school, because the unit **is** a building
and FacDB carries its BIN. The join needs no geometry.

It does not fit a park. Only 33% of the park units hold a BIN, and that BIN
belongs to a comfort station or a recreation building, never to the park. The
DPR acquisition date is the vintage that describes the site: the year the land
became a park. **Warning: acquisition is not construction.** A park that the
city acquired in 1936 can hold a playground of 2015.

A POPS plaza holds `year_completed`, which is the year of the space and not of
the tower beside it. The 92 DOT pedestrian plazas and the 20 city-state parks
carry no vintage in any NYC source, thus they drop out.

## The data on disk

| Path | Content |
|------|---------|
| `data/geo/nyc_buildings.parquet` | 1,082,872 footprints, pulled 2026-04-12 |
| `notebooks/cvpr/.proxy_cache/building_year_by_bin.parquet` | the slim BIN-to-year table |

Get the footprints again with `python data/geo/download_nyc_buildings.py`. The
source writes `0` for an unknown year, and `_proxies` drops it: a 0 that
reaches a correlation destroys it. A BIN carries more than 1 footprint row 13
times, and the module keeps the largest.

## The code

`notebooks/cvpr/_proxies.py`, the vintage section:

| Function | Purpose |
|----------|---------|
| `building_year_by_bin()` | The BIN-to-year table, with a disk cache |
| `building_vintage(facilities)` | A year for each library or school, by key |
| `park_vintage(facilities, units_gdf)` | A year for each park, by the DPR spatial join |
| `pops_vintage(facilities)` | A year for each POPS plaza, by BBL |
| `vintage_by_layer(vintage, layer)` | The mean year of each polygon |
| `vintage_coverage(facilities, vintage)` | The hit rate for each facility type |

Each case notebook adds its vintage rows to `proxy_agg`, and
`scripts/export_cvpr_results_table.py` puts them in the table. Run the 4 case
notebooks before the export script, or gate 2 stops it.

```bash
for nb in libraries schools parks plazas; do
  .venv-nightly/bin/python notebooks/cvpr/$nb/${nb}_validation.py
done
.venv-mllmsci-vllm025cu129/bin/python scripts/export_cvpr_results_table.py
```

## What the first run found

At the community-district layer, the vintage correlations are **negative** on
every case. The libraries row is the strongest: $r = -.50$ and $\tau = -.35$
with $p < .001$ for gemma-4-12b, and $-.42$ and $-.33$ for qwen3.5-9b. Both
models rate the areas with **older** library buildings higher.

The schools row is flat ($\tau = .01$), and qwen3.5-9b has no schools row at
all: it abstains on 99.7% of the schools pairs, thus 15 of 3,103 schools reach
5 comparisons and no polygon reaches the 3-unit floor.
