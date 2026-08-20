# CVPR notebooks

These marimo notebooks are load-bearing for the CVPR paper. They come in 2
kinds:

| Kind | Notebook | Question it answers |
|------|----------|---------------------|
| Validation by proxy | `<case>_validation.py` | Does the model score agree with an outside measurement? |
| Trace analysis | `<case>_traces.py` | What does the model say while it decides? |

Every figure and every table comes from the **canonical run registry**. Read
the next section before you run anything.

**Warning:** Run these notebooks from the canonical venv
`.venv-mllmsci-vllm025cu129`. It has marimo, geopandas, wordcloud, and
scikit-learn. `.venv-3.12` has no marimo. `.venv-nightly` still works but it is
**deprecated** (2026-08-12).

```bash
.venv-mllmsci-vllm025cu129/bin/marimo edit notebooks/cvpr/schools/schools_validation.py
```

## The canonical run registry — read this first

A notebook does **not** choose its own runs. The battery has run 3 times with 3
different prompts, and W&B holds all of them, thus a query to the network can
return a run that answers a question the paper no longer asks. The registry
fixes 1 run for each cell of the grid:

| Kind | Cases | Models | What reads it |
|------|-------|--------|---------------|
| `proxy` | 7 | 2 | the validation notebooks and the results table |
| `trace` | 7 | 2 | the word figures and the extraction panels |

```
notebooks/cvpr/canonical_data/
├── manifest.json                     <- the registry: 28 runs, with hashes
├── proxy/<case>__<model>/
│   ├── results.parquet -> the run's labels
│   ├── pairs.parquet   -> the pair manifest
│   └── stage           -> the run directory, for the Hydra config and the logs
└── trace/<case>__<model>/  ... the same 3 links
```

### The commands

```bash
# after a new battery lands
python scripts/register_canonical_runs.py register --stage-root '<glob>'
python scripts/register_canonical_runs.py verify     # the gate
python scripts/register_canonical_runs.py show       # what is registered

# then the exports, in this order
marimo export html notebooks/cvpr/<case>/<case>_validation.py -o /dev/null
python scripts/export_cvpr_results_table.py          # the LaTeX table
python scripts/export_cvpr_trace_figures.py          # the word figures
```

### What the gate tests

`register` refuses a run that fails any of these, and `verify` tests them again
against the disk:

- Each link resolves, and the file has the same size and SHA-256 as on the day
  of registration.
- The grid is complete: 7 cases x 2 models x 2 kinds.
- The 2 models of a case asked the **same question**.
- A gemma-4-12b run used `image_layout=interleaved_labels`. Without the anchor
  that arch does not bind the second image.
- No run is degenerate, which means 1 label above 98% of the rows.
- A trace run carries a trace on 95% of its rows.
- The results parquet joins 1-to-1 to `pairs.parquet`.

A problem that you accept stays in the manifest and prints as a warning on
every `verify`. Nothing is accepted in silence.

### The 2 downstream gates

- `export_cvpr_results_table.py` refuses when a case export is OLDER than the
  registry. An older export came from other runs.
- `_extractions.load()` refuses an extraction corpus whose `source_results_path`
  is not a registered trace run. The word figures then hold the words alone,
  and the panels are skipped with a message.
- `_ic.load()` applies the same test to the Integrative Complexity corpus, and
  `scripts/merge_trace_extractions.py` reports it when it writes a corpus.

### To use another source

`_provenance.discover_runs(..., source="wandb")` still queries the network. Use
it to FIND a run that you then register. No paper figure may use it.
`CVPR_ALLOW_UNVERIFIED=1` turns the gate off for a debug session only.

## Layout — 1 folder for each prompt

**Rule: everything for a prompt lives inside that prompt's folder.** The
notebooks, the recipe, the exported tables, and the figures. Nothing that
belongs to 1 prompt sits at the top level. Only the shared modules and this
file do.

```
notebooks/cvpr/
├── _provenance.py  _geography.py  _proxies.py    <- shared, every prompt uses them
├── _traces.py  _trace_notebook.py
├── _gen_trace_notebooks.py                       <- writes each <case>_traces.py
├── README.md
└── road/                                         <- 1 prompt
    ├── recipe.json                               <- its proxy sources
    ├── road_quality_validation.py                <- validation by proxy
    ├── road_quality_traces.py                    <- reasoning traces
    ├── outputs/                                  <- its tables
    └── figures/                                  <- its PNG files and word tables
```

The prompt folders are `road`, `subway`, `schools`, `libraries`,
`restaurants`, `parks`, `plazas`, and `street_photography`.

`parks` and `plazas` split 1 case into 2 proxies, thus the trace figures of that
case live in `parks/` alone. `street_photography` has no outside measurement,
thus it holds figures and no recipe.

`master/` is the 1 folder that is NOT a prompt. It holds the notebooks that
read across every prompt, and it keeps its own `outputs/`. A cross-prompt
notebook goes there, never at the top level.

**Warning: a folder name is not always the case name.** The recipe directories
were named before the cases were, thus the `road_quality` case lives in `road/`
and the `subway_safety` case lives in `subway/`. `_gen_trace_notebooks.py`
holds the map.

### Add a prompt

1. Make the folder and put `recipe.json` in it.
2. Copy the nearest `<case>_validation.py` into it. Set `CASE`,
   `OUTPUT_PREFIX`, and `RECIPE_DIR`.
3. Add the case to `_gen_trace_notebooks.py`, then run that script.

A notebook resolves its own paths, thus it needs no edit to find its folder:
`Path(__file__).parent / "outputs"` for a table, and
`_trace_notebook.figures_dir(__file__)` for a figure.

**Warning: import the shared modules from the parent.** A notebook sits 1 level
down, thus it must put `Path(__file__).resolve().parent.parent` on `sys.path`,
never its own directory.

## Files

| File | Purpose |
|------|---------|
| `_canonical.py` | The canonical run registry: the ground truth for every figure |
| `canonical_data/manifest.json` | The 28 registered runs, with their hashes |
| `_provenance.py` | Run discovery, from the registry by default and from W&B on request |
| `_geography.py` | Aggregation into the 3 NYC geographies |
| `_proxies.py` | Recipe reader, SODA client, and the proxy comparison |
| `_traces.py` | Trace discovery, word counts, and the distinctive-word score |
| `_extractions.py` | Typed extractions over the traces: rates, the risk panel, and the distinctive-claim score |
| `_ic.py` | The Integrative Complexity report: corpus, codes, and figures |
| `_ic_words.py` | Word blocks over the IC spans, 1 for each ingredient type |
| `_cuisine.py` | The restaurants ranking by cuisine, and its 3 bands |
| `_maps.py` | Choropleths of a case on 1 layer, and the small-multiples pair |
| `_ic_link.py` | Does a complex trace agree with the proxy more often? |
| `master/ic_linking.py` | The IC-to-proxy link notebook |
| `master/ic_complexity.py` | The IC notebook over every case |
| `_style.py` | The house palette and the camera-ready figure style |
| `_results_table.py` | The 3 metrics and the camera-ready results table |
| `master/results_table.py` | The table over every prompt |
| `_trace_notebook.py` | The shared body of a per-prompt trace notebook |
| `_gen_trace_notebooks.py` | Writes each `<case>_traces.py` from 1 template |
| `scripts/register_canonical_runs.py` | Registers the runs and holds the gate |
| `scripts/export_cvpr_results_table.py` | Writes the LaTeX table without marimo |
| `scripts/export_cvpr_trace_figures.py` | Writes every word figure without marimo |
| `scripts/export_cvpr_ic_figures.py` | Writes the IC tables and figures without marimo |
| `scripts/launch_ic_corpus.sh` | Submits the IC extraction over the registered trace runs |
| `<prompt>/recipe.json` | The proxy sources for that prompt |
| `<prompt>/<case>_validation.py` | Validation by proxy |
| `<prompt>/<case>_traces.py` | The reasoning-trace report |
| `restaurants/restaurants_cuisine.py` | Which cuisines the model chooses |
| `street_photography/street_photography_map.py` | The score against income, mapped |
| `<prompt>/outputs/` | The exported tables. Give the paper all of them together. |
| `<prompt>/figures/` | The exported PNG files and word tables |

## The master results table

`master/results_table.py` builds 1 camera-ready table over every prompt: the
case at the first indent, its proxies under it, and a column block for each
canonical model.

It is **downstream**. It reads `<prompt>/outputs/*_aggregated_*.parquet` and
`*_proxy_*.parquet`, and it touches neither W&B nor a results parquet. Run the
validation notebooks first, then this one. A prompt that has not exported is
named in the report rather than dropped in silence.

| Column | Meaning | Chance |
|--------|---------|--------|
| agr. | Share of units on the same side of BOTH medians | 0.50 |
| `r` | Pearson correlation | 0 |
| tau | Kendall tau-b | 0 |

**agr. and tau answer different questions.** Agreement splits each series at
its own median, thus it needs no shared scale and it survives any monotone
distortion of either one — but it throws away the size of a gap. Kendall tau
counts concordant pairs, thus it reads the whole ordering: tau = 0.2 means a
random pair is ordered the same way 60% of the time. A high agreement beside a
low tau means the model finds the good half but cannot rank inside it.

**Read `r` beside tau, never alone.** Pearson assumes a linear relation, and
income is right-skewed.

### Warning: never flip a sign in the table

Every proxy arrives oriented "higher is better". `_proxies` applies the sign
BEFORE it exports: crime density and pothole repairs are already negated in the
parquet, and the restaurant inspection score is already flipped. Thus a
positive number always means agreement, and `_results_table.py` applies no sign
of its own.

A negative crime row is a real disagreement, not a sign artefact. A flip there
would hide it.

## The procedure

Follow these 9 steps for every case. The steps keep the cases comparable.

1. Read the canonical runs of the case from the registry (`_canonical.py`). Do
   not name a run directory in a notebook, and do not query W&B for a figure.
2. Record the provenance of each run.
3. Join the results parquet to `pairs.parquet` on `pair_id`.
4. Score each unit from the pairwise labels.
5. Join the units to their FacDB coordinates.
6. Aggregate to the 3 geographies.
7. Read `<case>/recipe.json` and get the proxy.
8. Join the proxy to the same units, and report the match rate.
9. Aggregate the proxy to the same 3 geographies, then correlate.

## Proxies and `recipe.json`

Each case directory holds a `recipe.json`. It names the NYC Open Data sources
for the proxy. Add a source with the next index.

```json
{
  "0": {
    "name": "School Quality Reports Data",
    "endpoint": "https://data.cityofnewyork.us/api/v3/views/dnpx-dfnc/query.json",
    "documentation": "https://dev.socrata.com/foundry/..."
  }
}
```

`_proxies.py` reads the Socrata id out of `endpoint`. These keys are optional:
`resource_id`, `select`, `where`, and `group`.

A recipe holds 2 kinds of source. The `role` key separates them.

| `role` | Purpose | Keys |
|--------|---------|------|
| `proxy` | The measurement to compare against | `endpoint`, `key` |
| `geocode` | Gives each proxy key a position | `file`, `key`, `joins_to` |

A `geocode` entry names a local file in the case directory. The schools case
uses the DOE School Point Locations shapefile, where the `ATS` column holds the
DBN.

**Prefer a key join over a name join.** A key join cannot make a false match. A
name join can, because 2 schools can share a name inside a borough. For the
schools case the DBN join reaches 99.9% of proxy schools, and the name join
reaches 97.9%. The key join also puts the proxy on the map without any reference
to the model unit set, so a bad name match cannot move a school.

### Credentials

**Warning: never write a credential into this repository.** The SODA key lives
in `.env` at the repository root. Git ignores `.env`. The code names 2
environment variables and never holds a value:

- `NYC_API_ID` — the key id
- `NYC_API_SKEY` — the key secret

Do not print them. Do not put them in a notebook cell, a log line, or an error
message. `_proxies.has_credentials()` reports presence only.

### Warning: filter on the server

The School Quality Reports source holds more than 1,000,000 rows. A whole-table
fetch reaches the row cap and returns a **wrong answer that looks correct** — cut
at 1,000,000 rows, `Average Student Attendance` returns 3 schools, not 1,841.

`fetch` raises an error when it reaches the cap. Use `fetch_metric`, which sends
the metric and the year to the server.

### Warning: full coverage is not possible

The proxy covers public schools. The model unit set comes from FacDB, which also
holds private, charter, and postgraduate schools. Those have no DBN, so no proxy
row can ever describe them.

The 2 sets describe different school populations. Thus compare them at the
geography level, not school by school. The notebook prints the match rate so the
gap stays visible.

## What makes a run canonical

A run must meet all 3 conditions. `_provenance.RunRecord.is_canonical` tests
them.

| Condition | Value |
|-----------|-------|
| Model | `qwen3.5-9b/instruct` or `gemma-4-12b/instruct`, exactly |
| Start date | On or after 2026-08-11 |
| Sweep | `canonical_qwen9b` or `canonical_gemma12b` |

**Warning:** Match the model string exactly. A substring test accepts
`gemma-4-12b/instruct_thinking`, which uses different sampling parameters. Its
labels are not comparable.

**Warning:** The date test is not decoration. The battery changed on 2026-08-11
to 7 cases, a minimal prompt with no persona, abstention always on, and greedy
decoding. An earlier run used a different prompt. Do not mix the two. A search
of the schools case without these tests returns 18 runs. Only 2 are canonical.

### Warning: the schools prompt changed again on 2026-08-13

`CONSOLIDATION_DATE` is 1 date for all 7 cases, but the schools case has a
second one. The prompt moved from
`deprecated/pairwise_school_send_child_ordinal` to
`pairwise_school_better_ordinal`.

| | Before 2026-08-13 | After |
|---|---|---|
| Question | "Which school would you rather send your child to?" | "Which looks to be the better school?" |
| Asks about | the reader's preference | the school |
| Shares a subject with the report-card proxy | no | yes |

The old question mixed school quality with commute, neighbourhood, safety, and
class. Thus a weak correlation could not say whether the model failed to read
quality, or whether preference and quality simply differ.

**The canonical schools runs of 2026-08-12 used the old prompt**, and so does
the thinking run of 2026-08-13. `is_canonical` still accepts them, because it
tests only the 2026-08-11 date. Thus the schools numbers in `outputs/` describe
the deprecated question. A run of the new prompt is needed before the schools
result can support a claim about school quality. Do not pool the two.

## The provenance chain

A W&B stage run does not record the case name or the output path. Resolve them
in this order:

1. `wandb-metadata.json` gives `args[0]`, the submitit job directory.
2. The stage directory is 2 levels above that.
3. `<stage_dir>/.hydra/overrides.yaml` gives the pipeline, the model, and the
   sweep.
4. `<stage_dir>/outputs/pairwise/` holds the results parquet and `pairs.parquet`.

`wandb-metadata.json` also gives the git commit and the Python executable.

## Data facts

- The results parquet holds the labels. It has no unit or geographic columns.
  `pairs.parquet` holds `unit_uid_a/b` and the coordinates. Join on `pair_id`.
- Read `facilities.parquet` with `pandas`. It has no geo metadata, so
  `geopandas.read_parquet` raises an error.
- Take the unit position from FacDB, not from the pairs manifest. The pairs
  manifest holds the camera position. The camera sits up to 80 ft away and can
  fall in the neighbor polygon.
- An abstention becomes NaN and drops out. A `NotSure` is not a judgment of
  "equal", so it must not pull a mean toward 0.

## The cases

| Case | Unit table | Proxy | Join to units |
|------|-----------|-------|---------------|
| Schools | `curation/facdb_schools_k_12/facilities.parquet` | School Quality Reports, from SODA | The proxy has its own points, by DBN |
| Restaurants | `curation/dohmh_restaurants_inspected_all/restaurants_aggregated.parquet` | DOHMH inspections, already in the unit table | `uid` matches `unit_uid` exactly |
| Libraries | `curation/facdb_libraries/facilities.parquet` | ACS median household income, on disk | The proxy is an area value, not a unit value |
| Subway | `curation/subway_entrances_all/entrances.parquet` | Income and NYPD complaints | Area values only |
| Parks | `curation/facdb_parks_plazas/facilities.parquet`, DPR rows | Parks Inspection Program | **Unit level**, by `gispropnum` |
| Plazas | `curation/facdb_parks_plazas/facilities.parquet`, POPS and DOT rows | Income and NYPD complaints | Area values only |

### Warning: split parks from plazas

The FacDB "PARKS AND PLAZAS" group holds 2 populations, and only 1 has an
inspector.

| Source | Units | Rated by PIP? |
|--------|-------|---------------|
| `dpr_parksproperties` | 2,035 | yes |
| `dcp_pops` | 392 | no |
| `dot_pedplazas` | 92 | no |
| state and other | 63 | no |

A mixed table lets PIP look complete when it covers 79% of the group. The split
also changes the answer: area-level PIP agreement fell from +0.23 to +0.02 once
the plazas left the table. `_proxies.split_parks_plazas` does the split.

### Warning: compare unit by unit when a proxy rates the unit

An area correlation can appear where no unit correlation exists, because
aggregation into polygons creates its own agreement. The parks case shows it
plainly, with gemma-4-12b and PIP cleanliness:

| Scope | n | Spearman rho |
|-------|---|--------------|
| Community district | 60 | +0.022 |
| NTA | 206 | +0.085 |
| **Park (unit)** | **1,048** | **-0.005** |

Use `_proxies.correlate_units` whenever a proxy rates the same object the model
rates. Report the unit value. Use the area value only to show the gap.

This is not a ceiling effect: 35.8% of parks failed a cleanliness inspection and
49.6% failed a condition inspection. Among only the parks with a failure, rho
stays near zero (+0.081 and +0.031).

### Warning: FacDB park geometry is a building, not the park

The `geom_wkb` of a park row has a median area of 0.0038 km^2, while Central
Park covers 3.41 km^2. FacDB matched a nearby building, because a park row
usually carries no BIN. Never use that polygon for a park.

`_proxies.attach_park_property_id` takes the unit POINT and joins it to the true
DPR polygon: 82.7% fall inside a park, and 335 of the remaining 352 sit within
500 ft, so 98.0% get a `gispropnum`. That id is what makes the unit comparison
possible.

### Warning: an area proxy supports a weaker claim

Schools and restaurants have a proxy that measures the unit. An inspection
grades a restaurant. A report card grades a school.

Income grades neither a library nor its upkeep. It describes the households
around it. Thus a libraries correlation can only say that the look of a building
tracks the wealth of its neighborhood. It cannot say that the model reads
library quality. Write that limit into any claim.

### Warning: 2 of the 3 layers report a mean, not a median

| Layer | Method | Is it a median? |
|-------|--------|-----------------|
| Census tract | Direct join | Yes, the true ACS median |
| NTA | Population-weighted mean of tract medians | **No** |
| Community district | Population-weighted mean of tract medians | **No** |

You cannot average medians. No median of medians is a median of the whole area.

The tract-to-area step needs **no** NYC atomic polygons (`wgbs-damt`), because
the geographies nest. Checked 2026-08-12: every one of the 2,325 tracts puts its
interior point in exactly 1 NTA and exactly 1 community district, and no tract
falls in 2 polygons. Use an interior point, never a strict `within` test — a
shared border makes `within` match only 681 of the 2,325 tracts.

### Warning: a small unit set limits a case

NYC holds about 236 public libraries. With `MIN_UNITS_PER_POLYGON = 3`, that
keeps 48 community districts, 12 NTAs, and 1 census tract. Read the
community-district layer as the libraries result. The census-tract layer is
structurally unusable, because 3 libraries almost never share 1 tract.

The restaurants case is the simpler shape. The curation tooling in
`dagspaces/common/curation/dohmh` already got the inspections, cleaned them, and
aggregated them to 1 row for each restaurant, with the position. Thus the
notebook makes no network call and needs no name join.

### Warning: check the orientation of a proxy

A proxy does not always point the same way as the model score.

The DOHMH inspection score counts violation points, so a **low** raw score is a
**clean** restaurant. The model answers "which would you rather eat at", so a
**high** model score is a **better** restaurant. The 2 scales are opposed.

`RESTAURANT_METRICS` states `higher_is_better` and a `sign` for each metric.
`restaurant_proxy` multiplies the raw value by the sign, so every proxy value
leaves the module oriented as "higher is better". A positive correlation then
always means agreement. **Do not flip a sign again inside a notebook.** State the
orientation in the paper.

### Warning: the vintage proxies carry NO orientation

A year measures age, not quality, so no sign makes it point "higher is better".
A positive value on a vintage row says only that the model calls a **newer**
unit better. State it as a finding. Never read it as agreement.
`_results_table.UNORIENTED_PROXIES` names the 3 keys.

| Case | Field | Source | Key | Units with a year |
|------|-------|--------|-----|-------------------|
| Libraries | `construction_year` | Building footprints, on disk | `bin` | 96.4% |
| Schools | `construction_year` | Building footprints, on disk | `bin` | 97.5% |
| Parks | `acquisitiondate` | DPR Parks Properties | `gispropnum` | 93.6% |
| Plazas | `year_completed` | POPS | `bbl` | 68.8% |

Each case takes the field that describes ITS unit. A building year fits a
library and a school, because the unit is a building. It does not fit a park:
33% of the park units hold a BIN, and that BIN is a comfort station. See
`vlm-narratives-docs/vintage-proxies.md`.

## Geographies

| Layer | File | Key | CRS on disk |
|-------|------|-----|-------------|
| Neighborhood tabulation area | `nynta2020_26b/nynta2020.shp` | `NTA2020` | EPSG:2263 |
| Community district | `Community_Districts_20260812.geojson` | `boro_cd` | EPSG:4326 |
| Census tract | `2020_Census_Tracts_20260304.geojson` | `boroct2020` | EPSG:4326 |

The files do not share a CRS. `_geography` puts every layer and every point into
EPSG:2263 before the spatial join.

## The trace notebooks

There is 1 trace notebook for each prompt, `<case>_traces.py`, in the same
shape as the validation-by-proxy notebooks. `_trace_notebook.py` holds the body
they share, so a change reaches all of them. They use `_traces.py`, not
`_provenance.py`, because they want the runs that `_provenance` drops on
purpose.

**Only runs from 2026-08-11 forward enter a trace notebook.** That is the
consolidation date. An earlier run repeats the persona and the cue list of the
old prompt, so its cloud describes the prompt, not the model. The `think10k`
sweep of 2026-07-11 is the largest trace corpus on disk and it is entirely
pre-consolidation, thus it does NOT appear. Tick "include runs from before" to
see it.

`_traces.MIN_ROWS` also drops a run under 1,000 rows. The 2 subway runs of 18
rows are the layout probes that found the image-binding bug, and 1 of them ran
the broken layout, where 15 of 18 traces say the model saw only one image.

### Section 4: structured extractions

Sections 1 to 3 of a trace notebook count **words**. Section 4 counts
**claims**: typed spans with attributes and character offsets, which the
`extract_traces` pipeline writes. `_extractions.py` reads them.

| Question | Where the answer is |
|----------|--------------------|
| Which image holds a cue? | `visual_evidence.image` |
| Is the cue good or bad? | `visual_evidence.valence` |
| Does the model go past the pixels? | class `inference` |
| Does it reason about the people present? | class `person_reference` |
| Can I quote the sentence? | `is_quotable` and the offsets |

The data comes from `data/trace_extractions/`, 1 parquet for each case. Build
it with `scripts/merge_trace_extractions.py`. A notebook that finds no parquet
prints a message and keeps its word sections.

**Warning: count a quotable span only.** A `match_lesser` span is a sentence
the model COMPOSED from the trace plus its own words, and its offsets point at
a fragment of it. `_extractions.load` drops those rows, and about 2% to 3% of
spans fall there.

**Warning: a rate needs every trace in the denominator.** A trace whose spans
were all dropped still asked the question. `_extractions.trace_totals` reads
the totals from the files, never from the filtered frame.

**Warning: an attribute value is a free string, not an enum.** The guided
schema fixes the class names and the attribute names only. The model wrote
`kind` values the prompt never lists, such as `architecture` and `prestige`,
and 8% of `image_artifact.kind` values move into `other`. Read
`vocabulary_report` before you quote a rate, and raise `SCHEMA_VERSION` when
you change a list.

#### Which cues go with a win

The extraction names the image a cue sits on, thus the question is
**directional**: when the model attaches a cue to image A, does A win? A test
of presence cannot ask this, because both images live in 1 trace.

`plot_win_block` sets the cues as a block of text, the best win rate first, and
colours them on the same scale. `plot_win_bars` gives the same numbers with a
Wilson interval, and that is the figure to quote.

**Use the block, not the cloud.** `plot_win_cloud` still exists, and it sizes
each cue by how often the model names it. But a cloud spends its area on 2
variables at one time, and area is the channel a reader compares worst. The
block drops the size: reading ORDER carries the win rate and the colour repeats
it. The count of each cue stays in the table.

**Warning: the colour diverges around the BASE RATE, not around 50%.** A cue
sits on the winning image about 60% of the time in every case, because the
model lists more cues for the image it prefers. Centre the scale at 0.5 and
almost every word turns green, which says nothing.

**Warning: this is an association, not a cause.** The model narrates while it
decides, thus a cue may follow the judgment rather than drive it.

3 rules keep the number honest, and `win_rates` applies all 3:

| Rule | Why |
|------|-----|
| 1 vote for each comparison | A trace that names "trash can" 3 times still describes 1 comparison |
| A cue on BOTH images is dropped | A fence in A and in B separates nothing |
| The rate is shrunk toward the base rate | A cue seen 4 times would otherwise read 100% and take the top of the scale |

#### The photograph rows

`plot_win_block(..., unit_photos=6)` puts a band of photographs above the words
and another below: the units the model ranks highest, and the units it ranks
lowest. `unit_win_rates` scores them.

**A unit win rate is not a cue win rate.** A cue rate asks what the model says.
A unit rate asks which places it prefers, and only that one can put a
photograph on the page. The 2 scales also differ in the middle:

| Statistic | Middle of the scale | Why |
|-----------|--------------------|-----|
| Cue | about 0.60 | The model lists more cues for the image it prefers |
| Unit | exactly 0.50 | Each decided pair gives 1 winner and 1 loser |

**Warning: take the unit base rate BEFORE the `min_pairs` filter.** After the
filter it reads about 0.37, because a unit reaches 8 decided comparisons only
when the model keeps deciding about it, and it decides more readily against a
place. Shrink toward that value and every unit moves down.

**Warning: the swap decides which unit the model called image A.**
`presented_label` describes the image the model saw FIRST, and `is_swapped`
says which side that was. Read `unit_uid_a` as image A on a swapped pair and
both rows come out inverted. `presented_left_path` and `presented_right_path`
already carry the presented order, so use those for the pictures.

**Warning: a unit rests on few pairs.** With 10,000 pairs over about 2,000
units, a unit that passes `min_pairs=8` still holds only 9 to 16 decided
comparisons. The frame shows the shrunk rate and the caption shows the raw
record, so a 100% of 9 cannot pass for a settled number. Raise `min_pairs` for
a figure that goes in the paper.

`valence_consistency` then tests the model against itself: a cue the model
called `good` should sit on the image that won. On schools it does — `good`
wins 88%, `bad` wins 6%, `neutral` 43%.

**Warning: the diverging ramp is coral-to-teal, NOT red-to-green.** About 8% of
men cannot separate red from green. `_style.CMAP_DIV` also keeps every stop
about equally dark, because the ramp colours the WORDS: a pale middle
disappears on the white ground. `_style._check_ramp()` proves it.

**Warning: the `decision` class is scaffold.** Every trace ends with "I'll go
with ...", thus that span ranks high in every case and says nothing.
`distinctive` leaves it out. Its label is already a column.

### Colours

`_style.py` holds the palette, sampled from
`UAIR/notebooks/colm-camera-ready/corpus_descriptives_two_corpora.py`, so a
figure here and a figure there read as one system. Call
`_style.apply_house_style()` one time in a notebook; `_traces.make_cloud` picks
up the palette on its own.

**Warning: a fill colour is not an ink colour.** `mint`, `cream`,
`periwinkle`, `slate`, and `amber` are made to sit UNDER black text. As text on
the light ground of a word cloud they are unreadable. `_style.INK` holds only
the swatches that pass a luminance test, and `_style._check_ink()` proves it.
Amber is the trap: at 0.511 it is lighter than green at 0.427, so it fails the
same test the fills fail.

**Warning: the word cloud is seeded.** `_style.WORDCLOUD_SEED` fixes the layout,
so the same counts always draw the same picture. Without it no reader can tell a
real change from a reshuffle.

### Why a case notebook reads every case

The `distinctive` score asks what 1 prompt says that the others do not, thus it
needs the others as a background. A notebook counts every case and draws only
its own. Read the run count in the report to see what the background was.

| Notebook | Runs it keeps |
|----------|---------------|
| `<case>_validation.py` | The canonical set only. Greedy, label only. |
| `<case>_traces.py` | Any run that holds a trace, including a thinking run. |

**A canonical run has no trace.** The battery runs greedy with
`max_tokens=128`, thus `model_reasoning` is empty. Only a thinking sweep fills
it. `_traces.discover_trace_runs` reads the head of each parquet and skips a
run whose trace column is empty.

### Warning: use the `distinctive` mode for a comparison

Every trace names the 2 images, lists cues, then picks a label. Thus a plain
frequency cloud looks nearly the same for each prompt. The `distinctive` mode
scores a word by how much more this prompt uses it than the other prompts, with
the log-odds ratio and an informative Dirichlet prior (Monroe, Colaresi, and
Quinn, 2008). The prior corrects the variance of a rare word, so a word that
appears 3 times cannot beat a word that appears 3,000 times.

The mode needs 2 prompts or more. With 1 prompt it falls back to `frequency`.

### Warning: do not mix 2 prompt eras

A run from before 2026-08-11 used a persona and a cue list, and the trace
repeats that text. A cloud that mixes the 2 eras shows the old prompt, not the
model. `_traces.BOILERPLATE` removes the words that the consolidated prompt
puts in the model's mouth. It does not know the older text, so add those words
in the notebook control.

### Warning: skip a `.presplit.parquet` file

A gemma-4-e2b run needs a post-hoc split of its trace from its answer
(`scripts/split_e2b_thinking_reasoning.py`). The tool keeps the original file
beside the fixed one. `_traces.find_results_parquet` skips it. The generic
`_provenance._find_parquets` does **not**, so it can return the wrong file.

### Cost and cache

A 10k-pair run holds about 40 MB of trace text. The first count costs some
seconds for each run. `_traces` keeps the counts in `.trace_cache/`, keyed by
the parquet path, its modify time, and the filters, so a later call is
immediate. Discovery caches to `.trace_cache/discovery.json`. Press **Refresh
from W&B** in the notebook to see a sweep that landed after the last read.

**Warning:** never glob `multirun/`. That tree sits on NFS and a full walk costs
minutes. Discovery goes through W&B, which names each stage directory.

## Versions

Each module and each notebook carries a `__version__` or a `NOTEBOOK_VERSION`.
The export file names carry the notebook version. Raise the version when you
change how a number is computed.
