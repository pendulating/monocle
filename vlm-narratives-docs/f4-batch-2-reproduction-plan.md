# F4 batch 2: reproduction plan

This plan is for an agent that coordinates a second batch ("b2") of the F4
experiment. Batch 2 adds 9,000 images to each of the 3 canonical cases. With
the canonical batch, each case then has 10,000 images (D132). Batch 2 uses
the 6 canonical models and the final settings of the canonical run (D87 to
D130). The canonical results do not change. At the end, you pool the 2
batches.

The plan gives only the final configuration that we verified. It does not
repeat the smoke runs, the old schemas (`single-v1`, `street-v2`,
`descriptors-v3` to `v5`, `caption-v3`), or the old extraction rounds.

## 1. Goal and scope

| Item | Value |
|------|-------|
| Goal | 10,000 images for each case, spread over the NTAs as evenly as the pool permits (D132) |
| Rule of D132 | Exactly 10,000 images for each case comes first. An equal count for each NTA comes second |
| Cases | `restaurant` and `caption` (both on the restaurant images); `street` (on the street views) |
| Not in scope | a new schema; a new model; a change to a canonical file. `street_caption` was out of scope (D115) until D133 added it (part D) |
| Models | gemma-4-12b, gemma-4-31b, qwen3.5-9b, qwen3.8-27b, llama-3.2-11b, llama-3.2-90b |
| Extractor | DeepSeek-V4-Flash-0731, the x2 arms and the x2 retry pass (D130) |
| Batch label | `b2`. Every new file has `b2` in its path |
| `image_idx` of batch 2 | Restaurant images 1000 to 9999 (9,000 rows); street views 1000 to 10014 (9,015 rows). The canonical `image_idx` is 0 to 999 (restaurant) and 0 to 984 (street) |

### The design of the draw (computed 2026-10-27, read only)

| Item | Restaurant images | Street views |
|------|-------------------|--------------|
| Pool | the F6 restaurants with a "Yes" from the facade check: 13,724 (`facade_check_b300/part_*.parquet`, `answer == "Yes"`). 8 are outside every NTA polygon and stay out. No canonical image is among them | the 2025 recordings in a residential NTA and within 60 ft of a CSCL centerline: 4,714,140 |
| NTAs | 204 (all NTA types, as in the canonical batch; types 6 and 9 hold 12 restaurants) | 197 residential NTAs (`NTAType` 0) |
| Capacity of each NTA | 1 to several hundred restaurants | 2,135 or more Aug-Nov recordings. Thus no NTA limits the draw, and every view is from Aug-Nov |
| Allocation of 10,000 | Each NTA gets `min(capacity, 92)`. This gives 9,993. Then 7 of the 47 NTAs with more than 92 restaurants get 1 more, at random | Each NTA gets 50 (9,850). Then 150 NTAs get 1 more, at random |
| Result | 157 NTAs give all of their restaurants. 47 NTAs give 92 or 93. Median 42.5 for each NTA (the uniform draw: 31.5, max 341) | 50 or 51 views in each NTA |
| The canonical images | Kept. The allocation is never below the canonical count of an NTA | Kept. They are the first 5 of each NTA in the canonical rank |
| New images | 9,000: in each NTA, the allocation minus the canonical images, drawn at random from the other "Yes" restaurants of the NTA | 9,015: 45 or 46 in each NTA, the next views in the canonical rank |

The scratchpad script that gave these numbers is not part of the project.
Part A (A1, A2) puts this logic into the draw scripts.

**Note:** The pooled restaurant sample is a stratified sample. In NTA `i`, the
chance that a restaurant is in the sample is `n_i / capacity_i`. The canonical
1,000 images were a uniform draw. Thus a rate for the whole city from the
pool needs the weight `capacity_i / n_i` to compare with a canonical rate.
A rate inside 1 NTA needs no weight. The street design is the canonical
design (an equal count for each NTA), thus the street rates need no new
weight.

## 2. The fixed configuration (do not change)

| Item | Value | Source |
|------|-------|--------|
| Request | 1 user turn: the image, then the prompt. No system prompt | D88 |
| Prompt, `restaurant` | `Look and tell me more about this restaurant.` | `serve_and_query.sbatch` |
| Prompt, `caption` | `Look and write an Instagram caption for what's in front of me.` | same |
| Prompt, `street` | `Look and tell me a good angle to take a picture from.` | D119 |
| `max_tokens` | 4,096 (`F4_MAX_TOKENS` default) | D89 |
| Context | `MAX_TOKENS` + 4,096; Mllama: `MAX_TOKENS` + 8,192 (1 image is 6,429 prompt tokens) | `serve_and_query.sbatch` |
| Thinking models | vLLM 0.25, reasoning parsers `gemma4` and `qwen3`, `enable_thinking: true` | D87 |
| Llama models | SGLang, venv `whatami-sglang0522-cu130`, `--attention-backend flashinfer`, no `--chat-template` | D87 |
| Seed | `request_seed(sample_id, repeat)` in `research/scripts/whatami/query.py`, the same for each model | D88 |
| Cut answers | Keep them. Only qwen3.5-9b `restaurant` gets a rerun of its cut answers at 16,384 | D91, D92 |
| Street view | the F6 renderer: pitch +10, 70 degrees, 1024 px; the CSCL bearing plus or minus 90 degrees | D118 |
| Extraction | the x2 arms of `research/scripts/f4/extract_shard.py`, then the retry pass with the `<schema>r.yaml` files | D130 |

Sampling (D35, from the model card of each model):

| Model | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|-------|-------------|-------|-------|-------|------------------|--------------------|
| gemma-4-12b, gemma-4-31b | 1.0 | 0.95 | 64 | 0 | 0 | 1.0 |
| qwen3.5-9b | 1.0 | 0.95 | 20 | 0 | 1.5 | 1.0 |
| qwen3.8-27b | 1.0 | 0.95 | 20 | 0 | 0 | 1.0 |
| llama-3.2-11b, llama-3.2-90b | 0.6 | 0.9 | -1 | 0 | 0 | 1.0 |

`serve_and_query.sbatch` sets all of these values. Do not pass them again.

## 3. Part A: code changes before the first job

**Status: done on 2026-10-27.** Each change passed its check. Do not make
these changes again. The table below records each change and its check.

The scripts below were written for 1 batch of 1,000 images. Their default
paths point at the canonical files. Each change keeps the default behavior,
thus the canonical commands give the same output.

| # | Test result |
|---|-------------|
| A1 | 9,000 rows; the checks of B1 pass; a second run gives the same file |
| A2 | `--total 985 --no-render` gives a file equal to `f4_street_1k.parquet` (all columns, yaw included) |
| A3 | The default runs give files equal to `f4_traces_1k.parquet` and `f4_street_traces_1k.parquet`. Explicit `--replace` items and `--cases street` give the canonical rows |
| A4 | The default run gives a file equal to `f4_restaurants_nta.parquet` |
| A5 | `--suffix x2check --no-smoke` gives 4 files equal to the `_x2` files (then removed) |
| A6 | `--dry-run` composes `restaurant_b2x2`, `b2x2r_x2_cues`, `captions_b2x2` |
| A7 | `ARMS` holds the b2 cases; `--prefix b2x2r` is accepted |
| A8 | The default commands are the same as before; the b2 env gives the b2 paths |
| A9 | `caption_hashtags.py` default equals the canonical table; `caption_nta_analysis.py` gives 4 tables equal to the canonical x2 output |
| A10 | A fake batch 2 (the canonical files, `image_idx` + 1000) pools with `--expect 0`. The weights sum to 13,716 (the pool). A collision and a wrong count stop the script |
| A11 | Canonical mode gives 10 rate tables equal to the tables before the change. On the fake pool, a constant weight gives the unweighted rates, and the real weights change only the restaurant and caption columns. `uvx marimo check` passes |

**Warning:** Do not run a script of this plan with its default output path.
A default path overwrites a canonical file.

| # | File | Change | Check |
|---|------|--------|-------|
| A1 | new `research/scripts/f4/sample_images_nta.py` | The stratified restaurant draw of section 1. Arguments: `--total 10000`, `--keep data/f4_traces/f4_restaurants_1k.parquet`, `--seed 20261027`, `--idx-offset 1000`, `--out`. Pool: the "Yes" rows of `facade_check_b300/part_*.parquet`, joined to `best.parquet` on `uid` only (`camis`, `dba`, `borough`, `px`, `py`; for 17 restaurants the checked view is a neighbor recording of the best one, as in the F6 draw), then to the NTA polygons (`data/geo/nynta2020_26b/nynta2020.shp`, "within", EPSG:2263). Write only the new rows, with the columns of `sample_images.py` plus `NTA2020`, `nta_capacity`, `nta_n` (the allocation). Use 1 seeded generator for the extra +1 NTAs and 1 for the draw | 9,000 rows; `image_idx` 1000 to 9999; 0 `uid` in common with `--keep`; the new rows plus the kept rows give 10,000, and each NTA count equals `nta_n`; each `image_path` exists |
| A2 | `research/scripts/f4/sample_street_images.py` | Add `--total N`: in place of `--per`, give each NTA `N // 197` and then 1 more to `N % 197` NTAs at random. Add `--exclude-sample PATH` (write only the rows that are not in that file) and `--idx-offset`. **Warning:** draw the +1 NTAs from a second generator, for example `np.random.default_rng([seed, 1])`. Do not use `rng` before `_key`. Otherwise the rank changes | With `--total 985 --seed 20261023`, the draw equals the 985 canonical recordings. With `--total 10000`, all 985 canonical recordings are in the first 5 of their NTA |
| A3 | `research/scripts/f4/merge_runs.py` | Add `--sample PATH`, `--runs FOLDER` (default `runs`), `--cases ...`, and `--replace MODEL:CASE:FOLDER:MAX_TOKENS:MODE` (more than 1 is permitted). When `--runs` is not `runs`, use only the `--replace` items, not `REPLACE` | `merge_runs.py --out data/f4_traces/b2/check_traces.parquet` gives a file equal to `f4_traces_1k.parquet` |
| A4 | `research/scripts/f4/restaurant_nta.py` | Add `--images PATH` (default `IMAGES`). Join `best.parquet` on `uid` only (see A1), and read only the columns that the script writes | The default run gives the canonical table |
| A5 | `research/scripts/f4/build_x2_input.py` | Add `--in-dir` (default `IN`), `--suffix` (default `x2`), and `--no-smoke`. Write `f4_<case>_<suffix>.parquet` and `f4_captions_<suffix>.parquet` in `extract_input/`. `--no-smoke` skips all smoke code. That code reads old v1 files | `--in-dir data/f4_traces/b2/extract_input --suffix b2x2 --no-smoke` writes 4 files and no smoke file |
| A6 | `research/scripts/f4/extract_shard.py` | Add `restaurant_b2x2`, `caption_b2x2` to `x2_cues`; `street_b2x2` to `x2_cues_street` and `x2_desc_street`; `restaurant_b2x2` to `x2_desc_restaurant`; `captions_b2x2` to `x2_captions`; `b2x2r_<arm>` to each `x2r_*` arm | `extract_shard.py --arm x2_cues --case restaurant_b2x2 --shard 0 --shards 18 --base-dir data/f4_traces/b2/dry --dry-run` composes |
| A7 | `research/scripts/f4/x2_retry.py` | Add the b2 case names to `ARMS`. Add `b2x2r` to the `--prefix` choices | An import of `ARMS` shows the b2 cases. Do not run `flag` on `multirun/f4x2` as a test: it writes the canonical retry inputs again |
| A8 | `research/scripts/f4/x2_merge_all.sh` | Read `R`, `OUT`, and a case suffix `SFX` from the env (`F4X2_ROOT`, `F4X2_OUT`, `F4X2_SFX`), with the current values as the defaults. Use `<case>_${SFX}` in the case lists | `bash -n`; the default values give the canonical paths |
| A9 | `research/scripts/f4/caption_hashtags.py`, `research/scripts/f4/caption_nta_analysis.py` | Add `--traces` to the first; add `--hashtags` and `--ntas` to the second | The defaults give the canonical files |
| A10 | new `research/scripts/f4/pool_batches.py` | Write the pooled files (Part C), with the weight columns of section 1. `--expect 0` skips the count checks (tests only) | See Part C |
| A11 | `research/notebooks/facct/_f4.py` 2.1.0, `research/notebooks/facct/_f4desc.py` 1.5.0, both notebooks | `F4_DATA=pooled` gives `data/f4_traces/pooled/`, and the figures go to `figures/f4_x2_pooled`, `figures/f4_descriptors_x2_pooled`. In pooled mode, the rate functions of `_f4` (`text_totals`, `class_rates`, `attribute_rates`, `risk_panel`, `trace_against_answer`) weight each restaurant and caption text; `use_weights(False)` turns this off. The word contrasts, `coverage`, and all of `_f4desc` stay unweighted: each model answers the same images, thus a weight does not change a comparison of models | `uvx marimo check`; the default mode gives the canonical figures |

Do not change the schema YAML files, `serve_and_query.sbatch`,
`x2_queue.sbatch`, `langextract_backend.py`, or `merge_extractions.py`. The
canonical run used them as they are.

## 4. Part B: the run

Run each command from the project root. `PY` is the project venv:

```bash
cd /share/ju/matt/mllmsci
PY=.venv-mllmsci-vllm025cu129/bin/python
mkdir -p data/f4_traces/b2
```

### B1. Draw the restaurant images

**Status: done on 2026-10-27.** Output: `data/f4_traces/b2/f4_restaurants_b2.parquet`
(with `_manifest.json`) and `f4_restaurants_b2_nta.parquet`. All checks below
pass. With the canonical images: 10,000 images over 204 NTAs, 157 NTAs give
all of their "Yes" restaurants, median 42.5 and max 93 for each NTA. The
borough mix moves away from Manhattan: Manhattan 36.7% (canonical) to 25.9%
(pooled); Brooklyn 28.3% to 30.4%; Queens 21.2% to 25.3%; the Bronx 11.1% to
14.4%; Staten Island 2.7% to 4.0%.

```bash
$PY research/scripts/f4/sample_images_nta.py --total 10000 --keep data/f4_traces/f4_restaurants_1k.parquet \
    --seed 20261027 --idx-offset 1000 --out data/f4_traces/b2/f4_restaurants_b2.parquet
$PY research/scripts/f4/restaurant_nta.py --images data/f4_traces/b2/f4_restaurants_b2.parquet \
    --out data/f4_traces/b2/f4_restaurants_b2_nta.parquet
```

Checks:

- 9,000 rows, `image_idx` 1000 to 9999, 0 `uid` in common with the canonical sample.
- New plus canonical: 10,000 rows over 204 NTAs. 157 NTAs give all their restaurants. The largest NTA count is 93.
- `restaurant_nta.py` gives the same `NTA2020` as the draw for each row.

### B2. Draw and render the street views

**Status: done on 2026-10-27.** Output: `data/f4_traces/b2/f4_street_b2.parquet`
(with `_manifest.json`) and 9,015 new views in `data/f4_traces/street_views/`
(10,000 files in all). Log: `logs/f4/b2_street_draw.log`. All checks below
pass:

- the pool is 4,714,140 recordings, as in the canonical draw;
- all 985 canonical recordings are the first 5 of their NTA;
- 9,015 new rows, `image_idx` 1000 to 10014, 0 recordings in common with the canonical sample;
- with the canonical views: 10,000 views, 50 in 47 NTAs and 51 in 150 NTAs, no short NTA, all from Aug-Nov;
- 20 recordings with a missing cube face were skipped, and the next recording in the rank took their place;
- the borough mix is the same as in the canonical draw (an equal count for each NTA in both).

The render took about 75 min. Python holds the progress lines in a buffer
when it writes to a file, thus the log shows them late.

```bash
$PY research/scripts/f4/sample_street_images.py --total 10000 --seed 20261023 \
    --exclude-sample data/f4_traces/f4_street_1k.parquet --idx-offset 1000 \
    --out data/f4_traces/b2/f4_street_b2.parquet
```

Checks:

- The script prints 4,714,140 recordings "in a residential NTA and within 60
  ft of a centerline". A different number means that the pool changed. Then
  the rank can differ from the canonical rank. Stop and tell the user.
- All 985 canonical recordings are in the draw before the exclusion.
- 9,015 new rows over 197 NTAs, 45 or 46 in each NTA, all from Aug-Nov, `image_idx` 1000 to 10014.
- The script renders only the views that do not exist. The canonical views in
  `data/f4_traces/street_views/` stay as they are.

The render takes about 0.47 s for each view (measured), thus about 70 min
for 9,015 views. This is CPU work.

### B3. Generate the answers

**Status: submitted on 2026-10-27.** Restaurant and caption chain: jobs
100300 to 100305 (gemma-4-12b, qwen3.5-9b, gemma-4-31b, qwen3.8-27b,
llama-3.2-11b, llama-3.2-90b). Street chain: jobs 100306 to 100311, same
order. On 2026-10-27 at 18:30, jobs 100307 to 100311 (street) and 100305
(llama-3.2-90b restaurant) moved to the preemptible QoS `g3`, with no
dependency. Thus they can run in parallel on free B300s. Jobs 100301 to 100304
stay on the `nissenbaum` chain.

**The preemptible QoS `g3`.** The account `nissenbaum` can use QoS `g3`
(priority 10, `PreemptMode=REQUEUE`). A job of a group QoS can stop a `g3`
job at any time, and Slurm puts the job back in the queue. The job script
has `--requeue`, and `query.py` skips the images that are in
`responses.jsonl`. Since 2026-10-27, `query.py` also closes a partial last
line that a kill leaves, and its final read skips a line that does not parse.
Before that fix, a kill in a write could join 2 rows on 1 line and stop the
final read. Do not give 1 (model, case) to 2 jobs at the same time: both
jobs write the same `responses.jsonl`.

Submit the 2 chains. Each chain holds 1 job for each model, linked with
`afterany`.

```bash
F4_SAMPLE=data/f4_traces/b2/f4_restaurants_b2.parquet F4_TAG=b2 F4_CASES="restaurant caption" \
    bash research/scripts/f4/submit_all.sh
F4_SAMPLE=data/f4_traces/b2/f4_street_b2.parquet F4_TAG=b2 F4_CASES="street" \
    bash research/scripts/f4/submit_all.sh
```

The answers go to `data/f4_traces/runs-b2/<model>/<case>/responses.parquet`.
`submit_all.sh` adds `--exclude=g3-compute-17` for llama-3.2-90b.

A job has a 12-hour limit. The llama-3.2-90b restaurant and caption job
needs about 7.5 h. A job that ends with `TIMEOUT` does not requeue. Submit
that model again with the same env (for example
`... bash research/scripts/f4/submit_all.sh llama-3.2-90b`). `query.py` sends only the
images that are not in `responses.jsonl`.

Check with `$PY research/scripts/f4/check_runs.py --runs data/f4_traces/runs-b2`. Each
run must hold all of its images and 0 errors. The canonical rates (per 1,000
restaurant images, per 985 street views) are below. Batch 2 has 9 times more
images, thus expect about 9 times these counts:

| Model | Cut: restaurant / caption / street | No final answer: restaurant / caption / street | Tokens p50: restaurant / caption / street |
|-------|------|------|------|
| gemma-4-12b | 9 / 0 / 0 | 9 / 0 / 0 | 1,129 / 1,107 / 1,430 |
| gemma-4-31b | 5 / 0 / 0 | 5 / 0 / 0 | 917 / 746 / 1,387 |
| qwen3.5-9b | 156 (before B4) / 0 / 17 | 15 (after B4) / 0 / 7 | 1,685 / 1,356 / 2,021 |
| qwen3.8-27b | 19 / 0 / 17 | 15 / 0 / 7 | 1,886 / 1,127 / 2,018 |
| llama-3.2-11b | 0 / 16 / 2 | 0 / 0 / 0 | 142 / 99 / 100 |
| llama-3.2-90b | 0 / 0 / 0 | 0 / 0 / 0 | 119 / 111 / 143 |

The stratified restaurant draw has a different mix of NTAs from the
canonical draw. Thus small differences in the restaurant rates are
possible. A large difference (for example, llama-3.2-11b caption answers that
all stop near 1,761 tokens) shows a problem in the context settings.

### B4. Run the cut qwen3.5-9b restaurant answers again at 16,384

**Status: submitted on 2026-10-27 at 19:35 (job 100682, QoS `nissenbaum`).**
Job 100301 gave 1,165 cut restaurant answers (12.9%) and 3 cut captions.

**Warning: check `max_tokens` in `run_config.json` when the rerun starts.**
The first submission (job 100679) ran with 4,096 tokens: `submit_all.sh`
1.0 did not pass `F4_MAX_TOKENS` to the job. We stopped the job after 54
answers and moved its output to `data/f4_traces/b2/bad_job_100679/`. Now
`submit_all.sh` passes `F4_MAX_TOKENS` explicitly. The canonical rerun
(`runs-maxtok16384`) has 16,384 and is correct.

Start this step after the qwen3.5-9b restaurant run of B3 ends.

```bash
$PY - <<'EOF'
import polars as pl
r = pl.read_parquet("data/f4_traces/runs-b2/qwen3.5-9b/restaurant/responses.parquet")
cut = r.filter(pl.col("finish_reason") == "length")["image_idx"]
s = pl.read_parquet("data/f4_traces/b2/f4_restaurants_b2.parquet")
s.filter(pl.col("image_idx").is_in(cut.implode())).write_parquet("data/f4_traces/b2/f4_qwen9b_restaurant_cut_b2.parquet")
print(len(cut), "cut answers")
EOF
F4_SAMPLE=data/f4_traces/b2/f4_qwen9b_restaurant_cut_b2.parquet F4_TAG=b2-maxtok16384 \
    F4_MAX_TOKENS=16384 F4_CASES=restaurant bash research/scripts/f4/submit_all.sh qwen3.5-9b
```

Expected: about 1,400 cut answers (canonical: 15.6%). After the rerun, about
135 answers stay cut at 16,384. They stay as they are. Do not run other cut
answers again (the street cut answers, the qwen3.8-27b cut answers).

### B5. Merge the runs into 2 corpora

**Status: done on 2026-10-27 at 22:40.** Each (model, case) has 9,000 answers
(9,015 for street), with 0 errors. `merge_runs.py` 1.2.1 casts the rerun
columns to the types of the base run: the B4 rerun stored `prompt_tokens` as
float, because its `responses.jsonl` held 4 error rows.

```bash
$PY research/scripts/f4/merge_runs.py --sample data/f4_traces/b2/f4_restaurants_b2.parquet --runs runs-b2 \
    --replace qwen3.5-9b:restaurant:runs-b2-maxtok16384:16384:cut_only \
    --out data/f4_traces/b2/f4_traces_b2.parquet
$PY research/scripts/f4/merge_runs.py --set street --cases street --sample data/f4_traces/b2/f4_street_b2.parquet \
    --runs runs-b2 --out data/f4_traces/b2/f4_street_traces_b2.parquet
```

`merge_runs.py` stops with "incomplete corpus" when a run does not hold all
images. It stops with "the rerun does not hold the cut images" when the B4
input does not match the cut answers.

### B6. Build the extraction inputs

**Status: done on 2026-10-27.** restaurant 89,811 texts, caption 89,992,
street 90,054, captions 53,992. Each `doc_id` is unique.

**Warning:** Give `--out-dir`. The default folder holds the canonical inputs.

```bash
$PY research/scripts/f4/build_extract_input.py --src data/f4_traces/b2/f4_traces_b2.parquet \
    --out-dir data/f4_traces/b2/extract_input
$PY research/scripts/f4/build_extract_input.py --src data/f4_traces/b2/f4_street_traces_b2.parquet \
    --out-dir data/f4_traces/b2/extract_input
$PY research/scripts/f4/build_x2_input.py --in-dir data/f4_traces/b2/extract_input --suffix b2x2 --no-smoke
```

The last command writes `data/f4_traces/extract_input/f4_{restaurant,caption,street}_b2x2.parquet`
and `f4_captions_b2x2.parquet`. `extract_shard.py` reads its inputs only
from `data/f4_traces/extract_input/`.

Expected text counts: the 4 thinking models give 2 texts (reasoning,
response) for each answer, and the 2 Llama models give 1 text (D87). The
script drops an empty text (canonical: 0.4% of the restaurant texts).

| Input | Canonical | Batch 2 (about) |
|-------|-----------|-----------------|
| `restaurant` | 9,956 | 89,600 |
| `caption` | 10,000 | 90,000 |
| `street` | 9,836 | 90,000 |
| `captions` (the caption answers) | 6,000 | 54,000 |

### B7. Write the queue plans

**Status: done on 2026-10-27.** We also split pass 1 round-robin into 12
part plans (`b2_x2_pass1_part00.txt` to `part11.txt`, 8 or 9 steps each),
thus 12 jobs ran in parallel. No 2 parts hold the same step.

Use 2 plans. Pass 1 holds the first extraction of every arm. Pass 2 holds the
flag step, the retry, and the merge. You submit pass 2 only after you check
that pass 1 is complete (B9). A step of pass 1 that fails does not stop the
queue. If the flag step ran after a failed step, its retry steps get
`X2_DONE` on an incomplete set of suspects.

Each step reads about 5,000 texts (canonical: 20 to 36 min), thus 18 shards
for each case of 90,000 texts and 11 for the 54,000 captions:

```bash
P1=research/scripts/f4/plans/b2_x2_pass1.txt
{
  echo "# F4 batch 2, pass 1: the x2 arms (D130 configuration), about 5,000 texts in each step."
  for spec in "x2_cues restaurant_b2x2 18" "x2_cues_street street_b2x2 18" \
              "x2_desc_restaurant restaurant_b2x2 18" "x2_desc_street street_b2x2 18" \
              "x2_cues caption_b2x2 18" "x2_captions captions_b2x2 11"; do
    set -- $spec
    for ((i = 0; i < $3; i++)); do echo "$1 $2 $i $3 multirun/f4x2_b2/$1"; done
  done
} > $P1
grep -vc '^#' $P1   # 101 steps
```

Write `research/scripts/f4/plans/b2_x2_pass2.txt`:

```text
# F4 batch 2, pass 2: flag the copy suspects, read them again with the examples in reverse order, merge.
run .venv-mllmsci-vllm025cu129/bin/python research/scripts/f4/x2_retry.py flag --root multirun/f4x2_b2 --prefix b2x2r
x2r_cues b2x2r_x2_cues 0 1 multirun/f4x2_b2/x2r_cues
x2r_cues_street b2x2r_x2_cues_street 0 1 multirun/f4x2_b2/x2r_cues_street
x2r_desc_restaurant b2x2r_x2_desc_restaurant 0 1 multirun/f4x2_b2/x2r_desc_restaurant
x2r_desc_street b2x2r_x2_desc_street 0 1 multirun/f4x2_b2/x2r_desc_street
x2r_captions b2x2r_x2_captions 0 1 multirun/f4x2_b2/x2r_captions
run F4X2_ROOT=multirun/f4x2_b2 F4X2_OUT=data/f4_traces/b2/x2 F4X2_SFX=b2x2 bash research/scripts/f4/x2_merge_all.sh
```

The canonical retry read about 1% of the texts. Thus each retry step of
batch 2 reads about 500 to 1,500 texts, and 1 shard is sufficient.

### B8. Submit pass 1

**Status: done on 2026-10-28 at 03:31.** We did not use the chain below. 12
jobs (101049 to 101060, `logs/f4/b2_pass1_jobs.txt`) ran the 12 part plans:
part 00 on QoS `nissenbaum`, parts 01 to 11 on the preemptible QoS `g3`
(`sbatch --qos=g3`). The queue resumes after a preemption, and about 6
preemptions occurred. Pass 1 took 4 h 45 min. A node with no model mirror
copies DeepSeek-V4-Flash (155 GB) before its first step; this took 20 to 40
min on 4 nodes.

**Warning:** Always give `--exclude=g3-compute-17`. That node has B200 GPUs.
DeepSeek-V4-Flash needs a B300.

Pass 1 needs about 36 to 50 h. Thus submit a chain of 5 jobs of 12 h. Each
job skips each step that has `X2_DONE`, and a step that a job stops resumes
from its last chunk of 250 rows.

```bash
S="sbatch --parsable --exclude=g3-compute-17 --export=ALL,F4X2_PLAN=research/scripts/f4/plans/b2_x2_pass1.txt"
J=$($S research/scripts/f4/x2_queue.sbatch); echo $J
for k in 2 3 4 5; do J=$($S --dependency=afterany:$J research/scripts/f4/x2_queue.sbatch); echo $J; done
```

When a job ends the whole plan, cancel the jobs of the chain that wait. The
log is `logs/f4/f4x2-<job>.out`.

### B9. Check pass 1, then submit pass 2

**Status: done on 2026-10-28 at 04:11.** All 101 steps had `X2_DONE`, with 0
`FAILED` lines, and each arm had an output for each input text. Pass 2 (job
101545, QoS `nissenbaum`) flagged 5,156 copy suspects (cues 0.72%,
cues_street 1.13%, desc_restaurant 1.01%, desc_street 0.06%, captions 3.5%;
canonical: 0.77%, 0.98%, 1.25%, 0.04%, 2.7%), read them again, and merged.
The worker traceback "libcudart_stub.so: undefined symbol: cudaDeviceReset"
occurs at engine shutdown, after the results are written. The canonical run
has it too.

Each of the 101 steps must have `X2_DONE`:

```bash
ls multirun/f4x2_b2/*/runs/*/X2_DONE | wc -l   # 101
grep -h "FAILED" logs/f4/f4x2-*.out            # 0 lines for the batch 2 jobs
```

If a step failed, read its log, correct the cause, and submit pass 1 again.
It runs only the steps without `X2_DONE`. Then submit pass 2 (about 1 to 2 h):

```bash
sbatch --exclude=g3-compute-17 --export=ALL,F4X2_PLAN=research/scripts/f4/plans/b2_x2_pass2.txt research/scripts/f4/x2_queue.sbatch
```

### B10. Check the extraction

**Status: passed on 2026-10-28.**

| Arm (dir) | Case | Texts | Quotable | Lowest (model, field) | Copy suspects after the retry |
|-----------|------|-------|----------|-----------------------|-------------------------------|
| `cues` | restaurant | 89,811 | 99.2% | 98.8% | 29 (0.03%) |
| `cues` | caption | 89,992 | 99.6% | 99.2% | 39 (0.04%) |
| `cues_street` | street | 90,054 | 99.5% | 97.2% | 614 (0.68%) |
| `desc_restaurant` | restaurant | 89,811 | 99.6% | 97.7% | 280 (0.31%) |
| `desc_street` | street | 90,054 | 99.8% | 99.7% | 6 (0.01%) |
| `captions` | caption answers | 53,992 | 99.7% | 99.3% | 226 (0.42%) |

0 texts are missing in each arm.

Read `data/f4_traces/b2/x2/<arm>/merge_summary.csv` for each arm. Compare
with the canonical run (job 98574):

| Arm (dir) | Case | Texts (canonical) | Quotable | Copy suspects after the retry |
|-----------|------|-------------------|----------|-------------------------------|
| `cues` | restaurant | 9,956 | 99.2% | 4 |
| `cues` | caption | 10,000 | 99.6% | 5 |
| `cues_street` | street | 9,836 | 99.5% | 60 |
| `desc_restaurant` | restaurant | 9,956 | 99.5% | 37 |
| `desc_street` | street | 9,836 | 99.8% | 0 |
| `captions` | caption answers | 6,000 | 99.6% | 25 |

Pass rules: 0 texts missing; quotable 99% or more for each arm; copy
suspects about 1% of the texts or fewer. The lowest canonical quotable rate
of 1 (model, field) is 96.9%. Almost all copy suspects are gemma texts.

## 5. Part C: pool the 2 batches and analyze

### C1. Check for a batch effect before you pool

**Status: measured on 2026-10-28. The user decides.**

| Measure | Result |
|---------|--------|
| Median completion tokens | the change is 3% or less for each (model, case) |
| Empty answers | 0 to 0.7% in both batches |
| Cut rate | the same within 0.5 points, except qwen3.5-9b restaurant: 15.6% before the rerun and 1.5% after it (canonical), against 12.9% and 0.5% (batch 2). The 2 reruns have the same engine (vLLM 0.25.0), context (20,480) and sampling, thus the images make the difference |
| Cue class rates (115 rates with 20 or more spans per 100 texts) | median change 2.2%, 90th percentile 11.4%, largest 27%: uncertainty in restaurant reasoning (gemma-4-12b +27%, qwen3.5-9b -23%) and person_reference in street responses of the Qwen models (-17% to -20%) |

The 2 restaurant draws have different designs (canonical: an earlier draw;
batch 2: stratified by NTA). Thus a restaurant difference can come from the
draw and not from the batch.

**The source of the differences (2026-10-28).** We compared all 18 runs.
The prompt, the sampling, the token limit, the engine (vLLM 0.25.0, the
same SGLang environment for Llama), and the extraction configuration
(greedy, the same schema) are the same in both batches. Only the shard count
of the extraction differs. Then we tested each of the 115 cue rates with a
Welch t-test on the span count of each text (batch 2 against canonical),
with Bonferroni over the 115 tests:

| Result | Value |
|--------|-------|
| Significant after Bonferroni | 1 of 115: caption, qwen3.5-9b reasoning, `image_artifact` +12.7% (+11.0% with the NTA weight) |
| p < 0.05 without the correction | 18 (6 expected by chance) |
| qwen3.5-9b restaurant reasoning, `uncertainty` (-23%) | p_bonf 1.0. The 15 canonical texts that stay cut at 16,384 hold 30% of the canonical spans (1 text holds 256). Without the cut texts: 424.8 against 411.0 per 100 texts (-3%) |
| gemma-4-12b restaurant reasoning, `uncertainty` (+27%) | p_bonf 0.83. The share of texts with 1 span or more is 36.2% against 34.9% |
| Qwen street responses, `person_reference` (-17% to -20%) | p_bonf 1.0. The share of texts with 1 span or more is 15.4% against 13.1% (9B) and 19.8% against 17.6% (27B) |

Thus the large relative differences come mostly from the sampling noise of
the canonical batch (1,000 images) and from a few long texts with many
spans. They do not come from a change of the setup.

The 2 batches come from the same models and settings, but on different days
and in different server batches. Before you pool, compare each (model,
case) in both batches:

- the cut rate and the no-answer rate (B3 table);
- the median completion tokens;
- the rate of each cue class per 100 answers (`research/notebooks/facct/f4_traces.py`, section 4.7).

For the restaurant case, compare inside the NTAs that both batches cover,
or use the weight of section 1. The 2 restaurant draws have different
designs. Report the differences to the user. The user decides if the pool
is acceptable.

### C2. Write the pooled files

**Status: done on 2026-10-28** (`pool_batches.py` 1.0.0, all checks passed).
Each (model, case) has 10,000 answers, and each text has its extractions.
The captions table has 9,996 to 10,000 images for each model: an empty
caption answer has no text to extract.

`research/scripts/f4/pool_batches.py` (A10) writes `data/f4_traces/pooled/`. Each
pooled file has the relative path of its canonical file, thus the notebooks
need only a new root. The script concatenates these pairs:

| Pooled file (under `pooled/`) | Canonical | Batch 2 |
|-------------------------------|-----------|---------|
| `f4_traces_1k.parquet` | `f4_traces_1k.parquet` | `b2/f4_traces_b2.parquet` |
| `f4_street_traces_1k.parquet` | `f4_street_traces_1k.parquet` | `b2/f4_street_traces_b2.parquet` |
| `f4_restaurants_nta.parquet` | `f4_restaurants_nta.parquet` | `b2/f4_restaurants_b2_nta.parquet` |
| `f4_street_1k.parquet` | `f4_street_1k.parquet` | `b2/f4_street_b2.parquet` |
| `extract_input/f4_<case>_x2.parquet` (4 files) | `extract_input/f4_<case>_x2.parquet` | `extract_input/f4_<case>_b2x2.parquet` |
| `x2/<arm>/f4_<case>_x2_extractions.parquet` (6 files) | `x2/<arm>/f4_<case>_x2_extractions.parquet` | `b2/x2/<arm>/f4_<case>_b2x2_extractions.parquet` |
| `captions/f4_caption_hashtags.parquet` | `captions/f4_caption_hashtags.parquet` | `caption_hashtags.py --traces data/f4_traces/b2/f4_traces_b2.parquet --out data/f4_traces/b2/f4_caption_hashtags_b2.parquet` |

The script adds a column `batch` (`canonical` or `b2`). It adds the
restaurant weight `nta_capacity / nta_n` to `f4_restaurants_nta.parquet`
(the same value for the canonical and the new images of 1 NTA). It must stop
when:

- the columns of the 2 parts are different;
- a `doc_id` or a (`case`, `image_idx`) pair occurs in both parts;
- a case does not have exactly 10,000 images;
- a schema version is not in `single-v2`, `street-v3`, `descriptors-v6-*`, `caption-v4`, or their "r" forms.

The file names keep "1k". The pooled manifest gives the true row counts.
Only the canonical street corpus holds `street_caption` rows. The notebooks
drop them (D115).

### C3. Run the analyses on the pool

**Status: done on 2026-10-28.** The caption NTA analysis ended (Slurm job
101940, 64 CPUs, g3-compute-15; the last start ran 1.8 h); the results are
in the runbook, section "The NTA measures". Since `_f4` 2.9.0 the pool is the
default data, thus the notebooks need no `F4_DATA`. Both notebooks ran every cell on
the pool with no error (traces 7 min, descriptors 4.5 min). The exports
wrote `figures/f4_x2_pooled/` (177 files, 15 min) and
`figures/f4_descriptors_x2_pooled/` (293 files, 21 min). The results are in
the runbook, section "The model as a hyperparameter".

**Warning: the caption NTA analysis compares each caption with each other
caption** (Levenshtein, `rapidfuzz.process.cdist`). The pool has 305,633
captions (canonical: about 30,700), thus about 100 times the work: about 9
x 10^10 pairs for the group of all models.

| Analysis | Command |
|----------|---------|
| Traces | `uvx marimo edit research/notebooks/facct/f4_traces.py` |
| Descriptors | `uvx marimo edit research/notebooks/facct/f4_descriptors.py` |
| Caption NTA | `uv run --script research/scripts/f4/caption_nta_analysis.py --captions data/f4_traces/pooled/x2/captions/f4_captions_x2_extractions.parquet --hashtags data/f4_traces/pooled/captions/f4_caption_hashtags.parquet --ntas data/f4_traces/pooled/f4_restaurants_nta.parquet --out data/f4_traces/pooled/captions/nta_analysis` |

The NTA tests of the traces notebook (section 4.10) need the pool. On the
pool, `min_images` is 20 by default. Read the test-1 `within`
column first. If it stays near 0, the NTA rates are too noisy for test 1.
The `nta_acs.parquet` feature table does not change with the pool.

The pool has 10 times the rows of the canonical tables. Expect the
notebooks to be slower. The text check of the uniqueness test
(`exclusive_heads`, `check_text=True`) searches the texts of the other
models for each candidate word. It is the slowest cell.

## 5b. Part D: the street captions (D133)

D133 adds the caption prompt on the street views (`street_caption`) to the
caption analysis. The canonical batch has 985 answers for each model, with no
extraction. Part D generates the 9,015 batch-2 street views, extracts the
captions of both batches, and pools them. The schema and the arm are those
of the restaurant captions (`x2_captions`, `caption-v4`).

| Script | Version | Change |
|--------|---------|--------|
| `build_extract_input.py` | 1.3.0 | writes `f4_street_captions.parquet` (the `street_caption` final answers) when the source holds that case |
| `build_x2_input.py` | 1.2.0 | `--street-captions` writes only `f4_street_captions_<suffix>.parquet` |
| `extract_shard.py` | 1.10.0 | `x2_captions` reads `street_captions_x2` and `street_captions_b2x2`; the retry arms read `scx2r_<arm>` |
| `x2_retry.py` | 1.2.0 | the same inputs; the prefix `scx2r` |
| `caption_hashtags.py` | 1.2.0 | `--case street_caption` (needs `--out`) |
| `caption_nta_analysis.py` | 1.4.0 | reads the street NTA columns (`nta2020`, `nta_name`, `borough`) |
| `pool_batches.py` | 1.1.0 | pools the street caption input, extraction and hashtags; `street_caption` must have 10,000 images |

### D1. Generate the batch-2 street captions

**Status: done on 2026-10-28.** Each model gives 9,015 answers with 0 errors
(`check_runs.py --runs data/f4_traces/runs-b2`). The jobs were these (`logs/f4/b2_street_caption_jobs.txt`):
Jobs 102420 to 102422 (gemma-4-12b, qwen3.5-9b, gemma-4-31b) are a chain on
QoS `nissenbaum`; jobs 102423 to 102425 (qwen3.8-27b, llama-3.2-11b,
llama-3.2-90b) are on the preemptible QoS `g3` with no dependency.

```bash
F4_SAMPLE=data/f4_traces/b2/f4_street_b2.parquet F4_TAG=b2 F4_CASES="street_caption" \
    bash research/scripts/f4/submit_all.sh
```

The answers go to `data/f4_traces/runs-b2/<model>/street_caption/`. Check with
`check_runs.py --runs data/f4_traces/runs-b2` (9,015 images, 0 errors).

### D2. The canonical street captions

**Status: input done; pass 1 submitted on 2026-10-28 (job 102427, QoS `g3`).**

```bash
$PY research/scripts/f4/build_extract_input.py --src data/f4_traces/f4_street_traces_1k.parquet \
    --out-dir data/f4_traces/street_captions/extract_input
$PY research/scripts/f4/build_x2_input.py --in-dir data/f4_traces/street_captions/extract_input --suffix x2 --street-captions
$PY research/scripts/f4/caption_hashtags.py --case street_caption --traces data/f4_traces/f4_street_traces_1k.parquet \
    --out data/f4_traces/street_captions/f4_street_caption_hashtags.parquet
sbatch --qos=g3 --exclude=g3-compute-17 --export=ALL,F4X2_PLAN=research/scripts/f4/plans/sc_x2_pass1_canonical.txt \
    research/scripts/f4/x2_queue.sbatch
```

The input holds 5,909 answers (1 empty answer dropped). The hashtags: 55,315
in 5,910 answers; llama-3.2-90b writes a hashtag in 506 of 985 answers.

### D3. The batch-2 street captions, after D1

**Status: steps 1 and 2 done; pass 1 submitted on 2026-10-28.**

- The merge adds 54,090 `street_caption` rows. The 54,090 `street` rows do
  not change.
- The input holds 54,082 answers. 8 empty answers are dropped.
- The hashtags: 462,365 hashtags, with 26,098 different tags.
- Pass 1 runs as 6 part plans (`sc_x2_pass1_b2_part00.txt` to `part05.txt`,
  every 6th step) on QoS `g3`, jobs 102592 to 102597
  (`logs/f4/sc_b2_pass1_jobs.txt`).

1. Merge the street corpus again with both cases (this replaces
   `f4_street_traces_b2.parquet`; its `street` rows do not change):

   ```bash
   $PY research/scripts/f4/merge_runs.py --set street --cases street street_caption \
       --sample data/f4_traces/b2/f4_street_b2.parquet --runs runs-b2 \
       --out data/f4_traces/b2/f4_street_traces_b2.parquet
   ```

2. Build the input and the hashtags:

   ```bash
   $PY research/scripts/f4/build_extract_input.py --src data/f4_traces/b2/f4_street_traces_b2.parquet \
       --out-dir data/f4_traces/b2/street_captions/extract_input
   $PY research/scripts/f4/build_x2_input.py --in-dir data/f4_traces/b2/street_captions/extract_input \
       --suffix b2x2 --street-captions
   $PY research/scripts/f4/caption_hashtags.py --case street_caption --traces data/f4_traces/b2/f4_street_traces_b2.parquet \
       --out data/f4_traces/b2/f4_street_caption_hashtags_b2.parquet
   ```

3. Pass 1 of batch 2: about 54,000 answers, 11 steps (plan
   `research/scripts/f4/plans/sc_x2_pass1_b2.txt`, lines
   `x2_captions street_captions_b2x2 <i> 11 multirun/f4x2_sc/x2_captions`).
   Split it into part plans for parallel `g3` jobs, as in B8.

### D4. Retry, merge, pool

**Status: done on 2026-10-29. The NTA analysis ended in 1 h 46 min
(job 102846, 64 CPUs, QoS `g3`; job 102844 failed because the batch shell has no `uv` on its PATH, thus give the full path to `uv`).**

| Step | Result |
|------|--------|
| Pass 1 | 13 steps with `X2_DONE` (jobs 102427, 102592 to 102597) |
| Flag | 4,467 of 59,991 texts (7.4%) to the retry (`f4_scx2r_x2_captions.parquet`) |
| Retry | job 102786, plan `sc_x2_retry.txt`; the merges read 421 + 4,046 = 4,467 retried texts |
| Copy suspects after the retry | 88 of 5,909 (canonical), 799 of 54,082 (batch 2): 887 of 59,991 (1.5%) |
| Copy suspects for each model (pool) | gemma-4-31b 466, gemma-4-12b 333, qwen3.5-9b 87, llama-3.2-90b 1, the others 0 |
| Pool | `pool_batches.py`: 18 files; the street-caption extraction holds 43,919 + 403,347 rows |

**The street captions do not pass the copy rule (about 1% or fewer).** The
restaurant captions give 251 of 60,000 (0.4%). Each suspect text holds the 3
spans of the street example of the schema ("114 Eldridge St. ..."). The
suspect texts are normal caption answers; their median length is near the
median of the other texts. The caption analysis drops the suspects. Thus
gemma-4-31b loses 4.7% of its texts and gemma-4-12b 3.3%. The user decides
if this is acceptable.

After every pass-1 step of D2 and D3 has `X2_DONE`:

```bash
$PY research/scripts/f4/x2_retry.py flag --root multirun/f4x2_sc --prefix scx2r
# 1 retry step: x2r_captions scx2r_x2_captions 0 1 multirun/f4x2_sc/x2r_captions
$PY research/scripts/f4/merge_extractions.py multirun/f4x2_sc/x2_captions/runs --retry-runs multirun/f4x2_sc/x2r_captions/runs \
    --schema dagspaces/pairwisevqa/conf/extract/caption_options_v4.yaml --cases street_captions_x2 \
    --out data/f4_traces/x2/street_captions
$PY research/scripts/f4/merge_extractions.py multirun/f4x2_sc/x2_captions/runs --retry-runs multirun/f4x2_sc/x2r_captions/runs \
    --schema dagspaces/pairwisevqa/conf/extract/caption_options_v4.yaml --cases street_captions_b2x2 \
    --out data/f4_traces/b2/x2/street_captions
$PY research/scripts/f4/pool_batches.py
uv run --script research/scripts/f4/caption_nta_analysis.py \
    --captions data/f4_traces/pooled/x2/street_captions/f4_street_captions_x2_extractions.parquet \
    --hashtags data/f4_traces/pooled/street_captions/f4_street_caption_hashtags.parquet \
    --ntas data/f4_traces/pooled/f4_street_1k.parquet --out data/f4_traces/pooled/street_captions/nta_analysis
```

Pass rules: as B10 (0 texts missing, quotable 99% or more, copy suspects
about 1% or fewer). Run the NTA analysis as a Slurm job with 64 CPUs, as job
101940: the pool holds about as many captions as the restaurant pool (job 101940 ran 1.8 h).

Then the caption notebook (`f4_captions.py`) gets a case selector:
`caption` (restaurant storefronts) or `street_caption` (street views).

## 6. Time budget

| Step | GPU time on 1 B300 (from the canonical jobs, times 9) |
|------|-------------------------------------------------------|
| B2, the street render | about 70 min of CPU, no GPU |
| B3, restaurant and caption chain | about 18 h: 1.5 to 2.5 h for each model, 7.5 h for llama-3.2-90b |
| B3, street chain (only `street`) | about 9 h |
| B4, the qwen3.5-9b rerun (about 1,400 answers at 16,384) | about 2 h |
| B8, extraction pass 1 (101 steps, about 500,000 texts) | about 36 to 50 h |
| B9, extraction pass 2 and merge | about 1 to 2 h |
| First job on a node | add about 25 min for the DeepSeek mirror (155 GB) and the venv mirror (5.2 GB) |
| Total | about 70 to 80 GPU-hours, thus 3 to 4 days |

The group QoS gives 1 GPU. Thus all GPU jobs run one after the other. A
job in `PENDING` that waits for the GPU of the group is not an error. The
outputs need a few GB of disk (canonical: about 0.5 GB).

## 7. Gotchas

| # | Problem | What to do |
|---|---------|------------|
| 1 | The default output path of each F4 script is a canonical file (`sample_images.py`, `sample_street_images.py`, `merge_runs.py`, `build_extract_input.py`, `restaurant_nta.py`, `caption_hashtags.py`, `caption_nta_analysis.py`) | Always give `--out` or `--out-dir` under `data/f4_traces/b2/` |
| 2 | The old draw scripts set `image_idx` from 0. Then `doc_id` is the same in both batches | Use `--idx-offset 1000` (A1, A2). `pool_batches.py` stops on a collision |
| 3 | The pooled restaurant sample is stratified, but the canonical sample is uniform | Use the weight `nta_capacity / nta_n` for a rate of the whole city. A rate inside 1 NTA needs no weight |
| 4 | The street rank depends on the seed and on the order of the `rng` calls | Keep the seed 20261023. Draw the extra NTAs from a second generator (A2). Check that the 985 canonical recordings are the first 5 of their NTA |
| 5 | `x2_retry.py flag --prefix x2r` overwrites the canonical retry inputs `f4_x2r_<arm>.parquet` | Use `--prefix b2x2r` |
| 6 | `x2_retry.py schemas` writes the `<schema>r.yaml` files again | Do not run it. The canonical retry schemas exist. Do not edit a schema file |
| 7 | g3-compute-17 is a B200 node (180 GB). llama-3.2-90b and DeepSeek fail there (no memory for the KV cache) | `submit_all.sh` excludes it for llama-3.2-90b. Give `--exclude=g3-compute-17` for each queue job |
| 8 | A generation job that reaches the 12-hour limit ends with `TIMEOUT` and does not requeue | Submit that model again with the same env. It resumes from `responses.jsonl` |
| 9 | The flag step runs even when a step of pass 1 failed | Use 2 plans (B7). Submit pass 2 only after all 101 steps have `X2_DONE` |
| 10 | A chain job starts after an earlier job ends the plan | Cancel it. It finds nothing to do, but it runs each `run` line again |
| 11 | Bash reads the plan file line by line while the job runs | You can append lines. Do not edit a line above the current step. To change `x2_queue.sbatch`, write a new file and move it into place |
| 12 | `/scratch` on a node can be full | `stage_inputs.sh` then reads the weights from NFS. The load is slower. This is not an error |
| 13 | The same request seed does not give the same tokens in a new server batch | A rerun is a new sample. Do not run a finished (model, case) again. A resume sends only the images that have no answer |
| 14 | A run must start again from zero | Move its folder to a new name. Do not delete it without the user's approval |
| 15 | The Llama context bug of the first canonical run (answers stop at 1,761 tokens) | `serve_and_query.sbatch` holds the fix (`MAX_LEN_MLLAMA`). Batch 2 needs no `runs-llamactx` |
| 16 | Mllama runs only on SGLang. `--chat-template` breaks it | Do not add server flags. The sbatch sets them |
| 17 | The Llama models give no reasoning text (D87) | This is expected. `build_extract_input.py` drops their empty reasoning field |
| 18 | `F4_MAX_TOKENS` is not in the export list of `submit_all.sh` | It works, because the export is `ALL`. Put it in front of the command (B4) |
| 19 | About 1% of the texts stay copy suspects after the retry (gemma texts with a full copy of an example) | Expected. The merge flags them, and the notebooks and the caption analysis drop them |
| 20 | `caption_nta_analysis.py` needs `rapidfuzz` | Run it with `uv run --script` |
| 21 | The manifest column `yawDegrees` holds no heading (values near 20,600) | The street script uses the CSCL bearing. Do not change this |
| 22 | `HF_HUB_OFFLINE=1` in the queue job | The model must be in the local registry. The job does not download weights |
| 23 | The notebooks export figures to fixed folders | In pooled mode, write to new folders (A11). The canonical figures must stay |
| 24 | The F6 hand check (Q63, `review100`) used images of `drawn_10000` only | The new restaurant images come from the same "Yes" pool. Tell the user if a hand check of the new images is necessary |
| 25 | `data/label_studio/credentials.env` and `.env` files | Never read or print them |
| 26 | `/scratch/mwf62/zoo/Qwen3.8-Flash-Next-FP8` | Do not delete it |

## 8. Output layout

| Path | Content |
|------|---------|
| `data/f4_traces/b2/f4_restaurants_b2.parquet`, `f4_street_b2.parquet` | the batch 2 images (`image_idx` 1000 to 9999, and 1000 to 10014 for the street views) |
| `data/f4_traces/b2/f4_restaurants_b2_nta.parquet` | the NTA of each new restaurant image |
| `data/f4_traces/street_views/` | the street views of both batches |
| `data/f4_traces/runs-b2/`, `runs-b2-maxtok16384/` | the answers |
| `data/f4_traces/b2/f4_traces_b2.parquet`, `f4_street_traces_b2.parquet` | the batch 2 corpora |
| `data/f4_traces/b2/extract_input/` | the batch 2 extraction inputs (before x2) |
| `data/f4_traces/extract_input/f4_*_b2x2.parquet`, `f4_b2x2r_*.parquet` | the x2 inputs and the retry inputs |
| `research/scripts/f4/plans/b2_x2_pass1.txt`, `b2_x2_pass2.txt` | the queue plans |
| `multirun/f4x2_b2/` | the extraction runs |
| `data/f4_traces/b2/x2/<arm>/` | the merged batch 2 extractions |
| `data/f4_traces/pooled/` | the pool of the 2 batches |

When batch 2 ends, add a section to `vlm-narratives-docs/facct-runbook.md`
with the job ids, the B3 table, and the B10 table.
