# Structured trace extraction with LangExtract

This page gives the plan that replaces the word cloud in `notebooks/cvpr` with
typed, grounded extractions. It is a plan, not a record of work. No code exists
yet.

Read [reasoning-trace-analysis.md](reasoning-trace-analysis.md) first. It says
where the traces are and which runs are usable.

## Why we change the method

The word cloud counts words. The paper needs claims. A trace says which image
holds a cue, whether the cue is good or bad, and whether the cue moved the
judgment. A word count throws all 3 away.

| Question the paper asks | Word cloud | Typed extraction |
|-------------------------|-----------|------------------|
| Which cues does the model name? | partly | yes |
| Does the cue belong to image A or image B? | no | yes |
| Is the cue good or bad in the model's reading? | no | yes |
| Does the model infer wealth, class, or crime? | no | yes |
| Does a cue predict the label? | no | yes |
| Can I quote the sentence in the paper? | no | yes, with offsets |

LangExtract adds 4 things we do not want to write again: a few-shot prompt
builder, a tolerant JSON parser, a fuzzy aligner that maps each extraction back
to a character span of the source trace, and an HTML view of the spans.

## What we measured before we planned

Checked on 2026-08-13, with `langextract==1.6.0` (Apache-2.0).

| Fact | Value |
|------|-------|
| Install without the Google stack | 4 packages: `langextract`, `absl-py`, `ml-collections`, `more-itertools` |
| `google.genai` at import time | never imported; the providers load by name, on demand |
| A custom engine plugs in | `lx.extract(..., model=<BaseLanguageModel>)` works |
| Alignment | returns `CharInterval` and `MATCH_EXACT` on our trace text |
| Trace length, subway 11k run | mean 3,568 chars, median 3,258, p90 5,751, p99 8,973, max 19,252 |

**Warning: a plain `uv pip install langextract` downgrades `websockets` from
17.0.1 to 16.1.1** in `.venv-mllmsci-vllm025cu129`. The pull comes from
`google-genai`, which we never call. Install with `--no-deps` and name the 3
extra packages. See Phase 0.

## The architecture decision

LangExtract must not drive the GPU. Our stage layer already owns batches,
chunk streaming, data-parallel shards, W&B, and the SLURM launchers.

| Option | How it works | Verdict |
|--------|--------------|---------|
| A. LangExtract talks to a vLLM server | `_resolve_server_url` and `VLLM_SERVER_URL` already exist in `vllm_inference.py` | Keep for a laptop-scale debug run only. A server is a second job to watch, and it gives no shard. |
| B. A custom `BaseLanguageModel` that calls an in-process vLLM engine | `lx.extract` hands `infer()` a whole batch of prompts. The adapter calls `LLM.generate` one time for that batch. | **Chosen.** Smallest code, supported API, full batch. |
| C. LangExtract as prompt builder and resolver only, around `run_vllm_inference` | Most control. Uses `prompting`, `chunking`, and `resolver` directly. | Fallback. Take it only if option B costs too much throughput. |

Option B keeps us on the public API. The resume problem that option B does not
solve, the stage solves: the stage cuts the trace table into row ranges and
writes 1 parquet for each range.

## The parts to build

| File | State | Role |
|------|-------|------|
| `dagspaces/common/langextract_backend.py` | new | `VLLMLanguageModel`, the schema bridge, and the example loader |
| `dagspaces/urbanpairvqa/stages/trace_extract.py` | new | The stage: read traces, extract, write the long parquet |
| `dagspaces/urbanpairvqa/stages/__init__.py` | change | Export the stage function |
| `dagspaces/urbanpairvqa/orchestrator.py` | change | Add `"trace_extract": TraceExtractRunner()` to `_STAGE_REGISTRY` |
| `dagspaces/urbanpairvqa/conf/extract/urban_cues_v1.yaml` | new | The schema, the few-shot examples, and the chunk settings |
| `dagspaces/urbanpairvqa/conf/pipeline/extract_traces.yaml` | new | The 1-node pipeline, on `slurm_gpu_1x` |
| `notebooks/cvpr/_extractions.py` | new | Load the extractions, count them, and score them |
| `notebooks/cvpr/_trace_notebook.py` | change | Add the extraction report beside the cloud |
| `pyproject.toml` | change | Add `langextract`, `absl-py`, `ml-collections`, `more-itertools` |

The stage lives in `urbanpairvqa` because its input is a pairvqa results
parquet. A new dagspace is not needed. 1 stage and 1 config group is the whole
surface.

## The adapter

```python
class VLLMLanguageModel(BaseLanguageModel):
    """Run a LangExtract batch on the in-process vLLM engine."""

    def infer(self, batch_prompts, **kwargs):
        # 1 generate() call for the whole batch. Ignore max_workers:
        # the engine schedules the batch itself.
        ...
```

3 rules for the adapter:

1. Ignore `max_workers`. Thread parallelism is wrong for 1 local engine.
2. Set `temperature=0.0`. The extractor tags text; it does not write prose.
3. Build `guided_json` from `GeminiSchema.from_examples(examples).schema_dict`.
   That dict is a plain JSON schema, so vLLM guided decoding accepts it. Guided
   decoding removes almost every parse failure.

**Warning:** `lx.extract` warns when `batch_length < max_workers`. Set
`max_workers=1` and `batch_length=512` to silence it and to state the truth.

## The extraction schema, version 1

The schema follows the shape that the traces already have. A gemma-4-12b subway
trace names the image, lists what it sees, compares the two, then decides.

| Class | What it captures | Attributes |
|-------|------------------|------------|
| `visual_evidence` | A thing the model says it sees | `image` (A, B), `valence` (good, bad, neutral), `category` (people, cleanliness, upkeep, greenery, light, signage, traffic, construction, commerce, architecture) |
| `inference` | A claim the pixels do not hold | `image`, `kind` (safety, quality, upkeep, wealth, class, crime, demographic) |
| `person_reference` | A statement about the people present | `image`, `used_in_judgment` (yes, no) |
| `image_artifact` | A statement about the photo, not the place | `image`, `kind` (blur, angle, occlusion, time of day, weather) |
| `comparison` | An explicit A-against-B statement | `direction` (A, B, neither), `dimension`, `hedged` (yes, no) |
| `uncertainty` | Hedge or abstention language | `reason` (looks equal, cannot see, out of view) |
| `decision` | The sentence that states the answer | `label` |

`person_reference` and `inference[kind=wealth|class|demographic]` carry the risk
result that the framework exists to find. A word cloud cannot count them,
because the words are ordinary.

Version the schema. Write `schema_version` into every output row. A schema
change makes a new version, never an edit in place.

## Config that matters

| Setting | Value | Why |
|---------|-------|-----|
| `max_char_buffer` | 12000 | 1 trace must be 1 chunk. p99 is 8,973 chars. A split trace breaks the A-against-B binding, which is the point of the schema. |
| `batch_length` | 512 | The batch that reaches `generate()` at one time. |
| `max_workers` | 1 | The engine schedules. Threads do not help. |
| `temperature` | 0.0 | A tagger is not a generator. |
| `extraction_passes` | 1 | Raise it only if the audit shows low recall. Each pass costs a full run. |
| `fence_output` | false | Guided decoding gives raw JSON. |
| `max_tokens` | 8192 | A cut answer loses the whole trace, not its tail. See silent failure 2. |
| `require_extraction` | true | `minItems: 1` stops the empty answer. See silent failure 1. |
| `max_model_len` | 16384 | The prompt reaches about 7,000 tokens, and the answer 8,192 more. The model default of 6,144 is for an image prompt. |

**Warning: the default `max_char_buffer` is 1000.** At that value a median trace
becomes 4 chunks and 4 model calls, and no chunk sees both images. Set the value
in the config, and do not accept the default.

**Warning: a trace repeats the prompt.** The model writes "The user wants me to
compare the safety of two subway station entrances". That is prompt echo, not
evidence. Put a negative example in the few-shot set that shows the echo and
extracts nothing from it.

**Warning: the built-in example check raises on a negative example.**
`lx.extract` aligns each example span against its own example text before it
runs. That check is worth keeping, because it catches a paraphrased span, which
can never align. But its aligner raises `ValueError` when an example holds no
span, and a negative example holds none on purpose.

`langextract_backend.validate_examples` runs the same check over the examples
that have a span, and `extract_documents` passes
`prompt_validation_level=OFF` to turn the built-in one off. Call
`validate_examples` before any run. It returns 1 message for each problem, and
an empty list when every span aligns exactly.

## The output contract

1 row for each extraction. Long format. The file joins back to the results
parquet on `pair_id`.

| Column | Source |
|--------|--------|
| `pair_id`, `presented_label`, `presented_score` | The results parquet |
| `case`, `judge_model`, `sweep`, `wandb_id`, `question` | The run record from `_traces.TraceRun` |
| `extraction_index`, `extraction_class`, `extraction_text` | LangExtract |
| `attributes_json` | LangExtract, as a JSON string |
| `char_start`, `char_end`, `alignment_status` | LangExtract |
| `is_quotable` | The stage: true for `match_exact` and `match_fuzzy` only |
| `extractor_model`, `schema_version`, `extract_run_id` | The stage |

A trace with no extraction still writes 1 row with a null class. Without it a
reader cannot separate "the model said nothing" from "the stage did not run".

## Phases

### Phase 0 — Environment — DONE 2026-08-13

1. Installed into `.venv-mllmsci-vllm025cu129`:
   `uv pip install --no-deps langextract absl-py ml-collections more-itertools`.
2. `import langextract` loads no Google module, and `websockets` stays at
   17.0.1.
3. `pyproject.toml` carries the group `langextract`, with the `--no-deps` rule
   in a comment.

### Phase 1 — The adapter — DONE 2026-08-13

`dagspaces/common/langextract_backend.py` holds:

| Part | Role |
|------|------|
| `ExtractionSpec` | The task: the description, the examples, the buffer, and the schema version |
| `spec_from_config` | Reads a `conf/extract/*.yaml` group |
| `validate_examples` | Tests each example span against its own text |
| `VLLMLanguageModel` | The LangExtract model class; 1 engine call for each batch |
| `VLLMEngine` | Builds the engine, renders the chat template, strips a thought block, and shuts the workers down |
| `extract_documents` | The `lx.extract` call, with our settings |
| `annotated_to_rows` | The long output rows |

`tests/test_langextract_backend.py` holds 18 tests. They use a stub engine, so
they need no GPU. They cover the batch shape, the length-mismatch guard, the
guided-JSON dict, the example checks, the char intervals, and the empty-trace
row.

### Phase 2 — The stage — DONE 2026-08-13

| File | Role |
|------|------|
| `dagspaces/urbanpairvqa/stages/trace_extract.py` | The stage: read, shard, extract, write |
| `dagspaces/urbanpairvqa/orchestrator.py` | `TraceExtractRunner`, in the registry as `trace_extract` |
| `dagspaces/urbanpairvqa/conf/extract/urban_cues_v1.yaml` | The 7 classes, 3 examples, and the chunk settings |
| `dagspaces/urbanpairvqa/conf/pipeline/extract_traces.yaml` | The 1-node pipeline, on `slurm_gpu_1x` |

Command:

```bash
python -m dagspaces.urbanpairvqa.cli -m pipeline=extract_traces \
    trace_extract.results_path=/share/.../subway_safety_mvp_20260813_013722.parquet \
    runtime.sample_n=200
```

The 200-trace smoke run on the subway thinking run of 2026-08-13, with
qwen3.5-9b as the extractor:

| Measure | Value |
|---------|-------|
| Extractions | 8,128 from 200 traces, 40.6 for each trace |
| Traces with no extraction | 0 |
| Quotable (`match_exact` or `match_fuzzy`) | 91.7% |
| Answers that reached the token cap | 11, and the repair saved all 11 |
| Time | 90 s to load the engine, 605 s for 200 traces |

Every one of the 7 classes appears.

**Warning: a plain install is not enough. Sync the venv mirror.** A stage job
starts from the node-local `/scratch` mirror, thus a package that reaches only
the NFS venv is absent at run time. The first smoke run stopped with
`ModuleNotFoundError: No module named 'langextract'` on klara. Run this after
any install, as an sbatch job pinned to the node:

```bash
sbatch --partition=pierson -w klara --wrap="bash scripts/sync_venv_to_scratch.sh"
```

See [scratch-mirrors.md](scratch-mirrors.md).

### Phase 3 — The full extraction — DONE 2026-08-14

The run started 02:01 and ended 06:51: 4 cases, 6 shards for each, 24 GPU jobs,
gemma-4-12b as the extractor.

| Case | Traces | Extractions | For each trace | Quotable |
|------|--------|-------------|----------------|----------|
| subway_safety | 11,000 | 275,233 | 25.0 | 97.7% |
| libraries | 11,000 | 234,750 | 21.3 | 98.3% |
| schools | 11,000 | 219,437 | 19.9 | 98.3% |
| road_quality | 11,000 | 215,631 | 19.6 | 99.2% |
| **Total** | **44,000** | **945,051** | | |

No trace came back silent. No trace was read two times. 107 answers of 44,000
reached the token cap, and the repair saved all 107.

`scripts/merge_trace_extractions.py` joins the 24 shards into 1 parquet for
each case, under `data/trace_extractions/`. It tests the coverage and never
repairs a gap.

```bash
python scripts/merge_trace_extractions.py \
    multirun/2026-08-14_URBANPAIRVQA/02-01-38 --out data/trace_extractions
```

**Warning: `array_parallelism` is 1 on every launcher.** That default is right
for a model sweep, where 2 jobs would fight for the same GPU. It is wrong for
shards: the 24 jobs then run one after the other, and the run takes 36 hours.
Pass `hydra.launcher.array_parallelism=6` to match the free GPUs.

**Warning: an attribute value is not an enum.** The guided schema fixes the
class names and the attribute NAMES, but an attribute value is a free string.
The model wrote `kind` values that the prompt never lists, such as
`architecture`, `prestige`, and `cleanliness`, and `other` reaches 128 for each
100 schools traces. Thus the analysis must normalise the values, and it must
report what falls in `other`. See Phase 4.

### Phase 3 — the original plan

Run the 4 thinking runs of 2026-08-13, and each earlier run that passes the
consolidation-date rule.

| Run | Rows | Est. model calls |
|-----|------|------------------|
| subway safety, gemma-4-12b | 11,000 | 11,000 |
| libraries, gemma-4-12b | 11,000 | 11,000 |
| schools, gemma-4-12b | 11,000 | 11,000 |
| road quality, gemma-4-12b | 11,000 | 11,000 |

About 2,500 prompt tokens and 2,600 output tokens for each call. That is 5x the
output the first plan assumed, thus 1 case costs about 7 GPU-hours. Run each
case as 8 shards to get it in about 55 minutes. See "Speed" above.

Checkpoint: 4 parquets on disk, and the run table names each source run.

### Phase 4 — The notebook report — DONE 2026-08-14

| File | Role |
|------|------|
| `notebooks/cvpr/_extractions.py` | Load, normalise, rate, score, quote, and export |
| `notebooks/cvpr/_gen_trace_notebooks.py` | Section 4 of the template |
| `notebooks/cvpr/<case>/<case>_traces.py` | The 4 notebooks, at version 1.1.0 |
| `tests/test_extractions.py` | 14 tests over the rate, the vocabulary, and the unit |

Section 4 of each trace notebook holds: the coverage table, the class mix by
case, the risk panel, the distinctive-claim table, the vocabulary report, and a
quote table. The export button writes them beside the word-cloud files, in that
prompt's `figures/` folder.

Proved on 2026-08-14: `marimo export html` runs the subway notebook end to end,
with 3 figures and the real numbers.

The word sections stay. They answer a different question, and a notebook that
finds no extraction parquet still draws its clouds.

### Phase 4 — the original plan

1. Write `notebooks/cvpr/_extractions.py`.
2. Add the report to `_trace_notebook.py`, then run
   `_gen_trace_notebooks.py`.

Reuse, do not rewrite:

- `_traces.discover_trace_runs` finds the runs. Discovery, the era rule, the
  `MIN_ROWS` rule, and the `presplit` rule all stay.
- `_traces.mixed_question_cases` still splits a case that asked 2 questions.
- `_traces.distinctive_scores` takes any `Counter`. Feed it a counter of
  `class:attribute` keys instead of words. The log-odds score with the Dirichlet
  prior works the same on either unit.
- `_style.py` keeps the palette.

New figures:

| Figure | Reads |
|--------|-------|
| Cue category by case | Which cues each question calls up |
| Valence by image and by label | Whether the model's cues agree with its answer |
| Risk panel | The rate of `person_reference` and of a wealth, class, or demographic `inference` |
| Grounded quote table | 10 quotes for each class, with `pair_id` and offsets |

Keep the word clouds. They stay valid, and the paper may still use them.

### Phase 5 — Validation

The extraction is a measurement. It needs error bars.

| Test | Method | Report |
|------|--------|--------|
| Grounding | Count `alignment_status` | The share that is `MATCH_EXACT` |
| Human audit | 100 traces, 2 people, the same schema | Precision and recall for each class |
| Self-consistency | Extract 500 traces 2 times | Jaccard of the 2 extraction sets |
| Extractor swap | Extract 1,000 traces with qwen3.5-9b and with gemma-4-12b | The correlation of the class counts |

The extractor swap answers the first reviewer question: does the finding come
from the judge model, or from the extractor?

## Speed

Measured on 1 A6000 on klara, over the same 200 subway traces, on 2026-08-14.
The work is decode-bound: the model writes about 2,600 tokens for each trace.

| Setting | Time for 200 traces | Against the baseline | Verdict |
|---------|--------------------|----------------------|---------|
| Baseline: `max_num_seqs=64`, no prefix cache | 605 s | — | |
| **`enable_prefix_caching=true`, `max_num_seqs=128`** | **467 s** | **1.4x** | **Kept** |
| Guided JSON off | 345 s | 1.36x more | **Rejected** |
| qwen3.5-4b instead of 9b | 412 s | 1.13x more | Not now |

**Prefix caching is the free win.** Every prompt starts with the same 2,000
tokens: the instructions and the 3 examples. Only the trace differs. Without
the cache the engine prefills those 2,000 tokens again for each of 11,000
traces. The pairvqa model configs leave the cache off, and rightly so — an
image prompt shares no prefix. A text prompt with a fixed preamble is the
opposite case.

**Do not turn guided JSON off to go faster.** It is 1.36x faster and it brings
back the empty answer: 70 of 200 traces came back silent, because `minItems: 1`
lives in the schema. Speed that loses a third of the data is not speed.

**qwen3.5-4b is not the win it looks like.** It decodes 1.5x faster for each
token, but it writes 59 extractions for each trace against 44, so the wall
clock falls by only 13%. Its extra output is over-extraction, not more signal.
Judge it in Phase 5 on quality, not here on speed.

### Shards are the real lever

No trace needs another trace, thus the work splits perfectly.
`trace_extract.shard_count` and `shard_index` cut the table with a stride, and
a Hydra multirun runs 1 job for each shard on its own GPU.

```bash
python -m dagspaces.urbanpairvqa.cli -m pipeline=extract_traces \
    trace_extract.results_path=/share/.../subway_safety_mvp_20260813_013722.parquet \
    trace_extract.shard_count=8 \
    trace_extract.shard_index=0,1,2,3,4,5,6,7
```

| Layout | Wall clock for 1 case of 11,000 traces |
|--------|----------------------------------------|
| 1 GPU, baseline | about 9.2 h |
| 1 GPU, prefix cache | about 7.1 h |
| 8 GPUs, prefix cache | about 55 min |

Proved on 2 shards of 100 traces on 2026-08-14: the 2 jobs ran together, they
shared no trace, and together they held every trace.

The stride matters. A contiguous cut puts the long traces of 1 region in 1 job,
and that job then runs much longer than the others.

**Sharding buys wall clock, not GPU hours.** The total stays near 7 GPU-hours
for each case. The only way to spend less is to extract less, and that is a
question about the science, not about the engine.

## The 3 silent failures that Phase 2 found

Each one loses data without an error. Each one now has a number in the
metadata. Watch all 3 on every run.

### 1. An empty answer

The model answers `{"extractions": []}` in about 11 tokens. At the first
setting, 113 of 200 traces came back this way, and a long trace failed far more
often: the silent traces had a median of 4,487 characters against 2,785 for the
rest.

The fix is a lower bound in the schema. `require_extraction: true` puts
`minItems: 1` on the array, thus the grammar cannot write an empty one. That
took the count to 0.

### 2. A cut answer

Guided decoding writes valid JSON up to `max_tokens`. The cap then cuts it, and
the parser drops the WHOLE answer — every extraction, not only the last. The
log line is `Skipping chunk: parse error`.

Two defences: `max_tokens: 8192`, and `repair_truncated_json`, which cuts back
to the last extraction that closed and shuts the array. `answers_truncated` and
`answers_repaired` report both.

### 3. A composed sentence with a fragment offset

This is the subtle one. The aligner reports `match_lesser` when the model
builds a sentence out of the text plus its own words. The row looks grounded,
and its offsets point at a fragment.

| Extracted | Aligned to |
|-----------|-----------|
| "Same is often the best fit for very similar urban environments" | "Same" |
| "Image B looks definitely safer for pedestrians" | "Image B looks" |
| "safety is subjective" | "safety" |

`match_fuzzy` is NOT this problem. A fuzzy row holds the source text word for
word; the aligner only took the tolerant path because the words repeat.

Thus the rule: **count `match_exact` and `match_fuzzy`, and drop
`match_lesser`.** The `is_quotable` column holds that test, and
`quotable_rate` reports it. It was 91.7% on the smoke run.

## Traps

**An extraction describes the text, not the image.** The model can name a trash
can that is not there. A count of `visual_evidence` is a count of what the model
says. Write that limit into any claim.

**The judge model and the extractor model are different roles.** The extractor
tags text and never scores an image. Do not report an extractor label as a
judgment.

**The era rule still holds.** A run from before 2026-08-11 repeats the persona
and the cue list of the old prompt. Its extractions describe the prompt. The
date filter in `_traces.discover_trace_runs` already removes it.

**The 18-row subway runs stay out.** They are the layout probes that found the
image-binding bug. `MIN_ROWS` removes them.

**gemma-4-12b needs `interleaved_labels`.** A trace from the broken layout says
the model saw 1 image. Extraction cannot repair that.

**Never glob `multirun/`.** Discovery goes through W&B.

**An ungrounded extraction is a defect, not data.** Drop it from the counts, and
report the rate you dropped.

## Open choices

| Choice | Recommendation |
|--------|----------------|
| Which model extracts | gemma-4-12b instruct, text only. It already runs in the venv, and Phase 5 tests the swap. |
| Where the stage lives | `urbanpairvqa`. Its input is a pairvqa parquet. |
| Whether to keep the clouds | Keep them. The 2 methods answer different questions. |
