# Structured trace extraction with LangExtract

This page records the extraction that replaces the word cloud in
`research/notebooks/cvpr` with typed, grounded extractions. The first sections give
the design. The phases give what ran and what it measured. The section
"Review of 2026-09-20" gives what the review found, and "Version 2" gives
what changed because of it.

Read [reasoning-trace-analysis.md](reasoning-trace-analysis.md) first. It says
where the traces are and which runs are usable.

**Warning: read "Review of 2026-09-20" before you quote a rate from a version 1
corpus.** The class rates of `uncertainty`, `inference`, and `image_artifact`
change 2x to 10x when the extractor changes and the traces do not.

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
| `dagspaces/pairwisevqa/stages/trace_extract.py` | new | The stage: read traces, extract, write the long parquet |
| `dagspaces/pairwisevqa/stages/__init__.py` | change | Export the stage function |
| `dagspaces/pairwisevqa/orchestrator.py` | change | Add `"trace_extract": TraceExtractRunner()` to `_STAGE_REGISTRY` |
| `research/conf/pairwisevqa/extract/urban_cues_v1.yaml` | new | The schema, the few-shot examples, and the chunk settings |
| `research/conf/pairwisevqa/pipeline/extract_traces.yaml` | new | The 1-node pipeline, on `slurm_gpu_1x` |
| `research/notebooks/cvpr/_extractions.py` | new | Load the extractions, count them, and score them |
| `research/notebooks/cvpr/_trace_notebook.py` | change | Add the extraction report beside the cloud |
| `pyproject.toml` | change | Add `langextract`, `absl-py`, `ml-collections`, `more-itertools` |

The stage lives in `pairwisevqa` because its input is a pairvqa results
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

### Settings of backend 1.2.0 and 1.3.0 (F4 x2 round, D130)

The single-image F4 schemas (`single-v2`, `street-v3`, `descriptors-v6`,
`caption-v4`) do not need the A-against-B binding. Thus they cut a long
trace into chunks, and use these settings:

| Setting | Value | Why |
|---------|-------|-----|
| `max_char_buffer` | 6000 | Long qwen3.5-9b traces made the extractor loop to the token limit. A chunk of 6,000 characters stops the loop, and the tail of a trace gets spans (about 40% of the spans of a long trace lie in its second half). |
| `context_window_chars` | 600 | LangExtract puts the last 600 characters of the chunk before it in front of a chunk ("[Previous text]"). |
| `max_tokens` | 6144 | 1 chunk needs less. |
| `rescue_substrings` | true | The exact phase of the 1.6.0 aligner runs 1 difflib match over all spans; a span whose block starts inside it gets a 1-token interval at a wrong place (`match_lesser`). `rescue_offsets` finds the span in the text again. |
| `trace_extract.context_column` | `context` | The stage passes the column to `Document.additional_context`: 1 line that names the field, the model, and the prompt. LangExtract puts it after the description, before the examples. |

`rescue_offsets(text, span, prefer_from, hint)` gives `(start, end, mode)`.
It tries the modes in this order:

| Mode | Search |
|------|--------|
| `exact` | the span as it is |
| `casefold` | the span with no case |
| `markup` (1.3.0) | no case, no Markdown marks (`*`, `_`, `` ` ``, `#`), 1 space for each run of white space; the extractor drops the bold marks inside a span |

A rescued row has `alignment_rescue` set, `is_quotable` true, and
`is_verbatim` true only for `exact`.

**Warning: the examples teach more than the rules.** The extractor sometimes
gives the answer of 1 example in place of an answer for the input (LangExtract
issue 209). A rule cannot stop this. `research/scripts/f4/x2_retry.py` flags these
texts and reads them again with the examples in reverse order. See the x2
section of `facct-runbook.md`.

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
| `dagspaces/pairwisevqa/stages/trace_extract.py` | The stage: read, shard, extract, write |
| `dagspaces/pairwisevqa/orchestrator.py` | `TraceExtractRunner`, in the registry as `trace_extract` |
| `research/conf/pairwisevqa/extract/urban_cues_v1.yaml` | The 7 classes, 3 examples, and the chunk settings |
| `research/conf/pairwisevqa/pipeline/extract_traces.yaml` | The 1-node pipeline, on `slurm_gpu_1x` |

Command:

```bash
python -m dagspaces.pairwisevqa.cli -m pipeline=extract_traces \
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

See [models-and-mirrors.md](../../docs/models-and-mirrors.md).

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

`research/scripts/merge_trace_extractions.py` joins the 24 shards into 1 parquet for
each case, under `data/trace_extractions/`. It tests the coverage and never
repairs a gap.

```bash
python research/scripts/merge_trace_extractions.py \
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
| `research/notebooks/cvpr/_extractions.py` | Load, normalise, rate, score, quote, and export |
| `research/notebooks/cvpr/_gen_trace_notebooks.py` | Section 4 of the template |
| `research/notebooks/cvpr/<case>/<case>_traces.py` | The 4 notebooks, at version 1.1.0 |
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

1. Write `research/notebooks/cvpr/_extractions.py`.
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

### Phase 6 — The qwen3.5-9b half — DONE 2026-09-05

The corpus of 2026-08-28 reads the gemma-4-12b judge alone, with qwen3.5-9b
as the extractor. The qwen3.5-9b judge takes gemma-4-12b as the extractor,
because the extractor and the judge must differ. Thus the extractor differs
between the 2 halves, and a difference between the halves can come from the
judge or from the extractor. The Phase 5 swap test on 1,000 traces is the
way to separate the 2.

| | The gemma half | The qwen half |
|---|---|---|
| Judge | `gemma-4-12b/instruct_thinking` | `qwen3.5-9b/instruct_thinking` |
| Extractor | qwen3.5-9b | gemma-4-12b |
| Source runs | the registered trace runs | the registered trace runs |
| Experiment name | `CUESV1` | `CUESV1Q` |
| Merged corpus | `data/trace_extractions/` | `data/trace_extractions_qwen9b/` |

The 2 corpora sit in 2 roots on purpose. `merge_trace_extractions.py` names
a file by its case alone, thus 1 root can hold 1 judge for each case. The
readers take the root as an argument: `_extractions.load(root=...)`.

| Script | Role |
|--------|------|
| `research/scripts/launch_trace_extraction.sh qwen_by_gemma` | 7 multiruns, 1 for each case, 8 shards each, 120 s apart |
| `research/scripts/check_trace_extraction.sh CUESV1Q` | counts the monitors that landed a job and the shards that finished |

The launch: `model=gemma-4-12b/instruct`, `slurm_gpu_preempt`, 8 shards of
1,375 traces, chunks of 250 rows. After the last shard finishes:

```bash
python research/scripts/merge_trace_extractions.py multirun/<date>_CUESV1Q/* \
    --out data/trace_extractions_qwen9b
```

The merge prints a registry note for each case. Every case must read "from
the registered run, OK".

The run started 13:58 and the last shard ended before 19:00: 56 jobs on the
`gpu` partition, every one landed, 6 of them restarted once after a
preemption and 1 twice, no failure. The merge of the same day:

| Case | Traces | Extractions | For each trace | Quotable |
|------|--------|-------------|----------------|----------|
| subway_safety | 11,000 | 492,065 | 44.7 | 94.9% |
| libraries | 11,000 | 349,119 | 31.7 | 95.8% |
| schools | 11,000 | 362,211 | 32.9 | 96.0% |
| road_quality | 11,000 | 342,477 | 31.1 | 96.5% |
| parks_plazas | 11,000 | 383,356 | 34.9 | 95.9% |
| restaurants | 11,000 | 344,656 | 31.3 | 94.7% |
| street_photography | 11,000 | 273,885 | 24.9 | 95.8% |
| **Total** | **77,000** | **2,547,769** | | |

No trace came back silent, and no trace was read two times. Every case
reads "from the registered run, OK", and `_extractions.registry_mismatch`
on the qwen root is empty. The gemma extractor writes about 10% more spans
for each trace than the qwen extractor did on the gemma half, with a
quotable rate 4 to 7 points higher.

### Phase 7 — The 2 self-extraction arms — DONE 2026-09-08

Phases 3 and 6 cross the judge and the extractor, thus a difference between
the 2 halves can come from either role. Two more arms close the 2 x 2: each
judge's traces extracted by the judge itself.

| Arm | Judge | Extractor | Experiment | Corpus root |
|-----|-------|-----------|------------|-------------|
| gemma by qwen (2026-08-28) | gemma-4-12b | qwen3.5-9b | `CUESV1` | `data/trace_extractions/` |
| qwen by gemma (Phase 6) | qwen3.5-9b | gemma-4-12b | `CUESV1Q` | `data/trace_extractions_qwen9b/` |
| gemma self | gemma-4-12b | gemma-4-12b | `CUESV1GG` | `data/trace_extractions_gemma12b_self/` |
| qwen self | qwen3.5-9b | qwen3.5-9b | `CUESV1QQ` | `data/trace_extractions_qwen9b_self/` |

With the 4 cells, a class rate reads 2 ways: down a column, the judge
changes and the extractor holds; along a row of the same judge, the
extractor changes and the traces hold. The self cells also answer the echo
question directly: an extractor that reads its own traces can be compared
with the other extractor on the same traces.

Every arm runs the same way: `research/scripts/launch_trace_extraction.sh <arm>`, 7
multiruns 120 s apart, 8 shards of 1,375 traces, chunks of 250 rows, the
`slurm_gpu_preempt` launcher. `research/scripts/check_trace_extraction.sh
<experiment>` counts the landed monitors and the finished shards. The 2
self arms started on 2026-09-05 at 19:12, one after the other. The merge
of each arm goes to its own root:

```bash
python research/scripts/merge_trace_extractions.py multirun/2026-09-05_CUESV1GG/* \
    --out data/trace_extractions_gemma12b_self
python research/scripts/merge_trace_extractions.py multirun/2026-09-05_CUESV1QQ/* \
    --out data/trace_extractions_qwen9b_self
```

**Warning: a root holds 1 judge for each case.** The merge names a file by
its case alone, so a merge into a root that another arm owns overwrites it.

Both arms finished on 2026-09-05: 112 shards, every one landed, no failure.
The 2 merge jobs that waited on the shards with `--dependency=afterok`
never ran: SLURM cancelled both with "Dependency", because a requeued shard
does not end in the state `afterok` asks for. The merges ran by hand on
2026-09-08, and both roots pass `registry_mismatch`.

| Arm | Traces | Extractions | For each trace | Quotable |
|-----|--------|-------------|----------------|----------|
| gemma self, subway_safety | 11,000 | 275,978 | 25.1 | 97.9% |
| gemma self, libraries | 11,000 | 205,709 | 18.7 | 98.2% |
| gemma self, schools | 11,000 | 215,835 | 19.6 | 98.2% |
| gemma self, road_quality | 11,000 | 213,214 | 19.4 | 99.1% |
| gemma self, parks_plazas | 11,000 | 218,949 | 19.9 | 97.9% |
| gemma self, restaurants | 11,000 | 193,332 | 17.6 | 97.6% |
| gemma self, street_photography | 11,000 | 223,694 | 20.3 | 97.6% |
| qwen self, subway_safety | 11,000 | 865,185 | 78.7 | 89.2% |
| qwen self, libraries | 11,000 | 497,425 | 45.2 | 85.4% |
| qwen self, schools | 11,000 | 618,884 | 56.3 | 86.9% |
| qwen self, road_quality | 11,000 | 490,934 | 44.6 | 94.5% |
| qwen self, parks_plazas | 11,000 | 615,238 | 55.9 | 89.4% |
| qwen self, restaurants | 11,000 | 701,452 | 63.8 | 87.6% |
| qwen self, street_photography | 11,000 | 416,103 | 37.8 | 90.1% |

No silent trace and no repeated trace in either arm. The 4 cells of the
2 x 2, spans for each trace over the 7 cases:

| Judge \ Extractor | qwen3.5-9b | gemma-4-12b |
|---|---|---|
| gemma-4-12b | 26.9 to 43.7 | 17.6 to 25.1 |
| qwen3.5-9b | 37.8 to 78.7 | 24.9 to 44.7 |

**Warning: do not wait on a preemptable array with `afterok`.** A shard that
SLURM requeues passes through a state that `afterok` reads as a failure, and
the dependent job is cancelled. Run the merge by hand, or wait with
`afterany` and let the merge's own coverage test catch a missing shard.

## Review of 2026-09-20

The review read the code, the 4 corpora, and the source traces. It found 7
problems. Each one has a fix below.

### 1. The qwen traces split

The design says 1 trace must be 1 chunk. A qwen3.5-9b trace is about 3x as
long as a gemma-4-12b trace, and the version 1 buffer of 12,000 characters
did not hold it.

| Judge | Case | Median chars | p90 | Over 12,000 |
|-------|------|--------------|-----|-------------|
| qwen3.5-9b | subway_safety | 9,278 | 24,494 | 36% |
| qwen3.5-9b | restaurants | 8,079 | 23,427 | 30% |
| qwen3.5-9b | schools | 10,183 | 22,230 | 28% |
| qwen3.5-9b | libraries | 7,532 | 21,244 | 27% |
| qwen3.5-9b | parks_plazas | 7,390 | 20,789 | 26% |
| qwen3.5-9b | road_quality | 4,704 | 10,751 | 8% |
| qwen3.5-9b | street_photography | 3,560 | 8,759 | 5% |
| gemma-4-12b | every case | 1,869 to 3,109 | under 5,500 | 0% |

Every qwen manifest recorded the count as `traces_over_buffer`, and Phases 6
and 7 did not report it. A split trace gets the full prompt for each chunk,
and `minItems: 1` makes each chunk answer. On 150 long qwen subway traces, a
middle chunk gave 7.8 `decision` spans under the gemma extractor, and 96% of
middle chunks held a decision. Thus a version 1 qwen corpus holds 2.0 to 3.6
decisions for each trace.

Fix: version 2 holds the longest qwen trace in 1 chunk, and
`_extractions.load` keeps the last decision of each trace.

### 2. The repair loses the tail

"The repair saved all" hides a loss. `repair_truncated_json` cuts back to the
last closed extraction, and the tail of the answer is gone. The decision sits
in the tail. Truncated answers reached 9% of a gemma-by-qwen subway shard and
22% of a qwen-self subway shard. That is one reason 5% to 11% of the traces in
the qwen-extractor arms carry no decision.

Fix: version 2 raises `max_tokens` to 12,288. Watch `answers_truncated` on
every run.

### 3. The 2 extractors pick different spans

The Phase 5 swap test was planned and never ran. The review ran it on 300
shared traces for each case, with a 50% character-overlap match.

| Measure | gemma traces | qwen traces |
|---------|--------------|-------------|
| qwen-extractor spans with a counterpart | 35% to 38% | 21% to 34% |
| gemma-extractor spans with a counterpart | 47% to 51% | 31% to 40% |
| Same class, when matched | 83% to 90% | 74% to 85% |
| Same `image` | 99% | 99% |
| Same `valence` | 91% | 91% to 93% |
| Same `category` | 72% to 85% | 74% to 86% |
| Same `decision.label` | 99% to 100% | 99% to 100% |

The attributes of a shared span agree. The set of spans does not. On the
same 11,000 gemma subway traces, the qwen extractor wrote 148,690
`uncertainty` spans and the gemma extractor 24,889.

`_extractions.compare_arms` and `_extractions.rate_stability` now read the
2 x 2 for any rate table. A rate `holds` when the 2 extractors stay within
1.5x in every case.

| Class | gemma judge | qwen judge |
|-------|-------------|------------|
| `visual_evidence` | holds, 1.07x | holds, 1.13x |
| `comparison` | holds, 1.19x | holds, 1.18x |
| `decision` | holds, 1.29x | holds, 1.24x |
| `person_reference` | mixed, 1.42x | mixed, 1.50x |
| `image_artifact` | mixed, 1.83x (up to 27x) | mixed, 1.34x (up to 6x) |
| `uncertainty` | mixed, 1.64x (up to 2.9x) | extractor, 3.78x |
| `inference` | extractor, 1.94x | mixed, 2.16x |

The ratio is the median over the 7 cases, after `load` drops the duplicate
rows and the unanswered traces (problems 6 and 7). Before those 2 filters,
`uncertainty` read 3.7x and 6.5x and `comparison` 1.4x and 1.7x: about half
of the extractor gap was repeated spans. Thus `cue-hierarchy.md` stands on a
class that holds. The abstention channel of `trace-channels.md` is a fact
about the extractor pair, and the paper must report it from both extractors
or not at all. The `trace_channels` notebook now shows this table.

### 4. A fuzzy span is not the source text

The rule "count `match_exact` and `match_fuzzy`" assumed that a fuzzy span
holds the source text word for word. It does not. The aligner accepts a span
when 75% of its tokens match. On 3,000 sampled fuzzy rows, 12% (gemma
extractor) to 20% (qwen extractor) differed from the text at the offsets, and
some changed the meaning: "the construction in A" for "the cars in A".

Fix: the stage writes `is_verbatim`, which is True when the source text at
the offsets IS the span. `research/scripts/verify_span_text.py` added the column to
the 4 version 1 corpora on 2026-09-20. A quote needs `is_verbatim`.
`_extractions.quotes` applies it. A count keeps `is_quotable`, because a
near-verbatim span is still a statement the model made.

| Arm | Verbatim, of all rows | Verbatim, of quotable rows |
|-----|-----------------------|----------------------------|
| gemma by qwen | 83% to 94% | 91% to 97% |
| qwen by gemma | 91% to 94% | 96% to 98% |
| gemma self | 96% to 99% | 99% to 100% |
| qwen self | 77% to 87% | 87% to 92% |

### 5. The values drift

An attribute value was a free string. `other` took 43% to 67% of
`uncertainty.reason`, 44% to 70% of `image_artifact.kind`, and 25% to 37% of
`inference.kind` in the qwen-extractor arms. `safety` appeared as a
`visual_evidence` category 1,537 times in 1 file.

Fix: version 2 puts an enum on every declared value.

### 6. A silent judge carries the label `Same`

The version 2 smoke run showed it. A judge that reaches its 8,192-token cap
inside the thought block writes no answer, and the pairwise stage then gives
the row the label `Same` (`pairwise_vqa._normalize_label` returns `Same` when
no label matches). That row is a fallback, not a judgment.

| Judge | Case | Traces with no answer | Share of the `Same` labels |
|-------|------|-----------------------|----------------------------|
| qwen3.5-9b | restaurants | 632 (5.7%) | |
| qwen3.5-9b | subway_safety | 580 (5.3%) | 73% |
| qwen3.5-9b | schools | 518 (4.7%) | |
| qwen3.5-9b | libraries | 487 (4.4%) | |
| qwen3.5-9b | parks_plazas | 426 (3.9%) | |
| qwen3.5-9b | road_quality | 88 (0.8%) | |
| qwen3.5-9b | street_photography | 14 (0.1%) | |
| gemma-4-12b | every case | 0 to 8 | |

Every such qwen trace is about 31,000 characters long: the cap cut it. Thus
the qwen judge's `Same` rate in a thinking run is mostly this fallback, and a
`decision` span in such a trace cannot agree with `presented_label`.

Fix: the stage writes `judge_answered`, `research/scripts/verify_span_text.py` added it
to the 4 version 1 corpora, and `_extractions.load(answered_only=True)` drops
those traces from the frame and from the denominator. Every rate in the
notebooks now excludes them. The pairwise stage itself still writes `Same`;
that is a separate change, for the judgment runs.

### 7. A capped answer is a loop

The version 2 smoke runs showed it. An answer that reaches the token cap is
almost never a long answer. It is a loop: the qwen3.5-9b extractor wrote
"Let's look at the hot again" and 3 variants of it 388 times for 1 trace, as
`uncertainty` spans, after it had covered 98% of the trace. 12 of 120 qwen
answers on qwen traces looped at 12,288 tokens, and the same 12 at 16,384. The
gemma extractor looped on 4 to 7 of 120.

`repair_truncated_json` keeps the head of such an answer, which holds the
real spans and then the loop. The hallucinated spans do not align, thus
`is_quotable` drops most of them. The repeated real spans do align.

The loop is the extreme case of a general habit. The prompt says "a repeated
thought is 1 extraction, not 2", and the extractors do not obey it. A later
copy of the same class and text inside 1 trace, in the version 1 corpora:

| Arm | Duplicate rows | Of them, at the same offset as the first copy |
|-----|----------------|-----------------------------------------------|
| gemma by qwen | 14% to 29% | 74% |
| qwen by gemma | 9% to 20% | |
| gemma self | 1% to 5% | 72% |
| qwen self | 17% to 35% | 65% |

The qwen extractor repeats; the gemma extractor mostly does not. On the same
gemma traces, the qwen extractor wrote 43.7 spans for each trace against 25.1
for the gemma extractor, and 12 of the 18 extra spans were copies.

Fix: the stage writes `is_duplicate`, True for every later copy of the same
class and text inside 1 trace. `research/scripts/verify_span_text.py` added it to the
4 version 1 corpora, and `_extractions.load(dedupe=True)` drops those rows.
The count `answers_truncated` in the manifest is the count of loops.

### What the review did not find

No human audit exists. Phase 5 names one, and nothing in the repository
stands in for it. The audit runs in Label Studio after the version 2 arms
finish.

## Version 2

`research/conf/pairwisevqa/extract/urban_cues_v2.yaml`, 2026-09-20. It
keeps the 7 classes, the attribute names, and the value lists of version 1.

| Change | Value | Why |
|--------|-------|-----|
| `attribute_values` | an enum for each declared value | Problem 5. The grammar cannot write a value outside the list, and a declared attribute is required. `comparison.dimension` stays free. |
| A 4th example | schools trace `unit_00006634` of the qwen3.5-9b run | Every version 1 example was a gemma trace. The qwen example names "Less", says "Wait", and decides "MuchLess". |
| Rule 6 | 1 decision: the LAST sentence that states the answer | Problem 1. An earlier candidate is a comparison or an uncertainty. |
| Rule 7 | a list of the labels is not evidence | A qwen trace lists the 6 labels before it decides. |
| `max_char_buffer` | 40,000 | Problem 1. The longest qwen trace is 35,598 characters. |
| `max_model_len` | 32,768 | About 2,700 tokens of preamble, up to 10,000 of trace, and the answer. The pipeline reads `${extract.max_model_len}`. |
| `max_tokens` | 16,384 | Problem 2. A trace in 1 chunk writes more, and the qwen extractor writes up to 280 spans for a 30,000-character trace. At 12,288, 14 of 120 qwen answers still reached the cap. |

**Warning: do not force compact JSON.** Both extractors indent the answer,
which costs about 2x the tokens, thus
`structured_outputs_config.disable_any_whitespace` looked like a free win. The
smoke run of 2026-09-20 with `backend: xgrammar` and that flag gave 45 silent
traces of 120 for gemma-4-12b (`{"extractions": [{}]}`, 0 under the default
backend) and 2.5 echo spans for each trace for qwen3.5-9b (0.1 under the
default). The compact grammar moves the model off its own JSON habit. Keep
the default backend and pay the tokens.

The 31B does not save it. Measured on 2026-09-20 on klara, 2 x A6000, the
same 60 qwen traces: compact JSON took 1,367 s against 1,651 s, but the
duplicate rows rose from 15% to 41%, the exact alignment fell from 77% to
52%, and the decisions from 0.98 to 0.93 for each trace. The tokens that the
indentation saved went into repeated spans. The 17% was the shared-memory
gain, not the grammar.

**The 31B on 2 x A6000 is bound by the PCIe all-reduce.** The UAIR project
measured 58 tokens per second for gemma-4-31b at TP=2 on A6000, and our
answers of about 2,000 tokens give the same rate: 27.5 s for each trace, or
about 84 two-GPU hours for a case of 11,000 traces. NCCL shared memory on
(`NCCL_SHM_DISABLE=0`, which `VLLMEngine` now forces for TP>1) is the one
lever that holds. Measured on klara on the same 60 qwen traces, 2026-09-20:
1,259 s with shared memory on against 1,651 s with it off, a 24% gain at the
same quality. Thus the 31B reads about 21 s for each trace on 2 A6000, or
about 64 two-GPU hours for a case of 11,000 traces. TP=4 does not help
(UAIR: the collectives double), and FP8 KV does not run on sm_86.

`langextract_backend` applies the enums in `ExtractionSpec.guided_json`, and
`validate_examples` tests each example value against them. Select the
version with `extract=urban_cues_v2` on the CLI. The output file name now
carries `${extract.schema_version}`.

**Warning: 2 versions never pool.** `_extractions.load` reads a v1 corpus or
a v2 corpus, and it stops on a root that holds both. Read a v2 arm against a
v2 arm.

### The extractor

Decided on 2026-09-20, after a literature search. No published paper applies
LangExtract to reasoning traces. The closest work reads a trace with a model
from another family: Thought Anchors (Bogdan et al., 2025) labels
DeepSeek-R1-Distill-Qwen traces with GPT-4o; ReasoningFlow (2026) annotates
QwQ, R1, and GPT-oss traces with Gemini and picks the annotator by F1 against
a 31-trace human gold set. A same-family judge carries a lift of 3 to 8
points even when it never sees its own output (Awuni et al., 2026), and a
stronger model in the same family does not remove it (Chen et al., 2025).
One extractor is not enough: Baumann et al. (2025) find that about 31% of
published annotation conclusions flip under another annotator, and Camuffo
et al. (2026) recommend 2 model families for each construct.

We hold no third-family model of comparable size with a context that holds a
trace: Phi-4 stops at 16K tokens, and GPT-OSS-20B has 3.6B active
parameters. Thus each judge is read by the strongest model of the OTHER
family, and the same-family cell of the same size is the control for the
family effect.

| Judge | Primary extractor | Family control | Small-model row |
|-------|-------------------|----------------|-----------------|
| gemma-4-12b | Qwen3.8-27B | Gemma-4-31B | qwen3.5-9b, gemma-4-12b |
| qwen3.5-9b | Gemma-4-31B | Qwen3.8-27B | gemma-4-12b, qwen3.5-9b |

`_extractions.PRIMARY_ARMS` names the primary cell for each judge. The
paper reports a rate from the primary cell, with the other cells of the same
judge as the sensitivity result, and the Label Studio gold set selects
between the 2 large extractors by F1 for each class.

A 27B or 31B model in bf16 fills 54 to 62 GB, thus these arms run on 2 GPUs:
launcher `slurm_gpu_preempt_2x`, `trace_extract.tensor_parallel_size=2`,
`trace_extract.max_num_seqs=32`, `runtime.job_memory_gb=128`. The launch
script sets all 4. Qwen3.8-27B (released 2026-08-14, Apache 2.0,
`Qwen3_5ForConditionalGeneration`) sits at `zoo/models/Qwen3.8-27B`, with
the config `model=qwen3.8-27b/instruct`.

### Submit without a monitor

Decided on 2026-09-20. The Hydra multirun path puts a MONITOR job above each
shard. The monitor composes the config, submits the GPU stage job, blocks on
`job.result()`, and then writes `pipeline_manifest.json`. That is a workflow
engine for a DAG, and `extract_traces` is a graph of 1 node. The 2 qwen arms
of version 2 held 112 monitors, 1 CPU and 8 GB each, for 6 hours while their
stage jobs waited in the queue: 640 GB of lisbeth and 29 of the 56 CPUs of
klara, for nothing. A smoke job on klara then could not start.

**Do we need a monitor at all?** No, not for a graph of 1 node. Three things a
monitor does, and what does them instead:

| The monitor did | Now |
|-----------------|-----|
| Compose the config of each shard | `submit_trace_shards.py`, one time, on the login node |
| Submit the stage job and wait for it | SLURM. The shards are 1 job array. A preempted task is requeued by SLURM, and it skips the chunks it wrote. |
| Write `pipeline_manifest.json` | `run_shard_job`, inside the GPU job, after the stage returns |

A graph of 2 or more nodes still needs an order. The right tool for that is
also SLURM, not a waiting process: submit every node at once with
`--dependency=afterany:<parent>`, and let the child test its inputs when it
starts. `afterok` is wrong here, because a requeued task passes through a
state that `afterok` reads as a failure (see Phase 7). No pipeline of this
project has more than 1 node today, thus that path is not built.

```bash
python -m dagspaces.pairwisevqa.submit_trace_shards --arm qwen_by_gemma31_v2 --dry-run
python -m dagspaces.pairwisevqa.submit_trace_shards --arm qwen_by_gemma31_v2
bash research/scripts/check_trace_extraction.sh CUESV2QG31
python research/scripts/merge_trace_extractions.py multirun/<date>_CUESV2QG31/runs --out data/trace_extractions_v2_qwen9b_by_gemma31b
```

The arms, the engine shape of each, and the registered source runs live in
`submit_trace_shards.ARMS`. The output layout is
`<base-dir>/runs/<case>__<shard>/`, and `submitted_jobs.txt` names the array
task of each shard. `--only-missing` with the same `--base-dir` runs the
shards that hold no final parquet again. `research/scripts/launch_trace_extraction.sh`
is the old path and stays for the record of the version 1 arms.

The 2 qwen arms of version 2 (`CUESV2Q`, `CUESV2QQ`) moved to this path on
2026-09-20 at 21:00: their monitors and stage jobs were cancelled, about 10
resume chunks were lost, and both arms were submitted again as 2 arrays.

### The production configuration of 2026-09-20

Fixed at 22:00, after the smoke runs of the day. Every arm runs through
`submit_trace_shards.py`, schema `urban_cues_v2`, chunks of 250 rows, and
resumes from its chunks. A shard that hits its walltime is not requeued: run
the arm again with `--only-missing` and the same `--base-dir`.

| Arm | Judge traces | Extractor | Traces | Shards | Launcher | Walltime | Array |
|-----|--------------|-----------|--------|--------|----------|----------|-------|
| `gemma_by_qwen27_v2` (primary) | gemma-4-12b | Qwen3.8-27B | all 11,000 | 8 | `slurm_gpu_preempt_2x` | 6 h | 618532 |
| `qwen_by_gemma31_v2` (primary) | qwen3.5-9b | Gemma-4-31B | all 11,000 | 16 | `slurm_gpu_preempt_2x`, 5 cases | 8 h | 619252 |
| `qwen_by_gemma31_v2` (primary) | qwen3.5-9b | Gemma-4-31B | all 11,000 | 16 | `slurm_gpu_2x` on klara, subway_safety and restaurants | 15 h | 619256 |
| `qwen_by_gemma_v2` | qwen3.5-9b | gemma-4-12b | all 11,000 | 8 | `slurm_gpu_preempt` | 7 h | 618500 |
| `qwen_self_v2` | qwen3.5-9b | qwen3.5-9b | all 11,000 | 8 | `slurm_gpu_preempt` | 7 h | 618502 |
| `gemma_by_qwen_v2` | gemma-4-12b | qwen3.5-9b | all 11,000 | 8 | `slurm_gpu_preempt` | 7 h | 619268 |
| `gemma_self_v2` | gemma-4-12b | gemma-4-12b | all 11,000 | 8 | `slurm_gpu_preempt` | 7 h | 619275 |
| `gemma_by_gemma31_v2` (control) | gemma-4-12b | Gemma-4-31B | 2,000 for each case | 8 | `slurm_gpu_preempt_2x` | 5 h | 619287 |
| `qwen_by_qwen27_v2` (control) | qwen3.5-9b | Qwen3.8-27B | 2,000 for each case | 8 | `slurm_gpu_preempt_2x` | 5 h | 619297 |

**Why the controls read a sample.** The 31B reads a qwen trace in about 21 s
on 2 A6000 with shared memory on, thus a full control arm costs about 450
two-GPU hours, and the 2 controls together about the same as the 2 primary
arms. A family-effect estimate needs no 77,000 traces: 2,000 for each case
(`runtime.sample_n=2000`, seed 777, the same 2,000 in every shard split) give
14,000 traces for each control arm. The paper's numbers come from the primary
cells, and the controls give the family-effect delta with a confidence
interval.

**What the smoke runs said, 120 traces each, 2 x A6000:**

| Extractor | Traces | s for each trace | Exact | Verbatim | Duplicate | Last decision = label |
|-----------|--------|------------------|-------|----------|-----------|----------------------|
| Gemma-4-31B, shared memory on | qwen | 21 | 77% | 95% | 15% | 99% |
| Qwen3.8-27B, shared memory on | gemma | 7.7 | 98% | 99% | 1% | 100% |
| gemma-4-12b, 1 GPU | qwen | 15 | 56% | 78% | | 90% |
| qwen3.5-9b, 1 GPU | qwen | 10 | 36% | 69% | | 94% |

Qwen3.8-27B is the best extractor on every measure, and it is fast because
most of its layers use linear attention: its KV cache holds 454,000 tokens on
2 A6000 where the 31B holds 87,000. The Blackwell single-card smokes never
landed: those 28 GPUs were allocated to their owners all day.

### Phase 8 — The version 2 arms — DONE 2026-09-22

All 8 arms finished on 2026-09-22 by 09:00: 504 shards, every one
`COMPLETED`, no silent trace, no repeated trace, every case from the
registered run. Two bad nodes (`nikola-compute-18`, `seo-compute-01`) killed
21 shards at start-up; both are excluded and the shards ran again. No shard
hit its walltime.

| Arm | Judge traces | Extractor | Traces for each case | Spans for each trace | Quotable | Verbatim | Duplicate | Exact | Last decision = label |
|-----|--------------|-----------|----------------------|----------------------|----------|----------|-----------|-------|----------------------|
| `gemma_by_qwen27_v2` (primary) | gemma-4-12b | Qwen3.8-27B | 11,000 | 26.4 | 99.3% | 99.2% | 1.3% | 96.9% | 99.4% |
| `gemma_by_gemma31_v2` (control) | gemma-4-12b | Gemma-4-31B | 2,000 | 20.1 | 99.1% | 98.7% | 0.8% | 96.6% | 99.7% |
| `gemma_by_qwen_v2` | gemma-4-12b | qwen3.5-9b | 11,000 | 34.6 | 93.8% | 86.0% | 31.4% | 57.4% | 99.1% |
| `gemma_self_v2` | gemma-4-12b | gemma-4-12b | 11,000 | 19.4 | 97.6% | 96.6% | 5.2% | 89.1% | 99.5% |
| `qwen_by_gemma31_v2` (primary) | qwen3.5-9b | Gemma-4-31B | 11,000 | 32.2 | 96.9% | 93.8% | 17.2% | 76.7% | 99.4% |
| `qwen_by_qwen27_v2` (control) | qwen3.5-9b | Qwen3.8-27B | 2,000 | 65.9 | 97.5% | 95.1% | 24.5% | 71.5% | 99.0% |
| `qwen_by_gemma_v2` | qwen3.5-9b | gemma-4-12b | 11,000 | 27.3 | 95.4% | 90.9% | 19.2% | 72.6% | 98.0% |
| `qwen_self_v2` | qwen3.5-9b | qwen3.5-9b | 11,000 | 50.3 | 91.3% | 78.5% | 44.2% | 43.7% | 98.3% |

The quality columns are means over the 7 cases. `Duplicate` is the share of
rows that `is_duplicate` marks; `Exact` is `match_exact`; the last column
compares the last decision of each answered trace with `presented_label`,
after the duplicate rows leave.

Three things the table says:

1. The large extractors are exact. Qwen3.8-27B on gemma traces reads 97%
   of its spans exactly and repeats 1% of them. The small extractors repeat
   19% to 44% of their spans, and the 9B on its own traces reads 44% exactly.
2. The long qwen traces cost every extractor. The 31B falls from 97% exact
   on gemma traces to 77% on qwen traces, and the 27B from 97% to 72%, with
   duplicates at 17% and 25%. The loops of problem 7 sit here.
3. The last decision agrees with the label on 98% to 99.7% of answered
   traces in every arm. The decision channel is safe from every extractor.

**The 2 x 2 with the large extractors.** `compare_arms` over the primary and
the control of each judge, class rates for each 100 traces, the ratio of the
2 extractors in the median case:

| Class | gemma-4-12b traces: Qwen3.8-27B against Gemma-4-31B | qwen3.5-9b traces: Gemma-4-31B against Qwen3.8-27B |
|-------|------|------|
| `decision` | holds, 1.03x | holds, 1.02x |
| `visual_evidence` | holds, 1.08x | mixed, 1.37x |
| `comparison` | holds, 1.12x | mixed, 1.39x |
| `image_artifact` | mixed, 1.43x | holds, 1.18x |
| `person_reference` | mixed, 1.20x, 1 case at 4.2x | mixed, 1.07x, 1 case at 4.3x |
| `uncertainty` | mixed, 2.01x | extractor, 5.87x |
| `inference` | extractor, 2.26x | extractor, 5.01x |

The picture of the review holds with the large models: `decision`,
`visual_evidence`, and `comparison` are properties of the traces;
`inference` and `uncertainty` are properties of the extractor. Qwen3.8-27B
reads about 2x the `inference` spans of Gemma-4-31B on gemma traces (523
against 222 for each 100 traces) and 4.5x on qwen traces (1,308 against
288). The same-family cell reads MORE, not less: Qwen3.8-27B on qwen traces
writes 2,365 `visual_evidence` spans for each 100 traces against 1,698 for
Gemma-4-31B. The family effect is real, and it is an effect of span count,
not of the attributes on a shared span.

**Warning: the rate denominator follows the corpus.** `class_rates`,
`attribute_rates`, and `risk_panel` take `root`, and `compare_arms` passes
the root of each arm. Before 2026-09-22 they read the default root, and a
control arm of 2,000 traces for each case then showed rates 5.5x too low.

The paper's numbers come from the 2 primary cells:
`data/trace_extractions_v2_gemma12b_by_qwen27b/` for the gemma-4-12b judge
and `data/trace_extractions_v2_qwen9b_by_gemma31b/` for the qwen3.5-9b
judge. `_extractions.DEFAULT_ROOT` names the first. The v1 corpora stay on
disk for the record.

## Version 3, draft of 2026-09-28

`conf/extract/urban_cues_v3.yaml`. Not yet run. It answers the finding of the
gold-set review: the version 2 examples take a phrase where the annotator
takes the clause, thus the extractors reached half the human spans at a
precision of 0.9. The extraction task and the evaluation task must share one
convention, or the score measures the gap between 2 conventions and not the
extractor.

**The classes, on the Integrative Complexity codebook.**

| IC category | Class | Attributes | Version 2 |
|-------------|-------|------------|-----------|
| 1, descriptive observation | `dimension` | image, valence, free `name` | `visual_evidence` with a `category` enum |
| 1, evaluative observation | `inference` | image, kind, free `name` | `inference`, phrase-length |
| 2, evaluative standard | `perspective` | free `name`, favors, wins, source | lumped into `inference` |
| 3, comparative weighing | `weighing` | mechanism, favors | none |
| 3, the reason | `justification` | of, role (reason, condition) | none |
| 3, compartmentalization | `dismissal` | free `target`, about (cue, perspective, premise) | none |
| 3, pseudo-differentiation | `verdict` | label | a `comparison` |
| 3, tolerance for ambiguity | `uncertainty` | reason, marker (qualifier, contrastive, explicit_doubt, reconsideration) | `uncertainty`, reason |
| scaffold | `decision`, `comparison`, `person_reference`, `image_artifact` | as before | as before |

The rule of the IC pipeline holds: the prompt asks for the span, never for
the judgment. "Justified" is a located `justification` span; "developed" is
a perspective named more than one time; "revised" is a verdict whose label
differs from the decision. The trace-level codes of category 3 derive on the
CPU from the spans, as `research/ic/ic_codes.py` does for the IC
ingredients. `dismissal.about = premise` separates "I don't see a library"
from a cue set aside, which the IC pilot found to be 39% of dismissals.

**The examples match the gold set.** The 4 worked examples (3 of version 2
and the first trace of `ic_examples_v2.json`, which holds a weighing and a
dismissal) were re-annotated in the annotator's style, 93 spans, and read
against the 10 gold traces on the statistics that separated version 2 from
the annotator:

| Statistic | Version 2 examples | Version 3 examples | Gold, 10 traces |
|-----------|--------------------|--------------------|-----------------|
| span length, median words | 4 | 8 | 8 |
| spans for each 1,000 characters | 9.7 | 16.8 | 16.0 |
| share of the text inside a span | 32% | 80% | 77% |
| nested pairs | 0 | 15 | 65 |
| prompt size, about | 2,700 tokens | 6,900 tokens | |

A nested span aligns by the fuzzy pass of the aligner, and
`validate_examples` now accepts a fuzzy match whose slice equals the span.

**Warning: the gold set still carries the version 2 labels.** `visual_evidence`
maps to `dimension` by rename; the perspectives, verdicts, weighings,
dismissals, and justifications need a relabel. The annotator does that
later (decided 2026-09-28). The Label Studio config of the gold set must
carry the 12 classes before the remaining 20 traces are annotated.

**The gold set on version 3.** `research/scripts/build_gold_set.py --schema v3` (the
default from 2026-09-28) writes the 12 classes into the Label Studio
interface: a Choices control for each listed attribute and a TextArea for
the free `name`, `target`, and `dimension`. A carried annotation maps
`visual_evidence` onto `dimension`; the other new classes need the
annotator. Project 4, "Urban cues gold set v4 (IC classes)", holds the 30
tasks and the 10 annotations of project 3 with that rename. The alignment
module reads both class sets.

**The interface of 2026-09-30.** The per-image depiction check
(`shows_a`, `shows_b`) is gone, and a free `notes` box (a TextArea on the
task) takes its place: depiction notes when they arise, or anything else
about the trace. The instructions in Label Studio now state the version 3
criteria in full: the whole clause, nesting, the 12 classes with the
dimension-to-perspective process and the mode words, verdict against
decision, the uncertainty markers, the free names, and the notes box.
`research/scripts/label_studio_setup.py --update-config --gold-dir <set>` pushes
both onto an existing project; Label Studio refuses to drop a control that
an annotation used, thus the 2 depiction results were stripped from the
11 annotations that held them (1 in project 5, 10 carried in project 4)
before the update. `build_gold_set.py --config-only` writes the interface
alone. `_gold_alignment.load_export` reads `notes` into the task table.

**The clear 50-trace set (2026-09-30).** The annotator found traces in
the 50-trace set whose reasoning turns on "is this what the image shows?"
even when the unit is there, at the edge of the frame or behind the
foreground. The facing-filter confidence does not separate those traces:
by a word test on the trace (`build_gold_set.REFERENT_DOUBT`: "I don't
see", "is there a", "hard to tell", "the prompt says", ...) 50% to 82% of
the unit-case traces doubt the referent, and the share in the top decile of
confidence is 35% to 80%, about the same as below the median. Thus the
clear set draws on 3 filters at once, on top of the earlier ones:

| Filter | Value | Reads |
|--------|-------|-------|
| `--conf-quantile` | 0.9 | the weaker image in the top decile of its case |
| `--max-delta-bearing` | 30 | both images: the unit at most 30 degrees off the face centre |
| `--max-distance-ft` | 120 | both images: the camera at most 120 feet from the unit |
| `--exclude-referent-doubt` | | the trace never doubts that the image shows the item |

The pool after every filter runs from 10 (parks, qwen) to 52 (subway,
gemma) traces for each judge and case, enough for 3 or 4. The set drawn:
25 traces of each judge, none with referent doubt, the unit 13 to 23
degrees off centre and 35 to 90 feet away on average by case, and outside
both earlier sets. It lives at `data/gold_set/set50c/` and is Label Studio
project 6, "Urban cues gold set 50 clear (IC classes)", with the notes box
and no depiction check. The 2 image-only cases (road quality, street
photography) take only the doubt filter. `sample.csv` carries `db_max`,
`dist_max`, and `referent_doubt` for each trace.

**The 50-trace set for version 3 (2026-09-29).** `research/scripts/build_gold_set.py
--n-per-judge 25 --in-v3-sample --exclude-sample data/gold_set/sample.csv
--carry-annotations none --verify-sharing --seed 20260930 --out
data/gold_set/set50`. It draws 25 traces of each judge, 3 or 4 of each
case, from the 1,000-row sample the version 3 arms read (reproduced by
`v3_sample_pair_ids`: the traces of 200 characters or more, then
`sample(n=1000, random_state=777)`), above the confidence floor, answered,
under the length cap, and outside the 30-trace set. `--verify-sharing`
reads the merged arms against that sample: both gemma-trace arms hold
exactly the 1,000 rows of each case, and the 36 finished shards of the
qwen-trace Qwen27 arm sit inside it (2,379 rows, 100%). All 50 traces sit
in their judge's sample. The set lives under the local-files root, thus its
images serve from `set50/images/`. It is Label Studio project 5, "Urban
cues gold set 50 (IC classes)", with the 12 classes and no annotation.

**The run ladder of 2026-09-28.** The class set was confirmed the same day,
after 2 corrections by the annotator: a perspective is the mode of
valuation itself, the evaluative word inside the clause that applies it
("classic", "prestigious", "utilitarian"), and a word that names a state of
the scene ("well-lit") stays a dimension. The examples hold 121 spans.

| Rung | What | Rows | Gate |
|------|------|------|------|
| 1, smoke | `gemma_by_qwen27_v3` and `gemma_by_gemma31_v3` on subway_safety, 1 shard | 120 (`runtime.sample_n=120`) | `answers_truncated` under 10%, verbatim over 95%, duplicates under 5%, every class present, no empty answer |
| 2, full | the 4 large arms, 7 cases, 4 shards of 250 | 1,000 for each case (`runtime.sample_n=1000`) | every shard's final parquet on disk; then merge |

The stage samples BEFORE it shards (`load_traces`: `_sample`, then
`_shard`), with the seed 777 of `pair_sampler.pair_seed`. Thus the 2 arms
that read 1 judge draw the same 1,000 rows of each case, and a rerun draws
them again. The smokes write to `multirun/2026-09-28_<experiment>_smoke`,
apart from the full run's base directory, which carries the date alone.

**Rung 1, measured.** The Qwen3.8-27B smoke (job 607888, snavely-compute-02)
took 39 minutes for 120 gemma subway traces: 6 minutes of engine start and
16 seconds for each trace at 32 concurrent. It passed on truncation (0),
verbatim (99.6%), duplicates (2.7%), and empties (0), with 74 spans and 9
perspectives for each trace and 22% of spans nested. It found no weighing,
and the text holds 1 weighing word in 120 traces, thus the gate asks only
for the 5 core classes. It gave a decision on 92% of traces: a hedged last
line ('"Same" seems most appropriate.') was read as an uncertainty alone.
Rule 7 of the prompt now names that case, and the gate asks for a decision
on 95% of traces. The Gemma-4-31B smoke (job 608321) ran after on
owens-compute-01, which first copied the 58 GB model to its scratch: 51
minutes, truncation 3.3%, verbatim 98.2%, a decision on 96% of traces, 57
spans and 4.8 perspectives for each trace, 10% nested. It fails the
duplicate gate at 13.4% (the same extractor gave 17% under v2 on qwen
traces). A duplicate row carries `is_duplicate` and the loader drops it,
thus the arm went out as the control it is, with the rate on record. It
names the perspective of the first example, "NYC baseline", 154 times in
120 traces, against 89 for the 27B: read that name as an example leak.

**Rung 2, sent 2026-09-29.** Shards of 125 rows, 8 for each case, so that
a qwen shard (traces 2.5x longer) stays under the 180-minute limit of
`slurm_gpu_preempt_2x`.

| Arm | Array | Base | Sent |
|-----|-------|------|------|
| gemma_by_qwen27_v3 | 615424 | `multirun/2026-09-29_CUESV3GQ27` | after the Qwen27 smoke passed |
| qwen_by_qwen27_v3 | 616287, cancelled; 643681 | `multirun/2026-09-29_CUESV3QQ27` | again at 13:10 with the 64k caps, 112 shards of 63 |
| gemma_by_gemma31_v3 | 620317 | `multirun/2026-09-29_CUESV3GG31` | after the Gemma31 smoke, 01:40 |
| qwen_by_gemma31_v3 | 620418, cancelled; the chunk run | `multirun/2026-09-29_CUESV3QG31` | 23:30, chunks of 6,000 characters, 112 shards of 63 |

**The qwen traces broke the cap (2026-09-29, 09:30).** The first 8 shards
of `qwen_by_qwen27_v3` came back with 22% to 35% of answers cut at 16,384
tokens and a decision on 64% to 74% of traces: the decision sits in the
tail, and the tail is what the cap loses. A qwen trace runs to 24,500
characters at the 90th percentile and 35,600 at most, and version 3
writes about 190 spans for one, against 74 for a gemma trace. Both
qwen-trace arrays (616287, 620418) were cancelled after 8 and 0 shards,
and their base directories moved to `..._cap16k`. Both extractors hold a
long context (Gemma-4-31B: 131k with sliding-window local layers;
Qwen3.8-27B: 262k with full attention on every 4th layer). Rung 1 runs
again for the 2 qwen-trace arms at `extract.max_tokens=40000` and
`extract.max_model_len=65536` on 63 subway traces (jobs 632745, 632779,
bases `..._smoke64k`), and rung 2 follows with the same overrides in
shards of 63. The gemma-trace arms keep the 16k cap: 0.1% of their answers
were cut.

**The 64k smoke of the Qwen27 qwen-trace arm (job 632745, 13:10).** 63
subway traces in 35 minutes: 2 answers cut (3.2%, against 22% to 35%),
verbatim 99.3%, 271 spans and 23 perspectives for each trace. Of the 8
traces with no decision, 4 had no answer from the judge (the judge's own
cap cut the trace at 32,000 characters), and 4 carried the final answer as
a `verdict` ("Therefore, \"More\"."). Thus the loader promotes the last
verdict of a trace without a decision (`_extractions.promote_last_verdict`,
on by default), and the gate reads the decision share over the answered
traces after that promotion: 59 of 59. The arm went out again at 13:20 as
16 shards of 63 rows for each case, with the caps in its arm table.

**The Gemma31 qwen-trace arm loops (2026-09-29, 17:30).** Its 64k smoke
(job 632779, 2 A6000s) took 4.9 hours for 63 traces and failed on every
number: 35% of answers cut at 40,000 tokens, 70% duplicate spans, exact
alignment 26%. The shape of the failure: 36 of 63 traces came out clean,
and 27 wrote a cycle of 50 to 77 spans 8 to 27 times over. After the last
real span the grammar allows "one more item" or "close the list", and the
31B picks one more item and starts again. The wider cap gave it more room
to loop, and the KV cache of 2 x 48 GB held about 2 sequences at a time.
The same extractor under version 2 (16k cap) gave 17% duplicates on qwen
traces, and 6% to 11% on gemma traces under version 3: the fault grows
with the length of the answer.

The mitigation, smoked as job 652366 replaced by the `_big_rp` smoke: vLLM's
`repetition_penalty` at 1.05, plumbed from `extract.repetition_penalty`
through `ExtractionSpec` to `VLLMEngine.generate` (1.0 keeps the old
behaviour, and every pending shard keeps it); `max_char_buffer=20000`, so
that the longest qwen traces split in 2 chunks and each answer is shorter;
and the `slurm_gpu_preempt_2x_big` launcher (2 x 96 GB Blackwell), whose
KV cache holds about 4x the sequences.

**The penalty did nothing; the chunk size is the lever (2026-09-29, 22:45).**
The penalty smoke (job 659756, a 399k-token cache, 74 minutes) came back
the same: 69% duplicate spans, 40% cut, 23 of 63 traces in a loop. The
value reached vLLM (`_build_sampling_params` keeps it), thus the penalty
is the wrong tool: it also penalises the copy of a span out of the prompt.
The loops follow the length of the text the model answers in 1 call:

| Trace characters | Traces | In a loop | Distinct spans, median |
|------------------|--------|-----------|------------------------|
| under 6,000 | 12 | 0 | 65 |
| 6,000 to 12,000 | 25 | 3 | 95 |
| 12,000 to 20,000 | 14 | 9 | 115 |
| over 20,000 (in 20k chunks) | 12 | 11 | 154 |

A loop starts after about 100 distinct spans, whatever the cap. Thus the
Gemma31 qwen-trace arm takes `max_char_buffer=6000`: a chunk of 6,000
characters holds about 65 spans, under the loop threshold, and the 16k cap
and the 32k window come back. A qwen trace then splits in 2 to 6 chunks.
The cost: a `decision` for each chunk that states an answer, which
`keep_last_decision` resolves by position; a comparison or a perspective
that spans a chunk boundary is lost; `wins` is judged inside the chunk.
The gemma-trace Gemma31 arm and both Qwen27 arms keep 1 trace in 1 chunk.
Smoked as `multirun/2026-09-29_CUESV3QG31_smoke_chunk6k` (job 670211): 26
minutes for 63 traces, no trace in a loop, 2 answers cut (3.2%), verbatim
98.5%, exact 87%, duplicates 5.9%, a decision on 58 of 58 answered traces,
186 spans and 12.5 perspectives for each trace. 51 of the 63 traces split
in 2 or more chunks, and 44 traces carry 2 to 5 `decision` rows as a
result, 1 for each chunk that states the answer; `keep_last_decision`
keeps the last by position. The arm went out at 23:30 with the chunk
settings in its arm table, 112 shards of 63 rows.

**The merged version 3 corpora (2026-09-30, 08:15).** Every arm reads the
same 1,000 traces of each case for its judge (`build_gold_set
--verify-sharing` checked it against the reproduced sample).

| Arm | Root | Traces | Extractions | Verbatim | Cut | Duplicates | Decisions, of answered |
|-----|------|--------|-------------|----------|-----|------------|------------------------|
| gemma by Qwen27 | `data/trace_extractions_v3_gemma12b_by_qwen27b` | 7,000 | 440,760 | 98.8 to 99.8% | 0 to 0.1% | 1.5 to 6.5% | 95 to 97% |
| gemma by Gemma31 | `data/trace_extractions_v3_gemma12b_by_gemma31b` | 7,000 | 320,776 | 98.2 to 99.7% | 0 to 2.5% | 0.8 to 11% | 97 to 100% |
| qwen by Qwen27, 64k window | `data/trace_extractions_v3_qwen9b_by_qwen27b` | 7,000 | 1,343,332 | 98.9 to 99.6% | 0.2 to 3.4% | 3.6 to 11% | 98 to 100% |
| qwen by Gemma31, 6k chunks | `data/trace_extractions_v3_qwen9b_by_gemma31b` | 7,000 | 859,558 | 97.7 to 98.9% | 0.6 to 4% | 5 to 8.5% | 100% |

A qwen trace by Qwen27 carries 100 to 249 spans and 10 to 37 perspectives;
a gemma trace by Qwen27 carries 44 to 77 spans. The duplicate share is the
one gate every arm misses on some shards; the loader drops those rows. The
ladder closed on 2026-09-30 at 11:00, after 3 smokes and 2 relaunches of
the qwen-trace arms. Through the notebook loader (quotable, not duplicate,
judge answered, last decision, last verdict promoted):

| Arm | Rows | Traces | Decision on | Perspectives for each trace |
|-----|------|--------|-------------|-----------------------------|
| gemma by Qwen27 | 425,420 | 6,996 | 99.3% | 9.8 |
| gemma by Gemma31 | 299,602 | 6,996 | 98.2% | 4.9 |
| qwen by Qwen27 | 1,109,009 | 6,763 | 99.1% | 16.4 |
| qwen by Gemma31 | 709,888 | 6,763 | 99.9% | 7.7 |

The Gemma31 arms write about half the spans of the Qwen27 arms on the same
traces, as under version 2.

**The analysis code on version 3 (2026-09-30).** `_extractions` now defaults
to schema v3: `DEFAULT_ROOT` is the gemma-trace Qwen27 corpus,
`PRIMARY_ARMS` names the 2 cross-family v3 arms (`PRIMARY_ARMS_V2` keeps
the old pair), `VOCABULARY` is the union of the v2 and the v3 enums (the
class names are disjoint or identical, thus `normalize` needs no version
switch), `CLASSES_BY_VERSION` lists each version's classes, and
`SCAFFOLD_CLASSES` adds `verdict`. `_cues_across.load` bridges a v3 corpus
onto the cue module's names: a `dimension` row reads as `visual_evidence`
with `cue_class_source` set, and its free `name` fills `category`.

**Warning: the cue hierarchy over v3 dimensions alone is thin.** Under v3
the descriptive `dimension` carries almost only `neutral` valence, and the
evaluative content moved to `inference` and `perspective`. The bridge gives
7 keys, all neutral, against about 30 under v2. A cue ladder on v3 must
read `dimension` and `inference` together, or the perspective names, and
that is an open design choice; the paper figures still come from the v2
corpus until it is made. `research/scripts/export_trace_figures.py --root <v2 root>`
reads the v2 corpus on purpose.

**The class rates across the 2 extractors of a judge (v3, 7 cases each),
from `compare_arms` + `rate_stability` (ratio of the larger rate to the
smaller, stable at 1.5 or under):**

| Class | gemma traces, ratio median | qwen traces, ratio median | Verdict |
|-------|----------------------------|---------------------------|---------|
| decision | 1.01 | 1.01 | holds |
| dimension | 1.10 | 1.09 | holds |
| comparison | 1.2 | 1.24 | holds |
| image_artifact | 1.18 | 1.16 | holds, mixed |
| inference | 1.45 | 1.88 | mixed on gemma, extractor on qwen |
| perspective | 1.97 | 2.36 | extractor |
| uncertainty | 1.82 | 2.58 | extractor |
| dismissal, justification, weighing | 2.5 to 20 | 2.5 to 3 | extractor, and rare |

The Qwen27 extractor finds about 2x the perspectives and the uncertainty
of the Gemma31 extractor on the same traces.

**Span agreement between the 2 extractors of a judge** (`compare_arm_pairs`,
20 shared traces for each case, the positional rule at half the shorter
span, the primary arm as the reference):

| Class | gemma traces: F1 | precision of Gemma31 | recall of Qwen27 | qwen traces: F1 | precision of Qwen27 | recall of Gemma31 |
|-------|------------------|----------------------|------------------|-----------------|---------------------|-------------------|
| dimension | 0.80 | 0.84 | 0.77 | 0.71 | 0.70 | 0.71 |
| inference | 0.69 | 0.86 | 0.58 | 0.53 | 0.40 | 0.80 |
| perspective | 0.58 | 0.87 | 0.43 | 0.38 | 0.29 | 0.58 |
| uncertainty | 0.61 | 0.91 | 0.46 | 0.43 | 0.31 | 0.75 |
| decision | 0.90 | 0.91 | 0.89 | 0.78 | 0.79 | 0.78 |
| all classes | 0.69 | 0.84 | 0.59 | 0.55 | 0.45 | 0.71 |
| any class | 0.79 | 0.96 | 0.68 | 0.71 | 0.58 | 0.91 |

On the gemma traces the control (Gemma31) is a subset of the primary
(Qwen27): its spans sit on the primary's at 0.96 under any class, and it
marks fewer. On the qwen traces the roles swap, and the control (Qwen27)
marks far more than the primary (Gemma31), 164 against 105 spans for each
trace, thus its precision against the primary falls to 0.58 under any
class. The 2 x 2 says the same thing from both sides: Qwen3.8-27B marks
about 1.5x the spans of Gemma-4-31B on the same trace, and the classes the
gold set found hardest for the extractors (perspective, uncertainty) are
the classes the extractors part on. As under version 2: a
dimension rate and a decision are facts about the traces; a perspective,
an inference, or an uncertainty rate is a fact about the extractor, and
the paper reports it from both arms or not at all.

`research/scripts/check_trace_smoke.py <base> --gate` reads every shard of a base
directory. A shard that hits the limit reruns with `--only-missing
--base-dir <base>`. When every shard has its final parquet:
`python research/scripts/merge_trace_extractions.py <base>/runs --out data/trace_extractions_v3_<judge>_by_<extractor>`.

## The gold set

Phase 5 named a human audit, and no paper claim should rest on the extractor
without one. Built on 2026-09-22.

| Piece | Where |
|-------|-------|
| The sample and the tasks | `research/scripts/build_gold_set.py` -> `data/gold_set/` |
| The labelling server | `research/scripts/label_studio_gold.sh` (Label Studio 1.23, port 8925, `.venv-labelstudio`) |
| The project | `research/scripts/label_studio_setup.py`: 1 project, 30 blind tasks, 2 annotations for each task |
| The scorer | `research/scripts/score_gold_set.py <export.json>` |
| The status script | `python research/scripts/check_trace_arms.py` names the arms; `bash research/scripts/label_studio_gold.sh status` the server |

**The sample.** 30 traces: 15 of each judge, 2 for each case and 3 for
subway_safety, seed 20260922. Every trace has an answer, is under 9,000
characters, and sits in the 2,000-trace subset that the control arm read,
thus all 4 large-extractor cells hold spans on it. The cap keeps a task at a
size an annotator can finish (median 3,749 characters, 112,000 in all), and
it biases the sample toward shorter traces; the long qwen traces that loop
the extractors are under-represented on purpose, and a second, long-trace
sample is the follow-up if the scores differ by length.

**Blind labelling.** The tasks carry the text and the metadata only. The
spans of the 4 large-extractor arms are written beside them as
`predictions_<arm>.json` and are NOT imported, thus the annotator never sees
a machine span. `show_collab_predictions` is off on the project.

**The interface.** 7 span labels, one for each class, and a per-region
choice for each attribute (image, valence, category, kind, used_in_judgment,
direction, hedged, reason, label), with the value lists of
`urban_cues_v2.yaml`. The instructions on the project restate rules 1 to 7
of the prompt: copy the span as it stands, mark nothing in the opening
sentence, exactly 1 decision, a repeated thought is 1 span.

**Perspectives are inferences.** The IC codebook (dimensions, perspectives,
differentiation mechanisms) names a perspective, the evaluative standard by
which a dimension counts for or against a scene. The `urban_cues` schema has
no such class, and the runs lump those spans in with `inference`. For the
annotation and the coding of the traces the 2 names are 1: a perspective is
an `inference` span that states an evaluative standard, and a descriptive
`visual_evidence` span is a dimension. See `ic-ingredient-extraction.md`.

**The images (2026-09-23).** Each task shows the 2 photographs above the
trace, as the judge saw them: Image A is `presented_left_path`, Image B is
`presented_right_path`. Half the pairs are swapped, thus `image_path_a` is
NOT always Image A. The builder copies the files to `data/gold_set/images/`
and the task points at `/data/local-files/?d=images/<task_id>_A.jpg`. The
annotator answers, for each image, whether it shows the item the prompt
names (`shows_a`, `shows_b`: yes, no, unclear). The judges doubt it in some
traces ("I don't see a subway entrance in Image B"), and the answer records
whether the doubt is right. The check is required before a task submits.

**Warning: Label Studio serves a local file only from a registered storage.**
`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` is not enough; the file must sit
under a Local Files storage that the project registered, or the route answers
404. `research/scripts/label_studio_setup.py` registers `data/gold_set/images` on the
new project, with no sync. The tasks come from the JSON.

**The confidence floor (2026-09-23).** A judge sometimes argues with its
prompt ("I don't see a subway entrance in Image B"). That is a separate
reasoning task, and the gold set keeps it out. The facing filter gives each
image an `attribution_confidence` in [0, 1]: the share of the unit inside the
face, times a cubic falloff in the distance to the unit. The score is
geometric, not a model's read, and it sits in the image manifest that the
run's Hydra config names (`data.parquet_path`). The builder joins it to both
images of a pair. The score runs low, and no case but subway has a pair whose
weaker image reaches 0.5:

| Case | Mean over the images | Floor (q75 of the pair minima) |
|------|----------------------|--------------------------------|
| subway_safety | 0.38 | 0.371 |
| restaurants | 0.21 | 0.207 |
| parks_plazas | 0.16 | 0.142 |
| libraries | 0.13 | 0.129 |
| schools | 0.09 | 0.067 |
| road_quality, street_photography | no score: the image is the unit | none |

Thus a pair qualifies when its weaker image (`conf_min`) sits at or above the
0.75 quantile of the pair minima of its case and judge (`--conf-quantile`).
The traces a person already annotated stay in (`--pin`, default: the
annotated tasks of the carried export), and 2 of the 3 subway traces of
2026-09-22 sit below the floor (quantiles 0.20 and 0.49). Their `conf_min`
and `conf_quantile` are in the task data and in `sample.csv`, and the
alignment notebook counts them. The set of 2026-09-23 is project 3, "Urban
cues gold set v3 (confident pairs)". Projects 1 and 2 stay on the server as
they were.

**Carried annotations.** A rebuild of the set keeps the work: the builder
reads the newest export in `research/notebooks/cvpr/cues_annotations/` and copies each
annotation onto the task with the same `task_id` (`--carry-annotations`,
`none` for a clean set). The project of 2026-09-23 is project 2, "Urban cues
gold set v2 (with images)", with the 3 annotations of project 1 carried.
Project 3 (the confidence floor, below) replaced it the same day; annotate in
project 3. A project title has 50 characters at most.

**The 2 x 2 in the alignment notebook (2026-09-30).** Section 11 of
`master/gold_alignment.py` reads the 2 extractors of each judge against
each other on a seeded draw of their shared traces (`compare_arm_pairs`,
default 20 for each case) with the positional rule, the primary arm as the
reference, and `arm_profile` gives 1 line for each rung: traces, spans for
each trace, decision share, perspectives for each trace, class shares. The
arm picker defaults to the schema the export was annotated in (a `dimension`
span means v3). The export writes `gold_alignment_arm_profile_*.csv` and
`gold_alignment_arm_pairs_*.csv`.

**The paper tables (2026-10-01).** The export cell of `master/gold_alignment.py`
(notebook 1.4.0, `_gold_alignment` 1.6.0) also writes 2 LaTeX tables into
`research/notebooks/cvpr/master/outputs/`. The format comes from `research/notebooks/cvpr/_latex.py`.

| File | `\label` | Shows |
|------|----------|-------|
| `gold_alignment_positional_alignment_v<version>.tex` | `tab:gold-positional-alignment` | Span F1 of each extractor against the human, by class, the arms grouped by judge (`latex_positional_alignment`) |
| `gold_alignment_extractor_agreement_v<version>.tex` | `tab:gold-extractor-agreement` | The 2 x 2: F1, P, and R of the control against the primary arm of each judge, by class (`latex_extractor_agreement`) |

The first table reads the 4 arms of the 2 large extractors (Qwen3.8-27B and
Gemma-4-31B). A class row prints when it holds 10 spans or more
(`min_spans`). On the 10 v2 traces, only `visual_evidence`, `inference`, and
`uncertainty` pass. The ledger, the coverage, the nesting, and the decisions
stay in CSV form.
They explain the 2 tables, but a paper does not print them.

**The alignment notebook.** `research/notebooks/cvpr/master/gold_alignment.py` reads
the newest export in `research/notebooks/cvpr/cues_annotations/` and the spans of
every arm that holds the annotated traces, with `research/notebooks/cvpr/_gold_alignment.py`.
The first annotated trace (2026-09-22, gemma, subway_safety) shows 2 facts
about the human spans that the 1-to-1 scorer cannot read:

| Fact | What the annotator does | What the notebook does |
|------|-------------------------|------------------------|
| Nested spans | An `inference` USES a piece of `visual_evidence`, and both get a span: the evidence inside the inference. One region can carry 2 labels. | Reads the characters. A character counts in each class that covers it. `nesting` reports which side each extractor took. |
| Clause against phrase | The human marks a clause (124 spans on the first trace); the extractor quotes a phrase (35 for the primary, 18 for the control). | `ledger` places each extractor span: exact, inside, contains, partial, other class, none. `coverage` reads each human span: hit, touched, missed. |
| Every decision | The human marks each sentence that states an answer (8 on the first trace). The paper keeps the last extractor decision. | `decisions` reports whether the extractor's decision is the human's last. |
| Attributes | The panel below the text holds them; the first task set none. | `attributes` compares them on the matched spans when they exist. |

The first trace, primary arm (Qwen3.8-27B on the gemma trace): 80% of the
extractor spans overlap a same-class human span, and 60% sit inside one.
Character precision 0.68, recall 0.11, F1 0.19. The human's 43
`uncertainty` spans (each "Wait, let me look again") are the gap: no arm
reads them. The main class disagreement is human `inference` against
extractor `visual_evidence` ("well-lit, clean-looking street"). Only the
primary arm and the 9B v2 arm took the last decision; the gemma arms took an
earlier candidate. One trace, one annotator: read these as a first look.

**The positional reading (2026-09-28).** The user re-annotated 10 of the
30 traces of project 3, 6 gemma and 4 qwen, 540 spans, and asked for a
reading by position rather than by exact bounds. `_gold_alignment.scaffold_mask`
masks the scaffold the judge prints at the head of a row (`Image A`,
`Image B`, a bold header such as `**Vibe:**`, a list marker, white space) on
both sides. Then a span matches a span of the same class when their shared
content characters cover at least half of the SHORTER span's content, 1 to
1. The `any_class` row ignores the class and reads the boundary alone. The
4 large arms, micro over the 10 tasks:

| Extractor | Judge | Spans (human 540) | P | R | F1 | F1, any class |
|-----------|-------|-------------------|---|---|----|---------------|
| Qwen3.8-27B | gemma | 161 | 0.71 | 0.41 | 0.52 | 0.69 |
| Qwen3.8-27B | qwen | 177 | 0.64 | 0.43 | 0.51 | 0.67 |
| Gemma-4-31B | gemma | 119 | 0.58 | 0.25 | 0.35 | 0.55 |
| Gemma-4-31B | qwen | 92 | 0.61 | 0.21 | 0.32 | 0.47 |

Pooled over both judges: Qwen3.8-27B F1 0.52 (any class 0.68), Gemma-4-31B
F1 0.33 (any class 0.51). The order holds on 10 of 10 tasks but 1 (a road
trace where the 2 gemma arms tie). The lead is recall: the 31B extractor
marks about 60% as many spans and reads `inference` at a recall of 0.07
against 0.26. Under `any_class`, precision is 0.88 to 0.94 for every arm:
nearly every extractor span sits on a human span. The 2 gaps that remain
are the class of an `inference` (the extractor writes `visual_evidence` on
14 to 33 of them for each arm) and the `uncertainty` spans, which no arm
reads above a recall of 0.33. The rule's share (0.3 to 0.5) moves the
numbers by 0.01. The notebook holds the section and a slider; the tables
are in `master/outputs/gold_alignment_positional_*_v1.2.0.csv`.

**Scoring.** A predicted span matches a human span when the class is the
same and the character ranges overlap by at least 0.5 Jaccard. The scorer
reports span precision, recall, and F1 for each class and arm, attribute
agreement on the matched spans, and the same between the 2 annotators as the
ceiling. Krippendorff's alpha is not computed: the unit is a span the
annotator draws, not a fixed item, and span F1 between annotators is the
agreement measure that ReasoningFlow (2026) used for the same design.

```bash
bash research/scripts/label_studio_gold.sh            # start; log in data/label_studio/server.log
python research/scripts/label_studio_setup.py         # idempotent
# ... 2 people label the 30 tasks at http://lisbeth:8925/projects/1 ...
python research/scripts/score_gold_set.py data/gold_set/export.json
```

**Warning: the API token is off.** Label Studio 1.23 ships a new
organization with the legacy token disabled, thus `Authorization: Token`
answers 401. `label_studio_setup.py` logs in through the web form and uses
the session. The first user and its password are in
`data/label_studio/credentials.env` (mode 600); change the password after
the first login and add each annotator as a member.

**Warning: the server is a login-node process.** It runs under `nohup` on
lisbeth with its database on NFS under `data/label_studio/data/`. A reboot
of lisbeth stops it; `bash research/scripts/label_studio_gold.sh` starts it again on
the same database.

### Phase 8 — the plan of 2026-09-20

The launch script holds 4 arms: `qwen_by_gemma_v2` (`CUESV2Q`),
`qwen_self_v2` (`CUESV2QQ`), `gemma_by_qwen_v2` (`CUESV2`), and
`gemma_self_v2` (`CUESV2GG`). The qwen half runs first, because version 1
split its traces. The gemma half runs after, so that the 2 x 2 reads 1
version.

```bash
nohup bash research/scripts/launch_trace_extraction.sh qwen_by_gemma_v2 > multirun/cuesv2q_launch.log 2>&1 &
nohup bash research/scripts/launch_trace_extraction.sh qwen_self_v2 > multirun/cuesv2qq_launch.log 2>&1 &
bash research/scripts/check_trace_extraction.sh CUESV2Q
python research/scripts/merge_trace_extractions.py multirun/<date>_CUESV2Q/* --out data/trace_extractions_v2_qwen9b
```

The roots are in `_extractions.ARMS`. The merge summary carries
`verbatim_rate`, `unanswered_traces`, and `duplicate_rate` beside
`quotable_rate`.

#### The smoke runs of 2026-09-20

120 subway traces of the qwen3.5-9b judge, 46 of them over 12,000
characters. Both arms of the qwen half, with the final version 2 config.

| Measure | gemma-4-12b extracts | qwen3.5-9b extracts |
|---------|----------------------|---------------------|
| Traces over the buffer | 0 | 0 |
| Spans for each trace | 37.0 | 57.1 |
| Silent traces | 0 | 0 |
| Values outside the enum | 0 | 0 |
| Spans inside the opening echo | 0 | 0.10 for each trace |
| Decisions for each trace | 0.98 | 0.90 |
| Traces with a decision | 96% | 89% |
| Last decision = `presented_label` | 90% | 94% |
| Quotable | 95% | 90% |
| Verbatim | 78% | 69% |
| Answers at the cap (loops) | 5 | 12 |
| Time, 120 traces on 1 GPU | 30 min | 19 min |

Under version 1 the same traces gave 3.6 decisions for each trace. The
disagreements between the last decision and `presented_label` are mostly
the unanswered traces of problem 6, and the loops of problem 7 take the
decision with the tail.

The 2 arms first launched on 2026-09-20 at 14:34 (`CUESV2Q`) and 14:35
(`CUESV2QQ`) through the monitor path, and moved to `submit_trace_shards.py`
at 21:00 (see "Submit without a monitor"): 7 cases, 8 shards each,
`slurm_gpu_preempt`. Check them with `research/scripts/check_trace_extraction.sh
CUESV2Q`, then merge `multirun/<date>_CUESV2Q/runs` into the roots of
`_extractions.ARMS`. Run the gemma half (`gemma_by_qwen_v2`,
`gemma_self_v2`) after, so that the 2 x 2 reads 1 version.

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
python -m dagspaces.pairwisevqa.cli -m pipeline=extract_traces \
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

`match_fuzzy` is a smaller version of the same problem. The review of
2026-09-20 found that 12% to 20% of the fuzzy rows differ from the source text
at their offsets. See "Review of 2026-09-20", problem 4.

Thus the rule: **count `match_exact` and `match_fuzzy`, drop `match_lesser`,
and quote `is_verbatim` only.** The `is_quotable` column holds the count test,
`is_verbatim` holds the quote test, and the merge summary reports both rates.

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

**gemma-4-12b needs `images_then_labels`.** A trace from the broken
`images_then_text` layout says the model saw 1 image. Extraction cannot repair
that.

**Never glob `multirun/`.** Discovery goes through W&B.

**An ungrounded extraction is a defect, not data.** Drop it from the counts, and
report the rate you dropped.

## Open choices

| Choice | Recommendation |
|--------|----------------|
| Which model extracts | gemma-4-12b instruct, text only. It already runs in the venv, and Phase 5 tests the swap. |
| Where the stage lives | `pairwisevqa`. Its input is a pairvqa parquet. |
| Whether to keep the clouds | Keep them. The 2 methods answer different questions. |

## Who reads this corpus

| Page | Class |
|------|-------|
| `cue-hierarchy.md` | `visual_evidence` |
| `trace-channels.md` | `comparison`, `uncertainty`, `inference`, `image_artifact` |
