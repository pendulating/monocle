# The canonical run registry

Every number and every figure in the CVPR paper must come from a named run. The
registry names them. It lives at `research/notebooks/cvpr/canonical_data/`. A cell of
the grid is a **kind**, an **arm**, a **case**, and a **model**.

| Kind | What the run is | What reads it |
|------|-----------------|---------------|
| `proxy` | the greedy, label-only run | the validation notebooks and the LaTeX results table |
| `trace` | the thinking run, with the thought channel kept | the word-block figures and the extraction panels |

## The arm

An arm says which prompt a run used. Every arm runs the same 2 model strings
and the same 7 cases, thus the model alone cannot tell an ablation from a paper
run. The arm carries that difference.

| Arm | What changes | Pairs for each case |
|-----|--------------|---------------------|
| `canonical` | nothing — the prompt the paper reports | 1,000,000 |
| `noabstain` | the decoder cannot emit `NotSure` | 1,000,000 |
| `nocity` | the prompt does not name the city | 100,000 |

`register` reads the arm from what the run DID, never from the sweep name. A
sweep can be renamed or copied; the resolved Hydra config cannot.

| Arm | Signal in `.hydra/config.yaml` |
|-----|-------------------------------|
| `noabstain` | `prompt.structured_output.allow_not_sure` is false |
| `nocity` | the pipeline name ends in `_nocity` |
| `canonical` | neither signal |

`_canonical.runs()` defaults to `arm="canonical"`. Thus a caller that knows
nothing about arms reads a paper run, and an ablation needs an explicit ask.

**Warning:** do not put a number of one arm beside a number of another without
a label. The 3 arms answer 3 different prompts.

## Why it exists

The battery ran 3 times with 3 different prompts (2026-08-11, 2026-08-13, and
2026-08-14). W&B holds all of them. A notebook that queried the network could
read a run that answers a question the paper no longer asks, and no reader could
see it. The registry removes that choice from the notebook.

## The layout

```
research/notebooks/cvpr/canonical_data/
├── manifest.json                        <- every run, each with its SHA-256
├── proxy/canonical/<case>__<model>/
│   ├── results.parquet -> the labels
│   ├── pairs.parquet   -> the pair manifest
│   └── stage           -> the run directory, for the Hydra config and the logs
├── proxy/noabstain/<case>__<model>/     ... the same 3 links
├── proxy/nocity/<case>__<model>/        ... the same 3 links
└── trace/canonical/<case>__<model>/     ... the same 3 links
```

The arm sits between the kind and the cell. A path that omits it comes from
before schema 2 and no longer resolves.

A notebook reads through the **symlink**, not through the run directory. Thus a
provenance table names the registry, and a reader sees at once whether a figure
used a canonical run.

## The commands

```bash
# 1. register a new battery — give EVERY root, see the warning below
python research/scripts/register_canonical_runs.py register \
    --stage-root 'multirun/million_merged/*' \
                 'multirun/noabstain_merged/*' 'multirun/noabstain_merged_il/*' \
                 'multirun/nocity_merged/*' 'multirun/nocity_merged_il/*' \
                 'multirun/textfirst_merged/*' 'multirun/textfirst_merged_il/*' \
                 'multirun/matched_merged/*' \
                 'multirun/2026-08-14_URBANPAIRVQA/21-28-12/*' \
                 'multirun/2026-08-14_URBANPAIRVQA/21-28-32/*' \
                 'multirun/robustness_b_merged/paraphrase/*' \
                 'multirun/robustness_b_merged/persona/*' --force
# 81 runs on 2026-09-04

# 2. the gate
python research/scripts/register_canonical_runs.py verify

# 3. what is registered
python research/scripts/register_canonical_runs.py show

# 4. the exports, in this order
marimo export html research/notebooks/cvpr/<case>/<case>_validation.py -o /dev/null
python research/scripts/export_cvpr_results_table.py
python research/scripts/export_cvpr_ablations.py
python research/scripts/export_cvpr_trace_figures.py
```

**Warning: `register` REBUILDS the manifest from the roots you give it. It
does not merge with the manifest on disk.** On 2026-09-02 a register call
with 1 root wrote a 7-run manifest over the 70-run one, and pruned every other
link directory. Always pass every root of every arm, and the 2 trace sweep
directories of 2026-08-14. `--force` is needed because the canonical gemma
runs carry a recorded layout problem.

When 2 runs match a cell, `register` takes the newest and prints a note.
Thus the `nocity_merged_il` and `textfirst_merged_il` gemma runs of
2026-09-02 win their cells over the first gemma halves, and the qwen halves
still come from `nocity_merged` and `textfirst_merged`.

The `matched` arm holds gemma-4-12b alone (`ARM_MODELS` in the script), thus
the grid check does not ask for its qwen half. See `prompt-ablations.md`.

## What the gate tests

`register` refuses a run that fails a test, and `verify` tests the registry
against the disk again:

| Test | Why |
|------|-----|
| The link resolves, and the size and the SHA-256 match | A moved or rewritten file is not the registered run |
| The grid of EACH arm is complete | A missing cell leaves a hole in the table |
| The row count clears the floor of its (kind, arm) | A smaller run is a smoke test |
| The 2 models of a case asked the same question | A case name is not a question |
| gemma-4-12b used a BINDING layout | Two adjacent image blocks read as one image on that arch |
| The gemma runs do not MIX layouts | A battery must ask every case the same way |
| No label takes more than 98% of the rows | A degenerate run carries no ordering |
| A trace run holds a trace on 95% of its rows | A run with no trace makes no word figure |
| The results join 1-to-1 to `pairs.parquet` | Every downstream step needs that join |

An accepted problem stays in the manifest and prints as a warning on every
`verify`. Nothing is accepted in silence.

## The 2 downstream gates

- `export_cvpr_results_table.py` refuses when a case export is OLDER than the
  registry, because an older export came from other runs.
- `_extractions.load()` refuses an extraction corpus whose `source_results_path`
  is not a registered trace run. The word figures then hold the words alone.

## To move the paper to a new battery

1. Run the sweeps. See `looks-like-rerun.md`.
2. `register`, then `verify`.
3. Run the 7 validation notebooks, the results table, and the trace figures.
4. Run the extraction stage on the new trace runs, then
   `research/scripts/merge_trace_extractions.py`, then the trace figures again. Until you
   do, the extraction panels stay out of the figures.

## What happened on 2026-08-17

- Registered the "looks like" battery of 2026-08-14 (28 runs).
- The gate found 1 problem: the qwen3.5-9b schools proxy run answers `NotSure`
  on 99.7% of its rows. The run is registered with that note, and the table
  shows a dash for that cell.
- Regenerated the results table for the 3 geography layers, and the word-block
  figures for all 7 cases.
- Archived the extraction panels of the 4 older cases under
  `<case>/figures/stale_pre_looks/`. They came from the 2026-08-13 traces, which
  answer the old questions.

## The required layout (2026-08-20)

`REQUIRED_LAYOUT` in `research/scripts/register_canonical_runs.py` names the ONE layout
a gemma-4-12b run may register with:

| Layout | Block order | Status |
|---|---|---|
| `images_then_labels` | image, "Image A.", image, "Image B.", prompt | REQUIRED |
| `interleaved_labels` | "Image A:", image, "Image B:", image, prompt | rejected |
| `images_then_text` | image, image, prompt | rejected |

Two rules meet in the required layout. Text must sit BETWEEN the two images,
or the arch reads them as one image. An image must open the message, which is
the modality order the gemma-4 card asks for. `interleaved_labels` keeps the
first rule and breaks the second, thus it no longer registers.

Every run registered before 2026-08-20 used `interleaved_labels`. They fail
`verify` by design, and the failure names the case to re-run.
