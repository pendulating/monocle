# The canonical run registry

Every number and every figure in the CVPR paper must come from a named run. The
registry names them. It lives at `notebooks/cvpr/canonical_data/` and it holds
28 runs: 7 cases x 2 models x 2 kinds.

| Kind | What the run is | What reads it |
|------|-----------------|---------------|
| `proxy` | the greedy, label-only run | the validation notebooks and the LaTeX results table |
| `trace` | the thinking run, with the thought channel kept | the word-block figures and the extraction panels |

## Why it exists

The battery ran 3 times with 3 different prompts (2026-08-11, 2026-08-13, and
2026-08-14). W&B holds all of them. A notebook that queried the network could
read a run that answers a question the paper no longer asks, and no reader could
see it. The registry removes that choice from the notebook.

## The layout

```
notebooks/cvpr/canonical_data/
├── manifest.json                     <- 28 runs, each with its SHA-256
├── proxy/<case>__<model>/
│   ├── results.parquet -> the labels
│   ├── pairs.parquet   -> the pair manifest
│   └── stage           -> the run directory, for the Hydra config and the logs
└── trace/<case>__<model>/  ... the same 3 links
```

A notebook reads through the **symlink**, not through the run directory. Thus a
provenance table names the registry, and a reader sees at once whether a figure
used a canonical run.

## The commands

```bash
# 1. register a new battery
python scripts/register_canonical_runs.py register --stage-root 'multirun/<sweep-dir>/*'

# 2. the gate
python scripts/register_canonical_runs.py verify

# 3. what is registered
python scripts/register_canonical_runs.py show

# 4. the exports, in this order
marimo export html notebooks/cvpr/<case>/<case>_validation.py -o /dev/null
python scripts/export_cvpr_results_table.py
python scripts/export_cvpr_trace_figures.py
```

## What the gate tests

`register` refuses a run that fails a test, and `verify` tests the registry
against the disk again:

| Test | Why |
|------|-----|
| The link resolves, and the size and the SHA-256 match | A moved or rewritten file is not the registered run |
| The grid is complete | A missing cell leaves a hole in the table |
| The 2 models of a case asked the same question | A case name is not a question |
| gemma-4-12b used `image_layout=interleaved_labels` | Without the anchor that arch does not bind image B |
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
   `scripts/merge_trace_extractions.py`, then the trace figures again. Until you
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
