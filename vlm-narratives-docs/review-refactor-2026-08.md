# Review Refactor Round - August 2026

## Summary

This round shrank the pipeline plumbing and tidied the log paths. One shared
SLURM driver replaced nine copied orchestration blocks. The W&B logger lost
two helper modules to their own files. A subprocess worker moved to its own
module. Two P0 fixes came back after they were lost in an edit. About 4,000
dead lines left the tree. Stage code now logs through one stdlib path, and
SLURM scripts write their output to one root.

## Shared SLURM driver

| Item | State |
|---|---|
| Module | `dagspaces/common/stage_submission.py` |
| Settings type | `DagspaceSettings` (frozen dataclass) |
| Entrypoints | `execute_stage_job`, `run_experiment`, `_submit_and_wait`, `_recover_slurm_job` |

Nine dagspace orchestrators repeated one submit-wait-recover block and one
pipeline loop. The copies differed in small ways: job-name prefix, GPU
preflight on or off, dotenv call, W&B summary detail, recovery mode, manifest
fields. `DagspaceSettings` holds each difference as an explicit field. Each
orchestrator builds one settings object and delegates. Public names stayed
stable, so CLIs, tests, and scripts kept working.

Two driver styles exist:

- `monitor`: per-stage W&B run, GPU preflight, submit-wait-recover.
- `inline`: plain submit, block on the job, no recovery.

Job recovery has two modes. `basic` polls squeue, then reads the result
pickle. `nfs_poll` also polls the pickle for 120 seconds and checks the
outcome, because a fast job can finish before NFS shows its pickle.

Node dicts travel to jobs through `serialize_node` and come back through
`deserialize_node`. The rebuild uses `PipelineNodeSpec.from_config`.

## WandB logger split

`wandb_logger.py` was too large. Two helper groups moved out:

| New module | Contents |
|---|---|
| `dagspaces/common/compute_metadata.py` | `collect_compute_metadata`, checkpoint-name derivation |
| `dagspaces/common/host_info.py` | GPU/CPU/memory detection with a cached probe |

The logger class and its table logic stayed in place. Call sites import the
helpers from their new homes.

## DP worker module move

The data-parallel worker entry moved from `vllm_inference.py` to
`dagspaces/common/dp_worker_full.py`. The parent spawns it as a subprocess
with a task pickle and a result pickle. Its prints stay prints: its stdout is
the transport to the SLURM output file.

## P0 fixes restored

Two fixes went missing during concurrent edits, and this round restored them:

| Fix | Location | Role |
|---|---|---|
| `_prepare_image_content` | `vqa/stages/vqa.py` | Normalizes PIL images, file paths, URLs, and base64 strings into vLLM chat content blocks |
| `log_grouped_metrics` | `dagspaces/common/wandb_logger.py` | Logs grouped stage metrics under one key |

## Dead code removal

About 4,000 lines left the tree:

- Nine copied SLURM blocks in the dagspace orchestrators.
- Twelve bespoke runner classes in embed, artifact_gen, and urbanvit.
- The unused `_NoOpLogger` class.
- Old helper code that moved to `compute_metadata.py` and `host_info.py`.
- Worker code that moved to `dp_worker_full.py`.

## Caching

| Cache | Location | Key |
|---|---|---|
| Jinja compiled templates | `vqa/stages/vqa.py` | Template string, `lru_cache(512)` |
| Gemma4 unified chat template | `common/vllm_inference.py` | Model source |
| Sampling params objects | `common/vllm_inference.py` | Source dict identity |
| Resume chunks | `common/vllm_inference.py` | Prompt hash, opt-in |
| Host probe results | `common/host_info.py` | Process, `lru_cache(1)` |
| Street graphs | `urbanroamvqa/graph/cache.py` | Config fingerprint + parquet path |
| Prebuilt pair tables | `pairwisevqa/orchestrator.py` | Sampler fingerprint sidecar |

Parquet reads also prune columns and probe the schema arrow metadata before a
full read. Image inputs decode locally from base64 data URIs; remote URLs are
not fetched.

## Logging design

### Problem

Stage code held about 1,247 bare `print()` calls against 36 logger calls.
Ten ad-hoc `basicConfig` calls fought over the root logger. No handler wrote
to a file. SLURM outputs landed in `logs/`, `outputs/`, and `.slurm_jobs/`
with no pattern.

### Agreement

- Prints stay inside subprocess workers. Their stdout is the transport.
- In-process stages get one stdlib logger path with a stage-prefixed format.
- Only ops-valuable prints convert. No new framework.

### API

| Symbol | Behavior |
|---|---|
| `get_stage_logger(name)` | Returns a stdlib logger with one stderr handler |
| Format | `%(asctime)s %(name)s %(levelname)s %(message)s` |
| Level | `MLLMSCI_LOG_LEVEL` env var, default INFO |
| Idempotence | Repeated calls add no second handler |
| Propagation | Off, so each line prints once |

Module: `dagspaces/common/log_setup.py`. Stdlib only.

### Converted sites

| File | Message | Old | New |
|---|---|---|---|
| `common/vllm_inference.py` | `{n_errors} request errors — rows marked with __postprocess_error__` | print | warning |
| `common/vllm_inference.py` | `DP rank {rank}: {n} rows, CUDA_VISIBLE_DEVICES=...` | print | info |
| `common/vllm_inference.py` | `DP rank {rank}: {n} inputs, CUDA_VISIBLE_DEVICES=...` | print | info |
| `common/vllm_inference.py` | `skipping remote image URL (not fetched): ...` | print | warning |
| `urbanroamvqa/graph/builder.py` | `Dropped {n} recordings with NaN/inf coordinates` | print | warning |
| `urbanroamvqa/graph/builder.py` | `Graph built: {n} nodes, avg {x} neighbors` | print | info |
| `embed/stages/embed.py` | `{n} rows to embed (streaming={flag})` | print | info |
| `embed/stages/embed.py` | `Runtime env: {...}` | print | info |
| `pairwisevqa/stages/pairwise_vqa.py` | `Preparing {n} pairs...` | print | info |
| `artifact_gen/stages/raster.py` | `Embedding matrix: {shape}` | print | info |
| `common/curation/cli.py` | `basicConfig` setup | ad-hoc root config | `get_stage_logger("sv.nyc.curation")` |

Notes on text changes:

- The literal `WARNING:` token inside two old messages is gone. The level
  name now carries the severity. Greps on the rest of the text still match.
- The curation CLI format changed from `%H:%M:%S %-7level :: message` to the
  standard stage format. The `-v` flag still forces DEBUG.

All other prints stay. `dp_worker_full.py` keeps every print.

## SLURM output roots

Every root-level `.sub` script now writes to
`/share/ju/matt/mllmsci/logs/<script-name>-%j.out` and
`...-%j.err`. Old jobs wrote where the table shows; new jobs write to
`logs/`.

| Script | Old stdout/stderr | New stdout/stderr |
|---|---|---|
| build_cyclomedia_catalog.sub | `.slurm_jobs/cyclomedia_catalog/%j.{out,err}` | `logs/build_cyclomedia_catalog-%j.{out,err}` |
| gepa_pairwise_subway.sub | `outputs/gepa_pairwise/slurm_%j.out`, stderr merged | `logs/gepa_pairwise_subway-%j.{out,err}` |
| gepa_probe.sub | `outputs/gepa_pairwise/slurm_%j.out`, stderr merged | `logs/gepa_probe-%j.{out,err}` |
| labelling-server.sub | `.slurm_jobs/labelling-server/%j.{out,err}` | `logs/labelling-server-%j.{out,err}` |
| materialize_open_restaurants_cyclomedia.sub | `.slurm_jobs/curation_open_restaurants/%j.{out,err}` | `logs/materialize_open_restaurants_cyclomedia-%j.{out,err}` |
| materialize_scaffolding_cyclomedia.sub | `.slurm_jobs/curation_scaffolding/%j.{out,err}` | `logs/materialize_scaffolding_cyclomedia-%j.{out,err}` |
| rebuild_cyclomedia_queens.sub | `.slurm_jobs/cyclomedia_catalog/%j.{out,err}` | `logs/rebuild_cyclomedia_queens-%j.{out,err}` |
| run_nli_classify.sub | `.slurm_jobs/nli-classify/%j.{out,err}` | `logs/run_nli_classify-%j.{out,err}` |
| run_qwen30b_vllm.sub | `logs/qwen30b-vllm-%j.{out,err}` | `logs/run_qwen30b_vllm-%j.{out,err}` |
| run_qwen3vl30b_vllm.sub | `.slurm_jobs/qwen3vl30b-vllm/%j.{out,err}` | `logs/run_qwen3vl30b_vllm-%j.{out,err}` |

Old files stay where they are. Move nothing by hand; greps over old runs
keep their old paths.

## Validation

- `get_stage_logger` called twice in one process: same logger, one handler.
- `py_compile` on every touched Python file: pass.
- `pytest tests/test_roaming_vqa.py -q`: pass.
- `bash -n` on all ten edited `.sub` files: pass.
