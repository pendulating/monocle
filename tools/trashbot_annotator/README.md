# TrashBot deployment footage annotator

This tool watches TrashBot deployment videos. It samples frames, sends them
to a vLLM-served `google/gemma-4-12B-it`, and writes interaction,
normativity, and proxemics annotations per time window. The taxonomy
matches the `/share/ju/robot_norms` corpus.

Four channels cover the research questions:

| Channel | Question | Commands |
|---------|----------|----------|
| Windows | Who talks to or about the robot; norm evidence | `manifest`, `run`, `consolidate` |
| Proxemics | At what distance people deposit, toss, look, talk, or react | `run`, `consolidate`, `report` |
| Speech | Utterances to the robot; overheard talk about it | `speech` |
| Repeats | The same people across clips and days | `embed-frames`, `repeats` |

## Layout

| Path | Role |
|------|------|
| `setup_env.sh` | Vendors OpenCV, FFmpeg linkage, and VideoPipe under `third_party/`. Idempotent. |
| `env.sh` | Exports library paths for shells. Source it before builds or direct binary runs. |
| `CMakeLists.txt`, `src/` | The C++ pipeline app `trashbot_pipe`. |
| `config/default.json` | Endpoint, model, sampling, and request settings. |
| `config/prompt_proxemics.txt` | The default annotation prompt: talk, norms, proxemic events. |
| `config/prompt_normativity.txt` | The old prompt without Task C. Keep it for reruns of the old contract. |
| `config/prompt_speech.txt` | The utterance classification prompt. |
| `trashbot_annotator/` | Python driver package (manifest, run, consolidate, speech, proxemics, repeats). |
| `scripts/build_run_manifest.py` | Turns a manifest into run inputs: probe, proxy paths, shards, corpus. |
| `scripts/serve_gemma4.sub` | SLURM launcher for the vLLM server. |
| `scripts/embed_events.sub` | SLURM launcher for the keyframe embedding job. |
| `tests/` | Pytest suite with a stub OpenAI-compatible server. |
| `third_party/patches/` | Vendor patches for GStreamer-free builds. Kept in git. |

## Environment variables

**Warning:** do not write to `/tmp`. The cluster gives it 15 GB for all
users, and one batch of runs filled it.

| Variable | Purpose | Default |
|----------|---------|---------|
| `TRASHBOT_MAX_PENDING_FRAMES` | Cap for the analyser input queue | 48 |
| `TRASHBOT_TMPDIR` | Folder for the per-call config file | `<out-dir>/.work` |
| `TRASHBOT_LOG_DIR` | Folder for the VideoPipe log file | `./logs/vp` |
| `TRASHBOT_LOG_LEVEL` | `ERROR`, `WARN`, `INFO`, or `DEBUG` | `WARN` |

The cap stops the memory growth on long videos. The source decodes
faster than the model answers, so the queue needs a limit.

**Note:** the manifest skips Insta360 `.insv` files. They hold a
dual-fisheye stream which the model cannot read. Stitch them to mp4
first if you need them.

## Setup

Run the setup script once. It skips completed stages on reruns.

```bash
cd tools/trashbot_annotator
bash setup_env.sh          # vendors third_party/, builds libvideo_pipe.so
source env.sh              # library paths for this shell
cmake -S . -B build -DCMAKE_PREFIX_PATH="$PWD/third_party/opencv/install" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8    # produces build/trashbot_pipe
uv sync                    # python driver venv
```

No root access and no conda are needed. The build runs on CPUs.

## Serve the model

Submit the launcher. It follows the project conventions: it sources
`scripts/activate_stage_venv.sh` (node-local `/scratch` venv mirror,
CUDA 12.9 toolkit, FFmpeg libs for torchcodec) and resolves the model
through the `/scratch` registry mirrors with marker validation.

```bash
sbatch scripts/serve_gemma4.sub                 # zoo gemma-4-12B-it via registry
```

Useful overrides. **Warning:** this cluster's sbatch does not forward
the caller environment, so pass variables with `--export`:

```bash
sbatch --export=ALL,MODEL_NAME=gemma-4-12B-it scripts/serve_gemma4.sub   # registry name
sbatch --export=ALL,MODEL_ID=<hf id or path> scripts/serve_gemma4.sub    # skip resolution
sbatch --export=ALL,PORT=8001 scripts/serve_gemma4.sub
```

One-time mirror setup per node:

```bash
bash scripts/sync_venv_to_scratch.sh
sbatch -p pierson -w klara -c 4 --mem=8G --wrap \
  'bash /share/pierson/matt/mllmsci/scripts/sync_model_registry_to_scratch.sh'
```

The server needs about 24 GB of GPU memory in bf16. On smaller GPUs, use
the QAT checkpoint `google/gemma-4-12B-it-qat-w4a16-ct`. The server must
allow 8 images per prompt; the script sets this flag already. Clients
address it by the served name `google/gemma-4-12B-it`.

## Annotate footage

Three commands cover the workflow. Run them inside `tools/trashbot_annotator`.

```bash
uv run trashbot-annotate manifest --data-root /share/ju/robot_norms/data --out outputs/manifest.parquet
uv run trashbot-annotate run --manifest outputs/manifest.parquet --server-url http://localhost:8000 --out-dir outputs/windows --limit 5
uv run trashbot-annotate consolidate --out-dir outputs/windows --out outputs/annotations.parquet --manifest outputs/manifest.parquet
```

Notes:

* `run` calls `build/trashbot_pipe` once per video. It passes `--sample-id`
  so output records join manifest rows by id.
* `--fps 1.0 --batch 8` gives one request per 8 seconds of footage.
  Lower `--fps` to cut cost.
* `consolidate` expands each window to one row per interaction, one row
  per norm dimension, and one row per proxemic event. Invalid model
  labels land in `category_raw` / `dimension_raw` / `event_raw` with a
  warning.
* `--data-root` repeats for more roots. One manifest then covers all of
  them. The walk skips `consent` and `Trashbot_backup_img` folders.
* The manifest normalizes place slugs. All Albee Square folder
  spellings (`albee-square`, `albee_square`, `albee_square_pilot`)
  yield `albee-square`, so rows join the transcript corpus by place.
* The manifest parses a `control_mode` column (`auton_t2t`,
  `auton_waypoint`, `joy_t2t`) from the summer 2026 folders.

### Proxemics report

The default prompt asks for proxemic events with a distance estimate
per event. After `consolidate`, build the distance report:

```bash
uv run trashbot-annotate report --annotations outputs/annotations.parquet --out outputs/proxemics_report.parquet --by place --by control_mode
```

The report answers one question per event type: deposit distance, toss
distance, first-gaze distance, talk distance, annoyance distance, and
avoidance distance. Each block shows zone counts (intimate, personal,
social, public, after Hall) and metre quantiles. The metre values are
model estimates with noise. The zone counts are the robust view.

### Speech channel

The speech channel classifies transcript utterances, not frames. The
default corpus is `/share/ju/robot_norms/embeddings/chunk_corpus.parquet`
(2712 timestamped chunks). The same vLLM server answers text-only calls.

```bash
uv run trashbot-annotate speech --server-url http://localhost:8000 --jsonl outputs/speech/speech.jsonl --out outputs/speech/speech.parquet
```

The output marks each utterance:

* `address` — `to_robot`, `about_robot`, or `unrelated`.
* `kind` — command, greeting, question, praise, insult, warning,
  explanation, joke, or other.
* `operator` — true for likely operator or researcher talk.
* `eavesdrop` — true for bystander talk about the robot.

The run streams JSONL and resumes by `doc_id`. Rows join the window
annotations by `sample_id` and time overlap.

### Repeat interactions

Three steps find repeat-visit candidates:

```bash
uv run trashbot-annotate embed-frames --annotations outputs/annotations.parquet --out-dir outputs/repeats
sbatch --export=ALL,FRAMES_MANIFEST=$PWD/outputs/repeats/frames_manifest.parquet,OUT=$PWD/outputs/repeats/embeddings.parquet scripts/embed_events.sub
uv run trashbot-annotate repeats --embeddings outputs/repeats/embeddings.parquet --frames-manifest outputs/repeats/frames_manifest.parquet --out outputs/repeats/pairs.parquet
```

* `embed-frames` cuts one keyframe per window that holds a person
  event. It uses the zoo ffmpeg build.
* `embed_events.sub` embeds the keyframes with Qwen3-VL-Embedding-8B
  on a GPU node. It follows the venv and registry mirror conventions.
  Override with `MODEL_NAME=<zoo name>` or `MODEL_ID=<path>`, but keep
  a trained embedding model. A generative model (Qwen3-Omni, a VL
  Instruct checkpoint) has no embedding head, so its pooled states give
  uncalibrated similarities.
* `repeats` scores cosine similarity between events from different
  clips. It writes a pairs parquet and a clusters parquet. Pairs at
  one place across different dates are the repeat-visit candidates.

**Warning:** the keyframe embedding sees the person and the scene
together. Same-spot backgrounds inflate the similarity. Review the
candidate pairs in the viewer before you count them.

### Batch processing of one place

`scripts/albee_overnight.sub` processes a whole place through all four
channels in one SLURM job: parallel shard annotation with resume
passes, consolidate, report, speech, keyframe embedding, and repeat
scoring. Build the shard manifests first, then:

```bash
sbatch --export=ALL,SERVER_URL=http://klara:8001 scripts/albee_overnight.sub
```

The job needs a running vLLM server. `run --skip-existing` makes every
pass a resume pass, so a resubmit continues where the last job stopped.

### Direct binary call

```bash
source env.sh
build/trashbot_pipe --video VIDEO.mp4 --config config/default.json \
  --jsonl out.jsonl [--overlay out_dir/prefix] [--fps 1.0] [--batch 8] \
  [--max-windows N] [--sample-id ID]
```

`--overlay` writes an annotated mp4 next to the given prefix. The exact
file name gets a timestamp suffix from VideoPipe.

## Output schema

One JSON line per window, keys in this order:

`schema_version`, `video_path`, `sample_id`, `window_idx`, `t_start_s`,
`t_end_s`, `frame_indices`, `timestamps_s`, `model`, `response_raw`,
`annotation`, `error`.

`annotation` follows the prompt contract:

```json
{"interactions": [{"category": "instruction|gossip|self_talk", "address": "to_robot|about_robot|unrelated", "who": "...", "evidence": "..."}],
 "norm_dimensions": [{"dimension": "<exact name>", "relevance": 0, "evidence": "..."}],
 "proxemic_events": [{"event": "trash_deposit|trash_toss|gaze_at_robot|talk_to_robot|annoyance|avoidance", "who": "...", "distance_m": 1.5, "zone": "personal", "evidence": "..."}],
 "summary": "..."}
```

The consolidated parquet holds these columns: `dataset`, `place`,
`capture_type`, `control_mode`, `sample_id`, `video_path`,
`window_idx`, `start_s`, `end_s`, `category`, `address`, `dimension`,
`relevance`, `event`, `distance_m`, `zone`, `who`, `evidence`,
`summary`, `model`, `annotator`, `created_at`, plus `category_raw` /
`address_raw` / `dimension_raw` / `event_raw` / `zone_raw` for rejected
labels. `annotator` is always `trashbot_gemma4`. Records from the old
prompt contract consolidate too; the new columns stay null there. When
the model gives no zone, the zone derives from `distance_m`.

## Taxonomy

Talk categories map to the robot_norms labels:

| Key | Label |
|-----|-------|
| `instruction` | Instruction — commanding the robot |
| `gossip` | Gossip — talking about it with others |
| `self_talk` | Self-talk — muttering to oneself |

Norm dimensions come from `embeddings/norms_rerank.parquet` in
`/share/ju/robot_norms`: Anthropomorphism & positive affect; Conditional
acceptance & adoption; Discomfort, fear & distrust; Functional
expectations & competence; Interaction norms & legibility; Labor, jobs &
value; Legitimacy, safety & accountability; Privacy & surveillance;
Shared-sidewalk & right-of-way.

Relevance uses the robot_norms scale: 0 not relevant, 1 weak, 2 moderate,
3 strong.

Proxemic distance zones follow Hall:

| Zone | Range |
|------|-------|
| `intimate` | under 0.45 m |
| `personal` | 0.45 to 1.2 m |
| `social` | 1.2 to 3.7 m |
| `public` | over 3.7 m |
| `unknown` | no readable distance |

## Viewer

The viewer is a marimo app in `viewer.py`. It lists the annotation windows
from `outputs/**/*.jsonl`, plays each window as a spliced clip, and shows
the talk categories, norm scores, and raw model response beside the clip.

Start the service:

```bash
cd tools/trashbot_annotator
sbatch scripts/serve_viewer.sub
```

The job runs on partition `pierson` and binds port 8925 on the compute node.
Logs land in `.slurm_jobs/trashbot-viewer/`. Get the node name from `squeue`
or from the job log, then open a tunnel from your machine:

```bash
ssh -L 8925:<node-hostname>:8925 <login-host>
```

Open http://localhost:8925 in a browser.

The viewer scans `outputs/` next to `viewer.py` by default. Point
`TRASHBOT_VIEWER_DIR` at another directory to scan that one instead:

```bash
TRASHBOT_VIEWER_DIR=/share/pierson/matt/mllmsci/tools/trashbot_annotator/outputs/e2e \
  sbatch scripts/serve_viewer.sub
```

Set `TRASHBOT_FFMPEG` to force an ffmpeg binary. The default order is the
zoo build first, then the system build.

The pipe can write an annotated mp4 per sample into an `overlays/` folder
next to the data dir. File names follow `{sample_id}_{timestamp}.mp4`, and
the viewer picks the newest file per sample id. The detail view shows a
toggle between the annotated clip and the raw splice when that file fits
the 64 MB cap. Larger files show a note with the path and the size.

## Tests

```bash
uv run pytest -q        # 46 tests; the real-binary test needs build/trashbot_pipe
```

The runner injects vendored library paths into child processes, so tests
work from a clean shell.
