# TrashBot footage annotator

Tool location: `tools/trashbot_annotator/`. Usage guide:
`tools/trashbot_annotator/README.md`. This page records the design, the
evidence path, and the limits.

## Purpose

TrashBot deployments produce hours of park footage. People talk to the
robot, about it, and about sidewalk norms. This tool annotates that
footage with a vision language model. Each output row names an interaction
type, a norm dimension, or a proxemic event, with evidence.

Four channels serve the research questions:

| Channel | What it measures |
|---------|------------------|
| Windows | Talk categories and norm relevance per 8-second window |
| Proxemics | Person-robot distance at deposit, toss, gaze, talk, annoyance, avoidance |
| Speech | Transcript utterances to the robot; overheard talk about it |
| Repeats | The same people across clips and days, by keyframe embeddings |

## Architecture

A C++ VideoPipe pipeline handles video. A Python driver handles
orchestration. A vLLM server hosts the model.

* VideoPipe decodes the file and pushes frames through nodes.
  `vp_frame_batch_analyser_node` samples frames at a target fps, packs
  batches of JPEG frames, and calls the server once per batch.
* Two controls keep the memory flat. `main.cpp` gives the source a
  `skip_interval`, so the source drops the frames which the analyser
  does not sample. A hook on the source then holds the produce thread
  while the analyser input queue is above `TRASHBOT_MAX_PENDING_FRAMES`.
  See "Flow control" below.
* The analyser writes one JSON line per window. It draws a summary line on
  the last frame of each window for overlay videos.
* The Python CLI builds manifests from `/share/ju/robot_norms/data`, drives
  the binary per video, and consolidates JSONL into one parquet.
* The model is `google/gemma-4-12B-it` behind the OpenAI-compatible vLLM
  API. Frames go in as interleaved text+image content parts with mm:ss
  labels. Temperature 0 and a fixed seed keep runs deterministic.

## Flow control

The source node reads the file at the real-time rate and pushes decoded
frames into a queue with no limit. The analyser blocks on each HTTP
call, so it reads much slower than the source writes. The queue grew
until the job died. Every video longer than about 20 minutes failed
this way with `rc=-9`.

Three changes fix it:

| Change | Effect |
|--------|--------|
| `skip_interval` on the source | The source drops 29 of 30 frames before it decodes them |
| A hook on the source | The produce thread waits while the queue is above the cap |
| `source_fps` on the analyser | The analyser reads the true rate from the caller |

**Warning:** `vp_file_src_node` keeps its fps in an `int` and divides it
by `skip_interval + 1`. That division truncates to 0 as soon as the skip
interval is larger than the fps. The frame meta then carries a wrong
rate, and the analyser samples 25 times too slowly. `main.cpp` passes
the true rate in `batch_analyser_params.source_fps`, so the analyser
never reads the rate from the meta.

Environment variables:

| Variable | Purpose |
|----------|---------|
| `TRASHBOT_MAX_PENDING_FRAMES` | Input queue cap. Default 48, about 130 MB at 720p |
| `TRASHBOT_LOG_DIR` | Folder for the VideoPipe log file. Default `./logs/vp` |
| `TRASHBOT_LOG_LEVEL` | `ERROR`, `WARN` (default), `INFO`, or `DEBUG` |
| `TRASHBOT_TMPDIR` | Folder for the per-call config file. Default `<out-dir>/.work` |

## Disk and log discipline

**Warning:** do not write to `/tmp`. This cluster gives `/tmp` 15 GB for
all users.

The vendor logger writes at DEBUG level to the console and to a file. It
writes some lines per frame. One overnight batch left two log files of
2.96 GB and 2.27 GB in `/tmp`, and a shard driver then died with
`OSError: [Errno 28] No space left on device`. The console half of the
same output went to the driver, which holds it in memory.

The tool now sets the level to WARN, turns the console off, and writes
the log file under the output folder. The per-call config file follows
the same rule.

## Proxemics channel

The default prompt (`config/prompt_proxemics.txt`) adds Task C: proxemic
events. The model reports each visible event with a metre estimate
(`distance_m`) and a Hall zone (`intimate` under 0.45 m, `personal` to
1.2 m, `social` to 3.7 m, `public` over 3.7 m). The six event types map
one-to-one to the deployment questions:

| Event | Question |
|-------|----------|
| `trash_deposit` | How close does the robot come before people drop trash? |
| `trash_toss` | At what distance do people toss trash into TrashBot? |
| `gaze_at_robot` | How close is the robot before people look at it? |
| `talk_to_robot` | How close is the robot before people talk to it? |
| `annoyance` | How close is too close? |
| `avoidance` | At what distance do people step aside? |

`trashbot-annotate report` aggregates the events into zone counts and
metre quantiles, overall and per group (`--by place`, `--by
control_mode`). The metre values are model estimates with noise. Read
the zone counts as the robust result. The distributions describe the
footage; they are not causal thresholds.

The C++ binary needed no change. It parses the annotation JSON
generically, so the new fields pass through the window records.

## Speech channel

`trashbot-annotate speech` classifies the robot_norms utterance corpus
(`embeddings/chunk_corpus.parquet`, 2712 chunks with timestamps) with
text-only calls to the same served model. Each utterance gets:

* `address`: `to_robot`, `about_robot`, or `unrelated`.
* `kind`: command, greeting, question, praise, insult, warning,
  explanation, joke, other.
* `operator`: likely operator or researcher talk.
* `eavesdrop`: bystander talk about the robot (address `about_robot`
  and operator not true).

The classifier sees the utterance with two neighbor utterances on each
side. The run streams JSONL and resumes by `doc_id`. Rows join the
window annotations by `sample_id` and time overlap, and the windows
channel marks the same distinction visually with the `address` field on
each interaction.

## Repeat interactions

The repeat channel finds candidates for the same person across clips:

1. `embed-frames` cuts one keyframe per window with a person event.
2. `scripts/embed_events.sub` embeds the keyframes with
   Qwen3-VL-Embedding-8B (vLLM pooling runner, registry mirror rules).
   This is the strongest trained multimodal embedder in the zoo. A
   generative model (Qwen3-Omni, a VL Instruct checkpoint) is not a
   substitute: it has no embedding head, so cosine similarity over its
   pooled states is not calibrated.
3. `repeats` scores the cosine similarity between events from
   different clips, and clusters the pairs with union-find.

**Warning:** the embedding sees the person and the scene together.
Same-spot backgrounds inflate the similarity, so cross-day pairs at one
place are the meaningful candidates, and each needs a manual check in
the viewer. A person-crop embedding (the repo holds YOLO pose weights)
is the next step when the scene confound proves too strong.

## Corpus coverage

The manifest now covers the full data root in one pass:

* Multiple `--data-root` flags combine into one manifest.
* Place slugs normalize to hyphen form. All Albee Square spellings
  (`albee-square`, `albee_square`, `albee_square_pilot`) yield
  `albee-square`, which matches the transcript corpus.
* The summer 2026 folders yield a `control_mode` column (`auton_t2t`,
  `auton_waypoint`, `joy_t2t`).
* The walk skips `consent` and `Trashbot_backup_img` folders.
* The walk drops the Insta360 `.insv` files. See "Dropped footage".
* A full walk of `/share/ju/robot_norms/data` gives 642 videos; 58 sit
  at Albee Square (dataset `Bu_TrashbotWoz`).
* `scripts/build_run_manifest.py` turns a manifest into the run inputs:
  it probes each video, gives a proxy path to each video above 720p,
  splits the rows into shards of equal duration, and filters the
  utterance corpus for the same place. It also drops a file which holds
  no video stream.

## Dropped footage

The manifest no longer collects `.insv` files. They hold a dual-fisheye
stream. OpenCV reads it as one flat frame, so the model sees two
distorted circles, not a scene. The files also crashed the binary in
earlier runs.

**Warning:** this cut is not neutral at Albee Square. All of the
landfill and the recycle footage there is `.insv`.

| Capture type | Videos | Format | State |
|--------------|--------|--------|-------|
| interview | 58 | mp4 | Kept |
| landfill | 22 | insv | Dropped |
| recycle | 22 | insv | Dropped |

The proxemics channel measures deposit and toss distance. That behaviour
sits in the landfill and the recycle clips, not in the interviews. To
get it back, stitch the `.insv` files to mp4, or crop one fisheye half
with ffmpeg, then point `--data-root` at the result. The speech channel
does not lose anything: it reads transcripts, so all 494 Albee
utterances stay.

We found no folder for footage from Albert. The name appears only
inside transcripts as a person at the 2025 little-italy deployments.
When that footage arrives, one more `--data-root` flag covers it.

## Data alignment

The output schema mirrors the robot_norms annotation tables
(`notebooks/annotations.parquet`, `interaction_annotations.parquet`):

* Talk categories: instruction, gossip, self-talk. They map to the exact
  labels from `interaction_explorer.py`.
* Nine norm dimensions come from `norms_rerank.parquet`, with the 0-3
  relevance scale from the same notebooks.
* `sample_id` follows the `baseline_videos.parquet` convention: relative
  path with separators replaced by `_`, no extension.
* `dataset`, `place`, and `capture_type` derive from the path segments,
  so rows join the existing corpus by id or by columns.

## Vendor stack

No root access and no conda on the cluster shaped these choices:

* OpenCV 4.10 builds from source into `third_party/opencv/install`, with
  FFmpeg decode against the shared tree at
  `/share/pierson/matt/zoo/ffmpeg-libs/n7.1`.
* VideoPipe clones at a pinned commit. Two patches make it work without
  GStreamer: one gates configure checks and drops the RTSP sink node; one
  switches the file source/sink nodes to default OpenCV backends. The
  patches live in git under `third_party/patches/`.
* `setup_env.sh` reproduces all of it and skips finished stages.

## Verification evidence

Claims and the evidence that supports them:

| Claim | Evidence |
|-------|----------|
| Toolchain builds without root | `cmake` + `make` exit 0; `ldd` resolves all 47 entries; fresh-clone patch diff identical |
| Binary runs on real footage | 3-window smoke on a 3840x1920 GoPro clip, exit 0 in 22 s |
| Records match the contract | All 12 keys present and ordered; timestamps increase; error null |
| Python path works end to end | manifest 281 rows; run + consolidate produced a 116-row parquet with correct labels |
| Taxonomy matches robot_norms | Exact-string tests over both label sets |
| Suite health | 46 pytest tests pass (2026-08-30) |
| Proxemics + speech + repeats logic | Unit tests cover the flatten path, the report quantiles, the classifier stream with resume, and the pair scoring |
| Manifest covers the full corpus | Real walk: 642 videos, 58 at albee-square, control modes parsed. The .insv files stay out |
| Speech loader reads the real corpus | 2712 utterances load; 494 at albee-square; the prompt renders |
| Albee Square smoke, all four channels | 2026-08-30: 3 clips × 4 windows → 40 rows, 0 invalid labels; deposit at 0.3 m found; 28/28 utterances classified; 2 keyframes embedded (8B, dim 4096); 1 cross-day pair scored |
| Flow control holds the memory flat | A/B on one 708 s clip, stub server at 20 s per call: no cap grew 93 MB to 467 MB in 225 s; cap 48 rose to 250 MB and stayed there. Same 11 windows, 0 errors, same timestamps |
| Skip and rate stay consistent | 52.8 s clip gives 7 windows over 0 to 51.1 s, exit 0. Window starts land on 0.0, 8.008, 16.016 s |
| Suite health after the fixes | 52 pytest tests pass (2026-09-02) |
| Live model path works | One window on real footage against the served gemma-4-12B-it; 9 dimensions rated, gossip interaction found, 27 s wall |

## Limits

* The launcher follows the project conventions: `activate_stage_venv.sh`
  picks the node-local `/scratch` venv mirror, and the model resolves
  through `/scratch/$USER/registry/models` with marker validation.
* The cpp_llmlib client collapses HTTP failures to an empty string, so
  error records say "empty response" rather than the HTTP status.
* The source paces itself at the real-time rate, so a run never takes
  less wall time than the footage. The model latency dominates anyway:
  one 8-second window costs about 27 s.
* The `skip_interval` applies only without `--overlay`. An overlay video
  needs every frame, so an overlay run keeps the old memory profile.
  Use `--max-windows` for an overlay run on a long video.
* With `cycle=false`, the file source never signals EOF downstream. The
  app detects EOF by frame count and flushes the tail window through a
  control message.
* Overlay output uses the mp4v codec through FFmpeg after the patch. The
  exact output file name carries a timestamp suffix.
* The video path stays audio-free. The speech channel covers audio
  through the existing transcript corpus instead. Footage without a
  transcript needs an urbanspeech ASR pass first.
* The metre estimates in the proxemics channel come from a monocular
  VLM. Validate a sample against known object sizes before a paper
  claim rests on the metre scale.
* The repeat channel embeds whole keyframes, not person crops. The
  scene confound needs a manual review of each candidate pair. The
  Albee smoke shows the size of it: two frames from different days at
  one deposit opening score 0.94.
* This cluster's sbatch does not forward the caller environment. Pass
  job variables with `sbatch --export=ALL,VAR=value`.
* One transient segfault (rc=-11) hit two `.insv` clips while the
  speech classifier ran against the same server. Four retries all
  passed. When a clip fails, run it again before you debug.
