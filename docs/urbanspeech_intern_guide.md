# UrbanSpeech: How to Transcribe Your Video Files

A beginner-friendly guide to running the speech-to-text pipeline on your own videos.

This pipeline takes a folder of video clips, pulls the audio out of each one, and
uses an AI speech model (IBM **granite-speech**) to write down what's being said.
It runs on the lab's **JU** GPU machines through SLURM — you don't manage the GPUs
yourself, you just submit a job and wait for the results.

---

## 0. One-time setup

Do these once when you first log in to the cluster.

```bash
# Go to the project
cd /share/ju/mllmsci

# Turn on the Python environment (you'll do this every fresh terminal session)
source .venv/bin/activate
```

> 💡 If you open a new terminal later, you **only** need the `cd` and the
> `source .venv/bin/activate` lines again before running anything below.

---

## 1. Where your videos go

Put (or wait for) your video files here:

```
/share/ju/robot_norms/data/little-italy/
```

The pipeline searches **recursively** for `.mp4` files, so subfolders are fine —
even nested ones with spaces in the name (e.g. `Oct 1/Recycle/clip.mp4`). Each
video becomes one row in the final spreadsheet of transcripts.

A few things that are handled automatically, so you don't have to worry about them:

- **Subfolders & spaces** in folder names work fine.
- **Many video formats** are processed: `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`,
  `.ts`, `.m4v`, and `.insv` (the raw Insta360 format). Other files (images,
  zips, etc.) are skipped automatically.
- **Duplicate-looking files** like `clip(1).mp4` are treated as separate clips
  (they each get their own row).

### Skipping folders you don't want (e.g. `Interview`)

If some subfolders shouldn't be transcribed, use the **blacklist** option. Add
`data.video_exclude=[Interview]` to your command and any video whose path
contains "Interview" (case-insensitive) is skipped:

```bash
python -m dagspaces.urbanspeech.cli -m \
  pipeline=asr_videos \
  data.video_dir=/share/ju/robot_norms/data/little-italy \
  'data.video_exclude=[Interview]'
```

> ⚠️ Keep the **single quotes** around `'data.video_exclude=[Interview]'` — the
> square brackets confuse the shell otherwise.

You can list more than one folder/word to skip, separated by commas:
`'data.video_exclude=[Interview,Test,scratch]'`.

---

## 2. The basic command

Once the videos are in place, this is the whole thing:

```bash
cd /share/ju/mllmsci
source .venv/bin/activate

python -m dagspaces.urbanspeech.cli -m \
  pipeline=asr_videos \
  data.video_dir=/share/ju/robot_norms/data/little-italy
```

That's it. Let's break down what each part means:

| Part | What it does |
|------|--------------|
| `-m` | **Always include this.** It tells the system to submit SLURM jobs (the `-m` is required — the pipeline won't run correctly without it). |
| `pipeline=asr_videos` | The recipe: "extract audio, then transcribe." |
| `data.video_dir=...` | The folder with your videos. Change this to point at your data. |

The command returns quickly, but the actual work runs on the cluster in the
background. See [section 5](#5-checking-on-your-job) to watch its progress.

---

## 3. Do a tiny test run first (recommended!)

Before transcribing hundreds of videos, run on just a few to make sure everything
works. Add two options:

```bash
python -m dagspaces.urbanspeech.cli -m \
  pipeline=asr_videos \
  data.video_dir=/share/ju/robot_norms/data/little-italy \
  runtime.sample_n=3 \
  runtime.debug=true
```

- `runtime.sample_n=3` → only process the first 3 videos.
- `runtime.debug=true` → extra logging so you can see what's happening.

If that finishes and produces transcripts, you're ready for the full run (just
drop those two lines).

---

## 4. Where your results land

Each run creates a timestamped folder under `multirun/`:

```
multirun/<date>_URBANSPEECH/<time>/0/outputs/asr/transcripts.parquet
```

For example, the demo run lives at:

```
multirun/2026-06-09_URBANSPEECH/15-37-10/0/outputs/asr/transcripts.parquet
```

The output is a **parquet** file (a spreadsheet format). To peek at it in Python:

```bash
python -c "import pandas as pd; df = pd.read_parquet('multirun/2026-06-09_URBANSPEECH/15-37-10/0/outputs/asr/transcripts.parquet'); print(df[['sample_id','transcript']].head())"
```

(Swap in the path to *your* run's `transcripts.parquet`.)

Useful columns in the output:

| Column | Meaning |
|--------|---------|
| `sample_id` | An ID for each clip (derived from the filename). |
| `video_path` | The original video file. |
| `transcript` | **The transcribed speech** — what you came for. |
| `audio_duration_s` | How long the clip's audio was. |
| `has_audio` | `False` if the video had no audio track. |
| `asr_error` / `extract_error` | Filled in only if something went wrong on that clip. |

---

## 5. Checking on your job

After you submit, the work runs on the cluster. To see your running jobs:

```bash
squeue --me
```

You'll see two stages run one after another: first `extract_audio` (CPU), then
`asr` (GPU). When `squeue --me` shows nothing, the run is done.

To watch the live log of a run (replace the path with your run's folder):

```bash
tail -f multirun/<date>_URBANSPEECH/<time>/0/URBANSPEECH.log
```

(`Ctrl-C` to stop watching — that does **not** stop the job.)

To cancel a job if you started it by mistake:

```bash
scancel <job_id>      # the job id is shown by `squeue --me`
```

---

## 6. Optional: using the bigger, more accurate model

By default the pipeline uses the **2B** model (fast, 1 GPU). There's also an **8B**
model that can be more accurate but needs **2 GPUs**. To use it, add these two lines
(this is exactly what the demo run did):

```bash
python -m dagspaces.urbanspeech.cli -m \
  pipeline=asr_videos \
  data.video_dir=/share/ju/robot_norms/data/little-italy \
  model=granite_speech_3_3_8b \
  pipeline.graph.nodes.asr.launcher=slurm_gpu_ju_2x
```

> Start with the default 2B model. Only switch to 8B if you find the 2B
> transcripts aren't accurate enough — the 8B run is slower and uses more GPUs.

---

## 7. Quick troubleshooting

| Problem | What to check |
|---------|---------------|
| `command not found` / import errors | Did you run `source .venv/bin/activate`? |
| "No videos found" | Is `data.video_dir` pointing at a folder that actually has `.mp4` files? |
| Job sits in the queue forever | The JU GPUs may be busy — `squeue --me` will show state `PD` (pending). Just wait. |
| `transcript` is empty/`None` for a clip | That clip may have had no speech or no audio track — check the `has_audio` column. |
| Something else weird | Look at the run's `URBANSPEECH.log` (section 5) — errors get printed there. |

---

## Cheat sheet

```bash
# every session
cd /share/ju/mllmsci
source .venv/bin/activate

# tiny test (3 clips)
python -m dagspaces.urbanspeech.cli -m pipeline=asr_videos \
  data.video_dir=/share/ju/robot_norms/data/little-italy \
  runtime.sample_n=3 runtime.debug=true

# full run
python -m dagspaces.urbanspeech.cli -m pipeline=asr_videos \
  data.video_dir=/share/ju/robot_norms/data/little-italy

# full run, skipping the Interview folder
python -m dagspaces.urbanspeech.cli -m pipeline=asr_videos \
  data.video_dir=/share/ju/robot_norms/data/little-italy \
  'data.video_exclude=[Interview]'

# check progress
squeue --me
```

Good luck! When in doubt, run the tiny test first and ask Matt if a run errors out.
