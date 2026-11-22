# Gotchas

These problems cost real time in this repo. Each one gives a wrong or slow
result with no clear error. Read this page before a large run.

## Quick list

| Symptom | Cause | Fix |
|---------|-------|-------|
| Rows with an empty answer and an `__image_load_error__` column | An image of the row did not load, so the model did not see the row | [Images that do not load](#images-that-do-not-load) |
| `import vllm` stops with `Could not load this library: .../libtorchcodec_core*.so` | No FFmpeg libraries on `LD_LIBRARY_PATH` when python started | [FFmpeg and torchcodec](#ffmpeg-and-torchcodec) |
| `RuntimeError: FlashInfer requires GPUs with sm75 or higher` on a new GPU | The CUDA toolkit is older than 12.9 | [Blackwell GPUs and FlashInfer](#blackwell-gpus-and-flashinfer) |
| `ValueError: Configuration is missing required 'pipeline' section` | The `pipeline=` name did not resolve, for example a typo | [A pipeline name that does not resolve](#a-pipeline-name-that-does-not-resolve) |
| Jobs of a sweep run code that you did not intend | Each job imports the code from the checkout when it starts | [Do not change the checkout during a sweep](#do-not-change-the-checkout-during-a-sweep) |
| More shards make the sweep slower | The shards overload the NFS server with image reads | [More shards can be slower](#more-shards-can-be-slower) |
| A stage takes minutes to start | The job imports the venv from NFS, not from the node-local mirror | [The mirror falls back with no error](#the-mirror-falls-back-with-no-error) |
| A job on the wrong GPU model, or a failed job that does not run again | A SLURM constraint or `--requeue` does less than it seems | [SLURM](#slurm) |

## Images that do not load

`run_vllm_inference` opens the images of each batch just before the model call.
When an image of a row does not load, the model does not see that row. The row
gets an empty answer and an `__image_load_error__` column, and the run
continues. At the end, the stage log gives the count:

```
WARNING [vqa] 3 rows have images that did not load. The model did not see these rows ...
```

| Image value | What happens |
|-------------|--------------|
| A path to a file that does not exist, or a file that PIL cannot read | The row gets `__image_load_error__` |
| An `http://` or `https://` URL | The stage does not download it. The row gets `__image_load_error__`. |
| A local path, `file://` path, `data:` base64 URI or PIL image | The image loads |

Set `runtime.strict_images=true` to stop the stage on the first row with a
missing image.

**Warning:** check the image paths before a large run. A row with an error
costs no GPU time, but it has no answer. For example:

```python
import os, pandas as pd
df = pd.read_parquet("my_images.parquet")
missing = df[~df["image_path"].map(os.path.isfile)]
print(len(missing), "missing images")
```

Download remote images to a local folder first. Then make a table with
`scripts/make_image_table.py`.

## FFmpeg and torchcodec

vLLM 0.25 imports torchcodec, and torchcodec needs the FFmpeg libraries
(`libav*.so`, versions 4 to 7). A node without them stops at `import vllm`.

The dynamic loader reads `LD_LIBRARY_PATH` one time, when the process starts.
Thus a setting that python loads (for example `server.env`) cannot fix it. The
folder must be on `LD_LIBRARY_PATH` before python starts.

- The SLURM launchers source `scripts/activate_stage_venv.sh`, which adds
  `IMAGE_FORWARD_FFMPEG_LIBDIR` from `server.env`. A stage job needs nothing more.
- In your own shell (`hydra/launcher=local`, a notebook, a test of a stage),
  source the same script after you activate the venv:

```bash
source .venv/bin/activate
source scripts/activate_stage_venv.sh
```

## Blackwell GPUs and FlashInfer

On a Blackwell GPU (sm_120, for example an RTX PRO 6000), FlashInfer can stop
with `FlashInfer requires GPUs with sm75 or higher`. The message names the wrong
cause. FlashInfer compiles its kernels for the target `compute_120f`, and only
nvcc 12.9 or later can build that target. With an older toolkit the list of
targets is empty, and the check reports sm75.

The fix:

1. Install a CUDA 12.9 toolkit with `scripts/install_cuda_129_toolkit.sh [PREFIX]`.
   It needs no root and about 290 MB. The default PREFIX is `$HOME/cuda-12.9`.
2. Set `IMAGE_FORWARD_CUDA_HOME` to it in `server.env`.
   `scripts/activate_stage_venv.sh` then sets `CUDA_HOME` in each job.

**Note:** the pip package `nvidia-cuda-nvcc-cu12` is not a toolkit. It holds
`ptxas` only. Use the NVIDIA redistributable tarballs, as the script does.

**Note:** a `CUDA_HOME` in `~/.bashrc` does not stop the fix.
`activate_stage_venv.sh` sets `CUDA_HOME` from `IMAGE_FORWARD_CUDA_HOME` when the
key is set, and the stage log prints the change.

## A pipeline name that does not resolve

The defaults list of each dagspace has `optional pipeline:`. When the name of
`pipeline=` does not resolve, Hydra composes the config with no pipeline and no
error. The run then stops with:

```
ValueError: Configuration is missing required 'pipeline' section
```

Check the name for a typo. A pipeline in `sv/conf/` or `research/conf/` needs
the search-path plugin, so it is not found when
`IMAGE_FORWARD_NO_PROJECT_ROOTS=1` is set. To list the pipelines that Hydra
sees, run `dagspace <name> --help` and read the `pipeline:` line.

## Do not change the checkout during a sweep

A SLURM stage job imports `dagspaces` (and `research`, `sv`) from the repo
folder when the job starts, not when you submit it. A job that waits in the
queue thus runs the code of the checkout at its start time.

**Warning:** do not edit the code, switch the branch or pull while a sweep
runs from this checkout. Shards that start later run other code than the first
shards, and a moved file stops them. Develop in a second clone or in a
`git worktree` during a long sweep.

## More shards can be slower

Each image read from a deep NFS tree costs about 8 round trips to the server.
When many shards read images at the same time, the server slows down for each
shard. We measured this on a 322-shard sweep:

| Shards at the same time | Rows/s for each shard | Rows/s in total |
|-------------------------|-----------------------|-----------------|
| 15 to 17 | 12 to 14 | about 210 |
| 30 | 4.3 | about 130 |
| 78 to 80 | 1.2 | about 95 |

Above about 30 shards, the total rate falls. Shards then run past their time
limit.

- Hold the number of running shards at about 20: `#SBATCH --array=0-321%20`,
  or `scontrol update JobId=<id> ArrayTaskThrottle=20` for a running array.
- For a large table, build an image pack (see
  [streaming-and-resume.md](streaming-and-resume.md)). The pack gave 3.5 times
  the rate on the same GPU.
- Do not increase `model.image_decode_workers` when the images are on NFS with
  no pack. More readers make the server slow down sooner.

## The mirror falls back with no error

When a job cannot use the node-local venv mirror, it uses the NFS venv. The
result is the same, but the start can take 10 minutes or more. The job gives no
error. Read the stage log:

```
[stage_venv] node-local venv on <node>: ...                       the mirror is in use
[stage_venv] no matching scratch mirror on <node> ...; the stage runs from NFS
```

The usual causes:

- The monitor started from a different venv than `IMAGE_FORWARD_VENV_ACTIVATE`.
- The node has no mirror yet, and the bootstrap tarball does not exist.

**Warning:** in a SLURM script of your own, `source <mirror>/bin/activate` does
not change the interpreter. The mirror is a copy, so its `activate` script
names the NFS venv. Source `scripts/activate_stage_venv.sh`, and run
`"${IMAGE_FORWARD_STAGE_PYTHON:-python}"`.

**Warning:** a venv holds its own absolute path in many files. Do not copy or
move a venv to a new folder. Build it again with `scripts/build_venv.sh`.

See [models-and-mirrors.md](models-and-mirrors.md).

## SLURM

- A `--constraint` selects a node, not a GPU. A node can hold 2 GPU models, and
  `--gres=gpu:1` can then get either one. Give the type when it matters, for
  example `--gres=gpu:nvidia_rtx_a6000:1`.
- `--requeue` covers a preemption and a node failure. It does not run a job
  again after an error exit or a `TIMEOUT`. Use `model.resume_dir`, and submit
  the shards that did not finish again.
- A free GPU does not mean that your job starts. The job must still outrank the
  queue.
