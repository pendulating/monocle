# image-forward

image-forward runs vision-language models over large image tables on GPU
clusters. You give a table of images and a prompt. The framework runs the model
with vLLM, in a local process or in SLURM jobs, and writes a parquet table of
the answers.

- **Configs, not code.** Hydra configs give the data, the model, the prompt and
  the pipeline. A pipeline is a DAG of stages.
- **SLURM.** A small monitor process submits one GPU job for each stage and
  passes the outputs between the stages.
- **Large runs.** Batches stream through `llm.chat`. A stage can resume after a
  preemption, and it can read images from a pack in place of many NFS files.
- **New tasks.** `scripts/new_dagspace.py` makes a new task from a template
  that runs on the 10 example images.

## Quickstart

You need Linux x86_64, an NVIDIA GPU with a CUDA 12.9 driver, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:pendulating/monocle.git image-forward
cd image-forward
bash scripts/build_venv.sh            # the locked environment in .venv (vLLM 0.25, torch 2.11)
source .venv/bin/activate
```

On a node with a GPU, describe the 10 example images:

```bash
dagspace vqa -m hydra/launcher=local pipeline=example_images
```

The first run downloads `Qwen/Qwen2.5-VL-3B-Instruct` from Hugging Face. The
output is a parquet table with one answer for each image:

```
multirun/<date>_VQA_EXAMPLE/<time>/0/outputs/vqa/example_images.parquet
```

On a SLURM cluster, write `server.env` first (see
[docs/site-setup.md](docs/site-setup.md)). Then send the stage to a GPU job:

```bash
cp server.env.example server.env      # set the partition and the venv
dagspace vqa -m hydra/launcher=local pipeline=example_images runtime.stage_launcher=slurm_gpu_1x
```

**Note:** use `-m` in each `dagspace` command. The SLURM launchers work only in
Hydra multirun mode.

## Make your own task

```bash
python scripts/new_dagspace.py signs                 # dagspaces/signs/ and its tests
pytest tests/test_signs_dagspace.py                  # no GPU necessary
dagspace signs -m hydra/launcher=local pipeline=describe_images
```

Then edit the prompt in `dagspaces/signs/conf/prompt/describe.yaml`, and give
your own images:

```bash
python scripts/make_image_table.py /data/my_images my_images.parquet --prompt "What does the sign say?"
dagspace signs -m pipeline=describe_images data.parquet_path=$PWD/my_images.parquet data.image_root=null
```

See [docs/add-a-dagspace.md](docs/add-a-dagspace.md) and
[docs/add-a-stage.md](docs/add-a-stage.md).

## The dagspaces

| Dagspace | Task |
|----------|------|
| `vqa` | One question for each image (or image set). The answer can be JSON. |
| `ocr` | Text spotting with boxes. Large images are cut into tiles. |
| `pairwisevqa` | A comparison of 2 images on an ordinal scale |
| `embed` | Image and text embeddings |
| `speech` | Speech recognition over video clips |
| `_template` | The smallest dagspace: the source of `scripts/new_dagspace.py` |

`dagspace <name>` runs `python -m dagspaces.<name>.cli`. For a dagspace outside
`dagspaces/`, give the module path, for example `dagspace research.dagspaces.urbanroamvqa`.

## The layout

| Folder | Content |
|--------|---------|
| `dagspaces/` | The framework: the dagspaces and `common/` (the DAG engine, SLURM, vLLM inference, W&B) |
| `dagspaces/common/conf/` | The shared data, model and launcher configs |
| `examples/images/` | 10 public-domain photos and a table of them |
| `scripts/` | The venv build, the mirrors, the image table, the image pack, `new_dagspace.py` |
| `tests/` | The framework tests |
| `docs/` | The framework guides |
| `site/` | The launchers of one cluster (Cornell Tech). Delete or replace it on your cluster. |
| `sv/` | Project code: the Cyclomedia and NYC street-view data |
| `research/` | Project code: interpretability, notebooks, experiment scripts, papers |

The framework does not import `site/`, `sv/` or `research/`. A Hydra plugin adds
their config folders to the search path. Set `IMAGE_FORWARD_NO_PROJECT_ROOTS=1`
to use the framework configs only.

## Docs

| Page | Content |
|------|---------|
| [docs/architecture.md](docs/architecture.md) | The parts, a run, the pipeline format, the config search order |
| [docs/add-a-dagspace.md](docs/add-a-dagspace.md) | A new task from the template |
| [docs/add-a-stage.md](docs/add-a-stage.md) | A stage function and a pipeline node |
| [docs/site-setup.md](docs/site-setup.md) | `server.env` and the SLURM launchers |
| [docs/models-and-mirrors.md](docs/models-and-mirrors.md) | The model configs and the node-local mirrors |
| [docs/streaming-and-resume.md](docs/streaming-and-resume.md) | Large runs: batches, resume, image packs |

## Development

```bash
bash scripts/build_venv.sh --extra wandb --extra research   # more extras: see pyproject.toml
pytest -m "not gpu and not slurm and not data"              # the tests of the CI
ruff check .
```

The CI (`.github/workflows/ci.yml`) runs the tests, ruff and pyright on each
push.
