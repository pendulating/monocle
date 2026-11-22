# Architecture

image-forward runs vision-language models on large image tables. It uses vLLM
for inference, Hydra for configs and SLURM for the GPU jobs. W&B logging is
optional.

## The parts

| Part | Folder | Role |
|------|--------|------|
| Dagspace | `dagspaces/<name>/` | One kind of task: a CLI, an orchestrator, stages and configs |
| Shared code | `dagspaces/common/` | The DAG engine, the SLURM submission, vLLM inference, W&B, the factory |
| Shared configs | `dagspaces/common/conf/` | The data, model and launcher configs that each dagspace can use |
| Examples | `examples/images/` | 10 public-domain images and a parquet table of them |
| Generic scripts | `scripts/` | The venv build, the mirrors, the image table, the image pack, a new dagspace |
| Site configs | `site/conf/` | The launchers of one cluster. The framework does not need them. |
| Project code | `sv/`, `research/` | The code of this project. The framework does not import it. |

The framework dagspaces:

| Dagspace | Task |
|----------|------|
| `vqa` | One question for each image (or each image set). The answer can be JSON. |
| `ocr` | Text spotting with boxes. Large images are cut into tiles. |
| `pairwisevqa` | A comparison of 2 images on an ordinal scale |
| `embed` | Image and text embeddings |
| `speech` | Speech recognition over video clips |
| `_template` | The smallest dagspace. `scripts/new_dagspace.py` copies it. |

## A dagspace

Each dagspace has the same files:

| File | Role |
|------|------|
| `cli.py` | The Hydra entry point. It cleans the SLURM variables, then calls `run_experiment`. |
| `orchestrator.py` | It gives `get_stage_registry`, `execute_stage_job` and `run_experiment`. |
| `stages/` | The stage functions |
| `conf/config.yaml` | The top config: the defaults of each config group |
| `conf/pipeline/` | The pipelines (DAGs of stage nodes) |
| `conf/prompt/` | The prompts |

A new dagspace makes its orchestrator with `make_dagspace` (see
[add-a-dagspace.md](add-a-dagspace.md)). The older dagspaces write the same
3 functions by hand.

## A run

```
dagspace vqa -m pipeline=example_images
  → cli.py: Hydra composes the config (data, model, prompt, pipeline)
  → run_experiment: the pipeline graph, in topological order
  → for each node:
       launcher is null → the node runs in this process
       launcher is set  → submitit sends execute_stage_job to a SLURM job
  → the stage runner: input parquet → stage function → output parquet
  → <output_root>/pipeline_manifest.json: the outputs and the metadata of each node
```

The CLI process is the monitor. It waits for each stage job and gives the
outputs of a node to the nodes that depend on it. On a cluster, run the monitor
as a small CPU job (`hydra/launcher=slurm_monitor`) or on a login node
(`hydra/launcher=local`).

**Note:** use `-m` (multirun) for each command. The SLURM launchers work only in
multirun mode.

## A pipeline

A pipeline is a YAML file in `conf/pipeline/`. Each node names a stage, its
inputs and its outputs:

```yaml
# @package _global_
defaults:
  - override /data: example_images
  - _self_

pipeline:
  output_root: ${runtime.output_root}
  graph:
    nodes:
      describe:
        stage: describe                    # a key of get_stage_registry()
        depends_on: []
        outputs:
          results: outputs/describe/results.parquet
        launcher: ${runtime.stage_launcher}  # null: run in the monitor
      summarize:
        stage: summarize
        depends_on: [describe]
        inputs:
          dataset: describe.results          # <node>.<output>
        outputs:
          results: outputs/summarize/results.parquet
        launcher: slurm_gpu_1x
```

| Node key | Meaning |
|----------|---------|
| `stage` | The stage name in the registry of the dagspace |
| `depends_on` | The nodes that must finish first |
| `inputs` | The input name and the output of another node |
| `outputs` | The output name and the path below `output_root` |
| `overrides` | Config values for this node only, for example `model.batch_size: 8` |
| `launcher` | A launcher config name, or null |
| `max_attempts` | The number of tries for the node |

**Warning:** a pipeline file that overrides a config group (`/data`, `/model`,
`/prompt`) must start with `# @package _global_`. Put each `override` line
after the plain lines of the defaults list.

## The configs

Hydra composes the config from these groups:

| Group | Content | Where |
|-------|---------|-------|
| `data` | The parquet path, the image root, the column names | `dagspaces/common/conf/data/` |
| `model` | The model path, the vLLM engine settings, the batch size | `dagspaces/common/conf/model/` |
| `prompt` | The system prompt, the user template, the JSON schema | `dagspaces/<name>/conf/prompt/` |
| `pipeline` | The DAG | `dagspaces/<name>/conf/pipeline/` |
| `hydra/launcher` | The SLURM settings of a job | `dagspaces/common/conf/hydra/launcher/` |

Hydra reads the folders in this order. The first folder that holds a config
name wins:

1. `dagspaces/<name>/conf/`
2. The project roots: `site/conf/`, `sv/conf/`, `sv/nyc/conf/` and
   `research/conf/`. The plugin `hydra_plugins/mllmsci_searchpath/` adds them.
   In each root, the `<name>/` folder comes before the `common/` folder.
3. `dagspaces/common/conf/`

Set `IMAGE_FORWARD_NO_PROJECT_ROOTS=1` to remove the project roots. Then you get
the framework configs only. `tests/test_framework_configs.py` uses this switch.

## The VQA data flow

```
input parquet → DataFrame
  → preprocess: the prompt template, the image paths, the sampling params
  → run_vllm_inference: batches of llm.chat calls
  → postprocess: the JSON or the answer text
  → output parquet
```

A row gives its images as a local path, a `file://` path, a `data:` base64 URI
or a PIL image. A relative path joins `data.image_root`. Inference opens the
images batch by batch, thus the memory holds about one batch of images.

**Warning:** the stage does not download an `http(s)` URL. A row with an image
that does not load gets an empty answer and `__image_load_error__`, and the run
continues. Set `runtime.strict_images=true` to stop the run. See
[gotchas.md](gotchas.md).
