# Add a dagspace

A dagspace is one kind of task: for example, describe images, count objects or
read signs. Make a new one from the template. It works on the 10 example
images before you change a line.

## 1. Copy the template

```bash
python scripts/new_dagspace.py signs
```

The script writes these files:

| File | Content |
|------|---------|
| `dagspaces/signs/cli.py` | The CLI |
| `dagspaces/signs/orchestrator.py` | The stage registry, with one stage `describe` |
| `dagspaces/signs/stages/describe.py` | The stage function |
| `dagspaces/signs/conf/config.yaml` | The defaults: example data, a Qwen model, the `describe` prompt |
| `dagspaces/signs/conf/pipeline/describe_images.yaml` | A pipeline with one node |
| `dagspaces/signs/conf/prompt/describe.yaml` | The prompt |
| `tests/test_signs_dagspace.py` | 5 tests. They need no GPU. |

The name must be a lowercase Python name. To put the dagspace in a different
folder, give `--dest`, for example `--dest research/dagspaces`.

## 2. Run the tests

```bash
pytest tests/test_signs_dagspace.py -v
```

A fake model answers in these tests. They check the pipeline, the prompt, the
output table and the pickle of the job function.

## 3. Run the example

On a node with a GPU, run the stage in the CLI process:

```bash
dagspace signs -m hydra/launcher=local pipeline=describe_images
```

On a SLURM cluster, send the stage to a GPU job. Set up `server.env` first (see
[site-setup.md](site-setup.md)):

```bash
dagspace signs -m hydra/launcher=local pipeline=describe_images \
    runtime.stage_launcher=slurm_gpu_1x
```

The output is in `multirun/<date>_SIGNS/<time>/0/outputs/describe/results.parquet`.
It has one row for each image, with the `answer` column.

To use a different model, give a model config or a path:

```bash
dagspace signs -m pipeline=describe_images model=vllm_multimodal_qwen3_vl_2b
dagspace signs -m pipeline=describe_images model.model_source=/path/to/model
```

## 4. Use your own images

Make a table of a folder of images:

```bash
python scripts/make_image_table.py /data/my_images my_images.parquet \
    --prompt "Is there a sign in this image? Copy its text."
```

Then run with it:

```bash
dagspace signs -m pipeline=describe_images \
    data.parquet_path=$PWD/my_images.parquet data.image_root=null
```

To keep the settings, write a data config `dagspaces/signs/conf/data/my_images.yaml`:

```yaml
parquet_path: /data/my_images.parquet
image_root: null          # the table holds absolute paths
columns:
  sample_id: sample_id
  image_path: image_path
  prompt: prompt
```

With `--relative`, the table holds paths below the folder. Then set
`image_root` to the folder.

## 5. Change the task

| To change | Edit |
|-----------|------|
| The question | `conf/prompt/describe.yaml`, or the `prompt` column of the table |
| The answer format | `postprocess` in `stages/describe.py` |
| The token limit, the temperature | `sampling_params` in `conf/config.yaml` |
| The batch size, the GPU memory | `model.batch_size`, `model.engine_kwargs.*` |
| More stages | See [add-a-stage.md](add-a-stage.md) |

## The orchestrator

`make_dagspace` makes the 3 functions that the framework needs:

```python
from dagspaces.common.dagspace_factory import make_dagspace
from dagspaces.common.runners.dataframe import DataFrameStageRunner

from .stages.describe import run_describe_stage

_dagspace = make_dagspace(
    __name__,
    name="signs",
    stages={"describe": DataFrameStageRunner("describe", run_describe_stage)},
)

get_stage_registry = _dagspace.get_stage_registry
execute_stage_job = _dagspace.execute_stage_job
run_experiment = _dagspace.run_experiment
```

**Warning:** keep the 3 names at the bottom as they are. Submitit sends
`execute_stage_job` to each SLURM job by its module and name. `run_experiment`
stops with an error when the names do not agree.

Give a keyword to `make_dagspace` to change a setting of `DagspaceSettings`, for
example `recovery_mode="nfs_poll"`. An unknown keyword gives a `TypeError`.
