# Add a stage

A stage reads a table, does one step and writes a table. Most stages are a
function `(df, cfg) -> DataFrame`.

## 1. Write the stage function

Put it in `stages/` of the dagspace. This stage asks the model to count the
people in each image and reads a JSON answer:

```python
# dagspaces/signs/stages/count.py
from dagspaces.common.stage_utils import extract_last_json

from .describe import make_preprocess


def postprocess(row):
    out = {k: v for k, v in row.items() if k in ("sample_id", "image_path")}
    parsed = extract_last_json(row.get("generated_text") or "") or {}
    out["people"] = parsed.get("people")
    return out


def run_count_stage(df, cfg):
    from dagspaces.common.vllm_inference import run_vllm_inference

    return run_vllm_inference(
        df, cfg, preprocess=make_preprocess(cfg), postprocess=postprocess, stage_name="count"
    )
```

`run_vllm_inference` calls `preprocess` on each row. The row must then hold:

| Key | Content |
|-----|---------|
| `messages` | The chat messages. An image is `{"type": "image", "image": "/path.jpg"}`. |
| `sampling_params` | A dict of vLLM sampling params, for example `max_tokens` |

Give an image as a path, not as pixels. Then the stage opens the images batch
by batch, and the memory holds one batch only.

After the model call, each row has `generated_text`. `postprocess` makes the
output row from it.

A stage with no model is a plain function, for example a filter:

```python
def run_filter_stage(df, cfg):
    return df[df["people"].fillna(0) > 0]
```

## 2. Register the stage

Add a runner to `stages=` in `orchestrator.py`:

```python
from .stages.count import run_count_stage

_dagspace = make_dagspace(
    __name__,
    name="signs",
    stages={
        "describe": DataFrameStageRunner("describe", run_describe_stage),
        "count": DataFrameStageRunner("count", run_count_stage),
    },
)
```

`DataFrameStageRunner` does the work around the function:

1. It reads the `dataset` input of the node, or else `data.parquet_path`.
2. It renames the columns with `data.columns` and takes `runtime.sample_n` rows.
3. It joins `data.image_root` to each relative `image_path`.
4. It calls the function and writes the result to the `results` output.
5. It logs a sample of the rows to W&B, when W&B is on.

Give `input_key=` or `output_key=` to use different names.

## 3. Add the stage to a pipeline

```yaml
# dagspaces/signs/conf/pipeline/count_people.yaml
# @package _global_
defaults:
  - override /prompt: count
  - _self_

pipeline:
  output_root: ${runtime.output_root}
  graph:
    nodes:
      count:
        stage: count
        depends_on: []
        outputs:
          results: outputs/count/results.parquet
        launcher: ${runtime.stage_launcher}
```

And the prompt:

```yaml
# dagspaces/signs/conf/prompt/count.yaml
system: null
user_template: |
  How many people are in this image? Return JSON: {"people": <number>}.
```

To feed the output of one node to the next node, write
`inputs: {dataset: count.results}` and `depends_on: [count]`.

## 4. Test it

Copy a test from `tests/test_<name>_dagspace.py`. Replace
`run_vllm_inference` with a fake function, as `test_the_runner_reads_the_example_table_and_writes_the_results`
does. Then the test needs no GPU.

## A stage with its own runner

When a stage does not map one table to one table, write a `StageRunner`
subclass. Its `run(context)` method gets:

| Field | Content |
|-------|---------|
| `context.cfg` | The config of the node (the node overrides are applied) |
| `context.node` | The node spec: `key`, `stage`, `inputs`, `outputs` |
| `context.inputs` | The input name and its file path |
| `context.output_paths` | The output name and its file path |
| `context.output_dir` | The folder of the stage |
| `context.logger` | The W&B logger, or None |

The method returns `StageResult(outputs={name: path}, metadata={...})`. See
`dagspaces/common/runners/dataframe.py` for an example.
