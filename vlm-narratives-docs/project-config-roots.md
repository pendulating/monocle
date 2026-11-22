# Project config roots

The framework in `dagspaces/` holds only generic configs. The dataset configs
are in project folders outside the framework. A Hydra plugin adds these
folders to the search path of each dagspace.

## The roots

| Root | Content |
|------|---------|
| `sv/conf/` | The Cyclomedia configs |
| `sv/nyc/conf/` | The NYC configs (curated facilities, scaffolding permits, bayflood, nexar) |
| `research/conf/` | The research configs: the pairwisevqa `pipeline/`, `prompt/`, `sweep/` and `extract/` groups, and the GEPA configs (`vqa/prompt_opt/`) |

Each root has this layout:

| Folder | Search path of | Example |
|--------|----------------|---------|
| `<root>/<dagspace>/` | that dagspace only | `sv/nyc/conf/vqa/pipeline/vqa_bayflood.yaml` |
| `<root>/common/` | all dagspaces | `sv/conf/common/data/cyclomedia_manhattan.yaml` |

## How it works

`hydra_plugins/mllmsci_searchpath/mllmsci_searchpath.py` is a Hydra
`SearchPathPlugin`. It reads the dagspace name from the main config path
(`dagspaces/<name>/conf`). Then it appends these folders, in this order:

1. `<root>/<name>/` of each root
2. `<root>/common/` of each root

A config name can occur in more than one folder. Then the first folder in the
search path gives the config. The order is:

1. the dagspace `conf/`
2. the dagspace folders of the roots
3. the `common/` folders of the roots
4. `dagspaces/common/conf/` (Hydra adds the `hydra.searchpath` of `config.yaml` after the plugin folders)

Thus a dagspace-local config wins over a shared config, as in `dagspaces/`. For
example, `data=cyclomedia_manhattan` in the OCR dagspace gives
`sv/conf/ocr/data/cyclomedia_manhattan.yaml`.

**Note:** A project config also wins over a framework config in
`dagspaces/common/conf/` with the same name. Do not use a framework config
name for a project config.

## Rules

| Rule | Reason |
|------|--------|
| Run the commands from the repo root. | Hydra finds the plugin through the `hydra_plugins` namespace package. The repo root must be on `sys.path`. Phase 4 of `repo-cleanup-plan.md` adds the package to `pyproject.toml`. |
| Do not name a project config root in a framework config. | The framework must work without the project. |
| Give a project OCR handler as an import path. | `handler: sv.ocr.cyclomedia_handler:CyclomediaHandler`. The framework knows only `generic`. |
| Give a project stage as an import path. | See "Project stages" below. |

## Add a root

1. Make the folder, for example `research/conf/`, with `common/` and `<dagspace>/` subfolders.
2. Add the folder to `PROJECT_ROOTS` in the plugin.
3. Compose a config of each dagspace that uses it, with `--cfg job`.

## Project stages

A pipeline can use a stage that the dagspace does not register. Name the
runner in the `pipeline` block:

```yaml
pipeline:
  stage_runners:
    ic_extract: research.ic.runner:ICExtractRunner
  graph:
    nodes:
      ic_extract:
        stage: ic_extract
```

`stage_registry_for()` in `dagspaces/common/stage_submission.py` adds these
runners to the dagspace registry. The monitor and the stage job both use it.
The value is a `StageRunner` subclass or instance. A pipeline runner replaces a
dagspace runner with the same name. The stage name stays short, thus the W&B
metric names (`ic_extract/duration_s`) do not change.
