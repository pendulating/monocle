# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language: ASD-STE100

Write all text in ASD-STE100 Simplified Technical English (STE).

This rule applies to:

- Replies to the user
- Code comments and docstrings
- Comments in config files
- Documentation in `research/docs/` and `.dev_history/`, and commit messages

### Writing rules

| Rule | Requirement |
|------|-------------|
| Words | Use only approved words. Technical names and technical verbs are permitted. |
| One meaning | Give each word one meaning and one part of speech. |
| Sentence length | Instructions: 20 words maximum. Descriptions: 25 words maximum. |
| Paragraphs | Write one topic in one paragraph. Use 6 sentences maximum. |
| Voice | Use the active voice. |
| Tense | Use the simple tenses. Do not use the perfect or the progressive tenses. |
| Verbs that end in -ing | Do not use them, unless they are a technical name. Write a clause. |
| Instructions | Write one instruction in one sentence. |
| Articles | Write `the` or `a` where you can. |
| Noun clusters | Use 3 words maximum. |
| Lists | Put complex data in a vertical list. |
| Warnings | Write the warning before the step, not after it. |

### Words to replace

| Do not write | Write |
|--------------|-------|
| utilize | use |
| perform | do |
| prior to | before |
| in order to | to |
| due to | because of |
| ensure | make sure |
| via | with, by |
| obtain | get |
| attempt | try |
| terminate | stop |
| initiate | start |
| additional | more |
| approximately | about |
| however | but |
| therefore | thus |

### Examples

| Do not write | Write |
|--------------|-------|
| Removing the persona changed the judgments. | The judgments changed when we removed the persona. |
| The cache is being shared by concurrent jobs. | Concurrent jobs share the cache. |
| It's uninterpretable without an anchor. | You cannot read this value without an anchor. |
| This has been fixed. | We fixed this. |

**Note:** The STE dictionary has about 900 approved words. Follow the rules and
the style above. If a word is necessary for accuracy and no approved word is
equivalent, use it as a technical name.

## Documentation

The documentation of this repo has 2 folders:

| Folder | Content |
|--------|---------|
| `.dev_history/` | Gitignored. The development history of the code: plans, upgrade logs, incident reports, reviews, proposals |
| `research/docs/` | The research notes: experiments, results, paper plans, runbooks |

| Note | Folder |
|------|--------|
| How the framework works now | The `docs/` folder of image-forward, not this repo |
| A research experiment, a result, a runbook | `research/docs/` |
| How the code changed: a plan, an upgrade log, an incident, a fix, a review, a proposal | `.dev_history/` (not in git) |

A page in `research/docs/` must not link to `.dev_history/`. A clone
does not have that folder. Put the current facts in the guide, and keep the
story of the change in `.dev_history/`.

The old Obsidian wiki (`vault/context/`), the UAIR docs archive and the old
one-off scripts left git on 2026-11-22. The tag `archive/pre-cleanup-2026-11-21`
holds them. Do not add them again.

**When to write documentation:**

- After a structural change to the codebase. This includes a new dagspace, a new
  stage, a new config group, or a module that you rename.
- After you add, remove, or change a pipeline stage or a shared utility.
- After you correct a bug or a performance problem.
- When the user asks you to document something.

**Do not** write documentation for a small change. A typo, a comment, or a change
to a test does not need documentation. Document the architecture, not each commit.

**Conventions:** Use `kebab-case.md` for file names. Prefer tables and lists.
Write the text in STE (see the section above).

## Commit messages

Do not add AI-tool lines (`Co-Authored-By: Claude ...`, `Claude-Session: ...`)
to commit messages.

## Project Overview

monocle is the research repo of the study of how vision-language models judge
urban images (MLLMSCI / UAIR). It holds the experiments, the street-view data
code, the J-lens and the paper material.

The pipelines run on **image-forward** (`pendulating/image-forward`), a separate
repo: the dagspaces, the launchers, vLLM inference and the scripts. Its import
path is `dagspaces`. On this cluster the checkout is `/share/ju/matt/image-forward`,
and `IMAGE_FORWARD_HOME` in `server.env` names it. Read its `CLAUDE.md` and
`docs/` for the framework. Change framework code there, not here.

**Warning:** do not put generic framework code in this repo. Do not put project
code, datasets or research configs in image-forward.

## Common Commands

```bash
# Run a pipeline. Run the commands in this repo: image-forward finds the
# project config roots from the current folder.
dagspace vqa -m pipeline=vqa_cyclomedia_scaffolding runtime.debug=true runtime.sample_n=100
dagspace vqa -m hydra/launcher=slurm_gpu_4x pipeline=vqa_cyclomedia_scaffolding
dagspace pairwisevqa -m pipeline=extract_traces runtime.sample_n=10
dagspace research.dagspaces.urbanroamvqa -m pipeline=...

# Tests (CI runs the selection with no GPU, SLURM or local data)
pytest -m "not gpu and not slurm and not data"
ruff check .
```

**IMPORTANT:** Always use `-m` (multirun/submitit) when giving CLI commands to run a dagspace pipeline. Pipelines are designed to be orchestrated via SLURM — the monitor node submits stage jobs.

## How monocle uses image-forward

| Item | Where |
|------|-------|
| The framework version | `[tool.uv.sources] image-forward` in `pyproject.toml` (a git rev). Keep the checkout at the same commit. |
| The config roots | `[tool.image-forward] conf-roots` in `pyproject.toml`: `site/conf`, `sv/conf`, `sv/nyc/conf`, `research/conf` |
| The site settings | `server.env` in this repo (gitignored), with `IMAGE_FORWARD_HOME` |
| The cluster launchers | `site/conf/common/hydra/launcher/`. They take the place of the generic launchers with the same name. See `site/README.md`. |
| The framework scripts | `scripts/*.sh` and `scripts/*.py` are shims to `$IMAGE_FORWARD_HOME/scripts/` |
| The research dagspaces | `research/dagspaces/`: `urbanroamvqa`, `urbanvit`, `opf_vision_labels`, `artifact_gen`, `lensread` (J-lens readouts, see `research/docs/lensread.md`) |
| A project stage | `pipeline.stage_runners` in a pipeline (for example `research/ic/` adds `ic_extract`) |

The dataset configs (Cyclomedia, NYC, bayflood, nexar) are in `sv/conf/` and
`sv/nyc/conf/`. Thus `pipeline=vqa_bayflood` and `data=cyclomedia_manhattan`
work in each dagspace.

### Environments and node-local mirrors

On this site, `.venv-vllm025cu129` (a symlink to `.venv-mllmsci-vllm025cu129`)
is the default venv for all launchers. `server.env` sets it. It holds editable
installs of the image-forward checkout and of this repo. `.venv-3.12` (vLLM
0.19) and `.venv-nightly` (vLLM 0.23) stay on disk; override
`IMAGE_FORWARD_VENV_ACTIVATE` to use one.

**Warning:** do not change the site venv while jobs run. Check `squeue -u $USER` first.

A stage job starts from a node-local `/scratch` mirror of the venv when the
mirror stamp agrees with the venv. After a change to the venv, run
`bash scripts/sync_venv_to_scratch.sh` and then `--make-tarball`. See
`docs/models-and-mirrors.md` in image-forward. The history of the mirrors and of
the vLLM 0.25 upgrade is in `.dev_history/`.

## File Locations

- `<output_root>/pipeline_manifest.json` — The orchestrator writes it in each run. It holds the stage results and the metadata. Git ignores it.
- `site/` — The launchers of this cluster and `site/speech-guide.md`. The framework does not need them.
- `sv/` — The street-view code. The framework does not import it.
  - `sv/cyclomedia/`, `sv/cyclomedia_catalog/` — The NYC layer of the Cyclomedia source. The code (readers, catalog builder, `cyclomedia` CLI) is in image-forward: `dagspaces.sources.cyclomedia`. `sv/cyclomedia_catalog/nyc.py` holds the NYC settings (folders, boroughs, box, `America/New_York`). The other modules are shims at the old paths.
  - `sv/ocr/` — A shim. The OCR handler is `handler: dagspaces.sources.cyclomedia.ocr:CyclomediaHandler`.
  - `sv/nyc/curation/` — The NYC Socrata curation (`python -m sv.nyc.curation`).
  - `sv/conf/`, `sv/nyc/conf/` — The dataset configs, in `common/` (all dagspaces) and `<dagspace>/` folders.
  - `sv/scripts/`, `sv/nyc/scripts/`, `sv/tests/`, `sv/nyc/tests/` — The data scripts and their tests.
  - `sv/docs/` — The design notes of the catalog and the NYC curation.
- `research/` — The research code. The framework does not import it. The editable install maps `research` and `sv`.
  - `research/monocle/` — The interpretability code (the J-lens). Import it as `research.monocle`. `readout.py` is the core of the `lensread` dagspace.
  - `research/notebooks/` — The analysis notebooks (cvpr, facct).
  - `research/scripts/` — The experiment scripts (f4 to f10, batteries, exports).
  - `research/traces/` — The shard submitter of the trace extraction (`python -m research.traces.submit_trace_shards`).
  - `research/figures/`, `research/tables_export/` — The outputs of the notebooks and the scripts.
  - `research/papers/`, `research/notes/` — The paper drafts and the method notes.
  - `research/docs/` — The research documentation: experiments, results, runbooks (for example `facct-runbook.md`).
  - `research/conf/` — The research configs: the pairwisevqa batteries (`pipeline/`, `prompt/`, `sweep/`, `extract/`) and the GEPA configs. The plugin adds them to the search path.
  - `research/ic/` — The Integrative Complexity schema, codes, examples and the `ic_extract` stage. A pipeline adds the stage with `pipeline.stage_runners`.
  - `research/prompt_opt/` — The GEPA prompt optimization (single-image and pairwise). The `gepa` package is not in the venv.
- `scripts/` — Shims to the scripts of image-forward (`_image_forward_home.sh` finds the checkout), and `build_browser_index.py`.
- `conftest.py` — Loads `server.env`, sets the markers `gpu`, `slurm` and `data`, and gives the path fixtures. CI (`.github/workflows/ci.yml`) runs `pytest -m "not gpu and not slurm and not data"` and ruff.
- `tests/test_roaming_vqa.py` — Roaming VQA tests (graph construction, bearings, checkpoints)
- `.dev_history/` — Gitignored. The development history: the cleanup plan, upgrade logs, incident reports, reviews and proposals.
- `artificial-aesthetic/` — The FAccT paper repo. The LaTeX is the submodule `paper/`. `paperkit` copies the figures, tables and numbers into it, and checks them. See `research/docs/paper-repo-sync.md`.
- `archive/` — Gitignored. Old research scripts that left git on 2026-11-22, at their old paths. Read them for reference. Do not import them.
