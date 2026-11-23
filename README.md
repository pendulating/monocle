# monocle

monocle is the research repo of the study of how vision-language models judge
urban images. It holds the experiments, the street-view data code, the
interpretability code (the J-lens) and the paper material.

The pipelines run on [image-forward](https://github.com/pendulating/image-forward),
a separate repo. image-forward holds the generic framework: the dagspaces, the
SLURM launchers, the vLLM inference and the scripts. monocle installs it and
adds its own configs to the Hydra search path.

## The layout

| Folder | Content |
|--------|---------|
| `research/` | The research code: the J-lens (`research/monocle/`), the notebooks, the experiment scripts, the research dagspaces, the figures and the papers |
| `research/conf/` | The research configs: the pairwisevqa batteries, the prompts, the sweeps, the extraction schemas |
| `research/docs/` | The research notes: experiments, results, runbooks |
| `sv/` | The street-view data code: Cyclomedia, the catalog and the NYC curation, with their configs |
| `site/` | The launchers of the Cornell Tech cluster (`pierson`, `ju`, `klara`) |
| `scripts/` | Shims to the scripts of image-forward. The browser index is the `build_browser_index` stage of the image-forward `embed` dagspace (`pipeline=browser_index_cyclomedia`). |
| `tests/` | The tests of the research and street-view code |

## The 2 checkouts

Put the 2 checkouts side by side, and install both in one venv:

```
work/
  image-forward/    the framework (pendulating/image-forward)
  monocle/          this repo
```

```bash
git clone git@github.com:pendulating/image-forward.git
git clone git@github.com:pendulating/monocle.git
cd monocle
uv sync --extra urban --extra research --extra notebooks   # image-forward from git, at the pinned commit
uv pip install -e ../image-forward                         # the checkout, for the SLURM launchers
```

**Warning:** the SLURM launchers source the scripts of the image-forward
checkout. A git install of image-forward has no `scripts/` folder. Thus a
cluster venv needs the editable checkout. Keep the checkout at the commit that
`pyproject.toml` pins (`[tool.uv.sources]`).

Then write `server.env` in this repo. Copy `server.env.example` from
image-forward, and add the key of the checkout:

```bash
IMAGE_FORWARD_HOME=/path/to/image-forward
```

## Run a pipeline

Run the commands in this repo. image-forward finds the project from the
current folder: the `[tool.image-forward]` table of `pyproject.toml` names the
config roots (`site/conf`, `sv/conf`, `sv/nyc/conf`, `research/conf`).

```bash
dagspace pairwisevqa -m pipeline=extract_traces runtime.sample_n=10
dagspace vqa -m pipeline=vqa_bayflood
dagspace research.dagspaces.urbanroamvqa -m pipeline=...
```

**Note:** use `-m` in each `dagspace` command. The SLURM launchers work only in
Hydra multirun mode.

The job files in `research/scripts/` call `scripts/activate_stage_venv.sh` and
the other framework scripts by their old paths. The files in `scripts/` are
shims: they find the image-forward checkout (`IMAGE_FORWARD_HOME`) and run the
script there.

## Development

```bash
pytest -m "not gpu and not slurm and not data"   # the tests of the CI
ruff check .
```

To take a new image-forward commit, change the `rev` of `image-forward` in
`[tool.uv.sources]`, run `uv lock`, and move the checkout to the same commit.

## Docs

| Page | Content |
|------|---------|
| [image-forward docs](https://github.com/pendulating/image-forward/tree/main/docs) | The framework guides: architecture, stages, launchers, mirrors, project repos |
| [research/docs/](research/docs/) | The research notes |
| [site/README.md](site/README.md) | The launchers of this cluster |
