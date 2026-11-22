# image-forward documentation

These pages tell you how the framework works and how to extend it. Read the
repo `README.md` first for the install and the quickstart.

| Page | Content |
|------|---------|
| [architecture.md](architecture.md) | The parts of the framework and how a run goes through them |
| [add-a-dagspace.md](add-a-dagspace.md) | Make a new dagspace from the template |
| [add-a-stage.md](add-a-stage.md) | Write a stage function and add it to a pipeline |
| [site-setup.md](site-setup.md) | `server.env`, the SLURM launchers and the site configs |
| [models-and-mirrors.md](models-and-mirrors.md) | The model configs, the model zoo and the node-local mirrors |
| [streaming-and-resume.md](streaming-and-resume.md) | Large runs: chunked batches, resume after a preemption, image packs |
| [gotchas.md](gotchas.md) | Problems that give a wrong or slow result with no clear error. Read it before a large run. |

More pages about this repo:

| Page | Content |
|------|---------|
| [project-config-roots.md](project-config-roots.md) | The project config folders (`site/`, `sv/`, `research/`) and the search-path plugin |
| [gpu-census-tool.md](gpu-census-tool.md) | `scripts/gpu_census.py`: the GPU supply of a SLURM partition |
| [gemma-4-best-practices.md](gemma-4-best-practices.md) | How the pairwise and lens paths follow the Gemma 4 model card |

The research notes are in `research/docs/`.
