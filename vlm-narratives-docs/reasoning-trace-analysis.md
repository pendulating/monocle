# Reasoning-trace analysis

The consolidated pairvqa battery answers with a label only. A **thinking** sweep
also keeps the text that the model writes before it answers. This page says
where those traces are, and how to read them.

## Why a separate kind of run

| Run kind | Sampling | `max_tokens` | `model_reasoning` |
|----------|----------|--------------|-------------------|
| Canonical battery | greedy, `temperature=0.0` | 128 | empty |
| Thinking sweep | `temperature=1.0`, `top_p=0.95`, `top_k=64` | 8192 | full |

A canonical run holds no trace. The 128-token cap and the label grammar leave no
room for one. Thus trace work needs its own sweep.

**Warning:** a thinking run uses different sampling, so its labels are not
comparable with the battery. Do not pool the two. `_provenance.py` drops a
thinking run on purpose, and `notebooks/cvpr/_traces.py` collects it instead.

**Warning:** thinking mode must not be greedy. Greedy decoding repeats itself
over a long trace, which destroys the text that the run exists to collect.

## Where the traces are

| Sweep | Date | Cases | Models | Pairs |
|-------|------|-------|--------|-------|
| `multirun/think10k` | 2026-07-11 | subway safety, street photography | gemma-4-12b, qwen3.5-9b, gemma-4-e2b | 10,000 |
| `thinking_public_investment` | 2026-08-13 | subway safety, libraries, schools, road quality | gemma-4-12b | 10,000 |

The config is `dagspaces/urbanpairvqa/conf/sweep/thinking_public_investment_10k.yaml`.

## The notebook

There is 1 notebook for each prompt, `notebooks/cvpr/<case>_traces.py`, in the
same shape as the validation-by-proxy notebooks. `_trace_notebook.py` holds the
body they share. `notebooks/cvpr/README.md` gives the operating detail. Run one
from the canonical venv:

```bash
.venv-mllmsci-vllm025cu129/bin/marimo edit notebooks/cvpr/schools/schools_traces.py
```

**Only runs from 2026-08-11 forward enter a trace notebook.** That is the
consolidation date. Thus the `think10k` sweep does NOT appear, because it is
entirely pre-consolidation.

## Traps

**A frequency cloud cannot separate 2 prompts.** Every trace names the 2 images,
lists cues, then picks a label. That scaffold fills the cloud. Use the
`distinctive` mode, which scores a word by how much more 1 prompt uses it than
the others.

**A trace repeats the prompt.** The words `image`, `Image A`, and the 6 labels
appear in every trace of every case. `_traces.BOILERPLATE` removes them.

**2 prompt eras do not mix.** A run from before 2026-08-11 used a persona and a
cue list, and its traces repeat that text. The `think10k` sweep is in the older
era. A cloud that mixes the eras shows the prompt, not the model.

**gemma-4-e2b needs a post-hoc split.** Its trace and its answer arrive in 1
field. `scripts/split_e2b_thinking_reasoning.py` separates them and keeps the
original as `*.presplit.parquet`. Never read the `presplit` file.

**Not every sweep has a Hydra record.** The `think10k` stage directories hold no
`.hydra/overrides.yaml`, thus the W&B provenance chain cannot name their
pipeline. They also put 2 cases in 1 stage directory. `_traces.scan_sweep_dir`
reads such a sweep from disk and takes the case from each parquet file name.

**Never glob `multirun/`.** That tree sits on NFS and a full walk costs minutes.
Discovery goes through W&B, which names each stage directory.

## Warning: gemma-4-12b needs `interleaved_labels`

In thinking mode, gemma-4-12b cannot see the second image under the default
`images_then_text` layout. Measured on 16 pairs, 2026-08-13:

| Layout | "only one image" in the trace | NotSure |
|--------|-------------------------------|---------|
| `images_then_text` | 15 of 18 | 83% |
| `interleaved_labels` | 0 of 18 | 5.6% |

The prompt holds about 696 input tokens either way, which is 2 images' worth, so
the pixels are present. The model fails to bind them as 2 images. This is the
encoder-free `gemma4_unified` prompt-replacement path.

**A label-only run cannot show this.** Without a trace, the failure looks like an
honest high abstention rate. This is the strongest argument for the trace runs.
