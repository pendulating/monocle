# lensread: J-lens readouts as a pipeline

The dagspace `research.dagspaces.lensread` reads the J-lens on a table of
(images, prompt) rows. It does the read of the F10 scripts
(`research/scripts/f10/hop_lean.py`) for any prompt, any image table and many
position kinds. The core is `research/monocle/readout.py`.

A read at layer l and position p is `unembed(lens.transport(h_l[p], l))`.
`h_l` is the output of block l. The model is a HF transformers model at fp32
(decision D154). The lens is the released J++ lens of the model.

## Run it

```bash
# The 10 example images of image-forward, the F10 question, qwen3.5-9b
dagspace research.dagspaces.lensread -m hydra/launcher=local pipeline=readout_example

# Your table, another model, the default positions
dagspace research.dagspaces.lensread -m hydra/launcher=local pipeline=readout \
    model=jpp_gemma-4-31b data.parquet_path=/path/rows.parquet data.image_root=null

# The F10 positions for another prompt, with the full store
dagspace research.dagspaces.lensread -m hydra/launcher=local pipeline=readout \
    readout=f10 readout.noun=street readout.adjective=nice readout.store=true ...
```

The stage runs in a SLURM job on the launcher of the model config
(`runtime.stage_launcher: ${model.launcher}`). Set
`runtime.stage_launcher=null` to run it in the CLI process on a GPU node.

## The input rows

| Column | Content |
|--------|---------|
| `sample_id` | The row id. The default is the row number. |
| `image_path` or `image_paths` | 1 image, or a list of images. `data.image_root` joins a relative path. |
| `prompt` | The user text. `readout.prompt` takes its place for every row. |

`readout.layout` puts the images before the text (`image_first`, the default)
or after it (`text_first`). The chat template is the template of the model,
with the generation prompt and no thinking. `readout.force_prefix` adds text
after the generation prompt.

## The segments of a sequence

| Segment | Positions |
|---------|-----------|
| `prefix` | before the first image token |
| `image` | the image tokens (all images) |
| `text` | after the first image token, not image and not `gen` |
| `gen` | the generation prompt and `force_prefix`, at the end |
| `all` | every position |

**Note:** with `layout: text_first`, the prompt words are in `prefix`.

## The position selectors

`readout.positions` is a list of selectors. The stage reads the union. Each
position in the output names all the selectors that chose it (column
`selectors`). Each selector has a `label` (a default label is made from its
kind). A selector that selects nothing stops the stage, unless it has
`optional: true`.

| Kind | Keys (default) | Selects |
|------|----------------|---------|
| `last` | | the last token: the answer position |
| `tail` | `n` (1), `segment` (`all`) | the last n positions of the segment |
| `head` | `n` (1), `segment` (`text`) | the first n positions of the segment |
| `match` | `text`, `which` (`last`), `occurrence` (`all`) | each occurrence of the token sequence of `text`: its last token, or (`which: all`) all its tokens. `occurrence: first` or `last` keeps 1 occurrence. |
| `index` | `at` | absolute indices; a negative index counts from the end |
| `segment` | `name` (`image`) | each position of the segment |
| `top_delta` | `k` (5), `metric` (`read_js`), `segment` (`all`), `layers` (all) | the k positions with the largest change between 2 consecutive read layers |

The 2 metrics of `top_delta`:

| Metric | Change between layers l and l' |
|--------|-------------------------------|
| `read_js` | the Jensen–Shannon divergence (nats) of the 2 lens distributions over the full vocabulary |
| `residual` | the relative L2 change of the residual, `‖h_l' − h_l‖ / ‖h_l‖` |

The score of a position is the largest change over the layer pairs. `layers:
[lo, hi]` keeps the read layers in that range. The `delta` column holds the score.

The 2 presets:

| Preset | Positions |
|--------|-----------|
| `readout=default` | `last`, `tail` 8, `head` 3 of `text`, `top_delta` 5 (`read_js`) |
| `readout=f10` | the 6 positions of the F10 stores (`after_image`, `first_text`, `place`, `after_place`, `after_cool`, `final`) for "Is this {noun} {adjective}? Answer with one word: yes or no." |

## The outputs

| File | Content |
|------|---------|
| `outputs/readout/results.parquet` | 1 row for each (sample, position, layer, rank): `sample_id, position, segment, token_at, selectors, delta, layer, rank, token, token_id, logit, logprob` |
| `outputs/readout/answers.parquet` | 1 row for each sample: `top` (the top token of the model output), `tokens`, `n_patches`, `positions`, `yes_no`, `margin`, `lp_<form>` |
| `outputs/readout/manifest.json` | the model, the lens, the layers, the selectors, the vocabulary |
| `outputs/readout/store/` (`readout.store=true`) | `reads-<chunk>.npy` [positions, layers, words] fp16, `residuals-<chunk>.npy` [positions, layers, d] fp16, `index-<chunk>.parquet` (the sample and the position of each row), `vocab.parquet`, `manifest.json` |

- `logit` is the lens logit. `logprob` is its log-softmax over the full vocabulary.
- `readout.vocab: words` (the default) keeps the tokens ` [a-z]{3,}`, the word list of the F10 stores. `all` keeps the full vocabulary.
- `margin` is log p(any yes form) − log p(any no form) of the model output, over `readout.answer_forms` (the 8 F10 forms). `yes_no` uses `Yes` and `No` only.
- The read layers are `range(0, model.n_layers, readout.layer_step)` (step 4), or the list `readout.layers`.

**Warning:** the store holds about 6 MB for each position with the `words`
vocabulary and 8 layers. 10,000 images with 15 positions is about 1 TB. Keep
`readout.store=false` unless you need the mean readouts or the contrasts.

The stage works in chunks of `readout.chunk_rows` rows (50). The answers file
of a chunk is its last write. A rerun with the same output folder skips the
done chunks. A chunk file of other rows stops the stage.

## Models and GPUs

| `model=` | Blocks | GPUs (fp32) | Launcher |
|----------|--------|-------------|----------|
| `jpp_qwen3.5-9b` (default) | 31 | 1 × 96 GB | `slurm_gpu_preempt_big` |
| `jpp_gemma-4-31b` | 59 | 2 × 96 GB (`device_map: auto`) | `slurm_gpu_preempt_2x_big` |
| `jpp_qwen3.6-27b` | 63 | 2 × 96 GB (`device_map: auto`) | `slurm_gpu_preempt_2x_big` |

The lenses are in `outputs/_monocle/jlens/jpp/<model>/lens.pt` (from HF
`koayon/jpp-lenses`). The weights come from `$IMAGE_FORWARD_MODEL_ZOO`, or from
the node-local mirror when the node has one.

**Note:** the F10 runs used 1 B300 on the `g3` cluster. submitit tracks a job
with `sacct` and no `-M` option, thus a launcher for `g3` needs a change in
image-forward first.

## Agreement with F10

The same model, lens, prompt and images give the reads of the F10.30 store
(`data/f10/hop_more/lean/<model>`). The check on 2026-11-22 used the
`readout=f10` preset on the first 3 images of `images_k10.parquet`:

| Model | GPU | Reads at `place`, `after_place`, `final` | Answer margin |
|-------|-----|------------------------------------------|---------------|
| qwen3.5-9b | 1 × RTX PRO 6000 | max difference 0.4% of the largest logit; the same top-1 word at all 9 points; the same word list (43,563 tokens) | difference 0.02 or less |
| gemma-4-31b (2 images) | 2 × RTX PRO 6000, `device_map: auto` | max difference 0.05% of the largest logit; the same top-1 word at all 6 points; the same word list (53,999 tokens) | difference 0.0004 or less |

The F10.30 store is fp16 and came from a B300, thus a small difference is
correct. `tests/test_lensread.py::test_f10_parity_qwen35_9b` (markers `gpu`,
`data`) does this check again on a GPU node.
