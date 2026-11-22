# Large runs: batches, resume and image packs

These settings matter when a stage reads more than about 100,000 rows.

## Batches

`run_vllm_inference` sends the rows to vLLM in batches of `model.batch_size`
rows. Each batch is one `llm.chat` call. Before the call of batch k, a thread
opens the images of batch k+1. After the call, the stage drops the images.
Thus the memory holds about 2 batches of images.

Give each image as a path in the chat message:

```python
{"type": "image", "image": "/data/images/0001.jpg"}
```

**Warning:** do not open the images in `preprocess`. The stage calls
`preprocess` on each row at the start. With PIL images in the rows, the memory
holds all the images of the table.

## Resume after a preemption

Set `model.resume_dir` to a folder. Then the stage writes one chunk parquet for
each `model.resume_chunk_rows` rows (2000 by default). A job that starts again
reads the chunks and runs only the rows that remain.

```bash
dagspace vqa -m pipeline=my_pipeline model.resume_dir=/data/runs/my_run/resume
```

| Property | Effect |
|----------|--------|
| A chunk holds the raw text and the token counts | A change to `postprocess` does not make a chunk stale |
| A guard file holds a fingerprint of the prompts | A change to the prompts, the seed or the rows stops the stage. It does not mix 2 runs. |
| A chunk gets a temporary name, then an atomic rename | A preemption during a write leaves a whole file or no file |
| A short or bad chunk is dropped | Its rows run again |
| A chunk ends on a batch boundary | A resumed job sends the same batches to vLLM |

**Warning:** do not put `${now:...}` in the resume path. A requeued job makes a
new time, and then it does not find its chunks.

The `pairwisevqa` dagspace sets `model.resume_dir` for each shard when
`runtime.resume=true`.

## Where the output goes

Each stage job writes below `runtime.output_root`. Without a value, Hydra gives
`multirun/<date>_<EXPERIMENT>/<time>/<job number>/`.

**Warning:** 2 sweeps that start in the same second get the same folders, and
one writes over the other. Give each sweep its own `runtime.output_root`, for
example `runtime.output_root=/data/runs/sweep_a/${hydra:job.num}`.

## Shards

To split a large table over many GPU jobs, use a Hydra sweep over a shard
index. The `pairwisevqa` dagspace has `submit_shards.py` for this, and
`scripts/merge_pairwise_shards.py` joins the shard outputs:

```bash
python scripts/merge_pairwise_shards.py --help
```

## Image packs

A table of a million images on NFS opens a million files. Each open costs
several NFS round trips, and many shards at the same time can overload the
server. An image pack puts the JPEG bytes into large blob files. A read is then
one large read in place of many small ones.

Build the pack one time:

```bash
python scripts/build_image_pack.py --help
```

| File | Content |
|------|---------|
| `blob-NNNN.bin` | The image bytes, one after the other (16,384 images in each blob) |
| `index.parquet` | The path, the blob, the offset and the length of each image |
| `missing.txt` | The source paths that the build could not read |
| `manifest.json` | The marker of a complete pack. The build writes it last. |

Give the pack to a run:

```bash
dagspace vqa -m pipeline=my_pipeline runtime.image_pack=/data/image_pack
```

The stage reads the bytes of each batch from the pack. A path that is not in the
pack reads the file as before. The bytes are the same, thus the results do not
change. The stage log prints `image pack ON` at the start and the counts of
hits and misses at the end. A miss count above 0 tells you that the pack does
not cover the table.

**Warning:** the reader refuses a pack with no `manifest.json`. Let the build
finish before a run uses the pack.
