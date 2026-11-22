# NFS load from the pairvqa sweeps on /share/ju

A large `pairwisevqa` sweep on the `gpu` partition makes `/share/ju` slow for
everyone. This page gives the cause, the measurements, and the fix options.
Measured 2026-09-03 on the noabstain gemma battery (array 217915, 322 shards).

## Cause

Each pairvqa row opens 2 JPEG files under `/share/ju/cyclomedia/raw`. The
stage reads each file with `PIL.Image.open` on the main thread, one file after
the other, before each batch of 256 rows. The tree is deep, and the 1.1 million
unique images of the battery sit in 980,000 directories. Thus one cold image
costs about 8 NFS round trips, not 1.

| NFS operation | Round trips for each cold image |
|---------------|--------------------------------|
| LOOKUP        | 2.4                            |
| ACCESS        | 2.5                            |
| OPEN          | 1.0                            |
| READ (236 kB) | 1.0                            |
| CLOSE         | 1.0                            |
| GETATTR       | 0.4                            |
| Total         | about 8.4                      |

The battery reads 15.4 million images, but only 1.1 million are unique. A
shard reads each of its images about 1.5 times, thus the page cache of a node
removes little. The repeats sit across shards and across cases.

| Case              | Rows      | Unique images |
|-------------------|-----------|---------------|
| libraries         | 1,100,000 | 11,561        |
| subway_safety     | 1,100,000 | 45,889        |
| parks_plazas      | 1,100,000 | 94,375        |
| schools           | 1,100,000 | 112,303       |
| restaurants       | 1,100,000 | 395,665       |
| road_quality      | 1,100,000 | 490,802       |
| street_photography| 1,100,000 | 490,802       |
| All 7 cases       | 7,700,000 | 1,100,682     |

## The server collapses above about 30 shards

The chunk timestamps under `resume/` give the row rate of each shard. The
`sacct` start and end times give the number of shards that ran at the same
time.

| Hour (2026-09-03) | Shards at once | Rows/s for each shard (median) | Rows/s in total |
|-------------------|----------------|--------------------------------|-----------------|
| 00h to 07h        | 15 to 17       | 12 to 14                       | about 210       |
| 08h               | 30             | 4.3                            | about 130       |
| 09h               | 57             | 1.6                            | about 90        |
| 10h to 11h        | 78 to 80       | 1.2                            | about 95        |

The total rate FALLS when more shards run. At 80 shards the sweep does less
work than at 17, and it holds 4 times the GPUs. Under this load each NFS round
trip to `ju-nfs-02` waits 20 to 50 ms, and one cold image open takes 240 ms
(median) on the login node. The image loads alone then cap a shard at about
2 rows/s, which agrees with the measured 1.2 rows/s. A 24,000-row shard at
1.2 rows/s needs 5.5 hours, thus it ends in TIMEOUT inside the 180-minute
walltime and must run again with `--only-missing`.

## Fix options

### 1. Now: lower the array throttle

The user can hold the running sweep at about 20 shards:

```bash
scontrol update JobId=217915 ArrayTaskThrottle=20
```

This changes nothing in the running jobs. New tasks start only when fewer
than 20 run. The measured total rate at 17 shards is about 2 times the rate
at 80, and `/share/ju` recovers. The cost is that the sweep holds fewer GPUs.
Use `--parallel 20` in `submit_shards.py` for the next sweep.

### 2. Done 2026-09-03: the image pack on /share/pierson

2026-10-07: `/share/pierson` is being retired. The project, and the pack under
`multirun/pair_tables_1m/image_pack`, now live on `/share/ju`, the same server as
the raw images. A pack read is still 1 READ of about 240 kB in place of 8.4 round
trips, but the load now lands on ju-nfs-02. Watch per-shard rows/s on the first
sweep after the move.

The pack is built one time, before a sweep:

```bash
sbatch scripts/build_image_pack.sub      # wraps scripts/build_image_pack.py
```

`scripts/build_image_pack.py` reads the unique image paths of the pair
tables and copies the raw JPEG bytes into blob files of 16,384 images
(about 4 GB each) under `multirun/pair_tables_1m/image_pack/`. The size is
about 275 GB for 1.1 million images. The script reads `/share/ju` with 8
threads, and a run that stops keeps its finished blobs. It writes
`manifest.json` last, and the reader refuses a pack without it.

| File | Content |
|------|---------|
| `blob-NNNN.bin` | the image bytes, one after the other |
| `index.parquet` | path, blob, offset, length; sorted by path |
| `missing.txt` | the source paths that could not be read |
| `manifest.json` | the completeness marker and the sources |

A stage reads the pack when `runtime.image_pack` (or `model.image_pack`)
names the directory:

```bash
python -m dagspaces.pairwisevqa.submit_shards --sweep ... \
    --set runtime.image_pack=/share/ju/matt/mllmsci/multirun/pair_tables_1m/image_pack
```

The reader is `dagspaces/common/image_pack.py`. The single-process path
prefetches the images of 1 batch (256 rows) with 8 threads before the
generate call and drops them after it. The DP worker does the same for 1
chunk inside its image thread pool. Both loaders serve the bytes from memory,
and a path that is not in the index reads the file as before. The bytes are
the same JPEG bytes, thus the results do not change (checked pixel for pixel
on 2 images). The NFS cost falls from 8.4 round trips for each image on
`/share/ju` to 1 READ of about 240 kB on `/share/pierson`. The stage log
prints `image pack ON` at start and `image_pack hits=... misses=...` at the
end; a miss count above 0 means the pack does not cover the table.

`tests/test_image_pack.py` covers the build, the byte identity, the prefetch
cache, a missing source and an incomplete pack.

### 3. Fallback (obsolete after 2026-10): a mirror of the raw files on /share/pierson

`rsync` the 1.1 million files to `/share/pierson` and rewrite the path prefix
in `image_path`. This keeps the 8.4 round trips for each image and moves them
to the group server. It is the smallest change, and the weakest fix.

## Do not

- Do not add a thread pool to the single-process image hydration as the fix.
  More client concurrency makes the collapse worse. The serial load is now the
  only throttle.
- Do not shrink the images in the pack. The model input must stay the same.

## Verification, 2026-09-03 evening

The 200 unfinished shards ran again as array 252078 with the pack, 32 at a
time. The chunk timestamps give the row rate, and `sacct` gives the card.

| Run | Card | Shards at once | Rows/s for each shard (median) |
|-----|------|----------------|--------------------------------|
| Old array, raw files | RTX 6000 Ada | 30 to 80 | 1.5 |
| Old array, raw files | RTX PRO 6000 Blackwell | 15 to 17 | 13.8 |
| Old array, raw files | A40 | 15 to 17 | 4.7 |
| New array, pack | RTX 6000 Ada | 8 to 16 | 5.3 |

On the same card the pack gives 3.5 times the rate of the collapsed run, and
the rate now sits where the card class sits (A40 and L40S gave 4.7 and 7.6
when the server was not loaded). A finished shard reports
`image_pack hits=2810 misses=0`. A cold image open on `/share/ju` fell from
about 250 ms to about 110 ms once the sweep read the pack instead.

## Prefetch of the next batch, 2026-09-03 evening

After the pack, about 7 seconds of image work for each batch of 256 rows
stayed on the main thread (3 s of pack reads, 4 s of JPEG decode), against
about 48 s of GPU time on an RTX 6000 Ada. `_BatchPrefetcher` in
`dagspaces/common/vllm_inference.py` now does that work 1 batch ahead:

- Before the loop, the stage lists the batches that run, with the cached
  resume chunks left out, and submits the first batch to 1 background thread.
- When the loop reaches batch k, it takes the decoded PIL images of k and
  submits k+1. The generate call of k then runs while k+1 loads.
- The decode uses `model.image_decode_workers` threads. The default is 4
  with a pack and 1 without: on the raw tree more readers make the server
  collapse faster.
- A batch that the prefetcher does not hold (a damaged chunk that runs
  again, or a failed prefetch) loads its images in line, as before.
- The prefetcher drops the pack byte cache after the decode, thus memory
  holds 2 decoded batches at most (about 3 GB at 1024x1024).

The log prints `image prefetch ON` at start and, at the end, how long the
GPU waited for images in total. The DP worker keeps its own per-chunk pool
and is unchanged. `tests/test_image_prefetch.py` covers the prefetcher.

Measured on array 252078, RTX 6000 Ada, road_quality, 2026-09-03 22:20:

| Code | Chunk intervals | Rows/s for each shard (median) |
|------|-----------------|--------------------------------|
| Pack only | 146 | 5.2 |
| Pack and prefetch | 13 | 5.6 |

The first finished shard on the new code reports 22 batches and a total GPU
wait of 2.7 s for images, about 0.1 s for each batch. The image path is off
the critical path now; the rate that remains is the GPU and vLLM.
