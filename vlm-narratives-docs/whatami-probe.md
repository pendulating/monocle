# The "what am I looking at?" probe

Started 2026-10-20. This is an exploratory study, not a battery for the CVPR
paper. The CVPR cases ask questions that a smart-glasses user does not ask.
This probe asks the question that users ask most.

## The question

A user of smart glasses says "what am I looking at?". The glasses take a
photo and send it to a cloud model. The glasses then speak the answer. Users
on TikTok and in the press record this use often.

The probe gives a street-view image and this question to a model. It asks the
same question 10 times for each image, with sampling on. Thus we can see what
the models say about NYC streets, and how much the answer changes between
calls.

## The design

| Item | Value |
|------|-------|
| Images | 1,000, `data/whatami/sample_1k.parquet` |
| Prompt | `What am I looking at?` (one user turn: the image, then the text; no system prompt) |
| Repeats | 10 for each image, each with its own seed; 10,000 requests for each model |
| Sampling | temperature 0.6, top_p 1.0, top_k off, min_p 0, no penalties, max_tokens 1024 |
| Thinking | off (`enable_thinking: false`) for gemma and qwen |
| Engine | one server for all 10,000 requests: `vllm serve` (gemma, qwen), SGLang (llama) |
| GPU | 1 B300 on g3, account and QoS `nissenbaum` |

The request sets each sampling value. Thus the defaults in a model's
`generation_config.json` do not change the result, and all models sample at
plain temperature 0.6. The seed of a request comes from the sha256 of
`sample_id:repeat`. Thus each model gets the same seeds.

Thinking is off, because the glasses speak the answer at once. A run with
thinking on is a possible second arm.

### The image sample

`scripts/whatami/sample_images.py` draws the sample (seed 20261020).

1. The frame is the 2025 Cyclomedia coverage: the `year=2025` partitions of the
   five `*_2025_1k` datasets in `/share/ju/cyclomedia/catalog/v1`. The 2017
   partitions of manhattan and queens are not in the frame.
2. Only horizontal cube faces (F, B, L, R) are in the frame, as in the CVPR
   paper.
3. Rows with `manifest_ok` false or a file of 50 KB or less are removed (catalog
   check 8). 19,780,260 faces in 4,950,270 recordings remain.
4. The script draws 1,000 recordings uniformly, then one face at random in each
   recording. Thus no two images show the same spot.

Each borough thus gets weight in proportion to its coverage:

| Borough | Images |
|---------|--------|
| bronx | 159 |
| brooklyn | 242 |
| manhattan | 73 |
| queens | 364 |
| staten_island | 162 |

The faces are B 237, F 259, L 231, R 273. Each image is a 1024 x 1024 JPEG.

**Warning:** do not use `sample()` inside `group_by().agg()` in polars to take
one row from each group. The seed is the same in each group, thus it takes the
same position each time. The first draft did this, and 998 of 1,000 images
were L faces.

## The models

| Key | Weights in the zoo | Server | Notes |
|-----|--------------------|--------|-------|
| `gemma-4-12b` | `gemma-4-12B-it` | vLLM 0.25.0 (the default venv) | as in the CVPR battery |
| `qwen3.5-9b` | `Qwen3.5-9B` | vLLM 0.25.0 | `max_pixels` 1048576, as in the battery |
| `llama-3.2-11b` | `Llama-3.2-11B-Vision-Instruct` | SGLang 0.5.22 | downloaded 2026-10-20 |
| `llama-3.2-90b` | `Llama-3.2-90B-Vision-Instruct` | SGLang 0.5.22 | downloaded 2026-10-20, about 177 GB in BF16 |
| (later) | `Llama-Guard-3-11B-Vision` | SGLang 0.5.22 | downloaded 2026-10-20; not in the job table yet |

The Llama downloads skip the `original/` folders (the Meta-format copy of the
weights). Use the token in `/share/ju/matt/cache/huggingface/token`. The token
in `~/.cache/huggingface/token` is not valid (2026-10-20).

### Llama 3.2 Vision runs on SGLang, not vLLM

Llama 3.2 Vision uses the `MllamaForConditionalGeneration` architecture.
Llama Guard 3 Vision uses the same architecture.

| Try | Result |
|-----|--------|
| vLLM 0.25 | no Mllama: vLLM removed it in 0.10.2 with the V0 engine (`_PREVIOUSLY_SUPPORTED_MODELS` in `vllm/model_executor/models/registry.py`) |
| vLLM 0.10.1.1, the last release with Mllama | needs torch 2.7.1; the newest build is cu128 and has no B300 kernels. Job 70187: `CUDA error: no kernel image is available for execution on the device` in model init |
| SGLang 0.5.22 on CUDA 13 | has `models/mllama.py` and the Mllama image processor (`multimodal/processors/mlama.py`) |

The SGLang stack:

- The pins are in `scripts/whatami/requirements-sglang0522-cu130.txt`: torch
  2.14.1+cu130, sglang-kernel 0.4.9+cu130, transformers 5.19.0, flashinfer
  0.7.1.
- Take sglang-kernel from the cu130 index (`docs.sglang.ai/whl/cu130`). Its
  `common_ops` holds plain `sm_100` SASS, which runs on a 10.x chip such as the
  B300 (10.3). The PyPI build holds only `sm_100a` and `sm_120a`, which do not.
- The SGLang 0.5.22 wheels are `manylinux_2_34`. Compile for
  `x86_64-manylinux_2_39` (glibc 2.39 on g3), not 2.28.
- `scripts/whatami/ensure_scratch_venv.sh` builds the venv on node `/scratch`
  in the job. The marker holds the sha of the pins.
- The job sets `CUDA_HOME=/usr/local/cuda-13.3` for this venv.
- The job sets `--attention-backend flashinfer`. On Blackwell, SGLang selects
  the TRT-LLM backend, which failed on the Mllama cross-attention layers in the
  first prefill (job 70604: `Tensor match failed` in `kvcache.cuh`). The
  FlashInfer backend handles `encoder_lens`.
- The job sets `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. The venv
  python is the anaconda python, which loads an old libstdc++. The SGLang JIT
  kernels need `GLIBCXX_3.4.31` (job 70482). The repo launchers use the same
  preload.
- The job sets `SGLANG_CACHE_DIR=/scratch/$USER/sglang-cache`. The default is
  `~/.cache/sglang`, on the home quota (496 MB after one start).

A Llama prompt is 6,421 tokens: 4 image tiles of 1,601 tokens and 17 text
tokens. The 17 text tokens show that no system prompt is present. SGLang logs
the model defaults (temperature 0.6, top_p 0.9), but the value in each request
takes precedence, thus top_p is 1.0, as for gemma and qwen.

**Warning:** do not pass `--chat-template` to SGLang for these models. Its
`llama_3_vision` template adds a system prompt ("You are a helpful language
and vision assistant..."). Without the flag, SGLang uses the model's own chat
template. That template adds no system prompt when an image is present, thus
the prompt agrees with the gemma and qwen prompts.

**Warning:** compile a lock file with `uv pip compile --no-config`. Without it,
uv reads the repo `pyproject.toml`, and its override `transformers>=5.5`
replaces the transformers pin.

### Three faults on the B300, now fixed

| Fault | Cause | Fix |
|-------|-------|-----|
| The vLLM server took 1,280 s to start (job 70086) | the mirror's `bin/vllm` shebang names the NFS python, thus each process imported torch and vLLM over NFS | start the server with `"$PY" -m vllm.entrypoints.cli.main serve`; 220 s after the fix (job 70249) |
| qwen: `fatal error: cublasLt.h: No such file or directory` in a FlashInfer JIT build for `103a` (job 70186) | `/share/ju/matt/cuda-12.9` has no cuBLAS headers; qwen gets the FlashInfer attention backend, gemma gets Triton | `FLASHINFER_EXTRA_CUDAFLAGS=-I<venv>/nvidia/cublas/include` |
| The first gemma requests took about 28 s each | a Triton attention kernel compiles during the first requests | none needed: it occurs one time for each server |

A first copy of the weights from NFS to `/scratch` ran at 36 to 200 MB/s on
2026-10-20. The 90B copy (about 177 GB) can thus take 15 to 80 minutes.

## How to run

```bash
# from the project root, on any host that can submit to g3
bash scripts/whatami/submit_all.sh                         # the 4 models in a chain
bash scripts/whatami/submit_all.sh llama-3.2-90b           # one model
WHATAMI_TAG=smoke WHATAMI_LIMIT=5 WHATAMI_REPEATS=2 \
  bash scripts/whatami/submit_all.sh gemma-4-12b           # a smoke test
```

QoS `nissenbaum` gives the group 1 GPU. Thus the jobs run one after the other.
The chain uses `afterany`, thus a failed model does not block the next one.

### What one job does

`scripts/whatami/serve_and_query.sbatch`:

1. It gets the venv. The vLLM 0.25 venv comes from its `/scratch` mirror
   (`activate_stage_venv.sh`). The SGLang venv is built on `/scratch`.
2. `scripts/whatami/stage_inputs.sh` copies the weights and the 1,000 images to
   `/scratch/$USER/whatami/`. Each copy has a `.sync_complete` marker.
3. It starts the server on `127.0.0.1` and waits for `/health`. vLLM gets
   `--allowed-local-media-path` set to the image copy. SGLang reads any local
   path.
4. `scripts/whatami/query.py` sends the requests (128 at a time). The repeats of
   one image go together, thus the server can reuse the image encoding.
5. It stops the server.

## The outputs

`multirun/whatami/sample_1k[-<tag>]/<model>/`:

| File | Content |
|------|---------|
| `responses.jsonl` | one line for each request; a restart appends to it |
| `responses.parquet` | one row for each (image_idx, repeat): the last good result |
| `run_config.json` | the prompt, the sampling values, the sample path |

The columns are `model_key`, `served_model`, `image_idx`, `sample_id`,
`repeat`, `seed`, `prompt`, `sampling`, `chat_template_kwargs`, `started_at`,
`response`, `reasoning`, `finish_reason`, `prompt_tokens`,
`completion_tokens`, `latency_s` and `error`. Join `sample_id` to the sample
file for the borough, the location and the face.

A requeue or a second job skips each key that has a good result. The job ends
with exit code 1 when a key has no good result.

The server log is `logs/whatami/server-<model>-<job>.log`, and the job log is
`logs/whatami/whatami-<model>-<job>.out`.

## Run status (2026-10-20)

| Model | Job | Result |
|-------|-----|--------|
| `gemma-4-12b` | 70461 | 10,000 of 10,000 good; about 8 min at 20 req/s; median answer 203 tokens; 14 answers at the 1,024 cap |
| `qwen3.5-9b` | 70462 | 10,000 of 10,000 good; about 10 min at 17 req/s; median answer 363 tokens; 3 answers at the cap |
| `llama-3.2-11b` | 70705 | 10,000 of 10,000 good; about 10 min at 17 req/s; median answer 104 tokens; 1 answer at the cap |
| `llama-3.2-90b` | 70706 | 10,000 of 10,000 good; about 27 min at 6.2 req/s; median answer 131 tokens; 0 answers at the cap; the weight copy took 5,370 s (166 GB) |

A job that lands on a new node pays about 10 to 20 min of setup: the venv, the
weights and the images go to `/scratch`. Pin a node that holds them with
`--nodelist` when you run the job again.

First counts (substring matches, not a classification; one answer can name
more than one place):

| Term | gemma-4-12b | qwen3.5-9b | llama-3.2-11b | llama-3.2-90b |
|------|-------------|------------|---------------|---------------|
| New York | 23.1% | 39.2% | 6.1% | 3.3% |
| Brooklyn | 2.9% | 15.6% | 2.2% | 0.8% |
| Philadelphia | 2.4% | 13.8% | 0.0% | 0.0% |
| Boston | 1.6% | 10.3% | 0.0% | 0.0% |
| street view | 8.0% | 4.8% | 9.1% | 9.3% |

The "street view" row matters for validity. Some answers say that the image is
street-view imagery. A photo from smart glasses is not, thus the cube faces
are an imperfect stand-in for a glasses photo.

## Open items

- Llama Guard 3 Vision: it is a classifier with its own prompt format. It needs
  a separate script that scores the (image, question, answer) turns.
- A thinking-on arm for gemma and qwen.
- A system prompt like the one a glasses product uses.
