# Neuronpedia viewer for the F10.6 reads

The Jacobian Lens viewer of [Neuronpedia](https://github.com/hijohnnylin/neuronpedia)
is open source (Apache 2.0). We run a local copy that shows the F10.6 reads of
`data/f10/hop_all/`. The copy needs no GPU, no model server and no upload.

## How it works

A Neuronpedia share is a database row with the URL of a JSON file. The viewer reads the
file in the browser. We write our reads in that format, and we add the rows to a private
database. The page shows a saved run when no inference host serves the model.

| Part | Location | Note |
|------|----------|------|
| Source, pinned at `e2573d3` | `/scratch/$USER/neuronpedia/src` | Node-local. |
| Postgres 16 with pgvector | `/scratch/$USER/neuronpedia/pgdata` | From the wheel `pgserver`. Port 55432, 127.0.0.1 only. |
| Webapp | port 3001, 127.0.0.1 only | `next dev`. Telemetry off. Vercel analytics off by default. |
| Gate | port 3000, all interfaces | `research/scripts/f10/neuronpedia_gate.py`. Accepts the login node (`F10_VIEWER_ALLOW`) and 127.0.0.1 only. |
| Runs | `data/f10/hop_all/viewer/` | From `research/scripts/f10/hop_viewer.py`. |

## Commands

| Command | Purpose |
|---------|---------|
| `sbatch research/scripts/f10/hop_viewer.sbatch` | Write the runs from the store. About 1,200 files, CPU only. |
| `bash research/scripts/f10/neuronpedia_up.sh` | Install if necessary, load the runs, start the server on this node. |
| `pytest tests/test_hop_viewer.py -v` | Test the export. |

To open the viewer from a laptop, forward the port through the login node:

```bash
ssh <user>@<login-node> -L 3000:<compute-node>:3000
```

Then open `http://localhost:3000/f10/index.html`. The page lists each image with its
answer and links to the group runs.

The node has a routable address, and the viewer has no login. Thus the gate holds the
open port and refuses other addresses. `gate.log` shows each refused address. For a
different login node, set `F10_VIEWER_ALLOW` and run `neuronpedia_up.sh` again.

**Warning:** The server stops when the SLURM job of the session stops. Run
`neuronpedia_up.sh` again on the node. On a different node, the script installs again.

**Warning:** The database password is the default of Neuronpedia, and the server has no
login. Other users of this node and of the login node can reach the viewer.

**Note:** The viewer fetches a run with `force-cache`, thus a browser keeps an old run at
the same URL. `neuronpedia_load.py` adds a new `?v=` stamp to each URL at each load. After
a new export, run `neuronpedia_up.sh`, and do not copy the files only.

## How to read a run

| View | Content |
|------|---------|
| Token row | The prompt. The patch tokens show as 1 token, `[image]`. |
| Column `J++ Lens` | The words with the highest lens logits. |
| Column `Contrast` | The words with the highest contrast of the run. The header of the run gives the contrast. |
| Click on a token | The top word at each layer, for each column. |
| Layer bar | Limits the side lists to a range of layers. |

| Run | Contrast |
|-----|----------|
| 1 image | The read minus the mean of all images. |
| `all` | The mean minus the same text with no image. |
| 1 borough | The borough minus the other images. |
| `high-yes-no`, `low-yes-no` | The top third of `yes_no` minus the bottom third, and the reverse. |

## Limits

- The viewer shows the image as 1 token with no read. The patch residuals and top words
  are in `data/f10/hop_all/{model}/patches/`.
- The tokens before the image cannot see the image, thus they have no contrast.
- A read at a token predicts the next token.
- Only each 4th layer has a read.
- The list holds only the store words (` [a-z]{3,}`), thus no digits or symbols.
- The percent values are a softmax over the store words. A contrast is not a logit.
  Read the rank, not the number.
- Each layer has its own lens. A change in words across layers can be a change of the
  lens and not a change in the model.
- The viewer shows 1 run at a time. It gives no statistics across images.
- The buttons `STEER` and `SWAP` and the chat need a model server, thus they do not work.
- Neuronpedia inference takes text only. We found no image input in the inference app at
  `e2573d3`. A live read of an image prompt needs new code in Neuronpedia and in
  `interp-engine`.

## Local patches

`neuronpedia_up.sh` changes 2 kinds of text in the local copy and sends nothing upstream:

- The labels of the `LOGIT_LENS` column now say `Contrast`.
- `.env.localhost` uses port 55432 and lists our 3 models (`gemma-4-31b`, `qwen3.5-9b`, `qwen3.6-27b`) as J++ models. Qwen3.6-27B joined on 2026-11-02; see [neuronpedia-public-release.md](neuronpedia-public-release.md).

A static copy of the share pages, for reviewers, comes from `research/scripts/f10/share_static/`. See [neuronpedia-public-release.md](neuronpedia-public-release.md), section 8.
