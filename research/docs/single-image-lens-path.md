# The single-image lens path

The lens work uses one image and a 3-point rating, from 2026-08-21. The
pairwise prompt template stays for the registered battery. This page gives the
reason for the change and the new prompt specification.

The architecture is in
[gemma4-12b-architecture.md](gemma4-12b-architecture.md). The attribution
results are in [patch-attribution.md](patch-attribution.md). The pairwise
prompt specification is in
[canonical-lens-prompt-path.md](canonical-lens-prompt-path.md).

## Reason for the change

The pairwise prompt is not symmetric. Measured on 20 pairs of subway_safety,
attention from the last token position:

| Layer | on the image tokens of A | on the image tokens of B |
|---|---|---|
| 42 | 0.12 | 0.26 |
| 46 | 0.12 | 0.24 |

The model attends to image B broadly. It attends to image A mostly through the
final image token of A. The image order is our choice, so this asymmetry is
present in every difference map we made.

| Property | Pairwise | Single image |
|---|---|---|
| image spans with bidirectional attention | 2 | 1 |
| a map requires | (sum A) - (sum B) | nothing |
| prompt length | about 703 tokens | about 250 |
| text share at layer 47 | 94% | higher image share |

There is a second reason. Every case in the registry records
`image_layout: interleaved_labels`, and the lens rejects that layout from
2026-08-20. Thus the whole pairwise battery needs a re-run anyway. The new
path starts from a specification we control.

The layer result this path produced is in
[where-the-judgment-lives.md](where-the-judgment-lives.md).

## What it tests

The change is also an experiment. If the final image token still receives
80-105x the attention of any other image token with 1 image, the aggregation
is intrinsic. If it stops, the aggregation belongs to the comparison. Both
answers are useful.

## The prompt specification

| Piece | Value |
|---|---|
| prompt | `sv/conf/vqa/prompt/rating_2point.yaml` (the 3-point form is removed: its `Neutral` label collapsed, see the head of the file) |
| pipeline | `sv/conf/vqa/pipeline/vqa_cyclomedia_rating.yaml` |
| lens module | `research/monocle/single.py` |
| label rule | `research/monocle/labels.py` |
| forced answer prefix | `{\n  "answer": "` — the same as the pairwise path |
| layout | the image first, then the question |

The lens reads the prompt from the vqa config. It does not keep a copy.
That is the rule the pairwise path follows, and it prevents a divergence
between the run and the lens.

## The labels

The lens reads one position. Thus a label must be 1 token, and 2 labels must
not share a token. `research/monocle/labels.py` holds the rule and raises when a label
breaks it. A broken label fails silently: the readout returns the logit of a
shared prefix, which carries no judgment.

| Set | Result |
|---|---|
| `Safe` / `Neutral` / `Dangerous` | 1 token each |
| `Good` / `Neutral` / `Poor` | 1 token each |
| `Very safe` / `Very unsafe` | 2 tokens, and both start with `Very` |
| `1` .. `7` | 1 token each, but see below |

**Warning:** do not use the digits. They pass the token rule and still fail.
The share of image tokens whose top-1 lens readout is a bare digit is 0.0%
at L30-L38 and then:

| L40 | L41 | L42 | L43 | L46 |
|---|---|---|---|---|
| 8.3% | 11.3% | 10.6% | 6.7% | 17.1% |

That is the attribution window. Thus a digit logit difference shares a
direction with about 9% of the image tokens it scores, in the layers we read.
`Safe`, `Neutral` and `Dangerous` give 0.00% over the same image tokens.

## The cases

The subject and the question track the pairwise battery. subway_safety keeps
Safe/Dangerous. The other 6 cases ask about quality, so they share
Good/Poor, and the logit-difference direction is 1 vector across them.

| Case | Question | Answer words |
|---|---|---|
| subway_safety | How safe does this station entrance look? | Safe / Dangerous |
| libraries | How well maintained does this library building look? | Good / Poor |
| schools | How good a school does this look to be? | Good / Poor |
| road_quality | What condition is this roadway in? | Good / Poor |
| parks_plazas | How well maintained does this park or plaza look? | Good / Poor |
| restaurants | How appealing is this restaurant to eat at? | Good / Poor |
| street_photography | How appealing is this block for a street photography photoshoot? | Good / Poor |

## The sample

`load_single_rows` draws the images from the same pairs the pairwise battery
uses, and it keeps both images of each pair. Thus the 2 prompt templates cover
the same images, and a difference between them is the prompt template and not
the sample.

## Commands

```bash
pytest tests/test_single.py tests/test_labels.py -v

# Check a new label set BEFORE you use it.
python -m research.monocle.labels Safe Neutral Dangerous

# Prove the path end to end on the GPU.
sbatch research/monocle/monocle.sub monocle.single --case subway_safety --n-images 6

# The production run.
python -m dagspaces.vqa.cli -m pipeline=vqa_cyclomedia_rating \
    runtime.debug=true runtime.sample_n=64
```
