# The answer-token readout

`monocle/answer_tokens.py` reads what the answer position is disposed to say,
at every depth, over the whole vocabulary.

    logits_l[answer] = unembed( J_l @ h_l[answer] )   for the fitted layers
    logits_47[answer] = the model's own logits        (exact, no transport)

then top-k over all 262,144 tokens at every layer.

## Why it is separate from safety_workspace

`safety_workspace.py` reads the same position, but it logs only what it was
told to look for: the label probability, a restricted argmax, and the mass on 2
fixed word lists. It never records WHICH tokens the transport carries. This
module tallies them.

## What it writes

Per case, under `--out-dir/<case>/`:

| File | One row per |
|---|---|
| `answer_tokens.parquet` | (pair, cond, pos, layer, rank) — the top-k tokens |
| `answer_labels.parquet` | (pair, cond, pos, layer) — the label metrics |
| `summary.json` | the corpus tally: most frequent top-1 token per layer |

Both frames carry `case`, so the per-case files concatenate.

## Read the L42-L47 window

Two earlier results bound what the rest can mean.

- Under the production prompt the label token carries no mass through L36.
  Before L42 the answer is not in the channel.
- Layers at or below L30 are corpus-dominated. The wikitext and urban lenses
  emit near-disjoint token sets there. An early layer reads out its fitting
  corpus, not the image.

A token at L6 is an artefact of the instrument. A token at L42 is a finding.

## Raw top-k, with a word flag

The top-k is over the whole vocabulary, with no filter. At the answer position
the leading tokens are often JSON syntax or a control token, and to hide them
would misrepresent the channel. The `is_word` column annotates instead, so a
readable view is a query:

```python
df[df.is_word]                      # the word channel
top_tokens_by_layer(df, "prod", words_only=True)
```

`prob` is the softmax over the full vocabulary, so it stays comparable across
layers and across the word / non-word split.

## Options that matter

| Option | Note |
|---|---|
| `--cases` | all 7 by default |
| `--kind` | `proxy`; `trace` enables thinking and moves the answer position |
| `--force-prefix` | `natural` by default. Do not use `compact` — see the prompt-path doc |
| `--lens` | the wikitext reference lens by default |
| `--shard i/n` | strided, so every shard sees the same label mix |

Resume is by `pair_id`, checkpointed every 25 pairs, with an atomic parquet
write. A pre-emption costs at most 25 pairs.

## Cost

1.5 s per pair for 2 conditions and 2 read positions, on an A6000. 7 cases at
1,000 pairs each is about 2.9 GPU-h.

The 262k-token word mask is cached to `cache/monocle/`. Without the cache each
shard rebuilds it, which took 20 minutes on a contended node.

## Commands

```bash
pytest tests/test_answer_tokens.py -v

sbatch --time=8:00:00 monocle/monocle.sub monocle.answer_tokens \
    --n-pairs 1000 --out-dir outputs/_monocle/answer_tokens
```
