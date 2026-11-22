# The last-token readout

`research/monocle/answer_tokens.py` decodes the token distribution at the last token
position, at every layer, over the whole vocabulary.

    logits_l[answer] = unembed( J_l @ h_l[answer] )   for the fitted layers
    logits_47[answer] = the model's own logits        (exact, no translation)

then top-k over all 262,144 tokens at every layer.

## Why it is separate from safety_workspace

`safety_workspace.py` (the late-layer readout at the last token position)
reads the same position, but it logs only the preselected quantities: the
label probability, a restricted argmax, and the mass on 2 fixed word lists. It
never records which tokens the translated readout contains. This module
counts them.

## What it writes

Per case, under `--out-dir/<case>/`:

| File | One row per |
|---|---|
| `answer_tokens.parquet` | (pair, cond, pos, layer, rank) — the top-k tokens |
| `answer_labels.parquet` | (pair, cond, pos, layer) — the label metrics |
| `summary.json` | the corpus count: most frequent top-1 token per layer |

Both data frames carry `case`, so the per-case files concatenate.

## Interpretation of the L42-L47 window

Two earlier results constrain the interpretation of the others.

- Under the production prompt the label token carries no mass through L36.
  Before L42 the answer is not in the last-token readout.
- Layers at or below L30 are corpus-dominated. The wikitext and urban lenses
  emit near-disjoint token sets there. An early layer decodes its fitting
  corpus, not the image.

A token at L6 is an artifact of the method. A token at L42 is a finding.

## Raw top-k, with a word flag

The top-k is over the whole vocabulary, with no filter. At the last token
position the leading tokens are often JSON syntax or a control token, and a
filter would misrepresent the readout. The `is_word` column annotates them
instead, so a readable view is a query:

```python
df[df.is_word]                      # the word channel
top_tokens_by_layer(df, "prod", words_only=True)
```

`prob` is the softmax over the full vocabulary, so it stays comparable across
layers and across the word / non-word split.

## Options

| Option | Note |
|---|---|
| `--cases` | all 7 by default |
| `--kind` | `proxy`; `trace` enables thinking and moves the last token position |
| `--force-prefix` | `natural` by default. Do not use `compact` — see the prompt-path doc |
| `--lens` | the wikitext reference lens by default |
| `--shard i/n` | strided, so every shard sees the same label mix |

Resume is by `pair_id`, checkpointed every 25 pairs, with an atomic parquet
write. A pre-emption costs at most 25 pairs.

## Cost

1.5 s per pair for 2 conditions and 2 readout positions, on an A6000. 7 cases
at 1,000 pairs each is about 2.9 GPU-h.

The 262k-token word mask is cached to `cache/monocle/`. Without the cache each
shard rebuilds it, which took 20 minutes on a contended node.

## Commands

```bash
pytest tests/test_answer_tokens.py -v

sbatch --time=8:00:00 research/monocle/monocle.sub monocle.answer_tokens \
    --n-pairs 1000 --out-dir outputs/_monocle/answer_tokens
```

## See also

- [gemma4-12b-architecture.md](gemma4-12b-architecture.md) — the 48 layers, the two attention types, the encoder-free vision path.
