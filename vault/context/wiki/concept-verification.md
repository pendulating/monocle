---
title: "Answer Verification"
category: concept
created: 2026-04-06
updated: 2026-04-06
tags:
  - concept
  - verification
  - nli
  - embeddings
---

# Answer Verification

Post-inference answer filtering and scoring in UrbanVQA.

## Overview

After the VQA stage produces raw model answers, the verification system scores and filters those answers against a taxonomy of expected categories. This catches hallucinations, off-topic responses, and ambiguous answers before they propagate to downstream analysis.

## Key File

`dagspaces/urbanvqa/verification_core.py`

## VerificationConfig

The `VerificationConfig` dataclass controls all verification behavior:

| Field | Default | Description |
|-------|---------|-------------|
| `method` | `"combo"` | Verification method: `off`, `embed`, `nli`, `combo`, `combo_judge` |
| `top_k` | `3` | Number of top matching categories to return |
| `sim_threshold` | `0.55` | Minimum cosine similarity for embedding match |
| `entail_threshold` | `0.85` | Minimum entailment probability for NLI |
| `contra_max` | `0.05` | Maximum contradiction probability allowed |
| `embed_model_name` | `"intfloat/multilingual-e5-base"` | Embedding model for similarity scoring |
| `nli_model_name` | `"MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"` | NLI model for entailment scoring |
| `device` | `None` | Device override (`"cuda"`, `"cpu"`, or auto-detect) |

## Verification Methods

### `off`

No verification. Raw model answers pass through unchanged.

### `embed` -- Embedding Similarity

Computes cosine similarity between the model's answer and each taxonomy category using a sentence embedding model.

1. Encode the answer text with E5-style prefix (`"query: "`)
2. Encode taxonomy categories with passage prefix (`"passage: "`)
3. Compute cosine similarity matrix
4. Return top-k categories above `sim_threshold`

Uses mean-pooling over the last hidden state with L2 normalization. The embedding model (`intfloat/multilingual-e5-base`) supports multilingual text.

### `nli` -- Natural Language Inference

Uses an NLI model to check whether the answer text entails each taxonomy category.

1. Construct premise-hypothesis pairs: (answer_sentence, category_text)
2. Run through NLI model to get entailment/neutral/contradiction probabilities
3. Accept categories where entailment > `entail_threshold` and contradiction < `contra_max`

The NLI model (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) handles cross-lingual entailment.

### `combo` -- Combined Embed + NLI

Runs both embedding similarity and NLI, then merges results:

- A category must pass both the similarity threshold and the NLI entailment threshold
- Contradiction above `contra_max` vetoes a match even if similarity is high

This is the default method and provides the best balance of precision and recall.

### `combo_judge` -- LLM as Tie-Breaker

Extends `combo` with an LLM judge for ambiguous cases:

- When embed and NLI signals conflict (e.g., high similarity but low entailment, or vice versa), the system falls back to an LLM to make the final determination
- Useful for edge cases where the answer is a valid paraphrase that confuses simpler methods

## Core Functions

### `init_verification(cfg)`

Initializes the verification pipeline:

1. Parses the `VerificationConfig` from the Hydra config
2. Detects device (CUDA if available, otherwise CPU)
3. Loads and caches the embedding model and tokenizer (global `_EMBED_MODEL`, `_EMBED_TOKENIZER`)
4. Loads and caches the NLI model and tokenizer (global `_NLI_MODEL`, `_NLI_TOKENIZER`)
5. Builds the label-to-text mapping from the taxonomy

Models are cached globally so they are loaded only once per process.

### `verify_answer(answer_text, top_k, method)`

Scores a single answer against the taxonomy:

1. Splits the answer into sentences via `_split_sentences()`
2. Runs the configured verification method
3. Returns scored category matches

### `parse_thresholds_string(s)`

Parses a compact threshold string into `(sim, ent, contra)` floats:

```
"sim=0.55,ent=0.85,contra=0.05" -> (0.55, 0.85, 0.05)
```

This allows thresholds to be specified as a single string in config files.

## Internal Helpers

| Function | Description |
|----------|-------------|
| `_encode_embeddings(texts, is_query)` | Encode texts with E5-style prefixes, mean-pool, L2-normalize |
| `_cosine_sim_matrix(a, b)` | Compute cosine similarity matrix between two embedding arrays |
| `_split_sentences(text)` | Rule-based sentence splitter (regex on `.!?` and newlines) |
| `_nli_probs(premises, hypotheses)` | Run NLI model, return (entailment, neutral, contradiction) probability arrays |
| `_build_label_to_text_map(taxonomy)` | Convert taxonomy dict to numbered label-to-text mapping |
| `_mean_pool(hidden_state, attention_mask)` | Mean pooling over transformer hidden states |

## Configuration in Hydra

Verification is configured under the `verify` group in UrbanVQA configs:

```yaml
verify:
  method: combo
  thresholds: "sim=0.55,ent=0.85,contra=0.05"
  top_k: 3
  embed_model_name: "intfloat/multilingual-e5-base"
  nli_model_name: "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
```

## See Also

- [[urban-vqa]] -- the UrbanVQA dagspace that uses verification
