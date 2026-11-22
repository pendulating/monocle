# The gemma-4-12b architecture

`gemma-4-12B-it` is the encoder-free member of the gemma-4 family. Its
`model_type` is `gemma4_unified`, and its class is
`Gemma4UnifiedForConditionalGeneration`.

Every number here comes from the checkpoint at
`/share/ju/matt/zoo/models/gemma-4-12B-it` — `config.json`, the tensor
names in `model.safetensors`, and the transformers source in `.venv-nightly`.
We read the weights. We did not read the model card.

This page is the reference for [patch-attribution.md](patch-attribution.md),
[where-the-judgment-lives.md](where-the-judgment-lives.md) and
[canonical-lens-prompt-path.md](canonical-lens-prompt-path.md).

## The stack

| Property | Value |
|---|---|
| Blocks | 48, indices 0 to 47 |
| `hidden_size` | 3840 |
| `intermediate_size` | 15360 |
| Attention heads | 16 |
| Vocabulary | 262,144, tied embeddings |
| `final_logit_softcapping` | 30.0 |
| `rms_norm_eps` | 1e-6 |
| MoE | off (`enable_moe_block: false`) |

## Two kinds of block, not one

`layer_types` names 8 blocks `full_attention` and the other 40
`sliding_attention`. The global blocks are **5, 11, 17, 23, 29, 35, 41, 47** —
every 6th block, and the last block is always global.

The 2 kinds do not only differ by the mask. **They are different attention
mechanisms.**

| Tensor | Sliding block (40) | Global block (8) |
|---|---|---|
| `q_proj` | (4096, 3840) — 16 x 256 | (8192, 3840) — 16 x **512** |
| `k_proj` | (2048, 3840) — **8** KV heads | (512, 3840) — **1** KV head |
| `v_proj` | (2048, 3840) | **absent** |
| `o_proj` | (3840, 4096) | (3840, 8192) |
| `q_norm` / `k_norm` | (256,) | (512,) |

A global block has **no value projection**. `attention_k_eq_v: true`, so the
key vectors ARE the value vectors. We checked every block: the 8 blocks with
no `v_proj` are exactly the 8 global blocks.

Thus a global block is twice as wide for each head, it holds 1 shared KV head
(multi-query), and it reads its values out of its keys.

The 2 kinds also use different RoPE:

| | `rope_type` | `rope_theta` | `partial_rotary_factor` |
|---|---|---|---|
| sliding | default | 10,000 | — |
| full | proportional | 1,000,000 | 0.25 |

**Warning:** `sliding_window` is 1024, and a pairwise prompt holds about 703
tokens. Thus the window covers the whole sequence and no token is hidden by
it. That is a fact about the MASK only. A sliding block and a global block
still compute differently. Do not say "every layer is global here".

## Inside one block

Each block holds the gemma sandwich of 4 norms:

```
x -> input_layernorm -> self_attn -> post_attention_layernorm -----> + x
  -> pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> + x
```

The MLP is gated: `gate_proj` and `up_proj` both go 3840 -> 15360, the
activation is `gelu_pytorch_tanh`, and `down_proj` returns 15360 -> 3840.

Each block also holds a `layer_scalar` of shape (1,) — 1 learned scalar.
**OPEN:** we did not check how the block applies it.

## The vision path

There is no vision tower. The larger siblings (26B A4B, 31B) run a 550 M
SigLIP-like encoder. This model replaces all of it with about 35 M parameters,
and 26 M of those are the projection alone.

```
raw 48x48x3 patch (6912 values)
  -> patch_ln1  (6912)
  -> patch_dense (6912 -> 3840)
  -> patch_ln2  (3840)
  -> + pos_embedding[x] + pos_embedding[y]      (1120, 2, 3840)
  -> pos_norm
  -> embed_vision.embedding_projection (3840 -> 3840)
  -> a sequence position
```

Two properties matter for every read of this model.

1. **The embedder holds no attention.** It reads each patch alone. Thus all
   the work that mixes patches happens inside the language model.
2. **The only spatial signal is additive.** 1 vector for the column, 1 for the
   row, from a shared table. All the patches of a row share the row vector.

**Warning:** `patch_dense` casts its input to bfloat16. Thus the model is a
step function at fine scale, and a pixel gradient describes a float32 model
that does not exist. See [patch-attribution.md](patch-attribution.md).

## Patch geometry

A token budget sets the patch count, not a division of the image. The
processor resizes to fit `max_soft_tokens: 280`, and both sides must divide
by 48. Measured with the real processor:

| Image | Grid | Patches |
|---|---|---|
| 1024 x 1024 | 16 x 16 | 256 |
| 1600 x 1200 | 19 x 14 | 266 |
| 2048 x 1024 | 23 x 11 | 253 |

Cyclomedia faces are square, so ours are always 256.

`pixel_values` always pads to the full 280 slots. The live patches are the
ones whose `image_position_ids` is not -1.

## The attention mask

`use_bidirectional_attention: "vision"`. **The image patches attend to each
other in both directions.** Only the text stays causal.

`get_block_sequence_ids_for_mask` builds the mask from `mm_token_type_ids`, so
each contiguous run of vision tokens is its own bidirectional island. Two
images give 2 islands. Image A cannot see image B at all. Image B can see
image A, because the blocks stay causal against each other.

Measured geometry of a pairwise prompt (`seq_len` 706):

| | Block A | Block B |
|---|---|---|
| positions | 8 to 263 | 269 to 524 |
| token before | 255999 (`boi`) | 255999 |
| token after | 258882 (`eoi`) | 258882 |

`mm_token_type_ids` holds only [0, 1], so `boi` and `eoi` are **not** vision.
The island is exactly the 256 patches.

This is why the prompt layout matters. Two adjacent image blocks with no text
between them read as 1 image. See
[canonical-lens-prompt-path.md](canonical-lens-prompt-path.md) and
[gemma-4-best-practices.md](../../docs/gemma-4-best-practices.md).

The island is vision-only, and `blockwise_overlay` proves it: it unmasks a
pair only when both positions hold the same island id, and text carries -1.
The overlay is OR'd with the causal mask. **Thus a patch reads every token
before it, plus its own island, and never text that follows.** With the image
first, the patch residual is bit-identical under 2 different questions, and
with the text first it is not. See [modality-order.md](modality-order.md).

## What we measured about depth

| Depth | What happens |
|---|---|
| L18-L24 | patch maps look legible to a person, but hold no judgment signal |
| L30 | about 2 register patches carry nearly all the decision signal |
| L34 | a register floor: `<pad>` is the top readout of 82.5% of patches |
| L39 to L40 | a sharp switch: mean r goes 0.398 to 0.766. **OPEN** |
| L40-L45 | the attribution window; L43 is the peak at 0.805 |
| L36-L44 | ordinary patches read scene words; L42 is the best at 78% |
| L46 | 1 patch holds 43% of the map; the readout collapses to `ed` |
| below L30 | the image reaches the answer here. A whole-image swap above L30 does nothing |
| L22-L29 | the handover: the patches pass the judgment to the QUESTION tokens |
| L30-L41 | the question tokens pass it to the ANSWER position |
| L42-L47 | the judgment is already at the answer; image attention is idle |
| L47 | `dc/dh[patch]` is identically 0 — `c` reads `h_47[answer]` alone |

Because `c` reads only `h_47[answer]`, the sole path from a patch to the
answer at L46 is block 47's attention. Block 47 is a **global** block: 1 KV
head, K = V, head width 512. The final patch of each image block carries 117x
the gradient of any other patch there, and the answer attends to it 80-105x
more than to any other patch.

Cutting the whole stream off from the image confirms it. Blinding every
non-patch position to the patches costs 7.68 across all 48 blocks, but only
0.458 across blocks 30-47, and 0.131 when only the final patch is cut at
block 47.

**Warning:** that is an ADDRESS, not a route. A residual-stream swap moves `c`
no more when it replaces the last patch than when it replaces an ordinary one,
at every depth tested. And a swap of ALL 256 patches changes `c` only below
L30. Thus the image is spent early, and the deep attention concentration is
causally inert. See [patch-attribution.md](patch-attribution.md).

Full numbers in [patch-attribution.md](patch-attribution.md). The lens method
is in [canonical-lens-prompt-path.md](canonical-lens-prompt-path.md) and
[answer-token-readout.md](answer-token-readout.md).

## Open questions

| Question | Status |
|---|---|
| What does `layer_scalar` do? | not checked |
| `vocab_size_per_layer_input` is 262144 but `hidden_size_per_layer_input` is 0 | the gemma-3n per-layer embeddings appear off. Not confirmed |
| Why does L39 to L40 switch so sharply? | global blocks sit at 35 and 41. Not tested |
| Does block 47 attention confirm the last-patch route? | YES. 80-105x on 3 cases. The ramp starts at block 17 |

## Serving

vLLM 0.25.0 in `.venv-vllm025cu129`, and transformers 5.12 in `.venv-nightly`
for the lens work. `limit_mm_per_prompt.image` must be 2 for a pairwise
prompt. See
[gemma-4-best-practices.md](../../docs/gemma-4-best-practices.md).
