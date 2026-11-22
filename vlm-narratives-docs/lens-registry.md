# Jacobian-lens registry

Generated: 2026-09-07. Do not edit this page by hand.

```bash
.venv-3.12/bin/python research/scripts/register_lenses.py
```

## What this page is

A lens file is a measurement method. Its readout depends on the corpus that fitted it. This page names each Jacobian lens on disk, and gives the hash that identifies it.

The machine-readable copy is `outputs/_monocle/jlens/LENS_REGISTRY.json`. It carries the full hash, the layer list, and the `meta` dict.

## The lenses

| Name | Path | sha256 | Prompts or images | Layers | Dtype | Corpus | Estimand |
|---|---|---|---|---|---|---|---|
| `wikitext` | `outputs/_monocle/jlens/gemma4_12b_lens.pt` | `c90ffd4e2343` | 128 | 8 (6..46) | float16 | 128 WikiText prompts | gemma-4 maps residual-stream content to logits, broad text |
| `urban` | `outputs/_monocle/jlens/urban/gemma4_12b_lens.pt` | `b0545b248263` | 128 | 8 (6..46) | float16 | corpus_urban_1k.json | the same map, under urban text |
| `mm` | `outputs/_monocle/jlens/mm/gemma4_12b_lens.pt` | `65c78be50d57` | 128 | 8 (6..46) | float16 | 128 cyclomedia images, describe prompts | image-token content in a street image decodes to these words |
| `mm_all47` | `outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt` | `f66d058cb896` | 128 | 47 (0..46) | float16 | 128 cyclomedia images | the same map, at every layer |
| `ladder_R0` | `outputs/_monocle/jlens/ladder/R0/gemma4_12b_lens.pt` | `392c68ad1758` | 128 | 8 (6..46) | float16 | 128 cyclomedia images, describe prompts | the baseline of the lens series (the ladder directory); it duplicates the `mm` lens |
| `ladder_R1` | `outputs/_monocle/jlens/ladder/R1/gemma4_12b_lens.pt` | `be9b4ab94ccb` | 128 | 8 (6..46) | float16 | 128 cyclomedia images, describe prompts | the target-span positions alone |
| `ladder_R2` | `outputs/_monocle/jlens/ladder/R2/gemma4_12b_lens.pt` | `791e6bf741de` | 128 | 8 (6..46) | float16 | 128 cyclomedia images, rating prompt, case subway_safety | the prompt-text positions; the answer-token step of the series |
| `lrp_wikitext` | `outputs/_monocle/jlens/lrp/gemma4_12b_lens.pt` | `d9818392c7b9` | 128 | 8 (6..46) | float16 | 128 WikiText prompts | the WikiText lens under the LRP rules |
| `lrp_mm` | `outputs/_monocle/jlens/mm/lrp/gemma4_12b_lens.pt` | `7ef2aa1a1e36` | 128 | 8 (6..46) | float16 | 128 cyclomedia images, describe prompts | the `mm` lens under the LRP rules |
| `varA_half0` | `outputs/_monocle/jlens/varA/half0/gemma4_12b_lens.pt` | `af0f8d70f003` | 128 | 8 (6..46) | float32 | WikiText prompts 0-127 of corpus_wikitext_1k.json | the wikitext map again; half 0 of the split-half variance estimate |
| `varA_half1` | `outputs/_monocle/jlens/varA/half1/gemma4_12b_lens.pt` | `997dc87a729b` | 128 | 8 (6..46) | float32 | WikiText prompts 128-255 of corpus_wikitext_1k.json | the wikitext map again; half 1 of the split-half variance estimate |
| `wikitext1k` | `outputs/_monocle/jlens/wikitext1k/gemma4_12b_lens.pt` | `04a2ae78ce67` | 1000 | 8 (6..46) | float32 | 1,000 WikiText prompts, the reference scale | the wikitext map at the paper's scale |
| `mm_textafter` | `outputs/_monocle/jlens/mm_textafter/gemma4_12b_lens.pt` | `e2bdde9137d9` | 128 | 47 (0..46) | float32 | 128 cyclomedia images, the 7 single-image case prompts in rotation | the text positions after the image map to logits; the question readout |
| `mm_textafter_even` | `outputs/_monocle/jlens/mm_textafter/half_even/gemma4_12b_lens.pt` | `09abea391999` | 64 | 47 (0..46) | float32 | the even shards of mm_textafter, 64 images | the question readout again; half 0 of its split-half variance estimate |
| `mm_textafter_odd` | `outputs/_monocle/jlens/mm_textafter/half_odd/gemma4_12b_lens.pt` | `a6c35a1ce406` | 64 | 47 (0..46) | float32 | the odd shards of mm_textafter, 64 images | the question readout again; half 1 of its split-half variance estimate |

## A named lens that is not on disk

The wiki names these lenses, but no file holds them. Do not cite a result from one of them.

| Name | Path | Estimand |
|---|---|---|
| `ladder_R3` | `outputs/_monocle/jlens/ladder/R3/gemma4_12b_lens.pt` | the image-content positions; the faithful lens |

## The lens-series shard parts

A merge reads these. Do not read them as a lens. They are here so that a merge is reproducible.

| Name | sha256 | Prompts | Layers | Dtype | Step |
|---|---|---|---|---|---|
| `ladder_R0_shard0of8` | `721664ff66ff` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R0_shard1of8` | `61e205c7331f` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R0_shard2of8` | `b84ab03cbfbb` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R0_shard3of8` | `86814cbaee4b` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R0_shard4of8` | `65cde4c152d2` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R0_shard5of8` | `23300f5b4369` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R0_shard6of8` | `ebd21944dc5e` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R0_shard7of8` | `790be3804047` | 16 | 8 (6..46) | float16 | R0 |
| `ladder_R1_shard0of8` | `1e269ed65ee4` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R1_shard1of8` | `9de64a38e46a` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R1_shard2of8` | `042e050acda5` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R1_shard3of8` | `20e762e9038e` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R1_shard4of8` | `c87c51b540cb` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R1_shard5of8` | `0e41f644dcbf` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R1_shard6of8` | `8f226ea18415` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R1_shard7of8` | `6629a9c3d3af` | 16 | 8 (6..46) | float16 | R1 |
| `ladder_R2_shard0of8` | `b394e2c7506f` | 16 | 8 (6..46) | float16 | R2 |
| `ladder_R2_shard1of8` | `5d978c635322` | 16 | 8 (6..46) | float16 | R2 |
| `ladder_R2_shard2of8` | `b7ca3342aa6c` | 16 | 8 (6..46) | float16 | R2 |
| `ladder_R2_shard3of8` | `28b94104b98f` | 16 | 8 (6..46) | float16 | R2 |
| `ladder_R2_shard4of8` | `45a37acd7657` | 16 | 8 (6..46) | float16 | R2 |
| `ladder_R2_shard5of8` | `570070e40978` | 16 | 8 (6..46) | float16 | R2 |
| `ladder_R2_shard6of8` | `3c6fa05b8589` | 16 | 8 (6..46) | float16 | R2 |
| `ladder_R2_shard7of8` | `45d1d0e55d10` | 16 | 8 (6..46) | float16 | R2 |

## What we do not know

A file that a fit wrote before review item C7 has no `meta` key. For such a file the table above shows an annotation that this script holds, not a fact that the file carries. These items stay unknown:

| Item | Why it matters |
|---|---|
| The corpus, in the file | A second copy of the file cannot be traced back to its fit. |
| `dim_batch` | The batch changes the numerics. Two fits at different batches are not the same lens. |
| The fit jobs | We cannot name the SLURM jobs, the code commit, or the model revision. |

Files with no `meta` key:

- `outputs/_monocle/jlens/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/urban/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/mm/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/mm_all47/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/ladder/R0/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/ladder/R1/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/ladder/R2/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/lrp/gemma4_12b_lens.pt`
- `outputs/_monocle/jlens/mm/lrp/gemma4_12b_lens.pt`

A new fit writes the corpus, the estimator, `dim_batch`, and the date into the file's `meta` dict (review item C7). Read `meta` first. Use the annotation only when `meta` is absent.
