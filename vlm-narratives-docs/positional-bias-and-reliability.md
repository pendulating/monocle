# Positional bias and test-retest reliability

The pairwise battery carries 2 sources of error that a proxy correlation
cannot see. The first is the side: a model can favour the first image no
matter what it shows. The second is the repeat: the same 2 places, judged
again, can get a different label. This page says how the battery controls
each one, how we measure it, and what the canonical runs show.

## Where the code is

| File | Purpose |
|------|---------|
| `research/notebooks/cvpr/_reliability.py` | The 2 measures, the cache, and the 3 LaTeX tables |
| `research/notebooks/cvpr/master/reliability.py` | The marimo notebook over every canonical run |
| `research/notebooks/cvpr/master/outputs/positional_bias_v1.0.0.tex` | The bias table, 5 columns |
| `research/notebooks/cvpr/master/outputs/test_retest_v1.0.0.tex` | The reliability table, 5 columns |
| `research/notebooks/cvpr/master/outputs/order_effects_v1.0.0.tex` | Both in 1 table, 7 columns |
| `research/notebooks/cvpr/master/outputs/*_v1.0.0.csv` | The long frames, every measure |
| `research/notebooks/cvpr/master/outputs/reliability_cache/` | 1 JSON file for each run, named by the run hash |

The command below writes the same files without marimo. The registry gate
runs first.

```bash
.venv-vllm025cu129/bin/python research/notebooks/cvpr/_reliability.py
```

## What removes the positional bias

The 50% swap removes it, not the repeats.

1. The sampler runs with `counterbalance_mode: balanced`. It swaps the
   presented order on every second observation, by the parity of the
   observation index. See `dagspaces/pairwisevqa/samplers/cyclomedia_pairs.py`.
2. The row records the swap in `is_swapped` and `presented_order`.
3. The stage maps the presented label to its mirror on a swapped row, and
   writes the result to `relative_label`. "More" becomes "Less", and "Same"
   and "NotSure" do not change. See `dagspaces/pairwisevqa/stages/pairwise_vqa.py`.
4. Every proxy score reads `relative_label` only.

A left-side preference adds a positive shift to half the rows and a negative
shift to the other half. In the unit mean, the 2 halves cancel. The bias
cancels in the population, not inside each pair.

## What the 10% repeats do

`repeat_fraction: 0.10` draws 100,000 more observations of canonical pairs
the battery already holds. A repeat gets its swap state from the same parity
rule, thus about half the repeats come in the same order and half in the
opposite order. The sampler does not force an inverted duplicate.

The repeats measure reliability. The 2 halves answer 2 questions:

| Half | Facility case | Image case |
|------|---------------|------------|
| same order | fresh photographs, same side | same photographs, same side: decoding stability |
| opposite order | fresh photographs, other side | same photographs, other side: order sensitivity |

A facility case draws fresh photographs of both units on each observation.
Only 0.2% to 1.6% of its repeats show the same 2 images. Road quality and
street photography are image cases, thus 100% of their repeats show the
identical 2 images.

**Warning:** do not write that the repeats compensate for the positional
bias. They do not. A repeat enters the unit mean as an ordinary row.

## The measures

| Column | Meaning | No effect |
|--------|---------|-----------|
| first | Share of decided judgments that favour the first image | 0.50 |
| s-bar | Mean score on the -2 to +2 scale, after the de-swap | 0 |
| same | Sign agreement of a repeat that kept the order | 1 |
| swap | Sign agreement of a repeat that swapped the sides | 1 |

A decided judgment is any label other than Same or NotSure. A sign agreement
keeps the repeats where both judgments are decided and counts the share that
name the same winner. 0.5 is chance there, and a value below 0.5 means the
side beats the content. The long CSV files also hold the 6-label exact
agreement, Cohen kappa, the Spearman correlation of the scores, and the
abstention agreement.

Two floors run before a table prints. A bias cell needs 10,000 decided
judgments, and a reliability cell needs 1,000 decided repeats. Schools with
qwen3.5-9b abstains on 99.7% of the rows, thus both floors dash it.

## What the canonical runs show, 2026-09-02

Every cell holds 1,100,000 rows, 50% swapped, at temperature 0.0.

| Case | Model | first | s-bar | same | swap |
|------|-------|-------|-------|------|------|
| Subway safety | gemma-4-12b | 0.73 | +0.001 | 0.87 | 0.61 |
| Subway safety | qwen3.5-9b | 0.87 | -0.001 | 0.85 | 0.24 |
| Libraries | gemma-4-12b | 0.60 | +0.001 | 0.80 | 0.75 |
| Libraries | qwen3.5-9b | 0.75 | -0.001 | 0.74 | 0.41 |
| Schools | gemma-4-12b | 0.68 | +0.001 | 0.98 | 0.96 |
| Schools | qwen3.5-9b | dash | dash | dash | dash |
| Road quality | gemma-4-12b | 0.46 | -0.002 | 0.97 | 0.70 |
| Road quality | qwen3.5-9b | 0.69 | +0.001 | 0.97 | 0.44 |
| Parks / Plazas | gemma-4-12b | 0.53 | 0.000 | 0.80 | 0.75 |
| Parks / Plazas | qwen3.5-9b | 0.47 | +0.005 | 0.67 | 0.54 |
| Restaurants | gemma-4-12b | 0.67 | 0.000 | 0.78 | 0.64 |
| Restaurants | qwen3.5-9b | 0.99 | +0.004 | 0.98 | 0.04 |
| Street photography | gemma-4-12b | 0.75 | -0.001 | 0.99 | 0.50 |
| Street photography | qwen3.5-9b | 0.71 | -0.001 | 0.98 | 0.55 |

Three points follow.

- **The bias is real and the swap removes it.** Both models favour the
  first image in 11 of 13 printed cells. Qwen picks the first image in 99%
  of its decided restaurant judgments. After the de-swap, the mean sits
  within 0.005 of zero in every printed cell.
- **Greedy decoding is not fully stable.** On the same 2 images in the same
  order, 1% to 3% of decided verdicts change. The temperature is 0.0, thus
  the batch composition in vLLM causes it.
- **At the level of 1 pair, the side often beats the content.** When the
  same 2 images swap sides, gemma keeps its street photography verdict 50%
  of the time, which is chance. Qwen keeps its restaurant verdict 4% of the
  time. The proxy correlations survive because each unit or polygon rests on
  hundreds to thousands of comparisons, and the positional pull cancels
  across them. Do not present a single pairwise verdict as a measurement.

## Related pages

- `million-pair-battery.md` for the shards and the repeat cut.
- `canonical-run-registry.md` for the gate the notebook runs first.
- `proxy-arms.md` for the ablation arms, which the notebook can also read.
