# Judge vocabulary (v3)

## The question

Which keys does 1 judge use that the other judge does not, in each case?

- Judges: Gemma-4-12B and Qwen3.5-9B.
- Cases: subway_safety, libraries, schools, road_quality, parks_plazas, restaurants, street_photography.
- Classes of the version 3 schema: `dimension` (the attribute observed), `inference` (the attribute evaluated), `perspective` (the mode of valuation).
- A key is the normalised free `name` attribute of a span.

## The confound

An extractor model writes the name when it reads the trace of the judge. The
judge does not write it. The 2 extractors (Qwen3.8-27B, Gemma-4-31B) differ in
vocabulary and in volume. Thus a contrast of the 2 primary arms mixes the judge
and the extractor.

The design holds the extractor fixed. It compares the 2 judges in each row of
the table below, then keeps the contrasts that hold in both rows.

| Extractor | Gemma-4-12B traces | Qwen3.5-9B traces |
|-----------|--------------------|-------------------|
| Qwen3.8-27B | `gemma_by_qwen27_v3` | `qwen_by_qwen27_v3` |
| Gemma-4-31B | `gemma_by_gemma31_v3` | `qwen_by_gemma31_v3` |

A key that is distinctive of 1 judge under both extractors is a fact about the
judge. A key that is distinctive under 1 extractor only is not.

## The method

| Step | Rule |
|------|------|
| Load | `X.load(root=..., require_canonical=True)`, the paper filters. A trace is `(case, pair_id)`. |
| Normalise | `_ic_v3.normalise_names` (thus `_cues_across.normalise_cue`): case, punctuation, hyphen, article, degree adverb, plural. No synonym list. |
| Count | The document count: the traces that use the key 1 time or more. The denominator is `X.trace_counts` (every answered trace). |
| Contrast | The weighted log-odds ratio of Monroe, Colaresi and Quinn (2008), with an informative Dirichlet prior equal to the pooled counts of both judges in the case and extractor. The z is positive for a Gemma-4-12B key. |
| Floor | A key needs `min_traces` (default 10) traces of the 2 judges together under an extractor. |
| Robust | 1 sign and \|z\| >= 1.96 under both extractors, and the floor under both. |
| Agreement | The Spearman of the 2 extractors' z, over the keys that both extractors see, per case and class. |
| Cross case | The number of cases in which a key is robust. `general` is 3 cases or more. |

**Warning: the z compares shares of each judge's vocabulary, not per-trace
rates.** The qwen traces hold more distinct keys in 1 trace than the gemma
traces (the measured ratio is in the table below). Thus a key that both judges
use in almost every trace can be a larger share of the gemma vocabulary. Such a
key takes a positive z, but its per-trace rate is lower for gemma. The column
`rate_agrees` and a dagger in the figures and tables mark this case.

| Class | Ratio of distinct keys in 1 trace, qwen / gemma (Qwen3.8-27B; Gemma-4-31B) |
|-------|------------------------------------------------------------------------|
| dimension | 1.49; 1.51 |
| inference | 2.10; 1.85 |
| perspective | 1.38; 1.39 |

**Warning: the z has no correction for the number of keys.**

## The headline numbers (2026-10-01, `min_traces` = 10)

| Class | Significant pairs, Qwen3.8-27B | Significant pairs, Gemma-4-31B | Robust Gemma (rate agrees) | Robust Qwen (rate agrees) | Survive | Median Spearman (range) |
|-------|-----|-----|-----|-----|-----|-----|
| dimension | 98 | 64 | 15 (3) | 25 (25) | 0.33 | 0.80 (0.73 to 0.90) |
| inference | 230 | 94 | 29 (7) | 32 (32) | 0.23 | 0.69 (0.53 to 0.81) |
| perspective | 158 | 63 | 22 (14) | 22 (22) | 0.25 | 0.68 (0.54 to 0.85) |

A "pair" is a (case, key). `Survive` is the robust pairs over the pairs that
are significant under 1 extractor or more. Thus about 1 in 4 of the
single-extractor contrasts holds under the swap.

The robust keys in 3 cases or more:

| Class | Gemma-4-12B | Qwen3.5-9B |
|-------|-------------|------------|
| dimension | architecture (4), signage (3); both have the lower per-trace rate | lighting (6), building fixture (4), other (3) |
| inference | standardness (6), maintenance (4), aesthetic sentiment (3), modernity (3); all have the lower per-trace rate | location (6), architecture (5), vehicle type (3) |
| perspective | nyc baseline (6; the rate agrees in 5), maintenance (4; lower rate), modern (3) | utilitarian (4), gritty (3) |

Read the result as follows. Every robust qwen key is also more frequent per
trace. Most robust gemma keys are shared keys (for example `maintenance`) that
fill a larger share of a smaller gemma vocabulary. The gemma keys that are
also more frequent per trace are mostly perspectives (`nyc baseline`,
`scenic`, `high end`).

Strict uniqueness: 289 (case, class, extractor, key) rows have 10 traces or
more for 1 judge and 0 for the other. Only 13 keys are strictly unique to the
same judge under both extractors (for example qwen `brutalist` in libraries,
qwen `socioeconomic status` in schools, gemma `quaint` in street photography).

## The files

Module: `notebooks/cvpr/_judge_vocab.py`. Notebook:
`notebooks/cvpr/master/judge_vocabulary.py`. Headless export:
`scripts/export_judge_vocab.py`. Tests: `tests/test_judge_vocab.py`.

All outputs are in `notebooks/cvpr/master/outputs/judge_vocab/`:

| File | Label | Holds |
|------|-------|-------|
| `judge_vocab_<class>.png`, `.pdf`, `.json` | `fig:judge-vocab-<class>` | The top k keys of each judge in each case, 1 mark for each extractor; a filled mark is robust |
| `judge_vocab_<class>_keys_v1.0.0.tex` | `tab:judge-vocab-<class>` | The top 5 robust keys of each judge in each case, with the document frequencies |
| `judge_vocab_agreement_v1.0.0.tex` | `tab:judge-vocab-agreement` | The Spearman and the robust counts per case and class |
| `judge_vocab_{scores,robust,unique,agreement,cross_case,summary,breadth}_v1.0.0.csv` | none | The full tables |

The `.json` sidecar comes from `_figmeta.save_figure`. It holds the 4 caption
keys and a `provenance` block.
