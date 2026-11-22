# The answer-word projection against the unit proxies

Date: 2026-09-03. Model: gemma-4-12b, the canonical pairwise runs
(`research/notebooks/cvpr/canonical_data/proxy/canonical/<case>__gemma-4-12b`).

The single-image activation-addition work (`steer-coord.md`) reads out the
judgment at the last token position as a projection on a vector from the
Jacobian lens. This page evaluates that readout with the same procedure as
the CVPR validation-by-proxy: the same units, the same proxies, the same
statistic. The question is whether a single-image readout from one forward
pass tracks the external proxies as well as the pairwise rating that the
paper validates.

## The readouts

| readout | definition |
|---|---|
| pairwise | the CVPR unit rating: the mean ordinal score over the comparisons of the unit, 5 or more (`_provenance.score_units`) |
| c | the single-image logit difference between the two answer words, positive answer word minus negative answer word, mean over the sampled images of the unit |
| contrast_L<l> | the centered projection of the residual stream at the last token position, layer l, on `pinv(J_l)·d`, with `d` the logit-difference direction; mean over the images of the unit |
| pos_L<l>, neg_L<l> | the same on the vector of one answer word alone: `pinv(J_l)` applied to the unembedding gradient of that answer word |

The answer-word vectors come from `monocle.pole_sample` (16 calibration
images per case). `monocle.pole_score` scores the sampled images in batches
of 8 (one forward pass, no padding, about 0.4 to 1.3 s per image).
`research/scripts/pole_score_sample.py` samples the units and images, and
`research/scripts/pole_score_join.py` runs the join and the bootstrap (400
resamples over units, the CVPR seed).

## The sample

| folder | units with a rating and a proxy | sampled | images (3 per unit) | proxy |
|---|---|---|---|---|
| restaurants | 12,151 | 1,500 | 4,396 | inspection score history |
| subway (station) | 419 | 419 | 1,257 | transit crime per rider |
| libraries | 228 | 228 | 682 | building violations, total |
| schools | 1,136 | 1,136 | 3,405 | building violations, total |
| parks | 1,054 | 1,054 | 3,153 | PIP acceptable rate |
| plazas | 395 | 395 | 1,184 | DOB and ECB violations |
| road (image level) | 382,176 images | 3,000 images | 3,000 | DOT pavement rating, nearest segment |

Street photography has no proxy and is not sampled. All proxies are
oriented "higher is better" by their own modules; the join applies no
sign.

## Result: Spearman rho vs the unit proxy, same units

| folder | pairwise | c | logit-difference projection, layer 42 | logit-difference projection, layer 45 |
|---|---|---|---|---|
| restaurants (1,500) | -0.09 [-0.14, -0.04] | -0.06 [-0.12, -0.01] | -0.05 [-0.10, -0.01] | -0.06 [-0.11, 0.00] |
| subway (419) | 0.21 [0.12, 0.29] | 0.14 [0.06, 0.23] | 0.06 [-0.04, 0.16] | 0.09 [0.00, 0.19] |
| libraries (228) | -0.37 [-0.48, -0.25] | -0.22 [-0.34, -0.09] | 0.02 [-0.12, 0.14] | 0.15 [0.02, 0.29] |
| schools (1,136) | 0.17 [0.11, 0.23] | 0.18 [0.12, 0.24] | 0.14 [0.08, 0.20] | 0.14 [0.08, 0.20] |
| parks (1,054) | -0.02 [-0.08, 0.03] | -0.01 [-0.07, 0.05] | -0.04 [-0.10, 0.02] | 0.00 [-0.06, 0.06] |
| plazas (395) | 0.16 [0.06, 0.24] | 0.19 [0.08, 0.29] | 0.20 [0.10, 0.29] | 0.00 [-0.10, 0.11] |
| road (3,000 images) | 0.09 [0.05, 0.12] | 0.23 [0.20, 0.27] | 0.23 [0.20, 0.27] | 0.18 [0.15, 0.22] |

Figure: `outputs/_monocle/pole_score/figures/proxy_forest.png`. Table:
`outputs/_monocle/pole_score/proxy_correlations.parquet`. Unit scores:
`outputs/_monocle/pole_score/unit_scores.parquet`.

How the readouts relate to the pairwise rating of the same units (Spearman):

| folder | c | logit-difference projection, layer 42 | logit-difference projection, layer 45 |
|---|---|---|---|
| restaurants | 0.74 | 0.65 | 0.74 |
| subway | 0.71 | 0.49 | 0.62 |
| libraries | 0.77 | 0.43 | -0.33 |
| schools | 0.66 | 0.57 | 0.43 |
| parks | 0.78 | 0.72 | 0.17 |
| plazas | 0.68 | 0.66 | -0.23 |
| road | 0.44 | 0.45 | 0.45 |

## What the table shows

1. **The single-image logit difference c matches the pairwise rating.**
   Against the proxy it is equal to or above the pairwise rating in
   schools, plazas and road, within the bootstrap interval in
   restaurants, subway and parks, and below it in libraries only. Road
   shows the largest advantage: one forward pass per image correlates
   with the pavement rating at 0.23, where the few comparisons of the
   image itself correlate at 0.09.
2. **The Jacobian-lens logit-difference projection at layer 42 matches
   the pairwise rating in 4 of the 6 folders with a proxy:** restaurants,
   schools, plazas and road. It fails in libraries (0.02) and subway
   (0.06), where its correlation with the judgment itself is weak (0.43
   and 0.49 against the pairwise rating).
3. **Layer 45 is not a general readout.** It matches layer 42 for
   restaurants and schools, exceeds it for subway, and fails for plazas
   and libraries, where its projection correlates negatively with the
   judgment.
4. **The vector of one answer word alone does not read out polarity.** In
   libraries the 'pos' and 'neg' projections both correlate positively
   with the proxy (0.26 and 0.29); in subway both negatively. The two
   answer-word vectors share a component that tracks answer strength, and
   only their difference tracks polarity. Use the logit-difference
   vector.
5. **Parks shows no association for any readout**, as for the published
   pairwise number.

## What this establishes

The interpretability argument holds in this form: the pairwise judgment
is a scalar that one forward pass reads out at the last token position,
and that readout tracks the external proxies as well as the pairwise
ratings do. The Jacobian-lens projection at layer 42 is that readout for
4 of 6 folders, and it is the same coordinate that the steering vectors
change. Where the projection fails, the logit difference c still tracks
the proxy, so the failure is in the Jacobian-lens readout for those
cases, not in the single-image prompt template.

## Repeat

```bash
sbatch research/monocle/monocle.sub monocle.pole_sample --case <case>          # pole vectors, 60 eval images
cd research/notebooks/cvpr && ../../.venv-3.12/bin/python ../../scripts/pole_score_sample.py
sbatch research/monocle/monocle.sub monocle.pole_score --case <case> --shard <i> --n-shards <n>
.venv-3.12/bin/python research/scripts/pole_score_join.py --cases all
.venv-3.12/bin/python research/scripts/pole_score_figure.py
```

Warning: run the sampler from `research/notebooks/cvpr`, it imports
`_unit_validation` by path. Delete a shard from a test run before the
join: the join globs every `scores_shard*.parquet` in the case directory.

## In the paper

The CVPR draft holds the readouts in methods 3.6, results 4.3, and
`tables/single_image_read.tex`. The proxy overview figure draws the
layer-42 Jacobian-lens readout as its own marker at the community
district; see `proxy-overview.md`, section "The single-image series".

## Related

- `steer-coord.md`, parts 2 to 6: the vectors, the activation addition,
  the image tokens.
- `outputs/_monocle/pole_sample/<case>/`: the answer-word vectors, the
  60-image projections, the contact sheets (`figures/poles_L42.png`,
  `poles_L45.png`), and the per-unit TrueSkill join
  (`per_unit_join_L<l>.parquet`, `per_unit_report.txt`).
