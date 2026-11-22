# Robustness battery on the canonical setup

The CVPR appendix asks whether the subway-safety signal is a property of the
model or of our elicitation. The July 2026 battery answered that for a prompt
the paper no longer uses. This page describes the re-run of 2026-09-04 on
the canonical setup. The plan and its checklist are in
`robustness-battery-rerun-plan.md`.

## Where the code is

| File | Purpose |
|------|---------|
| `dagspaces/pairwisevqa/conf/sweep/robustness_1k_<rater>.yaml` | The base of 1 rater: 1,000 pairs, 1 shard, the rater's production layout |
| `dagspaces/pairwisevqa/conf/prompt/pairwise_subway_safety_{paraphrase,persona,enumrev,flipped}.yaml` | The 4 wording arms, each 1 change against the canonical prompt |
| `scripts/launch_robustness_battery.sh` | One submit for each (rater, arm); tier A, tier B, and the smoke mode |
| `scripts/subset_pair_tables.py` | Cuts the 1k and 25k pair tables out of the 1M table |
| `notebooks/cvpr/_robustness.py` | The measures, the proof of each arm, and the 3 LaTeX tables |
| `notebooks/cvpr/_robustness_estimand.py` | Tier B: the station and tract proxies of each arm against the matched canonical control, and the order halves |
| `notebooks/cvpr/master/outputs/robustness_v1.0.0.{tex,csv}` | The table the paper reads |
| `notebooks/cvpr/master/outputs/robustness_estimand*_v1.0.0.csv` | The numbers of the estimand paragraph |

## The design

| Item | Value |
|------|-------|
| Case | Subway safety |
| Raters | gemma-4-12b, qwen3.5-9b, qwen3.5-2b, gemma-4-e2b |
| Pairs | 1,000 canonical pairs (1,091 rows with repeats), rows of the 1M table |
| Base prompt | `pairwise_subway_safety_ordinal`, no system turn, abstention on |
| Decoding | Greedy, `max_tokens` 128 |
| Images | `runtime.image_pack` of the 1M table |

The 1,000 pairs are the first 1,000 canonical pairs of the 100k draw, which
is nested in the 1M draw. They are the same station pairs the July battery
scored. The images differ from July, because the 1M draw took other faces.

### Arms

Every arm is one change against the rater's own base.

| Id | Arm | Override |
|----|-----|----------|
| R0 | retest | none |
| W1 | paraphrase | `prompt=pairwise_subway_safety_paraphrase` |
| W2 | persona added | `prompt=pairwise_subway_safety_persona` |
| W3 | option order reversed | `prompt=pairwise_subway_safety_enumrev` |
| W4 | negated question | `prompt=pairwise_subway_safety_flipped`; the analysis mirrors the labels |
| W5 | city name removed | `prompt=pairwise_subway_safety_nocity` |
| F1 | halved image budget | qwen: `max_pixels=262144`; gemma: `max_soft_tokens=140` |
| F2 | free decoding | `prompt.structured_output.enabled=false` |
| F3 | temperature 0.6 | `sampling_params_vqa.temperature=0.6` |
| L1 | the other layout | both gemma: `images_then_labels`; both qwen: `interleaved_labels` |
| L2 | text first | both gemma: `text_first_interleaved`; both qwen: `text_first` |

### The 2 anchors

Decoding is greedy, thus a retest at the same temperature is not a noise
ceiling any more. The battery reads 2 anchors:

| Anchor | What it is | Who has it |
|--------|------------|------------|
| retest (R0) | the base config, run again in the battery | every rater |
| canonical | the registered 1M run, on the same `pair_id` rows | gemma-4-12b, qwen3.5-9b |

R0 against the canonical run reads the batch-effect floor: the same prompt
on the same images in another batch. Every arm reads against R0.

## Commands

```bash
# The pair tables (done once)
python scripts/subset_pair_tables.py --parent multirun/pair_tables_1m \
    --select multirun/pair_tables_select_1k --out-dir multirun/pair_tables_1k_of_1m --cases subway_safety

# Tier A: 4 raters x 11 arms, 1 job each
bash scripts/launch_robustness_battery.sh --dry-run
bash scripts/launch_robustness_battery.sh --base multirun/robustness_a_<stamp>

# Tier B: paraphrase + persona at 25,000 pairs, 2 raters, 2 shards each
bash scripts/launch_robustness_battery.sh --tier b

# The tables
.venv-mllmsci-vllm025cu129/bin/python notebooks/cvpr/_robustness.py --battery multirun/robustness_a_<stamp>
```

## Traps

- **The shard submitter records only the base overrides.** The
  `.hydra/overrides.yaml` of a run does not name the arm. `_robustness.py`
  reads `.hydra/config.yaml` and refuses a run whose prompt text, layout,
  temperature, or image budget does not match its arm name.
- **A cut table serves 1 `max_pairs` only.** The runner accepts the cut when
  `pair_sampler.max_pairs` equals the number of canonical pairs it kept. A
  smoke test with 20 pairs must run without `--pairs-dir`.
- **Use `++` for a key the base config may not hold.** `prompt.image_layout`
  exists in the gemma-4-12b sweep and not in the others; `max_soft_tokens`
  exists in no model config.
- **The working tree edit to `million_proxy_gemma12b.yaml` said
  `images_then_labels`.** The registry and the decision of 2026-09-02 say
  `interleaved_labels`. The line was corrected on 2026-09-04. Read a run's
  `.hydra/config.yaml`, never a sweep file.
- **gemma-4-e2b does not bind 2 adjacent images either.** The 20-pair smoke
  of 2026-09-04 gave Same on 100% of the rows under the default layout and
  on 91% under `text_first`, and a spread under `interleaved_labels`. The
  100-pair probe over 5 layouts confirmed it: default 100% Same,
  `images_then_labels` 99% Same, `text_first` 92% Same,
  `text_first_interleaved` 54% Same + 45% NotSure, `interleaved_labels` 70%
  Same with a spread. Only `interleaved_labels` binds on e2b, thus its base
  is `interleaved_labels` and it takes the 12b layout arms. Its L1 arm
  (`images_then_labels`) is thus a read of a layout that does not bind.

- **The merge recomposed the config and lost the arm.** `write_hydra_record`
  in `scripts/merge_pairwise_shards.py` composed the record from the sweep,
  the model, and the pipeline, with no `--set` override. The merged
  paraphrase and persona runs thus recorded the canonical prompt, the
  registry classed them `canonical`, and it dropped them under the row
  floor without a word. Since 2026-09-04 the merge copies the
  `.hydra/config.yaml` of the shards (they must agree on the prompt
  section) and composes only when no shard holds one. After a merge, read
  the prompt in the merged record before you register.

## Results (tier A, 2026-09-04, `multirun/robustness_a_20260904_210129`)

44 of 44 runs landed, 1,091 rows each, and every run proved its arm. The
generated tables are in `notebooks/cvpr/master/outputs/robustness*_v1.0.0.tex`
and in the paper under `tables/robustness*.tex`.

Linear-weighted kappa against the retest (R0). The first row is R0 against
the registered 1M run on the same rows.

| Arm | 12B | 9B | 2B | E2B |
|-----|-----|----|----|-----|
| Retest vs canonical (floor) | 0.96 | 0.94 | -- | -- |
| Paraphrase | 0.61 | 0.44 | 0.98 | 0.13 |
| Persona added | 0.57 | 0.39 | 0.98 | 0.19 |
| Option-order rev. | 0.89 | 0.67 | 0.85 | 0.28 |
| City name removed | 0.91 | 0.84 | 0.96 | 0.68 |
| Halved image budget | 0.75 | 0.41 | 0.98 | 0.21 |
| Free decoding | 0.83 | 0.65 | -0.02 | 0.10 |
| Temperature 0.6 | 0.87 | 0.36 | 0.59 | 0.66 |
| Other layout | 0.68 | 0.53 | 0.94 | 0.05 |
| Negated question | 0.25 | 0.52 | -0.97 | 0.30 |
| Text-first layout | 0.61 | 0.36 | 0.77 | 0.08 |

What the numbers say:

- **Greedy decoding has a floor.** The same prompt on the same images in
  another batch agrees at 0.96 (12B) and 0.94 (9B). Read every arm against
  that, not against 1.0.
- **The 12B rater is stable under option order, city name, free decoding,
  and temperature** (0.83 or more). A paraphrase or a persona costs it more
  (0.61, 0.57): both cut its Same share from 29% to about 8%, and the persona
  removes its abstention (6% to 0.1%).
- **The 9B rater is less stable.** It holds under the city name, option order,
  and free decoding (0.65 to 0.84). Wording, budget, temperature, and both
  layouts move it to 0.53 or less. Several arms move its side bias too: the
  paraphrase lowers it from 84% to 54%, free decoding raises it to 100%, and
  text-first to 95%.
- **The negated question breaks every rater.** The 12B keeps its polarity but
  answers Same on 52% of pairs (29% at base). The 9B answers with the second
  image on 100% of decided pairs. The 2B gives the same presented label as
  under the positive question, thus its mirrored labels anti-correlate
  (-0.97).
- **qwen3.5-2b is a function of the side, not the images.** It favours the
  second image on at least 98% of its decided judgments in every arm except
  free decoding (44% of its answers name no label) and temperature 0.6. Its
  0.98 kappas on the wording arms are vacuous. The July battery could not see
  this: at temperature 0.6 the side lock was diluted by sampling.
- **gemma-4-e2b binds 2 images only under `interleaved_labels`**, and even
  then its judgments change under 8 of 10 arms (kappa 0.30 or less), with a
  side bias that swings between 0% and 96% from arm to arm.

Differences from the July table that the reader must know:

| Item | July | Now |
|------|------|-----|
| 12B paraphrase / persona | 0.81 / 0.81 | 0.61 / 0.57 |
| 12B negated question | 0.54 | 0.25 |
| 9B wording arms | 0.52 to 0.55 | 0.39 to 0.67 |
| 2B | noisy but image-driven | side-locked |
| Anchor | temp-0.6 retest, temp-0 ceiling | greedy retest, batch floor |

The July prompt carried a persona and a cue list, and its temperature 0.6
hid the side lock of the 2B. The minimal canonical prompt is more sensitive
to wording on both large raters. The kappa alone cannot tell a wording
sensitivity from a side-bias shift, thus the paper prints the side-bias
table beside the agreement table.

## Tier B (2026-09-04, `multirun/robustness_b_20260904_212400`)

Paraphrase and persona at 25,000 pairs (27,442 rows with repeats), gemma-4-12b
and qwen3.5-9b, 2 shards each. Merged to
`multirun/robustness_b_merged/{paraphrase,persona}/subway_safety__<model>` and
registered as the arms `paraphrase` and `persona` (81 runs, verify OK). The
control of each arm is the canonical 1M run on the same `pair_id` rows.

```bash
# merge (one call for each of the 4 runs)
python scripts/merge_pairwise_shards.py --sweep-dir multirun/robustness_b_20260904_212400/gemma12b/W1/runs \
    --sweep robustness_1k_gemma12b --out-dir multirun/robustness_b_merged/paraphrase
# register: give EVERY root (see canonical-run-registry.md), then verify
# the estimand numbers
cd notebooks/cvpr && ../../.venv-mllmsci-vllm025cu129/bin/python _robustness_estimand.py
```

Abstention: gemma paraphrase 10.8%, persona 0.1%; qwen paraphrase 12.2%,
persona 1.5%.

| Read | Rater | Paraphrase / control | Persona / control |
|------|-------|----------------------|-------------------|
| Station rho, transit crime per rider (419 stations) | 12B | 0.20 / 0.20 | 0.25 / 0.20 |
| | 9B | 0.23 / 0.23 | 0.17 / 0.23 |
| Tract r, median household income (642 tracts) | 12B | 0.39 / 0.39 | 0.47 / 0.39 |
| | 9B | 0.41 / 0.35 | 0.22 / 0.35 |
| Tract r, crime density (664 tracts) | 12B | -0.10 / -0.13 | -0.21 / -0.13 |
| | 9B | -0.27 / -0.21 | -0.11 / -0.21 |
| Tract-score correlation, arm against control | 12B | 0.955 | 0.934 |
| | 9B | 0.750 | 0.637 |

Every station and tract coupling keeps its sign; the station reads hold at
p < 0.001 and the crime-density reads at p <= 0.03. The persona moves the
two raters in opposite directions: it sharpens the 12B and blunts the 9B.

Order halves of the canonical 1M run (550,000 rows each): tract income r
0.421 / 0.423 (12B) and 0.458 / 0.454 (9B); the two half tract scores
correlate 0.995 (12B) and 0.968 (9B).
