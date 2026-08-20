---
title: "The Self-Explanation Ladder — Can a VLM Re-Derive Its Own Question?"
category: concept
created: 2026-07-12
updated: 2026-07-12
tags:
  - concept
  - pairwise
  - gepa
  - interpretability
  - self-distillation
  - prompt-optimization
---

# The Self-Explanation Ladder

A black-box interpretability experiment on the [[urban-pair-vqa]] judges. **Gemma-4-12B is simultaneously the task model, the reflection model, and the label source**: we hold out its own production labels on a case and let [GEPA](https://github.com/gepa-ai/gepa) search for the prompt that best reproduces them. The question: **how much of its own question can the model reconstruct from its own behavior?**

The experiment is an **ablation over what the reflector is allowed to see** — an escalating "semantics channel." Each rung opens one more channel and asks whether the recovered prompt finally names the *judgment* (safety / appeal) rather than a visual proxy.

Companion result to the Reviewer-2 robustness battery (see [[urban-pair-vqa]]); the sequel that settles its meaning is [[concept-activation-probe]].

## Setup

| Element | Value |
|---|---|
| Harness | `dagspaces/urbanpairvqa/prompt_opt/gepa_pairwise.py` (in-process vLLM, `.venv-nightly`, klara) |
| Launcher | `scripts/gepa_pairwise_subway.sub` |
| Supervision | the model's own production labels, **presented orientation**, `NotSure` dropped |
| Split | 2,000 train / 1,000 val, stratified, disjoint |
| Metric | ordinal agreement `1 − |Δ| / 4` on the 5-point scale (exact-match tracked alongside) |
| Reflective feedback | **strictly task-neutral** (expected label, predicted label, ordinal distance — nothing else); leak-asserted in `tests/test_gepa_pairwise.py` |
| Fixed anchor | the "Image A relative to Image B" suffix stays outside the evolvable candidate |

**Axis-slot mode** (`--candidate-mode axis`) freezes the calibration scaffold — polarity/flip discipline, Same-suppression, conservative `Much*` — that blind search kept rediscovering, and evolves **only the attribute phrase** in *"Which photograph looks more `X`?"*. What the search recovers is then a single interpretable phrase instead of a paragraph.

**Brackets.** Every run sits between two references: a **constant baseline** (always predict the majority direction) and a **true-axis ceiling** (drop the case's real attribute phrase into the identical scaffold).

## The rungs

| # | Channel the reflector sees | Flag |
|---|---|---|
| 1 | the *real* production prompt as seed (control) | `--seed-mode yaml` |
| 2 | **labels only** — blind seed "Compare the two images." | `--seed-mode generic` |
| 3 | + the model's own **factual captions** (comparison forbidden) | `--reflection-descriptions` |
| 4 | + **contrastive captions** (comparatives allowed, verdict forbidden) | `--caption-style contrastive` |
| 5 | + **the mispredicted image pairs themselves** | `--reflection-images` |

## Results (ordinal, best-by-val)

| rung | subway | restaurants | schools |
|---|---|---|---|
| 2 — blind (labels only) | 0.804 | 0.720 | 0.738 |
| 3 — factual captions | 0.793 | 0.799 | 0.748 |
| 4a — contrastive, free prompt | **0.863** | 0.776 | 0.761 |
| 4b — **axis-slot** (captions) | 0.825 | 0.825 | 0.810 |
| 5 — **axis-slot (IMAGES)** | 0.820 | 0.795 | 0.799 |
| *always-"More" constant* | *0.823* | *0.775* | *0.728* |
| **true-axis ceiling** | **0.895** | **0.875** | **0.889** |

Recovered axis phrases:

| case | from captions (rung 4b) | from **images** (rung 5) |
|---|---|---|
| subway safety | "open and airy urban atmosphere" | **"brightly lit"** |
| restaurant appeal | "pedestrian-oriented commercial activity" | **"commercial activity"** |
| school appeal | "educational or institutional campus setting" | **"architectural style of the buildings"** |

## Findings

1. **The production prompt is locally optimal.** Seeded with the real subway prompt, 94 iterations never beat it (0.948 ord / 0.857 exact). The 0.857 is the temperature-0.6 noise floor in the targets.
2. **Style is recoverable; semantics are not.** From a blind seed, exact-match climbs 7% → 41–55% — but every recovered prompt encodes only *response geometry*: A-vs-B directionality discipline, Same-suppression, conservative `Much*`. **None mentions safety, food, schools, or upkeep.**
3. **Captions pass the domain through, never the judgment.** Storefronts, "Public Library," construction sites all appear. The *question asked about them* does not.
4. **The axis-slot names visual proxies, never the evaluative concept** — openness, commercial activity, campus presence. Real correlates of the judgment; not the judgment.
5. **Rung 5 is a clean negative, and it is the headline.** Showing the reflector the actual photographs **does not help** (flat-to-worse: −0.005 / −0.030 / −0.011). The axes get **more literal, not more evaluative** — "open and airy urban atmosphere" collapses to bare "brightly lit," and schools *loses* the domain captions had found. Mid-search the reflector proposes what it can literally *count*: "number of parked cars," "building height," "amount of overhead structure."
6. **The invariant.** Across all four channels — its own labels, factual captions, contrastive captions, and the pixels themselves — recovery plateaus **0.05–0.09 ordinal below the true-axis ceiling** and the evaluative concept is **never named**. The text bottleneck was never the binding constraint.

> A "safe" rating is really a *brightness-and-openness* rating; an "appealing restaurant" rating is really a *commercial-activity* rating — and the model cannot articulate the substitution **even while looking at the photographs it judged**.

**Crucially, this is a limit on *articulation*, not on *representation*** — see [[concept-activation-probe]], which recovers the safety judgment from the same forward pass with a linear probe.

## Verification performed

- **0 meta-prompts** across 76/73/68 axis proposals (the custom `AXIS_REFLECTION_TEMPLATE` fix held — gepa's default template made the reflector propose *prompts that generate phrases*, testing zero actual axes).
- **0 verdict leakage** in contrastive captions (the "safety" hits are construction netting/barriers — object descriptions, not judgments).
- Harness smoke-tested end-to-end (oracle stub = 1.0); leak assertions on feedback and caption-prompt text; 28 tests in `tests/test_gepa_pairwise.py`.

## Gotchas

> [!warning] `[val-eval]` is not the returned best — this put wrong numbers in the paper once.
> `evaluate()` logs a `[val-eval]` line for **every** full-val candidate evaluation, so the **last line is the last candidate tried, not the winner**. Read `max(val_aggregate_scores)` from `metrics.json` (task evals are temp-0, hence deterministic). The harness now also writes `best_val_ordinal` so this cannot recur.

> [!note] Multimodal reflection plumbing
> `make_reflective_dataset` embeds `gepa.Image(path=…)`; gepa swaps in `[IMAGE-N]` placeholders and calls `reflection_lm` with an OpenAI multimodal **list**, not a `str`. `reflect()` therefore accepts `str | list`, decodes the base64 data-URIs back to PIL, renders N `<|image|>` blocks (text first), and passes `multi_modal_data`. `limit_mm_per_prompt` is auto-raised to `2 × minibatch`. Mutually exclusive with `--reflection-descriptions`.

## Artifacts

| What | Where |
|---|---|
| Run dirs | `outputs/gepa_pairwise/*_gemma12b_{generic,generic_desc,generic_contrastive,axis_contrastive,axis_multimodal}_*/` |
| Ceilings / constants | `outputs/gepa_pairwise/probe_20260711_1517/` |
| Sweep summary | `outputs/gepa_pairwise/SWEEP_SUMMARY_generic_seed_20260711.md` |
| One-pager | `machine-beholder/ONE_PAGER_gepa_prompt_retranslation.md` |
| Paper | `papers/cvpr27_machine-perception-public-spaces` — methods `sec:self-reflection`, results `sec:results-selfreflection` + `tab:selfreflection` |

## See also

- [[concept-activation-probe]] — the sequel: is the judgment *represented* even though it cannot be *said*? (Yes.)
- [[urban-pair-vqa]] — the dagspace and the case prompts
- [[concept-counterbalancing]] — presented vs canonical orientation (why supervision uses `presented_label`)
