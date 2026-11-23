# Gemma-4 best practices: what we follow

Date: 2026-08-20. Scope: the `pairwisevqa` dagspace and the `monocle` lens
study. Source: the Gemma 4 model card, "Best Practices".

## The audit

| Card item | Our path | Status |
|---|---|---|
| 1. temp 1.0, top_p 0.95, top_k 64 | greedy for the label battery; the card values for thinking runs | deliberate deviation |
| 2. Thinking-mode control tokens | `thinking_mode` → `chat_template_kwargs` → the model's own template | correct |
| 3. No thought text in the history | every prompt holds one user turn | not applicable |
| 4. Image before the text | qwen: yes. gemma-4-12b: a text anchor comes first | deviation, measured |
| 5. Visual token budget | the default 280 cap, never tuned | measured 2026-08-20 |
| 6. Audio prompt shapes | no audio in these dagspaces | not applicable |
| 7. Audio and video length | still images only | not applicable |

### Sampling: why the battery is greedy

The label battery runs `temperature=0.0`. The answer is one of six enum tokens
under guided decoding, thus sampling adds noise and gives nothing back: at the
old `temperature=0.6`, qwen3.5-9b reproduced only 39% of its own labels on an
identical prompt. See the note in `conf/config.yaml`.

The thinking sweeps DO follow the card, because the thought channel is free
text and carries no grammar. Three sweeps missed `top_k=64` before 2026-08-20:
`thinking_gemma12b`, `thinking_street_photography_1k`, and
`thinking_street_subway_10k`. They now set it. The runs of 2026-07-11 sampled
the thought channel from the full vocabulary.

### Thinking mode

The contract needs no special code. `resolve_thinking_mode` reads
`model.thinking_mode`, and `model.chat_template_kwargs` reach
`apply_chat_template` and `llm.chat`. The model's own `chat_template.jinja`
then does what the card describes:

| Condition | What the template writes |
|---|---|
| thinking on | `<\|think\|>` at the top of a system turn, which it creates |
| thinking off, 12B and 31B | an empty thought block before the answer |
| thinking off, E2B and E4B | nothing — as the card says |

`structured_outputs_config.reasoning_parser: gemma4` holds the answer grammar
until the model closes the channel.

### Two images in one prompt

The gemma configs now pin `limit_mm_per_prompt: {image: 2}`. The pin states
the contract. It unlocks NOTHING.

Measured 2026-08-20: vLLM 0.19.0, 0.23.1 and 0.25.0 all give an unspecified
modality a limit of 999 ("Unspecified modality is set to 999 by default",
`vllm/config/multimodal.py`). Thus both images always reached the model, and
this pin lowers the allowance from 999 to 2. Raise it before you send a third
image with these shared configs.

Warning: the `qwen3.5-9b/instruct.yaml` comment says a missing cap makes vLLM
drop the second image. No version on this disk behaves that way. Do not use
the cap to explain a lost image.

## Modality order: the anchor problem

The card asks for image content before the text. gemma-4-12b does not obey the
simple form of that rule. Under `images_then_text` (image, image, prompt) the
model binds the two images as one image. Measured on 16 pairs, 2026-08-13:

| Layout | "only one image" in the trace | NotSure |
|---|---|---|
| `images_then_text` | 15 of 18 | 83% |
| `interleaved_labels` | 0 of 18 | 5.6% |

`interleaved_labels` fixes it with a text anchor BEFORE each image, which
breaks the card's rule.

`images_then_labels` (new, 2026-08-20) puts each anchor AFTER its image:
image A, "Image A.", image B, "Image B.", prompt. An image opens the message,
and the anchor still separates the two images.

### What the probe found

Schools case, 70 pairs, gemma-4-12b, 2026-08-20. Every arm holds the same
pairs, thus the labels compare pair by pair.

| Path | Layout | NotSure | "only one image" | Exact agreement |
|---|---|---|---|---|
| thinking | `images_then_labels` | 10.0% | 0 of 70 | 65.7% |
| thinking | `interleaved_labels` | 15.7% | 0 of 70 | — |
| thinking | `images_then_text` | 81.4% | **55 of 70** | 25.7% |
| greedy | `images_then_labels` | 61.4% | no traces | 82.9% |
| greedy | `interleaved_labels` | 48.6% | no traces | — |
| greedy | `images_then_text` | 71.4% | no traces | 74.3% |

The `images_then_text` arm ran on 2026-08-20 with the `limit_mm_per_prompt`
pin in place and on vLLM 0.25.0. It reproduces the 2026-08-13 result almost
exactly (83% NotSure then, 81.4% now). The engine is thus NOT the cause: the
model reads two adjacent image blocks as one image. The traces say it plainly
— "there is only one" (19), "only one image" (22), "I only see one" (14).

Three readings:

1. `images_then_labels` BINDS both images. No trace of 70 speaks of one image,
   and the thinking abstention does not change (McNemar p=0.344). The card's
   rule and this architecture can hold together.
2. Adjacency is what breaks the binding, not the order. Text between the two
   images is necessary; text BEFORE the first image is not.
3. On the greedy path the new layout abstains MORE than the incumbent: 10
   pairs moved into NotSure and 1 moved out (McNemar p=0.012). The battery
   runs greedy, thus this matters.

The probe reproduces the registered battery, so the harness is sound: the
greedy incumbent arm gives 48.6% against 51.5% for the registered 1.1M-row
schools run, and the thinking arm gives 15.7% against 11.7% for the 11k trace
run.

**Decision 2026-09-02: the canonical layout is `interleaved_labels`.** This
REVERSES the decision of 2026-08-20 below. The whole 1,000,000-pair canonical
gemma battery and the 14 trace runs ran `interleaved_labels`, and the matched
run of 2026-09-02 measured the greedy abstention of the 2 layouts pair for
pair, on about 100,900 shared pairs for each case:

| Case | `interleaved_labels` | `images_then_labels` | Delta |
|------|------|------|------|
| Restaurants | 28.7% | 44.8% | +16.1 |
| Parks / Plazas | 43.8% | 54.3% | +10.5 |
| Libraries | 14.3% | 24.6% | +10.4 |
| Schools | 51.6% | 61.7% | +10.1 |
| Subway safety | 6.8% | 13.0% | +6.2 |
| Road quality | 0.4% | 0.1% | -0.2 |
| Street photography | 0.0% | 0.0% | 0.0 |

Both layouts bind the 2 images. The card's "image first" rule was the only
reason for the switch, and the abstention cost weighs more. Thus:

- `research/scripts/register_canonical_runs.py` demands `interleaved_labels` again.
  The 14 canonical gemma runs verify. The `noabstain`, `nocity` and
  `textfirst` gemma arms, which ran under the 2026-08-20 rule, carry a
  recorded layout problem until they run again. The `nocity` and `textfirst`
  gemma halves re-ran on 2026-09-02 (`prompt-ablations.md`). The `noabstain`
  gemma half has NOT.
- The text-first probe of the canonical layout is `text_first_interleaved`
  (prompt, "Image A:", A, "Image B:", B), added 2026-09-02.
- `research/monocle/canonical.py` and `tests/test_canonical_prompt.py` demand
  `interleaved_labels` now. The lens work of 2026-08-21 to 2026-09-01 was built
  on `images_then_labels`. The lens reads the layout that the run recorded.

**Decision 2026-08-20 (superseded): production moves to `images_then_labels`.**
It obeys the card and it binds. These sweeps set it at the time:

- `looks_proxy_gemma12b`, `looks_thinking_gemma12b`
- `million_proxy_gemma12b`, `million_noabstain_gemma12b`
- `thinking_public_investment_10k`, `thinking_remaining_cases_10k`

`canonical_gemma12b` keeps `interleaved_labels`: that file exists only to
repeat the runs of 2026-08-11..14.

Three gates now demand `images_then_labels` and REJECT every other layout,
`interleaved_labels` included:

| Gate | File | Constant |
|---|---|---|
| registration and verify | `research/scripts/register_canonical_runs.py` | `REQUIRED_LAYOUT` |
| the lens prompt path | `research/monocle/canonical.py` | `REQUIRED_LAYOUT` |
| the test suite | `tests/test_canonical_prompt.py` | reads the constant |

The stage itself still SUPPORTS the other layouts — the probes need them for
comparison. The gates govern what may enter the registry, not what may run.

**History.** From 2026-08-20 to 2026-09-02 the registry gate demanded
`images_then_labels`, thus `verify` reported 14 problems against the canonical
gemma runs. The open item of that period, the greedy abstention rise of the
schools probe (10 pairs into NotSure, 1 out, McNemar p=0.012, n=70), was
measured at 100,000 pairs on 2026-09-02 and held on every case that abstains.
That measurement is what reversed the decision. See the table above.

## Visual token budget

No config sets the budget, thus every gemma run uses the 280 cap. The count is
not the cap: the processor rounds the resize down to whole 48-pixel cells
(patch 16 x pooling 3), so a square 1024x1024 cyclomedia face lands under it.
Measured through the processor, 2026-08-20:

| `max_soft_tokens` | Tokens for each image | For each pair | Prompt | Pixels for a face |
|---|---|---|---|---|
| 70 | 64 | 128 | ~240 | 384 x 384 |
| 140 | 121 | 242 | ~354 | 528 x 528 |
| 280 (today) | 256 | 512 | 624 | 768 x 768 |
| 560 | 529 | 1058 | ~1170 | 1104 x 1104 |
| 1120 | 1089 | 2178 | ~2290 | 1584 x 1584 |

vLLM accepts only these five values, as
`model.engine_kwargs.mm_processor_kwargs.max_soft_tokens`. Any other number
falls back to the model default and gives no error.

The card recommends a low budget for classification. A pairwise ordinal
judgment is a classification by its output, but the evidence is a facade or a
pavement. The probe (schools, 70 pairs, greedy, same pairs for each arm):

| Budget | NotSure | Exact agreement with 280 | Same side | McNemar vs 280 |
|---|---|---|---|---|
| 70 | 54.3% | 78.6% | 91.4% | p=0.344 |
| 140 | 60.0% | 67.1% | 92.9% | p=0.115 |
| 280 | 48.6% | — | — | — |

**Decision: keep 280.** Neither cut is free. The abstention rate rises at both
smaller budgets, and 140 keeps only 67% of the labels. The rise is not
significant at 70 pairs, but nothing here supports a cut, and a cut would cost
a re-run of the whole battery.

Warning: agreement does not prove a budget is safe. A model that cannot resolve
the facade can still guess the same side of a pair. Read the traces before you
trust a smaller budget.

## The probe files

| File | What it varies |
|---|---|
| `conf/sweep/layout_probe_gemma12b_proxy.yaml` | layout, greedy path |
| `conf/sweep/layout_probe_gemma12b_thinking.yaml` | layout, thinking path — the decider |
| `conf/sweep/budget_probe_gemma12b.yaml` | `max_soft_tokens` |

Each probe runs 64 pairs of the schools case on `klara_1x`. The thinking probe
is the one that can SEE a binding failure: a label-only run shows the failure
as an honest abstention rate, and the two look the same.

## See also

- [gemma4-12b-architecture.md](gemma4-12b-architecture.md) — the 48 blocks, the two attention kinds, the encoder-free vision path.
