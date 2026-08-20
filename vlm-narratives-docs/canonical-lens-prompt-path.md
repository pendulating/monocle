# The canonical prompt path for the lens study

`monocle/canonical.py` builds the forward pass that the Jacobian lens reads. It
makes the lens read the same prompt the CVPR battery ran.

## Why it exists

The lens modules were written before the 2026-08-11 prompt consolidation. They
pointed at the 2026-06-29 subway run and at a `conf/prompt/` YAML. The
consolidation changed 3 things, and each one breaks a lens read.

| Mismatch | The old lens build | The canonical battery |
|---|---|---|
| Image layout | `images_then_text` | `interleaved_labels` |
| System turn | a persona string | `system: null` — no system turn |
| Abstention | 5 labels, no `NotSure` | `NotSure` in the enum and in the text |

The layout mismatch is the worst of the 3. The registry gate demands
`interleaved_labels` for gemma-4-12b, because that architecture does not bind
image B without the text anchor. A lens read under `images_then_text` measures
a different forward pass. Every per-patch map over image B would be wrong.

Thus the rung-B and rung-C results from 2026-07-24 describe a superseded
prompt. Re-run them on this path before you cite them.

## The design rule

**Call the production code. Do not copy it.**

Every string that reaches the model comes from
`dagspaces/urbanpairvqa/stages/pairwise_vqa.py` and
`dagspaces/common/vllm_inference.py`. `monocle/canonical.py` holds no prompt
text of its own, except the 2 contrast templates below. Thus the prompt cannot
drift away from the battery.

`tests/test_canonical_prompt.py` renders the chat text twice — once through
`monocle.canonical`, once through the production chain — and asserts that the
2 strings are equal, for all 7 cases.

## The source of truth

The prompt config comes from the registered run's own
`stage/.hydra/config.yaml`, reached through the registry symlink. It does not
come from the YAML that sits in `conf/prompt/` today. A prompt file can change
after a run. The run's recorded config cannot.

See [canonical-run-registry.md](canonical-run-registry.md).

## Scope

gemma-4-12b only. Monocle binds `Gemma4UnifiedForConditionalGeneration`, and
all 3 fitted lenses are gemma-4-12b (48 layers, d_model 3840). A qwen3.5-9b
study needs its own lens fit first. `registry_dir` refuses another model.

**Warning:** prefer the `proxy` kind. The `trace` runs set
`enable_thinking=True` and sample at temperature 1.0, so the answer position
sits after a sampled reasoning block. The residual there then depends on
sampled text, not on the images alone.

## The contrast conditions

`build_conditions(case, kind)` returns one config per arm. Each arm is the
canonical config with **only** `prompt.user_template` changed.

| Arm | Question |
|---|---|
| `prod` | the registered question, byte-identical |
| `neutral` | "Compare the two images." — names no attribute, unit, or city |
| `axis` | "Which image looks more `<phrase>`?" — only where a phrase exists |

The system turn stays absent, the layout stays `interleaved_labels`, and the
abstention guidance still appends in every arm. Thus a prod-minus-neutral
difference measures the question, and nothing else.

The old build did not do this. Its neutral arm carried "You are a helpful
assistant.", its axis arm carried a long judge persona, and its prod arm
carried the old subway persona. That contrast moved the persona and the
question together.

## Abstention

`label_classes` puts `NotSure` in the first-token class set whenever the run
enabled it. `MuchLess` and `MuchMore` still collapse to `Much*`, because they
share the token "Much".

`NotSure` is **not** a point on the ordinal scale. `collapsed_ordinal_agreement`
returns NaN when either side abstains, and the summary reports `n_ordinal`,
`read_abstain_rate`, and `label_abstain_rate` next to the mean.

Read the abstention rate first. It is very different between cases.

| Case | NotSure rate (proxy) |
|---|---|
| street_photography | 0.000 |
| road_quality | 0.004 |
| subway_safety | 0.068 |
| libraries | 0.144 |
| restaurants | 0.287 |
| parks_plazas | 0.438 |
| schools | 0.518 |

## Presented order

`presented_images(row)` returns `presented_left_path` then
`presented_right_path`. The battery swaps the presentation order on half the
rows. A read that used `image_path_a` / `image_path_b` would mirror every
per-patch map on those rows. The function refuses a row that lacks the
presented columns.

## The teacher-forced prefix

The lens reads the position that emits the label. To put the sequence there,
the build appends a JSON prefix. **The prefix must match how the model writes
the object.** Under greedy decoding gemma-4-12b emits:

```
{
  "answer": "More"
}
```

— a newline and a 2-space indent. `canonical.FORCE_PREFIX` is that form, and it
is the default.

The 2026-07-24 rung-B run used a compact `{"answer": "` instead. That prefix is
off-distribution, and **it fails silently, per case**. Measured on 150 pairs
per case, mean p(prod label) at L47 under the production prompt (jobs 199648
and 199649):

| Case | compact | natural |
|---|---|---|
| subway_safety | 0.778 | 0.774 |
| road_quality | **0.001** | 0.769 |
| restaurants | **0.001** | 0.771 |

Two things follow.

1. subway_safety is the only case rung B ran, and it is unaffected. That
   result stands.
2. A multi-case study on the compact prefix would have reported a flat zero for
   road_quality and restaurants, with nothing to separate that from "the
   judgment never enters the channel". The failure has no signature.

Under the natural prefix all 3 cases land at L47 near 0.77, which also matches
the 0.770 rung B reported. `canonical.FORCE_PREFIX_COMPACT` stays only to
reproduce that run.

The neutral arm is insensitive to the choice (0.19-0.26 at L47 under both), so
the effect is specific to the prompt that asks for the judgment.

## Guided decoding

The battery sampled with guided decoding against the JSON schema, so the enum
was enforced. The lens reads a free softmax at the forced-prefix position. `p(label token)` is thus not what guided decoding emitted. This is
the correct choice — you want the model's unconstrained disposition — but state
it wherever the numbers appear.

## What uses this path

| Module | Change |
|---|---|
| `monocle/safety_workspace.py` | `--case` / `--kind`; pairs, prompts, and labels all come from the registry |
| `monocle/jlens_steer.py` | the same; `--parquet` and `--prompt-yaml` are gone |
| `monocle/answer_tokens.py` | the open-vocabulary answer readout — see [answer-token-readout.md](answer-token-readout.md) |

## Commands

```bash
# CPU gate — run this after any prompt or stage change
pytest tests/test_canonical_prompt.py -v

# GPU smoke on the rebuilt path
sbatch monocle/monocle.sub monocle.safety_workspace --smoke \
    --case subway_safety --kind proxy
```

## Verification, 2026-08-18

`tests/test_canonical_prompt.py` — 50 tests pass, including chat-text identity
on all 7 cases. The monocle suites total 168 tests and pass.

GPU smoke, job 189405, klara: 2 pairs, 3 conditions, 3 lenses.

- Image blocks `[256, 256]`, `seq_len=703` — 2 equal patch blocks under the
  interleaved layout.
- `NotSure` present in the first-token map, id 4348.
- Pairs read from the registry: `subway_safety/proxy: 2/100000`.
- 3.2 s per pair for 3 conditions and 3 lenses.
