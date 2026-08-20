---
title: "Activation Probe — Is the Judgment Represented, or Only the Proxy?"
category: concept
created: 2026-07-12
updated: 2026-07-12
tags:
  - concept
  - pairwise
  - interpretability
  - activations
  - probing
  - gemma-4
  - transformers
---

# Activation Probe

The sequel to [[concept-self-explanation-ladder]]. That experiment established a clean negative: across **four** self-report channels (its own labels, factual captions, contrastive captions, and the pixels themselves) Gemma-4-12B never names the evaluative concept behind its own judgments, and every recovered prompt plateaus **0.05–0.09 ordinal below the true-axis ceiling**.

That negative is ambiguous between two very different worlds:

| | Hypothesis | Implication |
|---|---|---|
| **H1** | **"Cannot say it."** Safety *is* represented internally and drives the judgment — the model just cannot verbalize it. | A self-explanation / interpretability failure. The residual is a *verbalization* gap. |
| **H2** | **"Does not represent it."** There is no safety representation; the judgment literally *is* a brightness/openness computation, and "safe" in the prompt merely steers a generic visual-salience circuit. | A **validity** failure. A VLM's "safety perception" is a brightness detector wearing a safety costume. |

Every GEPA channel is *behavioral*, so none can distinguish these. The activation probe reads the model's internals instead.

**Preliminary answer: H1.** The information is in the forward pass the whole time.

## Extraction recipe (verified, `.venv-nightly` + klara)

> [!tip] `Gemma4UnifiedForConditionalGeneration` is **native in `transformers` 5.12.1**
> No `trust_remote_code`, no vendored architecture. `AutoProcessor` resolves `Gemma4UnifiedProcessor` cleanly.

Reuse the **exact production prompt path** (see `_gemma4_unified_chat_template()` in [[vllm-inference]]) or the probe measures a different model than the one that produced the labels:

```python
is_g4u, tmpl = _gemma4_unified_chat_template(MODEL_DIR)      # loads chat_template.jinja
proc  = AutoProcessor.from_pretrained(MODEL_DIR)             # -> Gemma4UnifiedProcessor
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0").eval()

messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": [
        {"type": "image"}, {"type": "image"},          # g4u matches on type == "image"
        {"type": "text", "text": user_text}]},
]
text   = proc.apply_chat_template(messages, tokenize=False,
                                  add_generation_prompt=True, chat_template=tmpl)
inputs = proc(text=[text], images=[pair_images], return_tensors="pt").to("cuda:0")
out    = model(**inputs, output_hidden_states=True, use_cache=False)
```

| Property | Value |
|---|---|
| Weights | 22.3 GiB bf16 — **one A6000**, loads in ~13 s |
| Hidden states | **48 layers × 3840 dims** |
| Image tokens | 512 (256/image, id `258880`), seq ≈ 732 |
| Throughput | **0.33 s/pair/condition**, forward-only (1,500 pairs × 3 conditions ≈ 27 min) |
| Faithfulness | HF greedy vs production labels = 79% exact / **0.948 ordinal** — same model vLLM served |

**Read position.** Teacher-force the JSON prefix `{"answer": "` (4 tokens) so the *final* position is the one that literally emits the label token; the position 4 back is then the last real prompt token. Both read positions perform equivalently (`@label` ≈ `@last`), so one forward yields both.

## ⚠️ The Gemma massive-activation trap

> [!danger] Raw cosine between residual streams is **1.0000** across completely different pairs. This is an ARTIFACT. It nearly killed the experiment.
> **Dim 1750 alone has |mean| = 246.9**, while the median dimension is 0.25. `||mean vector|| = 251.8` vs `mean ||x − mean|| = 1.7` — the shared component is ~**150×** the pair-specific signal, so every vector looks parallel.
>
> **Centered**, the honest geometry appears: mean cosine **0.0005**, range −0.85 to 0.92. States are pair-specific and near-orthogonal.
>
> **Always center / standardize before probing or plotting. Never trust a raw cosine on a Gemma residual stream.** Read at face value, the raw number says "all activations identical, probe is dead" — which cannot be true, since the model emits different labels for those same pairs.

## Preliminary result (subway, 1,500 pairs)

Ridge on the 3,840-d residual (standardized), 1,125 train / 375 test, ordinal `1 − |Δ|/4`. Layers {12, 24, 36, 47}.

Constant always-"More" on this split = **0.817** (vs 0.823 in the GEPA sweeps → split and metric validated).

| activations taken under… | probe → **safety** label | that prompt's *behavioral* score |
|---|---|---|
| `prod` — the real safety prompt | **0.950** (exact 0.82, L47) | 0.895 *(true-axis ceiling)* |
| `axis` — **"brightly lit"** (rung-5 winner) | **0.886** (exact 0.56, L47) | **0.820** |
| `neutral` — blind "Compare the two images." | **0.877** (exact 0.53, L24) | 0.804 |
| *always-"More"* | *0.817* | *0.823* |

### Reading

- **When the model is asked which photo is *brighter*, its internal state predicts its own *safety* judgment at 0.886 — 0.066 above what it actually says (0.820).** Its lift over the constant (+0.069) is essentially the lift the *true safety prompt* achieves behaviorally (+0.072).
- Even the **blind** prompt's activations carry the judgment (0.877).
- → The GEPA negative is **not** a representational gap. The failure is specifically in **articulation**. This *completes* the self-explanation story rather than contradicting it: it locates where the break happens.
- `prod` = 0.950 is the **extraction upper bound, not a fair comparison** — it reads out the answer the model is about to emit. Reassuringly it lands exactly on the independently measured greedy self-agreement (0.948), i.e. the probe saturates the temperature-0.6 noise ceiling. This confirms extraction is sound.

## Caveats — not yet a paper claim

1. **Decodability ≠ use.** A linear probe can find information the model never reads downstream. The **causal** ablation/steering check is what licenses the strong claim: ablate the safety direction under the production prompt and see whether judgments collapse onto the brightness judgments; add it under a neutral prompt and see whether safety-like judgments appear.
2. **`probe(neutral) ≈ probe(axis)`** invites the deflationary reading — *"safety is decodable from generic visual features"* — which is close to circular (of course a readout on a VLM's image representation predicts that VLM's own judgments). The **projection test** kills that: collect the model's own *brightness* labels, fit `w_bright`, project it out of the `prod` activations, and ask whether safety survives. If it does, safety ⊄ brightness.
3. **Ordinal is forgiving** under the 60%-"More" label skew (the constant already scores 0.817). Report exact-match and a balanced metric alongside.
4. Layer grid was coarse (12/24/36/47); L47 best for `prod`/`axis`, L24 for `neutral`.
5. Keep probes **linear** (standard discipline against reading probe capacity as representation) and include a shuffled-label control.

## Next steps

1. **Projection test** — the decisive one (needs the model's own brightness labels; cheap).
2. **Causal steering / ablation** — moves from *decodable* to *used*.
3. Extend to restaurants + schools; finer layer sweep.

## Artifacts

| What | Where |
|---|---|
| Smoke (shapes, faithfulness, A/B-swap, speed) | `outputs/_actprobe/act_probe_smoke.py` + `.sub` — job 849227 |
| Preliminary probe (3 conditions × 2 positions × 4 layers) | `outputs/_actprobe/act_probe_smoke2.py` + `.sub` — job 849256 |
| Results | `outputs/_actprobe/probe_prelim.json`, `outputs/act_probe_{smoke,prelim}_*.out` |
| Supervision | `multirun/2026-06-29_URBANPAIRVQA/14-37-17/0/outputs/pairwise/subway_safety_mvp_20260629_143729.parquet` |

Environment: `.venv-nightly` + `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`, klara, 1×A6000 (same recipe as the GEPA runs — see [[slurm-deployment]]).

## See also

- [[concept-self-explanation-ladder]] — the behavioral experiment this settles
- [[urban-pair-vqa]] — the dagspace, cases, and production labels
- [[vllm-inference]] — `_gemma4_unified_chat_template()` and the `gemma4_unified` prompt path
- [[troubleshooting]] — the massive-activation trap and the `[val-eval]` scoring trap
