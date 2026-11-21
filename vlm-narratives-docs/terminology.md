# Terminology for the white-box study

Written 2026-09-17. This page fixes the vocabulary of the white-box
documents and of Paper 2. Each standard term comes from the papers in
`white-box-related-work.md`; 4 scans read the full text of 40 papers and
checked each quotation. A term marked (statistics) is textbook usage that
the scanned papers do not define.

Rules:

1. Use the standard term. Do not use the retired term, except as a code
   identifier (`rho_partial`, `null_q95`, `rd`, a file or directory name).
2. On first use in a document, give a code identifier its standard name:
   "the control-word threshold (`null_q95`)".
3. Reserve "probe" for a trained classifier on activations, "feature" for
   a concept the model represents, "latent" for a learned unit of a
   sparse autoencoder, and "patch" for a square of input pixels.
4. Where the literature has no name, describe the thing in plain words.
   Do not coin a name.

## Statistics and design

| Retired term | Standard term | Source |
|--------------|---------------|--------|
| density partialled; the density partial; `rho_partial` | partial Spearman correlation, controlling for population density | "controlling for" (Naik 2017; Hofmann 2024); partial correlation (statistics) |
| the brightness partial | partial correlation that also controls for mean image luminance; luminance as a covariate; the luminance control | the same; covariate, confounder (Manvi 2024; Nwatu 2023) |
| the pool; pool words | control words (247 neutral scene words) | "serve as the control" (Manvi 2024); attribute and target words (Caliskan 2017) |
| the pool bar; `null_q95`; clears the pool bar | control-word null distribution; its 95th percentile is the significance threshold; "exceeds the control-word threshold" | null hypothesis, "expected by chance" (Caliskan 2017; Hofmann 2024); null distribution (statistics) |
| a word reads; a read (as a finding) | a significant association: the confidence interval excludes 0 and the correlation exceeds the control-word threshold | "significantly above zero" (Hofmann 2024) |
| reversed | a significant association of the opposite sign | "the exact opposite trend" (Hofmann 2024) |
| dose-response | correlation across all 197 neighborhoods ("monotonic correlation with ground truth") | Manvi 2024; Spearman's rank correlation (Nwatu 2023) |
| extremes; extremes AUC | extreme-groups comparison (the 10 highest against the 10 lowest neighborhoods); AUC | (statistics) |
| AUC with the NTA cluster interval | AUC with a 95% cluster-bootstrap confidence interval (neighborhoods resampled) | "95% bootstrapped confidence intervals" (Bai 2025); cluster bootstrap (statistics) |
| matched AUC; density-matched redraw | density-matched comparison; density-matched sample | "matched" (Hofmann 2024) |
| the cross-read check | test of independence from the wealth readout: the association is re-estimated controlling for the wealth readout | "controlling for", "hold up in multivariate specifications" (Naik 2017) |
| arm | condition (model x token position); setting | "treatment condition", "control condition" (Hofmann 2024) |
| the draw; own draw; union | the sample; the task's own sample; the pooled sample | "random sample", "stimuli" (Hofmann 2024; Bai 2025) |
| proxy (a census variable, in the white-box documents) | ground-truth variable; census variable; socioeconomic or demographic indicator | "ground truth" (Manvi 2024); "demographic variables" (Gebru 2017). The audit documents keep "proxy" for a measured stand-in of a latent quality, as Paper 1 defines it. |
| the read of record | the primary readout, selected a priori and blind to the ground truth | (statistics: primary outcome) |
| refused / answered; abstain label | refusal; refused and answered images; the abstention option | Arditi 2024; "refuses to answer" (Manvi 2024) |
| probe (a contrast pair with its question and proxy); round 1, 2, 3 | contrast task; Experiment 1, 2, 3 | target and attribute words (Caliskan 2017) |
| the word census | single-word logit-lens analysis | |
| lead; question-word prior | prompt-induced prior (the prompt names the word) | |
| stigmatized pole, poor pole; plain pole, rich pole | negative-valence terms (poverty terms); positive-valence terms (affluence terms) | "valence", "favorability" (Caliskan 2017; Hofmann 2024) |

## Interpretability

| Retired term | Standard term | Source |
|--------------|---------------|--------|
| the plain lens; a read; lens read | logit lens; logit-lens readout ("we decode the hidden state with the unembedding matrix") | Belrose 2023; Neo 2025 |
| transport (of the Jacobian lens) | the Jacobian lens is a tuned-lens variant: a fitted linear translator per layer into the final-layer basis; "translate" | "translators" (Belrose 2023) |
| block; L42 | layer; layer 42; early, middle and late layers | all scanned papers |
| residual (noun, alone) | residual stream; hidden state | Neo 2025; Belrose 2023 |
| the answer position | the last token position | Neo 2025; Zhang 2025; Arditi 2024 |
| the patches; patch positions; cells; soft tokens | image tokens (visual tokens); image-token positions. "Image patch" stays for the 48-pixel input square. | Neo 2025; Basu 2024; Kaduri 2025 |
| patch mean read | mean logit-lens log-probability over the image tokens | cf. "internal confidence" (Jiang 2025, a maximum, not a mean) |
| the relay | information flow: image tokens, then question tokens, then the last token | "information flow" (Zhang 2025); "information transfer" (Basu 2024) |
| swap; donor; recovery | activation patching; source image; fraction of the effect recovered (patching effect) | Zhang & Nanda 2024; Meng 2022 |
| removal map; causal patch map; ablation map | token-ablation map: the effect on the logit difference of ablating 1 image token | "token ablation" (Neo 2025) |
| the saliency null; cheap instruments fail | gradient-based attribution does not recover the token-ablation map | "gradient-based ... baselines" (Neo 2025) |
| `c`; contrast; the read `c` | logit difference (between the 2 answer words) | Zhang & Nanda 2024 |
| steering; injection; dose; alpha | activation addition; steering vector; coefficient alpha (intervention strength) | Rimsky 2024; Arditi 2024; Li 2023 |
| the carrier; the judgment axis | the principal judgment direction: the direction of largest between-image variance of the logit difference | cf. difference-in-means direction (Arditi 2024) |
| the handle; the pinv vector | the pseudo-inverse steering direction (minimum-norm pre-image of the logit-difference gradient under the lens) | cf. the measurement and intervention representations of Park 2024 |
| the massive coordinate; the centering trap | massive activations (rogue dimensions); mean-center the activations before any cosine similarity | Sun 2024; Timkey & van Schijndel 2021 |
| the block-47 rewrite | the final-layer transformation (no standard name): the layer's output scalar and its MLP output replace most of the incoming residual stream | cf. Sun 2024; Belrose 2023 |
| the component split; `rd`; raw projection | direct logit attribution: the contribution of the incoming residual stream, the attention output and the MLP output to a token's unnormalized logit | standard mechanistic-interpretability usage (Elhage 2021; Nanda) |
| the h47 probe; gone or moved | linear probe with held-out neighborhoods; it tests if the information stays linearly decodable | Gurnee & Tegmark 2024; Li 2023 |
| workspace; ignition | late layers (40 to 45) at the last token position; the layer at which the judgment first becomes decodable | |

## Post-training

| Retired term | Standard term | Source |
|--------------|---------------|--------|
| base; instruct; checkpoint | base model; instruction-tuned (IT) model; "gemma-4-12B" and "gemma-4-12B-it" | Gemma Scope 2024; Arditi 2024 ("chat model") |
| mask; masked at the output; lipstick | suppressed at the output: represented in intermediate layers and not expressed ("bypassed, not removed"; a "wrapper") | Lee 2024; Jain 2024; "shallow safety alignment" (Qi 2025) |
| absence; removed at every depth | removed: represented by the base model, and by the IT model at no layer | "deleted altogether" (Jain 2024, as the alternative) |
| left alone | preserved: expressed at the output of both models | cf. "unshifted positions" (Lin 2024) |
| the refusal hides a judgment | the refused judgment stays linearly decodable: a gap between the internal representation and the output | Li 2023; Hofmann 2024 ("conceal ... on the surface") |
| the handle at block 45 writes any pole | one direction, added at layer 45, is sufficient to elicit the refused answer | "necessary and sufficient" (Arditi 2024) |

## Sparse autoencoders

| Retired term | Standard term | Source |
|--------------|---------------|--------|
| feature (a learned unit) | latent; "feature" stays for the concept | Gemma Scope 2024; Gao 2025 |
| dictionary; width | sparse autoencoder (SAE); dictionary size (number of latents) | Gao 2025; Bussmann 2024 |
| FVU; dead; L0 | fraction of variance unexplained (FVU); dead latents; L0 (mean number of active latents) | Rajamanoharan 2024; Gao 2025 |
| the splice test; verdict flips under the reconstruction | spliced-in evaluation: the share of decisions that change when the SAE reconstruction replaces the activation | "splice the SAE into the LM's forward pass" (Rajamanoharan 2024) |
| seed match; direction-stable | latents shared across seeds: matched by decoder cosine similarity (threshold 0.7) | Paulo & Belrose 2025; Fel 2025 ("stability") |
| concept-stable; a code, not directions | matched by activation correlation and not by decoder direction (activation similarity against decoder similarity; no standard name) | cf. Leask 2025 |
| the working set | shared latents; the others are orphan latents | Paulo & Belrose 2025 |
| gate G1; gate G2 | criterion G1 (reconstruction fidelity); criterion G2 (stability across seeds) | "reconstruction fidelity" (Rajamanoharan 2024) |
| conditional centering; frame identity | per-template (or per-position) mean subtraction; the share of variance that the prompt template explains | cf. "pre-encoder bias" (Gao 2025) |

## Attribution over image tokens

| Retired term | Standard term | Source |
|--------------|---------------|--------|
| cell; square (a unit of the 16 x 16 grid) | image token (1 of 256); "image patch" for the input pixels | Neo 2025 |
| the measured map; the truth | the token-ablation map (the measured effect of ablating each image token on the logit difference) | "token ablation" (Neo 2025) |
| the analytic reader; g*h | gradient-times-input attribution at the residual stream (the gradient of the logit difference times the hidden state) | Shrikumar 2017; "gradient-based attribution" (Neo 2025) |
| the fitted reader; the reader | the fitted ablation predictor: a ridge regression from gradient-times-input to the token-ablation effect | (no standard name; describe) |
| instrument | attribution method | Neo 2025 |
| gate; Gate B; the power gate | criterion; criterion B; the statistical-power test | (statistics) |
| the deletion battery | the token-ablation experiment | Neo 2025 |
| hole map; content map | ablation-effect map; content map (the map of what each token depicts) | |
| the grain | the ablation unit (1 token, a 2 x 2 block, a set) | |
| a pointer, not a heat map | the predictor ranks the top image tokens; it does not reproduce the full map | |
| energy (of a direction, a map) | variance; squared norm. "Edge energy" stays as an image feature. | |
| null (as a noun, a baseline) | null model; baseline (model-free baseline, judgment-free baseline) | (statistics) |

## Document vocabulary

| Retired term | Standard term |
|--------------|---------------|
| Traps | Pitfalls |
| the ledger; stands; reversed; open | summary of claims; supported; not supported (or refuted); open |
| frame; open frame; closed frame; raw frame | prompt template; open-ended prompt; forced-choice prompt; completion-style prompt (no chat template) |
| verdict (of the model) | decision; judgment |
| instrument | method; measurement |
| NOT concluded | Open questions |
