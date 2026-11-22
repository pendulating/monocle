# How interpretability researchers analyze and validate lens readouts

Evidence labels used in these notes:

- **[raw]**: I read the primary text or code directly.
- **[summary]**: I read a page summary from a fetch tool. I did not check each number against the raw text.
- **[snippet]**: I saw only a search result or an abstract.

The short source names below link to these documents:

- GW paper: Gurnee, Sofroniew, ... Lindsey (2026), "Verbalizable Representations Form a Global Workspace in Language Models". [arXiv HTML](https://arxiv.org/html/2607.15495v1).
- jlens repo: [anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens). Local copy: `/share/ju/matt/mllmsci/sub/jacobian-lens`.
- J++ post: Ayonrinde and Lindsey (2026). [LessWrong](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace). I read the [GreaterWrong mirror](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful).
- GPT-2 post: Cairns, yix and Roe (2026). [GreaterWrong mirror](https://www.greaterwrong.com/posts/GeKveaNdruATcj2Fj/even-gpt-2-passes-most-functional-global-workspace-tests).

## What does the Global Workspace / Jacobian lens paper do to analyze readouts?

### Takeaway
The GW paper treats a readout as a hypothesis, and it tests the hypothesis with an intervention. It uses four kinds of evidence: rank statistics over a layer band, a prompt or word control, a causal swap or ablation, and a test that does not use the lens. The paper has no general fix for a fixed token bias. It uses controls and contrasts case by case. The Neuronpedia blog post `jacobian-lens` has almost no analysis method.

### Cited Findings

**Source check**
- The Neuronpedia post is an announcement. It has no statistics, no baselines and no bias discussion. It says only that small models "oversteer" swaps and need fewer layers. [Neuronpedia blog](https://www.neuronpedia.org/blog/jacobian-lens) [summary]
- The methods are in the GW paper, the jlens `data/` READMEs, and the [Neuronpedia docs](https://docs.neuronpedia.org/jacobian-lens). The docs say the lens skips about the first 1/3 of layers because it "is not reliable there". [summary]

**The lens and the three read forms** ([GW paper §2.1, §2.5](https://arxiv.org/html/2607.15495v1)) [raw]
- The lens is `softmax(W_U norm(J_l h_l))`. J_l averages over source positions, later target positions, and 1000 prompts of 128 tokens.
- Form 1 is the ranked list of top tokens.
- Form 2 is a per-token probe: the score or cosine of h_l against one lens vector v_t. The paper uses this form to test a named concept against a threshold.
- Form 3 is a sparse non-negative decomposition by gradient pursuit. The paper uses this form to get a discrete list of active concepts. It gives a less redundant set than top-k.

**Readout statistics** ([GW paper](https://arxiv.org/html/2607.15495v1)) [raw]

| Statistic | Definition | Where |
|---|---|---|
| Hit | Target token at rank 1 at any (layer, position) in the band | [experiments README](https://github.com/anthropics/jacobian-lens/blob/main/data/experiments/README.md) |
| pass@k | Mean over items of the fraction of intermediates whose min-over-layers rank is ≤ k | [evaluations README](https://github.com/anthropics/jacobian-lens/blob/main/data/evaluations/README.md) |
| pass@k AUC | Area under pass@k against log k, normalized so that rank 1 always gives 1 | §A.6 |
| Spearman | Lens against next-token logits, for 10 candidates in 14 categories, at 3 workspace layers | §3.1, Fig. 6 |
| Word set minus control | Mean lens log-prob of 2 target words, minus the mean of 100 unrelated control words, averaged over workspace layers | §A.21 |
| Top-k count | Count of a fixed word list in the top-10, summed over band layers | §5.4 |
| Top-25 presence | A word counts as present if its best rank over the band is ≤ 25 | §4.2, §5.5 |
| Median rank per token | Median over band layers, then the best value over a turn | §6.1 |
| Median-over-layers top-k | Fraction of tokens with a word in the top-10 of the median-over-layers rank | §6.2, Fig. 44 |
| Workspace loading | Cosine of residual and lens vector, averaged over relevant positions | §3.4 |
| Excess kurtosis | Kurtosis of the logit distribution for one (position, layer) | §4.1 |
| Top-1 autocorrelation | Same top-1 token at t and t+Δ, relative to a position-shuffled null | §4.1 |
| Occupancy | K at which the marginal reconstruction gain falls below that of random directions | §4.2 |
| Excess variance | Variance explained by top-K lens vectors, above a same-size random set | §4.2 |

**Baselines and controls** ([GW paper](https://arxiv.org/html/2607.15495v1)) [raw]
- No-instruction baseline: the paper shows that the prompt context alone does not put the target in the readout. The rate is about zero (§3.2).
- Unrelated control words: 100 concrete nouns such as recipe, bicycl and guitar (§A.21). Grey control words that never occur in the prompt appear in §4.2.
- Control conditions, not only control words. The preference test has 3 controls. It shows that false, despite and although also appear on control prefills, so only BUT is specific (§6.2).
- Shuffled-position null for autocorrelation (§4.1). Shuffled-prompt control for co-occupancy (§A.17): 0.46 against 0.53.
- Same-size random direction sets for occupancy and variance (§4.2).
- A covariance-matched random baseline for SAE kurtosis (§A.18, Fig. 75).
- Matched-norm perturbations and random-direction ablation for behavior effects (§3.5).
- Layer-matched random head sets for broadcast ablation (§4.3).
- Model contrast: post-trained model against base model on the same prompts (§6.1). Clean checkpoint against misaligned checkpoints (§5.4).
- Readout z-score against the no-instruction baseline, over stimulus positions and workspace layers (§A.12).

**Causal validations** ([GW paper](https://arxiv.org/html/2607.15495v1)) [raw]
- Coordinate swap: with V=[v_s v_t], c=V⁺h, set h' = h + V(σ(c) − c). The orthogonal part of h stays fixed (§2.5).
- Verbal report swap: the target comes from the same category and is not in the top-10 outputs. Pure lens vectors reach the top-5 on 88% of trials (§3.1).
- Two-hop swap over 50 prompts: top-1 success is 54% on Haiku 4.5, 70% on Sonnet 4.5, 70% on Opus 4.5 (§3.3).
- Privilege test. The paper splits a concept vector into a J-space part and a remainder. J-space part: 59% to top-5. Remainder: 5%. Pure lens vectors: 88%. All perturbations have the same magnitude (§3.1).
- Clamp test. The paper clamps the J-lens coordinates to clean-pass values. The remainder effect then falls to zero in the swap test. In the two-hop test it falls from 28% to 6% (§3.1, §3.3).
- Injection test: the paper adds a lens vector on the user turn and scores the reciprocal rank of the word at the open quote, with a strength sweep over n=100 concepts (§3.1).
- Ablation effect: project out one lens direction at the readout position and report the output KL (§A.6).
- Swap failures concentrate where the source concept has low workspace loading (§3.4). A failed swap thus does not disprove a lens word by itself, and a weak source predicts failure.
- Lens-independent localization: mean-difference patching uses no lens. It finds the same layers as the lens (§A.24.1).
- Lens-independent ignition measure: projection share on the line between pure-A and pure-B activations (§4.1.1).
- Independent readout agreement: the eval-awareness score separates NLA-flagged positions with AUC 0.853 on a stratified subset (§A.21).
- Dose response: the score moves monotonically under steering at −0.1, 0, +0.1 on all 6 suites (§A.21).
- Paired stimulus edit: realism edits lower the score on all 11 suites, and 8 reach significance at the 95% level (§A.21).
- A probe from one word set can include spurious cases. The paper proposes post-hoc LLM filtering (§A.21).

**Statistical reporting** [raw]
- Wilson 95% intervals for rates, IQR bands for reciprocal rank, SEM over stimuli, and paired-difference 95% CIs ([GW paper](https://arxiv.org/html/2607.15495v1), Figs. 7, 8, 16, 67, 82).

**Lens bias**
- The paper does not name a lens bias on single tokens. The nearest statements are these. [raw]
  - "at some positions and layers, the top lens tokens are ones we cannot make sense of", and "We have not characterized this systematically" (§9.1).
  - Template lens: "a small set of words appear in the top template lens readouts on a high fraction of transcripts". The paper filters them out, and it says this "is not a principled approach" (§A.9.1).
  - Concept vectors are "mean-subtracted over a baseline set of 100 other concepts" (§3.1). Templates are centered and whitened (§A.9.1).
  - Tuned lens: a bias term carries most of its gain over the logit lens in early layers (§A.6).

**Layer differences** ([GW paper §4.1, §2.5, §9.1](https://arxiv.org/html/2607.15495v1)) [raw]
- Layers are reindexed to 0–100, and the paper reports 25 evenly spaced layers.
- CKA between lens-vector similarity matrices gives early, middle and late blocks.
- Four statistics set the band at about L38–L92: top-k next-token accuracy, excess kurtosis, top-1 autocorrelation, and effective dimensionality.
- Above the band, the lens is "motor". It reads the next token.
- The paper says these statistics come from the lens and so could be artifacts. It then uses a test without the lens (ignition) to support the onset layer.
- The band boundary is "somewhat post-hoc" (§9.1).
- Results mostly aggregate over the band with best rank, median rank, or mean log-prob. The paper does not state a logit-scale normalization across layers.

### Inferences
- A `W_U J h` readout is homogeneous and linear before the norm step. Thus a logit contrast between two conditions is close to the lens applied to the activation difference. The `norm` step makes this approximate.
- The paper's own favorite form for a named concept is word set minus control word set, averaged over band layers. Our contrasts follow this form.
- The strongest precedents for us are §A.21 (word score with controls) and §6.1 (reaction concepts with a contrast of conditions).

### Gaps
- I found no statement of how the band should be set on a multimodal model.
- The paper does not give a general method for a token that leads at most positions.
- I did not read every appendix figure. Fig. 6 values are not in the text.

## What is J++ and how is it evaluated?

### Takeaway
J++ keeps one linear map per layer, but it filters noisy Jacobians with clustered "expert" Jacobians and an LRP backward pass. It then removes tokens with no letter or digit from the ranking. The evaluation is recall@10 of known intermediates plus an intermediate swap. J++ does not fix a word-like fixed bias such as ` de` or ` la`. Its authors tried z-scored readouts and report mixed results.

### Cited Findings
- `koayon/jpp-lenses` is a Hugging Face model repo, not a GitHub repo. The GitHub path returns 404. Code is at [safety-research/jpp_lens](https://github.com/safety-research/jpp_lens), and lenses are at [koayon/jpp-lenses](https://huggingface.co/koayon/jpp-lenses). [raw]
- Step 1, Jacobian Filtering: k-means with E=8 clusters per layer, one expert Jacobian per cluster, and a learned signed weight per expert and layer. [jpp_lens README](https://github.com/safety-research/jpp_lens) [raw]
- Step 2: an LRP backward pass as in the R-Lens. [jpp_lens README](https://github.com/safety-research/jpp_lens) [raw]
- Step 3, Readout Filtering: remove tokens with no letter or digit. That is 6,116 tokens. [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [raw]
- Fit data: 64 sequences of 128 tokens from WikiText-103, skipping the first 16 positions. A small dev set of latent-variable problems sets the expert weights. The dev set is disjoint from the eval set. [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [raw]
- Readout metric: recall@10 macro-averaged over 5 tasks, using best rank over 7 layers, with 95% bootstrap intervals over items. [jpp_lens README](https://github.com/safety-research/jpp_lens) [raw]
- The scorer has item-weighted and pair-weighted pass@k. Only items the model answers correctly count, except typo and association. [scoring code](https://github.com/safety-research/jpp_lens/blob/main/src/lens_evals/readout_evals/readout_eval_scoring.py) [raw]
- Results on Qwen3.6-27B: J++ 55.2%, J-Lens 35.7%, R-Lens 37.7%. [jpp_lens README](https://github.com/safety-research/jpp_lens) [raw]
- Median uplift across six models: 63.5% over J-Lens, 42.2% over R-Lens. [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [raw]
- Intermediate swap: 59.7% against 49.3% (J) and 50.7% (R). [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [raw]
- Ablation of the pooled Jacobian costs 11.7 points, of LRP 6.6, of Readout Filtering 2.9. [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [summary]
- Our models: Qwen3.5-9B has layers 0–30, target 31, recall@10 55.0%. Gemma 4 31B has layers 0–58, target 59, recall@10 58%. The Gemma value drops the poetry task, because 1 of 98 items passed the correctness filter. [HF card](https://huggingface.co/koayon/jpp-lenses) [summary]
- The fixed-token passage: "a few tokens appear in the top readouts of many activations despite seeming unrelated to the context". The Jacobian transport "is a homogeneous linear map rather than an affine map that can capture input-independent reweighting". [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [raw]
- Bias Addition: the authors consider an affine bias and "z-scoring the lens readouts". They "find these approaches to have a mixed influence on performance". [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [raw]
- Open limits: single-token vocabulary, poetry at most 5% for all lenses, and multi-digit numbers invisible on Qwen because of digit tokens. [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [summary]
- A reader comment asks whether the filter weights generalize to tasks without a pre-set target word. The post does not answer. [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [summary]
- R-Lens: fit as J-lens, with LRP rules in the backward pass. It reduces early "trash tokens" (for example `锁定`, `euw`, `tav`). It shows no gain on the smallest models tested. [R-Lens post](https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers) [raw]
- The [Neuronpedia J++ note](https://www.neuronpedia.org/blog/jpp-drift-ui) has no layer metrics. [summary]

### Inferences
- Readout Filtering removes only tokens with no letter or digit. ` de`, ` la`, ` que` and ` own` pass the filter. Our Gemma bias is thus outside the J++ fix.
- The expert weights are tuned on recall of named latent variables. A judgment such as "cool" is not in that task set. J++ gains may not transfer to an evaluative concept.
- The J++ recall values are for text prompts. They do not validate image-token positions or a judgment task.
- "Mixed influence" of z-scoring refers to recall of known intermediates. It does not say z-scoring is wrong for a contrast study.

### Gaps
- I found no independent replication of the J++ numbers.
- I found no per-layer J++ table for Gemma 4 31B or Qwen3.5-9B in the parts I read.
- I do not know if J++ on Gemma shows the same fixed words in the authors' runs. The post does not name ` de` or ` la`.

## What are the Neuronpedia oracle lens and NLA, and can they describe a residual in words?

### Takeaway
Both can describe a residual in words, and both have a confabulation risk. Neither has a released adapter for Gemma-4-31B or Qwen3.5-9B that I could find. Thus using them means training a new adapter, or using them only on a nearby model. NLA has a direct precedent for verbalizing activation differences.

### Cited Findings

**Oracle lens in the GW paper** ([§A.9.2](https://arxiv.org/html/2607.15495v1)) [raw]
- It has four stages on Haiku 4.5: a reconstructor (phrase to template vector), a dictionary of about 3.4 million phrase vectors, a teacher (non-negative OMP on 1 million activations), and an oracle trained by supervision and then RL.
- Reward: the fraction of whitened activation variance that the phrases explain.
- At inference: sample K phrases, map to directions, and refit coefficients with NNLS.
- The RL oracle explains on average 31% of whitened variance on held-out assistant-turn positions.
- Reads are at layer L67 with N=4, K=10. Some panels show the highest-FVE of 5 decodes.
- At most positions it predicts upcoming text. At delimiter tokens it reads as commentary on the situation.
- The paper suspects less confabulation than NLA, because the reconstructor is frozen and the reconstruction is linear. It also has lower reconstruction accuracy.

**Oracle lens in Neuronpedia** ([oracle.py](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/inference/neuronpedia_inference/endpoints/lens/oracle.py)) [raw]
- The endpoint is `/v1/lens/oracle`. It captures `resid_post` at the requested layers and positions. An adapter-on model then writes up to `max_bullets` bullets (cap 5) per position and layer.
- The registry holds Qwen3.6-27B, Qwen3.5-0.8B, Qwen3.5-2B, and Qwen3-4B. It holds no Gemma 4 model and no Qwen3.5-9B.
- [interp_engine/oracle.py](https://github.com/decoderesearch/interp-engine/blob/main/interp_engine/oracle.py) says: the base model plus a PEFT adapter "trained to describe one residual vector in words". The vector replaces the embedding of one marker character. The transform is `alpha * h / ||h||`. The trained band default is layers 20–60, step 4. Reads are greedy. [raw]
- The fixed prompt says: "Produce distinct concepts that encode this activation, each as a '- ' bullet on its own line." [raw]
- Files in [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens): `gemma-4-31b` has `jlens` and `jpp`, and `qwen3.5-9b` has `jpp`. Neither has an `oracle` folder (HF API listing, 2026-10-31). [raw]

**NLA** ([Fraser-Taliente et al. 2026](https://transformer-circuits.pub/2026/nla/))
- A verbalizer writes text from one activation at a middle-to-late layer. A reconstructor recovers the activation from the text. [raw]
- FVE = 1 − L / E‖h − h̄‖². An FVE of 0 equals predicting the mean. NLAs in the paper reach 0.6–0.8. [raw]
- Confabulation: claims can be false in detail but "typically thematically faithful". Claims that repeat across adjacent tokens are more trustworthy. [raw and summary]
- The paper checks claims with prediction tasks where ground truth is known by construction. [raw]
- Stated limits include confabulation, no mechanistic grounding, too much expressivity, and cost. [summary]
- The [NLA server](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/nla/README.md) treats round-trip MSE as the quality measure. Its default is Qwen 2.5 7B at layer 20. [raw]
- Public checkpoints under `kitft`: Qwen2.5-7B L20, Gemma3-12B L32, Gemma3-27B L41, Llama-3.3-70B L53 ([HF API](https://huggingface.co/kitft), 2026-10-31). The Neuronpedia post also names a Qwen 1.5B NLA, "surprisingly workable, especially for simple concepts", but it "couldn't replicate" poetry planning. [Neuronpedia blog](https://www.neuronpedia.org/blog/jacobian-lens) [summary]

**NLA on activation differences** ([arXiv 2610.30934](https://arxiv.org/html/2610.30934)) [raw for abstract, methods and discussion]
- It defines Δh = h_pi(x_a, θ_a) − h_pj(x_b, θ_b), and it feeds the difference to the verbalizer.
- Scoring: Hit rate over K samples at temperature T (default K=4, T=0.7), in a sample form and an any-of-K form.
- Verbalizations of composed analogy activations mostly name the target, and they rarely mention the subtracted word.
- It also has controls for false positives (its Appendices D.3 and E.4).
- FVE baselines use one fixed mean and variance from neutral prompts, so values stay comparable.

**Other free-text methods**
- The GW paper groups these together: patching into a template, supervised question-answer oracles, and NLA. It says they carry "additional risk of confabulations". [GW paper §8](https://arxiv.org/html/2607.15495v1) [raw]

### Inferences
- A difference vector is a valid input in principle, because the oracle transform unit-normalizes the vector. It is off-distribution for an adapter trained on single activations. The NLA paper gives support for differences only for NLA, not for the oracle.
- Text about an aesthetic judgment is a high-risk case for confabulation. A verbalizer can write a plausible story about urban quality, and we have no ground truth by construction. Use these tools to generate hypotheses, then test the hypotheses with word sets and swaps.
- Adjacent-token repetition is a cheap trust filter that we can use across positions.
- For a first pass, a word list plus a coherence score costs far less than training an adapter.

### Gaps
- I did not find the training recipe for the Neuronpedia adapters. I cannot say whether they follow §A.9.2 stage by stage.
- I found no FVE report for the Neuronpedia oracle adapters.
- I found no oracle or NLA test on image-token activations.
- I did not test a contrast vector on any adapter.

## How do autointerp and feature-labeling methods label a feature and score a label?

### Takeaway
Autointerp shows an LLM top activating examples, then scores the label on held-out examples. Most scorers need activations over examples. Our token scores over 591 images give exactly that. Two methods work on a word list alone: word intrusion and NPMI coherence. An LLM or human judge with a measured agreement is the third option.

### Cited Findings

**Labeling**
- Neuronpedia autointerp wraps EleutherAI [delphi](https://github.com/EleutherAI/delphi), formerly `sae-auto-interp`. [autointerp README](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/autointerp/README.md) [raw]
- `explain_default` uses `DefaultExplainer` with threshold 0.6 on activation examples. [explain route](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/autointerp/neuronpedia_autointerp/routes/explain/default.py) [raw]
- "MaxAct" in Neuronpedia means the `np_max-act` explainer type. The default UI type is `np_max-act-logits`. A webapp comment describes its sign-flipped form as "the most negative examples and the negative logits". This suggests that the method shows both examples and logits. [explanations pane](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/webapp/components/panes/explanations-pane.tsx), [autointerp.ts](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/webapp/lib/utils/autointerp.ts) [snippet]
- Delphi's `ContrastiveExplainer` adds non-activating examples. FAISS finds "hard negatives", which are non-activating examples that are semantically close to activating examples. [delphi README](https://github.com/EleutherAI/delphi) [raw]
- The J-lens can correct a MaxAct label. An SAE feature fires on technical prose, which suggests "technical exposition". Its J-lens words are fake, fraud and suspicious. Steering against it cuts the verbal test-scenario call-out from 28/50 to 10/50. [GW paper §A.24.3](https://arxiv.org/html/2607.15495v1) [raw]

**Scoring** ([Paulo et al. 2024](https://arxiv.org/html/2410.13928)) [raw for sections, not all tables]

| Scorer | Test | Needs |
|---|---|---|
| Simulation (OpenAI) | LLM predicts activations from the label. Scored mostly on activating examples. High cost | Per-token activations |
| Detection | LLM says if a whole sequence activates the feature, given the label | Activating and non-activating examples |
| Fuzzing | LLM says if marked tokens are correctly marked | Per-token marks |
| Surprisal | A label should lower a base model's cross-entropy on activating contexts | Log-probs |
| Embedding | Label embedding should be closer to activating than to non-activating examples | Embedding model |
| Intervention | Mean change in log p(label) for text generated with the intervention, against without it | A steering hook |

- The paper finds that intervention scoring explains features that existing methods miss. [Paulo et al.](https://arxiv.org/abs/2410.13928) [summary]
- Neuronpedia exposes fuzz, detection and embedding scorers, with 5 examples at a time. An example counts as activating when the sum of its values is above 0. [fuzz route](https://github.com/hijohnnylin/neuronpedia/blob/main/apps/autointerp/neuronpedia_autointerp/routes/score/fuzz_detection.py) [raw]
- Evaluation without a label: intruder detection uses 4 activating examples and 1 non-activating intruder. Example embedding scoring is a cheap proxy. Reported correlations over 56 latents: 0.48 with human intruder scores, 0.51 with LLM intruder scores. LLM and human intruder scores correlate at 0.84. [Paulo and Belrose](https://arxiv.org/html/2507.08473v1) [snippet]
- Delphi `IntruderScorer` has a "word" or "sentence" type. [intruder.py](https://github.com/EleutherAI/delphi/blob/main/delphi/scorers/classifier/intruder.py) [raw]
- Bills et al. wrote GPT-4 explanations of GPT-2 neurons and scored them by simulation. [Summary page](https://learn.aisingapore.org/2024/02/language-models-can-explain-neurons-in-language-models/) [snippet]
- Huang et al. audit those explanations. They report high error rates and little causal efficacy. [arXiv 2309.10312](https://arxiv.org/html/2309.10312v1) [snippet]

**Scores for a word list**
- Word intrusion: add an intruder word to the top words, and ask annotators to find it (Chang et al. 2009). [Lecture notes](https://courses.engr.illinois.edu/cs598jhm/sp2013/Slides/Lecture10.pdf) [snippet]
- NPMI coherence with an external reference corpus, and automated intruder detection (Lau, Newman and Baldwin 2014). Word intrusion and observed coherence correlate strongly. [ACL Anthology E14-1056](https://aclanthology.org/E14-1056/) [snippet]
- LatentLens judge: an LLM judge sees the image region and the top-5 descriptions. It labels each as concrete, abstract or global. A token is interpretable if at least one of the top-5 passes. Agreement with humans on 1,020 instances is Cohen's κ=0.68. [LatentLens §3.3](https://arxiv.org/html/2602.00462v5) [raw]
- Lens kurtosis scores whether a direction loads on few words. High kurtosis is necessary but not sufficient, because motor features also have it. A feature is "motor" if the actual next token is in its top-10 lens tokens more than 10% of the time, over its strongest activations. [GW paper §A.18](https://arxiv.org/html/2607.15495v1) [raw]
- Multilingual grouping is evidence of a concept: `limits`, `限`, `Limit`, `lím`, `limit` for one feature. [GW paper §A.18](https://arxiv.org/html/2607.15495v1) [raw]
- Sparse Readout Prism assigns a logit or logit difference to sparse readout features plus a residual. The sum is exact. Ablating a feature shifts the margin as predicted, with r² 0.83–0.93 against at most 0.06 for random directions. [SRP on HF](https://huggingface.co/papers/2609.01936) [summary]

### Inferences
- Treat one lens token at one (layer, position) as a feature, and treat its score over 591 images as the activation vector. All example-based scorers then apply.
- MaxAct for images: show the top and bottom images for a word. A VLM or human writes a label. Score it with detection on held-out images.
- Hard negatives matter here. Use visually close images with low token score, so that the label does not simply describe scene type.
- Word-list scores (intrusion, NPMI, LLM judge) measure coherence of the list. They do not show that the list reflects model content. A fixed-bias list can be coherent, because ` de`, ` la` and ` que` form a clear group.
- Intervention scoring is the closest autointerp match to the swap: steer with a word, then ask if the output shifts toward the label.

### Gaps
- I did not read the Bills paper or Chang 2009 directly.
- I found no published use of NPMI or word intrusion on J-lens readouts. Using them there is new.
- I found no study that scores contrast word lists with autointerp.

## Which methods show that a readout word reflects model content and not a lens artifact?

### Takeaway
No single test settles this. The best current recipe combines four checks: an interpretable contrast against a matched control, agreement across lenses and fit corpora, a causal test with random-direction controls, and a check for the answer-readout confound. Critics warn that swaps can pass by construction. Thus a passed swap is necessary evidence, not sufficient evidence.

### Cited Findings

**Prior lens failure modes**

| Lens | Failure mode | Source |
|---|---|---|
| Logit | Noise below about L58: `vah`, `valea`, `general`. Works only near the output. Reliability varies by model | [GW §A.5](https://arxiv.org/html/2607.15495v1) [raw]; [LatentLens](https://arxiv.org/html/2602.00462v5) [raw] |
| Logit | Author's caveat: it shows "some of what is going on, not all of it" | [nostalgebraist](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) [summary] |
| Tuned | "Skips to the output". Reads `red` from L38 on the multihop prompt | [GW §A.5](https://arxiv.org/html/2607.15495v1) [raw] |
| Tuned | Early predictive gain comes from the bias term alone, so it ignores the input there | [GW §A.6](https://arxiv.org/html/2607.15495v1) [raw] |
| Template | Also skips ahead. A few words appear without relevance | [GW §A.9.1](https://arxiv.org/html/2607.15495v1) [raw] |
| J-lens | Early third noisy. Single-token vocabulary. Top words at some positions make no sense | [GW §9.1](https://arxiv.org/html/2607.15495v1) [raw] |
| J-lens | Punctuation, brackets and whitespace dominate many readouts | [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [summary] |
| Fit corpus | English-fit and Chinese-fit lenses report different scripts for the same frozen Qwen3.5-9B state on 39 of 80 prompts | [SRP](https://arxiv.org/html/2609.01936v1) [raw] |

- Conflict: Belrose et al. call the tuned lens "more predictive, reliable and unbiased" than the logit lens, and their causal experiments support it. [Abstract](https://arxiv.org/abs/2303.08112) [snippet]. Gurnee et al. find it "less useful" for internal computation, and the J-lens has the worst next-token agreement by design. [GW §2.4, §A.6](https://arxiv.org/html/2607.15495v1) [raw]. The two sources measure different goals: output prediction against intermediate recovery.

**Two different centerings**
- Vocabulary-axis centering: subtract the mean J-lens vector over the vocabulary. On GPT-2, raw vectors have mean cosine 0.62–0.78 at layers 0–10, and this becomes 0.00. The readouts do not change, because the shared direction shifts every token equally. Interventions improve: 38/38 against 1/38 on verbal report. [GPT-2 post](https://www.greaterwrong.com/posts/GeKveaNdruATcj2Fj/even-gpt-2-passes-most-functional-global-workspace-tests) [raw]
- Activation-axis centering: subtract a mean activation before the read. Examples are GW concept vectors, GW templates ([§3.1, §A.9.1](https://arxiv.org/html/2607.15495v1)) [raw], and FVE, where FVE 0 equals the mean activation ([NLA](https://transformer-circuits.pub/2026/nla/)) [raw].
- Principal-component removal: on LLaVA-1.5-7B and Qwen2.5-VL-7B, image patches decode through the logit lens to structural syntax such as `,`, `a`, `in`. Removing the top PCs of a text subspace gives words such as `bench`, `refrigerator`, `deck`. [arXiv 2605.08245](https://arxiv.org/html/2605.08245v1) [raw]

**Causes of artifact and tests by construction**
- "By construction", lens vectors point at the tokens that the model is disposed to say, so swaps raise those tokens. The privilege tests may "largely measure which part points at the word". [GPT-2 post](https://www.greaterwrong.com/posts/GeKveaNdruATcj2Fj/even-gpt-2-passes-most-functional-global-workspace-tests) [summary]
- GPT-2 small passes the verbal report, injection, reasoning and part of the flexible tests. It lacks the band, ignition, kurtosis rise and amplification. [GPT-2 post](https://www.greaterwrong.com/posts/GeKveaNdruATcj2Fj/even-gpt-2-passes-most-functional-global-workspace-tests) [summary]
- Theory: the J-lens prefers directions that amplify over depth, stay constant across token positions, and hold across contexts. The study uses toy models only. [Ostrow 2026](https://arxiv.org/html/2610.31955) [raw for abstract and limits]

**The answer-readout confound**
- [solarkyle/jspace](https://github.com/solarkyle/jspace) works on Gemma-4-12B with 25,340 prompts. On a slice with a constant correct answer, a probe on internal features reaches AUC about 1.0 because it reads the answer the model is about to give. Honest varying-truth values on the same datasets are 0.55–0.78. The study is a pre-registered public repo with an audit by another LLM, not a peer-reviewed paper. [raw for README]
- The same repo saw frozen classifiers miss a pre-registered transfer test: mean increment −0.016, positive on 2 of 5 datasets. Lens-based signals can fail by task family. [raw for README]

**Motor separation**
- Late layers read the imminent output. The GW paper's test is whether the next token sits in the top-10 of a feature more than 10% of the time. [GW §A.18](https://arxiv.org/html/2607.15495v1) [raw]
- Top-down summoning design: same stimulus, two questions, metric Q2 − Q1 of positions with an expected word in the top-k, then swap label and foil at stimulus positions and measure the answer shift under each question. [experiments README](https://github.com/anthropics/jacobian-lens/blob/main/data/experiments/README.md) [raw]

**Agreement checks**
- Inter-lens comparison: cosine of same-token vectors, symmetric KL of readouts, and top-1 agreement by layer. All three lenses merge only in the last layers. [GW §A.6](https://arxiv.org/html/2607.15495v1) [raw]
- Feature-level agreement across fit corpora: the same SRP feature dominates under both lenses on 77 of 80 prompts (96%, CI 0.90–0.99). Nulls give 0/159 for unrelated tokens and 0/240 for shuffled prompts. [SRP](https://arxiv.org/html/2609.01936v1) [raw]
- A good readout tool agrees with an independent readout tool at the same positions: AUC 0.853 against NLA. [GW §A.21](https://arxiv.org/html/2607.15495v1) [raw]
- J-lens fit size hardly changes the result: it beats baselines with as few as 10 prompts. Recipe variants agree. [GW §A.7](https://arxiv.org/html/2607.15495v1) [raw]

### Inferences
- The two centerings answer different problems. Vocabulary-axis centering makes interventions work, but it cannot change ranks. It cannot remove ` de` from the top of a raw read.
- The fixed Gemma bias shows that the same tokens lead across activations. In a linear read this is about `W_U J h̄`, where h̄ is the mean activation. If so, a contrast or a subtraction of the background mean removes it to first order. We can test this directly by comparing the top words of `W_U J h̄` with the observed fixed words.
- Vocabulary-axis centering still matters for any swap we run. Check the mean cosine of our lens vectors first.
- Our prompt has a fixed text and a binary answer. That is the setting of the answer-readout confound. A yes-vs-no contrast at late layers will read the answer unless we exclude it.
- From the Ostrow theory, a persistent direction may read strongly even when it does not carry the local computation. This is my inference, not a claim of that paper.
- A word that survives the contrast may reflect image content and not the judgment. Yes-images and no-images differ in scene content, so a word such as "tree" can follow from the images alone. This is the main threat for a mechanism claim.

### Gaps
- I found no source that compares raw, z-scored and mean-subtracted J-lens readouts on a contrast task.
- I found no test of whether high-frequency tokens lead because of h̄ or because of J. This should be measured on our data.
- I found no validation of swaps at image-token positions.
- The GPT-2 post and the jspace repo are not peer reviewed.

## Is there work that applies a lens or activation readout to multimodal models, or to aesthetic, subjective or evaluative judgments?

### Takeaway
Multimodal logit-lens work is real and growing. It targets objects and hallucination, not judgments. I found no Jacobian lens work on image tokens. I found no lens study of aesthetic judgment. The nearest evaluative precedents are the Tylenol safety read, the preference-violation read, emotion directions, and a creativity contrast.

### Cited Findings

**Multimodal**
- LLaVA: visual tokens become more interpretable in vocabulary space with depth. The last token position then extracts the object information. Accuracy falls over 70% when object-specific tokens are removed. [arXiv 2410.07149](https://arxiv.org/abs/2410.07149) [snippet]
- The same paper is reported to find about 23.7% of object patches decoding to the correct class in each image's best layer, with a Qwen2VL-2B peak at layer 25.1 of 29. [arXiv PDF](https://arxiv.org/pdf/2410.07149) [snippet, not checked]
- LatentLens replaces the unembedding with nearest neighbors among contextual text activations. Over 15 VLMs, it labels 68% of visual tokens interpretable. EmbeddingLens gets 32%, and LogitLens 24%. Tuned Lens does not improve visual-token interpretability. [LatentLens](https://arxiv.org/html/2602.00462v5) [raw]
- Prompt order: per-patch logit-lens concept mass shifts with the question when the question comes first, but the answer token barely attends to it. Causal attention knockout confirms that the answer reads the question only when it follows the image. [arXiv 2607.15565](https://arxiv.org/abs/2607.15565) [snippet]
- Generic-token bias in VLMs: see 2605.08245 in the previous section. [raw]
- Gaze heads: a small set of mid-layer heads controls where a VLM looks and what it reports looking at. [GW §8 citing Gandikota and Bau](https://arxiv.org/html/2607.15495v1) [raw for the citation]
- Steering a vision SAE feature on an empty image, then asking the LLM what it sees, gives a causal label for a visual feature. [arXiv 2603.22593](https://arxiv.org/abs/2603.22593) [snippet]
- Logit-lens methods on layer stages in VLMs: [arXiv 2509.19191](https://arxiv.org/pdf/2509.19191), [V-SEAM 2509.14837](https://arxiv.org/pdf/2509.14837) [snippet].
- Oracle code notes that a vision-language Qwen3.5/3.6 keeps its text trunk under `model.language_model`. It works on the text trunk. [oracle.py](https://github.com/decoderesearch/interp-engine/blob/main/interp_engine/oracle.py) [raw]
- Corpus for fit: J-lens uses pretraining-like text. J++ uses WikiText-103. [GW §2.1](https://arxiv.org/html/2607.15495v1) [raw]; [J++ post](https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/paper-j-lens-jacobian-filtering-enables-more-faithful) [raw]

**Evaluative and subjective**
- Tylenol: at the `is` of "my pain is gone" (L58), the post-trained model reads safely, safe, maximum on 1000 mg, and unsafe, dangerous, WARNING on 8000 mg. The base model reads pain, now, feels on both. This is a same-prompt contrast with a change of one fact. [GW §6.1](https://arxiv.org/html/2607.15495v1) [raw]
- Reaction suites: a fixed list of reaction words per prompt, median rank over band layers per token, best rank per turn. [GW §6.1](https://arxiv.org/html/2607.15495v1) [raw]
- Preference violation: BUT appears after prefilled dispreferred options, on world-state pairs with no correct answer. Three controls separate it from general error tokens. [GW §6.2](https://arxiv.org/html/2607.15495v1) [raw]
- Sentiment classification is "essentially unaffected even under heavy ablation", as are other shallow classification tasks. [GW §3.5.2](https://arxiv.org/html/2607.15495v1) [raw]
- The same-direction emotion result: internal directions for an emotion hold whether it is attributed to the user, a third person, or the Assistant (Sofroniew et al. 2026). [GW §6.1 citing this work](https://arxiv.org/html/2607.15495v1) [raw for the citation]
- Creativity: internal state differences between "boringly" and "creatively" prompts give a measure that "corresponds strongly with human judgment". [arXiv 2412.06060](https://arxiv.org/abs/2412.06060) [snippet]
- Drift Calibrator: difference-in-means vectors and cosine similarity track six traits. [Neuronpedia](https://www.neuronpedia.org/blog/jpp-drift-ui) [summary]
- Hobby repos exist, for example [eric-tramel/jlens-mood](https://github.com/eric-tramel/jlens-mood) (emotion axis) and [hareem-arshad/Jacobian-lens-bias](https://github.com/hareem-arshad/Jacobian-lens-bias). I did not evaluate them.

### Inferences
- The sentiment result raises a real risk for us. A "cool" judgment could be automatic and run outside the J-space. An absence of readout content would then not show an absence of a mechanism, and an ablation might have no effect on the answer.
- J++ and J-lens were fit on text, so image-token positions are outside the fit distribution. Text positions after the image are closer to the fit distribution. Reads at the final text tokens are thus better supported than reads on image patches.
- The VLM debias result suggests that structural tokens at image positions may be a real over-alignment and not only lens noise. Our Gemma bias is function words, so this is a plausible link. It is not established.
- ` de`, ` la` and ` que` are Romance function words. The GW paper reports common French words as a context signal in readouts ([§A.24.3](https://arxiv.org/html/2607.15495v1)) [raw]. The SRP result shows that fit corpus changes the reported language. One hypothesis is thus a language or script channel. High token frequency is another, as the J++ post suggests. Test both.
- Question-before-image and image-before-question prompts can differ in what image tokens carry. Check our prompt order before comparing positions across the 2 models.

### Gaps
- I found no study that applies a J-lens or J++ to image tokens.
- I found no lens work on beauty, taste, or "cool" judgments. The search is not exhaustive. I ran one extended search for this topic.
- I found no work on urban perception with lens readouts.
- I did not read the full Sofroniew paper or the gaze heads paper.

## What can a study of ours reuse?

### Takeaway
Reuse the paper's named-concept score, a matched control design, and the swap. Report the raw bias as a measured object, not as noise. Separate three signals: the fixed lens component, the image content, and the answer.

### Cited Findings
- Same-stimulus, two-question contrast with Q2 − Q1 and a swap: [experiments README](https://github.com/anthropics/jacobian-lens/blob/main/data/experiments/README.md) [raw]
- Word set minus 100 control words, averaged over the band: [GW §A.21](https://arxiv.org/html/2607.15495v1) [raw]
- Swap target from the same category, outside the top-10 outputs: [GW §3.1](https://arxiv.org/html/2607.15495v1) [raw]
- Answer-readout decomposition: [solarkyle/jspace](https://github.com/solarkyle/jspace) [raw]
- Judge validation against humans with Cohen's κ: [LatentLens](https://arxiv.org/html/2602.00462v5) [raw]
- Wilson intervals and paired differences: [GW paper](https://arxiv.org/html/2607.15495v1) [raw]; bootstrap over items: [jpp_lens README](https://github.com/safety-research/jpp_lens) [raw]

### Inferences

**Suggested analysis order for the bias**
1. Compute `W_U J_l h̄_l` per layer, with h̄ from a background set. Compare its top words with the fixed words.
2. Run the same on a text-only background. This shows whether the bias needs an image.
3. Report raw top words once. Then use contrasts. Do not hide the bias.
4. Compare J-lens and J++ on Gemma. `gemma-4-31b` has both in the Neuronpedia repo. This separates a fit effect from a model effect.
5. Check the mean cosine of lens vectors before any swap. Center over the vocabulary if it is high.

**Suggested contrasts**
1. Same image, different question. For example "Is this place cool?" against a neutral question such as "Is this place big?". This removes most image content by design.
2. Same question, yes-images against no-images. This keeps image content, so match the images on scene content, or use hard negatives.
3. Report contrast 1 as the primary mechanism evidence. Contrast 2 alone cannot separate content from judgment.

**Layer handling**
1. Do not compare raw logits across layers. Each layer has its own lens, and no source gives a shared scale.
2. Z-score each token within a layer against the background, then aggregate. The GW paper uses a z-score against a no-instruction baseline in §A.12. This use is an extension.
3. Report ranks with a full-vocabulary denominator next to any aggregate.
4. Find each model's band with kurtosis and autocorrelation against a shuffled null. Do not copy L38–L92. GPT-2 shows no clear band.
5. Mark late layers as motor. Exclude `yes` and `no` variants from the word list and report them separately.

**Word-level validation**
1. Fix the word list before the analysis. Split the 591 images into discovery and held-out sets.
2. Score each word by its score over images. Use MaxAct images and a detection score on held-out images.
3. Use word intrusion or an LLM judge for list coherence. Measure judge agreement with a human sample.
4. Require multilingual or synonym support where it exists. One token alone is weak.
5. Test agreement between Gemma and Qwen with readout features or word groups, not token identity.

**Causal validation**
1. Swap a contrast word with a same-category foil at band layers, on final text tokens. Use `yes`/`no` flip rate as the outcome, with a Wilson interval.
2. Run ablation with a random-direction control and a matched-norm control.
3. Report workspace loading next to swap success, because low loading predicts failure.
4. Expect a possible null result. Sentiment classification resists ablation, so a judgment may not route through the J-space.
5. State that a passed swap is partly expected by construction, as the GPT-2 post argues. Pair it with the no-lens test: a mean-difference patch from the contrast, with a sweep over layer.

**Statistics**
- Use the simplest textbook tests that fit each comparison: paired t or sign test for paired contrasts, binomial with Wilson intervals for rates, and Bonferroni across words. This matches the project rule in memory. The sources use the same simple tools.

### Gaps
- The neutral control question needs a pilot. A poor control can itself shift image attention.
- The discovery and held-out sizes need a power check. The sources use small sets (9 to 100 items) for most evaluative suites, and they do not justify the sizes.
- No source sets a threshold for a "valid" readout word. Any threshold we use is ours.
