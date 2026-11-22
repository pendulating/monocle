# Jacobian-lens literature review

Date: 2026-11-02. Scope: how the FAccT paper (`artificial-aesthetic/paper/`)
uses the Jacobian lens to study aesthetic judgments, and which other
computer-science papers apply the lens for an empirical analysis.

All quotations in this document are verbatim. We copied them from the
source page on 2026-11-02. A quotation that we could only get from a tool
summary, not from the page text, carries the mark **[unverified]**. Check
it against the page before you cite it.

## 0. Peer-review status

No source in this review is peer-reviewed. We checked each arXiv listing
on 2026-11-02 for a journal reference, a venue comment, or an OpenReview
page. None has one.

| Source | Type | Venue field on arXiv | Status |
|---|---|---|---|
| Gurnee et al., workspace paper | Lab report and arXiv 2607.15495 | Comments absent, Journal-ref absent | Not peer-reviewed. Three invited commentaries, not reviews. |
| Blank, Bhatia, Nanda, R-lens | LessWrong post | Not on arXiv | Not peer-reviewed |
| Ayonrinde and Lindsey, J++ Lens | LessWrong post | Not on arXiv | Not peer-reviewed |
| Ostrow et al., theory | arXiv 2610.31955 | "27 pages, 19 figures" | Not peer-reviewed |
| Mosafer, safety J-space | arXiv 2610.04316 | "26 pages" | Not peer-reviewed. Single author. |
| Wang and Reid, looped transformers | arXiv 2609.01924 | Comments absent | Not peer-reviewed |
| Liu, lie readouts | LessWrong post and repo | Not on arXiv | Not peer-reviewed |
| Standard Logits, R-lens and S-lens | Model card | Not on arXiv | Not peer-reviewed |
| Sönning, 12B workspace | GitHub thread | Not on arXiv | Not peer-reviewed |
| ec75hash, template lens | Model card | Not on arXiv | Not peer-reviewed |
| tao-hpu, replication | Repo | "arXiv link forthcoming" | Not peer-reviewed |
| Nanda, review | LessWrong post | Not on arXiv | Not peer-reviewed |
| Rahbar, Ignition Index | arXiv 2608.05160 | "26 pages, 10 figures" | Not peer-reviewed. Does not use the lens. |
| Pettyjohn et al., OmniLens | arXiv 2608.10260 | Comments absent | Not peer-reviewed. Does not use the lens. |

The lens is four months old. No paper that uses it has passed review yet.
The paper's own claim rests on the method, not on these results. Cite the
Anthropic report and the J++ post as the method sources. Cite the rest only
as context, and mark them as preprints in the text.

What has independent support:

- The core readout claims of Gurnee et al. replicate on Qwen3.6-27B in
  Nanda's review (4.1). Poetry and arithmetic did not replicate.
- The J++ recall numbers for the three lenses the paper uses are on the
  Neuronpedia card (2.3). They come from the lens authors, not from a third
  party.
- The local project measured the calibration critique and the R-lens
  itself. See `causal-lens-retrospective.md` and `r-lens.md`.

### 0.1 Who the authors are

Checked on 2026-11-02. The affiliation column is verbatim from the first
page of each arXiv paper. The record column comes from Google Scholar,
DBLP, the ML Anthology, ICLR and NeurIPS pages, and institution pages.
"Not verified" means we found no page that confirms the item.

| Paper | Printed affiliation | Author record | Weight |
|---|---|---|---|
| Gurnee et al., workspace (2607.15495) | Anthropic (all 16 authors) | Gurnee: PhD, MIT Operations Research Center, Feb 2025 ([DSpace](https://dspace.mit.edu/handle/1721.1/158869)); "Language Models Represent Space and Time", ICLR 2024 ([poster](https://iclr.cc/virtual/2024/poster/18036)); "Finding Neurons in a Haystack", TMLR 2023 ([arXiv 2305.01610](https://arxiv.org/abs/2305.01610); the ML Anthology lists it under TMLR 2023). Lindsey: PhD, Columbia, 2019 to 2023, advisor Litwin-Kumar ([Krell](https://www.krellinst.org/csgf/profile/lindsey2019)); NeurIPS 2020 and 2022 papers with Litwin-Kumar ([DBLP mirror](https://rankdblp.com/authors/__name__ashok_litwin-kumar.html)). | Strong. Industry lab, authors with peer-reviewed records. |
| Ostrow et al., theory (2610.31955) | "Department of Brain and Cognitive Sciences, Massachusetts Institute of Technology" (Ostrow, Qian, Fiete); "Department of Physics, Cornell University" (Liu) | Ostrow: "Beyond Geometry ... Dynamical Similarity Analysis", NeurIPS 2023 ([proceedings](https://proceedings.nips.cc/paper_files/paper/2023/file/6ac807c9b296964409b277369e55621a-Paper-Conference.pdf)). Fiete: professor of brain and cognitive sciences, MIT; director, K. Lisa Yang ICoN Center; Swartz Prize 2022; HHMI Faculty Scholar 2016 ([McGovern](https://mcgovern.mit.edu/profile/ila-fiete/)). Qian, Liu: not verified. | Strong. Accredited institutions, senior author with a long record. |
| Wang and Reid, looped (2609.01924) | "Fin AI Research", emails at intercom.io | Reid: PhD, University College Dublin, 2013; VP of AI at Intercom ([UCD](https://www.ucd.ie/cs/news/alumnusinconversation-fergalreidvpofaiatintercomphd2013/)). Wang: "Tiny Recursive Reasoning with Mamba-2 Attention Hybrid", ICLR 2026 workshop on Latent and Implicit Thinking ([ICLR](https://iclr.cc/virtual/2026/workshop/10000773)). No main-track paper found for either. | Medium. Industry group, one workshop paper, one PhD. |
| Mosafer, safety (2610.04316) | None printed. Footnote email at umail.umz.ac.ir. | One SSRN preprint, Oct 2026, with two co-authors at the University of Mazandaran ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7615929)). No peer-reviewed paper found. No Scholar or DBLP page found. A different person, Mohammad Mosaferi at Tabriz, is not this author. | Weak. Single author, no printed affiliation, no reviewed record. Do not cite for a claim. |
| Rahbar, Ignition Index (2608.05160) | "Dialpad, Inc." and "Independent Research" | Senior Applied Scientist, Dialpad ([site](https://srahbar.com/), [Scholar](https://scholar.google.com/citations?user=JYGPswkAAAAJ&hl=en)). A 2019 thesis and a 2018 paper on Scholar. An EMNLP W-NUT 2026 workshop paper is claimed on LinkedIn and not verified. | Medium to weak. Does not use the lens, so it does not matter here. |
| Pettyjohn et al., OmniLens (2608.10260) | "University of Chicago", "Illinois Institute of Technology", "Argonne National Laboratory", "Colorado School of Mines" | Foster: Globus co-founder, Argonne division director, SC25 Test of Time Award ([HPCwire](https://www.hpcwire.com/off-the-wire/globusworld-2018-program-and-speakers-announced/)). Others not checked. | Strong. Does not use the lens, so it does not matter here. |
| Ayonrinde and Lindsey, J++ (post) | Post byline: Lindsey, Anthropic. Ayonrinde: no affiliation on the post. | Ayonrinde: Scholar lists UK AI Security Institute ([Scholar](https://scholar.google.com/citations?user=j40ixccAAAAJ&hl=en)); co-author of SAEBench, ICML 2025; ICLR workshop 2025; NeurIPS workshop 2024 ([ML Anthology](https://mlanthology.org/authors/a/ayonrinde-kola/)). Code is under the GitHub org `safety-research`. | Medium to strong. One main-track paper as co-author. The lens weights come from these authors. |
| Blank, Bhatia, Nanda, R-lens (post) | Post names MATS scholars with Nanda. | Nanda: leads the language model interpretability team at Google DeepMind (the Anthropic commentary PDF states this). Blank and Bhatia: not verified. | Medium. Known mentor, unverified students. |
| Liu, lie readouts (post) | None | Not verified. | Weak. |
| Sönning, ec75hash, tao-hpu, Standard Logits | None | Not verified. Two of the four state that they wrote the work with Claude. | Weak. Context only. |

Two more notes:

- The Ignition Index listing has an August identifier (2608) and a May
  submission date. We did not resolve this.
- The Gurnee et al. report had three invited commentaries. The commentary
  authors are Dehaene (Collège de France), Naccache (Sorbonne), Butlin,
  Shiller, Plunkett, and Long (Eleos AI Research), and Nanda (DeepMind).
  These are reviews by name, but they are not anonymous peer review.

## 1. What the paper does with the lens

The paper reads the judgment of three models along the prompt and the
layers. It does not fit a lens. It uses three pre-fit J++ lenses.

| Item | Value |
|---|---|
| Models | Gemma-4-31B, Qwen3.5-9B, Qwen3.6-27B |
| Lens | J++ Lens (Ayonrinde and Lindsey, 2026), pre-fit, from Neuronpedia |
| Prompt | `{image} Is this place cool? Answer with one word: yes or no.` Repeated with `nice` and `ordinary`. |
| Images | 591 street-view images of New York City |
| Token positions | `after_image`, `place`, `cool`, `?`, `model`, the thought tokens, `<channel|>`, last token |
| Contrasts | C1 image minus mean; C2 mean minus no-image; C3 borough minus the rest; C4 top third by answer margin minus bottom third; C5 the reverse |
| Beside the lens | Ridge decoder of the answer margin on the raw residuals, five folds by neighborhood |

The method section states the claim the paper tests (`sec/3_methods.tex`):

> "The Jacobian lens (J-lens) family of work from the field of mechanistic
> interpretability claims to uncover a 'global workspace' in language
> models. This global workspace, dubbed the 'J-space', is shown to be used
> by models when asked to think about a concept, and readouts at the same
> token position across many successive layers can show intermediate
> 'hops' in the reasoning process a model uses to answer a prompt"

The paper defines its terms:

> "A \emph{readout} is the ranked list of vocabulary tokens that the lens
> gives at one token position and model layer. We keep lower-case English
> tokens, which we call \emph{words}. A \emph{mean readout} averages the
> lens logits behind this list over a set of images, and then ranks the
> words again."

The results section gives the finding in three parts (`sec/4_results.tex`):

> "Three parts separate in the data, and all three replicate across models
> ... First, the image alone carries most of the verdict: a ridge decoder on
> the mean image patch, before any question, recovers the answer margin at
> out-of-fold $R^2=.82$ ($.75$, $.56$) ... Second, the question adds a step:
> the gain over the mean patch rises from $.07$ ($.08$, $.07$) at 'place' to
> $.12$ ($.18$, $.19$) at 'cool'. Third, at 'cool' the image picks the
> \emph{sense} of the word: the largest axis of the neighborhood readouts at
> layer 24 holds 63\% (49\%, 57\%) of the variance, and runs from
> temperature words (coolant, humidity) to social words (boutique, hipster)."

The paper keeps the causal claim narrow:

> "The prompt is fixed, so this is an order along the prompt, not a causal
> order inside the model."

The borough contrast is the paper's most specific use of the lens:

> "For the Bronx, the contrast holds music and craft words through layer 16
> ("spotify", "banjo", "handmade"), and then sharply diverges at layer 20
> from the generally-positive tone of ... the mean readout: "graffiti",
> "fucking", "shopping", "gallery", and "homeless"."

> "We read this split as the sense of the word, not as a difference in
> judgment. Where the model reads a scene as urban, 'cool' takes its social
> sense; where it does not, as in the average Staten Island or Queens
> scene, 'cool' takes its temperature sense."

The paper's use of the lens is new in three ways. No published work that we
found reads the lens on image inputs. No published work reads it on a
subjective judgment. No published work averages readouts over a population
of inputs and ranks the differences (the contrasts C1 to C5).

## 2. The lens family: primary sources

### 2.1 The Jacobian lens (Gurnee et al., Anthropic, 2026)

- Page: <https://transformer-circuits.pub/2026/workspace/>
- arXiv: <https://arxiv.org/abs/2607.15495>
- Code: <https://github.com/anthropics/jacobian-lens>
- Authors: Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski,
  Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan
  Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel
  Ameisen, Joshua Batson, Jack Lindsey. Published July 6, 2026.

Definition:

> "Concretely, it computes, for each layer, the average linearized effect
> of an activation on the model's likelihood of producing a particular
> token (now or in the future), averaging over a large corpus of contexts
> (see Methods for details)."

> "The averaging step is key, as it distinguishes representations that are
> verbalizable—poised to be spoken about, should the occasion arise—from
> those that merely happen to be verbalized in one particular context."

> "The result is a single d_model × d_model matrix per layer that maps from
> a source layer ℓ to the final layer L."

> "The logit lens applies the unembedding matrix directly to the
> intermediate residual stream, which corresponds to setting J_ℓ = I in our
> formulation."

The workspace claim and the layer band:

> "The analyses above all identify a similar range of layers, beginning
> about a third of the way through (~L38) and ending shortly before the
> output (~L92), as the region where the J-space carries persistent,
> abstract content distinct from both the input tokens and the imminent
> output."

> "The excess variance explained is modest, never exceeding 10%, indicating
> that the model's activations are dominated by information outside the
> J-space."

The multi-hop experiment that the paper cites (spider, legs, 8):

> "When we swap the spider lens vector for ant, the model's top output
> changes from "8" to "6", the number of legs on an ant."

> "The Jacobian-lens coordinate swap succeeds in 54% of trials on Haiku
> 4.5, 70% on Sonnet 4.5, and 70% on Opus 4.5."

> "Across 90 two-hop prompts, swapping the probes' J-space components flips
> the model's answer to the swapped-in intermediate on 61% of trials,
> matching the 60% achieved by swapping the raw J-lens token vectors as in
> the preceding experiments."

> "Swapping the non-J-space components, despite carrying the bulk of the
> variance, flips the answer on only 28% of trials"

Applied uses in the same paper (alignment audits):

> "For the cases we have examined, however, the J-lens often makes silent
> strategic reasoning and situational awareness visible around the token
> positions where the relevant decisions are being made."

> "The ablation reduces verbalized eval-awareness, the rate at which the
> model explicitly states in its reasoning that the scenario is a test,
> from 71% to 3%"

> "Under no ablation, the model attempts blackmail in 0 of 180 rollouts;
> under ablation, it does so in 13 of 180 (7%; Figure 36, bottom left),
> matching previous findings that evaluation awareness inhibits misaligned
> actions."

> "In both cases, a disposition implanted by fine-tuning leaves a signature
> in the J-space at response-start positions, on ordinary prompts where the
> model's surface behavior gives no indication of it."

Limits that the authors state:

> "The Jacobian lens is an imperfect tool, which we believe only
> approximately and incompletely captures the model's underlying workspace
> structure."

> "For instance, it only identifies vectors associated with concepts that
> correspond to single tokens in the model's vocabulary, but many important
> concepts correspond to multiple tokens"

> "Note that in roughly the first third of the model, the readouts are
> noisy and largely uninterpretable"

> "However, because these metrics are derived from the J-lens, it is
> possible that these layer-wise effects are artifacts of the J-lens
> methodology, rather than reflecting something fundamental about the
> model."

### 2.2 R-lens (Blank, Bhatia, Nanda, 2026)

- Post: <https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers>
- Weights: <https://huggingface.co/camilablank/workspace-lenses>
- Date: 5 Aug 2026.

> "We introduce the R-lens: a drop-in replacement for J-lens that produces
> clearer readouts on earlier layers."

> "R-Lens is identical to J-Lens, except that we make minor and
> low-overhead changes to the backwards pass, following
> layerwise-relevance propagation, allowing us to reduce the propagation of
> errors."

> "On the token "sushi", R-lens surfaces the intermediate "Japan" at layer
> 2, while J-lens does not surface it until layer 14"

> "There is no R-lens advantage for both the smallest dense and MoE models
> we tested."

### 2.3 J++ Lens (Ayonrinde and Lindsey, 2026)

- Post: <https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace>
- Mirror: <https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace>
- Code: <https://github.com/safety-research/jpp_lens>
- Weights: <https://huggingface.co/koayon/jpp-lenses> and
  <https://huggingface.co/neuronpedia/jacobian-lens>
- Date: 8 Oct 2026. Authors: Kola Ayonrinde, Jack Lindsey.

This is the lens that the paper uses. The paper's bib entry
(`kolaayonrinde2026lensjacobian`) names only Ayonrinde and has no URL. Add
Lindsey and the URL above.

> "The J++ Lens filters noisy gradients before they enter the averaged
> Jacobian."

> "To isolate such source activations, we partition the activation space at
> each layer into clusters with k-means clustering. We then fit one
> averaged Jacobian per cluster, using only the source activations in that
> cluster."

> "The resulting map is a drop-in replacement for the J-Lens with no extra
> inference cost, but with noisy Jacobians effectively filtered out."

> "The J++ Lens surfaces the correct intermediate readouts at 55.2%
> recall@10, compared to 35.7% and 37.7% for the J-Lens and R-Lens
> respectively."

> "We see the strongest performance uplift in early layers where the J-Lens
> struggles the most. We see the J++ Lens' improved performance as evidence
> that there is useful information in the workspace earlier than was
> suggested by Gurnee et al. (2026)."

> "Unless stated otherwise, all experiments use Qwen3.6-27B. We also
> evaluate readouts on five further models: Qwen3.5-9B, Gemma 4 31B, Olmo
> 3 32B, Qwen3.5-122B-A10B and DeepSeek-V4-Flash."

> "We fit each lens on 64 sequences of 128 tokens from WikiText-103,
> skipping the first 16 positions of each sequence."

Limits:

> "The J++ Lens reads out only single vocabulary tokens."

> "Single-token lenses depend heavily on the tokeniser: for example, we
> could not evaluate an arithmetic task on Qwen models, which tokenise
> numbers digit by digit, so multi-digit numbers were not visible in a
> readout."

> "Yang et al. (2025) show that models sometimes "shortcut" to the final
> answer without representing the intermediate variable, and in these cases
> we would not expect any lens to surface the intermediate variable
> reliably."

The Neuronpedia model card (<https://huggingface.co/neuronpedia/jacobian-lens>)
lists the three lenses the paper uses, with the recall@10 of each:

| Model | Recall@10 | Fitted by |
|---|---|---|
| Qwen/Qwen3.6-27B | 55.2% | Ayonrinde and Lindsey |
| Qwen/Qwen3.5-9B | 55.0% | Ayonrinde and Lindsey |
| google/gemma-4-31B | 58% (poetry task dropped) | Ayonrinde and Lindsey |

The card gives the fit corpus as WikiText-103 and the readout rule as
`logits = unembed(h @ J[ℓ].T)`. It also states: "We cast the maps from
fp32 to fp16 (relative error 2.7e-4 or less; 1.1e-3 for Gemma 4 31B, which
has very small values in some layers)."

### 2.4 A mathematical theory of the Jacobian lens (Ostrow et al., 2026)

- arXiv: <https://arxiv.org/abs/2610.31955>
- Authors: Mitchell Ostrow, Daoyuan Qian, Toni J.B. Liu, Ila Fiete.
  Submitted 22 Oct 2026.

> "we introduce a mathematical theory of the J-lens, and show that it
> preferentially identifies coherent features -- directions that act like
> leading, spatially coherent Lyapunov modes."

> "Small perturbations along these directions amplify across layers, remain
> roughly constant across token positions, and are consistent across
> different contexts."

> "We validate our analysis on a Bayes filter and a toy transformer, each
> optimally predicting next tokens emitted from a Hidden Markov Model
> (HMM)."

> "This is commensurate with Gurnee et al. (2026) finding that the J-space
> is never more than 10% of activation variance."

This theory matters for the paper. It says the lens finds directions that
stay constant across token positions. The paper's contrasts C1 to C5 use
exactly that property: they compare the same position across images.

## 3. Empirical applications of the lens by others

We rank these by how close they are to the paper's use.

### 3.1 Safety-content accessibility in J-space (Mosafer, 2026)

- arXiv: <https://arxiv.org/abs/2610.04316>
- HTML: <https://arxiv.org/html/2610.04316>
- Code: <https://github.com/mohsafer/jspace-safety-testbed>
- Submitted 3 Oct 2026. Single author.

This is the closest published analogue. It scores readouts with fixed word
lists and compares two prompt populations with a rank statistic. The paper's
contrasts C4 and C5 do the same with the answer margin.

> "Jacobian-space (J-space) readouts, linear maps from hidden states to the
> output vocabulary via the model's input-output Jacobian, have been
> proposed as a window on behaviorally accessible internal content, and a
> first safety protocol (JADR) showed that danger recognition is readable
> there."

> "Per layer we take the top-k (k=100; top-200 stored) tokens of the lens
> logits and score them with six frozen English lexicons (safety,
> compliance, evasion, softening, hedging, harm), scored by the
> rank-discounted counter"

> "Two contrasts are computed by Mann–Whitney rank AUC between danger
> prompts (StrongREJECT ..., 313) and safe controls (XSTest ..., safe half,
> 250): SafetyAUC on the safety axis and ComplAUC on the compliance axis"

> "Across four small model pairs (135M to 1.5B, three lens-fit seeds),
> tuned checkpoints concentrate recognition in upper-middle layers, and at
> 0.5B DPO installs refusal while J-space recognition drops to chance:
> training can widen the accessibility gap exactly where behavioral safety
> looks best."

> "steering shows the amplification profile is descriptive, not causal"

### 3.2 Lie readouts at the newline token (Liu, 2026)

- Post: <https://www.lesswrong.com/posts/bhvaAbSPZgiG3oJPu/on-qwen-the-j-lens-readouts-at-the-newline-token-reveal-lies-4>
- Code and data: <https://github.com/MPJL9/jlens-lie-readouts>
- Date: 19 Oct 2026. Author: Jimmy Liu.

This work reads the lens at a template token after the answer, and counts
word families. It is the nearest analogue to the paper's readouts at
`<channel|>` and the last token.

> "We put four models (Qwen3.6-27B, two Gemma-3 models, gpt-oss-20b) in
> MASK's pressure scenarios, where a system prompt gives the model a role
> and a reason to lie about a fact it was told."

> "At the newline token right after Qwen's response, the `lie` family
> appears after 84% of lies and 9% of truths (78% and 7% on the held-out
> scenarios)."

> "Eight of the ten families used as a detector catch 86% of held-out lies
> but mistakenly flag 13% of truths (AUROC 0.87)."

> "On Qwen3.6-27B, the deception-related signal in the residual stream is
> concentrated at the newline token among the five tokens at the turn
> boundary."

> "The signal does not carry to the other models."

### 3.3 R-lens and S-lens on Gemma 4 12B with image inputs (Standard Logits, 2026)

- Card: <https://huggingface.co/stdlogits/rs-lenses>
- Code: <https://github.com/stdlogits/neuron-lens>
- Date: 2026-11-01.

This is the only lens fit on image inputs that we found. It fits on a
Gemma 4 model, which is the paper's model family.

> "Linear lenses that turn a vector inside a model into a ranking over
> tokens, for `Qwen/Qwen3-8B` and `google/gemma-4-12B`, fitted with one
> estimator on the same 1,000 web documents"

> "Gemma 4 takes images and sound through linear embedders, so its lenses
> are also fitted on images and on sounds"

> "The R-lens predicts what steering a neuron does."

> "For Gemma 4 12B, from L11: 0.53 at L12-L23, 0.82 at L24-L35 and 0.97 at
> L36-L47, ahead of the logit lens."

> "The audio S-lens read is at chance."

> "This is stability within the corpus, not independence from it."

### 3.4 Looped transformers under the Jacobian lens (Wang and Reid, 2026)

- arXiv: <https://arxiv.org/abs/2609.01924>
- HTML: <https://arxiv.org/html/2609.01924v1>
- Submitted 1 Sep 2026. Authors: Wenlong Wang, Fergal Reid.

> "We extend the Jacobian lens to iterated architectures using a
> virtual-unrolling adapter."

> "We apply the full workspace suite — lens fitting, readout, and eleven
> causal experiment families — to Ouro-2.6B (48 layers looped 4 times,
> deeply supervised) and Huginn-0125 (a 4-layer core recurred 16 times,
> trained for latent reasoning), using Qwen3.6-27B (64 untied layers) as
> the standard baseline."

> "Huginn reroutes self-computed content into words nearly perfectly (98%)
> yet never verbalises externally injected content (0/97), which only
> explicitly deep-supervised Ouro achieves"

> "Across the boundary the Italy projection drops to ∼0.1 of the target
> state's value with no concept-versus-control discrimination"

### 3.5 Workspace geometry of a 12B model (Sönning, 2026)

- Thread: <https://github.com/google-deepmind/gemma/discussions/725>
- Code: <https://github.com/soenning-ai/jlens-small-models>
- Date: 12 Jul 2026. Author: Sebastian Sönning (community, not DeepMind).

> "Measured with Anthropic's open-source Jacobian-Lens on a single 12 GB
> consumer GPU, Gemma 4 12B shows the "global workspace" signature reported
> for frontier models — participation ratio ≈ 6–7 directions (of 3840) in
> layers 33–40, with the top 25 directions carrying ~86 % of transport
> energy."

> "The from-scratch control has clean mid-stack semantics (France→Paris,
> triangle→vertices) but no workspace geometry — its transport stays
> diffuse (PR up to ~2,000)"

> "Short answer: a 12B distilled model has the signature, and it has it
> more strongly than the paper's frontier models. A from-scratch model of
> comparable size does not."

> "The lower half of Gemma 4's stack was never measured — it doesn't fit in
> 12 GB."

> "n = 1 — a data point, not a law."

### 3.6 Template lens for Qwen3.6-27B (ec75hash, 2026)

- Card: <https://huggingface.co/ec75hash/template-lens-qwen3.6-27b>
- Dates: fit 2026-08-19 and 2026-08-20, correction 2026-09-23.

> "The J-lens is healthy on single tokens and degrades sharply with concept
> length; the template lens stays flat — the paper's core claim,
> replicated."

> "Top-1 word identification among 12 (chance 0.083) climbs monotonically
> with depth: 0.31 (embeddings) → 0.63 (L12) → 0.83 (L36) → 0.979 (final
> layer)."

> "Nothing here yet separates talking about deception from being
> deceptive."

### 3.7 Independent replication on small open models (tao-hpu, 2026)

- Repo: <https://github.com/tao-hpu/jspace-replication>

> "An independent, third-party replication of the core experiments in
> Verbalizable Representations Form a Global Workspace in Language Models
> (Anthropic, 2026) on small open-weights models, with systematic
> bookkeeping of which claims hold up and which do not."

> "covert content survives almost exclusively for context registers
> (language identity, intended form of a typo), not content plans."

> "a mid-band entity swap rewrites the model's restatement of the question
> itself, stably across a 1.7B-14B ladder, while self-report about the edit
> changes shape at every scale."

### 3.8 Tooling

| Tool | Link | Note |
|---|---|---|
| Neuronpedia J-lens docs | <https://docs.neuronpedia.org/jacobian-lens> | "We have pre-fitted J-lenses for 36+ models, and the largest is Llama 70B." Lens choices: Jacobian, J++, Logit. "By default the J-lens skips about the first 1/3 of layers, because it is not reliable there." |
| Neuronpedia J++ and drift UI | <https://www.neuronpedia.org/blog/jpp-drift-ui> | UI research post |
| nnsight tutorial | <https://nnsight.net/tutorials/mini-papers/jacobian-lens/> | "pre-fitted lenses for 38 open models are published at `neuronpedia/jacobian-lens`." Uses gemma-2-2b. |
| nnsight vLLM example | <https://nnsight.net/vllm/examples/jacobian-lens/> | Lens readout with vLLM |
| interpretune issue | <https://github.com/speediedan/interpretune/issues/225> | Planned read, probe, and steer ops |
| Workspace-lens experiments | <https://github.com/BrennanLagasse/workspace-lens-experiments> | Experiments on J, R, and J++ lenses |

## 4. Commentary and reviews

### 4.1 Neel Nanda, review and replication on Qwen3.6-27B

- Post: <https://www.lesswrong.com/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper>
- Date: 6 Jul 2026. Replication by Camila Blank and Agam Bhatia.

> "We have replicated the core claims on Qwen 3.6 27B, and also share
> preliminary evidence of extending this work by finding abstract
> "interpretative meta-tokens"..."

> "At first glance multihop factual recall seems weak but effective, but on
> further examination swapping the answer turned out to strictly dominate."

> "Poetry, and arithmetic both failed to replicate, but this is plausibly
> due to experimenter error or worse model capabilities."

> "I do not expect it to reliably flag everything important going on, and I
> expect it to have many false positives, whether from errors of the method
> or our misinterpretation of it (and I expect the authors would agree with
> me)."

> "I expect J-Lens would be best used as something to run on prompts in the
> initial steps of an investigation in a more exploratory way..."

A reader comment on that post reports a test of the released Qwen2.5-7B lens
against a TF-IDF baseline: the lens "does not clear a TF-IDF baseline
(0.690) with my pre-registered margin." This is one comment, not the post.

### 4.2 Dehaene and Naccache, cognitive neuroscience

- PDF: <https://unicog.org/wp_2025/wp-content/uploads/2026/07/Dehaene-and-Naccache-Workspace-commentary-on-Gurnee-Lindsey-June-2026.pdf>
- Also in the Anthropic collection:
  <https://www-cdn.anthropic.com/files/4zrzovbb/website/cc4be2488d65e54a6ed06492f8968398ddc18ebe.pdf>

> "To identify such reportable representations, they developed an elegant
> tool, the Jacobian lens."

> "The J-space is shown to be limited in capacity, but the paper does not
> establish the nonlinear, competitive, all-or-none entry into the
> workspace which, according to GNW and several experiments, is a reliable
> signature of conscious access in human and animal brains."

> "Tools such as neuronal population recordings (in brains) or the Jacobian
> Lens (in LLMs) allow us to dissect the architecture of the system, and
> find that it actually contains sophisticated and structured
> representations of concepts."

### 4.3 Butlin, Shiller, Plunkett, and Long (Eleos AI Research)

- In the Anthropic collection PDF above.

> "The paper provides strong evidence of privileged representations in
> LLMs, but our impression is that more evidence is needed to conclusively
> establish the existence of a workspace-like structure."

> "If it turns out the J-space is missing significant portions of the
> W-space, or that it includes many elements we think don't belong, we
> should expect the results we see to be distorted."

> "This is one place where the acknowledged distortions of the J-lens
> straightforwardly limit our evidence."

### 4.4 Explainers

These are secondary. Do not cite them for numbers.

| Source | Link |
|---|---|
| Louapre, "J-Space: Yet Another LLM Mind Reader?" (HF blog) | <https://huggingface.co/blog/dlouapre/j-space> |
| AlphaSignal | <https://alphasignalai.substack.com/p/how-anthropics-jacobian-lens-reads> |
| VentureBeat | <https://venturebeat.com/technology/anthropics-new-j-lens-reveals-a-silent-workspace-inside-claude-that-mirrors-a-leading-theory-of-consciousness> |
| IBM Think | <https://www.ibm.com/think/news/what-anthropic-j-space-research-means-future-ai> |
| learnmechinterp | <https://learnmechinterp.com/topics/jacobian-lens/> |

Louapre relays the paper's capacity numbers: "In the middle workspace
layers, the median occupancy plateaus at around 25."

## 5. Papers that the search returns but that do not use the lens

| Paper | Link | Why it is not a lens paper |
|---|---|---|
| The Ignition Index (Rahbar, 2026) | <https://arxiv.org/abs/2608.05160> | Uses linear probes, the logit lens, and the tuned lens. Does not name the Jacobian lens. |
| Interpreting Language Model Hidden States at Scale, OmniLens (Pettyjohn et al., 2026) | <https://arxiv.org/abs/2608.10260> | Cites the lens once: "The Jacobian lens avoids translator training by decoding average local output sensitivity at a hookpoint." Does not use it. |
| Decoding Hidden Deception in Reasoning LLMs (2026) | <https://arxiv.org/abs/2606.17478> | Activation explainers. Predates the lens. |
| Rubric-decomposed fusion of VLM aesthetic judges (2026) | <https://arxiv.org/abs/2609.27110> | Aesthetic judges, no interpretability. |

## 6. Gaps in the literature

The searches below found no paper. Each gap is a point the FAccT paper can
claim.

| Search | Result |
|---|---|
| Jacobian lens on vision-language models or image tokens | No paper. The Standard Logits card (3.3) fits a lens on images but reports no readout study. |
| Jacobian lens on aesthetic judgment, image quality, or LLM-as-judge | No paper. |
| Jacobian lens on social bias, fairness, stereotype, or persona | No paper. The only "bias" in the lens literature is the reward-model bias in Gurnee et al. |
| Jacobian lens readouts averaged over a population of inputs and contrasted between groups | Only Mosafer (3.1), with text prompts and fixed word lists. |

## 7. What this means for the paper

1. **Cite the J++ Lens with both authors and the URL.** The bib entry
   names only Ayonrinde.
2. **Keep one citation key for the workspace paper.** `main.bib` holds
   `anthropic2026workspace` and `group.bib` holds `verbalizable`. The
   methods TODO already notes this.
3. **State the gap.** No prior work reads the lens on images or on a
   subjective judgment. Section 6 gives the searches.
4. **Cite Mosafer (3.1) and Liu (3.2) as the nearest methods.** Both score
   readouts with word lists and compare prompt populations. The paper's
   contrasts C3 to C5 are the same idea with images and boroughs.
5. **Cite Ostrow et al. (2.4) for the position-invariance of lens
   directions.** It gives a reason that a contrast at one token position
   is well-posed.
6. **Cite Nanda (4.1) for the false-positive caution.** The paper's rule
   that a word must be in the top 12 is a partial answer to it.
7. **Cite the Standard Logits card (3.3) with care.** It is the only lens
   fit on image inputs, but it is a model card with no paper, dated one
   day before this review.
8. **Note the Gemma 4 31B caveat.** The Neuronpedia card marks its recall
   with a star: the poetry task was dropped.

## 8. Related local documents

- `jacobian-lens-review.md`: review of the local lens code and parity.
- `lens-corpus-faithfulness.md`: estimand and corpus preregistration.
- `r-lens.md`: local R-lens replication.
- `causal-lens-retrospective.md`: the calibration critique.
- `f10-judgment-mechanism.md`: the source analyses for the prompt figure.
