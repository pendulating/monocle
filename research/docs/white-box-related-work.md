# White-box study: prior and concurrent work

Written 2026-09-16. A reading list for the white-box record ("Inside the
Judgment"). Each venue was checked on the OpenReview, proceedings or
journal page on 2026-09-16. An arXiv-only paper appears only when it is
the canonical source of a method we use, or comes from a top lab.

## The 5 closest papers

| Paper | Venue | Why it is close |
|-------|-------|-----------------|
| Hofmann, Kalluri, Jurafsky, King. *AI generates covertly racist decisions about people based on their dialect* | Nature 2024 | The same shape of result in text: alignment removes the overt stereotype and leaves the covert one, and the gap grows with alignment. Our "mask over the rich pole, absence elsewhere" sits beside it. |
| Lee et al. *A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity* | ICML 2024, oral | DPO does not remove the toxic directions; it learns an offset that routes around them, and they come back under a small edit. The closest mechanistic statement of "a mask, not a removal". |
| Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction* | NeurIPS 2024 | The refusal direction, its ablation and its addition. Our block-45 handle and the refusal-hides-a-judgment result are the multimodal version. |
| Neo et al. *Towards Interpreting Visual Information Processing in Vision-Language Models* | ICLR 2025 | The logit lens on image tokens across layers in LLaVA, with token ablation. The direct precedent for our lens on the patches and the relay. |
| Manvi et al. *Large Language Models are Geographically Biased* | ICML 2024 | Model outputs correlated with place-level socioeconomic ground truth, with a bias score against the truth. Our census-proxy dose-response is the same design, on images and inside the model. |

## 1. The lens on image tokens

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Neo et al. *Towards Interpreting Visual Information Processing in VLMs* | ICLR 2025 | Logit lens on LLaVA image tokens: object tokens become vocabulary-readable in the middle layers; ablation of object tokens hurts. | Prior to the patch lens and the relay (section 3). |
| Jiang, Kachinthaya, Petryk, Gandelsman. *Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations* | ICLR 2025 | Projects image-token residuals onto the vocabulary, then edits them with a text-derived direction to remove hallucinated objects. | Lens plus a linear edit at the patches; our steering at the patches (section 6). |
| Wu et al. *The Semantic Hub Hypothesis* | ICLR 2025 | Logit lens on images, audio and other languages: the middle layers of a multimodal LM share one semantic space that the dominant language names. | Why a text lens reads image tokens at all. |
| Zhang, Yadav, Han, Shutova. *Cross-modal Information Flow in Multimodal LLMs* | CVPR 2025 | Attention knockout in LLaVA: image information moves to the question tokens first, then to the answer position, in 2 stages. | The relay (image, question, answer) in section 3. |
| Kaduri, Bagon, Dekel. *What's in the Image?* | CVPR 2025 | Image information is compressed into a few tokens and read in early-middle layers; the last image token matters. | The heavy last patch (SAE trap) and the relay. |
| Gandelsman, Efros, Steinhardt. *Interpreting CLIP's Image Representation via Text-Based Decomposition* | ICLR 2024 | Decomposes CLIP's image embedding by head and position into text-labelled directions. | The vocabulary-readable patch read, on the encoder side. |
| Belrose et al. *Eliciting Latent Predictions from Transformers with the Tuned Lens* | arXiv 2023, canonical | A fitted affine map per block into the unembedding. | The method our Jacobian lens extends (section 1). |

## 2. Causal patching and attribution

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Basu et al. *Understanding Information Storage and Transfer in Multi-modal LLMs* | NeurIPS 2024 | Causal tracing in LLaVA: which image tokens and which layers carry a fact; gradient-based attribution to image regions. | The causal patch map and its readers (sections 4, 5). |
| Zhang & Nanda. *Towards Best Practices of Activation Patching* | ICLR 2024 | The metric, the corruption and the granularity all change the answer. | The reference, substitute and donor controls in section 4. |
| Ghandeharioun et al. *Patchscopes* | ICML 2024 | One framework for lens-style reads by patching a residual into a prompt. | An alternative read of the workspace. |

## 3. Linear directions and steering

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Park, Choe, Veitch. *The Linear Representation Hypothesis and the Geometry of LLMs* | ICML 2024 | Defines linear representations through probing and steering, with a causal inner product. | The frame for "axis", "handle", "pole" in section 6. |
| Rimsky et al. *Steering Llama 2 via Contrastive Activation Addition* | ACL 2024 | Mean-difference steering vectors at one layer, with the layer sweep. | The single-image steering recipe. |
| Li et al. *Inference-Time Intervention* | NeurIPS 2023 | Shifts a few attention heads along truth directions at inference. | A narrow handle that works where a broad shift does not. |
| Gurnee & Tegmark. *Language Models Represent Space and Time* | ICLR 2024 | Linear probes recover place coordinates from the residual. | Place is linearly present; our proxies are place-level. |

## 4. Alignment as a shallow layer over the base model

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Arditi et al. *Refusal Is Mediated by a Single Direction* | NeurIPS 2024 | One direction ablates or induces refusal across 13 chat models. | The refusal object; our handle at block 45. |
| Lee et al. *DPO and Toxicity* | ICML 2024, oral | DPO learns an offset that avoids the toxic directions; they are still there. | "Mask, not removal", in the weights. |
| Qi et al. *Safety Alignment Should Be Made More Than Just a Few Tokens Deep* | ICLR 2025 | Alignment changes mainly the first output tokens; the base distribution is under them. | Our refusal at the output over a judgment held inside. |
| Lin et al. *The Unlocking Spell on Base LLMs* (URIAL) | ICLR 2024 | Base and aligned models differ on few token positions; in-context prompts recover alignment from the base. | The base-model arm (section 7). |
| Jain et al. *Mechanistically Analyzing the Effects of Fine-Tuning* | ICLR 2024 | Fine-tuning adds a thin wrapper over base capabilities; the base mechanism survives and returns. | The mask reading of block 47. |
| Prakash et al. *Fine-Tuning Enhances Existing Mechanisms* | ICLR 2024 | Fine-tuning strengthens circuits the base already has. | The absence reading: what the base has, the instruct keeps or drops, never builds new. |
| Wei et al. *Assessing the Brittleness of Safety Alignment* | ICML 2024 | Safety lives in a small set of weights, separable from utility. | Where a mask would live. |
| Wollschläger et al. *The Geometry of Refusal* | ICML 2025 | Refusal is a cone of directions, not one; independent directions coexist. | The 3 orthogonal steering directions (section 6). |

## 5. Hidden bias that survives alignment

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Hofmann et al. *Covertly racist decisions based on dialect* | Nature 2024 | Overt stereotypes go down with alignment; covert dialect stereotypes stay or go up. | The closest result in text. |
| Bai, Wang, Sucholutsky, Griffiths. *Explicitly unbiased LLMs still form biased associations* | PNAS 2025 | Word-association tests find implicit bias in aligned models that pass explicit tests. | Our single-word census is an inside-the-model association test. |
| Manvi et al. *LLMs are Geographically Biased* | ICML 2024 | Outputs on place-level socioeconomic topics against ground truth, with a bias score. | The dose-response design over 197 NTAs. |
| Nwatu, Ignat, Mihalcea. *Bridging the Digital Divide* | EMNLP 2023 | VLM performance drops with the household income of the image's origin (Dollar Street). | Income-linked VLM behaviour on images. |

## 6. Sparse dictionaries

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Gao et al. *Scaling and Evaluating Sparse Autoencoders* | ICLR 2025, oral (OpenAI) | TopK SAEs, the aux loss on dead latents, scaling laws and evaluation. | Our trainer's recipe (section 8). |
| Bussmann, Leask, Nanda. *BatchTopK Sparse Autoencoders* | NeurIPS 2024 workshop, canonical | The batch-level top-k and the threshold at read time. | The exact variant we train. |
| Rajamanoharan et al. *Jumping Ahead: JumpReLU SAEs* | arXiv 2024 (DeepMind) | The JumpReLU threshold encoder. | Our read-time encoder. |
| Lieberum et al. *Gemma Scope* | arXiv 2024 (DeepMind) | Open SAEs on every layer of Gemma 2. | Why we train our own: it stops at Gemma 2. |
| Templeton et al. *Scaling Monosemanticity* | Anthropic 2024 | Features in a production model; feature steering; safety-relevant features. | The reference for a feature census (stage 3). |
| Marks et al. *Sparse Feature Circuits* | ICLR 2025 | Circuits over SAE features; removes a spurious (gender) feature from a classifier. | Stage 5 and 6: causal tests and attribution on features. |
| Leask et al. *SAEs Do Not Find Canonical Units of Analysis* | ICLR 2025 | Features split with width; there is no one true dictionary. | Our answer code: many equivalent bases. |
| Paulo & Belrose. *SAEs Trained on the Same Data Learn Different Features* | arXiv 2025 (EleutherAI), canonical | Seed changes 30 percent of features; the stable ones carry the signal. | The seed-match gate G2 and the working-set rule. |
| Fel et al. *Archetypal SAE* | ICML 2025 | A stability fix for SAEs on vision models. | An alternative if the working set is too small. |
| Kantamneni et al. *Are SAEs Useful? Sparse Probing* | ICML 2025 | SAE features do not beat probes for detection. | Stage 4: the labels select, they do not train. |
| Lim et al. *SAEs Reveal Selective Remapping of Visual Concepts* | ICLR 2025 | Patch-level SAEs on CLIP (PatchSAE). | The patch dictionary. |

## 7. Massive activations and the rewrite

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Sun, Chen, Kolter, Liu. *Massive Activations in LLMs* | COLM 2024 (with Meta) | A few coordinates hold values 1,000 times the rest; they act as constant biases. | The gemma centering trap; the 286 coordinate. |
| Timkey & van Schijndel. *All Bark and No Bite: Rogue Dimensions* | EMNLP 2021 | A rogue dimension makes every raw cosine near 1. | Why we center before any cosine. |
| Dettmers et al. *LLM.int8()* | NeurIPS 2022 | Outlier features emerge at scale and break quantization. | The same coordinates from the systems side. |

## 8. Street imagery and socioeconomic prediction

| Paper | Venue | What it does | Relation |
|-------|-------|--------------|----------|
| Gebru et al. *Street View and the demographic makeup of neighborhoods* | PNAS 2017 | Cars in street view predict race, income and voting by precinct. | The application-side prior: the image holds the proxies. |
| Naik et al. *Computer vision uncovers predictors of physical urban change* | PNAS 2017 | Street view features predict neighborhood change. | The same. |
| Dubey et al. *Deep Learning the City* (Place Pulse 2.0) | ECCV 2016 | Pairwise human perception of safety, wealth and liveliness from street view. | Our pairwise judgment task is the model-side version. |
| Jean et al. *Satellite imagery and machine learning to predict poverty* | Science 2016 | Transfer features from imagery to consumption and assets. | The canonical imagery-to-wealth result. |

## Not included

Shrivastava & Holtzman *Linearly Decoding Refused Knowledge* (arXiv 2025) and Pach et al. *SAEs Learn Monosemantic Features in VLMs* (arXiv 2025) are close but arXiv-only from outside the listed labs on 2026-09-16.
