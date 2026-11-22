# F10 manuscript proposals

Written 2026-11-01 against `artificial-aesthetic/paper` at `e2b7620`. This page proposes
edits to the manuscript. It makes no edit. Each item stands alone: accept or reject it.

**Note:** We shortened the paper on 2026-11-01 (D168). Line numbers in `3_methods.tex`, `4_results.tex` and the appendix are from before that edit. Find the text by its words. The paper is still over the limit, thus accept proposals that add text with care.

The commentary of this page is in STE. The proposed manuscript text is in the author's
voice, in quoted blocks or LaTeX blocks.

| Item | Value |
|------|-------|
| Files read in full | `sec/0_abstract.tex`, `1_intro.tex`, `2_rw.tex`, `3_methods.tex`, `4_results.tex`, `5_discussion.tex`, `A_addtl-methods.tex`, `B_addtl-results.tex` |
| Records read | `facct-claim-map.md`, `f10-analysis-walkthrough.md`, `f10-judgment-mechanism.md` (the report), `neuronpedia-viewer.md`, `lens-registry.md`, `facct-runbook.md` F7 and F10.6 to F10.18 |
| Code read | `research/monocle/jlens_read.py`, `research/scripts/f10/hop_all.py`, the header of `research/scripts/f10/hop_depth.py` |
| Data read | `research/figures/f10/borough/borough_words.csv`, `research/figures/f10/stage/stage_words.csv`, `research/figures/f10/probe/probe_borough.csv`, `probe_place.csv`, `research/figures/f10/sense/sense_links.csv`, `research/figures/f10/mechanism/phase_layers.csv`, `mech_borough_layers.png`, `data/f10/hop_depth/images.parquet` |
| Literature | Gurnee et al., "Verbalizable Representations Form a Global Workspace", Transformer Circuits, 2026 (the first third of the page); Ayonrinde and Lindsey, "J++ Lens", LessWrong, 2026 (title and the summary in the runbook F7; the page did not render) |
| Value pairs | A pair such as .82, .75 gives Gemma-4-31B, then Qwen3.5-9B |

**Warning:** No Qwen3.6-27B number is in this page. The margin fault makes them invalid.
No `nice` or `ordinary` number exists yet. P26 marks a slot with no numbers.

**Status (2026-11-01):** The terms of section 2 are in the working copy of the manuscript, with
P2 and P3. Other items of section 3 are open.

| Change from this page | Reason |
|-----------------------|--------|
| The verdict rule says "the yes-no margin of the readout", not "the decoded margin" | `phase_layers.py` uses the lens margin (`logit_margin.csv`), not the decoder |
| The cited-word rule says top 12, not top ten | The borough prose cites words at ranks 11 and 12 |
| The caption of the borough figure stays as the author set it | The author chose the text |
| "surface" stays when the subject is a readout or a contrast | The word is part of the author's voice |

**Applied on 2026-11-01:** P2, P5 (short form), P6, P11, P13 to P15 (short forms), P17, P18, P20, P21, P22 (with P23 in the form of the 3-model Manhattan result), and the parts of U1 to U11 that these items cover. The manuscript also has the F10.24 results in a new subsection, "Across models and adjectives".

**Literature check (2026-11-01):** A check of the sources changes 3 items of this page.

| Item | Correction | Source |
|------|------------|--------|
| The rule for `J-space` | Too strict. "its J-space contains", "in its J-space" and "the J-space of a model" are normal use. "average J-space", "the J-space surfaces" and `vector` for a word stay wrong | Gurnee et al. (paper and blog post) |
| P1 | Withdrawn in part. The sources do report `legs` at intermediate layers. `8` is the output | J++ Lens post, Introduction; Gurnee et al., spider example |
| Names | `Jlens` is not a prose name. Use Jacobian lens (J-lens), J++ Lens, R-Lens | Both sources |

## 1. Top items

| # | Change | Reason |
|---|--------|--------|
| T1 | Collapse the vocabulary: `J-space` is the workspace, not the readout. Replace `J-space vectors`, `most unique vectors`, `most-activated vector` and `the J-space surfaces` with `readout` and `word`. Section 2 gives the rule and each site. | The paper averages readouts and ranks words. It never uses a J-lens vector. A reader of Gurnee et al. reads `J-space` as a region of the residual stream, and `vector` as a word direction. The current text mixes the three objects. |
| T2 | Define 4 quantities once, in the methods: the readout, the mean readout, the image contrast (C2), and the answer margin. The methods now define C2 twice (`3_methods.tex:86` and `:89`) and the margin in 2 forms. | The results use all 4 with different names. See P2, P4. |
| T3 | Correct the prompt text at `3_methods.tex:84`: `Answer in one word` to `Answer with one word`. | The run used `Answer with one word` (`hop_depth.py:55`, `QUESTION`). The TikZ figure already shows the correct text. |
| T4 | Sort the city-mean readouts by the author's 2 estimands. The `cool` synonyms and `vibe` are embedded (same rank with no image). `place` and `?` words are image-caused. The `model` words are top words of the image contrast C2 only, not of the mean readout. Write the sentence at `4_results.tex:243` with the measured order: scene, sense, standard, verdict. | `stage_words.csv`: `vibe` ranks 3 with the image and 6 without; `location` ranks 3 against 20,094. The report gives the `model` words at rank 330 to 11,460 in the mean readout and 4 to 19 in C2. The sentence at `:243` also lacks a word. See P8 to P11. |
| T5 | Correct 5 statements of the borough paragraphs against `borough_words.csv`. English `recognize` and `familiarity` are already at Bronx `place` L44. `ninguna` is at L56. Every borough holds words in several languages, thus narrow "no other borough" to the geography and recognition content. The Bronx `cool` readout at L8 to L16 holds music and craft words, not nonsense. Tourism words sit at L44 to L48, past the last column of the figure. | See P13 to P17. The sample design makes the C3 lists valid. The corrections are about what the lists hold. |
| T6 | Replace the hypothesis at `4_results.tex:265` with the sense-of-the-word reading. | The hypothesis makes a layer claim ("fails to do by layer 20") that no layer test supports. The temperature words start at L12 in both boroughs. The `urban` probe and the traces support the direction without a layer. See P18. |
| T7 | State the main F10 result in prose, in 1 paragraph before `fig:prompt-readout`: a scene code holds .82, .75 of the margin; the question adds .12, .18 at `cool`; the image picks the sense of `cool`; the sense does not track the verdict. Add the Qwen3.5-9B replication in the same paragraph. | Claim 1 rests on the figure alone. The prose never states the gain, the sense axis or the second model. See P6. |
| T8 | Add the 2 strongest F10 results to claims 3 and 4: the value order of `cool` from the 53 probes, and the object-and-value result from the traces. | These are the tested results behind "values" and "used differently". The graffiti anecdote at `:349` has no test. The report gives the chi-square with Holm. See P21, P22. |
| T9 | Add the `in New York` probe for "not always correct". | It is the only direct measure. `fig:misnaming-map` and `fig:hashtag-map` are unresolved references. See P20. |
| T10 | Fix the 2 small faults in the methods: the typo sentence at `3_methods.tex:86` ("shows what images generally to the readout") and the spider example at `:80`. | The source paper does not report `legs` in a readout, and `8` is the output. See P1, P3. |

## 2. Terminology

### 2.1 What the code computes

The chain, from `research/scripts/f10/hop_all.py` and `research/monocle/jlens_read.py`:

| Step | Code | Object |
|------|------|--------|
| 1 | `ActivationRecorder(lens_model.layers, at=layers)` | The **residual** `h` at each token position, at the output of block `l`. Layers 0, 4, 8, ... (`STEP = 4`). |
| 2 | `lens.transport(h, l)` | `J_l h`: the J++ lens maps the layer-`l` residual to the final residual space. One linear map for each layer. |
| 3 | `lens_model.unembed(...)` | Final norm, `lm_head` and softcap: **logits** over the vocabulary. |
| 4 | `x[..., keep]` | The **readout**: the logits of the 53,999 store words (` [a-z]{3,}`) at one (image, token position, layer) cell. `reads/reads-*.npy`. |
| 5 | `x.topk(30)` | The **top-k words** of the whole vocabulary, `topk/ids-*.npy`. |
| 6 | `text_reads.npy` | The readout of the same text with no image. |
| 7 | `yes_no`, `margin` | The **answer margin** from the model's own output at the last token. `yes_no` is `logp(Yes) - logp(No)`. `margin` pools 8 forms. |
| 8 | `hop_borough.py` and the viewer | A **contrast**: one mean readout minus another, at the same cell. C1 to C5. |

The paper uses none of the J-lens vectors. It does not steer, project or ablate. Thus the
words `vector`, `direction` and `activation` have no referent in the paper's analyses.

### 2.2 Canonical terms

| Term | Definition (for the paper) | What it is not | Canonical use |
|------|----------------------------|----------------|---------------|
| **Jacobian lens** | A family of per-layer linear maps from the residual stream of layer $\ell$ to the output vocabulary, fit as the average Jacobian of the final residual with respect to the layer-$\ell$ residual \citep{anthropic2026workspace}. | Not the workspace. Not a readout. | Name the family once in the methods. |
| **J++ lens** | The variant of Ayonrinde and Lindsey that filters the Jacobians before the average. We use their released lenses for Gemma-4-31B and Qwen3.5-9B. | Not "Jlens" in the plural. Not a new fit. | The tool of this paper. Write "the J++ lens" after the first mention. |
| **Readout** | The vector of lens logits over the vocabulary at one token position and one layer, for one image. The "readout of a cell". Its ranked word list is "the top words of the readout". | Not the J-space. Not a vector in residual space. | The noun for the measurement. One word: `readout`. Do not use `read` as a noun. |
| **Mean readout** | The readout averaged over a set of images (all 591, or a borough). | Not "the average J-space". | "the mean readout over 591 images", "the Bronx mean readout". |
| **No-image readout** | The readout of the same text with the image and its slot removed (`text_reads.npy`). It gives the **embedded** conception. | Not a baseline of "no information": the prompt is still there. | "the readout with no image". |
| **Contrast** | The difference of 2 mean readouts at the same cell. C1: image minus the mean. C2: the mean minus the no-image readout. C3: a borough minus the other boroughs. C4, C5: the top third of the answer margin minus the bottom third, and the reverse. | Not a logit. Rank its words; do not report its value. | "contrast C3 at `cool`". |
| **Image-caused** | The rank of a word in contrast C2 (`rank_added`). A word that ranks high in C2 and low with no image rises because of the image. | Not a cause inside the model. The no-image text also lacks the vision tokens. | Use C2 and "image-caused" as one thing. |
| **Embedded** | A word with about the same rank with and without the image. | Not a failure of the image. | "embedded in the model". |
| **J-space** | In Gurnee et al., the set of points expressible as a sparse non-negative combination of J-lens vectors; the region of the residual stream that they argue works as a global workspace. | Not a readout. Not a list of words. Not a thing that "surfaces" tokens. | Use only for the workspace hypothesis (methods, 1 sentence; the figure (b) label). |
| **Workspace layers** | The middle band of layers in which readouts carry reportable content, between a "sensory" band and a "motor" band \citep{anthropic2026workspace}. | Not a measured band of this paper. Panel (b) bands are ours. | Panel (b) only. |
| **Token position** | A position in the prompt: the token `cool`, the token `?`, the last token. | Not a word of a readout. | "at the `cool` token" or "at the `cool` position". |
| **Word** (of a readout) | An entry of the store vocabulary: a lower-case token of 3 letters or more. It can be a word piece (`anarch`, `prostit`). | Not a "vector". Not a "token" of the prompt. | "the top words", "the word `vibe`". |
| **Top-k words** | The k words with the largest value in a readout or a contrast, at one cell. State k and the quantity. | Not "most-activated". | "a top-10 word of the mean readout at `?` L40". |
| **Logit** | The pre-softmax score of one vocabulary entry. A readout is a vector of lens logits. The model's own logits at the last token give the answer. | Not a probability. Not a contrast. | "lens logits", "the yes and no logits". |
| **Answer margin** | $\log p(\text{yes}) - \log p(\text{no})$ at the last token of the prompt, from the model's own output, with no lens. For 2 tokens this equals the logit difference. The analyses pool 4 forms of each word (`Yes`, `yes`, with and without a leading space). | Not a readout value. | "the answer margin", defined once. |
| **Residual** | The residual-stream activation at one token position after block $\ell$: the input of the lens. | Not an "activation" of a word. | "the residual at `cool`, layer 24". |
| **Layer** | The index $\ell$ of a block. We save every 4th layer. "L24" is the output of block 24. The last saved layer is L56 (Gemma-4-31B, 59 blocks) and L28 (Qwen3.5-9B, 31 blocks). | Not "the last layer of the model": L56 is the last saved layer, not block 58. | "L24"; "the last saved layer". |
| **Sense band, standard band, verdict band** | Layer bands set by the rules of `3_methods.tex:91`. | Not workspace layers. | Panel (b) and the borough figure shading. |

True synonyms that must collapse to 1 word:

| Found in the paper | Canonical |
|--------------------|-----------|
| readout, read (noun), reads, "the raw J-space", "the J-space" (as a list), "aggregated readout", "city-averaged readouts", "average J-space" | **readout**, **mean readout** |
| J-space vectors, most unique vectors, most-activated vector, tokens (of a readout), readout tokens, top-k tokens | **word**, **top-k words** |
| readout at the token position, token readout | **readout at the `cool` token** |
| contrastive transformation, contrastive output | **contrast** |
| difference in the yes logit and the no logit, answer margin, yes/no judgment | **answer margin** |

Distinct objects that the paper now merges:

| Pair | Distinction |
|------|-------------|
| J-space and readout | A region of the residual stream against a vector of logits. The paper reads out of the residual; it does not read "the J-space". |
| Jacobian lens and J++ lens | The family against the tool. "Jlens-family" is fine once; after that, "the J++ lens". |
| R-lens and J++ lens | `4_results.tex:16` uses an R-lens on Gemma-4-12B. The R-lens is a different lens on a different model. Keep it apart or cut the teaser. |
| token (position) and word (of a readout) | "the `cool` token" against "the word `vibe`". |
| contrast C2 and the image-caused readout | They are 1 thing. Define it once. |
| the answer margin and a lens value | The margin comes from the model's output, not from the lens. The bars of `fig:prompt-readout` decode the margin from residuals; the lens adds words, not the $R^2$. |

### 2.3 Site-by-site corrections

| Site | Quoted phrase | Correction |
|------|---------------|------------|
| `3_methods.tex:80` | "readouts at the same token position across many successive layers can show intermediate 'hops'" | Keep. This is the one correct use of `readout` with `token position` and `layers`. |
| `3_methods.tex:80` | "the J-space first shows 'spider', then 'legs', then finally, '8' at the final layer" | See P1. The source reports `spider` in the readout; `8` is the output. |
| `3_methods.tex:82` | "two pre-fit Jlens (specifically, the J++ lens...)" | "two released J++ lenses \cite{kolaayonrinde2026lensjacobian}, one for Gemma-4-31B and one for Qwen3.5-9B" |
| `3_methods.tex:82` | "surfacing faithful readouts" | "faithful readouts" is the source's word. Keep. |
| `3_methods.tex:84` | "J-space readout task" | "readout task" |
| `3_methods.tex:86` | "contrastive outputs", "Each contrastive output ranks words by the difference between two reads" | "contrasts", "Each contrast ranks words by the difference between two mean readouts" |
| `3_methods.tex:86` | "C1 is the readout of image $I$ minus the average readout of all 591 images ... image $i$" | One symbol: $i$. "mean readout". |
| `3_methods.tex:86` | "C2 is the $i$ average minus the same text with no image, and shows what images generally to the readout" | See P2. |
| `3_methods.tex:86` | "C4 is the top-third of images, ranked by the difference in the \emph{yes} logit and the \emph{no} logit" | "C4 is the mean readout of the top third of images by answer margin minus that of the bottom third; C5 is the reverse" (`neuronpedia-viewer.md`, "How to read a run"). |
| `3_methods.tex:89` | "The read with no image gives ... the mean read over images minus that read" | `read` (noun) to `readout`. Merge with C2: see P2. |
| `3_methods.tex:91` | "The read at the last token gives $r\geq.95$" | "The decoded margin at the last token". The verdict rule uses the ridge decoder on residuals, not the lens (`phase_layers.csv`, `peak` .988, .961 is the decoder $r$). Check the script before you change the word. |
| `4_results.tex:16` | "late-layer token readout from an R-lens" | Keep `readout`. Add that the R-lens is a separate lens on Gemma-4-12B, or cut with the teaser (P5). |
| `4_results.tex:18` | "Jlens, and the J-space workspace they reveal" | "the J++ lens" |
| `4_results.tex:20` | "depending on if we read the raw J-space or a contrastive transformation (e.g., C1-5)" | "depending on whether we rank the mean readout or a contrast (C1--C5)" |
| `4_results.tex:20` | "All cited J-space vectors for a finding are at least a top-ten most-activated vector at a token position and model layer" | "Every word we cite ranks in the top ten of the mean readout, or of the named contrast, at its token position and layer" |
| `4_results.tex:20` | "Neuropedia" | "Neuronpedia" |
| `4_results.tex:20` | "Gemma-4-31B's readouts size to 44.6 GB" | Holds: `data/f10/hop_all/gemma-4-31b/reads` is 45 GB and the Qwen folder 19 GB. Say "the word readouts" (the full store with residuals is 96 GB per model, runbook F10.6). |
| `4_results.tex:208` caption | "Reads at each token of Gemma-4-31B" | "Readouts at each token" |
| `4_results.tex:213` heading | "A generalizable readout of the task" | "The mean readout over the city" |
| `4_results.tex:215` | "We first performs readouts from the average J-space constituted by all 591 images" | "We first rank the mean readout over all 591 images" |
| `4_results.tex:217` | "the mean read over 591 images", "both reads" | "mean readout", "both readouts" |
| `4_results.tex:223` caption | "Embedded and image-caused reads", "the image read only" | "readouts", "the image readout only" |
| `4_results.tex:228` | "the raw readout begins to surface \emph{types} of places" | "the mean readout" (`raw` means "not a contrast"; say so once at `:20` and drop it after) |
| `4_results.tex:230` | "the idea of 'vibe' becomes prominent in the J-space" | "in the readout" |
| `4_results.tex:236` | "the J-space surfaces more specific aesthetic values" | "the readout holds" |
| `4_results.tex:243` | "top-k tokens of the aggregated readout" | "top words of the mean readout" |
| `4_results.tex:245` heading | "Borough-specific readouts" | Keep. |
| `4_results.tex:246` | "we take the average J-space of the 111 images from the Bronx and subtract the average J-space of the other 480 images; this operation reveals directions in the J-space that are surfaced uniquely" | "we take the mean readout of the 111 Bronx images and subtract the mean readout of the other 480; the top words of this contrast are the words that Bronx images raise above the rest of the city" |
| `4_results.tex:257` | "the raw, city-averaged readouts" | "the mean readout" |
| `4_results.tex:259` | "the J-space surfaces city-related tokens" | "the contrast holds city words" |
| `4_results.tex:259` | "No other borough's J-space exhibits this behavior: Staten Island's most unique vectors ... Brooklyn's most unique vectors ... Queens' most unique vectors ... Manhattan's most unique vectors" | "No other borough's contrast at \texttt{place} holds geography or recognition words: Staten Island's top words ... Brooklyn's ... Queens' ... Manhattan's" (see P14 for the content) |
| `4_results.tex:261` | "the readout is nonsensical", "the J-space surfaces tokens relating to tourism" | See P15. "the contrast holds tourism words" |
| `4_results.tex:263` | "their layers 20--24 contain tokens relating to temperature", "no harmful content is surfaced" | "words", "no such word appears at \texttt{cool} at any layer" (scope: see section 4) |
| `4_results.tex:263` | "Manhattan's readout is also unique. It surfaces European nationality at layer 8" | Keep `readout`; say `contrast` once in the paragraph so a reader knows these are C3 lists. |
| `4_results.tex:265` | "layers 20-24 surface aesthetic \emph{properties}" | See P18. |
| `4_results.tex:252` caption | "Words of top-12 contrast C3" | The figure shows 6 words per cell (`mech_borough_layers.png`). "Six of the top-12 words of contrast C3". |

A global rule for the fix: replace `J-space` by `readout` or `contrast` at every site
except `3_methods.tex:80` (the workspace sentence) and the panel (b) label. Replace
`vector` by `word` at every site. Replace noun `read` by `readout`.

## 3. Itemized proposals

Each item gives the location, the type, the current text, the proposed text, the claim
it supports (1 to 4), the evidence, and the confidence.

### Methods

**P1** `3_methods.tex:80` | edit | claim 1 | confidence: medium

Current: "the J-space first shows 'spider', then 'legs', then finally, '8' at the final layer of the model."

The source (Gurnee et al., Methods and the spider example) reports that the lens "confirms that spider is represented at the relevant token positions", and that a swap of the `spider` vector for `ant` moves the output from `8` to `6`. The part of the page we read does not report `legs` in a readout. `8` is the output. Proposed:

> for example, when asked 'the number of legs on the animal that spins webs is', the readout at intermediate layers shows 'spider', a word that is in neither the prompt nor the answer, before the model outputs '8' \cite{anthropic2026workspace}.

Check the second half of the source page before you keep `legs`.

**P2** `3_methods.tex:86` and `:89` | edit | claim 1 | confidence: high

C2 is defined at `:86` and again at `:89`. The sentence at `:86` lacks a verb. Merge into one definition and name the two estimands there. Proposed replacement of the C2 sentence at `:86`:

> C2 is the mean readout over all images minus the readout of the same text with no image. We use it to separate two quantities: the readout with no image gives the model's \emph{embedded} conception at a token, and C2 gives the change that an image causes. The no-image text omits the vision tokens, so C2 covers the image and its slot together.

Then cut the first two sentences of `:89` ("We separate two quantities ... image and its slot together.") and keep its last sentence (the fixed prompt and the $R^2$ limit).

Source: `hop_all.py:219-227` (`text_reads.npy`); report section "2 estimands".

**P3** `3_methods.tex:84` | edit | claim 1 | confidence: high

Current: "Answer in one word: yes or no."
Proposed: "Answer with one word: yes or no."

Source: `research/scripts/f10/hop_depth.py:55`.

**P4** `3_methods.tex:86`, after the C5 sentence | add | claims 1, 4 | confidence: high

The results use "answer margin" (figure caption), "yes/no judgment" (`:239`) and "the difference in the yes logit and the no logit" (`:86`). Define it once:

> Throughout, the \emph{answer margin} of an image is the log-odds of \emph{yes} against \emph{no} at the last token of the prompt, read from the model's own output distribution with no lens. We pool the four surface forms of each word (\emph{Yes}, \emph{yes}, with and without a leading space).

Source: `hop_all.py:65-73` (`ANSWERS`, `pooled_margin`). **Note:** the C4/C5 split in the viewer uses `yes_no` (`Yes` minus `No`), not the pooled form (`neuronpedia-viewer.md`). State which one the paper's C4/C5 lists use, or make them the same.

**P5** `3_methods.tex`, after `:91` | add | claim 1 | confidence: medium

The results figure reports $R^2$ gains, a sense axis and a sense score, but the methods state only the band rules. Add one paragraph:

> Beside the lens, we fit a ridge decoder of the answer margin on the raw residuals, with five folds by neighborhood (NTA) and three repeats. The decoder on the mean image patch is the base; the \emph{gain} of a token is the out-of-fold $R^2$ that its residual adds to that base, tested with a paired $t$ on NTA errors and Holm correction over the cells of a model. To read the sense of a word, we take the first axis of an SVD of the NTA mean readouts at a cell, and score each image on a fixed list of temperature words against a fixed list of social words. To read the model's own answer under other adjectives, we ask the same question with 53 adjectives in place of \emph{cool} and read the answer margin of each.

Source: report "Methods" table (F10.9, F10.14, F10.15, F10.17); report Q1 ("5 folds by NTA and 3 repeats", "Holm covers all cells of 1 model, and a paired t on NTA errors").

### Results, claim 1

**P6** `4_results.tex:22`, replace the one-line lead | add | claim 1 | confidence: high

Current: "We first report findings for Gemma-4-31B. \autoref{fig:prompt-readout} gives an overview of the reads along the prompt, and the sections below give the detail."

Proposed:

> \autoref{fig:prompt-readout} gives the judgment along the prompt and the layers, for both models. Three parts separate in the data. First, the image alone carries most of the verdict: a ridge decoder on the mean image patch, before any question, recovers the answer margin at out-of-fold $R^2=.82$ (Qwen3.5-9B: $.75$), and the same patches decode \emph{beautiful} ($.88$) or \emph{green} ($.87$) about as well, so this is a scene code rather than a stored answer. Second, the question adds a step: the gain over the mean patch is small at \emph{this} ($.04$, $.05$) and \emph{place} ($.07$, $.08$), steps up at \emph{cool} ($.12$, $.18$), and then stays about flat through the instruction. Third, at \emph{cool} the image picks the \emph{sense} of the word. The largest axis of the readouts across neighborhoods at layer 24 holds 63\% of the variance (Qwen 49\%) and runs from \emph{coolant}, \emph{temperature}, \emph{humidity} to \emph{boutique}, \emph{hipster}, \emph{bodega}; an image's position on it barely tracks the verdict ($r=.10$; Qwen $-.11$). Both models show the same three parts at different layers (\autoref{fig:prompt-readout}b). The sections below give the detail for Gemma-4-31B.

Sources: report Q1 tables (R2 .820, .753; gains .043/.051, .067/.078, .124/.178; `beautiful` .88, `green` .87); report Q2 table (.63, .49; r .10 p .051, $-.11$ p .008); axis words table (Gemma L24).

**P7** `4_results.tex:208` caption | edit | claim 1 | confidence: medium

The callouts of panel (a) mix two kinds of word list. Callout 3 gives "Manhattan ballroom, gala, champagne; Staten Island canola, mulch, zucchini". These are from the held-out $t$-ranked lists (report, "Lists that hold in both halves": Gemma Manhattan `place` L36 "dazzling, sequins, ballroom, gala, champagne, peacock"). The borough figure (`fig:borough-layers`) ranks by the raw C3 difference, and at Manhattan `place` L36 that list holds `podium, escalator, venues, ballroom` but not `gala` or `champagne` (`borough_words.csv`, `by_raw`). Add one clause to the caption so the two lists do not look like one:

> Borough words in callout 3 are from the $t$-ranked contrast on held-out neighborhoods; the lists of \autoref{fig:borough-layers} rank the raw contrast.

Also: callout 5 says "L40--44" but the status $r=.71$ is at `?` L36 (report: "Status pair ... 0.707 (`?` L36)"). Either move the layer label to L36--44 or say that the standard band is L32--44.

**P8** `4_results.tex:228` | edit | claim 2 | confidence: high

Current: "At the 'place' token position, the raw readout begins to surface \emph{types} of places ("metropolitan", "beachfront"), and continues to do so until layer 24, where tokens relating to \emph{localization} emerge: nearby, relocated, surroundings, and, in layer 28, 'location', which continues to be prominent until the last layer of the model."

`stage_words.csv`: `beachfront` ranks 4 at L12 with the image and 4 without (embedded); `metropolitan` ranks 273 at L24 in the mean readout (not a top word there); `nearby` ranks 3 at L36 against 19,873 without the image; `location` ranks 3 at L40 against 20,094. Proposed:

> At the \emph{place} token, the mean readout first holds types of place ("beachfront" ranks fourth at layer 12 with and without the image, so this is embedded), and from layer 24 words of \emph{localization} rise: nearby, relocated, surroundings, and, from layer 28, 'location', which stays near the top until the last saved layer. These are image-caused: with no image, 'nearby' and 'location' rank near 20,000 (\autoref{fig:ranks}).

Check the layer at which `metropolitan` is a top word before you keep it.

**P9** `4_results.tex:230` | edit | claim 1 (embedded conception) | confidence: high

Current: "At the 'cool' token position, the raw readout surfaces an assortment of nouns (cookware, kiln, canyon) prior to layer 16, and then shifting to \emph{synonyms} of 'cool' (amazing, awesome, funky) through layer 32. After that, the idea of 'vibe' becomes prominent in the J-space, as well as 'idea' and 'opinion, and 'vibe' remains prominent through the last layer of the model."

The words are real, and they rank the same with no image: `amazing` 8 against 11 (L16), `awesome` 8 against 9 (L20), `funky` 5 against 6 (L28), `idea` 2 against 2 (L32), `vibe` 3 against 6 (L36). This is the author's estimand (1). Say so:

> At the \emph{cool} token, the mean readout holds an assortment of nouns (cookware, kiln, canyon) before layer 16, then \emph{synonyms} of 'cool' (amazing, awesome, funky) through layer 32, and from there 'vibe', 'idea' and 'opinion', with 'vibe' near the top through the last saved layer. Unlike the words at \emph{place}, these rank the same with no image (\autoref{fig:ranks}: 80 of the top 100 words at layer 36 are also in the top 100 with no image). The conception of 'cool' is embedded in the model; the image does not supply it.

Source: `stage_words.csv`; report F10.21 table (`cool` L36: 80, 0, 0). Note: `opinion` is not in `stage_words.csv`; the rank-cells file holds all words if you want to check it.

**P10** `4_results.tex:234` | edit | claim 3 | confidence: high

Current: "We observe signs of the model wielding aesthetic \emph{values} at the 'model' token position. Layer 32 exposes "renovated", "fashionable", and "decorated", layer 40 exposes "ugly", "handsome", and "favorite", ..."

These words rank 330 to 11,460 in the mean readout and 4 to 19 in contrast C2 (report, "The stage of each paper word"). The paragraph must name the contrast:

> At the 'model' token, the image contrast C2 (what the image adds over the same text with no image) holds aesthetic \emph{values}: layer 32 raises "renovated", "fashionable", and "decorated", layer 40 "ugly", "handsome", and "favorite", layer 44 "inexpensive" and "aesthetics", layer 48 "believable", and layer 52 "authenticity", "austerity", and "enjoyment". None of these is a top word of the mean readout itself; they are what the image adds.

Source: report F10.13 table (`model` row: rank 616 to 4,403 in the mean, about 53,000 with no image, `rank_added` 4 to 8); claim-map check 17.

**P11** `4_results.tex:243` | edit | claim 1 | confidence: high

Current: "So, according to the J-space, this aesthetic evaluative task involves the following, in a sequential ordering: localization, defining the aesthetic standard (e.g., 'cool'), applying relevant aesthetic standards (e.g., 'attractiveness'), and ultimately making an aesthetic judgment. However, we do not observe any in top-k tokens of the aggregated readout at any position; this motivates us to examine C3, C4, and C5."

The sentence lacks a noun ("we do not observe any ___"). The order "localization, standard, application, judgment" is read from onset by token, and the report finds that onset does not separate stages (F10.13: "Partial r starts at the first tokens for nearly all sets"). The measured order is the one in \autoref{fig:prompt-readout}b. Proposed:

> So the mean readout gives the task in parts that we can place along the prompt: a scene code in the image patches, localization at \emph{place}, the sense of the word at \emph{cool}, and an explicit standard at \emph{?}, before the verdict at the last token. We stress that the prompt is fixed, so this is an order along the prompt, not a causal order inside the model. The mean readout holds no borough, no image and no verdict, which motivates the contrasts C3, C4, and C5.

Source: report F10.13; report "Note" on the fixed prompt; `phase_layers.csv`.

**P12** `4_results.tex:239` | edit | claim 1 (embedded) | confidence: medium

Current: "at the '<channel|>' token position that ends the prompt, we observe expressions of uncertainty and acknowledgments of subjectivity from layer 36 onwards: "inconclusive", "cannot", "subjective", "idk", and "jury", before the final yes/no judgment surfaces at layer 56."

`inconclusive` ranks 4 against 6, `idk` 8 against 14, `cannot` 26 against 32 with and without the image (L40). `subjective` at L52 ranks 2 against 7. These are embedded, as the `cool` words are. Add one clause: "These rank about the same with no image: the hedge belongs to the prompt and the model, not to the image." Also "layer 56" is the last saved layer, not the final block (58); write "by the last saved layer (56)".

Source: `stage_words.csv` (`end` rows); `hop_depth.py:52-53` (59 layers, step 4).

### Results, boroughs (claim 2 and 4)

The sample is stratified over all 197 residential NTAs, 3 images each (`images.parquet`: Queens 177, Brooklyn 159, Bronx 111, Manhattan 96, Staten Island 48). The C3 mean contrast is an unbiased read of each borough. The items below correct what the lists hold, against `borough_words.csv` (`by_raw`, Gemma-4-31B). They do not question the lists.

**P13** `4_results.tex:259`, Bronx `place` | edit | claim 2 | confidence: high

Current: "By layer 40, the J-space surfaces city-related tokens for images in the Bronx, in both English, Spanish, and Polish ('miasta'). Recognition-related tokens appear starting at Layer 44, in Spanish and Portuguese (not English): 'identificar', 'reconhecimento', 'ninguna' (none), before English equivalents emerge at the last model layer: 'recognizes', 'familiarity'."

The file gives, Bronx `place`: L40 `cidades, cidade` (Portuguese), `miasta` (Polish), `villes` (French), `cities`; L44 `cities, identificar, reconhecimento, recognize, identific, familiarity`; L56 `ninguna, recognizes, exacta, familiarity, recognize, unknown, unremarkable, identifiable`. Thus English recognition words are already at L44, and `ninguna` is at L56. Proposed:

> By layer 40, the contrast holds city words in English, Portuguese, French, and Polish ('cities', 'cidades', 'villes', 'miasta'). Recognition words appear at layer 44 in several languages at once: 'identificar', 'reconhecimento', 'recognize', 'familiarity'. At the last saved layer the contrast turns to non-recognition: 'ninguna' (none), 'unknown', 'unremarkable', 'identifiable'.

**P14** `4_results.tex:259`, "No other borough" | edit | claim 2 | confidence: high

Current: "No other borough's J-space exhibits this behavior: ..."

Every borough's `place` contrast holds words in several languages (Brooklyn `negozio, loja, tiendas, biznes`; Queens `vivienda, hogares, keluarga, rumah, woning`; Staten Island `petani, paysans, tanah, pulau`; Manhattan `wisata, turist, museo`). What is specific to the Bronx is the geography and recognition content. Proposed:

> No other borough's contrast at \emph{place} holds geography or recognition words: Staten Island's top words orient around island descriptors ("landfall", "coastal") and rural descriptors ("acreage", "homestead", "drought", "farmer"), Brooklyn's relate to commerce ("storefront", "pharmacies", "businesses"), Queens' resemble family life ("homeowners", "vivienda", "homeschooling"), and Manhattan's align with its renown ("gastronomy", "attractions", "tourists", "busiest"). Words in Spanish, Portuguese, Indonesian, and Polish appear in every borough's contrast.

All quoted words hold in `by_raw` at `place`: Staten Island `landfall` L12, `coastal` L16 to L24, `acreage` L44, `homestead` L44, `drought` L32, `farmer` L36; Brooklyn `storefront` L44, `pharmacies` L44, `businesses` L48; Queens `homeowners` L48, `vivienda` L44, `homeschooling` L24; Manhattan `gastronomy` L44, `attractions` L48, `tourists` L48, `busiest` L48.

**P15** `4_results.tex:261`, Bronx `cool` | edit | claim 2 | confidence: high

Current: "For the Bronx, the readout is nonsensical until about layer 20, where it sharply diverges ... from layer 40 to the last model layer, the J-space surfaces tokens relating to tourism."

The file gives L8 `instagram, spotify, musica, tiktok, musicians, youtube`; L12 `museum, railroad, banjo, music, arte`; L16 `parlor, handmade, wedding, jewelry, heirloom, ornament`. L40 holds `shopping, market, farmland, city, outskirts, town`; tourism words are at L44 (`tourism, tourist, visit, bustling`) and L48 (`touristic, tourist, tourism, walkable`). The figure ends at L40. Proposed:

> For the Bronx, the contrast holds music and craft words through layer 16 ("spotify", "musica", "banjo", "handmade", "heirloom"), and then sharply diverges at layer 20 from the generally-positive tone of the mean readout: "graffiti", "fucking", "shopping", "gallery", and "homeless". Layer 24 holds "porn", "anarch" (a word piece), "illegal", "vandalism", and "gangs". These severe descriptors disappear by layer 28, where "borough"/"bairro" stays prominent until layer 40; at layers 44 and 48, past the last column of \autoref{fig:borough-layers}, the contrast turns to tourism ("tourism", "tourist", "walkable", "bustling"), and at the last saved layer to "places" and "notorious".

Alternative: extend the figure to L56 and keep "from layer 44". The `mech_borough_layers.png` columns are L8, 12, 16, 20, 24, 28, 32, 40.

**P16** `4_results.tex:263`, Manhattan | check | claim 4 | confidence: high

"words relating to tourism and consumption dominate the rest of the model layers" after L32. The file gives L40 `venues, crowded, shopping, destination`; L44 `favorite, attractions, tourism, visitors`; L48 `tourist, hotspots, splurge`. L40 is in the figure; L44 and L48 are not. Add "(layers 40 to 48)" or extend the figure as in P15.

**P17** `4_results.tex:263`, Staten Island and Queens | edit | claim 2 | confidence: high

Current: "their layers 20--24 contain tokens relating to temperature and hydrology ("temperature", "seawater", "humidity", "equatorial"), and no harmful content is surfaced at any layer."

The temperature words start at layer 12 in both boroughs (Queens L12 `evapor, temperat, thermostat, humidity`; Staten Island L12 `evapor, sulphur, thermostat, temperature`), and in Staten Island they hold through the last saved layer (L56 `temperaturas, thermometer, humidity`). Proposed:

> Staten Island and Queens are not congruent with the other boroughs: from layer 12 their contrasts hold words of temperature and hydrology ("temperature", "seawater", "humidity", "equatorial"), Staten Island's through the last saved layer, and no word of the Bronx and Brooklyn kind appears at \emph{cool} at any layer.

Scope note: the "no harmful content" claim holds at `cool`. At `place`, the Queens contrast holds `incest` (L28 to L40), `psychotic`, `traumatic` (L32). Keep the claim scoped to `cool`.

**P18** `4_results.tex:265` | edit | claims 1, 2 | confidence: high

Current: "And so, we hypothesize that, if the model recognizes an image as being in an urban context (which it fails to do by layer 20 for the average Staten Island and Queens scene), layers 20-24 surface aesthetic \emph{properties}."

Does the hypothesis follow? The direction holds, the layer does not, and "aesthetic properties" is not what layers 20 to 24 hold:

- The split is the sense axis of \autoref{fig:prompt-readout} callout 4. The Staten Island and Queens words sit at the temperature end (axis rank 53,837 to 53,998 of 53,999), the Bronx and Brooklyn words at the social end (median rank 43 to 69) (report F10.18).
- The urban reading is supported by a probe, not a layer: the sense score of an image correlates with the model's own `urban` answer at $r=.70$, $.62$ (`sense_links.csv`), and by borough the `urban` probe is $-1.56$ in Staten Island and $-0.51$ in Queens against $+0.62$ in Manhattan (`probe_borough.csv`, Gemma, in SD of the NTA means). The `in a city` probe is positive on only 75\% (Qwen 94\%) of Staten Island images (`probe_place.csv`).
- The traces state the same choice: 23\%, 26\% of reasoning traces name temperature; in the temperature decile of images 52\%, 56\% do, in the social decile 6.5\%, 8.8\% (`sense_links.csv` share; report Q2).
- No layer test exists. "Fails to do by layer 20" cannot be claimed, and the temperature words begin at L12 (P17).
- The social end holds the sense of the word (boutique, hipster, bodega, bohemian), not aesthetic properties.

Proposed:

> We read this split as the sense of the word, not as a difference in judgment. Where the model reads the scene as urban, \emph{cool} takes its social sense at layers 12--32 ("boutique", "hipster", "bodega"); where it does not, as in the average Staten Island or Queens scene, \emph{cool} takes its temperature sense. An image's position between the two ends tracks the model's own answer to "Is this place urban?" ($r=.70$; Qwen $.62$), and the reasoning traces state the same choice: 23\% of Gemma-4-31B's traces (Qwen 26\%) ask whether 'cool' means temperature, 52\% in the most temperature-like decile of images against 6.5\% in the most social decile. The sense barely tracks the verdict ($r=.10$; Qwen $-.11$): an image can take the social sense and still get a 'no'.

**P19** `4_results.tex:261`, Brooklyn | add | claim 4 | confidence: medium

After the Brooklyn sentence, one clause separates Brooklyn from the Bronx with a tested number: the Brooklyn `cool` list holds on held-out neighborhoods (16 of 16, report "Paper words on the sense axis"; stability .76 over 40 splits, `sense_splits.csv` via report), and sits mostly on the sense axis (stability .43 with axis 1 out). Proposed:

> Brooklyn's list is stable across random splits of neighborhoods (mean overlap .76 over 40 splits); most of it sits on the same social sense of 'cool' (overlap .43 once that axis is removed).

**Caution:** do not add the parallel Bronx stability number (.22) as a verdict on the Bronx list. The author rejected that reading: the C3 contrast is unbiased by design, and a low split stability is consistent with a skewed borough mean (report: 10 of 111 images hold 56% of the top-word effect at `cool` L24).

### Results, claim 2 (place, not always correct)

**P20** `4_results.tex:273`, after "locations cited by the model for an image are not always correct" | add | claim 2 | confidence: high

The two map references are unresolved. The probe is the direct measure:

> A direct test makes the point without the traces. We ask both models "Is this place in New York?" over the 591 stratified street images, all of which are in New York City. Gemma-4-31B answers yes on 61.6\% of images and Qwen3.5-9B on 88.5\%, with a steep gradient from the core to the edge: Manhattan 83.3\% and 94.8\%, Brooklyn 78.0\% and 93.1\%, the Bronx 69.4\% and 91.9\%, Queens 41.2\% and 83.1\%, Staten Island 20.8\% and 72.9\%.

**Update (2026-11-02):** Qwen3.6-27B now has the same probe. See P45 for the 3-model sentence.

Source: `research/figures/f10/probe/probe_place.csv` rows `in New York`. **Note:** this is a forced one-word probe, not the accuracy of a stated location in free text (claim map, "No result exists for").

### Results, claims 3 and 4

**P21** `4_results.tex`, new paragraph after `:331` or in the hierarchy section after `:349` | add | claims 3, 4 | confidence: high

The value order of 'cool', from the 53 probes:

> What does 'cool' reward? We ask the same question with 53 other adjectives in place of 'cool' and relate each answer margin to the 'cool' margin, holding 'good', 'nice', and the borough fixed. The partial correlations order the values: interesting ($.87$; Qwen $.56$), unique ($.60$, $.43$), stylish ($.54$, $.46$), trendy ($.44$, $.49$) rise with 'cool'; ordinary ($-.44$, $-.56$), boring ($-.71$, $-.35$), clean ($-.38$, $-.29$), safe ($-.26$, $-.34$) fall with it. Value adjectives alone recover the 'cool' margin at $R^2=.92$ ($.81$), properties such as green or old at $.84$ ($.75$), and a neutral set at $.02$ ($.02$). For both models 'cool' reads as \emph{interesting, not ordinary} more than as \emph{pretty}; clean and safe work against it once general approval is held fixed.

Source: report Q3 and Q4 tables (`probe_links.csv`, `probe_model.csv`). The models disagree on `old` (.55, $-.02$), `gritty` (.47, .07) and `ugly` (.20, $-.37$); one clause could carry this to the discussion item "model is a hyperparameter".

**P22** `4_results.tex:349`, replace the graffiti sentence as the proof | edit | claims 3, 4 | confidence: high

Current (last two sentences): "This shows that, for a model, observed cues are used differently to justify different aesthetic judgments, a behavior that humans also exhibit: graffiti in Bushwick ... is a more negative signal."

The claim map notes that the restaurant block photographs are F6 views that the model did not judge, and that the Bushwick contrast has no test. The `cool` traces give the tested form:

> The traces of the 'cool' prompt make this explicit. We tag the object and the value in each span of 7,092 reasoning traces (three repeats of each image, two prompt forms). The value, not the object, gives the sign: brick counts for 'cool' in 91\% of spans under \emph{heritage} and in 0\% under \emph{utility} (Qwen 98\% and 9\%); graffiti in 86\% under \emph{edge} and 0\% under \emph{order} (97\%, 5\%); trees in 92\% under \emph{nature} and 2\% under \emph{utility} (99\%, 7\%). The sign depends on the value for 18 of 22 objects (Qwen 20 of 25; $\chi^2$, Holm). Repeats of one image agree more on the object than on the value (ICC $.55$ against $.31$; Qwen $.55$ against $.28$): the model is firmer in what it cites than in the standard it applies.

Source: report Q5, "One object, 2 signs" (`argument_cells.csv`, `argument_flips.csv`, `argument_repeat.csv`). Keep the Bushwick sentence as an illustration if you wish, but not as the proof.

**P23** `4_results.tex:338` section, after `:349` | add | claim 4 | confidence: medium

The borough order of 'cool' from the probes, as one sentence:

> Under the one-word prompt, the 'cool' margin orders the boroughs: in SD of the neighborhood means, Manhattan sits at $+0.95$ (Qwen $+1.16$), Brooklyn at $0.0$, Queens at $-0.20$ ($-0.31$), Staten Island at $-0.26$ ($-0.28$), and the Bronx at $-0.42$ ($-0.37$) ($F=11.9$, $18.7$; $\eta^2=.20$, $.28$). Status adjectives carry the top of that order: trendy, upscale, expensive, and fashionable give the largest Manhattan effects (about $+1.4$).

Source: `probe_borough.csv` (cool: $-.418$, $.015$, $.955$, $-.200$, $-.255$; $F$ 11.87, 18.74; $\eta^2$ .198, .281; trendy Manhattan 1.41); claim map C5.20. **Caution:** a borough is not an SES measure. The sentence supports the "hierarchy" half of claim 4; the "socioeconomic gradient" half stays on C5.1, C5.3, C5.4 and C5.6 (walkthrough, claim 4).

**P24** `4_results.tex:347`, "ordered consistently (by neighborhood, $\tau=[0.46,0.51]$ across different models" | check | claim 4 | confidence: medium

The claim map gives C4.7: area $\tau$ .46, .51, .51 at tract, NTA and district, and a pair-by-pair $\kappa$ of only .23. The sentence is correct at the area level; add "at the neighborhood level" so a reader does not take it as pair-level agreement. The F10 cross-model agreement on the 591 images is a separate, image-level number: the 'cool' margin agrees across the two models at $r=.72$ (C4.18, `probe_cross.csv`; not read by us, cite from the claim map).

### Results, claim 1, cross-model

**P25** `4_results.tex:22` area | add | claim 1 | confidence: high

The text reports Gemma-4-31B alone; the figure carries Qwen3.5-9B. One sentence states the replication:

> Qwen3.5-9B shows the same three parts at earlier layers (scene code L0--24, sense L4--28, standard L12--28, verdict at L28 against L0--32, L12--32, L32--44 and L44--56 for Gemma-4-31B), and the sense score of an image agrees between the two models at $r=.67$.

Source: `phase_layers.csv`; report Q2 ("The score of each image agrees between the 2 models (r 0.67)").

**P26** `4_results.tex`, end of the J-space section | add | claim 1 | confidence: low (slot only)

Mark a slot. State no number.

> % SLOT: Qwen3.6-27B (released J++ lens) and the prompts "Is this place nice?" and "Is this place ordinary?" are in progress. Report here (a) whether the three parts hold on a third model, and (b) whether the sense axis at the adjective token is specific to 'cool'. Do not cite the first Qwen3.6-27B verdict or margin files: a margin fault made them invalid.

The caller's note gives provisional cross-model word correlations (Gemma--Q27 .28, Gemma--Q9 .26, Q27--Q9 .44). We did not read their file. If you cite them, mark them provisional and name the file.

### Small fixes

| # | Site | Current | Proposed |
|---|------|---------|----------|
| P27 | `4_results.tex:215` | "We first performs readouts" | "We first rank the mean readout" |
| P28 | `4_results.tex:20` | "Neuropedia" | "Neuronpedia" |
| P29 | `4_results.tex:236` | "residencial", "archictectural" | Check both against the store. `residencial` may be a real Spanish or Portuguese word of the readout; `archictectural` is not a word of the store pattern and looks like a typo. |
| P30 | `4_results.tex:252` caption | "Words of top-12 contrast C3 at the \texttt{cool} token" | "Six of the top-12 words of contrast C3 at the \texttt{cool} token, layers 8 to 40" (the PNG shows 6 words per cell and no L36 column) |
| P31 | `4_results.tex:16` | "split into inferrable 16 chunks each by the model" | The 16 blocks are 4 by 4 pools that we set, not blocks the model makes (claim-map check 11). Write "pooled into 16 blocks". `fig:teaser` is also an unresolved reference. |
| P32 | `3_methods.tex:82` | "We generate readouts with a single NVIDIA B300 GPU." | The F10.6 workers ran on the `g3` partition as preemptible array tasks (report F10.23; `hop_all.sbatch`). Check which GPUs ran the 591-image stores before you keep "a single B300". |
| P33 | `4_results.tex:280` heading | "Models demonstrate an awareness of the task requirements" | The claim is commented out in the list at `:7`. The subsubsection still carries its heading. Flag only. |
| P34 | `main.bib:71` | author `{Kola Ayonrinde}` | The LessWrong post lists Kola Ayonrinde and Jack Lindsey. Flag only. |

## 4. Claims without support

Prose claims that we could not trace, or that the data contradict. A contradiction is a
finding; we report it and do not smooth it.

| # | Site | Claim | Status | Evidence |
|---|------|-------|--------|----------|
| U1 | `4_results.tex:259` | "Recognition-related tokens appear starting at Layer 44, in Spanish and Portuguese (not English)" | Contradicted | `borough_words.csv`: Bronx `place` L44 `by_raw` holds `recognize` and `familiarity`. See P13. |
| U2 | `4_results.tex:259` | "'ninguna' (none)" at layer 44 | Contradicted | `ninguna` is at L56, not L44. |
| U3 | `4_results.tex:259` | "No other borough's J-space exhibits this behavior" (multilingual localization) | Partly contradicted | Every borough holds words in several languages at `place`. The geography and recognition content is Bronx-specific. See P14. |
| U4 | `4_results.tex:261` | "the readout is nonsensical until about layer 20" (Bronx `cool`) | Contradicted | L8 to L16 hold music and craft words. See P15. |
| U5 | `4_results.tex:261` | "from layer 40 to the last model layer, the J-space surfaces tokens relating to tourism" | Off by one column | Tourism words sit at L44 and L48. L40 holds `shopping, market, farmland, city`. The figure ends at L40. |
| U6 | `4_results.tex:263` | "their layers 20--24 contain tokens relating to temperature" | Incomplete | The temperature words start at L12 in both boroughs. |
| U7 | `4_results.tex:265` | "(which it fails to do by layer 20 for the average Staten Island and Queens scene)" | No test | No layer-resolved test of urban recognition exists. The `urban` and `in a city` probes are final answers. See P18. |
| U8 | `4_results.tex:265` | "layers 20-24 surface aesthetic \emph{properties}" | Not what the data hold | The social end of the axis holds the sense of the word (boutique, hipster, bodega). The report calls it "the social sense of the word". |
| U9 | `4_results.tex:243` | The order "localization, defining the standard, applying standards, judgment" | Not supported by onset | Report F10.13: "Onset alone does not separate stages." Use the token order of the figure (P11). |
| U10 | `4_results.tex:228` | "metropolitan" as a top word at `place` | Not found at L24 | `stage_words.csv` gives rank 273 at L24. Check other layers. |
| U11 | `4_results.tex:234` | The `model` words as top words of the readout | Contradicted for the mean readout | They rank 330 to 11,460 in the mean readout, and 4 to 19 in C2. See P10. |
| U12 | `3_methods.tex:80` | The spider example shows 'spider', then 'legs', then '8' | `legs` not found | The first third of the source reports `spider` in the readout and `8` as the output. See P1. |
| U13 | `3_methods.tex:84` | "Answer in one word" | Contradicted | The run used "Answer with one word". |
| U14 | `4_results.tex:349` | "graffiti in Bushwick ... is more likely to be perceived as 'trendy', whereas graffiti in Kip's Bay ... is a more negative signal" | No test | No neighborhood-level test of the graffiti sign exists in the records we read. The object-by-value test (P22) is the tested form. |
| U15 | `4_results.tex:347` | "avoids restaurants adjacent to construction, older infrastructure, and working-class businesses (e.g., laundromats)" | Weak for laundromats | Claim map, limits of claim 4: `laundromat` lift $-0.11$ in pairwise traces and $-0.02$ under F9. |
| U16 | `4_results.tex:16`, `:273` | `fig:teaser`, `fig:misnaming-map`, `fig:hashtag-map` | Unresolved references | The maps are commented out at `:288` to `:300`. P20 gives the probe as the proof of "not always correct". |
| U17 | `1_intro.tex:8` and the runbook | Deadline Nov 15 against 2026-11-03 | Conflict | Flag only. |
| U18 | `fig:prompt-readout` callout 7 | "The read of this image holds the words that the model cites for it in a separate run with thinking on." | Supported, with a limit | 194 of 375 cells (Qwen 73 of 192) pass Holm (report F10.12). The image holds both the readout and the argument, so this is shared content, not a cause. Add "in a separate run" is already there; consider "shares" in place of "holds". |

## 5. F10.26: 14 prompts (2026-11-02)

Written against `artificial-aesthetic/paper` at `6de388d`. Line numbers are from that commit.
Each item gives the site, the text now, the proposed text and the result that supports it.
The proposed text is in the author's voice. The support is in `f10-judgment-mechanism.md`,
section "14 prompts: results (F10.26)", and in `research/figures/f10/prompts/`. A cell is 1 prompt in 1
model, of 42.

**P35** `sec/4_results.tex:6` | change | claim 1 | confidence: high

Text now: "Three parts separate in the data, and all three replicate across models (values give Gemma-4-31B, then Qwen3.5-9B and Qwen3.6-27B in parentheses)."

Proposed:

> Three parts separate in the data. All three replicate across models (values give Gemma-4-31B, then Qwen3.5-9B and Qwen3.6-27B in parentheses), and the first two hold across 14 prompts as well, whereas the third is specific to 'cool' (see below).

Support: `t2_counts.csv`. Scene code 39 of 42, step at the adjective 41 of 42, sense band 6 of 42 (the 6 `cool` cells).

**P36** `sec/4_results.tex:202`, the comment slot | add | claim 1 | confidence: high

Text now: "% F10.26 (14 prompts ...): add here which parts of the judgment generalize over the 14 prompts, and which do not."

Proposed paragraph:

> \paragraph{What generalizes over 14 prompts.} We repeat the measures of \autoref{fig:prompt-readout} on all 14 prompts and count the prompt--model cells, of 42, in which each result holds. The scene code holds in 39: a decoder on the mean patch recovers the margin at out-of-fold $R^2\geq.50$ in every cell except Gemma-4-31B on 'ugly', which the model answers for 20 and 83 of the 591 images only, and Qwen3.6-27B on 'scene ordinary' ($.47$, with a yes rate of $.98$). The step at the adjective holds in 41, and so does the gain at '?' (the exception is again Gemma-4-31B 'place ugly'). The embedded conception at the adjective holds in all 42: at mid depth, at least 50 of the top-100 words of the mean readout are in the top 100 with no image. The sense part does not generalize. The temperature-to-social axis passes the band rule in the six 'cool' cells and in none of the 36 others, and for the other adjectives the largest axis of the readout at the adjective follows the answer (29 of 36 cells, $|r|$ up to $.97$ for 'urban'), whereas for 'cool' it never does ($|r|\leq.19$). Its ends name kinds of place rather than senses of the word: for 'nice', 'ordinary' and 'fancy' in the Qwen models, the axis runs from residential and suburban words to tourist and commercial words. The verdict band at $r\geq.95$ is reached in 34 cells; the eight misses peak between $.92$ and $.94$. The layers of each band follow the model, not the prompt. The noun changes the level of the answer but not its order: 'scene' moves the yes rate of 'cool' from $.38$ to $.72$ in Gemma-4-31B and from $.92$ to $.39$ in Qwen3.6-27B, while the neighborhood means of the two nouns correlate at $r=.78$--$.89$ and the Manhattan sign is the same for all 21 adjective--model pairs.

Support: `t2_grid.csv`, `t2_counts.csv`, `t3_sense_tests.csv`, `t3_axis_words.csv`, `t6_noun_margins.csv`, `t1_outcomes.csv`.

**P37** `sec/4_results.tex:199`, end of the paragraph "One aesthetic direction across adjectives and models" | add | claim 1 | confidence: high

Proposed sentences, after "...but the top words of that direction do not name visual objects":

> A ridge decoder of each margin from the same residual at the noun, fit in five folds by neighborhood, gives the same picture without the lens. The 'nice' and 'pretty' decoders share a direction (cosine $.63$, $.78$, $.76$), 'cool' sits near them ($.36$--$.56$ with 'nice'), 'ugly' runs against them (cosine $-.09$ to $-.60$), and the 'ordinary' and 'urban' decoders are orthogonal to all of them ($|\mathrm{cosine}|\leq.33$). The first singular direction of the 13 other decoders, fit in the training folds, predicts a held-out 'nice' or 'pretty' margin at out-of-fold $R^2=.52$--$.79$, but 'ordinary' and 'urban' at $\leq.18$; a rank-3 fit recovers them ($R^2=.68$--$.88$), so the noun holds at least three answer-relevant directions, of which the aesthetic one is shared. 'Ugly' is the mirror of 'pretty' in the answer (neighborhood means correlate at $r=-.86$ to $-.91$) but not in the residual (decoder cosine $-.09$ to $-.60$).

Support: `t7_cosines_noun.csv`, `t7_loo.csv`, `t7_rank_noun.csv`, `t4_pretty_ugly.csv`.

**P38** `sec/4_results.tex:212`, last sentence of "Across boroughs" | change | claims 2 and 4 | confidence: high

Text now: "The three models share an order of boroughs (\autoref{results:hierarchy-boroughs}) but not its words: the top-50 words of C3 at 'cool' barely overlap between any pair of models."

Proposed:

> The three models share an order of boroughs on each adjective (\autoref{results:hierarchy-boroughs}), but the order depends on the adjective, and the words depend on both. Manhattan leads on 'cool', 'fancy' and 'urban' in all 18 prompt--model cells and sits last on 'ordinary' in all six; Staten Island leads on 'nice' (six of six) and 'pretty' (four of six), and the Bronx on 'ugly' (five of six). The top-50 words of C3 at the adjective barely overlap between any pair of models, or between any two adjectives within a model (Jaccard $\leq.01$), although each list is stable across random halves of the neighborhoods (split-half $r=.59$--$.78$ in every cell).

Support: `t8_borough_order.csv`, `t8_jaccard.csv`, `t8_stability.csv`.

**P39** `sec/4_results.tex:212`, the sentence on "Is this place urban?" | change | claim 2 | confidence: high

Text now: "...an image's position between the two senses tracks the model's answer to \"Is this place urban?\" ($r=.70$; Qwen3.5-9B $.62$)."

Proposed:

> ...an image's position between the two senses tracks the model's answer to "Is this \{place, scene\} urban?" (neighborhood means, $r=.78$, $.74$, $.68$ for 'place' and $.81$, $.61$, $.72$ for 'scene') far more than it tracks the 'cool' answer itself ($r\leq.22$). The 'urban' answer has one borough order in all three models and both nouns (Manhattan, Brooklyn, Bronx, Queens, Staten Island), and the borough differences of the other adjectives remain when the 'urban' margin is held fixed (34 of 36 cells).

Support: `t5_urban_sense.csv`, `t5_urban_borough.csv`, `t5_urban_partial.csv`. The old values were image-level r for 2 models; the new values are NTA means for 3 models.

**P40** `sec/4_results.tex:251`--`252` | change | claim 4 | confidence: high

Text now: "The one-word prompts give the same order at the level of boroughs, in all three models. Relative to the other boroughs, Manhattan's neighborhoods sit higher on 'cool' ($t=6.7$, $8.5$, $7.5$) and lower on 'ordinary' ($-6.7$, $-7.1$, $-6.5$); these six differences pass Bonferroni correction over the nine tests ($p<.001$). They also sit higher on 'nice' ($2.3$, $2.9$, $2.3$), but only the Qwen3.5-9B difference passes ($p=.04$; Gemma-4-31B $.23$, Qwen3.6-27B $.20$)."

Proposed:

> At the level of boroughs, the order depends on the adjective. Over the 14 prompts and three models (42 tests, Holm correction), Manhattan's neighborhoods sit higher than the rest on 'cool', 'fancy' and 'urban' in every cell ($t=4.2$--$10.5$) and lower on 'ordinary' in five of six ($t=-3.6$ to $-7.1$). On 'nice' the difference passes in no cell, on 'pretty' in one (Qwen3.5-9B, 'place', $p=.04$) and on 'ugly' in two (Qwen3.5-9B, negative); on 'nice' and 'pretty', Staten Island's neighborhoods sit highest in ten of twelve cells.

Support: `t8_manhattan.csv` (`p_holm`, family 42), `t8_borough_order.csv`.

**P41** `sec/3_methods.tex:79` | add | methods | confidence: high

Text now ends: "\emph{Verdict}: the yes-no margin of the readout at the last token gives $r\geq.95$ with the answer margin."

Proposed addition:

> The sense rule is a rule for 'cool'; for the other adjectives we ask whether the first axis of the readout at the adjective follows the answer. To test generality, we repeat the decoder, band and borough measures on all 14 prompts and count the prompt--model cells, of 42, in which a result holds; a result generalizes when it holds in every cell.

Support: `t2_counts.csv`, D174.

**P42** `sec/3_methods.tex:81`, after "The decoder on the mean image patch is the base..." | add | methods | confidence: medium

Proposed:

> As a check on the shared direction without the lens, we also fit one ridge decoder per prompt from the same residual at the noun, take the first singular direction of the unit decoders of 13 prompts in the training folds, and test it on the 14th prompt out of fold.

Support: `t7_loo.csv`, D175. Add only if P37 is accepted.

**P43** `sec/4_results.tex`, the paragraph of P36 or the limitations | add | claims 1 and 3 | confidence: high

Proposed:

> Gemma-4-31B declines to answer 'ugly' for 571 of the 591 images with 'place' and 508 with 'scene', and 'nice' for 295 with 'place'; the Qwen models decline none. Its refusals of 'nice' are not uniform: the first token is neither yes nor no for 60\% of the Bronx images and 32\% of the Manhattan images. Our results on Gemma-4-31B 'ugly' rest on 20 and 83 answers and we do not interpret them.

Support: `t1_outcomes.csv`, `t1_outcome_tests.csv` (borough x outcome, Holm p < .001 for Gemma `place nice`).

**P44** `sec/4_results.tex:199`, the sentence "'Ordinary', then, is not the opposite of the aesthetic adjectives" | keep | claim 1 | confidence: high

No change. F10.26 supports it in the answer too: the 'cool' and 'ordinary' margins correlate at $r=-.05$ to $-.39$ over neighborhoods, against $-.86$ to $-.91$ for 'pretty' and 'ugly'. Support: `t4_pretty_ugly.csv`.

### Edits that F10.26 does not support

| Site | Do not write | Reason |
|------|--------------|--------|
| Any | "all three parts generalize over the 14 prompts" | The sense part is specific to `cool` (6 of 42) |
| Any | "Manhattan sits higher on every aesthetic adjective" | `nice` 0 of 6, `pretty` 1 of 6; Staten Island leads there |
| Any | "the shared direction is the only answer-relevant direction at the noun" | `ordinary` and `urban` need 2 more directions |
| Any | a socioeconomic reading of the borough order | F10.26 holds no ACS test; F10.28 does |
| Any | Qwen word lists of the shared direction through the CPU lens as exact | The CPU readout reproduces the Qwen stores at r .70 |

### Place probes on the third model (2026-11-02)

**P45** `4_results.tex:269`, the sentence "Gemma-4-31B answers yes on 61.6\% of images and Qwen3.5-9B on 88.5\%" | replace | claim 2 | confidence: high

The Qwen3.6-27B probe run exists now. The gradient is the same in all 3 models:

> Gemma-4-31B answers yes on 61.6\% of images, Qwen3.5-9B on 88.5\% and Qwen3.6-27B on 88.8\%, with a steep gradient from Manhattan (83.3\%, 94.8\%, 94.8\%) to Staten Island (20.8\%, 72.9\%, 70.8\%).

Source: `research/figures/f10/probe/probe_place.csv` rows `in New York`. The larger Qwen is not more accurate than the small one: Queens .870 against .831, Staten Island .708 against .729.

## 6. F10.29: seven questions (2026-11-02)

Source: `f10-29-seven-questions.md`. Questions 5 and 7 (steering, other tokens) are not in these proposals yet.

**P46** `4_results.tex`, the ACS paragraph of section 4.2 (`tab:prompt-ses`) | add | claim 4 | confidence: high

> The Black share of an NTA carries no gradient of its own. Given median household income, the partial rank correlation of the expected answer with the Black share is zero in all 18 cells of `ugly`, `nice` and `cool` (median |rho| .02, no cell passes Bonferroni), while income given the Black share passes in 17 of 18 (median |rho| .31).

Source: `research/figures/f10/hierarchy/ses_partial.csv`. **Do not write** that income causes the judgment, or that the model reads income. The test says that 1 gradient carries the other.

**P47** `4_results.tex`, the shared-direction paragraph of section 4.2 | add | claim 4 | confidence: high

> The direction is shared across models and not only across prompts. The mean of the directions of two models, on the 36,490 lens words that all three share, predicts the NTA-level answers of the third model on its own 14 prompts: all 14 pass for Qwen3.5-9B, 11 for Qwen3.6-27B and 6 for Gemma-4-31B (Bonferroni), with the sign reversed for `ugly` in all three. For `nice` and `pretty` the transferred direction reaches the model's own leave-one-prompt-out ceiling.

Source: `research/figures/f10/transfer/transfer_test_layer.csv`.

**P48** `3_methods.tex`, the sample paragraph, or the limits | add | all claims | confidence: high

> Three images per NTA give a noisy NTA score: the NTA explains a median 19% of the image-level variance (ICC(1)), and the three-image mean has a Spearman-Brown reliability of .42. The gradients we report rest on the 197 NTAs, not on the three images: with one image per NTA, the income test passes in 1 cell of 38, against 25 with three. The noise attenuates the measured gradients; the median |rho| with income is .33 measured and .50 after the reliability correction.

Source: `research/figures/f10/sample/summary.csv`, `cells.csv`. **Do not write** "even a small sample reveals the bias".

**P49** `3_methods.tex:79`, the phase rules | replace the fixed layers with a sentence on the thresholds | claim 1 | confidence: high

> The scene code, the step at the adjective and the verdict hold over a wide range of thresholds (R2 .30 to .70: 42 to 33 of 42 cells; gain .02 to .08: 41 to 37; verdict r .85 to .92: 41 of 42, .95: 34). The layer of each band moves by one saved layer when the threshold moves by .05. The temperature sense of `cool` on its first axis is a property of that word and of layers 12 to 32 (6 of 42 at every threshold).

Source: `research/figures/f10/robust/sweep.csv`, `band_edges.csv`. The Figure 1(b) bands should carry the threshold in the caption.

**P50** `4_results.tex`, the Gemma refusal note | add | claim 3 | confidence: moderate

> Gemma-4-31B declines the verdict on `ugly` for 97% of images and on `nice` for 50%, with the first token "Subjective", "Beauty" or "Unable", and it declines least in Manhattan (32% on `nice`, against 59% to 60% in Brooklyn and the Bronx). At the last token, the lens words `subjective`, `opinion` and `beauty` rise with the refusal share across NTAs (r .4 to .7). The refusal marks the task as contestable; it does not show awareness of harm, which no prompt asks about.

Source: `research/figures/f10/aware/unable_share.csv`, `refusal_words_by_layer.csv`.

### Edits that F10.29 does not support

| Site | Do not write | Reason |
|------|--------------|--------|
| Any | "the model reads income" or any causal order from a partial correlation | The test holds 1 feature fixed; the model sees pixels |
| Any | "the models share a mechanism" from the transfer | The transfer is in the lens-word space on the same images; it shows a shared ranking |
| Any | "3 images per NTA are enough" | The reliability of the NTA score is .42 |
| Any | "the model is aware of downstream harms" | A refusal with "subjective" is a sign that the task is contestable, not of harm awareness |

## 7. F10.30: more images for each NTA (2026-11-03)

**Applied on 2026-11-03** in the paper at `3b28f30` (on top of the Overleaf pull `03992d0`): P51
with P56, P52 (the table through the manifest, `artificial-aesthetic` at `f926a3d`), P53, P55,
P57 and P58. P54 followed on the user's request (paper at `0f4223e`): the Manhattan sentence uses the Welch t of the 50-image means of E, with the 3 `urban` tests. The
paperkit build writes the PDF (33 pages), but `latexmk` returns 12 because `main.bib` from
Overleaf holds 3 repeated entries (lines 1313, 1415, 1584); this is not from these edits.

Source: `f10-30-more-images.md`. Written against `artificial-aesthetic/paper` at `666cf8c`.
F10.30 scored the answers of the 3 models to the 14 prompts on 10,000 street images (50 or 51 for
each of the 197 NTAs; the 591 images of the paper are the first 3 of each NTA), and captured the
lens readouts on the first 10 of each NTA (1,970 images). The paper's 591-image numbers are not
wrong. The proposals below replace them with the numbers at 50 images for each NTA where the
paper reports an answer gradient, and add 1 sentence where the lens results hold on new images.
A cell is 1 prompt in 1 model. k is the number of images for each NTA.

| Decision for the author | Recommendation |
|-------------------------|----------------|
| Which k for the answer gradients of section 4.2 and `tab:prompt-ses` | k = 50 (P51, P52). The reliability of the NTA mean is .90 against .39 at k = 3, and the test is the same |
| Which k for the lens results (the direction, Figure 3, the transfer) | Keep the 591-image fit as the object of the paper. Add 1 sentence that the direction, its words and the transfer hold on the 10-image means (P55 to P57) |
| The Black-share sentence | It must change. At k = 50 the Black share passes on its own in 14 of 24 aesthetic cells, but given income in 0 of 18 (P51) |

**P51** `sec/4_results.tex:160`, the ACS paragraph of section 4.2 | replace | claim 4 | confidence: high

Text now (the parts with numbers): "...across the 197 neighborhoods of the 591 images. ... On the aesthetic prompts ('cool', 'nice', 'pretty', 'fancy'), $E$ rises with median income in 23 of 24 pairs of model and prompt ($\rho$ from $.16$ to $.46$), with the share of college graduates in 16, and with the share of White residents in 14. 'Ugly' reverses in every pair (income $\rho$ from $-.44$ to $-.34$), and 'ordinary' and 'urban' pass in none. Where Gemma-4-31B declines to answer, it declines more often in poorer neighborhoods ($\rho$ with income from $-.44$ to $-.36$). ... Every correlation of an aesthetic answer with the share of Black residents is negative ($\rho$ from $-.19$ to $-.04$), and none passes; with median income held fixed, the share of Black residents relates to none of the 18 answers to 'cool', 'nice', and 'ugly' (median partial $|\rho|=.02$), while income, with the share of Black residents held fixed, relates to 17 ($.31$; Bonferroni over 18)."

Proposed:

> The J-space and the traces of the one-word prompts carry the same gradient across the 197 neighborhoods. We score the answer to each of the 14 prompts as the expected answer $E=P(\text{yes})-P(\text{no})$, which places an answer that is neither yes nor no between the two, and relate its neighborhood mean over 50 images to four ACS features (Spearman $\rho$, Bonferroni over 188 tests; \autoref{tab:prompt-ses} in the appendix). On the aesthetic prompts ('cool', 'nice', 'pretty', 'fancy'), $E$ rises with median income in all 24 pairs of model and prompt ($\rho$ from $.29$ to $.68$), with the share of White residents in 24, and with the share of college graduates in 23. 'Ugly' reverses in 5 of 6 pairs (income $\rho$ from $-.59$ to $-.42$; the sixth is Gemma-4-31B on 'place', which declines the verdict on 97\% of images), and 'ordinary' and 'urban' pass in none. Where Gemma-4-31B declines to answer, it declines more often in poorer neighborhoods ($\rho$ with income $-.45$ and $-.53$ for 'nice', $-.42$ for 'ugly' on 'scene'). The shared direction and the net valence of the cited evidence follow the answers (income $\rho=.32$ and $.25$ for each; Bonferroni over 20 tests). Every correlation of an aesthetic answer with the share of Black residents is negative ($\rho$ from $-.40$ to $-.14$; 14 of 24 pass), but with median income held fixed, the share of Black residents relates to none of the 18 answers to 'cool', 'nice', and 'ugly' (partial $\rho$ from $-.12$ to $.13$), while income, with the share of Black residents held fixed, relates to 17 (median $|\rho|=.44$; Bonferroni over 18).

Support: `research/figures/f10/more/ses.csv` (k = 50: 120 of 188 tests pass, against 81 at k = 3),
`partial.csv`. The sentence on the shared direction and the evidence keeps the 591-image numbers,
because those measures are lens and trace measures (see P56). **Do not write** that the Black
share "carries no gradient" or that "none passes": at 50 images it passes on its own in 14 cells.
The claim that survives is the partial one: given income, it adds nothing that the test can see.
This is not a causal order (the methods already say so).

**P52** `sec/B_addtl-results.tex:241` and the table `tab:prompt-ses` | replace | claim 4 | confidence: high

Text now: "The estimates are conservative: with three images, a neighborhood mean has a median reliability of $.42$, and the median income $|\rho|$ rises from $.33$ to $.50$ after correction for it."

Proposed:

> Each neighborhood mean takes 50 images, and has a median reliability of $.90$ (ICC(1) $.15$; Spearman--Brown). With the first three images of each neighborhood, the 591 images of the lens analyses, the reliability is $.39$, the median income $|\rho|$ is $.33$, and the same correlations rise to $.50$ after correction for the reliability; at 50 images the measured value is $.47$. The table at three images is in the supplement.

The table: `research/notebooks/facct/tables/f10_prompt_ses_k50.tex` holds `tab:prompt-ses` at k = 50, in
the same layout (120 of 188 tests pass). `paperkit` can copy it over `tables/facct/prompt-ses.tex`.
Keep the k = 3 table as `tables/facct/prompt-ses-k3.tex` in the supplement, or drop it.

Support: `research/figures/f10/more/reliability.csv`, `ses.csv`, `summary.json`.

**P53** `sec/A_addtl-methods.tex:200`, the reliability paragraph | replace | all claims | confidence: high

Text now: "We take ICC(1) of the answer margin over the three images of each NTA (one-way ANOVA), and the reliability of the mean of three images by the Spearman--Brown rule. The corrected $\rho$ is the measured $\rho$ divided by the square root of that reliability; it assumes that the ACS feature has no noise."

Proposed:

> We take ICC(1) of $E$ over the images of each NTA (one-way ANOVA), and the reliability of the mean of $k$ images by the Spearman--Brown rule. ICC(1) is $.15$ to $.19$ (median over the 14 prompts, each model) whether an NTA holds 3, 10 or 50 images; the reliability of the mean is $.39$, $.67$ and $.90$. The corrected $\rho$ is the measured $\rho$ divided by the square root of that reliability; it assumes that the ACS feature has no noise.

Support: `research/figures/f10/more/reliability.csv`.

**P54** `sec/4_results.tex:162`, the Manhattan sentence | optional | claim 4 | confidence: moderate

Text now: "Relative to the other boroughs, Manhattan's neighborhoods sit higher on 'cool' ($t=6.7$, $8.5$, $7.5$) and lower on 'ordinary' ($-6.7$, $-7.1$, $-6.5$); these six differences pass Bonferroni correction over the nine tests ($p<.001$)."

At k = 50, with $E$ as the score (not the margin), the Welch $t$ of the NTA means of 'place' are: 'cool' $9.9$, $14.2$, $10.1$; 'ordinary' $-5.3$, $-10.6$, $-8.6$; 'urban' $6.5$, $11.5$, $11.9$. All 9 pass. The sentence now uses the margin on 591 images, where Gemma-4-31B 'ordinary' is at the bound (97\% yes) and $E$ gives $t=-2.4$ at k = 3. Thus the 2 scores differ on that cell. If the author moves the sentence to $E$ at 50 images, write:

> Relative to the other boroughs, Manhattan's neighborhoods sit higher on 'cool' ($t=9.9$, $14.2$, $10.1$) and lower on 'ordinary' ($-5.3$, $-10.6$, $-8.6$); all six differences pass Bonferroni correction over the nine tests ($p<.001$), as do the three on 'urban'.

Otherwise leave the sentence. Support: `research/figures/f10/more/manhattan.csv`.

**P55** `sec/4_results.tex`, the shared-direction paragraph of section 4.2 (the text of P47) | add 1 sentence | claim 4 | confidence: high

Proposed, after the transfer sentence of P47:

> The transfer also holds on held-out images: with the neighborhood means of ten new images per NTA as the target, the transferred direction passes for 9, 14 and 14 of the 14 prompts (Gemma-4-31B, Qwen3.5-9B, Qwen3.6-27B), against 6, 14 and 11 on the three images it was fit on.

Support: `research/figures/f10/more/transfer/transfer_test_layer.csv`, variant `main`. The source
directions are the 591-image fits; the target NTA means are from the 10-image summaries, of which
7 of 10 images are new.

**P56** `sec/4_results.tex:160`, the sentence "The shared direction and the net valence of the cited evidence follow the answers (income $\rho=.32$ and $.25$ for each; Bonferroni over 20 tests)" | add a clause | claim 4 | confidence: high

Proposed:

> The shared direction and the net valence of the cited evidence follow the answers (income $\rho=.32$ and $.25$ for each; Bonferroni over 20 tests), and the direction keeps its gradient on ten new images per neighborhood ($\rho=.41$ and $.25$).

Support: `research/figures/f10/more/ses_readout.csv`. Qwen3.6-27B gives $.20$ there and does not pass
(Bonferroni over 12). The lens gradient is smaller than the answer gradient at the same k; do not
write that they are equal.

**P57** Figure 3 (`shared_direction_words`) and its prose in section 4.2 | change the quoted words; add 1 caption sentence | claim 4 | confidence: high

The direction fit again on the 10-image summaries has cosine $.94$ to $.97$ with the 591-image
direction at the test layers. The layer-mean word lists of F10.28 agree at Spearman $\rho$ $.95$ to
$.97$ over the vocabulary, and 10 to 15 of the top 20 words at each end are the same.

Proposed caption sentence:

> The lists change little when the direction is fit on ten images per neighborhood instead of three (Spearman $\rho$ from $.95$ to $.97$ over the vocabulary; 10 to 15 of the 20 words at each end are the same).

Quoted words to keep or drop, by their rank at k = 3 and at k = 10:

| Model | Keep (rank k = 3, k = 10) | Drop or replace (rank k = 3, k = 10) |
|-------|---------------------------|--------------------------------------|
| Gemma-4-31B | flower (2, 1), beautiful (7, 11), bouquet (12, 9), cherished (4, 7); welding (9, 10), smelting (4, 12), contaminants (6, 13), trabajadores (14, 17) | ornaments (20, 26); industrial (8, 44), replace with `crude` or `contaminate` |
| Qwen3.5-9B | exquisite (3, 2), majestic (4, 3), elegant (5, 5), queen (7, 4), stunning (1, 14); abuses (15, 9), slaughter (11, 10), `hitam` (12, 12) | regal (10, 21); warehouses (3, 39), unemployed (20, 31), dumpsters (8, 21), replace with `abusive` or `industriale` |
| Qwen3.6-27B | exquisite (3, 1), elegant (12, 4), beautifully (5, 7), stunning (7, 9); dumpster (1, 1), gang (8, 12), thugs (7, 16), job (2, 19) | splendid (28, 41); factory (15, 89), warehouses (13, 59), replace with `stripper` or `makeshift` |

Support: `research/figures/f10/more/words.csv`, `words_agree.csv`. The figure itself stays on the
591-image direction. `hitam` holds its rank in both fits.

**P58** `sec/3_methods.tex`, the sample sentence of the F10 analyses, or `sec/A_addtl-methods.tex`, lane B of the flow chart | add 1 sentence | all claims | confidence: high

The flow chart already says "985 images, then 10,000; the first 5 of each NTA; the first 50 or 51 for the larger batch". The text must say which analyses use which batch:

> The answer gradients of \autoref{results:hierarchy-boroughs} use the first 50 images of each NTA (9,850 images, scored with vLLM); the lens readouts, the shared direction and the reasoning traces use the first three (591 images); the held-out tests of the direction use the first ten (1,970 images).

Support: `data/f10/hop_more/images_manifest.json`; `f10-30-more-images.md`, "The sample". The vLLM
scores reproduce the HF scores on the 591 images (Spearman $\rho$ $.96$ to $1.00$ in 38 of 42 cells;
the 4 others are cells at the bound where mean $E$ agrees to 2 decimals).

### Edits that F10.30 does not support

| Site | Do not write | Reason |
|------|--------------|--------|
| Section 4.2 | "the share of Black residents carries no gradient" | At 50 images it passes on its own in 14 of 24 aesthetic cells. Only the partial given income is 0 of 18 |
| Section 4.2 | "more images make the gradient steeper" as a property of the model | The reliability of the NTA mean grows with k; ICC(1) does not. The gradient was there, under noise |
| Section 4.2 | "the readout carries the same gradient as the answer" | At k = 10 the direction gives income $\rho$ .41, .25, .20; the answers give .40 to .68 at k = 50 |
| Appendix A | "the test layer is L8" (or L28, L60) as a fixed fact | The leave-1-out rule picks L4 and L12 on the 10-image fit for 2 models; the held-out r is flat over the layers. Name the band |
| Any | "the vLLM and HF answers are identical" | They agree in rank order; the 4 cells at the bound differ in the tail |

## 8. The review of the paper against the record pages (2026-11-03)

The user asked for a review of the results of the paper against the record pages
of 2026-10-29 to 2026-11-03. The review found 1 strong result and 6 moderate
results that the paper did not hold, and 1 caveat that the review page
`f9-h9-review.md` ranks high. All of them are applied (paper commit after
`a9bce82`, 2026-11-03). Each item names its source page and its site in the paper.

| Item | Source | Site in the paper | Status |
|------|--------|-------------------|--------|
| P59. The steering test: the shared direction, added from the noun token onward, moves the verdict in all 3 models, each in 1 band of layers; a random vector does not; `ugly` reverses | `f10-29-seven-questions.md` question 5 | New paragraph "The direction moves the answer" in section 4.1, after "One aesthetic direction"; new Appendix B subsection `app:steering` with `tab:steering-dose` (40-image dose run, by layer) | applied |
| P60. The refusals name the task as a matter of taste, and the last-token readout says `subjective`, `opinion`, `matter` on `nice` and `beauty`, `unable`, `cannot` on `ugly`; the 4 thinking models take no side in 40% to 85% of the judgment answers | `f10-29-seven-questions.md` question 4; `f9-subjective-prompt.md` section 4.7 | Section 4.1, the paragraph "The standard sets the sign of an object", after the doubt sentence | applied |
| P61. The refusal share of `nice` falls with the White share (-.47, -.51) and rises with the Black share (.31, .32) at 50 images; Manhattan has the fewest refusals | `f10-30-more-images.md` | Section 4.2, the ACS paragraph, the refusal sentence | applied |
| P62. Qwen3.6-27B on the "in New York" probe: .888, Manhattan .948, Staten Island .708 (P45) | `f10-judgment-mechanism.md` task 5 | Section 4.3, the "in New York" sentence, now 3 models | applied |
| P63. Each standard ranks the boroughs in its own order at 50 images: Manhattan first on `cool`, `fancy`, `urban` (18 of 18), Staten Island first on `nice`, `pretty`, `ordinary` (18 of 18), the Bronx first on `ugly` (5 of 6) and last on `nice`, `pretty`, `fancy` (18 of 18) | new stage `hop_more_answers.py --borough`, `research/figures/f10/more/borough_order.csv` | Section 4.2, after the Manhattan sentence; new Appendix B subsection `app:borough-order` with `tab:borough-order` | applied |
| P64. `ugly` mirrors `pretty` in the answers (r -.86, -.88, -.91) but not in the residual (cosine -.09, -.37, -.60), because `ugly` + `urban` lands near minus `pretty` | `f10-judgment-mechanism.md` task 4; `f10-34-sum-directions.md` | Section 4.1, the pair sentence of "One aesthetic direction" | applied |
| P65. The noun changes the level of the answer, not the order: `cool` yes share .38 to .72 (Gemma), .92 to .39 (Qwen3.6-27B); NTA means agree at r .61 to .99, Manhattan sign the same in 21 of 21 | `f10-judgment-mechanism.md` task 6 | Section 4.2, after the borough order | applied |
| P66. H9 in both forms: raw 34 of 60 (pre-registered), standardized 23 of 60, about 1.4 times against 4; the descriptive net valence is exactly 0 for 65% to 86% of restaurant answers | `f9-subjective-prompt.md` section 4.4; `f9-h9-review.md` sections 3.1 and 3.5 | Section 4.3, the H9 sentence; Appendix B `app:desc-presc` paragraph before `tab:h9-summary` (the schema note) | applied |
| P67. The in-view uncertainty of the restaurant answers rises with the White share, the bachelor's share and income in 20 of 60 cells, and in 0 street cells | `f9-inference-valence.md` `trace_ses` | Section 4.3, the "Popularity" paragraph; `fig:trace-ses` in Appendix B (the figure was in the paper folder and unused) | applied |

Not changed: the object-sign counts (17 of 22 with Bonferroni in the paper; 18 of 22
with Holm in the claim map). Both are correct for their test, and the paper keeps the
simpler one.

### Moves to the appendix (2026-11-03, after the review)

The user chose 9 of the 10 appendix candidates of the review. Each move keeps the
result sentence in the main text and puts the mechanics and the numbers in the appendix.

| Move | From | To |
|------|------|----|
| A1. The transfer mechanics (36,490 shared words, the test layer, the Gemma `cool` exception, the ceilings) | Section 4.1, "One aesthetic direction" | Appendix A, "Transfer" |
| A2. The borough word lists with the foreign-language words | Section 4.1, "Across boroughs" | Appendix B, `app:borough-words` |
| A3. The per-standard word examples of Figure 2 | Section 4.1, the Figure 2 paragraph | cut; the figure holds them. The counts stay |
| A4. The fixed-effects counts, the pairwise Kendall tau values, and the 591-image readout gradient | Section 4.2, the ACS paragraph | Appendix A `app:fixed-effects` (already there), Appendix B `app:pairwise` (already there) and `app:prompt-ses` |
| A5. The per-borough pass counts | Section 4.2, the borough paragraph | Appendix B, `app:borough-order`. The Manhattan 1 of 6 and 0 of 6 stay |
| A6. The 95 of 100 and 80 of 100 rank counts | Section 4.3, "What the image supplies" | Appendix B, `app:ranks` |
| A7. The sample sizes, the 24-test family and the float32 replication of the borough-name probe | Section 4.3 | Appendix A, "The borough-name probe", new label `app:borough-name` |
| A8. The uncertainty shares by case (75.9%, 7.1% to 79.8%, 2.9%, 0.2%) | Section 4.3, "Popularity" | the tables `tab:uncertainty-by-case` and `tab:uncertainty-reasons` hold them |
| A9. The lens lineage (R-lens, the replication, the theory paper, "since July 2026") | Section 3.2 | Appendix A, new paragraph "The lens family" before "Lenses and readouts" |

| A10. The Qwen dose sentence and the verdict-direction flips of the steering paragraph | Section 4.1, "The direction moves the answer" | Appendix B, `app:steering` |

A10 followed on the user's choice, in the same session.

New assets in `manifest/assets.toml`: `paper/tables/facct/steering-dose.tex`
(`research/scripts/f10/hop_steer_table.py`), `paper/tables/facct/borough-order.tex`
(`research/scripts/f10/hop_more_answers.py --borough`), `paper/figures/facct/trace_ses.pdf`
(`research/notebooks/facct/_trace_ses.py`).

## Related

- [facct-claim-map.md](facct-claim-map.md): the claim rows (C1 to C6) that this page cites
- [f10-30-more-images.md](f10-30-more-images.md): the 10,000-image answers and the 10-image readouts behind section 7
- [f10-judgment-mechanism.md](f10-judgment-mechanism.md): the report with the tables
- [f10-analysis-walkthrough.md](f10-analysis-walkthrough.md): the 12 steps against the 4 claims
- [neuronpedia-viewer.md](neuronpedia-viewer.md): the contrasts of the viewer
