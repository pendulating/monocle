# Population-level methods for stages in residuals and lens reads

Scope: methods from computational neuroscience and interpretability, for the F10.6 store. Local facts come from the repo files named in each line. "Inference" means my own reasoning, not a published result.

## Objective: which methods find stages, and which do not depend on the lens

### Takeaway
Only residual-based methods are free of the lens, and they cannot name the content. The best design crosses residual layer with lens layer: every lens is 1 linear map, so you can compute the words of residual layer a through lens b. No published study that I found does this. Treat it as a new contribution, and test any stage with a swap.

### Cited Findings

Facts on the lens and the store:
- At inference, the J++ lens is "one linear map per layer", and the read is `W_U J_l h` — [J++ repo](https://github.com/safety-research/jpp_lens); `monocle/jlens_read.py` loads 1 Jacobian for each layer.
- The J-lens authors flag the open question. They cannot rule out that "these layer-wise effects are artifacts of the J-lens methodology". The first third of the model gives noisy reads — [Workspace paper](https://transformer-circuits.pub/2026/workspace/index.html).
- The swap succeeds on 54% to 70% of trials, by model. The authors call the lens "an imperfect tool" — [Workspace paper](https://transformer-circuits.pub/2026/workspace/index.html).
- Local warning: J++ and the R-lens share the top-1 token on only 0.0 to 1.6% of image blocks — `vlm-narratives-docs/facct-runbook.md`, F7.
- Local record: a community audit found that banded Gemma models do not resolve concepts through the fitted lens, and that a Qwen model does. I did not read that audit — `vlm-narratives-docs/jacobian-lens-review.md`.
- Store: 197 neighbourhoods with 3 images each give 591 images. Gemma-4-31B layers are 0 to 56 of 59. Qwen3.5-9B layers are 0 to 28 of 31. The word axes hold 53,999 and 43,563 tokens — `data/f10/hop_all/*/manifest.json`, `scripts/f10/hop_depth.py`.
- The margin `yes_no` is the log probability of `Yes` minus that of `No` at the last position — `scripts/f10/hop_depth.py`.
- The decoder is causal, and the image comes first, thus patches cannot see the question — `vlm-narratives-docs/facct-runbook.md`, F7.

### Method comparison

Lens column: **Free** = uses residuals only. **Bound** = the result changes with the lens. **Direct** = measures the lens effect.

| Method | Form on our arrays | Finds | Lens | Fit to our data | Rank |
|---|---|---|---|---|---|
| Crossed lens read | `z(a,b) = unembed(J_b h_a)` for residual layer a and lens layer b | Word change split into model term and lens term | Direct | Exact; all J are on disk | 1 |
| Second-order RSA on residuals | 1 RDM per cell, 174,345 entries; cell-by-cell Spearman matrix (375², 192²) | Blocks of cells with 1 geometry | Free | Very good; no alignment needed | 2 |
| Generalization of a ridge decoder | Train at a cell, test at all cells (375², 192²); dual ridge on 591×591 kernels | Sustained code, new code, recurrence | Free | Very good; residual space is shared in 1 model | 3 |
| Model RDM regression, commonality | Cell RDM ~ scene RDM + lexicon RDM + margin RDM | Which content leads in each cell | Free for the cell; lexicon RDM is Bound if it comes from reads | Good | 4 |
| Variance partition with grouped ridge | margin ~ scene + cell residual, by neighbourhood folds | Unique information of a cell | Free | Good, but the limit is the feature set | 5 |
| Reduced-rank regression | Residual at (t, l) predicts residual at (t', l+4) | Dimensionality of the channel between cells | Free | Good; `n` limits rank | 6 |
| CP / TCA on residuals | [images, tokens, layers, top PCs], image mean of each cell removed | Components with depth and token profiles | Free | Good after compression | 7 |
| PARAFAC2 on reads | Word loadings vary by layer | Hops, with the shift of the word axis modelled | Bound, but it models the shift | Fair; library form is 3-way | 8 |
| dPCA | Bin the margin; d × margin bins × tokens × layers | Margin subspace, apart from token and layer | Free on residuals | Fair; needs bins | 9 |
| Change-point on the RDM series | Kernel cost on 15 or 8 layer steps | Boundaries in depth | Free on residuals | Fair; very few points | 10 |
| Subspace angles, Procrustes | Compare top PCs or shapes of 2 cells | Rotation of the code | Free | Good as a check | 11 |
| Debiased CKA | Debiased HSIC on Gram matrices | The same use as RSA | Free | Use only as a check | 12 |
| HMM, HSMM | Sequence of layers for each token and image | Stage onset that varies by image | Free | Weak; 8 to 15 points | 13 |
| DSA, jPCA | Fit dynamics over a sequence | Dynamics or rotations | Free | Weak; sparse depth samples | 14 |
| CEBRA | Pool (image, cell) samples, labels are the margin and the cell | A nonlinear embedding | Free | Weak; only 591 unique images | 15 |
| RSA or CP on reads alone | Same forms, on reads | Word geometry | Bound | Use only with row 1 | n/a |

### Inferences
- Array sizes, by my arithmetic: tail residuals are 4.8 GB and 1.9 GB in float32. Full tail reads are 48 GB and 20 GB. Patch residuals are 49 GB and 79 GB. Thus tail-residual methods fit in memory, and patch methods need a pooled form.
- Grid logic: the grid is a DAG. Cell (t, l) can receive only from cells with token ≤ t at lower layers. Use this rule to reject impossible flow claims.
- The last cell is trivial. `final` at the deepest layers must decode the margin, by construction. A useful stage is one that appears earlier than this.
- Definition of a stage for the notes: a contiguous block of cells with a shared geometry and a shared decoder, which differs from the next block, in residuals, and which a swap confirms.
- Suggested order:
  1. Run RSA and decoder generalization on residuals, to find boundaries without the lens.
  2. Run the crossed read at the boundaries only, to name the content and to measure the lens term.
  3. Run the 2 analyses on Qwen3.5-9B, and compare by second-order RSA.
  4. Test each stage with a swap, chosen before the swap runs.
- Statistics: resample neighbourhoods, not images. Use Bonferroni over cells, in line with the user's rule for simple tests.

### Gaps
- I found no published study that crosses a residual layer with the lens of another layer. Absence of a result in my searches is not proof that none exists.
- Many entries rest on abstracts or search excerpts. I read these sources directly: dPCA (first 100,000 characters), Nili 2014, Biran 2024, the Workspace paper (first 100,000 characters), the Lindsey post (same limit), TCA (same limit), and the TensorLy, MNE and J++ pages. For the others, verify before you quote a detail.

## Representational similarity analysis

### Takeaway
RSA fits the store well, and it needs no alignment across cells, tokens or models. RSA on residuals against RSA on reads separates lens change from model change only in part; the crossed RDM design closes the gap.

### Cited Findings
- RSA relates multi-channel activity measures to each other and to computational models, and it avoids an explicit spatial correspondence map — [Kriegeskorte 2008](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/text).
- Noise ceiling: it is "the expected RDM correlation achieved by the (unknown) true model, given the noise in the data". The upper bound uses the mean of all subject RDMs. The lower bound uses leave-one-subject-out. It needs several independent RDM estimates — [Nili 2014](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003553).
- I found no confirmation that the 2008 paper introduced the ceiling. A lecture slide says so, with a wrong journal name. Attribute the ceiling to Nili 2014 — [Hebart slides](https://fmrif.nimh.nih.gov/images/2/23/MVPA_8.pdf).
- Nili recommends Kendall's τA when some models predict ties, and Spearman when none do. τA is much slower. Pairwise model tests use a bootstrap of stimuli, with FDR by default. Stimulus-label randomization does not suit tests of a difference between models — [Nili 2014](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003553).
- rsatoolbox gives `boot_noise_ceiling` and `cv_noise_ceiling` — [rsatoolbox docs](https://rsatoolbox.readthedocs.io/en/latest/rsatoolbox.inference.noise_ceiling.html). Its aims include evaluation of neural network models — [eLife preprint](https://elifesciences.org/reviewed-preprints/107828).
- 2-factor cross-validation and bootstrap generalize to new subjects and new conditions, and the authors validate them on simulated deep-network data — [Schütt 2021](https://arxiv.org/abs/2112.09200).
- Crossvalidated distances give unbiased estimates on a ratio scale, and continuous measures are more reliable than classifier accuracy — [Walther 2016](https://ir.lib.uwo.ca/brainpub/913). Noise normalization did not consistently improve reliability in a later study — [Ritchie](https://iris.unitn.it/retrieve/e3835199-7164-72ef-e053-3705fe0ad821/Ritchie_2022_NeuroImage.pdf).
- PCM, encoding analysis and RSA share 1 framework. PCM gives the most powerful test, and RSA gives a useful summary with weaker assumptions — [Diedrichsen and Kriegeskorte 2017](https://kriegeskortelab.zuckermaninstitute.columbia.edu/sites/default/files/content/DiedrichsenKriegeskorte_2017_PLoSComputBiol.pdf).
- Fusion precedent: dissimilarity comparison across MEG and fMRI separated low-level V1 signals from later IT signals — [Cichy 2014](https://www.ewi-psy.fu-berlin.de/psychologie/arbeitsbereiche/neural_dyn_of_vis_cog/data_on_cms_server/publication_pdfs/Cichy_et_al_NN_2014.pdf).
- Use on language models: a 2026 preprint computes, layer by layer, the Pearson correlation between hidden-state and reference similarity matrices — [arXiv 2602.00628](https://arxiv.org/pdf/2602.00628). Abnar 2019 uses a stability variant of RSA, to see where deeper processing occurs — [Abnar 2019](https://preview.aclanthology.org/fix_video/W19-4820.pdf).
- A web project page reports early layers with high similarity, varied middle layers, and late convergence. It is not peer reviewed — [sidn.baulab.info](https://sidn.baulab.info/evolution).

### Inferences
- Concrete form: at each cell, take `H` [591, d], remove the cell mean, compute `D`, keep the 174,345 upper entries. Correlate `D` across cells, with Spearman. The result is a [375, 375] or [192, 192] matrix. Block structure along layers gives stages.
- The cell mean removal also removes token identity, thus the matrix shows how image variation changes, not which token sits there.
- Reads and residuals have a simple link. If the unembed is linear, then `d_read²(i,j) = (h_i-h_j)ᵀ M_l (h_i-h_j)`, with `M_l = J_lᵀ Wᵀ W J_l`. So a read RDM is a residual RDM under a lens metric. Only `M_l` separates the 2 RDMs. The final norm and softcap add a small nonlinearity.
- Reading the 2 RDM series:

| Residual RDM | Read RDM | Reading |
|---|---|---|
| changes | changes | Model change, with a possible lens part |
| stable | changes | Lens change, or a low-variance model change that the lens picks up |
| changes | stable | Model change in dimensions that the lens does not see |
| stable | stable | No change at this scale |

- Row 2 is the weak point. Euclidean RSA follows high-variance directions, thus a small verbal signal can pass unseen. RSA alone cannot settle row 2.
- The crossed RDM settles it. Compute a read RDM for each pair (residual layer a, lens layer b). This gives 225 RDMs for each Gemma token. Regress second-order similarity on "same a" and "same b". The 2 coefficients give the share of model and of lens.
- The exact word split for a step from l1 to l2, with `z(a,b)` as above:
  - Model term: `½[z(l2,l1) − z(l1,l1) + z(l2,l2) − z(l1,l2)]`
  - Lens term: `½[z(l1,l2) − z(l1,l1) + z(l2,l2) − z(l2,l1)]`
  - The 2 terms sum to the observed change for any unembed, because only 4 reads add and subtract.
- Warning: `J_b` on a residual from layer a is off the fit domain of the lens. Treat `z(a,b)` as a descriptive contrast. Biran-style back-patching gives a causal check on it: patch `h_a` at layer b and read the output.
- Noise ceiling: the forward pass is deterministic, so the classical ceiling has no noise to measure. Do not report one from these data alone. A real ceiling needs repeat measures, such as paraphrases of the prompt or small crops of each image. With 2 models, the subject factor of Schütt 2021 has too few levels.
- Crossnobis needs independent partitions. The 3 images of a neighbourhood could act as trials, for a 197×197 RDM. But variation among those images is signal, not noise, and the question changes. Use the plain distance first.
- Model RDMs:
  - Scene: CLIP embedding (external), and low-level image statistics.
  - Internal scene: pooled patch residuals. These are free of the question, because patches come before it.
  - Criteria: mass on a fixed lexicon. This one is Bound to the lens.
  - Answer: `|margin_i − margin_j|`.
- Continuous RDMs have no ties, thus use Spearman. τA is too slow for 174,345 entries and 70,125 cell pairs.
- Only 591 of 174,345 pairs (0.34%) share a neighbourhood. Keep or drop them; the choice matters little. The bootstrap unit still has to be the neighbourhood.

### Gaps
- I found no published use of a noise ceiling for a deterministic network, and no RSA over a token-by-layer grid.
- The rsatoolbox list of RDM comparators and its handling of bootstrap duplicates need a check in the docs.

## Temporal generalization

### Takeaway
Train at one cell and test at all others. This is the most direct test of a stage on residuals, because 1 model's residual stream keeps the same coordinates. Run it first along depth, for each token.

### Cited Findings
- Reference: King and Dehaene, Trends Cogn Sci 18(4):203-210, 2014. A serial chain of activations gives a diagonal generalization matrix. Sustained activity gives a square matrix — [King and Dehaene](https://pmc.ncbi.nlm.nih.gov/articles/PMC5635958).
- `GeneralizingEstimator` tests each model at all time points. The last data dimension is the axis of generalization. The diagonal equals plain decoding over time. Off-diagonal values can show persistence or recurrence — [MNE tutorial](https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html).

### Inferences
- Full form: a train cell (t, l) and a test cell (t', l') give a 4-way score array with 140,625 values for Gemma and 36,864 for Qwen.
- Practical slices:
  - Depth by depth at 1 token: [15, 15] or [8, 8]. This is the direct analog of the time-by-time matrix.
  - Token by token at 1 layer: [25, 25] or [24, 24].
  - Full map: cluster the [375, 375] matrix, then read the blocks.
- Dual ridge: with `n << d`, use `ŷ' = H' Hᵀ (H Hᵀ + λI)⁻¹ y`. Precompute the cross-cell kernels once. Then each fold only selects rows.
- Scale and mean shift across cells. Centre the test cell with training-fold images. Score with Pearson or Spearman over test images, not R². Otherwise a norm change across layers looks like a loss of the code.
- Reading the map for a stage:

| Pattern at 1 token | Reading |
|---|---|
| Square block over layers | The same code persists |
| Narrow diagonal | The code rotates in depth |
| Two blocks with a low cross score | 2 stages, if the diagonal stays high |
| Late score that needs early weights | Recurrence of a code |

- The lens gives a second, exact version. A lens row is a linear decoder for 1 word. So a crossed read is the generalization matrix of the lens. A lens that generalizes well across layers, with a stable word, shows a stable code.
- The target can change. The margin is 1 target. Scene PCs and lexicon scores are others. A decoder of scene PCs and a decoder of the margin may have different onsets, and that gap is a candidate for a hop.
- If each block adds to a shared residual stream, a low cross score is real rotation or a new code, not a change to a separate layer space. This is an assumption. A check is cheap: compare identity distance with Procrustes distance for the same cell pair.

### Gaps
- I could not read the full King and Dehaene text. PMC blocked the fetch. Other patterns (reactivated, below-chance) come from my memory. Verify them before use.
- I found no decoder generalization study on a transformer token-by-layer grid.

## Cross-validated decoding, variance partition, encoding models

### Takeaway
Grouped ridge from residual to margin gives the curve of value information over the grid. Variance partition then asks what each cell adds over the scene. All cells are functions of the image, so "unique" means unique for the chosen features only.

### Cited Findings
- Small samples give large error bars: about ±10% at 100 samples, and the standard error across folds strongly underestimates them — [Varoquaux 2018](https://arxiv.org/pdf/1706.07581).
- Leave-one-out gives unstable and biased estimates. Repeated random splits are better — [Varoquaux 2017](https://arxiv.org/pdf/1606.05201).
- Decoder weights are hard to read. Haufe et al. convert a backward model to a forward model. Large weights do not imply a strong link with the condition — [Haufe 2014](https://pubmed.ncbi.nlm.nih.gov/24239590).
- Banded ridge gives each feature space its own penalty. Variance partition is usually limited to 2 or 3 spaces, because it yields 2^m − 1 values. Banded ridge scales further and discards redundant spaces — [Dupré la Tour 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9807218/). The package is Himalaya — [bioRxiv](https://www.biorxiv.org/node/2743674).
- A 2025 abstract reports that variance partition was more robust than 2 residual methods — [Schultheiss 2025](https://2025.ccneuro.org/abstract_pdf/Schultheiss_2025_Comparing_Variance_Partitioning_Residual_Method_Interpreting.pdf).
- Commonality analysis estimates the part of a dissimilarity structure tied to 1 variable and not to another, at each time point. It showed a dynamic sequence of task and object processes — [Hebart 2018](https://elifesciences.org/articles/32816).

### Inferences
- Form: `GroupKFold` with neighbourhood groups; dual ridge at each cell; out-of-fold prediction; 1 score per cell. Repeat with reshuffled group folds.
- Precision: `n = 591` with 197 clusters of 3. Expect wide intervals at the cell level. Bootstrap neighbourhoods for the interval, and apply Bonferroni over 375 or 192 cells.
- Null for the scan: permute the margin among neighbourhoods as blocks.
- Three blocks for the partition:
  - Scene: CLIP PCs and low-level statistics.
  - Lexicon: lens-free if you use a text-based score; Bound if you use reads.
  - Cell: residual at the cell, reduced to PCs inside each fold.
- Use the result this way: a cell with no unique margin information over the scene block carries no value signal beyond a linear scene model. A nonlinear scene model could absorb more, so state the model class.
- Read the decoder with the Haufe pattern, `cov(h, ŷ)`, not the weights. You can then read that pattern through the lens. The decoder stays Free and only the label is Bound.
- The commonality form on RDMs is the simplest stage plot: for each cell, plot the unique and shared parts of scene, lexicon and margin RDMs. A shift from scene to margin across cells is the expected picture.

### Gaps
- I found no study that uses this partition on LLM activations with an image scene model.

## Dimensionality reduction across the grid

### Takeaway
CP/TCA is the natural fit to a 4-way array. dPCA fits only if you bin the margin. jPCA does not fit. CEBRA adds a nonlinear picture but little proof. On reads, plain CP assumes a fixed word axis over layers, and that is the lens problem in a new form.

### Cited Findings
- TCA reduces `NTK` data points to `R(N+T+K)` values, 1 factor for each mode. It "often has a unique solution without further assumptions", but PCA factors can rotate with no change in error — [Williams 2018](https://www.biorxiv.org/content/10.1101/211128v1.full).
- Rank selection uses an error plot and a similarity plot over random starts. The tensor needs a fixed `T` for each trial. Non-negative TCA can give clearer models, and the authors used it because their traces were non-negative — [Williams 2018](https://www.biorxiv.org/content/10.1101/211128v1.full).
- PCA and ICA failed to recover the signal in the simulated network test — [tensor review](https://arxiv.org/pdf/2402.16638).
- `parafac` accepts `mask`, `init`, `l2_reg`, `normalize_factors` — [TensorLy parafac](https://tensorly.org/stable/modules/generated/tensorly.decomposition.parafac.html).
- `parafac2` fits third-order data as `X_i ≈ B_i diag(a_i) Cᵀ`, with `B_iᵀ B_i` constant over slices — [TensorLy parafac2](https://tensorly.org/stable/modules/generated/tensorly.decomposition.parafac2.html).
- dPCA minimizes reconstruction of each marginalization, with a ridge penalty that cross-validation selects. It "works only with discrete parameters", all parameter combinations must be present, and a dataset can be non-demixable — [Kobak 2016](https://elifesciences.org/articles/10989).
- jPCA projects population activity into planes where rotations appear — [Churchland 2012](https://web.stanford.edu/~shenoy/GroupPublications/ChurchlandEtAlNature2012.pdf).
- Tensor maximum entropy surrogates test if a structure is an expected byproduct of simpler features. Motor cortex dynamics passed this test — [Elsayed 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5577566).
- CEBRA has a supervised mode and a self-supervised mode, and it combines nonlinear ICA ideas with contrastive learning — [Schneider 2023](https://www.nature.com/articles/s41586-023-06031-6).

### Inferences

| Method | 4-way? | Use for us |
|---|---|---|
| PCA | No, unfold | Compress `d` to top PCs, a step for other methods |
| CP / TCA | Yes | Residual form: [591, tokens, layers, PCs], cell mean removed |
| Tucker | Yes | Compression before CP; factors are not unique |
| PARAFAC2 | 3-way in TensorLy | Fix 1 token, then slices = layers, varying mode = words |
| dPCA | With bins | Bin margin in 5: 5 × 25 × 15 = 1,875 condition means of `d` |
| jPCA | No | Needs a smooth time axis; 8 to 15 sparse depth points do not give one |
| CEBRA | Flatten | Labels are margin, token, layer; a picture, not a test |

- CP sizes: reads with all words are too large. Restrict to about 2,000 high-variance words (about 0.9 GB for Gemma), or use Tucker first.
- A CP component reads as a hop: a word or PC loading, a depth profile, a token profile and an image score. Correlate the image score with the margin and with scene PCs.
- Remove the cell mean first. Otherwise token identity and layer norm take the first components. Decide on the scale step in advance, because unit scale for each cell hides amplitude change.
- Non-negative CP does not fit residuals, because they have a sign. Logits also have no zero.
- CP on reads assumes that 1 word loading holds at all layers. A lens shift then splits 1 concept into several components, or merges them. PARAFAC2 relaxes this. Its constant cross-product rule means that word relations stay fixed while the axis turns. That is a partial model of a lens rotation, not a proof of it.
- Run CP on residuals and on reads. Components in both are safer. Components in reads only need the crossed test.
- Elsayed-style surrogates fit CP directly: build tensors that keep the covariance of each mode, and ask if the stage structure exceeds them.

### Gaps
- Published tests of the Lebedev critique of jPCA did not appear in my search. I do not cite it.
- I did not check TensorLy for memory limits on a tensor of this size, or for a `non_negative_parafac` mask form.

## Trajectories, state sequences, subspaces, shape

### Takeaway
Depth is sampled at each 4th layer, so sequence models have 8 to 15 points. Use change-point on second-order series with care. Use shape and subspace measures as checks of rotation. Use debiased CKA only, because `d > n` here.

### Cited Findings
- HSMM-MVPA pairs hidden semi-Markov models with multivariate pattern analysis, and each state is a processing stage — [Anderson 2016](https://www.psychologie.uzh.ch/dam/jcr:3a92cb65-16c0-4a4c-8e43-878baa0ce7ac/Anderson.PsychRev.123.2016.pdf). Borst and Anderson extend it to EEG — [Groningen record](https://research.rug.nl/en/publications/discovering-processing-stages-by-combining-eeg-with-hidden-markov/).
- HMMs learn time scales from data, with no preset window. This comes from a search excerpt — [Quinn 2018](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00603/pdf).
- `ruptures` gives offline change-point detection, with PELT and kernel costs — [Truong, ruptures](https://arxiv.org/pdf/1801.00826). A detection method has 3 parts: cost function, search method, and a constraint on the number of changes — [Truong 2020](https://arxiv.org/pdf/1801.00718).
- "neither CCA nor any other statistic that is invariant to invertible linear transformation" can measure meaningful similarity when the dimension exceeds the number of data points — [Kornblith 2019](https://arxiv.org/abs/1905.00414).
- CKA is sensitive to simple transformations and outliers, and its value can change while model behaviour stays the same — [Davari 2022](https://arxiv.org/abs/2210.16156). This is a preprint on the page I read.
- Biased CKA gives high scores for random matrices when data are few and features are many. "Positive alignment of true stimuli-driven responses is only achieved by using debiased CKA" — [Murphy 2024](https://arxiv.org/abs/2405.01012).
- Shape metrics give true distances on representations, with Procrustes as a base, and they satisfy the triangle inequality. They embed well in Euclidean space — [Williams 2021](https://arxiv.org/abs/2110.14739v2). Procrustes distance is equivalent to a Bures distance — [Harvey 2024](https://2024.ccneuro.org/pdf/545_Paper_authored_2024_duality_CCN.pdf).
- Communication subspace: V2 tracks a small set of V1 patterns, apart from the largest V1 fluctuations — [Semedo 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6449210). Reduced rank regression is the main estimator, and it takes an L2 penalty in closed form — [RRR tutorial](https://arxiv.org/pdf/2512.12467).
- DSA compares dynamics and not geometry, with a modified Procrustes on fitted linear dynamics. It can find similar dynamics under different geometry — [Ostrow 2023](https://arxiv.org/abs/2306.10168v3).

### Inferences
- `d/n` is 9.1 for Gemma and 6.9 for Qwen, thus Kornblith's limit applies. Do not use CCA on raw residuals. CCA is possible only on few PCs, with cross-validation.
- Biased CKA is unsafe here for the same reason. Debiased CKA with a linear kernel is close to second-order RSA on Gram matrices, so it adds a check, not a new result.
- Outliers: the Davari finding means that a few odd street images can move CKA. A leave-one-neighbourhood-out jackknife tests this.
- HMM: with cell means left in, the states only copy the layer index. With cell means removed, the sequence has image signal, but only 8 to 15 steps. Its 1 advantage is a stage onset that varies by image. A simple alternative: for each image, find the first layer where a held-out decoder passes a level.
- Change-point: feed the row of the second-order matrix for each layer, or the RDM series, to a kernel cost. With 15 points, and 8 on Qwen, the penalty choice matters. Report boundary stability under the neighbourhood bootstrap. A segment fit on the second-order matrix is a simple equivalent.
- Lens-free rotation curve: top-k PCs of the image-varying residual at each cell, and principal angles between adjacent layers.
- Lens-only rotation curve: the same angles between row spaces of `W J_l`. If the read shifts where this curve jumps, and the residual curve does not, suspect the lens.
- Coverage: the share of image-varying residual variance inside the top singular subspace of `J_l`. A fall with depth means that the lens loses sight of the signal.
- Reduced-rank flow: predict (t', l+4) from (t, l). Compare a same-token parent with an earlier-token parent, each with the other as control. This gives a map of which cell feeds which, in the DAG. It is a statistical flow, not a causal one, and `n` limits rank.
- Flow from the scene: pooled patch residuals at layer l predict tail cells at layer l+4.
- DSA needs a longer series than we hold. Use it only if a run saves all layers.

### Gaps
- I did not find a source for subspace angles in this context. The method is standard linear algebra. I also did not verify the Davari peer-review status.

## Mediation and causal tests

### Takeaway
Observational mediation fails here for the same reason as in neuroscience: other parts of the state confound a chosen mediator. But a network has an escape that a brain lacks. You can compute cross-world counterfactuals by patch. So use mediation only to select candidates, then measure direct and indirect effects by swap.

### Cited Findings
- Mediation needs a clear definition, a stated identification assumption, and a sensitivity analysis. The core risk is an unobserved confounder of mediator and outcome. Under sequential ignorability the effects are identified — [Imai 2010](https://imai.sites.fas.harvard.edu/research/BaronKenny.html).
- No unmeasured mediator-outcome confounding is "critical and hardly testable" — [Estimating Causal Mediation Effects under Correlated Errors](https://arxiv.org/pdf/1410.7217). Task-unrelated activity is a likely confounder there — same source.
- With sparse neural recordings, confounding makes mediation analysis impossible — [arXiv 1905.02024](https://arxiv.org/pdf/1905.02024).
- Standard remedies do not adapt to spatially varying confounding — [BASMU](https://arxiv.org/pdf/2407.04142).
- Causal abstraction treats a network and an explanation as causal models. Interchange interventions do the same intervention on both, then compare outputs — [Geiger](https://arxiv.org/pdf/2301.04709); [comparison post](https://alignmentforum.org/posts/uLMWMeBG3ruoBRhMW).
- Causal scrubbing swaps activations that the hypothesis calls irrelevant, under the same high-level values — [comparison post](https://alignmentforum.org/posts/uLMWMeBG3ruoBRhMW).
- A strong two-hop swap: Texas features replaced by California features changed "Austin" to "Sacramento", and other swaps gave Atlanta, Victoria, Beijing and Constantinople — [Lindsey 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).
- The same authors got "satisfying insight" on "about a quarter of the prompts", and their examples are existence proofs and a biased sample — [Lindsey 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).
- Spider to ant changes the output from 8 to 6. In an arithmetic prompt, 21, 42 and 49 appear in order, and activation patching confirms the same depths — [Workspace paper](https://transformer-circuits.pub/2026/workspace/index.html).
- Warning from a known system: lesions and correlation analyses found structure in a microprocessor but no hierarchy of information processing, for example a "Donkey Kong transistor" — [Jonas and Kording 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5230747/).
- Local record: the removal test holds on Qwen3.5-9B and fails on Gemma-4-31B, and 0 of 90 gated area tests pass. The swap was validated on 2-hop facts, not evaluative questions — `vlm-narratives-docs/facct-runbook.md`, F10.

### Inferences
- Observational form: `X` = scene score, `M` = read or residual projection at an early cell, `Y` = margin. Other cells at the same depth share causes with `M`. Sequential ignorability is not plausible for a single projection.
- The "sparse record" objection is partly absent here. A whole layer, at all positions, screens off earlier layers from later ones. But we hold each 4th layer, and tail positions plus patches. Single projections do not screen off anything.
- Cross-world effects are computable. For images A and B at a chosen stage subspace `S`:
  - Indirect effect: `Y(A, M=M_B) − Y(A, M=M_A)`.
  - Direct effect: `Y(B, M=M_A) − Y(A, M=M_A)`.
  - In a deterministic network, these need no ignorability. They need only a well-defined patch.
- This gives a clean link from correlation to cause:
  1. Name a stage from residual analyses, with the block of cells and a subspace, before any swap.
  2. Select image pairs with a large gap in the stage score and similar scenes.
  3. Swap the subspace at the stage. Measure the change in margin, with a dose series.
  4. Test necessity by ablation of the subspace, and test sufficiency by a scrub of the other dimensions.
  5. Add a placebo: swap a random subspace of the same variance, and swap at a cell after the stage.
  6. Report the share of the margin gap that the swap moves.
- A stage subspace can come from the decoder's Haufe pattern or a CP loading, so the intervention itself does not need the lens. This matters, because activation addition on the lens direction gave a null result on Gemma-4-12B (`vlm-narratives-docs/jacobian-lens-review.md`, F1).
- Off-manifold risk: a patched vector can be unlike any real state. Swaps with real vectors from another image are safer than synthetic directions.
- Evaluative judgments may not have a single bridge word. A subspace swap may work where a word swap fails. Failed word swaps thus do not disprove a stage.

### Gaps
- I found no study that measures natural direct and indirect effects by patch in an LLM. The idea here is my inference.
- I did not read the original causal scrubbing post. The comparison post is my source.

## Comparison across 2 models

### Takeaway
Second-order RSA across models needs no alignment and uses the 591 shared images. Tokens align almost 1 to 1. Normalized depth is only a first guess; a monotone path through the cross-model RSA matrix estimates the true depth map.

### Cited Findings
- Hyperalignment maps responses into a common space of shared tuning functions — [Haxby 2011](https://collaborate.princeton.edu/en/publications/a-common-high-dimensional-model-of-the-representational-space-in-/); [Haxby 2020](https://elifesciences.org/articles/56601.pdf).
- SRM handles different functional topographies and improves detection of a shared response — [Chen 2015](https://papers.nips.cc/paper/2015/hash/b3967a0e938dc2a6340e258630febd5a-Abstract.html).
- Depth precedent: mean first layer of bridge decoding differs by model, for example 8.9 of 32 in LLaMA 2 7B, 27.1 of 80 in LLaMA 3 70B, and 5.4 of 32 in Pythia 6.9B — [Biran 2024](https://arxiv.org/html/2406.12775v2).
- Local warning: in a comparison of 3 models, layers paired by relative depth "cannot separate the lens from the model" — `vlm-narratives-docs/facct-runbook.md`, F7.
- Patch reads name the scene best at L45 of 59 for Gemma-4-31B and at L20 to L24 of 31 for Qwen3.5-9B — `vlm-narratives-docs/facct-runbook.md`, F7.

### Inferences
- Token map, from the manifests: Gemma has 2 tokens after the image, Qwen has 1. Both have the same 14 question tokens, and 9 template tokens after them. Only the Gemma `\n\n` lacks a partner. Thus the token axis aligns with no method.
- Depth map: normalized depth is `l/59` and `l/31`. The sampled range is 0 to 0.95, and 0 to 0.90.
- The local scene peak is at 0.76 on Gemma and 0.65 to 0.77 on Qwen. That is a mild support for normalized depth, on 8 images and with the lens in the loop.
- Better: build the [375, 192] matrix of Spearman values between Gemma and Qwen RDMs. Find the best monotone path through it. This is a dynamic time warp on second-order similarity. The path is the depth map, and it needs no lens and no alignment.
- Stage test across models: fix stage boundaries on Gemma, and ask if the same RDM blocks appear on Qwen. Bootstrap neighbourhoods.
- The bound: the 2 margins correlate at some value over images. Cross-model stage similarity near the answer cannot exceed what that value allows. Report it next to any cross-model claim.
- SRM or Procrustes across models: `d` exceeds `n`, thus a fit can be perfect on the training images (Kornblith limit). Project to `k << 591` PCs, fit on training neighbourhoods, and test on the rest. With 2 models, SRM reduces to something near a 2-view alignment, so its gain over Procrustes in PC space is small.
- Decoder across models: train a ridge on Gemma, apply through the aligned PCs on Qwen. This is generalization across models, and it uses the same logic as the grid.
- Reads across models: the word axes differ. Intersect the 2 lower-case word lists. The result is Bound to 2 different lenses, and a Gemma lens issue is on record. Do not rest a cross-model claim on it.

### Gaps
- I found no published use of DTW on cross-model RSA for layer correspondence.
- A search excerpt says that hyperalignment was applied to hidden layers of neural networks. I could not find the original paper, so I do not cite it.
- I did not compute the margin correlation between the 2 models. The store holds it.

## Precedent in interpretability and image-computable aesthetics

### Takeaway
The stage evidence in LLMs comes mostly from factual multi-hop prompts, and it is now in dispute. Aesthetic science offers a clear template (features, then integration, then value) and it suggests that a large part of a beauty judgment is perceptual. I found no stage study of an aesthetic judgment in an LLM.

### Cited Findings

Language models:
- Biran et al. use Patchscopes on the hidden state at the last token of the first hop. They prefer it over vocabulary projection. The bridge entity resolves early and the second hop later, and in a figure this happens "almost exclusively during the early layers" — [Biran 2024](https://arxiv.org/html/2406.12775v2).
- Bridge decoding rates are 41% to 78%. Back-patching corrects about 32% to 66% of incorrect cases, but a success means that any source-target pair works — [Biran 2024](https://arxiv.org/html/2406.12775v2).
- Yang et al. test the 2 hops by entity recall and by the effect of a boost. Up to 23% of fact composition types show strong evidence — [Yang 2024](https://arxiv.org/html/2402.16837v2).
- Shortcut-free follow-up over 41 LLMs: best latent composability is 80% for countries as a bridge and 5% for years — [Yang follow-up](https://arxiv.org/html/2411.16679v2).
- Logit flow finds 4 stages in single-hop prediction, and back attention lets lower layers read higher layers — [Yu 2025](https://arxiv.org/abs/2502.10835). This may be the "cross-layer attention" work in the brief.
- Direct conflict: answer entities for later hops can decode earlier than bridge entities, and this "layer-order inversion" grows with hop count. The authors propose shallow MLP recall followed by selective extraction in deeper attention layers — [Liu 2026](https://arxiv.org/abs/2601.03542). This contradicts the hop-aligned reading of [Biran 2024](https://arxiv.org/html/2406.12775v2).
- Early, middle and late regions appear on a 0-100 layer scale, with a workspace at about 38 to 92 — [Workspace paper](https://transformer-circuits.pub/2026/workspace/index.html).

Aesthetics:
- Preference for art is partly a systematic integration over visual features, and the features emerge hierarchically in a network trained for object recognition — [Iigaya 2021](https://ideas.repec.org/a/nat/nathum/v5y2021i6d10.1038_s41562-021-01124-6.html).
- The brain shows a graded hierarchy of low-level and high-level features in the visual system, and integration in parietal and prefrontal cortex — [Iigaya 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9873760).
- Models with no affect training explain most of the explainable variance in human ratings of arousal, valence and beauty — [Conwell 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11789064/).
- Sensory regions rise linearly with rating, but a frontal network shows a step-like rise only for the most moving works, and this includes default mode regions — [Vessel 2012](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2012.00066/pdf).
- Agreement is higher for natural landscapes than for exterior architecture, while architecture and art show strong individual differences — [Vessel 2018](https://www.aesthetics.mpg.de/en/newsroom/news/news-article/article/beautiful-nature-controversial-art.html).
- Memorability behaves as an intrinsic, image-computable property, with an observer-independent score — [Isola 2011](https://vision.princeton.edu/projects/2011/memorability/index.html).
- Place Pulse 2.0 has about 1.17 million pairwise comparisons over about 111,000 images, on 6 attributes that include `beautiful` and `lively`. The exact image count differs among sources — [Dubey 2016](https://arxiv.org/pdf/1608.01769).

### Inferences
- Hypothesis to test, in order of the Iigaya template:
  1. Scene stage: tail geometry matches pooled patches and CLIP.
  2. Integration stage: geometry departs from the scene and moves to the margin, with unique margin information over the scene block.
  3. Answer stage: `yes or no` and template tokens at depth carry a low-dimensional code that generalizes square.
- Competing hypothesis, from Conwell: no integration stage exists. A linear read of scene features gives the margin. Then the scene block leaves no unique margin information at any cell before the answer. This is a clear and useful null result.
- Liu 2026 warns against a fixed order: the answer may decode before a "criteria" word. Do not assume scene → criteria → answer. Let the grid show the order, and read onset with error bars.
- Vessel 2012 suggests a shape test. For each cell, compare a linear fit of the decoder score against a step fit on the margin. A move from linear to step in depth would be a neat analog. It is a loose analogy only.
- Place Pulse offers a ready external scene model: a `beautiful` or `lively` score from a model trained on human votes, as a model RDM. Then the margin can compare with human judgment of the same kind of image. `cool` is not in the 6 attributes.
- Lens check for all of this: Biran avoided vocabulary projection, the Workspace authors flag artifacts, and the lens is linear by layer. Thus a word sequence alone does not count as a hop in this project.

### Gaps
- I found no LLM study of stages of an aesthetic or evaluative judgment. The local swap record notes that the source validated swaps on facts only.
- I found no primary source for a peer-reviewed "stage" framing of LLM layers by RSA. The only explicit page is a web project.
- Neural work on aesthetics has its own limits. fMRI timing does not establish a stage order. Commentary on Iigaya that says higher layers predict better came from a search excerpt.
