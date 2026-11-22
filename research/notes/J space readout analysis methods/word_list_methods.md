# Text-analysis methods for weighted word lists, applied to lens readouts

Marking rule for these notes:

- "Cited Findings" holds claims from a source, with the URL.
- "Inferences" holds my own reasoning. It has no source. Test it before you use it.
- "Gaps" holds items that I could not verify.
- "Full text" means that I read the method text myself. "Abstract or excerpt" means that I read only a search excerpt or a tool summary.

## For each method: required input, validity on logits, what it answers, main failure mode

### Takeaway
Logits are log-scale scores that have no sample size. Thus no count-based variance is valid on them. A method is valid when it uses logit differences as an effect and takes its variance from the variation between neighbourhoods. The default test is a Welch t-test or OLS on neighbourhood means. A small number of pre-set word sets is a better primary test than 54,000 word-level tests.

### Cited Findings
- Monroe et al. define `b_w = log(p_w) - log(p_1)` and `p_w = exp(b_w) / sum_j exp(b_j)`. This is a softmax over log-scale scores, with one reference word. (Full text, eq. 6-7.) — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- In Fightin' Words, variance depends on "the absolute frequency of a word". The approximation is `sigma^2 = 1/(y_i + a_i) + 1/(y_j + a_j)`. (Full text, eq. 17-20.) — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- The project lens is `logits_l[answer] = unembed(J_l @ h_l[answer])`. — [answer-token-readout.md](/share/ju/matt/mllmsci/research/docs/answer-token-readout.md)
- In the project code, unembed is final norm + `lm_head` + softcap. `J = I` gives the plain logit lens. — [jacobian-lens-review.md](/share/ju/matt/mllmsci/research/docs/jacobian-lens-review.md)
- Log-ratios are the standard way to get subcompositional coherence. This principle says that an analysis of a subset of parts must agree with the analysis of the full composition. (Abstract or excerpt.) — [UPF working paper 1106](https://econ-papers.upf.edu/papers/1106.pdf); a later paper accepts near coherence — [Aitchison's CoDA 40 Years On](https://ar5iv.labs.arxiv.org/html/2201.05197)
- A 2026 preprint treats softmax attention rows as compositional data with centred log-ratio coordinates. It is one preprint. (Abstract or excerpt.) — [arXiv 2608.14712](https://arxiv.org/pdf/2608.14712)

### Comparison table

Column key: "Valid on logits" means that the inference holds on real-valued scores. The table text is my synthesis. The per-method sections below give the sources.

| Method | Required input | Valid on logits? | Form on our data | What it answers | Main failure mode |
|---|---|---|---|---|---|
| Fightin' Words z | Counts | No, not the z. The point estimate has a direct analogue | Top-k membership counts, or Welch t on centred logits | Which words separate the groups | Fake sample size |
| TF-IDF | Counts | No | Do not use. Double centring does the same job | Which words are distinctive | Two-group `idf` breaks down |
| Dunning G2 | Counts | No | Chi-square on images with word in top-k, one vote per unit | Which words differ in list membership | Overdispersion, if pooled |
| Scaled F-score | Non-negative counts | No | Top-k counts only. Keep the plot idea | Which words are frequent and precise | Negative inputs give nonsense |
| LDA, STM | Counts | No | Do not use as the primary method | Latent themes | Reads noise as themes |
| BERTopic, Top2Vec | Documents as strings | Not as given | Cluster the fixed vocabulary, once | Word groups | Unstable clusters |
| Word clusters | Word vectors only | Yes. Clusters are fixed sets | Fixed sets, then set scores | Meaning groups from outside the data | Labels by eye |
| Lexicon score | Word ratings | Yes | Regress or correlate centred logits on the ratings, per image | Is a read concrete, valent | Coverage and sense errors |
| Semantic axes | Word vectors, poles | Yes | Project and correlate with logits | Position on a set contrast | Pole choice |
| Weighted centroid | Weights >= 0, vectors | Yes, after you pick a weight | Softmax or top-k mean vector | Where a read sits in meaning space | Temperature decides the result |
| WMD | Distributions | Partly | Top-k only | Distance between reads | Adds little over L1 |
| Soft cosine | Vectors, term similarity | Yes | Two reads, sparse term matrix | Similarity with synonyms | Similarity matrix choice |
| Ridge, PLS | Real-valued matrix | Yes | Grouped cross-fit to `yes_no` | Is the margin in the words | Leakage of the answer |
| Lasso | Real-valued matrix | Yes | Same, but as a second choice | A short word list | Unstable selection |
| MNIR | Counts | No | Use the Gaussian analogue | Words that move with the margin | Wrong likelihood |
| Procrustes | Separate spaces | Not needed | Words already share one axis | Rotation between spaces | Solves a problem that we do not have |
| RBO | Ranked lists | Yes | Adjacent layers | How much the top list changes | Choice of `p` |

### Inferences
- **Algebra (my derivation).** Let `l_w` be a logit and `Z = sum_j exp(l_j)`. Then `log p_w = l_w - log Z`. For a rare word, `log(p_w / (1 - p_w))` is close to `log p_w`. So the Monroe log-odds-ratio for a rare word is `(l_w^A - log Z^A) - (l_w^B - log Z^B)`. Thus a difference of logits is the Monroe point estimate, apart from a shift in `log Z`. Only the variance has no analogue.
- **Log-ratios survive the vocabulary restriction.** `l_a - l_b` does not depend on the other words. So a contrast between two word sets stays valid on 40,000 to 54,000 words. Softmax probabilities do not stay valid, because the mass of the omitted tokens is unknown. The readout does not hold all of the vocabulary. So you cannot recover `log Z` exactly, unless the store holds it.
- **Softcap.** The softcap and the final norm are non-linear. So a logit difference is not exactly a difference in the linear read. I assume that the softcap compresses high logits. Check the size of this effect at the top of the list.
- **Double centring is the required first step.** Subtract the per-image mean over the fixed vocabulary. This removes the free constant. Then subtract the per-word mean over images. This removes the "this word is always high" effect, and it does the job of `idf`. Use one fixed vocabulary for all comparisons. If 40,000 and 54,000 are two vocabularies, use their intersection.
- **Scale.** Logit spread can change with the layer. So do not compare raw logit sizes across layers. Divide by the per-read standard deviation, or use ranks in the read.
- **Unit of analysis.** There are 591 / 197 = 3 images per neighbourhood. Images in one neighbourhood are not independent. The neighbourhood is the safe unit.
  - Borough is a neighbourhood attribute. So the images are pseudo-replicates for borough. Use neighbourhood means.
  - A tercile split on images can put images from one neighbourhood in both groups. Make the split on the neighbourhood mean of `yes_no`, or use cluster-robust errors.
  - Five boroughs are few. Treat borough as a fixed group of neighbourhoods. Do not generalise to other boroughs.
- **Mechanical link to the outcome.** `yes_no` comes from the same run as the word logits. At late layers at the answer position, the word logits and the margin share the same hidden state. So a late-layer correlation can be a restatement of the answer. Cross-fit does not remove this. Read the relation layer by layer. Also remove the `yes` and `no` token directions, and read what stays.
- **Word-level Bonferroni power (my calculation, scipy).**
  - `alpha = 0.05 / 54,000 = 9.3e-7`.
  - 197 v 197 images, df 392: `|t| >= 4.99`. Cohen's d is about 0.50 at 50% power. This case ignores clustering, so it is an upper limit on valid power.
  - 66 v 66 neighbourhoods, df 130: `|t| >= 5.15`. Cohen's d is about 0.90 at 50% power.
  - Correlation with `n = 197` neighbourhoods: `|r| >= 0.34`.
  - Each extra layer or token position divides `alpha` again.
- **Recommended stack, in order.**
  1. Fix a vocabulary, double centre, and aggregate to neighbourhoods.
  2. Pre-set 5 to 10 word sets or axes (concreteness, valence, aesthetic terms, status). Compute one score per image.
  3. Test each score with a t-test or OLS, per layer. Apply Bonferroni over sets and layers.
  4. Do a word-level Welch t for description. Report it as a ranked list, with effect size.
  5. Use ridge with grouped cross-fit for a yes/no answer to "is the margin in the words?".
  6. Find themes on half of the neighbourhoods. Confirm them on the other half.

### Gaps
- I found no paper that applies these text methods to lens readouts or to logit vectors. The mapping above is my own. Each entry needs a check on simulated data.
- I did not look for a method paper on the softcap effect on logit contrasts.

## Keyness and distinguishing words: log-odds with Dirichlet prior, TF-IDF, Dunning G2, Scattertext

### Takeaway
All four methods need counts for their inference. The honest count route is top-k membership with one vote per unit. The stronger route uses the logits directly, with a Welch t-test on neighbourhoods. The corpus-linguistics literature itself recommends this t-test over G2.

### Cited Findings
- **Fightin' Words point estimate.** `delta_w = log[(y_i + a_i) / (n_i + a_0 - y_i - a_i)] - log[(y_j + a_j) / (n_j + a_0 - y_j - a_j)]`. (Full text, eq. 16.) — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- **The z-score.** `zeta_w = delta_w / sqrt(sigma^2)`. (Full text, eq. 22.) — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- **The likelihood.** The model is multinomial on the counts in a group, so one group total enters. (Full text, eq. 10-11.) In my text search of the paper, "overdispersion" does not occur. — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- **The prior.** `a_w = a_0 * p_w_hat` shrinks estimates toward a global rate. Its size is a "prior sample". The authors note that the prior can also come from outside sources. (Full text, eq. 23, footnote 18.) — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- **Selection limit.** Every word gets a non-zero weight. The authors note a risk of overfit if the weights feed a later model. (Full text, section 3.4.) — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- **TF-IDF.** Monroe et al. say that "the logic of logging the document frequency breaks down in a collection of two documents". The unlogged variant correlates at 0.997 with a raw difference of proportions in their example. They say that neither version has "clear value" for their purpose. (Full text, section 3.2.7.) — [Monroe et al. 2008](https://languagelog.ldc.upenn.edu/myl/Monroe.pdf)
- **G2 and word independence.** G2 and chi-square treat each word as an independent sample, so they ignore how a word spreads over texts. Lijffijt et al. call these tests anti-conservative. (Tool summary of the full text.) — [Lijffijt et al. 2016](https://academic.oup.com/dsh/article/31/2/374/2462707)
- **Size of the G2 problem.** In random splits of texts, chi-square rejected 57.6% of words, and G2 rejected 65%. The t-test, Wilcoxon and bootstrap tests rejected 3.6% to 4.8%. (Tool summary of the full text.) — [Lijffijt et al. 2016](https://academic.oup.com/dsh/article/31/2/374/2462707)
- **One word.** "Matilda" gets `P < 0.0001` from G2. It gets `P = 0.44` from Welch's t. It occurs in 5 of 409 texts. (Tool summary of the full text.) — [Lijffijt et al. 2016](https://academic.oup.com/dsh/article/31/2/374/2462707)
- **Their unit and tests.** Each text is the unit. The authors recommend the t-test, Wilcoxon rank-sum or bootstrap test. (Tool summary of the full text.) — [Lijffijt et al. 2016](https://academic.oup.com/dsh/article/31/2/374/2462707)
- A 2022 paper agrees that the likelihood-ratio default has "statistical inadequacy", and it lists Wilcoxon and dispersion-based measures as better. (Abstract or excerpt.) — [CLLT 2022](https://www.degruyterbrill.com/document/doi/10.1515/cllt-2022-0116/pdf)
- **Scaled F-score code.** `precision = c / (c + not_c)`. `recall = c / sum(c)`. Each passes through a normal CDF, then a harmonic mean. The code has no check for negative values. (Tool read of the source code.) — [ScaledFScore.py](https://raw.githubusercontent.com/JasonKessler/scattertext/master/scattertext/termscoring/ScaledFScore.py)
- Dense rank in Scattertext is a transform for the plot, in the examples that I saw. (Abstract or excerpt.) — [Scattertext PyData notebook](https://nbviewer.jupyter.org/github/JasonKessler/Scattertext-PyData/blob/master/PyData-Scattertext-Part-1.ipynb)
- The moderated t shrinks word-level (there, gene-level) variances toward a pooled value. It gives "far more stable inference" when the sample is small, with a t-distribution and more degrees of freedom. (Abstract or excerpt.) — [Smyth 2004](https://www.degruyterbrill.com/document/doi/10.2202/1544-6115.1027)

### Inferences
- **Fightin' Words on logits.** Keep the point estimate, as a difference of centred logits. Replace the count variance with the standard error between neighbourhoods. The result is a Welch t. This satisfies the project rule.
- **The prior on logits.** The Dirichlet prior protects rare words with low counts. A logit has no low-count problem. But word-level variances differ, and a word with a small variance can give a large t. The moderated t is the analogue. It is a fair second choice if plain t ranks noisy words high. Report plain t first.
- **Logits as pseudo-counts.** The conversion `count = N * softmax(l)` is invalid for inference. `N` is free, so the z or G2 can take any size. Also, `softmax` depends on the temperature and on the missing vocabulary. Use this only for a ranked description, and only if you state the `N`.
- **Top-k counts.** This route is valid only with one vote per unit. For each word, cross "word in top-k of this neighbourhood's mean read" with the group. Then use chi-square or Fisher. Pooled counts with `n = k * images` repeat the Lijffijt error. Slots in one list are not independent, and three images share a neighbourhood. Report at more than one `k` (for example 10, 30, 100). The top-30 list of the whole vocabulary suits this route, but it holds fragments.
- **Direct logit tests against top-k tests.** A t-test on logits uses all images, so it sees small moves in the middle of the list. But those moves do not reach the model's output. Top-k tests see only words that the read puts at the top. The two answer different questions. Report which one you use.
- **Scattertext.** Do not use its scores on logits. Its plot layout (rank in group A against rank in group B) works with any two scores.

### Gaps
- I did not confirm that Scattertext can take a custom score matrix from outside the corpus.
- The tool summary of Lijffijt was long, and it covered only the first 100,000 characters. The second half of the paper was not read.
- I did not read Dunning 1993 itself. The claims here come from the later critics.
- A search summary says that a paper in the *Journal of Corpus Linguistics* defends ranks by log-likelihood when you add Bayes factors. I could not match this to a URL, so I do not cite it. A report writer must not state this as a finding.

## Topic models and clustering: LDA, STM, BERTopic, Top2Vec, word-embedding clusters

### Takeaway
Do not fit LDA, STM or BERTopic on readouts. They need documents and counts, and a readout list is not a document. The best fit is a clustering of the fixed vocabulary, done once in outside word vectors, and then treated as a set of lexicons. Themes that you read by eye need a held-out test, and blind raters if you can get them.

### Cited Findings
- LDA and PLSA use bag-of-words documents as input. (Abstract or excerpt.) — [Angelov 2020](https://ar5iv.arxiv.org/html/2008.09470)
- STM adds document metadata. Covariates enter the prior on topic prevalence, the prior on topic content, or both. (Abstract or excerpt.) — [stm vignette](https://mirror.linux.duke.edu/pub/cran/web/packages/stm/vignettes/stmVignette.pdf)
- BERTopic runs embed, UMAP, HDBSCAN, bag-of-words per cluster, then c-TF-IDF. c-TF-IDF needs word frequencies per cluster. (Tool summary of the docs.) — [BERTopic algorithm](https://maartengr.github.io/BERTopic/algorithm/algorithm.html)
- BERTopic accepts `fit_transform(docs, embeddings)`, but the docs example still passes the document strings. (Tool summary of the docs.) — [BERTopic embeddings](https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html)
- Top2Vec puts document and word vectors in one space. It finds the number of topics by itself. (Abstract or excerpt.) — [Angelov 2020](https://ar5iv.arxiv.org/html/2008.09470)
- Weights can replace raw counts in LDA. Wilson and Chew report better cross-language retrieval. I did not confirm that their weights can be real-valued. (Abstract or excerpt.) — [Wilson and Chew 2010](https://aclanthology.org/N10-1070/)
- A topic model is a factor model for normalized counts. (Full text.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **Risk of reading by eye.** "Multiple topic model runs can lead to multiple different interpretations", and posteriors are often multimodal. The authors suggest supervision to build interpretability. (Full text, section 3.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **Held-out fit is not meaning.** Models with better held-out likelihood can give less meaningful topics. (Abstract or excerpt.) — [Chang et al. 2009](https://papers.neurips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html)
- **Word intrusion.** Take a topic's top five words, add one low-probability word that is common in another topic, and ask people to find it. (Abstract or excerpt.) — [Chang et al. 2009](https://home.cs.colorado.edu/~jbg/docs/nips2009-rtl.pdf)
- **Coherence.** Röder et al. build coherence measures from combined parts. New combinations correlate better with human ratings. (Abstract or excerpt.) — [Röder et al. 2015](https://ris.uni-paderborn.de/record/50761)
- **Coherence can mislead.** Automated scores can name a winner when human scores do not. — [Hoyle et al. 2021](https://arxiv.org/pdf/2107.02173) (Abstract or excerpt.)
- **Topic labels.** Ying et al. add crowd tasks that test the label, as well as the topic. They find that extensive validation is rarely reported. They also say that no single method fits all cases. (Abstract or excerpt.) — [Ying et al. 2022](https://par.nsf.gov/servlets/purl/10427560)
- **Split-sample rule.** Discovery and estimation on the same documents cause an identification and overfit problem. The fix is a training set for discovery and a test set for estimation. (Abstract or excerpt.) — [Egami et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9581481)

### Inferences
- **Why readouts are a bad fit for LDA.**
  - A top-k list has fixed length and no repeats. So the model has no word-frequency signal in a document.
  - Lists from adjacent layers or images are near copies. So the count of "documents" is inflated.
  - The model would learn the shared vocabulary of the lens, and not a mechanism.
- **The best fit: a fixed vocabulary partition.**
  1. Embed the 40,000 to 54,000 words in outside vectors.
  2. Cluster them (k-means on normalized vectors, or similar).
  3. Treat each cluster as a lexicon.
  4. Compute a cluster score per image.

  The clusters come from outside the data, so they do not use the group labels. That removes the double use of the data. Run several values of `k` and several seeds, and report which clusters repeat.
- **Decomposition of the logits.** PCA on the double-centred image-by-word matrix is the real-valued form of a topic model. It uses each read as given, with no fake counts. Components are not clusters, and a sign is free. Read both ends of a component. The first component can be a global lens scale, so check this before you read it as a theme. I have no source that tests this on logits.
- **Protocol for labels.**
  1. Split neighbourhoods into discovery and test halves.
  2. Name a theme on the discovery half only.
  3. Write the word list and the label.
  4. Test the score on the other half.
  5. Ask a blind rater to do word intrusion on the list, if you claim the label in the paper.
- **Lens-specific risk.** A theme in the top words can be a property of the unembedding matrix, and not of the image. Run the same read on a control input or a shuffled read. The theme must exceed that control. This risk matches the Dubossarsky control design in the semantic change section.

### Gaps
- *Text as Data* (Grimmer, Roberts, Stewart) lists "Validations are Essential and Depend on the Theory and the Task". I saw this only in a table of contents. A listing shows six principles. — [book listing](https://bookrastinating.com/book/337439/s/text-as-data)
- I did not retrieve Grimmer and Stewart 2013 ("promise and pitfalls"). My memory says it gives four principles, but I could not verify this. Do not cite it from these notes.
- I did not verify the exact NPMI and reference corpus settings in Röder et al.
- I did not verify the exact input format of Top2Vec.
- I found no study of how word intrusion works on a list that a model readout produces.

## Lexicons and semantic axes

### Takeaway
Lexicons and axes are the best fit, because they work on a signed score with no sample-size claim. Concreteness and NRC VAD have good coverage for the scene-to-evaluation question. I found no rated lexicon for aesthetic or status vocabulary. Build small lists from the aesthetics literature and from the Kozlowski class pairs.

### Cited Findings

**Lexicons**

| Resource | Size and form | Note | Source |
|---|---|---|---|
| Brysbaert concreteness | 37,058 words, 2,896 two-word terms, scale 1 to 5 | Lemmas | [abstract](https://experts.mcmaster.ca/scholarly-works/57343) |
| Warriner VAD | 13,915 lemmas | Extends ANEW (1,034 words) | [Warriner et al. 2013](https://macsphere.mcmaster.ca/bitstream/11375/22965/1/Warriner%20et%20al%2c%202013%2c%20Behavior%20Research%20Methods.pdf) |
| NRC VAD v1 | About 20,000 words, 0 to 1, Best-Worst Scaling | Commercial use needs consent | [Mohammad 2018](https://aclanthology.org/P18-1017/); [README](https://elearning.unito.it/scienzeumanistiche/pluginfile.php/763390/mod_folder/content/0/NRC-VAD%20lexicon/README) |
| NRC VAD v2 | 44,928 words, 10,073 multi-word terms, -1 to 1 | Adds to v1 | [Mohammad 2025](https://ar5iv.labs.arxiv.org/html/2503.23547) |
| SentiWordNet 3.0 | All WordNet synsets, automatic scores | Scores attach to a sense, not a word form | [Baccianella et al. 2010](https://aclanthology.org/L10-1531/) |
| LIWC-22 | Proprietary dictionary | Needs a licence, and raw categories cannot be shared | [manual](https://www.liwc.app/static/documents/LIWC-22%20Manual%20-%20Development%20and%20Psychometrics.pdf); [pyliwc](https://pypi.org/project/pyliwc) |
| Empath | About 200 categories, new categories from seed words | Fiction embedding, r = 0.906 with LIWC | [Fast et al. 2016](https://arxiv.org/pdf/1602.06979) |
| General Inquirer | 11,789 words, 182 labels | Includes Lasswell values | [Harvard page](https://inquirer.sites.fas.harvard.edu/homecat.htm) |
| SEANCE | 254 core indices, 20 components | A bundle of free dictionaries | [SEANCE](https://soletlab.asu.edu/seance/) |

All rows above are from an abstract or excerpt, except NRC VAD v2 (tool summary of the full text).

- Pollock argues that mean concreteness can hide the real judgments for a large class of words. (Abstract or excerpt.) — [Pollock 2017](https://link.springer.com/article/10.3758/s13428-017-0938-y)

**Aesthetic vocabulary**

- A 2021 study rates 75 aesthetic emotion terms on GRID features. It finds 15 clusters and four dimensions: valence, power, arousal, novelty. — [Frontiers 2021](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.667173/full)
- Augustin et al. start from 77 words for aesthetic impressions of visual art. "Beautiful" and "wonderful" apply about equally across art forms. — [Augustin et al. 2012](https://journals.sagepub.com/doi/10.1068/i0511aap)
- Jacobsen et al. collected free associations to "aesthetics" from 311 German students. More than 90% said "beautiful", and "ugly" came second. — [Jacobsen et al. 2004](https://home.uni-leipzig.de/biocog/archived-20210701/index.php@page=abstract&signature=jacobsen_t2004psycholrep94_1253.html)
- These three are from an abstract or excerpt. All three study art or emotion. None of them studies streets or the built environment.

**Axes**

- **SemAxis.** `V_axis = V+ - V-`, the difference of the two pole means. `score(w) = cos(v_w, V_axis)`. (Full text, eq. 1-2.) — [An et al. 2018](https://arxiv.org/pdf/1806.05521)
- **SemAxis pole sensitivity.** With two pole words, AUC ranges from 67.2 to 92.4, with a mean of 78.2. With the `l = 10` nearest neighbours added, the mean is 80.6 and the minimum is above 71.2. ANEW poles give 77.8 and LabMT poles give 67.5. (Full text, section 4.3.) — [An et al. 2018](https://arxiv.org/pdf/1806.05521)
- **SemAxis axes.** The paper gives 732 pre-set axes from ConceptNet antonyms. The mean cosine between axes is 0.062. (Full text.) — [An et al. 2018](https://arxiv.org/pdf/1806.05521)
- **SemAxis training.** The authors pre-train on a large background corpus, then update on the target text. (Full text.) — [An et al. 2018](https://arxiv.org/pdf/1806.05521)
- **Kozlowski dimension.** Take the arithmetic mean of a set of antonym pair differences. The projection is a cosine with the dimension, because the vectors are normalized. They warn that a weighted sum fits their survey better but is fragile when you cannot survey. (Full text of the 2018 preprint.) — [Kozlowski et al. 2018](https://arxiv.org/pdf/1803.09288)
- **Kozlowski class pairs.** rich-poor, richer-poorer, richest-poorest, affluence-poverty, affluent-impoverished, expensive-inexpensive, luxury-cheap, opulent-needy. (Full text, Table 1.) — [Kozlowski et al. 2018](https://arxiv.org/pdf/1803.09288)
- **Kozlowski validation.** 398 survey respondents, 59 words per scale. Gender correlations are 0.78 to 0.90. Class correlations are 0.40 to 0.60. (Full text, Table 2 text.) — [Kozlowski et al. 2018](https://arxiv.org/pdf/1803.09288)
- **Kozlowski uncertainty.** The authors estimate uncertainty with a subsample method over texts. They find six to ten pairs better than a single pair. (Full text.) — [Kozlowski et al. 2018](https://arxiv.org/pdf/1803.09288)
- **Bolukbasi direction.** Take the top principal component of 10 normalized pair differences. Occupation projections correlate with crowd ratings at Spearman 0.51. Projections on two embeddings agree at 0.81. (Tool summary of the full text.) — [Bolukbasi et al. 2016](https://ar5iv.labs.arxiv.org/html/1607.06520)
- **Rocchio.** A revised query moves toward the centroid of relevant documents and away from the centroid of non-relevant documents. Negative term weights are set to 0. (Abstract or excerpt.) — [Manning et al., IR book](https://nlp.stanford.edu/IR-book/html/htmledition/the-rocchio71-algorithm-1.html)

**Set tests from genomics (a word set against a ranked list has the same form)**

- GSEA asks if set members sit near the top or bottom of a ranked list. It uses a weighted running sum, and `p = 0` gives the Kolmogorov-Smirnov statistic. The null comes from permuted labels. (Abstract or excerpt.) — [GenePattern GSEA docs](https://software.broadinstitute.org/cancer/software/genepattern/modules/docs/GSEA/17)
- Goeman and Bühlmann split tests into competitive and self-contained tests. They also split p-values into gene sampling and subject sampling. (Abstract or excerpt.) — [Goeman and Bühlmann 2007](https://doc.rero.ch/record/303530?ln=en)
- A later review says that competitive tests with gene sampling rest on an untenable independence assumption. The excerpt did not show which of two reviews says this. (Abstract or excerpt.) — [Briefings in Bioinformatics review](https://academic.oup.com/bib/article/15/4/504/407653); [PMC4103537](https://pmc.ncbi.nlm.nih.gov/articles/PMC4103537)
- Wu and Smyth show that correlation between genes inflates set test variance, and they correct it with a variance inflation factor. (Abstract or excerpt.) — [Wu and Smyth 2012](https://www.doi.org/10.1093/nar/gks461)

### Inferences
- **Per-image lexicon loading.** For each image and layer, regress the centred logits on the rating over the covered words. Include log word frequency as a control. The slope is one number per read. It is linear in the logits, and it uses no softmax. Then test slopes with a t-test or OLS on neighbourhoods. A Spearman correlation per read is a robust form.
- **Set contrast.** For an unrated word set, compute (mean logit in set) minus (mean logit in matched background words). Match the background on word frequency. Use this contrast for aesthetic terms and status terms.
- **The scene-to-evaluation question.** Compute two scores per layer: the concreteness loading and an evaluative loading (absolute valence distance from neutral, or the set contrast for aesthetic terms). The mechanism claim predicts that concreteness falls and evaluation rises with depth. Test this with paired differences between layers, per neighbourhood.
- **Valence is not aesthetics.** Valence measures good against bad in general. Use a pole set that is specific to aesthetics as a separate score. A street-specific list is a project product. It needs its own validation.
- **Coverage.** The lexicons hold lemmas. Our vocabulary holds word forms. Lemmatize before the join, and report the share of logit mass or words covered. Words outside the lexicon are not random. Rare and technical words drop out first.
- **Sense.** A logit attaches to a token, not a sense. So SentiWordNet needs a sense guess, and I rank it low.
- **Axes in logit space.** You can compute `corr(l, axis scores)` over the vocabulary. This treats the axis as a rating lexicon, and it uses all words. Do not use only the top-30 words.
- **Pole lists.** SemAxis shows that the pole choice has a large effect. So report each axis over more than one pole list. Pre-set the lists, and use the Kozlowski mean of pairs. It has no fit to our data, so it avoids double use. A Bolukbasi PCA and a Rocchio centroid give similar axes. Rocchio is the same mean-of-poles idea with images as the source. If you use your own top images as poles, you use the outcome twice. Use a split sample in that case.
- **Lexicon sets in the gene-set frame.** A word set against a ranked word list has the same form as a gene set against a ranked gene list. Use a self-contained test: one score per unit, with units as the sample. This matches the t-test rule.
- **Do not sample words.** Words in a lexicon correlate, as genes do. So a test that samples words, such as a Mann-Whitney test of lexicon words against other words, has an invalid null. Sample neighbourhoods, not words.

### Gaps
- I found no rated word-level lexicon for status, prestige, taste or luxury.
- I found no published lexicon that rates aesthetic words for streets, cities or buildings. I did not search urban perception work on its own.
- I did not verify the current licence terms of NRC VAD v2 or LIWC-22.
- I did not confirm which General Inquirer categories SEANCE uses.
- The sum-to-1 rule in SentiWordNet is confirmed for the 2006 version only.
- The final ASR version of Kozlowski may differ from the 2018 preprint that I read.

## Embeddings of readout words

### Takeaway
A weighted centroid is valid, but its result depends on the weight. Softmax weights make it a function of a few top words and the temperature. A linear form, such as regression of logits on embeddings, avoids that choice. WMD adds cost and has weak evidence. Fragments need a filter, because no embedding gives them a stable meaning.

### Cited Findings
- Kozlowski et al. compare Google News, GloVe and other embeddings. Results differ by corpus: the race correlation is 0.70 on Google News and 0.17 on Google Ngrams. (Full text.) — [Kozlowski et al. 2018](https://arxiv.org/pdf/1803.09288)
- fastText sums the vectors of character n-grams, so it gives a vector to a word that it never saw. (Abstract or excerpt.) — [Bojanowski et al. 2017](https://arxiv.org/pdf/1607.04606)
- SIF is a weighted average of word vectors with the first principal component removed. — [Arora et al. 2017](https://mlanthology.org/iclr/2017/arora2017iclr-simple) A critique finds that the SIF gain over a plain average falls when you remove the components from both. (Abstract or excerpt.) — [Karipbayeva and Sorokina](https://ar5iv.arxiv.org/html/1909.13494)
- WMD treats a document as a distribution over words, and takes an optimal transport distance in embedding space. (Secondary source.) — [Sato et al. 2022](https://proceedings.mlr.press/v162/sato22b.html)
- Sato et al. find that classic baselines match WMD after L1 normalization, and they find that WMD values resemble L1-normalized bag-of-words values in high dimensions. (Abstract or excerpt.) — [Sato et al. 2022](https://proceedings.mlr.press/v162/sato22b.html)
- Soft cosine is `a'Sb / sqrt(a'Sa * b'Sb)`, with `S` a term similarity matrix. It reduces to cosine when `S` is the identity. (Abstract or excerpt.) — [arXiv 2602.14755](https://arxiv.org/pdf/2602.14755); [Sidorov et al. 2014](https://cys.cic.ipn.mx/index.php/CyS/article/view/2043)
- gensim gives a sparse term similarity matrix for this. — [gensim tutorial](https://nbviewer.jupyter.org/github/piskvorky/gensim/blob/3.8.1/docs/notebooks/soft_cosine_tutorial.ipynb)
- Nearest-neighbour results from a trained embedding are sensitive to small corpus changes, more so in small corpora. The authors advise an average over bootstrap samples. (Abstract or excerpt.) — [Antoniak and Mimno 2018](https://aclanthology.org/Q18-1008/)
- The project logit lens decodes the hidden state with the unembedding matrix. — [terminology.md](/share/ju/matt/mllmsci/research/docs/terminology.md)

### Inferences
- **Centroid forms.**
  - `c = sum_w softmax(l_w / T) * v_w`. The result depends on `T`, and a low `T` gives one or two words.
  - `c = sum_{top k} softmax(l_w) * v_w`. The result depends on `k`.
  - `c = sum_{top k} (1/k) * v_w`. This is the simplest form. It uses ranks only.

  Report more than one setting. Do not compare centroids across layers without a fix for logit scale.
- **Linear form with no weight choice.** Ridge-regress the centred logits on the embedding matrix, over the vocabulary. This gives a vector `b` in embedding space, for each read. The cross-validated R-squared says how much of the read the word meanings explain. Use folds of words. A low R-squared warns that centroid and axis results describe a small part of the read. Word vectors of related words correlate, so folds of words give an optimistic result.
- **The model's own vectors.** The lens is a linear map through `lm_head` before softcap. A column of `lm_head` is a word vector in the model's own space, and it has no out-of-vocabulary case. An axis made from these columns is close to a direct probe of the hidden state. Two limits apply. First, this space is not validated against human ratings. Second, you can compute the answer without any word list, so a claim about "word meaning" needs an outside embedding too. Use the outside embedding for the claim about meaning. Use the model's vectors as a check.
- **Stability.** The Antoniak problem applies to embeddings that you train yourself. We use pre-trained vectors. Our uncertainty is the embedding choice. Repeat each main result on two embeddings (for example GloVe and fastText).
- **Fragments.** The lower-case English list avoids most of them. The top-30 list of the whole vocabulary can hold fragments. Options:
  - Drop tokens that fail a word-list check, and report the dropped share.
  - Keep them as a separate "fragment mass" variable. Their share across layers can be a result.
  - Do not trust fastText vectors for fragments. They come from the n-grams of the string, so they measure spelling. For example, a fragment of a word has a vector near words that share those letters, not near its source word.
- **WMD and soft cosine.** Both need a reduced list. A 54,000-square transport problem is not practical. On top-k lists, WMD compares two reads, but not a read with an outcome. Soft cosine is the cheaper of the two. Rank both low. Neither gives a test by itself, so a test needs a permutation of units.

### Gaps
- I did not verify the exact SIF weight formula from the paper. My memory says `a / (a + p(w))`.
- I found no study of how sentence-transformers handle a single word fragment.
- I did not retrieve Kusner et al. 2015. The WMD claims come from Sato et al.
- I found no source on how much logit variance word embeddings explain in a lens read. This is a project measurement.

## Text regression: which words relate to a continuous outcome

### Takeaway
Use ridge or PLS with folds grouped by neighbourhood to ask whether the margin is in the words. Do not use the coefficients as a word list. A word list comes from marginal t or correlation tests, with Bonferroni. MNIR needs counts, but its idea has a simple Gaussian form that fits logits.

### Cited Findings
- **Penalized models.** Gentzkow et al. recommend penalized linear models for most text regression. They cite interpretability and software. (Full text, section 3.1.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **Correlated inputs.** Selection can give poor forecasts when predictors correlate highly. In a case where all predictors are the target plus noise, lasso is inferior to a simple average of the predictors. (Full text, section 3.1.2.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **Word coefficients.** Coefficients "are typically only interpretable in cases where the true model is extremely sparse". Otherwise multicollinearity makes the selected set "highly unstable". (Full text.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **Use of coefficients.** Inspection of the largest coefficients is still useful as a sanity check. (Full text.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **Penalty choice.** BIC tends to pick simpler models than cross-validation. Zou et al. recommend BIC when selection, not prediction, is the goal. (Full text.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **PCR and PLS.** PCR reduces predictors without the outcome. PLS uses the covariance with the outcome. `K` can be tuned by cross-validation. Neither is scale invariant, so features are often standardized first. (Full text.) — [Gentzkow et al. 2019](https://web.stanford.edu/%7Egentzkow/research/text-as-data.pdf)
- **MNIR.** Taddy models word counts as multinomial, given the sentiment. The sufficient reduction score is `z_i = phi' f_i`, with `f_i = x_i / m_i`. Estimation uses Laplace priors and a gamma-lasso. (Abstract or excerpt.) — [Taddy 2013](https://arxiv.org/pdf/1012.2098)
- Software for MNIR is the R package `textir`. — [textir manual](https://taddylab.r-universe.dev/textir/doc/manual.html)
- Split-sample logic applies to measures that you discover from text. (Abstract or excerpt.) — [Egami et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9581481)

### Inferences
- **Specific form.**
  - `X` is 591 by V, centred per read and per word. Or use neighbourhood means, which gives 197 by V.
  - `y` is `yes_no`.
  - Folds are grouped by neighbourhood, so no neighbourhood appears in both train and test.
  - Report out-of-fold R-squared. Get the null from permuted `y` at the neighbourhood level.
  - Repeat per layer. The curve of R-squared over layers is the main result.
- **Ridge over lasso.** Word logits correlate strongly, because they come from one low-rank hidden state. By the Gentzkow argument, that favours averages over selection. Ridge is cheap in the `n << p` case, as a dual problem with an n-by-n matrix.
- **Limit of R-squared.** A hidden state of a few thousand dimensions projects into all word logits. So a high R-squared at late layers does not show that the words carry the meaning. It can show only that the logits are a linear image of the hidden state. Compare with the same regression on the hidden state, or on a random projection of the logits. If the random projection does as well, the word identity adds nothing.
- **Gaussian MNIR.** Regress each word's logit on `yes_no`, one word at a time. Take the slopes as `phi`, and form `z = phi' l`. This is the inverse regression idea with a Gaussian likelihood in place of the multinomial. It is close to a one-component PLS. Fit `phi` on the training folds only, because a fit on all data leaks. This gives the word-level result too: the slope and its t. This is the most direct way to reconcile a word list with the prediction.
- **Word list for the paper.** Use marginal slopes with Bonferroni on neighbourhoods. With `n = 197`, a word needs `|r| >= 0.34`. Then apply the theme methods to the list only as description.
- **Controls.** Include image covariates only if they are not on the path of the judgment. Scene content is a likely mediator. A control for it changes the question from "what relates to the margin" to "what relates beyond content".

### Gaps
- I read only the abstract and excerpts of Taddy 2013. I did not check if a real-valued or Gaussian form appears in that paper.
- I found no source on the use of cross-fit or double machine learning for lens readouts.
- I found no direct source for ridge in this `p >> n`, low-rank setting on logits.
- I searched for no Gentzkow-style text on grouped folds. The grouped fold rule is my own reasoning.

## Conceptual change across a sequence: semantic change methods applied across layers

### Takeaway
Semantic change methods align separate embedding spaces, and we do not need that step. All layers score the same words, so a word's position is already comparable after a scale fix. The method to borrow is the control design. Measured change often has an artefact source, so each claim of change needs a shuffled control.

### Cited Findings
- Hamilton et al. align embeddings from different periods with orthogonal Procrustes. This keeps cosine similarities unchanged. (Abstract or excerpt.) — [Hamilton et al. 2016](https://arxiv.org/pdf/1605.09096)
- Their reported laws say that the rate of change falls with word frequency, and that polysemous words change faster. (Abstract or excerpt.) — [Hamilton et al. 2016](https://arxiv.org/pdf/1605.09096)
- Dubossarsky et al. rebuilt the analysis on a shuffled corpus, with text randomly reassigned across decades. There, no real change exists. The laws appear again. They conclude that the frequency effect is largely an artefact of the word representations. (Abstract or excerpt.) — [Dubossarsky et al. 2017](https://aclanthology.org/D17-1118/)
- A survey confirms that all three laws appeared in the shuffled corpus too. (Abstract or excerpt.) — [Survey of Computational Approaches to Lexical Semantic Change](https://arxiv.org/pdf/1811.06278)
- Rank-biased overlap (RBO) compares incomplete ranked lists. It handles lists with few shared items, it weights top ranks more, and it gives bounds for the unseen part. It is a weighted mean of overlap at each depth, with geometric weights. (Abstract or excerpt.) — [Webber et al. 2010](https://people.eng.unimelb.edu.au/ammoffat/abstracts/wmz10acmtois.html); [formula restated](https://arxiv.org/pdf/1408.3587)
- Top-10 sets can have zero overlap across bootstrap samples for some words. (Abstract or excerpt.) — [Antoniak and Mimno 2018](https://par.nsf.gov/servlets/purl/10057829)

### Inferences
- **No alignment.** Procrustes solves arbitrary rotation between spaces. A logit vector has fixed word axes in all layers. So skip it. The change that we need to remove is scale and shift. Double centring and a per-read scale fix handle that.
- **Mapping of terms.**
  - A word in a time slice maps to a word in a layer.
  - A word's vector in Hamilton maps to the word's profile over images: its 591 logits at that layer.
  - Word change between layers maps to `1 - corr(profile at L, profile at L+1)`.
  - Read change between layers maps to `1 - corr(read at L, read at L+1)`, or to RBO on the top lists.
- **Three scales of change.**
  - **Sets:** loading per layer for concreteness and aesthetic terms. Test with paired differences. This one answers the main question.
  - **Lists:** RBO between adjacent layers. A sharp fall marks a candidate turning point. Use it as description.
  - **Words:** which words move most. Use it to read the turn. Apply the discovery and test split.
- **Controls, from Dubossarsky.**
  - Shuffle layer labels inside each image. A real trend must exceed this null.
  - Run a control prompt with no evaluation (a content question on the same images), if the data exist. The trend must be absent, or be weaker.
  - Check if change depends on word frequency or on the norm of the `lm_head` column. A dependence on these points to an artefact.
- **The J lens as a confound.** Each layer has its own fitted `J_l`. So a change across layers can come from the change in the fit, and not from the model's content. A readout difference between adjacent layers mixes the two. Compare with the plain logit lens, which has `J = I`.
- **Multiple tests.** Pre-set a small set of layers, or fit one trend per score. A test at each layer pair multiplies Bonferroni.

### Gaps
- I did not verify a second-order similarity method for semantic change. My memory says that Hamilton et al. have a related paper. It would fit the profile idea above.
- I found no paper that applies semantic change methods to layers of a network.
- I did not choose an RBO persistence `p`. Report it with the result. I found no guide for lists of this kind.
- I did not check if a control prompt run exists in the project data.
