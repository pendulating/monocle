# Statistics for ranked lists and word sets in lens readouts

Evidence tags on each cited line:

- **[read]**: I read the source text (full text, or a section of it).
- **[fetch]**: A fetch tool extracted this from the source page. I did not read every line.
- **[snippet]**: This comes from a search summary or a secondary source only. Check it before you cite it in a paper.

All arithmetic in the Inferences parts is mine, unless a source is named. Numbers assume 197 NTAs, a two-sided test and alpha 0.05.

## Rank-list similarity: which measure fits a long tail, and how do you get an inference?

### Takeaway

Use rank-biased overlap (RBO) with a preset depth, or Jaccard at a few preset k, as a per-image score. Do not use full-list Kendall or Spearman, because they weight the noisy tail the same as the top. Get the inference from images (t test on NTA means against a cross-image baseline), not from words.

### Cited Findings

- Kendall tau is unweighted. It puts "as much emphasis on disorder at the bottom of the ranking as it does on disorder at the top". [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- Webber et al. apply the objection to correlation with indefinite lists to Kendall, to other rank correlations, and to their weighted and top-k forms. [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- For a prefix of a long list, "random and negatively correlated rankings will look essentially the same". Thus the test must be for departure "not from randomness, but from agreement". [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- RBO uses a user model. The user compares overlap at increasing depth, and continues to the next rank with probability p. [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- The weight of the first d ranks is Eq. 21: W(1:d) = 1 - p^(d-1) + ((1-p)/p) * d * (ln(1/(1-p)) - sum_{i=1}^{d-1} p^i / i). [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- With p = 0.9, the first 10 ranks hold 86% of the weight. With p = 0.98, the first 50 ranks hold the same weight. [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- W(1:d) does not change with list length, so the tail does not dominate the head. [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- RBO gives a lower bound, a residual and an extrapolated value (RBO_EXT). 1 - RBO is a proved metric (rank-biased distance). [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- RBO uses set overlap, so it accepts non-conjoint lists. Kendall tau requires conjoint lists. [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- Inference: Webber et al. say that significance tests on one pair of lists are "problematic". Confidence intervals on rank correlation "tend to be very wide". [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- The authors put the sampled unit at the query level: "the (conceptual) random selection is not of documents, but of queries". Then inference "can proceed along the standard lines". [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- `scipy.stats.weightedtau` uses hyperbolic weights 1/(r+1) by default. It returns no p value, because the null distribution is unknown. [fetch] — [SciPy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html); the method is Vigna 2015.
- Tau-AP is top-weighted, but it is not symmetric and it needs conjoint lists. [read] — [Webber et al. 2010](https://codalism.com/research/papers/wmz10_tois.pdf)
- OrderedList scores two lists with exponential weights on overlap at each rank, and compares the score with a random reference to get an empirical p value. [snippet] — [OrderedList manual](https://bioc.r-universe.dev/OrderedList/doc/manual.html)
- Rank-rank hypergeometric overlap (RRHO) steps through two lists and tests overlap at each step. The R package offers permutation FWER control and BY FDR control. [snippet] — [RRHO manual](https://bioc.r-universe.dev/RRHO/doc/manual.html)
- Correspondence-at-the-top curves plot top-N overlap against N. [snippet] — [matchBox docs](https://rdrr.io/github/marchion/matchBox/man/computeCat.html)
- Sequential rank agreement uses the standard deviation of ranks at each depth, for two or more lists. It compares the curve with a permutation reference. [fetch] — [Ekstrom et al.](https://arxiv.org/abs/1508.06803)

Comparison of measures for lists of about 50,000 words:

| Measure | Top-weighted | Conjoint needed | Tail effect | p value on one pair |
|---|---|---|---|---|
| Spearman | No | Yes | Tail dominates | Formula exists, not valid here |
| Kendall tau | No | Yes | Tail dominates | Formula exists, not valid here |
| Weighted tau | Yes, slow decay | Yes | Reduced | None (SciPy gives nan) |
| Jaccard at k | Hard cut | No | None | Hypergeometric, not valid here |
| RBO | Yes, tunable | No | Set by p | None |
| Sequential rank agreement | Depth curve | No | Shown by depth | Permutation reference |

### Inferences

- The list here is fully observed, not indefinite. RBO bounds and extrapolation matter only if you store a top-k prefix. If you store full lists, RBO to full depth has no real residual.
- Choose p from Eq. 21 and fix it first. My values: p = 0.99 puts 85% of weight on the top 100, and p = 0.999 puts 85% on the top 1,000. Report a second p as a check.
- Do not use the p value of Spearman or Kendall at n = 50,000. Words are not a random sample, and they are correlated. At this n, a p value is near zero for trivial agreement.
- The null "RBO = 0" has no use. Most lens lists share a common top, because the model has a word prior. Two different images can then have high RBO.
- A valid design for question (b):
  1. For each image, compute RBO between condition A and condition B (two layers, two positions, or two models).
  2. Compute the same RBO between image i at A and an image from another NTA at B.
  3. Take the difference, and average it within each NTA.
  4. Do a one-sample t test on the 197 NTA values.
  5. As a check, shuffle the image-to-image pairs between A and B by whole NTA. This is the analog of sample permutation.
- This design tells you whether a list is stable for an image beyond the shared prior. Another way is to remove the mean logit of each word across images before you rank. This removes the shared prior directly. If you use it, compute the means on training images only.
- Across models, vocabularies differ. Restrict to the shared words first. If you do not, RBO counts a vocabulary difference as a disagreement.
- RRHO is useful as a picture. Its hypergeometric p values have the word-independence fault (see the next section). Use only the permutation form, and permute images, not words.
- Keep Jaccard at preset k (for example 10, 100, 1000) as the plain secondary report. It is easy to read, but it ignores order inside the top k.

### Gaps

- I found no source that gives a null distribution or significance test for RBO itself. A search for this returned no direct result. The per-image design above is my own proposal, built on the Webber et al. statement about query-level inference.
- I did not check how RBO is usually tested in IR practice. A search tool suggested a paired permutation test over queries, but it gave no source.
- I did not read Fagin et al. 2003, Yilmaz et al. 2008 or Vigna 2015 directly. Claims on them come through Webber et al. or SciPy.
- I did not read the Plaisier et al. or Irizarry et al. papers.

## Enrichment: which set test fits, what is the fault of item permutation, and what is the analog of sample permutation?

### Takeaway

Item permutation and the hypergeometric test treat words as independent sampled units. Words here are not independent, so these tests are too liberal, and their p values do not refer to new images. The analog of sample permutation is to shuffle or sign-flip at the NTA level. I found no prior use of GSEA on a language model vocabulary.

### Cited Findings

Fault of item-level tests:

- Goeman and Buhlmann separate gene sampling (the gene is the sampling unit) from subject sampling (the subject is the sampling unit). The 2x2 table methods are competitive and gene-sampling methods. [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- Under gene sampling, a replication means "repeating the experiment for new genes and the same subjects". [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- With correlated genes, the hypergeometric test "is anti-conservative; it may greatly understate the true P-values". A significant result can "simply indicate that the genes are dependent". [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- Wu and Smyth report that gene-permutation tests "are shown here to be extremely sensitive to inter-gene correlation". [fetch] — [Wu and Smyth 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458527/)
- Their Table 1 has set size 100, correlation 0.05, nominal 0.05. Rejection rates: geneSetTest 0.4157, geneSetTest on ranks 0.4144, sigPathway 0.3880, PAGE 0.3709, CAMERA 0.0477, CAMERA on ranks 0.0473. [fetch] — [Wu and Smyth 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458527/)
- The variance inflation factor (VIF) "can be much larger than one when the test set contains many genes", even for a small correlation. The formula renders as an image on the page, so I could not read it there. [fetch] — [Wu and Smyth 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458527/)
- A later paper says that the gene-sampling competitive framework has "serious errors" because of its assumption of independence across genes. [snippet] — [Briefings in Bioinformatics article](https://pmc.ncbi.nlm.nih.gov/articles/PMC4103537)

GSEA:

- ES is the maximum deviation from zero of a running sum down the ranked list. With p = 0 it is the Kolmogorov-Smirnov statistic. The authors use p = 1. [fetch] — [Subramanian et al. 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC1239896/)
- GSEA permutes phenotype labels, and "the permutation of class labels preserves gene-gene correlations". The paper uses 1,000 permutations and reports sets at FDR <= 0.25. [fetch] — [Subramanian et al. 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC1239896/)
- Sample permutation "inevitably alters the hypothesis that is being tested". GSEA is "a hybrid test for which the null and alternative hypotheses are difficult to characterize". [fetch] — [Wu and Smyth 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458527/)
- With subject permutation, the null becomes the complete null that no gene in the set or outside it is differentially expressed. [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- Efron and Tibshirani propose the maxmean statistic and restandardization, which uses column permutation with a gene-level adjustment. [snippet] — [Efron and Tibshirani 2007](https://arxiv.org/pdf/math/0610667v2); [GSA manual](https://cran.r-universe.dev/GSA/doc/manual.html)
- Preranked GSEA: lab pages and explainers say that gene-set permutation is the usual mode when only a ranked vector exists, and that phenotype permutation needs replicates. The cutoffs differ (7 per group in tool warnings, more than 20 on one lab page). I could not get the official guide text. [snippet] — [Bader Lab GSEA parameters](https://baderlab.org/CancerStemCellProject/VeroniqueVoisin/AdditionalResources/GSEA/parameters)

CAMERA and ROAST:

- CAMERA is a competitive test that adjusts the variance of the set mean for inter-gene correlation. It estimates one correlation parameter. [fetch] — [Wu and Smyth 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3458527/)
- `cameraPR` takes a precomputed statistic. Its default is `inter.gene.cor = 0.01`. Setting NA makes limma estimate a correlation for each set. [snippet] — [limma camera help](https://rdrr.io/bioc/limma/man/camera.html)
- Gordon Smyth said of the fixed default: it "will not control the type I error rate correctly, but this was unavoidable given we wanted to improve the gene set rankings". [fetch] — [Bioconductor support](https://support.bioconductor.org/p/86278/)
- ROAST is self-contained. It rotates residuals in the residual space of a linear model, and keeps inter-probe correlation. [fetch] — [Wu et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922896/)
- Rotation helps at small n, because "the number of rotations does not depend on sample size". Permutation is said to need "a moderate to large number of replicates". [fetch] — [Wu et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922896/)
- ROAST offers mean, msq, mean50 and floormean. Mean suits a set where most members move. msq suits a set where few members move. The authors suggest mean50 as a compromise. [fetch] — [Wu et al. 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2922896/)

Single-sample scores:

- GSVA gives sample-wise scores as a function of genes inside and outside the set. [snippet] — [Hanzelmann et al. 2013](https://bmcbioinformatics.biomedcentral.com/track/pdf/10.1186/1471-2105-14-7)
- ssGSEA scores each sample by the normalized difference of ECDFs of ranks inside and outside the set. Its final normalization can be turned off. [snippet] — [GSVA vignette](https://bioconductor.org/packages/devel/bioc/vignettes/GSVA/inst/doc/GSVA.html)
- singscore is rank-based and independent of sample composition. Its authors say GSVA, z-score, PLAGE and ssGSEA scores can be unstable with less data. [snippet] — [Foroutan et al. 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6219008/)

Use on words or in NLP:

- ttgsea runs GSEA on genes, then predicts scores for words in gene set names. This is a reverse use, not a test on a word ranking. [snippet] — [ttgsea vignette](https://bioconductor.org/packages/release/bioc/vignettes/ttgsea/inst/doc/ttgsea.html)
- WEAT is the nearest NLP analog. Its p value comes from a permutation over partitions of the target words. I read this in secondary descriptions only. [snippet] — [Ethayarajh et al. 2019](https://arxiv.org/abs/1908.06361) describe WEAT, and [a test_WEAT help page](https://rdrr.io/cran/PsychWordVec/man/test_WEAT.html) implements it.
- WEAT can give opposite signs for the same target pair with different attribute words, and both results are "highly statistically significant". [read] — [Ethayarajh et al. 2019](https://arxiv.org/pdf/1908.06361)
- Interpretability work often reads a coherent concept from top logit tokens by inspection, not with a ranked test. [snippet] — [Actionable interpretability survey](https://arxiv.org/pdf/2601.14004)
- "Enrichment neurons" in recent work means a unit that adds a residual direction back. This is a different sense of the word. [snippet] — [Gerstner and Schutze 2025](https://arxiv.org/html/2505.17936v1)
- One embedding study uses 110 semantic categories with more than 6,500 words to score embedding dimensions. This is a set test idea on static embeddings. [snippet] — [Semantic Structure and Interpretability of Word Embeddings](https://arxiv.org/pdf/1711.00331)

Comparison of set tests:

| Method | Null type | Sampled unit | Word correlation handled | Use here |
|---|---|---|---|---|
| Hypergeometric on top k | Competitive | Word | No | Not valid as a p value |
| Wilcoxon on words in one image | Competitive | Word | No | Use U as a score only |
| GSEA, item permutation | Competitive | Word | No | Not valid |
| Preranked GSEA | Competitive | Word | No | Not valid |
| GSEA, sample permutation | Hybrid | Image or NTA | Yes | Valid for (c), and for group contrast |
| CAMERA / cameraPR | Competitive | Word, with VIF | Yes, by one number | Valid if rho is estimated |
| ROAST | Self-contained | Image or NTA | Yes | Valid, but no gain at n = 197 |
| GSVA | Per-sample score | Image | Yes, after the score | Fit on training data only |
| ssGSEA, singscore | Per-sample score | Image | Yes, after the score | Good score, then t test |

### Inferences

- The analog map: gene = word, subject = image or NTA, phenotype = margin or group label, gene permutation = word shuffle, sample permutation = shuffle of margin or labels across NTAs.
- Question (a) has no phenotype, so nothing can shuffle labels. The analog is a sign flip of each NTA's set-minus-control score. This is the one-sample form of a permutation test.
- GSEA with sample permutation does not fit (a). It fits (c), with margin as a continuous phenotype, and it fits group contrasts. A plain Pearson t statistic for each word gives the ranked list.
- If the lens output is linear in the hidden state h, then logit(w) = u_w . h + b_w. The mean logit of set S is then a single projection, (mean u_S) . h. So word-word covariance across images comes from the unembedding geometry and the covariance of h. The effective number of free directions cannot exceed the hidden size, even with 54,000 words. Confirm that the lens has this form before you use this argument.
- This gives a simple check. Cosine similarity of the mean unembedding vectors of two sets shows how far the two sets are separate tests. If it is high, two "different" set results can be the same result.
- VIF with the standard form 1 + (m-1)rho (my use, consistent with the Wu and Smyth text, and the Moulton form in the clustered section):

| Set size m | rho | VIF | Effective words m/VIF |
|---|---|---|---|
| 100 | 0.05 | 5.95 | 16.8 |
| 100 | 0.20 | 20.8 | 4.8 |
| 500 | 0.05 | 25.95 | 19.3 |
| 500 | 0.20 | 100.8 | 5.0 |

- A larger set gives little more information past 1/rho effective words. Words are likely more correlated than genes in a typical set, so the genomics default `inter.gene.cor = 0.01` is not safe. Estimate rho from the residuals, or use a method on images that does not need rho. This is my judgment, and you must measure rho.
- The per-image score avoids all of this. The score is a single number for each image, and the t test over NTAs uses its observed variance. No VIF is needed.
- Per-image score options, from simple to rich:
  1. Mean logit of set minus mean logit of matched control.
  2. AUC of set against control in one image (U statistic divided by the product of set sizes). The null value is 0.5, so it has a natural anchor.
  3. Count of set words in the top k, minus the same count for control.
  4. ssGSEA with normalization off, or singscore.
- AUC weights all ranks the same. It can miss a set where only a few words rise to the top. Top-k count and ssGSEA catch that case. Choose one primary score first, because more scores add more tests.
- GSVA fits on all samples. If you use it, fit on training images only, or it leaks across a split.
- Do not get a p value by comparison of a set with many random control sets on the group mean. That is item sampling, with the same fault as gene permutation. Use resampled controls only as a robustness interval.

### Gaps

- I found no application of GSEA, CAMERA or ROAST to a language model vocabulary ranking. Two searches found none, but this does not prove that none exists.
- I could not open the official GSEA user guide for permutation advice or for preranked limits. All thresholds above are secondary. At 197 NTAs, neither threshold is a limit.
- I did not get the Caliskan et al. 2017 original. The statement on WEAT permutation is secondary. Ethayarajh et al. discuss WEAT significance, but the text I searched does not describe the permutation.
- I could not open Tamayo et al. 2016 or Gatti et al. 2010. I make no claim from them.
- Wu and Smyth do not state how correlation between the set and the background affects the test. I found no sentence on this point.

## The simplest valid default: per-image set score minus matched control, then a t test

### Takeaway

Yes, this default is valid, and it fits the project rule. It is neither a pure competitive test nor a pure self-contained test. It uses a competitive statistic with subject sampling, so the p value refers to new images with the same words. The t test must use NTA means (df = 196), and the control must match on frequency-type properties.

### Cited Findings

- A competitive test compares the set with its complement. A self-contained test compares the set with a fixed standard that does not depend on genes outside the set. [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- A test with a competitive statistic and a subject-sampling p value is "a hybrid form". [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- Goeman and Buhlmann prefer self-contained tests in general. They also say the choice "should depend on the biological interpretation of the null hypothesis". [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- Self-contained tests can be "too powerful": with many changed genes, almost all sets can be significant. [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- The competitive null is "much more difficult to test because its definition is closely tied to a gene sampling model". [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- A subject-sampling P value gives confidence about a new sample of subjects, but it "does not make any statement about replications to new genes". [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)
- Clark 1973 argues that you need a statistical basis to generalize from a sample of words. Judd et al. 2012 use crossed random effects for subjects and stimuli. [snippet] — [PNAS commentary on the names fallacy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9499510/); [The stimuli-as-fixed-effect fallacy](https://www.r-bloggers.com/2012/06/the-stimuli-as-a-fixed-effect-fallacy/)
- goseq shows the bias case in genomics. Longer genes have more power, so a plain hypergeometric null is wrong for RNA-seq. goseq weights each gene by a probability that depends on length or counts. [snippet] — [goseq help](https://rdrr.io/bioc/goseq/man/goseq.html)
- The goseq manual says that "Hypergeometric" should NEVER be used for results for biological interpretation. Wallenius is the recommended option. [snippet] — [goseq help](https://rdrr.io/bioc/goseq/man/goseq.html)
- The squared embedding norm is linear in the log probability of the word. Because of this, WEAT can be non-zero for unbiased sets unless the attribute words have equal frequency. [read] — [Ethayarajh et al. 2019](https://arxiv.org/pdf/1908.06361)
- Under this analysis, equal association needs p(x) = p(y). Caliskan et al. did not state this requirement. [read] — [Ethayarajh et al. 2019](https://arxiv.org/pdf/1908.06361)

### Inferences

Is this the same as competitive or self-contained? The answer depends on the score:

| Score | Null | Class |
|---|---|---|
| Raw mean set logit | Mean logit = 0 | Self-contained, but zero has no meaning |
| Set logit minus vocabulary mean | Set equals all words | Competitive statistic, subject sampling |
| Set logit minus matched control | Set equals control words | Competitive statistic, subject sampling |
| Slope of set score on margin | Slope = 0 | Self-contained form |
| Slope of (set - control) on margin | Slopes equal | Competitive statistic, subject sampling |

- A softmax does not change when you add a constant to all logits. So a raw logit has no fixed zero, and a self-contained test for (a) has no anchor. You must contrast with other words, or use ranks.
- For (c), a self-contained slope has a clear null, but a global shift can make all sets significant. A change in the size of h with margin can do this. This is the "too powerful" fault. Use the set-minus-control slope. Report the raw slope only as a secondary value.
- This t test is not exposed to the Goeman objection to the competitive null. The null here is a plain population statement about images: the mean difference is zero. The word set is fixed in the statistic, and the variance comes from images.
- Scope of the claim: the result applies to these words on new images from the same population. It does not apply to "evaluative words in general". For that claim, words must be a random factor. A crossed mixed model is possible but heavy. A simple check is leave-one-word-out, plus a report of the top contributing words.
- Matched control means that each set word gets control words that are alike on all properties, except for set membership. Candidate properties, in order of likely importance here:
  1. Unembedding norm, because of the norm-frequency link.
  2. Log corpus frequency.
  3. Model prior (mean logit on neutral or held-out inputs).
  4. Number of tokens, if the lens scores multi-token words.
  5. Character length, part of speech and casing.
- Warning: If you compute the model prior from the same images, the match depends on the outcome. Use held-out images or a neutral prompt.
- Use control words that are far from the set in the unembedding. Near synonyms in the control reduce the contrast.
- Fix the control sets before the analysis. Use several control words for each set word, so that the control mean is stable.
- Textbook forms that stay valid, if the unit is the NTA:
  - One-sample t test on 197 NTA means of the score.
  - Binomial sign test: count of NTAs with score above zero, against 0.5. This is a robust check.
  - Two-sample t test on NTA means, for a group label at NTA level.
  - Paired t test, for two conditions on the same images, or for two models.
  - Chi-square, only if each NTA gives one count to the table.
- Textbook forms that are not valid: a chi-square or Fisher table with words as units, a word-level t test, or an image-level table that ignores NTAs.

Unit-of-analysis table:

| Question | Score | Unit | Test | df or N |
|---|---|---|---|---|
| (a) Set is high at a cell | Set minus control | NTA mean | One-sample t | 196 |
| (a) Word is high at a cell | Word logit minus vocabulary mean | NTA mean | One-sample t | 196 |
| (b) List is stable | RBO minus cross-image RBO | NTA mean | One-sample t | 196 |
| (b) Set effect differs by layer | Difference of scores | NTA mean | Paired t | 196 |
| (b) Group difference, NTA-level label | Set score | NTA mean | Two-sample t | 195 |
| (b) Model difference | Difference of scores, same images | NTA mean | Paired t | 196 |
| (c) Set goes with margin, between NTAs | NTA mean score, NTA mean margin | NTA | Pearson t | 195 |
| (c) Set goes with margin, all images | Image score, image margin | Image, clustered by NTA | OLS with cluster SE, T(196) | 591 images |

- For (c), watch a built-in link. The margin is itself a difference of two logits (`yes` and `no`). At the output position, a set score and margin are two projections of the same h. Thus they can correlate because of unembedding geometry alone. Check the cosine of (mean u_S - mean u_C) with (u_yes - u_no), and exclude words close to `yes` or `no`. The strength of this effect in your data is unknown.

### Gaps

- I found no paper that tests this exact design (per-sample set minus matched control, then t test) against CAMERA or ROAST. The validity argument here is mine, from standard one-sample inference.
- I did not read Clark 1973 or Judd et al. 2012 directly.
- I found no standard for how to match control words in a lens analysis. The property list above is my judgment from the goseq and WEAT cases.

## Multiple testing: which correction is right for a position-by-layer grid?

### Takeaway

Use Holm for a small preset family. Use max-t sign-flip permutation over the grid for a claim about a specific cell, because it has strong FWER control and uses the correlation between cells. Use BH only for an exploratory map, and label it so. Do not use a cluster test to claim a specific layer or position.

### Cited Findings

- Holm rests on the Bonferroni inequality, so it holds for any joint distribution of the statistics. It has more power than Bonferroni. [snippet] — [Wikipedia, Holm-Bonferroni](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method); [Shaffer 1986](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1986-Shaffer-JASA.pdf)
- BH controls FDR when the test statistics are positively regression dependent on each true-null statistic. For arbitrary dependence, the bound gains the factor sum_{j=1}^{m} 1/j, which is BY. [snippet, abstract level] — [Benjamini and Yekutieli 2001](https://projecteuclid.org/euclid.aos/1013699998)
- PRDS does not carry over directly to two-sided tests. Yekutieli 2008 gives non-PRDS two-sided Gaussian examples. [snippet] — [Dobriban preprint](https://arxiv.org/pdf/2607.12208)
- A 2026 preprint claims a counterexample: BH fails nominal FDR for correlated two-sided Gaussian p values at alpha 0.01. I did not check this result, and it is not peer reviewed as far as I know. [snippet] — [Dobriban preprint](https://arxiv.org/pdf/2607.12208)
- Earlier support for BH in this case came from simulation (Reiner-Benaim 2007). [snippet] — [Dobriban preprint](https://arxiv.org/pdf/2607.12208)
- The cluster-based test checks that the data in the conditions come from the same distribution. Significance "only speaks to the null hypothesis and not to the 'where' or 'when' of this difference". [fetch] — [FieldTrip FAQ](https://www.fieldtriptoolbox.org/faq/stats/clusterstats_interpretation/)
- The cluster extent depends on the threshold: "a stringent threshold will lead to a focal effect, while a liberal threshold will produce a widespread effect". [fetch] — [FieldTrip FAQ](https://www.fieldtriptoolbox.org/faq/stats/clusterstats_interpretation/)
- Cluster-sum inference gives "weak control over the familywise error rate", and "no error control at individual timepoints". [fetch] — [Rousselet 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11670281/)
- Sassenhagen and Draschkow 2019: cluster tests do not establish significance of effect latency or location. [snippet] — [record of the paper](https://www.psych.ox.ac.uk/team/dejan-draschkow/publication_modal/965525)
- Max-t gives strong FWER control and pointwise thresholds, and it is conservative. BH and BY give only weak FWER control, and "limited claims about the location of effects". [fetch] — [Rousselet 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11670281/)
- In the Rousselet onset simulation, cluster-sum had low bias and variance among standard methods. So a cluster map can be a good estimate, even without pointwise error control. [fetch] — [Rousselet 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11670281/)
- The max statistic is recorded for each rearrangement, and it gives a single FWER threshold for the map. This "requires that the statistic has a distribution that is spatially homogeneous". [fetch] — [Winkler et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)
- TFCE combines the max-statistic approach with the sensitivity of cluster tests to extended effects. [snippet] — [Eelbrain docs](https://eelbrain.readthedocs.io/en/latest/auto_examples/mass-univariate-statistics/permutation-statistics.html)
- Yekutieli 2008 tests on a tree, and it is shown to control FDR only under independence. [snippet] — [Bogomolov et al. 2021](https://arxiv.org/pdf/1705.07529); citation at [RePEc](https://ideas.repec.org/a/bes/jnlasa/v103y2008mmarchp309-316.html)
- TreeBH controls error rates at several levels of resolution. A modified form controls them under any dependency structure. [snippet] — [Bogomolov et al.](https://arxiv.org/pdf/1705.07529)
- FDR control over all levels does not guarantee error control for discoveries within a level. [snippet] — [Bogomolov et al. 2021, Biometrika](https://odin.mdacc.tmc.edu/~cbpeterson/Bogomolov_Peterson_Biometrika_2021.pdf)

Comparison:

| Method | Error rate | Dependence | Cell-level claim | Fit for the grid |
|---|---|---|---|---|
| Bonferroni | FWER | Any | Yes | Valid, conservative |
| Holm | FWER | Any | Yes | Default for small families |
| Max-t sign flip | Strong FWER | Uses the observed one | Yes | Best for cell claims |
| BH | FDR | PRDS, two-sided unclear | Limited | Exploratory map |
| BY | FDR | Any | Limited | Check on BH |
| Cluster permutation | Weak FWER | Uses the observed one | No | Global "any effect" test |
| TreeBH | FDR at each level | Modified form: any | By level | Set, then position, then layer |

### Inferences

Bonferroni cost with n = 197 NTAs (df = 196). I computed these with SciPy. Detectable d is the mean divided by the SD of NTA scores, at 80% power, with a normal approximation:

| Family | Tests | alpha | t cutoff | Detectable d | Pearson r cutoff |
|---|---|---|---|---|---|
| One test | 1 | 0.05 | 1.972 | 0.20 | 0.14 |
| Positions | 25 | 0.002 | 3.132 | 0.28 | 0.22 |
| Grid, 8 layers | 200 | 2.5e-4 | 3.731 | 0.33 | 0.26 |
| Grid, 15 layers | 375 | 1.33e-4 | 3.898 | 0.34 | 0.27 |
| 8-layer grid, 10 sets | 2,000 | 2.5e-5 | 4.318 | 0.37 | 0.30 |
| 15-layer grid, 10 sets | 3,750 | 1.33e-5 | 4.468 | 0.38 | 0.31 |
| One cell, all words | 50,000 | 1e-6 | 5.052 | 0.42 | 0.34 |
| 15-layer grid, all words | 18.75 million | 2.67e-9 | 6.239 | 0.50 | 0.41 |

- The cost is mild at set level. A move from 1 test to 3,750 tests raises the detectable d from 0.20 to 0.38. So Bonferroni or Holm at set level is affordable, and it fits the project rule.
- The cost is larger at word level, and word-level p values on a grid are less useful in any case. Test sets first. Show single words only as a description within a significant set.
- BY factor: the harmonic sum is 6.51 for 375 tests and 8.81 for 3,750 tests.
- Cells near each other share signal. Layers are in a true order, so adjacency has meaning across layers. Token positions are in order, but adjacent tokens can have very different roles. A cluster over positions has a weak rationale. Thus cluster-forming over the layer axis is more defensible than over the position axis. This is my judgment.
- Question (a) asks about "a position and layer". That is a cell claim. A cluster test cannot answer it. Max-t can.
- Max-t procedure:
  1. For each cell, compute the t statistic over NTA scores. Use t, not the raw mean, to meet the homogeneity condition.
  2. For each of about 10,000 rearrangements, flip the sign of each NTA across all cells at once.
  3. Keep the largest absolute t over the grid.
  4. Use the 95th percentile as the cutoff for all cells.
- The same flip for all cells keeps the correlation between cells. So max-t is less strict than Bonferroni when cells correlate.
- Sign flip needs independence and symmetry (see the next section). Compare the cutoff with the Bonferroni t. If they are near each other, report Holm, because it is simpler.
- Define the family before the analysis. A sensible plan: one family for each preset set, a Holm or max-t correction inside the grid, and a Bonferroni correction across the number of preset sets.
- If there are about 10 preset sets, a simple hierarchy works without TreeBH. First test each set with one summary score, with Bonferroni across sets. Then test cells only inside the sets that pass, with the grid correction at the level of that set. Confirm the error properties before you claim FWER for this gate. TreeBH is the published form, if you use FDR.
- For effect size in a surviving cell, use held-out data. Selection by a corrected test still inflates the estimate.

### Gaps

- I could not read the original text of Maris and Oostenveld 2007, Holm 1979, Benjamini and Hochberg 1995, Nichols and Holmes 2002, Smith and Nichols 2009 or Yekutieli 2008. Claims are from secondary sources, or from abstracts.
- I found no paper on correction across a position-by-layer grid of a language model. The neuroimaging analogy is my own.
- I did not check whether a position-by-layer grid meets the smoothness that cluster methods expect. You can measure this from the correlation of t values between adjacent cells.
- I did not verify effective-number-of-tests methods (for example Nyholt 2004, Li and Ji 2005). I make no claim on them.
- The Rousselet simulation studies onset estimates in EEG, not grids of model readouts. Its numbers do not transfer directly.

## Clustered data: cluster-robust errors, NTA means, mixed models and permutation within strata

### Takeaway

At 197 balanced clusters, the few-cluster fault does not apply. For a variable at NTA level, a t test on NTA means is equal to the balanced nested model, so a mixed model adds nothing. For a variable that varies within an NTA, use OLS with NTA-clustered errors, or a mixed model, because NTA means discard the within-NTA signal.

### Cited Findings

- The variance inflation is about 1 + rho_x * rho_u * (Nbar - 1). This is exact for balanced clusters with rho_x = 1. [read] — [Cameron and Miller 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)
- "There is no specific point at which we need to worry about few clusters. Instead, 'more is better'. Current consensus appears to be that G = 50 is enough for state-year panel data." [read] — [Cameron and Miller 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)
- With a standard cluster variance and critical value 1.96, Bertrand et al. rejection rates were .063, .058, .080 and .115 at G = 50, 20, 10 and 6. [read] — [Cameron and Miller 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)
- There is no clear-cut definition of "few". It "may range from less than 20 to less than 50 clusters in the balanced case". [read] — [Cameron and Miller 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)
- Unbalanced clusters do worse, and recent theory shows a lower effective number of clusters in some cases. [read] — [Cameron and Miller 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)
- With equicorrelated errors, balanced clusters and regressors that do not vary in a cluster, FGLS equals OLS, so there is no efficiency gain. [read] — [Cameron and Miller 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf)
- Murtaugh compares a nested ANOVA and a t test on pond means. "The assumptions behind the two analyses are identical", viz. that the group means are independent and normal with equal variances. The choice is "a matter of stylistic preference". [read] — [Murtaugh 2007](https://www.sfu.ca/~lmgonigl/materials-qm/papers/murtaugh-2007-56.pdf)
- In that example, t = 7.469 and F = 55.782. Note that t squared equals F, up to rounding. [read] — [Murtaugh 2007](https://www.sfu.ca/~lmgonigl/materials-qm/papers/murtaugh-2007-56.pdf)
- The CLT supports the t test on means, as means of 48 observations are near normal. [read] — [Murtaugh 2007](https://www.sfu.ca/~lmgonigl/materials-qm/papers/murtaugh-2007-56.pdf)
- Exchangeable errors means that the joint distribution does not change under permutation. [fetch] — [Winkler et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)
- Sign flipping needs symmetric errors and independence. Independence is "required to allow sign flipping of one observation without perturbing others". [fetch] — [Winkler et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)
- Within-block permutation shuffles only inside a block. Whole-block permutation exchanges blocks in their entirety. With ISE, signs flip for all observations in a block at once. [fetch] — [Winkler et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4010955/)
- Multi-level block permutation controlled type I error at the nominal level. Power was only slightly lower than with free permutation. [snippet, author's blog] — [Winkler, Brainder blog](https://brainder.org/2015/12/07/permutation-tests-in-the-human-connectome-project/)
- Goeman and Buhlmann warn that permutation tests are not adequate when subjects do not follow the simple sampling scheme, for example with covariates. [read] — [Goeman and Buhlmann 2007](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL08/Goeman_2007.pdf)

### Inferences

- Verified for this data: G = 197 is about 4 times the G = 50 that the source calls enough, and clusters are balanced at 3 images. Use T(196) for critical values in any case. At this df, T and normal differ little. The source gives no specific result for 197.
- A caveat to "few-cluster limits do not apply": if a group label is rare, the useful clusters are few. If only 12 NTAs hold a label, treat that contrast as a small sample.
- With 3 images per NTA, image-level inflation for a NTA-level variable is 1 + 2 * ICC:

| ICC | Design effect | Effective images (of 591) |
|---|---|---|
| 0.1 | 1.2 | 492 |
| 0.3 | 1.6 | 369 |
| 0.5 | 2.0 | 295 |
| 0.8 | 2.6 | 227 |

- If you ignore clusters at ICC 0.5, standard errors are too small by a factor of about 1.41. Measure ICC for your scores and report it.
- Choice by the level of the variable:

| Variable varies at | Preferred | Why |
|---|---|---|
| NTA only | t test on NTA means | Equal to a balanced mixed model, simple |
| Image, inside NTA | OLS, NTA-clustered SE | Keeps the within-NTA signal |
| Image, within contrast | NTA fixed effects, clustered SE | Removes all NTA-level confounds |
| Same images, two conditions | Paired t on NTA mean differences | Differences cancel the image effects |

- A mixed model with a random NTA intercept is valid in all rows, but it assumes the form of the random effects. It adds little over the simple forms, at this G and balance. Use it when you have missing cells, so that the data are not balanced.
- For (c), an image-level margin has between-NTA and within-NTA parts. The two can have different slopes. Whole-block permutation tests association between NTAs. Within-block permutation tests it within NTAs. State which one you claim. If you want the claim "the image, not the neighbourhood", use within-block permutation or NTA fixed effects.
- Sign flip by NTA needs a roughly symmetric distribution of NTA scores. Means of 3 images of a bounded or skewed score may not meet this. A histogram is a sufficient check. The t test also relies on the CLT here, with n = 197, and it is a fine default.
- NTAs are not surely independent. Adjacent NTAs can share a look, a borough or a camera condition. I propose one sensitivity run with clusters at a larger level. With 5 boroughs that is few clusters, so it is only a rough check.
- Images are the sampled unit only if they are a fair draw from the NTA. If a rule selected the 3 images (for example the same scene type), the claim applies to that rule.

### Gaps

- I did not find a source that states the equality of NTA means and mixed model in general form. Murtaugh gives it by example, and his appendix (Ecological Archives E088-003-A1) has the proof, which I did not read.
- I did not check Hubbard et al. on GEE, or simulation guides on minimum clusters for mixed models.
- The search found nothing that confirmed the "rule of 42". I make no claim on it.
- I did not read the Winkler 2015 paper directly. Claims on it are from the author's blog.

## Selective inference and split-sample: how to pick word lists and test them without double dipping

### Takeaway

Preset word lists need no split. Any list or cell chosen from the readouts must be tested on held-out NTAs. Split by NTA, not by image, and fix the split before all selection. A single hold-out with a t test is the simplest valid plan, and its cost is known.

### Cited Findings

- Double dipping is "the use of the same data set for selection and selective analysis". [fetch] — [Kriegeskorte et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/)
- Selection can cause "distortion of estimates and invalidation of statistical tests". [fetch] — [Kriegeskorte et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/)
- The bias stays "even when the statistical criterion is stringent and the ROI highlights a truly activated region". [fetch] — [Kriegeskorte et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/)
- "contrast-vector orthogonality, by itself, is not sufficient to ensure independence". [fetch] — [Kriegeskorte et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/)
- "A simple way to ensure independence is to use independent data for selection and selective analyses". [fetch] — [Kriegeskorte et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/)
- The authors say to begin with a nonselective analysis, and to use independent data for selective analysis unless independence is shown. [fetch] — [Kriegeskorte et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/)
- Fithian et al. describe their proposal as related to data splitting, "but is more powerful". [fetch] — [Fithian et al.](https://arxiv.org/abs/1410.2597)
- Schultheiss et al. report that Fithian et al. "prove that completely discarding the fraction of data used for model selection in the inference stage leads to inadmissible tests". Carving keeps the selection data, except the information that the selection used. [read] — [Schultheiss et al. 2021](https://arxiv.org/pdf/2006.04613)
- A single split is a random process, so "the inference statements change if a different split is chosen". Meinshausen et al. call this the "p-value lottery". The instability is strong for an unstable selector, such as the Lasso. [read] — [Schultheiss et al. 2021](https://arxiv.org/pdf/2006.04613)
- Data carving itself can suffer from the instability of the selector. [read] — [Schultheiss et al. 2021](https://arxiv.org/pdf/2006.04613)

### Inferences

When a split is necessary:

| Item | Source | Split needed |
|---|---|---|
| Word list from an external lexicon, fixed first | Outside the data | No |
| Lexicon that you edit after you see readouts | Data | Yes |
| List from top words of the lens | Data | Yes |
| Words chosen for link with margin | Data | Yes |
| Best position or layer | Data | Yes, for the effect size |
| Control match on model prior | Data | Yes |
| Filter of tokens, or of k, or p of RBO, tuned on results | Data | Yes |
| Lens fit on these images | Data | Yes, keep them out of the fit |

- Split rule: assign whole NTAs to halves. If images of one NTA fall in both halves, shared neighbourhood content leaks across the split, and the test set is not independent. Stratify by borough or by group label, if a group contrast matters. Fix the seed, and record it before you read a result.
- Cost of a 50/50 split, with df from NTAs (98 discovery, 99 test):

| Plan | N | t cutoff | Detectable d |
|---|---|---|---|
| Full sample, one preset test | 197 | 1.972 | 0.20 |
| Hold-out, one test | 99 | 1.984 | 0.28 |
| Hold-out, 5 sets, Bonferroni | 99 | 2.627 | 0.35 |
| Hold-out, 10 sets, Bonferroni | 99 | 2.872 | 0.37 |

- Compare this with no split at the set-by-grid level (d 0.38 at 3,750 tests). So a split with few confirmatory tests has about the same power as a full-sample correction over a big family. It also gives an unbiased effect size. For open discovery, this favours the split.
- Keep the confirmatory family small. Take only a few lists and cells from discovery to test.
- Do the whole chain inside the discovery half: word filter, control match, score choice, p of RBO, k, and any threshold. The test half then gets one locked recipe.
- For (c): in the discovery half, rank words by the partial link with margin, group them, and form lists. In the test half, compute the per-image set score and test it against margin with NTA-clustered errors. A word list "fit" on the same images always shows a link.
- An independent test on the same model supports a claim about images. It does not cover new models or new prompts. Those need their own hold-out.
- The rule of Kriegeskorte et al. on orthogonality applies here. For example, selection on overall margin link and a later test on a group contrast are not automatically independent. Show it, or use a held-out half.
- Carving and multi-split methods have more power, but they are not textbook tests. Under the project rule, stay with one hold-out. Mention carving only if power fails.
- The p-value lottery is a real risk with one split. A fixed seed removes the freedom to shop for a split, but not the luck of the split. Lists from a stable rule (top words by a simple link with margin) vary less than lists from a sparse model. As a check, repeat the discovery step on other splits, and report how much the list changes. Do not use those repeats for the p value.
- Repeat splits give dependent p values, so you cannot apply a t test to them directly. If you use cross-fit scores over K folds, permute the whole chain to get a p value. This is my judgment, and I did not source it.

### Gaps

- I read the inadmissibility claim only through Schultheiss et al. I did not read the Fithian et al. proof. The abstract supports only "more powerful".
- I did not read Cox 1975, Wasserman and Roeder 2009 or Vul et al. 2009. I make no claim from them. I know Meinshausen et al. 2009 only through Schultheiss et al.
- I found no source on split-sample practice for lens readouts or for word set selection in language models.
- I found no source that quantifies leakage between images of one NTA. The rule to split by NTA follows from the cluster structure.
