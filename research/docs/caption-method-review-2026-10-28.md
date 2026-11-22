# Caption method review (2026-10-28)

The user asked an advisor (a Fable subagent): "Are we using the best and
most appropriate method to see if the meaning of / utterance of captions by
a model varies in meaningful semantic ways along sociodemographic axes?"
This file keeps the answer. The advisor changed no repository file. Its
scratch checks are in the session scratchpad (`advisor/check1.py` to
`check4.py`) and are not kept.

Status: advice only. The user has not decided on a method.

## 1. The answer

The current method (`f4_captions.py`: the rarefied entropy of the caption
words and hashtags of each NTA, Spearman with the ACS features) measures
lexical diversity, not meaning. It answers a different question: does a
model use a wide or a narrow set of words in an NTA? Keep it as a
homogeneity result, not as the test of semantic variation.

The advisor's read-only checks:

| Check | Result |
|-------|--------|
| Rarefied word entropy against the image count of the NTA | rho -0.46 (all models), -0.52 (gemma-4-12b), -0.55 (qwen3.5-9b): the measure still tracks the NTA size |
| Rarefied word entropy against the caption options in 1 answer | rho -0.66 (gemma-4-12b), -0.49 (gemma-4-31b): the answer format drives the measure |
| gemma-4-31b, % Asian | rho +0.35; +0.24 after a partial correlation on the options |
| qwen3.5-9b, % bachelor's | rho -0.45; -0.44 without place words; -0.46 after the partial |
| Place and restaurant names removed | rho moves by 0.05 or less |

## 2. Threats to validity, ranked

1. The measure (form) does not match the question (meaning).
2. The image content is a mediator: the cuisine, the signs and the
   storefront differ by NTA (Hendricks et al. 2018; Zhao et al. 2021).
3. The NTA size and the answer format are nuisance variables with no control.
4. The NTA is an ecological unit (Robinson 1950). Write "the captions shift
   with the resident mix of the area", not "the model treats people
   differently".
5. The rarefied `k` comes from the smallest NTA with 3 images (llama-3.2-90b
   hashtags: `k` = 3).
6. Neighbour NTAs are alike (spatial correlation), thus the p values are too
   small.

## 3. The recommended methods, simplest first

| Method | What it measures | Test | Cost |
|--------|------------------|------|------|
| A. Coded content categories (for example cozy/charm, upscale, authentic/hidden gem, cheap/value, gritty/hustle, vibrant/energy, culture/heritage, nostalgia), coded by an LLM judge and checked on 200 to 300 human-coded answers | the share of answers in an NTA with each category | Spearman or Pearson with the ACS features, Bonferroni (the current test) | about 1 day |
| B. Method A at the answer level, with controls | 1 answer: holds the category or not | linear probability model or logistic regression on the ACS feature, with cuisine and borough fixed effects, standard errors clustered by NTA (Cameron and Miller 2015) | about 1 day |
| C. Embedding screen | the caption content as a whole | PERMANOVA (Anderson 2001) over NTA centroids of sentence embeddings | an omnibus check only |
| D. Counterfactual cue | a belief about the residents | the same image with a changed neighborhood cue in the prompt; McNemar or paired t | 6 short GPU runs on 1,000 images |

The advisor's demos on the pooled data (a word lexicon, not a validated
coder): 37 of 240 category tests are significant. Examples: "cozy/charm"
rises with % bachelor's (gemma-4-12b, rho +0.51); "vibrant/energy" falls
with income (llama-3.2-90b, rho -0.60); "authentic/hidden gem" falls with
% bachelor's (gemma-4-31b, rho -0.46). With cuisine fixed effects, about 20%
to 25% of each effect goes away; the rest stays.

Only method D can support a claim that a model holds a belief about the
people of an area. Methods A to C show that the output tracks the area.

## 4. The advisor's next step

Do A and B: a category map or 1 LLM-judge pass on the caption answers, the
cuisine join (now in `_f4.image_cuisines`), 1 function in `_f4cap.py` of the
same shape as `entropy_correlations`, and 1 regression table for each model.
Keep the entropy figure with `k` from the NTAs with 20 images or more.
Decide on D for the bias claim of the paper.

## 5. References (found by web search in the review session)

- Anderson (2001), PERMANOVA. https://www.pelagicos.net/MARS6300/readings/Anderson_2001.pdf
- Cameron and Miller (2015), cluster-robust inference. https://jhr.uwpress.org/content/50/2/317
- Gilardi, Alizadeh and Kubli (2023), LLM text annotation. https://pmc.ncbi.nlm.nih.gov/articles/PMC10372638
- Grimmer and Stewart (2013), text as data. https://ideas.repec.org/a/cup/polals/v21y2013i03p267-297_01.html
- Hendricks et al. (2018), bias in captioning. https://openaccess.thecvf.com/content_ECCV_2018/html/Lisa_Anne_Hendricks_Women_also_Snowboard_ECCV_2018_paper.html
- Howard et al. (2024), SocialCounterfactuals. https://openaccess.thecvf.com/content/CVPR2024/html/Howard_SocialCounterfactuals_Probing_and_Mitigating_Intersectional_Social_Biases_in_Vision-Language_Models_CVPR_2024_paper.html
- Legendre and Anderson (1999), distance-based RDA. https://search.r-project.org/CRAN/refmans/vegan/html/dbrda.html
- Legendre and Fortin (2010), Mantel test (secondary source only). https://www.molecularecologist.com/2015/08/05/should-we-use-mantel-tests-in-molecular-ecology/
- Manvi et al. (2024), geographic bias in LLMs. https://proceedings.mlr.press/v235/manvi24a.html
- Monroe, Colaresi and Quinn (2008), log-odds with a Dirichlet prior. https://www.cambridge.org/core/journals/political-analysis/article/fightin-words-lexical-feature-selection-and-evaluation-for-identifying-the-content-of-political-conflict/81B3703230D21620B81EB6E2266C7A66
- Reimers and Gurevych (2019), Sentence-BERT. https://aclanthology.org/D19-1410
- Roberts et al. (2014), structural topic models. https://ideas.repec.org/a/wly/amposc/v58y2014i4p1064-1082.html
- Robinson (1950), ecological correlations. https://www.stats.uwo.ca/faculty/aim/2015/9938/articles/Robinson1950AmericanSociologicalReview.pdf
- Rodriguez, Spirling and Stewart (2023), embedding regression. https://www.cambridge.org/core/journals/american-political-science-review/article/embedding-regression-models-for-contextspecific-description-and-inference/4C90013E5C714C8483ED95CC699022FB
- Zhao, Wang and Russakovsky (2021), racial bias in captioning. https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Understanding_and_Evaluating_Racial_Biases_in_Image_Captioning_ICCV_2021_paper.html
- Ziems et al. (2024), LLMs for computational social science. https://aclanthology.org/2024.cl-1.8
