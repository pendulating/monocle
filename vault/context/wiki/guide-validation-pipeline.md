---
title: "Guide — Validation Pipeline for City Stakeholders"
category: guide
created: 2026-04-07
updated: 2026-04-07
tags:
  - guide
  - evaluation
  - validation
  - pseudo-labeling
  - annotation
  - stakeholder
  - urbanembed
---

# Guide — Validation Pipeline for City Stakeholders

How to rigorously evaluate an embedding-based retrieval + pseudo-labeling pipeline for urban infrastructure detection (e.g., scaffolding), producing evidence that satisfies government stakeholders on both precision and recall.

## The Validation Challenge

We use Qwen3-VL-Embedding-8B cosine similarity thresholds to pseudo-label ~85K scaffolding images from 1M+ street-level photos (see [[concept-embedding-thresholds]]). A downstream CNN is trained on these pseudo-labels. The city needs statistical guarantees that this system works before deploying it.

Key tensions:
- **Pseudo-labels are noisy** — embedding similarity is a proxy for relevance, not ground truth
- **Class imbalance** — scaffolding appears in ~5–10% of images; false positive rate matters enormously
- **Geographic equity** — the model must perform equally across all neighborhoods
- **Stakeholder trust** — city officials need interpretable metrics, not just "mAP = 0.82"

## Phase 1: Establish Human Baseline (500–1000 images)

### Why

Human inter-annotator agreement sets the ceiling for model performance. If annotators only agree 85% of the time on "is there scaffolding?", the model cannot be expected to exceed that. This reframes the conversation from "is the model perfect?" to "does it match human-level judgment?"

### Method

1. **Sample 500–1000 images** stratified across:
   - Score bands (high/medium/low cosine similarity — ensures coverage of boundary cases)
   - Geographic regions (neighborhoods, boroughs)
   - Image conditions (lighting, occlusion, season)
2. **3–5 independent annotators** per image, using clear labeling guidelines:
   - Define scaffolding categories: full sidewalk shed, partial scaffold, construction netting, temporary structures
   - Define edge cases: scaffolding partially visible, distant, occluded
   - Use binary label (scaffolding present: yes/no) plus optional severity/type tags
3. **Measure inter-annotator agreement**:
   - Cohen's kappa (2 raters): κ > 0.80 = strong agreement
   - Fleiss' kappa (3+ raters): target κ ≥ 0.75 for high-stakes applications
   - Document disagreement patterns — these predict where the model will struggle
4. **Resolve disagreements** via majority vote (3/5 or 2/3) for the gold-standard label

### Sources

- **Assessing Inter-Annotator Agreement** — Cohen's/Fleiss' kappa methodology with recommended thresholds for high-stakes annotation. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10062409/)
- **How Reliable Are Annotations via Crowdsourcing?** — Majority-vote consensus from multiple non-experts produces quality comparable to expert annotators. [ACM DL](https://dl.acm.org/doi/10.1145/1743384.1743478)
- **The Truth About Ground Truth** — Human annotations themselves contain 3–15% noise in geospatial tasks; validation methodology must account for annotator uncertainty. [IEEE](https://ieeexplore.ieee.org/document/8898003)
- **Thinking Like an Annotator** — Clear labeling instructions generated through model analysis improve annotator consistency. [arXiv:2306.14035](https://arxiv.org/abs/2306.14035)

## Phase 2: Pseudo-Label Quality Audit (1000–2000 images)

### Why

Before training a CNN on ~85K pseudo-labeled images, measure how accurate those pseudo-labels actually are. This tells us the noise floor of our training data.

### Method

1. **TREC-style pooling**: sample images from multiple retrieval thresholds (2σ, 2.5σ, 3σ, 3.5σ, 4σ) — not just the top results
2. **Human-annotate** the pooled sample (using Phase 1 annotators + guidelines)
3. **Compute per-threshold precision**:
   - "Of images with cosine sim ≥ 3σ, what % actually contain scaffolding?"
   - "Of images below 0.5σ (negative pseudo-labels), what % are truly negative?"
4. **Plot precision-recall curve** across thresholds — the operating point is where the city's tolerance for false positives meets acceptable recall
5. **Estimate label noise rate** at the chosen threshold — feeds into noise-aware training methods

### Key Metrics to Report

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Precision@threshold | % of pseudo-positives that are correct | ≥ 90% at chosen threshold |
| Negative precision | % of pseudo-negatives that are correct | ≥ 98% |
| Estimated noise rate | % of incorrect pseudo-labels in training set | < 10% |
| Coverage | % of true positives captured at threshold | Report, don't target |

### Sources

- **Evaluation in Information Retrieval** — TREC pooling methodology: pool top-K results from multiple systems/thresholds, judge pooled set, estimate recall. [Stanford IR Book Ch.8](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf)
- **FixMatch** — High-confidence pseudo-label filtering with consistency regularization; achieves 94.93% on CIFAR-10 with 250 labels. Validates that confidence thresholds effectively separate reliable from unreliable pseudo-labels. [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)
- **Self-Training with Noisy Student** — Iterative pseudo-labeling on 300M unlabeled images; noise injection (dropout, augmentation) during student training makes it robust to label noise. [arXiv:1911.04252](https://arxiv.org/abs/1911.04252)
- **In Defense of Pseudo-Labeling** — Uncertainty-aware filtering beyond raw confidence scores; ensemble disagreement identifies truly reliable pseudo-labels. [OpenReview](https://openreview.net/forum?id=-ODN6SbiUU)

## Phase 3: Stratified Test Set Construction (2000–5000 images)

### Why

The test set must be representative of the full deployment distribution. A biased test set produces misleading metrics.

### Method

1. **Define strata** — at minimum:
   - Geographic: community districts or zip codes (ensures all neighborhoods represented)
   - Temporal: different months/seasons (scaffolding appearance varies)
   - Image quality: resolution, lighting, occlusion levels
   - Class balance: oversample the minority class (scaffolding) to ≥30% of test set for statistical power
2. **Sample size calculation** — for 95% CI of ±3% on precision/recall with 10% prevalence: need ~385 positive and ~385 negative examples per stratum. Practically: 2000–5000 total images.
3. **Triple-annotate** the entire test set with majority vote resolution
4. **Never use test set images for training or threshold selection** — strict held-out discipline

### Sample Size Guidance

For a binary classifier with prevalence p and desired confidence interval width w at confidence level 1-α:

```
n ≥ (z² × p × (1-p)) / w²

Example: p=0.10 (10% scaffolding), w=0.03 (±3%), z=1.96 (95% CI)
n ≥ (1.96² × 0.10 × 0.90) / 0.03² = 384 per class minimum
```

### Sources

- **Sample Size Analysis for Machine Learning Clinical Validation (SSAML)** — Formal method for computing sample sizes for ML validation with desired confidence intervals. Key finding: many published evaluations use sample sizes too small for meaningful confidence intervals. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10045793/)
- **NIST AI 800-3: Expanding the AI Evaluation Toolbox** — Uses Generalized Linear Mixed Models to properly quantify uncertainty in benchmark evaluations; distinguishes benchmark accuracy from generalized accuracy. [NIST](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf)
- **Spatio-Temporal Stratified Sampling** — Divide geographic area into strata, sample proportionally. Reduces spatial autocorrelation bias. [MDPI](https://www.mdpi.com/2220-9964/11/8/451)

## Phase 4: Model Evaluation

### Metrics

Report a suite of metrics — different stakeholders care about different things:

| Metric | Audience | Interpretation |
|--------|----------|----------------|
| **Precision** | City inspectors | "When the model says scaffolding, how often is it right?" |
| **Recall** | Policy planners | "Of all scaffolding, how much does the model find?" |
| **F1** | Technical team | Harmonic mean of precision and recall |
| **F0.5** | Compliance officers | Precision-weighted; penalizes false alarms |
| **F2** | Safety officers | Recall-weighted; penalizes missed scaffolding |
| **Precision@K** | Report consumers | "Of the top 1000 detections, how many are correct?" |
| **mAP** | ML benchmarking | Area under precision-recall curve |

### Confidence Intervals

Report all metrics with 95% confidence intervals. For precision p on n samples:

```
CI = p ± z × sqrt(p(1-p)/n)
```

Example: "Precision = 92% ± 3% (95% CI, n=400)"

### Disaggregated Evaluation

Report metrics **per stratum** (neighborhood, image condition, time period). This catches geographic bias:

| Neighborhood | Precision | Recall | F1 | n |
|-------------|-----------|--------|----|---|
| Midtown | 94% ± 3% | 88% ± 4% | 0.91 | 120 |
| Harlem | 91% ± 4% | 85% ± 5% | 0.88 | 95 |
| LES | 93% ± 3% | 87% ± 4% | 0.90 | 110 |

If any neighborhood drops significantly, investigate and document why.

### Sources

- **Fingerprinting NYC's Scaffolding Problem with Longitudinal Dashcam Data** — Trained YOLOv7 on 2,214 annotated scaffolding images. Achieved 78% recall, 79% precision. Cross-validated against NYC Department of Buildings permit database as external ground truth. Identified 529 unpermitted structures (9.3% false positive rate). Temporal confirmation (≥6 detections in 80-ft grid) improves precision. [arXiv:2402.06801](https://arxiv.org/abs/2402.06801)
- **Robust CV-Based Construction Site Detection** — 88.56% static accuracy, 87.26% dynamic accuracy. Geographic variation: 75–100% by angle/distance. [arXiv:2503.04139](https://arxiv.org/abs/2503.04139)
- **Reporting Classifier Performance with Confidence Intervals** — Practical guide to computing and reporting CIs on classification metrics. [ML Mastery](https://machinelearningmastery.com/report-classifier-performance-confidence-intervals/)

## Phase 5: Calibration and Reliability Guarantees

### Why

City stakeholders need statements like: "When the model reports high confidence, it is correct X% of the time." Raw model scores are not probabilities.

### Method: Conformal Prediction

Conformal prediction provides **distribution-free, finite-sample coverage guarantees** — no assumptions about the model or data distribution.

1. Hold out a calibration set (separate from test set)
2. Compute nonconformity scores (e.g., 1 - model_confidence) on calibration set
3. Set threshold at the (1-α) quantile of calibration scores
4. Guarantee: on future data, the prediction set contains the true label with probability ≥ 1-α

**Stakeholder-friendly output**: "Our model's high-confidence scaffolding detections are correct at least 95% of the time (validated on 500 held-out images, distribution-free guarantee)."

### Method: Platt Scaling

Post-hoc calibration that transforms raw scores into calibrated probabilities:

1. Fit a logistic regression on validation set: P(correct) = σ(a × score + b)
2. After calibration: "score = 0.90 means 90% chance of being correct"

### Sources

- **A Gentle Introduction to Conformal Prediction** — Angelopoulos & Bates (2021). Distribution-free, non-asymptotic guarantees without model assumptions. Ideal for high-stakes applications where stakeholders need interpretable guarantees. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- **Reliable Decisions with Threshold Calibration** — Calibrate decision thresholds on validation set; report precision@threshold and recall@threshold with confidence intervals. [Paper](https://roshni714.github.io/papers/sahoo2021reliable.pdf)

## Phase 6: External Validation

### Cross-Reference Against City Records

The strongest validation is comparison against an independent data source:

- **NYC DOB Scaffold Permits** — The Department of Buildings maintains a permit database. Cross-reference model detections against known permit locations.
  - True positives: model detects scaffolding at permitted locations
  - Model-only detections: potential unpermitted scaffolding (value-add for the city!)
  - Permit-only locations: model misses (recall failures)

This was exactly the approach used in the NYC dashcam scaffolding paper (arXiv:2402.06801), which found 529 unpermitted structures — turning a validation exercise into a policy-relevant finding.

### Temporal Consistency

Multiple detections of the same structure across time increases confidence:

- If scaffolding is detected at the same GPS location across 3+ captures → high confidence
- Single-capture detections → flag for human review

## Phase 7: Stakeholder Deliverables

### Required Documentation

| Document | Content | Source |
|----------|---------|--------|
| **Model Card** | Architecture, training data, intended use, limitations, disaggregated metrics | [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993) |
| **Datasheet** | Data sources, collection methods, annotation protocol, known biases | [Gebru et al. 2021](https://arxiv.org/abs/1803.09010) |
| **Evaluation Report** | Full metrics with CIs, per-stratum breakdown, failure analysis | NIST AI 800-3 |
| **Failure Mode Analysis** | Confusion matrix, systematic error patterns, edge case gallery | [FMEA for ML](https://arxiv.org/abs/1911.11034) |

### Presenting to City Officials

Frame results in operational terms:

- **Not**: "mAP = 0.84, F1 = 0.87"
- **Instead**: "Of every 100 locations the system flags as having scaffolding, 92 actually do (precision). Of every 100 actual scaffolding locations, the system catches 85 (recall). This performance is consistent across all 59 community districts."

Include a **failure gallery** — showing the hardest cases the model gets wrong builds trust by demonstrating transparency, not weakness.

### Sources

- **NIST AI Risk Management Framework (AI 100-1)** — Foundational framework for trustworthy AI covering accuracy, explainability, fairness, robustness. [NIST](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf)
- **Model Cards for Model Reporting** — "Nutrition labels" for ML models; documents disaggregated evaluation across subgroups. [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)
- **Data Cards** — Essential facts about ML datasets for stakeholders across the dataset lifecycle. [ACM](https://dl.acm.org/doi/fullHtml/10.1145/3531146.3533231)
- **Urban Visual Intelligence** — Survey of studying cities with AI and street-level imagery; covers validation approaches for municipal applications. [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/24694452.2024.2313515)
- **Municipal AI Integration** — 8-phase framework emphasizing stakeholder involvement, transparency, and accountability throughout. [Springer](https://link.springer.com/article/10.1007/s44243-025-00056-3)

## Active Learning for Annotation Efficiency

To minimize annotation cost while maximizing test set quality:

1. **Uncertainty sampling**: prioritize images near the decision boundary (cosine sim ≈ threshold ± margin) where the model is least certain
2. **Diversity sampling**: ensure annotated images span the full geographic and visual diversity
3. **Expected error reduction**: select images that would most reduce model uncertainty if labeled

This can reduce required annotations by 50–70% while maintaining evaluation quality.

### Sources

- **Active Decision Boundary Annotation** — Prioritize annotating examples near the decision boundary; reduces annotation cost by 50–70% while maintaining performance. [arXiv:1703.06971](https://arxiv.org/abs/1703.06971)
- **Annotation Cost Efficient Active Learning for Content-Based Image Retrieval** — Active learning specifically for image retrieval; queries annotators on ambiguous pairs. [arXiv:2306.11605](https://arxiv.org/abs/2306.11605)

## Geographic Equity Considerations

City governments are increasingly sensitive to algorithmic bias across neighborhoods. Document and test for this explicitly.

- **Spatial bias in training data**: street-level imagery coverage varies by neighborhood; ensure training data isn't concentrated in wealthy areas
- **Performance disparities**: scaffolding may look different in different neighborhoods (building types, materials, density); disaggregate metrics geographically
- **Mapillary coverage analysis**: crowdsourced imagery over-represents urban, politically influential areas. [Coverage & Bias of Street View Imagery](https://arxiv.org/abs/2409.15386)
- **Equity framing**: present geographic equity analysis as a feature, not a compliance burden — "our system ensures equal monitoring quality across all communities"

### Sources

- **Urban Visual Intelligence** — Documents how CV algorithms can "render data-poor, marginalized geographies algorithmically invisible." [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/24694452.2024.2313515)
- **Reducing Social Inequity of Neighborhood Visual Environment** — Analyzes how street-view CV disparities affect communities in Los Angeles. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11781154/)

## Related Pages

- [[concept-embedding-thresholds]] — Threshold selection methods and empirical analysis
- [[urban-embed]] — Embedding pipeline and rerank stage
- [[guide-browser-search]] — Browser-based search interface
