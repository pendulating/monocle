---
title: "Concept — Embedding Similarity Thresholds"
category: concept
created: 2026-04-07
updated: 2026-04-07
tags:
  - concept
  - embedding
  - retrieval
  - threshold
  - pseudo-labeling
  - distillation
  - urbanembed
---

# Concept — Embedding Similarity Thresholds

Methods for finding the decision boundary in embedding space between query-matching and non-matching images, with application to pseudo-labeling for downstream supervised CNN training.

## Problem Statement

Given a query (e.g., "green sidewalk scaffolding") and 1M+ image embeddings from Qwen3-VL-Embedding-8B, we need to determine the cosine similarity threshold that separates **positive** (query-relevant) images from the **background** (irrelevant). This threshold can then be used to generate pseudo-labels for training a lightweight downstream classifier.

## Our Approach: Background Gaussian + Excess Signal

Fit a Gaussian to the bulk of the cosine similarity distribution (p10–p90) to model the background, then identify where observed counts significantly exceed expected counts.

### Empirical Results ("green sidewalk scaffolding" vs 1,038,932 Manhattan images)

| Threshold | σ | Observed | Expected (Gaussian) | Excess ratio | Interpretation |
|-----------|---|----------|---------------------|--------------|----------------|
| 1.0σ | 0.318 | 251K | 165K | 1.5x | Weakly enriched |
| 2.0σ | 0.355 | 138K | 24K | 5.8x | Signal emerging |
| 2.5σ | 0.373 | 106K | 6.5K | 16x | Strong signal |
| **3.0σ** | **0.391** | **85K** | **1.4K** | **60x** | **Clean separation** |
| 3.5σ | 0.409 | 68K | 242 | 282x | Deep signal |
| 4.0σ | 0.427 | 55K | 33 | 1,672x | High confidence |

The inflection point (right tail onset via second derivative) was detected at cosine sim ≈ 0.309 (~0.75σ), where the density transitions from bulk-Gaussian to enriched tail. The practical decision boundary sits at **2.5–3.0σ**, where the observed-to-expected ratio transitions from "moderately enriched" to "almost entirely signal."

### Choosing the Threshold for Pseudo-Labeling

| Use case | Recommended threshold | Rationale |
|----------|----------------------|-----------|
| High-recall pseudo-labels (noisy) | 2.0σ (~138K images) | Maximizes positive examples; ~17% noise |
| Balanced pseudo-labels | 2.5–3.0σ (~85–106K images) | Strong signal-to-noise; manageable set size |
| High-precision pseudo-labels | 3.5σ+ (~68K images) | Nearly pure positives; for clean training sets |
| Negative examples | Below 0.5σ (~650K images) | Deep in the background distribution |

## Literature Grounding

### Foundational Methods

**Background distribution modeling and excess detection** — Our approach mirrors techniques from particle physics (bump hunting) and anomaly detection, where a null hypothesis distribution is fit to data and deviations are quantified in units of σ.

- **Cosine Similarity Knowledge Distillation for Surface Anomaly Detection** (2024). Models normal embeddings with teacher features; uses cosine similarity thresholds to identify anomalies where similarity deviates from the learned background. [Nature Scientific Reports](https://www.nature.com/articles/s41598-024-58409-9)

- **Anomaly Detection by Clustering DINO Embeddings using a Dirichlet Process Mixture** (2025, MICCAI). Non-parametric DPMM fitting of background embedding distribution; anomaly scores as distance from mixture components. Shows that Gaussian assumptions can be relaxed with mixture models. [MICCAI 2025](https://papers.miccai.org/miccai-2025/0070-Paper2425.html)

- **On the Distribution of Cosine Similarity with Application to Biology** — Smith et al. (2023). Derives asymptotic moments of cosine similarity as a function of data covariance structure. Key finding: variance reaches minimum when covariance has equal eigenvalues. Provides theoretical basis for understanding when Gaussian fits to similarity distributions are justified. [arXiv:2310.13994](https://arxiv.org/abs/2310.13994)

### Knee/Elbow Detection

**Finding a "Kneedle" in a Haystack** — Satopaa et al. (2011, IEEE SIMPLEX). The foundational Kneedle algorithm for detecting inflection points in curves by finding local maxima of curvature. Widely used for threshold selection in ML (implemented in Python `kneed` package). Applicable to our score-rank curve to detect where relevance drops off. [Paper](https://raghavan.usc.edu/papers/kneedle-simplex11.pdf)

**Kneeliverse: A Universal Knee-Detection Library** (2025, SoftwareX). Implements multiple knee detection algorithms (Kneedle, Menger, L-method, DFDT, Z-Method) with multi-knee detection. Z-Method specifically designed for multi-peak scenarios relevant to graded relevance. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352711025001281)

### Embedding Geometry

**All-but-the-Top: Simple and Effective Postprocessing for Word Representations** — Mu & Viswanath (2018). Shows embeddings tend to occupy a narrow cone (anisotropy), making cosine similarities artificially concentrated. Proposes removing mean and top principal components to improve isotropy. Critical context: our Gaussian background fit implicitly accounts for this anisotropy by fitting the observed distribution rather than assuming uniform. [arXiv:1702.01417](https://arxiv.org/abs/1702.01417)

**Deep Metric Learning with Spherical Embedding** — Zhang et al. (2020, NeurIPS). Proposes Spherical Embedding Constraint to ensure embeddings fall on the same hypersphere with consistent normalization. Shows that angular losses alone "cannot guarantee embeddings are on the same hypersphere during training." Validates our use of L2-normalized embeddings where cosine similarity = dot product. [arXiv:2011.02785](https://arxiv.org/abs/2011.02785)

**Surpassing Cosine Similarity for Multidimensional Comparisons** (2024). Documents concentration of cosine similarity in high dimensions — random vectors become increasingly perpendicular. This explains why our 4096d embedding similarities are tightly concentrated (σ=0.036 for background) and why small deviations are meaningful. [arXiv:2407.08623](https://arxiv.org/abs/2407.08623)

### Threshold Selection for Pseudo-Labeling

**Relevance Filtering for Embedding-Based Retrieval** (2024). Maps raw cosine similarity to interpretable relevance scores via query-dependent calibration. Shows that raw cosine similarities are hard to interpret across queries; proposes calibration functions for stable thresholding. Directly relevant to our per-query Gaussian fitting approach. [arXiv:2408.04887](https://arxiv.org/abs/2408.04887)

**FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling** (2021, NeurIPS). Proposes dynamic per-class thresholds rather than global fixed thresholds for pseudo-label selection. Key insight: different concepts have different score distributions, so thresholds should be adaptive. Our per-query Gaussian fit naturally provides query-adaptive thresholds. [OpenReview](https://openreview.net/pdf?id=3qMwV98zLIk)

**Vision-Language Pseudo-Labels for Single-Positive Multi-Label Learning** (2024, CVPRW). Directly applies VLM cosine similarity thresholds to generate pseudo-labels for multi-label classification. Demonstrates that embedding similarity thresholds produce reliable training labels when carefully selected. [CVPR Workshop](https://openaccess.thecvf.com/content/CVPR2024W/LIMIT/papers/Xing_Vision-Language_Pseudo-Labels_for_Single-Positive_Multi-Label_Learning_CVPRW_2024_paper.pdf)

**Gaussian Mixture Models for Adaptive Thresholding in Visual Place Recognition** (2025). Shows that place-specific (i.e., query-specific) thresholds based on negative Gaussian mixture statistics significantly outperform global thresholds. Validates our approach of fitting per-query background distributions. [arXiv:2512.09071](https://arxiv.org/abs/2512.09071)

### Distillation from Embeddings to Classifiers

**EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval** — Kim et al. (2023). Distills large dual-encoder models to ~10% size while retaining 95–97% performance. Uses relative geometry among queries/documents rather than absolute scores. Shows that embedding-based pseudo-labels can effectively train compact models. [arXiv:2301.12005](https://arxiv.org/abs/2301.12005)

**VL2Lite: Task-Specific Knowledge Distillation from Large Vision-Language Models to Lightweight Networks** — Jang et al. (2025, CVPR). Distills VLMs (CLIP-like) into lightweight classifiers via visual and linguistic knowledge distillation. Achieves up to 7% classification improvement over direct transfer. [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_VL2Lite_Task-Specific_Knowledge_Distillation_from_Large_Vision-Language_Models_to_Lightweight_CVPR_2025_paper.pdf)

**DAIT: Distillation from Vision-Language Models with Adaptive Intermediate Teacher Transfer** (2026). Addresses task-irrelevant semantics in VLM representations via a trainable intermediate teacher. Achieves 12.6% gains on fine-grained classification. Suggests that raw embedding similarity thresholds can be improved with task-specific adaptation. [arXiv:2603.15166](https://arxiv.org/abs/2603.15166)

### Score Distribution Theory

**Score Distribution Modeling in Information Retrieval** — Arampatzis & Robertson (2011, Information Retrieval). Comprehensive study of modeling retrieval score distributions as normal-exponential mixtures (relevant=normal, non-relevant=exponential). Provides EM estimation methods and robustness analysis. Our Gaussian background fit is a special case of this framework. [Springer](https://link.springer.com/article/10.1007/s10791-010-9145-5)

**Exploration of a Threshold for Similarity Based on Uncertainty in Embedding Space** — Rekabsaz & Lupu (2017, ECIR). Proposes uncertainty-aware thresholds for word embedding similarity using confidence intervals around expected neighbor counts. Directly addresses threshold calibration under embedding uncertainty. [Paper](https://navid-rekabsaz.github.io/papers/ecir17-uncertainty.pdf)

## Recommended Pipeline for Pseudo-Label Generation

Based on the literature and our empirical analysis:

1. **Compute full-corpus cosine similarities** against the query embedding
2. **Fit background Gaussian** to the bulk (p10–p90) of the score distribution
3. **Select threshold at 2.5–3.0σ** for balanced positive pseudo-labels
4. **Select negatives from below 0.5σ** to ensure clean separation
5. **Optionally refine** with cross-encoder reranking on a sample to validate threshold quality
6. **Train downstream CNN** with binary labels: positive (above threshold) vs negative (below 0.5σ), excluding the ambiguous zone between 0.5σ and 2.5σ

The ambiguous zone exclusion follows the margin-based training principle from metric learning literature — training is most effective when positives and negatives are clearly separated, with borderline examples excluded.

## Related Pages

- [[urban-embed]] — Embedding pipeline and rerank stage
- [[guide-browser-search]] — Browser-based search using these embeddings
- [[concept-verification]] — Post-inference answer filtering
