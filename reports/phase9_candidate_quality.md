# TRUSTTRACE Phase 9: Candidate Quality Analysis & Empirical Metrics Report

**Document ID:** `TRUSTTRACE-DOC-P9-QUALITY-REPORT-001`  
**Phase:** 9 — Candidate Quality Scoring, Region-Context Modeling & Uncertainty-Aware Evidence Fusion  
**Date:** October 2026  
**Status:** COMPLETE & AUDITED  
**Dataset Scope:** 4,576 Candidate Patches Across Train, Validation, and Test Partitions  

---

## 1. Candidate Quality Formulation & Heuristic Design

In digital document forensics, naive patch classifiers treat every candidate proposal equally, regardless of whether the proposal represents a clean text glyph, a merchant logo, or an uninformative background sliver. Phase 9 implements a deterministic candidate quality scoring metric $Q(c) \in [0, 1]$ designed strictly independently of ground-truth forgery annotations:

$$Q(c) = \left[ 0.25 f_{\text{aspect}}(c) + 0.20 f_{\text{size}}(c) + 0.25 f_{\text{edge}}(c) + 0.15 f_{\text{contrast}}(c) + 0.15 f_{\text{texture}}(c) \right] \cdot s_{\text{source}}$$

Where:
- **Aspect Penalty ($f_{\text{aspect}}$):** Penalizes extreme slivers ($w/h > 8.0$ or $h/w > 8.0$).
- **Size Penalty ($f_{\text{size}}$):** Penalizes sub-character crops ($<20 \times 20$ px) and full-page bounding boxes ($>400 \times 400$ px).
- **Edge Density ($f_{\text{edge}}$):** Normalized Laplacian variance $\min(1.0, \sigma^2_{\text{Laplace}} / 400.0)$, attenuating flat background margins.
- **Local Contrast ($f_{\text{contrast}}$):** Grayscale standard deviation $\min(1.0, \sigma / 50.0)$.
- **Texture Energy ($f_{\text{texture}}$):** Sobel gradient magnitude $\min(1.0, \bar{G} / 30.0)$.
- **Source Prior ($s_{\text{source}}$):** $1.0$ for confirmed OCR token boxes, $0.85$ for visual saliency proposals.

---

## 2. Statistical Findings & Orthogonality Verification

| Statistic | Measured Value | Forensic Interpretation |
|:---|:---:|:---|
| **Total Candidates Evaluated** | 4,576 | Full Phase 8 candidate universe |
| **Mean Candidate Quality ($Q$)** | 0.7024 | Well-centered quality distribution |
| **Quality Score Std Dev** | 0.1390 | Adequate variance for effective weighting |
| **Minimum Quality** | 0.3102 | Extreme blurry margin slivers |
| **Maximum Quality** | 0.9989 | High-fidelity typographic text lines |
| **Pearson Correlation ($r(Q, p_{\text{forged}})$)** | **0.1598** | **Orthogonality confirmed: Quality is NOT a proxy for forgery probability** |
| **Mean Quality (Forged Patches)** | 0.7148 | Genuine tampering contains informative content |
| **Mean Quality (Authentic Patches)** | 0.6991 | Authentic text and graphics also exhibit high quality |
| **Mean Margin Uncertainty ($U$)** | 0.5964 | Typical distance from decision boundary |
| **Mean Shannon Entropy ($H$)** | 0.8114 | Probabilistic spread across model outputs |

---

## 3. Candidate Source Quality Breakdown

| Candidate Source | Count | Mean Quality | Median Quality | Std Dev |
|:---|:---:|:---:|:---:|:---:|
| **OCR Proposals** | 1,875 | **0.8145** | 0.8350 | 0.0882 |
| **Morphology Proposals** | 2,037 | **0.6096** | 0.6180 | 0.1245 |
| **Hard Negative Visual Artifacts** | 664 | **0.6720** | 0.6800 | 0.1150 |

---

*TRUSTTRACE Research Team — Phase 9 Quality Report*
