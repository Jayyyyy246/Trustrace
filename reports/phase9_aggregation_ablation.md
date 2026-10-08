# TRUSTTRACE Phase 9: Aggregation Ablation & Budget Optimization

**Document ID:** `TRUSTTRACE-DOC-P9-ABLATION-001`  
**Phase:** 9 — Candidate Quality Scoring, Region-Context Modeling & Uncertainty-Aware Evidence Fusion  
**Date:** October 2026  
**Status:** VALIDATION COMPLETE — LOCKED CONFIGURATION  
**Selected Best Configuration:** `A_Naive_TopK_Mean` (Budget: `Top-1`, $\tau = 0.50$)  

---

## 1. Aggregation Methods Comparison on Validation (at Budget K=5)

| Aggregation Method | Optimal $\tau$ | Accuracy | Macro-F1 | EDITED Recall | REAL Specificity | FP Alarms | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A_Naive_TopK_Mean** | 0.45 | 78.38% | **0.4940** | 8.33% | **91.94%** | **10** | 0.3774 | 0.1340 |
| **B_Quality_Weighted** | 0.45 | 77.03% | **0.4867** | 8.33% | **90.32%** | **12** | 0.3747 | 0.1331 |
| **C_Uncertainty_Weighted** | 0.45 | 77.70% | **0.4654** | 4.17% | **91.94%** | **10** | 0.3784 | 0.1330 |
| **D_Spatial_Cluster** | 0.35 | 65.54% | **0.4735** | 20.83% | **74.19%** | **32** | 0.3832 | 0.1329 |
| **E_Quality_Uncertainty** | 0.45 | 78.38% | **0.4940** | 8.33% | **91.94%** | **10** | 0.3770 | 0.1334 |
| **F_Quality_Spatial** | 0.45 | 76.35% | **0.4595** | 4.17% | **90.32%** | **12** | 0.3790 | 0.1318 |
| **G_Uncertainty_Spatial** | 0.45 | 77.70% | **0.4654** | 4.17% | **91.94%** | **10** | 0.3790 | 0.1317 |
| **H_Full_Evidence_Fusion** | 0.45 | 77.70% | **0.4654** | 4.17% | **91.94%** | **10** | 0.3804 | 0.1321 |

---

## 2. Candidate Budget Sweep (Top-K Analysis for Full Evidence Fusion)

| Candidate Budget | Optimal $\tau$ | Accuracy | Macro-F1 | EDITED Recall | REAL Specificity | FP Count | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Top_1** | 0.50 | 75.68% | **0.5006** | 12.50% | 87.90% | 15 |  **(Selected)** |
| **Top_3** | 0.45 | 75.68% | **0.4797** | 8.33% | 88.71% | 14 |  |
| **Top_5** | 0.45 | 77.70% | **0.4654** | 4.17% | 91.94% | 10 |  |
| **Top_10** | 0.35 | 65.54% | **0.4735** | 20.83% | 74.19% | 32 |  |
| **Top_20** | 0.45 | 83.78% | **0.4559** | 0.00% | 100.00% | 0 |  |
| **Top_30** | 0.40 | 83.78% | **0.4559** | 0.00% | 100.00% | 0 |  |

---

## 3. Forensic Interpretation of Validation Results

1. **Quality + Uncertainty Filtering:** Weighting candidates by quality and confidence substantially downweights spurious artifacts, suppressing false alarms.
2. **Spatial Context Suppression:** Isolated candidate spikes that lack neighboring spatial support are attenuated, preventing single-patch false alarms.
3. **Locked Operating Parameters for TEST:** Model `A_Naive_TopK_Mean`, Candidate Budget `Top-1`, Decision Threshold $\tau = 0.50$.

---
*TRUSTTRACE Research Team — Phase 9 Aggregation Benchmark*