# TRUSTTRACE Phase 9: Document-Level Benchmark Report

**Document ID:** `TRUSTTRACE-DOC-P9-DOC-001`  
**Phase:** 9 — Candidate Quality Scoring, Region-Context Modeling & Uncertainty-Aware Evidence Fusion  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY BENCHMARKED  
**Test Partition:** Held-Out Phase 6/7/8/9 Test Set (148 Receipts: 123 REAL, 25 EDITED)  

---

## 1. Document-Level Benchmark Comparison Across Systems (Requirement 34)

| System | Patch Model | Aggregation | EDITED Recall | REAL Specificity | Macro-F1 | ROC-AUC | PR-AUC | ECE |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 6 Frozen Baseline (Top-3 Mean)** | Phase 6 Frozen | Top-K Mean | **100.00%** | **0.00%** | **0.1445** | 0.4771 | 0.1678 | **0.7382** |
| **Phase 8 Compact CNN (Top-3 Mean)** | Compact CNN | Top-K Mean | **20.00%** | **69.11%** | **0.4463** | 0.4231 | 0.1540 | **0.4433** |
| **Phase 8 Doc-PatchFormer (Top-3 Mean)** | Doc-PatchFormer | Top-K Mean | **100.00%** | **11.38%** | **0.2594** | 0.4797 | 0.1650 | **0.6376** |
| **Phase 9 Quality-Weighted Top-3** | Compact CNN | Top-K Mean | **12.00%** | **85.37%** | **0.4852** | 0.4267 | 0.1529 | **0.1808** |
| **Phase 9 Uncertainty-Weighted Top-3** | Compact CNN | Top-K Mean | **12.00%** | **87.80%** | **0.4966** | 0.4010 | 0.1541 | **0.1759** |
| **Phase 9 Spatial-Cluster Aggregation** | Compact CNN | Spatial-Cluster Aggregation | **8.00%** | **81.30%** | **0.4465** | 0.3974 | 0.1493 | **0.1305** |
| **Phase 9 Full Evidence Fusion** | Compact CNN | Full Evidence Fusion | **12.00%** | **87.80%** | **0.4966** | 0.3915 | 0.1518 | **0.1938** |

---

## 2. Confusion Matrices & False-Alarm Suppression ($N=148$ Held-Out Test Receipts)

| Pipeline Configuration | True Negatives (TN / 123) | False Positives (FP) | False Negatives (FN / 25) | True Positives (TP) | Rescued FP Receipts |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Phase 6 Frozen Baseline (Top-3 Mean)** | 0 (0.0%) | **123** | 0 | 25 (100.0%) | **0** |
| **Phase 8 Compact CNN (Top-3 Mean)** | 85 (69.1%) | **38** | 20 | 5 (20.0%) | **0** |
| **Phase 8 Doc-PatchFormer (Top-3 Mean)** | 14 (11.4%) | **109** | 0 | 25 (100.0%) | **0** |
| **Phase 9 Quality-Weighted Top-3** | 105 (85.4%) | **18** | 22 | 3 (12.0%) | **20** |
| **Phase 9 Uncertainty-Weighted Top-3** | 108 (87.8%) | **15** | 22 | 3 (12.0%) | **23** |
| **Phase 9 Spatial-Cluster Aggregation** | 100 (81.3%) | **23** | 23 | 2 (8.0%) | **15** |
| **Phase 9 Full Evidence Fusion** | 108 (87.8%) | **15** | 22 | 3 (12.0%) | **23** |

---

## 3. Region-Level Forgery Ground-Truth Evaluation

- **Total Official VIA Forgery Boxes on Test Partition:** 86
- **Stage 1 (Candidate Discovery Recall):** 0 (0.0%)
- **Stage 2 (Conditional Classification Success):** 0 (0.0%)
- **Stage 3 (Weighted Region Influence):** 0 (0.0%)

---
*TRUSTTRACE Research Team — Phase 9 Document-Level Benchmark*