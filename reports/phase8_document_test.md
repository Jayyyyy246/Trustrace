# TRUSTTRACE Phase 8: Document-Level Benchmark Report

**Document ID:** `TRUSTTRACE-DOC-P8-DOC-001`  
**Phase:** 8 — Candidate-Aware Patch Forensics & False-Positive Suppression  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY BENCHMARKED  
**Test Partition:** Held-Out Phase 6/7/8 Test Set (148 Receipts: 123 REAL, 25 EDITED)  
**Operating Decision Threshold:** $\tau = 0.65$ (Pre-calibrated on Validation)  

---

## 1. Document-Level Benchmark Comparison Across Models & Candidate Sources

| Architecture | Candidate Source | Accuracy | Macro-F1 | EDITED Recall | REAL Specificity | EDITED Prec | EDITED F1 | ROC-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 6 Frozen Baseline** | OCR | 75.68% | **0.4565** | **4.00%** | **90.24%** | 7.69% | 0.0526 | 0.4725 |
| **Phase 6 Frozen Baseline** | MORPHOLOGY | 16.89% | **0.1445** | **100.00%** | **0.00%** | 16.89% | 0.2890 | 0.4751 |
| **Phase 6 Frozen Baseline** | COMBINED | 16.89% | **0.1445** | **100.00%** | **0.00%** | 16.89% | 0.2890 | 0.4920 |
| **Phase 8 Compact CNN** | OCR | 65.54% | **0.4973** | **28.00%** | **73.17%** | 17.50% | 0.2154 | 0.5083 |
| **Phase 8 Compact CNN** | MORPHOLOGY | 60.81% | **0.4463** | **20.00%** | **69.11%** | 11.63% | 0.1471 | 0.4228 |
| **Phase 8 Compact CNN** | COMBINED | 60.14% | **0.4534** | **24.00%** | **67.48%** | 13.04% | 0.1690 | 0.4361 |
| **Phase 8 Candidate-Aware Doc-PatchFormer** | OCR | 40.54% | **0.3893** | **72.00%** | **34.15%** | 18.18% | 0.2903 | 0.5008 |
| **Phase 8 Candidate-Aware Doc-PatchFormer** | MORPHOLOGY | 26.35% | **0.2594** | **100.00%** | **11.38%** | 18.66% | 0.3145 | 0.4780 |
| **Phase 8 Candidate-Aware Doc-PatchFormer** | COMBINED | 25.00% | **0.2472** | **92.00%** | **11.38%** | 17.42% | 0.2930 | 0.4738 |

---

## 2. Confusion Matrices Breakdown ($N=148$ Held-Out Test Receipts)

| System Pipeline | True Negatives (TN / 123) | False Positives (FP) | False Negatives (FN / 25) | True Positives (TP) | Total Errors |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Phase 6 Frozen Baseline (OCR)** | 111 (90.2%) | **12** | 24 | 1 (4.0%) | 36 |
| **Phase 6 Frozen Baseline (MORPHOLOGY)** | 0 (0.0%) | **123** | 0 | 25 (100.0%) | 123 |
| **Phase 6 Frozen Baseline (COMBINED)** | 0 (0.0%) | **123** | 0 | 25 (100.0%) | 123 |
| **Phase 8 Compact CNN (OCR)** | 90 (73.2%) | **33** | 18 | 7 (28.0%) | 51 |
| **Phase 8 Compact CNN (MORPHOLOGY)** | 85 (69.1%) | **38** | 20 | 5 (20.0%) | 58 |
| **Phase 8 Compact CNN (COMBINED)** | 83 (67.5%) | **40** | 19 | 6 (24.0%) | 59 |
| **Phase 8 Candidate-Aware Doc-PatchFormer (OCR)** | 42 (34.1%) | **81** | 7 | 18 (72.0%) | 88 |
| **Phase 8 Candidate-Aware Doc-PatchFormer (MORPHOLOGY)** | 14 (11.4%) | **109** | 0 | 25 (100.0%) | 109 |
| **Phase 8 Candidate-Aware Doc-PatchFormer (COMBINED)** | 14 (11.4%) | **109** | 2 | 23 (92.0%) | 111 |

---

## 3. Two-Stage Funnel Decomposition (Candidate Discovery vs Classification)

| Model | Total GT Boxes | Stage 1: Discovered by Candidates | Stage 1 Rate | Stage 2: Correctly Classified | Stage 2 Rate | Final Region Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Phase 6 Frozen Baseline** | 86 | 2 | 2.3% | 2 | **100.0%** | **2.3%** |
| **Phase 8 Compact CNN** | 86 | 2 | 2.3% | 1 | **50.0%** | **1.2%** |
| **Phase 8 Candidate-Aware Doc-PatchFormer** | 86 | 2 | 2.3% | 2 | **100.0%** | **2.3%** |

---
*TRUSTTRACE Research Team — Phase 8 Document-Level Benchmark*