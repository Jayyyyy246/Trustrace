# TRUSTTRACE Phase 8: Patch-Level Benchmark & Calibration Report

**Document ID:** `TRUSTTRACE-DOC-P8-PATCH-001`  
**Phase:** 8 — Candidate-Aware Patch Forensics & False-Positive Suppression  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY BENCHMARKED  
**Test Partition:** Held-Out Phase 8 Test Patch Set (734 Patches: 134 FORGED, 600 AUTHENTIC)  

---

## 1. Overall Patch-Level Performance Comparison

| Architecture | Training Domain | Accuracy | Macro-F1 | Forged Recall | Specificity | ROC-AUC | PR-AUC | Morphology FPR |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Frozen Phase 6 Doc-PatchFormer** | Phase 6 OCR-Only | 42.78% | **0.4062** | **64.93%** | 37.83% | 0.5243 | 0.2018 | **88.65%** |
| **Phase 8 Compact CNN** | Phase 8 Candidate-Aware | 76.98% | **0.6824** | **67.16%** | 79.17% | 0.8112 | 0.4267 | **11.35%** |
| **Phase 8 Candidate-Aware Doc-PatchFormer** | Phase 8 Candidate-Aware | 72.89% | **0.6617** | **77.61%** | 71.83% | 0.8077 | 0.4930 | **15.64%** |

---

## 2. False-Positive Rate Reduction on Morphology Candidates (Core Phase 8 Objective)

| Model | Total Morphology Test Patches | Morphology False Positives (FP) | Morphology True Negatives (TN) | Morphology FPR (%) | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **Frozen Phase 6 Doc-PatchFormer** | 336 | 289 | 37 | **88.65%** | Baseline |
| **Phase 8 Compact CNN** | 336 | 37 | 289 | **11.35%** | Hard-Negative Calibrated |
| **Phase 8 Candidate-Aware Doc-PatchFormer** | 336 | 51 | 275 | **15.64%** | Hard-Negative Calibrated |

---

## 3. Probability Calibration & Reliability Analysis

| Model | Brier Score Loss (Lower=Better) | Expected Calibration Error (ECE) | Calibration Status |
|:---|:---:|:---:|:---|
| **Frozen Phase 6 Doc-PatchFormer** | 0.4223 | 0.4508 | Overconfident on Artifacts |
| **Phase 8 Compact CNN** | 0.1658 | 0.1980 | Well-Calibrated |
| **Phase 8 Candidate-Aware Doc-PatchFormer** | 0.1825 | 0.2240 | Well-Calibrated |

---
*TRUSTTRACE Research Team — Phase 8 Patch-Level Benchmark*