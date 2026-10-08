# TRUSTTRACE Phase 7: Document-Level Test Benchmark Report

**Document ID:** `TRUSTTRACE-DOC-P7-TEST-001`  
**Phase:** 7 — Dense Multi-Scale Saliency Sampler & Candidate Discovery  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY BENCHMARKED  
**Evaluated Model:** Frozen Phase 6 Doc-PatchFormer (`models/doc_patchformer_best.pt`, 408,642 params)  
**Test Partition:** Held-Out Phase 6 Test Set (148 Receipts: 123 REAL, 25 EDITED)  

---

## 1. Document-Level Benchmark Comparison Across Candidate Pipelines

| Architecture / Pipeline | Candidate Source | Threshold ($\tau$) | Accuracy | Macro-F1 | EDITED Recall | EDITED Prec | EDITED F1 | ROC-AUC | PR-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | OCR-Only | 0.70 | 79.05% | **0.4714** | **4.00%** | 12.50% | 0.0606 | 0.4725 | 0.1608 |
| **B** | Morphology-Only | 0.70 | 16.89% | **0.1445** | **100.00%** | 16.89% | 0.2890 | 0.4751 | 0.1674 |
| **C** | Combined OCR+Morphology | 0.70 | 18.24% | **0.1622** | **100.00%** | 17.12% | 0.2924 | 0.4920 | 0.1741 |
| *Phase 4 (Global MobileNetV3)* | Global 224x224 | 0.45 | 71.62% | 0.5497 | 32.00% | 24.24% | 0.2759 | 0.5906 | 0.2784 |
| *Phase 5 (Global + OCR Fusion)* | Global 224x224 + OCR | 0.55 | 74.32% | 0.4935 | 20.00% | 16.13% | 0.1786 | 0.5906 | 0.2784 |
| *Phase 6 (Doc-PatchFormer Baseline)* | OCR Word Patches | 0.70 | 89.19% | 0.7941 | 60.00% | 71.43% | 0.6522 | 0.8800 | 0.7607 |

---

## 2. Confusion Matrices Breakdown ($N=148$ Held-Out Test Receipts)

| Pipeline | True Negatives (TN / 123) | False Positives (FP) | False Negatives (FN / 25) | True Positives (TP) | Total Errors |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Pipeline A (OCR-Only)** | 116 (94.3%) | 7 | 24 | 1 (4.0%) | 31 |
| **Pipeline B (Morphology-Only)** | 0 (0.0%) | 123 | 0 | 25 (100.0%) | 123 |
| **Pipeline C (Combined OCR+Morphology)** | 2 (1.6%) | 121 | 0 | 25 (100.0%) | 121 |

---

## 3. Computational Latency & Efficiency

- **Hardware Environment:** Multi-core CPU (Intel/AMD x86_64, Windows 11)
- **Inference Execution Time:** 47.86 seconds total (107.78 ms per document per pipeline)
- **GPU Required?** No. Lightweight inference operates comfortably on CPU.

---
*TRUSTTRACE Research Team — Phase 7 Document-Level Benchmark*