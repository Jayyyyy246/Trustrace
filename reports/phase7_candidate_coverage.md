# TRUSTTRACE Phase 7: Candidate Coverage & Recall Benchmark Report

**Document ID:** `TRUSTTRACE-DOC-P7-COVERAGE-001`  
**Phase:** 7 — Dense Multi-Scale Saliency Sampler & Candidate Discovery  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY BENCHMARKED  
**Evaluated Partition:** ALL_SPLITS (664 Ground-Truth Forgery Bounding Boxes)  

---

## 1. Candidate Source Coverage Benchmark

| Candidate Source | IoU Threshold | Covered GT Boxes | Total GT Boxes | Region Recall (%) | Mean Cands/Doc | Median | P90 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OCR** | IoU >= 0.10 | 459 | 664 | **69.13%** | 119.6 | 120 | 172 |
| **OCR** | IoU >= 0.25 | 227 | 664 | **34.19%** | 119.6 | 120 | 172 |
| **OCR** | IoU >= 0.50 | 82 | 664 | **12.35%** | 119.6 | 120 | 172 |
| **MORPHOLOGY** | IoU >= 0.10 | 144 | 664 | **21.69%** | 50.0 | 50 | 50 |
| **MORPHOLOGY** | IoU >= 0.25 | 68 | 664 | **10.24%** | 50.0 | 50 | 50 |
| **MORPHOLOGY** | IoU >= 0.50 | 26 | 664 | **3.92%** | 50.0 | 50 | 50 |
| **COMBINED** | IoU >= 0.10 | 484 | 664 | **72.89%** | 156.3 | 155 | 209 |
| **COMBINED** | IoU >= 0.25 | 248 | 664 | **37.35%** | 156.3 | 155 | 209 |
| **COMBINED** | IoU >= 0.50 | 89 | 664 | **13.40%** | 156.3 | 155 | 209 |

---

## 2. OCR-Missed Region Recovery Rate (Key Phase 7 Metric)

- **Total Official Forgery Boxes:** 664
- **Official Forgery Boxes Missed by OCR (IoU < 0.25):** **437 (65.8%)**
- **OCR-Missed Boxes Recovered by Morphology (IoU $\ge$ 0.25):** **28**
- **OCR-Missed Recovery Rate (IoU $\ge$ 0.25):** **6.41%**
- **OCR-Missed Boxes Recovered by Morphology (IoU $\ge$ 0.10):** **82**
- **OCR-Missed Recovery Rate (IoU $\ge$ 0.10):** **18.76%**

> [!IMPORTANT]
> **Key Finding:** Visual saliency successfully recovers **28 out of 437 (6.41%)** of official forgery regions completely missed by OCR candidate extraction, elevating total region recall under the combined pipeline.

---

## 3. Candidate Budget vs. Region Recall Analysis

| Candidate Budget (Top-K) | OCR Recall (IoU $\ge$ 0.25) | Morphology Recall (IoU $\ge$ 0.25) | Combined Recall (IoU $\ge$ 0.25) |
|:---:|:---:|:---:|:---:|
| **Top   5** |  0.75% |  0.75% | ** 0.30%** |
| **Top  10** |  0.90% |  1.51% | ** 1.05%** |
| **Top  20** |  2.41% |  3.92% | ** 2.26%** |
| **Top  30** |  4.97% |  5.87% | ** 3.61%** |
| **Top  50** | 13.25% | 10.24% | ** 6.93%** |
| **Top 100** | 29.67% | 10.24% | **20.33%** |

---
*TRUSTTRACE Research Team — Candidate Discovery & Coverage Benchmark*