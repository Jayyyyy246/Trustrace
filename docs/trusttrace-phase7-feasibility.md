# TRUSTTRACE Phase 7: Dataset Feasibility & OCR Coverage Audit

**Document ID:** `TRUSTTRACE-DOC-P7-FEASIBILITY-001`  
**Phase:** 7 — Dense Multi-Scale Saliency Sampler & OCR-Independent Candidate Discovery  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY VALIDATED  

---

## 1. Executive Summary

Phase 6 demonstrated that native-resolution patch transformers achieve strong macro-F1 (0.7941) when evaluated on candidate patches centered on OCR word detections.
However, Phase 6 also revealed that **OCR coverage dependence** is the single largest remaining vulnerability: when an altered number, character, or receipt header is not successfully recognized by the OCR engine, it is never extracted as a candidate patch.

Before building OCR-independent candidate generators, this audit establishes the **exact empirical gap** between official ground-truth forgery annotations and inference-time OCR candidate boxes.

---

## 2. Official Ground-Truth Forgery Annotation Inventory

- **Total Audited Receipts:** 987 (824 REAL, 163 EDITED)
- **Edited Receipts with Valid VIA Annotations:** **162 / 163 (99.4%)**
- **Total Official Ground-Truth Forgery Boxes:** **664**
- **Median Box Width:** 24.0 px
- **Median Box Height:** 34.0 px
- **Median Box Area:** 808.0 px²

---

## 3. Critical OCR-to-Ground-Truth Coverage Analysis

Every official forgery bounding box was evaluated against all OCR candidate bounding boxes (both individual words and synthesized line envelopes):

| IoU Threshold | Covered GT Boxes | Total GT Boxes | Region Recall (%) | OCR-Missed Regions |
|:---:|:---:|:---:|:---:|:---:|
| **IoU $\ge$ 0.10** | 461 | 664 | 69.43% | 203 |
| **IoU $\ge$ 0.25** | 227 | 664 | 34.19% | 437 |
| **IoU $\ge$ 0.50** | 82 | 664 | 12.35% | 582 |

> [!IMPORTANT]
> **The Core Scientific Bottleneck Identified:**  
> At the standard detection overlap threshold ($\text{IoU} \ge 0.25$), OCR candidate extraction captures **34.19%** of official forgery regions, leaving **437 out of 664 ground-truth regions (65.81%) completely uncovered**.
> This rigorously validates the primary hypothesis of Phase 7: an OCR-independent candidate sampler is essential to recover missing forgery evidence.

---

## 4. Coverage Breakdown Across Dataset Partitions

| Partition | Total GT Boxes | Covered @ IoU $\ge$ 0.25 | Recall (%) | OCR-Missed Regions | Zero Leakage? |
|:---|:---:|:---:|:---:|:---:|:---:|
| **TRAIN** | 480 | 161 | 33.5% | 319 | **PASS** |
| **VAL** | 98 | 34 | 34.7% | 64 | **PASS** |
| **TEST** | 86 | 32 | 37.2% | 54 | **PASS** |

---

## 5. Coverage Breakdown by Entity Category (IoU $\ge$ 0.25)

| Entity Category | Total Boxes | Covered by OCR | Recall (%) | OCR-Missed Count | Forensic Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Total/payment** | 315 | 123 | 39.1% | 192 | Critical financial fraud target |
| **Product** | 147 | 57 | 38.8% | 90 | Critical financial fraud target |
| **Metadata** | 132 | 30 | 22.7% | 102 | Critical financial fraud target |
| **Company** | 40 | 12 | 30.0% | 28 | Critical financial fraud target |
| **Other** | 30 | 5 | 16.7% | 25 | Critical financial fraud target |

---

## 6. Feasibility Conclusion & Phase 7 Mandate

1. **Feasibility Confirmed:** The dataset supports rigorous evaluation of candidate discovery because Level A ground-truth annotations exist across all 162 edited receipts.
2. **OCR Gap Quantified:** Over **40% of authentic forgery regions** fail to meet IoU $\ge$ 0.25 alignment under OCR candidate extraction.
3. **Phase 7 Mandate:** Implement a deterministic **Dense Multi-Scale Saliency Sampler** combining morphological gradients, edge density, and local texture response to recover these missed regions without exceeding computational candidate budgets.
