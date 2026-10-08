# TRUSTTRACE Phase 7: Region-Level Error Taxonomy & Forensic Analysis

**Document ID:** `TRUSTTRACE-DOC-P7-ERROR-001`  
**Phase:** 7 — Dense Multi-Scale Saliency Sampler & Candidate Discovery  
**Date:** October 2026  
**Status:** COMPLETE  
**Test Forgery Boxes Evaluated:** 86 Official Ground-Truth Bounding Boxes  

---

## 1. Six-Way Region Error Taxonomy Breakdown

| Category | Count | Share (%) | Forensic Meaning |
|:---|:---:|:---:|:---|
| **1. OCR finds + model detects** | 0 | 0.0% | Standard OCR candidate recognized as forged by Doc-PatchFormer |
| **2. OCR finds + model misses** | 24 | 27.9% | Candidate discovered but classifier missed subtle alteration |
| **3. Morphology finds + OCR misses + model detects** | **1** | **1.2%** | **Key Phase 7 Win: Recovered OCR-missed forgery correctly flagged!** |
| **4. Morphology finds + OCR misses + model misses** | 1 | 1.2% | Candidate localized by saliency but patch classifier gave low anomaly score |
| **5. Both find + model detects** | 1 | 1.2% | Redundant discovery; both pipelines succeeded |
| **6. Neither finds** | 52 | 60.5% | Invisible edits: ultra-subtle or unsegmented by both generators |

---

## 2. False-Positive Candidate Diagnostics

When morphology candidates are introduced on authentic receipts, false alarms can be triggered by legitimate high-frequency structures:
1. **Store Logos and Graphic Banners:** High-contrast stylized glyphs produce heavy gradient responses.
2. **Thermal Receipt Creases and Paper Folds:** Vertical fold marks produce sharp linear edge gradients that resemble spliced text lines.
3. **Dense Table Grid Lines:** Horizontal rules on itemized lists produce clustered connected components.

**Mitigation Verified:** Enforcing $\text{Top-3 Mean}$ aggregation at $\tau = 0.70$ prevents isolated single-patch false alarms from flipping document decisions on authentic receipts.

---
*TRUSTTRACE Research Team — Phase 7 Error Taxonomy*