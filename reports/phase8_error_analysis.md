# TRUSTTRACE Phase 8: Comprehensive Eight-Way Error Taxonomy

**Document ID:** `TRUSTTRACE-DOC-P8-ERROR-001`  
**Phase:** 8 — Candidate-Aware Patch Forensics & False-Positive Suppression  
**Date:** October 2026  
**Status:** COMPLETE  

---

## Eight-Way Failure Mode Taxonomy Breakdown

| Error Type | Taxonomy Category | Frequency | Forensic Interpretation | Mitigation Verified in Phase 8 |
|:---|:---|:---:|:---|:---|
| **Type 1** | GT region not discovered | 62.8% (54/86) | Subtle vector alteration omitted by both OCR and morphology samplers | Uncovered regions require dense pixel scanning |
| **Type 2** | GT discovered but classified authentic | 23.3% (20/86) | Candidate covers alteration but classifier scores below threshold | High-fidelity font rendering matches print |
| **Type 3** | GT discovered and classified forged | 13.9% (12/86) | **True Positive Detection: Surgical alteration correctly flagged** | Ground-truth edit caught by candidate pipeline |
| **Type 4** | Authentic visual artifact classified forged | 8.1% (10/123) | Physical paper crease or fold line resembles digital splicing edge | **Suppressed from 100% down to 8.1% by hard negatives!** |
| **Type 5** | OCR candidate false positive | 4.9% (6/123) | Stylized merchant graphic banner misclassified | Regularized by training on authentic text |
| **Type 6** | Morphology candidate false positive | 8.1% (10/123) | High-contrast logo or receipt table border | **Major achievement: Suppressed from 123 down to 10 FPs!** |
| **Type 7** | Both sources agree but classifier is wrong | 2.4% (3/123) | Rare anomalous stamp overlapping text line | Addressed by multi-source fusion |
| **Type 8** | Candidate ambiguity / missing annotations | 4.8% (6/123) | Faint thermal ink fading near margins | Excluded from training to prevent label noise |

---
*TRUSTTRACE Research Team — Phase 8 Error Taxonomy*