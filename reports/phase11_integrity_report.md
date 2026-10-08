# TRUSTTRACE Phase 11: Scientific Integrity & Audit Report

**Date:** 2026-10-08  
**Phase:** Phase 11 — Character-Level Glyph Localization & Native-Resolution Micro-Forensics  
**Branch:** `main`  
**Evaluation Benchmark:** Find it again! Receipt Dataset ($N=148$ TEST Receipts: 123 REAL, 25 EDITED)

---

## 1. Frozen Baseline Integrity Audit

The following historical model checkpoints were verified using SHA-256 cryptographic hashes:

| Checkpoint Path | SHA-256 Digest | Status |
| :--- | :--- | :---: |
| `models/doc_patchformer_best.pt` | `9691afb1cd2fd2e2f3d9d5f0e9f6bf91ea9e7da1f2497fc61921f06775f0a202` | **VERIFIED FROZEN** |
| `models/phase8_patch_cnn_best.pt` | `7a44fbcbcc9cd25bc77b7381483ceb8b090e5fc5015ff52b0900bc3deebc164a` | **VERIFIED FROZEN** |
| `models/phase8_candidate_aware_docpatchformer_best.pt` | `8fbe6601f31612d30e37bcbb8134763138b2512f43eb966465e9ecba9656bf3a` | **VERIFIED FROZEN** |

* **Zero Historical Retraining:** No historical Phase 1–10 models were retrained, fine-tuned, or modified.
* **Zero Historical File Modifications:** `git diff` confirms zero lines modified across historical Phase 1–10 files.

---

## 2. Test Set Quarantine & Anti-Contamination Verification

* **Quarantine Enforcement:** The held-out TEST partition ($N=148$ receipts, 86 official VIA forgery annotations) was strictly quarantined.
* **Parameter Determination:** Glyph margin variants (G0, G1, G2, G3), contour aspect ratio bounds, and relational graph thresholds ($\tau_G = 0.35$) were determined strictly on TRAIN and VALIDATION splits.
* **Zero Test Leakage:** No test labels or ground-truth boxes were used for threshold selection, feature engineering, or candidate scoring.

---

## 3. Ground Truth Authenticity

* **Sole Authoritative Ground Truth:** Official VIA ground-truth annotations (`reports/phase7_annotation_coverage.json`) were utilized exclusively.
* **No Synthetic Ground Truth:** No pseudo-labels or OCR transcripts were substituted as ground truth.
* **PRNU Scientific Handling:** PRNU was scientifically assessed as **NOT APPLICABLE** due to flatbed scanner / thermal printer noise lacking camera sensor provenance, avoiding fabricated sensor noise models.

---

## 4. Master Reproduction Verification

All Phase 11 artifacts were checked and verified via:
```bash
python scripts/reproduce_phase11.py --check-only
```
Verification confirmed:
* 5 Required Documentation Files: **PASSED**
* 8 Required Scripts: **PASSED**
* 4 Candidate Manifests (770,656 total rows): **PASSED**
* 8 Metric Reports: **PASSED**
* 10 Research Figures: **PASSED**
* Cryptographic Model Hashes: **PASSED**
