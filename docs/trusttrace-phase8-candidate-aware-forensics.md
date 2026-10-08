# TRUSTTRACE Phase 8: Candidate-Aware Patch Forensics & False-Positive Suppression

**Document ID:** `TRUSTTRACE-DOC-P8-METHOD-001`  
**Phase:** 8 — Candidate-Aware Patch Forensics & False-Positive Suppression  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY BENCHMARKED  
**Target Checkpoint:** [`models/phase8_candidate_aware_docpatchformer_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/phase8_candidate_aware_docpatchformer_best.pt)  
**Evaluated Dataset:** *Find it again! — Receipt Dataset for Document Forgery Detection* (987 receipts, 664 official forgery bboxes)  

---

## 1. The Core Scientific Dilemma

In forensic document verification, the transition from OCR-dependent patch extraction to unconstrained visual saliency candidate generation creates an acute **precision/recall dilemma**:

- **Phase 6 (OCR-Only Anchors):** High specificity ($95.12\%$) and clean macro-F1 ($0.7941$), but severely bounded sensitivity ($34.19\%$ GT region coverage), completely missing manipulations where numerals were skipped or degraded by OCR.
- **Phase 7 (Raw Visual Saliency):** High sensitivity ($100\%$ EDITED document recall, recovering $18.76\%$ of OCR-missed regions), but total false-positive collapse ($16.89\%$ accuracy, $0\%$ specificity), as legitimate paper folds, merchant logos, and table lines were misclassified as digital forgeries.

**Phase 8 Core Principle:**  
> *"Do not confuse finding something visually suspicious with proving something is digitally forged."*  
> An effective forensic architecture cannot treat raw visual saliency as synonymous with digital tampering. The patch classifier must learn the subtle, fine-grained distinction between natural high-frequency print/paper artifacts and true digital manipulation.

---

## 2. Hard-Negative Visual Taxonomy

Phase 8 constructs an explicit curriculum of **hard authentic visual negatives** sourced from verified authentic receipts:

1. **Merchant Graphic Logos & Stylized Banners:**
   - *Visual Property:* Dense, high-contrast, non-standard typography with heavy ink application.
   - *Phase 7 Failure Mode:* Saliency samplers placed high-scoring candidate boxes over logos; the frozen Phase 6 classifier interpreted non-standard font contours as splicing boundaries.
   - *Phase 8 Mitigation:* 400+ authentic logo crops explicitly labeled `AUTHENTIC_REGION`.
2. **Paper Folds, Hard Creases, and Scanning Shadows:**
   - *Visual Property:* Sharp linear intensity discontinuities running vertically or diagonally across receipts.
   - *Phase 7 Failure Mode:* Linear edge gradients resembled copy-move box edges or splicing seams.
   - *Phase 8 Mitigation:* Saliency candidates centered on creases from authentic receipts included in training with negative labels.
3. **Table Grid Dividers and Receipt Rules:**
   - *Visual Property:* Elongated horizontal rules with high aspect ratios ($w/h > 8.0$).
   - *Phase 7 Failure Mode:* High edge density triggered false alarms.
   - *Phase 8 Mitigation:* Mined rules and list boundaries trained with positive suppression.
4. **Thermal Print Fading & Paper Texture Noise:**
   - *Visual Property:* Speckled, jagged character edges resulting from low thermal head voltage.
   - *Phase 7 Failure Mode:* Discontinuous ink edges mimicked lossy compression interpolation artifacts.
   - *Phase 8 Mitigation:* Thermal print background samples added to negative training pools.

---

## 3. Candidate-Aware Architecture & Domain Calibration

### 3.1 Model Comparison: CNN vs. Doc-PatchFormer
To determine whether false-positive suppression requires transformer attention or merely appropriate training data:
- **Baseline CNN (`models/phase8_patch_cnn_best.pt`):** Compact 3-stage convolutional network ($93,954$ parameters) utilizing local $3 \times 3$ receptive fields.
- **Candidate-Aware Doc-PatchFormer (`models/phase8_candidate_aware_docpatchformer_best.pt`):** 2-stage CNN tokenizer yielding 64 spatial tokens ($d=128$), coupled with a 2-layer Transformer Encoder ($4$ heads, $d_{\text{ff}}=256$, $408,642$ parameters).

### 3.2 Provenance-Agnostic Representation
While candidate provenance (`OCR` vs `MORPHOLOGY`) is recorded as metadata for diagnostic reporting, the primary classification backbone remains **source-agnostic**. The model is forced to evaluate raw optical stroke and texture physics rather than relying on source metadata shortcuts.

---

## 4. Specificity-First Document-Level Decision Calibration

Because digital forensics prioritizes actionable legal integrity, false accusations against genuine receipts must be strictly bounded.

In [`scripts/evaluate_phase8_document_pipeline.py`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/scripts/evaluate_phase8_document_pipeline.py), the decision boundary was calibrated via a rigorous validation sweep:
- **Decision Pooling:** $\text{Top-3 Mean}$ aggregation over candidate patch probabilities.
- **Operating Threshold:** $\tau = 0.65$ (calibrated on Validation set).
- **Result:** Drastic false-positive reduction from $121$ down to $10$ on the test set, restoring REAL specificity above $90\%$ while retaining high sensitivity on genuine document tampering.

---
*TRUSTTRACE Research Team — Phase 8 Candidate-Aware Forensics*
