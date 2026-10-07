# TRUSTTRACE Phase 5: Design Review & Cross-Domain Compatibility Analysis

**Document ID:** `TRUSTTRACE-DOC-P5-DESIGN-001`  
**Phase:** 5 — Evidence Fusion & Forensic Analysis  
**Date:** October 2026  
**Status:** COMPLETE & APPROVED  
**Review Target:** Forensic Models, Datasets, and Fusion Feasibility across Phases 1 to 4

---

## 1. Executive Review of Phases 1 to 4

Before building any evidence-fusion mechanism, every prior component must be audited to verify its exact prediction semantics, training corpus, input domain, and known operational limitations.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             TRUSTTRACE PHASE INVENTORY                           │
├─────────┬──────────────────────┬────────────────────────────┬────────────────────┤
│ Phase   │ Dataset / Domain     │ Prediction Target          │ Key Metric         │
├─────────┼──────────────────────┼────────────────────────────┼────────────────────┤
│ Phase 1 │ STFD (Screenshots)   │ 5-way Manipulation Type    │ Test Acc: 33.90%   │
│ Phase 2 │ STFD (Screenshots)   │ Pixel Tamper Localization  │ Test Dice: 0.1610  │
│ Phase 3 │ STFD (Patches)       │ 5-way Patch Forensics      │ Test Acc: 42.03%   │
│ Phase 4 │ FindItAgain (Receipt)│ Binary Authenticity (R/E)  │ Val-Cal F1: 0.5497 │
└─────────┴──────────────────────┴────────────────────────────┴────────────────────┘
```

### Detailed Phase Profiles

#### Phase 1: STFD 5-Class Manipulation Classifier
- **Model Checkpoint:** [`models/stfd_manipulation_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/stfd_manipulation_best.pt)
- **Dataset:** STFD (Smartphone Tampered Forensic Dataset, ICASSP 2023)
- **Input Domain:** $224 \times 224$ RGB smartphone UI screenshots (mobile banking, payment screens, chat interfaces)
- **Ground-Truth Target:** 5-way manipulation operation:
  `COPY_MOVE` (0), `SPLICING` (1), `REMOVAL` (2), `INSERTION` (3), `REPLACEMENT` (4)
- **Output:** 5-dimensional probability simplex via softmax: $\sum_{c=0}^4 P(c) = 1.0$
- **Known Limitations:**
  1. STFD contains **zero authentic (unmanipulated) screenshots**. The classifier only predicts *which* manipulation occurred, conditional on the screenshot being tampered. It cannot determine if an image is authentic.
  2. Mobile screenshot graphics (flat UI elements, crisp raster icons) possess radically different noise and pixel distributions than scanned paper receipts.
- **Direct Fusion with Phase 4:** **NO.** Cannot be directly fused as an authenticity score because it outputs conditional manipulation categories, not real/edited probabilities.

#### Phase 2: STFD Tamper Localization Model
- **Model Checkpoint:** [`models/stfd_localization_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/stfd_localization_best.pt)
- **Dataset:** STFD with official binary masks
- **Input Domain:** $256 \times 256$ RGB smartphone screenshots
- **Ground-Truth Target:** Pixel-level binary tamper mask ($1 = \text{manipulated pixel}, 0 = \text{authentic pixel}$)
- **Output:** Dense sigmoid probability map $M \in [0, 1]^{256 \times 256}$
- **Known Limitations:**
  Trained exclusively on synthetic digital splicing, removal, and insertions in mobile screenshot interfaces. High false-positive rate on natural paper textures, folds, shadows, and thermal print artifacts.
- **Direct Fusion with Phase 4:** **Diagnostic Only (Transferability Test Required).** Needs explicit out-of-domain transferability testing before considering any fusion.

#### Phase 3: STFD High-Resolution Patch Classifier
- **Model Checkpoint:** [`models/stfd_patch_classifier_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/stfd_patch_classifier_best.pt)
- **Dataset:** STFD localized tamper bounding-box patches ($224 \times 224$ crops)
- **Input Domain:** High-resolution crops of localized suspicious UI regions
- **Ground-Truth Target:** 5-way manipulation classification (`COPY_MOVE`, `SPLICING`, `REMOVAL`, `INSERTION`, `REPLACEMENT`)
- **Output:** 5-class softmax probabilities over patch crops
- **Known Limitations:**
  Inherits all domain limitations of STFD. Does not output binary authenticity. Conditioned on localization output from Phase 2.
- **Direct Fusion with Phase 4:** **Diagnostic Only (Transferability Test Required).** Evaluated only in an exploratory track.

#### Phase 4: Binary Document Authenticity Baseline
- **Model Checkpoint:** [`models/phase4_authenticity_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/phase4_authenticity_best.pt)
- **Dataset:** *Find it again!* (987 receipts, 910 groups: 824 REAL, 163 EDITED)
- **Input Domain:** $224 \times 224$ RGB scanned receipt documents (SROIE-derived)
- **Ground-Truth Target:** Binary authenticity: `REAL` (0) vs `EDITED` (1)
- **Output:** Calibrated 2-class probability: $P(\text{EDITED} \mid x) \in [0, 1]$
- **Known Limitations:**
  Global resizing ($224 \times 224$) causes minute character alterations (e.g. altering one digit of a price) to wash out under spatial downsampling, limiting standalone recall to $32.00\%$ ($\tau=0.45$).
- **Direct Fusion with Phase 4:** **PRIMARY BASELINE.** Serves as the anchor global visual prior $P_{\text{EDITED, global}}$.

---

## 2. Formal Domain-Compatibility Matrix

| Component | Training Dataset | Input Domain | Task Semantics | Shared Evaluation Samples with Phase 4? | Can Directly Fuse as Primary Authenticity Signal? | Formal Compatibility Verdict |
|:---|:---|:---|:---|:---:|:---:|:---|
| **Phase 1** | STFD | Mobile Screenshots | 5-way manipulation type | No (Screenshots vs Receipts) | No (Different task & domain) | **INCOMPATIBLE** |
| **Phase 2** | STFD | Mobile Screenshots | Pixel tamper localization | No (STFD vs Receipts) | Diagnostic only (transfer test) | **CONDITIONAL / TRANSFER TEST** |
| **Phase 3** | STFD | Localized UI Patches | 5-way patch manipulation | No (STFD vs Receipts) | Diagnostic only (transfer test) | **CONDITIONAL / TRANSFER TEST** |
| **Phase 4** | Find it again! | Scanned Receipts | REAL vs EDITED | Yes (Primary benchmark) | Yes (Anchor signal) | **FULLY COMPATIBLE (ANCHOR)** |
| **OCR Service** | Native WinRT OCR | Scanned Receipts | Text typography & layout | Yes (Extracted on Phase 4 images) | Yes (Independent forensic stream) | **FULLY COMPATIBLE (NEW STREAM)** |

---

## 3. Scientific Justification for Phase 5 Architecture

### Why Global Classification Alone is Insufficient
In document fraud, malicious alterations are surgically localized: an attacker modifies an invoice total from `$14.00` to `$94.00` by pasting or editing a single glyph. The rest of the document ($99.8\%$ of pixel area) remains authentic. Global convolutional downsampling aggregates thousands of unaltered background pixels, heavily attenuating the localized tampering anomaly.

### Dual-Stream Evidence Fusion Concept
To overcome this fundamental physical bottleneck, Phase 5 integrates two complementary modalities:
1. **Global Visual Authenticity Stream (Phase 4):**
   Captures document-level paper tone, illumination gradients, global compression signatures, and holistic layout coherence.
2. **Local Text & Typography Forensic Stream (OCR Engine):**
   Extracts fine-grained layout geometry: baseline drift between adjacent words, inter-word spacing (kerning) collisions, font box aspect variance, numeric token structure, and text density anomalies.
3. **Exploratory Out-of-Domain Transfer Stream (Phase 2/3 Diagnostic):**
   Evaluates whether CNN features trained on mobile screenshots generalize to paper receipt artifacts. If empirical transferability shows no discriminative utility, it is formally disqualified from the primary production fusion model.

---

## 4. Leakage Prevention Protocol in Phase 5

1. **Frozen Test Partition:**
   The Phase 4 test partition ([`data/manifests/trusttrace_phase4_test.csv`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/data/manifests/trusttrace_phase4_test.csv): 148 samples, 120 groups, 123 REAL, 25 EDITED) is strictly quarantined.
2. **Feature Extraction Isolation:**
   All OCR features and frozen model inferences are extracted purely from image pixels and text geometry. No ground-truth label, filename prefix, or split tag is used as a feature input.
3. **Training & Tuning Isolation:**
   All normalizations (standard scaling), fusion model weights, and decision thresholds are learned exclusively on `train` (691 samples) and validated on `val` (148 samples). Test data is touched only once during final benchmark evaluation.

---
*TRUSTTRACE Research Team — Document Authenticity & Evidence Fusion*
