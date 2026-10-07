# TRUSTTRACE Phase 5: Cross-Domain Transferability Evaluation Report

> **TAG:** `OUT_OF_DOMAIN_EXPERIMENT`  
> **VERDICT:** **CONDITIONALLY TRANSFERABLE**

## 1. Objective of Diagnostic

This controlled diagnostic evaluated whether CNN models trained exclusively on synthetic manipulations in **smartphone UI screenshots** (STFD):
- **Phase 2:** UNet Tamper Localizer (`models/stfd_localization_best.pt`)
- **Phase 3:** Patch Manipulation Classifier (`models/stfd_patch_classifier_best.pt`)

transfer meaningfully to detect authentic (`REAL`) vs forged (`EDITED`) **scanned receipt documents** (*Find it again!* validation split, 148 samples).

## 2. Quantitative Diagnostic Results

- **Total Evaluated Validation Samples:** 148 (124 REAL, 24 EDITED)
- **Localizer Fallback Rate (no region $\ge 16$px at $	au=0.5$):** 6.08%
- **Mean Peak Tamper Probability on Authentic Receipts (REAL):** 0.5436
- **Mean Peak Tamper Probability on Forged Receipts (EDITED):** 0.5452
- **Mean Tamper Area Fraction on REAL:** 0.37%
- **Mean Tamper Area Fraction on EDITED:** 0.44%
- **Transferability Discrimination Power (ROC-AUC):** **0.5512**
- **Transferability Precision-Recall AUC (PR-AUC):** **0.2053** (Baseline Chance: 16.22%)

## 3. Scientific Analysis & Domain Mismatch Findings

### Finding: Weak Positive Transfer Detected
The Phase 2 localizer demonstrates weak discrimination power (ROC-AUC = 0.5512).
Ablation testing will compare inclusion vs exclusion in the final fusion model.
