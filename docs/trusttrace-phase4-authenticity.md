# TRUSTTRACE Phase 4: REAL vs EDITED Binary Authenticity Benchmark Report

**Document ID:** `TRUSTTRACE-DOC-P4-AUTH-001`  
**Phase:** 4 — Document Authenticity Classification Baseline  
**Date:** October 2026  
**Status:** COMPLETE & FROZEN  
**Target Artifact:** [`models/phase4_authenticity_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/phase4_authenticity_best.pt)  
**Evaluated Dataset:** *Find it again! — Receipt Dataset for Document Forgery Detection* (L3i Université de La Rochelle, SROIE-derived)

---

## 1. Executive Overview & Research Objective

In digital forensics and legal evidentiary verification, the primary security question is rarely merely:  
> *"What manipulation operation was performed?"* (addressed in Phase 1's 5-way taxonomy on STFD)

Rather, the fundamental operational question is:  
> *"Is this submitted evidence document genuine (`REAL`) or manipulated (`EDITED`)?"*

TRUSTTRACE Phase 4 establishes an independent, supervised binary authenticity classification benchmark. The model operates independently from the STFD screenshot dataset and the 5-way manipulation taxonomy, establishing a foundational baseline for visual document authenticity detection.

```
       [ Input Document Image (224x224 RGB) ]
                        │
                        ▼
         [ MobileNetV3-Small Backbone ]
                        │
                        ▼
           [ AdaptiveAvgPool2d((1, 1)) ]
                        │
                        ▼
               [ Dropout(p=0.2) ]
                        │
                        ▼
               [ Linear(576, 2) ]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   P(REAL | x)                   P(EDITED | x)
  (Authentic)                    (Manipulated)
```

---

## 2. Dataset Forensic Audit & Group-Leakage Prevention

### 2.1 Dataset Composition (*Find it again!*)
From the official archive (`findit2.zip`), exactly 987 receipt images were verified physically on disk across 987 transcription JSON pairs.
- **Total Samples:** 987
- **REAL (Authentic) Samples:** 824 (83.49%)
- **EDITED (Forged) Samples:** 163 (16.51%)
- **Imbalance Ratio:** 5.06 : 1 authentic-to-forged ratio.

### 2.2 Forensic Discovery of Official Split Leakage
A critical finding from our dataset audit ([`docs/trusttrace-phase4-dataset-audit.md`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/docs/trusttrace-phase4-dataset-audit.md)) was that the official release splits (`train.txt`, `val.txt`, `test.txt`) contain **severe pairwise leakage**:
- **7 direct suffix derivation pairs** were separated across splits (e.g., `X51009447842.png` [`EDITED`] in `train` vs `X51005447842.png` [`REAL`] in `test`), where both share the exact same receipt layout, vendor, transaction number (`CR0008955`), and date.
- In total, **22 groups (9 of which were mixed authentic/forged groups)** leaked across partitions in the official split.
- Evaluating a model on the official split would lead to severe data leakage, artificially inflating test scores because the model could simply memorize the receipt background and merchant layout.

### 2.3 Strict Group-Aware Partitioning (70% / 15% / 15%)
To prevent this, an indivisible connected-component grouping graph was constructed using:
1. Exact SHA-256 duplicate collision matching
2. Filename suffix derivation tracking (`X51005...` vs `X51009...`)
3. Perceptual hash near-duplicate clustering ($\text{Hamming distance} \le 2$)

This produced **910 indivisible receipt groups**. A greedy stratified allocation algorithm (seed 42) partitioned these groups into strictly isolated splits:

| Split | Sample Count | Sample % | Group Count | REAL Count | EDITED Count | EDITED Class % |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 691 | 70.01% | 647 | 577 | 114 | 16.50% |
| **Validation** | 148 | 14.99% | 143 | 124 | 24 | 16.22% |
| **Test (Held-Out)** | 148 | 14.99% | 120 | 123 | 25 | 16.89% |
| **Total** | **987** | **100.0%** | **910** | **824** | **163** | **16.51%** |

**Validation Verification:**
- Zero sample ID overlap across partitions.
- Zero SHA-256 hash overlap across partitions.
- Zero group leakage (no authentic source and forged variant split between train and test).
- Complete physical integrity on disk verified for all 987 files.

---

## 3. Model Architecture & Training Methodology

### 3.1 Model Specification
- **Backbone:** Lightweight `MobileNetV3-Small` initialized with pretrained ImageNet weights.
- **Classification Head:** `AdaptiveAvgPool2d((1, 1)) -> Flatten() -> Dropout(p=0.2) -> Linear(576, 2)`
- **Total Parameters:** 928,162
- **Trainable Parameters:** 928,162
- **Input Resolution:** $224 \times 224 \times 3$ RGB

### 3.2 Imbalance Mitigation (Option A: Inverse Class Frequency)
To prevent the classifier from collapsing to the majority class (`REAL`), training loss applied inverse class frequency weighting strictly derived from the training partition:
$$\text{Weight}(\text{REAL}) = \frac{N_{\text{train}}}{2 \cdot N_{\text{train, REAL}}} = \frac{691}{2 \times 577} = 0.5988 \quad \xrightarrow{\text{normalized}} \quad 0.3300$$
$$\text{Weight}(\text{EDITED}) = \frac{N_{\text{train}}}{2 \cdot N_{\text{train, EDITED}}} = \frac{691}{2 \times 114} = 3.0307 \quad \xrightarrow{\text{normalized}} \quad 1.6700$$
Ratio: $5.06 : 1$ penalty on misclassifying forged documents.

### 3.3 Training Hyperparameters
- **Optimizer:** `AdamW` ($\text{learning rate} = 3 \times 10^{-4}$, $\text{weight decay} = 1 \times 10^{-4}$)
- **Learning Rate Scheduler:** `CosineAnnealingLR` ($T_{\max}=10$, $\eta_{\min}=1 \times 10^{-6}$)
- **Batch Size:** 32
- **Epochs:** 10
- **Device:** Windows CPU (multi-core Intel execution)
- **Training Duration:** 1,108.8 seconds (18.48 minutes)
- **Augmentation (Train only):** Resize to $240 \times 240$, random crop to $224 \times 224$, mild color jitter (brightness 0.1, contrast 0.1). Destructive geometric transformations (shear, high rotation, cutouts) were avoided to preserve forensic edge cues.
- **Validation/Test Transforms:** Deterministic resize to $224 \times 224$, ImageNet normalization.

### 3.4 Checkpoint Selection
The primary checkpoint [`models/phase4_authenticity_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/phase4_authenticity_best.pt) was selected at **Epoch 5** based on peak validation Macro-F1 ($0.5027$).
Post-training calibration on the validation partition identified an optimal probability threshold of $\tau = 0.45$ (maximizing validation Macro-F1 to $0.5307$).

---

## 4. Test Benchmark Evaluation Results

The model was evaluated **exactly once** on the completely held-out test partition (148 images: 123 `REAL`, 25 `EDITED`) under both the default threshold ($\tau = 0.50$) and the validation-calibrated threshold ($\tau = 0.45$).

### 4.1 Primary Quantitative Metrics

| Metric | Default Threshold ($\tau = 0.50$) | Val-Calibrated Threshold ($\tau = 0.45$) | Notes |
|:---|:---:|:---:|:---|
| **Overall Accuracy** | 74.32% | 71.62% | Accuracy alone is not an informative forensic metric |
| **Macro-F1** | 0.4928 | **0.5497** | Unweighted average across both classes |
| **ROC-AUC** | **0.5906** | **0.5906** | Threshold-independent discrimination power |
| **PR-AUC** | **0.2784** | **0.2784** | Minority baseline prevalence is 16.89% (25/148) |
| **EDITED Recall** | 12.00% (3 / 25) | **32.00% (8 / 25)** | **+20.00% gain** via validation calibration |
| **EDITED Precision** | 15.79% (3 / 19) | **24.24% (8 / 33)** | **+8.45% gain** via validation calibration |
| **EDITED F1** | 0.1364 | **0.2759** | Significant improvement on minority class |
| **REAL Recall** | 86.99% (107 / 123) | 79.67% (98 / 123) | High specificity on authentic documents |
| **REAL Precision** | 82.95% (107 / 129) | 85.22% (98 / 115) | Consistent authentic classification |
| **REAL F1** | 0.8492 | 0.8235 | Robust majority performance |

### 4.2 Confusion Matrix Analysis

```
Default Threshold (tau = 0.50):
                        Predicted REAL    Predicted EDITED
  Actual REAL (123)          107 (TN)            16 (FP)
  Actual EDITED (25)          22 (FN)             3 (TP)

Validation-Calibrated Threshold (tau = 0.45):
                        Predicted REAL    Predicted EDITED
  Actual REAL (123)           98 (TN)            25 (FP)
  Actual EDITED (25)          17 (FN)             8 (TP)
```

---

## 5. Answers to Core Scientific Questions

### Q1: Can TRUSTTRACE distinguish REAL vs EDITED on the primary dataset?
**Answer:** The MobileNetV3-Small baseline demonstrates positive discrimination power ($\text{ROC-AUC} = 0.5906$, $\text{PR-AUC} = 0.2784$ vs 0.1689 baseline chance), indicating that visual authenticity cues are partially detectable even from resized $224 \times 224$ global document images. However, global downsampling severely constrains the model's ability to reliably separate subtle localized text forgeries from genuine print variations without high-resolution patch inspection.

### Q2: What is EDITED recall and precision?
**Answer:**
- Under default threshold ($\tau = 0.50$): **EDITED Recall is 12.00%**, **EDITED Precision is 15.79%**.
- Under validation calibration ($\tau = 0.45$): **EDITED Recall rises to 32.00%**, **EDITED Precision rises to 24.24%**.
In a security-sensitive context where missing a forged receipt carries a high penalty, operating at calibrated threshold $\tau = 0.45$ is substantially superior.

### Q3: What is Macro-F1, ROC-AUC, and PR-AUC?
**Answer:**
- **Macro-F1:** 0.4928 (default) / **0.5497** (calibrated)
- **ROC-AUC:** **0.5906**
- **PR-AUC:** **0.2784** (outperforming random positive prevalence by $+10.95\%$ absolute).

### Q4: Does performance remain reasonable without leakage?
**Answer:** Yes. Because all 910 groups were strictly isolated and zero template/variant pairs crossed the train/test divide, these numbers represent an honest, uncorrupted evaluation. If the official leaky splits had been used, test accuracy would have appeared artificially higher due to background layout memorization.

### Q5: What are the dominant false-positive and false-negative patterns?
**Answer:**
1. **False Negatives (EDITED $\to$ REAL, 17 samples at $\tau=0.45$):**
   - In *Find it again!*, forgeries are created by splicing or replacing small numbers of text characters (e.g., altering a price from `$12.50` to `$18.50` or replacing a vendor name).
   - When downsampling a $460 \times 1013$ receipt to $224 \times 224$, the altered characters occupy less than $0.2\%$ of the image pixels ($< 10 \times 10$ pixels).
   - Global average pooling washes out these minute local high-frequency boundary discrepancies, leading the model to perceive the document as genuine.
2. **False Positives (REAL $\to$ EDITED, 25 samples at $\tau=0.45$):**
   - Low-quality authentic scans with thermal paper fading, ink smudges, physical creases, or scanner dust produce local edge irregularities that the classifier mistakes for digital tampering artifacts.

### Q6: Are there signs of dataset-specific shortcut learning?
**Answer:**
- **Image Resolution / Aspect Ratio:** Audit confirmed both authentic and forged receipts originate from the SROIE capture pipeline; dimensions and aspect ratios are homogeneous between classes, preventing resolution from serving as a trivial shortcut.
- **Backgrounds:** Authentic and forged versions share similar paper backgrounds; because groups were isolated, the network could not exploit merchant backgrounds to guess labels.
- **Filename / Metadata Isolation:** Filenames and transcriptions were strictly excluded from input tensors. The model processed purely normalized RGB pixel values.

### Q7: Is the model strong enough to serve as the sole authenticity engine?
**Answer:** **No, not in isolation.** A global $224 \times 224$ classifier achieving $32\%$ EDITED recall is insufficient as a standalone fraud blocker. However, it provides a valuable global prior $P(\text{EDITED} \mid x_{\text{global}})$ that can be combined with local patch forensics.

### Q8: What should Phase 5 investigate?
**Answer:** Phase 5 should implement **Dual-Stream Multi-Scale Evidence Fusion**:
1. Global Document Authenticity Stream (Phase 4 MobileNetV3 baseline).
2. High-Resolution Tamper Localization & Patch Forensics Stream (derived from Phase 2/3 localization maps).
3. Text/OCR Inconsistency Detection (detecting font, alignment, and semantic price inconsistencies within extracted bounding boxes).

---

## 6. STFD Cross-Domain Evaluation Assessment

The user directive specified:
> *"After the primary Phase 4 benchmark is complete, optionally perform a separate evaluation using the frozen STFD data... If an appropriate real/unmanipulated STFD source cannot be established, skip this experiment and document why."*

### Finding & Scientific Justification for Skipping:
- The frozen STFD dataset (`C:\Users\jay\Downloads\STFD_ICASSP2023\STFD_ICASSP2023`) contains exactly 3,932 tampered screenshot images with 3,932 corresponding ground-truth masks across 5 manipulation classes.
- **STFD contains zero authentic (REAL) screenshot images.** The authors only released manipulated screenshots and their binary tamper masks.
- Creating artificial "REAL" labels from unmanipulated bounding box crops or non-tampered regions would manufacture unverified ground truth, violating our scientific integrity principles.
- Consequently, STFD cannot support a valid binary authenticity evaluation without genuine screenshot controls. This sanity check is intentionally and formally omitted with full scientific documentation.

---

## 7. Artifacts Summary & Verification

| Artifact Path | Type | Status | Description |
|:---|:---|:---:|:---|
| [`models/phase4_authenticity_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/phase4_authenticity_best.pt) | PyTorch Checkpoint | **FROZEN** | Best weights (Epoch 5), Calibrated threshold $\tau=0.45$ |
| [`data/manifests/phase4_dataset_master.csv`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/data/manifests/phase4_dataset_master.csv) | Manifest | Verified | 987 audited receipt images with SHA-256 and groups |
| [`data/manifests/trusttrace_phase4_train.csv`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/data/manifests/trusttrace_phase4_train.csv) | Manifest | Verified | 691 samples (577 REAL, 114 EDITED), 647 groups |
| [`data/manifests/trusttrace_phase4_val.csv`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/data/manifests/trusttrace_phase4_val.csv) | Manifest | Verified | 148 samples (124 REAL, 24 EDITED), 143 groups |
| [`data/manifests/trusttrace_phase4_test.csv`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/data/manifests/trusttrace_phase4_test.csv) | Manifest | Verified | 148 samples (123 REAL, 25 EDITED), 120 groups |
| [`reports/phase4_authenticity_config.json`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase4_authenticity_config.json) | Config | Verified | Architecture, hyperparameters, class weights |
| [`reports/phase4_authenticity_training_history.json`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase4_authenticity_training_history.json) | History | Verified | Epoch-by-epoch losses, F1, AUC, minority recall |
| [`reports/phase4_authenticity_training_history.png`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase4_authenticity_training_history.png) | Plot | Verified | 4-panel training and validation trajectories |
| [`reports/phase4_authenticity_test_report.json`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase4_authenticity_test_report.json) | Report | Verified | Complete metrics at default and calibrated thresholds |
| [`reports/phase4_authenticity_test_report.md`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase4_authenticity_test_report.md) | Report | Verified | Detailed markdown test evaluation report |
| [`reports/phase4_authenticity_confusion_matrix.png`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase4_authenticity_confusion_matrix.png) | Plot | Verified | Side-by-side confusion matrices ($\tau=0.50$ vs $\tau=0.45$) |
| [`reports/phase4_authenticity_examples.png`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase4_authenticity_examples.png) | Plot | Verified | Deterministic qualitative grid (TN, TP, FP, FN) |
| [`scripts/audit_phase4_dataset.py`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/scripts/audit_phase4_dataset.py) | Script | Executed | Dataset auditor, hash indexer, group identifier |
| [`scripts/create_phase4_splits.py`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/scripts/create_phase4_splits.py) | Script | Executed | Group-aware 70/15/15 stratified partition builder |
| [`scripts/train_phase4_authenticity.py`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/scripts/train_phase4_authenticity.py) | Script | Executed | MobileNetV3-Small training pipeline |
| [`scripts/evaluate_phase4_authenticity.py`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/scripts/evaluate_phase4_authenticity.py) | Script | Executed | Strict held-out test evaluation suite |
| [`scripts/reproduce_phase4_experiment.py`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/scripts/reproduce_phase4_experiment.py) | Script | Verified | End-to-end deterministic reproduction runner |

---

## 8. Frozen Baselines Verification

To ensure strict compliance with project integrity requirements:
- `models/stfd_manipulation_best.pt` (Phase 1): **Untouched** (SHA-verified).
- `models/stfd_localization_best.pt` (Phase 2): **Untouched** (SHA-verified).
- `models/stfd_patch_classifier_best.pt` (Phase 3): **Untouched** (SHA-verified).
- `C:\Users\jay\Downloads\STFD_ICASSP2023\STFD_ICASSP2023`: **Strictly Read-Only** (0 modifications).
- All Phase 1–3 manifests, reports, and benchmark metrics remain frozen.

---
*TRUSTTRACE Research Team — Document Authenticity & Forensic Evidence Verification*
