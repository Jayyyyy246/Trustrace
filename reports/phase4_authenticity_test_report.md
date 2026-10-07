# TRUSTTRACE Phase 4: Binary Authenticity Test Benchmark Report

## 1. Executive Summary

Phase 4 introduces the primary authenticity detection baseline for TRUSTTRACE: a binary classifier
answering whether a document evidence image is **REAL** or **EDITED**.
Unlike Phase 1–3 which evaluated 5-way manipulation classification and localization on smartphone screenshots (STFD),
Phase 4 establishes an independent document authenticity benchmark on the official *Find it again!* receipt dataset.

### Key Test Metrics Overview

| Metric | Default Threshold (0.50) | Val-Calibrated Threshold (0.45) |
|---|:---:|:---:|
| **Test Accuracy** | 74.32% | 71.62% |
| **Test Macro-F1** | 0.4928 | 0.5497 |
| **ROC-AUC** | 0.5906 | 0.5906 |
| **PR-AUC** | 0.2784 | 0.2784 |
| **EDITED Recall** (Security-Critical) | **12.00%** | **32.00%** |
| **EDITED Precision** | **15.79%** | **24.24%** |
| **EDITED F1** | 0.1364 | 0.2759 |
| **REAL Recall** | 86.99% | 79.67% |
| **REAL Precision** | 82.95% | 85.22% |
| **REAL F1** | 0.8492 | 0.8235 |

## 2. Confusion Matrix Breakdown

### Default Threshold (0.50)
- **True Negatives (TN - REAL correctly identified)**: 107 / 123
- **False Positives (FP - REAL misclassified as EDITED)**: 16
- **False Negatives (FN - EDITED missed as REAL)**: 22 / 25
- **True Positives (TP - EDITED correctly flagged)**: 3

### Validation-Calibrated Threshold (0.45)
- **True Negatives (TN)**: 98 / 123
- **False Positives (FP)**: 25
- **False Negatives (FN)**: 17 / 25
- **True Positives (TP)**: 8

## 3. Training & Architecture Parameters

- **Model Architecture**: MobileNetV3-Small
- **Total Parameters**: 928,162
- **Trainable Parameters**: 928,162
- **Input Dimensions**: 224x224 RGB
- **Optimizer**: AdamW (lr=0.0003, weight_decay=0.0001)
- **Loss Function**: CrossEntropyLoss with Inverse Class Frequency Weights (Option A)
- **Class Weights**: REAL: 0.3300, EDITED: 1.6700
- **Random Seed**: 42
- **Device**: cpu
- **Best Epoch**: 5

## 4. Leakage Prevention Audit

- **Group Isolation**: Verified across all 910 clusters (zero cross-split overlap).
- **Authentic/Forged Pair Protection**: Suffix-derived variant pairs are strictly confined to the same split.
- **Held-Out Test Set**: 148 samples (123 REAL, 25 EDITED) evaluated exactly once with frozen weights.
