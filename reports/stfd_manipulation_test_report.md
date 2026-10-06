# TRUSTTRACE: STFD 5-Class Manipulation Classifier Test Report

**Task:** STFD 5-Class Manipulation Type Classification  
**Model Backbone:** MobileNetV3-Small (Pretrained on ImageNet)  
**Input Resolution:** 224 × 224 × 3  
**Checkpoint Source:** Best Validation Macro-F1 Checkpoint (`stfd_manipulation_best.pt`, Epoch 4)  
**Evaluation Set:** Held-out Test Set (`trusttrace_stfd_clustered_test.csv`, 590 samples)  
**Date:** 2026-10-06 23:49:16  

> [!IMPORTANT]
> **Anti-Overclaim Notice:** STFD 5-class manipulation classification test accuracy = 33.90%. This evaluates manipulation-type categorization on confirmed-tampered screenshots, NOT the full TRUSTTRACE authenticity decision system.

---

## 1. Overall Held-Out Test Metrics

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Test Accuracy** | 0.3390 | **33.90%** |
| **Macro Precision** | 0.3021 | 30.21% |
| **Macro Recall** | 0.3194 | 31.94% |
| **Macro F1-Score** | 0.3024 | **30.24%** |
| **Weighted F1-Score** | 0.3195 | 31.95% |

---

## 2. Per-Class Breakdown

| Manipulation Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **COPY_MOVE** | 0.3687 | 0.5789 | 0.4505 | 114 |
| **SPLICING** | 0.2844 | 0.2480 | 0.2650 | 125 |
| **REMOVAL** | 0.4335 | 0.4934 | 0.4615 | 152 |
| **INSERTION** | 0.2714 | 0.1810 | 0.2171 | 105 |
| **REPLACEMENT** | 0.1525 | 0.0957 | 0.1176 | 94 |

---

## 3. Confusion Matrix

| True \ Predicted | COPY_MOVE | SPLICING | REMOVAL | INSERTION | REPLACEMENT |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COPY_MOVE** | 66 | 14 | 18 | 10 | 6 |
| **SPLICING** | 31 | 31 | 29 | 21 | 13 |
| **REMOVAL** | 28 | 21 | 75 | 10 | 18 |
| **INSERTION** | 35 | 19 | 19 | 19 | 13 |
| **REPLACEMENT** | 19 | 24 | 32 | 10 | 9 |

---

## 4. Hardware & Benchmark Performance
- **Inference Time:** 23.10 seconds
- **Throughput:** 25.5 screenshots / second
- **Device:** cpu

