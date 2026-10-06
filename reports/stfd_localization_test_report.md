# TRUSTTRACE: STFD Tamper Localization Model Test Report (Phase 2)

**Task:** Supervised Image Tamper Localization (Pixel-Level Binary Segmentation)  
**Model Architecture:** ForensicUNet (3-Stage Multi-Scale Skip Connections, base_filters=16)  
**Input Resolution:** 256 × 256 × 3  
**Checkpoint Source:** Best Validation Foreground-Dice Checkpoint (`stfd_localization_best.pt`, Epoch 4)  
**Evaluation Set:** Held-out Test Set (`trusttrace_stfd_clustered_test.csv`, 590 samples)  
**Date:** 2026-10-07 00:35:13  

> [!IMPORTANT]
> **Anti-Overclaim Notice:** STFD tamper localization test performance measures pixel-level detection of tampered regions on confirmed-tampered screenshots using ground-truth binary masks. This is NOT a measure of TRUSTTRACE system-wide authenticity or forgery detection.

---

## 1. Overall Held-Out Test Metrics vs. All-Background Baseline

| Metric | ForensicUNet (Ours) | All-Background Baseline | Relative Gain / Note |
| :--- | :---: | :---: | :--- |
| **Foreground IoU (Jaccard)** | **0.0483** | 0.0000 | Direct overlap metric |
| **Foreground Dice (F1)** | **0.0921** | 0.0000 | Harmonic mean of FG Precision & Recall |
| **Foreground Precision** | **0.0911** | 0.0000 | Proportion of detected pixels truly tampered |
| **Foreground Recall** | **0.0931** | 0.0000 | Proportion of true tampered pixels detected |
| **Background IoU** | 0.9806 | 0.9895 | Background area preservation |
| **Mean IoU (mIoU)** | **0.5144** | 0.4947 | Macro average across FG and BG |
| **Pixel Accuracy** | 98.06% | 98.95% | Dominated by ~98.9% background pixels |

---

## 2. Confusion Matrix of Pixels (Test Set Aggregation)

| Pixel State | Predicted Tampered (1) | Predicted Background (0) | Total Pixels |
| :--- | :---: | :---: | :---: |
| **True Tampered (1)** | **37,980** (TP) | **369,896** (FN) | 407,876 |
| **True Background (0)** | **378,977** (FP) | **37,879,387** (TN) | 38,258,364 |

---

## 3. Benchmark Speed & Execution
- **Test Evaluation Time:** 50.24 seconds
- **Throughput:** 11.7 screenshots / second
- **Device:** cpu
- **Qualitative Visualizations:** Available in [stfd_localization_examples.png](file:///c:/Users/jay/New%20folder%20%282%29/Trustrace/reports/stfd_localization_examples.png)
