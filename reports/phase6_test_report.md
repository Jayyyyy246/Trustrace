# TRUSTTRACE Phase 6: Doc-PatchFormer Test Benchmark Report

## 1. Executive Summary

TRUSTTRACE Phase 6 implements the **Document-Native Patch Embedding Transformer (Doc-PatchFormer)**,
directly resolving the resolution bottleneck of Phase 4 and Phase 5 by operating on 100% native-resolution
crops ($128 \times 128$) around candidate text glyphs and ground-truth manipulated entities.

### Key Findings:
1. **Patch-Level Discrimination Power:** Doc-PatchFormer achieves a Patch-Level ROC-AUC of **0.8398** and PR-AUC of **0.6563** (Accuracy: 74.40%, Forged Recall: 75.58%).
2. **Document-Level Aggregation:** Doc-PatchFormer achieves Document-Level Macro-F1 of **0.7941** and EDITED Recall of **60.00%** (ROC-AUC: 0.8800).
3. **Comparison Against Baselines:** Outperforms both Global Whole-Receipt CNN (Phase 4) and Patch-CNN without transformer across fine-grained character edit detection.

---

## 2. Document-Level Test Benchmark Comparison (148 Held-Out Receipts)

| Architecture | Input Representation | Threshold | Accuracy | Macro-F1 | EDITED Recall | EDITED Precision | EDITED F1 | ROC-AUC | PR-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline A: Global Alone** | Global 224x224 | 0.45 | 71.62% | 0.5497 | 32.00% | 24.24% | 0.2759 | 0.5906 | 0.2784 |
| **Baseline B: Patch-CNN** | Native 128x128 Patch | 0.7 | 81.76% | 0.6178 | 28.00% | 43.75% | 0.3415 | 0.8007 | 0.4363 |
| **Baseline C: Doc-PatchFormer** | Native 128x128 Patch | **0.7** | **89.19%** | **0.7941** | **60.00%** | **71.43%** | **0.6522** | **0.8800** | **0.7607** |
| **Baseline D: Phase 5 Fusion** | Global 224x224 + OCR | 0.55 | 74.32% | 0.4935 | 20.00% | 16.13% | 0.1786 | 0.5906 | 0.2784 |

---

## 3. Patch-Level Supervised Test Benchmark (Ground-Truth Supervised)

- **Total Test Patches Evaluated:** 418 (86 FORGED_REGION, 332 AUTHENTIC_REGION)
- **Patch-Level Accuracy:** 74.40%
- **FORGED Region Recall:** 75.58%
- **FORGED Region Precision:** 43.05%
- **FORGED Region F1-Score:** 0.5485
- **Patch ROC-AUC:** 0.8398
- **Patch PR-AUC:** 0.6563 (Baseline prevalence: 20.57%)

---

## 4. Confusion Matrices Breakdown

### Doc-PatchFormer (Baseline C @ τ=0.70)
- True Negatives (TN): 117 / 123
- False Positives (FP): 6
- False Negatives (FN): 10 / 25
- True Positives (TP): 15

### Patch-CNN Baseline (Baseline B @ τ=0.70)
- True Negatives (TN): 114 / 123
- False Positives (FP): 9
- False Negatives (FN): 18 / 25
- True Positives (TP): 7

---

## 5. Formal Ablation Summary

| Model | Input | Architecture | Supervision | Macro-F1 | EDITED Recall | ROC-AUC | Status |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| Baseline A (Global Alone) | Global 224x224 | MobileNetV3-Small | Image-level | 0.5497 | 0.32 | 0.5906 | VALID_BASELINE |
| Baseline B (Patch-CNN) | Native 128x128 Patch | 3-Stage CNN | Ground-Truth Localized | 0.6178 | 0.28 | 0.8007 | CNN_PATCH_BASELINE |
| Baseline C (Doc-PatchFormer) | Native 128x128 Patch | 2-Layer Patch Transformer | Ground-Truth Localized | 0.7941 | 0.6 | 0.88 | TRANSFORMER_MODEL |
| Baseline D (Phase 5 Fusion) | Global 224x224 + OCR | Logistic Regression Fusion | Image-level | 0.4935 | 0.2 | 0.5906 | MULTIMODAL_FUSION |

---

## 6. Scientific Verification & Answers to Core Questions

1. **Does native-resolution patch analysis outperform global 224x224?** Yes. Operating at native resolution preserves sub-millimeter glyph stroke edges that are completely destroyed by whole-document downsampling.
2. **Does it improve EDITED recall?** Yes, Doc-PatchFormer significantly elevates minority class sensitivity on character-level alterations.
3. **Does a transformer outperform a simpler CNN baseline?** Yes, self-attention across the 64 spatial tokens provides superior context modeling of character-background contrast compared to local convolutions alone.
4. **Zero Leakage:** All patches strictly inherited the 910 parent document groups with zero cross-split leakage.
