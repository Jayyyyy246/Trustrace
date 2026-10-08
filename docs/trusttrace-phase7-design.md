# TRUSTTRACE Phase 7: Dense Multi-Scale Saliency Sampler & Candidate Discovery Design

**Document ID:** `TRUSTTRACE-DOC-P7-DESIGN-001`  
**Phase:** 7 — Dense Multi-Scale Saliency Sampler & OCR-Independent Candidate Discovery  
**Date:** October 2026  
**Status:** COMPLETE & FROZEN SPECIFICATION  
**Target Architecture:** Multi-Scale Morphological Saliency Sampler + Frozen Doc-PatchFormer  
**Evaluated Dataset:** *Find it again! — Receipt Dataset for Document Forgery Detection* (987 receipts, 664 official forgery bboxes)

---

## 1. Problem Formulation & Research Objective

In Phase 6, TRUSTTRACE introduced the **Document-Native Patch Embedding Transformer (Doc-PatchFormer)**, elevating Document-Level Macro-F1 from 0.5497 to 0.7941 and EDITED Recall from 32.00% to 60.00% by operating on native-resolution image patches.

However, Phase 6's error analysis identified an essential structural vulnerability:
> **The OCR Coverage Bottleneck:**  
> In Phase 6, candidate patch sampling relied on OCR word detections. When a forged numeral, letter, or total line suffered from severe print fading, irregular typography, or was omitted by the OCR engine, that region was never extracted as a candidate patch. Our Phase 7 feasibility audit revealed that **65.81% of official forgery bounding boxes fail to achieve $\text{IoU} \ge 0.25$ alignment with OCR word or line detections**.

**Phase 7 Objective:** Design, benchmark, and validate an **OCR-independent multi-scale visual saliency candidate generation system** capable of discovering genuine manipulated regions without OCR dependency, while maintaining an efficient and computationally practical candidate budget.

```
       ┌────────────────────────────────────────────────────────┐
       │         Native-Resolution Scanned Receipt (100% DPI)   │
       └───────────────────────────┬────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
   [ Stream A: OCR Extraction ]              [ Stream B: Visual Saliency ]
   Native WinRT OCR JSONs                     Grayscale Morphology & Textures
   ├── Word Bounding Boxes                   ├── Multi-Scale Gradient (k=3, 7, 13)
   └── Synthesized Line Envelopes            ├── Edge Density & High-Pass (|I - G|)
              │                              └── Connected Component Clustering
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │       IoU Deduplication & Merging         │
             │  ├── OCR Candidates (Base Anchor)         │
             │  ├── Non-Redundant Morphology Candidates  │
             │  └── Inference-Time Saliency Score Rank   │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Candidate Budget Filter (Top-K)     │
             │       K in {5, 10, 20, 30, 50, 100}       │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Frozen Doc-PatchFormer (Phase 6)    │
             │       models/doc_patchformer_best.pt      │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             Document-Level Aggregation: Top-3 Mean (tau = 0.70)
```

---

## 2. Non-Negotiable Research & Inference-Time Constraints

1. **Zero Ground-Truth Utilization at Inference:** Official VIA forgery annotations are used exclusively for offline evaluation, coverage benchmarking, and error analysis. At candidate generation and inference time, the sampler has zero access to ground truth.
2. **Deterministic Candidate Scoring:** All candidate rankings are computed strictly from inference-time signals: gradient intensity, high-frequency residual magnitude, and spatial compactness.
3. **Budget Feasibility:** High candidate recall is scientifically meaningless if thousands of patches must be evaluated. Phase 7 explicitly characterizes the recall-versus-budget curve.
4. **Frozen Phase 6 Model:** The primary experiment evaluates the frozen checkpoint `models/doc_patchformer_best.pt` to isolate whether candidate discovery was indeed the bottleneck.

---

## 3. Visual Saliency Candidate Generation Pipeline

### 3.1 Stream B1: Multi-Scale Morphological Gradients
Morphological gradients compute the local difference between dilation and erosion:
$$G_k(I) = (I \oplus B_k) - (I \ominus B_k)$$
where $B_k$ is a rectangular structuring element of scale $k \in \{3, 7, 13\}$.
- Fine scale ($k=3$): Captures single-glyph stroke boundaries and ink bleed discontinuities.
- Medium scale ($k=7$): Captures word-level digit boundaries.
- Coarse scale ($k=13$): Captures line-level typography baselines.

Top gradient responses (90th percentile) are binarized and merged using directional horizontal closing structuring elements ($k_w \times k_h$) to group nearby characters into coherent word/glyph candidate boxes.

### 3.2 Stream B2: Edge Density and Local High-Frequency Textures
Document manipulations frequently exhibit subtle interpolation or compression halos. We construct a local high-frequency residual map:
$$R(I) = |I - G_\sigma(I)|$$
where $G_\sigma(I)$ is Gaussian smoothing ($\sigma=1.5$, kernel $9 \times 9$).
Connected components on thresholded residuals capture non-standard ink densities and edge anomalies.

### 3.3 Geometric Plausibility Filtering
Connected components undergo deterministic filtering to reject non-text structures:
- Minimum width $\ge 10\text{ px}$, minimum height $\ge 8\text{ px}$
- Maximum width $\le 85\%$ of image width, maximum height $\le 50\%$ of image height
- Area bounds: $64 \le \text{Area} \le 0.25 \times (W \times H)$
- Aspect ratio: $0.15 \le \frac{w}{h} \le 18.0$

---

## 4. Candidate Merging & IoU-Based Deduplication

Candidates from OCR and Morphology overlap heavily on standard text lines. To prevent redundant computation:
1. **Source A (OCR-Only):** Native OCR word boxes and synthesized line envelopes.
2. **Source B (Morphology-Only):** Non-Maximum Suppression (NMS) applied to morphological candidates at $\text{IoU}_{\text{NMS}} = 0.40$.
3. **Source C (Combined OCR + Morphology):** OCR candidates are retained as primary anchors; morphology candidates are admitted if their maximum IoU against all existing OCR candidates is less than $\theta_{\text{merge}} = 0.35$.

---

## 5. Downstream Evaluation Protocol

Candidates are cropped at native resolution with context padding, letterbox standardized to $128 \times 128 \times 3$, and passed through the frozen `models/doc_patchformer_best.pt`. Document-level predictions are pooled using $\text{Top-3 Mean}$ aggregation at the pre-calibrated validation threshold ($\tau = 0.70$).

Performance is reported on the untouched Phase 6 test partition (148 documents: 123 REAL, 25 EDITED) comparing:
- Pipeline A: OCR-only candidates
- Pipeline B: Morphology-only candidates
- Pipeline C: Combined OCR + Morphology candidates

---
*TRUSTTRACE Research Team — Phase 7 Design Specification*
