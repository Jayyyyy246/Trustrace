# TRUSTTRACE Phase 7: Dense Multi-Scale Saliency Sampler & Candidate Discovery

**Document ID:** `TRUSTTRACE-DOC-P7-CANDIDATES-001`  
**Phase:** 7 — Dense Multi-Scale Saliency Sampler & OCR-Independent Candidate Discovery  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY BENCHMARKED  
**Target Manifests:** [`data/manifests/phase7_candidates_master.csv`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/data/manifests/phase7_candidates_master.csv)  
**Evaluated Dataset:** *Find it again! — Receipt Dataset for Document Forgery Detection* (987 receipts, 664 official forgery bboxes)

---

## 1. Motivation: The OCR Discovery Bottleneck

Phase 6 established that native-resolution document patches evaluated with a 2-layer Transformer (Doc-PatchFormer) elevated Document Macro-F1 from 0.5497 to 0.7941 and EDITED Recall from 32.00% to 60.00%.

However, Phase 6's core vulnerability was its **dependency on OCR candidate anchors**:
- When a document manipulation altered a price total, tax identifier, or line item, the patch extractor could only sample the region if the local OCR engine produced a bounding box over that text.
- If the manipulated text suffered from irregular typography, low ink contrast, thermal fading, or was skipped by the OCR detector, the forged region was never presented to Doc-PatchFormer.
- In our Phase 7 feasibility audit ([`docs/trusttrace-phase7-feasibility.md`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/docs/trusttrace-phase7-feasibility.md)), we demonstrated that **65.81% of official ground-truth forgery bounding boxes fail to achieve $\text{IoU} \ge 0.25$ alignment with OCR word or line bounding boxes**.

Phase 7 introduces an **OCR-independent dense multi-scale visual saliency candidate discovery system** designed to locate suspicious, text-dense, and anomalous document regions directly from native raster pixels.

---

## 2. Multi-Scale Saliency Candidate Generation Methodology

Candidate discovery operates entirely without OCR recognition, optical character segmentation models, or ground-truth supervision. It combines three complementary low-level visual signals:

```
        Raw Native Receipt Image (Grayscale)
                         │
                         ▼
        [ Contrast Enhancement: CLAHE (clip=2.0) ]
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
[ Multi-Scale Morphological ]    [ High-Frequency Residual ]
[ Gradients (k = 3, 7, 13)  ]    [ R = |I - G_sigma(I)|    ]
         │                               │
         ▼                               ▼
[ Adaptive 90th% Threshold  ]    [ Adaptive 92nd% Threshold]
         │                               │
         ▼                               ▼
[ Horizontal Text Closing   ]    [ Glyph Cluster Morph     ]
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
        [ Connected Component Analysis ]
                         │
                         ▼
        [ Geometric Plausibility Filter ]
        ├── Width:  10 px <= w <= 0.85 * img_w
        ├── Height:  8 px <= h <= 0.50 * img_h
        ├── Area:   64 px² <= area <= 0.25 * img_area
        └── Aspect: 0.15 <= w/h <= 18.0
                         │
                         ▼
        [ Saliency Intensity Scoring & NMS ]
        └── Mean response inside candidate box
```

### 2.1 Multi-Scale Morphological Gradients
Morphological gradients compute the local boundary variation:
$$G_k(I) = (I \oplus B_k) - (I \ominus B_k)$$
where $B_k$ is a rectangular structuring element of dimension $k \in \{3, 7, 13\}$.
- **Scale $k=3$ (Fine):** Detects single-digit glyph stroke boundaries, ink bleed discontinuities, and sharp pixel boundaries.
- **Scale $k=7$ (Medium):** Detects word-level token contours and numeric clusters.
- **Scale $k=13$ (Coarse):** Detects entire text line fragments and receipt table rows.

### 2.2 High-Frequency Texture Anomalies
Forged document regions frequently introduce subtle high-frequency halos, edge ringing, or blur discrepancies relative to the surrounding paper background. We extract:
$$R(I) = |I - \mathcal{G}_{\sigma}(I)|$$
where $\mathcal{G}_{\sigma}$ represents Gaussian smoothing with $\sigma = 1.5$ (kernel $9 \times 9$). Top responses correspond to isolated stroke transitions and edge anomalies.

### 2.3 Connected Component Extraction & Plausibility Filtering
Connected components are extracted deterministically using 8-way connectivity. To discard background paper texture, full-page margins, and non-text artifacts, candidates must satisfy:
1. $w \ge 10\text{ px}$ and $h \ge 8\text{ px}$
2. $w \le 0.85 \times \text{img\_width}$ and $h \le 0.50 \times \text{img\_height}$
3. $64\text{ px}^2 \le \text{Area} \le 0.25 \times (\text{img\_width} \times \text{img\_height})$
4. Aspect ratio $\frac{w}{h} \in [0.15, 18.0]$

---

## 3. Candidate Deduplication & Merging Logic

Because OCR and morphology candidates can overlap on standard authentic text lines, deduplication prevents redundant patch extraction:
- **Non-Maximum Suppression (NMS):** Applied within morphological candidates at $\text{IoU}_{\text{NMS}} = 0.40$ based on candidate saliency scores.
- **Cross-Source Merging ($\theta_{\text{merge}} = 0.35$):** OCR candidate boxes are preserved as primary anchors. Morphological candidates are admitted only if:
  $$\max_{c_{\text{ocr}} \in \mathcal{C}_{\text{ocr}}} \text{IoU}(c_{\text{morph}}, c_{\text{ocr}}) < \theta_{\text{merge}}$$
  This ensures that every added morphology candidate explores genuinely unrepresented document regions.

---

## 4. Inference-Time Isolation & Budget Feasibility

- **Zero Test Leakage:** Candidate discovery operates strictly on the input image using deterministic morphology. No ground truth or validation statistics are utilized during inference.
- **Candidate Budget Analysis:** For each receipt, candidates are ranked by their visual saliency score and evaluated under budget constraints $K \in \{5, 10, 20, 30, 50, 100\}$.
- **Optimal Budget Selection:** As demonstrated in [`reports/phase7_candidate_coverage.md`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/reports/phase7_candidate_coverage.md), a budget of **Top-30 candidates** captures $75.6\%$ of all forgery regions while requiring only $\sim 28\text{ ms}$ of CPU inference time per document.

---
*TRUSTTRACE Research Team — Phase 7 Candidate Discovery*
