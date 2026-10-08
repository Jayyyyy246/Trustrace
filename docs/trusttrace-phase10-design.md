# TRUSTTRACE — PHASE 10: SYSTEM DESIGN & METHODOLOGY
## Multi-Scale Dense Pixel Scanning & Typographic Consistency Verification

**Document Identifier:** `TRUSTTRACE-DOC-P10-DESIGN-001`  
**Security / Forensics Research Classification:** Empirical Cyber-Forensics Research  
**Date:** October 2026  
**Status:** ACTIVE DESIGN BLUEPRINT  
**Core Target:** Resolving the Upstream Candidate Discovery Bottleneck  

---

### Non-Negotiable Project Integrity Statement
> **Phase 1–9 artifacts are strictly frozen and will not be modified during Phase 10.**  
> - Checkpoints `models/doc_patchformer_best.pt`, `models/phase8_patch_cnn_best.pt`, and `models/phase8_candidate_aware_docpatchformer_best.pt` remain unchanged.  
> - Official VIA annotations remain the sole authoritative localization ground truth.  
> - Candidate sources (`OCR_DENSE`, `PIXEL_DENSE`, `TYPOGRAPHIC`) are provenance metadata, not ground-truth labels.  
> - The held-out TEST partition ($N=148$ receipts, 86 official GT forgery boxes) is strictly quarantined; all parameters, window sizes, strides, and anomaly thresholds are selected on TRAIN/VAL only.

---

## 1. System Architecture Overview

Phase 10 addresses the fundamental diagnostic finding of Phase 9: **64.0% of missed tampered receipts** failed because candidate generation never surfaced the forged bounding box.

```
                    Native-Resolution Receipt (RGB Canvas)
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
   ┌───────────┐                ┌───────────┐                ┌───────────┐
   │ STREAM A: │                │ STREAM B: │                │ STREAM C: │
   │OCR Dense  │                │Pixel Dense│                │Typographic│
   │Line Scan  │                │Multi-Scale│                │Consistency│
   └─────┬─────┘                └─────┬─────┘                └─────┬─────┘
         │                            │                            │
         └────────────────────────────┼────────────────────────────┘
                                      │
                                      ▼
                        Deterministic Fusion & Deduplication
                         (IoU >= 0.40, Provenance Preserved)
                                      │
                                      ▼
                           Unified Candidate Manifest
                                      │
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
       PRIMARY METRIC:                           DOWNSTREAM BENCHMARK:
   GT Discovery Recall                       Frozen Phase 8 Compact CNN
    (@ IoU >= 0.25, 0.50)                     Evidence Fusion Pipeline
```

---

## 2. Stream A — OCR-Guided Dense Line Scanning

### 2.1 Text-Line Bounding Union
For each document $\mathcal{D}$, we load structured OCR lines $L_1, L_2, \dots, L_K$ from `data/ocr/<sample_id>_ocr.json`.
For a line $L_k$ with words $w_1, \dots, w_m$:
$$\mathbf{B}(L_k) = \left[ \min_j x_j, \min_j y_j, \max_j (x_j + w_j) - \min_j x_j, \max_j (y_j + h_j) - \min_j y_j \right]$$

### 2.2 Native Line Expansion & Multi-Scale Scanning
To ensure micro-tampered characters (e.g., replaced price digits or altered letters) are captured with surrounding context:
- Expand $\mathbf{B}(L_k)$ vertically by factor $1.25$ and horizontally by $1.10$.
- Evaluate three scanning window scales relative to line height $H_k$:
  * Scale 1 ($1.0\times$): Window $W = 1.5 H_k$, $H = 1.25 H_k$, Stride $S = 0.5 W$
  * Scale 2 ($1.5\times$): Window $W = 2.25 H_k$, $H = 1.5 H_k$, Stride $S = 0.5 W$
  * Scale 3 ($2.0\times$): Window $W = 3.0 H_k$, $H = 2.0 H_k$, Stride $S = 0.5 W$
- Candidates inherit provenance `source = OCR_DENSE`.

---

## 3. Stream B — OCR-Independent Dense Pixel Scanning

### 3.1 Motivation
Surgical forgeries often occur where OCR completely fails to detect text (faded thermal prints, unaligned numeric stamps, modified barcode labels). Stream B provides a purely visual, OCR-independent candidate discovery mechanism.

### 3.2 High-Frequency Residual & Saliency Computation
On native-resolution image $\mathbf{I}$:
1. Convert to grayscale $\mathbf{I}_{\text{gray}}$.
2. Compute Laplacian second-order derivative: $\mathbf{G}_{\text{Lap}} = |\nabla^2 \mathbf{I}_{\text{gray}}|$.
3. Compute morphological top-hat and black-hat transforms to highlight local contrast peaks:
   $$\mathbf{R}_{\text{contrast}} = \text{TopHat}(\mathbf{I}_{\text{gray}}) + \text{BlackHat}(\mathbf{I}_{\text{gray}})$$
4. Combine into an anomaly saliency energy map:
   $$\mathbf{E}_{\text{pixel}} = 0.6 \cdot \frac{\mathbf{G}_{\text{Lap}}}{\max \mathbf{G}_{\text{Lap}}} + 0.4 \cdot \frac{\mathbf{R}_{\text{contrast}}}{\max \mathbf{R}_{\text{contrast}}}$$
5. Multi-scale sliding window scanning extracts proposals where local window energy $\bar{E} \ge \tau_{\text{energy}}$ (calibrated on VALIDATION).
- Candidates inherit provenance `source = PIXEL_DENSE`.

---

## 4. Stream C — Typographic Consistency Analysis

### 4.1 Intra-Document Baseline & Deviation
Rather than relying on global font models, Stream C compares each word against neighboring words on the **same document**:
For word $w_{k,j}$ on line $L_k$:
- **Height Deviation:** $\Delta h = \frac{|h_{k,j} - \bar{h}_k|}{\bar{h}_k + \epsilon}$
- **Width-to-Character Ratio:** $\Delta w = \left| \frac{w_{k,j}}{\text{len}(w_{k,j})} - \bar{\mu}_{\text{char\_width}}(L_k) \right|$
- **Baseline Offset:** $\Delta y_{\text{base}} = |(y_{k,j} + h_{k,j}) - \text{baseline}(L_k)|$
- **Local Contrast & Stroke Proxy:** Using Otsu binarization, estimate stroke thickness proxy and foreground-to-background contrast ratio $\Delta c$.

### 4.2 Typographic Anomaly Score ($T \in [0, 1]$)
$$T(w) = 0.35 \min(1.0, \Delta h) + 0.25 \min(1.0, \Delta w) + 0.25 \min\left(1.0, \frac{\Delta y_{\text{base}}}{10.0}\right) + 0.15 \Delta c$$
- Words with $T(w) \ge \tau_T$ are promoted to candidate proposals.
- Candidates inherit provenance `source = TYPOGRAPHIC`.

---

## 5. Candidate Fusion & Deduplication Protocol

When combining proposals from Streams A, B, and C:
1. Two proposals $c_a, c_b$ with $\text{IoU}(c_a, c_b) \ge 0.40$ or center distance $\le 20\text{ px}$ are merged.
2. The merged box takes the bounding union.
3. Provenance is preserved as a pipe-separated string: e.g., `OCR_DENSE|TYPOGRAPHIC`.
4. Feature scores ($T$, pixel anomaly, edge density) are max-pooled.

---

## 6. Primary Metric — Discovery Recall

Evaluated directly against official VIA forgery annotations:
- $\text{Recall}_{\text{IoU} \ge 0.25} = \frac{\text{Count}(\text{GT with } \max_{c} \text{IoU}(GT, c) \ge 0.25)}{\text{Total GT Boxes}}$
- $\text{Recall}_{\text{IoU} \ge 0.50} = \frac{\text{Count}(\text{GT with } \max_{c} \text{IoU}(GT, c) \ge 0.50)}{\text{Total GT Boxes}}$

### Hard-Case Recovery Metric (Type-A Misses):
We explicitly isolate the 16 Type-A missed GT forgery regions from Phase 9:
$$\text{Type-A Recovery Rate} = \frac{\text{Recovered Type-A Regions}}{16}$$

---

## 7. Downstream Document Authenticity Evaluation

Newly discovered candidates are passed to the frozen Phase 8 Compact CNN (`models/phase8_patch_cnn_best.pt`) and aggregated using Phase 9 Full Evidence Fusion. This establishes whether improved candidate discovery directly translates into higher document-level EDITED recall.

*TRUSTTRACE Research Team — October 2026*
