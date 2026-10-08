# TRUSTTRACE Phase 8: Candidate-Aware Patch Forensics & False-Positive Suppression Design

**Document ID:** `TRUSTTRACE-DOC-P8-DESIGN-001`  
**Phase:** 8 — Candidate-Aware Patch Forensics & False-Positive Suppression  
**Date:** October 2026  
**Status:** COMPLETE & FROZEN SPECIFICATION  
**Target Models:**  
- Compact CNN Baseline ([`models/phase8_patch_cnn_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/phase8_patch_cnn_best.pt))  
- Candidate-Aware Doc-PatchFormer ([`models/phase8_candidate_aware_docpatchformer_best.pt`](file:///c:/Users/jay/New%20folder%20(2)/Trustrace/models/phase8_candidate_aware_docpatchformer_best.pt))  
**Evaluated Dataset:** *Find it again! — Receipt Dataset for Document Forgery Detection* (987 receipts, 664 official forgery bboxes)  

---

## 1. Motivation: The False-Positive Crisis of Phase 7

Phase 7 demonstrated that an OCR-independent multi-scale visual saliency sampler can successfully recover genuine forgery regions missed by OCR (recovering 18.76% of OCR-missed regions at IoU $\ge$ 0.10).

However, Phase 7 exposed an immediate downstream failure mode:
> **The Downstream Classifier Collapse:**  
> When unconstrained morphology candidates were fed into the frozen Phase 6 Doc-PatchFormer, document specificity collapsed to **16.89%** (with 123 false alarms on authentic receipts). Because Doc-PatchFormer was trained exclusively on clean OCR text crops, it perceived legitimate high-gradient non-text structures (paper folds, creases, merchant logos, and thermal ink noise) as digital forgeries.

**Phase 8 Core Objective:** Move beyond the frozen text-only assumption. Train patch-level classifiers directly on the **real multi-source candidate distribution**, incorporating deliberately mined **hard authentic visual negatives** to teach the model to distinguish genuine forgery from legitimate visual complexity.

```
       ┌────────────────────────────────────────────────────────┐
       │     Multi-Source Candidate Pool (Phase 7 Outputs)      │
       │     ├── OCR Word & Line Candidates                     │
       │     └── Multi-Scale Morphology Saliency Candidates     │
       └───────────────────────────┬────────────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
   [ Positive Forgery Candidates ]           [ Hard Authentic Negatives ]
   IoU(cand, GT) >= 0.25                     IoU(cand, GT) <= 0.05 on REAL
   ├── Verified Alterations                  ├── Merchant Logos & Graphics
   └── GT Anchor Inpainting                  ├── Paper Folds & Creases
              │                              └── Table Grid Rules & Noise
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Ambiguous Candidate Exclusion       │
             │       0.05 < IoU(cand, GT) < 0.25         │
             │       (Excluded to prevent label noise)   │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │     Candidate-Aware Model Training        │
             │  ├── Baseline 1: Compact CNN (~94k params)│
             │  ├── Model 2: Candidate-Aware ViT (408k)  │
             │  └── Class-Weighted BCEWithLogitsLoss     │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             Document-Level Aggregation: Top-3 Mean (tau = 0.65)
             (Validation-Calibrated Specificity-First Operating Point)
```

---

## 2. Mathematical Candidate Labeling Protocol

To prevent label noise, candidates are strictly partitioned using official VIA annotations:
1. **Positive (`FORGED_REGION`, label = 1):** $\text{IoU}(\text{candidate}, \text{GT}) \ge 0.25$.
2. **Negative (`AUTHENTIC_REGION`, label = 0):** $\text{IoU}(\text{candidate}, \text{GT}) \le 0.05$.
   - Includes all candidates from verified authentic parent receipts ($\text{GT} = \emptyset$).
   - Includes non-overlapping candidates from edited receipts ($\text{IoU} \le 0.05$).
3. **Ambiguous (`EXCLUDED`, label = -1):** $0.05 < \text{IoU}(\text{candidate}, \text{GT}) < 0.25$.
   - **Critical Rule:** Ambiguous candidates are **never silently labeled negative**. They are strictly omitted from training loss computation.

---

## 3. Hard-Negative Mining Architecture

To suppress morphology false alarms, the training pool deliberately samples high-saliency non-text visual artifacts from authentic receipts:
- **Logo and Stylized Typography Negatives:** Heavy ink boundaries that trigger edge detectors.
- **Crease and Fold Negatives:** Sharp linear gradients running across receipts.
- **Table Divider Rules:** Long horizontal edge structures.
- **Thermal Print Texture:** Speckled background noise from aged thermal cash register rolls.

**Mining Constraint:** Hard negatives are mined strictly from TRAIN and VAL partitions. The TEST partition remains completely unmined and quarantined until final evaluation.

---

## 4. Model Architectures & Experimental Hypotheses

1. **Compact CNN Baseline (`models/phase8_patch_cnn_best.pt`):**
   - 3-stage convolutional backbone (32 $\rightarrow$ 64 $\rightarrow$ 128 channels) with MaxPool and Linear classification head (93,954 parameters).
   - *Hypothesis:* Tests whether candidate-aware domain calibration can be achieved with simple local receptive fields.
2. **Candidate-Aware Doc-PatchFormer (`models/phase8_candidate_aware_docpatchformer_best.pt`):**
   - 2-stage CNN tokenizer (64 spatial tokens, $d=128$), 2-layer ViT Encoder (4 heads, $d_{\text{ff}}=256$, dropout 0.1, 408,642 parameters).
   - *Hypothesis:* Tests whether self-attention across visual tokens provides superior discrimination between stroke interpolation artifacts and natural ink bleed.

---

## 5. Two-Stage Funnel & Specificity-First Evaluation

The evaluation separates the problem into two distinct stages:
$$\text{Final Region Recall} = \text{Discovery Recall (Stage 1)} \times \text{Conditional Classification Success (Stage 2)}$$
- **Stage 1 (Discovery):** Did the candidate sampler place a box over the ground-truth region ($\text{IoU} \ge 0.25$)?
- **Stage 2 (Classification):** Did the patch model assign a forgery probability $P \ge \tau$ to that candidate?

Document-level authenticity is pooled via $\text{Top-3 Mean}$ aggregation at the pre-calibrated validation threshold ($\tau = 0.65$), explicitly prioritizing false-positive suppression on authentic documents.

---
*TRUSTTRACE Research Team — Phase 8 Design Specification*
