# TRUSTTRACE Phase 6: Dataset Feasibility & Patch Supervision Audit

**Supervision Classification:** **Level A (Ground-Truth Localized Patches)**  
**Date:** October 2026  
**Status:** COMPLETE & SCIENTIFICALLY VALIDATED  

---

## 1. Executive Summary

Phase 6 investigates the transition from whole-document downsampled representations ($224 \times 224$) to native-resolution document patch forensics.
Before implementing a patch transformer, the dataset was audited against four strict feasibility tiers:

1. **Level A — Ground-Truth Localized Patches:** Authentic region-level bounding boxes exist, permitting verified labels (`FORGED_REGION` vs `AUTHENTIC_REGION`).
2. **Level B — Paired Differencing:** Verified alignment permits pixel-differencing localization without registration errors.
3. **Level C — Weakly Supervised Patches:** Image-level labels only (`IMAGE_LEVEL_SUPERVISION`).
4. **Level D — Unsupported:** No scientifically valid supervision available.

### Audit Verdict
> **VERDICT:** **Level A (Ground-Truth Localized Patches) IS FULLY SATISFIED.**  
> The official *Find it again!* dataset contains explicit native-resolution VIA-format annotations for manipulated entities, detailing the exact bounding boxes, entity types, and modification categories.

---

## 2. Quantitative Ground-Truth Forgery Annotation Audit

- **Total Audited Receipts:** 987 (824 REAL, 163 EDITED)
- **Edited Receipts with Valid Region Annotations:** **162 / 163 (99.4%)**
- **Total Ground-Truth Forgery Bounding Boxes:** **664**
- **Median Forgery Bounding Box Width:** 24.0 pixels (native scan resolution)
- **Median Forgery Bounding Box Height:** 34.0 pixels (native scan resolution)
- **Median Forgery Bounding Box Area:** 808.0 pixels²

### Entity Types Annotated
| Entity Type | Count | Share (%) | Forensic Relevance |
|:---|:---:|:---:|:---|
| **Metadata** | 132 | 19.9% | High-value target for financial fraud |
| **Other** | 30 | 4.5% | High-value target for financial fraud |
| **Product** | 147 | 22.1% | High-value target for financial fraud |
| **Total/payment** | 315 | 47.4% | High-value target for financial fraud |
| **Company** | 40 | 6.0% | High-value target for financial fraud |

---

## 3. Partition Inventory & Leakage Protection

Ground-truth forgery bounding boxes are distributed across the group-aware partitions without group leakage:

| Split | Edited Receipts with BBoxes | Total Forgery BBoxes | Document Group Count | Zero Group Leakage? |
|:---|:---:|:---:|:---:|:---:|
| **Train** | 113 | 480 | 647 | **PASS** |
| **Val** | 24 | 98 | 143 | **PASS** |
| **Test** | 25 | 86 | 120 | **PASS** |

> [!IMPORTANT]
> **Grouping Hierarchy Rule:** `document group → parent image → patches`  
> Under this structure, patches from the same parent document group remain strictly confined to the same split. No patch from a test document or its twin variant appears in train or validation.

---

## 4. Native-Resolution Patch Extraction Strategy

Because ground-truth bounding boxes have a median size of $\sim 20 \times 30$ pixels, downsampling the entire document destroys character strokes.
We implement native-resolution extraction:
1. Load original native PNG image.
2. Extract candidate patches centered on:
   - **Positive class (`FORGED_REGION`):** Ground-truth forgery bounding boxes expanded with controlled context padding.
   - **Negative class (`AUTHENTIC_REGION`):** OCR word bounding boxes on authentic receipts, and non-overlapping OCR word boxes on edited receipts.
3. Standardize patch dimensions (evaluating 64×64, 128×128, 192×192) while preserving native spatial pixels.

This confirms that Phase 6 Doc-PatchFormer is scientifically grounded with authentic region-level supervision.
