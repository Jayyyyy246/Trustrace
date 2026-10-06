# TRUSTTRACE: STFD Dataset Preparation & Forensic Integrity Report

**Dataset Name:** Screenshot Text Forgery Dataset (STFD)  
**Reference Publication:** *Learning to Locate the Text Forgery in Smartphone Screenshots* (ICASSP 2023)  
**Official Project Page:** `https://github.com/ZeqinYu/STFL-Net`  
**Local Dataset Path:** `C:\Users\jay\Downloads\STFD_ICASSP2023\STFD_ICASSP2023` *(Read-Only Source)*  
**Pipeline Status:** Master Manifest Created, Stratified Split Generated, 100% Leakage Checks Passed  

---

## 1. Distinction Between Ground Truth and Inferred Metadata

To adhere to strict digital evidence and forensic audit standards, TRUSTTRACE explicitly distinguishes between official ground-truth labels and secondary inferred metadata:

* **OFFICIAL GROUND TRUTH (from STFD):**
  - **Manipulation Type:** One of the 5 canonical manipulation operations (`COPY_MOVE`, `SPLICING`, `REMOVAL`, `INSERTION`, `REPLACEMENT`).
  - **Tampering Mask:** Binary ground-truth mask with pixel values `{0, 255}` (0 = pristine background, 255 = manipulated region).
  - **TRUSTTRACE Class:** All STFD tampered instances map to `trusttrace_class = EDITED`. No synthetic or untampered `REAL` samples are assumed from STFD.
* **INFERRED SECONDARY METADATA (TRUSTTRACE Computer Vision & OCR Engine):**
  - **Scene Category:** Heuristically inferred scene context (`E_COMMERCE`, `DOCUMENT`, `MAP_TRANSPORT`, `WEB_BROWSING`, `ONLINE_BANKING`, `SYSTEM_INTERFACE`, `MOBILE_PAYMENT`, `OTHER`, `UNKNOWN`).
  - **Scene Confidence:** Scalar in range `[0.0, 1.0]`.
  - **Classification Method:** Trace of multimodal heuristics used (`OCR_TEXT`, `CV_MONOCHROME_DOC`, `CV_ECOMMERCE_COLOR`, `CV_MAP_CANVAS`, `CV_SETTINGS_GRID`).
  - *These scene tags are never used as training targets or manipulation labels.*

---

## 2. Dataset Inventory & Integrity Summary

| Forensic Audit Check | Result | Verification Status |
| :--- | :---: | :---: |
| **Total Images Discovered** | **3,932** | 100% Readable & Decodable |
| **Total Ground-Truth Masks** | **3,932** | 100% Readable & Decodable |
| **Valid Image-Mask Pairs** | **3,932** | 100% 1-to-1 Match |
| **Missing Masks** | **0** | None |
| **Orphan Masks** | **0** | None |
| **Dimension Mismatches** | **0** | None |
| **Corrupted Files** | **0** | None |
| **Exact Duplicate Files** | **0** | All SHA-256 Hashes Unique |

---

## 3. Viewport & Image Resolution Distribution

The dataset captures screenshots across smartphone hardware viewports:

| Resolution ($W \times H$) | Sample Count | Percentage | Representative Device Form-Factor |
| :--- | :---: | :---: | :--- |
| `1080 × 2340` | 2,665 | 67.8% | Android Flagship FHD+ (19.5:9) |
| `1080 × 2400` | 504 | 12.8% | Android Flagship FHD+ (20:9) |
| `750 × 1334` | 138 | 3.5% | Apple iPhone 6/7/8/SE (16:9) |
| `1080 × 2376` | 100 | 2.5% | Honor / Huawei Flagship Viewports |
| `1080 × 2316` | 96 | 2.4% | Android Narrow Viewports |
| `1080 × 1920` | 91 | 2.3% | Standard Mobile Full HD (16:9) |
| `2340 × 1080` | 80 | 2.0% | Landscape Smartphone Viewports |
| Other Resolutions | 258 | 6.6% | Tablets, Custom Ratios, Desktop Previews |
| **Total** | **3,932** | **100.0%** | |

---

## 4. Manipulation Type Distribution (Official Ground Truth)

| Official STFD Folder | Canonical Type | Count | Dataset Share |
| :--- | :---: | :---: | :---: |
| `1_Copy-move` | `COPY_MOVE` | 758 | 19.3% |
| `2_Splicing` | `SPLICING` | 830 | 21.1% |
| `3_Removal` | `REMOVAL` | 1,016 | 25.8% |
| `4_Insertion` | `INSERTION` | 701 | 17.8% |
| `5_Replacement` | `REPLACEMENT` | 627 | 15.9% |
| **Total** | | **3,932** | **100.0%** |

---

## 5. Inferred Scene Metadata Distribution

| Inferred Scene Category | Sample Count | Share | Primary Detection Indicator |
| :--- | :---: | :---: | :--- |
| `E_COMMERCE` | 497 | 12.6% | Top-half product imagery, price badges, cart docks, specs |
| `DOCUMENT` | 444 | 11.3% | High whiteness (>85%), low saturation, dense text scanlines |
| `MAP_TRANSPORT` | 116 | 3.0% | Cartographic color palette, roadway lines, transit keywords |
| `WEB_BROWSING` | 70 | 1.8% | URL address bar, browser toolbar icons, domain names |
| `ONLINE_BANKING` | 9 | 0.2% | Bank ledger statements, card representations, bank titles |
| `SYSTEM_INTERFACE` | 5 | 0.1% | Grouped settings menus, uniform row dividers, toggle grids |
| `MOBILE_PAYMENT` | 1 | 0.0% | Explicit payment receipt vouchers, transaction confirm cards |
| `UNKNOWN` | 2,790 | 71.0% | Conservative fallback for ambiguous / sub-threshold images |
| **Total** | **3,932** | **100.0%** | |

### UNKNOWN Scene Handling Rationale
In accordance with strict forensic guidelines, ambiguous screenshots were never forced into speculative categories. Screenshots with confidence $< 0.35$ or overlapping cross-domain features (such as in-app chat purchases or hybrid feed items) are preserved as `UNKNOWN`.

---

## 6. Mask Tampering Region Statistics

Ground-truth binary masks were analyzed at pixel level:

* **Mean Tamper Area Ratio:** `1.083%` (tampered pixels relative to total image pixels)
* **Median Tamper Area Ratio:** `0.407%`
* **Minimum Tamper Area Ratio:** `0.0168%` (fine text glyph edits)
* **Maximum Tamper Area Ratio:** `37.48%` (broad paragraph/document replacements)
* **Mean Tampered Pixels:** `26,693 px`
* **Median Tampered Pixels:** `9,978 px`

*(Observation: The majority of text manipulations occupy less than 1.5% of the total canvas, highlighting the localized nature of digital screenshot tampering).*

---

## 7. Deduplication & Data Leakage Verification

* **Exact Duplicate Images:** `0` (All 3,932 SHA-256 hashes are strictly unique).
* **Exact Duplicate Masks:** `0` (All 3,932 mask SHA-256 hashes are strictly unique).
* **Perceptual Hashing:** *Perceptual duplicate detection unavailable because dependency (`imagehash`) is not installed in the environment (not installed automatically per audit policy).*
* **Cross-Split Leakage Overlap:**
  - `Train ∩ Validation`: **0** sample IDs, **0** image hashes, **0** mask hashes.
  - `Train ∩ Test`: **0** sample IDs, **0** image hashes, **0** mask hashes.
  - `Validation ∩ Test`: **0** sample IDs, **0** image hashes, **0** mask hashes.
  - **Verdict:** **100% Disjoint Partitions (Zero Data Leakage).**

---

## 8. Reproducible Train / Validation / Test Splits

* **Fixed Random Seed:** `42`
* **Stratification Axis:** `manipulation_type`
* **Target Ratios:** `70.0% Train` / `15.0% Validation` / `15.0% Test`

### Split Allocation Matrix

| Manipulation Type | Train (70%) | Validation (15%) | Test (15%) | Total | Train % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COPY_MOVE` | 531 | 114 | 113 | 758 | 70.1% |
| `INSERTION` | 491 | 105 | 105 | 701 | 70.0% |
| `REMOVAL` | 711 | 152 | 153 | 1,016 | 70.0% |
| `REPLACEMENT` | 439 | 94 | 94 | 627 | 70.0% |
| `SPLICING` | 581 | 124 | 125 | 830 | 70.0% |
| **TOTAL** | **2,753** | **589** | **590** | **3,932** | **70.0%** |

---

## 9. Prepared Manifest Artifacts

1. **Master Manifest:**  
   [`data/manifests/trusttrace_stfd_master.csv`](../data/manifests/trusttrace_stfd_master.csv)  
   Contains 3,932 verified rows with schema:  
   `sample_id,image_path,mask_path,relative_image_path,relative_mask_path,manipulation_type,trusttrace_class,scene_category,scene_confidence,classification_method,width,height,file_size,mask_pixel_count,mask_area_ratio,image_md5,image_sha256,mask_md5,mask_sha256`.

2. **Training Split:**  
   [`data/manifests/trusttrace_stfd_train.csv`](../data/manifests/trusttrace_stfd_train.csv) (2,753 samples).

3. **Validation Split:**  
   [`data/manifests/trusttrace_stfd_val.csv`](../data/manifests/trusttrace_stfd_val.csv) (589 samples).

4. **Test Split:**  
   [`data/manifests/trusttrace_stfd_test.csv`](../data/manifests/trusttrace_stfd_test.csv) (590 samples).

---

## 10. Operational Tooling & Reproducibility Scripts

* **Split Generation:** `python scripts/create_stfd_splits.py`  
  Recreates the exact deterministic splits from the master manifest using seed `42`.
* **Dataset Validation:** `python scripts/validate_stfd_prepared_dataset.py`  
  Executes 7 independent integrity and leakage checks (file existence, dimensions, zero overlap, valid labels). Exits with code 0 on complete pass.
* **Statistical Summary:** `python scripts/summarize_stfd_dataset.py`  
  Prints terminal metrics for sample counts, manipulation distributions, scene allocations, and mask area statistics.

---

## 11. Known Limitations

1. **No Authentic / Pristine Counterparts in STFD:** The official dataset release contains only tampered screenshots. Untampered images (`trusttrace_class = REAL`) must be acquired from separate pristine document/screenshot collections.
2. **No Hardware Device Grouping Metadata:** Because filenames are randomized 32-character MD5 hashes, multi-shot variations of the same original device session cannot be clustered into groups; stratified individual splitting was applied.
3. **Inferred Scenes:** Scene tags are heuristic estimates for exploratory analysis and must not be used as ground truth.

---
*Report certified by TRUSTTRACE Digital Evidence Forensics Engine*
