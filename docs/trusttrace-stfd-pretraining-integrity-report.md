# TRUSTTRACE: Pre-Training Dataset Perceptual Integrity & Leakage Audit Report

- **Target Dataset:** STFD (Screenshot Text Forgery Dataset, ICASSP 2023)
- **Local Dataset Root:** `C:\Users\jay\Downloads\STFD_ICASSP2023\STFD_ICASSP2023` *(Read-Only)*
- **Audit Timestamp:** `2026-10-06 17:23:16 UTC`
- **Audit Runtime:** `41.18s`
- **Integrity Status:** **100% READ-ONLY COMPLIANT — ZERO TRAINING EXECUTED**

---

## 1. Executive Summary & Checkpoint Verdict

| Checkpoint Dimension | Confirmed Audit Result | Forensic Verdict |
| :--- | :---: | :---: |
| **Total Images Audited** | **3,932** | 100% Verified Decodable |
| **Exact Cryptographic Duplicates (SHA-256)** | **0** | **CONFIRMED UNIQUE** |
| **Dataset-Wide Exact Perceptual Duplicates (dist = 0)** | **473** | **REUSED BASE SCREENSHOT TEMPLATES DETECTED** |
| **Dataset-Wide Near-Duplicates (dist $\le$ 4)** | **1954** | **CONFIRMED REUSED BASE TEMPLATES** |
| **Cross-Split Exact Perceptual Duplicates (dist = 0)** | **216** | **PERCEPTUAL LEAKAGE DETECTED (Cross-Split Reused Templates)** |
| **Cross-Split Near-Duplicates (dist $\le$ 4)** | **884** | **PERCEPTUAL LEAKAGE DETECTED (Cross-Split Reused Templates)** |
| **Cross-Split Suspicious Pairs (dist $\le$ 8)** | **1421** | **INTER-SESSION VIEWPORT ALIGNMENT** |
| **Empty or Saturated Masks** | **0** | **100% VALID BINARY MASKS** |
| **Master Sample Count** | **3,932** | **100% ACCOUNTED** |
| **Cryptographic Split Disjointness** | **100% (2,753 / 589 / 590)** | **BYTE-LEVEL LEAK-FREE** |

> **CRITICAL FORENSIC AUDIT FINDING (PERCEPTUAL LEAKAGE):**  
> While the dataset possesses 0 exact cryptographic duplicates (all SHA-256 digests are unique due to altered pixel bytes in manipulated text areas), perceptual hashing reveals that **the authors of STFD created multiple distinct tampering operations (e.g. Splicing, Copy-Move, Replacement) on the same base smartphone screenshot templates**.  
> Because text forgery in STFD only affects an average of ~1.08% (median 0.41%) of the total pixels, the macro low-frequency visual appearance (pHash) of the underlying screenshot remains identical or near-identical.  
> Consequently, when splitting randomly by sample_id, **216 exact perceptual duplicates (dist = 0)** and **884 near-duplicates (dist $\le$ 4)** cross between TRAIN, VALIDATION, and TEST!

---

## 2. Perceptual Hashing Methodology

In compliance with project safety rules (*no unverified external packages installed*), deterministic perceptual comparison was executed using native OpenCV and NumPy algorithms:

1. **64-bit DCT pHash:**  
   - Viewport scaled to $32 \times 32$ grayscale using area interpolation (`cv2.INTER_AREA`).
   - 2D Discrete Cosine Transform (`cv2.dct`) computed to isolate frequency spectra.
   - Low-frequency $8 \times 8$ coefficient grid extracted.
   - Median coefficient thresholding (excluding DC component) generates a 64-bit fingerprint representing macro visual structure.
2. **64-bit dHash (Difference Hash):**  
   - Grayscale scaling to $9 \times 8$ with adjacent horizontal gradient comparison.
3. **Full Pairwise Comparison:**  
   - Exact pairwise Hamming distance calculated for all $\frac{3932 \times 3931}{2} = 7,728,246$ unique image pairs ($15,460,624$ full symmetric evaluations).
   - Conservative thresholding applied:
     - `dist = 0`: Identical low-frequency visual signature.
     - `dist <= 4`: Near-duplicate / visually near-identical layout ($\ge 93.75\%$ similarity).
     - `dist <= 8`: Suspicious structural similarity ($\ge 87.5\%$ similarity).

---

## 3. Perceptual Duplicate & Similarity Findings

* **Exact Perceptual Duplicates (pHash dist = 0):** **473** pairs across the dataset.
* **Dataset-Wide Near-Duplicates (pHash dist $\le$ 4):** **1954** pairs.
* **Dataset-Wide Suspicious Pairs (pHash dist $\le$ 8):** **3051** pairs (0.039% of total comparisons).

---

## 4. Cross-Split Perceptual Leakage Findings

Cross-split comparisons verify whether visually coupled screens leaked across partition boundaries:

* **Train vs Validation (dist $\le$ 4):** `415` pairs (including `104` exact dist=0)
* **Train vs Test (dist $\le$ 4):** `370` pairs (including `85` exact dist=0)
* **Validation vs Test (dist $\le$ 4):** `99` pairs (including `27` exact dist=0)

### Top Suspicious Cross-Split Pairs (pHash dist $\le$ 2):
| Sample A | Sample B | Split A | Split B | pHash Dist | dHash Dist | Manipulation A | Manipulation B |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| `stfd_5da962250f823c968d0e263955d6ab1c` | `stfd_9b1038338d3421b9483b0a0898e6639f` | VAL | TRAIN | 2 | 5 | `COPY_MOVE` | `COPY_MOVE` |
| `stfd_5da962250f823c968d0e263955d6ab1c` | `stfd_0abb7543b45ffc55fc5853dc5a6769b2` | VAL | TRAIN | 2 | 8 | `COPY_MOVE` | `INSERTION` |
| `stfd_5da962250f823c968d0e263955d6ab1c` | `stfd_9fa1fb9e86e4fc5acfa0160eb4954436` | VAL | TRAIN | 2 | 6 | `COPY_MOVE` | `INSERTION` |
| `stfd_63560d15bf1c2780d0bbab3b77a0c1a3` | `stfd_df88962fc5d7f6382da9c93b515ca47e` | VAL | TEST | 2 | 2 | `COPY_MOVE` | `REPLACEMENT` |
| `stfd_67317d7e9d6fc0356801510ca646898d` | `stfd_6bd591f40e5ca45b39e561df9aa09d03` | VAL | TRAIN | 2 | 4 | `COPY_MOVE` | `REMOVAL` |
| `stfd_67317d7e9d6fc0356801510ca646898d` | `stfd_396b6b0350e878e9315169add6cbd8dc` | VAL | TEST | 2 | 4 | `COPY_MOVE` | `INSERTION` |
| `stfd_67317d7e9d6fc0356801510ca646898d` | `stfd_f86998868f53a306544b569ad014c828` | VAL | TEST | 2 | 3 | `COPY_MOVE` | `INSERTION` |
| `stfd_67317d7e9d6fc0356801510ca646898d` | `stfd_866c377945ac629fdc66d3352b09c831` | VAL | TRAIN | 0 | 3 | `COPY_MOVE` | `REPLACEMENT` |
| `stfd_814e280ca6dcb93bd8dfa8fffa2cd0b9` | `stfd_d5bf13d1598e120e7678d209fa478651` | VAL | TRAIN | 2 | 1 | `COPY_MOVE` | `COPY_MOVE` |
| `stfd_b8f99757f8088b1fe06afb0b4a9d7762` | `stfd_a16ee7fe258e341674241faa1bfdc2dc` | TRAIN | VAL | 2 | 10 | `COPY_MOVE` | `INSERTION` |
| `stfd_0049b03c573ed561d785090ee63420ee` | `stfd_d688de3e3cf159770a858df6e656fa0a` | TEST | VAL | 2 | 9 | `SPLICING` | `INSERTION` |
| `stfd_0282474d93b008c23cab6696ff83e96a` | `stfd_b22e8d7d8742fea4c69e39696a82f67e` | TRAIN | TEST | 2 | 3 | `SPLICING` | `REMOVAL` |
| `stfd_0282474d93b008c23cab6696ff83e96a` | `stfd_a0bc071f046694c27d992a5ca74576bb` | TRAIN | VAL | 2 | 4 | `SPLICING` | `INSERTION` |
| `stfd_0282474d93b008c23cab6696ff83e96a` | `stfd_a660f13063e75581afb9919b6709eefe` | TRAIN | TEST | 2 | 2 | `SPLICING` | `REPLACEMENT` |
| `stfd_0752dd5c028496df1420a8de1c82d5ae` | `stfd_8b36adf41ed650ef791c3e119c7d6e3d` | TRAIN | TEST | 2 | 3 | `SPLICING` | `REMOVAL` |
| `stfd_076f040274fb7b77facfc0482a0a578c` | `stfd_65440f494727d951721fa574a9dcb409` | TEST | TRAIN | 0 | 9 | `SPLICING` | `INSERTION` |
| `stfd_076f040274fb7b77facfc0482a0a578c` | `stfd_5beda94fcb862ed78f8d8ee353a0f8e7` | TEST | VAL | 2 | 3 | `SPLICING` | `REPLACEMENT` |
| `stfd_07f3a5d7256defcb90753c225220352c` | `stfd_5e22cb87cff4a53db58c0e2ce138799d` | TRAIN | VAL | 0 | 3 | `SPLICING` | `REPLACEMENT` |
| `stfd_08d9b3851ea2795f75989aabd3afddea` | `stfd_73e22eb382336281d83cb5c1def3c0a0` | TRAIN | VAL | 0 | 2 | `SPLICING` | `REMOVAL` |
| `stfd_08d9b3851ea2795f75989aabd3afddea` | `stfd_725ae4101efd747f9819155e12b9a927` | TRAIN | VAL | 0 | 4 | `SPLICING` | `INSERTION` |
| `stfd_08d9b3851ea2795f75989aabd3afddea` | `stfd_106595213ff0b8f3db04476d120478e9` | TRAIN | VAL | 0 | 2 | `SPLICING` | `REPLACEMENT` |
| `stfd_0b5746c46964a7cb5a2816c593738ad5` | `stfd_d22018dd67066a7951a16289e2e3b5c3` | VAL | TRAIN | 2 | 1 | `SPLICING` | `SPLICING` |
| `stfd_0b5746c46964a7cb5a2816c593738ad5` | `stfd_6672bcb0d68fcb857fd544ca117a97f1` | VAL | TRAIN | 2 | 1 | `SPLICING` | `INSERTION` |
| `stfd_0bd3706c82c685061a57b0bb17e65d95` | `stfd_4e9532c25fdb3dfced0b506b9be7e12e` | TEST | TRAIN | 2 | 1 | `SPLICING` | `REPLACEMENT` |
| `stfd_12cd163527168dffe7fe99e82f4618d8` | `stfd_91163e5d94fdacd370ac5d313d653172` | TRAIN | TEST | 0 | 1 | `SPLICING` | `REPLACEMENT` |

---

## 5. Within-Split Near-Duplicates (Informational Only)

Near-duplicates (dist $\le$ 4) clustered within partitions:

- **TRAIN Internal Near-Duplicates:** `978` pairs
- **VALIDATION Internal Near-Duplicates:** `49` pairs
- **TEST Internal Near-Duplicates:** `43` pairs

*(Note: Within-split near-duplicates represent natural variations of the same base screenshot inside the training pool, which can benefit model robustness. However, cross-split pairs cause data leakage into evaluation sets).*

---

## 6. STFD Structural & Viewport Analysis

Distribution of top native device display resolutions in the dataset:

| Resolution ($W \times H$) | Image Count | Share | Typical Device Profile |
| :--- | :---: | :---: | :--- |
| `1080x2340` | 2665 | 67.8% | Smartphone Viewport |
| `1080x2400` | 504 | 12.8% | Smartphone Viewport |
| `750x1334` | 138 | 3.5% | Smartphone Viewport |
| `1080x2376` | 100 | 2.5% | Smartphone Viewport |
| `1080x2316` | 96 | 2.4% | Smartphone Viewport |
| `1080x1920` | 91 | 2.3% | Smartphone Viewport |
| `2340x1080` | 80 | 2.0% | Smartphone Viewport |
| `828x1792` | 68 | 1.7% | Smartphone Viewport |
| `1125x2436` | 56 | 1.4% | Smartphone Viewport |
| `1170x2532` | 53 | 1.3% | Smartphone Viewport |

---

## 7. Split-Level Mask Tampering Sanity Check

| Mask Metric | Master Dataset | TRAIN (70%) | VALIDATION (15%) | TEST (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Minimum Area Ratio** | 0.0168% | 0.0169% | 0.0168% | 0.0241% |
| **Maximum Area Ratio** | 37.48% | 34.74% | 29.32% | 37.48% |
| **Mean Area Ratio** | 1.083% | 1.079% | 1.051% | 1.135% |
| **Median Area Ratio** | 0.407% | 0.410% | 0.390% | 0.408% |
| **Extremely Small (<0.1%)** | 6.2% | 6.1% | 7.0% | 5.9% |
| **Extremely Large (>10%)** | 0.7% | 0.7% | 0.8% | 0.5% |
| **Empty Masks (0 px)** | **0** | **0** | **0** | **0** |
| **Saturated Masks (>90%)** | **0** | **0** | **0** | **0** |

---

## 8. Manipulation Distribution by Split (Official Ground Truth)

| Manipulation Type | TRAIN | VAL | TEST | TOTAL | Stratified Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `COPY_MOVE` | 531 | 114 | 113 | 758 | 70.1% / 15.0% / 14.9% |
| `INSERTION` | 491 | 105 | 105 | 701 | 70.0% / 15.0% / 15.0% |
| `REMOVAL` | 711 | 152 | 153 | 1016 | 70.0% / 15.0% / 15.1% |
| `REPLACEMENT` | 439 | 94 | 94 | 627 | 70.0% / 15.0% / 15.0% |
| `SPLICING` | 581 | 124 | 125 | 830 | 70.0% / 14.9% / 15.1% |
| **TOTAL** | **2753** | **589** | **590** | **3932** | **70.0% / 15.0% / 15.0%** |

---

## 9. Scene Metadata Distribution by Split (Inferred Metadata Only)

| Inferred Scene Category | TRAIN | VAL | TEST | TOTAL | Share |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DOCUMENT` | 304 | 66 | 74 | 444 | 11.3% |
| `E_COMMERCE` | 367 | 71 | 59 | 497 | 12.6% |
| `MAP_TRANSPORT` | 76 | 19 | 21 | 116 | 3.0% |
| `MOBILE_PAYMENT` | 1 | 0 | 0 | 1 | 0.0% |
| `ONLINE_BANKING` | 7 | 0 | 2 | 9 | 0.2% |
| `SYSTEM_INTERFACE` | 2 | 2 | 1 | 5 | 0.1% |
| `UNKNOWN` | 1949 | 421 | 420 | 2790 | 71.0% |
| `WEB_BROWSING` | 47 | 10 | 13 | 70 | 1.8% |
| **TOTAL** | **2753** | **589** | **590** | **3932** | **100.0%** |

---

## 10. Known Limitations & Forensic Audit Conclusions

1. **CONFIRMED:** Byte-level deduplication (SHA-256) is 100% unique (0 duplicate files).
2. **CONFIRMED:** Base screenshot template reuse exists extensively across manipulation categories in STFD (473 exact pHash duplicate pairs, 1,954 near-duplicate pairs).
3. **AUDIT VERDICT:** The current naive random-split dataset **FAILS** strict perceptual isolation: **216 exact perceptual duplicates (dist = 0)** and **884 near-duplicates (dist $\le$ 4)** cross between TRAIN, VALIDATION, and TEST.
4. **RECOMMENDATION BEFORE MODEL BENCHMARKING:** To prevent optimistic evaluation bias (where a model recognizes the familiar background template rather than true tampering traces), TRUSTTRACE should adopt a **perceptually clustered split**: grouping all near-duplicate base screenshot templates into connected clusters before assigning entire clusters to either Train, Val, or Test.
5. **ZERO MODEL TRAINING:** No training was performed. All operations were strictly read-only and non-destructive.

---
*Certified by TRUSTTRACE Digital Forensics Audit Suite*
