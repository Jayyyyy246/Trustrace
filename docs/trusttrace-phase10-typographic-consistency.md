# TRUSTTRACE Phase 10: Typographic Consistency Analysis (Stream C)

## 1. Executive Summary & Research Hypothesis

Phase 10 introduced an intra-document typographic consistency analysis stream (Stream C) to evaluate the hypothesis:

> **Research Hypothesis:** A forged text fragment, even if visually convincing to casual human inspection, will exhibit subtle geometric or stylistic deviations from the document's authentic local typographic norms (e.g., stroke thickness, baseline offset, character aspect ratio, kerning, or edge sharpness).

This document details the mathematical formulation, implementation, ablation findings, and **empirical null result** of Stream C.

---

## 2. Intra-Document Typographic Modeling

To prevent bias from global font variations across diverse merchant templates, Stream C measures **document-relative and line-relative deviations**, comparing each candidate word against its neighbors from the same receipt.

### 2.1 Feature Families Extracted

1. **Character / Word Geometry Proxies:**
   * Median word height $H_w$ and width $W_w$.
   * Line-relative height deviation:
     $$\Delta_h = \frac{|H_w - \bar{H}_{\text{line}}|}{\bar{H}_{\text{line}} + \epsilon}$$
   * Line-relative aspect-ratio deviation:
     $$\Delta_{ar} = \frac{|(W_w / H_w) - \overline{AR}_{\text{line}}|}{\overline{AR}_{\text{line}} + \epsilon}$$

2. **Baseline Alignment Proxies:**
   * Word baseline $y_{\text{base}} = y_1 + h$.
   * Line-relative baseline offset:
     $$\Delta_{\text{base}} = \frac{|y_{\text{base}} - \bar{y}_{\text{base, line}}|}{\bar{H}_{\text{line}} + \epsilon}$$

3. **Stroke Width & Visual Appearance Proxies:**
   * Binarized morphological distance transform to estimate stroke width proxy $S_w$.
   * Stroke thickness deviation:
     $$\Delta_{\text{stroke}} = \frac{|S_w - \bar{S}_{w, \text{line}}|}{\bar{S}_{w, \text{line}} + \epsilon}$$
   * Character darkness/contrast: Mean pixel intensity within foreground mask compared to local line background.

4. **Spacing & Kerning Proxies:**
   * Inter-word horizontal gap deviation relative to median gap on the line.

### 2.2 Unified Typographic Anomaly Score ($T$)

A deterministic score $T \in [0, 1]$ is constructed by convex combination:
$$T = \min\left(1.0, 0.35 \Delta_h + 0.25 \Delta_w + 0.20 \Delta_{\text{base}} + 0.20 \Delta_{\text{stroke}}\right)$$

> **CRITICAL FORENSIC DISTINCTION:**
> $T$ is **NOT a forgery probability**. It is purely a visual and geometric deviation proxy used for candidate discovery and contextual evidence scoring.

---

## 3. Quantitative Results & Evaluation

Evaluated against the official VIA ground-truth annotations across the held-out TEST partition ($N=148$ receipts, 86 official GT boxes):

| Metric | Stream C (Typographic Only) | Historical Baseline (Phase 7/9) | Combined Phase 10 |
| :--- | :---: | :---: | :---: |
| **GT Discovery Recall @ $\text{IoU} \ge 0.25$** | **0.00% (0 / 86)** | 39.53% (34 / 86) | 15.12% (13 / 86) |
| **GT Discovery Recall @ $\text{IoU} \ge 0.50$** | **0.00% (0 / 86)** | 17.44% (15 / 86) | 5.81% (5 / 86) |
| **Average Proposals / Receipt** | 3.0 | 48.2 | 222.0 |
| **Downstream Accuracy** | 79.73% | 59.46% | 59.46% |
| **Downstream EDITED Recall** | 4.00% (1 / 25) | 36.00% (9 / 25) | 36.00% (9 / 25) |
| **Downstream REAL Specificity** | 95.12% (117 / 123) | 64.23% (79 / 123) | 64.23% (79 / 123) |
| **Macro-F1** | 0.4744 | 0.4778 | 0.4778 |

---

## 4. Typographic Feature Ablation Study

To determine whether individual feature subsets contributed partial signal, features were evaluated in isolation:

| Feature Family | Top Decile Anomaly Density | Simulated Word Recall | Observed Test GT Recall |
| :--- | :---: | :---: | :---: |
| **Family A: Geometry Only** ($\Delta_h, \Delta_{ar}$) | 14.2% | 32.6% | 0.00% |
| **Family B: Baseline / Alignment Only** ($\Delta_{\text{base}}$) | 10.8% | 24.4% | 0.00% |
| **Family C: Stroke / Edge Proxies Only** ($\Delta_{\text{stroke}}$) | 8.5% | 20.9% | 0.00% |
| **Family D: Spacing Only** ($\Delta_{\text{gap}}$) | 6.9% | 16.3% | 0.00% |
| **Family E: All Typographic Features ($T$)** | **3.0%** | **N/A** | **0.00%** |

No individual feature family was capable of isolating forged regions under the official VIA ground-truth benchmark.

---

## 5. Scientific Root-Cause Diagnosis of the Null Result

Why did intra-document typographic consistency analysis fail to discover micro-forgeries?

1. **Digital Typeface Matching in Receipt Fraud:**
   Document tampering in modern financial fraud (e.g., altering a receipt total from `12.50` to `72.50`) is typically performed by digital image editors inserting digits rendered in standard system fonts (e.g., Arial, Lucida Console, Merchant Monospace). Because thermal receipt printers themselves use standardized monospaced or dot-matrix fonts, digital insertions blend almost seamlessly into the line geometry.

2. **Severe Natural Intra-Document Heterogeneity:**
   Real, unmanipulated receipts naturally exhibit dramatic typographical variations:
   * Merchant headers are printed in large, bold, or stylized fonts.
   * Tax / registration details are printed in condensed, low-contrast fonts.
   * Item descriptions and price columns frequently differ in kerning and alignment.
   * Subtotal and Total lines use heavy bold weights.
   As a result, natural legitimate variation produces typographic anomaly scores $T \in [0.25, 0.45]$, masking the subtle deviations of forged digits.

3. **Sub-Character Granularity Mismatch:**
   Typographic features were computed at the *word* level (from OCR bounding boxes). When an adversary alters a single digit within a word (e.g., `$15.00` altered to `$75.00`), the aggregate word height, aspect ratio, and baseline remain virtually unchanged.

---

## 6. Phase 11 Implication

Typographic consistency modeling cannot serve as a reliable upstream candidate proposal mechanism for receipt scans. Future work seeking to exploit typographic cues must:
1. Extract characters at isolated glyph-level segmentation masks.
2. Pair typographic geometry with high-frequency sensor noise (PRNU), compression quantization (JPEG DCT grids), or neural font-classifier embeddings rather than heuristic bounding-box measurements.
