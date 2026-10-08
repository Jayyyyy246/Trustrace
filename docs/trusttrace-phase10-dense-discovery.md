# TRUSTTRACE Phase 10: Multi-Scale Dense Discovery (Streams A & B)

## 1. Executive Overview

In Phase 9, diagnostic error decomposition revealed that **candidate discovery is the dominant system bottleneck**: approximately **64% of missed tampered receipts** failed because the upstream candidate sampler never proposed a window covering the forged region.

Phase 10 implemented two high-resolution candidate-discovery streams designed to operate directly on native-resolution document scans ($1080 \times 1527$ px) rather than downsampled document tensors:
1. **Stream A: OCR-Guided Dense Line Scanning**: Performs fine-grained, multi-scale sliding-window inspection along detected text lines.
2. **Stream B: OCR-Independent Dense Pixel Scanning**: Extracts multi-scale visual candidates based on local Laplacian high-frequency residuals, morphological edge gradient variance, and local contrast anomalies.

---

## 2. Stream Architecture & Methodology

### 2.1 Native-Resolution Requirement
Previous phases (Phases 1–6) processed document images at global downsampled resolutions ($224 \times 224$ or $512 \times 512$), which inevitably blurred sub-millimeter forgery seams, character kerning anomalies, and micro-digit alterations. 

In Phase 10:
* **Original Document Resolution:** $1080 \times 1527$ pixels.
* **Working Resolution:** Full native resolution ($1.0\times$).
* **Candidate Extraction:** Bounding box coordinates $[x_1, y_1, x_2, y_2]$ are defined directly in native pixel space.

### 2.2 Stream A: OCR-Guided Dense Line Scanning
For each text line extracted from the structured OCR transcription (`data/ocr/*.json`):
1. **Line Context Expansion:** The line bounding box is expanded by a vertical margin $m_v = 0.15 \times H_{\text{line}}$ and horizontal margin $m_h = 0.05 \times W_{\text{line}}$.
2. **Multi-Scale Windows:** Windows of height $H_{\text{win}} = s \cdot H_{\text{line}}$ and width $W_{\text{win}} = 1.6 \cdot H_{\text{win}}$ are evaluated across validation-tuned scales $s \in \{1.0, 1.5, 2.0\}$.
3. **Dense Sliding Stride:** Horizontal stride is set to $\Delta x = 0.5 \cdot W_{\text{win}}$, guaranteeing overlapping coverage across every word and digit.
4. **Candidate Budget:** Yields an average of **213.7 proposals per receipt**.

### 2.3 Stream B: OCR-Independent Dense Pixel Scanning
To recover micro-tamperings located outside recognized OCR lines (e.g., hand-written notes, logos, stamps, or text omitted by OCR engine failures):
1. **Multi-Scale Grid Decomposition:** Scans the document across window sizes $\{96 \times 96, 128 \times 128, 192 \times 192\}$ px with stride $0.65 \times W$.
2. **High-Frequency Residual Filtering:** Computes the Laplacian variance $\sigma^2(\nabla^2 I)$ to detect edge discontinuities and sharpening boundaries.
3. **Morphological Gradient Energy:** Computes morphological dilation minus erosion $(I \oplus K) - (I \ominus K)$ to identify anomalous stroke densities.
4. **Budget Pruning:** Retains the top-20 highest energy proposals per document (average **20.3 candidates per receipt**).

---

## 3. Quantitative Discovery Results

Evaluated on the held-out TEST benchmark partition ($N=148$ receipts, 86 official VIA ground-truth forgery annotations):

| Candidate Stream | Avg Candidates / Doc | GT Recall @ $\text{IoU} \ge 0.25$ | GT Recall @ $\text{IoU} \ge 0.50$ | Recovered GT Boxes | Missed GT Boxes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Phase 7/9 Baseline** | 48.2 | **39.53%** | **17.44%** | 34 / 86 | 52 |
| **Phase 10 Stream A (OCR Dense)** | 213.7 | 15.12% | 5.81% | 13 / 86 | 73 |
| **Phase 10 Stream B (Pixel Dense)** | 20.3 | 2.33% | 0.00% | 2 / 86 | 84 |
| **Stream A + Stream B Combined** | 221.8 | 15.12% | 5.81% | 13 / 86 | 73 |

---

## 4. Empirical Diagnosis & Failure Mode Analysis

### 4.1 The Dense Window Aspect-Ratio Mismatch (A2 Failure Mode)
Stream A was designed under the hypothesis that dense sliding windows along text lines would naturally encompass any forged character. However, empirical evaluation revealed a critical geometric disparity:
* **Ground-Truth Geometry:** In FindItAgain, annotators frequently marked *sub-character micro-boxes* (e.g., just the altered digit "3" in the string "13.50", dimensions $18 \times 28$ px, area $\approx 504\text{ px}^2$).
* **Candidate Window Geometry:** To capture line context, sliding windows have dimensions $H_{\text{win}} \approx 65\text{ px}$, $W_{\text{win}} \approx 104\text{ px}$ (area $\approx 6,760\text{ px}^2$).
* **Intersection over Union (IoU) Penalty:** Even when the sliding window perfectly centers on the altered digit, the maximum mathematical IoU is strictly upper-bounded:
  $$\text{IoU} = \frac{\text{Area}(\text{GT})}{\text{Area}(\text{Win})} = \frac{504}{6,760} \approx 0.075 \ll 0.25$$
* Consequently, **58.1% of all missed GT regions** (50 out of 86 boxes) are actually covered by the dense line windows, but are recorded as *misses* under the standard $\text{IoU} \ge 0.25$ criterion due to window-aspect mismatch.

### 4.2 Pixel Saliency Inefficacy in Digital Document Alterations
Stream B achieved only 2.33% recall (2 out of 86 boxes). Investigation of the false negatives revealed:
1. **Seamless Digital Paste:** Modern receipt fraud replaces characters with cleanly rendered digital vectors or anti-aliased font glyphs.
2. **Background Homogeneity:** Receipts have thermal paper or plain white background. Digital replacements do not produce sharp Laplacian energy spikes or high-frequency ringing artifacts compared to scanned photographic splices.
3. **Hard Negatives:** High-frequency pixel energy predominantly triggers on merchant logos, barcodes, decorative dividers, and paper creases, generating visual false alarms rather than forgery candidates.

---

## 5. Candidate Efficiency & Throughput

| Pipeline Stage | Processing Time (sec/doc) | Memory Footprint (RAM) | GPU Utilization |
| :--- | :---: | :---: | :---: |
| Native OCR Line Parsing | 0.02s | < 5 MB | 0.0% |
| Stream A Multi-Scale Sliding Windows | 0.18s | < 25 MB | 0.0% |
| Stream B Pixel Energy & Morphological Filter | 0.25s | < 45 MB | 0.0% |
| Non-Maximum Suppression & Fusion | 0.04s | < 10 MB | 0.0% |
| **Total Phase 10 Candidate Discovery** | **0.49s** | **< 60 MB** | **0.0%** |

Dense scanning is computationally efficient on standard CPU hardware (<0.5 seconds per receipt), but candidate budget alone cannot resolve the fundamental micro-box resolution mismatch without character-level glyph segmentation.
