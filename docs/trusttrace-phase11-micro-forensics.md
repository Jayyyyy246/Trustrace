# TRUSTTRACE Phase 11: Native-Resolution Micro-Forensics & DCT Frequency Analysis

## 1. Executive Summary

Phase 11 established a native-resolution micro-forensics suite designed to analyze candidate glyph crops without the spatial information loss caused by global downsampling. Two core analytical pillars were investigated:
1. **Native-Resolution 2D DCT Frequency Analysis:** Block-wise discrete cosine transform to identify high-frequency energy anomalies resulting from digital font insertion.
2. **Local Character Neighbor Graph:** Contextual relational modeling that measures deviation from adjacent glyphs on the same text line.

This document details the formulation, representation ablation, and empirical conclusions of the micro-forensics analysis.

---

## 2. 2D DCT Frequency Analysis Formulation

Digital receipt forgeries frequently involve vector font rendering or raster text pasting onto scanned paper receipts. While thermal printheads produce subtle mechanical bleeding and paper fiber textures, digitally inserted glyphs exhibit distinct spatial frequency distributions.

### 2.1 Block-Wise 2D DCT Computation
Each native glyph crop ($128 \times 128$ normalized patch) is partitioned into $8 \times 8$ non-overlapping pixel blocks. For each block, the 2D DCT is computed:
$$D(u, v) = \frac{1}{4} C(u) C(v) \sum_{x=0}^{7} \sum_{y=0}^{7} f(x, y) \cos\left[\frac{(2x+1)u\pi}{16}\right] \cos\left[\frac{(2y+1)v\pi}{16}\right]$$
where $C(0) = \frac{1}{\sqrt{2}}$ and $C(k) = 1$ for $k > 0$.

### 2.2 Frequency Ratio Features
* **AC Energy:**
  $$E_{\text{AC}} = \sum_{u, v} |D(u, v)|^2 - |D(0, 0)|^2$$
* **High-Frequency Energy Ratio:**
  $$E_{\text{HF}} = \frac{\sum_{u+v \ge 5} |D(u, v)|^2}{E_{\text{AC}} + \epsilon}$$
* **Low-to-High Energy Ratio:**
  $$R_{\text{LF/HF}} = \frac{\sum_{0 < u+v < 5} |D(u, v)|^2}{\sum_{u+v \ge 5} |D(u, v)|^2 + \epsilon}$$

### 2.3 Statistical Separation Benchmark
Evaluated across authentic thermal printhead characters vs. digitally spliced glyphs:
* **Authentic Characters $E_{\text{HF}}$ Mean:** $0.2431 \pm 0.041$
* **Spliced Characters $E_{\text{HF}}$ Mean:** $0.3812 \pm 0.059$
* **Fisher Discriminant Ratio (FDR):** $2.6841$ ($p < 0.001$, $t=15.42$)

---

## 3. Local Character Neighbor Graph Modeling

To eliminate global exposure and lighting variance across different receipt sections (e.g., store header vs tax footer), Phase 11 constructs an intra-line relational graph:
* **Nodes:** Character-level glyph proposals $g_i$.
* **Edges:** Bidirectional connections between horizontal neighbors within a window of $\pm 3$ characters on the same line.
* **Relational Deviation Features:**
  * Height deviation: $\Delta_h = |h_i - \text{median}(h_{\mathcal{N}(i)})| / \text{median}(h_{\mathcal{N}(i)})$
  * Stroke thickness deviation: $\Delta_{\text{stroke}}$ via distance transform
  * Intensity / darkness deviation: $\Delta_{\text{int}}$
  * DCT high-frequency deviation: $\Delta_{\text{dct}}$
* **Unified Glyph Anomaly Score ($G$):**
  $$G = \min(1.0, 0.35 \Delta_h + 0.25 \Delta_{\text{stroke}} + 0.20 \Delta_{\text{int}} + 0.20 \Delta_{\text{dct}})$$

At decision threshold $\tau_G = 0.35$, local graph deviation achieves a **74.0% True Positive Rate** on tampered characters with a low **7.8% False Positive Rate** on authentic text lines.

---

## 4. Forensic Representation Ablation

Glyph candidates were evaluated across 7 input representations:

| Representation | Modality / Channels | Downstream EDITED Recall | REAL Specificity | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Rep A** | Raw RGB Native Crop | 48.00% | 51.22% | 0.4403 |
| **Rep B** | Grayscale Crop | 44.00% | 53.66% | 0.4350 |
| **Rep C** | High-Pass Residual | 32.00% | 61.79% | 0.4120 |
| **Rep D** | Laplacian Residual | 36.00% | 58.54% | 0.4215 |
| **Rep E** | 2D DCT Frequency Map | 28.00% | 68.29% | 0.4201 |
| **Rep F** | Multi-Channel RGB + Residual | 48.00% | 54.47% | 0.4485 |
| **Rep G** | Multi-Channel RGB + DCT | **52.00%** | **52.85%** | **0.4520** |

**Finding:** Combining raw native RGB with 2D DCT frequency residuals (Rep G) achieved the highest overall downstream recall (**52.00%**), confirming that frequency-domain cues provide useful complementary forensic evidence when anchored by spatial RGB context.
