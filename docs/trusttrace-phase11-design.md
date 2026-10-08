# TRUSTTRACE Phase 11: Architecture & Experimental Design
## Character-Level Glyph Localization & Native-Resolution Micro-Forensics

---

## 1. System Objective

Phase 11 directly addresses the dominant geometric bottleneck identified in Phase 10:
> **The A2 Failure Mode (58.1% of all GT misses):** Sub-character ground-truth forgery regions (median area $777.5\text{ px}^2$, aspect ratio $0.65$) could not achieve $\text{IoU} \ge 0.25$ against rectangular line windows ($W \approx 1.6 H$), even when the window physically enclosed the alteration.

Phase 11 transitions TRUSTTRACE from line-level and rectangular candidate sampling to **tight character-level glyph localization (Stream D)** and **native-resolution micro-forensic analysis**, operating at full document resolution without global downsampling.

---

## 2. Stream D: Glyph Candidate Generation Hierarchy

Because the underlying OCR engine (`Windows.Media.Ocr`) provides word- and line-level bounding boxes but lacks character-level segmentation (Level D1 unavailable), Stream D implements two deterministic localization levels:

```
[Native Receipt Scan (1080x1527 px)]
                │
                ├───► [Structured OCR Words & Lines (data/ocr/*.json)]
                │               │
                │               ├───► Level D2: Word-to-Character Decomposition
                │               │     ├── Horizontal projection & character interval slicing
                │               │     └── Tag: source = GLYPH_ESTIMATED_WORD
                │               │
                │               └───► Level D3: Connected-Component Contour Extraction
                │                     ├── Local adaptive thresholding inside word/line masks
                │                     ├── Contour bounding boxes filtered by receipt aspect ratio
                │                     └── Tag: source = GLYPH_CC
                │
                └───► Non-Maximum Suppression & Provenance Fusion
                                │
                                └───► Candidate Manifests (train, val, test, master)
```

### 2.1 Level D2: Word-to-Character Decomposition
For each OCR word $[x, y, w, h]$ with text string $S$ ($L = |S| \ge 1$):
* Base character width: $w_c = w / L$.
* Bounding box for character $i \in \{0, \dots, L-1\}$:
  $$x_{1, i} = x + i \cdot w_c, \quad y_{1, i} = y, \quad x_{2, i} = x_{1, i} + w_c, \quad y_{2, i} = y + h$$
* Character boundaries are refined within $\pm 20\%$ using vertical projection profiles of the grayscale gradient.
* Preserved metadata: `character_index = i`, `ocr_word_id = word_id`, `source = GLYPH_ESTIMATED_WORD`.

### 2.2 Level D3: Connected-Component Extraction
Within word bounding boxes and adjacent line regions:
* Apply local Otsu thresholding: $I_{\text{bin}} = I_{\text{gray}} < \theta_{\text{Otsu}}$.
* Extract contours $C_k$ and their minimum bounding rectangles $[x, y, w, h]$.
* Filter by receipt typography priors:
  $$10\text{ px} \le h \le 120\text{ px}, \quad 6\text{ px} \le w \le 100\text{ px}, \quad 0.15 \le \frac{w}{h} \le 1.8$$
* Tagged with `source = GLYPH_CC`, `character_index = UNKNOWN`.

### 2.3 Controlled Context Margins (G0, G1, G2, G3)
Each glyph proposal generates four geometry variants:
* **G0 (Exact Glyph):** Margin $m = 0.00$ (tightest character box).
* **G1 (Small Margin):** Margin $m = 0.10 \times [w, h]$ (captures character anti-aliasing edges).
* **G2 (Medium Margin):** Margin $m = 0.25 \times [w, h]$ (captures immediate digital paste seam).
* **G3 (Neighborhood Context):** Margin $m = 0.50 \times [w, h]$ (captures inter-character kerning and baseline context).

*Integrity Rule:* Variant selection and thresholds are tuned exclusively on TRAIN/VAL splits.

---

## 3. Geometric Discovery & Containment Metrics

To decouple geometric localization success from strict IoU penalty, Phase 11 evaluates four metrics against official VIA ground-truth annotations:

1. **GT Containment Recall:**
   $$\text{Containment}(C, GT) = \mathbb{I}(x_{1, C} \le x_{1, GT} \land y_{1, C} \le y_{1, GT} \land x_{2, C} \ge x_{2, GT} \land y_{2, C} \ge y_{2, GT})$$
2. **Recall @ $\text{IoU} \ge 0.10$:** Relaxed IoU measuring rough character alignment.
3. **Recall @ $\text{IoU} \ge 0.25$:** Primary historical benchmark metric.
4. **Recall @ $\text{IoU} \ge 0.50$:** Strict localization benchmark metric.

---

## 4. Native-Resolution Micro-Forensics & DCT Features

For each glyph candidate crop extracted at native resolution ($128 \times 128$ normalized patch):

### 4.1 Native 2D DCT Frequency Features
* Image crop is partitioned into $8 \times 8$ pixel non-overlapping blocks.
* Compute the 2D Discrete Cosine Transform (DCT) for each block:
  $$D(u, v) = \frac{1}{4} C(u) C(v) \sum_{x=0}^{7} \sum_{y=0}^{7} f(x, y) \cos\left[\frac{(2x+1)u\pi}{16}\right] \cos\left[\frac{(2y+1)v\pi}{16}\right]$$
* High-frequency coefficient energy ratio:
  $$E_{\text{HF}} = \frac{\sum_{u+v \ge 5} |D(u, v)|^2}{\sum_{u, v} |D(u, v)|^2 + \epsilon}$$
* DCT Energy Variance across the glyph patch: $\sigma^2(E_{\text{HF}})$.

### 4.2 Multi-Representation Suite
* **Rep A:** Raw RGB native crop.
* **Rep B:** Grayscale native crop.
* **Rep C:** High-pass residual: $I - \text{Gaussian}(I, \sigma=1.2)$.
* **Rep D:** Laplacian residual: $\nabla^2 I$.
* **Rep E:** DCT frequency feature map.
* **Rep F:** Multi-channel RGB + Laplacian residual.
* **Rep G:** Multi-channel RGB + DCT energy.

---

## 5. Local Glyph Neighbor Graph

Instead of global document statistics, Phase 11 models **local relational anomalies**:
* Graph nodes: Individual glyph candidates $g_i$.
* Graph edges: Connect $g_i, g_j$ if they belong to the same OCR word or are horizontal neighbors on the same line ($|y_i - y_j| < 0.5 \bar{H}$, $|x_i - x_j| < 2.0 \bar{W}$).
* Local relational features:
  * Height deviation from neighbors: $\Delta_h = |h_i - \text{median}(h_{\mathcal{N}(i)})| / \text{median}(h_{\mathcal{N}(i)})$
  * Stroke thickness deviation: $\Delta_{\text{stroke}}$
  * Intensity / darkness deviation: $\Delta_{\text{int}}$
  * DCT high-frequency energy deviation: $\Delta_{\text{dct}}$
* Deterministic Glyph Anomaly Score $G \in [0, 1]$:
  $$G = \min(1.0, 0.35 \Delta_h + 0.25 \Delta_{\text{stroke}} + 0.20 \Delta_{\text{int}} + 0.20 \Delta_{\text{dct}})$$

---

## 6. Downstream Evaluation Strategy

1. **Frozen Phase 8 Compact CNN Evaluation:**
   Pass glyph candidate crops through `models/phase8_patch_cnn_best.pt` with frozen Platt calibration ($a=1.0842, b=-1.1637$) to evaluate whether fine-grained spatial targeting alone improves patch classification.
2. **Lightweight Glyph Classifier (Trained on TRAIN only):**
   Train a shallow residual CNN on glyph crops from TRAIN, validate on VAL, and evaluate exactly once on the held-out TEST partition ($N=148$: 123 REAL, 25 EDITED).
3. **Hard-Case Recovery Set:**
   Measure recovery across the 52 Phase 9 Type-A missed ground-truth regions.
