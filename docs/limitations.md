# TRUSTTRACE Forensic Limitations & Operational Boundaries

**Document Version:** 1.0.0  
**Compliance Standards:** Daubert Standard (*Known or Potential Rate of Error*), ISO/IEC 27037:2012, Federal Rules of Evidence 702  
**Purpose:** Comprehensive, transparent documentation of technical constraints, environmental assumptions, edge-case failure modes, and judicial boundaries.

---

## 1. Executive Summary & Epistemic Humility

Digital forensics is an empirical discipline constrained by the physical laws of signal degradation, lossy data compression, and information entropy. A core principle of TRUSTTRACE is **epistemic humility**: the platform explicitly disclaims false omniscience and refuses to fabricate certainty when underlying physical signals have been corrupted or destroyed.

When evidence is degraded, TRUSTTRACE returns the standardized finding:
> *"The available evidence was insufficient to determine authenticity."*

---

## 2. Technical & Signal Processing Constraints

### 2.1. Social Media & Messaging Platform Recompression
- **Mechanism:** Platforms such as WhatsApp, Telegram, Facebook, Instagram, and X (formerly Twitter) re-encode uploaded media using aggressive, non-standard lossy quantization profiles ($Q \le 60$), downscale spatial resolutions, and strip all EXIF/TIFF container headers.
- **Forensic Impact:**
  1. High-frequency DCT coefficients that carry micro-scale sensor noise (PRNU) are attenuated or discarded.
  2. Multi-generation compression creates heavy 8×8 block boundary artifacts that mimic localized editing in standard Error Level Analysis.
  3. Metadata analysis cannot determine original camera provenance from stripped containers.
- **System Mitigation:** The Evidence Decision Engine suppresses ELA manipulation warnings when the estimated JPEG quality factor drops below 65 (`RULE-FUSE-04`), falling back to `UNKNOWN` to prevent false-positive tampering accusations.

### 2.2. Dimensional Extremes & Micro-Images
- **Micro-Images ($< 32 \times 32$ pixels):**
  - Insufficient spatial support for scale-space pyramid construction required by SIFT/ORB feature extractors.
  - Median filtering and 8×8 block DCT lattice operations become mathematically degenerate.
  - The system returns neutral baseline metrics and logs dimension limitations.
- **Ultra-High Resolution Assets ($> 100$ MP):**
  - Requires substantial memory and compute allocations.
  - Guardrails enforce decompression safety ceilings (Pillow `MAX_IMAGE_PIXELS`) to prevent denial-of-service memory exhaustion.

### 2.3. Flat, Textureless, and Monochromatic Imagery
- **Homogeneous Surfaces:** Clean sky gradients, monochromatic studio backdrops, and solid-color graphic banners lack local luminance variation.
- **Forensic Impact:**
  - Zero keypoint descriptors can be extracted, rendering copy-move detection inactive.
  - ELA residual variance is artificially suppressed ($\sigma^2_{\text{ELA}} \approx 0$).
  - Laplacian focus variance registers near zero ($< 1.0$), mimicking motion blur.
- **System Mitigation:** The engine marks textureless assets as `AMBIGUOUS / INCONCLUSIVE` rather than asserting authentic capture.

---

## 3. Metadata & Provenance Constraints

### 3.1. Absence of Metadata is Not Evidence of Tampering
The complete absence of camera metadata (EXIF/JFIF/TIFF) is standard across modern web distribution pipelines.
- **Judicial Rule:** The platform **refuses to treat metadata omission as evidence of fraud**.
- In the absence of metadata, the system relies exclusively on intrinsic pixel-level signal metrics.

### 3.2. Donor Metadata Splicing & Tampered Containers
Adversarial actors can extract intact, authentic EXIF headers from an authentic optical camera photograph and inject them into a synthesized or manipulated image.
- **Forensic Impact:** Container inspection alone would falsely validate the asset as camera-original.
- **System Mitigation:** The Evidence Decision Engine evaluates cross-modal conflict rules (`RULE-FUSE-03`). If authentic hardware metadata conflicts with physical copy-move keypoint clusters or severe ELA disparities, the engine flags container tampering and prioritizes physical pixel evidence.

---

## 4. Machine Learning & Generative AI Boundaries

### 4.1. The Out-of-Distribution (OOD) Challenge
Deep learning classifiers generalize only across visual distributions representative of their training corpora.
- **Failure Mode:** Novel generative AI architectures (e.g., future diffusion iterations or novel GAN architectures) introduce unfamiliar latent artifacts that can cause uncalibrated neural networks to emit high-confidence false predictions.
- **System Mitigation:** TRUSTTRACE monitors Monte Carlo Dropout Shannon entropy. When predictive entropy breaches $H > 0.40$, the ML signal is automatically rejected and forced to `UNKNOWN`.

### 4.2. Refusal of Simulated Predictions
When trained model weights are not loaded in the active runtime, TRUSTTRACE explicitly refuses to simulate probabilistic outputs. The ML analyzer reports `model_status: NOT_AVAILABLE` and `predicted_label: UNKNOWN`.

---

## 5. Judicial & Evidentiary Boundaries

### 5.1. Technical Observation vs. Human Intent
Under Federal Rule of Evidence 702 and the Daubert standard:
- TRUSTTRACE generates **technical observations** (e.g., pixel variance, quantization tables, feature displacement vectors).
- Technical observations **do not prove legal intent, willful fabrication, or guilt**. A software signature (`Adobe Photoshop`) indicates that post-capture processing occurred; it does not prove malicious intent (it could reflect neutral color correction, resizing, or cropping).
- All reports mandate human forensic expert review before submission in legal proceedings.

### 5.2. Chain of Custody Pre-requisite
TRUSTTRACE cryptographically certifies that an ingested file has remained bitwise invariant from the exact moment of ingestion. It cannot certify the physical history of an asset prior to upload without external hardware provenance (e.g., C2PA cryptographic sensor signatures).
