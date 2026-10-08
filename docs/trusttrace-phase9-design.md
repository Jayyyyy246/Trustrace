# TRUSTTRACE — PHASE 9: SYSTEM DESIGN & METHODOLOGY
## Candidate Quality Scoring, Region-Context Modeling & Uncertainty-Aware Evidence Fusion

**Document Identifier:** `TRUSTTRACE-DOC-P9-DESIGN-001`  
**Security / Forensics Research Classification:** Empirical Cyber-Forensics Research  
**Date:** October 2026  
**Status:** ACTIVE DESIGN BLUEPRINT  
**Target Focus:** Bridging Patch Forensics to Document-Level Verifiable Authenticity  

---

### Non-Negotiable Project Integrity Statement
> **Phase 1–8 artifacts are treated as frozen historical baselines and must not be modified during Phase 9.**  
> - Checkpoints `models/doc_patchformer_best.pt`, `models/phase8_patch_cnn_best.pt`, and `models/phase8_candidate_aware_docpatchformer_best.pt` remain strictly untouched.  
> - Official VIA annotations remain the sole authoritative ground-truth labels for forgery regions.  
> - Candidate sources (`OCR` vs `MORPHOLOGY`) are provenance metadata, not ground-truth labels.  
> - No held-out TEST annotations may be used for hyperparameter tuning, threshold selection, quality parameter optimization, or edge distance thresholds.  
> - All parameters, thresholds, and weights are tuned on TRAIN/VAL only; the held-out TEST partition ($N=148$ receipts) remains quarantined until final verification.

---

## 1. Research Motivation: The Aggregation Bottleneck

Phase 8 established that retraining patch classifiers on the empirical candidate distribution with hard authentic visual negatives effectively cures the patch-level false-positive collapse:
- Morphology False-Positive Rate (FPR) dropped from **88.65%** down to **11.35%** with the Compact CNN.
- Expected Calibration Error (ECE) dropped from **0.4508** down to **0.1980**.

However, document-level authenticity decisions still accumulate errors under naive Top-K Mean pooling:
- On a document containing 30 candidate patches, even a low $11\%$ patch-level false-positive rate yields a high probability ($\sim 1 - (1 - 0.11)^{30} \approx 97\%$) that at least 1 or 2 patches false-alarm.
- When Top-3 Mean pooling is applied, if 3 out of 30 candidate patches produce spurious high probabilities on high-contrast logos or wrinkles, the entire receipt is flagged as `EDITED`.
- Furthermore, naive Top-K Mean treats every candidate equally, ignoring:
  1. **Candidate Quality:** Is the candidate an informative text/edge crop or an uninformative blurry background sliver?
  2. **Model Uncertainty:** Is the classifier highly confident, or is the prediction near the decision boundary ($\approx 0.5$)?
  3. **Spatial Context:** Is the candidate an isolated high-scoring artifact (likely noise) or part of a coherent cluster of tampered words/tokens (genuine tampering)?

Phase 9 addresses this fundamental evidence-aggregation bottleneck.

---

## 2. Mathematical Formalization

Let a receipt document $\mathcal{D}_i$ be represented by a set of candidate regions:
$$\mathcal{C}_i = \{c_{i,1}, c_{i,2}, \dots, c_{i,M}\}$$
where each candidate $c_{i,j}$ has bounding box coordinates $\mathbf{b}_{i,j} = (x_1, y_1, x_2, y_2)$, source provenance $s_{i,j} \in \{\text{OCR}, \text{MORPHOLOGY}\}$, and extracted patch image $\mathbf{x}_{i,j} \in \mathbb{R}^{128 \times 128 \times 3}$.

### 2.1 Candidate Quality Scoring ($Q$)
We define a deterministic quality scoring function $Q(c) \in [0, 1]$ independent of the forgery label:
$$Q(c) = w_{\text{area}} f_{\text{area}}(c) + w_{\text{aspect}} f_{\text{aspect}}(c) + w_{\text{edge}} f_{\text{edge}}(c) + w_{\text{contrast}} f_{\text{contrast}}(c) + w_{\text{context}} f_{\text{context}}(c)$$
where:
- $f_{\text{area}}(c)$: Penalizes patches that are too small ($<20 \times 20$ px) or excessively large ($>400 \times 400$ px).
- $f_{\text{aspect}}(c)$: Penalizes extreme aspect ratios ($w/h > 8$ or $h/w > 8$).
- $f_{\text{edge}}(c)$: Normalized Laplacian variance $\min(1.0, \sigma^2_{\text{Laplace}} / 500.0)$, rejecting flat, uninformative regions.
- $f_{\text{contrast}}(c)$: Standard deviation of grayscale pixel intensities, ensuring forensic signal exists.
- $f_{\text{context}}(c)$: Contextual overlap with other candidate boxes or OCR words.

### 2.2 Calibrated Probability ($p_{\text{cal}}$) & Uncertainty ($U$)
For raw model logit $z_{i,j}$, the temperature-scaled calibrated probability is:
$$p_{\text{cal}}(c_{i,j}) = \sigma(z_{i,j} / T)$$
where $T > 0$ is the optimal temperature parameter determined on the validation set.

We evaluate three uncertainty formulations:
- **Margin Uncertainty (Method A):** $U_{\text{margin}}(p) = 1 - |2p - 1| \in [0, 1]$ (Maximum uncertainty at $p=0.5$).
- **Normalized Entropy (Method B):** $U_{\text{entropy}}(p) = \frac{-p \log_2(p) - (1-p) \log_2(1-p)}{1.0} \in [0, 1]$.
- **Calibration Distance (Method C):** Uncertainty scaled by reliability bin error.

Confidence is defined as $C(p) = 1 - U(p)$.

### 2.3 Spatial Candidate Graph & Clustering
For document $\mathcal{D}_i$, we construct an undirected spatial graph $\mathcal{G}_i = (\mathcal{V}_i, \mathcal{E}_i)$:
- **Nodes $\mathcal{V}_i$:** The candidate set $\mathcal{C}_i$.
- **Edges $\mathcal{E}_i$:** An edge $(c_a, c_b) \in \mathcal{E}_i$ exists if:
  $$\text{dist}_{\text{center}}(c_a, c_b) \le R_{\text{spatial}} \quad \text{or} \quad \text{IoU}(c_a, c_b) \ge \tau_{\text{overlap}}$$
  where $R_{\text{spatial}}$ is normalized by receipt diagonal.

Connected components or DBSCAN in spatial coordinate space define clusters:
$$\mathcal{K}_i = \{K_{i,1}, K_{i,2}, \dots, K_{i,L}\}$$
For each cluster $K$, we calculate:
- Cluster Size: $|K|$
- Maximum / Mean Calibrated Probability: $\max_{c \in K} p_{\text{cal}}(c)$, $\bar{p}_K$
- Cluster Spatial Density: $\rho(K)$
- Cluster Evidence Support: $S(K) = \sum_{c \in K} p_{\text{cal}}(c) \cdot Q(c) \cdot C(c)$

### 2.4 Evidence Fusion Formulations
We evaluate an 8-way ablation matrix:
1. **Model A (Naive Top-K Mean):** $\mathcal{S}_A = \frac{1}{K} \sum_{j=1}^K p_{(j)}$
2. **Model B (Quality-Weighted Top-K):** $\mathcal{S}_B = \frac{\sum_{j=1}^K Q_{(j)} p_{(j)}}{\sum_{j=1}^K Q_{(j)}}$
3. **Model C (Uncertainty-Weighted Top-K):** $\mathcal{S}_C = \frac{\sum_{j=1}^K (1 - U_{(j)}) p_{(j)}}{\sum_{j=1}^K (1 - U_{(j)})}$
4. **Model D (Spatial-Cluster Aggregation):** $\mathcal{S}_D = \max_{K \in \mathcal{K}} \bar{p}_K \cdot \min(1.0, \log_2(|K| + 1))$
5. **Model E (Quality + Uncertainty):** $\mathcal{S}_E = \frac{\sum_{j=1}^K Q_{(j)} (1 - U_{(j)}) p_{(j)}}{\sum_{j=1}^K Q_{(j)} (1 - U_{(j)})}$
6. **Model F (Quality + Spatial Context):** Quality-weighted cluster scoring.
7. **Model G (Uncertainty + Spatial Context):** Confidence-weighted cluster scoring.
8. **Model H (Full Evidence Fusion):** Multi-factor evidence combining quality, confidence, and spatial cluster coherence.

---

## 3. Systematic Execution Roadmap

1. **Step 1 — Feasibility & Candidate Quality Audit:**
   - Execute `scripts/audit_phase9_candidate_quality.py`.
   - Document distributions in `docs/trusttrace-phase9-feasibility.md`.
2. **Step 2 — Candidate Feature Extraction:**
   - Execute `scripts/extract_phase9_candidate_features.py`.
   - Output `data/manifests/phase9_candidate_features_*.csv`.
3. **Step 3 — Probability Calibration:**
   - Execute `scripts/calibrate_phase9_probabilities.py`.
   - Fit temperature scaling and Platt scaling on validation set only.
   - Save metrics to `reports/phase9_calibration.json`.
4. **Step 4 — Spatial Graph & Clustering:**
   - Execute `scripts/build_phase9_candidate_graph.py`.
   - Construct adjacency graphs and cluster structures for train, val, and test.
5. **Step 5 — Aggregation Ablation & Budget Optimization:**
   - Execute `scripts/evaluate_phase9_aggregation.py`.
   - Evaluate Top-K budgets ($K \in [1, 3, 5, 10, 20, 30, 50]$) and fusion models A–H on VALIDATION only.
   - Select best configuration and lock operating threshold.
6. **Step 6 — Final Held-Out Benchmark & Test Evaluation:**
   - Execute `scripts/evaluate_phase9_document_pipeline.py`.
   - Evaluate on $N=148$ test receipts.
   - Generate all research figures and error taxonomy.
7. **Step 7 — Synthesis & Reporting:**
   - Create `docs/trusttrace-phase9-candidate-quality.md` and `docs/trusttrace-phase9-final.md`.
   - Validate reproduction suite via `scripts/reproduce_phase9.py`.
   - Commit and push to remote.

*TRUSTTRACE Research Team — Phase 9 Blueprint*
