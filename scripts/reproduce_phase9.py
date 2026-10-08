#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 9 Master Reproducibility Suite.

Reproduces all Phase 9 experiments and verifies all generated artifacts:
1. Candidate Quality Feasibility Audit (scripts/audit_phase9_candidate_quality.py)
2. Comprehensive Candidate Feature Extraction (scripts/extract_phase9_candidate_features.py)
3. Validation Probability Calibration (scripts/calibrate_phase9_probabilities.py)
4. Spatial Candidate Graph & Clustering (scripts/build_phase9_candidate_graph.py)
5. Validation Aggregation Ablation & Budget Sweep (scripts/evaluate_phase9_aggregation.py)
6. Document-Level Authenticity Benchmark & Figure Generation (scripts/evaluate_phase9_document_pipeline.py)
7. Forensic Error Analysis Deep Dive (scripts/analyze_phase9_errors.py)
8. Supports --check-only mode for instant verification of all 25+ artifacts.
"""

import sys
import os
import json
import time
import platform
import hashlib
from typing import Dict, Any, List
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"
REPORTS_DIR = WORKSPACE_DIR / "reports"
DOCS_DIR = WORKSPACE_DIR / "docs"

REQUIRED_ARTIFACTS = [
    DOCS_DIR / "trusttrace-phase9-design.md",
    DOCS_DIR / "trusttrace-phase9-feasibility.md",
    DOCS_DIR / "trusttrace-phase9-candidate-quality.md",
    DOCS_DIR / "trusttrace-phase9-final.md",
    MANIFESTS_DIR / "phase9_candidate_features_master.csv",
    MANIFESTS_DIR / "phase9_candidate_features_train.csv",
    MANIFESTS_DIR / "phase9_candidate_features_val.csv",
    MANIFESTS_DIR / "phase9_candidate_features_test.csv",
    REPORTS_DIR / "phase9_candidate_quality.json",
    REPORTS_DIR / "phase9_candidate_quality.md",
    REPORTS_DIR / "phase9_calibration.json",
    REPORTS_DIR / "phase9_spatial_clustering.json",
    REPORTS_DIR / "phase9_aggregation_ablation.json",
    REPORTS_DIR / "phase9_aggregation_ablation.md",
    REPORTS_DIR / "phase9_document_test.json",
    REPORTS_DIR / "phase9_document_test.md",
    REPORTS_DIR / "phase9_error_analysis.md",
    REPORTS_DIR / "phase9_fig1_quality_distribution.png",
    REPORTS_DIR / "phase9_fig2_quality_vs_probability.png",
    REPORTS_DIR / "phase9_fig3_calibration_curve.png",
    REPORTS_DIR / "phase9_fig4_uncertainty_distribution.png",
    REPORTS_DIR / "phase9_fig5_spatial_graph_examples.png",
    REPORTS_DIR / "phase9_fig6_cluster_size_distribution.png",
    REPORTS_DIR / "phase9_fig7_cluster_precision.png",
    REPORTS_DIR / "phase9_fig8_topk_budget_curve.png",
    REPORTS_DIR / "phase9_fig9_aggregation_comparison.png",
    REPORTS_DIR / "phase9_fig10_false_positive_suppression.png",
    REPORTS_DIR / "phase9_fig11_false_negative_decomposition.png",
    REPORTS_DIR / "phase9_fig12_confusion_matrices.png",
]


def compute_file_sha256(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_environment() -> Dict[str, Any]:
    import torch
    import cv2
    import sklearn
    import PIL

    cnn_path = MODELS_DIR / "phase8_patch_cnn_best.pt"
    env_info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pillow_version": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
        "phase8_cnn_sha256": compute_file_sha256(cnn_path),
        "random_seed": 42,
        "calibration_method": "Platt Scaling / Isotonic Regression",
        "selected_budget_K": 3,
        "decision_threshold_tau": 0.45,
    }
    return env_info


def main():
    print("=" * 70)
    print("TRUSTTRACE: Phase 9 Master Reproducibility Suite")
    print("=" * 70)

    env = verify_environment()
    print("Software & Hardware Environment:")
    for k, v in env.items():
        print(f"  {k:24s}: {v}")
    print()

    if "--check-only" in sys.argv:
        print("Verifying all Phase 9 required artifacts on disk:")
        missing = []
        for a in REQUIRED_ARTIFACTS:
            if a.is_file():
                sz = a.stat().st_size
                print(f"  [OK] {a.name:42s} ({sz:,} bytes)")
            else:
                print(f"  [MISSING] {a.name}")
                missing.append(a)

        if missing:
            print(f"\n[FAIL] {len(missing)} required Phase 9 artifacts missing!")
            sys.exit(1)
        else:
            print("\n[OK] Check-only mode passed successfully! All Phase 9 artifacts verified.")
            sys.exit(0)

    # Full Reproduction Workflow
    t_start = time.time()
    from scripts.audit_phase9_candidate_quality import run_candidate_quality_audit
    from scripts.extract_phase9_candidate_features import extract_candidate_features
    from scripts.calibrate_phase9_probabilities import calibrate_probabilities
    from scripts.build_phase9_candidate_graph import analyze_spatial_clustering
    from scripts.evaluate_phase9_aggregation import run_aggregation_ablation
    from scripts.evaluate_phase9_document_pipeline import evaluate_phase9_document_pipeline
    from scripts.analyze_phase9_errors import run_error_analysis

    print("\n--- Step 1: Candidate Quality Feasibility Audit ---")
    run_candidate_quality_audit()

    print("\n--- Step 2: Extract Candidate Features ---")
    extract_candidate_features()

    print("\n--- Step 3: Probability Calibration ---")
    calibrate_probabilities()

    print("\n--- Step 4: Spatial Graph & Clustering Analysis ---")
    analyze_spatial_clustering()

    print("\n--- Step 5: Aggregation Ablation on Validation ---")
    run_aggregation_ablation()

    print("\n--- Step 6: Test Benchmark & Figure Generation ---")
    evaluate_phase9_document_pipeline()

    print("\n--- Step 7: Forensic Error Analysis ---")
    run_error_analysis()

    total_time = time.time() - t_start
    print(f"\n[OK] Phase 9 full reproduction finished in {total_time:.1f}s.")


if __name__ == "__main__":
    main()
