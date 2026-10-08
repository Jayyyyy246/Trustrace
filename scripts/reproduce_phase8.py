#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 8 Master Reproducibility Suite.

Reproduces all Phase 8 experiments and validates all generated artifacts:
1. Audits candidate training pool and mathematical labeling protocol
2. Constructs candidate-aware patch datasets with hard authentic visual negatives
3. Validates group-aware split integrity and zero cross-partition leakage
4. Performs quantitative domain shift analysis between Phase 6 and Phase 7/8
5. Mines and audits hard authentic negatives
6. Trains Compact CNN baseline (models/phase8_patch_cnn_best.pt)
7. Trains Candidate-Aware Doc-PatchFormer (models/phase8_candidate_aware_docpatchformer_best.pt)
8. Evaluates patch-level benchmark, FPR reduction on morphology, and calibration
9. Evaluates document-level authenticity pipelines and two-stage funnels
10. Supports --check-only mode for instant integrity verification.
"""

import sys
import os
import json
import time
import platform
import csv
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
    DOCS_DIR / "trusttrace-phase8-design.md",
    DOCS_DIR / "trusttrace-phase8-feasibility.md",
    DOCS_DIR / "trusttrace-phase8-candidate-aware-forensics.md",
    DOCS_DIR / "trusttrace-phase8-final.md",
    MANIFESTS_DIR / "phase8_patch_master.csv",
    MANIFESTS_DIR / "phase8_patch_train.csv",
    MANIFESTS_DIR / "phase8_patch_val.csv",
    MANIFESTS_DIR / "phase8_patch_test.csv",
    MODELS_DIR / "phase8_patch_cnn_best.pt",
    MODELS_DIR / "phase8_candidate_aware_docpatchformer_best.pt",
    REPORTS_DIR / "phase8_training_pool_audit.json",
    REPORTS_DIR / "phase8_domain_shift.json",
    REPORTS_DIR / "phase8_domain_shift.md",
    REPORTS_DIR / "phase8_patch_metrics.json",
    REPORTS_DIR / "phase8_patch_metrics.md",
    REPORTS_DIR / "phase8_calibration.json",
    REPORTS_DIR / "phase8_document_test.json",
    REPORTS_DIR / "phase8_document_test.md",
    REPORTS_DIR / "phase8_error_analysis.md",
    REPORTS_DIR / "phase8_fig1_source_distribution.png",
    REPORTS_DIR / "phase8_fig2_label_distribution.png",
    REPORTS_DIR / "phase8_fig3_patch_performance.png",
    REPORTS_DIR / "phase8_fig4_morphology_fp_examples.png",
    REPORTS_DIR / "phase8_fig5_hard_negatives_composition.png",
    REPORTS_DIR / "phase8_fig6_roc_curves.png",
    REPORTS_DIR / "phase8_fig7_pr_curves.png",
    REPORTS_DIR / "phase8_fig8_calibration_curves.png",
    REPORTS_DIR / "phase8_fig9_confusion_matrices.png",
    REPORTS_DIR / "phase8_fig10_two_stage_funnel.png",
    REPORTS_DIR / "phase8_fig11_feature_comparison.png",
    REPORTS_DIR / "phase8_fig12_threshold_sweep.png",
]


def verify_environment() -> Dict[str, Any]:
    import torch
    import cv2
    import sklearn
    import PIL

    env_info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pillow_version": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    return env_info


def main():
    print("=" * 70)
    print("TRUSTTRACE: Phase 8 Master Reproducibility Suite")
    print("=" * 70)

    env = verify_environment()
    print("Software & Hardware Environment:")
    for k, v in env.items():
        print(f"  {k:24s}: {v}")

    # Check-only mode
    if "--check-only" in sys.argv:
        print("\nVerifying all Phase 8 required artifacts on disk:")
        missing = []
        for a in REQUIRED_ARTIFACTS:
            if a.is_file():
                print(f"  [OK] {a.name:42s} ({a.stat().st_size:,} bytes)")
            else:
                print(f"  [MISSING] {a.name}")
                missing.append(a)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} Phase 8 artifacts: {[m.name for m in missing]}")
        print("\n[OK] Check-only mode passed successfully! All Phase 8 artifacts verified.")
        return

    # Step 1: Feasibility Audit
    print("\n[Step 1/8] Running Training Pool Feasibility Audit...")
    from scripts.audit_phase8_training_pool import audit_candidate_pool
    audit_candidate_pool()

    # Step 2: Dataset Construction
    print("\n[Step 2/8] Creating Candidate-Aware Patch Dataset...")
    from scripts.create_phase8_patch_dataset import create_phase8_dataset
    create_phase8_dataset()

    # Step 3: Split Validation
    print("\n[Step 3/8] Validating Patch Split Integrity...")
    from scripts.validate_phase8_patch_split import validate_patch_splits
    validate_patch_splits()

    # Step 4: Domain Shift Analysis
    print("\n[Step 4/8] Running Domain Shift Analysis...")
    from scripts.analyze_phase8_domain_shift import analyze_domain_shift
    analyze_domain_shift()

    # Step 5: Hard Negative Mining
    print("\n[Step 5/8] Mining Hard Negatives...")
    from scripts.mine_phase8_hard_negatives import mine_hard_negatives
    mine_hard_negatives()

    # Step 6: Train Models
    print("\n[Step 6/8] Training Models...")
    from scripts.train_phase8_cnn import train_cnn
    train_cnn()
    from scripts.train_phase8_candidate_aware_docpatchformer import train_candidate_aware_docpatchformer
    train_candidate_aware_docpatchformer()

    # Step 7: Patch Evaluation
    print("\n[Step 7/8] Evaluating Patch Models...")
    from scripts.evaluate_phase8_patch_models import evaluate_patch_models
    evaluate_patch_models()

    # Step 8: Document Pipeline Evaluation
    print("\n[Step 8/8] Evaluating Document Pipelines...")
    from scripts.evaluate_phase8_document_pipeline import evaluate_document_pipeline
    evaluate_document_pipeline()

    # Final Verification
    print("\nVerifying all generated artifacts:")
    for a in REQUIRED_ARTIFACTS:
        assert a.is_file(), f"Missing artifact: {a}"
        print(f"  [OK] {a.name:42s} ({a.stat().st_size:,} bytes)")

    print("\n" + "=" * 70)
    print("PHASE 8 REPRODUCIBILITY SUITE PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
