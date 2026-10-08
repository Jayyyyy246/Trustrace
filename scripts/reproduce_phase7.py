#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 7 Reproducibility Suite.

Reproduces all Phase 7 experiments and validates all generated artifacts:
1. Validates dataset integrity and zero group/sample/hash leakage across splits
2. Audits official VIA ground-truth forgery annotations vs OCR candidate coverage
3. Generates multi-scale morphological and visual saliency candidate pools
4. Merges and deduplicates candidate manifests (OCR, Morphology, Combined)
5. Benchmarks candidate coverage at IoU >= 0.10, 0.25, 0.50 and candidate budgets
6. Evaluates frozen Doc-PatchFormer on document-level authenticity pipelines
7. Generates all required markdown reports, JSON metrics, and figures
8. Supports --check-only mode for instant CI/CD integrity verification.
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
    DOCS_DIR / "trusttrace-phase7-design.md",
    DOCS_DIR / "trusttrace-phase7-feasibility.md",
    DOCS_DIR / "trusttrace-phase7-candidate-discovery.md",
    DOCS_DIR / "trusttrace-phase7-final.md",
    MANIFESTS_DIR / "phase7_candidates_master.csv",
    MANIFESTS_DIR / "phase7_candidates_train.csv",
    MANIFESTS_DIR / "phase7_candidates_val.csv",
    MANIFESTS_DIR / "phase7_candidates_test.csv",
    REPORTS_DIR / "phase7_annotation_coverage.json",
    REPORTS_DIR / "phase7_candidate_coverage.json",
    REPORTS_DIR / "phase7_candidate_coverage.md",
    REPORTS_DIR / "phase7_document_test.json",
    REPORTS_DIR / "phase7_document_test.md",
    REPORTS_DIR / "phase7_error_analysis.md",
    REPORTS_DIR / "phase7_confusion_matrix.png",
    REPORTS_DIR / "phase7_candidate_coverage.png",
    REPORTS_DIR / "phase7_budget_recall.png",
    REPORTS_DIR / "phase7_candidate_distribution.png",
    REPORTS_DIR / "phase7_ocr_recovery.png",
    REPORTS_DIR / "phase7_examples.png",
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
    print("TRUSTTRACE: Phase 7 Reproducibility Suite")
    print("=" * 70)

    env = verify_environment()
    print("Software & Hardware Environment:")
    for k, v in env.items():
        print(f"  {k:24s}: {v}")

    # Check-only mode
    if "--check-only" in sys.argv:
        print("\nVerifying all Phase 7 required artifacts on disk:")
        missing = []
        for a in REQUIRED_ARTIFACTS:
            if a.is_file():
                print(f"  [OK] {a.name:36s} ({a.stat().st_size:,} bytes)")
            else:
                print(f"  [MISSING] {a.name}")
                missing.append(a)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} Phase 7 artifacts: {[m.name for m in missing]}")
        print("\n[OK] Check-only mode passed successfully! All Phase 7 artifacts verified.")
        return

    # Step 1: Split Validation
    print("\n[Step 1/5] Running Split Integrity Validation...")
    from scripts.validate_phase7_dataset import validate_splits
    validate_splits()

    # Step 2: Annotation Coverage Audit
    print("\n[Step 2/5] Running Ground-Truth & OCR Coverage Audit...")
    from scripts.audit_phase7_annotation_coverage import audit_coverage
    audit_coverage()

    # Step 3: Candidate Generation & Merging
    print("\n[Step 3/5] Running Candidate Generation & Manifest Merging...")
    from scripts.merge_candidate_sources import generate_all_candidates
    generate_all_candidates()

    # Step 4: Candidate Recall & Budget Benchmark
    print("\n[Step 4/5] Evaluating Candidate Recall & Budget Curves...")
    from scripts.evaluate_phase7_candidate_recall import benchmark_coverage
    benchmark_coverage()

    # Step 5: Document Pipeline Evaluation
    print("\n[Step 5/5] Evaluating Document-Level Authenticity Pipelines...")
    from scripts.evaluate_phase7_document_pipeline import evaluate_document_pipelines
    evaluate_document_pipelines()

    # Final artifact verification
    print("\nVerifying all generated artifacts:")
    for a in REQUIRED_ARTIFACTS:
        assert a.is_file(), f"Missing artifact: {a}"
        print(f"  [OK] {a.name:36s} ({a.stat().st_size:,} bytes)")

    print("\n" + "=" * 70)
    print("PHASE 7 REPRODUCIBILITY SUITE PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
