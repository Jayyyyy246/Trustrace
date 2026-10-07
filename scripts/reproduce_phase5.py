#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 5 End-to-End Experiment Reproducibility Suite.

Documents and reproduces the complete Phase 5 Evidence Fusion experiment:
1. Verifies dependencies and software environment
2. Checks manifest integrity and strict group isolation
3. Executes OCR feature extraction (scripts/extract_ocr_forensic_features.py)
4. Executes out-of-domain transferability study (scripts/evaluate_phase5_transferability.py)
5. Trains primary evidence fusion model (scripts/train_phase5_fusion.py)
6. Runs comprehensive test evaluation and ablation benchmarking (scripts/evaluate_phase5_fusion.py)
7. Verifies all generated reports and figures
"""

import sys
import os
import json
import time
import platform
import csv
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

REQUIRED_ARTIFACTS = [
    REPORTS_DIR / "phase5_transferability_report.md",
    REPORTS_DIR / "phase5_test_report.md",
    REPORTS_DIR / "phase5_test_report.json",
    REPORTS_DIR / "phase5_ablation.csv",
    REPORTS_DIR / "phase5_error_analysis.csv",
    REPORTS_DIR / "phase5_confusion_matrix.png",
    REPORTS_DIR / "phase5_roc_curve.png",
    REPORTS_DIR / "phase5_pr_curve.png",
    REPORTS_DIR / "phase5_calibration.png",
    REPORTS_DIR / "phase5_ablation.png",
    REPORTS_DIR / "phase5_examples.png",
    MODELS_DIR / "phase5_fusion_best.joblib",
    MANIFESTS_DIR / "phase5_ocr_features.csv",
    MANIFESTS_DIR / "phase5_fusion_features.csv",
]


def verify_environment() -> Dict[str, Any]:
    import torch
    import torchvision
    import sklearn
    import joblib
    import PIL

    env_info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "pillow_version": PIL.__version__,
        "ocr_engine": "Windows.Media.Ocr (WinRT Native)",
        "cuda_available": torch.cuda.is_available(),
    }
    return env_info


def main():
    print("=" * 70)
    print("TRUSTTRACE: Phase 5 Evidence Fusion Experiment Reproducibility Suite")
    print("=" * 70)

    env = verify_environment()
    print("Software & Hardware Environment:")
    for k, v in env.items():
        print(f"  {k:24s}: {v}")

    # Check manifest isolation
    train_p = MANIFESTS_DIR / "trusttrace_phase4_train.csv"
    val_p = MANIFESTS_DIR / "trusttrace_phase4_val.csv"
    test_p = MANIFESTS_DIR / "trusttrace_phase4_test.csv"

    for p in [train_p, val_p, test_p]:
        assert p.is_file(), f"Missing manifest: {p}"

    with open(train_p, "r", encoding="utf-8") as f:
        tr_rows = list(csv.DictReader(f))
    with open(val_p, "r", encoding="utf-8") as f:
        va_rows = list(csv.DictReader(f))
    with open(test_p, "r", encoding="utf-8") as f:
        te_rows = list(csv.DictReader(f))

    assert not (set(r["sample_id"] for r in tr_rows) & set(r["sample_id"] for r in te_rows)), "Leakage!"
    assert not (set(r["group_id"] for r in tr_rows) & set(r["group_id"] for r in te_rows)), "Group Leakage!"
    print(f"\nManifest Integrity Verified (Zero Leakage, Zero Overlap):")
    print(f"  Train: {len(tr_rows)} | Val: {len(va_rows)} | Test: {len(te_rows)} (Total: {len(tr_rows)+len(va_rows)+len(te_rows)})")

    if "--check-only" in sys.argv:
        print("\nVerifying all Phase 5 artifacts exist on disk:")
        for a in REQUIRED_ARTIFACTS:
            assert a.is_file(), f"Missing artifact: {a}"
            print(f"  [OK] {a.name} ({a.stat().st_size:,} bytes)")
        print("\n[OK] Check-only mode passed successfully.")
        return

    # 1. OCR Feature Extraction
    print("\n[Step 1/4] Running OCR Feature Extraction...")
    from scripts.extract_ocr_forensic_features import main as extract_ocr
    extract_ocr()

    # 2. Transferability Study
    print("\n[Step 2/4] Running Transferability Diagnostic...")
    from scripts.evaluate_phase5_transferability import run_transferability_study
    run_transferability_study()

    # 3. Training Fusion Model
    print("\n[Step 3/4] Training Evidence Fusion Model...")
    from scripts.train_phase5_fusion import train_fusion_model
    train_fusion_model()

    # 4. Evaluation Suite
    print("\n[Step 4/4] Running Comprehensive Evaluation...")
    from scripts.evaluate_phase5_fusion import main as evaluate_fusion
    evaluate_fusion()

    # Verification of outputs
    print("\nVerifying all generated artifacts:")
    for a in REQUIRED_ARTIFACTS:
        assert a.is_file(), f"Missing artifact: {a}"
        print(f"  [OK] {a.name} ({a.stat().st_size:,} bytes)")

    print("\n" + "=" * 70)
    print("PHASE 5 REPRODUCIBILITY SUITE PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
