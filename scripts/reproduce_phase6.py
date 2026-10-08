#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 6 Doc-PatchFormer Reproducibility Suite.

Documents and reproduces the complete Phase 6 experiment:
1. Environment and dependency audit
2. Feasibility audit of patch annotations (Level A)
3. Native-resolution patch extraction and manifest construction
4. Training of Doc-PatchFormer and Patch-CNN baseline
5. Comprehensive test evaluation, document aggregation, and ablations
6. Supports --check-only mode
"""

import sys
import os
import json
import time
import platform
import csv
from typing import Dict, Any
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
    DOCS_DIR / "trusttrace-phase6-feasibility.md",
    DOCS_DIR / "trusttrace-phase6-doc-patchformer.md",
    MANIFESTS_DIR / "phase6_patch_master.csv",
    MANIFESTS_DIR / "phase6_patch_train.csv",
    MANIFESTS_DIR / "phase6_patch_val.csv",
    MANIFESTS_DIR / "phase6_patch_test.csv",
    MODELS_DIR / "doc_patchformer_best.pt",
    MODELS_DIR / "patch_cnn_baseline.pt",
    REPORTS_DIR / "phase6_test_report.json",
    REPORTS_DIR / "phase6_test_report.md",
    REPORTS_DIR / "phase6_ablation.csv",
    REPORTS_DIR / "phase6_error_analysis.csv",
    REPORTS_DIR / "phase6_confusion_matrix.png",
    REPORTS_DIR / "phase6_roc_curve.png",
    REPORTS_DIR / "phase6_pr_curve.png",
    REPORTS_DIR / "phase6_resolution_ablation.png",
    REPORTS_DIR / "phase6_examples.png",
]


def verify_environment() -> Dict[str, Any]:
    import torch
    import torchvision
    import sklearn
    import PIL

    env_info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pillow_version": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    return env_info


def main():
    print("=" * 70)
    print("TRUSTTRACE: Phase 6 Doc-PatchFormer Reproducibility Suite")
    print("=" * 70)

    env = verify_environment()
    print("Software & Hardware Environment:")
    for k, v in env.items():
        print(f"  {k:24s}: {v}")

    # Check manifest isolation
    train_m = MANIFESTS_DIR / "phase6_patch_train.csv"
    val_m = MANIFESTS_DIR / "phase6_patch_val.csv"
    test_m = MANIFESTS_DIR / "phase6_patch_test.csv"

    if all(p.is_file() for p in [train_m, val_m, test_m]):
        with open(train_m, "r", encoding="utf-8") as f:
            tr_rows = list(csv.DictReader(f))
        with open(val_m, "r", encoding="utf-8") as f:
            va_rows = list(csv.DictReader(f))
        with open(test_m, "r", encoding="utf-8") as f:
            te_rows = list(csv.DictReader(f))

        tr_g = set(r["group_id"] for r in tr_rows)
        te_g = set(r["group_id"] for r in te_rows)
        assert not (tr_g & te_g), "Group leakage across patch splits!"
        print(f"\nPatch Manifests Integrity Verified (Zero Group Leakage):")
        print(f"  Train Patches: {len(tr_rows)} | Val Patches: {len(va_rows)} | Test Patches: {len(te_rows)} (Total: {len(tr_rows)+len(va_rows)+len(te_rows)})")

    if "--check-only" in sys.argv:
        print("\nVerifying all Phase 6 artifacts exist on disk:")
        for a in REQUIRED_ARTIFACTS:
            assert a.is_file(), f"Missing artifact: {a}"
            print(f"  [OK] {a.name} ({a.stat().st_size:,} bytes)")
        print("\n[OK] Check-only mode passed successfully.")
        return

    # 1. Feasibility Audit
    print("\n[Step 1/4] Running Supervision Feasibility Audit...")
    from scripts.audit_phase6_patch_supervision import audit_feasibility
    audit_feasibility()

    # 2. Patch Extraction
    print("\n[Step 2/4] Running Native-Resolution Patch Extraction...")
    from scripts.extract_document_patches import main as extract_patches
    extract_patches()

    # 3. Model Training
    print("\n[Step 3/4] Training Doc-PatchFormer and Patch-CNN Models...")
    from scripts.train_doc_patchformer import main as train_models
    train_models()

    # 4. Evaluation Suite
    print("\n[Step 4/4] Running Comprehensive Evaluation Suite...")
    from scripts.evaluate_doc_patchformer import main as evaluate_all
    evaluate_all()

    print("\nVerifying all generated artifacts:")
    for a in REQUIRED_ARTIFACTS:
        assert a.is_file(), f"Missing artifact: {a}"
        print(f"  [OK] {a.name} ({a.stat().st_size:,} bytes)")

    print("\n" + "=" * 70)
    print("PHASE 6 REPRODUCIBILITY SUITE PASSED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
