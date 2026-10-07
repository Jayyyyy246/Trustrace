#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 4 Binary Authenticity Experiment Reproducibility Suite.

Documents and reproduces the complete Phase 4 receipt authenticity experiment:
1. Validates dataset manifests and strict group-isolation integrity
2. Verifies software environment and dependencies
3. Ensures trained checkpoint exists or trains it reproducibly
4. Runs held-out test evaluation at default and validation-calibrated thresholds
5. Supports --check-only mode
"""

import sys
import os
import json
import time
import platform
import csv
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

MANIFESTS_DIR = WORKSPACE_DIR / "data" / "manifests"
REPORTS_DIR = WORKSPACE_DIR / "reports"
MODELS_DIR = WORKSPACE_DIR / "models"

MASTER_M = MANIFESTS_DIR / "phase4_dataset_master.csv"
TRAIN_M = MANIFESTS_DIR / "trusttrace_phase4_train.csv"
VAL_M = MANIFESTS_DIR / "trusttrace_phase4_val.csv"
TEST_M = MANIFESTS_DIR / "trusttrace_phase4_test.csv"

BEST_MODEL_PATH = MODELS_DIR / "phase4_authenticity_best.pt"
CONFIG_JSON_PATH = REPORTS_DIR / "phase4_authenticity_config.json"
TEST_REPORT_JSON_PATH = REPORTS_DIR / "phase4_authenticity_test_report.json"
TEST_REPORT_MD_PATH = REPORTS_DIR / "phase4_authenticity_test_report.md"
CONFUSION_MATRIX_PNG_PATH = REPORTS_DIR / "phase4_authenticity_confusion_matrix.png"
EXAMPLES_PNG_PATH = REPORTS_DIR / "phase4_authenticity_examples.png"


def verify_and_document_environment():
    import torch
    import torchvision
    import sklearn
    import numpy as np
    import PIL

    env_info = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    return env_info


def verify_partition_integrity():
    for p in [MASTER_M, TRAIN_M, VAL_M, TEST_M]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing required manifest: {p}")

    with open(TRAIN_M, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(VAL_M, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    with open(TEST_M, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    train_ids = set(r["sample_id"] for r in train_rows)
    val_ids = set(r["sample_id"] for r in val_rows)
    test_ids = set(r["sample_id"] for r in test_rows)
    assert not (train_ids & val_ids), "Train/Val sample overlap!"
    assert not (train_ids & test_ids), "Train/Test sample overlap!"
    assert not (val_ids & test_ids), "Val/Test sample overlap!"

    train_shas = set(r["sha256"] for r in train_rows)
    val_shas = set(r["sha256"] for r in val_rows)
    test_shas = set(r["sha256"] for r in test_rows)
    assert not (train_shas & val_shas), "Train/Val SHA-256 overlap!"
    assert not (train_shas & test_shas), "Train/Test SHA-256 overlap!"
    assert not (val_shas & test_shas), "Val/Test SHA-256 overlap!"

    train_groups = set(r["group_id"] for r in train_rows)
    val_groups = set(r["group_id"] for r in val_rows)
    test_groups = set(r["group_id"] for r in test_rows)
    assert not (train_groups & val_groups), "Train/Val group leakage!"
    assert not (train_groups & test_groups), "Train/Test group leakage!"
    assert not (val_groups & test_groups), "Val/Test group leakage!"

    return {
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "total_samples": len(train_rows) + len(val_rows) + len(test_rows),
    }


def main():
    print("=" * 70)
    print("TRUSTTRACE: Phase 4 Binary Authenticity Experiment Reproducibility Suite")
    print("=" * 70)

    env = verify_and_document_environment()
    print("Software Environment:")
    for k, v in env.items():
        print(f"  {k:22s}: {v}")

    counts = verify_partition_integrity()
    print(f"\nManifest Integrity Verified (Zero Leakage, Zero Overlap):")
    print(f"  Train: {counts['train_samples']} samples")
    print(f"  Val:   {counts['val_samples']} samples")
    print(f"  Test:  {counts['test_samples']} samples")
    print(f"  Total: {counts['total_samples']} samples")

    if not BEST_MODEL_PATH.is_file():
        print("\nModel checkpoint missing. Initiating training...")
        from scripts.train_phase4_authenticity import train_model
        train_model()
    else:
        print(f"\nVerified model checkpoint: {BEST_MODEL_PATH.name} ({BEST_MODEL_PATH.stat().st_size:,} bytes)")

    if "--check-only" in sys.argv:
        print("\n[OK] Check-only mode: All manifests, models, and dependencies verified.")
        return

    print("\nRunning Phase 4 held-out test evaluation...")
    from scripts.evaluate_phase4_authenticity import evaluate_authenticity
    evaluate_authenticity()

    print("\nVerifying produced report artifacts:")
    for p in [
        CONFIG_JSON_PATH,
        TEST_REPORT_JSON_PATH,
        TEST_REPORT_MD_PATH,
        CONFUSION_MATRIX_PNG_PATH,
        EXAMPLES_PNG_PATH,
    ]:
        assert p.is_file(), f"Missing report artifact: {p}"
        print(f"  [OK] {p.name}")

    print("\n" + "=" * 70)
    print("PHASE 4 EXPERIMENT REPRODUCED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
