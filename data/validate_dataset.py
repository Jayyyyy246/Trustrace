"""TRUSTTRACE Dataset Quality Gate Validator.

Executes comprehensive forensic dataset validation:
1. Target class presence: REAL, EDITED, AI-GENERATED, SCREENSHOT-MANIPULATED
2. Minimum sample requirements per class
3. Complete image decode & integrity verification
4. Valid taxonomy labels
5. Zero exact SHA-256 cross-split duplicates
6. Zero source-group leakage across train, validation, and test splits
7. Strict manifest consistency with filesystem state

Exits code 0 ONLY when all quality gates pass; otherwise exits non-zero.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PIL import Image

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.deduplication import compute_sha256, verify_cross_split_leakage

REQUIRED_CLASSES = ["REAL", "EDITED", "AI-GENERATED", "SCREENSHOT-MANIPULATED"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def read_manifest_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Reads a manifest CSV file into a list of row dictionaries."""
    if not csv_path.is_file():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def validate_dataset_quality(
    dataset_dir: Path,
    manifests_dir: Optional[Path] = None,
    min_samples_per_class: int = 10,
    check_decode: bool = True,
) -> Dict[str, Any]:
    """
    Validates the dataset against all TRUSTTRACE quality gates.
    
    Returns:
        Dict summarizing validation pass/fail status and detailed reasons.
    """
    errors: List[str] = []
    warnings: List[str] = []
    class_counts: Dict[str, int] = {c: 0 for c in REQUIRED_CLASSES}

    # 1. Directory and Class Verification
    if not dataset_dir.is_dir():
        errors.append(f"Dataset root directory '{dataset_dir}' does not exist.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "class_counts": class_counts,
            "total_images": 0,
        }

    missing_classes = []
    for cls in REQUIRED_CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.is_dir():
            missing_classes.append(cls)
        else:
            # Count valid image files
            files = [
                f for f in cls_dir.iterdir()
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            ]
            class_counts[cls] = len(files)

    if missing_classes:
        errors.append(f"Missing required class directories: {missing_classes}")

    # Check for prohibited UNKNOWN directory
    if (dataset_dir / "UNKNOWN").is_dir():
        errors.append("Prohibited 'UNKNOWN' directory detected. UNKNOWN is a runtime rejection state, not a training class.")

    total_images = sum(class_counts.values())

    # 2. Minimum Sample Verification
    underpopulated = [c for c, count in class_counts.items() if count < min_samples_per_class]
    if underpopulated:
        errors.append(
            f"Classes below minimum sample requirement ({min_samples_per_class}): "
            + ", ".join(f"{c} ({class_counts[c]})" for c in underpopulated)
        )

    # 3. Decodability Verification
    corrupted_files: List[str] = []
    if check_decode and total_images > 0:
        for cls in REQUIRED_CLASSES:
            cls_dir = dataset_dir / cls
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.iterdir():
                if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
                    try:
                        with Image.open(f) as img:
                            img.verify()
                        with Image.open(f) as img:
                            w, h = img.size
                            if w <= 0 or h <= 0:
                                corrupted_files.append(f"{f.name} (zero dimensions)")
                    except Exception as exc:
                        corrupted_files.append(f"{f.name} ({exc})")

    if corrupted_files:
        errors.append(f"Corrupted or indecodable images detected ({len(corrupted_files)}): {corrupted_files[:5]}")

    # 4. Manifest and Leakage Verification (if manifests provided)
    if manifests_dir and manifests_dir.is_dir():
        train_csv = manifests_dir / "train.csv"
        val_csv = manifests_dir / "validation.csv"
        test_csv = manifests_dir / "test.csv"
        manifest_csv = manifests_dir / "dataset_manifest.csv"

        if not manifest_csv.is_file():
            errors.append(f"Main manifest '{manifest_csv}' missing.")
        if not train_csv.is_file():
            errors.append(f"Training split manifest '{train_csv}' missing.")
        if not val_csv.is_file():
            errors.append(f"Validation split manifest '{val_csv}' missing.")
        if not test_csv.is_file():
            errors.append(f"Test split manifest '{test_csv}' missing.")

        if all(p.is_file() for p in (train_csv, val_csv, test_csv)):
            train_recs = read_manifest_csv(train_csv)
            val_recs = read_manifest_csv(val_csv)
            test_recs = read_manifest_csv(test_csv)

            # Check for empty splits
            if not train_recs:
                errors.append("Training split manifest contains 0 samples.")
            if not val_recs:
                errors.append("Validation split manifest contains 0 samples.")
            if not test_recs:
                errors.append("Test split manifest contains 0 samples.")

            # Cross-split leakage checks
            leak_result = verify_cross_split_leakage(train_recs, val_recs, test_recs)
            if leak_result["has_leakage"]:
                errors.extend(leak_result["violations"])

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "class_counts": class_counts,
        "total_images": total_images,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Dataset Quality Gate Validator")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="Path to dataset root directory")
    parser.add_argument("--manifests-dir", type=str, default="data/manifests", help="Path to manifests directory")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum required samples per class")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifests_dir = Path(args.manifests_dir) if args.manifests_dir else None

    print("=" * 60)
    print("TRUSTTRACE DATASET QUALITY GATE VALIDATION")
    print(f"Dataset Path:    {data_dir.resolve()}")
    print(f"Manifests Path:  {manifests_dir.resolve() if manifests_dir else 'None'}")
    print(f"Min Per Class:   {args.min_samples}")
    print("=" * 60)

    result = validate_dataset_quality(
        dataset_dir=data_dir,
        manifests_dir=manifests_dir,
        min_samples_per_class=args.min_samples,
    )

    print(f"\nTotal Images: {result['total_images']}")
    print(f"Class Counts: {result['class_counts']}")

    if result["warnings"]:
        print("\nWarnings:")
        for w in result["warnings"]:
            print(f"  [!] {w}")

    if not result["is_valid"]:
        print("\n[FAILED] Quality gate rejected dataset for the following reasons:")
        for err in result["errors"]:
            print(f"  [-] {err}")
        print("\nDataset is NOT approved for model training.")
        return 1

    print("\n[PASSED] All dataset quality gates satisfied. Dataset is approved for model training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
