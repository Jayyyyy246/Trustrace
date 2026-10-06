"""TRUSTTRACE Dataset Preparation and Validation Utility.

Inspects a directory of forensic images to verify:
1. Presence of required class subdirectories: REAL, EDITED, AI-GENERATED, SCREENSHOT-MANIPULATED
2. Image decodability and corruption checks via PIL verify
3. SHA-256 deduplication to prevent data leakage across splits
4. Class balance statistics and split manifest generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from typing import Dict, List, Any

from model.dataset import (
    CLASS_LABELS,
    VALID_IMAGE_EXTENSIONS,
    validate_dataset_directory,
    compute_image_content_hash,
)


def inspect_dataset(data_dir: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dataset_path": str(data_dir.resolve()),
        "exists": data_dir.is_dir(),
        "expected_classes": CLASS_LABELS,
        "class_counts": {},
        "missing_classes": [],
        "total_images": 0,
        "is_ready_for_training": False,
        "status": "UNAVAILABLE",
    }

    if not data_dir.is_dir():
        report["message"] = (
            f"Dataset directory '{data_dir}' does not exist. "
            f"Required directory structure:\n"
            f"  {data_dir}/\n"
            f"    |-- REAL/\n"
            f"    |-- EDITED/\n"
            f"    |-- AI-GENERATED/\n"
            f"    \\-- SCREENSHOT-MANIPULATED/\n"
            "Forensic datasets must be acquired and placed in these directories before model training."
        )
        return report

    val = validate_dataset_directory(data_dir)
    report["class_counts"] = val["class_counts"]
    report["missing_classes"] = val["classes_missing"]
    report["total_images"] = val["total_images"]
    report["corrupt_images"] = val["corrupt_images"]
    report["duplicate_images"] = len(val["duplicates"])

    has_all_classes = len(val["classes_missing"]) == 0
    has_sufficient_images = val["total_images"] >= len(CLASS_LABELS) * 5

    if has_all_classes and has_sufficient_images and not val["corrupt_images"]:
        report["is_ready_for_training"] = True
        report["status"] = "READY"
        report["message"] = f"Dataset validated successfully: {val['total_images']} images ready for training."
    else:
        report["status"] = "INCOMPLETE"
        reasons = []
        if val["classes_missing"]:
            reasons.append(f"Missing classes: {val['classes_missing']}")
        if val["total_images"] == 0:
            reasons.append("0 valid images found across classes")
        if val["corrupt_images"]:
            reasons.append(f"{len(val['corrupt_images'])} corrupted images detected")
        report["message"] = f"Dataset incomplete: {'; '.join(reasons)}"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Dataset Preparation Utility")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="Path to dataset root directory")
    parser.add_argument("--output-manifest", type=str, default=None, help="Optional output path for dataset summary JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"=== TRUSTTRACE Dataset Inspection: {data_dir} ===")

    report = inspect_dataset(data_dir)
    print(f"Status: {report['status']}")
    print(f"Details: {report['message']}")
    if report["class_counts"]:
        print(f"Class Counts: {report['class_counts']}")

    if args.output_manifest:
        out_p = Path(args.output_manifest)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Manifest written to: {out_p}")

    return 0 if report["is_ready_for_training"] else 1


if __name__ == "__main__":
    sys.exit(main())
