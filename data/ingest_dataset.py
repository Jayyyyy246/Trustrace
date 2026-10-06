"""TRUSTTRACE Dataset Ingestion & Manifest Builder.

Imports, validates, deduplicates, maps classes, and creates leak-free
deterministic splits for forensic datasets with full cryptographic provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.deduplication import compute_sha256, compute_dhash, detect_duplicates, verify_cross_split_leakage

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
TARGET_CLASSES = ["REAL", "EDITED", "AI-GENERATED", "SCREENSHOT-MANIPULATED"]

MANIFEST_FIELDNAMES = [
    "image_id",
    "sha256",
    "source_dataset",
    "source_id",
    "original_path",
    "target_class",
    "width",
    "height",
    "format",
    "file_size",
    "split",
]

REJECTED_FIELDNAMES = [
    "path",
    "reason",
    "sha256",
]


def load_source_mapping(mapping_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads source-to-class mapping specification."""
    if mapping_path is None:
        mapping_path = Path(__file__).resolve().parent / "source_mapping.json"
    if not mapping_path.is_file():
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_image_file(file_path: Path, min_dim: int = 32) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validates file existence, format, decode, and dimensions.
    
    Returns:
        (is_valid, rejection_reason, metadata_dict)
    """
    if not file_path.is_file():
        return False, "File does not exist or is not a regular file", None

    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported extension '{ext}'. Must be one of: {sorted(SUPPORTED_EXTENSIONS)}", None

    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "Empty file (0 bytes)", None

    try:
        sha256 = compute_sha256(file_path)
    except Exception as exc:
        return False, f"Failed calculating SHA-256: {exc}", None

    try:
        with Image.open(file_path) as img:
            img.verify()
    except Exception as exc:
        return False, f"Corrupted or indecodable image header: {exc}", sha256

    try:
        # Re-open after verify() to inspect dimensions and format
        with Image.open(file_path) as img:
            w, h = img.size
            img_format = img.format or ext.lstrip(".").upper()
            if w <= 0 or h <= 0:
                return False, f"Invalid zero/negative dimensions ({w}x{h})", sha256
            if w < min_dim or h < min_dim:
                return False, f"Resolution below minimum ({w}x{h} < {min_dim}x{min_dim})", sha256
            dhash = compute_dhash(img)
    except Exception as exc:
        return False, f"Failed reading image metadata/dhash: {exc}", sha256

    metadata = {
        "width": w,
        "height": h,
        "format": img_format,
        "file_size": file_size,
        "sha256": sha256,
        "dhash": dhash,
    }
    return True, None, metadata


def map_source_to_class(
    source_dataset: str,
    explicit_class: Optional[str] = None,
    source_mapping: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolves verified target class from source dataset name or explicit mapping.
    
    Returns:
        (target_class, unmapped_reason)
    """
    if explicit_class and explicit_class in TARGET_CLASSES:
        return explicit_class, None

    if source_mapping is None:
        source_mapping = load_source_mapping()

    if source_mapping and "classes" in source_mapping:
        src_clean = source_dataset.lower().replace("-", "_").replace(" ", "_")
        for cls_name, cls_cfg in source_mapping["classes"].items():
            accepted = [s.lower().replace("-", "_").replace(" ", "_") for s in cls_cfg.get("accepted_sources", [])]
            if src_clean in accepted:
                return cls_name, None

    return None, f"Source dataset '{source_dataset}' is UNMAPPED and not verified in taxonomy specification"


def build_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Performs source-group-aware stratified splitting.
    Images sharing the same source_group are placed exclusively in the same split.
    """
    rng = random.Random(seed)

    # Group records by (target_class, source_group)
    by_class: Dict[str, Dict[str, List[Dict[str, Any]]]] = {c: {} for c in TARGET_CLASSES}
    for r in records:
        cls = r["target_class"]
        # Group key: explicit source_group or base filename before __ or source_id
        src_grp = r.get("source_group") or r["source_id"].split("__")[0]
        by_class.setdefault(cls, {}).setdefault(src_grp, []).append(r)

    train_recs: List[Dict[str, Any]] = []
    val_recs: List[Dict[str, Any]] = []
    test_recs: List[Dict[str, Any]] = []

    for cls, groups_dict in by_class.items():
        groups = list(groups_dict.keys())
        rng.shuffle(groups)

        n_groups = len(groups)
        n_train = int(round(n_groups * train_ratio))
        n_val = int(round(n_groups * val_ratio))

        # Adjust boundary if small dataset
        if n_groups > 0 and n_train == 0:
            n_train = 1

        train_grps = set(groups[:n_train])
        val_grps = set(groups[n_train:n_train + n_val])
        test_grps = set(groups[n_train + n_val:])

        for grp, recs in groups_dict.items():
            if grp in train_grps:
                for r in recs:
                    r["split"] = "train"
                train_recs.extend(recs)
            elif grp in val_grps:
                for r in recs:
                    r["split"] = "validation"
                val_recs.extend(recs)
            else:
                for r in recs:
                    r["split"] = "test"
                test_recs.extend(recs)

    return train_recs, val_recs, test_recs


def ingest_dataset(
    source_dir: Optional[Path] = None,
    dataset_dir: Path = Path("data/dataset"),
    manifests_dir: Path = Path("data/manifests"),
    source_name: Optional[str] = None,
    explicit_class: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    copy_to_dataset: bool = False,
) -> Dict[str, Any]:
    """
    Core dataset ingestion and manifest generation pipeline.
    Supports ingesting a single source directory or multiple source directories.
    """
    manifests_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for c in TARGET_CLASSES:
        (dataset_dir / c).mkdir(parents=True, exist_ok=True)

    if sources is None:
        if source_dir is None:
            raise ValueError("Either 'source_dir' or 'sources' list must be provided to ingest_dataset.")
        sources = [{
            "source_dir": source_dir,
            "source_name": source_name or "default_source",
            "explicit_class": explicit_class,
        }]

    mapping = load_source_mapping()
    valid_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []
    seen_sha: Set[str] = set()
    exact_duplicates: List[str] = []

    # Find candidate files across all configured sources
    candidates: List[Tuple[Path, str, Optional[str]]] = []
    for src_cfg in sources:
        s_dir = Path(src_cfg["source_dir"])
        s_name = src_cfg.get("source_name", "unknown")
        s_cls = src_cfg.get("explicit_class")
        if s_dir.is_dir():
            for root, _, files in os.walk(s_dir):
                for f in sorted(files):
                    p = Path(root) / f
                    if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                        candidates.append((p, s_name, s_cls))

    warnings: List[str] = []

    # Process candidates
    idx = 1
    for p, s_name, s_cls in candidates:
        is_val, reason, meta = validate_image_file(p)
        if not is_val:
            rejected_records.append({
                "path": str(p.resolve()),
                "reason": reason or "Unknown validation failure",
                "sha256": meta if isinstance(meta, str) else "",
            })
            continue

        sha = meta["sha256"]
        if sha in seen_sha:
            exact_duplicates.append(str(p))
            rejected_records.append({
                "path": str(p.resolve()),
                "reason": f"Duplicate SHA-256 within ingestion set ({sha[:12]}...)",
                "sha256": sha,
            })
            continue
        seen_sha.add(sha)

        # Class mapping
        target_cls, unmapped_reason = map_source_to_class(s_name, s_cls, mapping)
        if not target_cls:
            rejected_records.append({
                "path": str(p.resolve()),
                "reason": unmapped_reason or "UNMAPPED source",
                "sha256": sha,
            })
            continue

        image_id = f"img_{idx:06d}"
        idx += 1

        dest_path = str(p.resolve())
        if copy_to_dataset:
            import shutil
            dest_file = dataset_dir / target_cls / f"{image_id}{p.suffix.lower()}"
            shutil.copy2(p, dest_file)
            dest_path = str(dest_file.resolve())

        valid_records.append({
            "image_id": image_id,
            "sha256": sha,
            "dhash": meta["dhash"],
            "source_dataset": source_name,
            "source_id": p.name,
            "original_path": dest_path,
            "target_class": target_cls,
            "width": meta["width"],
            "height": meta["height"],
            "format": meta["format"],
            "file_size": meta["file_size"],
            "split": "unassigned",
            "source_group": p.stem.split("__")[0],
        })

    # Partition into splits
    train_recs, val_recs, test_recs = build_splits(
        valid_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    all_split_recs = train_recs + val_recs + test_recs

    # Verify cross-split leakage
    leakage_check = verify_cross_split_leakage(train_recs, val_recs, test_recs)
    if leakage_check["has_leakage"]:
        warnings.extend(leakage_check["violations"])

    # Detect perceptual near-duplicates
    dup_res = detect_duplicates(all_split_recs)
    near_dup_count = len(dup_res["near_duplicates"])

    # Write manifests
    def write_csv(path: Path, recs: List[Dict[str, Any]], fieldnames: List[str]):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in recs:
                writer.writerow(r)

    manifest_csv = manifests_dir / "dataset_manifest.csv"
    train_csv = manifests_dir / "train.csv"
    val_csv = manifests_dir / "validation.csv"
    test_csv = manifests_dir / "test.csv"
    rejected_csv = manifests_dir / "rejected.csv"

    write_csv(manifest_csv, all_split_recs, MANIFEST_FIELDNAMES)
    write_csv(train_csv, train_recs, MANIFEST_FIELDNAMES)
    write_csv(val_csv, val_recs, MANIFEST_FIELDNAMES)
    write_csv(test_csv, test_recs, MANIFEST_FIELDNAMES)
    write_csv(rejected_csv, rejected_records, REJECTED_FIELDNAMES)

    # Class counts
    class_counts = {c: sum(1 for r in all_split_recs if r["target_class"] == c) for c in TARGET_CLASSES}
    split_counts = {
        "train": len(train_recs),
        "validation": len(val_recs),
        "test": len(test_recs),
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": len(all_split_recs),
        "classes": class_counts,
        "splits": split_counts,
        "duplicates": {
            "exact": len(exact_duplicates),
            "near_duplicates": near_dup_count,
        },
        "rejected": len(rejected_records),
        "unmapped": sum(1 for r in rejected_records if "UNMAPPED" in r["reason"]),
        "sources": sorted(list(set(r["source_dataset"] for r in all_split_recs))) if all_split_recs else [],
        "warnings": warnings,
    }

    report_path = dataset_dir.parent / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Dataset Ingestion & Manifest Builder")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory containing images to ingest")
    parser.add_argument("--source-name", type=str, required=True, help="Registered source dataset identifier")
    parser.add_argument("--target-class", type=str, choices=TARGET_CLASSES, default=None, help="Explicit target class override")
    parser.add_argument("--dataset-dir", type=str, default="data/dataset", help="Target dataset directory")
    parser.add_argument("--manifests-dir", type=str, default="data/manifests", help="Target manifests directory")
    parser.add_argument("--copy-files", action="store_true", help="Copy files into data/dataset/<CLASS>/")
    args = parser.parse_args()

    print(f"=== TRUSTTRACE Dataset Ingestion ===")
    print(f"Source Dir:    {args.source_dir}")
    print(f"Source Name:   {args.source_name}")
    print(f"Dataset Dir:   {args.dataset_dir}")
    print(f"Manifests Dir: {args.manifests_dir}")

    report = ingest_dataset(
        source_dir=Path(args.source_dir),
        dataset_dir=Path(args.dataset_dir),
        manifests_dir=Path(args.manifests_dir),
        source_name=args.source_name,
        explicit_class=args.target_class,
        copy_to_dataset=args.copy_files,
    )

    print("\n=== Ingestion Report Summary ===")
    print(f"Total Valid Images:  {report['total_images']}")
    print(f"Class Counts:        {report['classes']}")
    print(f"Splits:              {report['splits']}")
    print(f"Duplicates Rejected: {report['duplicates']['exact']}")
    print(f"Near Duplicates:     {report['duplicates']['near_duplicates']}")
    print(f"Total Rejected:      {report['rejected']}")
    print(f"Unmapped:            {report['unmapped']}")
    if report["warnings"]:
        print(f"Warnings:            {report['warnings']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
