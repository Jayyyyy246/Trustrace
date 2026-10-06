#!/usr/bin/env python3
"""
TRUSTTRACE: Reproducible, Leakage-Safe STFD Dataset Splitting Script.
Generates:
- data/manifests/trusttrace_stfd_train.csv (70%)
- data/manifests/trusttrace_stfd_val.csv   (15%)
- data/manifests/trusttrace_stfd_test.csv  (15%)

Uses fixed random seed 42 with stratified sampling across manipulation_type.
Strictly verifies zero sample_id, image hash, and mask hash leakage across splits.
"""

import sys
import csv
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = WORKSPACE_DIR / "data" / "manifests"
MASTER_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_master.csv"
TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_val.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_test.csv"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_master_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Master manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 3932:
        raise ValueError(f"Expected exactly 3,932 rows in master manifest, found {len(rows)}")

    # Validate required columns
    required_cols = {
        "sample_id",
        "image_path",
        "mask_path",
        "manipulation_type",
        "trusttrace_class",
        "image_sha256",
        "mask_sha256",
    }
    missing = required_cols - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns in master manifest: {missing}")

    return rows


def perform_stratified_split(
    rows: List[Dict[str, Any]],
    seed: int = RANDOM_SEED
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Splits rows deterministically using stratified sampling by manipulation_type.
    """
    # Group rows by manipulation_type
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        mtype = r["manipulation_type"]
        groups.setdefault(mtype, []).append(r)

    rng = np.random.RandomState(seed)

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []

    for mtype in sorted(groups.keys()):
        items = groups[mtype]
        # Sort items by deterministic sample_id before shuffling
        items = sorted(items, key=lambda x: x["sample_id"])
        
        # Shuffle with fixed RNG
        indices = rng.permutation(len(items))
        shuffled = [items[i] for i in indices]

        n = len(shuffled)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        # Remainder goes to test to guarantee sum(splits) == n
        n_test = n - n_train - n_val

        train_part = shuffled[:n_train]
        val_part = shuffled[n_train : n_train + n_val]
        test_part = shuffled[n_train + n_val :]

        assert len(train_part) == n_train
        assert len(val_part) == n_val
        assert len(test_part) == n_test

        train_rows.extend(train_part)
        val_rows.extend(val_part)
        test_rows.extend(test_part)

    # Sort each split deterministically by sample_id
    train_rows = sorted(train_rows, key=lambda x: x["sample_id"])
    val_rows = sorted(val_rows, key=lambda x: x["sample_id"])
    test_rows = sorted(test_rows, key=lambda x: x["sample_id"])

    return {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }


def verify_leakage(splits: Dict[str, List[Dict[str, Any]]]):
    """
    Strict verification that zero samples, image hashes, or mask hashes overlap.
    """
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]

    train_ids = set(r["sample_id"] for r in train)
    val_ids = set(r["sample_id"] for r in val)
    test_ids = set(r["sample_id"] for r in test)

    train_img_shas = set(r["image_sha256"] for r in train)
    val_img_shas = set(r["image_sha256"] for r in val)
    test_img_shas = set(r["image_sha256"] for r in test)

    train_mask_shas = set(r["mask_sha256"] for r in train)
    val_mask_shas = set(r["mask_sha256"] for r in val)
    test_mask_shas = set(r["mask_sha256"] for r in test)

    # Check intersections
    id_tv = train_ids & val_ids
    id_tt = train_ids & test_ids
    id_vt = val_ids & test_ids
    if id_tv or id_tt or id_vt:
        raise ValueError(f"CRITICAL: Sample ID overlap detected across splits! (TV: {len(id_tv)}, TT: {len(id_tt)}, VT: {len(id_vt)})")

    img_tv = train_img_shas & val_img_shas
    img_tt = train_img_shas & test_img_shas
    img_vt = val_img_shas & test_img_shas
    if img_tv or img_tt or img_vt:
        raise ValueError(f"CRITICAL: Image SHA-256 hash overlap detected across splits! (TV: {len(img_tv)}, TT: {len(img_tt)}, VT: {len(img_vt)})")

    mask_tv = train_mask_shas & val_mask_shas
    mask_tt = train_mask_shas & test_mask_shas
    mask_vt = val_mask_shas & test_mask_shas
    if mask_tv or mask_tt or mask_vt:
        raise ValueError(f"CRITICAL: Mask SHA-256 hash overlap detected across splits! (TV: {len(mask_tv)}, TT: {len(mask_tt)}, VT: {len(mask_vt)})")

    total = len(train) + len(val) + len(test)
    if total != 3932:
        raise ValueError(f"Total split count {total} does not match expected 3,932!")


def save_split_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        raise ValueError(f"Cannot save empty split to {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_distribution(splits: Dict[str, List[Dict[str, Any]]]):
    print("\n" + "=" * 60)
    print("TRUSTTRACE STFD DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Total Samples: {len(splits['train']) + len(splits['val']) + len(splits['test'])}")
    print(f"  TRAIN:      {len(splits['train']):>5} ({len(splits['train'])/3932*100:>5.1f}%)")
    print(f"  VALIDATION: {len(splits['val']):>5} ({len(splits['val'])/3932*100:>5.1f}%)")
    print(f"  TEST:       {len(splits['test']):>5} ({len(splits['test'])/3932*100:>5.1f}%)")

    # Manipulation breakdown
    all_types = sorted(list(set(r["manipulation_type"] for r in splits["train"])))
    print("\nManipulation Type Breakdown:")
    print(f"{'Manipulation Type':<16} | {'Train':>6} | {'Val':>6} | {'Test':>6} | {'Total':>6} | {'Train %':>7}")
    print("-" * 60)
    for mt in all_types:
        tr_c = sum(1 for r in splits["train"] if r["manipulation_type"] == mt)
        va_c = sum(1 for r in splits["val"] if r["manipulation_type"] == mt)
        te_c = sum(1 for r in splits["test"] if r["manipulation_type"] == mt)
        tot = tr_c + va_c + te_c
        pct = (tr_c / tot) * 100 if tot else 0.0
        print(f"{mt:<16} | {tr_c:>6} | {va_c:>6} | {te_c:>6} | {tot:>6} | {pct:>6.1f}%")

    print("\nData Leakage Check: 100% PASSED (0 sample, image hash, or mask hash overlaps)")
    print("=" * 60)


def main() -> int:
    print(f"Loading master manifest: {MASTER_MANIFEST_PATH}")
    rows = load_master_manifest(MASTER_MANIFEST_PATH)

    print("Performing stratified split by manipulation_type (seed=42)...")
    splits = perform_stratified_split(rows, seed=RANDOM_SEED)

    print("Verifying data leakage constraints...")
    verify_leakage(splits)

    print("Saving split CSVs...")
    save_split_csv(splits["train"], TRAIN_MANIFEST_PATH)
    save_split_csv(splits["val"], VAL_MANIFEST_PATH)
    save_split_csv(splits["test"], TEST_MANIFEST_PATH)

    print(f"Saved: {TRAIN_MANIFEST_PATH}")
    print(f"Saved: {VAL_MANIFEST_PATH}")
    print(f"Saved: {TEST_MANIFEST_PATH}")

    print_distribution(splits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
