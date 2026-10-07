#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 4 Group-Aware Partitioning Pipeline.

Creates deterministic, group-aware Train (70%), Validation (15%), and Test (15%) splits
for the "Find it again!" document authenticity dataset:
- Guarantees zero group overlap across splits
- Guarantees zero authentic/forged pair leakage across splits
- Guarantees zero sample ID and SHA-256 overlap
- Seed: 42 (deterministic)
- Stratified by REAL vs EDITED across groups
- Outputs:
  * data/manifests/trusttrace_phase4_train.csv
  * data/manifests/trusttrace_phase4_val.csv
  * data/manifests/trusttrace_phase4_test.csv
"""

import sys
import os
import csv
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"

MASTER_MANIFEST_PATH = MANIFESTS_DIR / "phase4_dataset_master.csv"
TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_phase4_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_phase4_val.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_phase4_test.csv"

RANDOM_SEED = 42
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def create_group_aware_splits() -> None:
    print("=" * 70)
    print("TRUSTTRACE: Creating Phase 4 Group-Aware Authenticity Splits")
    print("=" * 70)

    if not MASTER_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Master manifest not found: {MASTER_MANIFEST_PATH}")

    with open(MASTER_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} samples from master manifest.")

    # Group samples by group_id
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)

    print(f"Total indivisible groups: {len(groups)}")

    # Characterize groups for stratified allocation
    # Categories:
    # 1. Mixed groups (contain BOTH REAL and EDITED)
    # 2. Pure EDITED groups
    # 3. Pure REAL groups
    mixed_groups = []
    pure_edited_groups = []
    pure_real_groups = []

    for gid, items in groups.items():
        has_real = any(it["label"] == "REAL" for it in items)
        has_edited = any(it["label"] == "EDITED" for it in items)
        if has_real and has_edited:
            mixed_groups.append(gid)
        elif has_edited:
            pure_edited_groups.append(gid)
        else:
            pure_real_groups.append(gid)

    print(f"  Mixed groups (REAL + EDITED): {len(mixed_groups)}")
    print(f"  Pure EDITED groups:          {len(pure_edited_groups)}")
    print(f"  Pure REAL groups:            {len(pure_real_groups)}")

    # Deterministic shuffle
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(mixed_groups)
    rng.shuffle(pure_edited_groups)
    rng.shuffle(pure_real_groups)

    # Initialize partition buckets
    split_groups: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    split_counts: Dict[str, Dict[str, int]] = {
        s: {"total": 0, "REAL": 0, "EDITED": 0} for s in ["train", "val", "test"]
    }

    total_samples = len(rows)
    total_edited = sum(1 for r in rows if r["label"] == "EDITED")
    total_real = sum(1 for r in rows if r["label"] == "REAL")

    targets = {
        "train": {"total": 691, "REAL": 577, "EDITED": 114},
        "val": {"total": 148, "REAL": 124, "EDITED": 24},
        "test": {"total": 148, "REAL": 123, "EDITED": 25},
    }

    print("\nTarget Sample Counts:")
    for s, tgt in targets.items():
        print(f"  {s.upper():5s}: Total={tgt['total']:3d} | REAL={tgt['REAL']:3d} | EDITED={tgt['EDITED']:2d}")

    def add_group_to_split(split: str, gid: str) -> None:
        split_groups[split].append(gid)
        for it in groups[gid]:
            split_counts[split][it["label"]] += 1
            split_counts[split]["total"] += 1

    # 1. Mixed groups (9 groups): 70% train (6), 15% val (1), 15% test (2)
    for i, gid in enumerate(mixed_groups):
        if i < 6:
            add_group_to_split("train", gid)
        elif i < 7:
            add_group_to_split("val", gid)
        else:
            add_group_to_split("test", gid)

    # 2. Pure EDITED groups: allocate greedily to whichever split has highest remaining EDITED deficit
    for gid in pure_edited_groups:
        deficits = {s: targets[s]["EDITED"] - split_counts[s]["EDITED"] for s in ["train", "val", "test"]}
        best_split = max(deficits, key=deficits.get)
        add_group_to_split(best_split, gid)

    # 3. Pure REAL groups: allocate greedily to whichever split has highest remaining REAL deficit
    for gid in pure_real_groups:
        deficits = {s: targets[s]["REAL"] - split_counts[s]["REAL"] for s in ["train", "val", "test"]}
        best_split = max(deficits, key=deficits.get)
        add_group_to_split(best_split, gid)

    # Build row partitions
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for s, g_list in split_groups.items():
        for gid in g_list:
            for item in groups[gid]:
                item_copy = dict(item)
                item_copy["split"] = s
                split_rows[s].append(item_copy)

    print("\nActual Split Allocation:")
    for s in ["train", "val", "test"]:
        r_list = split_rows[s]
        n_real = sum(1 for r in r_list if r["label"] == "REAL")
        n_ed = sum(1 for r in r_list if r["label"] == "EDITED")
        pct = len(r_list) / total_samples * 100.0
        print(f"  {s.upper():5s}: {len(r_list):3d} samples ({pct:5.2f}%) [{len(split_groups[s]):3d} groups] | REAL: {n_real:3d} | EDITED: {n_ed:2d}")

    # Validation Checks
    print("\nRunning Strict Group-Aware Partition Validation:")
    validate_partition_isolation(split_rows, split_groups)

    # Write manifests
    fieldnames = list(rows[0].keys())
    if "split" not in fieldnames:
        fieldnames.append("split")

    dest_map = {
        "train": TRAIN_MANIFEST_PATH,
        "val": VAL_MANIFEST_PATH,
        "test": TEST_MANIFEST_PATH,
    }

    for s, path in dest_map.items():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows[s])
        print(f"  Saved {len(split_rows[s])} rows to {path.name}")

    print("\n" + "=" * 70)
    print("PHASE 4 GROUP-AWARE SPLITS CREATED SUCCESSFULLY!")
    print("=" * 70)


def validate_partition_isolation(
    split_rows: Dict[str, List[Dict[str, Any]]],
    split_groups: Dict[str, List[str]],
) -> None:
    train_ids = set(r["sample_id"] for r in split_rows["train"])
    val_ids = set(r["sample_id"] for r in split_rows["val"])
    test_ids = set(r["sample_id"] for r in split_rows["test"])

    # 1. No sample overlap
    assert not (train_ids & val_ids), f"FATAL: Train & Val share {len(train_ids & val_ids)} samples!"
    assert not (train_ids & test_ids), f"FATAL: Train & Test share {len(train_ids & test_ids)} samples!"
    assert not (val_ids & test_ids), f"FATAL: Val & Test share {len(val_ids & test_ids)} samples!"
    print("  [PASS] Zero sample ID overlap across partitions.")

    # 2. No SHA-256 overlap
    train_shas = set(r["sha256"] for r in split_rows["train"])
    val_shas = set(r["sha256"] for r in split_rows["val"])
    test_shas = set(r["sha256"] for r in split_rows["test"])

    assert not (train_shas & val_shas), "FATAL: Cryptographic SHA-256 overlap between Train and Val!"
    assert not (train_shas & test_shas), "FATAL: Cryptographic SHA-256 overlap between Train and Test!"
    assert not (val_shas & test_shas), "FATAL: Cryptographic SHA-256 overlap between Val and Test!"
    print("  [PASS] Zero SHA-256 hash overlap across partitions.")

    # 3. No group overlap
    train_g = set(split_groups["train"])
    val_g = set(split_groups["val"])
    test_g = set(split_groups["test"])

    assert not (train_g & val_g), "FATAL: Group leakage between Train and Val!"
    assert not (train_g & test_g), "FATAL: Group leakage between Train and Test!"
    assert not (val_g & test_g), "FATAL: Group leakage between Val and Test!"
    print("  [PASS] Zero group leakage across partitions.")

    # 4. No empty partitions
    assert len(train_ids) > 0, "Train partition is empty!"
    assert len(val_ids) > 0, "Val partition is empty!"
    assert len(test_ids) > 0, "Test partition is empty!"
    print("  [PASS] All partitions are populated.")

    # 5. Image files exist on disk
    for s, r_list in split_rows.items():
        for r in r_list:
            p = Path(r["image_path"])
            assert p.is_file(), f"Missing file: {p}"
    print("  [PASS] All image files verified physically on disk.")


if __name__ == "__main__":
    create_group_aware_splits()
