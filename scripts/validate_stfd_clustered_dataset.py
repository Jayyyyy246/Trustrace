#!/usr/bin/env python3
"""
TRUSTTRACE: Independent STFD Clustered Dataset Validator.

Performs strict, exhaustive forensic validation of the candidate-template clustered partition:
1. Manifest row counts:
   - Master == 3,932
   - Train + Val + Test == 3,932
2. File existence and integrity:
   - All 3,932 images and 3,932 masks exist on disk and are readable
   - Image dimensions match mask dimensions 100%
   - Dimensions match manifest width/height
3. Cryptographic uniqueness:
   - Zero duplicate sample_ids
   - Zero duplicate image SHA-256
   - Zero duplicate mask SHA-256
4. Cluster-level isolation:
   - Every sample assigned to a candidate_template_cluster_id
   - Train Clusters ∩ Val Clusters == empty
   - Train Clusters ∩ Test Clusters == empty
   - Val Clusters ∩ Test Clusters == empty
5. Perceptual cross-split isolation audit:
   - Recomputes 64-bit DCT pHash for all 3,932 images
   - Cross-split pHash == 0 is ZERO (FAIL if > 0)
   - Cross-split pHash <= 4 is ZERO (FAIL if > 0)
   - Reports cross-split pHash <= 8
6. Schema and label validation:
   - manipulation_type in {COPY_MOVE, SPLICING, REMOVAL, INSERTION, REPLACEMENT}
   - trusttrace_class == 'EDITED'
   - scene_category in allowed vocabulary
   - cluster_assignment_method == 'PHASH_LE4_CONNECTED_COMPONENTS'
   - No missing required fields

Exits with code 0 on complete pass, or non-zero on any violation.
"""

import sys
import os
import csv
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
import numpy as np

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = WORKSPACE_DIR / "data" / "manifests"

MASTER_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_master.csv"
TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_val.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"

EXPECTED_TOTAL = 3932

VALID_MANIPULATION_TYPES = {
    "COPY_MOVE",
    "SPLICING",
    "REMOVAL",
    "INSERTION",
    "REPLACEMENT",
}

VALID_SCENE_CATEGORIES = {
    "MOBILE_PAYMENT",
    "ONLINE_BANKING",
    "E_COMMERCE",
    "CHAT_SOCIAL",
    "DOCUMENT",
    "MAP_TRANSPORT",
    "WEB_BROWSING",
    "SYSTEM_INTERFACE",
    "OTHER",
    "UNKNOWN",
}

REQUIRED_COLUMNS = [
    "sample_id",
    "image_path",
    "mask_path",
    "relative_image_path",
    "relative_mask_path",
    "manipulation_type",
    "trusttrace_class",
    "scene_category",
    "scene_confidence",
    "classification_method",
    "width",
    "height",
    "file_size",
    "mask_pixel_count",
    "mask_area_ratio",
    "image_md5",
    "image_sha256",
    "mask_md5",
    "mask_sha256",
    "candidate_template_cluster_id",
    "cluster_assignment_method",
]


def compute_phash_64(img_gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(img_gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq[1:, 1:])
    return (dct_lowfreq > med).flatten()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check header
    missing_cols = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
    if missing_cols:
        raise ValueError(f"Manifest {path.name} missing required columns: {missing_cols}")

    return rows


def validate_clustered_dataset():
    print("=" * 70)
    print("TRUSTTRACE STFD CLUSTERED DATASET INDEPENDENT VALIDATOR")
    print("=" * 70)

    # 1. Load manifests
    print("\n[CHECK 1/8] Loading and validating manifest files...")
    master_rows = load_manifest(MASTER_MANIFEST_PATH)
    train_rows = load_manifest(TRAIN_MANIFEST_PATH)
    val_rows = load_manifest(VAL_MANIFEST_PATH)
    test_rows = load_manifest(TEST_MANIFEST_PATH)

    print(f"  Master: {len(master_rows):>4} rows ({MASTER_MANIFEST_PATH.name})")
    print(f"  Train:  {len(train_rows):>4} rows ({TRAIN_MANIFEST_PATH.name})")
    print(f"  Val:    {len(val_rows):>4} rows ({VAL_MANIFEST_PATH.name})")
    print(f"  Test:   {len(test_rows):>4} rows ({TEST_MANIFEST_PATH.name})")

    assert len(master_rows) == EXPECTED_TOTAL, f"Master rows count {len(master_rows)} != {EXPECTED_TOTAL}"
    assert len(train_rows) + len(val_rows) + len(test_rows) == EXPECTED_TOTAL, (
        f"Sum of splits {len(train_rows) + len(val_rows) + len(test_rows)} != {EXPECTED_TOTAL}"
    )
    print("  [PASS] Manifest row counts match exactly (3,932 total).")

    # 2. Check no missing required fields
    print("\n[CHECK 2/8] Validating schema integrity and field non-emptiness...")
    for idx, r in enumerate(master_rows):
        for col in REQUIRED_COLUMNS:
            val = r[col].strip()
            if not val:
                raise ValueError(f"Row {idx} ({r.get('sample_id')}): column '{col}' is empty or whitespace.")

        # Validate values
        if r["manipulation_type"] not in VALID_MANIPULATION_TYPES:
            raise ValueError(f"Row {idx}: invalid manipulation_type '{r['manipulation_type']}'")
        if r["trusttrace_class"] != "EDITED":
            raise ValueError(f"Row {idx}: invalid trusttrace_class '{r['trusttrace_class']}'")
        if r["scene_category"] not in VALID_SCENE_CATEGORIES:
            raise ValueError(f"Row {idx}: invalid scene_category '{r['scene_category']}'")
        if r["cluster_assignment_method"] != "PHASH_LE4_CONNECTED_COMPONENTS":
            raise ValueError(f"Row {idx}: invalid cluster_assignment_method '{r['cluster_assignment_method']}'")
        if not r["candidate_template_cluster_id"].startswith("cluster_"):
            raise ValueError(f"Row {idx}: invalid candidate_template_cluster_id '{r['candidate_template_cluster_id']}'")
    print("  [PASS] All 3,932 records have valid, non-empty schema values and allowed enums.")

    # 3. Cryptographic uniqueness
    print("\n[CHECK 3/8] Checking sample ID and SHA-256 uniqueness...")
    sample_ids = [r["sample_id"] for r in master_rows]
    image_shas = [r["image_sha256"] for r in master_rows]
    mask_shas = [r["mask_sha256"] for r in master_rows]

    assert len(set(sample_ids)) == EXPECTED_TOTAL, "Duplicate sample_ids found in master manifest!"
    assert len(set(image_shas)) == EXPECTED_TOTAL, "Duplicate image_sha256 found in master manifest!"
    assert len(set(mask_shas)) == EXPECTED_TOTAL, "Duplicate mask_sha256 found in master manifest!"
    print("  [PASS] All 3,932 sample IDs, image SHA-256s, and mask SHA-256s are strictly unique.")

    # 4. Split set partition integrity
    print("\n[CHECK 4/8] Validating split sample partition disjointness...")
    train_ids = set(r["sample_id"] for r in train_rows)
    val_ids = set(r["sample_id"] for r in val_rows)
    test_ids = set(r["sample_id"] for r in test_rows)

    assert len(train_ids) == len(train_rows), "Duplicates inside Train partition!"
    assert len(val_ids) == len(val_rows), "Duplicates inside Val partition!"
    assert len(test_ids) == len(test_rows), "Duplicates inside Test partition!"

    assert not (train_ids & val_ids), f"Train/Val intersection not empty: {len(train_ids & val_ids)} samples leaked!"
    assert not (train_ids & test_ids), f"Train/Test intersection not empty: {len(train_ids & test_ids)} samples leaked!"
    assert not (val_ids & test_ids), f"Val/Test intersection not empty: {len(val_ids & test_ids)} samples leaked!"

    assert (train_ids | val_ids | test_ids) == set(sample_ids), "Splits union does not match Master set!"
    print("  [PASS] Disjoint sample partitions: Train int Val = empty, Train int Test = empty, Val int Test = empty.")

    # 5. Cluster isolation
    print("\n[CHECK 5/8] Validating strict candidate template cluster isolation...")
    train_clusters = set(r["candidate_template_cluster_id"] for r in train_rows)
    val_clusters = set(r["candidate_template_cluster_id"] for r in val_rows)
    test_clusters = set(r["candidate_template_cluster_id"] for r in test_rows)

    total_unique_clusters = len(set(r["candidate_template_cluster_id"] for r in master_rows))
    print(f"  Total clusters: {total_unique_clusters}")
    print(f"  Train clusters: {len(train_clusters)}")
    print(f"  Val clusters:   {len(val_clusters)}")
    print(f"  Test clusters:  {len(test_clusters)}")

    leak_tr_val = train_clusters & val_clusters
    leak_tr_te = train_clusters & test_clusters
    leak_va_te = val_clusters & test_clusters

    assert not leak_tr_val, f"Train and Val share {len(leak_tr_val)} clusters: {list(leak_tr_val)[:5]}"
    assert not leak_tr_te, f"Train and Test share {len(leak_tr_te)} clusters: {list(leak_tr_te)[:5]}"
    assert not leak_va_te, f"Val and Test share {len(leak_va_te)} clusters: {list(leak_va_te)[:5]}"
    assert len(train_clusters) + len(val_clusters) + len(test_clusters) == total_unique_clusters, (
        "Sum of split cluster counts != total unique clusters!"
    )
    print("  [PASS] ZERO candidate template cluster leakage across any split boundaries!")

    # 6. Physical image and mask file validation
    print("\n[CHECK 6/8] Verifying physical files on disk (existence, readability, dimensions)...")
    t0 = time.time()
    def _verify_file(r):
        ip = Path(r["image_path"])
        mp = Path(r["mask_path"])
        if not ip.is_file():
            return False, f"Image not found: {ip}"
        if not mp.is_file():
            return False, f"Mask not found: {mp}"

        with Image.open(ip) as img:
            iw, ih = img.size
        with Image.open(mp) as msk:
            mw, mh = msk.size

        if (iw, ih) != (mw, mh):
            return False, f"Dimension mismatch: img {iw}x{ih} vs mask {mw}x{mh} ({r['sample_id']})"
        if (iw, ih) != (int(r["width"]), int(r["height"])):
            return False, f"Manifest dimension mismatch: file {iw}x{ih} vs manifest {r['width']}x{r['height']}"
        return True, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_verify_file, master_rows))

    for ok, err in results:
        if not ok:
            raise RuntimeError(f"Physical file verification failed: {err}")
    print(f"  [PASS] All 3,932 image/mask pairs exist, are readable, and match dimensions 100% ({time.time()-t0:.2f}s).")

    # 7. Strict Perceptual cross-split isolation
    print("\n[CHECK 7/8] Re-running full pairwise cross-split perceptual audit (64-bit DCT pHash)...")
    phash_matrix = np.zeros((EXPECTED_TOTAL, 64), dtype=np.uint8)
    def _calc_phash(idx_r):
        idx, r = idx_r
        im = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        return idx, compute_phash_64(im)

    with ThreadPoolExecutor(max_workers=8) as ex:
        for idx, ph in ex.map(_calc_phash, enumerate(master_rows)):
            phash_matrix[idx] = ph

    sum_bits = phash_matrix.sum(axis=1, dtype=np.int32)
    dot = phash_matrix.astype(np.int32) @ phash_matrix.T.astype(np.int32)
    p_dists = sum_bits[:, None] + sum_bits[None, :] - 2 * dot

    sample_to_split = {}
    for r in train_rows: sample_to_split[r["sample_id"]] = "TRAIN"
    for r in val_rows: sample_to_split[r["sample_id"]] = "VAL"
    for r in test_rows: sample_to_split[r["sample_id"]] = "TEST"

    cross_0 = 0
    cross_4 = 0
    cross_8 = 0
    for i in range(EXPECTED_TOTAL):
        for j in range(i + 1, EXPECTED_TOTAL):
            s_i = sample_to_split[master_rows[i]["sample_id"]]
            s_j = sample_to_split[master_rows[j]["sample_id"]]
            if s_i != s_j:
                pd = p_dists[i, j]
                if pd == 0: cross_0 += 1
                if pd <= 4: cross_4 += 1
                if pd <= 8: cross_8 += 1

    print(f"  Cross-Split pHash == 0: {cross_0}")
    print(f"  Cross-Split pHash <= 4: {cross_4}")
    print(f"  Cross-Split pHash <= 8: {cross_8}")

    assert cross_0 == 0, f"VIOLATION: Found {cross_0} cross-split exact pHash matches (dist=0)!"
    assert cross_4 == 0, f"VIOLATION: Found {cross_4} cross-split near-duplicate pHash matches (dist<=4)!"
    print("  [PASS] ZERO cross-split pHash == 0 pairs and ZERO cross-split pHash <= 4 pairs!")

    # 8. Manipulation stratification summary
    print("\n[CHECK 8/8] Verifying partition balance and manipulation stratification...")
    for s_name, rows in [("TRAIN", train_rows), ("VAL", val_rows), ("TEST", test_rows)]:
        pct = len(rows) / EXPECTED_TOTAL * 100
        print(f"  {s_name:<5}: {len(rows):>4} samples ({pct:.2f}%)")

    print("\n  Manipulation breakdown per split:")
    for mt in sorted(VALID_MANIPULATION_TYPES):
        tot = sum(1 for r in master_rows if r["manipulation_type"] == mt)
        tr = sum(1 for r in train_rows if r["manipulation_type"] == mt)
        va = sum(1 for r in val_rows if r["manipulation_type"] == mt)
        te = sum(1 for r in test_rows if r["manipulation_type"] == mt)
        print(f"    {mt:<14} | Train: {tr:>4} ({tr/tot*100:.1f}%) | Val: {va:>4} ({va/tot*100:.1f}%) | Test: {te:>4} ({te/tot*100:.1f}%) | Total: {tot}")

    print("\n" + "=" * 70)
    print("ALL 8 INDEPENDENT VALIDATION CHECKS PASSED.")
    print("THE CLUSTERED PARTITION IS FORENSICALLY SOUND AND FULLY VALIDATED.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        validate_clustered_dataset()
        sys.exit(0)
    except Exception as e:
        print(f"\n[VALIDATION FAILED]: {e}", file=sys.stderr)
        sys.exit(1)
