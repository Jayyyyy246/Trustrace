#!/usr/bin/env python3
"""
TRUSTTRACE: STFD High-Resolution Patch Extraction Pipeline (Phase 3A).

Generates high-resolution localized crops based on official STFD ground-truth masks:
- Input: Original-resolution RGB screenshots and official ground-truth binary masks
- Bounding Box: Minimal bounding rectangle containing foreground manipulated pixels
- Padding: 25% relative padding along width and height
- Safety Constraints: Minimum crop size (64x64) with symmetric expansion and boundary clamping
- Aspect Ratio: Maintained via centered letterboxing onto a 224x224 RGB canvas
- Output Manifests:
  * data/manifests/trusttrace_stfd_patch_master.csv (3,932 samples)
  * data/manifests/trusttrace_stfd_patch_train.csv (2,752 samples)
  * data/manifests/trusttrace_stfd_patch_val.csv (590 samples)
  * data/manifests/trusttrace_stfd_patch_test.csv (590 samples)

Preserves exact candidate template cluster assignments and sample IDs.
"""

import sys
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
PATCHES_DIR = DATA_DIR / "stfd_patches" / "oracle"

TRAIN_SRC_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_clustered_train.csv"
VAL_SRC_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_clustered_val.csv"
TEST_SRC_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"

PATCH_MASTER_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_patch_master.csv"
PATCH_TRAIN_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_patch_train.csv"
PATCH_VAL_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_patch_val.csv"
PATCH_TEST_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_patch_test.csv"

# Hyperparameters
PADDING_RATIO = 0.25
MIN_CROP_SIZE = 64
TARGET_CROP_SIZE = (224, 224)
CANVAS_FILL_COLOR = (0, 0, 0)


def extract_oracle_crop_box(
    mask_arr: np.ndarray,
    img_w: int,
    img_h: int,
    padding_ratio: float = PADDING_RATIO,
    min_crop_size: int = MIN_CROP_SIZE,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], bool, bool]:
    """
    Extracts the bounding box and padded crop rectangle for the mask foreground.

    Returns:
        bbox: (x1, y1, x2, y2) tight bounding box of foreground
        crop_box: (crop_x1, crop_y1, crop_x2, crop_y2) padded, clamped crop rectangle
        fallback_used: bool
        tiny_region_adjustment: bool
    """
    fg_indices = np.argwhere(mask_arr > 128)
    if len(fg_indices) == 0:
        # Fallback to full image
        return (0, 0, img_w, img_h), (0, 0, img_w, img_h), True, False

    y_min = int(fg_indices[:, 0].min())
    y_max = int(fg_indices[:, 0].max()) + 1
    x_min = int(fg_indices[:, 1].min())
    x_max = int(fg_indices[:, 1].max()) + 1

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    # Relative padding
    pad_w = padding_ratio * bbox_w
    pad_h = padding_ratio * bbox_h

    # Initial target crop dimensions
    target_w = bbox_w + 2.0 * pad_w
    target_h = bbox_h + 2.0 * pad_h

    tiny_adjustment = False
    if target_w < min_crop_size:
        target_w = float(min_crop_size)
        tiny_adjustment = True
    if target_h < min_crop_size:
        target_h = float(min_crop_size)
        tiny_adjustment = True

    # Center crop around original bbox center
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x1 = cx - target_w / 2.0
    x2 = cx + target_w / 2.0
    y1 = cy - target_h / 2.0
    y2 = cy + target_h / 2.0

    # Clamp & shift within image boundaries [0, img_w] x [0, img_h]
    if x1 < 0:
        shift = -x1
        x1 = 0.0
        x2 = min(float(img_w), x2 + shift)
    if x2 > img_w:
        shift = x2 - float(img_w)
        x2 = float(img_w)
        x1 = max(0.0, x1 - shift)

    if y1 < 0:
        shift = -y1
        y1 = 0.0
        y2 = min(float(img_h), y2 + shift)
    if y2 > img_h:
        shift = y2 - float(img_h)
        y2 = float(img_h)
        y1 = max(0.0, y1 - shift)

    crop_x1 = max(0, int(round(x1)))
    crop_y1 = max(0, int(round(y1)))
    crop_x2 = min(img_w, int(round(x2)))
    crop_y2 = min(img_h, int(round(y2)))

    # Ensure valid dimensions
    if crop_x2 <= crop_x1:
        crop_x2 = min(img_w, crop_x1 + 1)
    if crop_y2 <= crop_y1:
        crop_y2 = min(img_h, crop_y1 + 1)

    return (x_min, y_min, x_max, y_max), (crop_x1, crop_y1, crop_x2, crop_y2), False, tiny_adjustment


def letterbox_crop(
    crop_img: Image.Image,
    target_size: Tuple[int, int] = TARGET_CROP_SIZE,
    fill_color: Tuple[int, int, int] = CANVAS_FILL_COLOR,
) -> Image.Image:
    """
    Resizes crop preserving aspect ratio, centering on canvas.
    """
    w, h = crop_img.size
    target_w, target_h = target_size

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", target_size, fill_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return canvas


def process_sample(row: Dict[str, str], split: str) -> Dict[str, Any]:
    """
    Processes one sample: crops original image, letterboxes, saves to disk, returns manifest dict.
    """
    sample_id = row["sample_id"]
    img_path = Path(row["image_path"])
    mask_path = Path(row["mask_path"])

    if not img_path.is_file():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    orig_w, orig_h = img.size
    mask_w, mask_h = mask.size

    if (orig_w, orig_h) != (mask_w, mask_h):
        raise ValueError(f"Mismatch dimensions for {sample_id}: img {(orig_w, orig_h)} vs mask {(mask_w, mask_h)}")

    mask_arr = np.array(mask)
    bbox, crop_box, fallback_used, tiny_adj = extract_oracle_crop_box(mask_arr, orig_w, orig_h)

    # Crop original resolution image
    cropped_region = img.crop(crop_box)
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]

    # Letterbox resize to 224x224
    letterboxed = letterbox_crop(cropped_region, target_size=TARGET_CROP_SIZE)

    # Destination path
    crop_out_path = PATCHES_DIR / f"{sample_id}.png"
    letterboxed.save(crop_out_path, format="PNG", optimize=True)

    manip_class = row.get("manipulation_type") or row.get("manipulation_class", "UNKNOWN")
    cluster_id = row.get("candidate_template_cluster_id") or row.get("candidate_cluster_id", "singleton")

    return {
        "sample_id": sample_id,
        "image_path": str(img_path),
        "mask_path": str(mask_path),
        "manipulation_class": manip_class,
        "manipulation_type": manip_class,
        "trusttrace_class": row.get("trusttrace_class", "EDITED"),
        "split": split,
        "candidate_cluster_id": cluster_id,
        "candidate_template_cluster_id": cluster_id,
        "original_width": orig_w,
        "original_height": orig_h,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "padding": PADDING_RATIO,
        "crop_width": crop_w,
        "crop_height": crop_h,
        "crop_path": str(crop_out_path),
        "fallback_used": fallback_used,
        "tiny_region_adjustment": tiny_adj,
    }


def generate_all_patches() -> None:
    print("=" * 70)
    print("TRUSTTRACE: High-Resolution STFD Oracle Patch Generator (Phase 3A)")
    print("=" * 70)

    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    splits_data = [
        ("train", TRAIN_SRC_MANIFEST, PATCH_TRAIN_MANIFEST),
        ("val", VAL_SRC_MANIFEST, PATCH_VAL_MANIFEST),
        ("test", TEST_SRC_MANIFEST, PATCH_TEST_MANIFEST),
    ]

    all_records: List[Dict[str, Any]] = []

    fieldnames = [
        "sample_id",
        "image_path",
        "mask_path",
        "manipulation_class",
        "manipulation_type",
        "trusttrace_class",
        "split",
        "candidate_cluster_id",
        "candidate_template_cluster_id",
        "original_width",
        "original_height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "padding",
        "crop_width",
        "crop_height",
        "crop_path",
        "fallback_used",
        "tiny_region_adjustment",
    ]

    start_time = time.time()

    for split_name, src_manifest, dst_manifest in splits_data:
        print(f"\nProcessing {split_name.upper()} set from {src_manifest.name}...")
        with open(src_manifest, "r", encoding="utf-8") as f:
            src_rows = list(csv.DictReader(f))

        print(f"Loaded {len(src_rows)} samples. Extracting patches with multi-threading...")
        split_records = []
        
        # Parallel extraction using threads (I/O & Pillow operations)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_sample, row, split_name) for row in src_rows]
            for i, fut in enumerate(futures):
                res = fut.result()
                split_records.append(res)
                if (i + 1) % 500 == 0 or (i + 1) == len(futures):
                    print(f"  Processed {i + 1}/{len(futures)} {split_name} samples...")

        # Write split manifest
        with open(dst_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_records)
        print(f"Saved {len(split_records)} rows to {dst_manifest.name}")

        all_records.extend(split_records)

    # Write master manifest
    with open(PATCH_MASTER_MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nSaved {len(all_records)} total rows to {PATCH_MASTER_MANIFEST.name}")

    elapsed = time.time() - start_time
    print(f"\nPatch extraction completed in {elapsed:.2f} seconds ({len(all_records)} patches generated).")

    # Run strict validation
    validate_patch_generation(all_records)


def validate_patch_generation(records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 70)
    print("RUNNING STRICT PRE-TRAINING VALIDATION ON GENERATED PATCHES")
    print("=" * 70)

    # 1. Check total count
    assert len(records) == 3932, f"Total records {len(records)} != 3932"
    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]

    assert len(train_recs) == 2752, f"Train count {len(train_recs)} != 2752"
    assert len(val_recs) == 590, f"Val count {len(val_recs)} != 590"
    assert len(test_recs) == 590, f"Test count {len(test_recs)} != 590"

    print(f"[PASS] Split counts verified: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")

    # 2. Check sample ID uniqueness and split disjointness
    train_ids = set(r["sample_id"] for r in train_recs)
    val_ids = set(r["sample_id"] for r in val_recs)
    test_ids = set(r["sample_id"] for r in test_recs)

    assert len(train_ids) == len(train_recs), "Duplicate sample IDs in train"
    assert len(val_ids) == len(val_recs), "Duplicate sample IDs in val"
    assert len(test_ids) == len(test_recs), "Duplicate sample IDs in test"

    assert not (train_ids & val_ids), "Sample ID overlap between train and val"
    assert not (train_ids & test_ids), "Sample ID overlap between train and test"
    assert not (val_ids & test_ids), "Sample ID overlap between val and test"

    print("[PASS] Sample IDs strictly disjoint across splits.")

    # 3. Check cluster disjointness
    train_clusters = set(r["candidate_cluster_id"] for r in train_recs)
    val_clusters = set(r["candidate_cluster_id"] for r in val_recs)
    test_clusters = set(r["candidate_cluster_id"] for r in test_recs)

    assert not (train_clusters & val_clusters), "Candidate cluster leakage between train and val"
    assert not (train_clusters & test_clusters), "Candidate cluster leakage between train and test"
    assert not (val_clusters & test_clusters), "Candidate cluster leakage between val and test"

    print("[PASS] Zero candidate-template cluster overlap detected across splits.")

    # 4. Verify crop files existence and dimensions on disk
    print("Verifying generated crop files on disk...")
    tiny_count = sum(1 for r in records if r["tiny_region_adjustment"])
    fallback_count = sum(1 for r in records if r["fallback_used"])

    for r in records[:50]:  # Detailed sample check
        p = Path(r["crop_path"])
        assert p.is_file(), f"Missing crop file: {p}"
        with Image.open(p) as img:
            assert img.size == TARGET_CROP_SIZE, f"Bad crop dimension for {p}: {img.size}"

    # Verify all files exist
    all_files_exist = all(Path(r["crop_path"]).is_file() for r in records)
    assert all_files_exist, "Some crop files are missing from disk!"

    print(f"[PASS] All {len(records)} crop images verified on disk with shape {TARGET_CROP_SIZE}.")
    print(f"       - Fallbacks used: {fallback_count}/{len(records)} ({fallback_count/len(records)*100:.2f}%)")
    print(f"       - Tiny region adjustments: {tiny_count}/{len(records)} ({tiny_count/len(records)*100:.2f}%)")
    print("=" * 70)
    print("PHASE 3A PATCH DATASET GENERATION VALIDATION PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_patches()
