#!/usr/bin/env python3
"""
TRUSTTRACE: STFD Master Manifest Builder.
Constructs data/manifests/trusttrace_stfd_master.csv by joining:
1. stfd_audit_manifest.csv (forensic integrity audit)
2. data/stfd_scene_inventory.csv (inferred scene metadata)
3. Direct mask pixel verification & area ratio calculation

Zero modification to the original dataset.
"""

import os
import sys
import csv
import time
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
AUDIT_MANIFEST_PATH = WORKSPACE_DIR / "stfd_audit_manifest.csv"
SCENE_INVENTORY_PATH = DATA_DIR / "stfd_scene_inventory.csv"
OUTPUT_MASTER_PATH = MANIFESTS_DIR / "trusttrace_stfd_master.csv"

# Category mapping
MANIPULATION_MAPPING: Dict[str, str] = {
    "1_Copy-move": "COPY_MOVE",
    "2_Splicing": "SPLICING",
    "3_Removal": "REMOVAL",
    "4_Insertion": "INSERTION",
    "5_Replacement": "REPLACEMENT",
}

FIELDNAMES = [
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
]


def load_audit_manifest() -> Dict[str, Dict[str, Any]]:
    """Loads audit manifest separated into images and masks."""
    if not AUDIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing required audit manifest: {AUDIT_MANIFEST_PATH}")

    images: Dict[str, Dict[str, Any]] = {}
    masks: Dict[str, Dict[str, Any]] = {}

    with open(AUDIT_MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip non-images (like Readme.md)
            if row["image_valid"] != "True":
                continue

            rel_p = row["relative_path"]
            if row["is_mask"] == "True":
                # Key by (category, filename)
                cat = row["category"]
                fn = Path(rel_p).name
                masks[(cat, fn)] = row
            else:
                cat = row["category"]
                fn = Path(rel_p).name
                images[(cat, fn)] = row

    return {"images": images, "masks": masks}


def load_scene_inventory() -> Dict[str, Dict[str, Any]]:
    """Loads inferred scene inventory indexed by normalized absolute image path."""
    if not SCENE_INVENTORY_PATH.is_file():
        raise FileNotFoundError(f"Missing required scene inventory: {SCENE_INVENTORY_PATH}")

    inventory: Dict[str, Dict[str, Any]] = {}
    with open(SCENE_INVENTORY_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm_p = os.path.normpath(row["image_path"]).lower()
            inventory[norm_p] = row

    return inventory


def build_master_manifest() -> int:
    start_time = time.time()
    print("=== Building TRUSTTRACE STFD Master Manifest ===")
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    audit_data = load_audit_manifest()
    images = audit_data["images"]
    masks = audit_data["masks"]
    scene_inventory = load_scene_inventory()

    print(f"Loaded {len(images)} valid images and {len(masks)} masks from audit manifest.")
    print(f"Loaded {len(scene_inventory)} records from scene inventory.")

    if len(images) != 3932:
        raise ValueError(f"Expected exactly 3,932 images, found {len(images)}")
    if len(masks) != 3932:
        raise ValueError(f"Expected exactly 3,932 masks, found {len(masks)}")

    master_rows: List[Dict[str, Any]] = []
    seen_sample_ids = set()
    seen_image_hashes = set()
    unmatched_scene_count = 0

    print("Verifying masks and calculating pixel statistics...")
    for idx, ((cat, fn), img_row) in enumerate(sorted(images.items())):
        mask_row = masks.get((cat, fn))
        if mask_row is None:
            raise ValueError(f"Missing corresponding mask for image ({cat}, {fn})")

        img_path = img_row["absolute_path"]
        mask_path = mask_row["absolute_path"]

        # Deterministic sample_id based on image filename stem
        stem = Path(fn).stem
        sample_id = f"stfd_{stem}"
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id detected: {sample_id}")
        seen_sample_ids.add(sample_id)

        # Image SHA-256 uniqueness
        img_sha = img_row["sha256"]
        if img_sha in seen_image_hashes:
            raise ValueError(f"Duplicate image SHA-256 detected: {img_sha} for {img_path}")
        seen_image_hashes.add(img_sha)

        # Dimension checks
        w = int(img_row["width"])
        h = int(img_row["height"])
        mw = int(mask_row["width"])
        mh = int(mask_row["height"])
        if (w, h) != (mw, mh):
            raise ValueError(f"Dimension mismatch for {fn}: Image ({w}x{h}) vs Mask ({mw}x{mh})")

        # Read mask to calculate pixel statistics
        mask_mat = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_mat is None:
            raise ValueError(f"Failed to load mask image: {mask_path}")

        # Count positive (tampered) pixels
        pos_pixels = int(np.count_nonzero(mask_mat > 127))
        total_pixels = w * h
        area_ratio = round(pos_pixels / total_pixels, 8) if total_pixels > 0 else 0.0

        # Manipulation type
        manip_type = MANIPULATION_MAPPING.get(cat)
        if not manip_type:
            raise ValueError(f"Unknown manipulation category: {cat}")

        # Scene inventory join
        norm_img_p = os.path.normpath(img_path).lower()
        scene_row = scene_inventory.get(norm_img_p)

        if scene_row:
            scene_cat = scene_row["scene_category"]
            scene_conf = float(scene_row["scene_confidence"])
            scene_meth = scene_row["classification_method"]
        else:
            unmatched_scene_count += 1
            scene_cat = "UNKNOWN"
            scene_conf = 0.0
            scene_meth = "UNMATCHED"

        master_rows.append({
            "sample_id": sample_id,
            "image_path": img_path,
            "mask_path": mask_path,
            "relative_image_path": img_row["relative_path"],
            "relative_mask_path": mask_row["relative_path"],
            "manipulation_type": manip_type,
            "trusttrace_class": "EDITED",
            "scene_category": scene_cat,
            "scene_confidence": scene_conf,
            "classification_method": scene_meth,
            "width": w,
            "height": h,
            "file_size": int(img_row["file_size"]),
            "mask_pixel_count": pos_pixels,
            "mask_area_ratio": area_ratio,
            "image_md5": img_row["md5"],
            "image_sha256": img_row["sha256"],
            "mask_md5": mask_row["md5"],
            "mask_sha256": mask_row["sha256"],
        })

    # Write master manifest
    with open(OUTPUT_MASTER_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(master_rows)

    duration = round(time.time() - start_time, 2)
    print(f"Master manifest successfully generated: {OUTPUT_MASTER_PATH}")
    print(f"Total rows written: {len(master_rows)}")
    print(f"Unmatched scene records: {unmatched_scene_count}")
    print(f"Elapsed time: {duration}s")
    return 0


if __name__ == "__main__":
    sys.exit(build_master_manifest())
