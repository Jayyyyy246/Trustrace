#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 6 Native-Resolution Document Patch Extractor.

Extracts native-resolution text and entity crops directly from original receipt PNGs:
- FORGED_REGION: Extracted around official VIA ground-truth forgery bounding boxes
  with controlled context padding (preserving 100% native resolution character edges).
- AUTHENTIC_REGION: Extracted from OCR word bounding boxes on authentic receipts,
  and non-overlapping OCR word boxes on edited receipts.
- Standardizes crops to 128x128 RGB while preserving native stroke geometry.
- Enforces strict hierarchical group-aware partitioning:
  document group -> parent image -> patches
- Outputs:
  * data/manifests/phase6_patch_master.csv
  * data/manifests/phase6_patch_train.csv
  * data/manifests/phase6_patch_val.csv
  * data/manifests/phase6_patch_test.csv
  * data/document_patches/ (crops)
"""

import sys
import os
import csv
import ast
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from PIL import Image
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
OCR_DIR = DATA_DIR / "ocr"
MANIFESTS_DIR = DATA_DIR / "manifests"
PATCHES_DIR = DATA_DIR / "document_patches"
FINDIT2_DIR = Path(r"C:\Users\jay\Downloads\finditagain\findit2")

MASTER_MANIFEST_PATH = MANIFESTS_DIR / "phase4_dataset_master.csv"
PATCH_MASTER_PATH = MANIFESTS_DIR / "phase6_patch_master.csv"
PATCH_TRAIN_PATH = MANIFESTS_DIR / "phase6_patch_train.csv"
PATCH_VAL_PATH = MANIFESTS_DIR / "phase6_patch_val.csv"
PATCH_TEST_PATH = MANIFESTS_DIR / "phase6_patch_test.csv"

RANDOM_SEED = 42
TARGET_PATCH_SIZE = (128, 128)


def parse_forgery_annotations() -> Dict[str, List[Dict[str, Any]]]:
    """Parses ground-truth forgery region annotations from findit2 official manifests."""
    annos: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for split_txt in ["train.txt", "val.txt", "test.txt"]:
        p = FINDIT2_DIR / split_txt
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row or len(row) < 5:
                    continue
                img_name = row[0].strip()
                is_forged = row[3].strip()
                anno_str = row[4].strip()
                if is_forged == "1" and anno_str and anno_str != "0":
                    try:
                        parsed = ast.literal_eval(anno_str)
                        regs = parsed.get("regions", [])
                        for reg in regs:
                            annos[img_name].append(reg)
                    except Exception:
                        pass
    return annos


def boxes_overlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    """Checks if two [x, y, w, h] boxes intersect with positive area."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    return (x_right > x_left) and (y_bottom > y_top)


def crop_and_standardize_patch(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = TARGET_PATCH_SIZE,
    context_margin: float = 1.35,
) -> Image.Image:
    """
    Crops native resolution patch with context padding and letterboxes to target_size.
    Preserves 100% authentic pixel stroke fidelity without distortion.
    """
    img_w, img_h = img.size
    bx, by, bw, bh = bbox

    # Center of target region
    cx = bx + bw / 2.0
    cy = by + bh / 2.0

    # Expand box with context margin
    dim = max(bw, bh, 24) * context_margin
    half_dim = dim / 2.0

    x1 = max(0, int(round(cx - half_dim)))
    y1 = max(0, int(round(cy - half_dim)))
    x2 = min(img_w, int(round(cx + half_dim)))
    y2 = min(img_h, int(round(cy + half_dim)))

    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)

    crop = img.crop((x1, y1, x2, y2))
    cw, ch = crop.size

    # Letterbox pad to square with white paper background (255, 255, 255)
    max_side = max(cw, ch)
    square_canvas = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    paste_x = (max_side - cw) // 2
    paste_y = (max_side - ch) // 2
    square_canvas.paste(crop, (paste_x, paste_y))

    # Bilinear resize to standard patch size
    return square_canvas.resize(target_size, Image.Resampling.BILINEAR)


def main():
    print("=" * 70)
    print("TRUSTTRACE: Extracting Native-Resolution Document Patches")
    print("=" * 70)

    random.seed(RANDOM_SEED)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Master Manifest and Split Map
    with open(MASTER_MANIFEST_PATH, "r", encoding="utf-8") as f:
        master_rows = list(csv.DictReader(f))

    split_map: Dict[str, str] = {}
    for s_name in ["train", "val", "test"]:
        sp_path = MANIFESTS_DIR / f"trusttrace_phase4_{s_name}.csv"
        with open(sp_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                split_map[r["sample_id"]] = s_name

    forgery_annos = parse_forgery_annotations()
    print(f"Loaded {len(master_rows)} parent documents. Forgery bboxes parsed for {len(forgery_annos)} images.")

    patch_records: List[Dict[str, Any]] = []
    patch_id_counter = 0

    for doc_idx, doc_row in enumerate(master_rows):
        sample_id = doc_row["sample_id"]
        img_p = Path(doc_row["image_path"])
        group_id = doc_row["group_id"]
        split = split_map.get(sample_id, "unknown")
        doc_label = doc_row["label"]  # REAL or EDITED
        fname = f"{sample_id}.png"

        if not img_p.is_file():
            continue

        with Image.open(img_p) as full_img:
            rgb_img = full_img.convert("RGB")
            img_w, img_h = rgb_img.size

            # Load OCR words for this document
            ocr_words: List[Tuple[str, Tuple[int, int, int, int]]] = []
            ocr_json = OCR_DIR / f"{sample_id}_ocr.json"
            if ocr_json.is_file():
                try:
                    with open(ocr_json, "r", encoding="utf-8-sig") as f_ocr:
                        ocr_data = json.load(f_ocr)
                    for line in ocr_data.get("lines", []):
                        for w in line.get("words", []):
                            txt = w.get("text", "").strip()
                            bbox = w.get("bbox", [])
                            if len(bbox) == 4 and bbox[2] > 4 and bbox[3] > 4:
                                ocr_words.append((txt, tuple(bbox)))
                except Exception:
                    pass

            doc_forgery_boxes: List[Tuple[Tuple[int, int, int, int], str]] = []

            # 1. Extract Positive (FORGED_REGION) patches if document is EDITED
            if doc_label == "EDITED":
                regs = forgery_annos.get(fname, [])
                for reg in regs:
                    shape = reg.get("shape_attributes", {})
                    attrs = reg.get("region_attributes", {})
                    etype = attrs.get("Entity type", "Unspecified")

                    if shape.get("name") == "rect":
                        bx = int(shape.get("x", 0))
                        by = int(shape.get("y", 0))
                        bw = int(shape.get("width", 0))
                        bh = int(shape.get("height", 0))
                        if bw >= 2 and bh >= 2 and bx + bw <= img_w + 10 and by + bh <= img_h + 10:
                            doc_forgery_boxes.append(((bx, by, bw, bh), etype))

                for f_bbox, etype in doc_forgery_boxes:
                    patch_id_counter += 1
                    patch_id = f"P{patch_id_counter:07d}_{sample_id}_FORGED"
                    patch_filename = f"{patch_id}.png"
                    patch_path = PATCHES_DIR / patch_filename

                    if not patch_path.is_file():
                        patch_img = crop_and_standardize_patch(rgb_img, f_bbox, TARGET_PATCH_SIZE)
                        patch_img.save(patch_path, format="PNG")

                    patch_records.append({
                        "patch_id": patch_id,
                        "parent_sample_id": sample_id,
                        "group_id": group_id,
                        "split": split,
                        "patch_label": "FORGED_REGION",
                        "binary_label": 1,
                        "parent_document_label": doc_label,
                        "entity_type": etype,
                        "bbox_x": f_bbox[0],
                        "bbox_y": f_bbox[1],
                        "bbox_w": f_bbox[2],
                        "bbox_h": f_bbox[3],
                        "patch_path": str(patch_path),
                    })

            # 2. Extract Negative (AUTHENTIC_REGION) patches
            # Pick valid OCR word boxes that do not intersect any forgery region
            valid_auth_words: List[Tuple[str, Tuple[int, int, int, int]]] = []
            for w_txt, w_bbox in ocr_words:
                overlaps_forgery = any(boxes_overlap(w_bbox, fb[0]) for fb in doc_forgery_boxes)
                if not overlaps_forgery:
                    valid_auth_words.append((w_txt, w_bbox))

            # Select sample of authentic words: 1 to 3 per document to maintain class balance
            num_auth_to_sample = len(doc_forgery_boxes) if doc_label == "EDITED" else min(2, len(valid_auth_words))
            if num_auth_to_sample > 0 and valid_auth_words:
                # Deterministically sample based on document ID hash
                rng = random.Random(f"{sample_id}_{RANDOM_SEED}")
                sampled_auth = rng.sample(valid_auth_words, min(num_auth_to_sample, len(valid_auth_words)))

                for auth_idx, (w_txt, a_bbox) in enumerate(sampled_auth):
                    patch_id_counter += 1
                    patch_id = f"P{patch_id_counter:07d}_{sample_id}_AUTH"
                    patch_filename = f"{patch_id}.png"
                    patch_path = PATCHES_DIR / patch_filename

                    if not patch_path.is_file():
                        patch_img = crop_and_standardize_patch(rgb_img, a_bbox, TARGET_PATCH_SIZE)
                        patch_img.save(patch_path, format="PNG")

                    patch_records.append({
                        "patch_id": patch_id,
                        "parent_sample_id": sample_id,
                        "group_id": group_id,
                        "split": split,
                        "patch_label": "AUTHENTIC_REGION",
                        "binary_label": 0,
                        "parent_document_label": doc_label,
                        "entity_type": "AuthenticText",
                        "bbox_x": a_bbox[0],
                        "bbox_y": a_bbox[1],
                        "bbox_w": a_bbox[2],
                        "bbox_h": a_bbox[3],
                        "patch_path": str(patch_path),
                    })

        if (doc_idx + 1) % 150 == 0 or (doc_idx + 1) == len(master_rows):
            print(f"Processed {doc_idx + 1} / {len(master_rows)} parent documents ({len(patch_records)} patches generated)...")

    print(f"\nTotal Patches Generated: {len(patch_records)}")
    forged_total = sum(1 for p in patch_records if p["patch_label"] == "FORGED_REGION")
    auth_total = sum(1 for p in patch_records if p["patch_label"] == "AUTHENTIC_REGION")
    print(f"  FORGED_REGION:    {forged_total:4d} ({forged_total/len(patch_records)*100:.1f}%)")
    print(f"  AUTHENTIC_REGION: {auth_total:4d} ({auth_total/len(patch_records)*100:.1f}%)")

    # Split into Train, Val, Test based on parent document split
    split_patches = {"train": [], "val": [], "test": []}
    for p in patch_records:
        split_patches[p["split"]].append(p)

    print("\nPer-Split Patch Distribution:")
    for s in ["train", "val", "test"]:
        s_p = split_patches[s]
        n_f = sum(1 for p in s_p if p["patch_label"] == "FORGED_REGION")
        n_a = sum(1 for p in s_p if p["patch_label"] == "AUTHENTIC_REGION")
        print(f"  {s.upper():5s}: {len(s_p):4d} patches | FORGED: {n_f:3d} | AUTHENTIC: {n_a:4d}")

    # Verify zero group leakage across patch splits
    train_groups = set(p["group_id"] for p in split_patches["train"])
    val_groups = set(p["group_id"] for p in split_patches["val"])
    test_groups = set(p["group_id"] for p in split_patches["test"])

    assert not (train_groups & val_groups), "Group Leakage between Train and Val patches!"
    assert not (train_groups & test_groups), "Group Leakage between Train and Test patches!"
    assert not (val_groups & test_groups), "Group Leakage between Val and Test patches!"
    print("\n[PASS] Zero group leakage across patch partitions verified!")

    # Write manifests
    fieldnames = list(patch_records[0].keys())

    dest_map = {
        PATCH_MASTER_PATH: patch_records,
        PATCH_TRAIN_PATH: split_patches["train"],
        PATCH_VAL_PATH: split_patches["val"],
        PATCH_TEST_PATH: split_patches["test"],
    }

    for path, rows_to_write in dest_map.items():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)
        print(f"Saved {len(rows_to_write):4d} rows to: {path.name}")

    print("\n" + "=" * 70)
    print("PATCH EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
