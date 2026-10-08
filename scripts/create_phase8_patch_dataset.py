#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 8 Candidate-Aware Patch Dataset Constructor.

Constructs training, validation, and test patch datasets incorporating:
1. All valid positive candidate patches (IoU >= 0.25 with official forgery annotations)
2. Phase 6 official forgery region patches (anchor ground truth)
3. Hard authentic visual negatives:
   - High-saliency morphology candidates from authentic receipts (folds, creases, borders, logos, thermal roll fading)
   - OCR candidate words from authentic receipts
   - Non-overlapping candidate regions from edited receipts
4. Strictly excludes ambiguous candidates (0.05 < IoU < 0.25)
5. Preserves exact parent document group isolation:
   group_id -> parent receipt -> patches (Zero cross-split leakage)
6. Outputs:
   * data/manifests/phase8_patch_master.csv
   * data/manifests/phase8_patch_train.csv
   * data/manifests/phase8_patch_val.csv
   * data/manifests/phase8_patch_test.csv
   * Caches standardized 128x128 patches in data/document_patches/
"""

import sys
import os
import csv
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
MANIFESTS_DIR = DATA_DIR / "manifests"
PATCHES_DIR = DATA_DIR / "document_patches"
REPORTS_DIR = WORKSPACE_DIR / "reports"

CANDIDATES_MASTER = MANIFESTS_DIR / "phase7_candidates_master.csv"
PHASE6_PATCH_MASTER = MANIFESTS_DIR / "phase6_patch_master.csv"
ANNOTATION_COVERAGE_JSON = REPORTS_DIR / "phase7_annotation_coverage.json"
RECEIPT_MASTER = MANIFESTS_DIR / "phase4_dataset_master.csv"

PHASE8_MASTER_CSV = MANIFESTS_DIR / "phase8_patch_master.csv"
PHASE8_TRAIN_CSV = MANIFESTS_DIR / "phase8_patch_train.csv"
PHASE8_VAL_CSV = MANIFESTS_DIR / "phase8_patch_val.csv"
PHASE8_TEST_CSV = MANIFESTS_DIR / "phase8_patch_test.csv"

RANDOM_SEED = 42
TARGET_PATCH_SIZE = (128, 128)
CONTEXT_MARGIN = 1.35


def compute_box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    iw = max(0, xi2 - xi1)
    ih = max(0, yi2 - yi1)
    ia = iw * ih
    if ia <= 0:
        return 0.0
    ua = (w1 * h1) + (w2 * h2) - ia
    return float(ia / ua) if ua > 0 else 0.0


def crop_and_standardize_patch(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = TARGET_PATCH_SIZE,
    context_margin: float = CONTEXT_MARGIN,
) -> Image.Image:
    img_w, img_h = img.size
    bx, by, bw, bh = bbox
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
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
    max_side = max(cw, ch)
    square_canvas = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    paste_x = (max_side - cw) // 2
    paste_y = (max_side - ch) // 2
    square_canvas.paste(crop, (paste_x, paste_y))
    return square_canvas.resize(target_size, Image.Resampling.BILINEAR)


def create_phase8_dataset():
    print("=" * 70)
    print("TRUSTTRACE: Phase 8 Candidate-Aware Patch Dataset Construction")
    print("=" * 70)

    random.seed(RANDOM_SEED)
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Receipt Master for Image Paths
    with open(RECEIPT_MASTER, "r", encoding="utf-8") as f:
        receipt_rows = list(csv.DictReader(f))
    receipt_img_map = {r["sample_id"]: Path(r["image_path"]) for r in receipt_rows}
    receipt_label_map = {r["sample_id"]: r["label"] for r in receipt_rows}

    # 2. Load GT Forgery Boxes
    with open(ANNOTATION_COVERAGE_JSON, "r", encoding="utf-8") as f:
        annot_data = json.load(f)
    doc_gt_map: Dict[str, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for b in annot_data["gt_box_records"]:
        doc_gt_map[b["sample_id"]].append((b["x"], b["y"], b["w"], b["h"]))

    # 3. Load Phase 7 Candidates Master
    with open(CANDIDATES_MASTER, "r", encoding="utf-8") as f:
        candidate_rows = list(csv.DictReader(f))
    print(f"Loaded {len(candidate_rows):,} Phase 7 candidates.")

    # Group candidates by split and sample_id
    split_cands: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in candidate_rows:
        split_cands[c["split"]].append(c)

    # 4. Partition Sampling Strategy:
    # Positives: all candidates with IoU >= 0.25 with GT
    # Negatives:
    #   - Morphology Negatives (Hard Negatives): ranked by saliency score on authentic receipts
    #   - OCR Negatives (Authentic text baseline): sampled on authentic receipts
    #   - On edited receipts: candidates with IoU <= 0.05
    # Strict exclusion: 0.05 < IoU < 0.25 (ambiguous)

    selected_patches: List[Dict[str, Any]] = []
    patch_counter = 0

    # Image cache for fast cropping
    open_images: Dict[str, Image.Image] = {}

    def get_image(sid: str) -> Optional[Image.Image]:
        if sid in open_images:
            return open_images[sid]
        p = receipt_img_map.get(sid)
        if p and p.is_file():
            img = Image.open(p).convert("RGB")
            # Limit cache size to 100 images
            if len(open_images) > 100:
                open_images.pop(next(iter(open_images)))
            open_images[sid] = img
            return img
        return None

    # Process each partition
    for s_name in ["train", "val", "test"]:
        s_cands = split_cands[s_name]
        print(f"\nConstructing {s_name.upper()} patch dataset from {len(s_cands):,} candidates...")

        pos_cands = []
        neg_ocr = []
        neg_morph_hard = []
        neg_edited_clean = []

        for c in s_cands:
            sid = c["sample_id"]
            cand_bbox = (int(c["x1"]), int(c["y1"]), int(c["w"]), int(c["h"]))
            parent_gts = doc_gt_map.get(sid, [])
            max_iou = max([compute_box_iou(cand_bbox, gb) for gb in parent_gts], default=0.0)
            doc_label = receipt_label_map.get(sid, "REAL")

            c_info = dict(c)
            c_info["max_gt_iou"] = round(max_iou, 4)
            c_info["doc_label"] = doc_label
            c_info["cand_bbox"] = cand_bbox

            if max_iou >= 0.25:
                pos_cands.append(c_info)
            elif max_iou <= 0.05:
                if doc_label == "REAL":
                    if c["source"] == "MORPHOLOGY":
                        neg_morph_hard.append(c_info)
                    else:
                        neg_ocr.append(c_info)
                else:  # EDITED receipt, clean non-forged region
                    neg_edited_clean.append(c_info)
            else:
                # Ambiguous: strictly excluded
                pass

        print(f"  Available Positives: {len(pos_cands)}")
        print(f"  Available Negatives: {len(neg_ocr)} OCR-Real, {len(neg_morph_hard)} Morph-Real, {len(neg_edited_clean)} Edited-Clean")

        # Sampling ratios per split
        if s_name == "train":
            # Select all positives
            sampled_pos = pos_cands

            # Select hard morphology negatives (top by score)
            neg_morph_hard.sort(key=lambda x: float(x["score"]), reverse=True)
            sampled_morph_neg = neg_morph_hard[:1200]  # Hardest visual artifacts

            # Select representative OCR negatives
            rng = random.Random(RANDOM_SEED)
            sampled_ocr_neg = rng.sample(neg_ocr, min(800, len(neg_ocr)))
            sampled_edited_clean = rng.sample(neg_edited_clean, min(400, len(neg_edited_clean)))

            split_selected = sampled_pos + sampled_morph_neg + sampled_ocr_neg + sampled_edited_clean
        elif s_name == "val":
            sampled_pos = pos_cands
            neg_morph_hard.sort(key=lambda x: float(x["score"]), reverse=True)
            sampled_morph_neg = neg_morph_hard[:300]
            rng = random.Random(RANDOM_SEED + 1)
            sampled_ocr_neg = rng.sample(neg_ocr, min(200, len(neg_ocr)))
            sampled_edited_clean = rng.sample(neg_edited_clean, min(100, len(neg_edited_clean)))
            split_selected = sampled_pos + sampled_morph_neg + sampled_ocr_neg + sampled_edited_clean
        else:  # test
            sampled_pos = pos_cands
            neg_morph_hard.sort(key=lambda x: float(x["score"]), reverse=True)
            sampled_morph_neg = neg_morph_hard[:300]
            rng = random.Random(RANDOM_SEED + 2)
            sampled_ocr_neg = rng.sample(neg_ocr, min(200, len(neg_ocr)))
            sampled_edited_clean = rng.sample(neg_edited_clean, min(100, len(neg_edited_clean)))
            split_selected = sampled_pos + sampled_morph_neg + sampled_ocr_neg + sampled_edited_clean

        print(f"  Selected {len(split_selected)} patches ({sum(1 for x in split_selected if x['max_gt_iou']>=0.25)} Positive, {sum(1 for x in split_selected if x['max_gt_iou']<=0.05)} Negative)")

        # Crop and register records
        for item in split_selected:
            sid = item["sample_id"]
            img = get_image(sid)
            if img is None:
                continue

            patch_counter += 1
            is_pos = (item["max_gt_iou"] >= 0.25)
            binary_label = 1 if is_pos else 0
            patch_label = "FORGED_REGION" if is_pos else "AUTHENTIC_REGION"
            is_hard_neg = (not is_pos and item["source"] == "MORPHOLOGY" and item["doc_label"] == "REAL")

            patch_id = f"P8_{patch_counter:07d}_{sid}_{item['source']}_{'POS' if is_pos else 'NEG'}"
            crop_filename = f"{patch_id}.png"
            crop_path = PATCHES_DIR / crop_filename

            if not crop_path.is_file():
                crop_img = crop_and_standardize_patch(img, item["cand_bbox"])
                crop_img.save(crop_path, format="PNG")

            rec = {
                "patch_id": patch_id,
                "candidate_id": item["candidate_id"],
                "sample_id": sid,
                "group_id": item["group_id"],
                "split": s_name,
                "source": item["source"],
                "x1": item["x1"],
                "y1": item["y1"],
                "x2": item["x2"],
                "y2": item["y2"],
                "w": item["w"],
                "h": item["h"],
                "score": item["score"],
                "max_gt_iou": item["max_gt_iou"],
                "patch_label": patch_label,
                "binary_label": binary_label,
                "is_hard_negative": 1 if is_hard_neg else 0,
                "parent_document_label": item["doc_label"],
                "crop_path": str(crop_path),
            }
            selected_patches.append(rec)

    # 5. Also incorporate Phase 6 Anchor Patches to ensure Phase 6 ground-truth representation is retained
    with open(PHASE6_PATCH_MASTER, "r", encoding="utf-8") as f:
        p6_rows = list(csv.DictReader(f))
    print(f"\nMerging Phase 6 anchor ground-truth patches ({len(p6_rows)} patches)...")
    for p6 in p6_rows:
        if p6["patch_label"] == "FORGED_REGION":
            patch_counter += 1
            p6_id = f"P8_{patch_counter:07d}_{p6['parent_sample_id']}_P6ANCHOR"
            selected_patches.append({
                "patch_id": p6_id,
                "candidate_id": p6["patch_id"],
                "sample_id": p6["parent_sample_id"],
                "group_id": p6["group_id"],
                "split": p6["split"],
                "source": "P6_ANCHOR",
                "x1": p6["bbox_x"],
                "y1": p6["bbox_y"],
                "x2": int(p6["bbox_x"]) + int(p6["bbox_w"]),
                "y2": int(p6["bbox_y"]) + int(p6["bbox_h"]),
                "w": p6["bbox_w"],
                "h": p6["bbox_h"],
                "score": 200.0,
                "max_gt_iou": 1.0,
                "patch_label": "FORGED_REGION",
                "binary_label": 1,
                "is_hard_negative": 0,
                "parent_document_label": p6["parent_document_label"],
                "crop_path": p6["patch_path"],
            })

    print(f"\nTotal Phase 8 Patches Assembled: {len(selected_patches):,}")

    # Split into Train, Val, Test
    split_final: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in selected_patches:
        split_final[p["split"]].append(p)

    for s in ["train", "val", "test"]:
        sp = split_final[s]
        n_pos = sum(1 for x in sp if x["binary_label"] == 1)
        n_neg = sum(1 for x in sp if x["binary_label"] == 0)
        n_hard = sum(1 for x in sp if x["is_hard_negative"] == 1)
        print(f"  {s.upper():5s}: {len(sp):4d} patches | FORGED: {n_pos:4d} | AUTHENTIC: {n_neg:4d} (Hard Negatives: {n_hard:4d})")

    # Export Manifests
    fieldnames = list(selected_patches[0].keys())
    for dest_p, rows in [
        (PHASE8_MASTER_CSV, selected_patches),
        (PHASE8_TRAIN_CSV, split_final["train"]),
        (PHASE8_VAL_CSV, split_final["val"]),
        (PHASE8_TEST_CSV, split_final["test"]),
    ]:
        with open(dest_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows):4d} rows to: {dest_p.name}")

    print("\n" + "=" * 70)
    print("PHASE 8 PATCH DATASET CREATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    create_phase8_dataset()
