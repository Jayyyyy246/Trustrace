#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 9 Candidate Feature Extraction Pipeline.

Extracts comprehensive candidate-level features across Train, Val, and Test splits:
- Geometric features: w, h, area, aspect ratio
- Visual quality: brightness, contrast, edge density (Laplacian var), texture score (Sobel)
- Spatial context: local neighbor density (spatial_neighbors_150px)
- Candidate quality score: deterministic Q in [0, 1]
- Forensic model inference: frozen Phase 8 Compact CNN outputs (probability, margin uncertainty, entropy)
- Outputs:
  * data/manifests/phase9_candidate_features_master.csv
  * data/manifests/phase9_candidate_features_train.csv
  * data/manifests/phase9_candidate_features_val.csv
  * data/manifests/phase9_candidate_features_test.csv
"""

import sys
import os
import csv
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"

PHASE8_TRAIN_MANIFEST = MANIFESTS_DIR / "phase8_patch_train.csv"
PHASE8_VAL_MANIFEST = MANIFESTS_DIR / "phase8_patch_val.csv"
PHASE8_TEST_MANIFEST = MANIFESTS_DIR / "phase8_patch_test.csv"
PHASE8_CNN_PATH = MODELS_DIR / "phase8_patch_cnn_best.pt"

OUT_MASTER = MANIFESTS_DIR / "phase9_candidate_features_master.csv"
OUT_TRAIN = MANIFESTS_DIR / "phase9_candidate_features_train.csv"
OUT_VAL = MANIFESTS_DIR / "phase9_candidate_features_val.csv"
OUT_TEST = MANIFESTS_DIR / "phase9_candidate_features_test.csv"

from scripts.train_phase8_cnn import Phase8CompactCNN
from scripts.audit_phase9_candidate_quality import (
    compute_visual_features,
    compute_uncertainties,
    compute_candidate_quality,
)


def extract_candidate_features():
    print("=" * 70)
    print("TRUSTTRACE: Phase 9 Candidate Feature Extraction Pipeline")
    print("=" * 70)

    device = torch.device("cpu")
    print("Loading frozen Phase 8 Compact CNN...")
    cnn_model = Phase8CompactCNN(in_channels=3, num_classes=2)
    ckpt = torch.load(PHASE8_CNN_PATH, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cnn_model.load_state_dict(state)
    cnn_model.to(device)
    cnn_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    splits_data = [
        ("train", PHASE8_TRAIN_MANIFEST, OUT_TRAIN),
        ("val", PHASE8_VAL_MANIFEST, OUT_VAL),
        ("test", PHASE8_TEST_MANIFEST, OUT_TEST),
    ]

    all_extracted_rows = []

    for split_name, in_path, out_path in splits_data:
        print(f"\nProcessing {split_name} split from {in_path.name}...")
        with open(in_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"Loaded {len(rows)} patch crops for {split_name}.")

        # Group by document for spatial neighbor density
        doc_cands = defaultdict(list)
        for r in rows:
            doc_cands[r["sample_id"]].append(r)

        spatial_density_map = {}
        for sid, c_list in doc_cands.items():
            centers = []
            for c in c_list:
                cx = (float(c["x1"]) + float(c["x2"])) / 2.0
                cy = (float(c["y1"]) + float(c["y2"])) / 2.0
                centers.append((cx, cy))
            for idx, c in enumerate(c_list):
                pid = c["patch_id"]
                cx, cy = centers[idx]
                neighbors = sum(1 for j, (ox, oy) in enumerate(centers) if j != idx and math.hypot(cx - ox, cy - oy) <= 150.0)
                spatial_density_map[pid] = neighbors

        extracted_split_rows = []
        batch_tensors = []
        batch_meta = []

        for idx, r in enumerate(rows):
            pid = r["patch_id"]
            img_p = r["crop_path"]
            w = float(r["w"])
            h = float(r["h"])
            src = r["source"]
            vis = compute_visual_features(img_p)
            quality = compute_candidate_quality(w, h, vis["edge_density"], vis["contrast"], vis["texture_score"], src)

            with Image.open(img_p) as img:
                t = transform(img.convert("RGB"))

            batch_tensors.append(t)
            meta = {
                "patch_id": pid,
                "candidate_id": r["candidate_id"],
                "sample_id": r["sample_id"],
                "group_id": r["group_id"],
                "split": split_name,
                "source": src,
                "x1": r["x1"],
                "y1": r["y1"],
                "x2": r["x2"],
                "y2": r["y2"],
                "w": w,
                "h": h,
                "area": w * h,
                "aspect_ratio": round(max(w / max(h, 1.0), h / max(w, 1.0)), 2),
                "brightness": vis["brightness"],
                "contrast": vis["contrast"],
                "edge_density": vis["edge_density"],
                "texture_score": vis["texture_score"],
                "spatial_neighbors_150px": spatial_density_map.get(pid, 0),
                "candidate_quality_score": quality,
                "max_gt_iou": r["max_gt_iou"],
                "binary_label": r["binary_label"],
                "is_hard_negative": r["is_hard_negative"],
                "parent_document_label": r["parent_document_label"],
                "crop_path": img_p,
            }
            batch_meta.append(meta)

            if len(batch_tensors) >= 64 or idx == len(rows) - 1:
                batch_stacked = torch.stack(batch_tensors).to(device)
                with torch.no_grad():
                    logits = cnn_model(batch_stacked)
                    raw_probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                    logit_diffs = (logits[:, 1] - logits[:, 0]).cpu().tolist()

                for b_idx, m in enumerate(batch_meta):
                    p = raw_probs[b_idx]
                    m_unc, ent = compute_uncertainties(p)
                    m["raw_logit"] = round(logit_diffs[b_idx], 4)
                    m["forged_probability"] = round(p, 4)
                    m["margin_uncertainty"] = m_unc
                    m["entropy"] = ent
                    extracted_split_rows.append(m)

                batch_tensors = []
                batch_meta = []

        # Save split CSV
        fieldnames = list(extracted_split_rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(extracted_split_rows)
        print(f"Saved {len(extracted_split_rows)} rows to {out_path.name}.")

        all_extracted_rows.extend(extracted_split_rows)

    # Save Master CSV
    fieldnames = list(all_extracted_rows[0].keys())
    with open(OUT_MASTER, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_extracted_rows)
    print(f"\nSaved master candidate features ({len(all_extracted_rows)} total rows) to: {OUT_MASTER.name}")


if __name__ == "__main__":
    extract_candidate_features()
