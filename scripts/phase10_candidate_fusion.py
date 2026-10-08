#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 10 Candidate Fusion & Deduplication Pipeline.

Integrates Stream A (OCR dense line scan), Stream B (Pixel dense scan), and Stream C (Typographic consistency):
1. Extracts proposals from all three streams on native-resolution image canvas
2. Applies deterministic deduplication:
   - Merges proposals with IoU >= 0.40 or center distance <= 20 pixels
   - Preserves multi-stream provenance (e.g., OCR_DENSE|TYPOGRAPHIC)
3. Outputs standardized manifests:
   - data/manifests/phase10_candidates_train.csv
   - data/manifests/phase10_candidates_val.csv
   - data/manifests/phase10_candidates_test.csv
   - data/manifests/phase10_candidates_master.csv
"""

import sys
import os
import csv
import json
import math
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from PIL import Image

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
OCR_DIR = DATA_DIR / "ocr"

DOC_TRAIN_MANIFEST = MANIFESTS_DIR / "trusttrace_phase4_train.csv"
DOC_VAL_MANIFEST = MANIFESTS_DIR / "trusttrace_phase4_val.csv"
DOC_TEST_MANIFEST = MANIFESTS_DIR / "trusttrace_phase4_test.csv"

OUT_MASTER = MANIFESTS_DIR / "phase10_candidates_master.csv"
OUT_TRAIN = MANIFESTS_DIR / "phase10_candidates_train.csv"
OUT_VAL = MANIFESTS_DIR / "phase10_candidates_val.csv"
OUT_TEST = MANIFESTS_DIR / "phase10_candidates_test.csv"

from scripts.phase10_dense_scanner import extract_ocr_dense_candidates, extract_pixel_dense_candidates
from scripts.phase10_typographic_features import compute_typographic_features
from scripts.build_phase9_candidate_graph import compute_box_iou


def deduplicate_proposals(
    raw_candidates: List[Dict[str, Any]],
    iou_thresh: float = 0.40,
    dist_thresh: float = 20.0,
) -> List[Dict[str, Any]]:
    """Deduplicates overlapping proposals while strictly preserving multi-stream provenance."""
    if not raw_candidates:
        return []

    # Sort proposals by area or anomaly score
    sorted_cands = sorted(
        raw_candidates,
        key=lambda c: (c.get("typographic_score", 0.0) + c.get("pixel_anomaly_score", 0.0)),
        reverse=True,
    )

    merged = []
    used = set()

    for i, c1 in enumerate(sorted_cands):
        if i in used:
            continue

        cluster = [c1]
        used.add(i)

        b1 = (float(c1["x1"]), float(c1["y1"]), float(c1["x2"]), float(c1["y2"]))
        cx1 = (b1[0] + b1[2]) / 2.0
        cy1 = (b1[1] + b1[3]) / 2.0

        for j in range(i + 1, len(sorted_cands)):
            if j in used:
                continue

            c2 = sorted_cands[j]
            b2 = (float(c2["x1"]), float(c2["y1"]), float(c2["x2"]), float(c2["y2"]))
            cx2 = (b2[0] + b2[2]) / 2.0
            cy2 = (b2[1] + b2[3]) / 2.0

            iou = compute_box_iou(b1, b2)
            dist = math.hypot(cx1 - cx2, cy1 - cy2)

            if iou >= iou_thresh or dist <= dist_thresh:
                cluster.append(c2)
                used.add(j)

        # Merge cluster bounding box union
        all_x1 = min(c["x1"] for c in cluster)
        all_y1 = min(c["y1"] for c in cluster)
        all_x2 = max(c["x2"] for c in cluster)
        all_y2 = max(c["y2"] for c in cluster)

        # Aggregate multi-source provenance
        sources = sorted(list(set(c["source"] for c in cluster)))
        provenance = "|".join(sources)

        max_typo = max((c.get("typographic_score", 0.0) for c in cluster), default=0.0)
        max_pixel = max((c.get("pixel_anomaly_score", 0.0) for c in cluster), default=0.0)
        max_edge = max((c.get("edge_density", 0.0) for c in cluster), default=0.0)
        max_text = next((c.get("text", "") for c in cluster if c.get("text", "")), "")

        merged.append({
            "x1": all_x1,
            "y1": all_y1,
            "x2": all_x2,
            "y2": all_y2,
            "w": all_x2 - all_x1,
            "h": all_y2 - all_y1,
            "area": (all_x2 - all_x1) * (all_y2 - all_y1),
            "source": provenance,
            "typographic_score": round(max_typo, 4),
            "pixel_anomaly_score": round(max_pixel, 4),
            "edge_density": round(max_edge, 2),
            "text": max_text,
        })

    return merged


def generate_phase10_candidates():
    print("=" * 70)
    print("TRUSTTRACE: Phase 10 Candidate Fusion Pipeline")
    print("=" * 70)

    splits = [
        ("test", DOC_TEST_MANIFEST, OUT_TEST, 148),
        ("val", DOC_VAL_MANIFEST, OUT_VAL, 148),
        ("train", DOC_TRAIN_MANIFEST, OUT_TRAIN, 300),  # representative sample for training pool
    ]

    all_master_rows = []
    global_cand_counter = 0

    for split_name, manifest_path, out_csv, max_docs in splits:
        print(f"\nProcessing {split_name} partition from {manifest_path.name}...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            docs = list(csv.DictReader(f))[:max_docs]

        print(f"Loaded {len(docs)} documents for {split_name}.")
        split_candidates = []
        t0 = time.time()

        for d_idx, d in enumerate(docs):
            sid = d["sample_id"]
            gid = d.get("group_id", "group_unknown")
            img_path = Path(d["image_path"])

            if not img_path.is_file():
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            # Load OCR json if available
            ocr_file = OCR_DIR / f"{sid}_ocr.json"
            ocr_data = {}
            if ocr_file.is_file():
                with open(ocr_file, "r", encoding="utf-8-sig") as f:
                    ocr_data = json.load(f)

            # Stream A: OCR Dense Line Scan
            cands_a = extract_ocr_dense_candidates(img_bgr.shape, ocr_data, scales=[1.0, 1.5, 2.0], stride_ratio=0.50)

            # Stream B: Pixel Dense Scan
            cands_b = extract_pixel_dense_candidates(img_bgr, max_candidates=60, window_sizes=[48, 72, 96])

            # Stream C: Typographic Consistency Analysis
            cands_c, _ = compute_typographic_features(img_bgr, ocr_data, anomaly_threshold=0.30)

            # Combine and Deduplicate
            combined_raw = cands_a + cands_b + cands_c
            deduped = deduplicate_proposals(combined_raw, iou_thresh=0.40, dist_thresh=20.0)

            for c in deduped:
                global_cand_counter += 1
                cid = f"P10_{global_cand_counter:07d}_{sid}"
                c_row = {
                    "candidate_id": cid,
                    "sample_id": sid,
                    "group_id": gid,
                    "split": split_name,
                    "source": c["source"],
                    "x1": c["x1"],
                    "y1": c["y1"],
                    "x2": c["x2"],
                    "y2": c["y2"],
                    "w": c["w"],
                    "h": c["h"],
                    "area": c["area"],
                    "typographic_score": c["typographic_score"],
                    "pixel_anomaly_score": c["pixel_anomaly_score"],
                    "edge_density": c["edge_density"],
                    "text": c["text"],
                }
                split_candidates.append(c_row)
                all_master_rows.append(c_row)

        el_time = time.time() - t0
        print(f"Generated {len(split_candidates)} candidates across {len(docs)} documents in {el_time:.1f}s ({len(split_candidates)/float(max(len(docs),1)):.1f} cands/doc).")

        # Save split CSV
        fieldnames = list(split_candidates[0].keys()) if split_candidates else []
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_candidates)
        print(f"Saved {len(split_candidates)} rows to: {out_csv.name}")

    # Save Master CSV
    fieldnames = list(all_master_rows[0].keys()) if all_master_rows else []
    with open(OUT_MASTER, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_master_rows)
    print(f"\nSaved master candidate manifest ({len(all_master_rows)} total rows) to: {OUT_MASTER.name}")


if __name__ == "__main__":
    generate_phase10_candidates()
