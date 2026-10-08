#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 7 Morphological & Visual Saliency Candidate Generator.

Generates candidate regions directly from native-resolution receipt scans without OCR:
- B1: Multi-scale morphological gradient (dilation - erosion) across kernel scales
- B2: Deterministic edge density and high-gradient contour grouping
- B3: Local high-frequency texture anomaly response (|I - G_sigma(I)|)
- B4: Connected component extraction with geometric plausibility filtering
- Computes native-resolution candidate bounding boxes with visual saliency scores
- Strictly deterministic, zero ground-truth utilization at inference time.
"""

import sys
import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"


def extract_saliency_candidates_for_image(
    image_path: Path,
    scales: List[int] = [3, 7, 13],
    max_candidates_per_source: int = 40,
) -> List[Dict[str, Any]]:
    """
    Extracts multi-scale visual saliency candidates directly from uncompressed receipt pixels.
    Returns list of candidate dicts with native-resolution bounding box coordinates and saliency scores.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    img_h, img_w = img.shape
    candidates: List[Dict[str, Any]] = []

    # Contrast enhancement for receipts
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_img = clahe.apply(img)

    # -------------------------------------------------------------
    # B1: Multi-Scale Morphological Gradient
    # -------------------------------------------------------------
    for k_size in scales:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        grad = cv2.morphologyEx(norm_img, cv2.MORPH_GRADIENT, kernel)

        # Threshold top gradient responses
        th_val = np.percentile(grad, 90.0)
        _, binary = cv2.threshold(grad, max(int(th_val), 30), 255, cv2.THRESH_BINARY)

        # Horizontal text line dilation to form word-level clusters
        cluster_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, k_size * 2), max(2, k_size // 2)))
        clustered = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cluster_k)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clustered, connectivity=8)

        scale_cands = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            # Geometric plausibility filter
            if w < 10 or h < 8 or w > img_w * 0.85 or h > img_h * 0.5:
                continue
            if area < 64 or area > (img_w * img_h * 0.25):
                continue
            ar = w / float(h)
            if ar < 0.15 or ar > 18.0:
                continue

            # Saliency score: mean gradient intensity within candidate box
            box_patch = grad[y : y + h, x : x + w]
            score = float(np.mean(box_patch)) if box_patch.size > 0 else 0.0

            scale_cands.append({
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "w": w,
                "h": h,
                "source": "morph_gradient",
                "scale": f"k{k_size}",
                "score": round(score, 4),
            })

        scale_cands.sort(key=lambda c: c["score"], reverse=True)
        candidates.extend(scale_cands[:max_candidates_per_source])

    # -------------------------------------------------------------
    # B2 & B3: Edge Density and High-Frequency Texture Anomalies
    # -------------------------------------------------------------
    # High-frequency residual: |I - Gaussian(I)|
    blurred = cv2.GaussianBlur(norm_img, (9, 9), 1.5)
    hf_residual = cv2.absdiff(norm_img, blurred)

    th_hf = np.percentile(hf_residual, 92.0)
    _, hf_bin = cv2.threshold(hf_residual, max(int(th_hf), 25), 255, cv2.THRESH_BINARY)
    hf_clustered = cv2.morphologyEx(hf_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hf_clustered, connectivity=8)
    hf_cands = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if w < 10 or h < 8 or w > img_w * 0.85 or h > img_h * 0.5:
            continue
        if area < 64 or area > (img_w * img_h * 0.25):
            continue
        ar = w / float(h)
        if ar < 0.15 or ar > 18.0:
            continue

        box_patch = hf_residual[y : y + h, x : x + w]
        score = float(np.mean(box_patch)) if box_patch.size > 0 else 0.0

        hf_cands.append({
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
            "w": w,
            "h": h,
            "source": "hf_texture",
            "scale": "g9",
            "score": round(score, 4),
        })

    hf_cands.sort(key=lambda c: c["score"], reverse=True)
    candidates.extend(hf_cands[:max_candidates_per_source])

    return candidates


def nms_deduplicate(candidates: List[Dict[str, Any]], iou_threshold: float = 0.40) -> List[Dict[str, Any]]:
    """Performs Non-Maximum Suppression (NMS) on candidates based on saliency score."""
    if not candidates:
        return []

    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
    kept: List[Dict[str, Any]] = []

    for cand in sorted_cands:
        b1 = (cand["x1"], cand["y1"], cand["w"], cand["h"])
        suppressed = False
        for k in kept:
            b2 = (k["x1"], k["y1"], k["w"], k["h"])
            # IoU
            xi1 = max(b1[0], b2[0])
            yi1 = max(b1[1], b2[1])
            xi2 = min(b1[0] + b1[2], b2[0] + b2[2])
            yi2 = min(b1[1] + b1[3], b2[1] + b2[3])
            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_a = inter_w * inter_h
            if inter_a > 0:
                union_a = (b1[2] * b1[3]) + (b2[2] * b2[3]) - inter_a
                iou = inter_a / float(union_a) if union_a > 0 else 0.0
                if iou >= iou_threshold:
                    suppressed = True
                    break
        if not suppressed:
            kept.append(cand)

    return kept


if __name__ == "__main__":
    # Test on single receipt
    test_img = Path(r"C:\Users\jay\Downloads\finditagain\findit2\train\X51005568881.png")
    if test_img.is_file():
        t0 = time.time()
        cands = extract_saliency_candidates_for_image(test_img)
        dt = (time.time() - t0) * 1000.0
        deduped = nms_deduplicate(cands, iou_threshold=0.40)
        print(f"Extracted {len(cands)} raw candidates, {len(deduped)} after NMS in {dt:.1f}ms")
        for i, c in enumerate(deduped[:5]):
            print(f"  Top {i+1}: bbox=[{c['x1']},{c['y1']},{c['w']},{c['h']}], source={c['source']}, score={c['score']}")
