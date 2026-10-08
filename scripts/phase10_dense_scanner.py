#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 10 Dense Scanner (Stream A & Stream B).

Implements:
1. Stream A — OCR-Guided Dense Line Scanning:
   - Identifies text lines from OCR word boxes
   - Expands line strips with context margin
   - Multi-scale sliding windows (1.0x, 1.5x, 2.0x line height) at native resolution
   - High-density sampling across word boundaries
2. Stream B — OCR-Independent Dense Pixel Scanning:
   - Operates directly on native-resolution image canvas
   - Computes Laplacian second-order gradients and morphological contrast (TopHat + BlackHat)
   - Extracts multi-scale visual candidate windows (48px, 72px, 96px) on high-energy anomaly areas
   - Retains provenance metadata: source = OCR_DENSE or PIXEL_DENSE
"""

import sys
import os
import json
import math
from pathlib import Path
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


def extract_ocr_dense_candidates(
    image_shape: Tuple[int, int],
    ocr_data: Dict[str, Any],
    scales: List[float] = [1.0, 1.5, 2.0],
    stride_ratio: float = 0.50,
) -> List[Dict[str, Any]]:
    """Stream A: Extracts dense overlapping sliding windows along detected text lines."""
    img_h, img_w = image_shape[:2]
    candidates = []
    lines = ocr_data.get("lines", [])

    for line_idx, line in enumerate(lines):
        words = line.get("words", [])
        if not words:
            continue

        # Compute bounding union of words
        xs = [w["bbox"][0] for w in words]
        ys = [w["bbox"][1] for w in words]
        x2s = [w["bbox"][0] + w["bbox"][2] for w in words]
        y2s = [w["bbox"][1] + w["bbox"][3] for w in words]

        lx1, ly1 = max(0, min(xs)), max(0, min(ys))
        lx2, ly2 = min(img_w, max(x2s)), min(img_h, max(y2s))
        lw = lx2 - lx1
        lh = ly2 - ly1

        if lw < 10 or lh < 8:
            continue

        # Expand line bounding box slightly
        pad_y = int(lh * 0.20)
        pad_x = int(lh * 0.15)
        ex_y1 = max(0, ly1 - pad_y)
        ex_y2 = min(img_h, ly2 + pad_y)
        ex_x1 = max(0, lx1 - pad_x)
        ex_x2 = min(img_w, lx2 + pad_x)
        eff_h = ex_y2 - ex_y1

        for scale in scales:
            win_h = int(eff_h * scale)
            win_w = int(win_h * 1.6)  # aspect ratio ~1.6 for character groups
            stride_x = max(12, int(win_w * stride_ratio))

            if win_w > (ex_x2 - ex_x1):
                # Line is shorter than window, propose single centered box
                cx = (ex_x1 + ex_x2) // 2
                cy = (ex_y1 + ex_y2) // 2
                x1 = max(0, cx - win_w // 2)
                y1 = max(0, cy - win_h // 2)
                x2 = min(img_w, x1 + win_w)
                y2 = min(img_h, y1 + win_h)
                candidates.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "w": x2 - x1, "h": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1),
                    "scale": scale,
                    "source": "OCR_DENSE",
                    "ocr_line_id": line_idx,
                })
            else:
                curr_x = ex_x1
                while curr_x + win_w <= ex_x2 + stride_x // 2:
                    x1 = curr_x
                    y1 = max(0, (ex_y1 + ex_y2) // 2 - win_h // 2)
                    x2 = min(img_w, x1 + win_w)
                    y2 = min(img_h, y1 + win_h)

                    candidates.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "w": x2 - x1, "h": y2 - y1,
                        "area": (x2 - x1) * (y2 - y1),
                        "scale": scale,
                        "source": "OCR_DENSE",
                        "ocr_line_id": line_idx,
                    })
                    curr_x += stride_x

    return candidates


def extract_pixel_dense_candidates(
    image_bgr: np.ndarray,
    max_candidates: int = 80,
    window_sizes: List[int] = [48, 72, 96],
) -> List[Dict[str, Any]]:
    """Stream B: Extracts multi-scale visual proposals on high-frequency residual energy peaks."""
    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Laplacian high-frequency edge energy
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_energy = np.abs(laplacian)
    max_lap = np.max(lap_energy)
    norm_lap = (lap_energy / max_lap) if max_lap > 0 else lap_energy

    # 2. Morphological contrast peaks (TopHat + BlackHat)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    contrast_energy = (tophat.astype(float) + blackhat.astype(float)) / 255.0

    # Unified Anomaly Saliency Map
    saliency_map = 0.65 * norm_lap + 0.35 * contrast_energy

    # Find prominent local maxima
    blurred = cv2.GaussianBlur(saliency_map, (9, 9), 2.0)
    # Downscale for fast peak selection
    ds_factor = 4
    small_map = cv2.resize(blurred, (img_w // ds_factor, img_h // ds_factor))

    flat_indices = np.argsort(small_map.flatten())[::-1]
    candidates = []
    selected_centers = []

    for idx in flat_indices:
        if len(candidates) >= max_candidates:
            break

        sy = (idx // (img_w // ds_factor)) * ds_factor
        sx = (idx % (img_w // ds_factor)) * ds_factor
        val = small_map.flatten()[idx]

        if val < 0.12:  # Stop on low background energy
            break

        # Check suppression distance to avoid over-clustering
        if any(math.hypot(sx - ox, sy - oy) < 40 for ox, oy in selected_centers):
            continue

        selected_centers.append((sx, sy))

        # Propose windows at multiple scales
        for w_size in window_sizes:
            x1 = max(0, sx - w_size // 2)
            y1 = max(0, sy - w_size // 2)
            x2 = min(img_w, x1 + w_size)
            y2 = min(img_h, y1 + w_size)
            crop_gray = gray[y1:y2, x1:x2]

            edge_var = float(np.var(laplacian[y1:y2, x1:x2])) if crop_gray.size > 0 else 0.0
            contrast_val = float(np.std(crop_gray)) if crop_gray.size > 0 else 0.0

            candidates.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": x2 - x1, "h": y2 - y1,
                "area": (x2 - x1) * (y2 - y1),
                "scale": round(w_size / 48.0, 2),
                "source": "PIXEL_DENSE",
                "pixel_anomaly_score": round(float(val), 4),
                "edge_density": round(edge_var, 2),
                "texture_score": round(contrast_val, 2),
            })

    return candidates
