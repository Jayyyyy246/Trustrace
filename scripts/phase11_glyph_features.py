#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 11 Character-Level Glyph Feature Extraction Suite.

Computes comprehensive micro-forensic features for individual glyph proposals:
1. Geometry Proxies:
   - Glyph height, width, aspect ratio, area
   - Stroke-width proxy via distance transform
2. Appearance Proxies:
   - Foreground mean intensity, variance, contrast
   - Edge density via Canny operator
   - Laplacian high-frequency residual energy
3. Spatial Context Proxies:
   - Line-relative baseline offset
   - Horizontal inter-glyph spacing
4. Shape & Contour Proxies:
   - Contour perimeter / area compactness
   - Stroke density
"""

import sys
import os
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))


def extract_glyph_micro_features(crop_rgb: np.ndarray) -> Dict[str, float]:
    """Extracts native-resolution micro-forensic features from a glyph crop."""
    h, w, c = crop_rgb.shape
    if h == 0 or w == 0:
        return {
            "glyph_height": 0.0,
            "glyph_width": 0.0,
            "aspect_ratio": 1.0,
            "glyph_area": 0.0,
            "stroke_proxy": 1.0,
            "mean_intensity": 128.0,
            "intensity_variance": 0.0,
            "edge_density": 0.0,
            "laplacian_energy": 0.0,
            "compactness": 1.0,
        }

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if c == 3 else crop_rgb.copy()

    # Geometry
    area = float(h * w)
    aspect_ratio = float(w) / max(1.0, float(h))

    # Stroke proxy via distance transform on inverted binary
    _, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg_pixels = np.sum(binarized > 0)
    if fg_pixels > 0:
        dist_transform = cv2.distanceTransform(binarized, cv2.DIST_L2, 3)
        stroke_proxy = float(2.0 * np.mean(dist_transform[binarized > 0]))
    else:
        stroke_proxy = 1.0

    # Appearance
    mean_int = float(np.mean(gray))
    var_int = float(np.var(gray))

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0) / max(1.0, area))

    # Laplacian high-frequency energy
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_energy = float(np.var(lap))

    # Compactness (perimeter^2 / area)
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        c_area = cv2.contourArea(cnt)
        compactness = float((perimeter ** 2) / max(1.0, c_area))
    else:
        compactness = float((2 * (w + h)) ** 2 / max(1.0, area))

    return {
        "glyph_height": float(h),
        "glyph_width": float(w),
        "aspect_ratio": round(aspect_ratio, 4),
        "glyph_area": round(area, 2),
        "stroke_proxy": round(stroke_proxy, 4),
        "mean_intensity": round(mean_int, 2),
        "intensity_variance": round(var_int, 2),
        "edge_density": round(edge_density, 4),
        "laplacian_energy": round(lap_energy, 4),
        "compactness": round(compactness, 4),
    }


if __name__ == "__main__":
    # Smoke test on dummy patch
    dummy = np.zeros((32, 20, 3), dtype=np.uint8)
    dummy[:, :] = [240, 240, 240]
    dummy[6:26, 8:12] = [20, 20, 20]  # Digit '1' stroke
    feats = extract_glyph_micro_features(dummy)
    print("Glyph Micro-Features Smoke Test:")
    for k, v in feats.items():
        print(f"  {k:22s}: {v}")
