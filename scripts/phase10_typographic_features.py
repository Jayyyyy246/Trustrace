#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 10 Typographic Consistency Analysis (Stream C).

Implements:
1. Intra-document character and word geometry measurement:
   - Height variance, width-to-character ratio, aspect ratio
   - Baseline vertical offset and alignment deviation
   - Visual stroke proxy and local foreground contrast
2. Line-relative and document-relative typographic comparisons
3. Deterministic Typographic Anomaly Score: T in [0, 1]
4. Generates candidate proposals on anomalous typographic regions:
   - source = TYPOGRAPHIC
5. Supports feature-family ablation:
   - Geometry, Baseline, Stroke/Contrast, Spacing, Combined
"""

import sys
import os
import json
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


def compute_typographic_features(
    image_bgr: np.ndarray,
    ocr_data: Dict[str, Any],
    anomaly_threshold: float = 0.30,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Analyzes intra-document typographic consistency and surfaces anomalous word regions."""
    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lines = ocr_data.get("lines", [])

    candidates = []
    all_word_records = []

    # 1. Document-level typographic reference
    all_word_heights = []
    for line in lines:
        for w in line.get("words", []):
            all_word_heights.append(float(w["bbox"][3]))
    doc_mean_h = float(np.mean(all_word_heights)) if all_word_heights else 30.0

    # 2. Intra-line comparative analysis
    for line_idx, line in enumerate(lines):
        words = line.get("words", [])
        if len(words) < 1:
            continue

        line_heights = [float(w["bbox"][3]) for w in words]
        line_baselines = [float(w["bbox"][1] + w["bbox"][3]) for w in words]
        line_char_widths = [float(w["bbox"][2]) / max(len(w["text"]), 1) for w in words]

        mean_lh = float(np.mean(line_heights))
        mean_base = float(np.mean(line_baselines))
        mean_cw = float(np.mean(line_char_widths))

        for w_idx, w_obj in enumerate(words):
            text = w_obj["text"]
            bx, by, bw, bh = w_obj["bbox"]
            n_chars = max(len(text), 1)

            # Feature A: Height deviation from line & document
            dev_h_line = abs(bh - mean_lh) / max(mean_lh, 1e-4)
            dev_h_doc = abs(bh - doc_mean_h) / max(doc_mean_h, 1e-4)
            feat_geometry = min(1.0, 0.7 * dev_h_line + 0.3 * dev_h_doc)

            # Feature B: Baseline offset
            curr_base = by + bh
            dev_base = abs(curr_base - mean_base) / max(mean_lh, 8.0)
            feat_baseline = min(1.0, dev_base)

            # Feature C: Character width / spacing deviation
            char_w = bw / n_chars
            dev_cw = abs(char_w - mean_cw) / max(mean_cw, 1e-4)
            feat_spacing = min(1.0, dev_cw)

            # Feature D: Stroke thickness & local contrast proxy
            cx1 = max(0, bx)
            cy1 = max(0, by)
            cx2 = min(img_w, bx + bw)
            cy2 = min(img_h, by + bh)
            crop_g = gray[cy1:cy2, cx1:cx2]

            if crop_g.size > 20:
                _, thresh = cv2.threshold(crop_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                fg_ratio = float(np.sum(thresh == 255)) / float(crop_g.size)
                local_std = float(np.std(crop_g))
                feat_stroke_contrast = min(1.0, abs(fg_ratio - 0.30) * 2.0 + (local_std / 80.0) * 0.3)
            else:
                feat_stroke_contrast = 0.20

            # Unified Typographic Anomaly Score T in [0, 1]
            T_score = (
                0.35 * feat_geometry
                + 0.25 * feat_baseline
                + 0.20 * feat_spacing
                + 0.20 * feat_stroke_contrast
            )

            word_meta = {
                "text": text,
                "bbox": [bx, by, bw, bh],
                "feat_geometry": round(feat_geometry, 4),
                "feat_baseline": round(feat_baseline, 4),
                "feat_spacing": round(feat_spacing, 4),
                "feat_stroke_contrast": round(feat_stroke_contrast, 4),
                "T_score": round(float(T_score), 4),
            }
            all_word_records.append(word_meta)

            # If T exceeds threshold, generate candidate proposal with margin
            if T_score >= anomaly_threshold:
                pad_x = max(8, int(bw * 0.25))
                pad_y = max(6, int(bh * 0.20))
                px1 = max(0, bx - pad_x)
                py1 = max(0, by - pad_y)
                px2 = min(img_w, bx + bw + pad_x)
                py2 = min(img_h, by + bh + pad_y)

                candidates.append({
                    "x1": px1, "y1": py1, "x2": px2, "y2": py2,
                    "w": px2 - px1, "h": py2 - py1,
                    "area": (px2 - px1) * (py2 - py1),
                    "scale": 1.0,
                    "source": "TYPOGRAPHIC",
                    "typographic_score": round(float(T_score), 4),
                    "text": text,
                    "feat_geometry": round(feat_geometry, 4),
                    "feat_baseline": round(feat_baseline, 4),
                    "feat_spacing": round(feat_spacing, 4),
                    "feat_stroke_contrast": round(feat_stroke_contrast, 4),
                })

    doc_stats = {
        "total_words_evaluated": len(all_word_records),
        "mean_typographic_score": round(float(np.mean([w["T_score"] for w in all_word_records])), 4) if all_word_records else 0.0,
        "max_typographic_score": round(float(np.max([w["T_score"] for w in all_word_records])), 4) if all_word_records else 0.0,
        "candidates_proposed": len(candidates),
    }

    return candidates, doc_stats
