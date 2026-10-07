#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 5 OCR & Typography Forensic Feature Extractor.

Extracts fine-grained layout, typography, and token consistency forensic features
from raw OCR documents stored in data/ocr/:
- Baseline drift (vertical misalignment characteristic of pasted/spliced text)
- Kerning and inter-word spacing collisions (negative gaps between words)
- Font height and bounding box scaling variance within lines
- Line spacing regularity and margin alignment drift
- Numeric token density, price repetitions, and date consistency
- Outputs: data/manifests/phase5_ocr_features.csv
"""

import sys
import os
import csv
import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

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

MASTER_MANIFEST_PATH = MANIFESTS_DIR / "phase4_dataset_master.csv"
OUTPUT_CSV_PATH = MANIFESTS_DIR / "phase5_ocr_features.csv"

# Precompiled regex patterns for forensic token parsing
PRICE_PATTERN = re.compile(r"^\$?\d+\.\d{2}$")
DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
NUMERIC_PATTERN = re.compile(r"^\d+(\.\d+)?$")


def extract_features_from_ocr_json(
    ocr_json_path: Path, img_w: int, img_h: int
) -> Dict[str, Any]:
    """
    Extracts rich typographic and linguistic forensic metrics from raw OCR output.
    """
    default_features = {
        "ocr_available": 0,
        "word_count": 0,
        "line_count": 0,
        "char_count": 0,
        "text_density": 0.0,
        "max_baseline_drift": 0.0,
        "mean_baseline_drift": 0.0,
        "anomalous_lines_count": 0,
        "kerning_irregularities_count": 0,
        "font_height_variance": 0.0,
        "font_anomaly_detected": 0,
        "numeric_token_count": 0,
        "numeric_token_ratio": 0.0,
        "repeated_token_count": 0,
        "repeated_token_ratio": 0.0,
        "price_pattern_count": 0,
        "duplicate_amounts_count": 0,
        "line_spacing_variance": 0.0,
        "left_margin_alignment_std": 0.0,
        "error_warning": "MISSING_JSON",
    }

    if not ocr_json_path.is_file():
        return default_features

    try:
        with open(ocr_json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        default_features["error_warning"] = f"JSON_CORRUPT: {e}"
        return default_features

    if data.get("status") != "AVAILABLE":
        default_features["error_warning"] = data.get("error", "ENGINE_UNAVAILABLE")
        return default_features

    lines = data.get("lines", [])
    raw_text = data.get("text", "")
    char_count = len(raw_text)
    doc_area = max(img_w * img_h, 1)

    # Collect all words and line metrics
    all_words: List[str] = []
    line_tops: List[float] = []
    line_lefts: List[float] = []
    baseline_drifts: List[float] = []
    anomalous_lines = 0
    kerning_collisions = 0
    line_font_height_vars: List[float] = []

    for line in lines:
        words = line.get("words", [])
        if not words:
            continue

        words_sorted = sorted(words, key=lambda w: w["bbox"][0])
        valid_words = [w for w in words_sorted if w["bbox"][2] > 2 and w["bbox"][3] > 2]
        if not valid_words:
            continue

        for w in valid_words:
            all_words.append(w["text"].strip())

        # Line layout metrics
        l_x = min(w["bbox"][0] for w in valid_words)
        l_y = min(w["bbox"][1] for w in valid_words)
        line_lefts.append(l_x)
        line_tops.append(l_y)

        # Baseline drift analysis within line (words in same line should share baseline)
        if len(valid_words) >= 2:
            baselines = [w["bbox"][1] + w["bbox"][3] for w in valid_words]
            heights = [w["bbox"][3] for w in valid_words]
            b_std = float(np.std(baselines))
            baseline_drifts.append(b_std)
            line_font_height_vars.append(float(np.var(heights)))

            median_h = float(np.median(heights))
            if b_std > 5.0 and (b_std / max(median_h, 1.0)) > 0.25:
                anomalous_lines += 1

            # Inter-word horizontal spacing (kerning) analysis
            for i in range(len(valid_words) - 1):
                gap = valid_words[i + 1]["bbox"][0] - (valid_words[i]["bbox"][0] + valid_words[i]["bbox"][2])
                if gap < -3:  # Box collision / overlap indicates synthetic text overlay
                    kerning_collisions += 1

    word_count = len(all_words)
    line_count = len(lines)
    max_b_drift = float(np.max(baseline_drifts)) if baseline_drifts else 0.0
    mean_b_drift = float(np.mean(baseline_drifts)) if baseline_drifts else 0.0
    font_h_var = float(np.mean(line_font_height_vars)) if line_font_height_vars else 0.0

    # Margin alignment and line spacing regularity
    left_margin_std = float(np.std(line_lefts)) if len(line_lefts) >= 2 else 0.0
    line_spacing_var = 0.0
    if len(line_tops) >= 2:
        sorted_tops = sorted(line_tops)
        spacings = [sorted_tops[i + 1] - sorted_tops[i] for i in range(len(sorted_tops) - 1)]
        line_spacing_var = float(np.var(spacings))

    # Token and semantic regularity analysis
    numeric_tokens = [w for w in all_words if NUMERIC_PATTERN.match(w)]
    price_tokens = [w for w in all_words if PRICE_PATTERN.match(w)]
    repeated_words = [count for w, count in Counter(all_words).items() if count > 1 and len(w) > 3]
    duplicate_amounts = [count for w, count in Counter(price_tokens).items() if count > 1]

    font_anomaly_detected = int(
        (anomalous_lines >= 1 and max_b_drift > 6.0) or (kerning_collisions >= 2)
    )

    return {
        "ocr_available": 1,
        "word_count": word_count,
        "line_count": line_count,
        "char_count": char_count,
        "text_density": round(float(char_count / doc_area * 1000.0), 4),
        "max_baseline_drift": round(max_b_drift, 3),
        "mean_baseline_drift": round(mean_b_drift, 3),
        "anomalous_lines_count": anomalous_lines,
        "kerning_irregularities_count": kerning_collisions,
        "font_height_variance": round(font_h_var, 3),
        "font_anomaly_detected": font_anomaly_detected,
        "numeric_token_count": len(numeric_tokens),
        "numeric_token_ratio": round(float(len(numeric_tokens) / max(word_count, 1)), 4),
        "repeated_token_count": sum(repeated_words),
        "repeated_token_ratio": round(float(sum(repeated_words) / max(word_count, 1)), 4),
        "price_pattern_count": len(price_tokens),
        "duplicate_amounts_count": sum(duplicate_amounts),
        "line_spacing_variance": round(line_spacing_var, 3),
        "left_margin_alignment_std": round(left_margin_std, 3),
        "error_warning": "NONE",
    }


def main():
    print("=" * 70)
    print("TRUSTTRACE: Extracting OCR Forensic Features")
    print("=" * 70)

    if not MASTER_MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Master manifest missing: {MASTER_MANIFEST_PATH}")

    with open(MASTER_MANIFEST_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} samples from master manifest.")
    print(f"Reading raw OCR results from {OCR_DIR}...")

    # Load partition split mapping from train/val/test manifests
    split_map: Dict[str, str] = {}
    for s_name in ["train", "val", "test"]:
        p = MANIFESTS_DIR / f"trusttrace_phase4_{s_name}.csv"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    split_map[r["sample_id"]] = s_name

    out_records: List[Dict[str, Any]] = []
    success_count = 0

    for idx, r in enumerate(rows):
        sample_id = r["sample_id"]
        img_p = r["image_path"]
        w = int(r.get("width", 800))
        h = int(r.get("height", 1200))
        split = split_map.get(sample_id, r.get("split_original", "unknown"))

        ocr_json_path = OCR_DIR / f"{sample_id}_ocr.json"
        features = extract_features_from_ocr_json(ocr_json_path, w, h)

        if features["ocr_available"] == 1:
            success_count += 1

        rec = {
            "sample_id": sample_id,
            "image_path": img_p,
            "split": split,
            "label": r["label"],
            "group_id": r["group_id"],
            **features,
        }
        out_records.append(rec)

    print(f"OCR successfully available for {success_count} / {len(rows)} samples ({success_count/len(rows)*100:.1f}%).")

    # Write output CSV
    fieldnames = list(out_records[0].keys())
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_records)

    print(f"Saved OCR forensic features manifest to: {OUTPUT_CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
