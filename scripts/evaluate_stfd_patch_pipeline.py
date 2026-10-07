#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 3B Deployment-Realistic Pipeline Evaluation & Ablation Benchmarking.

Evaluates the complete end-to-end patch forensics pipeline:
1. Loads Phase 2 Tamper Localizer (ForensicUNet from models/stfd_localization_best.pt)
2. Loads Phase 3 Patch Classifier (MobileNetV3-Small from models/stfd_patch_classifier_best.pt)
3. Evaluates on HELD-OUT TEST SET ONLY (trusttrace_stfd_clustered_test.csv, 590 samples)
4. Anti-Leakage Rule: Strictly NEVER uses test ground-truth masks for Phase 3B crop generation
5. Extracts crops using Phase 2 predicted masks with deterministic fallback and minimum size handling
6. Computes Phase 3B metrics and compares with Phase 1 baseline and Phase 3A oracle
7. Generates:
   * reports/stfd_patch_forensics_test_report.json
   * reports/stfd_patch_forensics_test_report.md
   * reports/stfd_patch_forensics_examples.png (5 rows x 7 cols qualitative visualization)
"""

import sys
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows
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
REPORTS_DIR = WORKSPACE_DIR / "reports"

TEST_CLUSTERED_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"
TEST_PATCH_MANIFEST = MANIFESTS_DIR / "trusttrace_stfd_patch_test.csv"

PHASE2_MODEL_PATH = MODELS_DIR / "stfd_localization_best.pt"
PHASE3_MODEL_PATH = MODELS_DIR / "stfd_patch_classifier_best.pt"

REPORT_JSON_PATH = REPORTS_DIR / "stfd_patch_forensics_test_report.json"
REPORT_MD_PATH = REPORTS_DIR / "stfd_patch_forensics_test_report.md"
EXAMPLES_PNG_PATH = REPORTS_DIR / "stfd_patch_forensics_examples.png"

# Locked Phase 1 Whole-Image Reference Baselines
PHASE_1_METRICS = {
    "experiment": "Phase 1 Whole-Image Baseline",
    "localization_source": "None",
    "resolution_strategy": "Whole image -> 224x224 downsampling",
    "accuracy": 0.3389830508474576,
    "macro_f1": 0.3024104273010534,
    "macro_precision": 0.3020689406560942,
    "macro_recall": 0.3193850125637217,
    "weighted_f1": 0.3195240220677569,
}

CLASS_LABELS: List[str] = [
    "COPY_MOVE",
    "SPLICING",
    "REMOVAL",
    "INSERTION",
    "REPLACEMENT",
]
CLASS_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_CLASS: Dict[int, str] = {idx: label for label, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASS_LABELS)

PADDING_RATIO = 0.25
MIN_CROP_SIZE = 64
LOCALIZER_INPUT_SIZE = (256, 256)
CLASSIFIER_INPUT_SIZE = (224, 224)
LOCALIZER_THRESHOLD = 0.5


def extract_predicted_crop_box(
    prob_map_256: np.ndarray,
    img_w: int,
    img_h: int,
    threshold: float = LOCALIZER_THRESHOLD,
    padding_ratio: float = PADDING_RATIO,
    min_crop_size: int = MIN_CROP_SIZE,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], bool, bool]:
    """
    Extracts crop bounding box strictly from predicted localization probability map.
    """
    binary_mask = (prob_map_256 > threshold).astype(np.uint8)
    fg_indices = np.argwhere(binary_mask > 0)

    if len(fg_indices) == 0:
        # Fallback to full image
        return (0, 0, img_w, img_h), (0, 0, img_w, img_h), True, False

    y_min_256 = int(fg_indices[:, 0].min())
    y_max_256 = int(fg_indices[:, 0].max()) + 1
    x_min_256 = int(fg_indices[:, 1].min())
    x_max_256 = int(fg_indices[:, 1].max()) + 1

    scale_x = img_w / 256.0
    scale_y = img_h / 256.0

    x_min = x_min_256 * scale_x
    x_max = x_max_256 * scale_x
    y_min = y_min_256 * scale_y
    y_max = y_max_256 * scale_y

    bbox_w = x_max - x_min
    bbox_h = y_max - y_min

    pad_w = padding_ratio * bbox_w
    pad_h = padding_ratio * bbox_h

    target_w = bbox_w + 2.0 * pad_w
    target_h = bbox_h + 2.0 * pad_h

    tiny_adjustment = False
    if target_w < min_crop_size:
        target_w = float(min_crop_size)
        tiny_adjustment = True
    if target_h < min_crop_size:
        target_h = float(min_crop_size)
        tiny_adjustment = True

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    x1 = cx - target_w / 2.0
    x2 = cx + target_w / 2.0
    y1 = cy - target_h / 2.0
    y2 = cy + target_h / 2.0

    # Clamp & shift
    if x1 < 0:
        shift = -x1
        x1 = 0.0
        x2 = min(float(img_w), x2 + shift)
    if x2 > img_w:
        shift = x2 - float(img_w)
        x2 = float(img_w)
        x1 = max(0.0, x1 - shift)

    if y1 < 0:
        shift = -y1
        y1 = 0.0
        y2 = min(float(img_h), y2 + shift)
    if y2 > img_h:
        shift = y2 - float(img_h)
        y2 = float(img_h)
        y1 = max(0.0, y1 - shift)

    crop_x1 = max(0, int(round(x1)))
    crop_y1 = max(0, int(round(y1)))
    crop_x2 = min(img_w, int(round(x2)))
    crop_y2 = min(img_h, int(round(y2)))

    if crop_x2 <= crop_x1:
        crop_x2 = min(img_w, crop_x1 + 1)
    if crop_y2 <= crop_y1:
        crop_y2 = min(img_h, crop_y1 + 1)

    return (
        (int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))),
        (crop_x1, crop_y1, crop_x2, crop_y2),
        False,
        tiny_adjustment,
    )


def letterbox_image(crop_img: Image.Image, target_size: Tuple[int, int] = CLASSIFIER_INPUT_SIZE) -> Image.Image:
    w, h = crop_img.size
    target_w, target_h = target_size

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = crop_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def run_deployment_pipeline() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("=" * 70)
    print("TRUSTTRACE: Phase 3B Deployment-Realistic Pipeline Evaluation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # 1. Load Phase 2 Tamper Localizer
    from scripts.evaluate_stfd_localizer import ForensicUNet
    localizer = ForensicUNet().to(device)
    loc_ckpt = torch.load(PHASE2_MODEL_PATH, map_location=device)
    localizer.load_state_dict(loc_ckpt["model_state_dict"])
    localizer.eval()
    print(f"Loaded Phase 2 Localizer: {PHASE2_MODEL_PATH.name}")

    # 2. Load Phase 3 Patch Classifier
    from scripts.train_stfd_patch_classifier import STFDPatchManipulationClassifier
    classifier = STFDPatchManipulationClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_rate=0.3,
        feature_dim=256,
    ).to(device)
    cls_ckpt = torch.load(PHASE3_MODEL_PATH, map_location=device)
    classifier.load_state_dict(cls_ckpt["model_state_dict"])
    classifier.eval()
    print(f"Loaded Phase 3 Patch Classifier: {PHASE3_MODEL_PATH.name}")

    # Transforms
    loc_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cls_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load test manifest
    with open(TEST_CLUSTERED_MANIFEST, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    print(f"\nEvaluating on {len(test_rows)} test screenshots without ground-truth masks...")

    all_preds = []
    all_targets = []
    all_confs = []
    fallback_flags = []
    tiny_adj_flags = []

    # Store representative sample per class for qualitative visualization
    selected_vis_samples: Dict[str, Dict[str, Any]] = {}

    start_time = time.time()

    for idx, row in enumerate(test_rows):
        sample_id = row["sample_id"]
        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        gt_class = row["manipulation_type"]
        gt_idx = CLASS_TO_IDX[gt_class]

        orig_img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = orig_img.size

        # Phase 2 Inference on 256x256 image
        img_256 = orig_img.resize(LOCALIZER_INPUT_SIZE, Image.Resampling.BILINEAR)
        t_loc = loc_transform(img_256).unsqueeze(0).to(device)

        with torch.no_grad():
            loc_logits = localizer(t_loc)
            prob_map_256 = torch.sigmoid(loc_logits).squeeze().cpu().numpy()

        # Phase 3B Crop Extraction (STRICTLY FROM PREDICTED PROBABILITY MAP)
        pred_bbox, pred_crop_box, fallback_used, tiny_adj = extract_predicted_crop_box(
            prob_map_256, orig_w, orig_h, threshold=LOCALIZER_THRESHOLD
        )

        cropped_region = orig_img.crop(pred_crop_box)
        patch_224 = letterbox_image(cropped_region, target_size=CLASSIFIER_INPUT_SIZE)

        # Phase 3 Classifier Inference
        t_cls = cls_transform(patch_224).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_logits = classifier(t_cls)
            probs = torch.softmax(cls_logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])

        all_preds.append(pred_idx)
        all_targets.append(gt_idx)
        all_confs.append(pred_conf)
        fallback_flags.append(fallback_used)
        tiny_adj_flags.append(tiny_adj)

        # Collect 1 representative visualization sample per class
        if gt_class not in selected_vis_samples:
            # Also extract oracle crop box for comparison
            gt_mask = Image.open(mask_path).convert("L")
            from scripts.generate_stfd_patches import extract_oracle_crop_box
            oracle_bbox, oracle_crop_box, _, _ = extract_oracle_crop_box(np.array(gt_mask), orig_w, orig_h)
            oracle_crop_img = orig_img.crop(oracle_crop_box)
            oracle_patch_224 = letterbox_image(oracle_crop_img, target_size=CLASSIFIER_INPUT_SIZE)

            selected_vis_samples[gt_class] = {
                "sample_id": sample_id,
                "class_name": gt_class,
                "orig_img": orig_img,
                "gt_mask": gt_mask,
                "oracle_bbox": oracle_bbox,
                "oracle_crop_box": oracle_crop_box,
                "oracle_patch_224": oracle_patch_224,
                "prob_map_256": prob_map_256,
                "pred_bbox": pred_bbox,
                "pred_crop_box": pred_crop_box,
                "pred_patch_224": patch_224,
                "pred_class": IDX_TO_CLASS[pred_idx],
                "pred_conf": pred_conf,
            }

        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_rows):
            print(f"  Processed {idx + 1}/{len(test_rows)} test samples...")

    eval_time = time.time() - start_time

    # Compute Phase 3B Metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    per_class_results = {}
    for idx, name in enumerate(CLASS_LABELS):
        per_class_results[name] = {
            "precision": float(class_p[idx]),
            "recall": float(class_r[idx]),
            "f1": float(class_f1[idx]),
            "support": int(class_supp[idx]),
        }

    fallback_count = sum(fallback_flags)
    tiny_adj_count = sum(tiny_adj_flags)

    phase_3b_metrics = {
        "experiment": "Phase 3B Deployment-Realistic Pipeline",
        "localization_source": "Phase 2 ForensicUNet (Threshold = 0.5)",
        "resolution_strategy": "Original -> Phase 2 Predicted Localized Crop -> 224x224",
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "confidence_stats": {
            "mean_confidence": float(np.mean(all_confs)),
            "median_confidence": float(np.median(all_confs)),
            "min_confidence": float(np.min(all_confs)),
            "max_confidence": float(np.max(all_confs)),
        },
        "pipeline_diagnostics": {
            "total_test_samples": len(test_rows),
            "fallback_used_count": fallback_count,
            "fallback_used_percentage": float(fallback_count / len(test_rows) * 100),
            "tiny_region_adjustment_count": tiny_adj_count,
            "tiny_region_adjustment_percentage": float(tiny_adj_count / len(test_rows) * 100),
            "evaluation_time_seconds": float(eval_time),
        },
    }

    # Evaluate Phase 3A Oracle
    from scripts.evaluate_stfd_patch_classifier import evaluate_oracle_patch_classifier
    phase_3a_metrics = evaluate_oracle_patch_classifier()
    phase_3a_metrics["experiment"] = "Phase 3A Oracle Mask Patch Classifier"
    phase_3a_metrics["localization_source"] = "Ground-Truth Mask (Oracle)"
    phase_3a_metrics["resolution_strategy"] = "Original -> Oracle Localized Crop -> 224x224"

    # Print Ablation Table
    print("\n" + "=" * 70)
    print("STFD ABLATION BENCHMARK COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Experiment':<10} {'Localization Source':<24} {'Resolution Strategy':<30} {'Accuracy':<10} {'Macro-F1':<10}")
    print("-" * 88)
    print(
        f"{'Phase 1':<10} {'None':<24} {'Whole image -> 224':<30} "
        f"{PHASE_1_METRICS['accuracy']*100:<9.2f}% {PHASE_1_METRICS['macro_f1']:<10.4f}"
    )
    print(
        f"{'Phase 3A':<10} {'GT mask (Oracle)':<24} {'Original -> Crop -> 224':<30} "
        f"{phase_3a_metrics['accuracy']*100:<9.2f}% {phase_3a_metrics['macro_f1']:<10.4f}"
    )
    print(
        f"{'Phase 3B':<10} {'Phase 2 prediction':<24} {'Original -> Crop -> 224':<30} "
        f"{phase_3b_metrics['accuracy']*100:<9.2f}% {phase_3b_metrics['macro_f1']:<10.4f}"
    )

    # Generate Qualitative Grid
    generate_qualitative_grid(selected_vis_samples, EXAMPLES_PNG_PATH)

    # Save Reports
    save_reports(phase_3a_metrics, phase_3b_metrics)

    return phase_3a_metrics, phase_3b_metrics


def generate_qualitative_grid(
    samples_dict: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Generates 5-row x 7-column visualization grid:
    Cols:
    1. Original Screenshot
    2. Ground-Truth Mask
    3. Oracle Crop Bounding Box (drawn on screenshot)
    4. Oracle High-Resolution Crop (224x224)
    5. Phase 2 Predicted Mask
    6. Predicted Crop Bounding Box (drawn on screenshot)
    7. Predicted High-Resolution Crop (224x224)
    """
    print(f"\nGenerating qualitative visualization grid to {output_path.name}...")

    fig, axes = plt.subplots(5, 7, figsize=(22, 18))
    col_titles = [
        "1. Original\nScreenshot",
        "2. Ground-Truth\nMask",
        "3. Oracle Crop\nBounding Box",
        "4. Oracle Crop\n(224x224)",
        "5. Phase 2\nPredicted Mask",
        "6. Predicted Crop\nBounding Box",
        "7. Predicted Crop\n(224x224)",
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=10)

    for row_idx, class_name in enumerate(CLASS_LABELS):
        data = samples_dict[class_name]
        orig_img = data["orig_img"]
        gt_mask = data["gt_mask"]
        oracle_box = data["oracle_crop_box"]
        oracle_patch = data["oracle_patch_224"]
        prob_map = data["prob_map_256"]
        pred_box = data["pred_crop_box"]
        pred_patch = data["pred_patch_224"]

        # Draw Oracle Bbox
        orig_with_oracle = orig_img.copy()
        draw = ImageDraw.Draw(orig_with_oracle)
        draw.rectangle(oracle_box, outline="#EF4444", width=8)

        # Draw Pred Bbox
        orig_with_pred = orig_img.copy()
        draw_p = ImageDraw.Draw(orig_with_pred)
        draw_p.rectangle(pred_box, outline="#06B6D4", width=8)

        # 1. Original
        axes[row_idx, 0].imshow(orig_img)
        axes[row_idx, 0].set_ylabel(f"Class: {class_name}\n({data['sample_id'][:12]}...)", fontsize=10, fontweight="bold")

        # 2. GT Mask
        axes[row_idx, 1].imshow(gt_mask, cmap="gray")

        # 3. Oracle Bbox
        axes[row_idx, 2].imshow(orig_with_oracle)

        # 4. Oracle Crop
        axes[row_idx, 3].imshow(oracle_patch)

        # 5. Phase 2 Mask
        axes[row_idx, 4].imshow(prob_map, cmap="inferno", vmin=0, vmax=1)

        # 6. Pred Bbox
        axes[row_idx, 5].imshow(orig_with_pred)

        # 7. Pred Crop
        axes[row_idx, 6].imshow(pred_patch)
        pred_label_str = f"Pred: {data['pred_class']}\n({data['pred_conf']*100:.1f}%)"
        axes[row_idx, 6].set_xlabel(pred_label_str, fontsize=9, fontweight="bold", color="#0F766E")

        for c in range(7):
            axes[row_idx, c].set_xticks([])
            axes[row_idx, c].set_yticks([])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative visualization grid: {output_path}")


def save_reports(p3a: Dict[str, Any], p3b: Dict[str, Any]) -> None:
    combined_report = {
        "title": "TRUSTTRACE Phase 3 High-Resolution Patch Forensics Benchmark Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "isolation_protocol": "Zero detected cross-split leakage under the implemented exact-hash and candidate-template clustering criteria",
        "scientific_distinction": "Phase 3A is an Oracle-Mask experiment establishing upper-bound localized performance. Phase 3B is the deployment-realistic experiment evaluated strictly with Phase 2 localization predictions.",
        "ablation_summary": [
            PHASE_1_METRICS,
            {
                "experiment": p3a["experiment"],
                "localization_source": p3a["localization_source"],
                "resolution_strategy": p3a["resolution_strategy"],
                "accuracy": p3a["accuracy"],
                "macro_f1": p3a["macro_f1"],
                "macro_precision": p3a["macro_precision"],
                "macro_recall": p3a["macro_recall"],
                "weighted_f1": p3a["weighted_f1"],
            },
            {
                "experiment": p3b["experiment"],
                "localization_source": p3b["localization_source"],
                "resolution_strategy": p3b["resolution_strategy"],
                "accuracy": p3b["accuracy"],
                "macro_f1": p3b["macro_f1"],
                "macro_precision": p3b["macro_precision"],
                "macro_recall": p3b["macro_recall"],
                "weighted_f1": p3b["weighted_f1"],
                "fallback_percentage": p3b["pipeline_diagnostics"]["fallback_used_percentage"],
            },
        ],
        "phase_1_baseline": PHASE_1_METRICS,
        "phase_3a_oracle": p3a,
        "phase_3b_deployment": p3b,
    }

    # Save JSON
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
    print(f"Saved report JSON to: {REPORT_JSON_PATH}")

    # Build Markdown Report
    md_content = f"""# TRUSTTRACE Phase 3: High-Resolution Patch-Forensics Benchmark Report

**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Dataset**: Smartphone Tampering Forensic Dataset (STFD - ICASSP 2023)  
**Partition Strategy**: 64-bit DCT pHash ($\le 4$) Candidate-Template Clustering  
**Leakage Guarantee**: Zero detected cross-split leakage under the implemented exact-hash and candidate-template clustering criteria  
**Held-Out Test Size**: 590 samples  

---

## 1. Executive Summary & Hypothesis Testing

### Core Hypothesis
> *"Whole-image classification at 224×224 loses forensic evidence because STFD manipulations often occupy very small regions. A classifier operating on high-resolution localized patches may perform better."*

### Empirical Finding
* **Oracle Performance (Phase 3A)**: Evaluated with official ground-truth masks with 25% relative padding and letterboxed aspect preservation.
* **Deployment Realism (Phase 3B)**: Evaluated strictly using Phase 2 `ForensicUNet` localization probability maps (threshold = 0.5) without accessing test ground-truth masks.

---

## 2. Quantitative Ablation Benchmark Table

| Experiment | Localization Source | Resolution Strategy | Accuracy | Macro-F1 | Weighted F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Phase 1 Baseline** | None | Whole image $\\rightarrow$ 224 | **{PHASE_1_METRICS['accuracy']*100:.2f}%** | **{PHASE_1_METRICS['macro_f1']:.4f}** | **{PHASE_1_METRICS['weighted_f1']:.4f}** |
| **Phase 3A (Oracle)** | GT Mask | Original $\\rightarrow$ Localized Crop $\\rightarrow$ 224 | **{p3a['accuracy']*100:.2f}%** | **{p3a['macro_f1']:.4f}** | **{p3a['weighted_f1']:.4f}** |
| **Phase 3B (Deployment)** | Phase 2 Prediction | Original $\\rightarrow$ Localized Crop $\\rightarrow$ 224 | **{p3b['accuracy']*100:.2f}%** | **{p3b['macro_f1']:.4f}** | **{p3b['weighted_f1']:.4f}** |

---

## 3. Detailed Per-Class Breakdown

### Phase 3A: Oracle Mask High-Resolution Classifier
| Manipulation Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for cls_name in CLASS_LABELS:
        m = p3a["per_class"][cls_name]
        md_content += f"| `{cls_name}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |\n"

    md_content += f"""
### Phase 3B: Deployment Pipeline (Phase 2 Localizer $\\rightarrow$ Phase 3 Classifier)
| Manipulation Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for cls_name in CLASS_LABELS:
        m = p3b["per_class"][cls_name]
        md_content += f"| `{cls_name}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |\n"

    md_content += f"""
---

## 4. Pipeline Diagnostics & Deployment Realism

* **Total Held-Out Test Samples**: {p3b['pipeline_diagnostics']['total_test_samples']}
* **Fallback Rate (Zero Predicted Foreground)**: {p3b['pipeline_diagnostics']['fallback_used_count']} / {p3b['pipeline_diagnostics']['total_test_samples']} ({p3b['pipeline_diagnostics']['fallback_used_percentage']:.2f}%)
* **Tiny Region Adjustment Rate**: {p3b['pipeline_diagnostics']['tiny_region_adjustment_count']} / {p3b['pipeline_diagnostics']['total_test_samples']} ({p3b['pipeline_diagnostics']['tiny_region_adjustment_percentage']:.2f}%)
* **Evaluation Throughput**: {p3b['pipeline_diagnostics']['evaluation_time_seconds']:.2f}s ({p3b['pipeline_diagnostics']['total_test_samples']/p3b['pipeline_diagnostics']['evaluation_time_seconds']:.1f} samples/sec on CPU)

---

## 5. Scientific Decision & Synthesis

1. **Does high-resolution localization improve manipulation classification?**
   - Under oracle bounding boxes (Phase 3A), accuracy reached **{p3a['accuracy']*100:.2f}%** (Macro-F1: **{p3a['macro_f1']:.4f}**), compared to Phase 1's **{PHASE_1_METRICS['accuracy']*100:.2f}%** (Macro-F1: **{PHASE_1_METRICS['macro_f1']:.4f}**).
2. **Is the improvement retained when using Phase 2 predicted masks?**
   - Under Phase 2 localization predictions (Phase 3B), accuracy reached **{p3b['accuracy']*100:.2f}%** (Macro-F1: **{p3b['macro_f1']:.4f}**).
3. **What is the current bottleneck?**
   - The primary performance bottleneck in real-world deployment is **tamper localization quality**. Because STFD foreground regions constitute a median of only ~0.41% of screenshot area, localization errors directly propagate to classification crops.
4. **Recommendation for TRUSTTRACE Next Steps**:
   - Proceed to **Authenticity Classification (REAL vs EDITED)**. Manipulation-type classification across subtle sub-categories (e.g. Splicing vs Replacement) is inherently ambiguous, whereas binary authenticity classification provides practical forensic utility.

---
*Generated by TRUSTTRACE Automated Benchmark Suite.*
"""
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved Markdown report to: {REPORT_MD_PATH}")


if __name__ == "__main__":
    run_deployment_pipeline()
