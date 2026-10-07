#!/usr/bin/env python3
"""
TRUSTTRACE: STFD Oracle Patch Classifier Evaluation Script (Phase 3A).

Evaluates the trained high-resolution patch classifier on the held-out test split:
- Manifest: data/manifests/trusttrace_stfd_patch_test.csv (590 samples)
- Checkpoint: models/stfd_patch_classifier_best.pt
- Strictly evaluates on the test partition without fitting or altering thresholds
- Computes:
  * Accuracy
  * Macro Precision, Macro Recall, Macro-F1
  * Weighted F1
  * Per-class metrics
  * 5x5 Confusion Matrix
  * Prediction confidence statistics
- Compares directly against Phase 1 whole-image reference baseline:
  * Phase 1 Accuracy: 33.90%
  * Phase 1 Macro-F1: 0.3024
"""

import sys
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

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

TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_patch_test.csv"
BEST_MODEL_PATH = MODELS_DIR / "stfd_patch_classifier_best.pt"

# Phase 1 Reference Baselines
PHASE_1_ACC = 0.3389830508474576  # 33.90%
PHASE_1_MACRO_F1 = 0.3024104273010534  # 0.3024

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


class STFDPatchTestDataset(Dataset):
    def __init__(self, manifest_path: Path):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.rows[idx]
        crop_path = Path(row["crop_path"])
        if not crop_path.is_file():
            raise FileNotFoundError(f"Crop not found: {crop_path}")
        img = Image.open(crop_path).convert("RGB")
        tensor_img = self.transform(img)

        manip = row.get("manipulation_class") or row.get("manipulation_type")
        label = CLASS_TO_IDX[manip]
        return tensor_img, label, row["sample_id"]


def evaluate_oracle_patch_classifier() -> Dict[str, Any]:
    print("=" * 70)
    print("TRUSTTRACE: Phase 3A Oracle Patch Classifier Test Evaluation")
    print("=" * 70)

    if not BEST_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {BEST_MODEL_PATH}")

    # Import model architecture
    from scripts.train_stfd_patch_classifier import STFDPatchManipulationClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from: {BEST_MODEL_PATH}")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model = STFDPatchManipulationClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_rate=checkpoint.get("dropout_rate", 0.3),
        feature_dim=checkpoint.get("feature_dim", 256),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded successfully (trained epoch: {checkpoint.get('epoch', 'unknown')}).")

    test_dataset = STFDPatchTestDataset(TEST_MANIFEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Evaluating on held-out test split: {len(test_dataset)} samples...")

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_confs: List[float] = []
    all_sample_ids: List[str] = []

    t_start = time.time()
    with torch.no_grad():
        for inputs, targets, sample_ids in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())
            all_sample_ids.extend(sample_ids)

    eval_time = time.time() - t_start

    # Metrics
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

    conf_stats = {
        "mean_confidence": float(np.mean(all_confs)),
        "median_confidence": float(np.median(all_confs)),
        "min_confidence": float(np.min(all_confs)),
        "max_confidence": float(np.max(all_confs)),
    }

    print("\n" + "=" * 70)
    print("PHASE 3A ORACLE PATCH CLASSIFIER RESULTS (HELD-OUT TEST SET)")
    print("=" * 70)
    print(f"Accuracy:        {acc * 100:.2f}% (Phase 1 Baseline: {PHASE_1_ACC * 100:.2f}%)")
    print(f"Macro-F1:        {macro_f1:.4f}  (Phase 1 Baseline: {PHASE_1_MACRO_F1:.4f})")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    print(f"Mean Confidence: {conf_stats['mean_confidence'] * 100:.2f}%")
    print(f"Evaluation Time: {eval_time:.2f}s ({len(test_dataset)/eval_time:.1f} samples/sec)")

    print("\nPer-Class Detailed Metrics:")
    print(f"{'Class':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 58)
    for name in CLASS_LABELS:
        m = per_class_results[name]
        print(f"{name:<14} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<8d}")

    print("\n5x5 Confusion Matrix:")
    header = f"{'True \\ Pred':<14}" + "".join(f"{name[:8]:>10}" for name in CLASS_LABELS)
    print(header)
    print("-" * len(header))
    for idx, name in enumerate(CLASS_LABELS):
        row_str = f"{name:<14}" + "".join(f"{cm[idx, j]:>10d}" for j in range(NUM_CLASSES))
        print(row_str)

    delta_acc = acc - PHASE_1_ACC
    delta_f1 = macro_f1 - PHASE_1_MACRO_F1
    print("\nBaseline Comparison (Phase 3A Oracle vs Phase 1 Whole-Image):")
    print(f"  Accuracy Delta: {delta_acc * 100:+.2f}% ({acc*100:.2f}% vs {PHASE_1_ACC*100:.2f}%)")
    print(f"  Macro-F1 Delta: {delta_f1:+.4f} ({macro_f1:.4f} vs {PHASE_1_MACRO_F1:.4f})")

    results = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_results,
        "confusion_matrix": cm.tolist(),
        "confidence_stats": conf_stats,
        "delta_accuracy_vs_phase1": float(delta_acc),
        "delta_macro_f1_vs_phase1": float(delta_f1),
        "evaluation_time_seconds": float(eval_time),
        "test_samples": len(test_dataset),
    }

    return results


if __name__ == "__main__":
    evaluate_oracle_patch_classifier()
