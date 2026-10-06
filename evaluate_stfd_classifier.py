#!/usr/bin/env python3
"""
TRUSTTRACE: STFD 5-Class Manipulation Classifier Evaluation Script.

Evaluates the FINAL SELECTED CHECKPOINT on the HELD-OUT TEST SET ONLY:
- Manifest: data/manifests/trusttrace_stfd_clustered_test.csv (590 samples)
- Computes Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
- Computes Per-class metrics and Support counts
- Generates 5x5 confusion matrix
- Saves:
  1. reports/stfd_manipulation_test_report.json
  2. reports/stfd_manipulation_test_report.md
  3. reports/stfd_manipulation_confusion_matrix.png

STRICT ANTI-OVERCLAIM: This evaluates the 5-class STFD manipulation type classifier,
NOT the final TRUSTTRACE authenticity decision layer.
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
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"
REPORTS_DIR = WORKSPACE_DIR / "reports"

TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"
BEST_MODEL_PATH = MODELS_DIR / "stfd_manipulation_best.pt"

TEST_REPORT_JSON = REPORTS_DIR / "stfd_manipulation_test_report.json"
TEST_REPORT_MD = REPORTS_DIR / "stfd_manipulation_test_report.md"
CONFUSION_MATRIX_PNG = REPORTS_DIR / "stfd_manipulation_confusion_matrix.png"

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
IMAGE_SIZE = (224, 224)


class STFDTestDataset(Dataset):
    def __init__(self, manifest_path: Path, transform=None):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Test manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.rows[idx]
        img_path = Path(row["image_path"])
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

        label = CLASS_TO_IDX[row["manipulation_type"]]
        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)

        return tensor_img, label, row["sample_id"]


class STFDManipulationClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = False,
        dropout_rate: float = 0.3,
        feature_dim: int = 256,
    ):
        super().__init__()
        weights = None
        backbone = models.mobilenet_v3_small(weights=weights)

        in_features = backbone.classifier[0].in_features  # 576
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        pooled = self.pool(feat_map)
        embedding = self.projection(pooled)
        logits = self.classifier(embedding)
        return logits


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label (Held-Out Test)",
        xlabel="Predicted Label",
        title="STFD 5-Class Manipulation Classifier\nConfusion Matrix (Held-Out Test Set)",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {save_path.name}")


def main():
    print("=" * 70)
    print("TRUSTTRACE: STFD 5-CLASS MANIPULATION CLASSIFIER EVALUATION")
    print("=" * 70)

    if not BEST_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {BEST_MODEL_PATH}. Train the model first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load checkpoint
    print(f"Loading checkpoint from {BEST_MODEL_PATH.name}...", flush=True)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

    model = STFDManipulationClassifier(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_epoch = checkpoint.get("epoch", "unknown")
    val_macro_f1 = checkpoint.get("val_macro_f1", 0.0)
    print(f"Loaded checkpoint trained to Epoch {best_epoch} (Val Macro-F1: {val_macro_f1:.4f}).")

    # 2. Test transform (strictly deterministic)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Load Test dataset
    test_dataset = STFDTestDataset(TEST_MANIFEST_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} test samples from {TEST_MANIFEST_PATH.name}.")

    # 4. Run inference
    print("\nEvaluating on held-out test set...", flush=True)
    t0 = time.time()
    all_preds = []
    all_targets = []
    all_probs = []
    sample_ids = []

    with torch.no_grad():
        for imgs, targets, sids in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
            sample_ids.extend(sids)

    eval_time = time.time() - t0
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 5. Compute metrics
    acc = float(accuracy_score(all_targets, all_preds))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    prec_per_class, rec_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    print("\n" + "=" * 55)
    print("HELD-OUT TEST RESULTS (STFD 5-CLASS MANIPULATION CLASSIFIER)")
    print("=" * 55)
    print(f"STFD 5-class manipulation classification test accuracy = {acc * 100:.2f}%")
    print(f"Test Macro Precision: {p_macro:.4f}")
    print(f"Test Macro Recall:    {r_macro:.4f}")
    print(f"Test Macro F1:        {f1_macro:.4f}")
    print(f"Test Weighted F1:     {f1_weighted:.4f}")
    print(f"Test Evaluation Time: {eval_time:.2f}s ({len(test_dataset)/eval_time:.1f} imgs/s)")

    print("\nPer-Class Breakdown (Held-Out Test Set):")
    print(f"{'Manipulation Class':<16} | {'Precision':>9} | {'Recall':>9} | {'F1-Score':>9} | {'Support':>7}")
    print("-" * 62)
    per_class_dict = {}
    for i, c_name in enumerate(CLASS_LABELS):
        p_c = float(prec_per_class[i])
        r_c = float(rec_per_class[i])
        f_c = float(f1_per_class[i])
        sup_c = int(support_per_class[i])
        per_class_dict[c_name] = {
            "precision": p_c,
            "recall": r_c,
            "f1": f_c,
            "support": sup_c,
        }
        print(f"{c_name:<16} | {p_c:>9.4f} | {r_c:>9.4f} | {f_c:>9.4f} | {sup_c:>7}")

    print("\nConfusion Matrix (Rows=True, Columns=Predicted):")
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{val:>4}" for val in row)
        print(f"  {CLASS_LABELS[i]:<14} [ {row_str} ]")

    # 6. Save Confusion Matrix Plot
    plot_confusion_matrix(cm, CLASS_LABELS, CONFUSION_MATRIX_PNG)

    # 7. Save JSON report
    report_data = {
        "model_architecture": "mobilenet_v3_small",
        "task": "STFD 5-Class Manipulation Classification",
        "checkpoint_epoch": best_epoch,
        "dataset": {
            "test_manifest": str(TEST_MANIFEST_PATH.name),
            "test_samples": len(test_dataset),
            "classes": CLASS_LABELS,
        },
        "metrics": {
            "test_accuracy": acc,
            "test_macro_precision": float(p_macro),
            "test_macro_recall": float(r_macro),
            "test_macro_f1": float(f1_macro),
            "test_weighted_f1": float(f1_weighted),
            "per_class": per_class_dict,
        },
        "confusion_matrix": cm.tolist(),
        "evaluation_time_seconds": eval_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": "This benchmark evaluates the STFD 5-class manipulation classification model, not the final TRUSTTRACE authenticity pipeline.",
    }

    with open(TEST_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nJSON test report saved to {TEST_REPORT_JSON.name}")

    # 8. Save Markdown report
    md_content = f"""# TRUSTTRACE: STFD 5-Class Manipulation Classifier Test Report

**Task:** STFD 5-Class Manipulation Type Classification  
**Model Backbone:** MobileNetV3-Small (Pretrained on ImageNet)  
**Input Resolution:** 224 × 224 × 3  
**Checkpoint Source:** Best Validation Macro-F1 Checkpoint (`stfd_manipulation_best.pt`, Epoch {best_epoch})  
**Evaluation Set:** Held-out Test Set (`trusttrace_stfd_clustered_test.csv`, {len(test_dataset)} samples)  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

> [!IMPORTANT]
> **Anti-Overclaim Notice:** STFD 5-class manipulation classification test accuracy = {acc * 100:.2f}%. This evaluates manipulation-type categorization on confirmed-tampered screenshots, NOT the full TRUSTTRACE authenticity decision system.

---

## 1. Overall Held-Out Test Metrics

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Test Accuracy** | {acc:.4f} | **{acc * 100:.2f}%** |
| **Macro Precision** | {p_macro:.4f} | {p_macro * 100:.2f}% |
| **Macro Recall** | {r_macro:.4f} | {r_macro * 100:.2f}% |
| **Macro F1-Score** | {f1_macro:.4f} | **{f1_macro * 100:.2f}%** |
| **Weighted F1-Score** | {f1_weighted:.4f} | {f1_weighted * 100:.2f}% |

---

## 2. Per-Class Breakdown

| Manipulation Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for c_name in CLASS_LABELS:
        d = per_class_dict[c_name]
        md_content += f"| **{c_name}** | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {d['support']} |\n"

    md_content += f"""
---

## 3. Confusion Matrix

| True \\ Predicted | {" | ".join(CLASS_LABELS)} |
| :--- | {" | ".join([":---:"] * NUM_CLASSES)} |
"""
    for i, row in enumerate(cm):
        md_content += f"| **{CLASS_LABELS[i]}** | " + " | ".join(str(val) for val in row) + " |\n"

    md_content += f"""
---

## 4. Hardware & Benchmark Performance
- **Inference Time:** {eval_time:.2f} seconds
- **Throughput:** {len(test_dataset)/eval_time:.1f} screenshots / second
- **Device:** {device}

"""
    with open(TEST_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown test report saved to {TEST_REPORT_MD.name}")


if __name__ == "__main__":
    main()
