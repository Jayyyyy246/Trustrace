#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 4 Binary Authenticity Classifier Training Script.

Trains a binary authenticity classifier (REAL vs EDITED) on receipt documents:
- Architecture: Pretrained MobileNetV3-Small backbone
                -> AdaptiveAvgPool2d((1, 1))
                -> Dropout(0.2)
                -> Linear(576, 2)
- Input: 224x224 RGB
- Targets: REAL (0), EDITED (1)
- Loss: CrossEntropyLoss with Inverse Class Frequency weights strictly computed from training split
- Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Checkpoint: models/phase4_authenticity_best.pt (selected on best validation Macro-F1)
- Reports:
  * reports/phase4_authenticity_config.json
  * reports/phase4_authenticity_training_history.json
  * reports/phase4_authenticity_training_history.png
"""

import sys
import os
import csv
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"
REPORTS_DIR = WORKSPACE_DIR / "reports"

TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_phase4_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_phase4_val.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_phase4_test.csv"

BEST_MODEL_PATH = MODELS_DIR / "phase4_authenticity_best.pt"
CONFIG_JSON_PATH = REPORTS_DIR / "phase4_authenticity_config.json"
HISTORY_JSON_PATH = REPORTS_DIR / "phase4_authenticity_training_history.json"
HISTORY_PNG_PATH = REPORTS_DIR / "phase4_authenticity_training_history.png"

# Hyperparameters
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2

CLASS_LABELS: List[str] = ["REAL", "EDITED"]
CLASS_TO_IDX: Dict[str, int] = {label: idx for idx, label in enumerate(CLASS_LABELS)}
IDX_TO_CLASS: Dict[int, str] = {idx: label for label, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = 2


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ReceiptAuthenticityDataset(Dataset):
    """
    Receipt Authenticity Dataset.
    Loads and caches RGB images in RAM to eliminate disk I/O bottlenecks.
    """
    def __init__(self, manifest_path: Path, transform=None, is_train: bool = False):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.transform = transform
        self.is_train = is_train
        self._cached_images: List[Image.Image] = []
        self._labels: List[int] = []

        print(f"Preloading {len(self.rows)} images from {manifest_path.name} into RAM...", flush=True)
        for r in self.rows:
            label_str = r["label"]
            if label_str not in CLASS_TO_IDX:
                raise ValueError(f"Unknown label '{label_str}' in row {r['sample_id']}")
            self._labels.append(CLASS_TO_IDX[label_str])

            img_p = Path(r["image_path"])
            with Image.open(img_p) as img:
                self._cached_images.append(img.convert("RGB"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self._cached_images[idx]
        target = self._labels[idx]

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.functional.to_tensor(img)

        return img_tensor, target


class MobileNetV3AuthenticityClassifier(nn.Module):
    """
    MobileNetV3-Small binary authenticity classifier:
    backbone.features -> AdaptiveAvgPool2d((1, 1)) -> Dropout -> Linear(576, 2)
    """
    def __init__(self, pretrained: bool = True, dropout_rate: float = DROPOUT_RATE):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        in_features = backbone.classifier[0].in_features  # 576
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, NUM_CLASSES),
        )

        # Initialize linear classification head
        nn.init.xavier_normal_(self.classifier[2].weight)
        nn.init.constant_(self.classifier[2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        pooled = self.pool(feat)
        logits = self.classifier(pooled)
        return logits


def check_leakage_safety() -> None:
    print("Verifying partition isolation before training...", flush=True)

    with open(TRAIN_MANIFEST_PATH, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(VAL_MANIFEST_PATH, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    with open(TEST_MANIFEST_PATH, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    train_ids = set(r["sample_id"] for r in train_rows)
    val_ids = set(r["sample_id"] for r in val_rows)
    test_ids = set(r["sample_id"] for r in test_rows)

    assert not (train_ids & val_ids), f"FATAL: Train and Val share {len(train_ids & val_ids)} samples!"
    assert not (train_ids & test_ids), f"FATAL: Train and Test share {len(train_ids & test_ids)} samples!"
    assert not (val_ids & test_ids), f"FATAL: Val and Test share {len(val_ids & test_ids)} samples!"

    train_shas = set(r["sha256"] for r in train_rows)
    val_shas = set(r["sha256"] for r in val_rows)
    test_shas = set(r["sha256"] for r in test_rows)

    assert not (train_shas & val_shas), "FATAL: Cryptographic SHA-256 overlap between Train and Val!"
    assert not (train_shas & test_shas), "FATAL: Cryptographic SHA-256 overlap between Train and Test!"
    assert not (val_shas & test_shas), "FATAL: Cryptographic SHA-256 overlap between Val and Test!"

    train_groups = set(r["group_id"] for r in train_rows)
    val_groups = set(r["group_id"] for r in val_rows)
    test_groups = set(r["group_id"] for r in test_rows)

    assert not (train_groups & val_groups), "FATAL: Group leakage between Train and Val!"
    assert not (train_groups & test_groups), "FATAL: Group leakage between Train and Test!"
    assert not (val_groups & test_groups), "FATAL: Group leakage between Val and Test!"

    print("[PASS] Pre-training leakage checks passed: Zero sample overlap, zero SHA-256 overlap, zero group leakage.")


def compute_class_weights(train_dataset: ReceiptAuthenticityDataset) -> torch.Tensor:
    counts = defaultdict(int)
    for lbl in train_dataset._labels:
        counts[lbl] += 1

    n_total = len(train_dataset)
    weights = []
    for c in range(NUM_CLASSES):
        w = n_total / (NUM_CLASSES * max(counts[c], 1))
        weights.append(w)

    w_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize so sum equals NUM_CLASSES
    w_normalized = w_tensor / w_tensor.sum() * NUM_CLASSES

    print("Inverse Class Frequency Weights (strictly computed from training split):")
    for idx, label in enumerate(CLASS_LABELS):
        print(f"  {label:6s}: count={counts[idx]:4d}, weight={w_normalized[idx]:.4f}")
    return w_normalized


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())

    n_total = len(all_targets)
    avg_loss = total_loss / max(n_total, 1)
    acc = accuracy_score(all_targets, all_preds)

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1], zero_division=0
    )
    macro_f1 = float(np.mean(f1))

    # ROC-AUC and PR-AUC
    try:
        roc_auc = float(roc_auc_score(all_targets, all_probs))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(all_targets, all_probs))
    except Exception:
        pr_auc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "real_precision": float(prec[0]),
        "real_recall": float(rec[0]),
        "real_f1": float(f1[0]),
        "edited_precision": float(prec[1]),
        "edited_recall": float(rec[1]),
        "edited_f1": float(f1[1]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "all_targets": all_targets,
        "all_probs": all_probs,
    }


def calibrate_threshold_on_val(val_targets: List[int], val_probs: List[float]) -> Tuple[float, float]:
    """
    Search candidate thresholds [0.05, 0.95] on validation set to maximize validation Macro-F1.
    """
    best_th = 0.50
    best_macro = -1.0

    for th in np.arange(0.05, 0.95, 0.05):
        preds = [1 if p >= th else 0 for p in val_probs]
        _, _, f1, _ = precision_recall_fscore_support(
            val_targets, preds, labels=[0, 1], zero_division=0
        )
        macro = float(np.mean(f1))
        if macro > best_macro:
            best_macro = macro
            best_th = float(th)

    return best_th, best_macro


def train_model() -> None:
    print("=" * 70)
    print("TRUSTTRACE: Training Phase 4 Binary Authenticity Baseline")
    print("=" * 70)

    set_seed(RANDOM_SEED)
    check_leakage_safety()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Forensic-safe realistic augmentations (avoid destructive transformations)
    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(224),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ReceiptAuthenticityDataset(
        TRAIN_MANIFEST_PATH, transform=train_transform, is_train=True
    )
    val_dataset = ReceiptAuthenticityDataset(
        VAL_MANIFEST_PATH, transform=val_transform, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Class imbalance Option A: Inverse class frequency weights
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model architecture
    model = MobileNetV3AuthenticityClassifier(pretrained=True, dropout_rate=DROPOUT_RATE).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: MobileNetV3-Small Binary Authenticity Classifier")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_real_f1": [],
        "val_edited_f1": [],
        "val_edited_recall": [],
        "val_edited_precision": [],
        "val_roc_auc": [],
        "val_pr_auc": [],
        "learning_rate": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    start_time = time.time()

    print("\nStarting Training Loop...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_dur = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_real_f1"].append(val_metrics["real_f1"])
        history["val_edited_f1"].append(val_metrics["edited_f1"])
        history["val_edited_recall"].append(val_metrics["edited_recall"])
        history["val_edited_precision"].append(val_metrics["edited_precision"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["learning_rate"].append(cur_lr)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS:2d} [{epoch_dur:5.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:5.2f}% | "
            f"Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"EDITED Rec: {val_metrics['edited_recall']*100:5.2f}% Prec: {val_metrics['edited_precision']*100:5.2f}% | "
            f"ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        # Save checkpoint based on best validation Macro-F1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val_metrics = {k: v for k, v in val_metrics.items() if k not in ["all_targets", "all_probs"]}

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": best_val_macro_f1,
                    "val_metrics": best_val_metrics,
                    "class_weights": class_weights.cpu().tolist(),
                    "class_labels": CLASS_LABELS,
                    "model_architecture": "MobileNetV3-Small",
                    "input_size": list(IMAGE_SIZE),
                    "random_seed": RANDOM_SEED,
                },
                BEST_MODEL_PATH,
            )
            print(f"  --> Saved new best checkpoint: Macro-F1 = {best_val_macro_f1:.4f} @ Epoch {epoch}")

    total_duration = time.time() - start_time
    print(f"\nTraining complete in {total_duration:.1f}s ({total_duration/60:.2f} min).")
    print(f"Best Validation Macro-F1: {best_val_macro_f1:.4f} at Epoch {best_epoch}")

    # Load best model to calibrate threshold on validation set
    print("\nCalibrating classification threshold on validation set using best checkpoint...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_eval = evaluate(model, val_loader, criterion, device)
    best_threshold, calib_macro_f1 = calibrate_threshold_on_val(
        val_eval["all_targets"], val_eval["all_probs"]
    )
    print(f"Validation Calibrated Threshold: {best_threshold:.2f} (Macro-F1: {calib_macro_f1:.4f})")

    # Update checkpoint with calibrated threshold
    checkpoint["val_calibrated_threshold"] = best_threshold
    checkpoint["val_calibrated_macro_f1"] = calib_macro_f1
    torch.save(checkpoint, BEST_MODEL_PATH)

    # Save training configuration
    config = {
        "model_architecture": "MobileNetV3-Small",
        "input_size": list(IMAGE_SIZE),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "classes": CLASS_LABELS,
        "num_classes": NUM_CLASSES,
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "CosineAnnealingLR",
        "optimizer": "AdamW",
        "dropout_rate": DROPOUT_RATE,
        "class_imbalance_handling": "Option A: Inverse Class Frequency CrossEntropyLoss",
        "class_weights": {lbl: float(w) for lbl, w in zip(CLASS_LABELS, class_weights.cpu().numpy())},
        "device": str(device),
        "total_training_duration_seconds": round(total_duration, 2),
        "best_epoch": best_epoch,
        "best_val_macro_f1": round(best_val_macro_f1, 4),
        "default_threshold": 0.50,
        "val_calibrated_threshold": round(best_threshold, 2),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }

    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {CONFIG_JSON_PATH}")

    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {HISTORY_JSON_PATH}")

    # Plot training curves
    plot_training_curves(history, HISTORY_PNG_PATH)
    print(f"Saved training history plot to {HISTORY_PNG_PATH}")

    print("\n" + "=" * 70)
    print("PHASE 4 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def plot_training_curves(history: Dict[str, List[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curve
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#2563eb")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#dc2626")
    axes[0, 0].set_title("Cross-Entropy Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy & Macro-F1
    axes[0, 1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc (%)", marker="o", color="#2563eb")
    axes[0, 1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Acc (%)", marker="s", color="#16a34a")
    axes[0, 1].plot(epochs, [f * 100 for f in history["val_macro_f1"]], label="Val Macro-F1 (%)", marker="^", color="#9333ea")
    axes[0, 1].set_title("Accuracy & Macro-F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # EDITED Class Recall & Precision
    axes[1, 0].plot(epochs, [r * 100 for r in history["val_edited_recall"]], label="EDITED Recall (%)", marker="o", color="#d97706")
    axes[1, 0].plot(epochs, [p * 100 for p in history["val_edited_precision"]], label="EDITED Precision (%)", marker="s", color="#0891b2")
    axes[1, 0].plot(epochs, [f * 100 for f in history["val_edited_f1"]], label="EDITED F1 (%)", marker="^", color="#dc2626")
    axes[1, 0].set_title("Minority Class (EDITED) Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ROC-AUC & PR-AUC
    axes[1, 1].plot(epochs, [auc * 100 for auc in history["val_roc_auc"]], label="Val ROC-AUC (%)", marker="o", color="#4f46e5")
    axes[1, 1].plot(epochs, [pr * 100 for pr in history["val_pr_auc"]], label="Val PR-AUC (%)", marker="s", color="#059669")
    axes[1, 1].set_title("Discrimination Power (AUC)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("AUC (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    train_model()
