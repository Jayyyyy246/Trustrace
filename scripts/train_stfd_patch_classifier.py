#!/usr/bin/env python3
"""
TRUSTTRACE: STFD High-Resolution Patch Classifier Training Script (Phase 3A).

Trains a 5-class manipulation classifier directly on high-resolution localized patches:
- Architecture: Pretrained MobileNetV3-Small backbone with forensic projection & classification head
- Input Resolution: 224x224 RGB localized patch crops
- Target Classes: COPY_MOVE, SPLICING, REMOVAL, INSERTION, REPLACEMENT
- Dataset: Clustered STFD Train (2,752) and Validation (590) patch manifests
- Controlled Comparison: Same backbone, optimizer, learning rate, and epochs as Phase 1
- Saves best checkpoint based on validation Macro-F1
- Outputs:
  * models/stfd_patch_classifier_best.pt
  * reports/stfd_patch_forensics_config.json
  * reports/stfd_patch_forensics_training_history.json
  * reports/stfd_patch_forensics_training_history.png
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_patch_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_patch_val.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_patch_test.csv"

BEST_MODEL_PATH = MODELS_DIR / "stfd_patch_classifier_best.pt"
CONFIG_JSON_PATH = REPORTS_DIR / "stfd_patch_forensics_config.json"
HISTORY_JSON_PATH = REPORTS_DIR / "stfd_patch_forensics_training_history.json"
HISTORY_PNG_PATH = REPORTS_DIR / "stfd_patch_forensics_training_history.png"

# Hyperparameters
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

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


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class STFDPatchDataset(Dataset):
    """
    STFD Patch Dataset for high-resolution localized crop classification.
    Reads pre-extracted letterboxed 224x224 crops and caches in RAM.
    """
    def __init__(self, manifest_path: Path, transform=None, is_train: bool = False):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.transform = transform
        self.is_train = is_train
        self._cache: Dict[int, Image.Image] = {}

        for idx, r in enumerate(self.rows):
            manip = r.get("manipulation_class") or r.get("manipulation_type")
            if manip not in CLASS_TO_IDX:
                raise ValueError(f"Row {idx}: unknown manipulation class '{manip}'")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self._cache:
            img = self._cache[idx]
        else:
            row = self.rows[idx]
            crop_path = Path(row["crop_path"])
            if not crop_path.is_file():
                raise FileNotFoundError(f"Crop file not found: {crop_path}")
            img = Image.open(crop_path).convert("RGB")
            self._cache[idx] = img

        row = self.rows[idx]
        manip = row.get("manipulation_class") or row.get("manipulation_type")
        label = CLASS_TO_IDX[manip]

        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)

        return tensor_img, label


class STFDPatchManipulationClassifier(nn.Module):
    """
    MobileNetV3-Small vision backbone with custom forensic projection and classification head.
    Identical capacity to Phase 1 baseline (~1.07M parameters) to strictly isolate localization effect.
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        feature_dim: int = 256,
    ):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
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

        # Initialize projection head
        nn.init.kaiming_normal_(self.projection[2].weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.projection[2].bias, 0.0)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)
        pooled = self.pool(feat_map)
        embedding = self.projection(pooled)
        logits = self.classifier(embedding)
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

    train_clusters = set(r["candidate_cluster_id"] for r in train_rows)
    val_clusters = set(r["candidate_cluster_id"] for r in val_rows)
    test_clusters = set(r["candidate_cluster_id"] for r in test_rows)

    assert not (train_clusters & val_clusters), "FATAL: Cluster leakage between Train and Val!"
    assert not (train_clusters & test_clusters), "FATAL: Cluster leakage between Train and Test!"
    assert not (val_clusters & test_clusters), "FATAL: Cluster leakage between Val and Test!"

    print("[PASS] Pre-training leakage checks passed: Zero sample overlap, zero cluster overlap under clustering criteria.")


def compute_class_weights(train_dataset: STFDPatchDataset) -> torch.Tensor:
    counts = defaultdict(int)
    for r in train_dataset.rows:
        manip = r.get("manipulation_class") or r.get("manipulation_type")
        counts[CLASS_TO_IDX[manip]] += 1

    n_total = len(train_dataset)
    weights = []
    for c in range(NUM_CLASSES):
        w = n_total / (NUM_CLASSES * max(counts[c], 1))
        weights.append(w)

    w_tensor = torch.tensor(weights, dtype=torch.float32)
    w_normalized = w_tensor / w_tensor.sum() * NUM_CLASSES
    print("Inverse Class Frequency Weights (strictly from training split):")
    for idx, label in enumerate(CLASS_LABELS):
        print(f"  {label:12s}: count={counts[idx]:4d}, weight={w_normalized[idx]:.4f}")
    return w_normalized


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    total = len(all_targets)
    val_loss = total_loss / total

    acc = accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    return val_loss, acc, macro_p, macro_r, macro_f1


def plot_training_history(history: Dict[str, List[float]], save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curves
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2563EB", lw=2)
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#DC2626", lw=2)
    axes[0].set_title("Phase 3A: Training & Validation Loss (Cross-Entropy)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Metric Curves
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], "^-", label="Val Accuracy (%)", color="#059669", lw=2)
    axes[1].plot(epochs, [f * 100 for f in history["val_macro_f1"]], "d-", label="Val Macro-F1 (%)", color="#7C3AED", lw=2)
    axes[1].set_title("Phase 3A: Validation Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training history curves to: {save_path}")


def train_patch_classifier() -> None:
    print("=" * 70)
    print("TRUSTTRACE: Training Phase 3A STFD High-Resolution Patch Classifier")
    print("=" * 70)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # Leakage safety check
    check_leakage_safety()

    # Image Normalization
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("\nLoading datasets and initializing memory cache...")
    t0 = time.time()
    train_dataset = STFDPatchDataset(TRAIN_MANIFEST_PATH, transform=train_transform, is_train=True)
    val_dataset = STFDPatchDataset(VAL_MANIFEST_PATH, transform=eval_transform, is_train=False)
    print(f"Train Dataset: {len(train_dataset)} samples")
    print(f"Val Dataset:   {len(val_dataset)} samples")
    print(f"Manifests loaded in {time.time() - t0:.2f}s")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\nInitializing MobileNetV3-Small Patch Model...")
    model = STFDPatchManipulationClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout_rate=0.3,
        feature_dim=256,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "val_macro_f1": [],
        "lr": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = -1

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nStarting training loop...")
    print("-" * 70)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<11} {'Val Loss':<10} {'Val Acc':<10} {'Val F1':<10} {'Time':<8}")
    print("-" * 70)

    train_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate_validation(model, val_loader, criterion, device)

        scheduler.step()
        epoch_time = time.time() - t_epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_macro_precision"].append(round(val_p, 4))
        history["val_macro_recall"].append(round(val_r, 4))
        history["val_macro_f1"].append(round(val_f1, 4))
        history["lr"].append(current_lr)

        print(
            f"{epoch:<6d} {train_loss:<12.4f} {train_acc * 100:<10.2f}% "
            f"{val_loss:<10.4f} {val_acc * 100:<9.2f}% {val_f1 * 100:<9.2f}% {epoch_time:<6.1f}s"
        )

        if val_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": best_val_macro_f1,
                "best_val_acc": val_acc,
                "class_labels": CLASS_LABELS,
                "class_to_idx": CLASS_TO_IDX,
                "model_architecture": "MobileNetV3-Small-Patch-Forensics",
                "input_size": IMAGE_SIZE,
                "feature_dim": 256,
                "dropout_rate": 0.3,
                "random_seed": RANDOM_SEED,
            }
            torch.save(checkpoint, BEST_MODEL_PATH)

    total_training_time = time.time() - train_start_time
    print("-" * 70)
    print(f"Training completed in {total_training_time:.1f}s ({total_training_time / 60:.2f} mins).")
    print(f"Best Validation Macro-F1: {best_val_macro_f1 * 100:.2f}% (Epoch {best_epoch})")
    print(f"Checkpoint saved to: {BEST_MODEL_PATH}")

    # Save training history JSON
    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {HISTORY_JSON_PATH}")

    # Plot curves
    plot_training_history(history, HISTORY_PNG_PATH)

    # Save experiment config JSON
    config = {
        "experiment_name": "stfd_oracle_patch_manipulation_classification",
        "phase": "PHASE_3A_ORACLE_PATCH_CLASSIFIER",
        "random_seed": RANDOM_SEED,
        "dataset": {
            "name": "Smartphone Tampering Forensic Dataset (STFD - ICASSP 2023) Oracle High-Resolution Patches",
            "isolation_method": "PHASH_LE4_CONNECTED_COMPONENTS_TEMPLATE_CLUSTERING",
            "train_manifest": str(TRAIN_MANIFEST_PATH),
            "val_manifest": str(VAL_MANIFEST_PATH),
            "test_manifest": str(TEST_MANIFEST_PATH),
            "samples": {
                "train": len(train_dataset),
                "val": len(val_dataset),
                "test": 590,
                "total": 3932,
            },
            "classes": CLASS_LABELS,
            "num_classes": NUM_CLASSES,
            "patch_extraction": {
                "strategy": "oracle_mask_bounding_box",
                "padding_ratio": 0.25,
                "min_crop_size": 64,
                "letterbox_aspect_ratio_preservation": True,
                "target_crop_size": [224, 224],
            },
        },
        "model": {
            "backbone": "mobilenet_v3_small",
            "pretrained": True,
            "pretrained_weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "feature_dim": 256,
            "dropout_rate": 0.3,
            "projection_head": "Flatten -> Dropout(0.3) -> Linear(576, 256) -> BatchNorm1d(256) -> SiLU -> Linear(256, 5)",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        },
        "hyperparameters": {
            "image_size": [224, 224],
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR (T_max=6, eta_min=1e-6)",
            "loss_function": "CrossEntropyLoss with Inverse Class Frequency Weighting",
            "gradient_clipping_max_norm": 5.0,
            "model_selection_metric": "Validation Macro-F1",
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "training_time_seconds": total_training_time,
        },
        "hardware": {
            "device": str(device),
            "torch_version": torch.__version__,
        },
    }

    with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved experiment configuration to: {CONFIG_JSON_PATH}")


if __name__ == "__main__":
    train_patch_classifier()
