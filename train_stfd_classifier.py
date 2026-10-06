#!/usr/bin/env python3
"""
TRUSTTRACE: STFD 5-Class Manipulation Classifier Training Script.

Trains a 5-class manipulation-type classifier on the clustered STFD dataset:
- Classes: COPY_MOVE, SPLICING, REMOVAL, INSERTION, REPLACEMENT
- Architecture: Pretrained MobileNetV3-Small backbone with forensic classification head
- Dataset: Clustered STFD Train (2,752) and Validation (590) manifests
- Strictly avoids accessing the Test set during training
- Saves best checkpoint based on validation Macro-F1
- Saves training history and loss/metric curves
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

WORKSPACE_DIR = Path(__file__).resolve().parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"
REPORTS_DIR = WORKSPACE_DIR / "reports"

TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_val.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"

BEST_MODEL_PATH = MODELS_DIR / "stfd_manipulation_best.pt"
HISTORY_JSON_PATH = REPORTS_DIR / "stfd_manipulation_training_history.json"
HISTORY_PNG_PATH = REPORTS_DIR / "stfd_manipulation_training_history.png"

# Hyperparameters & Configurations
RANDOM_SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Official 5 STFD Manipulation Classes
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
    # Determinism flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class STFDClusteredDataset(Dataset):
    """
    STFD Clustered Dataset loader for screenshot manipulation classification.
    Reads images dynamically from the original STFD paths without modifying disk files.
    Caches pre-resized PIL images in RAM for fast multi-epoch CPU training.
    """
    def __init__(self, manifest_path: Path, transform=None, is_train: bool = False):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.transform = transform
        self.is_train = is_train
        self._cache: Dict[int, Image.Image] = {}

        # Validate rows
        for idx, r in enumerate(self.rows):
            if r["manipulation_type"] not in CLASS_TO_IDX:
                raise ValueError(f"Row {idx}: unknown manipulation_type '{r['manipulation_type']}'")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self._cache:
            img = self._cache[idx]
        else:
            row = self.rows[idx]
            img_path = Path(row["image_path"])
            if not img_path.is_file():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img = Image.open(img_path).convert("RGB")
            # Downsample to target resolution once and cache
            img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            self._cache[idx] = img

        label = CLASS_TO_IDX[self.rows[idx]["manipulation_type"]]

        if self.transform:
            tensor_img = self.transform(img)
        else:
            tensor_img = transforms.ToTensor()(img)

        return tensor_img, label


class STFDManipulationClassifier(nn.Module):
    """
    MobileNetV3-Small vision backbone with custom forensic projection and classification head.
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
    """Strict pre-training leakage verification."""
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

    train_clusters = set(r["candidate_template_cluster_id"] for r in train_rows)
    val_clusters = set(r["candidate_template_cluster_id"] for r in val_rows)
    test_clusters = set(r["candidate_template_cluster_id"] for r in test_rows)

    assert not (train_clusters & val_clusters), "FATAL: Cluster leakage between Train and Val!"
    assert not (train_clusters & test_clusters), "FATAL: Cluster leakage between Train and Test!"
    assert not (val_clusters & test_clusters), "FATAL: Cluster leakage between Val and Test!"

    print("[PASS] Pre-training leakage checks passed: Zero sample overlap, zero cluster overlap.")


def compute_class_weights(train_dataset: STFDClusteredDataset) -> torch.Tensor:
    """Calculates inverse class frequency weights strictly from training set."""
    counts = defaultdict(int)
    for r in train_dataset.rows:
        counts[CLASS_TO_IDX[r["manipulation_type"]]] += 1

    n_total = len(train_dataset)
    weights = []
    for c in range(NUM_CLASSES):
        w = n_total / (NUM_CLASSES * max(counts[c], 1))
        weights.append(w)

    print("Training manipulation class counts and inverse frequency weights:")
    for c, label in enumerate(CLASS_LABELS):
        print(f"  {label:<14}: count = {counts[c]:>4}, weight = {weights[c]:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)
            loss = criterion(logits, targets)

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    _, _, f1_per_class, _ = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )
    per_class_f1 = {CLASS_LABELS[i]: float(f1_per_class[i]) for i in range(NUM_CLASSES)}

    return avg_loss, acc, f1_macro, per_class_f1, np.array(all_preds), np.array(all_targets)


def plot_and_save_history(history: Dict[str, List[float]], save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_title("STFD Manipulation Classifier - Loss History", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Macro-F1 plot
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Accuracy", color="#2ca02c")
    ax2.plot(epochs, history["val_acc"], "^-", label="Val Accuracy", color="#17becf")
    ax2.plot(epochs, history["val_macro_f1"], "s-", label="Val Macro-F1", color="#d62728", linewidth=2)
    ax2.set_title("STFD Manipulation Classifier - Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Training history visualization saved to {save_path.name}")


def main():
    print("=" * 70)
    print("TRUSTTRACE: STFD 5-CLASS MANIPULATION CLASSIFIER TRAINING")
    print("=" * 70)

    # 1. Setup directories and environment
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 2. Leakage verification
    check_leakage_safety()

    # 3. Data transforms (Gentle, realistic forensics-preserving augmentation for training)
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Create Datasets and DataLoaders
    print("\nInitializing datasets...", flush=True)
    train_dataset = STFDClusteredDataset(TRAIN_MANIFEST_PATH, transform=train_transform, is_train=True)
    val_dataset = STFDClusteredDataset(VAL_MANIFEST_PATH, transform=val_transform, is_train=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # NOTE: Test manifest is intentionally NOT loaded during training
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 5. Model, Loss, Optimizer, Scheduler
    print("\nConstructing MobileNetV3-Small model...", flush=True)
    model = STFDManipulationClassifier(
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout_rate=0.3,
        feature_dim=256
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 6. Training Loop
    print("\nStarting training loop...", flush=True)
    best_val_macro_f1 = -1.0
    best_epoch = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rate": [],
        "epoch_time_seconds": [],
    }

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            loss = criterion(logits, targets)
            loss.backward()

            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate on validation set
        val_loss, val_acc, val_f1, per_class_f1, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["learning_rate"].append(current_lr)
        history["epoch_time_seconds"].append(epoch_time)

        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            best_epoch = epoch
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "architecture": "mobilenet_v3_small",
                "class_labels": CLASS_LABELS,
                "num_classes": NUM_CLASSES,
                "image_size": IMAGE_SIZE,
                "seed": RANDOM_SEED,
            }, BEST_MODEL_PATH)
            star = " [*BEST CHECKPOINT SAVED]"
        else:
            star = ""

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({epoch_time:5.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:5.2f}%, Macro-F1: {val_f1:.4f}{star}",
            flush=True
        )
        print(f"  Val Per-Class F1: { {k: round(v, 3) for k, v in per_class_f1.items()} }", flush=True)

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Macro-F1: {best_val_macro_f1:.4f} (Achieved at Epoch {best_epoch})")
    print(f"Saved best model checkpoint to: {BEST_MODEL_PATH}")

    # 7. Save history JSON & plot
    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_architecture": "mobilenet_v3_small",
            "epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "total_training_time_seconds": total_training_time,
            "history": history,
            "class_labels": CLASS_LABELS,
            "random_seed": RANDOM_SEED,
            "device": str(device),
        }, f, indent=2)
    print(f"Training history saved to {HISTORY_JSON_PATH.name}")

    plot_and_save_history(history, HISTORY_PNG_PATH)


if __name__ == "__main__":
    main()
