#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 8 Compact CNN Baseline Training.

Trains a lightweight 3-stage convolutional neural network on the Phase 8 candidate-aware patch dataset:
- Evaluates whether Phase 8 improvements stem from training data domain calibration vs transformer architecture
- Lightweight footprint (~94,000 parameters)
- Class-weighted CrossEntropyLoss to address positive/negative candidate distribution
- Early stopping monitored on Validation Macro-F1
- Saves checkpoint to: models/phase8_patch_cnn_best.pt
"""

import sys
import os
import csv
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"

TRAIN_MANIFEST = MANIFESTS_DIR / "phase8_patch_train.csv"
VAL_MANIFEST = MANIFESTS_DIR / "phase8_patch_val.csv"
CHECKPOINT_PATH = MODELS_DIR / "phase8_patch_cnn_best.pt"

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 7
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Phase8PatchDataset(Dataset):
    def __init__(self, manifest_path: Path, transform=None):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.transform = transform

        # Pre-cache images in memory for rapid CPU training
        self.cached_images = []
        for r in self.rows:
            p = Path(r["crop_path"])
            with Image.open(p) as img:
                self.cached_images.append(img.convert("RGB"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        target = int(self.rows[idx]["binary_label"])
        if self.transform:
            img = self.transform(img)
        return img, target, self.rows[idx]


class Phase8CompactCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16x16

            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def train_cnn() -> Dict[str, Any]:
    print("=" * 70)
    print("TRUSTTRACE: Training Phase 8 Compact CNN Baseline")
    print("=" * 70)

    set_seed(RANDOM_SEED)
    device = torch.device("cpu")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading train and validation datasets into memory...")
    train_dataset = Phase8PatchDataset(TRAIN_MANIFEST, transform=transform_train)
    val_dataset = Phase8PatchDataset(VAL_MANIFEST, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Class balance weighting
    targets_all = [int(r["binary_label"]) for r in train_dataset.rows]
    n_pos = sum(targets_all)
    n_neg = len(targets_all) - n_pos
    pos_weight = n_neg / float(max(n_pos, 1))
    weights = torch.tensor([1.0, pos_weight], device=device)
    print(f"Training on {len(train_dataset)} patches ({n_pos} Forged, {n_neg} Authentic, pos_weight={pos_weight:.2f})")

    model = Phase8CompactCNN(in_channels=3, num_classes=2).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Compact CNN Parameter Count: {param_count:,}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_stats = {}

    t_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        # Validation evaluation
        model.eval()
        val_targets, val_preds, val_probs = [], [], []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().tolist()
                preds = torch.argmax(outputs, dim=1).cpu().tolist()
                val_targets.extend(labels.tolist())
                val_preds.extend(preds)
                val_probs.extend(probs)

        val_acc = accuracy_score(val_targets, val_preds)
        _, _, f1, _ = precision_recall_fscore_support(val_targets, val_preds, labels=[0, 1], zero_division=0)
        macro_f1 = float(np.mean(f1))
        val_roc = roc_auc_score(val_targets, val_probs)

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Macro-F1: {macro_f1:.4f} | Forged Recall: {f1[1]*100:.2f}% | ROC-AUC: {val_roc:.4f}")

        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_epoch = epoch
            best_stats = {
                "epoch": epoch,
                "val_acc": round(val_acc, 4),
                "macro_f1": round(macro_f1, 4),
                "forged_f1": round(float(f1[1]), 4),
                "roc_auc": round(float(val_roc), 4),
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "macro_f1": macro_f1,
                "param_count": param_count,
            }, CHECKPOINT_PATH)

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s. Best Epoch: {best_epoch} (Macro-F1: {best_val_macro_f1:.4f})")
    print(f"Saved best model checkpoint to: {CHECKPOINT_PATH.name}")

    best_stats["training_time_seconds"] = round(total_time, 2)
    best_stats["param_count"] = param_count
    return best_stats


if __name__ == "__main__":
    train_cnn()
