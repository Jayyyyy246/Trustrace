#!/usr/bin/env python3
"""
TRUSTTRACE: Phase 6 Document-Native Patch Embedding Transformer (Doc-PatchFormer).

Trains a document-native patch transformer on native-resolution text & entity crops:
- Architecture:
  Native Patch (128x128 RGB)
  -> Convolutional Patch Tokenizer (16x16 receptive field -> 64 spatial tokens)
  -> Prepend [CLS] token (65 tokens total)
  -> Learnable 1D Positional Embeddings
  -> 2-Layer Transformer Encoder (d_model=128, nhead=4, dim_feedforward=256, gelu)
  -> LayerNorm -> Dropout(0.1) -> Linear(128, 2) [AUTHENTIC_REGION vs FORGED_REGION]
- Also trains Baseline B: Patch-CNN (3-stage Conv-BatchNorm-ReLU-Pool without transformer)
  for a controlled architectural ablation.
- Enforces strict group-aware isolation:
  patches from parent document groups strictly remain in their respective splits.
- Checkpoints saved to:
  * models/doc_patchformer_best.pt
  * models/patch_cnn_baseline.pt
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
REPORTS_DIR = WORKSPACE_DIR / "reports"

PATCH_TRAIN_PATH = MANIFESTS_DIR / "phase6_patch_train.csv"
PATCH_VAL_PATH = MANIFESTS_DIR / "phase6_patch_val.csv"
PATCH_TEST_PATH = MANIFESTS_DIR / "phase6_patch_test.csv"

DOC_PATCHFORMER_BEST_PATH = MODELS_DIR / "doc_patchformer_best.pt"
PATCH_CNN_BEST_PATH = MODELS_DIR / "patch_cnn_baseline.pt"

RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
TARGET_SIZE = (128, 128)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DocumentPatchDataset(Dataset):
    """
    Loads native-resolution document patches cached in RAM to maximize CPU training throughput.
    """
    def __init__(self, manifest_path: Path, transform=None):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.transform = transform
        self._cached_images: List[Image.Image] = []
        self._labels: List[int] = []

        print(f"Preloading {len(self.rows)} patches from {manifest_path.name} into RAM...", flush=True)
        for r in self.rows:
            self._labels.append(int(r["binary_label"]))
            p = Path(r["patch_path"])
            with Image.open(p) as img:
                self._cached_images.append(img.convert("RGB"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        img = self._cached_images[idx]
        target = self._labels[idx]
        row = self.rows[idx]

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.functional.to_tensor(img)

        return tensor, target, row


class DocPatchFormer(nn.Module):
    """
    Document-Native Patch Embedding Transformer:
    - 2-stage CNN tokenizer producing 64 spatial token embeddings (d=128)
    - Learnable [CLS] token and 1D positional embeddings
    - 2-layer TransformerEncoder (4 heads, d_ff=256, GELU, dropout=0.1)
    - Linear binary classification head on [CLS] token
    """
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Tokenizer: 128x128 -> 32x32 -> 8x8 (64 tokens)
        self.tokenizer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),  # 32x32
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=4, stride=4),    # 8x8
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        num_tokens = 8 * 8  # 64 spatial tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.head[1].weight)
        nn.init.constant_(self.head[1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        feat = self.tokenizer(x)  # (b, 128, 8, 8)
        tokens = feat.flatten(2).transpose(1, 2)  # (b, 64, 128)

        cls_tokens = self.cls_token.expand(b, -1, -1)  # (b, 1, 128)
        x_tok = torch.cat((cls_tokens, tokens), dim=1)  # (b, 65, 128)
        x_tok = self.pos_drop(x_tok + self.pos_embed)

        encoded = self.transformer(x_tok)  # (b, 65, 128)
        cls_out = self.norm(encoded[:, 0])  # (b, 128)
        logits = self.head(cls_out)  # (b, 2)
        return logits


class PatchCNNBaseline(nn.Module):
    """
    Baseline B: Lightweight Convolutional Patch Classifier without Transformer:
    3 Conv-BatchNorm-ReLU-Pool stages -> AdaptiveAvgPool2d((1, 1)) -> Linear(128, 2)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        nn.init.xavier_uniform_(self.classifier[2].weight)
        nn.init.constant_(self.classifier[2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


def evaluate_patch_model(
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
        for inputs, targets, _ in dataloader:
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

    try:
        roc_auc = float(roc_auc_score(all_targets, all_probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(all_targets, all_probs))
    except Exception:
        pr_auc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "forged_precision": float(prec[1]),
        "forged_recall": float(rec[1]),
        "forged_f1": float(f1[1]),
        "authentic_precision": float(prec[0]),
        "authentic_recall": float(rec[0]),
        "authentic_f1": float(f1[0]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def train_single_architecture(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    save_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\n" + "-" * 70)
    print(f"Training Architecture: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = {}
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        ep_start = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets, _ in train_loader:
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
        ep_dur = time.time() - ep_start
        train_acc = train_correct / max(train_total, 1)

        val_metrics = evaluate_patch_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:2d}/{NUM_EPOCHS:2d} [{ep_dur:4.1f}s] | "
            f"Train Acc: {train_acc*100:5.2f}% | "
            f"Val Acc: {val_metrics['accuracy']*100:5.2f}% | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"FORGED Rec: {val_metrics['forged_recall']*100:5.2f}% Prec: {val_metrics['forged_precision']*100:5.2f}% | "
            f"ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics

            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_metrics": val_metrics,
                "random_seed": RANDOM_SEED,
            }, save_path)
            print(f"  --> Saved new best checkpoint @ Epoch {epoch} (Val Macro-F1: {best_val_macro_f1:.4f})")

    dur = time.time() - start_time
    print(f"Training {model_name} completed in {dur:.1f}s ({dur/60:.2f} min). Best Macro-F1: {best_val_macro_f1:.4f} @ Epoch {best_epoch}")
    return {"name": model_name, "best_epoch": best_epoch, "val_metrics": best_metrics, "duration": dur}


def main():
    print("=" * 70)
    print("TRUSTTRACE: Phase 6 Patch Model Training Pipeline")
    print("=" * 70)

    set_seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Conservative, forensic-safe augmentations
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DocumentPatchDataset(PATCH_TRAIN_PATH, transform=train_transform)
    val_dataset = DocumentPatchDataset(PATCH_VAL_PATH, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Compute class weights strictly on train
    n_forged = sum(1 for y in train_dataset._labels if y == 1)
    n_auth = sum(1 for y in train_dataset._labels if y == 0)
    n_tot = len(train_dataset)
    w_auth = n_tot / (2.0 * max(n_auth, 1))
    w_forged = n_tot / (2.0 * max(n_forged, 1))
    class_weights = torch.tensor([w_auth, w_forged], dtype=torch.float32).to(device)
    print(f"Inverse Class Frequency Weights (Train split): Authentic={w_auth:.4f}, Forged={w_forged:.4f}")

    # 1. Train Doc-PatchFormer
    patchformer = DocPatchFormer(embed_dim=128, num_heads=4, num_layers=2, mlp_dim=256, dropout=0.1).to(device)
    train_single_architecture(
        "Doc-PatchFormer",
        patchformer,
        train_loader,
        val_loader,
        class_weights,
        DOC_PATCHFORMER_BEST_PATH,
        device,
    )

    # 2. Train Baseline B: Patch-CNN
    patch_cnn = PatchCNNBaseline(in_channels=3, num_classes=2, dropout=0.1).to(device)
    train_single_architecture(
        "Patch-CNN Baseline",
        patch_cnn,
        train_loader,
        val_loader,
        class_weights,
        PATCH_CNN_BEST_PATH,
        device,
    )

    print("\n" + "=" * 70)
    print("PHASE 6 MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
