#!/usr/bin/env python3
"""
TRUSTTRACE: STFD Tamper Localization Model Training Script (Phase 2).

Trains a supervised image tamper-localization U-Net using official STFD binary masks:
- Input: RGB smartphone screenshot (256x256)
- Output: Single-channel binary tampering probability map (256x256)
- Loss: Combined BCEWithLogits (pos_weight=5.0) + Soft Dice Loss
- Partitions: Approved clustered STFD Train (2,752) and Validation (590) manifests
- Model selection: Best Validation Foreground Dice score
- Strict test set isolation: Test set is NEVER touched during training
- Checkpoint: models/stfd_localization_best.pt
"""

import sys
import os
import csv
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
MODELS_DIR = WORKSPACE_DIR / "models"
REPORTS_DIR = WORKSPACE_DIR / "reports"

TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_train.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_val.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"

BEST_MODEL_PATH = MODELS_DIR / "stfd_localization_best.pt"
HISTORY_JSON_PATH = REPORTS_DIR / "stfd_localization_training_history.json"
HISTORY_PNG_PATH = REPORTS_DIR / "stfd_localization_training_history.png"

# Hyperparameters & Configurations
RANDOM_SEED = 42
IMAGE_SIZE = (256, 256)  # 256x256 provides 30.6% more pixel density than 224x224
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 5.0  # Foreground weighting for severe background imbalance


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DoubleConv(nn.Module):
    """Standard double 3x3 convolution block with BatchNorm and ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ForensicUNet(nn.Module):
    """
    Lightweight, multi-scale U-Net for screenshot tamper localization.
    Preserves fine-grained character edge boundaries via multi-scale skip connections.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 16):
        super().__init__()
        bf = base_filters
        self.inc = DoubleConv(in_channels, bf)                                  # 16
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf * 2))     # 32
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 2, bf * 4)) # 64
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 4, bf * 8)) # 128

        self.up1 = nn.ConvTranspose2d(bf * 8, bf * 4, 2, stride=2)
        self.conv_up1 = DoubleConv(bf * 8, bf * 4)

        self.up2 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.conv_up2 = DoubleConv(bf * 4, bf * 2)

        self.up3 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.conv_up3 = DoubleConv(bf * 2, bf)

        self.outc = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        x = self.conv_up1(u1)

        u2 = self.up2(x)
        u2 = torch.cat([u2, x2], dim=1)
        x = self.conv_up2(u2)

        u3 = self.up3(x)
        u3 = torch.cat([u3, x1], dim=1)
        x = self.conv_up3(u3)

        logits = self.outc(x)
        return logits


class CombinedBceDiceLoss(nn.Module):
    """
    Combined weighted BCEWithLogits and Soft Dice Loss.
    Directly optimizes foreground boundary overlap while maintaining smooth gradients.
    """
    def __init__(self, pos_weight: float = POS_WEIGHT, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        pw = torch.tensor([pos_weight], dtype=torch.float32)
        self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    def to(self, device):
        super().to(device)
        self.bce_fn.pos_weight = self.bce_fn.pos_weight.to(device)
        return self

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # BCE component
        bce_loss = self.bce_fn(logits, targets)

        # Soft Dice component
        probs = torch.sigmoid(logits)
        eps = 1e-6
        intersection = (probs * targets).sum(dim=(-2, -1))
        cardinality = (probs.pow(2) + targets.pow(2)).sum(dim=(-2, -1))
        dice_score = (2.0 * intersection + eps) / (cardinality + eps)
        dice_loss = (1.0 - dice_score).mean()

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return total_loss, bce_loss, dice_loss


class STFDLocalizationDataset(Dataset):
    """
    Dataset for supervised tamper localization.
    Loads RGB screenshots and corresponding binary ground-truth masks.
    Uses nearest-neighbor interpolation for masks (never bilinear).
    Applies identical geometric transformations to image and mask.
    Caches resized PIL instances in RAM for fast CPU execution.
    """
    def __init__(self, manifest_path: Path, is_train: bool = False):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.is_train = is_train
        self._cache_img: Dict[int, Image.Image] = {}
        self._cache_mask: Dict[int, Image.Image] = {}

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx in self._cache_img:
            img = self._cache_img[idx]
            mask = self._cache_mask[idx]
        else:
            r = self.rows[idx]
            img_path = Path(r["image_path"])
            mask_path = Path(r["mask_path"])

            if not img_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"Missing file for sample {r['sample_id']}")

            img_raw = Image.open(img_path).convert("RGB")
            mask_raw = Image.open(mask_path).convert("L")

            # Verify original dimensions match
            if img_raw.size != mask_raw.size:
                raise ValueError(f"Dimension mismatch for {r['sample_id']}: img {img_raw.size} vs mask {mask_raw.size}")

            # Resize: bilinear for image, STRICT NEAREST for mask
            img = img_raw.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            mask = mask_raw.resize(IMAGE_SIZE, Image.Resampling.NEAREST)

            self._cache_img[idx] = img
            self._cache_mask[idx] = mask

        # Synchronized geometric augmentation for training (identical horizontal flip)
        if self.is_train and random.random() > 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Convert to tensor
        img_t = self.to_tensor(img)
        img_t = self.normalize(img_t)

        # Mask binary conversion (strictly 0.0 or 1.0)
        mask_np = np.array(mask, dtype=np.uint8)
        mask_binary = (mask_np > 128).astype(np.float32)
        mask_t = torch.from_numpy(mask_binary).unsqueeze(0)  # (1, H, W)

        return img_t, mask_t


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

    assert not (train_ids & val_ids), "FATAL: Train/Val sample overlap!"
    assert not (train_ids & test_ids), "FATAL: Train/Test sample overlap!"
    assert not (val_ids & test_ids), "FATAL: Val/Test sample overlap!"

    train_clusters = set(r["candidate_template_cluster_id"] for r in train_rows)
    val_clusters = set(r["candidate_template_cluster_id"] for r in val_rows)
    test_clusters = set(r["candidate_template_cluster_id"] for r in test_rows)

    assert not (train_clusters & val_clusters), "FATAL: Train/Val cluster overlap!"
    assert not (train_clusters & test_clusters), "FATAL: Train/Test cluster overlap!"
    assert not (val_clusters & test_clusters), "FATAL: Val/Test cluster overlap!"

    train_shas = set(r["image_sha256"] for r in train_rows)
    val_shas = set(r["image_sha256"] for r in val_rows)
    test_shas = set(r["image_sha256"] for r in test_rows)

    assert not (train_shas & val_shas), "FATAL: Train/Val SHA256 overlap!"
    assert not (train_shas & test_shas), "FATAL: Train/Test SHA256 overlap!"
    assert not (val_shas & test_shas), "FATAL: Val/Test SHA256 overlap!"

    print("[PASS] Pre-training leakage validation passed: Zero sample, SHA-256, or cluster overlap.")


def compute_localization_metrics(
    all_preds_binary: np.ndarray,
    all_targets_binary: np.ndarray
) -> Dict[str, float]:
    """
    Computes exhaustive binary segmentation metrics:
    - Pixel Accuracy
    - Foreground Precision, Recall, Dice, IoU
    - Background IoU
    - Mean IoU (mIoU)
    - All-Background baseline metrics
    """
    tp = float(np.sum((all_preds_binary == 1) & (all_targets_binary == 1)))
    fp = float(np.sum((all_preds_binary == 1) & (all_targets_binary == 0)))
    fn = float(np.sum((all_preds_binary == 0) & (all_targets_binary == 1)))
    tn = float(np.sum((all_preds_binary == 0) & (all_targets_binary == 0)))

    total_pixels = tp + fp + fn + tn
    eps = 1e-7

    pixel_acc = (tp + tn) / max(total_pixels, 1.0)
    fg_prec = tp / max(tp + fp, eps)
    fg_rec = tp / max(tp + fn, eps)
    fg_dice = (2.0 * tp) / max(2.0 * tp + fp + fn, eps)
    fg_iou = tp / max(tp + fp + fn, eps)

    bg_iou = tn / max(tn + fp + fn, eps)
    m_iou = (fg_iou + bg_iou) / 2.0

    # Baseline: predicting all zeros (background everywhere)
    bl_tn = tn + fp
    bl_fn = tp + fn
    bl_pixel_acc = bl_tn / max(total_pixels, 1.0)
    bl_bg_iou = bl_tn / max(bl_tn + bl_fn, eps)
    bl_m_iou = (0.0 + bl_bg_iou) / 2.0

    return {
        "pixel_accuracy": pixel_acc,
        "foreground_precision": fg_prec,
        "foreground_recall": fg_rec,
        "foreground_dice": fg_dice,
        "foreground_iou": fg_iou,
        "background_iou": bg_iou,
        "mean_iou": m_iou,
        "all_background_pixel_acc": bl_pixel_acc,
        "all_background_m_iou": bl_m_iou,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CombinedBceDiceLoss,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0

    all_preds_list = []
    all_targets_list = []

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)
            loss, _, _ = criterion(logits, targets)
            total_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            preds_bin = (probs > threshold).cpu().numpy().astype(np.uint8)
            targets_np = targets.cpu().numpy().astype(np.uint8)

            all_preds_list.append(preds_bin)
            all_targets_list.append(targets_np)

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)

    metrics = compute_localization_metrics(all_preds, all_targets)
    return avg_loss, metrics


def plot_and_save_history(history: Dict[str, List[float]], save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.set_title("STFD Tamper Localizer - Combined Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (BCE + Soft Dice)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Foreground Dice & IoU
    ax2.plot(epochs, history["val_fg_dice"], "s-", label="Val Foreground Dice", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, history["val_fg_iou"], "^-", label="Val Foreground IoU", color="#d62728", linewidth=2)
    ax2.plot(epochs, history["val_m_iou"], "d--", label="Val Mean IoU", color="#17becf")
    ax2.set_title("STFD Tamper Localizer - Validation Localization Metrics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score [0, 1]")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Training history visualization saved to {save_path.name}")


def main():
    print("=" * 70)
    print("TRUSTTRACE: STFD TAMPER LOCALIZATION MODEL TRAINING (PHASE 2)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Leakage verification
    check_leakage_safety()

    # 2. Datasets & Loaders
    print("\nInitializing datasets with RAM caching (256x256 resolution)...", flush=True)
    train_dataset = STFDLocalizationDataset(TRAIN_MANIFEST_PATH, is_train=True)
    val_dataset = STFDLocalizationDataset(VAL_MANIFEST_PATH, is_train=False)

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

    # 3. Model construction
    print("\nConstructing ForensicUNet (base_filters=16)...", flush=True)
    model = ForensicUNet(in_channels=3, out_channels=1, base_filters=16).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ForensicUNet Total Parameters: {total_params:,}")

    criterion = CombinedBceDiceLoss(pos_weight=POS_WEIGHT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 4. Training Loop
    print("\nStarting localization training loop...", flush=True)
    best_val_fg_dice = -1.0
    best_epoch = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_fg_dice": [],
        "val_fg_iou": [],
        "val_fg_prec": [],
        "val_fg_rec": [],
        "val_m_iou": [],
        "epoch_time_seconds": [],
    }

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        total_samples = 0

        for batch_idx, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(imgs)

            loss, bce_l, dice_l = criterion(logits, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

        scheduler.step()
        train_loss = running_loss / total_samples

        # Validation evaluation
        val_loss, metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)
        epoch_time = time.time() - epoch_start

        fg_dice = metrics["foreground_dice"]
        fg_iou = metrics["foreground_iou"]
        fg_prec = metrics["foreground_precision"]
        fg_rec = metrics["foreground_recall"]
        m_iou = metrics["mean_iou"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_fg_dice"].append(fg_dice)
        history["val_fg_iou"].append(fg_iou)
        history["val_fg_prec"].append(fg_prec)
        history["val_fg_rec"].append(fg_rec)
        history["val_m_iou"].append(m_iou)
        history["epoch_time_seconds"].append(epoch_time)

        is_best = fg_dice > best_val_fg_dice
        if is_best:
            best_val_fg_dice = fg_dice
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_fg_dice": fg_dice,
                "val_fg_iou": fg_iou,
                "architecture": "ForensicUNet",
                "base_filters": 16,
                "input_resolution": list(IMAGE_SIZE),
                "seed": RANDOM_SEED,
            }, BEST_MODEL_PATH)
            star = " [*BEST CHECKPOINT SAVED]"
        else:
            star = ""

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({epoch_time:5.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val FG-Dice: {fg_dice:.4f}, FG-IoU: {fg_iou:.4f}, mIoU: {m_iou:.4f}{star}",
            flush=True
        )
        print(f"  Val FG Precision: {fg_prec:.4f}, FG Recall: {fg_rec:.4f}", flush=True)

    total_training_time = time.time() - start_time
    print(f"\nTraining completed in {total_training_time/60:.2f} minutes.")
    print(f"Best Validation Foreground Dice: {best_val_fg_dice:.4f} (Achieved at Epoch {best_epoch})")
    print(f"Saved best model checkpoint to: {BEST_MODEL_PATH}")

    # 5. Save history
    with open(HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model_architecture": "ForensicUNet",
            "epochs": NUM_EPOCHS,
            "best_epoch": best_epoch,
            "best_val_fg_dice": best_val_fg_dice,
            "total_training_time_seconds": total_training_time,
            "history": history,
            "random_seed": RANDOM_SEED,
            "device": str(device),
            "input_resolution": list(IMAGE_SIZE),
        }, f, indent=2)
    print(f"Training history saved to {HISTORY_JSON_PATH.name}")

    plot_and_save_history(history, HISTORY_PNG_PATH)


if __name__ == "__main__":
    main()
