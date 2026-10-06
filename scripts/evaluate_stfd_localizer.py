#!/usr/bin/env python3
"""
TRUSTTRACE: STFD Tamper Localization Model Evaluation Script (Phase 2).

Evaluates the FINAL SELECTED LOCALIZATION CHECKPOINT on the HELD-OUT TEST SET ONLY:
- Manifest: data/manifests/trusttrace_stfd_clustered_test.csv (590 samples)
- Computes:
  * Pixel Accuracy
  * Foreground IoU (Jaccard)
  * Foreground Dice (F1)
  * Foreground Precision & Recall
  * Background IoU & Mean IoU
  * All-Background Baseline comparison
- Generates 5-row x 5-column qualitative visualization across all manipulation types:
  Original | Ground-Truth Mask | Predicted Probability Map | Thresholded Mask | Overlay
- Saves:
  1. reports/stfd_localization_test_report.json
  2. reports/stfd_localization_test_report.md
  3. reports/stfd_localization_examples.png
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

TEST_MANIFEST_PATH = MANIFESTS_DIR / "trusttrace_stfd_clustered_test.csv"
BEST_MODEL_PATH = MODELS_DIR / "stfd_localization_best.pt"

TEST_REPORT_JSON = REPORTS_DIR / "stfd_localization_test_report.json"
TEST_REPORT_MD = REPORTS_DIR / "stfd_localization_test_report.md"
EXAMPLES_PNG = REPORTS_DIR / "stfd_localization_examples.png"

IMAGE_SIZE = (256, 256)
THRESHOLD = 0.5


class DoubleConv(nn.Module):
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
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 16):
        super().__init__()
        bf = base_filters
        self.inc = DoubleConv(in_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bf * 4, bf * 8))

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


class STFDTestLocalizationDataset(Dataset):
    def __init__(self, manifest_path: Path):
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str, str, str]:
        r = self.rows[idx]
        img_raw = Image.open(r["image_path"]).convert("RGB")
        mask_raw = Image.open(r["mask_path"]).convert("L")

        img = img_raw.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        mask = mask_raw.resize(IMAGE_SIZE, Image.Resampling.NEAREST)

        img_t = self.normalize(self.to_tensor(img))
        mask_binary = (np.array(mask, dtype=np.uint8) > 128).astype(np.float32)
        mask_t = torch.from_numpy(mask_binary).unsqueeze(0)

        return img_t, mask_t, r["sample_id"], r["manipulation_type"], r["image_path"], r["mask_path"]


def generate_qualitative_examples(
    model: nn.Module,
    dataset: STFDTestLocalizationDataset,
    device: torch.device,
    save_path: Path
) -> None:
    """
    Generates a deterministic 5-row x 5-column qualitative grid across the 5 manipulation types.
    """
    categories = ["COPY_MOVE", "SPLICING", "REMOVAL", "INSERTION", "REPLACEMENT"]
    sample_indices = []

    for cat in categories:
        for idx, r in enumerate(dataset.rows):
            if r["manipulation_type"] == cat:
                sample_indices.append(idx)
                break

    model.eval()
    fig, axes = plt.subplots(len(categories), 5, figsize=(18, 3.6 * len(categories)))

    for row_idx, data_idx in enumerate(sample_indices):
        img_t, mask_t, sid, cat, img_path, _ = dataset[data_idx]

        with torch.no_grad():
            img_batch = img_t.unsqueeze(0).to(device)
            logit = model(img_batch)
            prob_map = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_mask = (prob_map > THRESHOLD).astype(np.uint8)

        # Original unnormalized image
        orig_img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        orig_np = np.array(orig_img)
        gt_mask_np = mask_t.squeeze().numpy().astype(np.uint8)

        # 1. Original Screenshot
        axes[row_idx, 0].imshow(orig_np)
        axes[row_idx, 0].set_title(f"{cat}\n(Original Screenshot)", fontsize=10, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 2. Ground-Truth Mask
        axes[row_idx, 1].imshow(gt_mask_np, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Ground-Truth Mask\n(STFD Binary Target)", fontsize=10)
        axes[row_idx, 1].axis("off")

        # 3. Predicted Probability Map
        im3 = axes[row_idx, 2].imshow(prob_map, cmap="plasma", vmin=0, vmax=1)
        axes[row_idx, 2].set_title("Predicted Probability\n(Continuous Heatmap)", fontsize=10)
        axes[row_idx, 2].axis("off")

        # 4. Thresholded Prediction
        axes[row_idx, 3].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 3].set_title(f"Thresholded Mask\n(tau = {THRESHOLD})", fontsize=10)
        axes[row_idx, 3].axis("off")

        # 5. Visual Overlay
        overlay = orig_np.copy()
        # Red highlight where pred_mask == 1
        red_mask = pred_mask == 1
        overlay[red_mask, 0] = np.clip(overlay[red_mask, 0] * 0.4 + 255 * 0.6, 0, 255).astype(np.uint8)
        overlay[red_mask, 1] = (overlay[red_mask, 1] * 0.4).astype(np.uint8)
        overlay[red_mask, 2] = (overlay[red_mask, 2] * 0.4).astype(np.uint8)

        axes[row_idx, 4].imshow(overlay)
        axes[row_idx, 4].set_title("Predicted Overlay\n(Superimposed Detection)", fontsize=10, fontweight="bold")
        axes[row_idx, 4].axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Qualitative localization examples saved to {save_path.name}")


def main():
    print("=" * 70)
    print("TRUSTTRACE: STFD TAMPER LOCALIZATION MODEL EVALUATION (PHASE 2)")
    print("=" * 70)

    if not BEST_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {BEST_MODEL_PATH}. Train the localizer first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load checkpoint
    print(f"Loading checkpoint from {BEST_MODEL_PATH.name}...", flush=True)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

    model = ForensicUNet(in_channels=3, out_channels=1, base_filters=checkpoint.get("base_filters", 16)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_epoch = checkpoint.get("epoch", "unknown")
    val_fg_dice = checkpoint.get("val_fg_dice", 0.0)
    print(f"Loaded checkpoint trained to Epoch {best_epoch} (Val FG-Dice: {val_fg_dice:.4f}).")

    # 2. Test loader
    test_dataset = STFDTestLocalizationDataset(TEST_MANIFEST_PATH)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Loaded {len(test_dataset)} test samples from {TEST_MANIFEST_PATH.name}.")

    # 3. Evaluate
    print("\nEvaluating on held-out test set...", flush=True)
    t0 = time.time()

    all_preds_list = []
    all_targets_list = []

    with torch.no_grad():
        for imgs, targets, _, _, _, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            preds_bin = (probs > THRESHOLD).cpu().numpy().astype(np.uint8)
            targets_np = targets.numpy().astype(np.uint8)

            all_preds_list.append(preds_bin)
            all_targets_list.append(targets_np)

    eval_time = time.time() - t0
    all_preds = np.concatenate(all_preds_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)

    # 4. Metrics calculation
    tp = float(np.sum((all_preds == 1) & (all_targets == 1)))
    fp = float(np.sum((all_preds == 1) & (all_targets == 0)))
    fn = float(np.sum((all_preds == 0) & (all_targets == 1)))
    tn = float(np.sum((all_preds == 0) & (all_targets == 0)))

    total_pixels = tp + fp + fn + tn
    eps = 1e-7

    pixel_acc = (tp + tn) / max(total_pixels, 1.0)
    fg_prec = tp / max(tp + fp, eps)
    fg_rec = tp / max(tp + fn, eps)
    fg_dice = (2.0 * tp) / max(2.0 * tp + fp + fn, eps)
    fg_iou = tp / max(tp + fp + fn, eps)

    bg_iou = tn / max(tn + fp + fn, eps)
    m_iou = (fg_iou + bg_iou) / 2.0

    # All-Background baseline
    bl_tn = tn + fp
    bl_fn = tp + fn
    bl_pixel_acc = bl_tn / max(total_pixels, 1.0)
    bl_bg_iou = bl_tn / max(bl_tn + bl_fn, eps)
    bl_m_iou = (0.0 + bl_bg_iou) / 2.0

    print("\n" + "=" * 65)
    print("HELD-OUT TEST RESULTS (STFD TAMPER LOCALIZATION MODEL)")
    print("=" * 65)
    print(f"STFD tamper localization test performance:")
    print(f"  Pixel Accuracy:         {pixel_acc * 100:.2f}% (All-Background Baseline: {bl_pixel_acc * 100:.2f}%)")
    print(f"  Foreground IoU:         {fg_iou:.4f} (All-Background Baseline: 0.0000)")
    print(f"  Foreground Dice (F1):   {fg_dice:.4f} (All-Background Baseline: 0.0000)")
    print(f"  Foreground Precision:   {fg_prec:.4f}")
    print(f"  Foreground Recall:      {fg_rec:.4f}")
    print(f"  Background IoU:         {bg_iou:.4f}")
    print(f"  Mean IoU (mIoU):        {m_iou:.4f} (All-Background Baseline: {bl_m_iou:.4f})")
    print(f"  Inference Time:         {eval_time:.2f}s ({len(test_dataset)/eval_time:.1f} screenshots/s)")

    # 5. Generate qualitative examples
    print("\nGenerating qualitative visualization grid...", flush=True)
    generate_qualitative_examples(model, test_dataset, device, EXAMPLES_PNG)

    # 6. Save JSON report
    report_data = {
        "model_architecture": "ForensicUNet",
        "task": "STFD Supervised Tamper Localization",
        "checkpoint_epoch": best_epoch,
        "input_resolution": list(IMAGE_SIZE),
        "dataset": {
            "test_manifest": str(TEST_MANIFEST_PATH.name),
            "test_samples": len(test_dataset),
        },
        "metrics": {
            "pixel_accuracy": pixel_acc,
            "foreground_iou": fg_iou,
            "foreground_dice": fg_dice,
            "foreground_precision": fg_prec,
            "foreground_recall": fg_rec,
            "background_iou": bg_iou,
            "mean_iou": m_iou,
        },
        "baselines": {
            "all_background_pixel_accuracy": bl_pixel_acc,
            "all_background_foreground_iou": 0.0,
            "all_background_foreground_dice": 0.0,
            "all_background_mean_iou": bl_m_iou,
        },
        "pixel_confusion": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        },
        "evaluation_time_seconds": eval_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "disclaimer": "This benchmark evaluates the STFD supervised tamper localization model, not the final TRUSTTRACE authenticity pipeline.",
    }

    with open(TEST_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"JSON test report saved to {TEST_REPORT_JSON.name}")

    # 7. Save Markdown report
    md_content = f"""# TRUSTTRACE: STFD Tamper Localization Model Test Report (Phase 2)

**Task:** Supervised Image Tamper Localization (Pixel-Level Binary Segmentation)  
**Model Architecture:** ForensicUNet (3-Stage Multi-Scale Skip Connections, base_filters=16)  
**Input Resolution:** 256 × 256 × 3  
**Checkpoint Source:** Best Validation Foreground-Dice Checkpoint (`stfd_localization_best.pt`, Epoch {best_epoch})  
**Evaluation Set:** Held-out Test Set (`trusttrace_stfd_clustered_test.csv`, {len(test_dataset)} samples)  
**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

> [!IMPORTANT]
> **Anti-Overclaim Notice:** STFD tamper localization test performance measures pixel-level detection of tampered regions on confirmed-tampered screenshots using ground-truth binary masks. This is NOT a measure of TRUSTTRACE system-wide authenticity or forgery detection.

---

## 1. Overall Held-Out Test Metrics vs. All-Background Baseline

| Metric | ForensicUNet (Ours) | All-Background Baseline | Relative Gain / Note |
| :--- | :---: | :---: | :--- |
| **Foreground IoU (Jaccard)** | **{fg_iou:.4f}** | 0.0000 | Direct overlap metric |
| **Foreground Dice (F1)** | **{fg_dice:.4f}** | 0.0000 | Harmonic mean of FG Precision & Recall |
| **Foreground Precision** | **{fg_prec:.4f}** | 0.0000 | Proportion of detected pixels truly tampered |
| **Foreground Recall** | **{fg_rec:.4f}** | 0.0000 | Proportion of true tampered pixels detected |
| **Background IoU** | {bg_iou:.4f} | {bl_bg_iou:.4f} | Background area preservation |
| **Mean IoU (mIoU)** | **{m_iou:.4f}** | {bl_m_iou:.4f} | Macro average across FG and BG |
| **Pixel Accuracy** | {pixel_acc * 100:.2f}% | {bl_pixel_acc * 100:.2f}% | Dominated by ~98.9% background pixels |

---

## 2. Confusion Matrix of Pixels (Test Set Aggregation)

| Pixel State | Predicted Tampered (1) | Predicted Background (0) | Total Pixels |
| :--- | :---: | :---: | :---: |
| **True Tampered (1)** | **{int(tp):,}** (TP) | **{int(fn):,}** (FN) | {int(tp+fn):,} |
| **True Background (0)** | **{int(fp):,}** (FP) | **{int(tn):,}** (TN) | {int(fp+tn):,} |

---

## 3. Benchmark Speed & Execution
- **Test Evaluation Time:** {eval_time:.2f} seconds
- **Throughput:** {len(test_dataset)/eval_time:.1f} screenshots / second
- **Device:** {device}
- **Qualitative Visualizations:** Available in [stfd_localization_examples.png](file:///c:/Users/jay/New%20folder%20%282%29/Trustrace/reports/stfd_localization_examples.png)
"""
    with open(TEST_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown test report saved to {TEST_REPORT_MD.name}")


if __name__ == "__main__":
    main()
