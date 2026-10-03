"""TRUSTTRACE Model Calibration CLI Script.

Performs post-hoc temperature scaling on a validation split to minimize Negative Log Likelihood (NLL)
and Expected Calibration Error (ECE). Updates model metadata with optimal temperature.

Usage:
    python calibrate.py --model-path model/checkpoints/trusttrace-v1.0.0.pt --val-dir data/val/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.calibrator import calibrate_model
from model.dataset import (
    CLASS_LABELS,
    DatasetConfig,
    ForensicDataset,
    build_transforms,
    collect_dataset_samples,
)
from model.predictor import ForensicPredictor
from model.versioning import ModelMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Model Calibration Suite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained PyTorch checkpoint (.pt)")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to validation dataset root containing class directories")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to model metadata JSON to update")
    parser.add_argument("--lr", type=float, default=0.01, help="Calibration optimization learning rate")
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum L-BFGS iterations")
    parser.add_argument("--batch-size", type=int, default=16, help="Validation batch size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    val_dir = Path(args.val_dir)

    print("=== TRUSTTRACE Probability Calibration ===")
    print(f"Model Checkpoint:     {model_path.resolve()}")
    print(f"Validation Directory: {val_dir.resolve()}")

    if not model_path.is_file():
        print(f"[ERROR] Model file not found: {model_path}")
        return 1

    if not val_dir.is_dir():
        print(f"[ERROR] Validation directory not found: {val_dir}")
        return 1

    try:
        predictor = ForensicPredictor(
            model_path=model_path,
            metadata_path=args.metadata_path,
            verify_checksum=True
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}")
        return 1

    # Load validation data
    config = DatasetConfig(batch_size=args.batch_size)
    eval_tf = build_transforms(is_training=False, config=config)
    val_samples = collect_dataset_samples(val_dir, expected_classes=CLASS_LABELS)

    if len(val_samples) == 0:
        print(f"[ERROR] No valid validation images found in {val_dir}")
        return 1

    val_ds = ForensicDataset(val_samples, transform=eval_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nRunning temperature scaling on {len(val_ds)} validation samples...")
    optimal_temp, report = calibrate_model(
        model=predictor.model,
        val_loader=val_loader,
        device=predictor.device,
        max_iter=args.max_iter,
        lr=args.lr,
    )

    print("\n--- Calibration Results ---")
    print(f"Optimal Temperature (T): {report['optimal_temperature']:.4f}")
    print(f"Pre-Calibration  ECE:    {report['pre_calibration']['ece']:.4f} | MCE: {report['pre_calibration']['mce']:.4f}")
    print(f"Post-Calibration ECE:    {report['post_calibration']['ece']:.4f} | MCE: {report['post_calibration']['mce']:.4f}")
    print(f"ECE Improvement:         {report['ece_reduction']:+.4f}")

    # Update metadata if available
    metadata_path = predictor._resolve_metadata_path(args.metadata_path)
    if metadata_path and metadata_path.is_file():
        meta = ModelMetadata.load(metadata_path)
        meta.temperature = optimal_temp
        meta.evaluation_metrics.setdefault("calibration", {})["post_temperature_scaling"] = report
        meta.save(metadata_path)
        print(f"\nModel metadata successfully updated at: {metadata_path}")
    else:
        print("\nNote: Metadata file not found; temperature parameter was not persisted to disk.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
