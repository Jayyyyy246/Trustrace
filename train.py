"""TRUSTTRACE Model Training CLI Script.

Usage:
    python train.py --data-dir dataset/ --epochs 10 --arch efficientnet_b0 --version trusttrace-v1.0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from model.dataset import DatasetConfig, validate_dataset_directory
from model.trainer import ForensicTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Forensic Model Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="Path to dataset root directory containing class folders")
    parser.add_argument("--output-dir", type=str, default="model", help="Directory where checkpoints, metadata, and evaluations are stored")
    parser.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["efficientnet_b0", "mobilenet_v3_small", "convnext_tiny", "custom_forensic_cnn"], help="Backbone neural architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--version", type=str, default="trusttrace-v1.0.0", help="Semantic version identifier for model output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--validate-only", action="store_true", help="Run dataset validation without starting training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"=== TRUSTTRACE ML Training Pipeline [Version: {args.version}] ===")
    print(f"Dataset directory: {data_dir.resolve()}")
    print(f"Target architecture: {args.arch}")

    if not data_dir.is_dir():
        print(f"\n[ERROR] Dataset directory '{data_dir}' does not exist.")
        print("Required dataset layout:")
        print("  <dataset_root>/")
        print("    |-- REAL/")
        print("    |-- EDITED/")
        print("    |-- AI-GENERATED/")
        print("    \\-- SCREENSHOT-MANIPULATED/")
        print("\nNote: Datasets must be provided by the operator. TRUSTTRACE does not fabricate synthetic metrics.")
        return 1

    # Validate dataset
    print("\n[1/3] Validating dataset integrity...")
    try:
        val_summary = validate_dataset_directory(data_dir)
        print(f"  Total valid images: {val_summary['total_images']}")
        print(f"  Class counts: {val_summary['class_counts']}")
        if val_summary["classes_missing"]:
            print(f"  [WARNING] Missing class directories: {val_summary['classes_missing']}")
        if val_summary["corrupt_images"]:
            print(f"  [WARNING] Corrupted images detected: {len(val_summary['corrupt_images'])}")
    except Exception as exc:
        print(f"[ERROR] Failed dataset validation: {exc}")
        return 1

    if args.validate_only:
        print("\nDataset validation completed.")
        return 0 if val_summary["is_valid"] else 1

    if val_summary["total_images"] == 0:
        print("\n[ERROR] Dataset contains 0 images across supported classes. Aborting training.")
        return 1

    # Configure and run training
    print("\n[2/3] Initializing training pipeline with leak-free partitioning...")
    config = DatasetConfig(
        batch_size=args.batch_size,
        seed=args.seed,
    )

    trainer = ForensicTrainer(
        dataset_dir=data_dir,
        output_dir=args.output_dir,
        architecture=args.arch,
        config=config,
        model_version=args.version,
    )

    print(f"\n[3/3] Training model for {args.epochs} epochs...")
    try:
        results = trainer.train(
            epochs=args.epochs,
            lr=args.lr,
        )
        print("\n=== Training Completed Successfully ===")
        print(f"Checkpoint saved: {results['checkpoint_path']}")
        print(f"Metadata saved:   {results['metadata_path']}")
        print(f"Evaluation saved: {results['evaluation_path']}")
        print(f"SHA-256 Checksum: {results['sha256']}")
        metrics = results["metrics"]["classification"]
        print(f"Test Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Test Macro-F1:    {metrics['macro_f1']:.4f}")
        return 0
    except Exception as exc:
        print(f"\n[FATAL] Training failed with exception: {exc}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
