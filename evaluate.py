"""TRUSTTRACE Model Evaluation CLI Script.

Computes exact confusion matrix, accuracy, precision, recall, macro-F1, per-class metrics,
Expected Calibration Error (ECE), and Out-of-Distribution (OOD) performance.

Usage:
    python evaluate.py --model-path model/checkpoints/trusttrace-v1.0.0.pt --test-dir data/test/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.dataset import (
    CLASS_LABELS,
    DatasetConfig,
    ForensicDataset,
    build_transforms,
    collect_dataset_samples,
)
from model.evaluator import (
    compute_ood_metrics,
    evaluate_dataloader,
)
from model.predictor import ForensicPredictor
from model.versioning import ModelMetadata, verify_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRUSTTRACE Forensic Model Evaluation Suite")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained PyTorch checkpoint (.pt)")
    parser.add_argument("--metadata-path", type=str, default=None, help="Path to associated model metadata (.json)")
    parser.add_argument("--test-dir", type=str, required=True, help="Path to test dataset root containing class directories")
    parser.add_argument("--ood-dir", type=str, default=None, help="Path to optional Out-of-Distribution test image directory")
    parser.add_argument("--output-file", type=str, default=None, help="Path to write evaluation JSON report")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)

    print("=== TRUSTTRACE ML Model Evaluation ===")
    print(f"Model Checkpoint: {model_path.resolve()}")
    print(f"Test Directory:   {test_dir.resolve()}")

    if not model_path.is_file():
        print(f"[ERROR] Model file not found: {model_path}")
        return 1

    if not test_dir.is_dir():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return 1

    # Load predictor to instantiate verified model
    try:
        predictor = ForensicPredictor(
            model_path=model_path,
            metadata_path=args.metadata_path,
            verify_checksum=True
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model or verify checksum: {exc}")
        return 1

    metadata = predictor.metadata
    temp = metadata.temperature if metadata else 1.0
    ood_cfg = metadata.ood_thresholds if metadata else None

    # Prepare Test DataLoader
    config = DatasetConfig(batch_size=args.batch_size)
    eval_tf = build_transforms(is_training=False, config=config)
    test_samples = collect_dataset_samples(test_dir, expected_classes=CLASS_LABELS)

    if len(test_samples) == 0:
        print(f"[ERROR] No valid test images found in {test_dir}")
        return 1

    test_ds = ForensicDataset(test_samples, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"\nEvaluating {len(test_ds)} test evidence samples across classes...")
    eval_results = evaluate_dataloader(
        model=predictor.model,
        dataloader=test_loader,
        device=predictor.device,
        temperature=temp,
        ood_thresholds=ood_cfg
    )

    # Optional Out-of-Distribution Evaluation
    if args.ood_dir:
        ood_path = Path(args.ood_dir)
        if ood_path.is_dir():
            print(f"\nComputing Out-of-Distribution metrics against {ood_path}...")
            ood_files = [p for p in ood_path.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".webp"}]
            if ood_files:
                id_confidences = []
                ood_confidences = []

                predictor.model.eval()
                with torch.no_grad():
                    # ID samples
                    for batch in test_loader:
                        logits = predictor.model(batch[0].to(predictor.device))
                        probs = torch.softmax(logits / temp, dim=-1)
                        max_p, _ = torch.max(probs, dim=-1)
                        id_confidences.extend(max_p.cpu().tolist())

                    # OOD samples
                    for img_p in ood_files:
                        try:
                            from model.dataset import load_image_safely
                            img = load_image_safely(img_p)
                            t = eval_tf(img).unsqueeze(0).to(predictor.device)
                            logits = predictor.model(t)
                            probs = torch.softmax(logits / temp, dim=-1)
                            max_p, _ = torch.max(probs, dim=-1)
                            ood_confidences.append(float(max_p.item()))
                        except Exception:
                            continue

                ood_results = compute_ood_metrics(id_confidences, ood_confidences)
                eval_results["out_of_distribution"] = ood_results
                print(f"  OOD AUROC: {ood_results['auroc']:.4f}")
                print(f"  OOD FPR95: {ood_results['fpr95']:.4f}")

    # Display Metrics Summary
    cls_metrics = eval_results["classification"]
    calib = eval_results["calibration"]

    print("\n--- Evaluation Summary ---")
    print(f"Total Samples:   {cls_metrics['total_samples']}")
    print(f"Accuracy:        {cls_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {cls_metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {cls_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:  {cls_metrics['macro_f1']:.4f}")
    print(f"ECE:             {calib['ece']:.4f}")
    print(f"MCE:             {calib['mce']:.4f}")

    print("\n--- Per-Class Performance ---")
    for cls_name, pstats in cls_metrics["per_class"].items():
        print(f"  [{cls_name:22s}] P: {pstats['precision']:.4f} | R: {pstats['recall']:.4f} | F1: {pstats['f1_score']:.4f} (N={pstats['support']})")

    print("\n--- Confusion Matrix ---")
    print("True \\ Pred:\t" + "\t".join(cls_metrics["class_labels"]))
    cm = cls_metrics["confusion_matrix"]
    for i, row in enumerate(cm):
        row_str = "\t".join(str(val) for val in row)
        print(f"{cls_metrics['class_labels'][i]:22s}\t{row_str}")

    # Output file
    out_file = Path(args.output_file) if args.output_file else Path(f"model/evaluation/{model_path.stem}_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nFull evaluation report saved to: {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
